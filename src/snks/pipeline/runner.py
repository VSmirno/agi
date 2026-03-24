"""Pipeline: end-to-end perception cycle and training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import numpy as np

from snks.daf.engine import DafEngine
from snks.daf.prediction import PredictionEngine
from snks.daf.types import PipelineConfig
from snks.encoder.encoder import VisualEncoder
from snks.encoder.text_encoder import TextEncoder
from snks.sks.detection import phase_coherence_matrix, cofiring_coherence_matrix, detect_sks
from snks.sks.metrics import compute_nmi
from snks.sks.tracking import SKSTracker


@dataclass
class CycleResult:
    """Result of one perception cycle."""
    sks_clusters: dict[int, set[int]]
    n_sks: int
    mean_prediction_error: float
    n_spikes: int
    cycle_time_ms: float


@dataclass
class TrainResult:
    """Result of training on a dataset."""
    n_cycles: int
    final_nmi: float
    mean_pe_history: list[float] = field(default_factory=list)
    sks_count_history: list[int] = field(default_factory=list)


class Pipeline:
    """Orchestrates: encode → inject → step DAF → detect SKS → track → predict."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.encoder = VisualEncoder(config.encoder)
        self.text_encoder = TextEncoder(config.encoder, device=config.device)
        self.engine = DafEngine(config.daf, enable_learning=True)
        self.tracker = SKSTracker()
        self.prediction = PredictionEngine(config.prediction)
        self._sks_config = config.sks
        self._motor_currents: torch.Tensor | None = None

    def inject_motor_currents(self, currents: torch.Tensor) -> None:
        """Set motor currents for dual injection (sensory + motor).

        Args:
            currents: (N,) motor current vector from MotorEncoder.
        """
        self._motor_currents = currents.to(self.engine.device)

    def perception_cycle(
        self,
        image: torch.Tensor | None = None,
        text: str | None = None,
    ) -> CycleResult:
        """Process one image and/or text through the full pipeline.

        Args:
            image: (H, W) float32 grayscale image. Optional.
            text: input string. Optional.
            At least one of image or text must be provided.

        Returns:
            CycleResult with detected SKS and metrics.
        """
        if image is None and text is None:
            raise ValueError("perception_cycle: укажите image или text")

        t0 = time.perf_counter()

        # 0. Reset dynamic state to resting (prevents carryover between stimuli)
        #    Preserves: col 2 (freq), col 3 (threshold), col 5-7 (aux)
        cfg = self.engine.config
        if cfg.oscillator_model == "kuramoto":
            self.engine.states[:, 0] = torch.rand(cfg.num_nodes, device=self.engine.device) * 2.0 * torch.pi
        else:
            self.engine.states[:, 0] = torch.randn(cfg.num_nodes, device=self.engine.device) * 0.1
            self.engine.states[:, 4] = 0.0  # w_recovery

        # 1. Encode inputs → currents (average when both modalities present)
        n_nodes = self.engine.config.num_nodes
        currents = None

        if image is not None:
            sdr = self.encoder.encode(image)
            currents = self.encoder.sdr_to_currents(sdr, n_nodes).to(self.engine.device)

        if text is not None:
            text_sdr = self.text_encoder.encode(text)
            text_currents = self.text_encoder.sdr_to_currents(text_sdr, n_nodes).to(self.engine.device)
            if currents is None:
                currents = text_currents
            else:
                currents = (currents + text_currents) / 2.0  # усреднение, не сумма

        # 1b. Dual injection: add motor currents if present
        if self._motor_currents is not None:
            motor = self._motor_currents
            if currents.dim() == 2 and motor.dim() == 1:
                # Add motor currents to channel 0 (voltage)
                currents[:, 0] = currents[:, 0] + motor
            elif currents.dim() == 1 and motor.dim() == 1:
                currents = currents + motor
            self._motor_currents = None  # consumed

        self.engine.set_input(currents)

        # 2. Step DAF
        step_result = self.engine.step(self.config.steps_per_cycle)

        # 3. Detect SKS
        sks_config = self._sks_config
        mode = sks_config.coherence_mode
        if mode == "auto":
            mode = "phase" if self.engine.config.oscillator_model == "kuramoto" else "rate"

        if mode == "rate":
            # Fast rate-based detection: threshold on firing rate
            fired_rate = step_result.fired_history.float().mean(dim=0)  # (N,)
            threshold = fired_rate.mean() + 3.0 * fired_rate.std()
            active = (fired_rate > threshold).nonzero(as_tuple=False).flatten()
            if len(active) >= sks_config.min_cluster_size:
                global_clusters = [{int(i) for i in active.tolist()}]
            else:
                global_clusters = []
        elif mode == "phase":
            coherence, active_idx = phase_coherence_matrix(
                step_result.states, top_k=sks_config.top_k
            )
            local_clusters = detect_sks(
                coherence, eps=sks_config.dbscan_eps,
                min_samples=sks_config.dbscan_min_samples,
                min_size=sks_config.min_cluster_size,
            )
            global_clusters = [
                {int(active_idx[i]) for i in cluster}
                for cluster in local_clusters
            ]
        else:
            coherence, active_idx = cofiring_coherence_matrix(
                step_result.fired_history, top_k=sks_config.top_k
            )
            local_clusters = detect_sks(
                coherence, eps=sks_config.dbscan_eps,
                min_samples=sks_config.dbscan_min_samples,
                min_size=sks_config.min_cluster_size,
            )
            global_clusters = [
                {int(active_idx[i]) for i in cluster}
                for cluster in local_clusters
            ]

        # 4. Track
        tracked = self.tracker.update(global_clusters)

        # 5. Predict
        predicted = self.prediction.predict()
        active_sks_ids = set(tracked.keys())
        self.prediction.observe(active_sks_ids)

        pe = self.prediction.compute_prediction_error(
            predicted, active_sks_ids,
            n_nodes=self.engine.config.num_nodes,
            sks_clusters=tracked,
        )
        mean_pe = float(pe.mean())

        # 6. Modulate STDP learning rate (for next cycle)
        lr_mod = self.prediction.get_lr_modulation(pe, self.config.prediction.pe_alpha)
        # Store for potential use in next step
        self._lr_modulation = lr_mod

        elapsed = (time.perf_counter() - t0) * 1000

        return CycleResult(
            sks_clusters=tracked,
            n_sks=len(tracked),
            mean_prediction_error=mean_pe,
            n_spikes=step_result.n_spikes,
            cycle_time_ms=elapsed,
        )

    def train_on_dataset(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 1,
    ) -> TrainResult:
        """Train on dataset and compute final NMI.

        NMI is computed by collecting firing-rate vectors from the last epoch,
        clustering them with k-means (k = number of true classes), and comparing
        the predicted cluster assignments to true labels.

        Args:
            images: (N, H, W) float32.
            labels: (N,) int64 true class labels.
            epochs: number of passes through dataset.

        Returns:
            TrainResult with training metrics.
        """
        from sklearn.cluster import KMeans

        n_images = images.shape[0]
        n_nodes = self.engine.config.num_nodes
        n_classes = int(labels.max().item()) + 1
        pe_history: list[float] = []
        sks_history: list[int] = []

        last_epoch_patterns: torch.Tensor | None = None
        last_epoch_order: torch.Tensor | None = None

        for epoch in range(epochs):
            perm = torch.randperm(n_images)

            if epoch == epochs - 1:
                last_epoch_patterns = torch.zeros(n_images, n_nodes)
                last_epoch_order = perm

            for i, idx in enumerate(perm):
                result = self.perception_cycle(images[idx])
                pe_history.append(result.mean_prediction_error)
                sks_history.append(result.n_sks)

                # Record firing rate vector on last epoch
                if epoch == epochs - 1:
                    fired_history = self.engine.get_fired_history()
                    if fired_history is not None:
                        last_epoch_patterns[i] = fired_history.float().mean(dim=0).cpu()

        # Compute NMI: cluster firing patterns → compare to true labels
        if last_epoch_patterns is not None and last_epoch_order is not None:
            X = last_epoch_patterns.numpy()
            true_labels = labels[last_epoch_order].numpy()

            km = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
            pred_labels = km.fit_predict(X)
            final_nmi = compute_nmi(pred_labels, true_labels)
        else:
            final_nmi = 0.0

        return TrainResult(
            n_cycles=len(pe_history),
            final_nmi=final_nmi,
            mean_pe_history=pe_history,
            sks_count_history=sks_history,
        )
