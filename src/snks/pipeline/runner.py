"""Pipeline: end-to-end perception cycle and training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import numpy as np

import torch as _torch

from snks.daf.engine import DafEngine
from snks.daf.hac_prediction import HACPredictionEngine
from snks.dcam.episodic_hac import EpisodicHACPredictor
from snks.daf.prediction import PredictionEngine
from snks.daf.types import ConfiguratorAction, PipelineConfig
from snks.dcam.hac import HACEngine
from snks.device import get_device
from snks.encoder.encoder import VisualEncoder
from snks.encoder.hebbian import HebbianEncoder
from snks.encoder.text_encoder import TextEncoder
from snks.language.grounding_map import GroundingMap
from snks.gws.workspace import GlobalWorkspace, GWSState
from snks.metacog.configurator import Configurator
from snks.metacog.cost_module import IntrinsicCostModule
from snks.metacog.monitor import MetacogMonitor, MetacogState
from snks.sks.detection import phase_coherence_matrix, cofiring_coherence_matrix, detect_sks
from snks.sks.embedder import SKSEmbedder
from snks.sks.meta_embedder import MetaEmbedder
from snks.sks.metrics import compute_nmi
from snks.sks.tracking import SKSTracker


@dataclass
class _CycleResultProxy:
    """Minimal proxy for MetacogMonitor.update before full CycleResult is built."""
    mean_prediction_error: float
    winner_pe: float = 0.0
    meta_pe: float = 0.0


@dataclass
class CycleResult:
    """Result of one perception cycle."""
    sks_clusters: dict[int, set[int]]
    n_sks: int
    mean_prediction_error: float
    n_spikes: int
    cycle_time_ms: float
    gws: GWSState | None = None
    metacog: MetacogState | None = None
    # Stage 9: HAC prediction fields
    winner_pe: float = 0.0
    winner_embedding: "_torch.Tensor | None" = None
    hac_predicted: "_torch.Tensor | None" = None
    # Stage 10: Hierarchical Prediction
    meta_embedding: "_torch.Tensor | None" = None
    meta_pe: float = 0.0
    # Stage 12: Intrinsic Cost Module
    mean_firing_rate: float = 0.0
    # Stage 13: Configurator
    configurator_action: "ConfiguratorAction | None" = None


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
        if config.encoder.hebbian:
            self.encoder = HebbianEncoder(config.encoder, lr=config.encoder.hebbian_lr)
        else:
            self.encoder = VisualEncoder(config.encoder)
        self._hebbian_cycle_count = 0
        self.text_encoder = TextEncoder(config.encoder, device=config.device)
        self.engine = DafEngine(config.daf, enable_learning=True)
        self.tracker = SKSTracker()
        self.prediction = PredictionEngine(config.prediction)
        self.gws = GlobalWorkspace(config.gws)
        self.metacog = MetacogMonitor(config.metacog)
        self._sks_config = config.sks
        self._motor_currents: torch.Tensor | None = None
        # Stage 9: HAC embedding + prediction
        # HAC runs on CPU: torch.fft.rfft on AMD ROCm gfx1151 causes segfault.
        # 2048-dim FFT on CPU is negligible (<1ms). DAF/encoder stay on GPU.
        self._hac = HACEngine(dim=config.sks_embed.hac_dim, device=_torch.device("cpu"))
        self.embedder = SKSEmbedder(
            n_nodes=config.daf.num_nodes,
            hac_dim=config.sks_embed.hac_dim,
            device=config.device,
        )
        if config.hac_prediction.use_episodic_buffer:
            self.hac_prediction: HACPredictionEngine | EpisodicHACPredictor = (
                EpisodicHACPredictor(self._hac, config.hac_prediction.episodic_capacity)
            )
        else:
            self.hac_prediction = HACPredictionEngine(self._hac, config.hac_prediction)
        self._broadcast_currents: torch.Tensor | None = None

        # Stage 10: Hierarchical Prediction (L2 meta-embedding + L2 predictor)
        from snks.daf.types import HACPredictionConfig as _HPC
        _l2_pred_cfg = _HPC(memory_decay=config.hierarchical.memory_decay, enabled=True)
        self.meta_embedder = MetaEmbedder(self._hac, config.hierarchical)
        self.l2_predictor = HACPredictionEngine(self._hac, _l2_pred_cfg)
        self._prev_l2_predicted: "_torch.Tensor | None" = None

        # Stage 12: Intrinsic Cost Module
        cost_cfg = config.cost_module
        if cost_cfg.firing_rate_target is None:
            cost_cfg.firing_rate_target = config.daf.homeostasis_target
        self.cost_module = IntrinsicCostModule(cost_cfg)

        # Stage 13: Configurator
        self.configurator = Configurator(
            config=config.configurator,
            daf_config=config.daf,
            metacog_config=config.metacog,
            hac_pred_config=config.hac_prediction,
        )

        # Stage 14: cached last result for EmbodiedAgent access
        self.last_cycle_result: "CycleResult | None" = None

        # Stage 19: cross-modal grounding map + priming
        self.grounding_map = GroundingMap()

    def save_checkpoint(self, path: str) -> None:
        """Save pipeline state: DAF weights + GroundingMap.

        Args:
            path: Base path prefix. Creates {path}_daf.safetensors,
                  {path}_grounding_* files, and {path}_pipeline.json.
        """
        import json as _json
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.engine.save_state(path)
        self.grounding_map.save(path + "_grounding")

        meta = {
            "steps_per_cycle": self.config.steps_per_cycle,
            "priming_strength": self.config.priming_strength,
            "num_nodes": self.config.daf.num_nodes,
            "grounding_vocab_size": self.grounding_map.vocab_size,
        }
        with open(path + "_pipeline.json", "w") as f:
            _json.dump(meta, f, indent=2)

    def load_checkpoint(self, path: str) -> None:
        """Load pipeline state: DAF weights + GroundingMap.

        Args:
            path: Base path prefix matching a previous save_checkpoint() call.
        """
        self.engine.load_state(path)
        self.grounding_map.load(path + "_grounding")

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

        # 1. Encode inputs → currents
        n_nodes = self.engine.config.num_nodes
        zones = self.engine.zones

        if zones is not None:
            # Stage 19: zone-based injection — each modality writes to its own zone
            self.engine._external_currents.zero_()

            if image is not None:
                sdr = self.encoder.encode(image)
                vis_zone = zones["visual"]
                vis_currents = self.encoder.sdr_to_currents(sdr, n_nodes, zone=vis_zone).to(self.engine.device)
                self.engine.set_input_zone(vis_currents, "visual")

            if text is not None:
                text_sdr = self.text_encoder.encode(text)
                ling_zone = zones["linguistic"]
                text_currents = self.text_encoder.sdr_to_currents(text_sdr, n_nodes, zone=ling_zone).to(self.engine.device)
                self.engine.set_input_zone(text_currents, "linguistic")

                # Stage 19: cross-modal priming
                if image is not None:
                    # Co-activation: register visual SDR for future priming
                    self.grounding_map.register_visual(text, vis_currents.detach())
                else:
                    # Text-only: inject priming current into visual zone
                    priming_sdr = self.grounding_map.word_to_visual_sdr(text)
                    if priming_sdr is not None:
                        strength = self.config.priming_strength
                        self.engine.set_input_zone(
                            priming_sdr.to(self.engine.device) * strength, "visual",
                        )

            # Motor and broadcast currents: applied globally (not zone-routed)
            if self._motor_currents is not None:
                self.engine._external_currents[:, 0] += self._motor_currents.to(self.engine.device)
                self._motor_currents = None

            if self._broadcast_currents is not None:
                self.engine._external_currents[:, 0] += self._broadcast_currents.to(self.engine.device)
                self._broadcast_currents = None
        else:
            # Legacy path: flat DAF, average modalities
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
                    currents = (currents + text_currents) / 2.0

            # 1b. Dual injection: add motor currents if present
            if self._motor_currents is not None:
                motor = self._motor_currents
                if currents.dim() == 2 and motor.dim() == 1:
                    currents[:, 0] = currents[:, 0] + motor
                elif currents.dim() == 1 and motor.dim() == 1:
                    currents = currents + motor
                self._motor_currents = None

            # 1c. [Stage 9] Inject broadcast currents from previous cycle
            if self._broadcast_currents is not None:
                broadcast = self._broadcast_currents.to(self.engine.device)
                if currents.dim() == 1:
                    currents = currents + broadcast
                else:
                    currents[:, 0] = currents[:, 0] + broadcast
                self._broadcast_currents = None

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

        # 4b. [Stage 9] Compute SKS embeddings in HAC space
        embeddings = self.embedder.embed(tracked)

        # 5. Predict (discrete, kept for backward compatibility / Exp 20 comparison)
        predicted = self.prediction.predict()
        active_sks_ids = set(tracked.keys())
        self.prediction.observe(active_sks_ids)

        pe = self.prediction.compute_prediction_error(
            predicted, active_sks_ids,
            n_nodes=self.engine.config.num_nodes,
            sks_clusters=tracked,
        )
        mean_pe = float(pe.mean())

        # 5b. [Stage 9] HAC prediction — predict_next before observe, then observe
        hac_predicted = self.hac_prediction.predict_next(embeddings)
        self.hac_prediction.observe(embeddings)
        # winner_pe is computed in step 7b after GWS selects winner

        # 6. Modulate STDP learning rate (for next cycle)
        lr_mod = self.prediction.get_lr_modulation(pe, self.config.prediction.pe_alpha)
        # Store for potential use in next step
        self._lr_modulation = lr_mod

        # 7. GWS: select dominant SKS winner
        gws_state = self.gws.select_winner(tracked, fired_history=step_result.fired_history)

        # 7b. [Stage 9] Compute per-winner PE in HAC space (after winner is known)
        winner_pe = 0.0
        winner_embedding = None
        if hac_predicted is not None and gws_state is not None:
            winner_id = gws_state.winner_id
            if winner_id in embeddings:
                winner_embedding = embeddings[winner_id]
                winner_pe = self.hac_prediction.compute_winner_pe(hac_predicted, winner_embedding)

        # 7c–7g. [Stage 10] Hierarchical Prediction (L2 meta-embedding)
        meta_embed = None
        meta_pe = 0.0
        if self.config.hierarchical.enabled:
            meta_embed = self.meta_embedder.update(embeddings)
            if meta_embed is not None:
                l2_input = {"meta": meta_embed}
                l2_predicted = self.l2_predictor.predict_next(l2_input)
                if l2_predicted is not None and self._prev_l2_predicted is not None:
                    meta_pe = self.l2_predictor.compute_winner_pe(
                        self._prev_l2_predicted, meta_embed
                    )
                self.l2_predictor.observe(l2_input)
                self._prev_l2_predicted = l2_predicted

        # 8. Metacognition: compute confidence, apply policy
        metacog_state = self.metacog.update(
            gws_state, _CycleResultProxy(mean_pe, winner_pe, meta_pe)
        )
        self.metacog.apply_policy(metacog_state, self.engine.config)

        # 8b. [Stage 9] Collect broadcast currents for next cycle
        broadcast = self.metacog.get_broadcast_currents(self.engine.config.num_nodes)
        if broadcast is not None:
            self._broadcast_currents = broadcast.to(self.engine.device)

        # 8c. [Stage 12] Intrinsic Cost Module
        mean_firing_rate = float(self.engine.states[:, 0].mean().item())
        if self.config.cost_module.enabled:
            cost_state = self.cost_module.compute(metacog_state, mean_firing_rate)
            metacog_state.cost = cost_state

        # 8d. [Stage 13] Configurator FSM
        configurator_action = None
        if self.config.configurator.enabled:
            configurator_action = self.configurator.update(metacog_state)

        elapsed = (time.perf_counter() - t0) * 1000

        result = CycleResult(
            sks_clusters=tracked,
            n_sks=len(tracked),
            mean_prediction_error=mean_pe,
            n_spikes=step_result.n_spikes,
            cycle_time_ms=elapsed,
            gws=gws_state,
            metacog=metacog_state,
            winner_pe=winner_pe,
            winner_embedding=winner_embedding,
            hac_predicted=hac_predicted,
            meta_embedding=meta_embed,
            meta_pe=meta_pe,
            mean_firing_rate=mean_firing_rate,
            configurator_action=configurator_action,
        )
        # Stage 40: Hebbian encoder learning — update after PE is computed
        if isinstance(self.encoder, HebbianEncoder) and image is not None:
            self._hebbian_cycle_count += 1
            if self._hebbian_cycle_count % self.config.encoder.hebbian_update_interval == 0:
                sdr_for_hebbian = self.encoder.encode(image)
                self.encoder.hebbian_update(image, sdr_for_hebbian, mean_pe)

        self.last_cycle_result = result  # Stage 14: cached for EmbodiedAgent
        return result

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
