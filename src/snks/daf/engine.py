"""DafEngine — orchestrates oscillator dynamics, coupling, integration, STDP, and homeostasis."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from snks.daf.coupling import compute_fhn_coupling, compute_kuramoto_coupling
from snks.daf.graph import SparseDafGraph
from snks.daf.homeostasis import Homeostasis
from snks.daf.integrator import DerivativeFn, integrate_n_steps
from snks.daf.oscillator import fhn_derivatives, init_states, kuramoto_derivatives
from snks.daf.stdp import STDP, STDPResult
from snks.daf.types import DafConfig
from snks.device import get_device


@dataclass
class StepResult:
    """Result of a DafEngine.step() call."""

    states: torch.Tensor  # (N, 8) final states
    fired_history: torch.Tensor  # (T, N) bool
    prediction_error: torch.Tensor  # (N,) float — placeholder for Stage 1
    mean_pe: float
    n_spikes: int
    elapsed_model_time: float
    stdp_result: STDPResult | None = None


class DafEngine:
    """Dynamic Attractor Fields engine.

    Orchestrates: oscillator dynamics → coupling → Euler-Maruyama integration
    → STDP learning → homeostatic regulation.
    """

    def __init__(self, config: DafConfig, enable_learning: bool = True) -> None:
        self.config = config
        self.device = get_device(config.device)

        N = config.num_nodes
        self.states = init_states(N, config.state_dim, config.oscillator_model, self.device,
                                   omega_std=config.omega_std)
        self.graph = SparseDafGraph.random_sparse(N, config.avg_degree, self.device)

        self._external_currents = torch.zeros(N, config.state_dim, device=self.device)
        self._last_fired_history: torch.Tensor | None = None
        self.step_count: int = 0

        # Learning components
        self.enable_learning = enable_learning
        self.stdp = STDP(config)
        self.homeostasis = Homeostasis(config, N, self.device)

    def set_input(self, currents: torch.Tensor) -> None:
        """Set external currents. currents: (N, 8) or (N,) for channel 0 only."""
        if currents.dim() == 1:
            self._external_currents.zero_()
            self._external_currents[:, 0] = currents
        else:
            self._external_currents.copy_(currents)

    def step(self, n_steps: int = 100) -> StepResult:
        """Run n integration steps with coupling, STDP, and homeostasis.

        Returns StepResult with cloned states (safe to hold reference).
        """
        derivative_fn = self._make_derivative_fn()

        spike_mode = "phase_crossing" if self.config.oscillator_model == "kuramoto" else "threshold"
        self.states, fired_history = integrate_n_steps(
            self.states,
            derivative_fn,
            n_steps=n_steps,
            dt=self.config.dt,
            noise_sigma=self.config.noise_sigma,
            spike_mode=spike_mode,
        )

        self._last_fired_history = fired_history
        self.step_count += n_steps

        # Apply learning
        stdp_result = None
        if self.enable_learning:
            stdp_result = self.stdp.apply(self.graph, fired_history)
            self.homeostasis.update(fired_history, self.states)

            # Structural plasticity: prune weak edges periodically
            if self.step_count % self.config.structural_interval == 0:
                self._structural_prune()

        n_spikes = int(fired_history.sum())

        return StepResult(
            states=self.states.clone(),
            fired_history=fired_history,
            prediction_error=torch.zeros(self.config.num_nodes, device=self.device),
            mean_pe=0.0,
            n_spikes=n_spikes,
            elapsed_model_time=n_steps * self.config.dt,
            stdp_result=stdp_result,
        )

    def get_states(self) -> torch.Tensor:
        """Return (N, 8) current states on device."""
        return self.states

    def get_fired_history(self) -> torch.Tensor | None:
        """Return (T, N) bool from last step() call, or None."""
        return self._last_fired_history

    def _make_derivative_fn(self) -> DerivativeFn:
        """Build derivative closure capturing graph, config, and external currents."""
        graph = self.graph
        config = self.config
        ext = self._external_currents

        if config.oscillator_model == "kuramoto":

            def derivative_fn(states: torch.Tensor) -> torch.Tensor:
                d = kuramoto_derivatives(states, config)
                coupling = compute_kuramoto_coupling(states, graph, config.coupling_strength)
                d[:, 0] += coupling + ext[:, 0]
                return d

        elif config.oscillator_model == "fhn":

            def derivative_fn(states: torch.Tensor) -> torch.Tensor:
                d = fhn_derivatives(states, config)
                coupling = compute_fhn_coupling(states, graph, config.coupling_strength)
                d[:, 0] += coupling + ext[:, 0]
                return d

        else:
            raise ValueError(f"Unknown oscillator model: {config.oscillator_model}")

        return derivative_fn

    def _structural_prune(self) -> None:
        """Remove edges with strength below prune threshold."""
        strengths = self.graph.get_strength()
        weak = strengths < self.config.structural_prune_threshold
        if weak.any():
            self.graph.remove_edges(weak)
