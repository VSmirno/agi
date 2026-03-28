"""DafEngine — orchestrates oscillator dynamics, coupling, integration, STDP, and homeostasis."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from snks.daf.compiled_step import integrate_fhn_compiled, make_compiled_integrate
from snks.daf.coupling import (
    build_coupling_csr,
    compute_fhn_coupling,
    compute_fhn_coupling_inplace,
    compute_kuramoto_coupling,
    update_coupling_csr_values,
)
from snks.daf.graph import SparseDafGraph
from snks.daf.homeostasis import Homeostasis
from snks.daf.integrator import DerivativeFn, euler_maruyama_step, euler_maruyama_step_fhn, integrate_n_steps
from snks.daf.oscillator import fhn_derivatives, fhn_derivatives_inplace, init_states, kuramoto_derivatives
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

    Uses CUDA Graphs on GPU for zero kernel-launch-overhead integration.
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

        # Pre-allocated buffers for zero-allocation hot loop
        E = self.graph.num_edges
        self._drift_buf = torch.zeros(N, config.state_dim, device=self.device)
        self._coupling_buf = torch.zeros(N, device=self.device)
        self._contrib_buf = torch.empty(E, device=self.device)
        self._src_v_buf = torch.empty(E, device=self.device)
        self._dst_v_buf = torch.empty(E, device=self.device)
        self._noise_buf = torch.empty(N, config.state_dim, device=self.device)
        self._noise_v = torch.empty(N, device=self.device)   # FHN: noise for v channel
        self._noise_w = torch.empty(N, device=self.device)   # FHN: noise for w channel
        self._edge_sign = (1.0 - 2.0 * self.graph.edge_attr[:, 3]).contiguous()

        # CSR sparse matrix for fast spmv coupling (cuSPARSE)
        # Skip on AMD ROCm for large N: torch.sparse_csr_tensor triggers slow HIP compilation
        if not config.disable_csr:
            self._coupling_csr, self._coupling_degree = build_coupling_csr(self.graph)
        else:
            self._coupling_csr = None
            self._coupling_degree = None
        self._spmv_out = torch.empty(N, device=self.device)
        self._spmv_failed: bool = config.disable_csr  # fallback to scatter_add if hipSPARSE unsupported

        # CUDA Graph state — disabled on HIP/ROCm (capture corrupts GPU context)
        is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_failed: bool = is_hip
        self._graph_n_steps: int = 0
        self._graph_fired_history: torch.Tensor | None = None

        # Compiled step: chunk-based FHN kernel reduces HIP dispatch overhead.
        # make_compiled_integrate() compiles a CHUNK_SIZE-step function; engine
        # calls it n_steps // chunk_size times per cycle.
        self._compiled_step_fn = None
        self._compiled_chunk_size: int = 0
        if config.oscillator_model == "fhn" and self.device.type == "cuda":
            from snks.daf.compiled_step import _COMPILE_CHUNK
            self._compiled_step_fn = make_compiled_integrate()
            self._compiled_chunk_size = _COMPILE_CHUNK if self._compiled_step_fn is not None else 0
        # Pre-computed edge weights for compiled path: sign * strength
        self._edge_weight = (self._edge_sign * self.graph.edge_attr[:, 0]).contiguous()

        # Learning components
        self.enable_learning = enable_learning
        self.stdp = STDP(config)
        self.homeostasis = Homeostasis(config, N, self.device)

    @property
    def num_nodes(self) -> int:
        """Number of oscillator nodes in the network."""
        return self.config.num_nodes

    def inject_external_currents(self, node_ids: list[int], value: float = 1.0) -> None:
        """Inject external current to specified nodes before next step.

        Writes to _external_currents[:, 0] (I_ext channel), NOT to states (voltage).
        Effect is transient — _external_currents is reset after each step() call.
        Used by ReplayEngine for approximate SKS re-activation.
        """
        valid = [nid for nid in node_ids if 0 <= nid < self.config.num_nodes]
        if valid:
            self._external_currents[valid, 0] += value

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
        use_compiled = (
            self._compiled_step_fn is not None
            and self.config.oscillator_model == "fhn"
        )
        use_graph = (
            self.device.type == "cuda"
            and self.config.oscillator_model == "fhn"
            and n_steps >= 10
            and not self._cuda_graph_failed
            and not use_compiled  # prefer compiled over CUDA Graphs
        )

        if use_compiled:
            self.states, fired_history = integrate_fhn_compiled(
                self.states,
                self._external_currents,
                self.graph.edge_index[0],
                self.graph.edge_index[1],
                self._edge_weight,
                n_steps=n_steps,
                K=self.config.coupling_strength,
                I_base=self.config.fhn_I_base,
                a=self.config.fhn_a,
                b=self.config.fhn_b,
                tau=self.config.fhn_tau,
                dt=self.config.dt,
                noise_sigma=self.config.noise_sigma,
                step_fn=self._compiled_step_fn,
                chunk_size=self._compiled_chunk_size,
            )
        elif use_graph:
            try:
                fired_history = self._step_cuda_graph(n_steps)
            except RuntimeError:
                # CUDA Graphs not supported (e.g. some AMD/HIP GPUs)
                self._cuda_graph_failed = True
                self._cuda_graph = None
                use_graph = False

        if not use_compiled and not use_graph:
            derivative_fn = self._make_derivative_fn()
            spike_mode = "phase_crossing" if self.config.oscillator_model == "kuramoto" else "threshold"
            self.states, fired_history = integrate_n_steps(
                self.states,
                derivative_fn,
                n_steps=n_steps,
                dt=self.config.dt,
                noise_sigma=self.config.noise_sigma,
                spike_mode=spike_mode,
                noise_buf=self._noise_buf,
            )

        self._last_fired_history = fired_history
        self.step_count += n_steps

        # Apply learning
        stdp_result = None
        if self.enable_learning:
            stdp_result = self.stdp.apply(self.graph, fired_history)
            self.homeostasis.update(fired_history, self.states)

            # Update CSR values and edge weights after STDP weight changes
            if self.config.oscillator_model == "fhn":
                if not self.config.disable_csr:
                    update_coupling_csr_values(
                        self._coupling_csr, self._coupling_degree, self.graph
                    )
                self._edge_weight = (self._edge_sign * self.graph.edge_attr[:, 0]).contiguous()

            # Structural plasticity: prune weak edges periodically
            if self.step_count % self.config.structural_interval == 0:
                self._structural_prune()

        # Reset external currents: transient per step
        self._external_currents.zero_()

        return StepResult(
            states=self.states.clone(),
            fired_history=fired_history,
            prediction_error=torch.zeros(self.config.num_nodes, device=self.device),
            mean_pe=0.0,
            n_spikes=-1,  # lazy: call fired_history.sum().item() if needed
            elapsed_model_time=n_steps * self.config.dt,
            stdp_result=stdp_result,
        )

    def _step_cuda_graph(self, n_steps: int) -> torch.Tensor:
        """Run integration via CUDA Graph capture/replay."""
        first_capture = (self._cuda_graph is None or self._graph_n_steps != n_steps)

        if first_capture:
            self._capture_cuda_graph(n_steps)
            # Capture already executed the loop — result is in states and fired_history
        else:
            self._cuda_graph.replay()

        return self._graph_fired_history

    def _capture_cuda_graph(self, n_steps: int) -> None:
        """Capture integration loop as CUDA Graph for zero-overhead replay."""
        N = self.config.num_nodes
        dt = self.config.dt
        noise_sigma = self.config.noise_sigma

        # Pre-allocate fired_history for the graph
        self._graph_fired_history = torch.empty(
            n_steps, N, dtype=torch.bool, device=self.device
        )

        derivative_fn = self._make_derivative_fn()
        step_fn = euler_maruyama_step_fhn
        step_args = (derivative_fn, dt, noise_sigma, self._noise_v, self._noise_w)

        # Warmup on side stream to initialize CUDA internal state
        states_backup = self.states.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step_fn(self.states, *step_args)
        torch.cuda.current_stream().wait_stream(s)
        self.states.copy_(states_backup)

        # Capture the integration loop
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            for t in range(n_steps):
                step_fn(self.states, *step_args)
                self._graph_fired_history[t] = self.states[:, 0] > 0.5

        self._graph_n_steps = n_steps
        # After capture, states contain the result of the captured execution.
        # This is the correct result for the first call.

    def get_states(self) -> torch.Tensor:
        """Return (N, 8) current states on device."""
        return self.states

    def get_fired_history(self) -> torch.Tensor | None:
        """Return (T, N) bool from last step() call, or None."""
        return self._last_fired_history

    def _make_derivative_fn(self) -> DerivativeFn:
        """Build derivative closure capturing graph, config, external currents, and buffers."""
        graph = self.graph
        config = self.config
        ext = self._external_currents

        # Capture pre-allocated buffers for zero-allocation hot path
        drift_buf = self._drift_buf
        coupling_buf = self._coupling_buf

        if config.oscillator_model == "kuramoto":

            def derivative_fn(states: torch.Tensor) -> torch.Tensor:
                d = kuramoto_derivatives(states, config)
                coupling = compute_kuramoto_coupling(states, graph, config.coupling_strength)
                d[:, 0] += coupling + ext[:, 0]
                return d

        elif config.oscillator_model == "fhn":
            A_csr = self._coupling_csr
            degree = self._coupling_degree
            K = config.coupling_strength
            tmp_buf = self._spmv_out  # (N,) temp for v*degree
            # scatter_add buffers for fallback path
            contrib_buf = self._contrib_buf
            src_v_buf = self._src_v_buf
            dst_v_buf = self._dst_v_buf
            edge_sign = self._edge_sign
            engine = self  # capture for _spmv_failed flag

            def derivative_fn(states: torch.Tensor) -> torch.Tensor:
                fhn_derivatives_inplace(states, config, drift_buf)
                if not engine._spmv_failed:
                    try:
                        v = states[:, 0]
                        torch.mv(A_csr, v, out=coupling_buf)
                        torch.mul(v, degree, out=tmp_buf)
                        coupling_buf.sub_(tmp_buf).mul_(K)
                    except RuntimeError:
                        engine._spmv_failed = True
                if engine._spmv_failed:
                    compute_fhn_coupling_inplace(
                        states, graph, K, coupling_buf,
                        contrib_buf, src_v_buf, dst_v_buf, edge_sign,
                    )
                drift_buf[:, 0].add_(coupling_buf).add_(ext[:, 0])
                return drift_buf

        else:
            raise ValueError(f"Unknown oscillator model: {config.oscillator_model}")

        return derivative_fn

    def _structural_prune(self) -> None:
        """Remove edges with strength below prune threshold."""
        strengths = self.graph.get_strength()
        weak = strengths < self.config.structural_prune_threshold
        if weak.any():
            self.graph.remove_edges(weak)
            self._reallocate_edge_buffers()
            self._cuda_graph = None  # invalidate — topology changed

    def _reallocate_edge_buffers(self) -> None:
        """Reallocate edge-sized buffers after graph topology changes."""
        E = self.graph.num_edges
        self._contrib_buf = torch.empty(E, device=self.device)
        self._src_v_buf = torch.empty(E, device=self.device)
        self._dst_v_buf = torch.empty(E, device=self.device)
        self._edge_sign = (1.0 - 2.0 * self.graph.edge_attr[:, 3]).contiguous()
        self._edge_weight = (self._edge_sign * self.graph.edge_attr[:, 0]).contiguous()
        # Rebuild CSR (structure changed)
        self._coupling_csr, self._coupling_degree = build_coupling_csr(self.graph)
