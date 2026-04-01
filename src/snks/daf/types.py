"""Core types and configuration for the DAF engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ZoneConfig:
    """A contiguous slice of DAF nodes belonging to one sensory zone."""
    start: int
    size: int


@dataclass
class DafConfig:
    """Configuration for the Dynamic Attractor Fields engine."""

    # Network topology
    num_nodes: int = 50_000
    state_dim: int = 8          # per-oscillator state: [v, amplitude, freq, threshold, w_recovery, adapt, aux1, aux2]
    avg_degree: int = 50
    zones: dict[str, ZoneConfig] | None = None   # Stage 19: zonal DAF (None = legacy flat mode)
    inter_zone_avg_degree: int = 10              # Stage 19: inter-zone edge density

    # Integration
    dt: float = 0.0001          # 0.1 ms model time
    noise_sigma: float = 0.01

    # Oscillator model: "fhn" (FitzHugh-Nagumo) or "kuramoto"
    oscillator_model: str = "fhn"

    # Kuramoto parameters
    omega_std: float = 1.0  # natural frequency spread: N(0, omega_std)

    # FitzHugh-Nagumo parameters
    fhn_a: float = 0.7
    fhn_b: float = 0.8
    fhn_tau: float = 12.5       # 1/0.08 — recovery time constant
    fhn_I_base: float = 0.5     # base intrinsic current

    # Coupling
    coupling_strength: float = 0.1

    # STDP
    stdp_mode: str = "timing"  # "timing" (FHN) or "rate" (Kuramoto rate-based Hebbian)
    stdp_a_plus: float = 0.01
    stdp_a_minus: float = 0.012
    stdp_tau_plus: float = 0.020    # 20 ms
    stdp_tau_minus: float = 0.020
    stdp_w_max: float = 1.0
    stdp_w_min: float = 0.0
    stdp_w_target: float = 0.5         # homeostatic weight target for STDP

    # Homeostasis
    homeostasis_target: float = 0.05    # target firing rate
    homeostasis_tau: float = 1.0        # adaptation time constant
    homeostasis_lambda: float = 0.001   # regularization strength

    # Structural plasticity
    structural_add_threshold: float = 0.8   # co-activation threshold to add edge
    structural_prune_threshold: float = 0.01  # weight below which to prune
    structural_interval: int = 1000          # steps between structural updates

    # Stage 43: Working Memory
    wm_fraction: float = 0.0   # fraction of nodes in WM zone (0 = disabled, 0.2 = 20%)
    wm_decay: float = 0.95     # per-cycle relaxation toward resting state

    # Device
    device: str = "auto"

    # Performance
    disable_csr: bool = False  # skip torch.sparse_csr_tensor (slow on AMD ROCm for large N)


@dataclass
class EncoderConfig:
    """Configuration for the visual encoder."""
    image_size: int = 64
    n_orientations: int = 8
    n_scales: int = 4
    n_phases: int = 4
    sdr_size: int = 4096
    sdr_sparsity: float = 0.04     # ~164 active bits
    gabor_kernel_size: int = 19    # max kernel size (all kernels padded to this)
    sdr_current_strength: float = 1.0  # current injected into DAF nodes per active SDR bit
    pool_h: int = 4                # AdaptiveAvgPool2d height: n_filters * pool_h * pool_w = sdr_size
    pool_w: int = 8                # AdaptiveAvgPool2d width
    # --- Stage 40: Hebbian encoder ---
    hebbian: bool = False          # use HebbianEncoder instead of frozen VisualEncoder
    hebbian_lr: float = 0.001      # Oja rule learning rate
    hebbian_update_interval: int = 5  # update weights every N perception cycles


@dataclass
class SKSConfig:
    """Configuration for SKS detection."""
    top_k: int = 5000
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 10
    min_cluster_size: int = 10
    coherence_mode: str = "auto"  # "phase" | "cofiring" | "auto"


@dataclass
class PredictionConfig:
    """Configuration for prediction engine."""
    pe_alpha: float = 1.0
    causal_window: int = 5
    causal_min_confidence: float = 0.3
    causal_decay: float = 0.95


@dataclass
class SKSEmbedConfig:
    """Configuration for SKS embedding (Stage 9)."""
    hac_dim: int = 2048  # совпадает с DcamConfig.hac_dim


@dataclass
class HACPredictionConfig:
    """Configuration for HAC associative memory predictor (Stage 9).

    Two backends:
        use_episodic_buffer=False (default): HACPredictionEngine — single bundle
            with exponential decay. Simple, O(1) memory. Degrades after ~20 steps.
        use_episodic_buffer=True: EpisodicHACPredictor — K-pair deque. Stable
            prediction quality over long episodes. O(K) lookup, K ≤ episodic_capacity.
    """
    memory_decay: float = 0.95          # bundle backend: exponential decay factor
    enabled: bool = True
    use_episodic_buffer: bool = False   # Stage 15: use EpisodicHACPredictor
    episodic_capacity: int = 32         # Stage 15: max pairs in episodic buffer


@dataclass
class GWSConfig:
    enabled: bool = True
    w_size: float = 1.0
    w_coherence: float = 0.0    # зарезервировано
    w_pred: float = 0.0         # зарезервировано


@dataclass
class MetacogConfig:
    enabled: bool = True
    alpha: float = 1/3          # вес dominance
    beta: float = 1/3           # вес stability
    gamma: float = 1/3          # вес (1 - pred_error_norm)
    delta: float = 0.0          # вес (1 - meta_pe), Stage 10. 0 = backward compatible
    policy: str = "null"        # "null" | "noise" | "stdp" | "broadcast"
    policy_strength: float = 1.0
    broadcast_threshold: float = 0.6  # confidence threshold for BroadcastPolicy


@dataclass
class DcamConfig:
    """Configuration for DCAM storage."""
    hac_dim: int = 2048
    lsh_n_tables: int = 32
    lsh_n_bits: int = 16
    max_nodes: int = 100_000
    episodic_capacity: int = 10_000
    consolidation_interval: int = 100  # perception cycles between consolidations
    consolidation_stc_threshold: float = 0.5   # importance threshold for STC tagging
    consolidation_coact_min: int = 3            # min co-activations to strengthen edge


@dataclass
class HierarchicalConfig:
    """Configuration for Stage 10: Hierarchical Prediction (L2 meta-embedding)."""
    enabled: bool = True
    meta_decay: float = 0.8      # EWA decay ≈ 5-cycle effective window
    memory_decay: float = 0.95   # L2 predictor memory decay


@dataclass
class CostState:
    """Intrinsic cost breakdown (Stage 12)."""
    total: float           # итоговая стоимость ∈ [0, 1]. Высокая = плохо.
    homeostatic: float     # отклонение от target firing rate ∈ [0, 1]
    epistemic_value: float # информационная ценность (PE) ∈ [0, 1]. Высокая = любопытно.
    goal: float            # задачная стоимость ∈ [0, 1].


@dataclass
class CostModuleConfig:
    """Configuration for Stage 12: Intrinsic Cost Module."""
    enabled: bool = True
    w_homeostatic: float = 0.3
    w_epistemic: float = 0.4
    w_goal: float = 0.3
    # None → взять из DafConfig.homeostasis_target при инициализации Pipeline
    firing_rate_target: float | None = None


@dataclass
class ConfiguratorConfig:
    """Configuration for Stage 13: Configurator FSM."""
    enabled: bool = True
    hysteresis_cycles: int = 8           # минимум циклов в режиме для смены
    max_explore_cycles: int = 32         # принудительный выход из EXPLORE (divergence guard)
    explore_cost_threshold: float = 0.65
    explore_epistemic_threshold: float = 0.45
    consolidate_cost_threshold: float = 0.35
    consolidate_stability_threshold: float = 0.70
    goal_cost_threshold: float = 0.10


@dataclass
class ConfiguratorAction:
    """Result of one Configurator.update() call."""
    mode: str                                    # текущий режим
    changed: dict[str, tuple[float, float]]      # param → (old, new)
    cycles_in_mode: int                          # сколько циклов в режиме


@dataclass
class StochasticPlanConfig:
    """Configuration for Stage 11: StochasticSimulator."""
    enabled: bool = False
    n_samples: int = 8
    temperature: float = 1.0


@dataclass
class PipelineConfig:
    """Top-level configuration."""
    daf: DafConfig = field(default_factory=DafConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dcam: DcamConfig = field(default_factory=DcamConfig)
    sks: SKSConfig = field(default_factory=SKSConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    gws: GWSConfig = field(default_factory=GWSConfig)
    metacog: MetacogConfig = field(default_factory=MetacogConfig)
    sks_embed: SKSEmbedConfig = field(default_factory=SKSEmbedConfig)
    hac_prediction: HACPredictionConfig = field(default_factory=HACPredictionConfig)
    steps_per_cycle: int = 100      # integration steps per perception cycle
    device: str = "auto"
    # --- Stage 19: cross-modal priming ---
    priming_strength: float = 0.3   # fraction of full SDR current for top-down priming
    # --- Stage 10 ---
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    # --- Stage 12 ---
    cost_module: CostModuleConfig = field(default_factory=CostModuleConfig)
    # --- Stage 13 ---
    configurator: ConfiguratorConfig = field(default_factory=ConfiguratorConfig)


@dataclass
class ConsolidationConfig:
    """Configuration for Stage 16: ConsolidationScheduler."""
    enabled: bool = False
    every_n: int = 10
    top_k: int = 50
    cold_threshold: float = 0.3
    node_threshold: float = 0.7
    save_path: str | None = None  # None = no save


@dataclass
class ReplayConfig:
    """Configuration for Stage 16: ReplayEngine."""
    enabled: bool = False
    top_k: int = 10
    n_steps: int = 50
    mode: str = "importance"  # "importance" | "recency" | "uniform"


@dataclass
class CausalAgentConfig:
    """Configuration for causal agent (Stage 6)."""
    # Pipeline (reuse from stage 3-5)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Environment
    grid_size: int = 8
    max_steps_per_episode: int = 200

    # Motor encoder
    motor_sdr_size: int = 512           # SDR bits per action
    motor_current_strength: float = 1.0

    # Causal model
    causal_min_observations: int = 3    # min obs before confident
    causal_confidence_threshold: float = 0.5
    causal_decay: float = 0.99
    causal_context_hash_bits: int = 64  # для быстрого поиска по контексту
    causal_context_bins: int = 64       # coarsen unstable DAF SKS IDs into N bins (stable perceptual-hash IDs kept as-is)

    # Motivation
    curiosity_epsilon: float = 0.2      # random exploration rate (or initial rate if decay used)
    curiosity_epsilon_min: float = 0.05  # minimum epsilon after decay
    curiosity_epsilon_horizon: int = 0   # decay horizon in steps (0 = no decay)
    curiosity_decay: float = 0.995      # novelty decay per visit

    # Mental simulation
    simulation_max_depth: int = 10
    simulation_min_confidence: float = 0.3

    # Stage 11: Stochastic planning
    stochastic_plan: StochasticPlanConfig = field(default_factory=StochasticPlanConfig)

    # Stage 15: DCAM episodic buffer (minimal integration)
    use_dcam_episodic: bool = False     # store transitions in EpisodicBuffer
    dcam: DcamConfig = field(default_factory=DcamConfig)
