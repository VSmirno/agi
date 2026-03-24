"""Core types and configuration for the DAF engine."""

from dataclasses import dataclass, field


@dataclass
class DafConfig:
    """Configuration for the Dynamic Attractor Fields engine."""

    # Network topology
    num_nodes: int = 50_000
    state_dim: int = 8          # per-oscillator state: [v, amplitude, freq, threshold, w_recovery, adapt, aux1, aux2]
    avg_degree: int = 50

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

    # Device
    device: str = "auto"


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
class PipelineConfig:
    """Top-level configuration."""
    daf: DafConfig = field(default_factory=DafConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dcam: DcamConfig = field(default_factory=DcamConfig)
    sks: SKSConfig = field(default_factory=SKSConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    steps_per_cycle: int = 100      # integration steps per perception cycle
    device: str = "auto"


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
    causal_context_bins: int = 16       # coarsen SKS IDs into N bins for generalization

    # Motivation
    curiosity_epsilon: float = 0.2      # random exploration rate
    curiosity_decay: float = 0.995      # novelty decay per visit

    # Mental simulation
    simulation_max_depth: int = 10
    simulation_min_confidence: float = 0.3
