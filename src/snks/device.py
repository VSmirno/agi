"""GPU device auto-detection. Works with CUDA (NVIDIA), ROCm (AMD), and CPU."""

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Auto-detect best available device.

    Args:
        prefer: "auto" (GPU if available), "cuda", "cpu"

    Returns:
        torch.device for computation.
        PyTorch uses the same torch.cuda API for both NVIDIA CUDA and AMD ROCm.
    """
    if prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"[SNKS] GPU: {props.name}, {vram_gb:.1f} GB VRAM")
        return dev

    print("[SNKS] No GPU found, using CPU")
    return torch.device("cpu")


def device_info() -> dict:
    """Return detailed device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": props.name,
            "vram_gb": round(props.total_memory / (1024 ** 3), 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })
    return info
