#!/usr/bin/env python3
"""Diagnose GPU availability and capabilities."""

import sys

def main():
    print("=" * 60)
    print("SNKS GPU Diagnostics")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available:  {torch.cuda.is_available()}")
        print(f"CUDA version:    {torch.version.cuda or 'N/A (ROCm?)'}")

        if hasattr(torch.version, 'hip'):
            print(f"HIP version:     {torch.version.hip or 'N/A'}")

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"GPU count:       {n}")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                vram = props.total_mem / (1024 ** 3)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    VRAM:              {vram:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Multi-processors:   {props.multi_processor_count}")

            # Quick benchmark
            print("\n--- Quick benchmark ---")
            device = torch.device("cuda")
            N = 50000
            x = torch.randn(N, 8, device=device)
            # Warmup
            for _ in range(10):
                y = x * 2 + torch.randn_like(x)
            torch.cuda.synchronize()

            import time
            start = time.perf_counter()
            iters = 10000
            for _ in range(iters):
                y = x * 2 + torch.randn_like(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f"  {iters} element-wise ops on ({N}, 8): {elapsed:.3f}s")
            print(f"  ~{iters / elapsed:.0f} ops/sec")
        else:
            print("\nNo GPU detected. Running on CPU.")
            print("For AMD ROCm: install pytorch-rocm")
            print("For NVIDIA:   install pytorch with CUDA")

    except ImportError:
        print("ERROR: PyTorch not installed!")
        print("  pip install torch")
        sys.exit(1)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
