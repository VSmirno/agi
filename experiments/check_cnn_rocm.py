"""Diagnostic: can CNNEncoder run on AMD ROCm GPU?

Tests:
1. GPU availability
2. Simple Conv2d forward on GPU
3. Full CNNEncoder forward on GPU
4. CNNEncoder backward (training step) on GPU
5. Training throughput on GPU
"""
from __future__ import annotations

import traceback

import torch
import torch.nn as nn


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Test 1: simple Conv2d
    print("\n=== Test 1: Simple Conv2d on GPU ===")
    try:
        conv = nn.Conv2d(3, 32, 3, padding=1).to(device)
        x = torch.randn(4, 3, 64, 64, device=device)
        y = conv(x)
        print(f"OK: input {list(x.shape)} output {list(y.shape)}")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()

    # Test 2: BatchNorm2d + Conv2d
    print("\n=== Test 2: BatchNorm2d + Conv2d chain ===")
    try:
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ).to(device)
        x = torch.randn(4, 3, 64, 64, device=device)
        y = model(x)
        print(f"OK: input {list(x.shape)} output {list(y.shape)}")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()

    # Test 3: Full CNNEncoder forward
    print("\n=== Test 3: CNNEncoder forward on GPU ===")
    try:
        from snks.encoder.cnn_encoder import CNNEncoder
        encoder = CNNEncoder().to(device)
        x = torch.randn(8, 3, 64, 64, device=device)
        out = encoder(x)
        print(f"OK: z_real={list(out.z_real.shape)}, near_logits={list(out.near_logits.shape)}")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()

    # Test 4: CNNEncoder backward
    print("\n=== Test 4: CNNEncoder backward on GPU ===")
    try:
        from snks.encoder.cnn_encoder import CNNEncoder
        encoder = CNNEncoder().to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        x = torch.randn(8, 3, 64, 64, device=device)
        out = encoder(x)
        loss = out.z_real.mean()
        loss.backward()
        optimizer.step()
        print(f"OK: loss={loss.item():.4f}")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()

    # Test 5: Training throughput
    print("\n=== Test 5: Training throughput GPU vs CPU ===")
    import time
    for dev_name in ["cuda", "cpu"]:
        dev = torch.device(dev_name)
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        try:
            from snks.encoder.cnn_encoder import CNNEncoder
            enc = CNNEncoder().to(dev)
            opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
            N = 10
            t0 = time.time()
            for _ in range(N):
                xb = torch.randn(256, 3, 64, 64, device=dev)
                out = enc(xb)
                out.z_real.mean().backward()
                opt.step()
                opt.zero_grad()
            elapsed = time.time() - t0
            print(f"  {dev_name}: {N} steps x256 in {elapsed:.1f}s = {N*256/elapsed:.0f} samples/sec")
        except Exception as e:
            print(f"  {dev_name} FAIL: {e}")
            traceback.print_exc()

    print("\n=== Done ===")
