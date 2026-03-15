#!/usr/bin/env python3
"""Quick inspect of latent cache without decoding."""

import argparse
import sys
import os

sys.path.insert(0, "/home/e/Dpipe/diffusion-trainers/musubi-tuner/src")

from safetensors.torch import safe_open


def main():
    parser = argparse.ArgumentParser(description="Quick inspect latent cache")
    parser.add_argument("latent_path", type=str, help="Path to latent cache file")
    args = parser.parse_args()

    print(f"Loading: {args.latent_path}")

    data = {}
    metadata = {}
    with safe_open(args.latent_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            data[key] = f.get_tensor(key)

    print("\n=== Metadata ===")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\n=== Tensor Info ===")
    for key, tensor in data.items():
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
        if len(tensor.shape) >= 3:
            # Calculate pixel dimensions for latents (VAE divides by 32)
            h_latent, w_latent = tensor.shape[-2], tensor.shape[-1]
            h_pixel = h_latent * 32
            w_pixel = w_latent * 32
            print(f"  → Pixel resolution: {h_pixel}x{w_pixel}")


if __name__ == "__main__":
    main()
