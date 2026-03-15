#!/usr/bin/env python3
"""Decode reference latent cache to verify what's actually stored."""

import argparse
import sys
import os

# Add musubi-tuner to path
sys.path.insert(0, "/home/e/Dpipe/diffusion-trainers/musubi-tuner/src")

import torch
import numpy as np
from safetensors.torch import safe_open
from PIL import Image
import av

from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder


def load_latent_cache(path: str) -> dict:
    """Load latent cache file and return metadata + latent tensor."""
    data = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            data[key] = f.get_tensor(key)
    return data, metadata


def decode_latent_to_video(vae, latent: torch.Tensor, fps: float = 25, output_path: str = "decoded_ref.mp4"):
    """Decode latent tensor to video file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = vae.to(device)
    latent = latent.to(device)
    vae.eval()

    with torch.no_grad():
        # latent shape from cache: [C, F, H, W] or other formats
        # VideoDecoder expects: [B, C, F, H, W] where C=128 for LTX
        print(f"Input latent shape: {latent.shape}")

        # LTX VideoDecoder expects C=128 channels
        # Shape [128, 10, 11, 23] likely means [C, F, H, W] = [128, 10, 11, 23]
        if len(latent.shape) == 4:
            C, F, H, W = latent.shape
            latent = latent.unsqueeze(0)  # [1, C, F, H, W]
            print(f"Reshaped to decoder format [B, C, F, H, W]: {latent.shape}")

        # Decode - VideoDecoder expects [B, C, F, H, W]
        decoded = vae(latent)  # [B, 3, F', H', W']
        decoded = decoded.squeeze(0)  # [3, F', H', W']
        decoded = decoded.permute(1, 2, 3, 0)  # [F', H', W', C]

    # Convert to uint8 on CPU
    decoded = decoded.cpu()
    decoded = decoded.float()  # Convert bfloat16 to float32 for numpy
    decoded = (decoded * 127.5 + 127.5).clamp(0, 255).numpy().astype(np.uint8)

    # Save as video
    height, width = decoded.shape[1], decoded.shape[2]
    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for frame in decoded:
        img = Image.fromarray(frame)
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"Saved decoded video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Decode reference latent cache for inspection")
    parser.add_argument("latent_path", type=str, help="Path to reference latent cache file")
    parser.add_argument("--vae", type=str, required=True, help="Path to LTX2 checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: same dir as latent, .mp4 extension)")
    parser.add_argument("--fps", type=float, default=25, help="Output video FPS")
    args = parser.parse_args()

    # Default output: save next to latent file with .mp4 extension
    if args.output is None:
        latent_dir = os.path.dirname(args.latent_path)
        latent_basename = os.path.splitext(os.path.basename(args.latent_path))[0]
        args.output = os.path.join(latent_dir, f"{latent_basename}.mp4")

    print(f"Loading latent cache: {args.latent_path}")
    data, metadata = load_latent_cache(args.latent_path)

    # Print info
    print("\n=== Latent Cache Info ===")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Find the latent tensor (key format varies: latents, latents_CxHxW_dtype, etc.)
    latent_key = None
    for key in data.keys():
        if key.startswith("latents"):
            latent_key = key
            break

    if latent_key is None:
        print("Available keys:")
        for key in data.keys():
            print(f"  {key}: {data[key].shape}")
        raise ValueError("No latent tensor found in cache file")

    latent = data[latent_key]
    print(f"\nLatent shape: {latent.shape}")
    print(f"  Frames: {latent.shape[0]}")
    print(f"  Channels: {latent.shape[1]}")
    print(f"  Height: {latent.shape[2]}")
    print(f"  Width: {latent.shape[3]}")

    # Calculate pixel resolution (VAE divides by 32)
    pixel_h = latent.shape[2] * 32
    pixel_w = latent.shape[3] * 32
    print(f"\nPixel resolution: {pixel_h}x{pixel_w}")

    # Load VAE decoder
    print(f"\nLoading VAE decoder from: {args.vae}")
    from musubi_tuner.ltx_2.model.video_vae import VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER

    vae = SingleGPUModelBuilder(
        model_path=str(args.vae),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
    vae.eval()

    # Decode
    print("\nDecoding latent to video...")
    decode_latent_to_video(vae, latent, fps=args.fps, output_path=args.output)


if __name__ == "__main__":
    main()
