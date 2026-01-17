#!/usr/bin/env python3
"""
Standalone test for LTX-2 VAE encode/decode pipeline.

This script tests:
1. Loading an image
2. Encoding it to latents with LTX-2 VAE encoder
3. Decoding it back with LTX-2 VAE decoder
4. Saving the result for visual inspection

Run: python test_ltx2_vae_encode_decode.py /path/to/image.png
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

# Add musubi-tuner to path
script_dir = Path(__file__).parent.resolve()
musubi_path = script_dir / "diffusion-trainers" / "musubi-tuner" / "src"
sys.path.insert(0, str(musubi_path))

# Also add the parent musubi-tuner directory for imports
sys.path.insert(0, str(script_dir / "diffusion-trainers" / "musubi-tuner"))


def load_vae(vae_path: str, device: torch.device, dtype: torch.dtype):
    """Load LTX-2 VAE encoder and decoder."""
    print(f"Loading VAE from {vae_path}...")

    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.model.video_vae.model_configurator import (
        VideoEncoderConfigurator,
        VideoDecoderConfigurator,
        VAE_ENCODER_COMFY_KEYS_FILTER,
        VAE_DECODER_COMFY_KEYS_FILTER,
    )

    # CRITICAL: Use bfloat16 for both encoder and decoder to match ltx-trainer
    # ltx-trainer uses float32 input + bfloat16 autocast for encoding
    vae_dtype = torch.bfloat16

    # Load encoder
    encoder = SingleGPUModelBuilder(
        model_path=str(vae_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=vae_dtype)
    encoder.eval()
    encoder.requires_grad_(False)
    print(f"Loaded VAE encoder (dtype={vae_dtype})")

    # Load decoder
    decoder = SingleGPUModelBuilder(
        model_path=str(vae_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=vae_dtype)
    decoder.eval()
    decoder.requires_grad_(False)
    print(f"Loaded VAE decoder (dtype={vae_dtype})")

    return encoder, decoder


def load_and_preprocess_image(image_path: str, max_side: int) -> tuple[torch.Tensor, int, int]:
    """
    Load and preprocess image for VAE encoding.

    Logic:
    1. Find the biggest side of the image
    2. Scale so biggest side = max_side (scale down if needed)
    3. Get the smallest side after scaling
    4. Find closest number LOWER than it that is divisible by 32
    5. Resize to that size (center crop if needed)

    Returns:
        (image_tensor, final_width, final_height) - Preprocessed tensor and final dimensions
    """
    print(f"Loading image from {image_path}...")

    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size
    print(f"Original image size: {orig_width}x{orig_height}")

    # Step 1: Find the biggest side
    biggest_side = max(orig_width, orig_height)
    print(f"Biggest side: {biggest_side}")

    # Step 2: Scale so biggest side = max_side (only if image is larger)
    if biggest_side > max_side:
        scale_factor = max_side / biggest_side
        print(f"Scaling down by factor: {scale_factor:.4f}")
        new_width = int(round(orig_width * scale_factor))
        new_height = int(round(orig_height * scale_factor))
    else:
        # Image is smaller than max_side, keep original size
        new_width = orig_width
        new_height = orig_height
        print(f"Image is smaller than {max_side}, keeping original size")

    print(f"After scaling: {new_width}x{new_height}")

    # Step 3: Get the smallest side after scaling
    smallest_side = min(new_width, new_height)
    print(f"Smallest side after scaling: {smallest_side}")

    # Step 4: Find closest number LOWER than smallest_side that is divisible by 32
    # Round down to nearest multiple of 32
    final_smallest_side = (smallest_side // 32) * 32
    # Ensure at least 32
    final_smallest_side = max(32, final_smallest_side)
    print(f"Final smallest side (divisible by 32): {final_smallest_side}")

    # Step 5: Calculate final dimensions
    if new_width > new_height:
        # Width is the bigger side
        final_width = max_side
        final_height = final_smallest_side
    else:
        # Height is the bigger side
        final_width = final_smallest_side
        final_height = max_side

    print(f"Final target size: {final_width}x{final_height}")

    # Convert to tensor and normalize to [0, 1]
    to_tensor = T.ToTensor()
    image_tensor = to_tensor(img)  # (C, H, W)
    print(f"Tensor shape: {image_tensor.shape}, range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

    # Resize to target size if needed
    current_height, current_width = image_tensor.shape[1:]

    if current_width != final_width or current_height != final_height:
        aspect_ratio = current_width / current_height
        target_aspect_ratio = final_width / final_height

        if aspect_ratio > target_aspect_ratio:
            # Image is wider - resize to match height, crop width
            resize_height = final_height
            resize_width = int(final_height * aspect_ratio)
        else:
            # Image is taller - resize to match width, crop height
            resize_height = int(final_width / aspect_ratio)
            resize_width = final_width

        print(f"Resizing to {resize_width}x{resize_height} (maintaining aspect ratio)")
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(resize_height, resize_width), mode="bilinear", align_corners=False
        )

        # Center crop to target dimensions
        h_start = (resize_height - final_height) // 2
        w_start = (resize_width - final_width) // 2
        image_tensor = image_tensor[:, :, h_start:h_start + final_height, w_start:w_start + final_width]
        print(f"Cropped to {final_width}x{final_height}")
    else:
        image_tensor = image_tensor.unsqueeze(0)

    # Add frame dimension and convert to [-1, 1]
    image_tensor = image_tensor.unsqueeze(2)  # (1, C, 1, H, W)
    image_tensor = (image_tensor * 2.0 - 1.0)
    print(f"Final tensor shape: {image_tensor.shape}, range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

    return image_tensor, final_width, final_height


def save_original_image(image_tensor: torch.Tensor, output_path: str):
    """Save the preprocessed image tensor as PNG."""
    import torchvision.transforms as T

    # Remove batch and frame dimensions: (1, C, 1, H, W) -> (C, H, W)
    # Use squeeze() without arguments to remove all size-1 dimensions
    img_tensor = image_tensor.squeeze().cpu()

    # Convert from [-1, 1] to [0, 1]
    img_tensor = ((img_tensor + 1.0) / 2.0).clamp(0.0, 1.0)

    # Convert to PIL Image
    to_pil = T.ToPILImage()
    pil_image = to_pil(img_tensor)

    pil_image.save(output_path)
    print(f"Saved preprocessed original to {output_path}")


def encode_image(image_tensor: torch.Tensor, encoder, device: torch.device, dtype: torch.dtype):
    """Encode image to latents using ltx-trainer approach (float32 input, bfloat16 autocast)."""
    print("\n=== ENCODING ===")
    print(f"Input shape: {image_tensor.shape}")

    # CRITICAL: Use float32 for VAE input (matching ltx-trainer)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # CRITICAL: Use bfloat16 autocast for encoding (matching ltx-trainer)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            encoded = encoder(image_tensor)

    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded dtype: {encoded.dtype}")
    print(f"Encoded range: [{encoded.min():.3f}, {encoded.max():.3f}]")
    print(f"Encoded mean: {encoded.mean():.3f}, std: {encoded.std():.3f}")

    # Check if latents are properly scaled
    if encoded.std() < 0.2 or encoded.abs().mean() < 0.15:
        print("WARNING: Encoded latents appear UNSCALED (std < 0.2). This will cause issues!")
    else:
        print("OK: Encoded latents appear properly scaled")

    return encoded


def decode_latents(latents: torch.Tensor, decoder, device: torch.device, dtype: torch.dtype):
    """Decode latents back to image."""
    print("\n=== DECODING ===")
    print(f"Input shape: {latents.shape}")
    print(f"Input dtype: {latents.dtype}")

    # CRITICAL: Keep latents in their encoded dtype (bfloat16), don't convert to float16
    latents = latents.to(device=device)

    with torch.no_grad():
        decoded = decoder(latents)

    print(f"Decoded shape: {decoded.shape}")
    print(f"Decoded dtype: {decoded.dtype}")
    print(f"Decoded range: [{decoded.min():.3f}, {decoded.max():.3f}]")

    # Convert to [0, 1]
    decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    print(f"Normalized range: [{decoded.min():.3f}, {decoded.max():.3f}]")

    return decoded


def save_images(original_tensor: torch.Tensor, decoded_tensor: torch.Tensor, output_dir: str):
    """Save preprocessed original and decoded images for comparison."""
    os.makedirs(output_dir, exist_ok=True)

    # Save preprocessed original (the one that was encoded)
    original_path_out = os.path.join(output_dir, "00_original.png")
    save_original_image(original_tensor, original_path_out)

    # Convert decoded tensor to PIL
    # Handle different output formats
    if decoded_tensor.dim() == 4:
        # [1, 3, H, W] -> [3, H, W]
        decoded_image = decoded_tensor[0].cpu().float()
    elif decoded_tensor.dim() == 5:
        # [1, 3, 1, H, W] -> [3, H, W]
        decoded_image = decoded_tensor[0, :, 0].cpu().float()
    else:
        raise ValueError(f"Unexpected decoded shape: {decoded_tensor.shape}")

    to_pil = T.ToPILImage()
    pil_image = to_pil(decoded_image)

    decoded_path = os.path.join(output_dir, "01_decoded.png")
    pil_image.save(decoded_path)
    print(f"Saved decoded to {decoded_path}")

    # Save side-by-side comparison
    from PIL import ImageDraw

    # Load the saved original for comparison
    original = Image.open(original_path_out)

    # Resize original to match decoded if needed
    if original.size != pil_image.size:
        original = original.resize(pil_image.size)

    width, height = pil_image.size
    comparison = Image.new("RGB", (width * 2, height))
    comparison.paste(original, (0, 0))
    comparison.paste(pil_image, (width, 0))

    comparison_path = os.path.join(output_dir, "02_comparison.png")
    comparison.save(comparison_path)
    print(f"Saved comparison to {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Test LTX-2 VAE encode/decode pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--vae", default="/home/e/Dpipe/models/ltx2/ltx-2-19b-dev.safetensors",
                       help="Path to LTX-2 checkpoint")
    parser.add_argument("--max_side", type=int, default=640,
                       help="Maximum side length (biggest side will be scaled to this)")
    parser.add_argument("--output", default="/home/e/Dpipe/test_vae_output",
                       help="Output directory")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"],
                       help="VAE dtype")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Load VAE
    encoder, decoder = load_vae(args.vae, device, dtype)

    # Load and preprocess image
    image_tensor, final_width, final_height = load_and_preprocess_image(args.image, args.max_side)
    print(f"\nFinal dimensions: {final_width}x{final_height}")

    # Encode
    latents = encode_image(image_tensor, encoder, device, dtype)

    # Decode
    decoded = decode_latents(latents, decoder, device, dtype)

    # Save results
    save_images(image_tensor, decoded, args.output)

    print(f"\nDone! Check the output directory: {args.output}")
    print("Files generated:")
    print("  - 00_original.png: Preprocessed input image (what was encoded)")
    print("  - 01_decoded.png: Result after encode/decode (should match original)")
    print("  - 02_comparison.png: Side-by-side comparison")
    print("\nIf the decoded image matches the original, the VAE is working properly.")
    print("If it looks like boxes/cyan, there's an issue with the encoding/decoding.")


if __name__ == "__main__":
    main()
