"""
Convert LTX-2 LoRA from ComfyUI format to training format

ComfyUI format:
  - Keys: diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
  - Uses dots as separators
  - No alpha keys (scale is folded into lora_B weights)

Training format:
  - Keys: lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
  - Uses underscores as separators
  - Has separate .alpha keys
"""

import safetensors.torch
import torch
import argparse
import os
from pathlib import Path


def convert_key_to_training(key):
    """
    Convert a ComfyUI format key to training format

    Example:
        diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
        -> lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
    """
    # Split into main part and weight part
    parts = key.split('.')
    if len(parts) < 3:
        print(f"Warning: Unexpected key format: {key}")
        return None

    # parts[0] = diffusion_model
    # parts[1:] = transformer_blocks.0.attn1.to_k.lora_A.weight

    # Remove 'diffusion_model' prefix
    if parts[0] != 'diffusion_model':
        print(f"Warning: Key doesn't start with 'diffusion_model': {key}")
        return None

    # Extract components
    # diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
    # parts = ['diffusion_model', 'transformer_blocks', '0', 'attn1', 'to_k', 'lora_A', 'weight']
    # We need: main='transformer_blocks.0.attn1.to_k', layer_type='lora_A', weight='weight'
    layer_type = parts[-2]  # lora_A or lora_B
    weight_part = parts[-1]  # weight
    main_part = '.'.join(parts[1:-2])  # transformer_blocks.0.attn1.to_k (exclude lora_A/B and weight)

    # Convert dots to underscores, but need to handle numeric parts carefully
    # transformer_blocks.0.attn1.to_k -> transformer_blocks_0_attn1_to_k

    converted = main_part

    import re

    # Step 1: Handle transformer_blocks.N pattern FIRST
    converted = re.sub(r'transformer_blocks\.(\d+)', r'transformer_blocks_\1', converted)

    # Step 2: Handle audio/video attention patterns (longest first to avoid partial matches)
    converted = converted.replace('.audio_to_video_attn.', '_audio_to_video_attn_')
    converted = converted.replace('.video_to_audio_attn.', '_video_to_audio_attn_')
    converted = converted.replace('.audio_attn1.', '_audio_attn1_')
    converted = converted.replace('.audio_attn2.', '_audio_attn2_')
    converted = converted.replace('.audio_ff.', '_audio_ff_')

    # Step 3: Handle regular (non-audio) attention patterns
    converted = converted.replace('.attn1.', '_attn1_')
    converted = converted.replace('.attn2.', '_attn2_')

    # Step 4: Handle to_out.N patterns (must come after attn replacements)
    # Match either .to_out.N or _to_out.N (after attn replacements)
    converted = re.sub(r'[_\.]to_out\.(\d+)', r'_to_out_\1', converted)

    # Step 5: Handle projection layers
    converted = converted.replace('.to_k.', '_to_k_')
    converted = converted.replace('.to_q.', '_to_q_')
    converted = converted.replace('.to_v.', '_to_v_')
    converted = re.sub(r'\.to_out\.', '_to_out_', converted)

    # Step 6: Handle feedforward layers
    converted = converted.replace('.ff.net.', '_ff_net_')
    converted = converted.replace('.ff.', '_ff_')
    converted = converted.replace('.net.', '_net_')
    converted = converted.replace('.proj', '_proj')

    # Step 7: Handle net.N patterns (ff.net.N)
    converted = re.sub(r'\.net\.(\d+)', r'_net_\1', converted)

    # Convert weight naming: lora_A -> lora_down, lora_B -> lora_up
    if layer_type == 'lora_A':
        layer_type = 'lora_down'
    elif layer_type == 'lora_B':
        layer_type = 'lora_up'
    else:
        print(f"Warning: Unknown layer type: {layer_type}")
        return None

    # Build the final key - use dot separator before layer_type
    training_key = f"lora_unet_model_{converted}.{layer_type}.{weight_part}"

    # Clean up multiple underscores
    training_key = re.sub(r'_+_', '_', training_key)

    return training_key


def extract_lora_name_from_comfy_key(key):
    """
    Extract the LoRA module name from a ComfyUI key.
    Example: diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
             -> transformer_blocks_0_attn1_to_k
    """
    parts = key.split('.')
    if len(parts) < 3 or parts[0] != 'diffusion_model':
        return None

    main_part = '.'.join(parts[1:-2])  # Remove diffusion_model, lora_A/B, and weight
    layer_type = parts[-2]

    # Convert to training format naming (without the layer_type)
    converted = main_part
    import re

    # Step 1: Handle transformer_blocks.N pattern FIRST
    converted = re.sub(r'transformer_blocks\.(\d+)', r'transformer_blocks_\1', converted)

    # Step 2: Handle audio/video attention patterns (longest first to avoid partial matches)
    converted = converted.replace('.audio_to_video_attn.', '_audio_to_video_attn_')
    converted = converted.replace('.video_to_audio_attn.', '_video_to_audio_attn_')
    converted = converted.replace('.audio_attn1.', '_audio_attn1_')
    converted = converted.replace('.audio_attn2.', '_audio_attn2_')
    converted = converted.replace('.audio_ff.', '_audio_ff_')

    # Step 3: Handle regular (non-audio) attention patterns
    converted = converted.replace('.attn1.', '_attn1_')
    converted = converted.replace('.attn2.', '_attn2_')

    # Step 4: Handle to_out.N patterns (must come after attn replacements)
    # Match either .to_out.N or _to_out.N (after attn replacements)
    converted = re.sub(r'[_\.]to_out\.(\d+)', r'_to_out_\1', converted)

    # Step 5: Handle projection layers
    converted = converted.replace('.to_k.', '_to_k_')
    converted = converted.replace('.to_q.', '_to_q_')
    converted = converted.replace('.to_v.', '_to_v_')
    converted = re.sub(r'\.to_out\.', '_to_out_', converted)

    # Step 6: Handle feedforward layers
    converted = converted.replace('.ff.net.', '_ff_net_')
    converted = converted.replace('.ff.', '_ff_')
    converted = converted.replace('.net.', '_net_')
    converted = converted.replace('.proj', '_proj')

    # Step 7: Handle net.N patterns (ff.net.N)
    converted = re.sub(r'\.net\.(\d+)', r'_net_\1', converted)

    converted = re.sub(r'_+_', '_', converted)
    return converted


def rerank_lora_weights(lora_down: torch.Tensor, lora_up: torch.Tensor, target_rank: int) -> tuple:
    """
    Rerank LoRA weights using SVD to preserve most important information.

    Args:
        lora_down: Original lora_down (A) weight [rank, in_dim]
        lora_up: Original lora_up (B) weight [out_dim, rank]
        target_rank: Target rank to convert to

    Returns:
        Tuple of (new_lora_down, new_lora_up) with target_rank
    """
    original_rank = lora_down.shape[0]
    device = lora_down.device
    dtype = lora_down.dtype

    if target_rank == original_rank:
        return lora_down, lora_up
    elif target_rank > original_rank:
        # For upranking, pad with zeros
        # lora_down: [rank, in] -> [target_rank, in]
        # lora_up: [out, rank] -> [out, target_rank]
        pad_down = target_rank - original_rank
        new_lora_down = torch.nn.functional.pad(lora_down, (0, 0, 0, pad_down))
        new_lora_up = torch.nn.functional.pad(lora_up, (0, pad_down))
        return new_lora_down.to(dtype=dtype), new_lora_up.to(dtype=dtype)
    else:
        # For downranking, use SVD to preserve most important components
        # W = B @ A gives the effective weight update
        # SVD: W = U @ S @ V^T
        # New A = sqrt(S[:r]) @ V^T[:r, :]
        # New B = U[:, :r] @ sqrt(S[:r])

        # Move to CPU for SVD (more stable) and convert to float32
        lora_down_cpu = lora_down.cpu().float()
        lora_up_cpu = lora_up.cpu().float()

        # Compute effective weight matrix: W = B @ A
        # lora_up is [out_dim, rank], lora_down is [rank, in_dim]
        # W is [out_dim, in_dim]
        W = lora_up_cpu @ lora_down_cpu

        # SVD on the weight matrix
        # For efficiency, we can use the fact that rank is small
        # U: [out_dim, out_dim], S: [min(out_dim, in_dim)], V^T: [in_dim, in_dim]
        # But we only need target_rank components
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Keep top target_rank singular values
        U_r = U[:, :target_rank]  # [out_dim, target_rank]
        S_r = S[:target_rank]     # [target_rank]
        Vh_r = Vh[:target_rank, :]  # [target_rank, in_dim]

        # Create new LoRA weights
        # B_new = U @ sqrt(S), A_new = sqrt(S) @ Vh
        sqrt_S = torch.sqrt(S_r)
        new_lora_up = U_r * sqrt_S  # [out_dim, target_rank]
        new_lora_down = sqrt_S[:, None] * Vh_r  # [target_rank, in_dim]

        return new_lora_down.to(device=device, dtype=dtype), new_lora_up.to(device=device, dtype=dtype)


def convert_comfy_to_training(input_path, output_path=None, alpha=None, target_rank=None, verbose=False):
    """
    Convert a LoRA file from ComfyUI format to training format

    Args:
        input_path: Path to the input ComfyUI LoRA file
        output_path: Path to save the converted LoRA (optional)
        alpha: Alpha value to use for all layers (optional, will try to detect from lora_A shape if not provided)
        target_rank: Target rank to convert to (optional, uses SVD for rank conversion)
        verbose: Print detailed conversion info

    Returns:
        Path to the output file
    """
    print(f"Loading ComfyUI LoRA from: {input_path}")

    # Load the ComfyUI LoRA
    comfy_state_dict = safetensors.torch.load_file(input_path)

    print(f"Input LoRA has {len(comfy_state_dict)} keys")

    # Group keys by layer and extract rank information
    lora_groups = {}
    for key, tensor in comfy_state_dict.items():
        if key.startswith('diffusion_model.') and ('.lora_A.' in key or '.lora_B.' in key):
            lora_name = extract_lora_name_from_comfy_key(key)
            if lora_name is None:
                print(f"Warning: Could not extract lora_name from: {key}")
                continue

            if lora_name not in lora_groups:
                lora_groups[lora_name] = {}

            if '.lora_A.' in key:
                lora_groups[lora_name]['lora_A'] = tensor
                # Rank is the output dimension of lora_A
                lora_groups[lora_name]['rank'] = tensor.shape[0]
            elif '.lora_B.' in key:
                lora_groups[lora_name]['lora_B'] = tensor

    # Detect alpha if not provided
    if alpha is None:
        # Try to detect alpha by checking if lora_B was scaled
        # In ComfyUI format, if lora_B has the folded scale, we need to reverse it
        # We'll use rank as the default alpha
        if lora_groups:
            sample_name = list(lora_groups.keys())[0]
            detected_rank = lora_groups[sample_name].get('rank', 4)
            alpha = detected_rank
            print(f"No alpha specified, using rank {alpha} as alpha")
        else:
            alpha = 4
            print(f"No alpha specified and could not detect rank, using default alpha={alpha}")

    # Check if rank conversion is needed
    if target_rank is not None:
        sample_name = list(lora_groups.keys())[0]
        original_rank = lora_groups[sample_name].get('rank', 4)
        if target_rank != original_rank:
            print(f"\n[Rank Conversion] Converting from rank {original_rank} to {target_rank} using SVD...")
            print(f"[Rank Conversion] This may take 10-30 seconds depending on model size...")
    else:
        target_rank = alpha  # Default to detected/original rank

    # Convert keys with optional rank conversion
    training_state_dict = {}
    converted = 0
    failed = 0
    alpha_created = 0
    reranked = 0

    # First pass: convert all non-LoRA weights and group LoRA weights by module
    non_lora_weights = {}
    converted_lora_groups = {}

    for key, tensor in comfy_state_dict.items():
        new_key = convert_key_to_training(key)

        if new_key is None:
            failed += 1
            if verbose:
                print(f"Failed to convert key: {key}")
        elif '.lora_down.' not in new_key and '.lora_up.' not in new_key:
            # Non-LoRA weight (e.g., alpha keys if they existed, though they don't in ComfyUI)
            non_lora_weights[new_key] = tensor
            converted += 1
        else:
            # LoRA weight - group by module
            # Extract module name from new_key
            # lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
            # -> module_name = transformer_blocks_0_attn1_to_k
            parts = new_key.split('.')
            # Remove 'lora_unet_model_' prefix
            main_part = parts[0][len('lora_unet_model_'):]
            # Remove .lora_down.weight or .lora_up.weight suffix
            if parts[1] == 'lora_down':
                weight_type = 'lora_down'
            elif parts[1] == 'lora_up':
                weight_type = 'lora_up'
            else:
                failed += 1
                continue

            module_name = main_part
            if module_name not in converted_lora_groups:
                converted_lora_groups[module_name] = {}

            converted_lora_groups[module_name][weight_type] = (new_key, tensor)

    # Second pass: process LoRA groups with optional rank conversion
    for module_name, weights in converted_lora_groups.items():
        if 'lora_down' not in weights or 'lora_up' not in weights:
            # Skip incomplete pairs
            continue

        lora_down_key, lora_down = weights['lora_down']
        lora_up_key, lora_up = weights['lora_up']

        # Get the original module name to find rank info
        # The original comfy name should match what we have in lora_groups
        # Need to map back - the converted module_name is like "transformer_blocks_0_attn1_to_k"

        # Unfold alpha scale from lora_up if needed (ComfyUI format has folded alpha)
        original_module_name = None
        for orig_name, orig_data in lora_groups.items():
            # orig_name from extract_lora_name_from_comfy_key gives us "transformer_blocks_0_attn1_to_k"
            if orig_name == module_name:
                original_module_name = orig_name
                break

        if original_module_name in lora_groups:
            rank = lora_groups[original_module_name].get('rank', alpha)
            scale = float(alpha) / float(rank)
            if scale != 1.0:
                lora_up = lora_up / scale
                if verbose:
                    print(f"Unfolded alpha for {module_name}: alpha={alpha} rank={rank} scale={scale}")

        # Apply rank conversion if needed
        current_rank = lora_down.shape[0]
        if target_rank != current_rank:
            lora_down, lora_up = rerank_lora_weights(lora_down, lora_up, target_rank)
            reranked += 1

        training_state_dict[lora_down_key] = lora_down
        training_state_dict[lora_up_key] = lora_up
        converted += 2

    # Add non-LoRA weights
    training_state_dict.update(non_lora_weights)

    # Create alpha keys for each LoRA module with target rank
    for module_name in converted_lora_groups.keys():
        alpha_key = f"lora_unet_model_{module_name}.alpha"
        training_state_dict[alpha_key] = torch.tensor(target_rank, dtype=torch.float32)
        alpha_created += 1
        if verbose:
            print(f"Created alpha key: {alpha_key} = {target_rank}")

    print(f"\nConversion summary:")
    print(f"  Converted: {converted} keys")
    print(f"  Created alpha keys: {alpha_created}")
    if reranked > 0:
        print(f"  Reranked: {reranked} LoRA modules from rank {original_rank} to {target_rank}")
    print(f"  Failed: {failed} keys")
    print(f"  Output LoRA has {len(training_state_dict)} keys")

    # Determine output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_training{input_file.suffix}"

    # Load metadata from the original file
    metadata = None
    try:
        with safetensors.safe_open(input_path, framework="pt") as f:
            metadata = f.metadata()
        if metadata:
            print(f"Preserving {len(metadata)} metadata entries")
    except Exception as e:
        print(f"Warning: Could not read metadata: {e}")

    # Save the converted LoRA
    print(f"\nSaving training format LoRA to: {output_path}")
    safetensors.torch.save_file(training_state_dict, output_path, metadata=metadata)

    print(f"[OK] Conversion complete!")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert LTX-2 LoRA from ComfyUI format to training format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input ComfyUI LoRA file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the converted LoRA (default: <input>_training.safetensors)"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="Alpha value for the training format (default: auto-detect from rank)"
    )
    parser.add_argument(
        "--target_rank",
        type=int,
        default=None,
        help="Target rank to convert LoRA to (uses SVD for rank reduction/padding)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed conversion information"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1

    try:
        output_path = convert_comfy_to_training(args.input, args.output, args.alpha, args.target_rank, args.verbose)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
