"""
Rerank a training format LTX-2 LoRA/LoKR checkpoint to a different rank using SVD.
This script works on files already in training format (keys start with lora_unet_model_).
Supports both standard LoRA (.lora_down/.lora_up) and LoKR (.lokr_w1/.lokr_w2) formats.
"""

import safetensors.torch
import torch
import argparse
import os
from pathlib import Path


def rerank_lora_weights(lora_down: torch.Tensor, lora_up: torch.Tensor, target_rank: int, device: str = 'cuda') -> tuple:
    """
    Rerank LoRA weights using SVD to preserve most important information.
    """
    original_rank = lora_down.shape[0]
    dtype = lora_down.dtype

    if target_rank == original_rank:
        return lora_down, lora_up
    elif target_rank > original_rank:
        # For upranking, pad with zeros
        pad_down = target_rank - original_rank
        new_lora_down = torch.nn.functional.pad(lora_down, (0, 0, 0, pad_down))
        new_lora_up = torch.nn.functional.pad(lora_up, (0, pad_down))
        return new_lora_down.to(dtype=dtype), new_lora_up.to(dtype=dtype)
    else:
        # For downranking, use SVD
        # Try to use GPU if available for speed
        device_to_use = device if torch.cuda.is_available() else 'cpu'
        lora_down_dev = lora_down.to(device=device_to_use).float()
        lora_up_dev = lora_up.to(device=device_to_use).float()

        # Compute effective weight matrix: W = B @ A
        W = lora_up_dev @ lora_down_dev

        # SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Keep top target_rank singular values
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vh_r = Vh[:target_rank, :]

        # Create new LoRA weights
        sqrt_S = torch.sqrt(S_r)
        new_lora_up = U_r * sqrt_S
        new_lora_down = sqrt_S[:, None] * Vh_r

        return new_lora_down.to(device=lora_down.device, dtype=dtype), new_lora_up.to(device=lora_up.device, dtype=dtype)


def rerank_lokr_weights(lokr_w1_b: torch.Tensor, lokr_w2_a: torch.Tensor, target_rank: int, device: str = 'cuda') -> tuple:
    """
    Rerank LoKR weights using SVD.

    LoKR decomposition: W = lokr_w1_b @ lokr_w2_a
    - lokr_w1_b shape: [out_features, rank]
    - lokr_w2_a shape: [rank, in_features]
    """
    original_rank = lokr_w1_b.shape[1]
    dtype = lokr_w1_b.dtype

    if target_rank == original_rank:
        return lokr_w1_b, lokr_w2_a
    elif target_rank > original_rank:
        # For upranking, pad with zeros
        pad = target_rank - original_rank
        new_w1_b = torch.nn.functional.pad(lokr_w1_b, (0, pad))
        new_w2_a = torch.nn.functional.pad(lokr_w2_a, (0, 0, 0, pad))
        return new_w1_b.to(dtype=dtype), new_w2_a.to(dtype=dtype)
    else:
        # For downranking, use SVD
        device_to_use = device if torch.cuda.is_available() else 'cpu'
        w1_b_dev = lokr_w1_b.to(device=device_to_use).float()
        w2_a_dev = lokr_w2_a.to(device=device_to_use).float()

        # Compute effective weight matrix: W = w1_b @ w2_a
        W = w1_b_dev @ w2_a_dev

        # SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Keep top target_rank singular values
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vh_r = Vh[:target_rank, :]

        # Create new LoKR weights
        sqrt_S = torch.sqrt(S_r)
        new_w1_b = U_r * sqrt_S
        new_w2_a = sqrt_S[:, None] * Vh_r

        return new_w1_b.to(device=lokr_w1_b.device, dtype=dtype), new_w2_a.to(device=lokr_w2_a.device, dtype=dtype)


def detect_checkpoint_format(state_dict: dict) -> str:
    """Detect if checkpoint is LoRA or LoKR format."""
    for key in state_dict.keys():
        # LoKR keys: lora_unet_model_<module>.lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b
        if '.lokr_w1_a' in key or '.lokr_w1_b' in key or '.lokr_w2_a' in key or '.lokr_w2_b' in key:
            return 'lokr'
        # Standard LoRA keys
        elif '.lora_down.weight' in key:
            return 'lora'
    return 'unknown'


def detect_rank_from_state_dict(state_dict: dict) -> int:
    """Detect rank from either LoRA or LoKR format."""
    # Try LoKR first - check all lokr keys
    # LoKR can have different weight pairs:
    # - lokr_w2_b + lokr_w2_a: w2_b[rank, out], w2_a[in, rank]
    # - lokr_w1_b + lokr_w2_a: w1_b[out, rank], w2_a[rank, in]
    # - lokr_w2_b + lokr_w1_a: w2_b[rank, out], w1_a[in, rank]
    for key in state_dict.keys():
        if '.lokr_w2_b' in key:
            tensor = state_dict[key]
            # lokr_w2_b shape is [rank, out_features]
            if len(tensor.shape) >= 2:
                return tensor.shape[0]  # rank is FIRST dimension
        elif '.lokr_w1_b' in key:
            tensor = state_dict[key]
            # lokr_w1_b shape is [out_features, rank]
            if len(tensor.shape) >= 2:
                return tensor.shape[1]  # rank is SECOND dimension
        elif '.lokr_w2_a' in key:
            tensor = state_dict[key]
            # lokr_w2_a shape is [in_features, rank] or [rank, in_features]
            if len(tensor.shape) >= 2:
                # Use smaller dimension to be safe
                return min(tensor.shape)
        elif '.lokr_w1_a' in key:
            tensor = state_dict[key]
            # lokr_w1_a shape varies, use smaller dimension
            if len(tensor.shape) >= 2:
                return min(tensor.shape)

    # Try standard LoRA
    for key in state_dict.keys():
        if '.lora_down.weight' in key:
            return state_dict[key].shape[0]

    # Print some keys for debugging
    sample_keys = list(state_dict.keys())[:10]
    raise ValueError(f"Could not detect rank from checkpoint - no recognized keys found. Sample keys: {sample_keys}")


def rerank_checkpoint(input_path: str, output_path: str, target_rank: int, device: str = 'cuda', verbose: bool = False):
    """Rerank a training format LoRA/LoKR checkpoint."""
    print(f"Loading checkpoint: {input_path}")
    state_dict = safetensors.torch.load_file(input_path)

    # Detect format and rank
    checkpoint_format = detect_checkpoint_format(state_dict)
    print(f"Detected format: {checkpoint_format}")

    original_rank = detect_rank_from_state_dict(state_dict)
    print(f"Original rank: {original_rank}, Target rank: {target_rank}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'}")

    if checkpoint_format == 'lokr':
        # Handle LoKR format
        # LoKR has different possible decompositions:
        # - lokr_w2_b + lokr_w2_a: W = w2_b.T @ w2_a.T, w2_b[rank, out], w2_a[in, rank]
        # - lokr_w1_b + lokr_w2_a: W = w1_b @ w2_a, w1_b[out, rank], w2_a[rank, in]
        # - lokr_w2_b + lokr_w1_a: W = w2_b @ w1_a, w2_b[rank, out], w1_a[in, rank]

        lokr_groups = {}
        for key, tensor in state_dict.items():
            # Group all LoKR weights by module
            for weight_type in ['lokr_w1_a', 'lokr_w1_b', 'lokr_w2_a', 'lokr_w2_b', 'lokr_w1']:
                if weight_type in key:
                    # Extract module name: lora_unet_model_<module>.<weight_type>
                    remaining = key.replace('lora_unet_model_', '')
                    module_name = remaining.replace('.' + weight_type, '')

                    if module_name not in lokr_groups:
                        lokr_groups[module_name] = {}
                    lokr_groups[module_name][weight_type] = (key, tensor)
                    break

        print(f"Reranking {len(lokr_groups)} LoKR modules...")

        for i, (module_name, weights) in enumerate(lokr_groups.items()):
            # Try to find the main weight pair and rerank
            if 'lokr_w2_b' in weights and 'lokr_w2_a' in weights:
                # Main pair: W = w2_b.T @ w2_a.T
                # w2_b: [rank, out], w2_a: [in, rank]
                w2_b_key, w2_b_tensor = weights['lokr_w2_b']
                w2_a_key, w2_a_tensor = weights['lokr_w2_a']

                original_rank = w2_b_tensor.shape[0]  # rank is first dimension of w2_b
                dtype = w2_b_tensor.dtype

                if target_rank == original_rank:
                    new_w2_b = w2_b_tensor
                    new_w2_a = w2_a_tensor
                elif target_rank > original_rank:
                    # Upranking - pad with zeros
                    pad = target_rank - original_rank
                    # Pad w2_b: [rank, out] -> [target_rank, out]
                    new_w2_b = torch.nn.functional.pad(w2_b_tensor, (0, 0, 0, pad))
                    # Pad w2_a: [in, rank] -> [in, target_rank]
                    new_w2_a = torch.nn.functional.pad(w2_a_tensor, (0, pad))
                else:
                    # Downranking using SVD
                    device_to_use = device if torch.cuda.is_available() else 'cpu'
                    w2_a_dev = w2_a_tensor.to(device=device_to_use).float()
                    w2_b_dev = w2_b_tensor.to(device=device_to_use).float()

                    # Compute effective weight: W = w2_b.T @ w2_a.T
                    # w2_b: [rank, out], w2_a: [in, rank]
                    # W = w2_b.T @ w2_a.T = [out, rank] @ [rank, in] = [out, in]
                    W = w2_b_dev.T @ w2_a_dev.T

                    # SVD
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                    # Keep top target_rank singular values
                    U_r = U[:, :target_rank]
                    S_r = S[:target_rank]
                    Vh_r = Vh[:target_rank, :]

                    # Create new weights: W_r = U_r @ diag(S_r) @ Vh_r
                    # We want: W_r = new_w2_b.T @ new_w2_a.T
                    # So: new_w2_b.T = U_r * sqrt(S_r), new_w2_a.T = sqrt(S_r) * Vh_r
                    sqrt_S = torch.sqrt(S_r)
                    new_w2_b = (U_r * sqrt_S).T  # [target_rank, out]
                    new_w2_a = (sqrt_S[:, None] * Vh_r).T  # [in, target_rank]

                    new_w2_b = new_w2_b.to(device=w2_b_tensor.device, dtype=dtype)
                    new_w2_a = new_w2_a.to(device=w2_a_tensor.device, dtype=dtype)

                # Update state dict
                state_dict[w2_b_key] = new_w2_b
                state_dict[w2_a_key] = new_w2_a

                # Note: lokr_w1 is tied to the 'factor' parameter (e.g., factor=4 means [4,4])
                # and should NOT be modified during rank changes. The factor is independent of rank.

            elif 'lokr_w1_b' in weights and 'lokr_w2_a' in weights:
                # Alternative pair: W = w1_b @ w2_a
                w1_b_key, w1_b_tensor = weights['lokr_w1_b']
                w2_a_key, w2_a_tensor = weights['lokr_w2_a']

                original_rank = w1_b_tensor.shape[1]  # rank is second dimension of w1_b
                dtype = w1_b_tensor.dtype

                if target_rank == original_rank:
                    new_w1_b = w1_b_tensor
                    new_w2_a = w2_a_tensor
                elif target_rank > original_rank:
                    pad = target_rank - original_rank
                    new_w1_b = torch.nn.functional.pad(w1_b_tensor, (0, pad))
                    new_w2_a = torch.nn.functional.pad(w2_a_tensor, (0, 0, 0, pad))
                else:
                    # Downranking
                    device_to_use = device if torch.cuda.is_available() else 'cpu'
                    w1_b_dev = w1_b_tensor.to(device=device_to_use).float()
                    w2_a_dev = w2_a_tensor.to(device=device_to_use).float()

                    # W = w1_b @ w2_a = [out, rank] @ [rank, in] = [out, in]
                    W = w1_b_dev @ w2_a_dev

                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                    U_r = U[:, :target_rank]
                    S_r = S[:target_rank]
                    Vh_r = Vh[:target_rank, :]

                    sqrt_S = torch.sqrt(S_r)
                    new_w1_b = U_r * sqrt_S
                    new_w2_a = sqrt_S[:, None] * Vh_r

                    new_w1_b = new_w1_b.to(device=w1_b_tensor.device, dtype=dtype)
                    new_w2_a = new_w2_a.to(device=w2_a_tensor.device, dtype=dtype)

                state_dict[w1_b_key] = new_w1_b
                state_dict[w2_a_key] = new_w2_a

            elif 'lokr_w2_b' in weights and 'lokr_w1_a' in weights:
                # Alternative pair: W = w2_b @ w1_a
                w2_b_key, w2_b_tensor = weights['lokr_w2_b']
                w1_a_key, w1_a_tensor = weights['lokr_w1_a']

                original_rank = w2_b_tensor.shape[0]
                dtype = w2_b_tensor.dtype

                if target_rank == original_rank:
                    new_w2_b = w2_b_tensor
                    new_w1_a = w1_a_tensor
                elif target_rank > original_rank:
                    pad = target_rank - original_rank
                    new_w2_b = torch.nn.functional.pad(w2_b_tensor, (0, 0, 0, pad))
                    new_w1_a = torch.nn.functional.pad(w1_a_tensor, (0, pad))
                else:
                    device_to_use = device if torch.cuda.is_available() else 'cpu'
                    w1_a_dev = w1_a_tensor.to(device=device_to_use).float()
                    w2_b_dev = w2_b_tensor.to(device=device_to_use).float()

                    # W = w2_b @ w1_a = [rank, out] @ [in, rank]
                    # Need to transpose: W = w2_b @ w1_a.T = [rank, out] @ [rank, in].T = [rank, out] @ [in, rank]
                    W = w2_b_dev @ w1_a_dev.T

                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                    U_r = U[:, :target_rank]
                    S_r = S[:target_rank]
                    Vh_r = Vh[:target_rank, :]

                    sqrt_S = torch.sqrt(S_r)
                    new_w2_b = (U_r * sqrt_S)
                    new_w1_a = (sqrt_S[:, None] * Vh_r).T

                    new_w2_b = new_w2_b.to(device=w2_b_tensor.device, dtype=dtype)
                    new_w1_a = new_w1_a.to(device=w1_a_tensor.device, dtype=dtype)

                state_dict[w2_b_key] = new_w2_b
                state_dict[w1_a_key] = new_w1_a
            else:
                print(f"  Warning: Module {module_name} missing required LoKR weight pair, skipping")

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(lokr_groups)} modules...")

    elif checkpoint_format == 'lora':
        # Handle standard LoRA format
        lora_groups = {}
        for key, tensor in state_dict.items():
            if key.endswith('.lora_down.weight'):
                module_name = key[len('lora_unet_model_'):key.rfind('.lora_down.weight')]
                if module_name not in lora_groups:
                    lora_groups[module_name] = {}
                lora_groups[module_name]['lora_down'] = (key, tensor)
            elif key.endswith('.lora_up.weight'):
                module_name = key[len('lora_unet_model_'):key.rfind('.lora_up.weight')]
                if module_name not in lora_groups:
                    lora_groups[module_name] = {}
                lora_groups[module_name]['lora_up'] = (key, tensor)

        print(f"Reranking {len(lora_groups)} LoRA modules...")

        for i, (module_name, weights) in enumerate(lora_groups.items()):
            if 'lora_down' in weights and 'lora_up' in weights:
                down_key, down_tensor = weights['lora_down']
                up_key, up_tensor = weights['lora_up']

                # Apply reranking
                new_down, new_up = rerank_lora_weights(down_tensor, up_tensor, target_rank, device)

                # Update state dict
                state_dict[down_key] = new_down
                state_dict[up_key] = new_up

                # Update alpha key
                alpha_key = f"lora_unet_model_{module_name}.alpha"
                state_dict[alpha_key] = torch.tensor(target_rank, dtype=torch.float32)

                if verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(lora_groups)} modules...")
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint_format}")

    # Preserve metadata
    metadata = None
    try:
        with safetensors.safe_open(input_path, framework="pt") as f:
            metadata = f.metadata()
    except:
        pass

    # Save
    print(f"Saving to: {output_path}")
    safetensors.torch.save_file(state_dict, output_path, metadata=metadata)
    print(f"[OK] Reranking complete!")


def main():
    parser = argparse.ArgumentParser(description="Rerank a training format LTX-2 LoRA/LoKR checkpoint")
    parser.add_argument("input", help="Path to input LoRA (training format)")
    parser.add_argument("-o", "--output", help="Output path (default: <input>_rank<N>.safetensors)")
    parser.add_argument("--target_rank", type=int, required=True, help="Target rank")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for SVD computation")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1

    if args.output is None:
        input_file = Path(args.input)
        args.output = str(input_file.parent / f"{input_file.stem}_rank{args.target_rank}{input_file.suffix}")

    try:
        rerank_checkpoint(args.input, args.output, args.target_rank, args.device, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
