"""
Rerank a training format LTX-2 LoRA checkpoint to a different rank using SVD.
This script works on files already in training format (keys start with lora_unet_model_).
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


def rerank_checkpoint(input_path: str, output_path: str, target_rank: int, device: str = 'cuda', verbose: bool = False):
    """Rerank a training format LoRA checkpoint."""
    print(f"Loading checkpoint: {input_path}")
    state_dict = safetensors.torch.load_file(input_path)

    # Detect original rank
    original_rank = None
    for key in state_dict.keys():
        if key.endswith('.lora_down.weight'):
            original_rank = state_dict[key].shape[0]
            break

    if original_rank is None:
        raise ValueError("Could not detect rank from checkpoint")

    print(f"Original rank: {original_rank}, Target rank: {target_rank}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'}")

    # Group LoRA weights by module
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

    # Rerank each module
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
    parser = argparse.ArgumentParser(description="Rerank a training format LTX-2 LoRA checkpoint")
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
