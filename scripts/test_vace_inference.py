#!/usr/bin/env python3
"""Test VACE inference using cached latents + LoRA."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from safetensors.torch import load_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from musubi_tuner.ltx2_vace_train import LTX2VaceTrainer
from musubi_tuner.utils.device_utils import clean_memory_on_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vace_lora_path", type=str, required=True, help="VACE LoRA .safetensors")
    p.add_argument("--vace_cached_latent", type=str, required=True, help="Cached VACE latent .safetensors")
    p.add_argument("--vace_scale", type=float, default=1.0)
    p.add_argument("--vace_lora_multiplier", type=float, default=1.0, help="LoRA merge multiplier")
    p.add_argument("--ltx2_checkpoint", type=str, default="/home/e/Dpipe/models/ltx2/ltx-2.3-22b-dev.safetensors")
    p.add_argument("--gemma_root", type=str, default="/home/e/Dpipe/models/text_encoders/gemma3")
    p.add_argument("--prompt", type=str, default="Two women with long brown hair dancing on the dance floor")
    p.add_argument("--negative_prompt", type=str, default="worst quality, inconsistent motion, blurry")
    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="workspace/output/vace/test")
    p.add_argument("--blocks_to_swap", type=int, default=45)
    p.add_argument("--fp8_base", action="store_true")
    p.add_argument("--fp8_scaled", action="store_true")
    p.add_argument("--8_bit_te", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Load cached VACE latent
    sd = load_file(args.vace_cached_latent)
    # keys like "vace_latents_11x16x8_bfloat16" -> 4D tensor (C, F, H, W)
    vace_key = [k for k in sd.keys() if k.startswith("latents_")][0]
    vace_latent = sd[vace_key]  # (C, F_lat, H_lat, W_lat)
    logger.info(f"Loaded cached VACE latent: {vace_key} shape={vace_latent.shape}")

    # Derive video dims from latent: spatial*32, temporal*8-1
    C, F_lat, H_lat, W_lat = vace_latent.shape
    height = H_lat * 32
    width = W_lat * 32
    frame_count = (F_lat - 1) * 8 + 1
    logger.info(f"Derived video dims: {width}x{height}, {frame_count} frames")

    # Build trainer args
    ta = SimpleNamespace(
        ltx2_checkpoint=args.ltx2_checkpoint,
        dit=args.ltx2_checkpoint,
        vae=args.ltx2_checkpoint,
        vae_dtype="bfloat16",
        ltx_mode="video",
        mixed_precision="bf16",
        blocks_to_swap=args.blocks_to_swap,
        flash_attn=False, flash3=False, sdpa=True, xformers=False,
        fp8_base=args.fp8_base, fp8_scaled=args.fp8_scaled,
        w8a8_mode="int8",
        sample_i2v_token_timestep_mask=True,
        sample_disable_flash_attn=True,
        sample_with_offloading=True,
        output_dir=args.output_dir,
        output_name="vace_test",
        gemma_root=args.gemma_root,
        gemma_safetensors=None,
        gemma_load_in_8bit=getattr(args, "8_bit_te", False),
        use_precached_sample_prompts=False,
        lora_weight=None, include_patterns=None, exclude_patterns=None,
        attn_mode="sdpa", compile=False,
        network_module=None, network_args=None,
        vace_scale=args.vace_scale, vace_layers=None, vace_freeze_dit=True,
        audio_vace_model_path=None, audio_vace_scale=1.0,
        enable_audio_xattn_in_vace=False,
        _vace_full_mode=True, _ltx2_3=False,
        sample_at_first=True, sample_every_n_steps=None, sample_every_n_epochs=None,
    )

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # Load transformer with VACE blocks
    trainer = LTX2VaceTrainer()
    trainer.blocks_to_swap = int(args.blocks_to_swap or 0)
    trainer.handle_model_specific_args(ta)

    loading_device = "cpu" if trainer.blocks_to_swap > 0 else device
    transformer = trainer.load_transformer(
        accelerator=SimpleNamespace(device=device),
        args=ta, dit_path=args.ltx2_checkpoint,
        attn_mode="sdpa", split_attn=False,
        loading_device=loading_device, dit_weight_dtype=None,
    )

    ltx_model = transformer.module if hasattr(transformer, "module") else transformer

    # Merge VACE LoRA into the VACE blocks
    logger.info(f"Merging VACE LoRA: {args.vace_lora_path}")
    from musubi_tuner.networks import lora_ltx2
    lora_sd = load_file(args.vace_lora_path)
    dit_dtype = trainer.dit_dtype or torch.float32
    net = lora_ltx2.create_arch_network_from_weights(
        args.vace_lora_multiplier, lora_sd, unet=ltx_model,
        for_inference=True,
        include_patterns=["vace_blocks"],
    )
    net.merge_to(None, ltx_model, lora_sd, device=next(ltx_model.parameters()).device, non_blocking=True)
    clean_memory_on_device(device)
    logger.info("VACE LoRA merged")

    # Attach VACE model to transformer
    ltx_model._vace_model = trainer._vace_model.to(dtype=dit_dtype)
    logger.info("VACE model attached to LTXModel")

    # Block swap
    if trainer.blocks_to_swap > 0:
        transformer.enable_block_swap(trainer.blocks_to_swap, device, supports_backward=False)
        if hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(device)
        if hasattr(transformer, "switch_block_swap_for_inference"):
            transformer.switch_block_swap_for_inference()

    # Patchify cached latent
    vace_latent_5d = vace_latent.unsqueeze(0)  # (1, C, F, H, W)
    from musubi_tuner.ltx_vace.vace_control_encoder import patchify_vace_context
    patchifier = getattr(ltx_model, "video_patchifier", None)
    with torch.no_grad():
        vace_tokens = patchify_vace_context(vace_latent_5d.to(device=device, dtype=dit_dtype), patchifier=patchifier)
    logger.info(f"VACE tokens: {vace_tokens.shape}")
    del vace_latent, vace_latent_5d
    clean_memory_on_device(device)

    # Monkey-patch transformer __call__ to inject VACE context
    _original_call = ltx_model.__class__.__call__

    def _patched_call(mod, *call_args, **call_kwargs):
        if "transformer_options" in call_kwargs:
            opts = dict(call_kwargs["transformer_options"]) if call_kwargs["transformer_options"] else {}
            opts["vace_context"] = vace_tokens
            opts["vace_scale"] = args.vace_scale
            call_kwargs["transformer_options"] = opts
        return _original_call(mod, *call_args, **call_kwargs)

    ltx_model.__class__.__call__ = _patched_call

    # Sample
    sample = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": height,
        "width": width,
        "frame_count": frame_count,
        "frame_rate": 25.0,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "discrete_flow_shift": 5.0,
        "seed": args.seed,
        "cfg_scale": None,
        "enum": 0,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("=" * 50)
    logger.info(f"VACE LoRA: {args.vace_lora_path} (multiplier={args.vace_lora_multiplier})")
    logger.info(f"Cached latent: {args.vace_cached_latent}")
    logger.info(f"Video: {width}x{height}, {frame_count} frames")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Steps: {args.sample_steps}, CFG: {args.guidance_scale}")
    logger.info("=" * 50)

    trainer.sample_images(
        accelerator=accelerator, args=ta,
        epoch=0, steps=0, vae=None,
        transformer=transformer,
        sample_parameters=[sample],
        dit_dtype=dit_dtype,
    )

    ltx_model.__class__.__call__ = _original_call
    logger.info(f"Done! Output: {os.path.join(args.output_dir, 'sample')}")


if __name__ == "__main__":
    main()
