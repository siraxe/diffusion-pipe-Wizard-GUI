"""
MiniMax H3 Training Command Builder

Constructs accelerate-launch commands for MiniMax H3 (T2VA / FL2VA) training,
latent caching, and text-encoder caching via the musubi-tuner H3 scripts.

This module currently ONLY builds and formats commands — it does not execute
them. Pair with a dispatcher when wiring into the training flow.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# H3-specific script names under diffusion-trainers/musubi-tuner/
H3_TRAIN_SCRIPT = "minimax_h3_train_network.py"
H3_CACHE_LATENTS_SCRIPT = "minimax_h3_cache_latents.py"
H3_CACHE_TEXT_SCRIPT = "minimax_h3_cache_text_encoder_outputs.py"

# Defaults pulled from minimax_h3.md
DEFAULTS = {
    "learning_rate": 1e-4,
    "max_grad_norm": 1.0,
    "blocks_to_swap": 0,
    "gradient_accumulation_steps": 1,
    "scheduler_type": "constant",
    "mixed_precision_mode": "bf16",
    "rank": 16,
    "alpha": 16,
    "optimizer_type": "AdamW8bit",
    "h3_training_mode": "fl2va",
    "output_dir": "output/mmh3_lora",
    "output_name": "mmh3_style",
    "task": "t2va",
}


class MMH3Run:
    """Builds MiniMax H3 commands (no execution)."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.musubi_root = self.project_root / "diffusion-trainers" / "musubi-tuner"

    @staticmethod
    def _find_project_root() -> Path:
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "flet_app").exists() or (parent / "diffusion-trainers").exists():
                return parent
        return Path.cwd()

    def _resolve_path(self, path: str) -> str:
        if not path:
            return ""
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        return str(p)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _get(d: Dict, key: str, default=None):
        v = d.get(key, default)
        return default if (v is None or (isinstance(v, str) and v.strip() == "")) else v

    # ----------------------------------------------------------------------
    # Cache Commands
    # ----------------------------------------------------------------------

    def build_cache_latents_command(
        self,
        dataset_config: str,
        video_vae: str,
        audio_vae: str,
        batch_size: int = 1,
        device: str = "cuda",
    ) -> List[str]:
        """minimax_h3_cache_latents.py --dataset_config --vae --audio_vae"""
        script = str(self.musubi_root / H3_CACHE_LATENTS_SCRIPT)
        return [
            sys.executable,
            script,
            "--dataset_config", self._resolve_path(dataset_config),
            "--vae", self._resolve_path(video_vae),
            "--audio_vae", self._resolve_path(audio_vae),
            "--device", device,
            "--batch_size", str(batch_size),
        ]

    def build_cache_text_encoder_command(
        self,
        dataset_config: str,
        text_encoder: str,
        tokenizer: str,
        task: str = "t2va",
        batch_size: int = 1,
        device: str = "cuda",
        cache_guidance_empty: bool = False,
        text_encoder_quantization: str = "none",
    ) -> List[str]:
        """minimax_h3_cache_text_encoder_outputs.py --dataset_config --text_encoder --tokenizer --task"""
        script = str(self.musubi_root / H3_CACHE_TEXT_SCRIPT)
        cmd = [
            sys.executable,
            script,
            "--dataset_config", self._resolve_path(dataset_config),
            "--text_encoder", self._resolve_path(text_encoder),
            "--tokenizer", self._resolve_path(tokenizer),
            "--task", task,
            "--device", device,
            "--batch_size", str(batch_size),
        ]
        if cache_guidance_empty:
            cmd.append("--cache_guidance_empty")
        if text_encoder_quantization and str(text_encoder_quantization).lower() in ("int8", "nf4"):
            cmd.extend(["--text_encoder_quantization", str(text_encoder_quantization).lower()])
        return cmd

    # ----------------------------------------------------------------------
    # Training Command
    # ----------------------------------------------------------------------

    def build_training_command(
        self,
        config: Dict,
        dataset_config: str,
        slider_config: Optional[str] = None,
        resume: Optional[str] = None,
        reset_optimizer: bool = False,
        reset_optimizer_params: bool = False,
    ) -> List[str]:
        """Build the full accelerate launch training command for MiniMax H3.

        Expected config sections: model, optimization, acceleration,
        training_strategy, checkpoints, lora, validation (all optional except model).

        slider_config / reset_optimizer / reset_optimizer_params are accepted to
        match the LTX2Run/WAN22Run API; H3 doesn't use them yet.
        """
        model = config.get("model", {})
        optimization = config.get("optimization", {})
        acceleration = config.get("acceleration", {})
        training_strategy = config.get("training_strategy", {})
        checkpoints = config.get("checkpoints", {})

        script = str(self.musubi_root / H3_TRAIN_SCRIPT)

        cmd: List[str] = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            script,
            "--dit", self._resolve_path(self._get(model, "model_path", "")),
            "--dataset_config", self._resolve_path(dataset_config),
            "--mixed_precision",
                self._get(acceleration, "mixed_precision_mode", DEFAULTS["mixed_precision_mode"]),
        ]

        # Attention backend: --flash_attn or --sdpa (default sdpa)
        if self.parse_bool(acceleration.get("flash_attn", False)):
            cmd.append("--flash_attn")
        else:
            cmd.append("--sdpa")

        # Gradient checkpointing
        if self.parse_bool(optimization.get("enable_gradient_checkpointing", True)) or \
           self.parse_bool(acceleration.get("gradient_checkpointing", True)):
            cmd.append("--gradient_checkpointing")

        # fp8_base + H2D block swap options
        if self.parse_bool(acceleration.get("fp8_base", False)):
            cmd.append("--fp8_base")

        blocks_to_swap = int(optimization.get("blocks_to_swap", acceleration.get("blocks_to_swap", 0)) or 0)
        if blocks_to_swap > 0:
            cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])
            if self.parse_bool(acceleration.get("block_swap_h2d_only", False)):
                cmd.append("--block_swap_h2d_only")
                ring_size = int(acceleration.get("block_swap_ring_size", 2) or 2)
                cmd.extend(["--block_swap_ring_size", str(ring_size)])
                if self.parse_bool(acceleration.get("use_pinned_memory_for_block_swap", True)):
                    cmd.append("--use_pinned_memory_for_block_swap")

        # H3-specific training mode (default fl2va; ref2va is rejected by the backend)
        h3_mode = self._get(training_strategy, "h3_training_mode", DEFAULTS["h3_training_mode"])
        if h3_mode:
            cmd.extend(["--h3_training_mode", str(h3_mode)])

        # Optional experimental guidance-distillation scale
        gds = self._get(training_strategy, "h3_guidance_distillation_scale", None)
        if gds is not None and str(gds).strip() not in ("", "0", "0.0"):
            cmd.extend(["--h3_guidance_distillation_scale", str(gds)])

        # LoRA network configuration
        lora = config.get("lora", {})
        rank = self._get(lora, "rank", DEFAULTS["rank"])
        alpha = self._get(lora, "alpha", DEFAULTS["alpha"])
        cmd.extend([
            "--network_module", "networks.lora",
            "--network_dim", str(rank),
            "--network_alpha", str(alpha),
        ])
        # Optional network_args: dropout, caption_dropout_rate
        network_args = []
        dropout = self._get(lora, "dropout", lora.get("network_dropout", 0.0))
        try:
            if float(dropout) > 0:
                network_args.append(f"dropout={dropout}")
        except (TypeError, ValueError):
            pass
        caption_dropout = self._get(lora, "caption_dropout_rate", 0.0)
        try:
            if float(caption_dropout) > 0:
                cmd.extend(["--caption_dropout_rate", str(caption_dropout)])
        except (TypeError, ValueError):
            pass
        if network_args:
            cmd.append("--network_args")
            cmd.append(" ".join(network_args))

        # Optimizer + LR + scheduler
        opt_type = self._get(optimization, "optimizer_type", DEFAULTS["optimizer_type"])
        cmd.extend([
            "--optimizer_type", str(opt_type),
            "--learning_rate", str(self._get(optimization, "learning_rate", DEFAULTS["learning_rate"])),
            "--lr_scheduler",
                self._get(optimization, "scheduler_type", DEFAULTS["scheduler_type"]),
            "--gradient_accumulation_steps",
                str(self._get(optimization, "gradient_accumulation_steps", DEFAULTS["gradient_accumulation_steps"])),
        ])

        max_grad_norm = self._get(optimization, "max_grad_norm", DEFAULTS["max_grad_norm"])
        try:
            if float(max_grad_norm) > 0:
                cmd.extend(["--max_grad_norm", str(max_grad_norm)])
        except (TypeError, ValueError):
            pass

        # Max train steps/epochs + save_every
        ckpt_mode = checkpoints.get("mode", "steps")
        interval = int(checkpoints.get("interval", 50) or 50)
        steps_or_epochs = int(optimization.get("max_steps", optimization.get("max_train_epochs", 10)) or 10)
        if ckpt_mode == "epochs":
            cmd.extend([
                "--max_train_epochs", str(steps_or_epochs),
                "--save_every_n_epochs", str(interval),
            ])
        else:
            cmd.extend([
                "--max_train_steps", str(steps_or_epochs),
                "--save_every_n_steps", str(interval),
            ])

        # Output
        output_dir = self._get(model, "output_dir", DEFAULTS["output_dir"])
        output_name = self._get(model, "output_name", DEFAULTS["output_name"])
        cmd.extend([
            "--output_dir", str(output_dir),
            "--output_name", str(output_name),
            "--log_with", "tensorboard",
            "--logging_dir", os.path.join(str(output_dir), ".tensorboard"),
        ])

        # Resume
        if resume and os.path.exists(resume):
            cmd.extend(["--resume", resume])

        # Extra top-level flags
        extra = config.get("extra_flags", "")
        if extra and str(extra).strip():
            cmd.extend(str(extra).strip().split())

        return cmd

    # ----------------------------------------------------------------------
    # Formatting + Printing
    # ----------------------------------------------------------------------

    @staticmethod
    def format_command(cmd: List[str]) -> str:
        """Pretty-print a command list as a backslash-continued multi-line shell string."""
        lines = []
        i = 0
        while i < len(cmd):
            arg = cmd[i]
            if arg.startswith("--") and i + 1 < len(cmd) and not str(cmd[i + 1]).startswith("-"):
                lines.append(f"  {arg} {cmd[i + 1]} \\")
                i += 2
            else:
                lines.append(f"  {arg} \\")
                i += 1
        if lines:
            lines[-1] = lines[-1].rstrip(" \\")
        return "\n".join(lines)

    def print_all_commands(
        self,
        config: Dict,
        dataset_config: str,
        video_vae: Optional[str] = None,
        audio_vae: Optional[str] = None,
        tokenizer: Optional[str] = None,
        resume: Optional[str] = None,
    ) -> None:
        """Build and print all H3 commands (cache + train) to stdout.

        video_vae, audio_vae, and tokenizer fall back to model section keys
        (video_vae_path / audio_vae_path / tokenizer_path) if not passed explicitly.
        """
        model = config.get("model", {})

        text_encoder = self._get(model, "text_encoder_path", "")
        video_vae = self._get(model, "vae_path", video_vae or "")
        audio_vae = self._get(model, "vae_audio_path", audio_vae or "")
        tokenizer = self._get(model, "tokenizer_path", tokenizer or "")

        print("=" * 78)
        print("MiniMax H3 — cache latents")
        print("=" * 78)
        if video_vae and audio_vae:
            cmd = self.build_cache_latents_command(dataset_config, video_vae, audio_vae)
            print(self.format_command(cmd))
        else:
            print("[skipped] video_vae and audio_vae paths are required")

        print()
        print("=" * 78)
        print("MiniMax H3 — cache text encoder outputs")
        print("=" * 78)
        if text_encoder and tokenizer:
            cmd = self.build_cache_text_encoder_command(dataset_config, text_encoder, tokenizer)
            print(self.format_command(cmd))
        else:
            print("[skipped] text_encoder_path and tokenizer_path are required")

        print()
        print("=" * 78)
        print("MiniMax H3 — train network")
        print("=" * 78)
        cmd = self.build_training_command(config, dataset_config, resume=resume)
        print(self.format_command(cmd))
        print()


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------

def create_training_command(config: Dict, dataset_config: str, resume: Optional[str] = None) -> List[str]:
    return MMH3Run().build_training_command(config, dataset_config, resume=resume)


def format_training_command(config: Dict, dataset_config: str, resume: Optional[str] = None) -> str:
    return MMH3Run().format_command(create_training_command(config, dataset_config, resume=resume))


def print_all_commands(
    config: Dict,
    dataset_config: str,
    video_vae: Optional[str] = None,
    audio_vae: Optional[str] = None,
    tokenizer: Optional[str] = None,
    resume: Optional[str] = None,
) -> None:
    MMH3Run().print_all_commands(config, dataset_config, video_vae, audio_vae, tokenizer, resume)


if __name__ == "__main__":
    # Demo: build commands from a minimal sample config to verify structure.
    sample_config = {
        "model": {
            "model_path": "models/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors",
            "text_encoder_path": "models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
            "tokenizer_path": "models/MiniMax-H3/tokenizer",
            "video_vae_path": "models/vae/minimax_h3_video_vae_fp16.safetensors",
            "audio_vae_path": "models/vae/minimax_h3_audio_vae_fp32.safetensors",
            "vae_path": "models/vae/minimax_h3_video_vae_fp16.safetensors",
            "vae_audio_path": "models/vae/minimax_h3_audio_vae_fp32.safetensors",
            "output_dir": "output/mmh3_lora",
            "output_name": "h3_style",
        },
        "lora": {"rank": 16, "alpha": 16, "dropout": 0.0, "caption_dropout_rate": 0.0},
        "optimization": {
            "optimizer_type": "AdamW8bit",
            "learning_rate": 1e-4,
            "scheduler_type": "constant",
            "gradient_accumulation_steps": 1,
            "max_train_epochs": 10,
            "max_grad_norm": 1.0,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            "flash_attn": True,
            "gradient_checkpointing": True,
            "fp8_base": True,
            "blocks_to_swap": 40,
            "block_swap_h2d_only": True,
            "block_swap_ring_size": 2,
            "use_pinned_memory_for_block_swap": True,
        },
        "training_strategy": {"h3_training_mode": "fl2va"},
        "checkpoints": {"mode": "epochs", "interval": 1},
    }
    print_all_commands(sample_config, "dataset.toml")
