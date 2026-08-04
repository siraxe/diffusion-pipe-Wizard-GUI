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
    def _get_any(d: Dict, keys: List[str], default=None):
        """
        Get the first non-empty value from a dict using multiple possible keys.

        This is intentionally tolerant of:
        - trailing spaces in keys
        - case differences in keys
        - empty string values
        - None values
        """
        if not isinstance(d, dict):
            return default

        normalized = {}
        for k, v in d.items():
            if isinstance(k, str):
                normalized[k.strip().lower()] = v

        for key in keys:
            target = str(key).strip().lower()
            if target not in normalized:
                continue

            v = normalized[target]

            if v is None:
                continue

            if isinstance(v, str):
                v = v.strip()
                if not v:
                    continue

            return v

        return default

    @staticmethod
    def _get(d: Dict, key: str, default=None):
        """
        Backwards-compatible single-key getter.
        """
        return MMH3Run._get_any(d, [key], default)

    @staticmethod
    def parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value

        if value is None:
            return False

        return str(value).strip().lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        """
        Safe int conversion.

        Handles:
        - None
        - empty strings
        - floats such as 1.0
        - numeric strings such as "25"
        """
        if value is None:
            return default

        if isinstance(value, bool):
            return int(value)

        try:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return default

            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_ckpt_mode(value) -> str:
        """
        Normalize checkpoint mode.

        Accepted epoch-like values:
        - epoch
        - epochs
        - ep

        Accepted step-like values:
        - step
        - steps
        - iteration
        - iterations
        """
        raw = str(value or "steps").strip().lower()

        if raw in {"epoch", "epochs", "ep"}:
            return "epochs"

        if raw in {"step", "steps", "iteration", "iterations"}:
            return "steps"

        return "steps"

    @staticmethod
    def _find_section(config: Dict, name: str) -> Dict:
        """
        Find a config section by name, tolerating trailing spaces and case differences.

        Example:
            "checkpoints", "checkpoints ", "Checkpoints"
        """
        if not isinstance(config, dict):
            return {}

        target = str(name).strip().lower()

        for k, v in config.items():
            if isinstance(k, str) and k.strip().lower() == target:
                return v if isinstance(v, dict) else {}

        return {}

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
        """
        minimax_h3_cache_latents.py --dataset_config --vae --audio_vae
        """
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
        """
        minimax_h3_cache_text_encoder_outputs.py --dataset_config --text_encoder --tokenizer --task
        """
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

        if text_encoder_quantization:
            quant = str(text_encoder_quantization).strip().lower()
            if quant in ("int8", "nf4"):
                cmd.extend(["--text_encoder_quantization", quant])

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
        """
        Build the full accelerate launch training command for MiniMax H3.

        Expected config sections:
            model
            optimization
            acceleration
            training_strategy
            checkpoints
            lora
            validation

        All sections are optional except model.

        slider_config / reset_optimizer / reset_optimizer_params are accepted to
        match the LTX2Run/WAN22Run API; H3 doesn't use them yet.
        """
        model = self._find_section(config, "model")
        optimization = self._find_section(config, "optimization")
        acceleration = self._find_section(config, "acceleration")
        training_strategy = self._find_section(config, "training_strategy")
        checkpoints = self._find_section(config, "checkpoints")

        # Some UIs/config writers accidentally use singular section names.
        if not checkpoints:
            checkpoints = self._find_section(config, "checkpoint")

        lora = self._find_section(config, "lora")
        # Backward compat: some non-ltx2 musubi saves use [adapter] instead of [lora].
        if not lora:
            lora = self._find_section(config, "adapter")

        script = str(self.musubi_root / H3_TRAIN_SCRIPT)

        cmd: List[str] = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            script,
            "--dit", self._resolve_path(self._get(model, "model_path", "")),
            "--dataset_config", self._resolve_path(dataset_config),
            "--mixed_precision", str(
                self._get(acceleration, "mixed_precision_mode", DEFAULTS["mixed_precision_mode"])
            ),
        ]

        # ------------------------------------------------------------------
        # Attention backend: --flash_attn or --sdpa (default sdpa)
        # ------------------------------------------------------------------
        if self.parse_bool(self._get(acceleration, "flash_attn", False)):
            cmd.append("--flash_attn")
        else:
            cmd.append("--sdpa")

        # ------------------------------------------------------------------
        # Gradient checkpointing
        # ------------------------------------------------------------------
        if (
            self.parse_bool(self._get(optimization, "enable_gradient_checkpointing", True))
            or self.parse_bool(self._get(acceleration, "gradient_checkpointing", True))
        ):
            cmd.append("--gradient_checkpointing")

        # ------------------------------------------------------------------
        # fp8_base + H2D block swap options
        # ------------------------------------------------------------------
        if self.parse_bool(self._get(acceleration, "fp8_base", False)):
            cmd.append("--fp8_base")

        # blocks_to_swap may live in [optimization], [acceleration], or top-level.
        blocks_to_swap = self._to_int(
            self._get(
                optimization,
                "blocks_to_swap",
                self._get(
                    acceleration,
                    "blocks_to_swap",
                    self._get(config, "blocks_to_swap", 0),
                ),
            ),
            0,
        )

        if blocks_to_swap > 0:
            cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])
            cmd.append("--block_swap_h2d_only")

            ring_size = self._to_int(
                self._get(acceleration, "block_swap_ring_size", 2),
                2,
            )
            cmd.extend(["--block_swap_ring_size", str(ring_size)])

            if self.parse_bool(self._get(acceleration, "use_pinned_memory_for_block_swap", True)):
                cmd.append("--use_pinned_memory_for_block_swap")

            # Granularity:
            # block mode maxes at 48 of 50; layer mode can offload all 50.
            granularity = str(
                self._get(acceleration, "block_swap_granularity", "block")
            ).strip().lower()

            if granularity == "layer" or blocks_to_swap > 48:
                cmd.extend(["--block_swap_granularity", "layer"])

        # ------------------------------------------------------------------
        # H3-specific training mode
        # Map 'i2va' to 'fl2va' (same checkpoint, different conditioning packing)
        # ------------------------------------------------------------------
        h3_mode = self._get(training_strategy, "h3_training_mode", DEFAULTS["h3_training_mode"])
        if h3_mode:
            h3_mode_str = str(h3_mode).strip().lower()
            # i2va uses the same FL2VA checkpoint as t2va/fl2va
            if h3_mode_str == "i2va":
                h3_mode_str = "fl2va"
            cmd.extend(["--h3_training_mode", h3_mode_str])

        # Optional experimental guidance-distillation scale
        gds = self._get(training_strategy, "h3_guidance_distillation_scale", None)
        if gds is not None:
            gds_str = str(gds).strip().lower()
            if gds_str not in ("", "0", "0.0", "false", "none"):
                cmd.extend(["--h3_guidance_distillation_scale", str(gds).strip()])

        # ------------------------------------------------------------------
        # LoRA network configuration
        # ------------------------------------------------------------------
        rank = self._to_int(self._get(lora, "rank", DEFAULTS["rank"]), DEFAULTS["rank"])
        alpha = self._to_int(self._get(lora, "alpha", DEFAULTS["alpha"]), DEFAULTS["alpha"])

        cmd.extend([
            "--network_module", "networks.lora_minimax_h3",
            "--network_dim", str(rank),
            "--network_alpha", str(alpha),
        ])

        # Optional network_args: dropout
        network_args = []

        dropout = self._get(lora, "dropout", self._get(lora, "network_dropout", 0.0))
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

        # ------------------------------------------------------------------
        # Optimizer + LR + scheduler
        # ------------------------------------------------------------------
        opt_type = self._get(optimization, "optimizer_type", DEFAULTS["optimizer_type"])
        learning_rate = self._get(optimization, "learning_rate", DEFAULTS["learning_rate"])
        scheduler = str(self._get(optimization, "scheduler_type", DEFAULTS["scheduler_type"])).strip().lower()
        grad_accumulation = self._to_int(
            self._get(optimization, "gradient_accumulation_steps", DEFAULTS["gradient_accumulation_steps"]),
            DEFAULTS["gradient_accumulation_steps"],
        )

        cmd.extend([
            "--optimizer_type", str(opt_type),
            "--learning_rate", str(learning_rate),
            "--lr_scheduler", scheduler,
            "--gradient_accumulation_steps", str(grad_accumulation),
        ])

        # Warmup steps — required when scheduler is constant_with_warmup / cosine_with_warmup
        if "warmup" in scheduler:
            lr_warmup_steps = self._to_int(
                self._get(optimization, "lr_warmup_steps", 50),
                50,
            )
            cmd.extend(["--lr_warmup_steps", str(lr_warmup_steps)])

        max_grad_norm = self._get(optimization, "max_grad_norm", DEFAULTS["max_grad_norm"])
        try:
            if float(max_grad_norm) > 0:
                cmd.extend(["--max_grad_norm", str(max_grad_norm)])
        except (TypeError, ValueError):
            pass

        # ------------------------------------------------------------------
        # Save state
        #
        # Fixed:
        # - read from checkpoints section
        # - fallback to top-level config
        # - support common aliases
        # - tolerate trailing spaces / boolean strings
        # ------------------------------------------------------------------
        save_state_keys = [
            "save_state",
            "save_state_enabled",
            "save_optimizer_state",
        ]

        save_state_value = self._get_any(checkpoints, save_state_keys, None)

        if save_state_value is None:
            save_state_value = self._get_any(config, save_state_keys, None)

        if save_state_value is None:
            save_state_value = self._get_any(optimization, save_state_keys, None)

        if save_state_value is None:
            save_state_value = self._get_any(acceleration, save_state_keys, False)

        if self.parse_bool(save_state_value):
            cmd.append("--save_state")

        # ------------------------------------------------------------------
        # Max train steps/epochs + save_every + keep_last_n
        #
        # Fixed:
        # - normalize mode
        # - read save_every_n_steps / save_every_n_epochs directly
        # - fallback to interval
        # - do not force 50 when an explicit valid value exists
        # - prefer correct max_train_epochs / max_steps based on mode
        # ------------------------------------------------------------------
        explicit_ckpt_mode = self._get_any(
            checkpoints,
            ["mode", "ckpt_mode", "checkpoint_mode", "save_mode"],
            None,
        )

        if explicit_ckpt_mode is None:
            # Infer mode if the user did not explicitly set one.
            if self._get_any(
                checkpoints,
                [
                    "save_every_n_epochs",
                    "save_every_n_epoch",
                    "save_every_epochs",
                    "save_every_epoch",
                ],
                None,
            ) is not None:
                ckpt_mode = "epochs"

            elif self._get_any(
                checkpoints,
                [
                    "save_every_n_steps",
                    "save_every_n_step",
                    "save_every_steps",
                    "save_every_step",
                ],
                None,
            ) is not None:
                ckpt_mode = "steps"

            elif (
                self._get_any(optimization, ["max_train_epochs", "max_epochs"], None) is not None
                and self._get_any(optimization, ["max_steps", "max_train_steps"], None) is None
            ):
                ckpt_mode = "epochs"

            # Backward compat: configs saved by build_toml_config_from_ui have no
            # [checkpoints] section and only top-level save_every_n_epochs / _steps.
            # Infer mode from which top-level key is present.
            elif self._get_any(
                config,
                ["save_every_n_epochs", "save_every_n_epoch"],
                None,
            ) is not None:
                ckpt_mode = "epochs"

            elif self._get_any(
                config,
                ["save_every_n_steps", "save_every_n_step"],
                None,
            ) is not None:
                ckpt_mode = "steps"

            else:
                ckpt_mode = "steps"
        else:
            ckpt_mode = self._normalize_ckpt_mode(explicit_ckpt_mode)

        if ckpt_mode == "epochs":
            interval_value = self._get_any(
                checkpoints,
                [
                    "save_every_n_epochs",
                    "save_every_n_epoch",
                    "save_every_epochs",
                    "save_every_epoch",
                    "save_every",
                    "interval",
                    # Fallbacks, in case the config only has step-style keys.
                    "save_every_n_steps",
                    "save_every_n_step",
                    "save_every_steps",
                    "save_every_step",
                ],
                None,
            )

            if interval_value is None:
                interval_value = self._get_any(
                    config,
                    [
                        "save_every_n_epochs",
                        "save_every_n_epoch",
                        "save_every_epochs",
                        "save_every_epoch",
                        "save_every",
                        "interval",
                    ],
                    None,
                )

            interval = self._to_int(interval_value, 50)

            # Epochs mode: only look for epoch-style totals. Don't fall back to
            # max_steps — that's a different unit and would produce
            # --max_train_epochs 2000 when the user actually has max_steps=2000.
            total_value = self._get_any(
                optimization,
                [
                    "max_train_epochs",
                    "max_epochs",
                    "epochs",
                ],
                None,
            )

            if total_value is None:
                total_value = self._get_any(
                    config,
                    [
                        "max_train_epochs",
                        "max_epochs",
                        "epochs",
                    ],
                    None,
                )

            if total_value is None:
                # No epoch-style total anywhere. Refuse to silently misinterpret
                # max_steps as epochs — fall back to a high default so the run
                # doesn't end prematurely. User can override via [checkpoints].mode.
                total_value = 100

            steps_or_epochs = self._to_int(total_value, 100)

            keep_value = self._get_any(
                checkpoints,
                [
                    "keep_last_n_epochs",
                    "keep_last_n_epoch",
                    "keep_last_n",
                    "keep_last",
                ],
                -1,
            )
            keep_last_n = self._to_int(keep_value, -1)

        else:
            interval_value = self._get_any(
                checkpoints,
                [
                    "save_every_n_steps",
                    "save_every_n_step",
                    "save_every_steps",
                    "save_every_step",
                    "save_every",
                    "interval",
                    # Fallbacks, in case the config only has epoch-style keys.
                    "save_every_n_epochs",
                    "save_every_n_epoch",
                    "save_every_epochs",
                    "save_every_epoch",
                ],
                None,
            )

            if interval_value is None:
                interval_value = self._get_any(
                    config,
                    [
                        "save_every_n_steps",
                        "save_every_n_step",
                        "save_every_steps",
                        "save_every_step",
                        "save_every",
                        "interval",
                    ],
                    None,
                )

            interval = self._to_int(interval_value, 50)

            # Steps mode: only look for step-style totals.
            total_value = self._get_any(
                optimization,
                [
                    "max_steps",
                    "max_train_steps",
                    "steps",
                ],
                None,
            )

            if total_value is None:
                total_value = self._get_any(
                    config,
                    [
                        "max_steps",
                        "max_train_steps",
                        "steps",
                    ],
                    None,
                )

            if total_value is None:
                # No step-style total anywhere. Don't silently use max_train_epochs.
                total_value = 2000

            steps_or_epochs = self._to_int(total_value, 2000)

            keep_value = self._get_any(
                checkpoints,
                [
                    "keep_last_n_steps",
                    "keep_last_n_step",
                    "keep_last_n",
                    "keep_last",
                ],
                -1,
            )
            keep_last_n = self._to_int(keep_value, -1)

        # Add checkpoint flags.
        #
        # If interval is explicitly 0 or negative, omit the save-every flag.
        # This allows users to disable periodic checkpointing without forcing 50.
        if ckpt_mode == "epochs":
            cmd.extend([
                "--max_train_epochs", str(steps_or_epochs),
            ])

            if interval > 0:
                cmd.extend([
                    "--save_every_n_epochs", str(interval),
                ])

            if keep_last_n > 0:
                cmd.extend([
                    "--save_last_n_epochs", str(keep_last_n),
                ])

        else:
            cmd.extend([
                "--max_train_steps", str(steps_or_epochs),
            ])

            if interval > 0:
                cmd.extend([
                    "--save_every_n_steps", str(interval),
                ])

            if keep_last_n > 0:
                cmd.extend([
                    "--save_last_n_steps", str(keep_last_n),
                ])

        # ------------------------------------------------------------------
        # Output dir/name
        # ------------------------------------------------------------------
        output_dir = self._get(
            config,
            "output_dir",
            self._get(model, "output_dir", DEFAULTS["output_dir"]),
        )

        output_name = self._get(
            config,
            "output_name",
            self._get(
                model,
                "output_name",
                self._get(model, "name", DEFAULTS["output_name"]),
            ),
        )

        cmd.extend([
            "--output_dir", str(output_dir),
            "--output_name", str(output_name),
            "--log_with", "tensorboard",
            "--logging_dir", os.path.join(str(output_dir), ".tensorboard"),
        ])

        # ------------------------------------------------------------------
        # Resume
        # ------------------------------------------------------------------
        if resume:
            resume_path = Path(resume).expanduser()
            if resume_path.exists():
                cmd.extend(["--resume", str(resume_path)])

        # ------------------------------------------------------------------
        # Extra top-level flags
        # ------------------------------------------------------------------
        extra = self._get(config, "extra_flags", "")
        if extra and str(extra).strip():
            cmd.extend(str(extra).strip().split())

        return cmd

    # ----------------------------------------------------------------------
    # Formatting + Printing
    # ----------------------------------------------------------------------

    @staticmethod
    def format_command(cmd: List[str]) -> str:
        """
        Pretty-print a command list as a backslash-continued multi-line shell string.
        """
        lines = []
        i = 0

        while i < len(cmd):
            arg = cmd[i]

            if (
                arg.startswith("--")
                and i + 1 < len(cmd)
                and not str(cmd[i + 1]).startswith("-")
            ):
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
        """
        Build and print all H3 commands (cache + train) to stdout.

        video_vae, audio_vae, and tokenizer fall back to model section keys
        (video_vae_path / audio_vae_path / tokenizer_path) if not passed explicitly.
        """
        model = self._find_section(config, "model")

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

def create_training_command(
    config: Dict,
    dataset_config: str,
    resume: Optional[str] = None,
) -> List[str]:
    return MMH3Run().build_training_command(config, dataset_config, resume=resume)


def format_training_command(
    config: Dict,
    dataset_config: str,
    resume: Optional[str] = None,
) -> str:
    return MMH3Run().format_command(
        create_training_command(config, dataset_config, resume=resume)
    )


def print_all_commands(
    config: Dict,
    dataset_config: str,
    video_vae: Optional[str] = None,
    audio_vae: Optional[str] = None,
    tokenizer: Optional[str] = None,
    resume: Optional[str] = None,
) -> None:
    MMH3Run().print_all_commands(
        config,
        dataset_config,
        video_vae,
        audio_vae,
        tokenizer,
        resume,
    )


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
        "lora": {
            "rank": 16,
            "alpha": 16,
            "dropout": 0.0,
            "caption_dropout_rate": 0.0,
        },
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
        "training_strategy": {
            "h3_training_mode": "fl2va",
        },
        "checkpoints": {
            "mode": "epochs",
            "interval": 1,
            "save_state": True,
            "keep_last_n": 3,
        },
    }

    print_all_commands(sample_config, "dataset.toml")