from __future__ import annotations

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

try:
    import safetensors.torch
except ImportError:
    safetensors = None

from .base import CommandBuilder

logger = logging.getLogger(__name__)


def is_comfy_format_lora(file_path: str) -> bool:
    """Check if a LoRA file is in ComfyUI format by examining the keys."""
    if safetensors is None:
        logger.warning("safetensors not available, cannot detect ComfyUI format")
        return False

    try:
        state_dict = safetensors.torch.load_file(file_path)
        if not state_dict:
            return False

        for key in list(state_dict.keys())[:5]:
            if key.startswith('diffusion_model.'):
                return True
            if key.startswith('lora_unet_model_'):
                return False

        for key in state_dict.keys():
            if '.lora_A.' in key or '.lora_B.' in key:
                return True

        return False
    except Exception as e:
        logger.warning(f"Error checking LoRA format for {file_path}: {e}")
        return False


def convert_comfy_to_training_with_rank(file_path: str, target_rank: int) -> Optional[str]:
    """Convert ComfyUI format LoRA to training format with optional rank conversion."""
    import subprocess

    try:
        input_file = Path(file_path).resolve()
        if not input_file.exists():
            logger.error(f"Source file does not exist: {file_path}")
            return None

        output_path = input_file.parent / f"{input_file.stem}_rank{target_rank}{input_file.suffix}"

        convert_script = Path(__file__).parent.parent.parent / 'scripts' / 'convert_comfy_to_training_lora.py'

        if not convert_script.exists():
            logger.error(f"Conversion script not found: {convert_script}")
            return None

        logger.info(f"Converting ComfyUI LoRA with rank conversion: {file_path} -> rank {target_rank}")

        cmd = [
            str(sys.executable),
            str(convert_script),
            str(input_file),
            '--target_rank', str(target_rank),
            '-o', str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode == 0:
            logger.info(f"Successfully created converted checkpoint: {output_path}")
            return str(output_path)
        else:
            logger.error(f"Conversion failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("Conversion timed out after 3 minutes")
        return None
    except Exception as e:
        logger.error(f"Failed to convert ComfyUI LoRA {file_path}: {e}")
        return None

H3_TRAIN_SCRIPT = "minimax_h3_train_network.py"
H3_SLIDER_TRAIN_SCRIPT = "minimax_h3_train_slider.py"

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


class MMH3Run(CommandBuilder):
    """Builds MiniMax H3 training commands (no execution)."""

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _get_any(d: Dict, keys: List[str], default=None):
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
        raw = str(value or "steps").strip().lower()

        if raw in {"epoch", "epochs", "ep"}:
            return "epochs"

        if raw in {"step", "steps", "iteration", "iterations"}:
            return "steps"

        return "steps"

    @staticmethod
    def _extra_flag_present(config: Dict, *flag_names: str) -> bool:
        """Detect any of --flag or --flag=... in the top-level extra_flags string."""
        extra = str(config.get("extra_flags", "") or "").strip() if isinstance(config, dict) else ""
        if not extra:
            return False
        tokens = extra.split()
        return any(
            t == flag_name or t.startswith(f"{flag_name}=")
            for flag_name in flag_names
            for t in tokens
        )

    @staticmethod
    def _find_section(config: Dict, name: str) -> Dict:
        if not isinstance(config, dict):
            return {}

        target = str(name).strip().lower()

        for k, v in config.items():
            if isinstance(k, str) and k.strip().lower() == target:
                return v if isinstance(v, dict) else {}

        return {}

    # ==========================================================================
    # LoRA Rank Detection & Conversion Helpers
    # ==========================================================================

    def get_lora_rank(self, file_path: str) -> int:
        """Detect the rank of a LoRA checkpoint."""
        if safetensors is None:
            logger.warning("safetensors not available, cannot detect LoRA rank")
            return 0

        try:
            state_dict = safetensors.torch.load_file(file_path)
            if not state_dict:
                return 0

            for key in state_dict.keys():
                if key.endswith('.lora_down.weight'):
                    return state_dict[key].shape[0]
                elif key.endswith('.lora_A.weight'):
                    return state_dict[key].shape[0]
                elif key.endswith('.lokr_w1_b'):
                    return state_dict[key].shape[1]
                elif key.endswith('.lokr_w2_a'):
                    return state_dict[key].shape[0]
                elif key.endswith('.lokr_w1_a') or key.endswith('.lokr_w2_b'):
                    shape = state_dict[key].shape
                    return min(shape)

            return 0
        except Exception as e:
            logger.warning(f"Error detecting LoRA rank for {file_path}: {e}")
            return 0

    def rerank_training_format_lora(self, file_path: str, target_rank: int) -> Optional[str]:
        """Rerank a training format LoRA checkpoint to a different rank."""
        import subprocess

        try:
            input_file = Path(file_path).resolve()
            if not input_file.exists():
                logger.error(f"Source file does not exist: {file_path}")
                return None

            output_path = input_file.parent / f"{input_file.stem}_rank{target_rank}{input_file.suffix}"

            sys_path = self.project_root / 'scripts'
            rerank_script = sys_path / "rerank_lora.py"

            if not rerank_script.exists():
                logger.error(f"Reranking script not found: {rerank_script}")
                return None

            cmd = [
                str(sys.executable),
                str(rerank_script),
                str(input_file),
                '--target_rank', str(target_rank),
                '-o', str(output_path),
                '--device', 'cuda'
            ]

            logger.info(f"Reranking checkpoint: {file_path} -> rank {target_rank}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                logger.info(f"Successfully reranked checkpoint: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Reranking failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Reranking timed out after 3 minutes")
            return None
        except Exception as e:
            logger.error(f"Failed to rerank LoRA {file_path}: {e}")
            return None

    def _build_initialization(self, config: Dict, lora: Dict) -> List[str]:
        """Builds flags for initializing from existing checkpoints."""
        cmd = []

        init_checkpoint = lora.get('init_from_existing', config.get('init_from_existing', ''))
        if not (init_checkpoint and str(init_checkpoint).lower() not in ('', 'null', 'none')):
            return cmd

        # Path resolution
        if not os.path.isabs(init_checkpoint):
            init_checkpoint = str(self.project_root / init_checkpoint)

        # Check if path is a directory and try to find .safetensors file inside
        if os.path.isdir(init_checkpoint):
            dir_path = init_checkpoint
            safetensors_files = [f for f in os.listdir(dir_path) if f.endswith('.safetensors')]
            if safetensors_files:
                init_checkpoint = str(Path(dir_path) / safetensors_files[0])
                logger.info(f"Directory detected, using found safetensors file: {init_checkpoint}")
            else:
                logger.warning(f"Directory detected but no .safetensors file found inside: {dir_path}")

        target_rank = lora.get('rank', DEFAULTS['rank'])

        if os.path.exists(init_checkpoint):
            # Check 1: ComfyUI format detection (convert regardless of rank)
            if is_comfy_format_lora(init_checkpoint):
                logger.info(f"Detected ComfyUI format checkpoint, converting to training format (rank {target_rank})")
                converted_path = convert_comfy_to_training_with_rank(init_checkpoint, target_rank)

                if converted_path:
                    init_checkpoint = converted_path
                    logger.info(f"Using converted checkpoint: {init_checkpoint}")
                else:
                    logger.warning(f"ComfyUI conversion failed, using original checkpoint (may cause errors)")
            # Check 2: Rank mismatch (only for training format)
            else:
                checkpoint_rank = self.get_lora_rank(init_checkpoint)

                if checkpoint_rank > 0 and checkpoint_rank != target_rank:
                    logger.info(f"Rank mismatch detected, reranking from {checkpoint_rank} to {target_rank}")
                    converted_path = self.rerank_training_format_lora(init_checkpoint, target_rank)

                    if converted_path:
                        init_checkpoint = converted_path
                        logger.info(f"Using converted checkpoint: {init_checkpoint}")
                    else:
                        logger.warning(f"Reranking failed, using original checkpoint (may cause errors)")
                else:
                    logger.info(f"Checkpoint rank {checkpoint_rank} matches target rank {target_rank}")
        else:
            logger.warning(f"Checkpoint file does not exist: {init_checkpoint}")

        cmd.extend(["--network_weights", init_checkpoint])
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

        dit_path = self._resolve_path(self._get(model, "model_path", ""))
        dit_name = Path(dit_path).name.lower()
        is_int8_convrot = "int8" in dit_name or "convrot" in dit_name

        mixed_precision = str(
            self._get(acceleration, "mixed_precision_mode", DEFAULTS["mixed_precision_mode"])
        )

        cmd: List[str] = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            script,
            "--mixed_precision", mixed_precision,
            "--dit", dit_path,
            "--dataset_config", self._resolve_path(dataset_config),
        ]

        # ------------------------------------------------------------------
        # Base weights: only LoRA adapter files; the H3 training script
        # rejects full checkpoints here (expects lora_unet_* keys).
        # ------------------------------------------------------------------
        adapter_path = str(self._get(model, "adapter", "") or "").strip()
        if adapter_path and adapter_path.lower() not in ("null", "none"):
            cmd.extend(["--base_weights", self._resolve_path(adapter_path)])

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

        if is_int8_convrot:
            cmd.append("--int8_convrot_base")
        elif self.parse_bool(self._get(acceleration, "fp8_base", False)):
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
            # i2va and t2va use the same FL2VA checkpoint
            if h3_mode_str in ("i2va", "t2va"):
                h3_mode_str = "fl2va"
            cmd.extend(["--h3_training_mode", h3_mode_str])

            # Ref2VA with a plain BF16 checkpoint: apply the doc-recommended
            # frozen-base reductions. Skipped for the pre-quantized ConvRot
            # checkpoint or when the user supplied the flags via extra_flags.
            if (
                h3_mode_str in ("ref2va", "ref2va_omni")
                and not is_int8_convrot
            ):
                has_fp8 = self.parse_bool(self._get(acceleration, "fp8_base", False))
                has_user_flags = self._extra_flag_present(config, "--h3_convrot_int8", "--h3_convrot_int8_fwd", "--h3_adaln_rank")

                # --h3_adaln_rank 16: compact AdaLN projections (~13B->~77M params).
                # Compatible with --fp8_base (replaces FP8's AdaLN quantization).
                if not has_user_flags:
                    cmd.extend(["--h3_adaln_rank", "16"])
                    logger.info("Ref2VA with BF16 base: added --h3_adaln_rank 16")

                # --h3_convrot_int8 --h3_convrot_int8_fwd bf16: quantize ConvRot
                # at load, keep BF16 forward. Only when NOT using --fp8_base
                # (they are alternative base-weight strategies).
                if (
                    not has_fp8
                    and not self._extra_flag_present(config, "--h3_convrot_int8", "--h3_convrot_int8_fwd")
                ):
                    cmd.extend(["--h3_convrot_int8", "--h3_convrot_int8_fwd", "bf16"])
                    logger.info("Ref2VA with BF16 base: added --h3_convrot_int8 --h3_convrot_int8_fwd bf16")

        # Optional experimental guidance-distillation scale
        gds = self._get(training_strategy, "h3_guidance_distillation_scale", None)
        if gds is not None:
            gds_str = str(gds).strip().lower()
            if gds_str not in ("", "0", "0.0", "false", "none"):
                cmd.extend(["--h3_guidance_distillation_scale", str(gds).strip()])

        # ------------------------------------------------------------------
        # Slider training: --slider when h3_slider = true (or t_type = 'slider')
        # ------------------------------------------------------------------
        h3_slider = self._get(training_strategy, "h3_slider", self._get(training_strategy, "slider", False))
        t_type = str(self._get(training_strategy, "t_type", "")).strip().lower()
        if self.parse_bool(h3_slider) or t_type == "slider":
            cmd.append("--slider")

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
        # Init from existing checkpoint
        # ------------------------------------------------------------------
        init_flags = self._build_initialization(config, lora)
        cmd.extend(init_flags)

        # ------------------------------------------------------------------
        # Optimizer + LR + scheduler
        # ------------------------------------------------------------------
        opt_type = self._get(optimization, "optimizer_type", DEFAULTS["optimizer_type"])
        if str(opt_type).strip().lower() == 'automagic':
            # musubi-tuner's H3 trainer expects 'automagic3' (its bundled
            # Automagic3 optimizer), not the upstream 'automagic' package.
            opt_type = 'automagic3'
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

        save_state_value = self._get(checkpoints, "save_state", None)

        if save_state_value is None:
            save_state_value = self._get(config, "save_state", None)

        if save_state_value is None:
            save_state_value = self._get(optimization, "save_state", None)

        if save_state_value is None:
            save_state_value = self._get(acceleration, "save_state", False)

        if self.parse_bool(save_state_value):
            cmd.append("--save_state")

        explicit_ckpt_mode = self._get(checkpoints, "mode", None)

        if explicit_ckpt_mode is None:
            # Infer mode if the user did not explicitly set one.
            if self._get(
                checkpoints,
                "save_every_n_epochs",
                None,
            ) is not None:
                ckpt_mode = "epochs"

            elif self._get(
                checkpoints,
                "save_every_n_steps",
                None,
            ) is not None:
                ckpt_mode = "steps"

            elif (
                self._get(optimization, "max_train_epochs", None) is not None
                and self._get(optimization, "max_steps", None) is None
            ):
                ckpt_mode = "epochs"

            elif self._get(
                config,
                "save_every_n_epochs",
                None,
            ) is not None:
                ckpt_mode = "epochs"

            elif self._get(
                config,
                "save_every_n_steps",
                None,
            ) is not None:
                ckpt_mode = "steps"

            else:
                ckpt_mode = "steps"
        else:
            ckpt_mode = self._normalize_ckpt_mode(explicit_ckpt_mode)

        if ckpt_mode == "epochs":
            interval_value = self._get(checkpoints, "interval", None)

            if interval_value is None:
                interval_value = self._get(checkpoints, "save_every_n_epochs", None)

            if interval_value is None:
                interval_value = self._get(config, "save_every_n_epochs", None)

            interval = self._to_int(interval_value, 50)

            total_value = self._get(optimization, "max_train_epochs", None)

            if total_value is None:
                total_value = self._get(config, "max_train_epochs", None)

            if total_value is None:
                total_value = 100

            steps_or_epochs = self._to_int(total_value, 100)

            keep_value = self._get(checkpoints, "keep_last_n", -1)
            if keep_value == -1:
                keep_value = self._get(checkpoints, "keep_last_n_epochs", -1)
            keep_last_n = self._to_int(keep_value, -1)

        else:
            interval_value = self._get(checkpoints, "interval", None)

            if interval_value is None:
                interval_value = self._get(checkpoints, "save_every_n_steps", None)

            if interval_value is None:
                interval_value = self._get(config, "save_every_n_steps", None)

            interval = self._to_int(interval_value, 50)

            # Steps mode: only look for step-style totals.
            total_value = self._get(optimization, "max_steps", None)

            if total_value is None:
                total_value = self._get(config, "max_steps", None)

            if total_value is None:
                # No step-style total anywhere. Don't silently use max_train_epochs.
                total_value = 2000

            steps_or_epochs = self._to_int(total_value, 2000)

            keep_value = self._get(checkpoints, "keep_last_n", -1)
            if keep_value == -1:
                keep_value = self._get(checkpoints, "keep_last_n_steps", -1)
            keep_last_n = self._to_int(keep_value, -1)

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
            tokens = str(extra).strip().split()
            # Cache-only flags consumed by mmh3_cache; the training script
            # does not accept them.
            cache_only = {"--cache_guidance_empty"}
            cmd.extend(t for t in tokens if t not in cache_only)

        # ------------------------------------------------------------------
        # H3 txt slider training: adapt this command for
        # minimax_h3_train_slider.py. No dataset caching is needed — the
        # prompts/latents come from the slider TOML itself.
        # ------------------------------------------------------------------
        txt_slider_active = (
            slider_config is not None
            and str(self._get(training_strategy, "h3_training_mode", "")).strip().lower() == "txt_slider"
        )
        if txt_slider_active:
            normal_script = str(self.musubi_root / H3_TRAIN_SCRIPT)
            if normal_script in cmd:
                cmd[cmd.index(normal_script)] = str(self.musubi_root / H3_SLIDER_TRAIN_SCRIPT)

            # The slider trainer uses the slider TOML as its dataset config.
            if "--dataset_config" in cmd:
                idx = cmd.index("--dataset_config")
                del cmd[idx:idx + 2]

            cmd.extend(["--slider_config", self._resolve_path(slider_config)])

            # Text slider mode requires --h3_training_mode fl2va.
            if "--h3_training_mode" in cmd:
                cmd[cmd.index("--h3_training_mode") + 1] = "fl2va"
            else:
                cmd.extend(["--h3_training_mode", "fl2va"])

            # Text encoder for prompt encoding (required in text mode).
            if "--text_encoder" not in cmd:
                te_path = str(self._get(model, "text_encoder_path", "") or "").strip()
                if te_path and te_path.lower() not in ("null", "none"):
                    cmd.extend(["--text_encoder", self._resolve_path(te_path)])
                tok_path = str(self._get(model, "tokenizer_path", "") or "").strip()
                if tok_path and tok_path.lower() not in ("null", "none"):
                    cmd.extend(["--tokenizer", self._resolve_path(tok_path)])
                te_quant = str(self._get(model, "text_encoder_quantization", "nvfp4") or "nvfp4").strip().lower()
                if te_quant and te_quant != "none":
                    cmd.extend(["--text_encoder_quantization", te_quant])
                blocks_to_stream = self._to_int(
                    self._get(model, "h3_text_encoder_blocks_to_stream", 50), 50
                )
                if blocks_to_stream > 0:
                    cmd.extend(["--h3_text_encoder_blocks_to_stream", str(blocks_to_stream)])

            # Frozen-base preservation loss (recommended for txt slider mode).
            if not self._extra_flag_present(config, "--h3_base_preservation_loss_weight"):
                preserve_weight = self._get(training_strategy, "h3_base_preservation_loss_weight", 0.02)
                try:
                    if float(preserve_weight) > 0:
                        cmd.extend(["--h3_base_preservation_loss_weight", str(preserve_weight)])
                        preserve_prob = self._get(training_strategy, "h3_base_preservation_probability", 0.25)
                        cmd.extend(["--h3_base_preservation_probability", str(preserve_prob)])
                except (TypeError, ValueError):
                    pass

            logger.info("H3 txt slider mode: using %s with %s", H3_SLIDER_TRAIN_SCRIPT, slider_config)

        return cmd

    # ----------------------------------------------------------------------
    # Formatting + Printing
    # ----------------------------------------------------------------------

    @staticmethod
    def format_command(cmd: List[str]) -> str:
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


if __name__ == "__main__":
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

    runner = MMH3Run()
    cmd = runner.build_training_command(sample_config, "dataset.toml")
    print(runner.format_command(cmd))