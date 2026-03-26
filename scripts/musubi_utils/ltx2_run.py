"""
LTX-Video-2 Training Command Builder

This module handles building commands for LTX-Video-2 training operations.
Refactored for maintainability, DRY compliance, and robustness.
"""

import os
import subprocess
import sys
import logging
import hashlib
from typing import Dict, List, Optional
from pathlib import Path

try:
    import safetensors.torch
except ImportError:
    safetensors = None

logger = logging.getLogger(__name__)


# ==========================================================================
# ComfyUI Format Detection and Conversion
# ==========================================================================

def is_comfy_format_lora(file_path: str) -> bool:
    """
    Check if a LoRA file is in ComfyUI format by examining the keys.

    ComfyUI format keys start with 'diffusion_model.'
    Training format keys start with 'lora_unet_model_'
    """
    if safetensors is None:
        logger.warning("safetensors not available, cannot detect ComfyUI format")
        return False

    try:
        state_dict = safetensors.torch.load_file(file_path)
        if not state_dict:
            return False

        # Check first few keys
        for key in list(state_dict.keys())[:5]:
            if key.startswith('diffusion_model.'):
                return True
            if key.startswith('lora_unet_model_'):
                return False

        # If no clear prefix, check for other ComfyUI patterns
        for key in state_dict.keys():
            if '.lora_A.' in key or '.lora_B.' in key:
                return True

        return False
    except Exception as e:
        logger.warning(f"Error checking LoRA format for {file_path}: {e}")
        return False


def convert_comfy_to_training_with_rank(file_path: str, target_rank: int) -> Optional[str]:
    """
    Convert ComfyUI format LoRA to training format with optional rank conversion.

    Returns the path to the new file, or None if failed.
    """
    import subprocess
    from pathlib import Path

    try:
        input_file = Path(file_path).resolve()
        if not input_file.exists():
            logger.error(f"Source file does not exist: {file_path}")
            return None

        output_path = input_file.parent / f"{input_file.stem}_rank{target_rank}{input_file.suffix}"

        # Path to conversion script
        # ltx2_run.py is at scripts/musubi_utils/ltx2_run.py
        # convert script is at scripts/convert_comfy_to_training_lora.py
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

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
        )

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


# Constants for CLI flags to prevent typos and centralize configuration
CONFIG_FLAGS = {
    "MIXED_PRECISION": "--mixed_precision",
    "GRADIENT_CHECKPOINTING": "--gradient_checkpointing",
    "FP8_BASE": "--fp8_base",
    "FP8_SCALED": "--fp8_scaled",
    "GEMMA_8BIT": "--gemma_load_in_8bit",
    "SPLIT_ATTN_TARGET": "--split_attn_target",
    "SPLIT_ATTN_MODE": "--split_attn_mode",
    "SPLIT_ATTN_CHUNK_SIZE": "--split_attn_chunk_size",
    "SAVE_STATE": "--save_state",
    "ENABLE_MASK": "--ltx2_enable_mask",
    "RESUME": "--resume",
    "SAVE_EVERY_N_STEPS": "--save_every_n_steps",
    "SAVE_EVERY_N_EPOCHS": "--save_every_n_epochs",
    "SAVE_LAST_N_STEPS": "--save_last_n_steps",
    "SAVE_LAST_N_EPOCHS": "--save_last_n_epochs",
    "MAX_TRAIN_STEPS": "--max_train_steps",
    "MAX_TRAIN_EPOCHS": "--max_train_epochs",
    "GRAD_ACCUMULATION": "--gradient_accumulation_steps",
    "LEARNING_RATE": "--learning_rate",
    "AUDIO_LR": "--audio_lr",
    "OPTIMIZER_TYPE": "--optimizer_type",
    "LR_SCHEDULER": "--lr_scheduler",
    "TIMESTAMP_SAMPLING": "--timestep_sampling",
    "LR_WARMUP_STEPS": "--lr_warmup_steps",
    "MAX_GRAD_NORM": "--max_grad_norm",
    "BLOCKS_TO_SWAP": "--blocks_to_swap",
    "OPTIMIZER_ARGS": "--optimizer_args",
    "CAPTION_DROPOUT_RATE": "--caption_dropout_rate",
    "OUTPUT_DIR": "--output_dir",
    "OUTPUT_NAME": "--output_name",
    "LOG_WITH": "--log_with",
    "LOGGING_DIR": "--logging_dir",
    "GEMMA_ROOT": "--gemma_root",
    "LTX2_CHECKPOINT": "--ltx2_checkpoint",
    "LTX2_MODE": "--ltx2_mode",
    "LTX_VERSION": "--ltx_version",
    "FIRST_FRAME_P": "--ltx2_first_frame_conditioning_p",
    "LORA_TARGET_PRESET": "--lora_target_preset",
    "SEPARATE_AUDIO_BUCKETS": "--separate_audio_buckets",
    "NETWORK_WEIGHTS": "--network_weights",
    "NETWORK_MODULE": "--network_module",
    "NETWORK_DIM": "--network_dim",
    "NETWORK_ALPHA": "--network_alpha",
    "INIT_LOKR_NORM": "--init_lokr_norm",
    "NETWORK_ARGS": "--network_args",
    "SAMPLE_INTERVAL": "--sample_every_n_steps", # Default to steps, logic handles epochs
    "SAMPLE_AT_FIRST": "--sample_at_first",
    "WIDTH": "--width",
    "HEIGHT": "--height",
    "NUM_FRAMES": "--sample_num_frames",
    "MERGE_AUDIO": "--sample_merge_audio",
    "OFFLOADING": "--sample_with_offloading",
    "TILED_VAE": "--sample_tiled_vae",
    "VAE_TILE_SIZE": "--sample_vae_tile_size",
    "VAE_TILE_OVERLAP": "--sample_vae_tile_overlap",
    "VAE_TEMPORAL_SIZE": "--sample_vae_temporal_tile_size",
    "VAE_TEMPORAL_OVERLAP": "--sample_vae_temporal_tile_overlap",
    "SAMPLE_PROMPTS": "--sample_prompts",
    "USE_PRECACHED_PROMPTS": "--use_precached_sample_prompts",
    "SAMPLE_CACHE": "--sample_prompts_cache",
    "LATENTS_CACHE": "--sample_latents_cache",
    # Preservation Flags
    "BLANK_PRESERVATION": "--blank_preservation",
    "BLANK_ARGS": "--blank_preservation_args",
    "DOP": "--dop",
    "DOP_ARGS": "--dop_args",
    "PRIOR_DIV": "--prior_divergence",
    "PRIOR_DIV_ARGS": "--prior_divergence_args",
    "CREPA": "--crepa",
    "CREPA_ARGS": "--crepa_args",
    "SELF_FLOW": "--self_flow",
    "SELF_FLOW_ARGS": "--self_flow_args",
    "CTS_LAMBDA_VIDEO_DRIVEN": "--cts_lambda_video_driven",
    "CTS_LAMBDA_AUDIO_DRIVEN": "--cts_lambda_audio_driven",
    "AUDIO_LOSS_BALANCE_MODE": "--audio_loss_balance_mode",
    "AUDIO_LOSS_BALANCE_TARGET_RATIO": "--audio_loss_balance_target_ratio",
    "AUDIO_LOSS_BALANCE_EMA_DECAY": "--audio_loss_balance_ema_decay",
}

DEFAULTS = {
    "learning_rate": 0.001,
    "max_grad_norm": 1.0,
    "blocks_to_swap": 0,
    "gradient_accumulation_steps": 4,
    "scheduler_type": "constant",
    "timestep_sampling_mode": "shifted_logit_normal",
    "output_dir": "output/ltx2_lora",
    "ltx_mode": "video",
    "first_frame_conditioning_p": 0.1,
    "rank": 64,
    "alpha": 64,
    "factor": 4,
    "sample_interval": -1,
    "sample_prompts_path": None, # Computed dynamically
}

class LTX2Run:
    """Builds commands for LTX-Video-2 training operations."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.musubi_root = self.project_root / "diffusion-trainers" / "musubi-tuner" / "src" / "musubi_tuner"
        # Cache configuration map for helper methods
        self.config_map = {} 

    @staticmethod
    def _find_project_root() -> Path:
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "flet_app").exists() or (parent / "diffusion-trainers").exists():
                return parent
        return Path.cwd()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to project root."""
        p = Path(path)
        if not p.is_absolute():
            return self.project_root / p
        return p

    # ==========================================================================
    # Hash-based Cache Validation
    # ==========================================================================

    def _compute_sample_cache_hash(self, validation: Dict) -> str:
        """Compute a hash of the sampling configuration to detect changes."""
        hasher = hashlib.sha256()

        # Hash all relevant validation parameters
        hash_fields = [
            'prompts', 'negative_prompt', 'video_dims', 'sample_steps',
            'guidance_scale', 'seed', 'start_images', 'interval'
        ]

        for field in hash_fields:
            value = validation.get(field)
            if value is not None:
                # For file paths, also hash the file contents
                if field == 'start_images' and value:
                    img_path = Path(value)
                    if not img_path.is_absolute():
                        img_path = self.project_root / img_path
                    if img_path.exists():
                        with open(img_path, 'rb') as f:
                            hasher.update(f.read())
                        hasher.update(str(img_path).encode())
                    else:
                        hasher.update(str(img_path).encode())
                else:
                    hasher.update(str(value).encode())

        return hasher.hexdigest()

    def _get_sample_cache_hash_file(self, output_dir: str) -> Path:
        """Get the path to the hash file for sample cache validation."""
        return Path(output_dir) / 'sample' / '.sample_cache_hash'

    def _should_rebuild_sample_cache(self, validation: Dict, output_dir: str) -> bool:
        """Check if sample cache needs to be rebuilt based on hash comparison."""
        hash_file = self._get_sample_cache_hash_file(output_dir)
        cache_path = Path(output_dir) / 'sample' / 'sample_prompts_cache.pt'

        # If cache doesn't exist, need to build
        if not cache_path.exists():
            return True

        # If hash file doesn't exist, need to build
        if not hash_file.exists():
            return True

        # Read stored hash and compare
        try:
            current_hash = self._compute_sample_cache_hash(validation)
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()

            if current_hash != stored_hash:
                logger.info(f"Sample config changed, cache will be rebuilt")
                return True

            logger.info(f"Sample config unchanged, using existing cache")
            return False
        except Exception as e:
            logger.warning(f"Error reading cache hash file: {e}, rebuilding cache")
            return True

    def _save_sample_cache_hash(self, validation: Dict, output_dir: str) -> None:
        """Save the hash of the current sampling configuration."""
        hash_file = self._get_sample_cache_hash_file(output_dir)
        hash_file.parent.mkdir(parents=True, exist_ok=True)

        current_hash = self._compute_sample_cache_hash(validation)
        with open(hash_file, 'w') as f:
            f.write(current_hash)

        logger.info(f"Saved sample cache hash: {current_hash[:16]}...")

    # ==========================================================================
    # Config Parsing Helpers
    # ==========================================================================

    def get_config_value(self, config: Dict, *keys, default=None):
        """Get a nested value from config dictionary."""
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        return config

    def parse_bool(self, value) -> bool:
        """Convert various input types to boolean."""
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'on')

    def parse_optimizer_args(self, optimizer_args: str) -> List[str]:
        if not optimizer_args:
            return []
        result = []
        current = []
        paren_depth = 0
        for char in optimizer_args:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == ',' and paren_depth == 0:
                arg = ''.join(current).strip()
                if arg:
                    result.append(arg.replace(';', ','))
                current = []
            else:
                current.append(char)
        if current:
            arg = ''.join(current).strip()
            if arg:
                result.append(arg.replace(';', ','))
        return result

    # ==========================================================================
    # Command Building Helpers (DRY)
    # ==========================================================================

    def _add_bool_flag(self, cmd: List[str], config_key: str, config_dict: Dict, flag_name: str):
        """Add a boolean flag if the condition is met."""
        if self.parse_bool(config_dict.get(config_key, False)):
            cmd.append(flag_name)

    def _add_value_flag(self, cmd: List[str], config_key: str, config_dict: Dict, default_val, 
                        flag_prefix: str = "--", convert_fn=str):
        """Add a value flag if the value is not None/empty/default."""
        val = self.get_config_value(config_dict, config_key)
        # Check if value exists and is not equal to default (for numeric types) or empty
        if val is not None:
            try:
                if convert_fn == int:
                    val_int = int(val)
                    if val_int != 0: # Assuming 0 is often a "no-op" for counts
                        cmd.extend([f"{flag_prefix}{config_key}", str(val_int)])
                elif val: # Non-empty string or non-zero float
                     cmd.extend([f"{flag_prefix}{config_key}", convert_fn(val)])
            except (ValueError, TypeError):
                logger.warning(f"Could not parse value for {config_key}: {val}")

    def _add_args_flag(self, cmd: List[str], config_key: str, config_dict: Dict, flag_name: str):
        """Add a list of arguments parsed from a string."""
        args_str = self.get_config_value(config_dict, config_key)
        if args_str:
            parsed = self.parse_optimizer_args(args_str)
            if parsed:
                cmd.append(flag_name)
                cmd.extend(parsed)

    # ==========================================================================
    # LoRA Rank Detection & Conversion Helpers
    # ==========================================================================

    def get_lora_rank(self, file_path: str) -> int:
        """Detect the rank of a LoRA/LoKR checkpoint."""
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

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180, 
            )

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

    # ==========================================================================
    # Command Building Logic (Modularized)
    # ==========================================================================

    def _build_base_flags(self, script: str, acceleration: Dict, model: Dict,
                          config_flag: str, config_path: str) -> List[str]:
        """Builds the base accelerate launch command."""
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            script,
            CONFIG_FLAGS["MIXED_PRECISION"], acceleration.get('mixed_precision_mode', 'bf16'),
            config_flag, str(config_path),
            CONFIG_FLAGS["GEMMA_ROOT"], model.get('text_encoder_path', ''),
            CONFIG_FLAGS["LTX2_CHECKPOINT"], model.get('model_path', ''),
            "--flash_attn",
        ]

        output_name = model.get('output_name', model.get('name', 'ltx2_lora'))
        if output_name:
            cmd.extend([CONFIG_FLAGS["OUTPUT_NAME"], output_name])

        return cmd

    def _build_optimization_flags(self, optimization: Dict, acceleration: Dict) -> List[str]:
        """Builds flags for optimization settings."""
        cmd = []

        self._add_bool_flag(cmd, 'enable_gradient_checkpointing', optimization, CONFIG_FLAGS["GRADIENT_CHECKPOINTING"])
        self._add_bool_flag(cmd, 'fp8_base', acceleration, CONFIG_FLAGS["FP8_BASE"])
        self._add_bool_flag(cmd, 'fp8_scaled', acceleration, CONFIG_FLAGS["FP8_SCALED"])
        self._add_bool_flag(cmd, '8_bit_te', acceleration, CONFIG_FLAGS["GEMMA_8BIT"])

        if self.parse_bool(optimization.get('attn_chunking', False)):
            cmd.extend([CONFIG_FLAGS["SPLIT_ATTN_TARGET"], "video",
                        CONFIG_FLAGS["SPLIT_ATTN_MODE"], "query",
                        CONFIG_FLAGS["SPLIT_ATTN_CHUNK_SIZE"], "512"])

        # Warmup steps for constant_with_warmup
        if optimization.get('scheduler_type') == 'constant_with_warmup':
            lr_warmup_steps = optimization.get('lr_warmup_steps', 50)
            cmd.extend([CONFIG_FLAGS["LR_WARMUP_STEPS"], str(lr_warmup_steps)])

        # Max grad norm
        max_grad_norm = optimization.get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            cmd.extend([CONFIG_FLAGS["MAX_GRAD_NORM"], str(max_grad_norm)])

        # Block swap
        blocks_to_swap = optimization.get('blocks_to_swap', 0)
        if blocks_to_swap > 0:
            cmd.extend([CONFIG_FLAGS["BLOCKS_TO_SWAP"], str(blocks_to_swap)])

        # Add default prodigy args if using prodigy and no optimizer_args provided
        optimizer_type = optimization.get('optimizer_type', 'AdamW')
        # if optimizer_type.lower() == 'prodigy' and not optimization.get('optimizer_args'):
        #     optimization['optimizer_args'] = 'd0=1e-5'

        self._add_args_flag(cmd, 'optimizer_args', optimization, CONFIG_FLAGS["OPTIMIZER_ARGS"])

        return cmd

    def _build_checkpoint_config(self, checkpoints: Dict, optimization: Dict) -> List[str]:
        """Builds flags for checkpointing and training loops."""
        cmd = []
        
        if self.parse_bool(checkpoints.get('save_state', False)):
            cmd.append(CONFIG_FLAGS["SAVE_STATE"])

        ckpt_mode = checkpoints.get('mode', 'steps')
        interval = checkpoints.get('interval', 50)
        steps = optimization.get('max_steps', 200)
        keep_last_n = checkpoints.get('keep_last_n', -1)

        if ckpt_mode == 'epochs':
            cmd.extend([CONFIG_FLAGS["SAVE_EVERY_N_EPOCHS"], str(interval)])
            if keep_last_n > 0:
                cmd.extend([CONFIG_FLAGS["SAVE_LAST_N_EPOCHS"], str(keep_last_n)])
            cmd.extend([CONFIG_FLAGS["MAX_TRAIN_EPOCHS"], str(steps)])
        else:
            cmd.extend([CONFIG_FLAGS["SAVE_EVERY_N_STEPS"], str(interval)])
            if keep_last_n > 0:
                cmd.extend([CONFIG_FLAGS["SAVE_LAST_N_STEPS"], str(keep_last_n)])
            cmd.extend([CONFIG_FLAGS["MAX_TRAIN_STEPS"], str(steps)])

        return cmd

    def _build_optimization_and_scheduler(self, optimization: Dict) -> List[str]:
        """Builds optimizer and scheduler specific flags."""
        cmd = []

        # Map optimizer names to module paths for external optimizers
        optimizer_type = optimization.get('optimizer_type', 'AdamW')
        if optimizer_type.lower() == 'prodigy':
            optimizer_type = 'prodigyopt.Prodigy'
        elif optimizer_type.lower() == 'came':
            optimizer_type = 'came_pytorch.CAME'
        elif optimizer_type.lower() == 'adamwschedulefree':
            optimizer_type = 'schedulefree.AdamWScheduleFree'

        cmd.extend([
            CONFIG_FLAGS["GRAD_ACCUMULATION"], str(optimization.get('gradient_accumulation_steps', 4)),
            CONFIG_FLAGS["LEARNING_RATE"], str(optimization.get('learning_rate', 0.001)),
            CONFIG_FLAGS["OPTIMIZER_TYPE"], optimizer_type,
            CONFIG_FLAGS["LR_SCHEDULER"], optimization.get('scheduler_type', 'constant'),
            CONFIG_FLAGS["TIMESTAMP_SAMPLING"],
                self.get_config_value(optimization, 'timestep_sampling_mode', default='shifted_logit_normal') # Or flow_matching section
        ])

        return cmd

    def _build_validation_flags(self, validation: Dict, output_dir: str, config: Dict = None, ckpt_mode: str = 'steps', dataset_config: str = None) -> List[str]:
        """Builds flags for sampling and validation."""
        cmd = []
        
        sample_interval = validation.get('interval', -1)
        sample_at_first = self.parse_bool(validation.get('sample_at_first', 'false'))
        sampling_enabled = (sample_interval != -1) or sample_at_first

        if not sampling_enabled:
            return cmd

        if sample_at_first:
            cmd.append(CONFIG_FLAGS["SAMPLE_AT_FIRST"])

        # Use the ckpt_mode parameter (passed from checkpoints config)
        # ckpt_mode is 'steps' or 'epochs' based on [checkpoints] mode setting

        interval = validation.get('interval', -1)
        if str(interval) != '-1':
            # Use the correct flag based on checkpoint mode
            if ckpt_mode == 'epochs':
                cmd.extend(["--sample_every_n_epochs", str(interval)])
            else:
                cmd.extend([CONFIG_FLAGS["SAMPLE_INTERVAL"], str(interval)])

        video_dims = validation.get('video_dims', '768, 512, 45')
        if str(video_dims) != '768, 512, 45':
            try:
                dims = [d.strip() for d in str(video_dims).split(',')]
                if len(dims) >= 3:
                    width = round(int(dims[0]) / 32) * 32
                    height = round(int(dims[1]) / 32) * 32
                    frames = max(round((int(dims[2]) - 1) / 8) * 8 + 1, 9)
                    cmd.extend([CONFIG_FLAGS["WIDTH"], str(width), 
                                CONFIG_FLAGS["HEIGHT"], str(height), 
                                CONFIG_FLAGS["NUM_FRAMES"], str(frames)])
            except Exception:
                pass

        if self.parse_bool(validation.get('generate_audio', False)):
            cmd.append(CONFIG_FLAGS["MERGE_AUDIO"])

        if self.parse_bool(validation.get('s_offload', True)):
            cmd.append(CONFIG_FLAGS["OFFLOADING"])
            
        if self.parse_bool(validation.get('tiled_vae', True)):
            cmd.extend([CONFIG_FLAGS["TILED_VAE"], 
                        CONFIG_FLAGS["VAE_TILE_SIZE"], "512",
                        CONFIG_FLAGS["VAE_TILE_OVERLAP"], "64",
                        CONFIG_FLAGS["VAE_TEMPORAL_SIZE"], "16",
                        CONFIG_FLAGS["VAE_TEMPORAL_OVERLAP"], "8"])

        # Sample prompts and caches
        sample_dir = Path(output_dir) / 'sample'
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_prompts_path = sample_dir / 'sample_prompts.txt'

        # Check if we should use sample prompts (inline config or file-based)
        use_prompts_file = self.parse_bool(validation.get('prompts_file', False))
        inline_prompts = validation.get('prompts')
        has_inline_config = bool(inline_prompts)

        # NOTE: Sample prompts caching is now handled by LTX2Cache.build_all_cache_commands
        # and will be run during the cache phase (in background thread with streaming)
        # Do NOT trigger caching here as it would block the UI

        # Add --sample_prompts flag if file exists or will be created by caching
        if sample_prompts_path.exists() or use_prompts_file or (has_inline_config and sampling_enabled):
            cmd.extend([CONFIG_FLAGS["SAMPLE_PROMPTS"], str(sample_prompts_path)])

            # Add cache flags if caches exist
            cache_path = sample_dir / 'sample_prompts_cache.pt'
            if cache_path.exists():
                cmd.extend([CONFIG_FLAGS["USE_PRECACHED_PROMPTS"], CONFIG_FLAGS["SAMPLE_CACHE"], str(cache_path)])

            latents_cache_path = sample_dir / 'sample_latents_cache.pt'
            if latents_cache_path.exists():
                cmd.extend([CONFIG_FLAGS["LATENTS_CACHE"], str(latents_cache_path)])

        return cmd

    def _build_preservation_flags(self, acceleration: Dict) -> List[str]:
        """Builds flags for preservation and regularization techniques."""
        cmd = []
        
        if self.parse_bool(acceleration.get('blank_preservation', False)):
            cmd.append(CONFIG_FLAGS["BLANK_PRESERVATION"])
            cmd.extend([CONFIG_FLAGS["BLANK_ARGS"], acceleration.get('blank_preservation_args', 'multiplier=0.5')])

        if self.parse_bool(acceleration.get('dop', False)):
            cmd.append(CONFIG_FLAGS["DOP"])
            cmd.extend([CONFIG_FLAGS["DOP_ARGS"], acceleration.get('dop_args', 'class=woman multiplier=1.0')])

        if self.parse_bool(acceleration.get('prior_divergence', False)):
            cmd.append(CONFIG_FLAGS["PRIOR_DIV"])
            cmd.extend([CONFIG_FLAGS["PRIOR_DIV_ARGS"], acceleration.get('prior_divergence_args', 'multiplier=0.1')])

        if self.parse_bool(acceleration.get('crepa', False)):
            cmd.append(CONFIG_FLAGS["CREPA"])
            crepa_mode = acceleration.get('crepa_mode', 'backbone')
            crepa_args = acceleration.get('crepa_args', 'student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2')
            cmd.extend([CONFIG_FLAGS["CREPA_ARGS"], f"mode={crepa_mode} {crepa_args}"])

        if self.parse_bool(acceleration.get('self_flow', False)):
            cmd.append(CONFIG_FLAGS["SELF_FLOW"])
            self_flow_args = acceleration.get('self_flow_args', 'teacher_mode=base student_block_ratio=0.3 teacher_block_ratio=0.7 lambda_self_flow=0.1')
            # Parse args and pass each individually (like optimizer_args)
            for arg in self_flow_args.split():
                cmd.append(CONFIG_FLAGS["SELF_FLOW_ARGS"])
                cmd.append(arg)

        if self.parse_bool(acceleration.get('cts_lambda', False)):
            cts_lambda_args = acceleration.get('cts_lambda_args', 'video_driven=0.3 audio_driven=0.1')
            # Parse the args string to extract individual values
            # Format: "video_driven=0.3 audio_driven=0.1"
            for arg in cts_lambda_args.split():
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    try:
                        val = float(value)
                        if val != 0:
                            if key == 'video_driven':
                                cmd.extend([CONFIG_FLAGS["CTS_LAMBDA_VIDEO_DRIVEN"], str(val)])
                            elif key == 'audio_driven':
                                cmd.extend([CONFIG_FLAGS["CTS_LAMBDA_AUDIO_DRIVEN"], str(val)])
                    except ValueError:
                        logger.warning(f"Could not parse cts_lambda arg: {arg}")

        return cmd

    def _build_network_config(self, lora: Dict, training_mode: str, training_strategy: Dict = None, optimization: Dict = None) -> List[str]:
        """Builds flags for network configuration (LoRA/LoKR)."""
        cmd = []

        network_dim = lora.get('rank', 64)
        network_alpha = lora.get('alpha', 64)
        lokr_factor = lora.get('factor', 4)

        if training_mode in ('lokr', 'loha'):
            cmd.extend([
                CONFIG_FLAGS["NETWORK_MODULE"], f"networks.{training_mode}",
                CONFIG_FLAGS["NETWORK_DIM"], str(network_dim),
                CONFIG_FLAGS["NETWORK_ALPHA"], str(network_alpha),
                CONFIG_FLAGS["INIT_LOKR_NORM"], "0.001",
                CONFIG_FLAGS["NETWORK_ARGS"], f"factor={lokr_factor}"
            ])
        else:
            cmd.extend([
                CONFIG_FLAGS["NETWORK_MODULE"], "networks.lora_ltx2",
                CONFIG_FLAGS["NETWORK_DIM"], str(network_dim),
                CONFIG_FLAGS["NETWORK_ALPHA"], str(network_alpha)
            ])

        # Dropout (via network_args)
        network_dropout = lora.get('network_dropout', lora.get('dropout', 0.0))
        if network_dropout > 0:
            cmd.append(f"--network_args")
            cmd.append(f"dropout={network_dropout}")

        # Stiefel-LoRA (via network_args) - derived from optimizer_type
        optimizer_type = (optimization or {}).get('optimizer_type', '').lower() if optimization else ''
        if optimizer_type == 'stiefel':
            cmd.append(f"--network_args")
            cmd.append(f"use_stiefel=True")

        # LoRA+ (via network_args) - increases UP side learning rate
        loraplus_ratio = (optimization or {}).get('loraplus_ratio', 0.0)
        if loraplus_ratio and float(loraplus_ratio) > 0:
            cmd.append(f"--network_args")
            cmd.append(f"loraplus_lr_ratio={loraplus_ratio}")

        # Caption dropout rate (from lora section)
        caption_dropout_rate = lora.get('caption_dropout_rate', 0.0)
        if caption_dropout_rate > 0:
            cmd.extend([CONFIG_FLAGS["CAPTION_DROPOUT_RATE"], str(caption_dropout_rate)])

        return cmd

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
            # Look for .safetensors files in the directory
            safetensors_files = [f for f in os.listdir(dir_path) if f.endswith('.safetensors')]
            if safetensors_files:
                # Use the first .safetensors file found
                init_checkpoint = str(Path(dir_path) / safetensors_files[0])
                logger.info(f"Directory detected, using found safetensors file: {init_checkpoint}")
            else:
                logger.warning(f"Directory detected but no .safetensors file found inside: {dir_path}")

        target_rank = lora.get('rank', 64)

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

        cmd.extend([CONFIG_FLAGS["NETWORK_WEIGHTS"], init_checkpoint])
        return cmd

    def build_training_command(
        self,
        config: Dict,
        dataset_config: str,
        slider_config: Optional[str] = None,
        resume: Optional[str] = None,
        reset_optimizer: bool = False,
        reset_optimizer_params: bool = False
    ) -> List[str]:
        """
        Build the full accelerate launch training command.

        Returns:
            List of command arguments for accelerate launch
        """
        # Extract config sections
        model = config.get('model', {})
        optimization = config.get('optimization', {})
        acceleration = config.get('acceleration', {})
        training_strategy = config.get('training_strategy', {})
        checkpoints = config.get('checkpoints', {})
        lora = config.get('lora', {})
        flow_matching = config.get('flow_matching', {})
        validation = config.get('validation', {})

        # Determine script (slider vs regular vs ic_lora vs vace)
        slider_enabled = self.parse_bool(training_strategy.get('slider', False))
        use_slider = slider_enabled and slider_config and os.path.exists(slider_config)

        # IC-LoRA uses standard script with v2v preset and reference_cache_directory
        ic_lora_enabled = self.parse_bool(training_strategy.get('ic_lora', False))
        use_ic_lora = ic_lora_enabled

        # VACE training detection: check if vace_lora is enabled in last_config.toml
        use_vace = False
        vace_dataset_config = None
        vace_lora_enabled = self.parse_bool(training_strategy.get('vace_lora', False))
        if vace_lora_enabled and dataset_config:
            # Check if dataset_config already points to _vace.toml (passed from start_button_handler)
            if dataset_config.endswith('_vace.toml'):
                use_vace = True
                vace_dataset_config = dataset_config
            else:
                # Otherwise, check for VACE-specific config file (_vace.toml)
                vace_config_path = dataset_config.replace('.toml', '_vace.toml')
                if os.path.exists(vace_config_path):
                    use_vace = True
                    vace_dataset_config = vace_config_path

        if use_slider:
            script = str(self.musubi_root / "ltx2_train_slider.py")
            config_flag = "--slider_config"
            config_path = slider_config
        elif use_vace:
            # VACE training uses ltx2_vace_train.py with the _vace.toml config
            script = str(self.musubi_root / "ltx2_vace_train.py")
            config_flag = "--dataset_config"
            config_path = vace_dataset_config or dataset_config
        else:
            script = str(self.musubi_root / "ltx2_train_network.py")
            config_flag = "--dataset_config"
            config_path = dataset_config

        # 1. Base Command
        cmd = self._build_base_flags(script, acceleration, model, config_flag, config_path)

        # 2. Output Directory & Logging
        output_dir = model.get('output_dir', DEFAULTS['output_dir'])
        cmd.extend([
            CONFIG_FLAGS["OUTPUT_DIR"], output_dir,
            CONFIG_FLAGS["LOG_WITH"], "tensorboard",
            CONFIG_FLAGS["LOGGING_DIR"], os.path.join(output_dir, ".tensorboard"),
        ])

        # 3. Training Strategy & Mode
        ltx_mode = training_strategy.get('ltx_mode', DEFAULTS['ltx_mode'])
        cmd.extend([
            CONFIG_FLAGS["LTX2_MODE"], ltx_mode,
            "--ltx2_first_frame_conditioning_p", str(training_strategy.get('first_frame_conditioning_p', 0.1)),
        ])

        # LTX version flag (2.3 vs 2.0)
        if self.parse_bool(training_strategy.get('ltx_2_3', False)):
            cmd.extend([CONFIG_FLAGS["LTX_VERSION"], "2.3"])

        # LoRA target preset: ic_lora (v2v) > audio > av > default (t2v)
        if use_ic_lora:
            cmd.extend(["--lora_target_preset", "v2v"])
        elif ltx_mode == 'audio':
            cmd.extend(["--lora_target_preset", "audio"])
        elif ltx_mode == 'av':
            cmd.extend(["--lora_target_preset", "full"])

            # Add audio loss balance mode for AV training
            audio_loss_balance_mode = training_strategy.get('audio_loss_balance_mode', 'ema_mag')
            cmd.extend([CONFIG_FLAGS["AUDIO_LOSS_BALANCE_MODE"], audio_loss_balance_mode])

            # Add EMA-specific settings if using ema_mag mode
            if audio_loss_balance_mode == 'ema_mag':
                target_ratio = training_strategy.get('audio_loss_balance_target_ratio', 0.33)
                ema_decay = training_strategy.get('audio_loss_balance_ema_decay', 0.99)
                cmd.extend([CONFIG_FLAGS["AUDIO_LOSS_BALANCE_TARGET_RATIO"], str(target_ratio)])
                cmd.extend([CONFIG_FLAGS["AUDIO_LOSS_BALANCE_EMA_DECAY"], str(ema_decay)])

        if self.parse_bool(training_strategy.get('separate_audio_buckets', False)):
            cmd.append("--separate_audio_buckets")

        # VACE-specific flags (only for VACE training)
        if use_vace:
            vace_config = training_strategy.get('vace', {})

            # VACE trainer inherits from hv_train_network which requires --dit flag
            cmd.extend(["--dit", model.get('model_path', '')])

            # --vace_scale: hint injection scale (default 1.0)
            vace_scale = vace_config.get('scale', 1.0)
            if vace_scale != 1.0 or 'scale' in vace_config:
                cmd.extend(["--vace_scale", str(vace_scale)])

            # --vace_layers: comma-separated DiT block indices (default every 4th)
            vace_layers = vace_config.get('layers', [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44])
            if isinstance(vace_layers, list):
                layers_str = ",".join(map(str, vace_layers))
            else:
                layers_str = str(vace_layers)
            cmd.extend(["--vace_layers", layers_str])

            # --vace_freeze_dit: freeze base DiT during training (default true)
            if self.parse_bool(vace_config.get('freeze_dit', True)):
                cmd.append("--vace_freeze_dit")

            # --vace_model_path: path to pre-trained VACE weights
            vace_model_path = vace_config.get('model_path')
            if vace_model_path and os.path.exists(vace_model_path):
                cmd.extend(["--vace_model_path", vace_model_path])

            # --enable_audio_xattn_in_vace: add audio cross-attention to video VACE
            if self.parse_bool(vace_config.get('enable_audio_xattn', False)):
                cmd.append("--enable_audio_xattn_in_vace")

            # Audio VACE scale (for joint AV training)
            audio_vace_scale = vace_config.get('audio_scale')
            if audio_vace_scale is not None:
                cmd.extend(["--audio_vace_scale", str(audio_vace_scale)])

            # VACE LoRA mode: train adapters instead of full VACE model
            # Enabled when training_mode is 'lora' (default) or when vace.lora is explicitly True
            training_mode = model.get('training_mode', 'lora')
            vace_lora_explicit = vace_config.get('lora', None)
            # LoRA mode enabled if: training_mode='lora' OR vace.lora=True
            # Full training when: training_mode='full' AND vace.lora not explicitly True
            if vace_lora_explicit is not None:
                vace_lora_mode = self.parse_bool(vace_lora_explicit)
            else:
                vace_lora_mode = (training_mode == 'lora')

            if vace_lora_mode:
                vace_lora_dim = vace_config.get('lora_dim', lora.get('rank', 32))
                vace_lora_alpha = vace_config.get('lora_alpha', lora.get('alpha', 32))
                cmd.extend([
                    "--network_module", "networks.lora_ltx2",
                    "--network_dim", str(vace_lora_dim),
                    "--network_alpha", str(vace_lora_alpha)
                ])

        # 4. Initialization Logic
        init_flags = self._build_initialization(config, lora)
        cmd.extend(init_flags)

        # 5. Network Configuration (skip for VACE - it trains its own parameters, not LoRA)
        if not use_vace:
            training_mode = model.get('training_mode', 'lora')
            net_flags = self._build_network_config(lora, training_mode, training_strategy, optimization)
            cmd.extend(net_flags)

        # 6. Optimization & Scheduler Flags
        opt_flags = self._build_optimization_flags(optimization, acceleration)
        sched_flags = self._build_optimization_and_scheduler(optimization)
        cmd.extend(opt_flags)
        cmd.extend(sched_flags)

        # Add audio_lr flag for av/full mode when audio_lr is not 0
        audio_lr = optimization.get('audio_lr', 0.0)
        if ltx_mode == 'av' and audio_lr != 0:
            cmd.extend([CONFIG_FLAGS["AUDIO_LR"], str(audio_lr)])

        # 7. Checkpoint Configuration
        ckpt_flags = self._build_checkpoint_config(checkpoints, optimization)
        cmd.extend(ckpt_flags)

        # 8. Validation & Sampling Flags
        ckpt_mode = checkpoints.get('mode', 'steps')
        val_flags = self._build_validation_flags(validation, output_dir, config, ckpt_mode, dataset_config)
        cmd.extend(val_flags)

        # 9. Preservation & Regularization
        pres_flags = self._build_preservation_flags(acceleration)
        cmd.extend(pres_flags)

        # 10. Resume Flag (Specific check at end as per original logic)
        if resume and os.path.exists(resume):
            cmd.extend([CONFIG_FLAGS["RESUME"], resume])

        # 11. Reset optimizer flags (for changing parameter groups like LoRA+)
        if reset_optimizer:
            cmd.append("--reset_optimizer")
        if reset_optimizer_params:
            cmd.append("--reset_optimizer_params")

        return cmd

    # NOTE: Sample prompt caching is now handled by LTX2Cache.build_all_cache_commands
    # and will be run during the cache phase (in background thread with streaming)
    # The cache_sample_wrapper.py script handles:
    # 1. Generating sample_prompts.txt from inline config
    # 2. Checking hash to see if caching is needed
    # 3. Running the actual caching subprocess if needed

    def prepare_for_training(self, config: Dict, dataset_config: str, slider_config: Optional[str] = None) -> None:
        """
        Prepare for training by caching sample prompts.

        This method delegates to LTX2CacheSample which:
        1. Generates sample_prompts.txt from inline config
        2. Caches text encoder outputs and image latents
        3. Uses hash-based validation to avoid redundant work

        Call this before starting training to ensure all prerequisites are ready.
        """
        # Just delegate to the caching method which handles everything
        self._cache_sample_prompts_if_needed(config, dataset_config)

    def format_training_command(self, **kwargs) -> str:
        """
        Format the training command as a multi-line string.
        
        Returns:
            Formatted command string with each argument on a new line
        """
        cmd = self.build_training_command(**kwargs)

        formatted_lines = []
        i = 0
        while i < len(cmd):
            arg = cmd[i]
            # If this is a flag that takes a value, include both on the same line
            if i + 1 < len(cmd) and not cmd[i + 1].startswith('-'):
                formatted_lines.append(f"  {arg} {cmd[i + 1]} \\")
                i += 2
            else:
                formatted_lines.append(f"  {arg} \\")
                i += 1

        # Remove trailing \ from last line
        if formatted_lines:
            formatted_lines[-1] = formatted_lines[-1].rstrip(' \\')

        return "\\\n".join(formatted_lines)


# ==========================================================================
# Convenience Functions
# ==========================================================================

def create_training_command(config: Dict, dataset_config: str, slider_config: Optional[str] = None, resume: Optional[str] = None, reset_optimizer: bool = False, reset_optimizer_params: bool = False) -> List[str]:
    """Convenience function to create training command arguments."""
    runner = LTX2Run()
    return runner.build_training_command(config, dataset_config, slider_config, resume, reset_optimizer, reset_optimizer_params)


def format_training_command(config: Dict, dataset_config: str, slider_config: Optional[str] = None, resume: Optional[str] = None, reset_optimizer: bool = False, reset_optimizer_params: bool = False) -> str:
    """Convenience function to format training command as string."""
    runner = LTX2Run()
    return runner.format_training_command(config=config, dataset_config=dataset_config, slider_config=slider_config, resume=resume, reset_optimizer=reset_optimizer, reset_optimizer_params=reset_optimizer_params)