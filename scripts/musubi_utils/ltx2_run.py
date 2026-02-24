"""
LTX-Video-2 Training Command Builder

This module handles building commands for LTX-Video-2 training operations.
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

try:
    import safetensors.torch
except ImportError:
    safetensors = None

logger = logging.getLogger(__name__)


class LTX2Run:
    """Builds commands for LTX-Video-2 training operations."""

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the LTX2 run command builder.

        Args:
            project_root: Path to the project root directory.
                        If None, will be auto-detected.
        """
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.musubi_root = self.project_root / "diffusion-trainers" / "musubi-tuner"

    @staticmethod
    def _find_project_root() -> Path:
        """Auto-detect the project root directory."""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "flet_app").exists() or (parent / "diffusion-trainers").exists():
                return parent
        return Path.cwd()

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to project root."""
        if os.path.isabs(path):
            return path
        return str(self.project_root / path)

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
    # LoRA Rank Detection & Conversion Helpers
    # ==========================================================================

    def get_lora_rank(self, file_path: str) -> int:
        """
        Detect the rank of a LoRA/LoKR checkpoint.

        Returns the rank (dimension of lora_down/lora_A/lokr_w1), or 0 if unable to detect.
        """
        if safetensors is None:
            logger.warning("safetensors not available, cannot detect LoRA rank")
            return 0

        try:
            state_dict = safetensors.torch.load_file(file_path)
            if not state_dict:
                return 0

            # Look for lora_down or lora_A weight to determine rank
            for key in state_dict.keys():
                # Training format: lora_unet_model_*.lora_down.weight
                if key.endswith('.lora_down.weight'):
                    return state_dict[key].shape[0]
                # ComfyUI format: diffusion_model.*.lora_A.weight
                elif key.endswith('.lora_A.weight'):
                    return state_dict[key].shape[0]
                # LoKR format: lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b
                # For lokr_w1_b and lokr_w2_a, rank is typically the last dimension
                # For lokr_w1_a and lokr_w2_b, rank is typically the first dimension
                elif key.endswith('.lokr_w1_b'):
                    # lokr_w1_b shape is [out_features, rank]
                    return state_dict[key].shape[1]
                elif key.endswith('.lokr_w2_a'):
                    # lokr_w2_a shape is [rank, dim]
                    return state_dict[key].shape[0]
                elif key.endswith('.lokr_w1_a'):
                    # lokr_w1_a shape - could be [rank, in_features] or [out_features, rank]
                    # Use smaller dimension as rank
                    shape = state_dict[key].shape
                    return min(shape)
                elif key.endswith('.lokr_w2_b'):
                    # lokr_w2_b shape - could be [rank, out_features] or [in_features, rank]
                    # Use smaller dimension as rank
                    shape = state_dict[key].shape
                    return min(shape)

            return 0
        except Exception as e:
            logger.warning(f"Error detecting LoRA rank for {file_path}: {e}")
            return 0

    def rerank_training_format_lora(self, file_path: str, target_rank: int) -> Optional[str]:
        """
        Rerank a training format LoRA checkpoint to a different rank.

        Creates a new file with _rank{target_rank} suffix.

        Returns the path to the new file, or None if failed.
        """
        import subprocess
        import sys

        try:
            # Determine output path
            input_file = Path(file_path)
            output_path = input_file.parent / f"{input_file.stem}_rank{target_rank}{input_file.suffix}"

            # Use the dedicated reranking script
            sys_path = os.path.join(self.project_root, 'scripts')
            rerank_script = os.path.join(sys_path, 'rerank_lora.py')

            logger.info(f"Reranking checkpoint: {file_path} -> rank {target_rank}")

            # Build command
            cmd = [
                sys.executable,
                rerank_script,
                file_path,
                '--target_rank', str(target_rank),
                '-o', str(output_path),
                '--device', 'cuda'
            ]

            # Run the conversion with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout
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
    # Training Command Building
    # ==========================================================================

    def build_training_command(
        self,
        config: Dict,
        dataset_config: str,
        slider_config: Optional[str] = None,
        resume: Optional[str] = None
    ) -> List[str]:
        """
        Build the full accelerate launch training command.

        Args:
            config: Full configuration dictionary
            dataset_config: Path to the dataset config file
            slider_config: Optional path to slider config
            resume: Optional path to state directory for resuming

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

        # Determine script (slider vs regular)
        slider_enabled = self.parse_bool(training_strategy.get('slider', False))
        use_slider = slider_enabled and slider_config and os.path.exists(slider_config)

        if use_slider:
            script = str(self.musubi_root / "ltx2_train_slider.py")
            config_flag = "--slider_config"
            config_path = slider_config
        else:
            script = str(self.musubi_root / "ltx2_train_network.py")
            config_flag = "--dataset_config"
            config_path = dataset_config

        # Base accelerate command
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            script,
            "--mixed_precision", acceleration.get('mixed_precision_mode', 'bf16'),
            config_flag, config_path,
            "--gemma_root", model.get('text_encoder_path', ''),
            "--ltx2_checkpoint", model.get('model_path', ''),
            "--flash_attn",
        ]

        # Output name
        output_name = model.get('output_name', model.get('name', 'ltx2_lora'))
        if output_name:
            cmd.extend(["--output_name", output_name])

        # Boolean flags
        if self.parse_bool(optimization.get('enable_gradient_checkpointing', True)):
            cmd.append("--gradient_checkpointing")
        if self.parse_bool(acceleration.get('fp8_base', True)):
            cmd.append("--fp8_base")
        if self.parse_bool(acceleration.get('fp8_scaled', True)):
            cmd.append("--fp8_scaled")
        if self.parse_bool(acceleration.get('8_bit_te', False)):
            cmd.append("--gemma_load_in_8bit")
        if self.parse_bool(acceleration.get('attn_chunking', False)):
            cmd.extend([
                "--split_attn_target", "video",
                "--split_attn_mode", "query",
                "--split_attn_chunk_size", "512"
            ])
        if self.parse_bool(checkpoints.get('save_state', False)):
            cmd.append("--save_state")
        if self.parse_bool(training_strategy.get('use_mask', False)):
            cmd.append("--ltx2_enable_mask")

        # Resume from state
        if resume and os.path.exists(resume):
            cmd.extend(["--resume", resume])

        # Checkpoint interval
        ckpt_mode = checkpoints.get('mode', 'steps')
        interval = checkpoints.get('interval', 50)
        steps = optimization.get('max_steps', 200)
        keep_last_n = checkpoints.get('keep_last_n', -1)

        if ckpt_mode == 'epochs':
            cmd.extend(["--save_every_n_epochs", str(interval)])
            if keep_last_n > 0:
                cmd.extend(["--save_last_n_epochs", str(keep_last_n)])
            cmd.extend(["--max_train_epochs", str(steps)])
        else:
            cmd.extend(["--save_every_n_steps", str(interval)])
            if keep_last_n > 0:
                cmd.extend(["--save_last_n_steps", str(keep_last_n)])
            cmd.extend(["--max_train_steps", str(steps)])

        # Network dropout (network_dim/alpha added later based on training_mode)
        network_dropout = lora.get('dropout', 0.0)
        if network_dropout > 0:
            cmd.extend(["--network_dropout", str(network_dropout)])

        # Optimizer and scheduler
        cmd.extend([
            "--gradient_accumulation_steps", str(optimization.get('gradient_accumulation_steps', 4)),
            "--learning_rate", str(optimization.get('learning_rate', 0.001)),
            "--optimizer_type", optimization.get('optimizer_type', 'AdamW'),
            "--lr_scheduler", optimization.get('scheduler_type', 'constant'),
            "--timestep_sampling", flow_matching.get('timestep_sampling_mode', 'shifted_logit_normal'),
        ])

        # Warmup steps for constant_with_warmup
        if optimization.get('scheduler_type') == 'constant_with_warmup':
            lr_warmup_steps = optimization.get('lr_warmup_steps', 50)
            cmd.extend(["--lr_warmup_steps", str(lr_warmup_steps)])

        # Max grad norm
        max_grad_norm = optimization.get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            cmd.extend(["--max_grad_norm", str(max_grad_norm)])

        # Block swap
        blocks_to_swap = optimization.get('blocks_to_swap', 0)
        if blocks_to_swap > 0:
            cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])

        optimizer_args = optimization.get('optimizer_args')
        if optimizer_args:
            parsed_args = self.parse_optimizer_args(optimizer_args)
            if parsed_args:
                cmd.append("--optimizer_args")
                cmd.extend(parsed_args)

        # Output directory
        output_dir = model.get('output_dir', 'output/ltx2_lora')
        cmd.extend([
            "--output_dir", output_dir,
            "--log_with", "tensorboard",
            "--logging_dir", os.path.join(output_dir, ".tensorboard"),
        ])

        # Training strategy
        ltx_mode = training_strategy.get('ltx_mode', 'video')
        cmd.extend([
            "--ltx2_mode", ltx_mode,
            "--ltx2_first_frame_conditioning_p", str(training_strategy.get('first_frame_conditioning_p', 0.1)),
        ])

        # LoRA target preset based on mode
        if ltx_mode == 'audio':
            cmd.extend(["--lora_target_preset", "audio"])

        # Separate audio buckets
        if self.parse_bool(training_strategy.get('separate_audio_buckets', False)):
            cmd.append("--separate_audio_buckets")

        # Load existing checkpoint with rank checking
        # Check both [lora] section and top-level config
        init_checkpoint = lora.get('init_from_existing', config.get('init_from_existing', ''))
        if init_checkpoint and str(init_checkpoint).lower() not in ('', 'null', 'none'):
            # Convert to absolute path if relative
            if not os.path.isabs(init_checkpoint):
                init_checkpoint = str(self.project_root / init_checkpoint)
                logger.info(f"Converted relative checkpoint path: {init_checkpoint}")

            # Get target rank from config
            target_rank = lora.get('rank', 64)

            # Check if the file exists
            if os.path.exists(init_checkpoint):
                logger.info(f"Checkpoint file exists: {init_checkpoint}")

                # Check for rank mismatch
                checkpoint_rank = self.get_lora_rank(init_checkpoint)
                logger.info(f"Detected checkpoint rank: {checkpoint_rank}, target rank: {target_rank}")

                if checkpoint_rank > 0 and checkpoint_rank != target_rank:
                    logger.info(f"Rank mismatch detected, reranking from {checkpoint_rank} to {target_rank}")
                    converted_path = self.rerank_training_format_lora(init_checkpoint, target_rank)

                    if converted_path:
                        init_checkpoint = converted_path
                        logger.info(f"Using converted checkpoint: {init_checkpoint}")
                    else:
                        logger.warning(f"Conversion failed, using original checkpoint (may cause errors)")
                else:
                    logger.info(f"Checkpoint rank {checkpoint_rank} matches target rank {target_rank}")
            else:
                logger.warning(f"Checkpoint file does not exist: {init_checkpoint}")

            cmd.extend(["--network_weights", init_checkpoint])

        # Network module and rank/alpha
        training_mode = model.get('training_mode', 'lora')
        network_dim = lora.get('rank', 64)
        network_alpha = lora.get('alpha', 64)
        lokr_factor = lora.get('factor', 4)  # Default factor=4 for LoKR

        if training_mode in ('lokr', 'loha'):
            # Use built-in musubi LoKR/LoHA module
            cmd.extend([
                "--network_module", f"networks.{training_mode}",
                "--network_dim", str(network_dim),
                "--network_alpha", str(network_alpha),
                "--init_lokr_norm", "0.001",
                "--network_args", f"factor={lokr_factor}",  # Use small factor for high threshold (allows rank 32+ low-rank mode)
            ])
        else:
            # Standard LoRA for LTX-2
            cmd.extend([
                "--network_module", "networks.lora_ltx2",
                "--network_dim", str(network_dim),
                "--network_alpha", str(network_alpha),
            ])

        # Validation/Sampling
        sample_interval = validation.get('interval', '-1')
        sample_at_first = self.parse_bool(validation.get('sample_at_first', 'false'))
        sampling_enabled = (sample_interval and str(sample_interval) != '-1') or sample_at_first

        if sampling_enabled:
            if sample_at_first:
                cmd.append("--sample_at_first")

            if sample_interval and str(sample_interval) != '-1':
                if ckpt_mode == 'epochs':
                    cmd.extend(["--sample_every_n_epochs", str(sample_interval)])
                else:
                    cmd.extend(["--sample_every_n_steps", str(sample_interval)])

            # Video dimensions
            video_dims = validation.get('video_dims', '768, 512, 45')
            if video_dims and str(video_dims) != '768, 512, 45':
                try:
                    dims = [d.strip() for d in str(video_dims).split(',')]
                    if len(dims) >= 3:
                        width = round(int(dims[0]) / 32) * 32
                        height = round(int(dims[1]) / 32) * 32
                        frames = max(round((int(dims[2]) - 1) / 8) * 8 + 1, 9)
                        cmd.extend(["--width", str(width), "--height", str(height), "--sample_num_frames", str(frames)])
                except:
                    pass  # Use defaults

            # Audio generation
            if self.parse_bool(validation.get('generate_audio', False)):
                cmd.append("--sample_merge_audio")

            # Sampling optimization flags
            if self.parse_bool(validation.get('s_offload', True)):
                cmd.append("--sample_with_offloading")
            if self.parse_bool(validation.get('tiled_vae', True)):
                cmd.append("--sample_tiled_vae")
                cmd.extend([
                    "--sample_vae_tile_size", "512",
                    "--sample_vae_tile_overlap", "64",
                    "--sample_vae_temporal_tile_size", "16",
                    "--sample_vae_temporal_tile_overlap", "8",
                ])

            # Sample prompts
            prompts = validation.get('prompts', '')
            if prompts:
                sample_dir = os.path.join(output_dir, 'sample')
                sample_prompts_path = os.path.join(sample_dir, 'sample_prompts.txt')
                if os.path.exists(sample_prompts_path):
                    cmd.extend(["--sample_prompts", sample_prompts_path])

                    # Check for pre-cached prompts
                    cache_path = os.path.join(sample_dir, 'sample_prompts_cache.pt')
                    if os.path.exists(cache_path):
                        cmd.extend(["--use_precached_sample_prompts", "--sample_prompts_cache", cache_path])

                    # Check for I2V latents cache
                    latents_cache_path = os.path.join(sample_dir, 'sample_latents_cache.pt')
                    if os.path.exists(latents_cache_path):
                        cmd.extend(["--sample_latents_cache", latents_cache_path])

        # Preservation & Regularization
        if self.parse_bool(acceleration.get('blank_preservation', False)):
            cmd.append("--blank_preservation")
            cmd.extend(["--blank_preservation_args", acceleration.get('blank_preservation_args', 'multiplier=0.5')])

        if self.parse_bool(acceleration.get('dop', False)):
            cmd.append("--dop")
            cmd.extend(["--dop_args", acceleration.get('dop_args', 'class=woman multiplier=1.0')])

        if self.parse_bool(acceleration.get('prior_divergence', False)):
            cmd.append("--prior_divergence")
            cmd.extend(["--prior_divergence_args", acceleration.get('prior_divergence_args', 'multiplier=0.1')])

        # CREPA
        if self.parse_bool(acceleration.get('crepa', False)):
            cmd.append("--crepa")
            crepa_mode = acceleration.get('crepa_mode', 'backbone')
            crepa_args = acceleration.get('crepa_args', 'student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2')
            crepa_args_with_mode = f"mode={crepa_mode} {crepa_args}"
            cmd.append("--crepa_args")
            cmd.extend(crepa_args_with_mode.split())

        return cmd

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

def create_training_command(config: Dict, dataset_config: str, slider_config: Optional[str] = None, resume: Optional[str] = None) -> List[str]:
    """
    Convenience function to create training command arguments.

    Args:
        config: Full configuration dictionary
        dataset_config: Path to the dataset config file
        slider_config: Optional path to slider config
        resume: Optional path to state directory for resuming

    Returns:
        List of command arguments for accelerate launch
    """
    runner = LTX2Run()
    return runner.build_training_command(config, dataset_config, slider_config, resume)


def format_training_command(config: Dict, dataset_config: str, slider_config: Optional[str] = None, resume: Optional[str] = None) -> str:
    """
    Convenience function to format training command as string.

    Args:
        config: Full configuration dictionary
        dataset_config: Path to the dataset config file
        slider_config: Optional path to slider config
        resume: Optional path to state directory for resuming

    Returns:
        Formatted command string
    """
    runner = LTX2Run()
    return runner.format_training_command(config=config, dataset_config=dataset_config, slider_config=slider_config, resume=resume)
