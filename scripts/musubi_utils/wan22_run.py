import os
import sys
import glob
import logging
from typing import Dict, List, Optional
from pathlib import Path

try:
    import safetensors.torch
except ImportError:
    safetensors = None

logger = logging.getLogger(__name__)


class WAN22Run:
    def __init__(self, project_root: Optional[str] = None):
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
        """Parse optimizer args string, handling parentheses and comma separation."""
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

    def is_comfy_format_lora(self, file_path: str) -> bool:
        """
        Check if a LoRA file is in ComfyUI format by examining the keys.

        ComfyUI format keys start with 'diffusion_model.'
        Training format keys start with 'lora_unet_model_'
        """
        if safetensors is None:
            logger.warning("safetensors not available, cannot check LoRA format")
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

    def convert_comfy_to_training_with_rank(self, file_path: str, target_rank: int) -> Optional[str]:
        """
        Convert ComfyUI format LoRA to training format with optional rank conversion.

        Creates a new file with _rank{target_rank} suffix instead of overwriting.

        Returns the path to the new file, or None if failed.
        """
        import subprocess
        import sys

        try:
            # Determine output path (with rank suffix)
            input_file = Path(file_path)
            output_path = input_file.parent / f"{input_file.stem}_rank{target_rank}{input_file.suffix}"

            # Path to conversion script
            sys_path = os.path.join(self.project_root, 'scripts')
            convert_script = os.path.join(sys_path, 'convert_comfy_to_training_lora.py')

            logger.info(f"Converting ComfyUI LoRA with rank conversion: {file_path} -> rank {target_rank}")

            # Build command
            cmd = [
                sys.executable,
                convert_script,
                '--input', file_path,
                '--output', str(output_path),
                '--rank', str(target_rank),
            ]

            # Run the conversion with timeout
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
                logger.error(f"ComfyUI conversion failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("ComfyUI conversion timed out after 3 minutes")
            return None
        except Exception as e:
            logger.error(f"Failed to convert ComfyUI LoRA {file_path}: {e}")
            return None

    def _resolve_dit_path(self, transformer_path: str) -> str:
        if not transformer_path:
            return ''

        # Resolve path
        dit_path = self._resolve_path(transformer_path)

        # If it's already a safetensors file, use it directly
        if dit_path.endswith('.safetensors') and os.path.isfile(dit_path):
            return dit_path

        # If it's a directory, find the first partial safetensors file
        if os.path.isdir(dit_path):
            # Look for safetensors files in the directory
            pattern = os.path.join(dit_path, '*.safetensors')
            safetensors_files = glob.glob(pattern)

            if safetensors_files:
                # Sort to get consistent ordering (usually -00001-of-XXXXX)
                safetensors_files.sort()
                return safetensors_files[0]

        # Fallback: return the original path
        return dit_path

    # ==========================================================================
    # Training Command Building
    # ==========================================================================

    def build_training_command(
        self,
        config: Dict,
        dataset_config: str,
        slider_config: Optional[str] = None,
        resume: Optional[str] = None,
        reset_optimizer: bool = False,
        reset_optimizer_params: bool = False
    ) -> List[str]:
        # Extract config sections
        model = config.get('model', {})
        optimization = config.get('optimization', {})
        acceleration = config.get('acceleration', {})
        adapter = config.get('adapter', {})
        training = config  # Top-level keys for WAN

        # Resolve the DiT path
        transformer_path = model.get('transformer_path', '')
        dit_path = self._resolve_dit_path(transformer_path)

        # Determine task type - use wan_task from config if available, otherwise infer from model type
        task = model.get('wan_task', '')
        if not task:
            model_type = model.get('type', '').lower()
            # Default task mapping
            task_map = {
                'i2v': 'i2v-A14B',
                't2v': 't2v-A14B',
            }
            task = 't2v-A14B'  # default
            for key, task_val in task_map.items():
                if key in model_type:
                    task = task_val
                    break

        # Get mixed precision mode - check acceleration.mixed_precision_mode first (from dropdown),
        # otherwise map from model.dtype (config file value)
        acceleration = config.get('acceleration', {})
        mixed_precision_raw = acceleration.get('mixed_precision_mode', model.get('dtype', 'bfloat16'))
        # Map common dtype names to accelerate's expected values
        mixed_precision_map = {
            'bfloat16': 'bf16',
            'float16': 'fp16',
            'float32': 'no',
            'no': 'no',
            'fp16': 'fp16',
            'bf16': 'bf16',
            'fp8': 'fp8',
        }
        mixed_precision = mixed_precision_map.get(str(mixed_precision_raw).lower(), 'bf16')

        # Base accelerate command
        script = str(self.musubi_root / "wan_train_network.py")

        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            "--mixed_precision", mixed_precision,
            script,
            "--task", task,
            "--dit", dit_path,
            "--dataset_config", dataset_config,
            "--flash_attn",
            "--mixed_precision", mixed_precision,
        ]

        # Acceleration flags from [acceleration] section
        if self.parse_bool(acceleration.get('fp8_base', True)):
            cmd.append("--fp8_base")
        if self.parse_bool(acceleration.get('fp8_scaled', True)):
            cmd.append("--fp8_scaled")
        # 8_bit_te can be in either acceleration or training section
        eight_bit_te = acceleration.get('8_bit_te', training.get('8_bit_te', False))
        if self.parse_bool(eight_bit_te):
            cmd.append("--fp8_t5")
        if self.parse_bool(acceleration.get('attn_chunking', False)):
            cmd.extend([
                "--split_attn_target", "video",
                "--split_attn_mode", "query",
                "--split_attn_chunk_size", "512"
            ])

        # Gradient checkpointing
        if self.parse_bool(training.get('activation_checkpointing', False)) or training.get('activation_checkpointing') == 'unsloth':
            cmd.append("--gradient_checkpointing")

        # Data loader settings
        cmd.extend([
            "--max_data_loader_n_workers", str(training.get('caching_batch_size', 2)),
        ])
        if self.parse_bool(training.get('persistent_data_loader_workers', True)):
            cmd.append("--persistent_data_loader_workers")

        # Optimizer settings - check optimizer_type_m first (musubi-specific), then optimizer_type
        optimizer_type = optimization.get('optimizer_type_m', optimization.get('optimizer_type', 'AdamW'))
        # Map optimizer types (matching UI dropdown options)
        optimizer_map = {
            'adamw': 'adamw',
            'adamw8bit': 'adamw8bit',
            'adafactor': 'adafactor',
            'prodigy': 'prodigy',
            'automagic': 'automagic',
            'came': 'came_pytorch.CAME',
        }
        mapped_optimizer = optimizer_map.get(optimizer_type.lower(), 'adamw')
        cmd.extend(["--optimizer_type", mapped_optimizer])

        # Always pass learning_rate (like LTX-Video-2), even for Automagic
        lr_value = optimization.get('learning_rate', optimization.get('lr', 2e-4))
        cmd.extend(["--learning_rate", str(lr_value)])

        # For Automagic, also pass optimizer_args
        optimizer_args = optimization.get('optimizer_args', '')
        if optimizer_args and 'automagic' in mapped_optimizer.lower():
            for arg in self.parse_optimizer_args(optimizer_args):
                cmd.extend(["--optimizer_args", arg])

        # Gradient settings - check optimization first, then training
        gradient_accumulation_steps = optimization.get('gradient_accumulation_steps', training.get('gradient_accumulation_steps', 1))
        cmd.extend([
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        ])

        # Gradient clipping - check optimization first, then training
        max_grad_norm = optimization.get('max_grad_norm', training.get('gradient_clipping', training.get('max_grad_norm', 1.0)))
        if max_grad_norm and float(max_grad_norm) > 0:
            cmd.extend(["--max_grad_norm", str(max_grad_norm)])

        # LoRA settings
        target_rank = adapter.get('rank', 32)
        cmd.extend([
            "--network_module", "networks.lora_wan",
            "--network_dim", str(target_rank),
        ])

        # LoRA+ (via network_args) - increases UP side learning rate
        loraplus_ratio = optimization.get('loraplus_ratio', 0.0) if optimization else 0.0
        if loraplus_ratio and float(loraplus_ratio) > 0:
            cmd.extend(["--network_args", f"loraplus_lr_ratio={loraplus_ratio}"])

        # Init from existing checkpoint - with rank checking and conversion
        init_checkpoint = adapter.get('init_from_existing', '')
        if init_checkpoint and str(init_checkpoint).lower() not in ('', 'null', 'none'):
            # Convert to absolute path if relative
            if not os.path.isabs(init_checkpoint):
                init_checkpoint = str(self.project_root / init_checkpoint)
                logger.info(f"Converted relative checkpoint path: {init_checkpoint}")

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

            # Check if the file exists
            if os.path.exists(init_checkpoint):
                logger.info(f"Checkpoint file exists: {init_checkpoint}")

                # Check if conversion is needed (ComfyUI format OR rank mismatch)
                needs_conversion = False
                conversion_reason = ""
                is_comfy = False

                # Check 1: ComfyUI format
                if self.is_comfy_format_lora(init_checkpoint):
                    needs_conversion = True
                    conversion_reason = "ComfyUI format"
                    is_comfy = True
                # Check 2: Rank mismatch (only for training format files)
                else:
                    checkpoint_rank = self.get_lora_rank(init_checkpoint)
                    if checkpoint_rank > 0 and checkpoint_rank != target_rank:
                        needs_conversion = True
                        conversion_reason = f"rank mismatch (checkpoint: {checkpoint_rank}, config: {target_rank})"
                    else:
                        logger.info(f"Checkpoint rank {checkpoint_rank} matches config rank {target_rank}")

                # Convert if needed
                if needs_conversion:
                    if is_comfy:
                        # ComfyUI format: convert to training format with target rank
                        converted_path = self.convert_comfy_to_training_with_rank(init_checkpoint, target_rank)
                    else:
                        # Training format but rank mismatch: just rerank
                        converted_path = self.rerank_training_format_lora(init_checkpoint, target_rank)

                    if converted_path:
                        init_checkpoint = converted_path
                        logger.info(f"Using converted checkpoint: {init_checkpoint}")
                    else:
                        logger.warning(f"Conversion failed, using original checkpoint: {init_checkpoint}")
                else:
                    logger.info(f"Checkpoint is already in correct format and rank")
            else:
                logger.warning(f"Checkpoint file does not exist: {init_checkpoint}")

            cmd.extend(["--network_weights", init_checkpoint])

        # Timestep sampling - read from model section
        timestep_sample_method = model.get('timestep_sample_method', 'shift')
        discrete_flow_shift = model.get('discrete_flow_shift', 3.0)

        # Map timestep sampling methods for WAN
        # WAN expects: sigma, uniform, sigmoid, shift, flux_shift, flux2_shift, qwen_shift, logsnr, qinglong_flux, qinglong_qwen, shifted_logit_normal
        timestep_map = {
            'shift': 'shift',
            'logit_normal': 'shifted_logit_normal',  # map logit_normal to shifted_logit_normal for WAN
            'logit': 'shifted_logit_normal',
            'shifted_logit_normal': 'shifted_logit_normal',
        }
        mapped_timestep = timestep_map.get(timestep_sample_method.lower(), 'shift')
        cmd.extend([
            "--timestep_sampling", mapped_timestep,
            "--discrete_flow_shift", str(discrete_flow_shift),
        ])

        # Min/Max timestep from min_t/max_t (multiplied by 1000)
        min_t = model.get('min_t', 0.0)
        max_t = model.get('max_t', 1.0)
        if min_t > 0 or max_t < 1.0:
            cmd.extend([
                "--min_timestep", str(int(min_t * 1000)),
                "--max_timestep", str(int(max_t * 1000)),
                "--preserve_distribution_shape",
            ])

        # Epochs and checkpointing - check optimization.max_steps, then training.epochs
        max_steps = optimization.get('max_steps', training.get('max_steps'))
        epochs = training.get('epochs', 16)
        if max_steps:
            cmd.extend(["--max_train_steps", str(max_steps)])
        else:
            cmd.extend(["--max_train_epochs", str(epochs)])

        save_every_n_epochs = training.get('save_every_n_epochs', 1)
        cmd.extend([
            "--save_every_n_epochs", str(save_every_n_epochs),
        ])

        # Output settings
        output_dir = training.get('output_dir', model.get('output_dir', 'output/wan_lora'))
        output_name = model.get('name', 'wan_lora')

        cmd.extend([
            "--output_dir", output_dir,
            "--output_name", output_name,
            "--log_with", "tensorboard",
            "--logging_dir", os.path.join(output_dir, ".tensorboard"),
        ])

        # Resume from state
        if resume and os.path.exists(resume):
            cmd.extend(["--resume", resume])

        # Reset optimizer flags (for changing parameter groups like LoRA+)
        if reset_optimizer:
            cmd.append("--reset_optimizer")
        if reset_optimizer_params:
            cmd.append("--reset_optimizer_params")

        # Block swap - check optimization first, then training
        blocks_to_swap = optimization.get('blocks_to_swap', training.get('blocks_to_swap', 0))
        if blocks_to_swap > 0:
            cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])

        return cmd

    def format_training_command(self, **kwargs) -> str:
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
    runner = WAN22Run()
    return runner.build_training_command(config, dataset_config, slider_config, resume, reset_optimizer, reset_optimizer_params)


def format_training_command(config: Dict, dataset_config: str, slider_config: Optional[str] = None, resume: Optional[str] = None, reset_optimizer: bool = False, reset_optimizer_params: bool = False) -> str:
    runner = WAN22Run()
    return runner.format_training_command(config=config, dataset_config=dataset_config, slider_config=slider_config, resume=resume, reset_optimizer=reset_optimizer, reset_optimizer_params=reset_optimizer_params)


if __name__ == "__main__":
    import sys
    import toml

    if len(sys.argv) < 2:
        print("Usage: python wan22_run.py <last_config.toml> [dataset_config.toml]")
        sys.exit(1)

    config_path = sys.argv[1]
    dataset_config = sys.argv[2] if len(sys.argv) > 2 else None

    if not dataset_config:
        ws_dir = os.path.dirname(config_path)
        dataset_config = os.path.join(ws_dir, 'last_data_musubi_config.toml')

    with open(config_path, 'r') as f:
        config = toml.load(f)

    runner = WAN22Run()

    # Print the resolved DiT path
    transformer_path = config.get('model', {}).get('transformer_path', '')
    dit_path = runner._resolve_dit_path(transformer_path)
    print(f"Resolved DiT path: {dit_path}")
    print()

    # Build and print the command
    cmd = runner.build_training_command(config, dataset_config)

    print("=== WAN22 Training Command ===\n")
    print(" ".join(cmd))
    print()
