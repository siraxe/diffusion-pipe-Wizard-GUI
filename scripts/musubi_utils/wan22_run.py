import os
import glob
from typing import Dict, List, Optional
from pathlib import Path


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
        resume: Optional[str] = None
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
        cmd.extend([
            "--network_module", "networks.lora_wan",
            "--network_dim", str(adapter.get('rank', 32)),
        ])

        # Init from existing checkpoint
        init_checkpoint = adapter.get('init_from_existing', '')
        if init_checkpoint and str(init_checkpoint).lower() not in ('', 'null', 'none'):
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

def create_training_command(config: Dict, dataset_config: str, slider_config: Optional[str] = None, resume: Optional[str] = None) -> List[str]:
    runner = WAN22Run()
    return runner.build_training_command(config, dataset_config, slider_config, resume)


def format_training_command(config: Dict, dataset_config: str, slider_config: Optional[str] = None, resume: Optional[str] = None) -> str:
    runner = WAN22Run()
    return runner.format_training_command(config=config, dataset_config=dataset_config, slider_config=slider_config, resume=resume)


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
