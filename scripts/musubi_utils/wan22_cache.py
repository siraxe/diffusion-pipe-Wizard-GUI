"""
WAN22 Cache Command Builder

This module handles building commands for WAN22 caching operations:
- Latent caching
"""

import os
from typing import Dict, List, Optional
from pathlib import Path


class WAN22Cache:
    """Builds commands for WAN22 caching operations."""

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

    def get_config_value(self, config: Dict, *keys, default=None):
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        return config

    def parse_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'on')

    def build_cache_latents_command(
        self,
        dataset_config: str,
        vae_path: str,
        i2v: bool = False,
        clip_path: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 2,
        device: str = "cuda"
    ) -> List[str]:
        script = str(self.musubi_root / "wan_cache_latents.py")

        cmd = [
            "python", script,
            "--dataset_config", dataset_config,
            "--vae", vae_path,
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--device", device
        ]

        if i2v:
            cmd.append("--i2v")
            if clip_path:
                cmd.extend(["--clip", clip_path])

        return cmd

    def build_cache_text_encoder_command(
        self,
        dataset_config: str,
        t5_path: str,
        fp8_t5: bool = False,
        batch_size: int = 1,
        num_workers: int = 2,
        device: str = "cuda"
    ) -> List[str]:
        script = str(self.musubi_root / "wan_cache_text_encoder_outputs.py")

        cmd = [
            "python", script,
            "--dataset_config", dataset_config,
            "--t5", t5_path,
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--device", device
        ]

        if fp8_t5:
            cmd.append("--fp8_t5")

        return cmd

    def build_all_cache_commands(
        self,
        config: Dict,
        dataset_config: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[str]]:
        model = config.get('model', {})
        training_strategy = config.get('training_strategy', {})
        acceleration = config.get('acceleration', {})

        # Get ckpt_path from model section and resolve to absolute path
        ckpt_path = model.get('ckpt_path', '')
        if ckpt_path and not os.path.isabs(ckpt_path):
            ckpt_path = str(self.project_root / ckpt_path)

        # Detect I2V mode: check wan_task field first, then training_strategy.i2v
        wan_task = model.get('wan_task', '').lower()
        i2v = 'i2v' in wan_task or self.parse_bool(training_strategy.get('i2v', False))

        # Check 8_bit_te from acceleration section first, then top-level (diffusion-pipe format)
        fp8_t5 = self.parse_bool(acceleration.get('8_bit_te', config.get('8_bit_te', False)))

        vae_path = os.path.join(ckpt_path, 'Wan2.1_VAE.pth') if ckpt_path else ''
        t5_path = os.path.join(ckpt_path, 'models_t5_umt5-xxl-enc-bf16.pth') if ckpt_path else ''

        # CLIP path for I2V mode (commented out for now)
        clip_path = None
        # if i2v and ckpt_path:
        #     clip_path = os.path.join(ckpt_path, 'models_clip_openclip ViT-H-14.safetensors')

        commands = {}
        commands['latents'] = self.build_cache_latents_command(
            dataset_config=dataset_config,
            vae_path=vae_path,
            i2v=i2v,
            clip_path=clip_path
        )
        commands['text_encoder'] = self.build_cache_text_encoder_command(
            dataset_config=dataset_config,
            t5_path=t5_path,
            fp8_t5=fp8_t5
        )
        return commands

    def format_all_cache_commands(self, config: Dict, dataset_config: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        commands = self.build_all_cache_commands(config, dataset_config, output_dir)
        return {
            cmd_type: " ".join(cmd_args)
            for cmd_type, cmd_args in commands.items()
        }


if __name__ == "__main__":
    import sys
    import toml

    if len(sys.argv) < 2:
        print("Usage: python wan22_cache.py <last_config.toml> [dataset_config.toml]")
        sys.exit(1)

    config_path = sys.argv[1]
    dataset_config = sys.argv[2] if len(sys.argv) > 2 else None

    if not dataset_config:
        ws_dir = os.path.dirname(config_path)
        dataset_config = os.path.join(ws_dir, 'last_data_musubi_config.toml')

    with open(config_path, 'r') as f:
        config = toml.load(f)

    cache = WAN22Cache()
    commands = cache.build_all_cache_commands(config, dataset_config)

    print("\n=== WAN22 Cache Commands ===\n")
    for cmd_type, cmd in commands.items():
        print(f"[{cmd_type.upper()}]")
        print(" ".join(cmd))
        print()
