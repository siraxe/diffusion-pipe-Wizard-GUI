import os
from typing import Dict, List, Optional
from pathlib import Path


class LTX2Cache:
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the LTX2 cache command builder.

        Args:
            project_root: Path to the project root directory.
                        If None, will be auto-detected.
        """
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.musubi_root = self.project_root / "diffusion-trainers" / "musubi-tuner"

    @staticmethod
    def _find_project_root() -> Path:
        """Auto-detect the project root directory."""
        # Start from current directory and search upward
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

    # ==========================================================================
    # Latent Caching Commands
    # ==========================================================================

    def build_cache_latents_command(
        self,
        dataset_config: str,
        ltx2_checkpoint: str,
        ltx2_mode: str = "video",
        vae_dtype: str = "bf16",
        batch_size: int = 1,
        device: str = "cuda"
    ) -> List[str]:
        script = str(self.musubi_root / "ltx2_cache_latents.py")

        return [
            "python",
            script,
            "--dataset_config", dataset_config,
            "--ltx2_checkpoint", ltx2_checkpoint,
            "--device", device,
            "--vae_dtype", vae_dtype,
            "--ltx2_mode", ltx2_mode,
            "--batch_size", str(batch_size)
        ]

    def format_cache_latents_command(self, **kwargs) -> str:
        """Format the cache latents command as a string."""
        cmd = self.build_cache_latents_command(**kwargs)
        return " ".join(cmd)

    # ==========================================================================
    # Text Encoder Caching Commands
    # ==========================================================================

    def build_cache_text_encoder_command(
        self,
        dataset_config: str,
        ltx2_checkpoint: str,
        gemma_root: str,
        ltx2_mode: str = "video",
        mixed_precision: str = "bf16",
        gemma_load_in_8bit: bool = True,
        batch_size: int = 1,
        device: str = "cuda"
    ) -> List[str]:
        script = str(self.musubi_root / "ltx2_cache_text_encoder_outputs.py")

        cmd = [
            "python",
            script,
            "--dataset_config", dataset_config,
            "--ltx2_checkpoint", ltx2_checkpoint,
            "--gemma_root", gemma_root,
            "--device", device,
            "--mixed_precision", mixed_precision,
            "--ltx2_mode", ltx2_mode,
            "--batch_size", str(batch_size)
        ]

        if gemma_load_in_8bit:
            cmd.append("--gemma_load_in_8bit")

        return cmd

    def format_cache_text_encoder_command(self, **kwargs) -> str:
        """Format the cache text encoder command as a string."""
        cmd = self.build_cache_text_encoder_command(**kwargs)
        return " ".join(cmd)

    # ==========================================================================
    # Sample Prompts Caching Commands
    # ==========================================================================

    def build_cache_sample_prompts_command(
        self,
        dataset_config: str,
        ltx2_checkpoint: str,
        gemma_root: str,
        sample_prompts: str,
        sample_prompts_cache: str,
        ltx2_mode: str = "video",
        mixed_precision: str = "bf16",
        gemma_load_in_8bit: bool = True,
        cache_i2v: bool = True
    ) -> List[str]:
        script = str(self.musubi_root / "ltx2_cache_text_encoder_outputs.py")

        cmd = [
            "python",
            script,
            "--dataset_config", dataset_config,
            "--ltx2_checkpoint", ltx2_checkpoint,
            "--gemma_root", gemma_root,
            "--device", "cuda",
            "--mixed_precision", mixed_precision,
            "--ltx2_mode", ltx2_mode,
            "--batch_size", "1",
            "--precache_sample_prompts",
            "--sample_prompts", sample_prompts,
            "--sample_prompts_cache", sample_prompts_cache
        ]

        if gemma_load_in_8bit:
            cmd.append("--gemma_load_in_8bit")

        if cache_i2v:
            cmd.append("--cache_i2v")

        return cmd

    def format_cache_sample_prompts_command(self, **kwargs) -> str:
        """Format the cache sample prompts command as a string."""
        cmd = self.build_cache_sample_prompts_command(**kwargs)
        return " ".join(cmd)

    # ==========================================================================
    # Batch Command Building
    # ==========================================================================

    def build_all_cache_commands(
        self,
        config: Dict,
        dataset_config: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[str]]:
        model = config.get('model', {})
        training_strategy = config.get('training_strategy', {})
        acceleration = config.get('acceleration', {})
        validation = config.get('validation', {})

        ltx2_checkpoint = model.get('model_path', '')
        gemma_root = model.get('text_encoder_path', '')
        ltx2_mode = training_strategy.get('ltx_mode', 'video')
        mixed_precision = acceleration.get('mixed_precision_mode', 'bf16')
        gemma_8bit = self.parse_bool(acceleration.get('8_bit_te', True))

        output_dir = output_dir or model.get('output_dir', 'output/ltx2_lora')
        sample_dir = os.path.join(output_dir, 'sample')

        commands = {}

        # Latent caching
        commands['latents'] = self.build_cache_latents_command(
            dataset_config=dataset_config,
            ltx2_checkpoint=ltx2_checkpoint,
            ltx2_mode=ltx2_mode
        )

        # Text encoder caching
        commands['text_encoder'] = self.build_cache_text_encoder_command(
            dataset_config=dataset_config,
            ltx2_checkpoint=ltx2_checkpoint,
            gemma_root=gemma_root,
            ltx2_mode=ltx2_mode,
            mixed_precision=mixed_precision,
            gemma_load_in_8bit=gemma_8bit
        )

        # Sample prompts caching (optional)
        if validation.get('prompts') and validation.get('cache_te', True):
            sample_prompts_path = os.path.join(sample_dir, 'sample_prompts.txt')
            sample_prompts_cache = os.path.join(sample_dir, 'sample_prompts_cache.pt')

            commands['sample_prompts'] = self.build_cache_sample_prompts_command(
                dataset_config=dataset_config,
                ltx2_checkpoint=ltx2_checkpoint,
                gemma_root=gemma_root,
                sample_prompts=sample_prompts_path,
                sample_prompts_cache=sample_prompts_cache,
                ltx2_mode=ltx2_mode,
                mixed_precision=mixed_precision,
                gemma_load_in_8bit=gemma_8bit,
                cache_i2v=self.parse_bool(validation.get('cache_i2v', True))
            )

        return commands

    def format_all_cache_commands(self, config: Dict, dataset_config: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        commands = self.build_all_cache_commands(config, dataset_config, output_dir)

        return {
            cmd_type: " ".join(cmd_args)
            for cmd_type, cmd_args in commands.items()
        }


# ==========================================================================
# Convenience Functions
# ==========================================================================

def create_cache_commands(config: Dict, dataset_config: str, output_dir: Optional[str] = None) -> Dict[str, List[str]]:
    cache = LTX2Cache()
    return cache.build_all_cache_commands(config, dataset_config, output_dir)


def format_cache_commands(config: Dict, dataset_config: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    cache = LTX2Cache()
    return cache.format_all_cache_commands(config, dataset_config, output_dir)
