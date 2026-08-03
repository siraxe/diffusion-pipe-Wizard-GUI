"""
MiniMax H3 cache command builder.

Mirrors the LTX2Cache layout: builds latent + text-encoder caching commands
for musubi-tuner's H3 scripts, and exposes a build_all_cache_commands entry
point that returns one command per cache stage.

H3 specifics:
  - latents need both a video VAE and an audio VAE;
  - text-encoder caching needs the BF16 safetensors checkpoint plus the
    released FL2VA text_encoder directory passed to --tokenizer;
  - --text_encoder_quantization is mutually exclusive: int8 (8_bit_te) or
    nf4 (nf4_te), never both.
"""

import os
import sys
from typing import Dict, List, Optional
from pathlib import Path

from loguru import logger


# H3-specific script names under diffusion-trainers/musubi-tuner/
H3_CACHE_LATENTS_SCRIPT = "minimax_h3_cache_latents.py"
H3_CACHE_TEXT_SCRIPT = "minimax_h3_cache_text_encoder_outputs.py"


class MMH3Cache:
    """Builds MiniMax H3 latent and text-encoder cache commands."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.musubi_root = self.project_root / "diffusion-trainers" / "musubi-tuner"

    # ----------------------------------------------------------------------
    # Path / config helpers
    # ----------------------------------------------------------------------

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
    # Individual command builders
    # ----------------------------------------------------------------------

    def build_cache_latents_command(
        self,
        dataset_config: str,
        vae: str,
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
            "--vae", self._resolve_path(vae),
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
    # Batch command building (config-driven)
    # ----------------------------------------------------------------------

    def _resolve_quantization(self, config: Dict) -> str:
        """8_bit_te and nf4_te are mutually exclusive. nf4 wins if both are set."""
        if self.parse_bool(config.get("nf4_te", False)):
            return "nf4"
        if self.parse_bool(config.get("8_bit_te", False)):
            return "int8"
        return "none"

    def build_all_cache_commands(
        self,
        config: Dict,
        dataset_config: str,
        slider_config: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Build all H3 cache commands from the config dict.

        Returns a dict keyed by cache stage ('latents', 'text_encoder').
        Stages missing required model paths are skipped (not emitted).
        """
        model = config.get("model", {})
        training_strategy = config.get("training_strategy", {})

        vae_path = self._get(model, "vae_path", "")
        vae_audio_path = self._get(model, "vae_audio_path", "")
        text_encoder_path = self._get(model, "text_encoder_path", "")
        tokenizer_path = self._get(model, "tokenizer_path", "")

        # task maps to h3_training_mode (default fl2va; ref2va is rejected by the backend at train time,
        # but the cache script accepts both for conditioning layout).
        task = self._get(training_strategy, "h3_training_mode", "fl2va")

        quantization = self._resolve_quantization(config)

        commands: Dict[str, List[str]] = {}

        if vae_path and vae_audio_path:
            commands["latents"] = self.build_cache_latents_command(
                dataset_config=dataset_config,
                vae=vae_path,
                audio_vae=vae_audio_path,
            )
        else:
            logger.warning("H3 latents cache skipped: vae_path and vae_audio_path are required")

        if text_encoder_path and tokenizer_path:
            commands["text_encoder"] = self.build_cache_text_encoder_command(
                dataset_config=dataset_config,
                text_encoder=text_encoder_path,
                tokenizer=tokenizer_path,
                task=task,
                text_encoder_quantization=quantization,
            )
        else:
            logger.warning("H3 text-encoder cache skipped: text_encoder_path and tokenizer_path are required")

        return commands

    def format_all_cache_commands(
        self,
        config: Dict,
        dataset_config: str,
        slider_config: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        commands = self.build_all_cache_commands(config, dataset_config, slider_config, output_dir)
        return {stage: " ".join(args) for stage, args in commands.items()}


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------

def create_cache_commands(
    config: Dict,
    dataset_config: str,
    slider_config: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    return MMH3Cache().build_all_cache_commands(config, dataset_config, slider_config, output_dir)


def format_cache_commands(
    config: Dict,
    dataset_config: str,
    slider_config: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    return MMH3Cache().format_all_cache_commands(config, dataset_config, slider_config, output_dir)
