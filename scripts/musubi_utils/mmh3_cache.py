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
import toml
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
    def _ensure_max_frames_from_frame_buckets(dataset_config_path: str) -> str:
        """Read dataset TOML and set max_frames = max(frame_buckets/target_frames) per dataset if missing.

        Modifies the original file in place (same pattern as LTX2). Returns the same path.
        """
        try:
            with open(dataset_config_path, 'r') as f:
                cfg = toml.load(f)
        except Exception as e:
            logger.warning(f"Failed to load dataset config for max_frames fixup: {e}")
            return dataset_config_path

        changed = False
        datasets = cfg.get('datasets', [])
        for ds in datasets:
            frame_buckets = ds.get('frame_buckets') or ds.get('target_frames')
            if isinstance(frame_buckets, list) and len(frame_buckets) > 0:
                existing_max = ds.get('max_frames')
                desired_max = max(frame_buckets)
                if not existing_max or existing_max != desired_max:
                    ds['max_frames'] = desired_max
                    changed = True

        if not changed:
            return dataset_config_path

        try:
            with open(dataset_config_path, 'w') as f:
                toml.dump(cfg, f)
            logger.info(f"Updated H3 dataset config with max_frames from frame_buckets: {dataset_config_path}")
        except Exception as e:
            logger.warning(f"Failed to write max_frames back to dataset config (will still cache): {e}")

        return dataset_config_path

    @staticmethod
    def _dataset_has_video_targets(dataset_config_path: str) -> bool:
        """True if any dataset entry has video targets (vs image/audio-only)."""
        try:
            with open(dataset_config_path, 'r') as f:
                cfg = toml.load(f)
        except Exception:
            return True  # unreadable config: keep the fl2va default
        for ds in cfg.get('datasets', []):
            if ds.get('target_video_directory') or ds.get('video_directory'):
                return True
        return False

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

    @staticmethod
    def _extra_flag_present(config: Dict, flag_name: str) -> bool:
        """Detect --flag or --flag=... in the top-level extra_flags string."""
        extra = str(config.get("extra_flags", "") or "").strip()
        if not extra:
            return False
        tokens = extra.split()
        return any(t == flag_name or t.startswith(f"{flag_name}=") for t in tokens)

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
        # Ensure max_frames is set from frame_buckets before caching
        dataset_config = self._ensure_max_frames_from_frame_buckets(dataset_config)

        model = config.get("model", {})
        training_strategy = config.get("training_strategy", {})

        vae_path = self._get(model, "vae_path", "")
        vae_audio_path = self._get(model, "vae_audio_path", "")
        text_encoder_path = self._get(model, "text_encoder_path", "")
        tokenizer_path = self._get(model, "tokenizer_path", "")

        # Use h3_training_mode from config to determine cache task type.
        # Slider variants are not cache tasks: img_slider (reference mode)
        # caches as plain fl2va; txt_slider and visual_slider need no dataset
        # caching at all (their prompts/presentations come from the slider
        # TOML) but map to fl2va here so any requested cache run stays valid.
        raw_task = str(self._get(training_strategy, "h3_training_mode", "fl2va")).strip().lower()
        task = "fl2va" if raw_task in ("img_slider", "txt_slider", "visual_slider") else raw_task
        # Image targets have no first/last frames for FL2VA conditioning; the
        # img slider on an image-only dataset caches as plain text (t2va) and
        # trains with the fl2va flag (the DiT consumes cached hidden states as-is).
        if task == "fl2va" and raw_task == "img_slider" and not self._dataset_has_video_targets(dataset_config):
            task = "t2va"
            logger.info("H3 img slider: image-only dataset, caching text encoder with --task t2va")

        quantization = self._resolve_quantization(config)

        # Guidance-consistent H3 training reads the empty-text branch from
        # cache, so the cache must be written with --cache_guidance_empty.
        # Pass the flag through when the user sets it in extra_flags.
        cache_guidance_empty = self._extra_flag_present(config, "--cache_guidance_empty")

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
                cache_guidance_empty=cache_guidance_empty,
            )
            if cache_guidance_empty:
                logger.info("Detected --cache_guidance_empty in extra_flags; enabling it for text-encoder cache")
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
