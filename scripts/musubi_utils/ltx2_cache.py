import os
import sys
import toml
import glob
from typing import Dict, List, Optional
from pathlib import Path

# Use loguru for consistent logging
from loguru import logger


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
            "--batch_size", str(batch_size),
            # "--vae_spatial_tile_size", "512",
            "--vae_temporal_tile_size", "96",
            "--vae_temporal_tile_overlap", "24",
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
    # Slider Mode Control Folder Caching
    # ==========================================================================

    def _create_control_dataset_config(
        self,
        dataset_config: str,
        control_dir: str,
        neg_cache_dir: str,
        pos_cache_dir: Optional[str] = None,
    ) -> str:
        """Create a temporary dataset config for caching the control folder.

        Args:
            dataset_config: Path to the original dataset config
            control_dir: Path to the control folder
            neg_cache_dir: Path to the negative cache directory
            pos_cache_dir: Path to positive cache dir (optional, used to match frame counts)

        Returns:
            Path to the created config file.
        """
        # Read the original dataset config
        with open(dataset_config, 'r') as f:
            orig_config = toml.load(f)

        # Get the general section (for caption extension, batch_size, etc.)
        general = orig_config.get('general', {})

        first_ds = orig_config.get('datasets', [{}])[0] if 'datasets' in orig_config else {}

        # For slider mode, control images don't need captions
        # Create empty caption files so the dataset loader doesn't filter them out
        if os.path.exists(control_dir):
            # Track which images already have captions to avoid duplicates
            existing_captions = set()
            caption_files = glob.glob(os.path.join(control_dir, "*.txt"))
            for cf in caption_files:
                existing_captions.add(os.path.splitext(os.path.basename(cf))[0])

            # Create empty caption files for media that don't have them
            media_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.PNG', '.JPG', '.JPEG', '.WEBP', '.BMP',
                          '.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM']
            for ext in media_exts:
                for media_file in glob.glob(os.path.join(control_dir, f"*{ext}")):
                    media_base = os.path.splitext(os.path.basename(media_file))[0]
                    if media_base not in existing_captions:
                        caption_file = os.path.join(control_dir, media_base + ".txt")
                        with open(caption_file, 'w') as f:
                            f.write("")  # Empty caption file
                        existing_captions.add(media_base)

        # Create a minimal dataset config for the control folder
        control_config = {
            'general': general,
            'datasets': []
        }

        # Check if the control directory has images or videos
        has_images = False
        if os.path.exists(control_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.PNG', '.JPG', '.JPEG', '.WEBP', '.BMP']:
                if glob.glob(os.path.join(control_dir, f"*{ext}")):
                    has_images = True
                    break

        if has_images:
            # Create dataset entry for images
            control_dataset = {
                'image_directory': control_dir,
                'cache_directory': neg_cache_dir,
                'num_repeats': 1,
                'enable_bucket': general.get('enable_bucket', True),
                'bucket_no_upscale': general.get('bucket_no_upscale', False),
            }
        else:
            # Create dataset entry for videos
            control_dataset = {
                'video_directory': control_dir,
                'cache_directory': neg_cache_dir,
                'num_repeats': 1,
                'enable_bucket': general.get('enable_bucket', True),
                'bucket_no_upscale': general.get('bucket_no_upscale', False),
            }
            # Copy video-specific settings from the first dataset in original config
            for key in ['target_frames', 'frame_extraction', 'target_fps', 'max_frames', 'enable_mask']:
                if key in first_ds:
                    control_dataset[key] = first_ds[key]


        # Copy resolution and AR bucketing settings from original dataset
        if 'resolution' in first_ds:
            control_dataset['resolution'] = first_ds['resolution']
        for key in ['enable_ar_bucket', 'min_ar', 'max_ar', 'num_ar_buckets']:
            if key in first_ds:
                control_dataset[key] = first_ds[key]

        control_config['datasets'].append(control_dataset)

        # Create temporary config file in the workspace
        temp_config_path = dataset_config.replace('.toml', '_control.toml')
        with open(temp_config_path, 'w') as f:
            toml.dump(control_config, f)

        logger.info(f"Created temporary control dataset config: {temp_config_path}")
        return temp_config_path



    # ==========================================================================
    # Batch Command Building
    # ==========================================================================

    def build_all_cache_commands(
        self,
        config: Dict,
        dataset_config: str,
        slider_config: Optional[str] = None,
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

        # Standard latent caching (main video dir)
        latents_cmd = self.build_cache_latents_command(
            dataset_config=dataset_config,
            ltx2_checkpoint=ltx2_checkpoint,
            ltx2_mode=ltx2_mode
        )

        # Add IC-LoRA reference caching arguments
        t_type = training_strategy.get('t_type', 'none') or 'none'
        ic_lora_enabled = (t_type == 'ic_lora')
        if ic_lora_enabled:
            ref_downscale = training_strategy.get('ref_downscale', 1)
            reference_frames = training_strategy.get('reference_frames', 1)
            latents_cmd.extend([
                "--reference_downscale", str(ref_downscale),
                "--reference_frames", str(reference_frames)
            ])

        commands['latents'] = latents_cmd

        # ==========================================================================
        # VACE Latent Caching (detects control/ directory with vid.mp4 + vid_mask.mp4)
        # ==========================================================================

        def _get_dataset_video_dir(ds_config: str) -> Optional[str]:
            """Extract video_directory path from dataset config."""
            try:
                with open(ds_config, 'r') as f:
                    cfg = toml.load(f)
                if 'datasets' in cfg and len(cfg['datasets']) > 0:
                    first_ds = cfg['datasets'][0]
                    return first_ds.get('video_directory', None)
            except Exception as e:
                logger.warning(f"Failed to read dataset config for VACE detection: {e}")
            return None

        def _detect_vace_structure(video_dir: str) -> Optional[str]:
            """
            Detect if control/ subdirectory exists with VACE structure.

            Expected structure:
                video_dir/
                    vid.mp4           # main training video
                video_dir/control/
                    vid.mp4           # control video (depth, pose, etc.)
                    vid_mask.mp4      # mask video (white=reactive)

            Returns path to control/ directory if VACE structure detected, None otherwise.
            """
            if not video_dir or not os.path.exists(video_dir):
                return None

            control_dir = os.path.join(video_dir, 'control')
            if not os.path.exists(control_dir) or not os.path.isdir(control_dir):
                return None

            # Check for VACE files (control videos and masks)
            has_vace_files = False
            for fname in os.listdir(control_dir):
                fpath = os.path.join(control_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                # Check for control video or mask files
                lower_fname = fname.lower()
                if any(lower_fname.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
                    has_vace_files = True
                    break

            if has_vace_files:
                logger.info(f"Detected VACE structure in control/ directory: {control_dir}")
                return control_dir

            return None

        # Only process VACE if t_type is vace_lora
        vace_lora_enabled = (t_type == 'vace_lora')

        # Detect VACE structure from dataset config
        video_dir = _get_dataset_video_dir(dataset_config)
        vace_control_dir = _detect_vace_structure(video_dir) if video_dir else None

        if vace_lora_enabled and vace_control_dir:
            logger.info(f"VACE-LoRA enabled - processing VACE structure")
            # Determine cache directory (control/cache_vace)
            vace_cache_dir = os.path.join(vace_control_dir, 'cache_vace')
            vace_cache_dir = os.path.abspath(vace_cache_dir)

            # Create a modified dataset config with VACE paths added
            with open(dataset_config, 'r') as f:
                vace_cfg = toml.load(f)

            # Add VACE paths to the first dataset entry
            if 'datasets' in vace_cfg and len(vace_cfg['datasets']) > 0:
                vace_cfg['datasets'][0]['vace_directory'] = vace_control_dir
                vace_cfg['datasets'][0]['vace_cache_directory'] = vace_cache_dir

                # Save modified config
                temp_vace_config = dataset_config.replace('.toml', '_vace.toml')
                with open(temp_vace_config, 'w') as f:
                    toml.dump(vace_cfg, f)
                logger.info(f"Created VACE dataset config: {temp_vace_config}")

            # Build VACE caching command using ltx2_cache_latents.py
            # The VACE caching is automatically triggered when dataset config has
            # vace_directory and vace_cache_directory set (which we added above)
            vace_cmd = self.build_cache_latents_command(
                dataset_config=temp_vace_config,
                ltx2_checkpoint=ltx2_checkpoint,
                ltx2_mode=ltx2_mode,
                vae_dtype=mixed_precision,
                batch_size=1,
                device="cuda"
            )

            commands['vace_latents'] = vace_cmd
            logger.info(f"Added VACE latent caching: {vace_control_dir} -> {vace_cache_dir}")
        elif vace_control_dir and not vace_lora_enabled:
            logger.info("VACE structure detected but vace_lora=false, skipping VACE caching")
        else:
            logger.debug("No VACE structure detected, skipping VACE caching")

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
        # Only cache if interval >= 1 to avoid caching for one-time sampling
        sample_interval = validation.get('interval', -1)
        if validation.get('prompts') and validation.get('cache_te', True) and sample_interval >= 1:
            # Use the wrapper script in scripts/musubi_utils that generates sample_prompts.txt and handles hash validation
            wrapper_script = str(self.project_root / "scripts" / "musubi_utils" / "ltx2_cache_sample.py")
            config_path = str(self.project_root / "workspace" / "last_config.toml")

            commands['sample_prompts'] = [
                "python",
                wrapper_script,
                "--config", config_path,
                "--dataset_config", dataset_config,
                "--project_root", str(self.project_root),
            ]

        # Slider mode: cache control folder to musubi_cache_negative
        t_type = training_strategy.get('t_type', 'none') or 'none'
        slider_enabled = (t_type == 'slider')
        if slider_enabled and slider_config:
            # Read the slider config to get the negative cache directory
            try:
                with open(slider_config, 'r') as f:
                    slider_cfg = toml.load(f)

                neg_cache_dir = slider_cfg.get('neg_cache_dir', None)
                pos_cache_dir = slider_cfg.get('pos_cache_dir', None)
                if not neg_cache_dir:
                    neg_cache_dirs = slider_cfg.get('neg_cache_dirs', [])
                    if neg_cache_dirs:
                        neg_cache_dir = neg_cache_dirs[0]  # Use first entry
                if neg_cache_dir:
                    # Derive control folder path (parent of musubi_cache_negative)
                    # neg_cache_dir is like: /path/to/dataset/control/musubi_cache_negative
                    # control folder is: /path/to/dataset/control
                    control_dir = os.path.dirname(neg_cache_dir)

                    # Only cache if control directory exists
                    if os.path.exists(control_dir) and os.path.isdir(control_dir):
                        # Create temporary dataset config for control folder
                        temp_control_config = self._create_control_dataset_config(
                            dataset_config,
                            control_dir,
                            neg_cache_dir,
                            pos_cache_dir=pos_cache_dir,
                        )

                        # Build cache command for control folder
                        latents_negative_cmd = self.build_cache_latents_command(
                            dataset_config=temp_control_config,
                            ltx2_checkpoint=ltx2_checkpoint,
                            ltx2_mode=ltx2_mode
                        )
                        commands['latents_negative'] = latents_negative_cmd

                        logger.info(f"Added latents_negative caching for slider mode: {control_dir} -> {neg_cache_dir}")
                    else:
                        logger.warning(f"Control directory does not exist, skipping negative caching: {control_dir}")
            except Exception as e:
                logger.error(f"Failed to add control folder caching for slider mode: {e}")

        return commands

    def format_all_cache_commands(self, config: Dict, dataset_config: str, slider_config: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, str]:
        commands = self.build_all_cache_commands(config, dataset_config, slider_config, output_dir)

        return {
            cmd_type: " ".join(cmd_args)
            for cmd_type, cmd_args in commands.items()
        }


# ==========================================================================
# Convenience Functions
# ==========================================================================

def create_cache_commands(config: Dict, dataset_config: str, slider_config: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, List[str]]:
    cache = LTX2Cache()
    return cache.build_all_cache_commands(config, dataset_config, slider_config, output_dir)


def format_cache_commands(config: Dict, dataset_config: str, slider_config: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, str]:
    cache = LTX2Cache()
    return cache.format_all_cache_commands(config, dataset_config, slider_config, output_dir)
