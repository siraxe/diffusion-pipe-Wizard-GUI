"""LTX-2 model handler - processes ltx-video-2 model configurations"""
import json
import math
import toml
import os
import subprocess
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _build_ltx2_pythonpath(current_dir: str, existing_pythonpath: str = '') -> str:
    """Build PYTHONPATH with LTX-2 package src directories."""
    ltx2_base = os.path.join(current_dir, "diffusion-trainers/LTX-2/packages")
    paths = [
        os.path.join(ltx2_base, "ltx-core/src"),
        os.path.join(ltx2_base, "ltx-pipelines/src"),
        os.path.join(ltx2_base, "ltx-trainer/src"),
    ]
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    return ":".join(paths)


def _get_popen_kwargs(cwd: str, env: dict) -> dict:
    """Configure subprocess.Popen kwargs with platform-specific settings."""
    kwargs = {
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    try:
        if os.name == 'posix':
            kwargs["preexec_fn"] = os.setsid
        elif os.name == 'nt':
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    except Exception:
        pass
    return kwargs


def _get_media_resolution(media_path: str) -> tuple[int, int] | None:
    """Get the resolution (width, height) of a video or image file.
    Returns None if the file type is not supported or cannot be read.
    """
    media_path = Path(media_path)
    suffix = media_path.suffix.lower()

    # For images, use PIL
    if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        try:
            from PIL import Image
            with Image.open(media_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"[DEBUG] Could not read image {media_path}: {e}")
            return None

    # For videos, use av/PyAV
    elif suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"]:
        try:
            import av
            with av.open(str(media_path)) as container:
                video_stream = container.streams.video[0]
                return (video_stream.width, video_stream.height)
        except Exception as e:
            print(f"[DEBUG] Could not read video {media_path}: {e}")
            return None

    return None


def _get_unique_aspect_ratios(dataset_json: str, sample_size: int = 10) -> list[tuple[float, int, int]]:
    """Get unique aspect ratios from the dataset by sampling media files.
    Returns list of tuples: (aspect_ratio, width, height) for unique aspect ratios found.
    """
    aspect_ratios = {}

    try:
        with open(dataset_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        # Determine media column name (common variants)
        media_column = None
        for col in ['media_path', 'video_path', 'video', 'path']:
            if col in data[0]:
                media_column = col
                break

        if not media_column:
            print("[DEBUG] No media column found in dataset")
            return []

        # Get data root for relative paths
        data_root = Path(dataset_json).parent

        # Sample files to find unique aspect ratios
        sample_count = min(sample_size, len(data))
        for i in range(sample_count):
            media_rel_path = data[i].get(media_column)
            if not media_rel_path:
                continue

            media_path = data_root / media_rel_path
            resolution = _get_media_resolution(str(media_path))

            if resolution:
                width, height = resolution
                aspect_ratio = round(width / height, 4)  # Round to avoid float precision issues
                if aspect_ratio not in aspect_ratios:
                    aspect_ratios[aspect_ratio] = (width, height)

        # Convert to list of tuples sorted by aspect ratio
        result = [(ar, w, h) for ar, (w, h) in aspect_ratios.items()]
        result.sort(key=lambda x: x[0])
        return result

    except Exception as e:
        print(f"[DEBUG] Error reading dataset for aspect ratios: {e}")
        return []


def _calculate_resolution_for_aspect_ratio(
    target_pixels: int,
    aspect_ratio: float,
    divisor: int = 32
) -> tuple[int, int]:
    """Calculate width and height that:
    - Has approximately target_pixels area (width * height)
    - Has the given aspect ratio (width / height)
    - Both dimensions are divisible by divisor (default 32)

    Example:
        target_pixels = 512 * 512 = 262144
        aspect_ratio = 715 / 366 ≈ 1.9536
        Returns: (704, 352) -> area = 247,808 pixels, 704/32=22, 352/32=11
    """
    # Calculate ideal dimensions from area and aspect ratio
    # area = width * height
    # aspect_ratio = width / height
    # Therefore: height = sqrt(area / aspect_ratio), width = height * aspect_ratio

    ideal_height = math.sqrt(target_pixels / aspect_ratio)
    ideal_width = ideal_height * aspect_ratio

    # Find closest values divisible by divisor
    height = int(round(ideal_height / divisor)) * divisor
    width = int(round(ideal_width / divisor)) * divisor

    # Ensure minimum size
    if height < divisor:
        height = divisor
    if width < divisor:
        width = divisor

    return (width, height)


def _adjust_to_divisible_by_32(value: int) -> int:
    """Adjust a value to be divisible by 32 by rounding to nearest multiple.
    Example: 896 -> 896, 897 -> 896, 913 -> 928
    """
    return int(round(value / 32)) * 32


def _create_captions_from_media_files(dataset_folder_path: str) -> bool:
    """Create captions.json from media files in the dataset folder with empty captions.
    Uses the existing load_dataset_captions function from dataset_utils.
    Returns True if captions.json was created, False if error or no media files found.
    """
    try:
        from flet_app.ui.dataset_manager.dataset_utils import load_dataset_captions
        from flet_app.settings import settings

        # Extract dataset name from folder path
        # dataset_folder_path should be like: /path/to/datasets/dataset_name
        if settings.DATASETS_DIR in dataset_folder_path:
            dataset_name = os.path.relpath(dataset_folder_path, settings.DATASETS_DIR)
        else:
            dataset_name = os.path.basename(dataset_folder_path)

        # Use existing function to get captions data
        captions_data = load_dataset_captions(dataset_name)

        if not captions_data:
            print(f"[Auto-Caption] No media files found in {dataset_folder_path}")
            return False

        # Write to captions.json
        captions_json_path = os.path.join(dataset_folder_path, "captions.json")
        with open(captions_json_path, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, indent=4)

        print(f"[Auto-Caption] Created captions.json with {len(captions_data)} entries (empty captions)")
        return True

    except Exception as e:
        print(f"[Auto-Caption] Error creating captions.json from media files: {e}")
        return False


def _create_captions_from_txt(dataset_folder_path: str, dataset_type: str = "video") -> bool:
    """Create captions.json from .txt files in the dataset folder.
    Returns True if captions.json was created/updated, False if error or no .txt files found.
    """
    try:
        import glob
        from flet_app.ui import settings

        captions_json_path = os.path.join(dataset_folder_path, "captions.json")

        # Load existing captions if any
        if os.path.exists(captions_json_path):
            with open(captions_json_path, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            if not isinstance(captions_data, list):
                captions_data = []
        else:
            captions_data = []

        # Get media extensions based on dataset type
        if dataset_type == "image":
            media_extensions = settings.IMAGE_EXTENSIONS
        else:
            media_extensions = list(dict.fromkeys(settings.VIDEO_EXTENSIONS + settings.IMAGE_EXTENSIONS))

        # Get all media files
        media_files = []
        for ext in media_extensions:
            media_files.extend(glob.glob(os.path.join(dataset_folder_path, f"*{ext}")))
            media_files.extend(glob.glob(os.path.join(dataset_folder_path, f"*{ext.upper()}")))

        media_files = sorted(list(set(os.path.normpath(f) for f in media_files)))

        # Build captions dict from existing data
        captions_dict = {os.path.basename(item['media_path']): item for item in captions_data if 'media_path' in item}

        # Update captions from .txt files
        updated_count = 0
        for media_path in media_files:
            base_filename, _ = os.path.splitext(os.path.basename(media_path))
            txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")

            if os.path.exists(txt_caption_path):
                with open(txt_caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()

                media_basename = os.path.basename(media_path)
                if media_basename in captions_dict:
                    captions_dict[media_basename]['caption'] = caption_text
                else:
                    captions_dict[media_basename] = {
                        "media_path": media_basename,
                        "caption": caption_text
                    }
                updated_count += 1

        if updated_count > 0:
            with open(captions_json_path, 'w', encoding='utf-8') as f:
                json.dump(list(captions_dict.values()), f, indent=4)
            print(f"[Auto-Caption] Created/updated captions.json with {updated_count} entries from .txt files")
            return True
        else:
            print("[Auto-Caption] No .txt files found to create captions.json")
            return False

    except Exception as e:
        print(f"[Auto-Caption] Error creating captions.json: {e}")
        return False


def _build_resolution_buckets(
    resolutions: list,
    frame_buckets: list,
    dataset_json: str | None = None,
    ar_buckets: list | None = None
) -> str:
    """Build resolution-buckets string from resolutions and frame buckets.

    Priority order:
    1. If ar_buckets is non-empty and resolutions is empty: use ar_buckets directly
    2. If dataset_json is provided and resolutions is non-empty: calculate resolution buckets
       based on the actual aspect ratios found in the dataset media files
    3. Fallback to square resolutions from resolutions list

    Args:
        resolutions: List of target resolution values (e.g., [512, 768])
        frame_buckets: List of frame counts (e.g., [1, 33, 65])
        dataset_json: Optional path to dataset JSON file for auto-resolution
        ar_buckets: List of [width, height] pairs (e.g., [[896, 1216], [512, 675]])

    Returns:
        Resolution bucket string like "704x352x33;512x512x49"
    """
    buckets = []

    # Priority 1: Use ar_buckets if resolutions is empty and ar_buckets has values
    if ar_buckets and not resolutions:
        print(f"\n[ar_buckets] Using ar_buckets (resolutions is empty):")
        for width, height in ar_buckets:
            # Adjust to be divisible by 32
            adjusted_width = _adjust_to_divisible_by_32(width)
            adjusted_height = _adjust_to_divisible_by_32(height)

            if adjusted_width != width or adjusted_height != height:
                print(f"  - {width}x{height} -> {adjusted_width}x{adjusted_height} (adjusted for divisibility by 32)")
            else:
                print(f"  - {adjusted_width}x{adjusted_height}")

            for fb in frame_buckets:
                buckets.append(f"{adjusted_width}x{adjusted_height}x{fb}")

        if buckets:
            return ";".join(buckets)

    # Priority 2: Calculate from dataset aspect ratios
    if dataset_json and resolutions and frame_buckets:
        # Get unique aspect ratios from the dataset
        aspect_ratios = _get_unique_aspect_ratios(dataset_json)

        if aspect_ratios:
            print(f"\n[Auto-Resolution] Found {len(aspect_ratios)} unique aspect ratio(s) in dataset:")
            for ar, w, h in aspect_ratios:
                print(f"  - {w}x{h} (aspect ratio: {ar:.4f})")

            # Calculate resolution buckets for each target resolution and aspect ratio
            for target_res in resolutions:
                target_pixels = target_res * target_res  # Area of square resolution

                for aspect_ratio, orig_w, orig_h in aspect_ratios:
                    width, height = _calculate_resolution_for_aspect_ratio(target_pixels, aspect_ratio)

                    for fb in frame_buckets:
                        buckets.append(f"{width}x{height}x{fb}")

                    print(f"[Auto-Resolution] Target {target_res}x{target_res} ({target_pixels:,} pixels) "
                          f"-> {width}x{height} ({width * height:,} pixels) for aspect ratio {aspect_ratio:.4f}")

            if buckets:
                return ";".join(buckets)

    # Priority 3: Fallback to square resolutions
    buckets = []
    if resolutions and frame_buckets:
        for res in resolutions:
            for fb in frame_buckets:
                buckets.append(f"{res}x{res}x{fb}")
    return ";".join(buckets) if buckets else "512x512x33"


def _extract_config_data(result: dict) -> tuple:
    """Extract and return configuration data from conversion result."""
    return (
        result.get('frame_buckets', [1, 33, 76]),
        result.get('resolutions', []),
        result.get('preprocessed_data_root'),
        result.get('model_path'),
        result.get('text_encoder_path'),
        result.get('yaml_path'),
        result.get('num_repeats', 1),
        result.get('ar_buckets', []),
    )


def _print_config_info(yaml_path: str, preprocessed_data_root: str, frame_buckets: list, resolutions: list, ar_buckets: list | None = None):
    """Print configuration information."""
    print(f"LTX2 - Converted config to: {yaml_path}")
    print(f"\ndata:")
    print(f"  preprocessed_data_root: {preprocessed_data_root}")
    print(f"\nDataset configuration:")
    print(f"  frame_buckets: {frame_buckets if frame_buckets else 'Not found'}")
    print(f"  resolutions: {resolutions if resolutions else 'Not found'}")
    if ar_buckets is not None:
        print(f"  ar_buckets: {ar_buckets if ar_buckets else 'Not found'}")


def _print_command(cmd_str: str, title: str = "Command"):
    """Print formatted command."""
    print("\n" + "="*80)
    print(f"{title}:")
    print("="*80)
    print(cmd_str)
    print("="*80 + "\n")


async def run_process_dataset(config_path: str, use_last_config: bool = False):
    """Run process_dataset.py for LTX-2 model. Returns process immediately for streaming."""
    from flet_app.ui.utils.process_cleanup import kill_existing_training_processes

    # Kill any existing training processes before starting new one
    kill_existing_training_processes()

    def _run():
        from flet_app.ui.utils.toml_to_yaml import convert_toml_to_ltx2_yaml

        # If use_last_config is True, check if YAML already exists and reuse it
        yaml_path = os.path.splitext(config_path)[0] + ".yaml"
        if use_last_config and os.path.exists(yaml_path):
            print(f"[Last Config] Reusing existing YAML: {yaml_path}")
            # Still need to extract config data, so read the existing YAML
            import yaml as yaml_lib
            with open(yaml_path, 'r') as f:
                yaml_config = yaml_lib.safe_load(f)

            # Extract data from existing YAML
            preprocessed_data_root = yaml_config.get('data', {}).get('preprocessed_data_root', '')
            # Remove /.precomputed suffix if present to get original root
            if preprocessed_data_root and preprocessed_data_root.endswith('/.precomputed'):
                preprocessed_data_root = preprocessed_data_root[:-14]

            # Build minimal result dict from existing YAML
            result = {
                'yaml_path': yaml_path,
                'frame_buckets': yaml_config.get('frame_buckets', [1, 33, 76]),
                'resolutions': yaml_config.get('resolutions', []),
                'ar_buckets': yaml_config.get('ar_buckets', []),
                'preprocessed_data_root': preprocessed_data_root,
                'model_path': yaml_config.get('model', {}).get('model_path', ''),
                'text_encoder_path': yaml_config.get('model', {}).get('text_encoder_path', ''),
                'num_repeats': yaml_config.get('num_repeats', 1),
            }
        else:
            # Convert TOML to YAML as usual
            result = convert_toml_to_ltx2_yaml(config_path)

        frame_buckets, resolutions, preprocessed_data_root, model_path, text_encoder_path, yaml_path, num_repeats, ar_buckets = _extract_config_data(result)
        _print_config_info(yaml_path, preprocessed_data_root, frame_buckets, resolutions, ar_buckets)

        # Debug: Show config source
        print(f"\n[DEBUG] Config file: {config_path}")
        print(f"[DEBUG] Extracted values -> frame_buckets: {frame_buckets}, resolutions: {resolutions}, ar_buckets: {ar_buckets}")

        dataset_json = os.path.join(preprocessed_data_root, 'captions.json')

        # Check if captions.json exists, if not try to create from .txt files
        if not os.path.exists(dataset_json):
            print(f"\n[Auto-Caption] captions.json not found at: {dataset_json}")
            print(f"[Auto-Caption] Attempting to create captions.json from .txt files...")
            if _create_captions_from_txt(preprocessed_data_root):
                print(f"[Auto-Caption] Successfully created captions.json from .txt files")
            else:
                print(f"[Auto-Caption] No .txt files found, creating captions.json from media files with empty captions...")
                if _create_captions_from_media_files(preprocessed_data_root):
                    print(f"[Auto-Caption] Successfully created captions.json from media files (empty captions)")
                    print(f"[Auto-Caption] Note: You can edit captions.json to add captions or use .txt files for future runs")
                else:
                    print(f"[Auto-Caption] Error: Could not create captions.json")
                    raise FileNotFoundError(f"Required file not found: {dataset_json}. Could not create from media files.")

        # Double-check the file exists before proceeding
        if not os.path.exists(dataset_json):
            raise FileNotFoundError(f"Required file not found: {dataset_json}. Please create captions.json first.")

        resolution_buckets_str = _build_resolution_buckets(resolutions, frame_buckets, dataset_json, ar_buckets)

        # Debug: Show final resolution buckets string
        print(f"\n[DEBUG] Final resolution-buckets string: {resolution_buckets_str}")
        current_dir = os.getcwd()
        script_path = os.path.join(current_dir, "diffusion-trainers/LTX-2/packages/ltx-trainer/scripts/process_dataset.py")

        # Read TOML config to get 8_bit_text_encoder and with_audio values
        use_8bit = False
        no_audio = False
        enable_ar_bucket = False
        try:
            with open(config_path, 'r') as f:
                toml_config = toml.load(f)
            # Check acceleration section for 8_bit_text_encoder
            acceleration = toml_config.get('acceleration', {})
            if acceleration.get('8_bit_text_encoder', True):
                use_8bit = True
            # Check training_strategy section for with_audio
            training_strategy = toml_config.get('training_strategy', {})
            with_audio = training_strategy.get('with_audio', True)
            if not with_audio:
                no_audio = True
            # Check enable_ar_bucket for smart AR routing
            if toml_config.get('enable_ar_bucket', False):
                enable_ar_bucket = True
        except Exception as e:
            print(f"[DEBUG] Could not read TOML config for flags: {e}")

        # Add -u flag for unbuffered output
        cmd = [
            "python", "-u",
            script_path,
            dataset_json,
            "--resolution-buckets",
            resolution_buckets_str,
            "--model-path",
            model_path,
            "--text-encoder-path",
            text_encoder_path,
            "--batch-size",
            str(num_repeats),
        ]

        # Add --load-text-encoder-in-8bit flag if 8_bit_text_encoder is True
        if use_8bit:
            cmd.append("--load-text-encoder-in-8bit")
            print("[LTX2] Added --load-text-encoder-in-8bit flag (8-bit text encoder enabled)")

        # Add --with-audio flag only if with_audio is True (default is False)
        if not no_audio:
            cmd.append("--with-audio")
            print("[LTX2] Added --with-audio flag (audio processing enabled)")

        # Add --enable-ar-bucket flag if enable_ar_bucket is True
        if enable_ar_bucket:
            cmd.append("--enable-ar-bucket")
            print("[LTX2] Added --enable-ar-bucket flag (smart AR routing enabled)")

        # Quote the resolution-buckets argument for shell-safe display
        cmd_str = " ".join(
            f'"{arg}"' if " " in arg or ";" in arg else arg
            for arg in cmd
        )
        _print_command(cmd_str, "Process Dataset Command")
        print("LTX2")

        env = os.environ.copy()
        env['PYTHONPATH'] = _build_ltx2_pythonpath(current_dir, env.get('PYTHONPATH', ''))
        env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
        popen_kwargs = _get_popen_kwargs(current_dir, env)

        # Create process but DON'T wait - let caller handle streaming
        proc = subprocess.Popen(cmd, **popen_kwargs)
        gc.collect()
        return proc, cmd_str

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        proc, cmd_str = await loop.run_in_executor(pool, _run)
    return proc, cmd_str


async def run_ltx_training(yaml_config_path: str, cache_only: bool = False):
    """Run LTX-2 training script with Python."""
    from flet_app.ui.utils.process_cleanup import kill_existing_training_processes

    # Kill any existing training processes before starting new one
    kill_existing_training_processes()

    def _run():
        import sys
        current_dir = os.getcwd()
        python_exe = sys.executable or "python"
        script_path = os.path.join(current_dir, "diffusion-trainers/LTX-2/packages/ltx-trainer/scripts/train.py")

        # Add -u flag for unbuffered output and --disable-progress-bars for text logging
        cmd = [python_exe, "-u", script_path, yaml_config_path, "--disable-progress-bars"]
        cmd_str = " ".join(cmd)
        _print_command(cmd_str, "LTX-2 Training Command")

        env = os.environ.copy()
        env['PYTHONPATH'] = _build_ltx2_pythonpath(current_dir, env.get('PYTHONPATH', ''))
        env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
        popen_kwargs = _get_popen_kwargs(current_dir, env)

        # Use PIPE for stdout to enable streaming output
        popen_kwargs['stdout'] = subprocess.PIPE
        popen_kwargs['stderr'] = subprocess.STDOUT

        proc = subprocess.Popen(cmd, **popen_kwargs)
        gc.collect()
        return proc, cmd_str

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        proc, cmd_str = await loop.run_in_executor(pool, _run)
    return proc, cmd_str


def _is_ltx2_in_content(content: str) -> bool:
    """Check if ltx-video-2 is present in file content."""
    return 'ltx-video-2' in content.lower()


def _is_ltx2_in_toml(content: str) -> bool:
    """Check if model type is ltx-video-2 via TOML parsing."""
    try:
        config = toml.loads(content)
        model_type = config.get('model', {}).get('type', '')
        return str(model_type).strip().lower() == 'ltx-video-2'
    except Exception as e:
        print(f"[DEBUG] TOML parsing failed: {e}")
        return False


def _read_config_file(config_path: str) -> str:
    """Read configuration file content."""
    with open(config_path, 'r') as f:
        return f.read()


def handle_ltx_model(config_path):
    """Check if model type is ltx-video-2."""
    try:
        content = _read_config_file(config_path)

        if _is_ltx2_in_content(content):
            print(f"[DEBUG] Detected LTX-2 model in config")
            return True

        if _is_ltx2_in_toml(content):
            print(f"[DEBUG] Detected LTX-2 model via TOML parsing")
            return True

        return False
    except Exception as e:
        print(f"[DEBUG] Could not check model type: {e}")
        return False
