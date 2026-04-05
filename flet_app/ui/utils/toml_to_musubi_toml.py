"""
Convert LTX data config TOML to Musubi data config TOML format.
"""
import os
import re
import toml
from loguru import logger


# Video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'}

# Image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'}


def detect_dataset_type(directory_path: str) -> str:
    """
    Detect whether a directory contains videos or images.

    Returns 'video' if videos found, 'image' if images found, 'empty' if neither.

    Priority: video > image > empty
    """
    if not directory_path or not os.path.isdir(directory_path):
        return 'empty'

    has_videos = False
    has_images = False

    try:
        for filename in os.listdir(directory_path):
            # Skip hidden files and directories
            if filename.startswith('.') or os.path.isdir(os.path.join(directory_path, filename)):
                continue

            ext = os.path.splitext(filename)[1].lower()

            if ext in VIDEO_EXTENSIONS:
                has_videos = True
                break  # Videos have priority, we can stop checking

            if ext in IMAGE_EXTENSIONS:
                has_images = True

    except (OSError, PermissionError):
        return 'empty'

    if has_videos:
        return 'video'
    elif has_images:
        return 'image'
    else:
        return 'empty'


def convert_toml_to_musubi_toml(last_data_config_path: str, last_config_path: str = None, output_path: str = None):
    """
    Convert last_data_config.toml to last_data_musubi_config.toml format.

    Args:
        last_data_config_path: Path to the input last_data_config.toml file
        last_config_path: Path to last_config.toml (for reading batch_size)
        output_path: Path to save the output TOML (default: same dir with _musubi suffix)

    Returns:
        Dict with keys: output_path, datasets_info (list of dataset info), resolutions, target_frames, dataset_type
    """
    if output_path is None:
        # Generate output path in same directory as input with _musubi suffix
        base_dir = os.path.dirname(last_data_config_path)
        output_path = os.path.join(base_dir, "last_data_musubi_config.toml")

    # Read the data config TOML
    with open(last_data_config_path, 'r') as f:
        data_config = toml.load(f)

    # Extract directory entries (can be multiple)
    # Each directory now contains all its own settings (per-dataset)
    directories = []
    if 'directory' in data_config:
        dirs = data_config['directory']
        if isinstance(dirs, list):
            for dir_entry in dirs:
                if isinstance(dir_entry, dict) and 'path' in dir_entry:
                    directories.append(dir_entry)
        elif isinstance(dirs, dict) and 'path' in dirs:
            directories.append(dirs)

    # Use first directory for compatibility with existing code
    video_directory = directories[0].get('path', '') if directories else ""

    # Detect dataset type (video or image) from first directory
    dataset_type = detect_dataset_type(video_directory)

    # For backwards compatibility: if settings are at global level, use them as defaults
    # Otherwise, each directory will have its own settings
    global_resolutions = data_config.get('resolutions', [])
    global_enable_ar_bucket = data_config.get('enable_ar_bucket', True)
    global_min_ar = data_config.get('min_ar', 0.5)
    global_max_ar = data_config.get('max_ar', 2.0)
    global_ar_buckets = data_config.get('ar_buckets', [])
    global_num_ar_buckets = data_config.get('num_ar_buckets', 7)
    global_frame_buckets = data_config.get('frame_buckets', [41])

    # Check for commented frame_buckets at global level (for backwards compatibility)
    if global_frame_buckets == [41]:
        try:
            with open(last_data_config_path, 'r') as f:
                raw_text = f.read()
            match = re.search(r'^[ 	]*#[ 	]*frame_buckets[ 	]*=[ 	]*(\[.*\])', raw_text, re.MULTILINE)
            if match:
                import json
                commented_list = match.group(1)
                parsed_list = json.loads(commented_list)
                if isinstance(parsed_list, list):
                    global_frame_buckets = parsed_list
        except Exception:
            pass

    # batch_size from last_config.toml [optimization] section
    batch_size = 4  # default
    # frame_extraction from last_config.toml [training_strategy] section (for musubi training)
    frame_extraction = 'head'  # default
    # use_mask from last_config.toml [training_strategy] section
    use_mask = False  # default
    # ltx_mode from last_config.toml [training_strategy] section (for audio-only mode)
    ltx_mode = 'video'  # default
    # target_fps from last_config.toml [training_strategy] section
    target_fps = 25.0  # default
    if last_config_path and os.path.exists(last_config_path):
        try:
            with open(last_config_path, 'r') as f:
                config = toml.load(f)
                batch_size = config.get('optimization', {}).get('batch_size', 4)
                frame_extraction = config.get('training_strategy', {}).get('frame_extraction', 'head')
                use_mask = config.get('training_strategy', {}).get('use_mask', False)
                ltx_mode = config.get('training_strategy', {}).get('ltx_mode', 'video')
                target_fps = float(config.get('training_strategy', {}).get('target_fps', 25))
                # Handle boolean conversion from string
                if not isinstance(use_mask, bool):
                    use_mask = str(use_mask).lower() in ['true', '1', 'yes', 'on']
        except Exception:
            pass

    # Check t_type dropdown value (replaces slider/ic_lora/vace_lora checkboxes)
    t_type = 'none'
    if last_config_path and os.path.exists(last_config_path):
        try:
            with open(last_config_path, 'r') as f:
                last_config = toml.load(f)
            training_strategy = last_config.get('training_strategy', {})
            # Get t_type directly or fall back to old checkbox format for backward compatibility
            if 't_type' in training_strategy:
                t_type = training_strategy.get('t_type', 'none')
            else:
                # Backward compatibility: check old checkbox keys
                if training_strategy.get('slider', False):
                    t_type = 'slider'
                elif training_strategy.get('ic_lora', False):
                    t_type = 'ic_lora'
                elif training_strategy.get('vace_lora', False):
                    t_type = 'vace_lora'
                else:
                    t_type = 'none'
        except Exception:
            pass
    # Derived boolean flags for backward compatibility with existing code
    slider_enabled = (t_type == 'slider')
    ic_lora_enabled = (t_type == 'ic_lora')
    vace_lora_enabled = (t_type == 'vace_lora')

    # Build the musubi config - create one dataset entry per directory
    datasets_list = []

    for dir_info in directories:
        dir_path = dir_info.get('path', '')
        dir_num_repeats = dir_info.get('num_repeats', 1)

        # Get per-dataset settings with fallback to global values (for backwards compatibility)
        dir_resolutions = dir_info.get('resolutions', global_resolutions)
        dir_enable_ar_bucket = dir_info.get('enable_ar_bucket', global_enable_ar_bucket)
        dir_min_ar = dir_info.get('min_ar', global_min_ar)
        dir_max_ar = dir_info.get('max_ar', global_max_ar)
        dir_ar_buckets = dir_info.get('ar_buckets', global_ar_buckets)
        dir_num_ar_buckets = dir_info.get('num_ar_buckets', global_num_ar_buckets)
        dir_frame_buckets = dir_info.get('frame_buckets', global_frame_buckets)
        # Get frame_extraction from per-dataset setting (only for LTX2 datasets)
        dir_frame_extraction = dir_info.get('frame_extraction', frame_extraction)
        # Get control_args for i2v preprocessing
        dir_control_args = dir_info.get('control_args', None)

        # Process resolutions (handle both flat list and list of lists)
        resolution_list = []
        is_multi_resolution = False

        if dir_resolutions and isinstance(dir_resolutions, list) and len(dir_resolutions) > 0:
            if isinstance(dir_resolutions[0], list):
                # Multiple resolutions: [[256, 256], [512, 512], [1024, 1024]]
                is_multi_resolution = True
                resolution_list = dir_resolutions
            else:
                # Single resolution as flat list: [512, 768] (convert to [[512, 768]])
                # If only one element like [256], normalize to [256, 256]
                is_multi_resolution = False
                if len(dir_resolutions) == 1:
                    resolution_list = [[dir_resolutions[0], dir_resolutions[0]]]
                else:
                    resolution_list = [dir_resolutions]
        else:
            # Default resolution
            resolution_list = [[512, 512]]

        # Detect dataset type for this specific directory
        dir_dataset_type = detect_dataset_type(dir_path)
        # Override to audio if ltx_mode is audio
        if ltx_mode == 'audio':
            dir_dataset_type = 'audio'

        # For multiple resolutions, create a dataset entry for each resolution
        for resolution in resolution_list:
            # cache_directory = path + /cache_musubi (or musubi_cache_positive for slider mode, or cache_ic_lora for ic_lora mode, or cache_vace for vace_lora mode)
            if slider_enabled:
                cache_directory = os.path.join(dir_path, "musubi_cache_positive") if dir_path else ""
            elif ic_lora_enabled:
                cache_directory = os.path.join(dir_path, "cache_ic_lora") if dir_path else ""
            elif vace_lora_enabled:
                cache_directory = os.path.join(dir_path, "cache_vace") if dir_path else ""
            else:
                cache_directory = os.path.join(dir_path, "cache_musubi") if dir_path else ""

            # For multiple resolutions, append resolution to cache directory to make them unique
            if is_multi_resolution:
                res_suffix = f"{resolution[0]}x{resolution[1]}"
                unique_cache_dir = f"{cache_directory}_{res_suffix}"
            else:
                unique_cache_dir = cache_directory

            # If AR bucketing is enabled, automatically enable bucketing
            # Note: Audio datasets don't support AR bucketing
            enable_bucket = dir_enable_ar_bucket if dir_dataset_type != 'audio' else False

            # Base config - no AR bucketing for audio datasets
            dataset_config = {
                'cache_directory': unique_cache_dir,
                'num_repeats': dir_num_repeats,
                'resolution': resolution,  # Each dataset has its own resolution
                'enable_bucket': enable_bucket,
                'bucket_no_upscale': False,
            }

            # Only add AR bucketing parameters for non-audio datasets
            if dir_dataset_type != 'audio':
                dataset_config.update({
                    'enable_ar_bucket': dir_enable_ar_bucket,
                    'min_ar': dir_min_ar,
                    'max_ar': dir_max_ar,
                    'num_ar_buckets': dir_num_ar_buckets,
                })

            # Set directory type based on detected content
            if dir_dataset_type == 'image':
                dataset_config['image_directory'] = dir_path
            elif dir_dataset_type == 'audio':
                dataset_config['audio_directory'] = dir_path
            else:
                dataset_config['video_directory'] = dir_path
                dataset_config['target_frames'] = dir_frame_buckets
                dataset_config['frame_extraction'] = dir_frame_extraction
                dataset_config['target_fps'] = target_fps
                # Automatically set max_frames to the largest value in target_frames
                if dir_frame_buckets:
                    dataset_config['max_frames'] = max(dir_frame_buckets)
                # Add control_args if present (for i2v preprocessing)
                if dir_control_args is not None:
                    dataset_config['control_args'] = dir_control_args
                # Add enable_mask if use_mask is true (only for video datasets)
                if use_mask:
                    dataset_config['enable_mask'] = True
                # Add reference_directory for IC-LoRA mode
                if ic_lora_enabled and dir_path:
                    # Check if control subdirectory exists
                    potential_control = os.path.join(dir_path, 'control')
                    if os.path.exists(potential_control) and os.path.isdir(potential_control):
                        dataset_config['reference_directory'] = potential_control
                        # Add per-dataset reference_cache_directory for proper multi-dataset support
                        dataset_config['reference_cache_directory'] = os.path.join(potential_control, 'cache_ref')
                # Add vace_directory for VACE-LoRA mode (uses control/ for consistency)
                if vace_lora_enabled and dir_path:
                    # Create control subdirectory for VACE control videos/masks
                    potential_control = os.path.join(dir_path, 'control')
                    if not os.path.exists(potential_control):
                        os.makedirs(potential_control, exist_ok=True)
                    dataset_config['vace_directory'] = potential_control

            datasets_list.append(dataset_config)

    # Build general config - exclude AR bucketing for audio-only mode
    general_config = {
        'caption_extension': '.txt',
        'batch_size': batch_size,
        'enable_bucket': global_enable_ar_bucket if ltx_mode != 'audio' else False,
        'bucket_no_upscale': False,
    }

    # Note: reference_cache_directory is now handled per-dataset for proper multi-dataset IC-LoRA support

    # Add vace_cache_directory for VACE-LoRA mode
    if vace_lora_enabled and datasets_list:
        first_ds = datasets_list[0]
        main_dir = first_ds.get('image_directory', first_ds.get('video_directory', ''))
        if main_dir:
            # VACE cache is in the control subdirectory (consistent with ic_lora)
            potential_control = os.path.join(main_dir, 'control')
            if not os.path.exists(potential_control):
                os.makedirs(potential_control, exist_ok=True)
            general_config['vace_cache_directory'] = os.path.join(potential_control, 'cache_vace')

    # Only add AR bucketing to general section if not audio-only mode
    if ltx_mode != 'audio':
        general_config.update({
            'enable_ar_bucket': global_enable_ar_bucket,
            'min_ar': global_min_ar,
            'max_ar': global_max_ar,
            'num_ar_buckets': global_num_ar_buckets,
        })

    musubi_config = {
        'general': general_config,
        'datasets': datasets_list
    }

    # Write to file with proper formatting
    _write_musubi_toml(output_path, musubi_config, dataset_type)

    # Check if slider mode is enabled and create slider config
    slider_config_path = None
    if datasets_list and last_config_path and os.path.exists(last_config_path):
        try:
            # Read t_type from last_config.toml (already computed above, but re-read for this scope)
            with open(last_config_path, 'r') as f:
                last_config = toml.load(f)

            training_strategy = last_config.get('training_strategy', {})
            # Use t_type dropdown value (replaces slider checkbox)
            if 't_type' in training_strategy:
                t_type_for_slider = training_strategy.get('t_type', 'none')
            else:
                # Backward compatibility
                if training_strategy.get('slider', False):
                    t_type_for_slider = 'slider'
                else:
                    t_type_for_slider = 'none'
            slider_enabled = (t_type_for_slider == 'slider')

            if slider_enabled:
                # Get sample_slider_range from config
                sample_slider_range_str = training_strategy.get('sample_slider_range', '-2.0, -1.0, 0.0, 1.0, 2.0')

                # Parse the slider range values
                slider_values = [float(x.strip()) for x in sample_slider_range_str.split(',') if x.strip()]

                # Create the slider config
                ws_dir = os.path.dirname(output_path)
                slider_config_path = os.path.join(ws_dir, 'last_data_musubi_slider_config.toml')

                import glob
                video_extensions = ['mp4', 'webm', 'mov', 'avi', 'mkv']
                has_source_videos = False

                # Check if we have source videos (for i2v mode detection)
                for ds in datasets_list:
                    pos_dir = ds.get('image_directory', ds.get('video_directory', ''))
                    if pos_dir and os.path.exists(pos_dir):
                        for ext in video_extensions:
                            if glob.glob(os.path.join(pos_dir, f"*.{ext}")) or \
                               glob.glob(os.path.join(pos_dir, f"*.{ext.upper()}")):
                                has_source_videos = True
                                break
                    if has_source_videos:
                        break

                # Build cache directory paths
                # For slider mode: use musubi_cache_positive and musubi_cache_negative
                # Determine the directory containing the datasets (parent of dataset dirs)
                pos_cache_dir = None
                neg_cache_dir = None
                text_cache_dir = None

                if datasets_list:
                    first_ds = datasets_list[0]
                    pos_dir = first_ds.get('image_directory', first_ds.get('video_directory', ''))
                    if pos_dir:
                        # Positive cache is musubi_cache_positive in the dataset directory
                        pos_cache_dir = os.path.join(pos_dir, 'musubi_cache_positive')
                        # Text cache is same as positive cache
                        text_cache_dir = pos_cache_dir

                        # Negative cache is musubi_cache_negative
                        # Check if there's a 'control' subdirectory within the dataset directory
                        potential_control = os.path.join(pos_dir, 'control')
                        if os.path.exists(potential_control) and os.path.isdir(potential_control):
                            # Control exists as a subdirectory - put negative cache there
                            neg_cache_dir = os.path.join(potential_control, 'musubi_cache_negative')
                        else:
                            # No control subdirectory - put negative cache alongside positive
                            neg_cache_dir = os.path.join(pos_dir, 'musubi_cache_negative')

                # Only create slider config if we have valid cache directories
                if pos_cache_dir and neg_cache_dir:
                    slider_lines = [
                        'mode = "reference"',
                        '',
                        '# Slider cache directories for reference mode training (list format)',
                        f'pos_cache_dirs = [ "{pos_cache_dir}", ]',
                        f'neg_cache_dirs = [ "{neg_cache_dir}", ]',
                        f'text_cache_dirs = [ "{text_cache_dir}", ]',
                        '',
                        f'batch_size = {batch_size}',
                        '',
                        f'sample_slider_range = [ {", ".join(str(v) for v in slider_values)},]',
                    ]

                    with open(slider_config_path, 'w') as f:
                        f.write('\n'.join(slider_lines) + '\n')

                    logger.info(f"Created slider config: {slider_config_path}")
                else:
                    logger.warning("Could not determine cache directories for slider config")
                    slider_config_path = None

        except Exception as e:
            logger.error(f"Error creating slider config: {e}")

    # For return value, use first directory's settings (for backwards compatibility)
    first_dir_resolution = [[512, 512]]
    first_dir_frame_buckets = global_frame_buckets
    first_cache_dir = ""
    if directories:
        first_dir = directories[0]
        first_dir_resolution = first_dir.get('resolutions', global_resolutions)
        first_dir_frame_buckets = first_dir.get('frame_buckets', global_frame_buckets)

    # Use first dataset's cache directory for backwards compatibility
    if datasets_list:
        first_cache_dir = datasets_list[0]['cache_directory']

    return {
        'output_path': output_path,
        'video_directory': video_directory,
        'cache_directory': first_cache_dir,
        'slider_config_path': slider_config_path,
        'resolutions': first_dir_resolution,
        'target_frames': first_dir_frame_buckets if dataset_type != 'image' else None,
        'dataset_type': dataset_type,
        'datasets_info': directories,  # Return list of directory info
    }


def _write_musubi_toml(output_path: str, config: dict, dataset_type: str = 'video'):
    """
    Write musubi TOML with proper formatting.

    Args:
        output_path: Path to write the TOML file
        config: Configuration dictionary
        dataset_type: 'video' or 'image' - determines which directory key to use
    """
    lines = []

    # [general] section
    lines.append("[general]")
    general = config['general']
    lines.append(f"caption_extension = \"{general['caption_extension']}\"")
    lines.append(f"batch_size = {general['batch_size']}")
    lines.append(f"enable_bucket = {str(general['enable_bucket']).lower()}")
    lines.append(f"bucket_no_upscale = {str(general['bucket_no_upscale']).lower()}")
    lines.append("")

    # [[datasets]] section - may have multiple datasets
    datasets = config['datasets']
    for i, dataset in enumerate(datasets):
        lines.append("[[datasets]]")

        # resolution is now per-dataset
        if 'resolution' in dataset:
            lines.append(f"resolution = {_format_list(dataset['resolution'])}")

        # Check if this is an image, audio, or video dataset based on which directory key exists
        if 'image_directory' in dataset:
            lines.append(f"image_directory = \"{dataset['image_directory']}\"")
        elif 'audio_directory' in dataset:
            lines.append(f"audio_directory = \"{dataset['audio_directory']}\"")
        elif 'video_directory' in dataset:
            lines.append(f"video_directory = \"{dataset['video_directory']}\"")
            if 'target_frames' in dataset:
                lines.append(f"target_frames = {_format_list(dataset['target_frames'])}")
            if 'frame_extraction' in dataset:
                lines.append(f"frame_extraction = \"{dataset['frame_extraction']}\"")
            if 'target_fps' in dataset:
                lines.append(f"target_fps = {float(dataset['target_fps'])}")
            if 'max_frames' in dataset:
                lines.append(f"max_frames = {dataset['max_frames']}")

        # Add reference_directory for IC-LoRA if present
        if 'reference_directory' in dataset:
            lines.append(f"reference_directory = \"{dataset['reference_directory']}\"")
        # Add per-dataset reference_cache_directory for IC-LoRA (multi-dataset support)
        if 'reference_cache_directory' in dataset:
            lines.append(f"reference_cache_directory = \"{dataset['reference_cache_directory']}\"")

        lines.append(f"cache_directory = \"{dataset['cache_directory']}\"")
        lines.append(f"num_repeats = {dataset['num_repeats']}")
        lines.append(f"enable_bucket = {str(dataset.get('enable_bucket', False)).lower()}")
        if dataset.get('enable_ar_bucket', False):
            lines.append(f"enable_ar_bucket = {str(dataset['enable_ar_bucket']).lower()}")
            lines.append(f"min_ar = {dataset.get('min_ar', 0.5)}")
            lines.append(f"max_ar = {dataset.get('max_ar', 2.0)}")
            lines.append(f"num_ar_buckets = {dataset.get('num_ar_buckets', 7)}")
        lines.append(f"bucket_no_upscale = {str(dataset.get('bucket_no_upscale', False)).lower()}")

        # Add blank line between datasets for readability
        if i < len(datasets) - 1:
            lines.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        f.write("\n")


def _format_list(lst):
    """Format a list as TOML array."""
    if not lst:
        return "[]"
    formatted_items = []
    for item in lst:
        if isinstance(item, str):
            # String elements need to be quoted in TOML
            formatted_items.append(f'"{item}"')
        else:
            formatted_items.append(str(item))
    return "[ " + ", ".join(formatted_items) + " ]"


def _format_list_of_lists(lst):
    """Format a list of lists as TOML array."""
    if not lst:
        return "[]"
    formatted_items = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            formatted_items.append(_format_list(item))
        else:
            formatted_items.append(str(item))
    return "[ " + ", ".join(formatted_items) + " ]"
