"""
Convert LTX (TOML) config to LTX-2 (YAML) format.
"""
import os
import yaml
import toml


def convert_toml_to_ltx2_yaml(toml_path: str, output_yaml_path: str = None):
    """
    Convert last_config.toml (LTX format) to last_config.yaml (LTX-2 format).

    Args:
        toml_path: Path to the input TOML file
        output_yaml_path: Path to save the output YAML (default: same dir, .yaml extension)

    Returns:
        Dict with keys: yaml_path, frame_buckets, resolutions, ar_buckets, preprocessed_data_root
    """
    if output_yaml_path is None:
        base_path = os.path.splitext(toml_path)[0]
        output_yaml_path = f"{base_path}.yaml"

    # Read TOML
    with open(toml_path, 'r') as f:
        toml_config = toml.load(f)

    # Build YAML config
    yaml_config = {}

    # =========================================================================
    # Model Configuration
    # =========================================================================
    model_section = toml_config.get('model', {})
    yaml_config['model'] = {
        'model_path': model_section.get('model_path', model_section.get('checkpoint_path', 'path/to/ltx-2-model.safetensors')),
        'text_encoder_path': model_section.get('text_encoder_path', 'path/to/gemma-text-encoder'),
        'training_mode': model_section.get('training_mode', 'lora'),
        'load_checkpoint': model_section.get('load_checkpoint', None),
    }

    # =========================================================================
    # LoRA Configuration
    # =========================================================================
    # For LTX2, LoRA config is in [lora] section
    lora_section = toml_config.get('lora', {})
    yaml_config['lora'] = {
        'rank': lora_section.get('rank', toml_config.get('model', {}).get('lora_rank', 32)),
        'alpha': lora_section.get('alpha', toml_config.get('model', {}).get('lora_alpha', 32)),
        'dropout': lora_section.get('dropout', toml_config.get('model', {}).get('lora_dropout', 0.0)),
        'target_modules': _build_target_modules(toml_config),
    }

    # =========================================================================
    # Training Strategy Configuration
    # =========================================================================
    # For LTX2, training_strategy is in [training_strategy] section
    training_strategy_section = toml_config.get('training_strategy', {})
    # Try to get with_audio from training_strategy, then [lora.target_modules], then [model], with default True
    with_audio = training_strategy_section.get('with_audio')
    if with_audio is None:
        # Fall back to [lora.target_modules] section
        lora_section = toml_config.get('lora', {})
        target_modules = lora_section.get('target_modules', {})
        with_audio = target_modules.get('with_audio')
    if with_audio is None:
        # Final fallback to [model] section or default
        with_audio = toml_config.get('model', {}).get('with_audio', True)
    # Ensure it's a boolean (handle string values from TOML)
    if isinstance(with_audio, str):
        with_audio = with_audio.strip().lower() in ('true', '1', 'yes', 'on')
    elif with_audio is not None:
        with_audio = bool(with_audio)
    yaml_config['training_strategy'] = {
        'name': 'text_to_video',
        'first_frame_conditioning_p': training_strategy_section.get('first_frame_conditioning_p', toml_config.get('model', {}).get('first_frame_conditioning_p', 0.5)),
        'with_audio': with_audio,
        'audio_latents_dir': 'audio_latents',
    }

    # =========================================================================
    # Optimization Configuration
    # =========================================================================
    # For LTX2, these are nested under [optimization] section
    optimization_section = toml_config.get('optimization', {})
    yaml_config['optimization'] = {
        'learning_rate': optimization_section.get('learning_rate', toml_config.get('learning_rate', 0.0001)),
        'steps': optimization_section.get('steps', toml_config.get('steps', 2000)),
        'batch_size': optimization_section.get('batch_size', toml_config.get('batch_size', 1)),
        'gradient_accumulation_steps': optimization_section.get('gradient_accumulation_steps', toml_config.get('gradient_accumulation_steps', 1)),
        'max_grad_norm': optimization_section.get('max_grad_norm', toml_config.get('max_grad_norm', 1.0)),
        'optimizer_type': optimization_section.get('optimizer_type', toml_config.get('optimizer_type', 'adamw8bit')),
        'scheduler_type': optimization_section.get('scheduler_type', toml_config.get('scheduler_type', 'linear')),
        'scheduler_params': optimization_section.get('scheduler_params', toml_config.get('scheduler_params', {})),
        'enable_gradient_checkpointing': optimization_section.get('enable_gradient_checkpointing', toml_config.get('gradient_checkpointing', True)),
    }

    # =========================================================================
    # Acceleration Configuration
    # =========================================================================
    # For LTX2, acceleration is in [acceleration] section
    acceleration_section = toml_config.get('acceleration', {})
    yaml_config['acceleration'] = {
        'mixed_precision_mode': acceleration_section.get('mixed_precision_mode', toml_config.get('mixed_precision', 'bf16')),
        'quantization': acceleration_section.get('quantization', toml_config.get('quantization', 'int8-quanto')),
        'load_text_encoder_in_8bit': acceleration_section.get('load_text_encoder_in_8bit', toml_config.get('load_text_encoder_in_8bit', True)),
    }

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # For LTX2, data config is in [data] section
    data_section = toml_config.get('data', {})
    dataset_path = data_section.get('preprocessed_data_root', toml_config.get('dataset', '/path/to/preprocessed/data'))
    preprocessed_root = _extract_preprocessed_root(dataset_path)

    # Append /.precomputed to the preprocessed data root
    if preprocessed_root and str(preprocessed_root).strip():
        preprocessed_root = str(preprocessed_root).rstrip('/') + '/.precomputed'

    yaml_config['data'] = {
        'preprocessed_data_root': preprocessed_root,
        'num_dataloader_workers': data_section.get('num_dataloader_workers', toml_config.get('num_workers', 2)),
    }

    # Store original preprocessed_root without .precomputed for later use
    preprocessed_root_original = _extract_preprocessed_root(dataset_path)

    # =========================================================================
    # Extract Dataset Variables (frame_buckets, resolutions, ar_buckets, num_repeats)
    # =========================================================================
    frame_buckets = [1, 33, 76]
    resolutions = []
    ar_buckets = []
    num_repeats = 1

    # Try to extract from dataset TOML if dataset_path points to a file
    if dataset_path and str(dataset_path).strip().endswith('.toml') and os.path.exists(dataset_path):
        try:
            data_config = toml.load(dataset_path)

            # Extract from top level first, then try [dataset] section
            if 'frame_buckets' in data_config:
                frame_buckets = data_config.get('frame_buckets', [1, 33, 76])
            if 'resolutions' in data_config:
                resolutions = data_config.get('resolutions', [])
            if 'ar_buckets' in data_config:
                ar_buckets = data_config.get('ar_buckets', [])
            if 'num_repeats' in data_config:
                num_repeats = data_config.get('num_repeats', 1)

            # If ar_buckets not found but we have num_ar_buckets, construct from min/max ar
            if not ar_buckets and 'num_ar_buckets' in data_config:
                num_ar = data_config.get('num_ar_buckets', 0)
                min_ar = data_config.get('min_ar', 0.5)
                max_ar = data_config.get('max_ar', 2.0)
                if num_ar > 0:
                    import numpy as np
                    # Create log-spaced ar_buckets
                    ar_buckets = np.logspace(np.log10(min_ar), np.log10(max_ar), num_ar).tolist()
        except Exception:
            pass

    # =========================================================================
    # Validation Configuration
    # =========================================================================
    validation_section = toml_config.get('validation', {})

    # Helper to convert string to bool
    def _to_bool(val, default=False):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ('true', '1', 'yes', 'on')
        return bool(val) if val is not None else default

    # Handle prompts - convert string to list for YAML format
    prompts_val = validation_section.get('prompts', 'Two woman with long brown hair')
    if isinstance(prompts_val, str):
        prompts_val = [prompts_val]

    # Handle video_dims - convert string to list if needed
    video_dims_val = validation_section.get('video_dims', '640, 416, 89')
    if isinstance(video_dims_val, str):
        # Parse comma-separated string to list
        video_dims_val = [int(x.strip()) for x in video_dims_val.split(',')]

    # Handle images - convert to list, convert "none" string to None
    images_val = validation_section.get('images', None)
    if isinstance(images_val, str) and images_val.lower() == 'none':
        images_val = None
    elif images_val is not None:
        # Ensure images is always a list
        if isinstance(images_val, str):
            images_val = [images_val]

    # Handle interval - convert "none" string to None
    interval_val = validation_section.get('interval', None)
    if isinstance(interval_val, str) and interval_val.lower() == 'none':
        interval_val = None

    yaml_config['validation'] = {
        'prompts': prompts_val,
        'negative_prompt': validation_section.get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted'),
        'images': images_val,
        'video_dims': video_dims_val,
        'frame_rate': validation_section.get('frame_rate', 25),
        'seed': validation_section.get('seed', 42),
        'inference_steps': validation_section.get('inference_steps', 30),
        'interval': interval_val,
        'videos_per_prompt': validation_section.get('videos_per_prompt', 1),
        'guidance_scale': validation_section.get('guidance_scale', 4.0),
        'stg_scale': 1.0,
        'stg_blocks': [29],
        'stg_mode': 'stg_av',
        'generate_audio': _to_bool(validation_section.get('generate_audio', False)),
        'skip_initial_validation': _to_bool(validation_section.get('skip_initial_validation', False)),
    }

    # =========================================================================
    # Checkpoint Configuration
    # =========================================================================
    # For LTX2, checkpoints config is in [checkpoints] section
    checkpoints_section = toml_config.get('checkpoints', {})
    yaml_config['checkpoints'] = {
        'interval': checkpoints_section.get('interval', toml_config.get('checkpoint_interval', 250)),
        'keep_last_n': checkpoints_section.get('keep_last_n', toml_config.get('keep_last_n', -1)),
        'precision': checkpoints_section.get('precision', 'bfloat16'),
    }

    # =========================================================================
    # Flow Matching Configuration
    # =========================================================================
    yaml_config['flow_matching'] = {
        'timestep_sampling_mode': 'shifted_logit_normal',
        'timestep_sampling_params': {},
    }

    # =========================================================================
    # Hugging Face Hub Configuration
    # =========================================================================
    yaml_config['hub'] = {
        'push_to_hub': toml_config.get('push_to_hub', False),
        'hub_model_id': toml_config.get('hub_model_id', None),
    }

    # =========================================================================
    # Weights & Biases Configuration
    # =========================================================================
    yaml_config['wandb'] = {
        'enabled': toml_config.get('wandb_enabled', False),
        'project': toml_config.get('wandb_project', 'ltx-2-trainer'),
        'entity': toml_config.get('wandb_entity', None),
        'tags': toml_config.get('wandb_tags', ['ltx2', 'lora']),
        'log_validation_videos': toml_config.get('log_validation_videos', True),
    }

    # =========================================================================
    # General Configuration
    # =========================================================================
    yaml_config['seed'] = toml_config.get('seed', 42)
    # For LTX2, output_dir is in [model] section; fall back to top level for other formats
    yaml_config['output_dir'] = toml_config.get('model', {}).get('output_dir', toml_config.get('output_dir', '/home/e/Dpipe/workspace/output/dir'))

    # Write YAML with custom formatting
    _write_yaml_formatted(output_yaml_path, yaml_config)

    print(f"Converted TOML to YAML: {output_yaml_path}")

    return {
        'yaml_path': output_yaml_path,
        'frame_buckets': frame_buckets,
        'resolutions': resolutions,
        'ar_buckets': ar_buckets,
        'preprocessed_data_root': preprocessed_root_original,
        'model_path': yaml_config['model']['model_path'],
        'text_encoder_path': yaml_config['model']['text_encoder_path'],
        'num_repeats': num_repeats,
    }


def _write_yaml_formatted(output_path: str, config: dict):
    """
    Write YAML with custom formatting:
    - Blank lines between major sections
    - No quotes around simple strings (scalars)
    - Quotes around strings in lists
    - Inline format for numeric lists
    - Vertical format for string lists
    """
    lines = []

    def _format_value_inline(value, in_list=False):
        """Format a simple value (scalar) for inline output."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Handle special case: string "null" should be treated as actual null
            if value.lower() == "null":
                return "null"
            # Always add quotes around strings
            return f'"{value}"'
        else:
            return str(value)

    def _is_numeric_list(lst):
        """Check if a list contains only numbers."""
        if not lst:
            return False
        return all(isinstance(item, (int, float)) for item in lst)

    def _write_dict_items(d, base_indent, is_root=False):
        """Recursively write dict items with proper indentation."""
        for key, val in d.items():
            if isinstance(val, dict):
                # If empty dict, write inline as {}
                if not val:
                    indent_str = ' ' * base_indent
                    lines.append(f"{indent_str}{key}: {{}}")
                else:
                    lines.append(f"{' ' * base_indent}{key}:")
                    _write_dict_items(val, base_indent + 2)
            elif isinstance(val, list):
                # Check if it's a numeric list - if so, use inline format
                if _is_numeric_list(val):
                    inline_list = "[ " + ", ".join(str(item) for item in val) + " ]"
                    lines.append(f"{' ' * base_indent}{key}: {inline_list}")
                else:
                    # String list - use vertical format with quotes
                    lines.append(f"{' ' * base_indent}{key}:")
                    for item in val:
                        if isinstance(item, (dict, list)):
                            # Complex nested structure
                            lines.append(f"{' ' * (base_indent + 2)}- {item}")
                        else:
                            # Simple item - add quotes if it's a string
                            lines.append(f"{' ' * (base_indent + 2)}- {_format_value_inline(item, in_list=True)}")
            else:
                # Scalar value
                lines.append(f"{' ' * base_indent}{key}: {_format_value_inline(val)}")

    # List of section names to maintain order and add spacing
    section_order = [
        'model',
        'lora',
        'training_strategy',
        'optimization',
        'acceleration',
        'data',
        'validation',
        'checkpoints',
        'flow_matching',
        'hub',
        'wandb',
        'seed',
        'output_dir',
    ]

    first_section = True
    for section_name in section_order:
        if section_name not in config:
            continue

        value = config[section_name]

        # Add blank line before section (except first)
        if not first_section:
            lines.append("")
        first_section = False

        # Write section
        if isinstance(value, dict):
            lines.append(f"{section_name}:")
            _write_dict_items(value, 2)
        else:
            lines.append(f"{section_name}: {_format_value_inline(value)}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        f.write("\n")


def _extract_preprocessed_root(dataset_path: str) -> str:
    """
    Extract the preprocessed data root path.

    If dataset_path is empty, return empty string.
    If dataset_path points to a TOML file, read it and extract the path from [[directory]] section.
    Otherwise, return the dataset_path as is.
    """
    if not dataset_path or not str(dataset_path).strip():
        return ""

    dataset_str = str(dataset_path).strip()

    # If it's a TOML file, read it and extract directory path
    if dataset_str.endswith('.toml') and os.path.exists(dataset_str):
        try:
            data_config = toml.load(dataset_str)
            # Look for [[directory]] section (stored as 'directory' key with list value)
            if 'directory' in data_config:
                dirs = data_config['directory']
                # If it's a list of dicts, take the first one
                if isinstance(dirs, list) and len(dirs) > 0:
                    first_dir = dirs[0]
                    if isinstance(first_dir, dict) and 'path' in first_dir:
                        return str(first_dir['path']).strip()
                # If it's a single dict
                elif isinstance(dirs, dict) and 'path' in dirs:
                    return str(dirs['path']).strip()
        except Exception:
            pass

    return dataset_str


def _build_target_modules(toml_config: dict) -> list:
    """
    Build target_modules list based on lora_targets boolean flags.

    When all_modules=true: includes basic attention modules (to_k, to_q, to_v, to_out.0)
    When all_modules=false: builds from specific flags:
    - video_attn: attn1/attn2 video attention modules
    - video_ff: ff.net video feed-forward modules
    - audio_attn: audio_attn1/attn2 audio attention modules
    - audio_ff: audio_ff.net audio feed-forward modules
    - cross_modal_attn: audio_to_video and video_to_audio cross-modal modules

    Returns a list of target module patterns.
    """
    target_modules = []

    # Get the lora_targets section - try [lora.target_modules] first (LTX2 format), then [adapter.lora_targets]
    lora_targets = toml_config.get('lora', {}).get('target_modules', None)
    if lora_targets is None:
        # Fall back to old format
        adapter_config = toml_config.get('adapter', {})
        lora_targets = adapter_config.get('lora_targets', {})

    if not lora_targets:
        lora_targets = {}

    # Helper to convert string to bool
    def _to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(val)

    # Check all_modules first
    all_modules = _to_bool(lora_targets.get('all_modules', False))

    if all_modules:
        # Include only basic attention modules
        target_modules = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ]
    else:
        # Build from individual flags
        # Video attention
        if _to_bool(lora_targets.get('video_attn', False)):
            target_modules.extend([m.strip() for m in "attn1.to_k, attn1.to_q, attn1.to_v, attn1.to_out.0".split(",")])
            target_modules.extend([m.strip() for m in "attn2.to_k, attn2.to_q, attn2.to_v, attn2.to_out.0".split(",")])

        # Video feed-forward
        if _to_bool(lora_targets.get('video_ff', False)):
            target_modules.extend([m.strip() for m in "ff.net.0.proj, ff.net.2".split(",")])

        # Audio attention
        if _to_bool(lora_targets.get('audio_attn', False)):
            target_modules.extend([m.strip() for m in "audio_attn1.to_k, audio_attn1.to_q, audio_attn1.to_v, audio_attn1.to_out.0".split(",")])
            target_modules.extend([m.strip() for m in "audio_attn2.to_k, audio_attn2.to_q, audio_attn2.to_v, audio_attn2.to_out.0".split(",")])

        # Audio feed-forward
        if _to_bool(lora_targets.get('audio_ff', False)):
            target_modules.extend([m.strip() for m in "audio_ff.net.0.proj, audio_ff.net.2".split(",")])

        # Cross-modal attention
        if _to_bool(lora_targets.get('cross_modal_attn', False)):
            target_modules.extend([m.strip() for m in "audio_to_video_attn.to_k, audio_to_video_attn.to_q, audio_to_video_attn.to_v, audio_to_video_attn.to_out.0".split(",")])
            target_modules.extend([m.strip() for m in "video_to_audio_attn.to_k, video_to_audio_attn.to_q, video_to_audio_attn.to_v, video_to_audio_attn.to_out.0".split(",")])

    return target_modules
