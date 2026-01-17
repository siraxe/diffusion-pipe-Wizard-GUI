"""
LTX2-specific configuration handling for save/load operations.
Handles the special TOML structure for LTX2 training.
"""

import os
from loguru import logger


def build_ltx2_toml_from_ui(training_tab_container) -> str:
    """Build TOML config for LTX2 from UI controls."""
    from .config_utils import extract_config_from_controls, expand_model_path
    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
    import os

    cfg = extract_config_from_controls(training_tab_container.config_page_content)

    # Helper functions
    def _get(name, default=None):
        return cfg.get(name, default)

    def _as_bool(val):
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ['1', 'true', 'yes', 'on']

    def _quote(s):
        if s is None:
            return "''"
        return f"'{str(s)}'"

    def _clean_value(val, is_numeric=False):
        """Clean value by removing type suffixes. If numeric fails, quote as string."""
        s = str(val).strip()
        # Remove type suffixes (f, d, etc) only from the end
        if s and s[-1] in 'fd':
            s = s[:-1].strip()

        if is_numeric:
            try:
                # Try to parse as float
                f_val = float(s)
                # Return as int if it's a whole number
                if f_val == int(f_val):
                    return int(f_val)
                return f_val
            except (ValueError, TypeError):
                # If it's not numeric, quote it as a string
                return _quote(s)
        return s

    # Get dataset path
    dataset_block = getattr(training_tab_container, 'dataset_page_content', None)
    dataset_path_val = ""
    if dataset_block and hasattr(dataset_block, 'get_selected_dataset'):
        selected_clean = dataset_block.get_selected_dataset()
        if selected_clean:
            base_dir, _dtype = _get_dataset_base_dir(selected_clean)
            ds_full = os.path.join(base_dir, selected_clean)
            dataset_path_val = ds_full.replace('\\', '/')

    # Get output_dir
    try:
        bb_field = getattr(training_tab_container, 'output_dir_field', None)
        if bb_field is not None and hasattr(bb_field, 'value') and getattr(bb_field, 'value', None):
            raw_output_dir = str(bb_field.value or '').strip()
        else:
            raw_output_dir = str(_get('output_dir', 'outputs/ltx2_lora') or '').strip()
    except Exception:
        raw_output_dir = str(_get('output_dir', 'outputs/ltx2_lora') or '').strip()

    raw_output_dir = raw_output_dir.replace('\\', '/')
    resolved_output_dir = os.path.normpath(os.path.join(os.getcwd(), raw_output_dir)).replace('\\', '/')

    # Build LTX2 TOML structure
    lines = []

    # [model]
    lines.append("[model]")
    lines.append("type = 'ltx-video-2'")

    model_path = _get('model_path', '')
    if model_path:
        model_path = expand_model_path(model_path)
    lines.append(f"model_path = {_quote(model_path)}")

    text_encoder_path = _get('text_encoder_path', '')
    if text_encoder_path:
        text_encoder_path = expand_model_path(text_encoder_path)
    lines.append(f"text_encoder_path = {_quote(text_encoder_path)}")

    # training_mode (default to 'lora' for LTX2)
    training_mode = _get('training_mode', 'lora')
    lines.append(f"training_mode = {_quote(training_mode)}")

    # load_checkpoint (optional - omit if empty)
    load_checkpoint = _get('load_checkpoint', '')
    if load_checkpoint and str(load_checkpoint).strip():
        lines.append(f"load_checkpoint = {_quote(load_checkpoint)}")
    lines.append("")

    # output_dir
    lines.append(f"output_dir = {_quote(resolved_output_dir)}")
    lines.append("")

    # [lora]
    lines.append("[lora]")
    rank_val = _clean_value(_get('rank', 32), is_numeric=True)
    lines.append(f"rank = {rank_val}")
    alpha_val = _clean_value(_get('alpha', 32), is_numeric=True)
    lines.append(f"alpha = {alpha_val}")
    dropout_val = _clean_value(_get('dropout', 0.0), is_numeric=True)
    lines.append(f"dropout = {dropout_val}")
    lines.append("")

    # [lora.target_modules]
    lines.append("[lora.target_modules]")
    lines.append(f"all_modules = {'true' if _as_bool(_get('all_modules', False)) else 'false'}")
    lines.append(f"video_attn = {'true' if _as_bool(_get('video_attn', True)) else 'false'}")
    lines.append(f"video_ff = {'true' if _as_bool(_get('video_ff', False)) else 'false'}")
    lines.append(f"audio_attn = {'true' if _as_bool(_get('audio_attn', False)) else 'false'}")
    lines.append(f"audio_ff = {'true' if _as_bool(_get('audio_ff', False)) else 'false'}")
    lines.append(f"cross_modal_attn = {'true' if _as_bool(_get('cross_modal_attn', False)) else 'false'}")
    lines.append("")

    # [training_strategy]
    lines.append("[training_strategy]")
    lines.append('name = "text_to_video"')
    ffc_val = _clean_value(_get('first_frame_conditioning_p', 1.0), is_numeric=True)
    lines.append(f"first_frame_conditioning_p = {ffc_val}")

    # with_audio - use with_audio value from UI
    with_audio = _as_bool(_get('with_audio', True))
    lines.append(f"with_audio = {'true' if with_audio else 'false'}")

    lines.append('audio_latents_dir = "audio_latents"')
    lines.append("")

    # [optimization]
    lines.append("[optimization]")
    lines.append(f"learning_rate = {_clean_value(_get('learning_rate', 0.0001), is_numeric=True)}")
    lines.append(f"steps = {_clean_value(_get('steps', 2000), is_numeric=True)}")
    lines.append(f"batch_size = {_clean_value(_get('batch_size', 1), is_numeric=True)}")
    lines.append(f"gradient_accumulation_steps = {_clean_value(_get('gradient_accumulation_steps', 1), is_numeric=True)}")
    lines.append(f"max_grad_norm = {_clean_value(_get('max_grad_norm', 1.0), is_numeric=True)}")
    lines.append(f"optimizer_type = {_quote(_get('optimizer_type', 'adamw8bit'))}")
    lines.append(f"scheduler_type = {_quote(_get('scheduler_type', 'linear'))}")
    lines.append("")
    lines.append("scheduler_params = { }")
    lines.append("enable_gradient_checkpointing = true")
    lines.append("")

    # [acceleration]
    lines.append("[acceleration]")
    lines.append(f"mixed_precision_mode = {_quote(_get('mixed_precision_mode', 'bf16'))}")
    lines.append(f"quantization = {_quote(_get('quantization', 'int8-quanto'))}")
    load_text_encoder_in_8bit = _as_bool(_get('load_text_encoder_in_8bit', False))
    lines.append(f"load_text_encoder_in_8bit = {'true' if load_text_encoder_in_8bit else 'false'}")
    lines.append("")

    # [data]
    lines.append("[data]")
    if dataset_path_val:
        # Convert to dataset.toml path
        try:
            base_dir, _dtype = _get_dataset_base_dir(os.path.basename(dataset_path_val))
        except Exception:
            base_dir = os.path.dirname(dataset_path_val)
        ds_name = os.path.basename(dataset_path_val.rstrip('/'))
        dataset_toml_abs = os.path.join(base_dir, f"{ds_name}.toml")
        dataset_path_to_write = dataset_toml_abs.replace('\\', '/')
        lines.append(f"preprocessed_data_root = {_quote(dataset_path_to_write)}")
    else:
        lines.append("preprocessed_data_root = \"\"")

    lines.append("num_dataloader_workers = 2")
    lines.append("")

    # [flow_matching]
    lines.append("[flow_matching]")
    timestep_mode = _get('timestep_sm', 'shifted_logit_normal')
    # Convert UI values to canonical names
    if str(timestep_mode).lower() == 'logit_normal':
        timestep_mode = 'shifted_logit_normal'
    lines.append(f"timestep_sampling_mode = {_quote(timestep_mode)}")
    lines.append("timestep_sampling_params = { }")
    lines.append("")

    # [checkpoints]
    lines.append("[checkpoints]")
    lines.append(f"interval = {_clean_value(_get('interval', 50), is_numeric=True)}")
    lines.append(f"keep_last_n = {_clean_value(_get('keep_last_n', -1), is_numeric=True)}")
    lines.append(f"precision = {_quote(_get('precision', 'bfloat16'))}")
    lines.append("")

    # [validation]
    lines.append("[validation]")
    lines.append(f"interval = {_clean_value(_get('validation_interval', 'none'), is_numeric=True)}")
    lines.append(f"skip_initial_validation = {_quote(_get('skip_initial_validation', 'false'))}")
    lines.append(f"generate_audio = {_quote(_get('generate_audio', 'false'))}")
    lines.append(f"prompts = {_quote(_get('prompts', 'Two woman with long brown hair'))}")
    lines.append(f"negative_prompt = {_quote(_get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted'))}")
    lines.append(f"images = {_quote(_get('images', 'none'))}")
    lines.append(f"video_dims = {_quote(_get('video_dims', '640, 416, 89'))}")
    lines.append(f"videos_per_prompt = {_clean_value(_get('videos_per_prompt', 1), is_numeric=True)}")
    lines.append(f"guidance_scale = {_clean_value(_get('guidance_scale', 4.0), is_numeric=True)}")
    lines.append(f"frame_rate = {_clean_value(_get('frame_rate', 25), is_numeric=True)}")
    lines.append(f"seed = {_clean_value(_get('seed', 42), is_numeric=True)}")
    lines.append(f"inference_steps = {_clean_value(_get('inference_steps', 30), is_numeric=True)}")
    lines.append("")

    return "\n".join(lines) + "\n"


def update_ltx2_ui_from_toml(training_tab_container, toml_data: dict) -> None:
    """Update UI controls from LTX2 TOML data."""
    from flet_app.ui.utils.config_utils import collapse_model_path
    import flet as ft

    def _set_field_value(label, value):
        """Helper to find and set a control's value by label."""
        try:
            found = False
            def _apply(control):
                nonlocal found
                if hasattr(control, 'controls') and control.controls:
                    for c in control.controls:
                        _apply(c)
                if hasattr(control, 'content') and control.content:
                    _apply(control.content)

                ctrl_label = getattr(control, 'label', None)
                if ctrl_label == label:
                    found = True
                    if isinstance(control, ft.TextField):
                        # Handle empty/None values - show as "null" string for specific fields
                        if value is None or value == '':
                            control.value = "null"
                        else:
                            control.value = str(value)
                    elif isinstance(control, ft.Dropdown):
                        control.value = str(value) if value is not None else ""
                    elif isinstance(control, ft.Checkbox):
                        if isinstance(value, bool):
                            control.value = value
                        else:
                            control.value = str(value).lower() in ['true', '1', 'yes', 'on']
                    if hasattr(control, 'page') and control.page:
                        control.update()

            config_content = getattr(training_tab_container, 'config_page_content', None)
            if config_content:
                _apply(config_content)
        except Exception as e:
            logger.warning(f"Error setting field {label} to {value}: {e}")

    try:
        # Model section - Set Model Type first (triggers visibility changes for LTX2)
        model = toml_data.get('model', {})
        if model:
            model_type = model.get('type', 'ltx-video-2')
            _set_field_value('Model Type', model_type)

            # Try to trigger the on_change event by finding and calling it
            try:
                def _trigger_model_change(control):
                    ctrl_label = getattr(control, 'label', None)
                    if ctrl_label == 'Model Type' and isinstance(control, ft.Dropdown):
                        if hasattr(control, 'on_change') and control.on_change:
                            control.on_change(ft.ControlEvent('change'))
                            return True
                    if hasattr(control, 'controls') and control.controls:
                        for c in control.controls:
                            if _trigger_model_change(c):
                                return True
                    if hasattr(control, 'content') and control.content:
                        return _trigger_model_change(control.content)
                    return False

                config_content = getattr(training_tab_container, 'config_page_content', None)
                if config_content:
                    _trigger_model_change(config_content)
                    # Force page update to ensure visibility changes take effect
                    page = getattr(training_tab_container, 'page', None)
                    if page:
                        page.update()
            except Exception:
                pass

            model_path = model.get('model_path', '')
            if model_path:
                model_path = collapse_model_path(model_path)
            _set_field_value('model_path', model_path)

            text_encoder_path = model.get('text_encoder_path', '')
            if text_encoder_path:
                text_encoder_path = collapse_model_path(text_encoder_path)
            _set_field_value('text_encoder_path', text_encoder_path)

            _set_field_value('load_checkpoint', model.get('load_checkpoint', ''))

        # Output dir
        output_dir = toml_data.get('output_dir', '')
        _set_field_value('output_dir', output_dir)

        # LoRA section
        lora = toml_data.get('lora', {})
        # Always load values (section may be empty dict)
        _set_field_value('rank', lora.get('rank', 32))
        _set_field_value('alpha', lora.get('alpha', 32))
        _set_field_value('dropout', lora.get('dropout', 0.0))

        # LoRA target modules section
        lora_target_modules = toml_data.get('lora', {}).get('target_modules', {})
        if not lora_target_modules:
            # Try as separate section [lora.target_modules]
            lora_target_modules = toml_data.get('lora.target_modules', {})
        # Always load values (section may be empty dict)
        _set_field_value('all_modules', lora_target_modules.get('all_modules', False))
        _set_field_value('video_attn', lora_target_modules.get('video_attn', True))
        _set_field_value('video_ff', lora_target_modules.get('video_ff', False))
        _set_field_value('audio_attn', lora_target_modules.get('audio_attn', False))
        _set_field_value('audio_ff', lora_target_modules.get('audio_ff', False))
        _set_field_value('cross_modal_attn', lora_target_modules.get('cross_modal_attn', False))
        _set_field_value('with_audio', lora_target_modules.get('with_audio', True))

        # Training strategy
        training_strategy = toml_data.get('training_strategy', {})
        # Always load values (section may be empty dict)
        _set_field_value('first_frame_conditioning_p', training_strategy.get('first_frame_conditioning_p', 1.0))
        # Load with_audio directly from training_strategy
        with_audio = training_strategy.get('with_audio', True)
        if not isinstance(with_audio, bool):
            with_audio = str(with_audio).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('with_audio', with_audio)

        # Optimization section
        optimization = toml_data.get('optimization', {})
        # Always load values (section may be empty dict)
        _set_field_value('learning_rate', optimization.get('learning_rate', 0.0001))
        _set_field_value('steps', optimization.get('steps', 2000))
        _set_field_value('batch_size', optimization.get('batch_size', 1))
        _set_field_value('gradient_accumulation_steps', optimization.get('gradient_accumulation_steps', 1))
        _set_field_value('max_grad_norm', optimization.get('max_grad_norm', 1.0))
        _set_field_value('optimizer_type', optimization.get('optimizer_type', 'adamw8bit'))
        _set_field_value('scheduler_type', optimization.get('scheduler_type', 'linear'))

        # Acceleration section
        acceleration = toml_data.get('acceleration', {})
        # Always load values (section may be empty dict)
        _set_field_value('mixed_precision_mode', acceleration.get('mixed_precision_mode', 'bf16'))
        _set_field_value('quantization', acceleration.get('quantization', 'int8-quanto'))
        _set_field_value('load_text_encoder_in_8bit', acceleration.get('load_text_encoder_in_8bit', False))

        # Checkpoints section
        checkpoints = toml_data.get('checkpoints', {})
        # Always load values (section may be empty dict)
        _set_field_value('interval', checkpoints.get('interval', 50))
        _set_field_value('keep_last_n', checkpoints.get('keep_last_n', -1))
        _set_field_value('precision', checkpoints.get('precision', 'bfloat16'))

        # Validation section - always load values (section may be empty dict)
        validation = toml_data.get('validation', {})
        # Always load validation values, even if section is empty (use defaults)
        # Don't use 'if validation:' as empty dict {} is falsy
        _set_field_value('validation_interval', validation.get('interval', 'none'))
        _set_field_value('skip_initial_validation', validation.get('skip_initial_validation', 'false'))
        _set_field_value('generate_audio', validation.get('generate_audio', 'false'))
        _set_field_value('prompts', validation.get('prompts', 'Two woman with long brown hair'))
        _set_field_value('negative_prompt', validation.get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted'))
        _set_field_value('images', validation.get('images', 'none'))
        _set_field_value('video_dims', validation.get('video_dims', '640, 416, 89'))
        _set_field_value('videos_per_prompt', validation.get('videos_per_prompt', 1))
        _set_field_value('guidance_scale', validation.get('guidance_scale', 4.0))
        _set_field_value('frame_rate', validation.get('frame_rate', 25))
        _set_field_value('seed', validation.get('seed', 42))
        _set_field_value('inference_steps', validation.get('inference_steps', 30))

        # Flow matching (timestep_sm)
        flow_matching = toml_data.get('flow_matching', {})
        if flow_matching:
            timestep_mode = flow_matching.get('timestep_sampling_mode', 'shifted_logit_normal')
            # Convert canonical name back to UI name
            if str(timestep_mode).lower() == 'shifted_logit_normal':
                timestep_mode = 'logit_normal'
            _set_field_value('timestep_sm', timestep_mode)

        # Dataset selection from 'preprocessed_data_root' key in [data] section
        try:
            data_section = toml_data.get('data', {})
            dataset_path = data_section.get('preprocessed_data_root')
            if dataset_path:
                clean_name = os.path.basename(str(dataset_path).replace('\\', '/'))
                # Strip .toml if present
                if clean_name.lower().endswith('.toml'):
                    clean_name = clean_name[:-5]
                # Get page from training_tab_container if available
                page = getattr(training_tab_container, 'page', None)
                ds_block = getattr(training_tab_container, 'dataset_page_content', None)
                if ds_block and hasattr(ds_block, 'set_selected_dataset'):
                    ds_block.set_selected_dataset(clean_name, page_ctx=page)
                # Also set on Data Config panel and refresh its indicator
                data_cfg = getattr(training_tab_container, 'data_config_page_content', None)
                if data_cfg and hasattr(data_cfg, 'dataset_block') and hasattr(data_cfg.dataset_block, 'set_selected_dataset'):
                    data_cfg.dataset_block.set_selected_dataset(clean_name, page_ctx=page)
                    if hasattr(data_cfg, 'refresh_indicator') and callable(getattr(data_cfg, 'refresh_indicator')):
                        try:
                            data_cfg.refresh_indicator(clean_name)
                        except Exception:
                            pass
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Error updating LTX2 UI from TOML: {e}")
