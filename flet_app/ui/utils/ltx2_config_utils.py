"""
LTX2-specific configuration handling for save/load operations.
Handles the special TOML structure for LTX2 training.
"""

import os
from loguru import logger


def build_ltx2_toml_from_ui(training_tab_container, config_name: str = None) -> str:
    """
    Build TOML config for LTX2 from UI controls.

    Args:
        training_tab_container: The training UI container
        config_name: Optional name for the config (will be added as 'name' field)
    """
    from .config_utils import extract_config_from_controls, expand_model_path
    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
    import os

    # Import for optimizer type conversion
    try:
        from .config.musubi.optimizer import get_musubi_optimizer_type_for_toml
    except ImportError:
        def get_musubi_optimizer_type_for_toml(ui_value: str) -> str:
            return str(ui_value).strip().lower()

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
        """Quote a string value for TOML - uses regular single quotes."""
        if s is None:
            return "''"
        # Use regular single quotes for simple strings
        s_str = str(s)
        # Only escape single quotes (backslashes are fine in single-quoted TOML strings)
        s_str = s_str.replace("'", "''")
        return f"'{s_str}'"

    def _quote_multiline(s):
        """Quote a string for TOML with triple quotes - for prompts and other long text."""
        if s is None:
            return '""""""'
        # Use triple double quotes for multi-line text support
        s_str = str(s).replace('"""', '\\"\\"\\"')
        return f'"""{s_str}"""'

    def _format_numeric(val):
        """Format a numeric value for TOML, avoiding scientific notation."""
        if isinstance(val, int):
            return str(val)
        # For floats, use decimal format instead of scientific notation
        # Convert to string with enough precision, then strip trailing zeros
        formatted = f"{val:.10f}".rstrip('0').rstrip('.')
        return formatted

    def _clean_value(val, is_numeric=False):
        """Clean value by removing type suffixes. If numeric fails, quote as string. Returns formatted string for numeric values."""
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
                    return str(int(f_val))
                # For floats, format without scientific notation
                return _format_numeric(f_val)
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

    # Write trainer field
    trainer = _get('Trainer', 'musubi')  # LTX2 uses musubi by default
    if trainer:
        lines.append(f"trainer = {_quote(trainer)}")

    # Add name field if config_name is provided (filename without .toml extension)
    if config_name:
        # Strip .toml extension if present
        clean_name = config_name.lower()
        if clean_name.endswith('.toml'):
            config_name = config_name[:-5]
        lines.append(f"name = {_quote(config_name)}")

    model_path = _get('model_path', '')
    if model_path:
        model_path = expand_model_path(model_path)
    lines.append(f"model_path = {_quote(model_path)}")

    text_encoder_path = _get('text_encoder_path', '')
    if text_encoder_path:
        text_encoder_path = expand_model_path(text_encoder_path)
    lines.append(f"text_encoder_path = {_quote(text_encoder_path)}")

    # training_mode from adapter dropdown (lora/lokr)
    training_mode = _get('adapter', 'lora')
    lines.append(f"training_mode = {_quote(training_mode)}")
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
    factor_val = _clean_value(_get('factor', 4), is_numeric=True)
    lines.append(f"factor = {factor_val}")
    network_dropout_val = _clean_value(_get('network_dropout', 0.0), is_numeric=True)
    if network_dropout_val and float(network_dropout_val) > 0:
        lines.append(f"network_dropout = {network_dropout_val}")
    caption_dropout_rate_val = _clean_value(_get('caption_dropout_rate', 0.0), is_numeric=True)
    if caption_dropout_rate_val and float(caption_dropout_rate_val) > 0:
        lines.append(f"caption_dropout_rate = {caption_dropout_rate_val}")

    # init_from_existing (optional - omit if empty)
    # First try to get from bottom bar field (if available), then fallback to config page
    init_from_existing = ''
    try:
        init_field = getattr(training_tab_container, 'init_from_existing_field', None)
        if init_field and hasattr(init_field, 'value'):
            init_from_existing = str(init_field.value or '').strip()
    except Exception:
        pass
    # Fallback to config page if bottom bar is empty
    if not init_from_existing:
        init_from_existing = _get('init_from_existing', '')
    if init_from_existing and str(init_from_existing).strip():
        init_from_existing = expand_model_path(init_from_existing)
        lines.append(f"init_from_existing = {_quote(init_from_existing)}")
    lines.append("")

    # [training_strategy]
    lines.append("[training_strategy]")
    ltx_mode_val = _get('ltx_mode', 'video')
    lines.append(f"ltx_mode = {_quote(ltx_mode_val)}")

    target_fps_val = _get('target_fps', '25')
    lines.append(f"target_fps = {_quote(target_fps_val)}")

    frame_extraction_val = _get('frame_extraction', 'head')
    lines.append(f"frame_extraction = {_quote(frame_extraction_val)}")
    separate_audio_buckets_val = _as_bool(_get('separate_audio_buckets', True))
    lines.append(f"separate_audio_buckets = {'true' if separate_audio_buckets_val else 'false'}")
    slider_val = _as_bool(_get('slider', False))
    lines.append(f"slider = {'true' if slider_val else 'false'}")
    ic_lora_val = _as_bool(_get('ic_lora', False))
    lines.append(f"ic_lora = {'true' if ic_lora_val else 'false'}")
    vace_lora_val = _as_bool(_get('vace_lora', False))
    lines.append(f"vace_lora = {'true' if vace_lora_val else 'false'}")
    # ref_downscale for IC-LoRA reference caching
    ref_downscale_val = _clean_value(_get('ref_downscale', 1), is_numeric=True)
    lines.append(f"ref_downscale = {ref_downscale_val}")
    ltx_2_3_val = _as_bool(_get('ltx_2_3', False))
    lines.append(f"ltx_2_3 = {'true' if ltx_2_3_val else 'false'}")
    use_mask_val = _as_bool(_get('use_mask', False))
    lines.append(f"use_mask = {'true' if use_mask_val else 'false'}")
    # use_stiefel is derived from optimizer_type in ltx2_run.py, no need to save here
    sample_slider_range_val = _get('sample_slider_range', '-2.0, -1.0, 0.0, 1.0, 2.0')
    lines.append(f"sample_slider_range = {_quote(sample_slider_range_val)}")
    # Build control_args from i2v_type and sample_each
    i2v_type_val = _get('i2v_type', 'jump')
    sample_each_val = _get('sample_each', '3')
    if i2v_type_val == 'reverse' or i2v_type_val == 'freeze':
        # For reverse and freeze, only need single element (num not needed)
        control_args_formatted = '["' + str(i2v_type_val) + '"]'
    else:
        # For jump and fade, need both type and num
        control_args_formatted = '["' + str(i2v_type_val) + '", "' + str(sample_each_val) + '"]'
    lines.append(f"control_args = {control_args_formatted}")
    ffc_val = _clean_value(_get('first_frame_conditioning_p', 0.1), is_numeric=True)
    lines.append(f"first_frame_conditioning_p = {ffc_val}")
    lines.append("")

    # [optimization]
    lines.append("[optimization]")
    lines.append(f"learning_rate = {_clean_value(_get('learning_rate', 0.0001), is_numeric=True)}")
    lines.append(f"audio_lr = {_clean_value(_get('audio_lr', 0.0000), is_numeric=True)}")
    lines.append(f"max_steps = {_clean_value(_get('max_steps', 2000), is_numeric=True)}")
    lines.append(f"batch_size = {_clean_value(_get('batch_size', 1), is_numeric=True)}")
    lines.append(f"gradient_accumulation_steps = {_clean_value(_get('grad_accum_steps', 1), is_numeric=True)}")
    lines.append(f"max_grad_norm = {_clean_value(_get('max_grad_norm', 1.0), is_numeric=True)}")
    lines.append(f"loraplus_ratio = {_clean_value(_get('loraplus_ratio', 0.0), is_numeric=True)}")
    lines.append(f"blocks_to_swap = {_clean_value(_get('blocks_to_swap', 0), is_numeric=True)}")
    caption_dropout_val = _clean_value(_get('caption_dropout', 0.0), is_numeric=True)
    lines.append(f"caption_dropout_rate = {caption_dropout_val}")
    # Convert UI optimizer type to TOML format
    opt_type_ui = _get('optimizer_type_m', 'AdamW')
    opt_type_toml = get_musubi_optimizer_type_for_toml(opt_type_ui)
    lines.append(f"optimizer_type = {_quote(opt_type_toml)}")
    lines.append(f"scheduler_type = {_quote(_get('scheduler_type', 'constant'))}")

    # Add optimizer_args if Automagic is selected
    if str(opt_type_toml).strip().lower() == 'automagic':
        optimizer_args = _get('optimizer_args', 'min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6, eps=(1e-30; 1e-3), clip_threshold=1.0, beta2=0.999, weight_decay=0.0, do_paramiter_swapping=False, paramiter_swapping_factor=0.1')
        lines.append(f"optimizer_args = {_quote(optimizer_args)}")

    # Add lr_warmup_steps if constant_with_warmup is selected (flattened, not nested)
    scheduler_type = _get('scheduler_type', 'constant')
    if str(scheduler_type).strip().lower() == 'constant_with_warmup':
        lr_warmup_steps = _get('lr_warmup_steps', 50)
        lines.append(f"lr_warmup_steps = {_clean_value(lr_warmup_steps, is_numeric=True)}")

    gradient_checkpointing_val = _as_bool(_get('gradient_checkpointing', True))
    lines.append(f"enable_gradient_checkpointing = {'true' if gradient_checkpointing_val else 'false'}")
    lines.append("")

    # [acceleration]
    lines.append("[acceleration]")
    lines.append(f"mixed_precision_mode = {_quote(_get('mixed_precision_mode', 'bf16'))}")
    fp8_base_val = _as_bool(_get('fp8_base', True))
    lines.append(f"fp8_base = {'true' if fp8_base_val else 'false'}")
    fp8_scaled_val = _as_bool(_get('fp8_scaled', True))
    lines.append(f"fp8_scaled = {'true' if fp8_scaled_val else 'false'}")
    load_text_encoder_in_8bit = _as_bool(_get('8_bit_te', True))
    lines.append(f"8_bit_te = {'true' if load_text_encoder_in_8bit else 'false'}")
    attn_chunking_val = _as_bool(_get('attn_chunking', False))
    lines.append(f"attn_chunking = {'true' if attn_chunking_val else 'false'}")
    blank_preservation_val = _as_bool(_get('blank_preservation', False))
    lines.append(f"blank_preservation = {'true' if blank_preservation_val else 'false'}")
    blank_preservation_args_val = _get('blank_preservation_args', 'multiplier=0.5')
    lines.append(f"blank_preservation_args = {_quote(blank_preservation_args_val)}")
    dop_val = _as_bool(_get('dop', False))
    lines.append(f"dop = {'true' if dop_val else 'false'}")
    dop_args_val = _get('dop_args', 'class=woman multiplier=1.0')
    lines.append(f"dop_args = {_quote(dop_args_val)}")
    prior_divergence_val = _as_bool(_get('prior_divergence', False))
    lines.append(f"prior_divergence = {'true' if prior_divergence_val else 'false'}")
    prior_divergence_args_val = _get('prior_divergence_args', 'multiplier=0.1')
    lines.append(f"prior_divergence_args = {_quote(prior_divergence_args_val)}")
    crepa_val = _as_bool(_get('crepa', False))
    lines.append(f"crepa = {'true' if crepa_val else 'false'}")
    crepa_mode_val = _get('crepa_mode', 'backbone')
    lines.append(f"crepa_mode = {_quote(crepa_mode_val)}")
    crepa_args_val = _get('crepa_args', 'student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2')
    lines.append(f"crepa_args = {_quote(crepa_args_val)}")
    self_flow_val = _as_bool(_get('self_flow', False))
    lines.append(f"self_flow = {'true' if self_flow_val else 'false'}")
    self_flow_args_val = _get('self_flow_args', 'teacher_mode=base student_block_ratio=0.3 teacher_block_ratio=0.7 lambda_self_flow=0.1')
    lines.append(f"self_flow_args = {_quote(self_flow_args_val)}")
    cts_lambda_val = _as_bool(_get('cts_lambda', False))
    lines.append(f"cts_lambda = {'true' if cts_lambda_val else 'false'}")
    cts_lambda_args_val = _get('cts_lambda_args', 'video_driven=0.3 audio_driven=0.1')
    lines.append(f"cts_lambda_args = {_quote(cts_lambda_args_val)}")
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

    # Add dataset_list for UI state (stores selected dataset names)
    dataset_names = []
    config_page = getattr(training_tab_container, 'config_page_content', None)
    if config_page:
        for ds_num in [1, 2, 3]:
            ds_block = getattr(config_page, f'dataset_{ds_num}_block', None)
            if ds_block and hasattr(ds_block, 'get_selected_dataset'):
                selected_name = ds_block.get_selected_dataset()
                if selected_name:
                    dataset_names.append(selected_name)

    if dataset_names:
        # Format as TOML string array
        dataset_list_formatted = "[" + ", ".join(f'"{name}"' for name in dataset_names) + "]"
        lines.append(f"dataset_list = {dataset_list_formatted}")

    lines.append("")

    # [flow_matching]
    lines.append("[flow_matching]")
    timestep_mode = _get('timestep_sm_m', 'shifted_logit_normal')
    # Convert UI values to canonical names (keep shifted_logit_normal as-is)
    lines.append(f"timestep_sampling_mode = {_quote(timestep_mode)}")
    lines.append("timestep_sampling_params = { }")
    lines.append("")

    # [checkpoints]
    lines.append("[checkpoints]")
    lines.append(f"mode = {_quote(_get('checkpoint_mode', 'steps'))}")
    save_state_val = _as_bool(_get('save_state', False))
    lines.append(f"save_state = {'true' if save_state_val else 'false'}")
    lines.append(f"interval = {_clean_value(_get('interval', 50), is_numeric=True)}")
    lines.append(f"keep_last_n = {_clean_value(_get('keep_last_n', -1), is_numeric=True)}")
    lines.append(f"precision = {_quote(_get('precision', 'bfloat16'))}")
    convert_to_comfy_val = _as_bool(_get('convert_to_comfy', True))
    lines.append(f"convert_to_comfy = {'true' if convert_to_comfy_val else 'false'}")
    lines.append("")

    # [validation]
    lines.append("[validation]")
    lines.append(f"interval = {_clean_value(_get('sample_every_n_interval', '-1'), is_numeric=True)}")
    sample_at_first_val = _as_bool(_get('sample_at_first', False))
    lines.append(f"sample_at_first = {'true' if sample_at_first_val else 'false'}")
    generate_audio_val = _as_bool(_get('generate_audio', False))
    lines.append(f"generate_audio = {'true' if generate_audio_val else 'false'}")
    lines.append(f"prompts = {_quote_multiline(_get('prompts', 'Two women with long brown hair dancing on the dance floor'))}")
    lines.append(f"negative_prompt = {_quote_multiline(_get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted'))}")
    lines.append(f"start_images = {_quote(_get('start_images', 'none'))}")
    lines.append(f"video_dims = {_quote(_get('video_dims', '768, 512, 45'))}")
    lines.append(f"sample_steps = {_clean_value(_get('sample_steps', '30'), is_numeric=True)}")
    lines.append(f"guidance_scale = {_clean_value(_get('guidance_scale', '4.0'), is_numeric=False)}")
    lines.append(f"seed = {_clean_value(_get('seed', '42'), is_numeric=True)}")
    s_offload_val = _as_bool(_get('s_offload', True))
    lines.append(f"s_offload = {'true' if s_offload_val else 'false'}")
    tiled_vae_val = _as_bool(_get('tiled_vae', True))
    lines.append(f"tiled_vae = {'true' if tiled_vae_val else 'false'}")
    cache_te_val = _as_bool(_get('cache_te', True))
    lines.append(f"cache_te = {'true' if cache_te_val else 'false'}")
    lines.append("")

    return "\n".join(lines) + "\n"


def update_ltx2_ui_from_toml(training_tab_container, toml_data: dict) -> None:
    """Update UI controls from LTX2 TOML data."""
    import sys
    print(f"[DEBUG] update_ltx2_ui_from_toml called, toml_data keys: {list(toml_data.keys())}", file=sys.stderr, flush=True)
    from flet_app.ui.utils.config_utils import collapse_model_path
    import flet as ft

    # Import for optimizer type conversion
    try:
        from .config.musubi.optimizer import get_musubi_optimizer_type_for_ui
    except ImportError:
        def get_musubi_optimizer_type_for_ui(toml_value: str) -> str:
            return str(toml_value).strip()

    def _set_field_value(label, value):
        """Helper to find and set a control's value by label or data attribute."""
        try:
            found = False
            matched_controls = []
            def _apply(control, depth=0):
                nonlocal found
                if hasattr(control, 'controls') and control.controls:
                    for c in control.controls:
                        _apply(c, depth + 1)
                if hasattr(control, 'content') and control.content:
                    _apply(control.content, depth + 1)

                ctrl_label = getattr(control, 'label', None)
                ctrl_data = getattr(control, 'data', None)
                # Match by label or data attribute
                if ctrl_label == label or ctrl_data == label:
                    found = True
                    matched_controls.append(f"{'  '*depth}{type(control).__name__}(label={ctrl_label}, data={ctrl_data})")
                    if isinstance(control, ft.TextField):
                        # Handle empty/None values - show as "null" string for specific fields
                        if value is None or value == '':
                            control.value = "null"
                        else:
                            control.value = str(value)
                        # Explicitly update the textfield
                        if hasattr(control, 'update'):
                            try:
                                control.update()
                            except Exception:
                                pass
                    elif isinstance(control, ft.Dropdown):
                        # Handle boolean values properly (convert to lowercase string)
                        if isinstance(value, bool):
                            control.value = 'true' if value else 'false'
                        else:
                            control.value = str(value) if value is not None else ""
                        # Explicitly update the dropdown
                        if hasattr(control, 'update'):
                            try:
                                control.update()
                            except Exception:
                                pass
                    elif isinstance(control, ft.Checkbox):
                        logger.info(f"Setting checkbox '{label}' to {value} (current={control.value})")
                        if isinstance(value, bool):
                            control.value = value
                        else:
                            control.value = str(value).lower() in ['true', '1', 'yes', 'on']
                        # Explicitly update the checkbox
                        if hasattr(control, 'update'):
                            try:
                                control.update()
                            except Exception:
                                pass
                    if hasattr(control, 'page') and control.page:
                        control.update()

            config_content = getattr(training_tab_container, 'config_page_content', None)
            if config_content:
                _apply(config_content)
                # Log if specific fields were not found
                if label in ['self_flow', 'cts_lambda', 'generate_audio']:
                    if found:
                        logger.info(f"{label} found and set to {value}. Matched: {matched_controls}")
                    else:
                        logger.warning(f"{label} NOT found in config_page_content")
        except Exception as e:
            logger.warning(f"Error setting field {label} to {value}: {e}")

    try:
        # Model section - Set Trainer first (affects model type options)
        model = toml_data.get('model', {})
        if model:
            # Store the original name from the config for later use when saving
            original_name = model.get('name', '')
            page = getattr(training_tab_container, 'page', None)
            if page:
                page.original_config_name = original_name

            trainer = model.get('trainer', 'musubi')  # LTX2 uses musubi by default
            _set_field_value('Trainer', trainer)

            # Trigger trainer change to update model type options BEFORE setting model type
            try:
                from flet_app.ui.pages.training_config import trainer_dropdown_ref
                if trainer_dropdown_ref and trainer_dropdown_ref.current:
                    if callable(getattr(trainer_dropdown_ref.current, 'on_change', None)):
                        trainer_dropdown_ref.current.on_change(ft.ControlEvent('change'))
            except Exception:
                pass

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

            # Explicitly set frame_extraction visibility on all dataset blocks for LTX2
            # This ensures is_ltx2_model["value"] is set before datasets are populated
            try:
                config_content = getattr(training_tab_container, 'config_page_content', None)
                if config_content:
                    for i in [1, 2, 3]:
                        ds_block = getattr(config_content, f'dataset_{i}_block', None)
                        if ds_block and hasattr(ds_block, 'set_frame_extraction_visible'):
                            ds_block.set_frame_extraction_visible(True)
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

            # training_mode -> adapter dropdown (lora/lokr)
            training_mode = model.get('training_mode', 'lora')
            # Strip quotes if present (TOML may have 'lokr' with quotes)
            if isinstance(training_mode, str):
                training_mode = training_mode.strip().strip("'").strip('"')
            _set_field_value('adapter', training_mode)

        # Output dir - set the bottom bar output_dir_field
        # output_dir can be at top level OR inside [model] section for LTX2
        output_dir = toml_data.get('output_dir', '')
        if not output_dir:
            model = toml_data.get('model', {})
            if isinstance(model, dict):
                output_dir = model.get('output_dir', '')
        if output_dir:
            # Collapse to relative path for UI display
            try:
                from flet_app.project_root import get_project_root
                proj_root = str(get_project_root()).replace('\\', '/').rstrip('/').lower()
                od = str(output_dir).replace('\\', '/').lower()
                if od.startswith(proj_root + '/'):
                    # Strip project root to get relative path
                    output_dir = output_dir.replace('\\', '/')[len(proj_root) + 1:]
            except Exception:
                pass

        # Set the bottom bar output_dir_field
        try:
            bb_field = getattr(training_tab_container, 'output_dir_field', None)
            if bb_field is not None and isinstance(bb_field, ft.TextField):
                bb_field.value = str(output_dir) if output_dir is not None else ''
                if bb_field.page:
                    bb_field.update()
        except Exception:
            pass

        # LoRA section
        lora = toml_data.get('lora', {})
        # Always load values (section may be empty dict)
        _set_field_value('rank', lora.get('rank', 32))
        _set_field_value('alpha', lora.get('alpha', 32))
        _set_field_value('factor', lora.get('factor', 4))
        _set_field_value('network_dropout', lora.get('network_dropout', lora.get('dropout', 0.0)))
        _set_field_value('caption_dropout_rate', lora.get('caption_dropout_rate', 0.0))
        _set_field_value('init_from_existing', lora.get('init_from_existing', ''))

        # Also set factor field directly via ref (in case _set_field_value didn't find it)
        try:
            from flet_app.ui.pages.training_config import factor_field_ref
            if factor_field_ref and factor_field_ref.current:
                factor_val = lora.get('factor', 4)
                factor_field_ref.current.value = str(factor_val)
                if factor_field_ref.current.page:
                    factor_field_ref.current.update()
        except Exception:
            pass

        # Also set the bottom bar init_from_existing_field directly
        # (since _set_field_value only searches config_page_content)
        try:
            init_from_val = lora.get('init_from_existing', '')
            if init_from_val:
                init_from_val = collapse_model_path(init_from_val)
            init_field = getattr(training_tab_container, 'init_from_existing_field', None)
            if init_field is not None and isinstance(init_field, ft.TextField):
                init_field.value = str(init_from_val) if init_from_val is not None else ''
                if init_field.page:
                    init_field.update()
        except Exception:
            pass

        # LoRA target modules section
        lora_target_modules = toml_data.get('lora', {}).get('target_modules', {})
        if not lora_target_modules:
            # Try as separate section [lora.target_modules]
            lora_target_modules = toml_data.get('lora.target_modules', {})
        # No LTX2 module checkboxes to load anymore

        # Training strategy
        training_strategy = toml_data.get('training_strategy', {})
        # Always load values (section may be empty dict)
        _set_field_value('first_frame_conditioning_p', training_strategy.get('first_frame_conditioning_p', 0.1))
        _set_field_value('ltx_mode', training_strategy.get('ltx_mode', 'video'))
        _set_field_value('target_fps', training_strategy.get('target_fps', '25'))
        _set_field_value('frame_extraction', training_strategy.get('frame_extraction', 'head'))
        separate_audio_buckets = training_strategy.get('separate_audio_buckets', True)
        if not isinstance(separate_audio_buckets, bool):
            separate_audio_buckets = str(separate_audio_buckets).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('separate_audio_buckets', separate_audio_buckets)
        slider = training_strategy.get('slider', False)
        if not isinstance(slider, bool):
            slider = str(slider).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('slider', slider)
        ltx_2_3 = training_strategy.get('ltx_2_3', False)
        if not isinstance(ltx_2_3, bool):
            ltx_2_3 = str(ltx_2_3).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('ltx_2_3', ltx_2_3)
        use_mask = training_strategy.get('use_mask', False)
        if not isinstance(use_mask, bool):
            use_mask = str(use_mask).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('use_mask', use_mask)
        ic_lora = training_strategy.get('ic_lora', False)
        if not isinstance(ic_lora, bool):
            ic_lora = str(ic_lora).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('ic_lora', ic_lora)
        vace_lora = training_strategy.get('vace_lora', False)
        if not isinstance(vace_lora, bool):
            vace_lora = str(vace_lora).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('vace_lora', vace_lora)
        ref_downscale = training_strategy.get('ref_downscale', 1)
        _set_field_value('ref_downscale', ref_downscale)

        # Trigger slider on_change to update sample_slider_range visibility
        # and ic_lora on_change to update ref_downscale visibility
        try:
            def _trigger_checkbox_change(control):
                ctrl_label = getattr(control, 'label', None)
                if ctrl_label in ('slider', 'ic_lora', 'vace_lora') and isinstance(control, ft.Checkbox):
                    if hasattr(control, 'on_change') and control.on_change:
                        control.on_change(ft.ControlEvent('change'))
                        return True
                if hasattr(control, 'controls') and control.controls:
                    for c in control.controls:
                        if _trigger_checkbox_change(c):
                            return True
                if hasattr(control, 'content') and control.content:
                    return _trigger_checkbox_change(control.content)
                return False

            config_content = getattr(training_tab_container, 'config_page_content', None)
            if config_content:
                _trigger_checkbox_change(config_content)
        except Exception:
            pass

        sample_slider_range = training_strategy.get('sample_slider_range', '-2.0, -1.0, 0.0, 1.0, 2.0')
        _set_field_value('sample_slider_range', sample_slider_range)

        # Parse control_args and set i2v_type and sample_each
        control_args = training_strategy.get('control_args', None)
        if control_args and isinstance(control_args, list):
            if len(control_args) >= 1:
                i2v_type_val = control_args[0]
                _set_field_value('i2v_type', i2v_type_val)
                if len(control_args) >= 2:
                    sample_each_val = control_args[1]
                    _set_field_value('sample_each', sample_each_val)
                else:
                    # For single-element modes (reverse, freeze), set default sample_each
                    _set_field_value('sample_each', '3')
            else:
                # Empty list, set defaults
                _set_field_value('i2v_type', 'jump')
                _set_field_value('sample_each', '3')
        else:
            # No control_args, set defaults
            _set_field_value('i2v_type', 'jump')
            _set_field_value('sample_each', '3')

        # Optimization section
        optimization = toml_data.get('optimization', {})
        # Always load values (section may be empty dict)
        _set_field_value('learning_rate', optimization.get('learning_rate', 0.0001))
        _set_field_value('audio_lr', optimization.get('audio_lr', 0.0000))
        # Backward compatibility: check for 'steps' if 'max_steps' not found
        max_steps_val = optimization.get('max_steps', optimization.get('steps', 2000))
        _set_field_value('max_steps', max_steps_val)
        _set_field_value('batch_size', optimization.get('batch_size', 1))
        _set_field_value('grad_accum_steps', optimization.get('gradient_accumulation_steps', 1))
        _set_field_value('max_grad_norm', optimization.get('max_grad_norm', 1.0))
        _set_field_value('blocks_to_swap', optimization.get('blocks_to_swap', 0))
        # caption_dropout_rate -> caption_dropout (UI field name)
        caption_dropout_val = optimization.get('caption_dropout_rate', 0.0)
        _set_field_value('caption_dropout', caption_dropout_val)
        # Convert TOML optimizer type (lowercase) to UI format (capitalized)
        opt_type_toml = optimization.get('optimizer_type', 'AdamW')
        opt_type_ui = get_musubi_optimizer_type_for_ui(opt_type_toml)
        _set_field_value('optimizer_type_m', opt_type_ui)
        _set_field_value('scheduler_type', optimization.get('scheduler_type', 'constant'))

        # Load optimizer_args if present (for Automagic optimizer)
        if 'optimizer_args' in optimization:
            _set_field_value('optimizer_args', optimization.get('optimizer_args'))

        # Load lr_warmup_steps (flattened in optimization section, with backward compatibility)
        lr_warmup_steps = optimization.get('lr_warmup_steps')
        if lr_warmup_steps is None:
            # Backward compatibility: check old nested format
            scheduler_params = optimization.get('scheduler_params', {})
            if isinstance(scheduler_params, dict):
                lr_warmup_steps = scheduler_params.get('num_warmup_steps', 50)
            else:
                lr_warmup_steps = 50
        _set_field_value('lr_warmup_steps', lr_warmup_steps)

        # Set lr_warmup_steps visibility based on scheduler_type value
        # This is needed because the on_change trigger might not work when loading from TOML
        try:
            scheduler_value = optimization.get('scheduler_type', 'constant')

            def _set_lr_warmup_steps_visibility(control, value):
                """Recursively find lr_warmup_steps and set its visibility."""
                if hasattr(control, 'controls') and control.controls:
                    for c in control.controls:
                        _set_lr_warmup_steps_visibility(c, value)
                if hasattr(control, 'content') and control.content:
                    _set_lr_warmup_steps_visibility(control.content, value)

                # Find lr_warmup_steps by label or data attribute
                ctrl_label = getattr(control, 'label', None)
                ctrl_data = getattr(control, 'data', None)
                if (ctrl_label == 'lr_warmup_steps' or ctrl_data == 'lr_warmup_steps') and isinstance(control, ft.TextField):
                    control.visible = (value == 'constant_with_warmup')
                    if hasattr(control, 'page') and control.page:
                        control.update()

            config_content = getattr(training_tab_container, 'config_page_content', None)
            if config_content:
                _set_lr_warmup_steps_visibility(config_content, scheduler_value)
        except Exception:
            pass

        # Note: optimizer_args visibility is now handled by the generic update_musubi_ui_from_toml
        # in config_utils_musubi.py, which is called for all musubi models including LTX2.

        # Also try to trigger on_change for consistency
        # Trigger scheduler_type on_change to update lr_warmup_steps visibility
        try:
            def _trigger_scheduler_change(control):
                ctrl_label = getattr(control, 'label', None)
                if ctrl_label == 'scheduler_type' and isinstance(control, ft.Dropdown):
                    if hasattr(control, 'on_change') and control.on_change:
                        control.on_change(ft.ControlEvent('change'))
                        return True
                if hasattr(control, 'controls') and control.controls:
                    for c in control.controls:
                        if _trigger_scheduler_change(c):
                            return True
                if hasattr(control, 'content') and control.content:
                    return _trigger_scheduler_change(control.content)
                return False

            config_content = getattr(training_tab_container, 'config_page_content', None)
            if config_content:
                _trigger_scheduler_change(config_content)
                page = getattr(training_tab_container, 'page', None)
                if page:
                    page.update()
        except Exception:
            pass

        # Note: optimizer_args on_change triggering is now handled by the generic update_musubi_ui_from_toml
        # in config_utils_musubi.py, which is called for all musubi models including LTX2.

        gradient_checkpointing = optimization.get('enable_gradient_checkpointing', True)
        if not isinstance(gradient_checkpointing, bool):
            gradient_checkpointing = str(gradient_checkpointing).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('gradient_checkpointing', gradient_checkpointing)

        # Acceleration section
        acceleration = toml_data.get('acceleration', {})
        # Always load values (section may be empty dict)
        _set_field_value('mixed_precision_mode', acceleration.get('mixed_precision_mode', 'bf16'))
        _set_field_value('fp8_base', acceleration.get('fp8_base', True))
        _set_field_value('fp8_scaled', acceleration.get('fp8_scaled', True))
        _set_field_value('8_bit_te', acceleration.get('8_bit_te', True))
        _set_field_value('attn_chunking', acceleration.get('attn_chunking', False))
        _set_field_value('blank_preservation', acceleration.get('blank_preservation', False))
        _set_field_value('blank_preservation_args', acceleration.get('blank_preservation_args', 'multiplier=0.5'))
        _set_field_value('dop', acceleration.get('dop', False))
        _set_field_value('dop_args', acceleration.get('dop_args', 'class=woman multiplier=1.0'))
        _set_field_value('prior_divergence', acceleration.get('prior_divergence', False))
        _set_field_value('prior_divergence_args', acceleration.get('prior_divergence_args', 'multiplier=0.1'))
        _set_field_value('crepa', acceleration.get('crepa', False))
        _set_field_value('crepa_mode', acceleration.get('crepa_mode', 'backbone'))
        _set_field_value('crepa_args', acceleration.get('crepa_args', 'student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2'))
        _set_field_value('self_flow', acceleration.get('self_flow', False))
        _set_field_value('self_flow_args', acceleration.get('self_flow_args', 'teacher_mode=base student_block_ratio=0.3 teacher_block_ratio=0.7 lambda_self_flow=0.1'))
        _set_field_value('cts_lambda', acceleration.get('cts_lambda', False))
        _set_field_value('cts_lambda_args', acceleration.get('cts_lambda_args', 'video_driven=0.3 audio_driven=0.1'))

        # Checkpoints section
        checkpoints = toml_data.get('checkpoints', {})
        # Always load values (section may be empty dict)
        # Backward compatibility: check for 'mode' or 'checkpoint_mode'
        checkpoint_mode_val = checkpoints.get('mode', checkpoints.get('checkpoint_mode', 'steps'))
        _set_field_value('checkpoint_mode', checkpoint_mode_val)
        save_state = checkpoints.get('save_state', False)
        if not isinstance(save_state, bool):
            save_state = str(save_state).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('save_state', save_state)
        _set_field_value('interval', checkpoints.get('interval', 50))
        _set_field_value('keep_last_n', checkpoints.get('keep_last_n', -1))
        _set_field_value('precision', checkpoints.get('precision', 'bfloat16'))
        _set_field_value('convert_to_comfy', checkpoints.get('convert_to_comfy', 'true'))

        # Validation section - always load values (section may be empty dict)
        validation = toml_data.get('validation', {})
        import sys
        print(f"[DEBUG] validation section: {validation}", file=sys.stderr, flush=True)
        # Always load validation values, even if section is empty (use defaults)
        # Don't use 'if validation:' as empty dict {} is falsy
        _set_field_value('sample_every_n_interval', validation.get('interval', '-1'))
        _set_field_value('sample_at_first', validation.get('sample_at_first', 'false'))
        generate_audio = validation.get('generate_audio', False)
        print(f"[DEBUG] generate_audio from TOML: {generate_audio} (type: {type(generate_audio)})", file=sys.stderr, flush=True)
        if not isinstance(generate_audio, bool):
            generate_audio = str(generate_audio).lower() in ['true', '1', 'yes', 'on']
        print(f"[DEBUG] calling _set_field_value('generate_audio', {generate_audio})", file=sys.stderr, flush=True)
        _set_field_value('generate_audio', generate_audio)
        _set_field_value('prompts', validation.get('prompts', 'Two women with long brown hair dancing on the dance floor'))
        _set_field_value('negative_prompt', validation.get('negative_prompt', 'worst quality, inconsistent motion, blurry, jittery, distorted'))
        _set_field_value('start_images', validation.get('start_images', 'none'))
        _set_field_value('video_dims', validation.get('video_dims', '768, 512, 45'))
        _set_field_value('sample_steps', validation.get('sample_steps', '30'))
        _set_field_value('guidance_scale', validation.get('guidance_scale', '4.0'))
        _set_field_value('seed', validation.get('seed', '42'))
        s_offload = validation.get('s_offload', True)
        if not isinstance(s_offload, bool):
            s_offload = str(s_offload).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('s_offload', s_offload)
        tiled_vae = validation.get('tiled_vae', True)
        if not isinstance(tiled_vae, bool):
            tiled_vae = str(tiled_vae).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('tiled_vae', tiled_vae)
        cache_te = validation.get('cache_te', True)
        if not isinstance(cache_te, bool):
            cache_te = str(cache_te).lower() in ['true', '1', 'yes', 'on']
        _set_field_value('cache_te', cache_te)

        # Flow matching (timestep_sm)
        flow_matching = toml_data.get('flow_matching', {})
        if flow_matching:
            timestep_mode = flow_matching.get('timestep_sampling_mode', 'shifted_logit_normal')
            # No conversion needed - keep as-is
            _set_field_value('timestep_sm_m', timestep_mode)

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

        # Sync dependent field visibility (sample_slider_range, preservation args)
        try:
            from flet_app.ui.pages.training_config import sync_dependent_field_visibility
            sync_dependent_field_visibility()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Error updating LTX2 UI from TOML: {e}")
