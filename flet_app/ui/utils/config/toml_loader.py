"""
TOML configuration loader for populating UI from TOML files.

Handles reading TOML data and applying values to Flet UI controls.
"""

import os
from typing import Any, Dict, Set
from loguru import logger
import flet as ft

from .constants import ALWAYS_INCLUDE_FIELDS
from .toml_formatting import to_bool, collapse_model_path, collapse_path_to_relative


# =============================================================================
# Section Populators
# =============================================================================

def populate_model_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [model] section."""
    from flet_app.ui.utils.model_settings import populate_label_vals_from_model

    model = toml_data.get('model', {}) or {}
    if not isinstance(model, dict):
        return

    # Load trainer first (before model type, since trainer affects model type options)
    if 'trainer' in model:
        label_vals['Trainer'] = model.get('trainer')

    normalized_type = populate_label_vals_from_model(model, label_vals)
    if not normalized_type:
        return

    label_vals['Model Type'] = normalized_type
    is_ltx2 = str(normalized_type).lower() in ('ltx-video-2', 'ltx2')

    if 'dtype' in model:
        label_vals['dtype'] = model.get('dtype')
    if 'transformer_dtype' in model:
        label_vals['transformer_dtype'] = model.get('transformer_dtype')

    if 'timestep_sample_method' in model:
        ts_method = model.get('timestep_sample_method')
        trainer = str(label_vals.get('Trainer', '')).lower() if 'Trainer' in label_vals else ''
        is_musubi = str(normalized_type).lower() in ('_wan22', 'ltx-video-2', 'ltx2') and trainer == 'musubi'

        if is_musubi and ts_method == 'logit_normal':
            ts_method = 'shifted_logit_normal'
        elif not is_ltx2 and not is_musubi and ts_method == 'shifted_logit_normal':
            ts_method = 'logit_normal'

        if is_musubi:
            label_vals['timestep_sm_m'] = ts_method
        else:
            label_vals['timestep_sm'] = ts_method

    if 'model_path' in model:
        label_vals['model_path'] = collapse_model_path(model.get('model_path', ''))
    if 'text_encoder_path' in model:
        label_vals['text_encoder_path'] = collapse_model_path(model.get('text_encoder_path', ''))
    if 'training_mode' in model:
        label_vals['adapter'] = model.get('training_mode')

    if is_ltx2:
        label_vals.pop('first_frame_conditioning_p', None)


def populate_optimizer_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [optimizer] or [optimization] section."""
    opt = toml_data.get('optimization', toml_data.get('optimizer', {})) or {}
    if not isinstance(opt, dict):
        return

    # Check if this is a musubi model - if so, skip setting optimizer_type_m
    # (will be handled by the musubi-specific loader)
    model = toml_data.get('model', {})
    is_musubi_model = False
    if isinstance(model, dict):
        model_type = str(model.get('type', '')).lower()
        trainer = str(model.get('trainer', '')).lower()
        is_musubi_model = (trainer == 'musubi' or 'ltx' in model_type or
                          'ltx2' in model_type or 'wan' in model_type)

    # Check for optimizer_type (musubi) or type (diffusion-pipe)
    if 'optimizer_type' in opt:
        opt_type = opt.get('optimizer_type')
    elif 'type' in opt:
        opt_type = opt.get('type')
    else:
        opt_type = None

    if opt_type:
        opt_type_str = str(opt_type).strip().lower()
        try:
            from flet_app.ui.pages.optimizer_field_config import get_optimizer_key
            mapped_key = get_optimizer_key(opt_type_str)
            if mapped_key:
                label_vals['optimizer_type'] = mapped_key
                if not is_musubi_model:
                    label_vals['optimizer_type_m'] = mapped_key
            else:
                label_vals['optimizer_type'] = opt_type_str
                if not is_musubi_model:
                    label_vals['optimizer_type_m'] = opt_type_str
        except Exception:
            label_vals['optimizer_type'] = opt_type_str
            if not is_musubi_model:
                label_vals['optimizer_type_m'] = opt_type_str

    # Common fields
    for k in ('lr', 'learning_rate', 'audio_lr', 'betas', 'weight_decay', 'eps'):
        if k in opt:
            label_vals[k] = opt.get(k)

    # Musubi optimization fields
    for k in ('max_steps', 'batch_size', 'max_grad_norm', 'loraplus_ratio', 'blocks_to_swap',
              'caption_dropout_rate', 'scheduler_type', 'optimizer_args', 'enable_gradient_checkpointing'):
        if k in opt:
            label_vals[k] = opt.get(k)

    # Handle gradient_accumulation_steps -> grad_accum_steps for UI
    if 'gradient_accumulation_steps' in opt:
        label_vals['gradient_accumulation_steps'] = opt.get('gradient_accumulation_steps')
        label_vals['grad_accum_steps'] = opt.get('gradient_accumulation_steps')

    # Handle caption_dropout_rate -> caption_dropout for UI
    if 'caption_dropout_rate' in opt:
        label_vals['caption_dropout_rate'] = opt.get('caption_dropout_rate')
        label_vals['caption_dropout'] = opt.get('caption_dropout_rate')

    # Prodigy-specific fields
    for k in ('beta3', 'd0', 'd_coef', 'schedulefree_c'):
        if k in opt:
            label_vals[k] = opt.get(k)

    # Automagic-specific fields
    for k in ('min_lr', 'max_lr', 'lr_bump', 'clip_threshold',
              'do_paramiter_swapping', 'paramiter_swapping_factor'):
        if k in opt:
            label_vals[k] = opt.get(k)


def populate_adapter_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [adapter] section."""
    ad = toml_data.get('adapter', {}) or {}
    if not isinstance(ad, dict):
        return

    if 'type' in ad:
        label_vals['adapter'] = ad.get('type')
    if 'rank' in ad:
        label_vals['a_rank'] = ad.get('rank')
    if 'dtype' in ad:
        label_vals['a_dtype'] = ad.get('dtype')
    if 'init_from_existing' in ad:
        label_vals['init_from_existing'] = ad.get('init_from_existing')


def populate_lora_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [lora] section."""
    lora = toml_data.get('lora', {}) or {}
    if not isinstance(lora, dict):
        return

    for k in ('rank', 'alpha', 'factor', 'network_dropout', 'caption_dropout_rate'):
        if k in lora:
            label_vals[k] = lora.get(k)
    # Backward compatibility: old 'dropout' field
    if 'dropout' in lora:
        label_vals['network_dropout'] = lora.get('dropout')


def populate_monitoring_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [monitoring] section."""
    mon = toml_data.get('monitoring', {}) or {}
    if not isinstance(mon, dict):
        return

    if 'enable_wandb' in mon:
        label_vals['enable_wandb'] = to_bool(mon.get('enable_wandb'))
    for k in ('wandb_api_key', 'wandb_tracker_name', 'wandb_run_name'):
        if k in mon:
            label_vals[k] = mon.get(k)


def populate_data_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [data] section."""
    data = toml_data.get('data', {}) or {}
    if not isinstance(data, dict):
        return

    if 'preprocessed_data_root' in data:
        label_vals['preprocessed_data_root'] = data.get('preprocessed_data_root')
    if 'dataset_list' in data:
        dataset_list = data.get('dataset_list', [])
        if isinstance(dataset_list, list):
            label_vals['dataset_list'] = dataset_list


def populate_checkpoints_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [checkpoints] section."""
    checkpoints = toml_data.get('checkpoints', {}) or {}
    if not isinstance(checkpoints, dict):
        return

    if 'mode' in checkpoints:
        label_vals['checkpoint_mode'] = checkpoints.get('mode')
    if 'save_state' in checkpoints:
        label_vals['Save State'] = to_bool(checkpoints.get('save_state'))
    for k in ('interval', 'keep_last_n', 'precision'):
        if k in checkpoints:
            label_vals[k] = checkpoints.get(k)
    # convert_to_comfy needs to be lowercase string for dropdown matching
    if 'convert_to_comfy' in checkpoints:
        val = checkpoints.get('convert_to_comfy')
        label_vals['convert_to_comfy'] = str(val).lower() if val is not None else 'false'


def populate_training_strategy_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [training_strategy] section."""
    ts = toml_data.get('training_strategy', {}) or {}
    if not isinstance(ts, dict):
        return

    for k in ('first_frame_conditioning_p', 'ltx_mode', 'target_fps'):
        if k in ts:
            label_vals[k] = ts.get(k)
    for k in ('separate_audio_buckets', 'slider', 'ic_lora', 'use_mask', 'ltx_2_3'):
        if k in ts:
            label_vals[k] = to_bool(ts.get(k, False))

    # Handle ref_downscale for IC-LoRA
    if 'ref_downscale' in ts:
        label_vals['ref_downscale'] = ts.get('ref_downscale')

    # Handle sample_slider_range
    if 'sample_slider_range' in ts:
        label_vals['sample_slider_range'] = ts.get('sample_slider_range')

    # Handle control_args - parse into i2v_type and sample_each
    if 'control_args' in ts:
        control_args = ts.get('control_args')
        if isinstance(control_args, list) and len(control_args) >= 1:
            label_vals['i2v_type'] = str(control_args[0])
            if len(control_args) >= 2:
                label_vals['sample_each'] = str(control_args[1])
            else:
                # For single-element modes (reverse, freeze), default sample_each to 3
                label_vals['sample_each'] = '3'
        else:
            # Default values if control_args is invalid
            label_vals['i2v_type'] = 'jump'
            label_vals['sample_each'] = '3'
    else:
        # Default values if control_args not present
        label_vals['i2v_type'] = 'jump'
        label_vals['sample_each'] = '3'


def populate_validation_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [validation] section."""
    val = toml_data.get('validation', {}) or {}
    if not isinstance(val, dict):
        return

    # For sample_at_first, keep as lowercase string for dropdown compatibility
    if 'sample_at_first' in val:
        label_vals['sample_at_first'] = str(val.get('sample_at_first', 'false')).lower()

    # For interval, check if this is an LTX2/musubi config by looking at the model section
    # If so, use sample_every_n_interval as the key; otherwise use validation_interval
    if 'interval' in val:
        model = toml_data.get('model', {})
        is_ltx2_or_musubi = False
        if isinstance(model, dict):
            model_type = str(model.get('type', '')).lower()
            trainer = str(model.get('trainer', '')).lower()
            is_ltx2_or_musubi = ('ltx' in model_type or 'ltx2' in model_type or
                                'wan' in model_type or trainer == 'musubi')
        interval_val = val.get('interval')
        if is_ltx2_or_musubi:
            label_vals['sample_every_n_interval'] = interval_val
        else:
            label_vals['validation_interval'] = interval_val

    bool_fields = ('generate_audio', 's_offload', 'tiled_vae', 'cache_te')
    for k in ('sample_steps', 'guidance_scale', 'seed'):
        if k in val:
            label_vals[k] = val.get(k)
    for k in bool_fields:
        if k in val:
            label_vals[k] = to_bool(val.get(k))
    for k in ('prompts', 'negative_prompt', 'start_images', 'video_dims'):
        if k in val:
            label_vals[k] = val.get(k)


def populate_acceleration_section(toml_data: dict, label_vals: dict) -> None:
    """Populate label_vals from the [acceleration] section."""
    acc = toml_data.get('acceleration', {}) or {}
    if not isinstance(acc, dict):
        return

    # Try musubi-specific handler first
    try:
        from flet_app.ui.utils.config_utils_musubi import populate_musubi_acceleration_section
        populate_musubi_acceleration_section(toml_data, label_vals, to_bool)
    except ImportError:
        if 'mixed_precision_mode' in acc:
            label_vals['mixed_precision_mode'] = acc.get('mixed_precision_mode')
        if 'fp8_base' in acc:
            label_vals['fp8_base'] = to_bool(acc.get('fp8_base', True))
        if 'fp8_scaled' in acc:
            label_vals['fp8_scaled'] = to_bool(acc.get('fp8_scaled', True))

    if '8_bit_te' in acc:
        label_vals['8_bit_te'] = to_bool(acc.get('8_bit_te', False))
    if 'load_text_encoder_in_8bit' in acc:
        label_vals['8_bit_te'] = to_bool(acc.get('load_text_encoder_in_8bit', False))


# =============================================================================
# Value Application
# =============================================================================

def apply_values_recursive(control: Any, label_vals: dict, page: Any) -> None:
    """Recursively apply values from label_vals to Flet controls."""
    try:
        if hasattr(control, 'controls') and control.controls:
            for c in control.controls:
                apply_values_recursive(c, label_vals, page)
        elif hasattr(control, 'content') and control.content:
            apply_values_recursive(control.content, label_vals, page)

        label = getattr(control, 'label', None)
        data = getattr(control, 'data', None)

        # Skip if no identifier or it's the Trainer dropdown
        if not (label or data) or label == 'Trainer':
            return

        # Check both label and data attribute against label_vals keys
        key = None
        if label in label_vals:
            key = label
        elif data in label_vals:
            key = data
        else:
            return

        val = label_vals[key]
        if isinstance(control, ft.TextField):
            control.value = str(val) if val is not None else ''
            if control.page:
                control.update()
        elif isinstance(control, ft.Dropdown):
            if val is None or str(val).strip() == '':
                control.value = None
            else:
                control.value = str(val)
            if control.page:
                control.update()
        elif isinstance(control, ft.Checkbox):
            control.value = to_bool(val) if val is not None else False
            if control.page:
                control.update()
    except Exception:
        pass


def apply_all_values(container: Any, label_vals: dict, page: Any) -> None:
    """Apply values to main config, dataset, and monitor containers."""
    # Handle Trainer dropdown first
    if 'Trainer' in label_vals:
        try:
            from flet_app.ui.pages.training_config import trainer_dropdown_ref
            if trainer_dropdown_ref and trainer_dropdown_ref.current:
                trainer_dropdown_ref.current.value = str(label_vals['Trainer'])
                if trainer_dropdown_ref.current.page:
                    trainer_dropdown_ref.current.update()
                if callable(getattr(trainer_dropdown_ref.current, 'on_change', None)):
                    class _E: pass
                    e = _E()
                    setattr(e, 'control', trainer_dropdown_ref.current)
                    setattr(e, 'page', page)
                    trainer_dropdown_ref.current.on_change(e)
        except Exception:
            pass

    # Apply to config page
    try:
        apply_values_recursive(container.config_page_content, label_vals, page)
    except Exception:
        pass

    # Apply to dataset page
    try:
        ds_content = getattr(container, 'dataset_page_content', None)
        if ds_content:
            apply_values_recursive(ds_content, label_vals, page)
    except Exception:
        pass

    # Apply to monitor page
    try:
        mon_content = getattr(container, 'monitor_page_content', None)
        if mon_content:
            apply_values_recursive(mon_content, label_vals, page)
    except Exception:
        pass

    # Handle optimizer_type dropdown - trigger visibility updates for automagic/prodigy rows
    if 'optimizer_type' in label_vals:
        try:
            from flet_app.ui.pages.training_config import (
                optimizer_type_dropdown_ref,
                on_optimizer_type_change,
            )
            if optimizer_type_dropdown_ref and optimizer_type_dropdown_ref.current:
                # Create a mock event object with the necessary attributes
                class _E: pass
                e = _E()
                setattr(e, 'control', optimizer_type_dropdown_ref.current)
                setattr(e, 'data', str(label_vals['optimizer_type']))
                setattr(e, 'page', page)
                # Call the change handler with from_toml_load=True to update visibility only
                on_optimizer_type_change(e, from_toml_load=True)
        except Exception:
            pass


def handle_dataset_selection(toml_data: dict, container: Any, page: Any) -> None:
    """Handle dataset selection based on TOML data."""
    try:
        dataset_path = toml_data.get('dataset')
        if not dataset_path:
            data_section = toml_data.get('data', {})
            if isinstance(data_section, dict):
                dataset_path = data_section.get('preprocessed_data_root')

        if not dataset_path:
            return

        clean_name = os.path.basename(str(dataset_path).replace('\\', '/'))
        if clean_name.lower().endswith('.toml'):
            clean_name = clean_name[:-5]

        ds_block = getattr(container, 'dataset_page_content', None)
        if ds_block and hasattr(ds_block, 'set_selected_dataset'):
            ds_block.set_selected_dataset(clean_name, page_ctx=page)

        data_cfg = getattr(container, 'data_config_page_content', None)
        if data_cfg and hasattr(data_cfg, 'dataset_block'):
            if hasattr(data_cfg.dataset_block, 'set_selected_dataset'):
                data_cfg.dataset_block.set_selected_dataset(clean_name, page_ctx=page)
            if hasattr(data_cfg, 'refresh_indicator'):
                try:
                    data_cfg.refresh_indicator(clean_name)
                except Exception:
                    pass
    except Exception:
        pass


def handle_dataset_list_selection(toml_data: dict, container: Any, page: Any) -> None:
    """Handle dataset_list to populate multiple dataset blocks."""
    try:
        dataset_list = None
        data_section = toml_data.get('data', {})
        if isinstance(data_section, dict):
            dataset_list = data_section.get('dataset_list', [])
        if not dataset_list or not isinstance(dataset_list, list):
            dataset_list = toml_data.get('dataset_list', [])
        if not dataset_list or not isinstance(dataset_list, list):
            return

        config_page = getattr(container, 'config_page_content', None)
        if not config_page:
            return

        # Clear all dataset blocks first
        for i in range(1, 4):
            ds_block = getattr(config_page, f'dataset_{i}_block', None)
            if ds_block and hasattr(ds_block, 'set_selected_dataset'):
                try:
                    ds_block.set_selected_dataset(None, page_ctx=page)
                except Exception:
                    pass

        # Populate with datasets from list
        for i, dataset_name in enumerate(dataset_list):
            if i >= 3:
                break
            ds_block = getattr(config_page, f'dataset_{i+1}_block', None)
            if ds_block and hasattr(ds_block, 'set_selected_dataset'):
                try:
                    ds_block.set_selected_dataset(dataset_name, page_ctx=page)
                except Exception:
                    pass
    except Exception:
        pass


# =============================================================================
# Main UI Updater
# =============================================================================

def update_ui_from_toml(container: Any, toml_data: dict) -> None:
    """Populate UI controls from TOML data."""
    page = getattr(container, 'page', None)
    label_vals = {}

    # Top-level keys
    top_keys = [
        'output_dir', 'epochs', 'micro_batch_size_per_gpu', 'pipeline_stages',
        'gradient_accumulation_steps', 'gradient_clipping', 'warmup_steps',
        'activation_checkpointing', 'eval_every_n_epochs', 'eval_micro_batch_size_per_gpu',
        'eval_gradient_accumulation_steps', 'save_every_n_epochs', 'checkpoint_every_n_minutes',
        'partition_method', 'save_dtype', 'caching_batch_size', 'steps_per_print', 'video_clip_mode',
        'init_from_existing'
    ]
    for k in top_keys:
        if k in toml_data:
            label_vals[k] = toml_data.get(k)

    # 8_bit_te (top-level)
    if '8_bit_te' in toml_data:
        label_vals['8_bit_te'] = to_bool(toml_data.get('8_bit_te', False))

    # Post-process output_dir for UI
    if 'output_dir' in label_vals and isinstance(label_vals['output_dir'], str):
        label_vals['output_dir'] = collapse_path_to_relative(label_vals['output_dir'])

    if 'eval_before_first_step' in toml_data:
        label_vals['eval_before_first_step'] = to_bool(toml_data.get('eval_before_first_step'))

    # Block swap
    if 'blocks_to_swap' in toml_data:
        label_vals['blocks_swap'] = toml_data.get('blocks_to_swap')
    if 'disable_block_swap_for_eval' in toml_data:
        label_vals['disable_bsfe'] = 'true' if to_bool(toml_data.get('disable_block_swap_for_eval')) else 'false'

    # Populate all sections
    populate_model_section(toml_data, label_vals)

    # Store the original name from the model section for later use when saving
    model = toml_data.get('model', {})
    if isinstance(model, dict) and 'name' in model and page:
        page.original_config_name = model.get('name', '')

    populate_optimizer_section(toml_data, label_vals)
    populate_adapter_section(toml_data, label_vals)
    populate_lora_section(toml_data, label_vals)
    populate_training_strategy_section(toml_data, label_vals)
    populate_acceleration_section(toml_data, label_vals)
    populate_checkpoints_section(toml_data, label_vals)
    populate_monitoring_section(toml_data, label_vals)
    populate_data_section(toml_data, label_vals)
    populate_validation_section(toml_data, label_vals)

    # Post-process init_from_existing
    if 'init_from_existing' in label_vals and isinstance(label_vals['init_from_existing'], str):
        label_vals['init_from_existing'] = collapse_path_to_relative(label_vals['init_from_existing'])

    # Trigger visibility updates before applying values
    try:
        from flet_app.ui.pages.training_config import model_type_dropdown_ref, optimizer_type_dropdown_ref
        from flet_app.ui.utils.model_settings import postprocess_visibility_after_apply
        postprocess_visibility_after_apply(label_vals, page, model_type_dropdown_ref)
    except Exception:
        pass

    # Musubi-specific loading
    try:
        model = toml_data.get('model', {})
        trainer = model.get('trainer', '')
        model_type = model.get('type', '')
        is_musubi = (trainer == 'musubi' or
                     any(m in str(model_type).lower() for m in ['ltx-video-2', 'ltx2', 'wan22', 'wan']))
        if is_musubi:
            from flet_app.ui.utils.config_utils_musubi import update_musubi_ui_from_toml
            update_musubi_ui_from_toml(container, toml_data)
    except Exception as e:
        logger.debug(f"Could not load musubi fields: {e}")

    # Apply values
    apply_all_values(container, label_vals, page)

    # Bottom bar fields
    try:
        if 'output_dir' in label_vals:
            bb_field = getattr(container, 'output_dir_field', None)
            if isinstance(bb_field, ft.TextField):
                bb_field.value = str(label_vals['output_dir']) if label_vals['output_dir'] else ''
                if bb_field.page:
                    bb_field.update()
    except Exception:
        pass

    try:
        if 'init_from_existing' in label_vals:
            bb_field = getattr(container, 'init_from_existing_field', None)
            if isinstance(bb_field, ft.TextField):
                bb_field.value = str(label_vals['init_from_existing']) if label_vals['init_from_existing'] else ''
                if bb_field.page:
                    bb_field.update()
    except Exception:
        pass

    # Sync dependent field visibility (factor field, etc.) after all values are applied
    try:
        from flet_app.ui.pages.training_config import sync_dependent_field_visibility
        sync_dependent_field_visibility()
    except Exception:
        pass

    # Dataset selection
    handle_dataset_selection(toml_data, container, page)
    handle_dataset_list_selection(toml_data, container, page)

    if page:
        try:
            page.update()
        except Exception:
            pass
