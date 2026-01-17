import os
import re
import traceback
from pathlib import Path
from flet_app.project_root import get_project_root as _get_project_root
from loguru import logger
import flet as ft

# Try importing TOML parsers
try:
    import tomllib as _toml_parser  # Python 3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as _toml_parser  # Fallback for older Python if available
    except Exception:
        _toml_parser = None

# Internal Imports
from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
from flet_app.ui.utils.model_settings import (
    append_model_specific_lines,
    populate_label_vals_from_model,
    postprocess_visibility_after_apply,
)


# =============================================================================
# Path & Utility Helpers
# =============================================================================

def get_project_root():  # type: ignore[no-redef]
    """Delegates to centralized project root helper."""
    return _get_project_root()

def _normalize_slashes(path: str) -> str:
    """Convert backslashes to forward slashes."""
    if not path:
        return path
    return str(path).strip().replace('\\', '/')

def _is_absolute_path(path: str) -> bool:
    """Check if a path is absolute (Windows drive, UNC, or POSIX root)."""
    if not path:
        return False
    return bool(re.match(r"^[A-Za-z]:[\\/]|^/|^\\\\", path))

def _quote(value):
    """Quote a string value for TOML output."""
    if value is None:
        return "''"
    return f"'{str(value)}'"

def _to_bool(value):
    """Convert a value to boolean for UI/TOML."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in ('1', 'true', 'yes', 'on')

def expand_model_path(path: str) -> str:
    """Expand relative model path to absolute path when saving to TOML."""
    if not path or not isinstance(path, str):
        return path

    path = _normalize_slashes(path)

    # If already absolute, return as-is
    if _is_absolute_path(path):
        return path

    # Get project root and expand
    project_root = get_project_root()
    expanded_path = os.path.join(str(project_root), path)

    return _normalize_slashes(expanded_path)

def collapse_model_path(path: str) -> str:
    """Collapse absolute model path to relative path when loading from TOML."""
    if not path or not isinstance(path, str):
        return path

    path = _normalize_slashes(path)

    # Get project root
    project_root = get_project_root()
    project_root_str = _normalize_slashes(str(project_root)).rstrip('/')

    # If path is under project root, make it relative
    if path.lower().startswith(project_root_str.lower() + '/'):
        relative_path = path[len(project_root_str) + 1:]
        return relative_path

    # If not under project root, return as-is
    return path

# =============================================================================
# Data Extraction Helpers
# =============================================================================

def extract_config_from_controls(control):
    """Recursively extract values from Flet controls into a dictionary."""
    result = {}
    
    def _extract(child):
        if hasattr(child, 'controls') and child.controls:
            for sub_child in child.controls:
                _extract(sub_child)
        elif hasattr(child, 'content') and child.content:
            _extract(child.content)
        elif isinstance(child, ft.TextField):
            if getattr(child, 'visible', True):
                result[child.label] = child.value
        elif isinstance(child, ft.Dropdown):
            if getattr(child, 'visible', True):
                result[child.label] = child.value
        elif isinstance(child, ft.Checkbox):
            if getattr(child, 'visible', True):
                key = getattr(child, 'data', None) or child.label
                # Handle special cases where label is the key
                if child.label in ['all_modules', 'video_attn', 'video_ff', 'audio_attn', 
                                   'audio_ff', 'cross_modal_attn', 'with_audio']:
                    key = child.label
                result[key] = child.value

    if control:
        _extract(control)
    return result

def _get_dataset_path_value(dataset_block):
    """Extract dataset path value from a UI dataset block."""
    if not dataset_block or not hasattr(dataset_block, 'get_selected_dataset'):
        return ""
    
    selected_clean = dataset_block.get_selected_dataset()
    if not selected_clean:
        return ""
    
    base_dir, _dtype = _get_dataset_base_dir(selected_clean)
    ds_full = os.path.join(base_dir, selected_clean)
    return _normalize_slashes(ds_full)

def _get_monitor_config(monitor_container):
    """Extract configuration from the monitor page container."""
    cfg = {}
    try:
        monitor_content = getattr(monitor_container, 'monitor_page_content', None)
        if monitor_content:
            cfg = extract_config_from_controls(monitor_content)
    except Exception:
        pass
    return cfg

# =============================================================================
# TOML Builder Helpers
# =============================================================================

def _build_training_section(lines, cfg, _get):
    """Writes training settings to lines list."""
    lines.append("# training settings")
    for key in [
        'epochs', 'micro_batch_size_per_gpu', 'pipeline_stages',
        'gradient_accumulation_steps', 'gradient_clipping', 'warmup_steps']:
        val = _get(key, None)
        if val is not None:
            lines.append(f"{key} = {val}")
    
    lr_sched = _get('lr_scheduler', 'constant')
    if lr_sched is not None and str(lr_sched).strip() != '':
        lines.append(f"lr_scheduler = {_quote(lr_sched)}")
    
    act_ckpt = _get('activation_checkpointing', 'unsloth')
    act_ckpt_str = str(act_ckpt).lower() if act_ckpt is not None else ''
    if act_ckpt_str in ['off', 'false'] or act_ckpt is False:
        lines.append("activation_checkpointing = false")
    elif act_ckpt_str in ['on', 'true'] or act_ckpt is True:
        lines.append("activation_checkpointing = true")
    else:
        lines.append(f"activation_checkpointing = {_quote(act_ckpt)}")

def _build_eval_section(lines, cfg, _get, _as_bool):
    """Writes eval settings to lines list."""
    lines.append("# eval settings")
    lines.append(f"eval_every_n_epochs = {_get('eval_every_n_epochs', 1)}")
    lines.append(f"eval_before_first_step = {'true' if _as_bool(_get('eval_before_first_step', True)) else 'false'}")
    lines.append(f"eval_micro_batch_size_per_gpu = {_get('eval_micro_batch_size_per_gpu', 1)}")
    lines.append(f"eval_gradient_accumulation_steps = {_get('eval_gradient_accumulation_steps', 1)}")

def _build_misc_section(lines, cfg, _get):
    """Writes misc settings to lines list."""
    lines.append("# misc settings")
    defaults = [
        ('save_every_n_epochs', 5),
        ('checkpoint_every_n_minutes', 10),
        ('partition_method', 'parameters'),
        ('save_dtype', 'bfloat16'),
        ('caching_batch_size', 1),
        ('steps_per_print', 1),
        ('video_clip_mode', 'single_beginning'),
    ]
    for key, dval in defaults:
        val = _get(key, dval)
        if isinstance(dval, str):
            lines.append(f"{key} = {_quote(val)}")
        else:
            lines.append(f"{key} = {val}")

def _build_model_section(lines, cfg, _get, expand_model_path_func):
    """Writes [model] section to lines list."""
    lines.append("[model]")
    model_source = _get('Model Type', '')
    lines.append(f"type = {_quote(model_source)}")

    mt_lower = model_source.strip().lower()

    # SDXL: write checkpoint_path immediately under type, if provided
    if mt_lower == 'sdxl':
        ckpt = _get('checkpoint_path', None)
        if ckpt is not None and str(ckpt).strip() != '':
            lines.append(f"checkpoint_path = {_quote(ckpt)}")

    # Base model paths (skip for SDXL and LTX models)
    if mt_lower not in ('sdxl', 'ltx-video', 'ltx', 'ltx-video-2'):
        diff_path = _get('diffusers_path', None)
        if diff_path is not None and str(diff_path).strip() != '':
            expanded_diff_path = expand_model_path_func(diff_path)
            lines.append(f"diffusers_path  = {_quote(expanded_diff_path)}")
        transf_path = _get('transformer_path', None)
        if transf_path is not None and str(transf_path).strip() != '':
            expanded_transf_path = expand_model_path_func(transf_path)
            lines.append(f"transformer_path   = {_quote(expanded_transf_path)}")
    
    # [model] - with_audio logic
    with_audio_val = False
    try:
        with_audio_control = getattr(_get('control_container'), 'with_audio_control', None)
        if with_audio_control:
            with_audio_val = with_audio_control.value
    except Exception:
        pass
    
    lines.append(f"with_audio = {'true' if with_audio_val else 'false'}")

    lines.append("")
    lines.append(f"dtype = {_quote(_get('dtype', 'bfloat16'))}")
    
    # transformer_dtype: handle UI 'None' by commenting out; else allow model-specific overrides
    mt_lower = str(_get('Model Type', '')).strip().lower()
    t_dtype = str(_get('transformer_dtype', 'float8'))
    
    if str(t_dtype).strip().lower() == 'none':
        lines.append("#transformer_dtype = 'float8'")
    else:
        try:
            if mt_lower == 'cosmos_predict2':
                f8e = _get('float8_e5m2', False)
                f8e_bool = isinstance(f8e, bool) and f8e or str(f8e).strip().lower() in ['1','true','yes','on']
                if f8e_bool:
                    t_dtype = 'float8_e5m2'
            if mt_lower == 'hidream':
                td = _get('t_dtype', False)
                td_bool = isinstance(td, bool) and td or str(td).strip().lower() in ['1','true','yes','on']
                if td_bool:
                    t_dtype = 'nf4'
            if mt_lower == 'longcat':
                td = _get('float8 t_dtype', False)
                td_bool = isinstance(td, bool) and td or str(td).strip().lower() in ['1','true','yes','on']
                if td_bool:
                    t_dtype = 'float8'
        except Exception:
            pass
        lines.append(f"transformer_dtype = {_quote(t_dtype)}")
    
    # Support renamed UI label 'timestep_sm' while saving canonical key
    _tsm = _get('timestep_sample_method', None)
    if _tsm is None or str(_tsm).strip() == '':
        _tsm = _get('timestep_sm', 'logit_normal')
    if str(_tsm).strip().lower() == 'none':
        lines.append("#timestep_sample_method = 'logit_normal'")
    else:
        lines.append(f"timestep_sample_method = {_quote(_tsm)}")
    
    # Append model-specific extra lines
    append_model_specific_lines(lines, _get, model_source)

def _build_optimizer_section(lines, cfg, _get):
    """Writes [optimizer] section to lines list."""
    lines.append("[optimizer]")
    lines.append(f"type = {_quote(_get('optimizer_type', 'adamw_optimi'))}")
    lines.append(f"lr = {_get('lr', 2e-5)}")
    lines.append(f"betas = {_get('betas', '[0.9, 0.99]')}")
    lines.append(f"weight_decay = {_get('weight_decay', 0.01)}")
    lines.append(f"eps = {_get('eps', 1e-8)}")

def _build_adapter_section(lines, cfg, _get):
    """Writes [adapter] section to lines list."""
    lines.append("[adapter]")
    lines.append(f"type = {_quote(_get('adapter', 'lora'))}")
    lines.append(f"rank = {_get('a_rank', 32)}")
    lines.append(f"dtype = {_quote(_get('a_dtype', 'bfloat16'))}")

def _build_monitoring_section(lines, monitor_cfg):
    """Writes [monitoring] section to lines list."""
    try:
        lines.append("")
        lines.append("[monitoring]")
        wandb_keys = ("enable_wandb", "wandb_api_key", "wandb_tracker_name", "wandb_run_name")
        
        mon_src = {}
        try:
            mon_src.update(monitor_cfg or {})
        except Exception:
            pass
        try:
            mon_src.update({k: v for k, v in (monitor_cfg or {}).items() if k in wandb_keys})
        except Exception:
            pass
        
        en_wandb = mon_src.get('enable_wandb', False)
        lines.append(f"enable_wandb = {'true' if (str(en_wandb).lower() in ['1','true','yes','on']) else 'false'}")
        lines.append(f"wandb_api_key = {_quote(mon_src.get('wandb_api_key', ''))}")
        lines.append(f"wandb_tracker_name = {_quote(mon_src.get('wandb_tracker_name', ''))}")
        lines.append(f"wandb_run_name = {_quote(mon_src.get('wandb_run_name', ''))}")
    except Exception:
        pass

# =============================================================================
# Main TOML Builder
# =============================================================================

def build_toml_config_from_ui(training_tab_container) -> str:
    """Builds TOML text from current UI controls matching the required schema."""
    
    # 1. Extract Configuration
    cfg = extract_config_from_controls(training_tab_container.config_page_content)
    monitor_cfg = _get_monitor_config(training_tab_container)
    dataset_path_val = _get_dataset_path_value(
        getattr(training_tab_container, 'dataset_page_content', None)
    )
    
    # 2. Resolve Paths
    from flet_app.project_root import get_project_root as _proj
    project_root = _proj()
    
    # Resolve Output Directory
    raw_output_dir = ""
    try:
        bb_field = getattr(training_tab_container, 'output_dir_field', None)
        if bb_field is not None and isinstance(bb_field, ft.TextField) and getattr(bb_field, 'value', None):
            raw_output_dir = str(bb_field.value or '').strip()
    except Exception:
        pass
    
    if not raw_output_dir:
        raw_output_dir = str(cfg.get('output_dir', 'workspace/output/dir') or '').strip()
    
    raw_output_dir = _normalize_slashes(raw_output_dir)
    
    if _is_absolute_path(raw_output_dir):
        resolved_output_dir = os.path.normpath(raw_output_dir)
    else:
        resolved_output_dir = os.path.normpath(os.path.join(str(project_root), *raw_output_dir.split('/')))
    
    resolved_output_dir = _normalize_slashes(resolved_output_dir)

    # Resolve Dataset TOML Path
    lines = []
    lines.append("# Output path for training runs. Each training run makes a new directory in here.")
    lines.append(f"output_dir = {_quote(resolved_output_dir)}")
    lines.append("")
    lines.append("# Dataset config file. This will be created next.")
    
    if dataset_path_val:
        try:
            base_dir, _dtype = _get_dataset_base_dir(os.path.basename(dataset_path_val))
        except Exception:
            base_dir = os.path.dirname(dataset_path_val)
        ds_name = os.path.basename(dataset_path_val.rstrip('/'))
        dataset_toml_abs = os.path.join(base_dir, f"{ds_name}.toml")
        dataset_path_to_write = _normalize_slashes(dataset_toml_abs)
    else:
        dataset_path_to_write = dataset_path_val
    
    lines.append(f"dataset = {_quote(dataset_path_to_write)}")
    lines.append("")

    # 3. Build Sections
    def _get(name, default=None):
        return cfg.get(name, default)

    def _as_bool(val):
        return _to_bool(val)

    _build_training_section(lines, cfg, _get)
    _build_eval_section(lines, cfg, _get, _as_bool)
    _build_misc_section(lines, cfg, _get)
    
    # Block swap section
    lines.append("# BLOCK SWAP (requires pipeline_stages=1)")
    blocks_swap_val = _get('blocks_swap', 0)
    try:
        blocks_swap_val = int(blocks_swap_val)
    except Exception:
        blocks_swap_val = 0
    lines.append(f"blocks_to_swap = {blocks_swap_val}")
    disable_bsfe_val = _get('disable_bsfe', 'true')
    lines.append(f"disable_block_swap_for_eval = {'true' if _as_bool(disable_bsfe_val) else 'false'}")
    lines.append("")

    _build_model_section(lines, cfg, _get, expand_model_path)
    _build_optimizer_section(lines, cfg, _get)
    _build_adapter_section(lines, cfg, _get)
    _build_monitoring_section(lines, monitor_cfg)

    return "\n".join(lines) + "\n"

# =============================================================================
# UI Updater Helpers
# =============================================================================

def _collapse_path_to_relative(path: str) -> str:
    """Collapse absolute path to relative path for UI display."""
    try:
        if isinstance(path, str):
            from flet_app.project_root import get_project_root as _proj
            proj_root = _proj()
            od = path.replace('\\', '/')
            proj_root_str = str(proj_root).replace('\\', '/').rstrip('/')
            if od.lower().startswith(proj_root_str.lower() + '/'):
                rel = od[len(proj_root_str.rstrip('/'))+1:]
                return rel
    except Exception:
        pass
    return path

def _populate_model_section(toml_data, label_vals, _to_bool_func):
    """Populate label_vals from the [model] section."""
    model = toml_data.get('model', {}) or {}
    if isinstance(model, dict):
        normalized_type = populate_label_vals_from_model(model, label_vals)
        if normalized_type:
            label_vals['Model Type'] = normalized_type
            is_ltx2 = str(normalized_type).lower() in ('ltx-video-2', 'ltx2')
            
            if 'dtype' in model:
                label_vals['dtype'] = model.get('dtype')
            if 'transformer_dtype' in model:
                label_vals['transformer_dtype'] = model.get('transformer_dtype')
            if 'timestep_sample_method' in model:
                label_vals['timestep_sm'] = model.get('timestep_sample_method')

            if 'model_path' in model:
                label_vals['model_path'] = collapse_model_path(model.get('model_path', ''))
            if 'text_encoder_path' in model:
                label_vals['text_encoder_path'] = collapse_model_path(model.get('text_encoder_path', ''))
            if 'load_checkpoint' in model:
                label_vals['load_checkpoint'] = model.get('load_checkpoint', '')

        if is_ltx2:
            label_vals.pop('first_frame_conditioning_p', None)

def _populate_optimizer_section(toml_data, label_vals):
    """Populate label_vals from the [optimizer] section."""
    opt = toml_data.get('optimizer', {}) or {}
    if isinstance(opt, dict):
        if 'type' in opt:
            label_vals['optimizer_type'] = opt.get('type')
        for k in ('lr', 'betas', 'weight_decay', 'eps'):
            if k in opt:
                label_vals[k] = opt.get(k)

def _populate_adapter_section(toml_data, label_vals):
    """Populate label_vals from the [adapter] section."""
    ad = toml_data.get('adapter', {}) or {}
    if isinstance(ad, dict):
        if 'type' in ad:
            label_vals['adapter'] = ad.get('type')
        if 'rank' in ad:
            label_vals['a_rank'] = ad.get('rank')
        if 'dtype' in ad:
            label_vals['a_dtype'] = ad.get('dtype')

        if 'lora_targets' in ad:
            lora_targets = ad['lora_targets']
            if isinstance(lora_targets, dict):
                for key in ['all_modules', 'video_attn', 'video_ff', 'audio_attn', 'audio_ff', 'cross_modal_attn']:
                    if key in lora_targets:
                        label_vals[key] = lora_targets[key]

def _populate_lora_section(toml_data, label_vals):
    """Populate label_vals from the [lora] section."""
    lora = toml_data.get('lora', {}) or {}
    if isinstance(lora, dict):
        if 'rank' in lora:
            label_vals['rank'] = lora.get('rank')
        if 'alpha' in lora:
            label_vals['alpha'] = lora.get('alpha')
        if 'dropout' in lora:
            label_vals['dropout'] = lora.get('dropout')
        
        if 'target_modules' in lora:
            target_modules = lora['target_modules']
            if isinstance(target_modules, dict):
                for key in ['all_modules', 'video_attn', 'video_ff', 'audio_attn', 'audio_ff', 'cross_modal_attn', 'with_audio']:
                    if key in target_modules:
                        label_vals[key] = target_modules[key]

def _populate_training_strategy_section(toml_data, label_vals, _to_bool_func):
    """Populate label_vals from the [training_strategy] section."""
    training_strategy = toml_data.get('training_strategy', {}) or {}
    if isinstance(training_strategy, dict):
        if 'first_frame_conditioning_p' in training_strategy:
            label_vals['first_frame_conditioning_p'] = training_strategy.get('first_frame_conditioning_p')
        if 'with_audio' in training_strategy:
            label_vals['with_audio'] = _to_bool(training_strategy.get('with_audio', False))

def _populate_optimization_section(toml_data, label_vals):
    """Populate label_vals from the [optimization] section."""
    optimization = toml_data.get('optimization', {}) or {}
    if isinstance(optimization, dict):
        for k in ('learning_rate', 'steps', 'batch_size', 'gradient_accumulation_steps', 'max_grad_norm'):
            if k in optimization:
                label_vals[k] = optimization.get(k)
        if 'optimizer_type' in optimization:
            label_vals['optimizer_type'] = optimization.get('optimizer_type')
        if 'scheduler_type' in optimization:
            label_vals['scheduler_type'] = optimization.get('scheduler_type')

def _populate_acceleration_section(toml_data, label_vals, _to_bool_func):
    """Populate label_vals from the [acceleration] section."""
    acceleration = toml_data.get('acceleration', {}) or {}
    if isinstance(acceleration, dict):
        if 'mixed_precision_mode' in acceleration:
            label_vals['mixed_precision_mode'] = acceleration.get('mixed_precision_mode')
        if 'quantization' in acceleration:
            label_vals['quantization'] = acceleration.get('quantization')
        if 'load_text_encoder_in_8bit' in acceleration:
            label_vals['load_text_encoder_in_8bit'] = _to_bool(acceleration.get('load_text_encoder_in_8bit', False))

def _populate_flow_matching_section(toml_data, label_vals):
    """Populate label_vals from the [flow_matching] section."""
    flow_matching = toml_data.get('flow_matching', {}) or {}
    if isinstance(flow_matching, dict):
        if 'timestep_sampling_mode' in flow_matching:
            ts_mode = flow_matching.get('timestep_sampling_mode')
            if str(ts_mode).lower() == 'shifted_logit_normal':
                ts_mode = 'logit_normal'
            label_vals['timestep_sm'] = ts_mode

def _populate_checkpoints_section(toml_data, label_vals):
    """Populate label_vals from the [checkpoints] section."""
    checkpoints = toml_data.get('checkpoints', {}) or {}
    if isinstance(checkpoints, dict):
        if 'interval' in checkpoints:
            label_vals['interval'] = checkpoints.get('interval')
        if 'keep_last_n' in checkpoints:
            label_vals['keep_last_n'] = checkpoints.get('keep_last_n')
        if 'precision' in checkpoints:
            label_vals['precision'] = checkpoints.get('precision')

def _populate_monitoring_section(toml_data, label_vals, _to_bool_func):
    """Populate label_vals from the [monitoring] section."""
    mon = toml_data.get('monitoring', {}) or {}
    if isinstance(mon, dict):
        if 'enable_wandb' in mon:
            label_vals['enable_wandb'] = _to_bool(mon.get('enable_wandb'))
        for k in ('wandb_api_key', 'wandb_tracker_name', 'wandb_run_name'):
            if k in mon:
                label_vals[k] = mon.get(k)

def _apply_values_recursive(control, label_vals, page):
    """Recursively apply values from label_vals to Flet controls."""
    try:
        if hasattr(control, 'controls') and control.controls:
            for c in control.controls:
                _apply_values_recursive(c, label_vals, page)
        elif hasattr(control, 'content') and control.content:
            _apply_values_recursive(control.content, label_vals, page)
        
        label = getattr(control, 'label', None)
        if not label:
            return
        
        if label in label_vals:
            val = label_vals[label]
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
                # Fire change handler for Model Type
                try:
                    if str(label).strip() == 'Model Type' and callable(getattr(control, 'on_change', None)):
                        class _E: pass
                        e = _E()
                        setattr(e, 'control', control)
                        setattr(e, 'page', page)
                        control.on_change(e)
                except Exception:
                    pass
            elif isinstance(control, ft.Checkbox):
                control.value = _to_bool(val)
                if control.page:
                    control.update()
    except Exception:
        pass

def _apply_all_values(training_tab_container, label_vals, page):
    """Apply values to main config, dataset, and monitor containers."""
    try:
        _apply_values_recursive(training_tab_container.config_page_content, label_vals, page)
    except Exception:
        pass
    
    try:
        dataset_content = getattr(training_tab_container, 'dataset_page_content', None)
        if dataset_content is not None:
            _apply_values_recursive(dataset_content, label_vals, page)
    except Exception:
        pass
    
    try:
        monitor_content = getattr(training_tab_container, 'monitor_page_content', None)
        if monitor_content is not None:
            _apply_values_recursive(monitor_content, label_vals, page)
    except Exception:
        pass

def _handle_dataset_selection(toml_data, training_tab_container, page):
    """Handle dataset selection based on TOML data."""
    try:
        dataset_path = toml_data.get('dataset')
        if not dataset_path:
            data_section = toml_data.get('data', {})
            if isinstance(data_section, dict):
                dataset_path = data_section.get('preprocessed_data_root')

        if dataset_path:
            clean_name = os.path.basename(str(dataset_path).replace('\\', '/'))
            if clean_name.lower().endswith('.toml'):
                clean_name = clean_name[:-5]
            
            ds_block = getattr(training_tab_container, 'dataset_page_content', None)
            if ds_block and hasattr(ds_block, 'set_selected_dataset'):
                ds_block.set_selected_dataset(clean_name, page_ctx=page)
            
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


# =============================================================================
# Main UI Updater
# =============================================================================

def update_ui_from_toml(training_tab_container, toml_data: dict):
    """Populate UI controls from TOML data produced by our builder schema."""
    page = getattr(training_tab_container, 'page', None)
    
    def _to_bool(v):
        return _to_bool(v)

    # Build label -> value map
    label_vals = {}
    top_keys = [
        'output_dir', 'epochs', 'micro_batch_size_per_gpu', 'pipeline_stages',
        'gradient_accumulation_steps', 'gradient_clipping', 'warmup_steps',
        'activation_checkpointing', 'eval_every_n_epochs', 'eval_micro_batch_size_per_gpu',
        'eval_gradient_accumulation_steps', 'save_every_n_epochs', 'checkpoint_every_n_minutes',
        'partition_method', 'save_dtype', 'caching_batch_size', 'steps_per_print', 'video_clip_mode'
    ]
    for k in top_keys:
        if k in toml_data:
            label_vals[k] = toml_data.get(k)

    # Post-process output_dir for UI: strip project root to show relative workspace path
    if 'output_dir' in label_vals and isinstance(label_vals['output_dir'], str):
        label_vals['output_dir'] = _collapse_path_to_relative(label_vals['output_dir'])
    
    if 'eval_before_first_step' in toml_data:
        label_vals['eval_before_first_step'] = _to_bool(toml_data.get('eval_before_first_step'))

    # Block swap
    if 'blocks_to_swap' in toml_data:
        label_vals['blocks_swap'] = toml_data.get('blocks_to_swap')
    if 'disable_block_swap_for_eval' in toml_data:
        label_vals['disable_bsfe'] = 'true' if _to_bool(toml_data.get('disable_block_swap_for_eval')) else 'false'

    # Populate Sections
    _populate_model_section(toml_data, label_vals, _to_bool)
    _populate_optimizer_section(toml_data, label_vals)
    _populate_adapter_section(toml_data, label_vals)
    _populate_lora_section(toml_data, label_vals)
    _populate_training_strategy_section(toml_data, label_vals, _to_bool)
    _populate_optimization_section(toml_data, label_vals)
    _populate_acceleration_section(toml_data, label_vals, _to_bool)
    _populate_flow_matching_section(toml_data, label_vals)
    _populate_checkpoints_section(toml_data, label_vals)
    _populate_monitoring_section(toml_data, label_vals, _to_bool)

    # Apply Values
    suppress_defaults_ctx = None
    try:
        from flet_app.ui.pages import training_config as _training_config_module
        suppress_defaults_ctx = getattr(_training_config_module, 'suppress_model_defaults', None)
    except Exception:
        pass

    if suppress_defaults_ctx:
        try:
            with suppress_defaults_ctx():
                _apply_all_values(training_tab_container, label_vals, page)
        except Exception:
            _apply_all_values(training_tab_container, label_vals, page)
    else:
        _apply_all_values(training_tab_container, label_vals, page)

    # Bottom bar output_dir
    try:
        if 'output_dir' in label_vals:
            bb_field = getattr(training_tab_container, 'output_dir_field', None)
            if bb_field is not None and isinstance(bb_field, ft.TextField):
                bb_field.value = str(label_vals['output_dir']) if label_vals['output_dir'] is not None else ''
                if bb_field.page:
                    bb_field.update()
    except Exception:
        pass
    
    # Data config panel
    try:
        data_config_content = getattr(training_tab_container, 'data_config_page_content', None)
        if data_config_content is not None and hasattr(data_config_content, 'dataset_block'):
            _apply_values_recursive(data_config_content.dataset_block, label_vals, page)
    except Exception:
        pass

    # Dataset selection
    _handle_dataset_selection(toml_data, training_tab_container, page)

    # LTX2-specific validation fields loading
    try:
        model = toml_data.get('model', {})
        model_type = model.get('type', '')
        if model_type and 'ltx-video-2' in str(model_type).lower():
            from .ltx2_config_utils import update_ltx2_ui_from_toml
            update_ltx2_ui_from_toml(training_tab_container, toml_data)
    except Exception as e:
        logger.debug(f"Could not load LTX2 validation fields: {e}")

    # Post-processing visibility
    try:
        from flet_app.ui.pages.training_config import model_type_dropdown_ref
        postprocess_visibility_after_apply(label_vals, page, model_type_dropdown_ref)
    except Exception:
        pass

    if page:
        try:
            page.update()
        except Exception:
            pass

# =============================================================================
# Image Processing Utilities
# =============================================================================

def _process_and_save_image(source_image_path, video_dims_tuple, dataset_name, target_filename, dataset_type="video"):
    # Unified datasets: always use 'datasets'
    base_datasets_dir = "datasets"
    dataset_sample_images_dir = Path("workspace") / base_datasets_dir / dataset_name / "sample_images"
    dataset_sample_images_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_sample_images_dir / target_filename

    try:
        from PIL import Image
        img = Image.open(source_image_path)
    except FileNotFoundError:
        logger.error(f"Error: Source image not found at {source_image_path} in _process_and_save_image.")
        return None
    except Exception as e:
        logger.error(f"Error opening image {source_image_path} in _process_and_save_image: {e}")
        return None

    original_width, original_height = img.size
    target_width, target_height = video_dims_tuple[0], video_dims_tuple[1]

    # Validate dimensions
    if original_width == 0 or original_height == 0 or target_width == 0 or target_height == 0:
        logger.warning(f"Invalid dimensions for scaling. Original: {img.size}, Target: {(target_width, target_height)}. Saving original.")
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(target_path)
        return str(target_path).replace('\\', '/')

    # Calculate scaling
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_factor = max(width_ratio, height_ratio)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    if new_width <= 0 or new_height <= 0:
        logger.warning(f"Calculated new dimensions are invalid ({new_width}x{new_height}). Saving original.")
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(target_path)
        return str(target_path).replace('\\', '/')

    try:
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        img_cropped = img_resized.crop((left, top, right, bottom))

        if img_cropped.mode == 'RGBA':
            img_cropped = img_cropped.convert('RGB')

        img_cropped.save(target_path)
        result_path = str(target_path).replace('\\', '/')
        logger.info(f"Image saved and scaled to: {result_path}")
        return result_path
    except Exception as e:
        logger.error(f"Error resizing or saving image {source_image_path} to {target_path}: {e}")
        logger.error(traceback.format_exc())
        return None

def _save_and_scale_image(source_image_path: str, video_dims_tuple: tuple, dataset_name: str,
                        target_filename: str, dataset_type: str = "video",
                        page: ft.Page = None, target_control: str = None,
                        image_display_c1=None, image_display_c2=None):

    if not source_image_path or not dataset_name:
        logger.warning("_save_and_scale_image: Missing source_image_path or dataset_name.")
        return None

    result_path = _process_and_save_image(source_image_path, video_dims_tuple, dataset_name, target_filename, dataset_type)
    
    if result_path and page is not None and target_control and target_control.lower() in ['c1', 'c2']:
        from .utils_top_menu import TopBarUtils
        logger.debug(f"Calling _load_cropped_image_into_ui for {target_control}")
        TopBarUtils._load_cropped_image_into_ui(
            page=page,
            image_path=result_path,
            target_control_key=target_control,
            image_display_c1=image_display_c1,
            image_display_c2=image_display_c2
        )
    
    return result_path