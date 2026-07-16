"""
TOML configuration builder for generating TOML from UI controls.

Handles extracting values from Flet controls and building TOML output.
"""

import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import flet as ft

from .constants import ALWAYS_INCLUDE_FIELDS, FIELD_TO_TOML_MAPPINGS
from .toml_formatting import quote, to_bool, toml_bool, normalize_slashes, is_absolute_path, expand_model_path

# Import for musubi optimizer type conversion
try:
    from .config.musubi.optimizer import get_musubi_optimizer_type_for_toml
except ImportError:
    # Fallback if musubi module not available
    def get_musubi_optimizer_type_for_toml(ui_value: str) -> str:
        return str(ui_value).strip().lower()


# =============================================================================
# Config Extraction
# =============================================================================

def extract_config_from_controls(control: Any) -> Dict:
    """Recursively extract values from Flet controls into a dictionary."""
    result = {}

    def _extract(child):
        if hasattr(child, 'controls') and child.controls:
            for sub_child in child.controls:
                _extract(sub_child)
        elif hasattr(child, 'content') and child.content:
            _extract(child.content)
        elif isinstance(child, ft.TextField):
            label = getattr(child, 'label', None)
            if label in ALWAYS_INCLUDE_FIELDS:
                result[label] = child.value
            elif getattr(child, 'visible', True):
                result[label] = child.value
        elif isinstance(child, ft.Dropdown):
            label = getattr(child, 'label', None)
            result_key = FIELD_TO_TOML_MAPPINGS.get(label, label)
            current_visible = getattr(child, 'visible', True)
            # Extract the key from the dropdown Option object or use the string value directly
            # (dropdown.value can be an Option object when set from UI, or a string when loaded from TOML)
            if child.value is None:
                dropdown_value = None
            elif hasattr(child.value, 'key'):
                dropdown_value = child.value.key  # Option object (fresh UI state)
            else:
                dropdown_value = child.value  # Already a string (loaded from TOML)
            if label in ALWAYS_INCLUDE_FIELDS:
                result[result_key] = dropdown_value
            elif current_visible:
                result[result_key] = dropdown_value
        elif isinstance(child, ft.Checkbox):
            if getattr(child, 'visible', True):
                key = getattr(child, 'data', None) or child.label
                if child.label in ['8_bit_te']:
                    key = child.label
                result[key] = child.value

    if control:
        _extract(control)
    return result


# =============================================================================
# Section Builders
# =============================================================================

def build_training_section(lines: List[str], cfg: Dict, _get: callable) -> None:
    """Write training settings to lines list."""
    lines.append("# training settings")
    for key in ['epochs', 'micro_batch_size_per_gpu', 'pipeline_stages',
                'gradient_accumulation_steps', 'gradient_clipping', 'warmup_steps']:
        val = _get(key, None)
        if val is not None:
            lines.append(f"{key} = {val}")

    lr_sched = _get('lr_scheduler', 'constant')
    if lr_sched is not None and str(lr_sched).strip() != '':
        lines.append(f"lr_scheduler = {quote(lr_sched)}")

    act_ckpt = _get('activation_checkpointing', 'unsloth')
    act_ckpt_str = str(act_ckpt).lower() if act_ckpt is not None else ''
    if act_ckpt_str in ['off', 'false'] or act_ckpt is False:
        lines.append("activation_checkpointing = false")
    elif act_ckpt_str in ['on', 'true'] or act_ckpt is True:
        lines.append("activation_checkpointing = true")
    else:
        lines.append(f"activation_checkpointing = {quote(act_ckpt)}")


def build_eval_section(lines: List[str], cfg: Dict, _get: callable) -> None:
    """Write eval settings to lines list."""
    lines.append("# eval settings")
    lines.append(f"eval_every_n_epochs = {_get('eval_every_n_epochs', 1)}")
    lines.append(f"eval_before_first_step = {toml_bool(_get('eval_before_first_step', True))}")
    lines.append(f"eval_micro_batch_size_per_gpu = {_get('eval_micro_batch_size_per_gpu', 1)}")
    lines.append(f"eval_gradient_accumulation_steps = {_get('eval_gradient_accumulation_steps', 1)}")


def build_misc_section(lines: List[str], cfg: Dict, _get: callable) -> None:
    """Write misc settings to lines list."""
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
            lines.append(f"{key} = {quote(val)}")
        else:
            lines.append(f"{key} = {val}")

    # 8-bit text encoder
    te_8bit = _get('8_bit_te', False)
    if te_8bit is not None:
        lines.append(f"8_bit_te = {toml_bool(te_8bit)}")


def build_model_section(lines: List[str], cfg: Dict, _get: callable) -> None:
    """Write [model] section to lines list."""
    from flet_app.ui.utils.model_settings import append_model_specific_lines

    lines.append("[model]")
    model_source = _get('Model Type', '')
    lines.append(f"type = {quote(model_source)}")

    trainer = _get('Trainer', 'diffusion-pipe')
    if trainer:
        lines.append(f"trainer = {quote(trainer)}")

    mt_lower = model_source.strip().lower()

    # SDXL: checkpoint_path
    if mt_lower == 'sdxl':
        ckpt = _get('checkpoint_path', None)
        if ckpt is not None and str(ckpt).strip() != '':
            lines.append(f"checkpoint_path = {quote(ckpt)}")

    # Base model paths (skip for SDXL and LTX)
    skip_path_models = ('sdxl', 'ltx-video', 'ltx', 'ltx-video-2')
    if mt_lower not in skip_path_models:
        diff_path = _get('diffusers_path', None)
        if diff_path and str(diff_path).strip():
            lines.append(f"diffusers_path  = {quote(expand_model_path(diff_path))}")
        transf_path = _get('transformer_path', None)
        if transf_path and str(transf_path).strip():
            lines.append(f"transformer_path   = {quote(expand_model_path(transf_path))}")
        if mt_lower == '_wan22':
            ckpt_path = _get('ckpt_path', None)
            if ckpt_path and str(ckpt_path).strip():
                lines.append(f"ckpt_path = {quote(ckpt_path)}")

    lines.append("")
    lines.append(f"dtype = {quote(_get('dtype', 'bfloat16'))}")

    # transformer_dtype (flux2/klein/krea2 use diffusion_model_dtype instead)
    t_dtype = str(_get('transformer_dtype', 'float8'))
    _diffusion_dtype_models = ('flux2', 'flux2_klein_4b', 'flux2_klein_9b', 'krea2')
    dtype_field = 'diffusion_model_dtype' if mt_lower in _diffusion_dtype_models else 'transformer_dtype'
    if t_dtype.strip().lower() == 'none':
        lines.append(f"#{dtype_field} = 'float8'")
    else:
        lines.append(f"{dtype_field} = {quote(t_dtype)}")

    # timestep_sample_method
    is_musubi_model = mt_lower in ('_wan22', 'ltx-video-2', 'ltx2') and trainer == 'musubi'
    if is_musubi_model:
        _tsm = _get('timestep_sm_m', None)
        if not _tsm or not str(_tsm).strip():
            _tsm = _get('timestep_sample_method', None)
            if _tsm == 'logit_normal':
                _tsm = 'shifted_logit_normal'
        if not _tsm or not str(_tsm).strip():
            _tsm = _get('timestep_sm', 'shifted_logit_normal')
    else:
        _tsm = _get('timestep_sample_method', None)
        if _tsm == 'shifted_logit_normal':
            _tsm = 'logit_normal'
        if not _tsm or not str(_tsm).strip():
            _tsm = _get('timestep_sm', 'logit_normal')

    if str(_tsm).strip().lower() == 'none':
        default = 'shifted_logit_normal' if is_musubi_model else 'logit_normal'
        lines.append(f"#timestep_sample_method = '{default}'")
    else:
        lines.append(f"timestep_sample_method = {quote(_tsm)}")

    # Musubi model section extras
    try:
        from flet_app.ui.utils.config_utils_musubi import append_musubi_model_section
        model_dict = {'type': model_source}
        append_musubi_model_section(lines, model_dict, _get)
    except ImportError:
        pass

    append_model_specific_lines(lines, _get, model_source)


def build_optimizer_section(lines: List[str], cfg: Dict, _get: callable) -> None:
    """Write [optimizer] or [optimization] section to lines list."""
    lines.append("")

    model_type = str(cfg.get('Model Type', '')).lower() if cfg.get('Model Type') else ''
    trainer = str(cfg.get('Trainer', '')).lower() if cfg.get('Trainer') else ''
    is_musubi = trainer == 'musubi'

    if is_musubi:
        lines.append("[optimization]")
        lines.append(f"learning_rate = {_get('learning_rate', _get('lr', 0.0001))}")
        lines.append(f"max_steps = {_get('max_steps', 2000)}")
        lines.append(f"batch_size = {_get('batch_size', 1)}")
        lines.append(f"gradient_accumulation_steps = {_get('grad_accum_steps', _get('gradient_accumulation_steps', 1))}")
        lines.append(f"max_grad_norm = {_get('max_grad_norm', 1.0)}")
        lines.append(f"loraplus_ratio = {_get('loraplus_ratio', 0.0)}")
        lines.append(f"blocks_to_swap = {_get('blocks_to_swap', 0)}")
        caption_dropout_val = _get('caption_dropout_rate', _get('caption_dropout', 0.0))
        if caption_dropout_val and float(caption_dropout_val) > 0:
            lines.append(f"caption_dropout_rate = {caption_dropout_val}")
        gc_val = _get('enable_gradient_checkpointing', _get('activation_checkpointing', 'false'))
        if isinstance(gc_val, bool):
            gc_bool = gc_val
        else:
            gc_str = str(gc_val).lower()
            gc_bool = gc_str in ['on', 'true', 'unsloth', '1', 'yes']
        lines.append(f"enable_gradient_checkpointing = {toml_bool(gc_bool)}")
    else:
        lines.append("[optimizer]")

    # Use optimizer_type_m only for musubi trainer, otherwise use optimizer_type
    if is_musubi:
        opt_type = _get('optimizer_type_m', 'AdamW')
    else:
        opt_type = _get('optimizer_type', 'adamw_optimi')
    opt_type_lower = str(opt_type).lower()

    if is_musubi:
        # Convert UI value (e.g., "Automagic") to TOML format (e.g., "automagic")
        opt_type_toml = get_musubi_optimizer_type_for_toml(opt_type)
        lines.append(f"optimizer_type = {quote(opt_type_toml)}")
    else:
        lines.append(f"type = {quote(opt_type)}")

    if is_musubi:
        lines.append(f"scheduler_type = {quote(_get('scheduler_type', 'constant'))}")

    is_automagic = 'automagic' in opt_type_lower
    is_prodigy = 'prodigy' in opt_type_lower

    # Non-musubi, non-automagic: write standard fields
    if not is_musubi and not is_automagic:
        lines.append(f"lr = {_get('lr', 2e-5)}")
        lines.append(f"betas = {_get('betas', '[0.9, 0.99]')}")
        lines.append(f"weight_decay = {_get('weight_decay', 0.01)}")
        lines.append(f"eps = {_get('eps', 1e-8)}")

    # Prodigy fields
    if is_prodigy:
        beta3_val = _get('beta3', None)
        if beta3_val is not None and str(beta3_val).strip() not in ('', 'None'):
            lines.append(f"beta3 = {beta3_val}")
        lines.append(f"d0 = {_get('d0', 1e-6)}")
        lines.append(f"d_coef = {_get('d_coef', 1.0)}")
        schedulefree_c_val = _get('schedulefree_c', 0.0)
        if schedulefree_c_val and float(schedulefree_c_val) != 0.0:
            lines.append(f"schedulefree_c = {schedulefree_c_val}")

    # Automagic fields (diffusion-pipe: also write lr for starting lr)
    if is_automagic:
        if not is_musubi:
            # Write lr as starting lr for automagic
            lr_val = _get('lr', 1e-6)
            lines.append(f"lr = {lr_val}")
        if is_musubi:
            try:
                from flet_app.ui.utils.config_utils_musubi import build_musubi_optimizer_args_line
                optimizer_args_line = build_musubi_optimizer_args_line(opt_type_lower, _get('optimizer_args', None))
                if optimizer_args_line:
                    lines.append(optimizer_args_line)
            except ImportError:
                _write_automagic_fields(lines, _get)
        else:
            _write_automagic_fields(lines, _get)


def _write_automagic_fields(lines: List[str], _get: callable) -> None:
    """Write automagic optimizer fields."""
    lines.append(f"min_lr = {_get('min_lr', 1e-7)}")
    lines.append(f"max_lr = {_get('max_lr', 1e-3)}")
    lines.append(f"lr_bump = {_get('lr_bump', 1e-6)}")
    lines.append(f"clip_threshold = {_get('clip_threshold', 1.0)}")
    lines.append(f"do_paramiter_swapping = {toml_bool(_get('do_paramiter_swapping', False))}")
    lines.append(f"paramiter_swapping_factor = {_get('paramiter_swapping_factor', 0.1)}")
    lines.append("eps = [1e-30, 1e-3]")


def build_adapter_section(lines: List[str], cfg: Dict, _get: callable) -> None:
    """Write [adapter] section to lines list."""
    lines.append("")
    lines.append("[adapter]")
    lines.append(f"type = {quote(_get('adapter', 'lora'))}")
    lines.append(f"rank = {_get('a_rank', 32)}")
    lines.append(f"dtype = {quote(_get('a_dtype', 'bfloat16'))}")
    init_from_existing = _get('init_from_existing', '')
    if init_from_existing and str(init_from_existing).strip():
        lines.append(f"init_from_existing = {quote(str(init_from_existing).strip())}")


def build_monitoring_section(lines: List[str], monitor_cfg: Dict) -> None:
    """Write [monitoring] section to lines list."""
    try:
        lines.append("")
        lines.append("[monitoring]")
        mon = monitor_cfg or {}
        en_wandb = mon.get('enable_wandb', False)
        lines.append(f"enable_wandb = {toml_bool(en_wandb)}")
        lines.append(f"wandb_api_key = {quote(mon.get('wandb_api_key', ''))}")
        lines.append(f"wandb_tracker_name = {quote(mon.get('wandb_tracker_name', ''))}")
        lines.append(f"wandb_run_name = {quote(mon.get('wandb_run_name', ''))}")
    except Exception:
        pass


# =============================================================================
# Helper Functions
# =============================================================================

def _get_dataset_path_value(dataset_block: Any) -> str:
    """Extract dataset path value from a UI dataset block."""
    if not dataset_block or not hasattr(dataset_block, 'get_selected_dataset'):
        return ""

    selected_clean = dataset_block.get_selected_dataset()
    if not selected_clean:
        return ""

    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
    base_dir, _dtype = _get_dataset_base_dir(selected_clean)
    ds_full = os.path.join(base_dir, selected_clean)
    return normalize_slashes(ds_full)


def _get_monitor_config(monitor_container: Any) -> Dict:
    """Extract configuration from the monitor page container."""
    cfg = {}
    try:
        monitor_content = getattr(monitor_container, 'monitor_page_content', None)
        if monitor_content:
            cfg = extract_config_from_controls(monitor_content)
    except Exception:
        pass
    return cfg


def _resolve_output_dir(raw_output_dir: str, project_root: Path) -> str:
    """Resolve output directory to absolute path."""
    from .toml_formatting import resolve_output_dir
    return resolve_output_dir(raw_output_dir, project_root)


# =============================================================================
# Main Builder
# =============================================================================

def build_toml_config_from_ui(container: Any) -> str:
    """Build TOML text from current UI controls."""
    from flet_app.project_root import get_project_root

    # Extract configuration
    cfg = extract_config_from_controls(container.config_page_content)
    monitor_cfg = _get_monitor_config(container)
    dataset_path_val = _get_dataset_path_value(
        getattr(container, 'dataset_page_content', None)
    )

    # Resolve paths
    project_root = get_project_root()

    # Resolve output directory
    raw_output_dir = ""
    try:
        bb_field = getattr(container, 'output_dir_field', None)
        if isinstance(bb_field, ft.TextField) and getattr(bb_field, 'value', None):
            raw_output_dir = str(bb_field.value or '').strip()
    except Exception:
        pass

    if not raw_output_dir:
        raw_output_dir = str(cfg.get('output_dir', 'workspace/output/dir') or '').strip()

    resolved_output_dir = _resolve_output_dir(raw_output_dir, project_root)

    # Resolve init_from_existing
    init_from_existing_val = ""
    try:
        bb_field = getattr(container, 'init_from_existing_field', None)
        if isinstance(bb_field, ft.TextField) and getattr(bb_field, 'value', None):
            init_from_existing_val = str(bb_field.value or '').strip()
    except Exception:
        pass

    resolved_init_from_existing = ""
    if init_from_existing_val:
        init_from_existing_val = normalize_slashes(init_from_existing_val)
        if is_absolute_path(init_from_existing_val):
            resolved_init_from_existing = os.path.normpath(init_from_existing_val)
        else:
            resolved_init_from_existing = os.path.normpath(
                os.path.join(str(project_root), *init_from_existing_val.split('/'))
            )
        resolved_init_from_existing = normalize_slashes(resolved_init_from_existing)

    if resolved_init_from_existing:
        cfg['init_from_existing'] = resolved_init_from_existing

    # Build lines
    lines = []
    lines.append("# Output path for training runs. Each training run makes a new directory in here.")
    lines.append(f"output_dir = {quote(resolved_output_dir)}")
    lines.append("")
    lines.append("# Dataset config file. This will be created next.")

    # Dataset path
    if dataset_path_val:
        try:
            from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
            base_dir, _dtype = _get_dataset_base_dir(os.path.basename(dataset_path_val))
        except Exception:
            base_dir = os.path.dirname(dataset_path_val)
        ds_name = os.path.basename(dataset_path_val.rstrip('/'))
        dataset_toml_abs = os.path.join(base_dir, f"{ds_name}.toml")
        dataset_path_to_write = normalize_slashes(dataset_toml_abs)
    else:
        dataset_path_to_write = dataset_path_val

    lines.append(f"dataset = {quote(dataset_path_to_write)}")

    # dataset_list
    dataset_names = []
    config_page = getattr(container, 'config_page_content', None)
    if config_page:
        for ds_num in [1, 2, 3]:
            ds_block = getattr(config_page, f'dataset_{ds_num}_block', None)
            if ds_block and hasattr(ds_block, 'get_selected_dataset'):
                selected_name = ds_block.get_selected_dataset()
                if selected_name:
                    dataset_names.append(selected_name)

    if dataset_names:
        lines.append("")
        dataset_list_formatted = "[" + ", ".join(f'"{name}"' for name in dataset_names) + "]"
        lines.append(f"dataset_list = {dataset_list_formatted}")

    # Helper
    def _get(name, default=None):
        return cfg.get(name, default)

    # Extra flags
    extra_flags_val = _get('extra_flags', '')
    if extra_flags_val and str(extra_flags_val).strip():
        lines.append("")
        lines.append(f"extra_flags = {quote(str(extra_flags_val).strip())}")

    lines.append("")

    # Build sections
    build_training_section(lines, cfg, _get)
    build_eval_section(lines, cfg, _get)
    build_misc_section(lines, cfg, _get)

    # Block swap
    lines.append("# BLOCK SWAP (requires pipeline_stages=1)")
    blocks_swap_val = _get('blocks_swap', 0)
    try:
        blocks_swap_val = int(blocks_swap_val)
    except Exception:
        blocks_swap_val = 0
    lines.append(f"blocks_to_swap = {blocks_swap_val}")
    disable_bsfe_val = _get('disable_bsfe', 'true')
    lines.append(f"disable_block_swap_for_eval = {toml_bool(disable_bsfe_val)}")
    lines.append("")

    build_model_section(lines, cfg, _get)
    build_optimizer_section(lines, cfg, _get)
    build_adapter_section(lines, cfg, _get)
    build_monitoring_section(lines, monitor_cfg)

    # Musubi acceleration section
    model_type = str(cfg.get('Model Type', '')).lower() if cfg.get('Model Type') else ''
    trainer = str(cfg.get('Trainer', '')).lower() if cfg.get('Trainer') else ''
    is_musubi = trainer == 'musubi' or model_type in ('ltx-video-2', 'ltx2', 'wan22', 'wan', '_wan22')

    if is_musubi:
        try:
            from flet_app.ui.utils.config_utils_musubi import append_musubi_acceleration_section
            append_musubi_acceleration_section(lines, _get)
        except ImportError:
            pass

    return "\n".join(lines) + "\n"
