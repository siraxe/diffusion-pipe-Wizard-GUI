import os
import re
import traceback
from pathlib import Path
from flet_app.project_root import get_project_root as _get_project_root
from loguru import logger
import flet as ft
try:
    import tomllib as _toml_parser  # Python 3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as _toml_parser  # Fallback for older Python if available
    except Exception:
        _toml_parser = None

from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
from flet_app.ui.utils.model_settings import (
    append_model_specific_lines,
    populate_label_vals_from_model,
    postprocess_visibility_after_apply,
)


def get_project_root():  # type: ignore[no-redef]
    return _get_project_root()  # delegate to centralized helper


def expand_model_path(path: str) -> str:
    """Expand relative model path to absolute path when saving to TOML."""
    if not path or not isinstance(path, str):
        return path

    # Remove any quotes and clean up
    path = str(path).strip().strip("'\"")

    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')

    # If already absolute, return as-is
    if path.startswith('/') or (len(path) > 1 and path[1] == ':'):
        return path

    # Get project root and expand
    project_root = get_project_root()
    expanded_path = os.path.join(str(project_root), path)

    # Normalize and convert to forward slashes
    return os.path.normpath(expanded_path).replace('\\', '/')


def collapse_model_path(path: str) -> str:
    """Collapse absolute model path to relative path when loading from TOML."""
    if not path or not isinstance(path, str):
        return path

    # Remove any quotes and clean up
    path = str(path).strip().strip("'\"")

    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')

    # Get project root
    project_root = get_project_root()
    project_root_str = str(project_root).replace('\\', '/').rstrip('/')

    # If path is under project root, make it relative
    if path.lower().startswith(project_root_str.lower() + '/'):
        relative_path = path[len(project_root_str) + 1:]
        return relative_path

    # If not under project root, return as-is
    return path


def build_toml_config_from_ui(training_tab_container) -> str:
    """Builds TOML text from current UI controls matching the required schema."""
    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir

    # Reuse extractor to get flat map of labels -> values
    cfg = extract_config_from_controls(training_tab_container.config_page_content)

    # Monitor panel values (optional)
    monitor_cfg = {}
    try:
        monitor_content = getattr(training_tab_container, 'monitor_page_content', None)
        if monitor_content is not None:
            monitor_cfg = extract_config_from_controls(monitor_content)
    except Exception:
        monitor_cfg = {}

    # Resolve dataset folder (not dataset TOML path yet)
    dataset_block = getattr(training_tab_container, 'dataset_page_content', None)
    dataset_path_val = ""
    if dataset_block and hasattr(dataset_block, 'get_selected_dataset'):
        selected_clean = dataset_block.get_selected_dataset()
        if selected_clean:
            base_dir, _dtype = _get_dataset_base_dir(selected_clean)
            ds_full = os.path.join(base_dir, selected_clean)
            dataset_path_val = ds_full.replace('\\', '/')

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

    lines = []
    # Output + dataset
    lines.append("# Output path for training runs. Each training run makes a new directory in here.")

    # Expand output_dir to an absolute path when saving
    # Prefer bottom bar field if present
    try:
        bb_field = getattr(training_tab_container, 'output_dir_field', None)
        if bb_field is not None and isinstance(bb_field, ft.TextField) and getattr(bb_field, 'value', None):
            raw_output_dir = str(bb_field.value or '').strip()
        else:
            raw_output_dir = str(_get('output_dir', 'workspace/output/dir') or '').strip()
    except Exception:
        raw_output_dir = str(_get('output_dir', 'workspace/output/dir') or '').strip()
    raw_output_dir = raw_output_dir.replace('\\', '/')
    from flet_app.project_root import get_project_root as _proj
    project_root = _proj()

    def _is_abs(p: str) -> bool:
        if not p:
            return False
        # Windows drive or UNC, or POSIX root
        return bool(re.match(r"^[A-Za-z]:[\\/]|^/|^\\\\", p))

    if _is_abs(raw_output_dir):
        resolved_output_dir = os.path.normpath(raw_output_dir)
    else:
        # Join relative to project root (so 'workspace/...' becomes
        # '/abs/path/to/project/workspace/...')
        resolved_output_dir = os.path.normpath(os.path.join(str(project_root), *raw_output_dir.split('/')))

    # Write normalized with forward slashes
    lines.append(f"output_dir = {_quote(resolved_output_dir.replace('\\\\', '/').replace('\\', '/'))}")
    lines.append("")

    lines.append("# Dataset config file. This will be created next.")

    # Ensure dataset points to dataset TOML file, not just folder
    if dataset_path_val:
        # dataset_path_val currently points to the dataset folder under datasets/ or datasets_img/
        # Convert to absolute TOML path under the same base
        try:
            base_dir, _dtype = _get_dataset_base_dir(os.path.basename(dataset_path_val))
        except Exception:
            base_dir = os.path.dirname(dataset_path_val)
        ds_name = os.path.basename(dataset_path_val.rstrip('/'))
        dataset_toml_abs = os.path.join(base_dir, f"{ds_name}.toml")
        dataset_path_to_write = dataset_toml_abs.replace('\\', '/')
    else:
        dataset_path_to_write = dataset_path_val

    lines.append(f"dataset = {_quote(dataset_path_to_write)}")
    lines.append("")

    # training settings
    lines.append("# training settings")
    for key in [
        'epochs', 'micro_batch_size_per_gpu', 'pipeline_stages',
        'gradient_accumulation_steps', 'gradient_clipping', 'warmup_steps']:
        val = _get(key, None)
        if val is not None:
            lines.append(f"{key} = {val}")
    # lr scheduler (string field)
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
        # keep as string option like 'unsloth'
        lines.append(f"activation_checkpointing = {_quote(act_ckpt)}")
    lines.append("")

    # eval settings
    lines.append("# eval settings")
    lines.append(f"eval_every_n_epochs = {_get('eval_every_n_epochs', 1)}")
    lines.append(f"eval_before_first_step = {'true' if _as_bool(_get('eval_before_first_step', True)) else 'false'}")
    lines.append(f"eval_micro_batch_size_per_gpu = {_get('eval_micro_batch_size_per_gpu', 1)}")
    lines.append(f"eval_gradient_accumulation_steps = {_get('eval_gradient_accumulation_steps', 1)}")
    lines.append("")

    # misc settings
    lines.append("# misc settings")
    for key, dval in [
        ('save_every_n_epochs', 5),
        ('checkpoint_every_n_minutes', 10),
        ('partition_method', 'parameters'),
        ('save_dtype', 'bfloat16'),
        ('caching_batch_size', 1),
        ('steps_per_print', 1),
        ('video_clip_mode', 'single_beginning'),
    ]:
        val = _get(key, dval)
        if isinstance(dval, str):
            lines.append(f"{key} = {_quote(val)}")
        else:
            lines.append(f"{key} = {val}")
    lines.append("")

    # Block swap section
    lines.append("# BLOCK SWAP (requires pipeline_stages=1)")
    blocks_swap_val = _get('blocks_swap', 0)
    try:
        blocks_swap_val = int(blocks_swap_val)
    except Exception:
        blocks_swap_val = 0
    if blocks_swap_val and blocks_swap_val > 0:
        lines.append(f"blocks_to_swap = {blocks_swap_val}")
    else:
        lines.append(f"#blocks_to_swap = {blocks_swap_val}")
    disable_bsfe_val = _get('disable_bsfe', 'true')
    lines.append(f"disable_block_swap_for_eval = {'true' if _as_bool(disable_bsfe_val) else 'false'}")
    lines.append("")

    # [model]
    lines.append("[model]")
    model_source = _get('Model Type', '')
    # Convert wan22 to wan when saving
    is_wan22 = (model_source == 'wan22')
    if is_wan22:
        model_source = 'wan'
    lines.append(f"type = {_quote(model_source)}")

    mt_lower = str(_get('Model Type', '')).strip().lower()
    # SDXL: write checkpoint_path immediately under type, if provided
    if mt_lower == 'sdxl':
        ckpt = _get('checkpoint_path', None)
        if ckpt is not None and str(ckpt).strip() != '':
            lines.append(f"checkpoint_path = {_quote(ckpt)}")

    # Base model paths (skip for SDXL; and only if provided)
    if mt_lower != 'sdxl':
        diff_path = _get('diffusers_path', None)
        if diff_path is not None and str(diff_path).strip() != '':
            # Expand to absolute path for TOML
            expanded_diff_path = expand_model_path(diff_path)
            lines.append(f"diffusers_path  = {_quote(expanded_diff_path)}")
        transf_path = _get('transformer_path', None)
        if transf_path is not None and str(transf_path).strip() != '':
            # Expand to absolute path for TOML
            expanded_transf_path = expand_model_path(transf_path)
            lines.append(f"transformer_path   = {_quote(expanded_transf_path)}")
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
                f8e_bool = False
                if isinstance(f8e, bool):
                    f8e_bool = f8e
                else:
                    f8e_bool = str(f8e).strip().lower() in ['1','true','yes','on']
                if f8e_bool:
                    t_dtype = 'float8_e5m2'
            if mt_lower == 'hidream':
                td = _get('t_dtype', False)
                td_bool = False
                if isinstance(td, bool):
                    td_bool = td
                else:
                    td_bool = str(td).strip().lower() in ['1','true','yes','on']
                if td_bool:
                    t_dtype = 'nf4'
            if mt_lower == 'longcat':
                td = _get('float8 t_dtype', False)
                td_bool = False
                if isinstance(td, bool):
                    td_bool = td
                else:
                    td_bool = str(td).strip().lower() in ['1','true','yes','on']
                if td_bool:
                    t_dtype = 'float8'
        except Exception:
            pass
        lines.append(f"transformer_dtype = {_quote(t_dtype)}")
    # Support renamed UI label 'timestep_sm' while saving canonical key
    # If UI selects "None", write a commented-out default line
    _tsm = _get('timestep_sample_method', None)
    if _tsm is None or str(_tsm).strip() == '':
        _tsm = _get('timestep_sm', 'logit_normal')
    if str(_tsm).strip().lower() == 'none':
        lines.append("#timestep_sample_method = 'logit_normal'")
    else:
        lines.append(f"timestep_sample_method = {_quote(_tsm)}")
    # Append model-specific extra lines (paths and flags)
    append_model_specific_lines(lines, _get, _get('Model Type', ''))
    lines.append("")

    # [optimizer]
    lines.append("[optimizer]")
    lines.append(f"type = {_quote(_get('optimizer_type', 'adamw_optimi'))}")
    lines.append(f"lr = {_get('lr', 2e-5)}")
    lines.append(f"betas = {_get('betas', '[0.9, 0.99]')}")
    lines.append(f"weight_decay = {_get('weight_decay', 0.01)}")
    lines.append(f"eps = {_get('eps', 1e-8)}")
    lines.append("")

    # [adapter]
    lines.append("[adapter]")
    lines.append(f"type = {_quote(_get('adapter', 'lora'))}")
    lines.append(f"rank = {_get('a_rank', 32)}")
    lines.append(f"dtype = {_quote(_get('a_dtype', 'bfloat16'))}")

    # [monitoring]
    try:
        lines.append("")
        lines.append("[monitoring]")
        # Support WandB fields originating either from Monitor tab or Config tab
        wandb_keys = ("enable_wandb", "wandb_api_key", "wandb_tracker_name", "wandb_run_name")
        mon_src = {}
        try:
            mon_src.update(monitor_cfg or {})
        except Exception:
            pass
        try:
            # Prefer values present on Config tab controls, if any
            mon_src.update({k: v for k, v in (cfg or {}).items() if k in wandb_keys})
        except Exception:
            pass
        en_wandb = mon_src.get('enable_wandb', False)
        lines.append(f"enable_wandb = {'true' if (str(en_wandb).lower() in ['1','true','yes','on']) else 'false'}")
        lines.append(f"wandb_api_key = {_quote(mon_src.get('wandb_api_key', ''))}")
        lines.append(f"wandb_tracker_name = {_quote(mon_src.get('wandb_tracker_name', ''))}")
        lines.append(f"wandb_run_name = {_quote(mon_src.get('wandb_run_name', ''))}")
    except Exception:
        pass

    return "\n".join(lines) + "\n"


def extract_config_from_controls(control):
    result = {}
    if hasattr(control, 'controls') and control.controls:
        for child in control.controls:
            child_result = extract_config_from_controls(child)
            if child_result:
                result.update(child_result)
    elif hasattr(control, 'content') and control.content:
        child_result = extract_config_from_controls(control.content)
        if child_result:
            result.update(child_result)
    elif isinstance(control, ft.TextField):
        if getattr(control, 'visible', True):
            result[control.label] = control.value
    elif isinstance(control, ft.Dropdown):
        if getattr(control, 'visible', True):
            result[control.label] = control.value
    elif isinstance(control, ft.Checkbox):
        if getattr(control, 'visible', True):
            result[control.label] = control.value
    return result


def update_ui_from_toml(training_tab_container, toml_data: dict):
    """Populate UI controls from TOML data produced by our builder schema."""
    page = getattr(training_tab_container, 'page', None)

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ('1', 'true', 'yes', 'on')

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
    try:
        if isinstance(label_vals.get('output_dir'), str):
            from flet_app.project_root import get_project_root as _proj
            proj_root = _proj()
            od = str(label_vals['output_dir']).replace('\\', '/')
            proj_root_str = str(proj_root).replace('\\', '/')
            if od.lower().startswith(proj_root_str.lower().rstrip('/') + '/'):
                rel = od[len(proj_root_str.rstrip('/'))+1:]
                label_vals['output_dir'] = rel
    except Exception:
        pass
    if 'eval_before_first_step' in toml_data:
        label_vals['eval_before_first_step'] = _to_bool(toml_data.get('eval_before_first_step'))

    # Block swap
    if 'blocks_to_swap' in toml_data:
        label_vals['blocks_swap'] = toml_data.get('blocks_to_swap')
    if 'disable_block_swap_for_eval' in toml_data:
        label_vals['disable_bsfe'] = 'true' if _to_bool(toml_data.get('disable_block_swap_for_eval')) else 'false'

    # Model
    model = toml_data.get('model', {}) or {}
    if isinstance(model, dict):
        normalized_type = populate_label_vals_from_model(model, label_vals)
        if normalized_type:
            label_vals['Model Type'] = normalized_type
        if 'dtype' in model:
            label_vals['dtype'] = model.get('dtype')
        if 'transformer_dtype' in model:
            label_vals['transformer_dtype'] = model.get('transformer_dtype')
        if 'timestep_sample_method' in model:
            # Map to UI label 'timestep_sm'
            label_vals['timestep_sm'] = model.get('timestep_sample_method')

    # Optimizer
    opt = toml_data.get('optimizer', {}) or {}
    if isinstance(opt, dict):
        if 'type' in opt:
            label_vals['optimizer_type'] = opt.get('type')
        for k in ('lr', 'betas', 'weight_decay', 'eps'):
            if k in opt:
                label_vals[k] = opt.get(k)

    # Adapter
    ad = toml_data.get('adapter', {}) or {}
    if isinstance(ad, dict):
        if 'type' in ad:
            label_vals['adapter'] = ad.get('type')
        if 'rank' in ad:
            label_vals['a_rank'] = ad.get('rank')
        if 'dtype' in ad:
            label_vals['a_dtype'] = ad.get('dtype')

    # Monitoring
    mon = toml_data.get('monitoring', {}) or {}
    if isinstance(mon, dict):
        if 'enable_wandb' in mon:
            label_vals['enable_wandb'] = _to_bool(mon.get('enable_wandb'))
        for k in ('wandb_api_key', 'wandb_tracker_name', 'wandb_run_name'):
            if k in mon:
                label_vals[k] = mon.get(k)

    # Apply values to controls by label
    def _apply_values_recursive(control):
        try:
            if hasattr(control, 'controls') and control.controls:
                for c in control.controls:
                    _apply_values_recursive(c)
            if hasattr(control, 'content') and control.content:
                _apply_values_recursive(control.content)
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
                    # Clear dropdown if value is empty/None; else set to provided value
                    if val is None or str(val).strip() == '':
                        control.value = None
                    else:
                        control.value = str(val)
                    if control.page:
                        control.update()
                    # If this is the Model Type dropdown, fire its change handler
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

    def _apply_all():
        try:
            _apply_values_recursive(training_tab_container.config_page_content)
        except Exception:
            pass
        try:
            dataset_content = getattr(training_tab_container, 'dataset_page_content', None)
            if dataset_content is not None:
                _apply_values_recursive(dataset_content)
        except Exception:
            pass

    suppress_defaults_ctx = None
    try:
        from flet_app.ui.pages import training_config as _training_config_module  # Delayed import to avoid cycles
        suppress_defaults_ctx = getattr(_training_config_module, 'suppress_model_defaults', None)
    except Exception:
        suppress_defaults_ctx = None

    if suppress_defaults_ctx:
        try:
            with suppress_defaults_ctx():
                _apply_all()
        except Exception:
            _apply_all()
    else:
        _apply_all()

    # Also update bottom bar output_dir field if available
    try:
        if 'output_dir' in label_vals:
            bb_field = getattr(training_tab_container, 'output_dir_field', None)
            if bb_field is not None and isinstance(bb_field, ft.TextField):
                bb_field.value = str(label_vals['output_dir']) if label_vals['output_dir'] is not None else ''
                if bb_field.page:
                    bb_field.update()
    except Exception:
        pass
    try:
        data_config_content = getattr(training_tab_container, 'data_config_page_content', None)
        if data_config_content is not None and hasattr(data_config_content, 'dataset_block'):
            _apply_values_recursive(data_config_content.dataset_block)
    except Exception:
        pass
    try:
        monitor_content = getattr(training_tab_container, 'monitor_page_content', None)
        if monitor_content is not None:
            _apply_values_recursive(monitor_content)
    except Exception:
        pass

    # Dataset selection from 'dataset' key
    try:
        dataset_path = toml_data.get('dataset')
        if dataset_path:
            clean_name = os.path.basename(str(dataset_path).replace('\\', '/'))
            # Strip .toml if present
            if clean_name.lower().endswith('.toml'):
                clean_name = clean_name[:-5]
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

    # Post-processing: ensure visibility for model-specific fields (delegated)
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


def _save_and_scale_image(source_image_path: str, video_dims_tuple: tuple, dataset_name: str,
                        target_filename: str, dataset_type: str = "video", # Added dataset_type
                        page: ft.Page = None, target_control: str = None,
                        image_display_c1=None, image_display_c2=None):
    from .utils_top_menu import TopBarUtils  # Import here to avoid circular imports

    if not source_image_path or not dataset_name:
        logger.warning("_save_and_scale_image: Missing source_image_path or dataset_name.")
        return None

    try:
        from PIL import Image
        img = Image.open(source_image_path)
    except FileNotFoundError:
        logger.error(f"Error: Source image not found at {source_image_path} in _save_and_scale_image.")
        return None
    except Exception as e:
        logger.error(f"Error opening image {source_image_path} in _save_and_scale_image: {e}")
        return None

    # Unified datasets: always use 'datasets'
    base_datasets_dir = "datasets"
    dataset_sample_images_dir = Path("workspace") / base_datasets_dir / dataset_name / "sample_images"
    dataset_sample_images_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_sample_images_dir / target_filename

    original_width, original_height = img.size
    target_width, target_height = video_dims_tuple[0], video_dims_tuple[1]

    if original_width == 0 or original_height == 0 or target_width == 0 or target_height == 0:
        logger.warning(f"Invalid dimensions for scaling. Original: {img.size}, Target: {(target_width, target_height)}. Saving original.")
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(target_path)
        return str(target_path).replace('\\', '/')

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

        if page is not None and target_control and target_control.lower() in ['c1', 'c2']:
            logger.debug(f"Calling _load_cropped_image_into_ui from _save_and_scale_image for {target_control} with path {result_path}")
            TopBarUtils._load_cropped_image_into_ui(
                page=page,
                image_path=result_path,
                target_control_key=target_control, # Pass as target_control_key
                image_display_c1=image_display_c1, # Pass along if available
                image_display_c2=image_display_c2  # Pass along if available
            )
        return result_path
    except Exception as e:
        logger.error(f"Error resizing or saving image {source_image_path} to {target_path}: {e}")
        logger.error(traceback.format_exc())
        return None
