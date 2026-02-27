import flet as ft
from flet_app.ui._styles import create_textfield
from flet_app.ui.pages.training_config import get_training_config_page_content
from flet_app.ui.pages.training_monitor import get_training_monitor_page_content
from flet_app.ui_popups import dataset_not_selected
# Import from new training module
from flet_app.ui.training import (
    cleanup_training_console,
    set_button_state,
    reset_to_start_button,
    terminate_process,
    handle_cancel_click,
    run_ltx2_training_flow,
)
import os
import subprocess
import signal
import shutil
import asyncio
import traceback
import tempfile
import json
from PIL import Image


def _get_workspace_last_config_paths():
    """
    Return the workspace directory along with the locked last_config and last_data_config paths.
    Ensures the workspace directory exists before returning the paths.
    """
    from flet_app.project_root import get_project_root

    project_root = str(get_project_root())
    ws_dir = os.path.abspath(os.path.join(project_root, "workspace"))
    os.makedirs(ws_dir, exist_ok=True)
    last_config_path = os.path.join(ws_dir, "last_config.toml")
    last_data_config_path = os.path.join(ws_dir, "last_data_config.toml")
    return ws_dir, last_config_path, last_data_config_path

# =====================
# Data/Utility Functions
# =====================

# Training console cleanup utilities are now in utils/console_cleanup.py


async def save_training_config_to_toml(training_tab_container):
    """
    Build TOML from UI and save to workspace/last_config.toml.
    Additionally, build and save workspace/last_data_config.toml from Data Config UI
    and update the training config to point its `dataset` to that file.
    Returns (absolute_path_to_last_config, toml_text_written).
    """
    import asyncio
    import re
    from concurrent.futures import ThreadPoolExecutor

    def _build_last_data_config_text():
        # Data config is now stored in per-dataset TOML files managed by data_config_panel.py
        # This function builds a consolidated last_data_config.toml for training
        # by reading from the individual dataset TOML files

        def _parse_json_list(val, default):
            import json as _json
            try:
                s = str(val) if val is not None else ""
                return _json.loads(s) if s.strip() != "" else default
            except Exception:
                return default

        def _fmt_list(lst):
            return "[" + ", ".join(str(x) for x in lst) + "]"

        def _fmt_list_of_lists(lst):
            def fmt_pair(p):
                # Handle both lists and single values
                if isinstance(p, (list, tuple)):
                    return "[" + ", ".join(str(x) for x in p) + "]"
                else:
                    return str(p)
            if not lst:
                return "[]"
            return "[" + ", ".join(fmt_pair(p) for p in lst) + "]"

        # Collect dataset directory path(s) from training config page (Dataset 1, 2, 3)
        datasets_to_save = []

        # Try to get datasets from training config page (Dataset 1, 2, 3)
        config_page = getattr(training_tab_container, 'config_page_content', None)
        if config_page and hasattr(config_page, 'dataset_1_block'):
            for ds_num in [1, 2, 3]:
                ds_block = getattr(config_page, f'dataset_{ds_num}_block', None)
                if ds_block and hasattr(ds_block, 'get_selected_dataset'):
                    selected_name = ds_block.get_selected_dataset()
                    if selected_name:
                        try:
                            from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
                            base_dir, _dtype = _get_dataset_base_dir(selected_name)
                            dir_path_val = os.path.join(base_dir, selected_name).replace('\\', '/')
                            # Get frame_extraction value from dataset block
                            frame_extraction_val = None
                            if hasattr(ds_block, 'get_frame_extraction'):
                                frame_extraction_val = ds_block.get_frame_extraction()
                            datasets_to_save.append({
                                'name': selected_name,
                                'path': dir_path_val,
                                'dtype': _dtype,
                                'frame_extraction': frame_extraction_val
                            })
                        except Exception:
                            pass

        # If no datasets from training config, try the single dataset block
        if not datasets_to_save:
            ds_block = getattr(training_tab_container, 'dataset_page_content', None)
            selected_name = None
            try:
                if ds_block and hasattr(ds_block, 'get_selected_dataset'):
                    selected_name = ds_block.get_selected_dataset()
                if selected_name:
                    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
                    base_dir, _dtype = _get_dataset_base_dir(selected_name)
                    dir_path_val = os.path.join(base_dir, selected_name).replace('\\', '/')
                    datasets_to_save.append({
                        'name': selected_name,
                        'path': dir_path_val,
                        'dtype': _dtype,
                        'frame_extraction': None
                    })
            except Exception:
                pass

        # Helper function to read config from a dataset's TOML file
        def _read_dataset_toml_config(dataset_name):
            """Read config values from a dataset's TOML file."""
            config = {}
            try:
                try:
                    import tomllib as _toml_reader
                except Exception:
                    _toml_reader = None

                if _toml_reader and dataset_name:
                    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
                    base_dir, _dtype = _get_dataset_base_dir(dataset_name)
                    clean_dataset_name = str(dataset_name)
                    dataset_full_path = os.path.join(base_dir, clean_dataset_name)
                    parent_dir = os.path.dirname(dataset_full_path)
                    toml_path = os.path.join(parent_dir, f"{clean_dataset_name}.toml")

                    if os.path.exists(toml_path):
                        with open(toml_path, 'rb') as f:
                            config = _toml_reader.load(f)
            except Exception:
                pass
            return config

        lines = []

        # Top-level fallback defaults (required by training code)
        lines.append("# Top-level fallback defaults")
        lines.append("resolutions = [512, 512]")
        lines.append("min_ar = 0.5")
        lines.append("max_ar = 2.0")
        lines.append("num_ar_buckets = 2")
        lines.append("")

        # Generate [[directory]] entries for each selected dataset
        for i, ds_info in enumerate(datasets_to_save):
            # Read config from the dataset's TOML file
            ds_config = _read_dataset_toml_config(ds_info['name'])

            # Get values from TOML config, with defaults
            resolutions_raw = ds_config.get('resolutions', [])
            ar_buckets_raw = ds_config.get('ar_buckets', [])
            enable_ar_bucket_val = ds_config.get('enable_ar_bucket', True)
            min_ar_val = ds_config.get('min_ar', 0.5)
            max_ar_val = ds_config.get('max_ar', 2.0)
            num_ar_buckets_val = ds_config.get('num_ar_buckets', 7)

            # Read control_path and negative_path from [[directory]] section
            directory_config = ds_config.get('directory', [{}])[0] if ds_config.get('directory') else {}
            # Also check for values directly at top level (old format)
            control_path_val = ds_config.get('control_path') or directory_config.get('control_path')
            negative_path_val = ds_config.get('negative_path') or directory_config.get('negative_path')

            # Read control_args from last_config.toml if slider mode is enabled
            control_args_val = None
            try:
                ws_dir, last_config_path, _ = _get_workspace_last_config_paths()
                if os.path.exists(last_config_path):
                    try:
                        import tomllib as _toml_reader
                    except Exception:
                        import tomli as _toml_reader
                    with open(last_config_path, 'rb') as f:
                        config = _toml_reader.load(f)
                    training_strategy = config.get('training_strategy', {})
                    slider_enabled = training_strategy.get('slider', False)
                    # Handle boolean conversion from string
                    if not isinstance(slider_enabled, bool):
                        slider_enabled = str(slider_enabled).lower() in ['true', '1', 'yes', 'on']
                    if slider_enabled:
                        control_args_val = training_strategy.get('control_args', None)
            except Exception:
                pass

            # Handle frame_buckets - check if commented/disabled
            frame_buckets_list = ds_config.get('frame_buckets', [])
            # Check if frame_buckets is disabled (commented in TOML or enable_frame_buckets is False)
            # Note: In the new format, if frame_buckets is at top level without enable_ flag,
            # we assume it's enabled if present
            enable_frame_buckets = bool(frame_buckets_list)

            # Get num_repeats from dataset block if available
            num_repeats_val = 1
            config_page = getattr(training_tab_container, 'config_page_content', None)
            if config_page:
                ds_block = getattr(config_page, f'dataset_{datasets_to_save.index(ds_info) + 1}_block', None)
                if ds_block and hasattr(ds_block, 'get_num_repeats'):
                    try:
                        num_repeats_val = int(ds_block.get_num_repeats())
                    except Exception:
                        num_repeats_val = 1

            lines.append("[[directory]]")
            lines.append("# The target images go in here. These are the images that the model will learn to produce.")
            lines.append(f"path = '{ds_info['path']}'")

            # Per-dataset settings
            lines.append(f"num_repeats = {num_repeats_val}")

            # Resolutions (per-dataset)
            if not resolutions_raw:
                lines.append("# resolutions = []")
            else:
                lines.append(f"resolutions = {_fmt_list(resolutions_raw)}")

            # AR bucket settings (per-dataset)
            lines.append(f"enable_ar_bucket = {'true' if enable_ar_bucket_val else 'false'}")
            lines.append("# Min and max aspect ratios, given as width/height ratio.")
            lines.append(f"min_ar = {min_ar_val}")
            lines.append(f"max_ar = {max_ar_val}")
            if not ar_buckets_raw:
                lines.append("# ar_buckets = []")
            else:
                lines.append(f"ar_buckets = {_fmt_list_of_lists(ar_buckets_raw)}")
            lines.append("# Total number of aspect ratio buckets, evenly spaced (in log space) between min_ar and max_ar.")
            lines.append(f"num_ar_buckets = {num_ar_buckets_val}")

            # Frame buckets (per-dataset)
            try:
                fb_line = f"frame_buckets = {_fmt_list(frame_buckets_list)}"
                if not enable_frame_buckets:
                    fb_line = "# " + fb_line
                lines.append(fb_line)
            except Exception:
                pass

            # Frame extraction (per-dataset, only if set)
            frame_extraction_val = ds_info.get('frame_extraction')
            if frame_extraction_val:
                lines.append(f"frame_extraction = \"{frame_extraction_val}\"")

            # control_args (per-dataset, only if set and slider mode is enabled)
            if control_args_val is not None:
                # Format as TOML array
                if isinstance(control_args_val, list):
                    formatted_args = "[" + ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in control_args_val) + "]"
                    lines.append(f"control_args = {formatted_args}")

            # control_path and negative_path (per-dataset, only if set)
            if control_path_val:
                lines.append(f"control_path = '{control_path_val}'")
            if negative_path_val:
                lines.append(f"negative_path = '{negative_path_val}'")

            # Blank line between datasets
            if i < len(datasets_to_save) - 1:
                lines.append("")

        return "\n".join(lines) + "\n"

    def _save_both():
        from .utils.config_utils import build_toml_config_from_ui, extract_config_from_controls

        ws_dir, last_config_path, last_data_config_path = _get_workspace_last_config_paths()

        def _write_atomic(path: str, text: str):
            """Write text to path atomically to avoid partially-written configs."""
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
                    tmp_file.write(text)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                os.replace(tmp_path, path)
            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                raise

        # 1) Build and save last_data_config.toml
        data_toml_text = _build_last_data_config_text()
        data_toml_path = last_data_config_path
        _write_atomic(data_toml_path, data_toml_text)

        # 2) Detect model type and build training TOML
        # Check if this is an LTX2 model
        config_page = getattr(training_tab_container, 'config_page_content', None)
        cfg = {}
        if config_page:
            try:
                cfg = extract_config_from_controls(config_page) or {}
            except Exception:
                pass

        model_type = cfg.get('Model Type', '').strip().lower() if cfg else ''

        # Use appropriate builder based on model type
        if model_type == 'ltx-video-2':
            from .utils.ltx2_config_utils import build_ltx2_toml_from_ui
            train_toml_text = build_ltx2_toml_from_ui(training_tab_container)
        else:
            train_toml_text = build_toml_config_from_ui(training_tab_container)
        data_toml_path_abs = os.path.abspath(data_toml_path).replace('\\', '/')

        # Replace the dataset pointer to point to our last_data_config.toml
        def _replace_dataset_line(content: str, new_path: str) -> str:
            # For standard models: dataset = '...'
            pattern = r"^(\s*dataset\s*=\s*)['\"]([^'\"]*)['\"]\s*$"
            repl = r"\1'" + new_path + r"'"
            return re.sub(pattern, repl, content, flags=re.MULTILINE)

        def _replace_preprocessed_data_root(content: str, new_path: str) -> str:
            # For LTX2 models: preprocessed_data_root = '...' in [data] section
            pattern = r"^(\s*preprocessed_data_root\s*=\s*)['\"]([^'\"]*)['\"]\s*$"
            repl = r"\1'" + new_path + r"'"
            return re.sub(pattern, repl, content, flags=re.MULTILINE)

        if model_type == 'ltx-video-2':
            train_toml_text = _replace_preprocessed_data_root(train_toml_text, data_toml_path_abs)
        else:
            train_toml_text = _replace_dataset_line(train_toml_text, data_toml_path_abs)

        # Convert wan22 to wan for runtime backend compatibility
        def _convert_wan22_to_wan(content: str) -> str:
            """Convert wan22 model type to wan for backend compatibility"""
            lines = content.split('\n')
            converted_lines = []
            for line in lines:
                # Look for "type = 'wan22'" in model section
                if line.strip().startswith("type = ") and "'wan22'" in line:
                    line = line.replace("'wan22'", "'wan'")
                converted_lines.append(line)
            return '\n'.join(converted_lines)

        # Convert flux2_klein variants to flux2 for last_config.toml
        def _convert_flux2_klein_to_flux2(content: str) -> str:
            """Convert flux2_klein_4b/9b model type to flux2 in last_config.toml"""
            lines = content.split('\n')
            converted_lines = []
            for line in lines:
                # Look for "type = 'flux2_klein_*'" in model section
                if line.strip().startswith("type = ") and ("'flux2_klein_4b'" in line or "'flux2_klein_9b'" in line):
                    line = line.replace("'flux2_klein_4b'", "'flux2'").replace("'flux2_klein_9b'", "'flux2'")
                converted_lines.append(line)
            return '\n'.join(converted_lines)

        train_toml_text = _convert_wan22_to_wan(train_toml_text)
        train_toml_text = _convert_flux2_klein_to_flux2(train_toml_text)

        # 3) Save last_config.toml with updated dataset pointer
        out_path = last_config_path
        _write_atomic(out_path, train_toml_text)

        return out_path, train_toml_text

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _save_both)
    print(f"Saved training TOML to {result[0]} and last_data_config.toml alongside it")
    return result

def _detect_gpu_count() -> int:
    """Best-effort detection of available GPUs. Returns 1 if unknown."""
    try:
        import torch  # type: ignore
        cnt = int(getattr(torch.cuda, 'device_count', lambda: 0)())
        if cnt and cnt > 0:
            return cnt
    except Exception:
        pass
    try:
        res = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
            if lines:
                return max(1, len(lines))
    except Exception:
        pass
    return 1

async def run_training_deepspeed(config_path: str, use_multi_gpu: bool, trust_cache: bool = False, resume_last: bool = False, cache_only: bool = False, resume_last_strength: float = 1.0, reset_optimizer_params: bool = False, reset_optimizer: bool = False):
    """Launch deepspeed training with env vars and computed GPU count.
    Returns (proc, cmd_string) where proc is a Popen object with stdout piped.
    """
    from flet_app.ui.utils.process_cleanup import kill_existing_training_processes

    # Kill any existing training processes before starting new one
    kill_existing_training_processes()

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _run():
        from flet_app.project_root import get_project_root
        project_root = str(get_project_root())
        num_gpus = _detect_gpu_count() if use_multi_gpu else 1
        cmd = [
            "deepspeed",
            f"--num_gpus={num_gpus}",
            "diffusion-trainers/diffusion-pipe/train.py",
            "--deepspeed",
            "--config",
            os.path.abspath(config_path),
        ]
        if trust_cache:
            cmd.append("--trust_cache")
        if resume_last:
            cmd.append("--resume_from_checkpoint")
            # Only add adapter scale flag if the training script supports it.
            if resume_last_strength < 1.0:
                try:
                    train_py = os.path.join(project_root, "diffusion-trainers","diffusion-pipe", "train.py")
                    with open(train_py, "r", encoding="utf-8") as f:
                        train_src = f.read()
                    if "resume_adapter_scale" in train_src:
                        cmd.extend(["--resume_adapter_scale", str(resume_last_strength)])
                except Exception:
                    # If detection fails, fall back to plain resume_from_checkpoint.
                    pass
        if cache_only:
            cmd.append("--cache_only")
        if reset_optimizer_params:
            cmd.append("--reset_optimizer_params")
        if reset_optimizer:
            cmd.append("--reset_optimizer")
        env = os.environ.copy()
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        popen_kwargs = dict(
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if os.name == 'posix':
                popen_kwargs["preexec_fn"] = os.setsid
            elif os.name == 'nt':
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        except Exception:
            pass
        proc = subprocess.Popen(cmd, **popen_kwargs)
        return proc, " ".join(cmd)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        launched = await loop.run_in_executor(pool, _run)
    proc, cmd_str = launched
    print(f"Started training with command: {cmd_str}")
    return proc, cmd_str

def is_dataset_selected(config_page_content):
    """
    Checks if a dataset is selected in the config page content.
    Returns the selected dataset or None.
    """
    dataset_block = getattr(config_page_content, 'dataset_block', None)
    if dataset_block and hasattr(dataset_block, 'get_selected_dataset'):
        return dataset_block.get_selected_dataset()
    return None

def check_and_delete_zone_identifier_files(dataset_name):
    """
    Checks for and deletes files with 'Zone.Identifier' in their names in the selected dataset folder.
    Returns the number of files deleted.
    """
    if not dataset_name:
        print("No dataset selected for Zone.Identifier check")
        return 0

    try:
        from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
        base_dir, _ = _get_dataset_base_dir(dataset_name)
        dataset_path = os.path.join(base_dir, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"Dataset path does not exist: {dataset_path}")
            return 0

        # Search for and delete files containing 'Zone.Identifier' in their names
        deleted_count = 0
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if "Zone.Identifier" in file:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted Zone.Identifier file: {file_path}")
                        deleted_count += 1
                    except Exception as delete_error:
                        print(f"Failed to delete {file_path}: {delete_error}")

        return deleted_count

    except Exception as e:
        print(f"Error checking/deleting Zone.Identifier files: {e}")
        return 0

def check_and_move_unmatched_control_images(dataset_name):
    """
    Checks for dataset images that don't have corresponding control images.
    Moves unmatched images to a '_miss' folder next to the control folder.
    Returns the number of moved files.
    """
    if not dataset_name:
        print("No dataset selected for control image check")
        return 0

    try:
        from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
        import glob

        base_dir, _ = _get_dataset_base_dir(dataset_name)
        dataset_path = os.path.join(base_dir, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"Dataset path does not exist: {dataset_path}")
            return 0

        control_dir = os.path.join(dataset_path, "control")
        if not os.path.exists(control_dir):
            print(f"Control directory does not exist: {control_dir}")
            return 0

        # Create _miss directory if it doesn't exist
        miss_dir = os.path.join(dataset_path, "_miss")
        os.makedirs(miss_dir, exist_ok=True)

        # Get all supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']

        # Find all images in the main dataset directory (excluding control and _miss directories)
        main_images = []
        for ext in image_extensions:
            main_images.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
            main_images.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))

        # Filter out files that might be in control or _miss directories
        main_images = [img for img in main_images if not img.startswith(control_dir + os.sep)
                      and not img.startswith(miss_dir + os.sep)]

        # Find all control images
        control_images = []
        for ext in image_extensions:
            control_images.extend(glob.glob(os.path.join(control_dir, f"*{ext}")))
            control_images.extend(glob.glob(os.path.join(control_dir, f"*{ext.upper()}")))

        # Create set of base names for control images (without extension)
        control_basenames = {os.path.splitext(os.path.basename(img))[0] for img in control_images}

        moved_count = 0
        for main_image in main_images:
            main_basename = os.path.splitext(os.path.basename(main_image))[0]

            # If main image doesn't have a corresponding control image
            if main_basename not in control_basenames:
                try:
                    # Move to _miss directory
                    target_path = os.path.join(miss_dir, os.path.basename(main_image))
                    shutil.move(main_image, target_path)
                    print(f"Moved unmatched image to _miss: {os.path.basename(main_image)}")
                    moved_count += 1
                except Exception as move_error:
                    print(f"Failed to move {main_image}: {move_error}")

        return moved_count

    except Exception as e:
        print(f"Error checking/moving unmatched control images: {e}")
        return 0

def has_control_enabled(training_tab_container, dataset_name=None):
    """
    Checks if dataset has a 'control' folder to determine if control is enabled.
    Returns True if control folder exists, False otherwise.
    """
    if not dataset_name:
        # Try to get dataset name from the container if not provided
        try:
            config_page_content = getattr(training_tab_container, 'config_page_content', None)
            if config_page_content:
                dataset_block = getattr(config_page_content, 'dataset_block', None)
                if dataset_block and hasattr(dataset_block, 'get_selected_dataset'):
                    dataset_name = dataset_block.get_selected_dataset()
        except Exception:
            pass

    if not dataset_name:
        print("Debug: No dataset available for control folder check")
        return False

    try:
        from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
        base_dir, _ = _get_dataset_base_dir(dataset_name)
        dataset_path = os.path.join(base_dir, dataset_name)

        control_path = os.path.join(dataset_path, "control")
        has_control = os.path.exists(control_path) and os.path.isdir(control_path)

        print(f"Debug: Control folder check - path: {control_path}, exists: {has_control}")
        return has_control

    except Exception as e:
        print(f"Debug: Error checking control folder: {e}")
        return False

# =====================
# GUI-Building Functions
# =====================

def build_navigation_rail(on_nav_change):
    """
    Builds the sub-navigation rail for the training tab.
    """
    config_dest_content = ft.Container(
        content=ft.Text("Config", size=10),
        padding=ft.padding.symmetric(vertical=0, horizontal=0),
        alignment=ft.alignment.center,
        expand=True
    )
    monitor_dest_content = ft.Container(
        content=ft.Text("Monitor", size=10),
        padding=ft.padding.symmetric(vertical=0, horizontal=0),
        alignment=ft.alignment.center,
        expand=True
    )
    return ft.NavigationRail(
        selected_index=0,
        on_change=on_nav_change,
        bgcolor=ft.Colors.TRANSPARENT,
        indicator_color=ft.Colors.TRANSPARENT,
        indicator_shape=ft.RoundedRectangleBorder(radius=0),
        label_type=ft.NavigationRailLabelType.NONE,
        destinations=[
            ft.NavigationRailDestination(icon=config_dest_content),
            ft.NavigationRailDestination(icon=monitor_dest_content),
        ]
    )

def build_main_content_row(sub_navigation_rail, content_area):
    """
    Builds the main content row containing the navigation rail and content area.
    """
    return ft.Row(
        [
            sub_navigation_rail,
            ft.VerticalDivider(),
            content_area,
        ],
        expand=True,
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.START
    )

def build_bottom_app_bar(on_start_click, multi_gpu_checkbox, trust_cache_checkbox, resume_last_checkbox, cache_only_checkbox, reset_opt_params_checkbox, reset_opt_checkbox, init_from_existing_field_ref, reset_opt_row_ref, init_from_existing_row_ref):
    """
    Builds the bottom app bar with the Start button and Multi-GPU checkbox.
    """
    # Output directory field moved to bottom bar (right-aligned, before Start)
    output_dir_field = create_textfield(
        "output_dir",
        "workspace/output/dir",
        width=250,
    )

    # Init from existing field (for loading existing adapter weights)
    init_from_existing_field = create_textfield(
        "init_from_existing",
        "",
        width=250,
        ref=init_from_existing_field_ref,
    )

    # Add Last Config checkbox
    last_config_checkbox = ft.Checkbox(label="Last Config", value=False)
    last_strength_field = create_textfield(
        "Last str",
        "1.0",
        width=100,
        keyboard_type=ft.KeyboardType.NUMBER,
        visible=False,
    )

    start_btn = ft.ElevatedButton(
        "Start",
        on_click=on_start_click,
        style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=14),
            shape=ft.RoundedRectangleBorder(radius=0)
        ),
        width=150,
        height=40,
    )
    bottom = ft.BottomAppBar(
        bgcolor=ft.Colors.BLUE_GREY_900,
        height=60,
        content=ft.Row(
            [
                # Container for option checkboxes
                ft.Container(
                    content=ft.Row([
                        multi_gpu_checkbox,
                        trust_cache_checkbox,
                        resume_last_checkbox,
                        # Reset optimizer options (hidden by default)
                        ft.Container(
                            content=ft.Row([
                                reset_opt_params_checkbox,
                                reset_opt_checkbox,
                            ], spacing=15),
                            ref=reset_opt_row_ref,
                            visible=False,
                        ),
                        # Init from existing field (shown by default)
                        ft.Container(
                            content=init_from_existing_field,
                            ref=init_from_existing_row_ref,
                            visible=True,
                        ),
                        last_strength_field,
                    ], spacing=15, alignment=ft.MainAxisAlignment.START),
                    alignment=ft.alignment.center_left,
                    expand=True,
                    padding=ft.padding.only(left=20) # Add some padding
                ),
                # Right side: Last Config checkbox then output_dir field then Start button
                ft.Container(
                    content=ft.Row([
                        last_config_checkbox,
                        cache_only_checkbox,
                        output_dir_field,
                        start_btn,
                    ], alignment=ft.MainAxisAlignment.END, spacing=12),
                    alignment=ft.alignment.center_right,
                    padding=ft.padding.only(right=20),
                ),
            ],
            expand=True,
        ),
    )
    # Expose the button for external control
    bottom.start_btn = start_btn
    # Expose the output_dir field for config read/write
    bottom.output_dir_field = output_dir_field
    # Expose the init_from_existing field for config read/write
    bottom.init_from_existing_field = init_from_existing_field
    # Expose the last_config_checkbox for external control
    bottom.last_config_checkbox = last_config_checkbox
    bottom.last_strength_field = last_strength_field
    return bottom

def build_main_container(main_content_row, bottom_app_bar):
    """
    Wraps the main content and bottom bar in a Stack for sticky footer.
    """
    return ft.Container(
        padding=ft.padding.only(top=0, bottom=0),
        content=ft.Stack(
            [
                ft.Container(main_content_row, expand=True),
                ft.Container(
                    bottom_app_bar,
                    left=0,
                    right=0,
                    bottom=0,
                    alignment=ft.alignment.bottom_center
                ),
            ],
            expand=True,
        ),
        expand=True
    )

# =====================
# Main Entry Point
# =====================

def get_training_tab_content(page: ft.Page):
    """
    Entry point for building the Training tab content. Sets up navigation, content, and event handlers.
    """
    page.snack_bar = ft.SnackBar(content=ft.Text("Training tab loaded! (debug)"), open=True)
    page.update()

    # Initialize config and monitor page content
    config_page_content = get_training_config_page_content()
    monitor_page_content = get_training_monitor_page_content()

    # Content area container
    content_area = ft.Container(
        content=config_page_content,
        expand=True,
        alignment=ft.alignment.top_left,
        padding=ft.padding.all(0)
    )

    def on_nav_change(e):
        try:
            # Switch content
            selected_idx = e.control.selected_index
            if selected_idx == 0:
                content_area.content = config_page_content
                # Refresh Config tab UI fields when switching to it
                try:
                    cfg_ds = getattr(config_page_content, 'dataset_block', None)
                    if cfg_ds and hasattr(cfg_ds, 'get_num_repeats') and hasattr(cfg_ds, 'set_num_repeats'):
                        current_repeats = cfg_ds.get_num_repeats()
                        cfg_ds.set_num_repeats(current_repeats, page_ctx=page)
                except Exception:
                    pass
            elif selected_idx == 1:
                content_area.content = monitor_page_content
            page.update()
        except Exception as ex:
            print(f"Error switching tabs: {ex}")
            page.update()

    sub_navigation_rail = build_navigation_rail(on_nav_change)
    main_content_row = build_main_content_row(sub_navigation_rail, content_area)

    # Add option checkboxes
    multi_gpu_checkbox = ft.Checkbox(label="Multi-GPU", value=False) # False by default
    trust_cache_checkbox = ft.Checkbox(label="Trust Cache", value=False)
    resume_last_checkbox = ft.Checkbox(label="Resume Last", value=False)
    cache_only_checkbox = ft.Checkbox(label="Cache Only", value=False)

    # Reset optimizer options (compact with tooltips, mutually exclusive)
    reset_opt_params_checkbox = ft.Checkbox(label="", value=False, tooltip="Reset Optimizer Params")
    reset_opt_checkbox = ft.Checkbox(label="", value=False, tooltip="Reset Optimizer")

    # Refs for visibility control
    reset_opt_row_ref = ft.Ref[ft.Container]()
    init_from_existing_row_ref = ft.Ref[ft.Container]()

    # Init from existing field reference
    init_from_existing_field_ref = ft.Ref[ft.TextField]()

    # Mutual exclusion logic for checkboxes
    def on_cache_only_change(e):
        if cache_only_checkbox.value:
            # Cache Only was checked, uncheck all others
            multi_gpu_checkbox.value = False
            trust_cache_checkbox.value = False
            resume_last_checkbox.value = False
            # Update UI
            if e.page:
                multi_gpu_checkbox.update()
                trust_cache_checkbox.update()
                resume_last_checkbox.update()

    def on_other_checkbox_change(e):
        if any([multi_gpu_checkbox.value, trust_cache_checkbox.value, resume_last_checkbox.value]):
            # Any other checkbox was checked, uncheck Cache Only
            cache_only_checkbox.value = False
            if e.page:
                cache_only_checkbox.update()

        # Toggle visibility based on resume_last_checkbox state
        resume_last_on = resume_last_checkbox.value
        if reset_opt_row_ref.current:
            reset_opt_row_ref.current.visible = resume_last_on
            reset_opt_row_ref.current.update()
        if init_from_existing_row_ref.current:
            init_from_existing_row_ref.current.visible = not resume_last_on
            init_from_existing_row_ref.current.update()

    # Attach event handlers
    cache_only_checkbox.on_change = on_cache_only_change
    multi_gpu_checkbox.on_change = on_other_checkbox_change
    trust_cache_checkbox.on_change = on_other_checkbox_change
    resume_last_checkbox.on_change = on_other_checkbox_change

    # Mutual exclusion for reset optimizer checkboxes
    def on_reset_opt_params_change(e):
        if reset_opt_params_checkbox.value:
            reset_opt_checkbox.value = False
            if e.page:
                reset_opt_checkbox.update()

    def on_reset_opt_change(e):
        if reset_opt_checkbox.value:
            reset_opt_params_checkbox.value = False
            if e.page:
                reset_opt_params_checkbox.update()

    reset_opt_params_checkbox.on_change = on_reset_opt_params_change
    reset_opt_checkbox.on_change = on_reset_opt_change

    async def handle_training_output(page=None, training_tab_container=None):
        """
        Handles saving config and running training, with error handling and user feedback.
        """

        try:
            if training_tab_container is None:
                msg = "Error: training_tab_container not passed to handle_training_output."
                if page is not None:
                    page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                    page.update()
                return

            current_selected_image_path_c1 = getattr(page, 'selected_image_path_c1', None)
            current_selected_image_path_c2 = getattr(page, 'selected_image_path_c2', None)

            # Save TOML config (or reuse the workspace "last" files if requested)
            _, last_config_path, last_data_config_path = _get_workspace_last_config_paths()
            last_config_checkbox = getattr(training_tab_container, 'last_config_checkbox', None)
            use_last_config = bool(last_config_checkbox and last_config_checkbox.value)
            last_strength_field = getattr(training_tab_container, 'last_strength_field', None)
            last_strength_value = 1.0
            if last_strength_field and isinstance(last_strength_field.value, str):
                raw_value = last_strength_field.value.strip()
            else:
                raw_value = ""
            if raw_value:
                try:
                    last_strength_value = float(raw_value)
                except ValueError:
                    print(f"Invalid Last str value '{raw_value}'. Defaulting to 1.0.")
                    last_strength_value = 1.0
            if last_strength_value <= 0:
                print(f"Last str must be > 0. Received {last_strength_value}. Defaulting to 1.0.")
                last_strength_value = 1.0
            if last_strength_field and last_strength_field.value != str(last_strength_value):
                last_strength_field.value = str(last_strength_value)
                try:
                    last_strength_field.update()
                except Exception:
                    pass
            if use_last_config and os.path.exists(last_config_path) and os.path.exists(last_data_config_path):
                print("Reusing existing workspace/last_config.toml")
                out_path = last_config_path
            else:
                out_path, _ = await save_training_config_to_toml(training_tab_container)

            # Check if trainer is musubi (for ltx-video-2, wan22 models)
            import toml
            try:
                with open(out_path, 'r') as f:
                    config = toml.load(f)
                    is_musubi_trainer = config.get('model', {}).get('trainer', 'diffusion-pipe') == 'musubi'
            except:
                is_musubi_trainer = False

            if is_musubi_trainer:
                # Use the new centralized musubi training flow
                await run_ltx2_training_flow(
                    out_path=out_path,
                    trust_cache=trust_cache_checkbox.value,
                    cache_only=cache_only_checkbox.value,
                    resume_last=resume_last_checkbox.value,
                    use_last_config=use_last_config,
                    main_container=main_container,
                    training_tab_container=training_tab_container,
                    page=page,
                    trust_cache_checkbox=trust_cache_checkbox,
                )
                return
            # Determine launch parameters
            use_multi_gpu = multi_gpu_checkbox.value
            trust_cache = trust_cache_checkbox.value
            resume_last = resume_last_checkbox.value
            cache_only = cache_only_checkbox.value
            # Prepare Training Console in Monitor tab
            try:
                monitor_content = getattr(training_tab_container, 'monitor_page_content', None)
                training_console_container = getattr(monitor_content, 'training_console_container', None)
                training_console_text = getattr(monitor_content, 'training_console_text', None)
                training_console_list = getattr(monitor_content, 'training_console_list', None)
                training_cmd_container = getattr(monitor_content, 'training_cmd_container', None)
                training_cmd_text = getattr(monitor_content, 'training_cmd_text', None)
                if training_console_container is not None:
                    training_console_container.visible = True
                if training_console_text is not None:
                    training_console_text.spans = []
                    # Ensure console starts clean
                    cleanup_training_console(training_console_text)
                if page is not None:
                    page.update()
            except Exception:
                pass

            proc, cmd_str = await run_training_deepspeed(out_path, use_multi_gpu, trust_cache, resume_last, cache_only, last_strength_value, reset_opt_params_checkbox.value, reset_opt_checkbox.value)

            # Store process handle and toggle Start->Cancel
            try:
                training_tab_container.training_proc = proc
                start_btn = getattr(training_tab_container, 'start_btn', None)
                if start_btn is not None:
                    start_btn.text = "Cancel"
                if page is not None:
                    page.update()
            except Exception:
                pass

            # Show command used above the console
            try:
                if training_cmd_text is not None:
                    training_cmd_text.value = cmd_str
                if training_cmd_container is not None:
                    training_cmd_container.visible = True
                if page is not None:
                    page.update()
            except Exception:
                pass

            # Stream output to Training Console
            try:
                from threading import Thread
                def _reader():
                    try:
                        if proc.stdout is None:
                            return
                        import re

                        def _spans_for_line(text: str):
                            spans_line = []
                            try:
                                s = text.rstrip("\n")
                                parts = re.split(r"(\[[^\]]*\])", s)
                                bracket_count = 0
                                for part in parts:
                                    if part is None or part == "":
                                        continue
                                    if part.startswith("[") and part.endswith("]"):
                                        bracket_count += 1
                                        if bracket_count <= 2:
                                            color = ft.Colors.with_opacity(0.3, ft.Colors.WHITE)
                                        else:
                                            color = ft.Colors.WHITE
                                        spans_line.append(ft.TextSpan(part, style=ft.TextStyle(color=color)))
                                    else:
                                        spans_line.append(ft.TextSpan(part, style=ft.TextStyle(color=ft.Colors.WHITE)))
                                spans_line.append(ft.TextSpan("\n"))
                            except Exception:
                                spans_line.append(ft.TextSpan(text, style=ft.TextStyle(color=ft.Colors.WHITE)))
                            return spans_line
                        line_counter = 0
                        # Simple throttle state for animation to avoid jitter
                        import time as _t
                        import asyncio as _a
                        _buffer = []  # pending spans batches (each element is a list[TextSpan])
                        _flush_running = {"v": False}

                        async def _flush_buffer_periodically():
                            _flush_running["v"] = True
                            try:
                                anim_wrap = getattr(monitor_content, 'training_console_anim', None)
                                while True:
                                    try:
                                        if not _buffer:
                                            # If process likely ended and buffer empty, stop
                                            if proc.poll() is not None:
                                                break
                                            await _a.sleep(0.08)
                                            continue
                                        # Drain buffer into a single batch
                                        drained = []
                                        while _buffer:
                                            drained.extend(_buffer.pop(0))
                                        # Apply once: append drained to existing spans (normal order)
                                        spans_current = list((training_console_text.spans or [])) if training_console_text is not None else []
                                        new_total = spans_current + drained
                                        if training_console_text is not None:
                                            training_console_text.spans = new_total

                                        # One smooth, scaled animation per batch
                                        try:
                                            if anim_wrap is not None:
                                                lines_added = 1
                                                try:
                                                    lines_added = max(1, sum(s.text.count('\n') for s in drained if hasattr(s, 'text')))
                                                except Exception:
                                                    pass
                                                top_pad = min(72, max(12, lines_added * 12))
                                                anim_wrap.padding = ft.padding.only(top=top_pad)
                                                anim_wrap.update()
                                                # small delay to allow implicit animate to kick in
                                                await _a.sleep(0.016)
                                                anim_wrap.padding = ft.padding.only(top=0)
                                                anim_wrap.update()
                                        except Exception:
                                            pass

                                        # Cleanup every batch to keep memory stable
                                        try:
                                            cleanup_performed = cleanup_training_console(training_console_text)
                                            if cleanup_performed:
                                                print("[AutoCleanup] Console cleaned post-batch")
                                        except Exception:
                                            pass

                                        try:
                                            training_console_text.update()
                                        except Exception:
                                            pass

                                        # Frame spacing between batches
                                        await _a.sleep(0.1)
                                    except Exception:
                                        await _a.sleep(0.1)
                                        continue
                            finally:
                                _flush_running["v"] = False

                        for line in proc.stdout:
                            try:
                                # Pre-split line into styled spans
                                new_line_spans = _spans_for_line(line) or []
                                # Enqueue to buffer for batched prepend
                                _buffer.append(new_line_spans)
                                # Start flusher if not running
                                if page is not None and not _flush_running["v"]:
                                    # Pass coroutine function to run_task (do not call it here)
                                    page.run_task(_flush_buffer_periodically)
                            except Exception:
                                pass
                    finally:
                        try:
                            rc = proc.wait(timeout=2)
                        except Exception:
                            rc = -1
                        try:
                            spans = list((training_console_text.spans or [])) if training_console_text is not None else []
                            spans.append(ft.TextSpan(f"\n[Done] Exit code: {rc}\n", style=ft.TextStyle(color=ft.Colors.WHITE)))
                            if training_console_text is not None:
                                training_console_text.spans = spans
                        except Exception:
                            pass
                        # Reset Start button and clear proc handle
                        try:
                            if hasattr(main_container, 'start_btn') and main_container.start_btn is not None:
                                main_container.start_btn.text = "Start"
                            training_tab_container.training_proc = None
                            main_container.training_proc = None
                        except Exception:
                            pass
                        try:
                            if page is not None:
                                page.update()
                        except Exception:
                            pass
                Thread(target=_reader, daemon=True).start()
            except Exception:
                pass

            msg = f"Saved config to {out_path}\nCmd: {cmd_str}\nTraining started."
            if page is not None:
                page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                page.update()
        except Exception as e:
            msg = f"Error: {e}"
            print(f"ERROR in handle_training_output: {e}")
            traceback.print_exc()
            if page is not None:
                page.snack_bar = ft.SnackBar(content=ft.Text(msg), open=True)
                page.update()

    async def handle_start_click(e, training_tab_container_arg):
        """
        Handles the Start button click: checks dataset selection and triggers training.
        """
        try:
            dataset_selected = is_dataset_selected(config_page_content)

            # Check for and delete Zone.Identifier files in the dataset
            deleted_count = check_and_delete_zone_identifier_files(dataset_selected)
            if deleted_count > 0:
                print(f"Deleted {deleted_count} Zone.Identifier files before training")

            # Check for unmatched control images if Has control is enabled
            # Pass both container and dataset name for cleaner folder detection
            has_control = has_control_enabled(training_tab_container_arg, dataset_selected)
            print(f"Has control enabled: {has_control}")  # Debug output
            if has_control:
                moved_count = check_and_move_unmatched_control_images(dataset_selected)
                if moved_count > 0:
                    print(f"Moved {moved_count} unmatched images to _miss folder")
                else:
                    print("No unmatched control images found")

            async def run_training():
                try:
                    await handle_training_output(e.page, training_tab_container_arg)
                except Exception as ex:
                    traceback.print_exc()
                    if e.page:
                        e.page.snack_bar = ft.SnackBar(
                            content=ft.Text(f"Error starting training: {ex}"),
                            open=True
                        )
                        e.page.update()

            if not dataset_selected:
                def on_confirm():
                    e.page.run_task(run_training)

                dataset_not_selected.show_dataset_not_selected_dialog(
                    e.page, "Dataset not selected, proceed?", on_confirm
                )
            else:
                await run_training()

        except Exception as ex:
            error_msg = f"Error in handle_start_click: {ex}"
            print(error_msg)
            traceback.print_exc()
            if e and e.page:
                e.page.snack_bar = ft.SnackBar(
                    content=ft.Text(error_msg),
                    open=True
                )
                e.page.update()

    # Create a wrapper function that can be called synchronously
    def start_button_click(e):
        try:
            # Switch to Monitor tab immediately to show console
            try:
                if sub_navigation_rail is not None:
                    sub_navigation_rail.selected_index = 1
                if content_area is not None:
                    content_area.content = monitor_page_content
                if e and e.page:
                    e.page.update()
            except Exception:
                pass
            # Check if this is musubi trainer (LTX2/WAN22) by checking the training container
            training_proc = None

            # Try to get training_proc from musubi specific container
            try:
                if hasattr(e.page, 'training_tab_container'):
                    training_tab = e.page.training_tab_container
                    # Check if this is musubi by checking the config
                    last_config_path = getattr(training_tab, 'last_config_path', None)
                    if last_config_path and os.path.exists(last_config_path):
                        import toml
                        with open(last_config_path, 'r') as f:
                            config = toml.load(f)
                            is_musubi_trainer = config.get('model', {}).get('trainer', 'diffusion-pipe') == 'musubi'
                        if is_musubi_trainer:
                            training_proc = getattr(training_tab, 'training_proc', None)
                            # Also check main_container as backup
                            if training_proc is None:
                                training_proc = getattr(main_container, 'training_proc', None)
            except Exception:
                pass

            # Fallback to main_container if not found
            if training_proc is None:
                training_proc = getattr(main_container, 'training_proc', None)

            if training_proc is not None:
                try:
                    alive = (training_proc.poll() is None)
                except Exception:
                    alive = False
                if alive:
                    # Use the new centralized cancel handler
                    if handle_cancel_click(e, main_container):
                        return

            # Safety check: if button shows "Cancel" but no process is running, reset it
            if hasattr(main_container, 'start_btn') and main_container.start_btn is not None:
                if main_container.start_btn.text == "Cancel" and training_proc is None:
                    # Button state is out of sync, reset to Start
                    main_container.start_btn.text = "Start"
                    if e and e.page:
                        e.page.update()
                    return  # Don't start training after resetting button

            # Otherwise, start training
            e.page.run_task(handle_start_click, e, main_container)
        except Exception as ex:
            print(f"ERROR in start_button_click: {ex}")

    # Build the bottom app bar with the wrapper function
    bottom_bar = build_bottom_app_bar(start_button_click, multi_gpu_checkbox, trust_cache_checkbox, resume_last_checkbox, cache_only_checkbox, reset_opt_params_checkbox, reset_opt_checkbox, init_from_existing_field_ref, reset_opt_row_ref, init_from_existing_row_ref)
    main_container = build_main_container(main_content_row, bottom_bar)

    # Attach Start button reference to main container for state toggling
    try:
        main_container.start_btn = getattr(bottom_bar, 'start_btn', None)
    except Exception:
        main_container.start_btn = None
    # Also expose output_dir field for config handling
    try:
        main_container.output_dir_field = getattr(bottom_bar, 'output_dir_field', None)
    except Exception:
        main_container.output_dir_field = None
    # Also expose init_from_existing field for config handling
    try:
        main_container.init_from_existing_field = getattr(bottom_bar, 'init_from_existing_field', None)
    except Exception:
        main_container.init_from_existing_field = None
    try:
        main_container.last_config_checkbox = getattr(bottom_bar, 'last_config_checkbox', None)
    except Exception:
        main_container.last_config_checkbox = None
    try:
        main_container.last_strength_field = getattr(bottom_bar, 'last_strength_field', None)
    except Exception:
        main_container.last_strength_field = None

    # Expose references to allow programmatic refreshes
    main_container.sub_navigation_rail = sub_navigation_rail
    main_container.content_area = content_area

    # Provide a helper to refresh Config panel UI state
    def refresh_config_panel():
        try:
            cfg = getattr(main_container, 'config_page_content', None)
            if cfg is None:
                return
            # Refresh Config tab UI fields
            cfg_ds = getattr(cfg, 'dataset_block', None)
            if cfg_ds and hasattr(cfg_ds, 'get_num_repeats') and hasattr(cfg_ds, 'set_num_repeats'):
                current_repeats = cfg_ds.get_num_repeats()
                cfg_ds.set_num_repeats(current_repeats, page_ctx=page)
            if page: page.update()
        except Exception:
            # Best effort; ignore if anything goes wrong
            pass
    main_container.refresh_config_panel = refresh_config_panel

    # Attach references for later extraction
    main_container.config_page_content = config_page_content
    main_container.monitor_page_content = monitor_page_content
    if hasattr(config_page_content, 'dataset_block'):
        main_container.dataset_page_content = config_page_content.dataset_block
    page.training_tab_container = main_container
    return main_container
