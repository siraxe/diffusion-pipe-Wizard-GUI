import flet as ft
import os
import json
from typing import Any
try:
    import tomllib as _toml_reader  # Python 3.11+
except Exception:  # pragma: no cover
    _toml_reader = None
from flet_app.ui._styles import create_textfield, add_section_title
from .training_dataset_block import get_training_dataset_page_content
from ..dataset_manager.dataset_utils import _get_dataset_base_dir
from typing import Any

def get_training_data_config_page_content():
    """Build the Data Config page with specified configuration fields and a Dataset Selection column."""

    # Replace the path field with: Has control + Data .toml exists indicator
    has_control_checkbox = ft.Checkbox(
        label="Has control",
        value=False,
        label_style=ft.TextStyle(size=12)
    )
    toml_exists_value = ft.Text("No", size=12, color=ft.Colors.RED_400)

    # Column 1 fields
    ar_buckets_field = create_textfield(
        "ar_buckets",
        "[[448, 576]]",
        hint_text="Aspect ratio buckets",
        expand=True
    )

    resolutions_field = create_textfield(
        "resolutions",
        "[512]",
        hint_text="Target resolutions",
        expand=True
    )

    frame_buckets_field = create_textfield(
        "frame_buckets",
        "[1, 33]",
        hint_text="Frame bucket values",
        expand=True
    )
    enable_frame_buckets_field = ft.Checkbox(
        label="enable_frame_buckets",
        value=False,
        label_style=ft.TextStyle(size=12)
    )

    # Column 2 fields
    enable_ar_bucket_field = ft.Checkbox(
        label="enable_ar_bucket",
        value=True,
        label_style=ft.TextStyle(size=12)
    )

    num_ar_buckets_field = create_textfield(
        "num_ar",
        "9",
        hint_text="Number of AR buckets",
        expand=False,
        width=60
    )

    min_ar_field = create_textfield(
        "min_ar",
        "0.5",
        hint_text="Minimum aspect ratio",
        expand=True,
        width=100
    )

    max_ar_field = create_textfield(
        "max_ar",
        "2.0",
        hint_text="Maximum aspect ratio",
        expand=True,
        width=100
    )

    # Dataset selection block (mirrors Config page)
    # Prepare Save button (styled like Monitor: default ElevatedButton)
    save_btn = ft.ElevatedButton(
        "Save Data Config",
        icon=ft.Icons.SAVE,
        on_click=None, # set after handler defined
    )
    dataset_block = get_training_dataset_page_content(extra_right_controls=[save_btn])

    # Create the main container
    page_controls = []

    # Helpers and save handler (defined early so UI can reference it)
    def _parse_list_field(text: str, default: Any):
        try:
            return json.loads(text)
        except Exception:
            return default

    def _on_save_click(e: ft.ControlEvent):
        try:
            ds_block = dataset_block
            selected = None
            if ds_block and hasattr(ds_block, 'get_selected_dataset'):
                selected = ds_block.get_selected_dataset()
            if not selected:
                if e and e.page:
                    e.page.snack_bar = ft.SnackBar(content=ft.Text("Select a dataset before saving."), open=True)
                    e.page.update()
                return

            base_dir, dataset_type = _get_dataset_base_dir(selected)
            clean_dataset_name = str(selected)
            dataset_full_path = os.path.join(base_dir, clean_dataset_name)
            parent_dir = os.path.dirname(dataset_full_path)
            out_toml_path = os.path.join(parent_dir, f"{clean_dataset_name}.toml")

            # Collect values
            resolutions_val = _parse_list_field(resolutions_field.value or "[]", [])
            ar_buckets_val = _parse_list_field(ar_buckets_field.value or "[]", [])
            enable_ar_bucket_val = bool(enable_ar_bucket_field.value)
            min_ar_val = float(min_ar_field.value) if (min_ar_field.value or "").strip() != "" else 0.0
            max_ar_val = float(max_ar_field.value) if (max_ar_field.value or "").strip() != "" else 0.0
            num_ar_buckets_val = int(num_ar_buckets_field.value) if (num_ar_buckets_field.value or "").strip() != "" else 0
            # num_repeats from dataset block
            if hasattr(ds_block, 'get_num_repeats'):
                num_repeats_val = int(ds_block.get_num_repeats())
            elif hasattr(ds_block, 'get_num_workers'):
                num_repeats_val = int(ds_block.get_num_workers())
            else:
                num_repeats_val = 1

            # Build TOML string
            def _fmt_list(lst):
                return "[" + ", ".join(str(x) for x in lst) + "]"
            def _fmt_list_of_lists(lst):
                def fmt_pair(p):
                    return "[" + ", ".join(str(x) for x in p) + "]"
                return "[" + ", ".join(fmt_pair(p) for p in lst) + "]"

            toml_lines = []
            toml_lines.append(f"resolutions = {_fmt_list(resolutions_val)}")
            toml_lines.append("")
            toml_lines.append(f"enable_ar_bucket = {'true' if enable_ar_bucket_val else 'false'}")
            toml_lines.append("")
            toml_lines.append("# Min and max aspect ratios, given as width/height ratio.")
            toml_lines.append(f"min_ar = {min_ar_val}")
            toml_lines.append(f"max_ar = {max_ar_val}")
            toml_lines.append(f"ar_buckets = {_fmt_list_of_lists(ar_buckets_val)}")
            toml_lines.append("")
            toml_lines.append("# Total number of aspect ratio buckets, evenly spaced (in log space) between min_ar and max_ar.")
            toml_lines.append(f"num_ar_buckets = {num_ar_buckets_val}")
            toml_lines.append(f"num_repeats = {num_repeats_val}")
            toml_lines.append("")
            # Frame buckets: always output; comment out if disabled
            try:
                fb_val = _parse_list_field(frame_buckets_field.value or "[]", [])
                def _fmt_list(lst):
                    return "[" + ", ".join(str(x) for x in lst) + "]"
                fb_line = f"frame_buckets = {_fmt_list(fb_val)}"
                if not bool(enable_frame_buckets_field.value):
                    fb_line = "# " + fb_line
                toml_lines.append(fb_line)
                toml_lines.append("")
            except Exception:
                pass
            toml_lines.append("[[directory]]")
            toml_lines.append("# The target images go in here. These are the images that the model will learn to produce.")
            # Use the selected dataset path
            dir_path_val = dataset_full_path
            # Escape backslashes minimally
            toml_lines.append(f"path = '{dir_path_val}'")
            # If Has control is on, write control_path = path + '/control'
            if bool(has_control_checkbox.value):
                control_path_val = os.path.join(dir_path_val, "control").replace("\\", "/")
                toml_lines.append(f"control_path = '{control_path_val}'")

            os.makedirs(parent_dir, exist_ok=True)
            with open(out_toml_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(toml_lines) + "\n")

            # Refresh TOML exists indicator after save
            try:
                toml_exists_value.value = "Yes"
                toml_exists_value.color = ft.Colors.GREEN_400
                if toml_exists_value.page:
                    toml_exists_value.update()
            except Exception:
                pass

            if e and e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Saved: {out_toml_path}"), open=True)
                e.page.update()
        except Exception as ex:
            if e and e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving: {ex}"), open=True)
                e.page.update()

    # Row: Left = Data Configuration, Right = Dataset Selection (titles aligned)
    # WandB controls have been moved to Config tab under Optimizer panel.

    # Wire handler after controls initialized
    save_btn.on_click = _on_save_click

    # Then show the Data Configuration (now containing the Data Processing panel) + Dataset Selection row
    page_controls.append(
        ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Data Configuration"),
                ft.Container(
                    content=ft.Column([
                        # Embed Data Processing Configuration inside Data Configuration (header removed by request)
                        ft.Container(
                            content=ft.Row([
                                # Column 1
                                ft.Column([
                                    ft.Text("Aspect Ratio & Resolution", size=12, weight=ft.FontWeight.BOLD),
                                    ar_buckets_field,
                                    resolutions_field,
                                    ft.Row([
                                        frame_buckets_field,
                                        enable_frame_buckets_field,
                                    ], spacing=10),
                                ], spacing=10, expand=True),

                                ft.Container(width=20),  # Spacer

                                # Column 2
                                ft.Column([
                                    ft.Text("Aspect Ratio Settings", size=12, weight=ft.FontWeight.BOLD),
                                    ft.Row([
                                        ft.Text("num_ar: ", size=12),
                                        num_ar_buckets_field,
                                        enable_ar_bucket_field,
                                    ], spacing=10),
                                    ft.Row([
                                        ft.Text("min_ar: ", size=12),
                                        min_ar_field,
                                    ], spacing=5),
                                    ft.Row([
                                        ft.Text("max_ar: ", size=12),
                                        max_ar_field,
                                    ], spacing=5),
                                ], spacing=10, expand=True),
                            ], spacing=10, alignment=ft.MainAxisAlignment.START),
                            padding=ft.padding.all(10),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                            border_radius=ft.border_radius.all(10),
                        ),
                        ft.Row([
                            has_control_checkbox,
                            ft.Text("Data .toml exists", size=12, weight=ft.FontWeight.BOLD),
                            toml_exists_value,
                        ], spacing=14, alignment=ft.MainAxisAlignment.END, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ], spacing=10),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                )
            ], col=6),
            ft.Column([
                *add_section_title("Dataset Selection"),
                ft.Container(
                    content=dataset_block,
                    padding=ft.padding.only(left=0, right=10, top=10, bottom=10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),
        ], spacing=20)
    )

    # Footer spacing and Save button spacer
    page_controls.append(ft.Container(height=20))

    # Update "Data .toml exists" on dataset selection changes
    def _load_toml_and_populate(toml_path: str):
        try:
            if not os.path.exists(toml_path):
                return
            if _toml_reader is None:
                return  # Skip if tomllib not available; avoid adding deps
            with open(toml_path, 'rb') as f:
                data = _toml_reader.load(f)
            # Top-level fields
            if isinstance(data.get('resolutions'), list):
                try:
                    resolutions_field.value = json.dumps(data['resolutions'])
                    if resolutions_field.page: resolutions_field.update()
                except Exception:
                    pass
            if isinstance(data.get('enable_ar_bucket'), bool):
                try:
                    enable_ar_bucket_field.value = bool(data['enable_ar_bucket'])
                    if enable_ar_bucket_field.page: enable_ar_bucket_field.update()
                except Exception:
                    pass
            if data.get('min_ar') is not None:
                try:
                    min_ar_field.value = str(data['min_ar'])
                    if min_ar_field.page: min_ar_field.update()
                except Exception:
                    pass
            if data.get('max_ar') is not None:
                try:
                    max_ar_field.value = str(data['max_ar'])
                    if max_ar_field.page: max_ar_field.update()
                except Exception:
                    pass
            if isinstance(data.get('ar_buckets'), list):
                try:
                    ar_buckets_field.value = json.dumps(data['ar_buckets'])
                    if ar_buckets_field.page: ar_buckets_field.update()
                except Exception:
                    pass
            if data.get('num_ar_buckets') is not None:
                try:
                    num_ar_buckets_field.value = str(data['num_ar_buckets'])
                    if num_ar_buckets_field.page: num_ar_buckets_field.update()
                except Exception:
                    pass
            # frame_buckets controls
            if isinstance(data.get('frame_buckets'), list):
                try:
                    frame_buckets_field.value = json.dumps(data['frame_buckets'])
                    if frame_buckets_field.page: frame_buckets_field.update()
                    enable_frame_buckets_field.value = True
                    if enable_frame_buckets_field.page: enable_frame_buckets_field.update()
                except Exception:
                    pass
            # num_repeats may be at top-level per our saved format
            if data.get('num_repeats') is not None:
                try:
                    num_repeats_val = int(data['num_repeats'])
                    if hasattr(dataset_block, 'set_num_repeats'):
                        dataset_block.set_num_repeats(num_repeats_val)

                    # Also try to directly update the UI field for better reliability
                    # Access the num_workers field (which is used for num_repeats)
                    if hasattr(dataset_block, 'num_workers_field_ref'):
                        num_workers_field_ref = getattr(dataset_block, 'num_workers_field_ref', None)
                        if num_workers_field_ref and num_workers_field_ref.current:
                            num_workers_field_ref.current.value = str(num_repeats_val)
                            if num_workers_field_ref.current.page:
                                num_workers_field_ref.current.update()
                except Exception as e:
                    print(f"Error setting num_repeats from TOML: {e}")
                    pass
            # directory array
            dirs = data.get('directory')
            if isinstance(dirs, list) and len(dirs) > 0 and isinstance(dirs[0], dict):
                d0 = dirs[0]
                # control_path presence controls checkbox
                has_ctrl = 'control_path' in d0 and bool(d0.get('control_path'))
                try:
                    has_control_checkbox.value = bool(has_ctrl)
                    if has_control_checkbox.page: has_control_checkbox.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_dataset_change_for_indicator(selected_dataset_name):
        try:
            exists = False
            if selected_dataset_name:
                base_dir, _ = _get_dataset_base_dir(selected_dataset_name)
                clean = str(selected_dataset_name)
                ds_full = os.path.join(base_dir, clean)
                parent_dir = os.path.dirname(ds_full)
                out_path = os.path.join(parent_dir, f"{clean}.toml")
                exists = os.path.exists(out_path)
                if exists:
                    _load_toml_and_populate(out_path)
            if not exists:
                try:
                    has_control_checkbox.value = False
                    if has_control_checkbox.page: has_control_checkbox.update()
                except Exception:
                    pass
            toml_exists_value.value = "Yes" if exists else "No"
            toml_exists_value.color = ft.Colors.GREEN_400 if exists else ft.Colors.RED_400
            if toml_exists_value.page:
                toml_exists_value.update()
        except Exception:
            pass

    # Save button moved above under Data Configuration, aligned right

    # Hook indicator to dataset selection changes and initialize state
    try:
        if hasattr(dataset_block, 'add_on_selection_change'):
            dataset_block.add_on_selection_change(_on_dataset_change_for_indicator)
        if hasattr(dataset_block, 'get_selected_dataset'):
            _on_dataset_change_for_indicator(dataset_block.get_selected_dataset())
        else:
            _on_dataset_change_for_indicator(None)
    except Exception:
        pass

    container = ft.Container(
        content=ft.Column(page_controls, spacing=8, scroll=ft.ScrollMode.AUTO),
        padding=ft.padding.all(5),
        expand=True,
    )

    # Expose dataset block for cross-page sync and a refresh method for TOML indicator
    container.dataset_block = dataset_block
    try:
        container.refresh_indicator = _on_dataset_change_for_indicator
    except Exception:
        pass
    return container
