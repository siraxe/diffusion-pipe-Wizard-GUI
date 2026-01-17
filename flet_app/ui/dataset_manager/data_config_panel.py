"""
Modular Data Configuration Panel for Dataset Layout Tab.
Contains aspect ratio, resolution, and frame bucket controls with auto-save TOML functionality.
"""
import flet as ft
import os
import json
import re
from typing import Any
try:
    import tomllib as _toml_reader  # Python 3.11+
except Exception:
    _toml_reader = None

from flet_app.ui._styles import create_textfield
from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir


# ======================================================================================
# Helper Functions for TOML Parsing and Formatting
# ======================================================================================

def _parse_list_field(value, default):
    """Parse a list field from string (handles JSON format)."""
    if not value or not value.strip():
        return default
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else default
    except Exception:
        return default


def _fmt_list(lst):
    """Format a list for TOML output."""
    return json.dumps(lst)


def _fmt_list_of_lists(list_of_lists):
    """Format a list of lists for TOML output."""
    return json.dumps(list_of_lists)


# ======================================================================================
# Data Configuration Panel Creation
# ======================================================================================

def create_data_config_panel(upload_button=None):
    """
    Create a modular Data Configuration panel with all UI elements.

    Args:
        upload_button: Optional upload button to add to the control row.

    Returns:
        tuple: (centered_panel, show_button, controls_dict, on_dataset_change_fn) where:
               - centered_panel: The panel with centered layout
               - show_button: Button to show the hidden panel
               - controls_dict: References to all input fields
               - on_dataset_change_fn: Function to call when dataset changes
    """
    # Create all the input fields
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
        "[33,81]",
        hint_text="Frame bucket values",
        expand=True
    )

    enable_frame_buckets_field = ft.Checkbox(
        label="frame_buckets",
        value=False,
        label_style=ft.TextStyle(size=12)
    )

    num_ar_buckets_field = create_textfield(
        "num_ar",
        "7",
        hint_text="Number of AR buckets",
        expand=True
    )

    enable_ar_bucket_field = ft.Checkbox(
        label="ar_buckets",
        value=True,
        label_style=ft.TextStyle(size=12)
    )

    min_ar_field = create_textfield(
        "min_ar",
        "0.5",
        hint_text="Minimum aspect ratio",
        expand=True
    )

    max_ar_field = create_textfield(
        "max_ar",
        "2.0",
        hint_text="Maximum aspect ratio",
        expand=True
    )

    has_control_checkbox = ft.Checkbox(
        label="Has control",
        value=False,
        label_style=ft.TextStyle(size=12)
    )

    toml_exists_value = ft.Text("No", size=12, color=ft.Colors.RED_400)

    # Create X button to hide the panel
    hide_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        icon_color=ft.Colors.GREY_400,
        icon_size=16,
        tooltip="Hide panel",
        on_click=None,  # Will be set after creating the toggle container
    )

    # ======================================================================================
    # TOML Loading Logic
    # ======================================================================================

    def _load_toml_and_populate(toml_path: str):
        """Load TOML file and populate all fields."""
        try:
            if not os.path.exists(toml_path):
                return
            if _toml_reader is None:
                return
            with open(toml_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            data = {}
            try:
                loads_fn = getattr(_toml_reader, 'loads', None)
                if loads_fn:
                    data = loads_fn(raw_text)
                else:
                    with open(toml_path, 'rb') as fb:
                        data = _toml_reader.load(fb)
            except Exception:
                data = {}

            # Check for commented fields using regex
            resolutions_commented = bool(re.search(r'^[ 	]*#[ 	]*resolutions[ 	]*=.*', raw_text, re.MULTILINE))
            ar_buckets_commented = bool(re.search(r'^[ 	]*#[ 	]*ar_buckets[ 	]*=.*', raw_text, re.MULTILINE))
            frame_buckets_commented = bool(re.search(r'^[ 	]*#[ 	]*frame_buckets[ 	]*=.*', raw_text, re.MULTILINE))

            # Extract values directly from raw text using regex (works even with invalid JSON)
            # This allows loading fields even when tomllib would fail on the whole file

            # Extract resolutions
            if not resolutions_commented:
                match = re.search(r'^[ 	]*resolutions[ 	]*=[ 	]*(.*)', raw_text, re.MULTILINE)
                if match:
                    resolutions_field.value = match.group(1).strip()
                    if resolutions_field.page:
                        resolutions_field.update()

            # Extract enable_ar_bucket
            match = re.search(r'^[ 	]*enable_ar_bucket[ ]*=[ ]*(true|false)', raw_text, re.MULTILINE)
            if match:
                enable_ar_bucket_field.value = match.group(1) == 'true'
                if enable_ar_bucket_field.page:
                    enable_ar_bucket_field.update()

            # Extract min_ar and max_ar
            match = re.search(r'^[ 	]*min_ar[ ]*=[ ]*([0-9.]+)', raw_text, re.MULTILINE)
            if match:
                min_ar_field.value = match.group(1)
                if min_ar_field.page:
                    min_ar_field.update()

            match = re.search(r'^[ 	]*max_ar[ ]*=[ ]*([0-9.]+)', raw_text, re.MULTILINE)
            if match:
                max_ar_field.value = match.group(1)
                if max_ar_field.page:
                    max_ar_field.update()

            # Extract ar_buckets
            if not ar_buckets_commented:
                match = re.search(r'^[ 	]*ar_buckets[ ]*=[ ]*(.*)', raw_text, re.MULTILINE)
                if match:
                    ar_buckets_field.value = match.group(1).strip()
                    if ar_buckets_field.page:
                        ar_buckets_field.update()

            # Extract num_ar_buckets
            match = re.search(r'^[ 	]*num_ar_buckets[ ]*=[ ]*([0-9]+)', raw_text, re.MULTILINE)
            if match:
                num_ar_buckets_field.value = match.group(1)
                if num_ar_buckets_field.page:
                    num_ar_buckets_field.update()

            # Extract frame_buckets and enable state
            # Check if the line is commented
            fb_commented = frame_buckets_commented
            if fb_commented:
                match = re.search(r'^[ 	]*#[ 	]*frame_buckets[ 	]*=[ ]*(\[.*\])', raw_text, re.MULTILINE)
                if match:
                    frame_buckets_field.value = match.group(1).strip()
                    if frame_buckets_field.page:
                        frame_buckets_field.update()
                    enable_frame_buckets_field.value = False
                    if enable_frame_buckets_field.page:
                        enable_frame_buckets_field.update()
            else:
                match = re.search(r'^[ 	]*frame_buckets[ ]*=[ ]*(.*)', raw_text, re.MULTILINE)
                if match:
                    frame_buckets_field.value = match.group(1).strip()
                    if frame_buckets_field.page:
                        frame_buckets_field.update()
                    enable_frame_buckets_field.value = True
                    if enable_frame_buckets_field.page:
                        enable_frame_buckets_field.update()

            # Handle has_control from directory section
            match = re.search(r'^[ 	]*control_path[ ]*=[ ]*(.*)', raw_text, re.MULTILINE)
            if match:
                has_control_checkbox.value = True
                if has_control_checkbox.page:
                    has_control_checkbox.update()
            else:
                has_control_checkbox.value = False
                if has_control_checkbox.page:
                    has_control_checkbox.update()

        except Exception as ex:
            print(f"Error loading TOML: {ex}")

    # ======================================================================================
    # TOML Auto-Save Logic
    # ======================================================================================

    def _save_toml_on_change(e=None):
        """Auto-save TOML when any field changes."""
        try:
            # Get selected dataset from page
            page = e.page if e else None
            if not page:
                return

            selected_dataset = getattr(page, 'selected_dataset', None)
            if not selected_dataset:
                return

            # Build paths
            base_dir, _ = _get_dataset_base_dir(selected_dataset)
            clean_dataset_name = str(selected_dataset)
            dataset_full_path = os.path.join(base_dir, clean_dataset_name)
            parent_dir = os.path.dirname(dataset_full_path)
            out_toml_path = os.path.join(parent_dir, f"{clean_dataset_name}.toml")

            # Read existing TOML to preserve num_repeats and other settings
            existing_num_repeats = 1
            existing_frame_extraction = None
            if os.path.exists(out_toml_path):
                try:
                    with open(out_toml_path, 'r', encoding='utf-8') as f:
                        raw_text = f.read()
                    # Extract num_repeats from [[directory]] section using regex
                    # Look for num_repeats = value after [[directory]] and before next [[directory]] or end
                    match = re.search(r'^\[\[directory\]\].*?^num_repeats[ ]*=[ ]*([0-9]+)', raw_text, re.MULTILINE | re.DOTALL)
                    if match:
                        existing_num_repeats = int(match.group(1))
                    # Extract frame_extraction if present
                    match = re.search(r'^\[\[directory\]\].*?^frame_extraction[ ]*=[ ]*\"([^\"]+)\"', raw_text, re.MULTILINE | re.DOTALL)
                    if match:
                        existing_frame_extraction = match.group(1)
                except Exception:
                    pass

            # Collect values from fields
            resolutions_raw = (resolutions_field.value or "").strip()
            ar_buckets_raw = (ar_buckets_field.value or "").strip()
            frame_buckets_raw = (frame_buckets_field.value or "").strip()
            enable_ar_bucket_val = bool(enable_ar_bucket_field.value)
            min_ar_val = float(min_ar_field.value) if (min_ar_field.value or "").strip() != "" else 0.0
            max_ar_val = float(max_ar_field.value) if (max_ar_field.value or "").strip() != "" else 0.0
            num_ar_buckets_val = int(num_ar_buckets_field.value) if (num_ar_buckets_field.value or "").strip() != "" else 0

            # Build TOML content
            toml_lines = []

            # resolutions (save raw value, or empty list if truly empty)
            if resolutions_raw:
                toml_lines.append(f"resolutions = {resolutions_raw}")
            else:
                toml_lines.append("resolutions = []")
            toml_lines.append("")

            # aspect ratio settings
            toml_lines.append(f"enable_ar_bucket = {'true' if enable_ar_bucket_val else 'false'}")
            toml_lines.append(f"min_ar = {min_ar_val}")
            toml_lines.append(f"max_ar = {max_ar_val}")

            # ar_buckets (save raw value, or comment out if empty)
            if ar_buckets_raw:
                toml_lines.append(f"ar_buckets = {ar_buckets_raw}")
            else:
                toml_lines.append("# ar_buckets = []")

            toml_lines.append(f"num_ar_buckets = {num_ar_buckets_val}")
            toml_lines.append("")

            # frame_buckets (save raw value, comment out if disabled)
            if frame_buckets_raw:
                fb_line = f"frame_buckets = {frame_buckets_raw}"
                if not bool(enable_frame_buckets_field.value):
                    fb_line = "# " + fb_line
                toml_lines.append(fb_line)
                toml_lines.append("")
            else:
                fb_line = "frame_buckets = []"
                if not bool(enable_frame_buckets_field.value):
                    fb_line = "# " + fb_line
                toml_lines.append(fb_line)
                toml_lines.append("")

            # Get dataset path for directory section
            dir_path_val = clean_dataset_name
            toml_lines.append("[[directory]]")
            toml_lines.append(f"path = '{dir_path_val}'")

            # Preserve num_repeats from existing TOML (set from config page)
            toml_lines.append(f"num_repeats = {existing_num_repeats}")

            # Preserve frame_extraction from existing TOML if present
            if existing_frame_extraction:
                toml_lines.append(f"frame_extraction = \"{existing_frame_extraction}\"")

            # Add control_path if has_control is checked
            if bool(has_control_checkbox.value):
                control_path_val = os.path.join(dir_path_val, "control").replace("\\", "/")
                toml_lines.append(f"control_path = '{control_path_val}'")

            # Write to file
            os.makedirs(parent_dir, exist_ok=True)
            with open(out_toml_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(toml_lines) + "\n")

            # Update indicator
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

    # ======================================================================================
    # Dataset Change Handler
    # ======================================================================================

    def _on_dataset_change(selected_dataset_name):
        """Handle dataset selection change - load TOML values for new dataset."""
        try:
            if not selected_dataset_name:
                # Clear fields when no dataset selected
                try:
                    ar_buckets_field.value = "[[448, 576]]"
                    resolutions_field.value = "[512]"
                    frame_buckets_field.value = "[33,81]"
                    num_ar_buckets_field.value = "7"
                    min_ar_field.value = "0.5"
                    max_ar_field.value = "2.0"
                    enable_ar_bucket_field.value = True
                    enable_frame_buckets_field.value = False
                    has_control_checkbox.value = False
                    toml_exists_value.value = "No"
                    toml_exists_value.color = ft.Colors.RED_400

                    # Update all fields
                    for field in [ar_buckets_field, resolutions_field, frame_buckets_field,
                                  num_ar_buckets_field, min_ar_field, max_ar_field]:
                        if field.page:
                            field.update()
                    for chk in [enable_ar_bucket_field, enable_frame_buckets_field, has_control_checkbox]:
                        if chk.page:
                            chk.update()
                    if toml_exists_value.page:
                        toml_exists_value.update()
                except Exception:
                    pass
                return

            # Check if TOML exists
            base_dir, _ = _get_dataset_base_dir(selected_dataset_name)
            clean = str(selected_dataset_name)
            ds_full = os.path.join(base_dir, clean)
            parent_dir = os.path.dirname(ds_full)
            out_path = os.path.join(parent_dir, f"{clean}.toml")
            exists = os.path.exists(out_path)

            # Load TOML if exists
            if exists:
                _load_toml_and_populate(out_path)

            # Update indicator
            try:
                toml_exists_value.value = "Yes" if exists else "No"
                toml_exists_value.color = ft.Colors.GREEN_400 if exists else ft.Colors.RED_400
                if toml_exists_value.page:
                    toml_exists_value.update()
            except Exception:
                pass

        except Exception as ex:
            print(f"Error in dataset change handler: {ex}")

    # ======================================================================================
    # Attach Change Handlers to All Fields
    # ======================================================================================

    # Text fields
    ar_buckets_field.on_change = _save_toml_on_change
    resolutions_field.on_change = _save_toml_on_change
    frame_buckets_field.on_change = _save_toml_on_change
    num_ar_buckets_field.on_change = _save_toml_on_change
    min_ar_field.on_change = _save_toml_on_change
    max_ar_field.on_change = _save_toml_on_change

    # Checkboxes
    has_control_checkbox.on_change = _save_toml_on_change
    enable_frame_buckets_field.on_change = _save_toml_on_change
    enable_ar_bucket_field.on_change = _save_toml_on_change

    # ======================================================================================
    # Build UI
    # ======================================================================================

    # Inner content with the 3 columns
    inner_content = ft.ResponsiveRow([
        # Column 1: Aspect Ratio & Resolution (col=4.5)
        ft.Container(
            content=ft.Column([
                ft.Text("Aspect Ratio & Resolution", size=11, weight=ft.FontWeight.BOLD),
                ar_buckets_field,
                resolutions_field,
                ft.Row([
                    frame_buckets_field,
                    enable_frame_buckets_field,
                ], spacing=10),
            ], spacing=8),
            col=4.5,
        ),

        # Column 2: Aspect Ratio Settings (col=4.5)
        ft.Container(
            content=ft.Column([
                ft.Text("Aspect Ratio Settings", size=11, weight=ft.FontWeight.BOLD),
                ft.Row([
                    ft.Text("num_ar: ", size=11),
                    num_ar_buckets_field,
                    enable_ar_bucket_field,
                ], spacing=8),
                ft.Row([
                    ft.Text("min_ar: ", size=11),
                    min_ar_field,
                ], spacing=5),
                ft.Row([
                    ft.Text("max_ar: ", size=11),
                    max_ar_field,
                ], spacing=5),
            ], spacing=8),
            col=4.5,
        ),

        # Column 3: Upload and controls (col=3, no name)
        ft.Container(
            content=ft.Column([
                # Row 1: Upload Files button
                upload_button if upload_button else ft.Container(),
                # Row 2: Data .toml exists indicator
                ft.Row([
                    ft.Text("Data .toml exists", size=11, weight=ft.FontWeight.BOLD),
                    toml_exists_value,
                ], spacing=5, alignment=ft.MainAxisAlignment.START),
                # Row 3: Has control checkbox
                has_control_checkbox,
                # Row 4: X button to hide panel
                ft.Row([
                    hide_button,
                ], alignment=ft.MainAxisAlignment.END),
            ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.START),
            col=3,
        ),
    ], spacing=10, alignment=ft.MainAxisAlignment.START)

    # Build the panel container with solid background (no opacity)
    panel = ft.Container(
        content=inner_content,
        padding=ft.padding.all(8),
        bgcolor=ft.Colors.SURFACE,
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=ft.border_radius.all(8),
    )

    # Create show button (cog icon, initially hidden - shown when panel is hidden)
    show_button = ft.IconButton(
        icon=ft.Icons.SETTINGS,
        on_click=None,  # Will be set after creating the toggle container
        visible=False,  # Initially hidden
        icon_color=ft.Colors.GREY_400,
        icon_size=20,
        tooltip="Show Data Config",
    )

    # Wrap panel in a centered row
    centered_panel = ft.ResponsiveRow([
        # Left spacer
        ft.Container(col=4.9),
        # Panel (centered with border)
        ft.Container(content=panel, col=7),
        # Right spacer
        ft.Container(col=0.1),
    ])

    # Define toggle functions
    def hide_panel(e):
        panel.visible = False
        show_button.visible = True
        if e and e.page:
            e.page.update()

    def show_panel(e):
        panel.visible = True
        show_button.visible = False
        if e and e.page:
            e.page.update()

    # Set the on_click handlers after defining the functions
    hide_button.on_click = hide_panel
    show_button.on_click = show_panel

    # Return panel container, show button, controls dict, and dataset change handler
    controls = {
        'ar_buckets_field': ar_buckets_field,
        'resolutions_field': resolutions_field,
        'frame_buckets_field': frame_buckets_field,
        'enable_frame_buckets_field': enable_frame_buckets_field,
        'num_ar_buckets_field': num_ar_buckets_field,
        'enable_ar_bucket_field': enable_ar_bucket_field,
        'min_ar_field': min_ar_field,
        'max_ar_field': max_ar_field,
        'has_control_checkbox': has_control_checkbox,
        'toml_exists_value': toml_exists_value,
    }

    return centered_panel, show_button, controls, _on_dataset_change
