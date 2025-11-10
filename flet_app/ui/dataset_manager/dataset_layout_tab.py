import flet as ft
import os
import json
import asyncio
import time
import shutil
from pathlib import Path
from flet_app.settings import settings

from flet_app.ui._styles import create_dropdown, create_styled_button, create_textfield, BTN_STYLE2
import flet_app.ui.dataset_manager.dataset_utils as dataset_utils
from flet_app.ui.dataset_manager.dataset_utils import (
    get_dataset_folders,
    get_videos_and_thumbnails,
    load_dataset_captions
)
from flet_app.ui.dataset_manager.dataset_thumb_layout import create_thumbnail_container, set_thumbnail_selection_state
from flet_app.ui.dataset_manager.dataset_actions import (
    on_change_fps_click, on_rename_files_click, on_bucket_or_model_change,
    on_add_captions_click_with_model, stop_captioning, on_delete_captions_click,
    perform_delete_captions,
    apply_affix_from_textfield, find_and_replace_in_captions, on_caption_to_json_click, on_caption_to_txt_click
)
from flet_app.ui.dataset_manager.dataset_controls import build_expansion_tile
from flet_app.ui.flet_hotkeys import is_d_key_pressed_global # Import global D key state

# ======================================================================================
# Global State (Keep track of UI controls and running processes)
# ======================================================================================

# References to UI controls
selected_dataset = {"value": None}
DATASETS_TYPE = {"value": None} # "image" or "video"
video_files_list = {"value": []}
thumbnails_grid_ref = ft.Ref[ft.GridView]()
refresh_button_ref = ft.Ref[ft.IconButton]()
thumbnails_refresh_in_progress = False
thumbnails_pending_refresh = False
thumbnails_pending_force_refresh = False
processed_progress_bar = ft.ProgressBar(visible=False)
processed_output_field = ft.TextField(
    label="Processed Output", text_size=10, multiline=True, read_only=True,
    visible=False, min_lines=6, max_lines=15, expand=True)
pending_uploads_count = {"value": 0}

# Multi-selection state
selected_thumbnails_set = set() # Stores video_path of selected thumbnails
last_clicked_thumbnail_index = -1 # Stores the index of the last clicked checkbox

# ABC Action Container (A, Duplicate, Delete)
abc_action_container = None

# Track if we're currently in the dataset tab
is_in_dataset_tab = {"value": False}

# Global controls (defined here but created in _create_global_controls)
bucket_size_textfield: ft.TextField = None
rename_textfield: ft.TextField = None
model_name_dropdown: ft.Dropdown = None
trigger_word_textfield: ft.TextField = None

# Global sorting state for datasets
dataset_sort_mode = {"value": "newest"}  # "newest", "oldest", "name_asc", "name_desc"
dataset_sort_controls_container: ft.Container | None = None

# References to controls created in dataset_tab_layout that need external access
dataset_dropdown_control_ref = ft.Ref[ft.Dropdown]()
dataset_dropdown_disabled_by_refresh = False
dataset_add_captions_button_ref = ft.Ref[ft.ElevatedButton]()
dataset_delete_captions_button_ref = ft.Ref[ft.ElevatedButton]()
caption_model_dropdown_ref = ft.Ref[ft.Dropdown]()
captions_checkbox_ref = ft.Ref[ft.Checkbox]() # 8-bit for LLaVA
hf_checkbox_ref = ft.Ref[ft.Checkbox]() # HF backend for Qwen
cap_command_textfield_ref = ft.Ref[ft.TextField]()
max_tokens_textfield_ref = ft.Ref[ft.TextField]()
change_fps_textfield_ref = ft.Ref[ft.TextField]() # Ref for the Change FPS textfield
joy_prompt_dropdown_ref = ft.Ref[ft.Dropdown]()

affix_text_field_ref = ft.Ref[ft.TextField]()
find_text_field_ref = ft.Ref[ft.TextField]()
replace_text_field_ref = ft.Ref[ft.TextField]()

# ======================================================================================
# GUI Update/Utility Functions (Functions that update the UI state)
# ======================================================================================

def set_bottom_app_bar_height():
    global bottom_app_bar_ref
    if bottom_app_bar_ref is not None and bottom_app_bar_ref.page:
        if processed_output_field.visible:
            bottom_app_bar_ref.height = 240
        else:
            bottom_app_bar_ref.height = 0
        bottom_app_bar_ref.update()

def _resolve_page_context(page_ctx: ft.Page | None, grid_control: ft.GridView | None) -> ft.Page | None:
    """Best effort page lookup so we can update UI state even when callers pass None."""
    if page_ctx:
        return page_ctx
    if grid_control and grid_control.page:
        return grid_control.page
    if refresh_button_ref.current and refresh_button_ref.current.page:
        return refresh_button_ref.current.page
    if dataset_dropdown_control_ref.current and dataset_dropdown_control_ref.current.page:
        return dataset_dropdown_control_ref.current.page
    return None


def _set_refresh_ui_state(is_running: bool, page_ctx: ft.Page | None, grid_control: ft.GridView | None):
    """Toggle refresh button/dropdown state while thumbnails rebuild."""
    global dataset_dropdown_disabled_by_refresh

    refresh_btn = refresh_button_ref.current
    if refresh_btn:
        refresh_btn.disabled = is_running
        refresh_btn.icon = ft.Icons.HOURGLASS_TOP if is_running else ft.Icons.REFRESH
        refresh_btn.tooltip = "Refreshing thumbnails..." if is_running else "Update dataset list and refresh thumbnails"
        if refresh_btn.page:
            refresh_btn.update()

    dropdown = dataset_dropdown_control_ref.current
    if dropdown:
        if is_running:
            if not dropdown.disabled:
                dataset_dropdown_disabled_by_refresh = True
            dropdown.disabled = True
        elif dataset_dropdown_disabled_by_refresh:
            dropdown.disabled = False
            dataset_dropdown_disabled_by_refresh = False
        if dropdown.page:
            dropdown.update()


def _build_thumbnails_loading_indicator(message: str = "Refreshing thumbnails...") -> ft.Container:
    return ft.Container(
        content=ft.Row(
            [
                ft.ProgressRing(width=18, height=18, stroke_width=2),
                ft.Text(message, size=11, color=ft.Colors.GREY_600),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        ),
        padding=10,
    )


def _on_thumbnail_checkbox_change(video_path: str, is_checked: bool, thumbnail_index: int, page_ctx=None):
    global selected_thumbnails_set, last_clicked_thumbnail_index
    import flet_app.ui.flet_hotkeys as ui_fh # Re-import to get a mutable reference to the module itself

    # Support range-select when Shift is held (web-safe) or legacy 'D' hotkey is used.
    is_range = False
    try:
        is_range = bool(getattr(ui_fh, 'is_shift_key_pressed_global', False)) or bool(ui_fh.is_d_key_pressed_global)
    except Exception:
        is_range = bool(ui.flet_hotkeys.is_d_key_pressed_global)

    if is_range and last_clicked_thumbnail_index != -1:
        start_index = min(last_clicked_thumbnail_index, thumbnail_index)
        end_index = max(last_clicked_thumbnail_index, thumbnail_index)

        for i in range(start_index, end_index + 1):
            if i < len(thumbnails_grid_ref.current.controls):
                control = thumbnails_grid_ref.current.controls[i]
                if isinstance(control, ft.Container) and isinstance(control.content, ft.Stack) and len(control.content.controls) > 1:
                    set_thumbnail_selection_state(control, is_checked)
                    
                    if is_checked:
                        selected_thumbnails_set.add(control.data)
                    else:
                        selected_thumbnails_set.discard(control.data)
        # Reset hotkey flags after consuming a range selection
        try:
            ui_fh.is_d_key_pressed_global = False
            if hasattr(ui_fh, 'is_shift_key_pressed_global'):
                ui_fh.is_shift_key_pressed_global = False
        except Exception:
            pass

    else:
        if is_checked:
            selected_thumbnails_set.add(video_path)
        else:
            selected_thumbnails_set.discard(video_path)

    last_clicked_thumbnail_index = thumbnail_index

    # Update ABC container visibility based on selection
    try:
        if page_ctx and hasattr(page_ctx, 'abc_container'):
            # Show ABC container if there are selected thumbnails
            page_ctx.abc_container.visible = len(selected_thumbnails_set) > 0
            page_ctx.abc_container.update()

        # Update local ABC container visibility
        update_abc_container_visibility()
    except Exception:
        # If page is not available or update fails, continue without error
        pass

def cleanup_old_temp_thumbnails(thumb_dir: str, max_age_seconds: int = 3600):
    current_time = time.time()
    if not os.path.exists(thumb_dir):
        return

    for filename in os.listdir(thumb_dir):
        if filename.endswith('.tmp_'):
            try:
                file_path = os.path.join(thumb_dir, filename)
                file_mtime = os.path.getmtime(file_path)
                if current_time - file_mtime > max_age_seconds:
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up old temp thumbnail {filename}: {e}")

def handle_dataset_sort_change(e):
    """Keep the dataset thumbnails sorted when the dropdown changes"""
    try:
        dataset_sort_mode["value"] = e.control.value
        if thumbnails_grid_ref and thumbnails_grid_ref.current and e.page:
            e.page.run_task(update_thumbnails, page_ctx=e.page, grid_control=thumbnails_grid_ref.current)
    except Exception as ex:
        print(f"Error in sort change: {ex}")

# ======================================================================================
# ABC Action Container Functions (A, Duplicate, Delete)
# ======================================================================================

def create_abc_action_container():
    """Create the A/B/C action container with duplicate and delete functionality"""
    global abc_action_container

    def on_duplicate_click(e):
        """Handle duplicate button click"""
        try:
            if not selected_thumbnails_set or not selected_dataset:
                return

            # Get selected items
            selected_items = list(selected_thumbnails_set)
            if selected_items:
                current_dataset = selected_dataset.get("value")
                if current_dataset:
                    from .dataset_utils import _get_dataset_base_dir
                    base_dir, _ = _get_dataset_base_dir(current_dataset)
                    dataset_path = os.path.join(base_dir, current_dataset)

                    duplicated_count = 0

                    for item_path in selected_items:
                        try:
                            item_name = os.path.basename(item_path)
                            name, ext = os.path.splitext(item_name)
                            new_name = f"{name}_copy{ext}"
                            new_path = os.path.join(dataset_path, new_name)

                            # Copy the file
                            shutil.copy2(item_path, new_path)
                            duplicated_count += 1

                            # Also copy .txt file if it exists
                            txt_path = os.path.splitext(item_path)[0] + '.txt'
                            if os.path.exists(txt_path):
                                txt_new_name = f"{name}_copy.txt"
                                txt_new_path = os.path.join(dataset_path, txt_new_name)
                                shutil.copy2(txt_path, txt_new_path)

                        except Exception as copy_error:
                            print(f"Error duplicating {item_path}: {copy_error}")

                    # Show success message
                    if duplicated_count > 0:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text(f"Duplicated {duplicated_count} item(s)"),
                            open=True
                        )
                    else:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text("No items duplicated"),
                            open=True
                        )

                    # Clear selections and refresh thumbnails
                    selected_thumbnails_set.clear()
                    global last_clicked_thumbnail_index
                    last_clicked_thumbnail_index = -1

                    # Refresh thumbnails
                    if thumbnails_grid_ref and thumbnails_grid_ref.current:
                        e.page.run_task(update_thumbnails,
                                       page_ctx=e.page,
                                       grid_control=thumbnails_grid_ref.current,
                                       force_refresh=True)

                    # Hide the container after operation
                    abc_action_container.visible = False
                    abc_action_container.update()
                    e.page.update()
        except Exception as ex:
            print(f"Error in duplicate click: {ex}")
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Error: {str(ex)}"),
                    open=True
                )
                e.page.update()

    def on_delete_click(e):
        """Handle delete button click"""
        try:
            if not selected_thumbnails_set or not selected_dataset:
                return

            # Get selected items
            selected_items = list(selected_thumbnails_set)
            if selected_items:
                current_dataset = selected_dataset.get("value")
                if current_dataset:
                    from .dataset_utils import _get_dataset_base_dir
                    base_dir, _ = _get_dataset_base_dir(current_dataset)
                    dataset_path = os.path.join(base_dir, current_dataset)

                    deleted_count = 0

                    for item_path in selected_items:
                        try:
                            # Delete the main file
                            if os.path.exists(item_path):
                                os.remove(item_path)
                                deleted_count += 1

                            # Also delete .txt file if it exists
                            txt_path = os.path.splitext(item_path)[0] + '.txt'
                            if os.path.exists(txt_path):
                                os.remove(txt_path)

                            # Delete the associated thumbnail
                            current_dataset = selected_dataset.get("value")
                            if current_dataset:
                                # Get thumbnail path using correct thumbnail directory
                                item_name = os.path.basename(item_path)
                                thumbnail_name = f"{os.path.splitext(item_name)[0]}.jpg"
                                thumbnail_path = os.path.join(settings.THUMBNAILS_BASE_DIR, current_dataset, thumbnail_name)

                                if os.path.exists(thumbnail_path):
                                    os.remove(thumbnail_path)
                                    print(f"Deleted thumbnail: {thumbnail_path}")

                        except Exception as delete_error:
                            print(f"Error deleting {item_path}: {delete_error}")

                    # Show success message
                    if deleted_count > 0:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text(f"Deleted {deleted_count} item(s)"),
                            open=True
                        )
                    else:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text("No items deleted"),
                            open=True
                        )

                    # Clear selections and refresh thumbnails
                    selected_thumbnails_set.clear()
                    global last_clicked_thumbnail_index
                    last_clicked_thumbnail_index = -1

                    # Refresh thumbnails
                    if thumbnails_grid_ref and thumbnails_grid_ref.current:
                        e.page.run_task(update_thumbnails,
                                       page_ctx=e.page,
                                       grid_control=thumbnails_grid_ref.current,
                                       force_refresh=True)

                    # Hide the container after operation
                    abc_action_container.visible = False
                    abc_action_container.update()
                    e.page.update()
        except Exception as ex:
            print(f"Error in delete click: {ex}")
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Error: {str(ex)}"),
                    open=True
                )
                e.page.update()

    def on_download_click(e):
        """Handle download button click"""
        try:
            if not selected_thumbnails_set or not selected_dataset:
                return

            # Get selected items
            selected_items = list(selected_thumbnails_set)
            if selected_items:
                current_dataset = selected_dataset.get("value")
                if current_dataset:
                    from .dataset_utils import _get_dataset_base_dir
                    import zipfile
                    import tempfile
                    from flet_app.project_root import get_project_root

                    base_dir, _ = _get_dataset_base_dir(current_dataset)
                    dataset_path = os.path.join(base_dir, current_dataset)

                    # Get workspace path for temporary zip file
                    project_root = get_project_root()
                    workspace_path = os.path.join(project_root, "workspace")
                    os.makedirs(workspace_path, exist_ok=True)

                    # Create temporary zip file
                    import uuid
                    zip_filename = f"selected_items_{uuid.uuid4().hex[:8]}.zip"
                    zip_path = os.path.join(workspace_path, zip_filename)

                    items_count = 0

                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for item_path in selected_items:
                            try:
                                if os.path.exists(item_path):
                                    # Add main file to zip
                                    arcname = os.path.basename(item_path)
                                    zipf.write(item_path, arcname)
                                    items_count += 1

                                    # Also add .txt file if it exists
                                    txt_path = os.path.splitext(item_path)[0] + '.txt'
                                    if os.path.exists(txt_path):
                                        txt_arcname = os.path.basename(txt_path)
                                        zipf.write(txt_path, txt_arcname)

                            except Exception as zip_error:
                                print(f"Error adding {item_path} to zip: {zip_error}")

                    # Show success message
                    if items_count > 0:
                        # Copy zip to static output directory
                        from flet_app.settings import settings
                        project_location = settings.get("project_location")
                        if project_location:
                            output_dir = Path(project_location) / "workspace" / "output"
                            output_dir.mkdir(parents=True, exist_ok=True)

                            # Copy to output directory with timestamp
                            import shutil
                            import datetime
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            static_zip_name = f"download_{timestamp}.zip"
                            static_zip_path = output_dir / static_zip_name
                            shutil.copy2(zip_path, static_zip_path)

                            # Generate URL accessible through Flet's static file serving
                            download_url = f"http://localhost:8550/output/{static_zip_name}"

                            # Show download link
                            e.page.snack_bar = ft.SnackBar(
                                content=ft.Row([
                                    ft.Text(f"Created zip with {items_count} item(s)"),
                                    ft.ElevatedButton(
                                        "Download",
                                        icon=ft.Icons.DOWNLOAD,
                                        on_click=lambda _: e.page.launch_url(download_url),
                                        style=ft.ButtonStyle(
                                            bgcolor=ft.Colors.BLUE_600,
                                            color=ft.Colors.WHITE
                                        )
                                    )
                                ], spacing=10),
                                open=True
                            )
                    else:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text("No items to download"),
                            open=True
                        )

                    # Clean up zip file after 5 minutes (background task)
                    import threading
                    def cleanup_zip():
                        import time
                        time.sleep(300)  # 5 minutes
                        try:
                            if os.path.exists(zip_path):
                                os.remove(zip_path)
                                print(f"Cleaned up temporary zip file: {zip_path}")
                        except Exception:
                            pass

                    threading.Thread(target=cleanup_zip, daemon=True).start()

        except Exception as ex:
            print(f"Error in download click: {ex}")
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Error: {str(ex)}"),
                    open=True
                )
                e.page.update()

    abc_action_container = ft.Container(
        content=ft.Row(
            [
                ft.Row([
                    ft.IconButton(
                        icon=ft.Icons.DOWNLOAD,
                        on_click=on_download_click,
                        icon_color=ft.Colors.GREEN_600,
                        tooltip="Download selected items",
                        icon_size=20
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CONTENT_COPY,
                        on_click=on_duplicate_click,
                        icon_color=ft.Colors.BLUE_600,
                        tooltip="Duplicate selected items",
                        icon_size=20
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE,
                        on_click=on_delete_click,
                        icon_color=ft.Colors.RED_600,
                        tooltip="Delete selected items",
                        icon_size=20
                    ),
                ], spacing=8),
            ],
            alignment=ft.MainAxisAlignment.START,
            expand=True
        ),
        top=0,
        right=160,  # 30px offset from the right
        padding=ft.padding.all(2),
        visible=False,  # Initially hidden
    )

    return abc_action_container

def create_sort_controls_container():
    global dataset_sort_controls_container
    if dataset_sort_controls_container is not None:
        return dataset_sort_controls_container

    sort_dropdown = create_dropdown(
        label=None,
        value=dataset_sort_mode["value"],
        options={
            "newest": "Newest",
            "oldest": "Oldest",
            "name_asc": "A-Z",
            "name_desc": "Z-A",
        },
        width=170,
        on_change=handle_dataset_sort_change,
        scale=0.8,
    )

    dataset_sort_controls_container = ft.Container(
        content=sort_dropdown,
        padding=ft.padding.only(left=0, right=0),
        border=None,
        bgcolor=ft.Colors.TRANSPARENT,
        top=0,
        right=0,
        visible=False,
        scale=0.8,
    )

    return dataset_sort_controls_container

def update_abc_container_visibility():
    """Update ABC container visibility based on selection state and tab state"""
    global abc_action_container
    if abc_action_container:
        # Main container visible only if we are in dataset tab AND have selections
        abc_action_container.visible = is_in_dataset_tab["value"] and len(selected_thumbnails_set) > 0
        
        if abc_action_container.page:
            abc_action_container.update()


def update_sort_controls_visibility():
    global dataset_sort_controls_container
    if dataset_sort_controls_container:
        dataset_sort_controls_container.visible = is_in_dataset_tab["value"]
        if dataset_sort_controls_container.page:
            dataset_sort_controls_container.update()

def on_main_tab_change(e):
    """Handle main tab switching - manage dataset selections and container visibility"""
    try:
        # Check which tab we're switching to
        new_tab_index = e.control.selected_index

        if new_tab_index == 1:  # Switching to Datasets tab
            # Set that we're in dataset tab
            is_in_dataset_tab["value"] = True
            # Update visibility (will show if there are preserved selections)
            update_abc_container_visibility()
            update_sort_controls_visibility()

            # Also update the page's abc_container to match our local container
            if hasattr(e.page, 'abc_container') and e.page.abc_container:
                e.page.abc_container.visible = is_in_dataset_tab["value"] and len(selected_thumbnails_set) > 0
                e.page.abc_container.update()
        else:  # Switching away from dataset tab
            # Set that we're not in dataset tab
            is_in_dataset_tab["value"] = False

            # DON'T clear selections - preserve them for when we return
            # Just hide the container
            update_abc_container_visibility()
            update_sort_controls_visibility()

            # Also update the page's abc_container
            if hasattr(e.page, 'abc_container') and e.page.abc_container:
                e.page.abc_container.visible = False
                e.page.abc_container.update()

    except Exception as ex:
        print(f"Error handling tab change: {ex}")

async def on_dataset_dropdown_change(
    ev: ft.ControlEvent,
    thumbnails_grid_control: ft.GridView,
    dataset_delete_captions_button_control: ft.ElevatedButton,
    bucket_size_textfield_control: ft.TextField,
    trigger_word_textfield_control: ft.TextField
):
    if processed_output_field.page:
        processed_output_field.visible = False
        set_bottom_app_bar_height()
    if processed_progress_bar.page:
        processed_progress_bar.visible = False

    # Clear selections when dataset changes
    global selected_thumbnails_set, last_clicked_thumbnail_index
    selected_thumbnails_set.clear()
    last_clicked_thumbnail_index = -1

    selected_dataset["value"] = ev.control.value

    base_dir, dataset_type = dataset_utils._get_dataset_base_dir(selected_dataset["value"])
    DATASETS_TYPE["value"] = dataset_type

    bucket_val, model_val, trigger_word_val = dataset_utils.load_dataset_config(selected_dataset["value"])
    # Preprocess panel removed; these controls may not be mounted. Guard updates.
    if bucket_size_textfield_control is not None:
        try:
            bucket_size_textfield_control.value = bucket_val
            if bucket_size_textfield_control.page:
                bucket_size_textfield_control.update()
        except Exception:
            pass
    if trigger_word_textfield_control is not None:
        try:
            trigger_word_textfield_control.value = trigger_word_val or ''
            if trigger_word_textfield_control.page:
                trigger_word_textfield_control.update()
        except Exception:
            pass

    await update_thumbnails(page_ctx=ev.page, grid_control=thumbnails_grid_control)

    if dataset_delete_captions_button_control:
        pass

    if ev.page:
        ev.page.update()

async def update_thumbnails(page_ctx: ft.Page | None, grid_control: ft.GridView | None, force_refresh: bool = False):
    global selected_thumbnails_set, last_clicked_thumbnail_index
    global thumbnails_refresh_in_progress, thumbnails_pending_refresh, thumbnails_pending_force_refresh
    
    if not grid_control:
        return

    page_ctx = _resolve_page_context(page_ctx, grid_control)

    if thumbnails_refresh_in_progress:
        thumbnails_pending_refresh = True
        thumbnails_pending_force_refresh = thumbnails_pending_force_refresh or force_refresh
        return

    thumbnails_refresh_in_progress = True
    _set_refresh_ui_state(True, page_ctx, grid_control)

    try:
        current_selection = selected_dataset.get("value")
        grid_control.controls.clear()
        grid_control.controls.append(_build_thumbnails_loading_indicator())
        if grid_control.page:
            grid_control.update()

        if not current_selection:
            grid_control.controls.clear()
            folders = get_dataset_folders()
            folders_exist = folders is not None and len(folders) > 0
            grid_control.controls.append(ft.Text("Select a dataset to view media." if folders_exist else "No datasets found."))
        else:
            thumbnail_paths_map, video_info = get_videos_and_thumbnails(current_selection, DATASETS_TYPE["value"], force_refresh)
            video_files_list["value"] = list(thumbnail_paths_map.keys())
            dataset_captions = load_dataset_captions(current_selection)

            grid_control.controls.clear()

            if not thumbnail_paths_map:
                grid_control.controls.append(ft.Text(f"No media found in dataset '{current_selection}'."))
            else:
                # Apply sorting based on current sort mode
                sort_mode = dataset_sort_mode.get("value", "newest")
                if sort_mode == "name_desc":
                    sorted_thumbnail_items = sorted(thumbnail_paths_map.items(), key=lambda item: item[0].lower(), reverse=True)
                elif sort_mode == "name_asc":
                    sorted_thumbnail_items = sorted(thumbnail_paths_map.items(), key=lambda item: item[0].lower())
                elif sort_mode == "oldest":
                    # Sort by modification time (oldest first)
                    sorted_thumbnail_items = sorted(thumbnail_paths_map.items(), 
                        key=lambda item: os.path.getmtime(item[0]) if os.path.exists(item[0]) else 0)
                else:  # newest (default)
                    # Sort by modification time (newest first)
                    sorted_thumbnail_items = sorted(thumbnail_paths_map.items(), 
                        key=lambda item: os.path.getmtime(item[0]) if os.path.exists(item[0]) else 0, reverse=True)

                for i, (video_path, thumb_path) in enumerate(sorted_thumbnail_items):
                    has_caption = any(
                        entry.get("media_path") == os.path.basename(video_path) and entry.get("caption", "").strip()
                        for entry in dataset_captions
                    )

                    grid_control.controls.append(
                        create_thumbnail_container(
                            page_ctx=page_ctx,
                            video_path=video_path,
                            thumb_path=thumb_path,
                            video_info=video_info,
                            has_caption=has_caption,
                            video_files_list=video_files_list["value"],
                            update_thumbnails_callback=update_thumbnails,
                            grid_control=grid_control,
                            on_checkbox_change_callback=_on_thumbnail_checkbox_change,
                            thumbnail_index=i,
                            is_selected_initially=(video_path in selected_thumbnails_set)
                        )
                    )

        if grid_control and grid_control.page:
            grid_control.update()

        try:
            if page_ctx and hasattr(page_ctx, 'abc_container'):
                page_ctx.abc_container.visible = len(selected_thumbnails_set) > 0
                page_ctx.abc_container.update()

            update_abc_container_visibility()
        except Exception:
            pass

        if force_refresh and current_selection:
            base_dir = settings.DATASETS_DIR
            dataset_folder_path = os.path.abspath(os.path.join(base_dir, current_selection))
            cleanup_old_temp_thumbnails(dataset_folder_path)

            thumb_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, current_selection)
            if os.path.exists(thumb_dir):
                cleanup_old_temp_thumbnails(thumb_dir)
    finally:
        thumbnails_refresh_in_progress = False
        _set_refresh_ui_state(False, page_ctx, grid_control)

    pending_refresh = thumbnails_pending_refresh
    pending_force = thumbnails_pending_force_refresh
    thumbnails_pending_refresh = False
    thumbnails_pending_force_refresh = False

    if pending_refresh and thumbnails_grid_ref and thumbnails_grid_ref.current:
        rerun_page = _resolve_page_context(page_ctx, thumbnails_grid_ref.current)
        rerun_grid = thumbnails_grid_ref.current
        if rerun_page and rerun_grid:
            rerun_page.run_task(update_thumbnails, rerun_page, rerun_grid, pending_force)

def update_dataset_dropdown(
    p_page: ft.Page | None,
    current_dataset_dropdown: ft.Dropdown,
    current_thumbnails_grid: ft.GridView,
    delete_button: ft.ElevatedButton
):
    # Clear selections when updating dataset list
    global selected_thumbnails_set, last_clicked_thumbnail_index
    selected_thumbnails_set.clear()
    last_clicked_thumbnail_index = -1

    folders = get_dataset_folders()
    current_dataset_dropdown.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in folders.items()] if folders else []
    current_dataset_dropdown.value = None
    selected_dataset["value"] = None

    bucket_val, model_val, trigger_word_val = dataset_utils.load_dataset_config(None)
    if bucket_size_textfield: bucket_size_textfield.value = bucket_val
    if model_name_dropdown: model_name_dropdown.value = model_val
    if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''

    if bucket_size_textfield: bucket_size_textfield.update()
    if model_name_dropdown: model_name_dropdown.update()
    p_page.run_task(update_thumbnails, page_ctx=p_page, grid_control=current_thumbnails_grid)

    if delete_button:
        pass

    if current_dataset_dropdown.page:
        current_dataset_dropdown.update()

    if p_page:
        p_page.snack_bar = ft.SnackBar(ft.Text("Dataset list updated! Select a dataset."))
        p_page.snack_bar.open = True

def reload_current_dataset(
    p_page: ft.Page | None,
    current_dataset_dropdown: ft.Dropdown,
    current_thumbnails_grid: ft.GridView,
    add_button: ft.ElevatedButton,
    delete_button: ft.ElevatedButton
):
    if processed_output_field.page:
        processed_output_field.visible = False
        set_bottom_app_bar_height()
    if processed_progress_bar.page:
        processed_progress_bar.visible = False

    # Clear selections when reloading datasets
    global selected_thumbnails_set, last_clicked_thumbnail_index
    selected_thumbnails_set.clear()
    last_clicked_thumbnail_index = -1

    folders = get_dataset_folders()
    current_dataset_dropdown.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in folders.items()] if folders else []
    current_dataset_dropdown.disabled = len(folders) == 0

    prev_selected_name = selected_dataset.get("value")

    if prev_selected_name and prev_selected_name in folders:
        current_dataset_dropdown.value = prev_selected_name
        selected_dataset["value"] = prev_selected_name
        bucket_val, model_val, trigger_word_val = dataset_utils.load_dataset_config(prev_selected_name)
        if bucket_size_textfield: bucket_size_textfield.value = bucket_val
        if model_name_dropdown: model_name_dropdown.value = model_val if model_val in settings.train_models else settings.train_def_model
        if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''
        p_page.run_task(update_thumbnails, page_ctx=p_page, grid_control=current_thumbnails_grid, force_refresh=True)
        snack_bar_text = f"Dataset '{prev_selected_name}' reloaded."
    else:
        current_dataset_dropdown.value = None
        selected_dataset["value"] = None
        bucket_val, model_val, trigger_word_val = dataset_utils.load_dataset_config(None)
        if bucket_size_textfield: bucket_size_textfield.value = bucket_val
        if model_name_dropdown: model_name_dropdown.value = model_val
        if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''
        p_page.run_task(update_thumbnails, page_ctx=p_page, grid_control=current_thumbnails_grid, force_refresh=True)
        snack_bar_text = "Dataset list reloaded. Select a dataset."

    if delete_button:
        pass

    if add_button:
         pass

    if p_page:
        p_page.snack_bar = ft.SnackBar(ft.Text(snack_bar_text))
        p_page.snack_bar.open = True
        p_page.update()

# ======================================================================================
# GUI Control Creation Functions (Build individual controls or groups)
# ======================================================================================

def _create_global_controls():
    global bucket_size_textfield, rename_textfield, model_name_dropdown, trigger_word_textfield

    if bucket_size_textfield is not None:
        return

    bucket_size_textfield = create_textfield(
        label="Bucket Size (e.g., [W, H, F] or WxHxF)",
        value=settings.DEFAULT_BUCKET_SIZE_STR,
        expand=True
    )

    rename_textfield = create_textfield(
        label="Rename all files",
        value="",
        hint_text="Name of videos + _num will be added",
        expand=True,
    )


    trigger_word_textfield = create_textfield(
        "Trigger WORD", "", col=9, expand=True, hint_text="e.g. 'CAKEIFY' , leave empty for none"
    )

    bucket_size_textfield.on_change = lambda e: e.page.run_task(on_bucket_or_model_change, e, selected_dataset, bucket_size_textfield, None, trigger_word_textfield)
    trigger_word_textfield.on_change = lambda e: e.page.run_task(on_bucket_or_model_change, e, selected_dataset, bucket_size_textfield, None, trigger_word_textfield)

def _build_dataset_creation_section(dataset_name_textfield: ft.TextField, add_dataset_button: ft.ElevatedButton):
    """Build the section for creating new datasets"""
    return ft.Column([
        ft.Container(height=5),  # Reduced spacing
        ft.ResponsiveRow([
            ft.Container(content=dataset_name_textfield, expand=True, col=7),
            ft.Container(content=add_dataset_button, expand=True, col=5),
        ], spacing=5),
        ft.Container(height=3),
        ft.Divider(),
    ], spacing=0)

def _build_dataset_selection_section(dataset_dropdown_control: ft.Dropdown, update_button_control: ft.IconButton):
    return ft.Column([
        ft.Container(height=10),
        ft.Row([
            ft.Container(content=dataset_dropdown_control, expand=True, width=160),
            ft.Container(content=update_button_control, alignment=ft.alignment.center_right, width=40),
        ], expand=True),
        ft.Container(height=3),
        ft.Divider(),
    ], spacing=0)

def _build_captioning_section(
    caption_model_dropdown: ft.Dropdown,
    captions_checkbox_container: ft.Container,
    cap_command_textfield: ft.TextField,
    max_tokens_textfield: ft.TextField,
    dataset_add_captions_button_control: ft.ElevatedButton,
    dataset_delete_captions_button_control: ft.ElevatedButton,
    joy_prompt_container: ft.Container,):
    return build_expansion_tile(
        title="1. Captions",
        controls=[
            ft.ResponsiveRow([captions_checkbox_container, caption_model_dropdown]),
            ft.ResponsiveRow([max_tokens_textfield, joy_prompt_container]),
            ft.ResponsiveRow([cap_command_textfield]),
            ft.Row([
                ft.Container(content=dataset_add_captions_button_control, expand=True),
                ft.Container(content=dataset_delete_captions_button_control, expand=True)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ],
        initially_expanded=True,
    )

# Preprocess panel removed per request

def _build_latent_test_section(update_thumbnails_func):
    find_replace=ft.Column([
        create_textfield(label="Find",value="",expand=True, ref=find_text_field_ref),
        create_textfield(label="Replace", value="", expand=True, ref=replace_text_field_ref),
        create_styled_button("Find and Replace", on_click=lambda e: e.page.run_task(find_and_replace_in_captions,
            e, selected_dataset, DATASETS_TYPE, find_text_field_ref, replace_text_field_ref, update_thumbnails_func, thumbnails_grid_ref
        ), 
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
        )
    ])
    prefix_suffix_replace=ft.Column([
        create_textfield(label="Text",value="",expand=True, ref=affix_text_field_ref),
        ft.ResponsiveRow([
            create_styled_button("Add prefix",col=6, on_click=lambda e: e.page.run_task(apply_affix_from_textfield,
                e, "prefix", selected_dataset, DATASETS_TYPE, update_thumbnails_func, thumbnails_grid_ref, affix_text_field_ref
            ),
            button_style=ft.ButtonStyle(
                text_style=ft.TextStyle(size=10),  # Smaller font
                shape=ft.RoundedRectangleBorder(radius=3)
            ),
            height=30  # Smaller height
            ),
            create_styled_button("Add suffix",col=6, on_click=lambda e: e.page.run_task(apply_affix_from_textfield,
                e, "suffix", selected_dataset, DATASETS_TYPE, update_thumbnails_func, thumbnails_grid_ref, affix_text_field_ref
            ),
            button_style=ft.ButtonStyle(
                text_style=ft.TextStyle(size=10),  # Smaller font
                shape=ft.RoundedRectangleBorder(radius=3)
            ),
            height=30  # Smaller height
            )
        ])
    ])
    return build_expansion_tile(
        title="Batch captions",
        controls=[
            find_replace,
            ft.Divider(thickness=1,height=3),
            prefix_suffix_replace
        ],
        initially_expanded=False,
    )

def _build_batch_section(change_fps_section: ft.ResponsiveRow, rename_textfield: ft.TextField, rename_files_button: ft.ElevatedButton,
                         caption_to_txt_button: ft.ElevatedButton,caption_to_json_button:  ft.ElevatedButton):
    # Create slicing controls in same style as change FPS
    slice_seconds_textfield = create_textfield(
        "seconds",
        "60",
        expand=True,
        hint_text="60 or 5.5 (seconds)",
        col=4
    )

    # Add re-encoding checkbox
    reencode_checkbox = ft.Checkbox(
        label="Re-encode",
        value=False,
        tooltip="Enable re-encoding for better compatibility (slower but more reliable)",
        label_style=ft.TextStyle(size=12)
    )

    slice_to_button = create_styled_button(
        "Slice to:",
        tooltip="Slice selected videos into chunks. Use integers for time-based splitting (e.g., '60') or floats for frame-based splitting (e.g., '5.5')",
        expand=True,
        col=8,
        on_click=lambda e: _on_slice_to_click(e, slice_seconds_textfield, reencode_checkbox),
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
    )

    # Create slice section following the same pattern as change_fps_section
    slice_section = ft.ResponsiveRow([
        ft.Container(content=slice_to_button, col=6,),
        ft.Container(content=slice_seconds_textfield, col=4,),
        ft.Container(content=reencode_checkbox, col=2,),
    ], spacing=5)

    return build_expansion_tile(
        title="Batch files",
        controls=[
            change_fps_section,
            ft.Divider(thickness=1),
            slice_section,
            ft.Divider(thickness=1),
            rename_textfield,
            rename_files_button,
            ft.Divider(thickness=1),
            ft.ResponsiveRow([
                ft.Container(content=caption_to_txt_button, expand=True,col=6, alignment=ft.alignment.center),
                ft.Container(content=caption_to_json_button, expand=True,col=6, alignment=ft.alignment.center)
            ])
        ],
        initially_expanded=False,
    )

def _build_bottom_status_bar():
    global bottom_app_bar_ref
    bottom_app_bar = ft.BottomAppBar(
        bgcolor=ft.Colors.BLUE_GREY_900,
        height=0,
        content=ft.Row([
            ft.Container(
                content=ft.Column([
                    processed_progress_bar,
                    processed_output_field,
                ], expand=True),
                expand=True,
            ),
        ], expand=True),
    )
    bottom_app_bar_ref = bottom_app_bar
    return bottom_app_bar

# ======================================================================================
# Event Handlers
# ======================================================================================

def _on_slice_to_click(e: ft.ControlEvent, seconds_textfield: ft.TextField, reencode_checkbox: ft.Checkbox):
    """Handle Slice to button click"""
    # Debug: Show that the function is being called
    print("Slice to button clicked!")

    try:
        # Support both integer and float values (e.g., "5" or "5.5")
        time_value = float(seconds_textfield.value or "60")
        reencode_enabled = reencode_checkbox.value
        print(f"Time value: {time_value}, Re-encode: {reencode_enabled}")

        # Show immediate feedback - indicate if it's frame-based
        if time_value.is_integer():
            mode_text = "re-encoding" if reencode_enabled else "smart stream copy"
            feedback_text = f"Processing chunking for {int(time_value)} seconds ({mode_text})..."
        else:
            mode_text = "re-encoding" if reencode_enabled else "smart stream copy"
            feedback_text = f"Processing chunking for {time_value} seconds (frame-based) ({mode_text})..."

        e.page.snack_bar = ft.SnackBar(ft.Text(feedback_text), open=True)
        e.page.update()

        # Create thumbnail update callback
        def thumbnail_update_callback():
            """Update thumbnails after chunking"""
            try:
                from .unified_popup_dialog import update_thumbnails_callback_factory
                thumbnail_func = update_thumbnails_callback_factory(e.page)
                thumbnail_func()
            except Exception:
                # Fallback: direct update
                e.page.run_task(update_thumbnails, page_ctx=e.page, grid_control=thumbnails_grid_ref.current, force_refresh=True)

        # Import and call the chunking function
        from flet_app.ui_popups import video_editor

        def run_chunking():
            video_editor.split_selected_videos_into_chunks(e.page, time_value, thumbnail_update_callback, reencode_enabled)

        e.page.run_thread(run_chunking)

    except ValueError:
        e.page.snack_bar = ft.SnackBar(ft.Text("Please enter a valid number of seconds."), open=True)
        e.page.update()
    except Exception as ex:
        print(f"Error in slice click: {ex}")
        e.page.snack_bar = ft.SnackBar(ft.Text(f"Error: {str(ex)}"), open=True)
        e.page.update()

# ======================================================================================
# Main GUI Layout Builder (Assembles the sections)
# ======================================================================================

def dataset_tab_layout(page=None):
    p_page = page

    if bucket_size_textfield is None:
        _create_global_controls()

    # Create ABC action container and return it for the main page to use
    abc_container_ref = create_abc_action_container()

    sort_controls_container_ref = create_sort_controls_container()

    # Initialize tab state (we're creating the dataset tab, so we're in it)
    is_in_dataset_tab["value"] = True

    folders = get_dataset_folders()
    folder_names = list(folders.keys()) if folders else []

    dataset_dropdown_control = create_dropdown(
        "Select dataset",
        selected_dataset["value"],
        {name: name for name in folder_names},
        "Select your dataset",
        expand=True,
    )
    dataset_dropdown_control_ref.current = dataset_dropdown_control

    thumbnails_grid_control = ft.GridView(
        ref=thumbnails_grid_ref,
        runs_count=5, max_extent=settings.THUMB_TARGET_W + 20,
        child_aspect_ratio=(settings.THUMB_TARGET_W + 10) / (settings.THUMB_TARGET_H + 80),
        spacing=7, run_spacing=7, controls=[], expand=True
    )

    dataset_dropdown_control.on_change = lambda ev: ev.page.run_task(
        on_dataset_dropdown_change,
        ev,
        thumbnails_grid_control,
        dataset_delete_captions_button_ref.current,
        bucket_size_textfield,
        trigger_word_textfield
    )

    update_button_control = ft.IconButton(
        ref=refresh_button_ref,
        icon=ft.Icons.REFRESH,
        tooltip="Update dataset list and refresh thumbnails",
        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=8)),
        icon_size=20,
        disabled=thumbnails_refresh_in_progress,
        on_click=lambda e: e.page.run_task(update_thumbnails, e.page, thumbnails_grid_control, True)
    )

    # Dataset creation fields
    dataset_name_textfield = create_textfield(
        "Dataset name",
        "",
        hint_text="Enter new dataset name",
        expand=True,
    )
    
    add_dataset_button = create_styled_button(
        "Add",
        tooltip="Add new dataset",
        expand=True,
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
    )

    # Event handler for the add dataset button
    async def on_add_dataset_click(e: ft.ControlEvent):
        """Handle the Add Dataset button click"""
        dataset_name = dataset_name_textfield.value
        if not dataset_name or dataset_name.strip() == "":
            e.page.snack_bar = ft.SnackBar(ft.Text("Please enter a dataset name"), open=True)
            e.page.update()
            return
            
        # Create the new dataset directory
        dataset_path = os.path.join(settings.DATASETS_DIR, dataset_name.strip())
        
        try:
            if os.path.exists(dataset_path):
                e.page.snack_bar = ft.SnackBar(ft.Text(f"Dataset '{dataset_name}' already exists"), open=True)
                e.page.update()
                return
                
            os.makedirs(dataset_path, exist_ok=True)
            
            # Update the dataset dropdown to include the new dataset
            folders = get_dataset_folders()
            dataset_dropdown_control.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in folders.items()] if folders else []
            dataset_dropdown_control.value = dataset_name
            selected_dataset["value"] = dataset_name
            await update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_control)
            
            e.page.snack_bar = ft.SnackBar(ft.Text(f"Dataset '{dataset_name}' created successfully"), open=True)
            e.page.update()
            
        except Exception as ex:
            e.page.snack_bar = ft.SnackBar(ft.Text(f"Error creating dataset: {str(ex)}"), open=True)
            e.page.update()

    add_dataset_button.on_click = lambda e: e.page.run_task(on_add_dataset_click, e)

    caption_model_dropdown = create_dropdown(
        "Captioning Model",
        settings.captions_def_model,
        settings.captions,
        "Select a captioning model",
        expand=True,col=9,
    )
    caption_model_dropdown_ref.current = caption_model_dropdown
    # Persist model selection when changed
    def _on_model_change(ev: ft.ControlEvent):
        try:
            ev.page.run_task(on_bucket_or_model_change, ev, selected_dataset, bucket_size_textfield, caption_model_dropdown_ref.current, trigger_word_textfield)
        except Exception:
            pass
        # Toggle visibility of 8-bit vs HF checkbox depending on model
        try:
            val = (caption_model_dropdown_ref.current.value or "").lower()
            is_llava = (val == "llava_next_7b")
            is_qwen = val.startswith("qwen3_vl")
            show_8bit = is_llava
            show_hf = is_qwen
            if captions_checkbox_ref.current:
                captions_checkbox_ref.current.visible = show_8bit
                if captions_checkbox_ref.current.page:
                    captions_checkbox_ref.current.update()
            if hf_checkbox_ref.current:
                hf_checkbox_ref.current.visible = show_hf
                if hf_checkbox_ref.current.page:
                    hf_checkbox_ref.current.update()
        except Exception:
            pass
    caption_model_dropdown.on_change = _on_model_change

    captions_checkbox = ft.Checkbox(
        label="8-bit", value=True, scale=1,
        visible=False,
        left="left",
        expand=True,
    )
    captions_checkbox_ref.current = captions_checkbox

    hf_checkbox = ft.Checkbox(
        label="HF", value=True, scale=1,
        visible=False,
        left="left",
        expand=True,
    )
    hf_checkbox_ref.current = hf_checkbox

    # Place both in same container; we toggle visibility based on model
    captions_checkbox_container = ft.Container(
        content=ft.Column([captions_checkbox, hf_checkbox], spacing=0),
        expand=True, col=3, scale=0.8,
        alignment=ft.alignment.bottom_center,
        margin=ft.margin.only(top=10)
    )

    # Set initial visibility based on current dropdown value (show 8-bit for LLaVA by default)
    try:
        _val0 = (caption_model_dropdown_ref.current.value or "").lower()
        _is_llava0 = (_val0 == "llava_next_7b")
        _is_qwen0 = _val0.startswith("qwen3_vl")
        if captions_checkbox_ref.current:
            captions_checkbox_ref.current.visible = _is_llava0
        if hf_checkbox_ref.current:
            hf_checkbox_ref.current.visible = _is_qwen0
    except Exception:
        pass

    cap_command_textfield = create_textfield(
        "Command",
        "Shortly describe the content of this video in one or two sentences.",
        expand=True,
        hint_text="command for captioning",
        col=12,
        multiline=True,
        min_lines=4,
        max_lines=8,
    )
    cap_command_textfield_ref.current = cap_command_textfield

    max_tokens_textfield = create_textfield("Max Tokens", "100",
                                        expand=True,
                                        hint_text="max tokens",col=4,
    )
    max_tokens_textfield_ref.current = max_tokens_textfield

    # JoyCaption prompt presets dropdown (hidden unless JoyCaption model is selected)
    joy_prompt_options = {"__include_none__": True}
    _key_to_prompt: dict[str, str] = {}
    try:
        import json as _json
        from flet_app.project_root import get_project_root
        pfile = str(get_project_root() / "scripts" / "prompt-examples.json")
        pfile = os.path.normpath(pfile)
        if os.path.exists(pfile):
            with open(pfile, "r", encoding="utf-8") as f:
                data = _json.load(f)
            prompts: list[str] = []
            for item in data:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict) and "prompt" in item:
                    prompts.append(str(item.get("prompt", "")))
            for i, pr in enumerate(prompts):
                key = f"p{i}"
                joy_prompt_options[key] = pr
                _key_to_prompt[key] = pr
    except Exception:
        pass

    joy_prompt_dropdown = create_dropdown(
        "Prompt Preset",
        "",
        joy_prompt_options,
        hint_text="Choose a preset to fill the command",
        expand=True,
        col=12,
    )
    joy_prompt_dropdown_ref.current = joy_prompt_dropdown

    def _on_preset_change(e):
        try:
            key = joy_prompt_dropdown.value or ""
            if key and key in _key_to_prompt and cap_command_textfield_ref.current is not None:
                cap_command_textfield_ref.current.value = _key_to_prompt[key]
                if cap_command_textfield_ref.current.page:
                    cap_command_textfield_ref.current.update()
        except Exception:
            pass

    joy_prompt_dropdown.on_change = _on_preset_change

    # Always show preset selector (also used for video models)
    joy_prompt_container = ft.Container(
        content=joy_prompt_dropdown,
        visible=True,
        col=8,
    )

    dataset_delete_captions_button_control = create_styled_button(
        "Delete",
        ref=dataset_delete_captions_button_ref,
        tooltip="Delete the captions.json file",
        expand=True,
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
    )
    dataset_delete_captions_button_ref.current = dataset_delete_captions_button_control

    dataset_add_captions_button_control = create_styled_button(
        "Add Captions",
        ref=dataset_add_captions_button_ref,
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
          # Smaller height
        expand=True
    )
    dataset_add_captions_button_ref.current = dataset_add_captions_button_control

    # Preprocess button and panel removed

    change_fps_textfield = create_textfield("Change fps", "24",
                                    expand=True,
                                    hint_text="fps",col=4,
    )
    change_fps_textfield_ref.current = change_fps_textfield

    change_fps_button = create_styled_button(
        "Change fps",
        tooltip="Change fps",
        expand=True,
        on_click=lambda e: e.page.run_task(on_change_fps_click,
            e, selected_dataset, DATASETS_TYPE, change_fps_textfield_ref, thumbnails_grid_ref, update_thumbnails, settings
        ),
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
    )
    change_fps_section = ft.ResponsiveRow([
        ft.Container(content=change_fps_textfield, col=4,),
        ft.Container(content=change_fps_button, col=8,),
    ], spacing=5)

    rename_files_button = create_styled_button(
        "Rename files",
        tooltip="Rename files",
        expand=True,
        on_click=lambda e: e.page.run_task(on_rename_files_click,
            e, selected_dataset, DATASETS_TYPE, rename_textfield, thumbnails_grid_ref, update_thumbnails
        ),
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
    )

    caption_to_json_button = create_styled_button(
        "txt > json",
        tooltip="txt > json",
        expand=True,
        on_click=lambda e: e.page.run_task(on_caption_to_json_click,
            e, selected_dataset, DATASETS_TYPE, update_thumbnails, thumbnails_grid_ref
        ),
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
    )

    caption_to_txt_button = create_styled_button(
        "json > txt",
        tooltip="json > txt",
        expand=True,
        on_click=lambda e: e.page.run_task(on_caption_to_txt_click,
            e, selected_dataset, DATASETS_TYPE
        ),
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),  # Smaller font
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        height=30  # Smaller height
    )

    update_button_control.on_click = lambda e: reload_current_dataset(
        e.page,
        dataset_dropdown_control_ref.current,
        thumbnails_grid_control,
        dataset_add_captions_button_ref.current,
        dataset_delete_captions_button_ref.current
    )
    dataset_add_captions_button_control.on_click = lambda e: e.page.run_task(
        on_add_captions_click_with_model,
        e,
        caption_model_dropdown_ref.current,
        captions_checkbox_ref.current,
        hf_checkbox_ref.current,
        cap_command_textfield_ref.current,
        max_tokens_textfield_ref.current,
        dataset_add_captions_button_ref.current,
        dataset_delete_captions_button_ref.current,
        thumbnails_grid_control,
        selected_dataset,
        DATASETS_TYPE,
        processed_progress_bar,
        processed_output_field,
        set_bottom_app_bar_height,
        update_thumbnails
    )
    dataset_delete_captions_button_control.on_click = lambda e: on_delete_captions_click(
        e,
        thumbnails_grid_control,
        selected_dataset,
        processed_progress_bar,
        processed_output_field,
        set_bottom_app_bar_height,
        update_thumbnails
    )

    # No preprocess button handler (panel removed)

    change_fps_button.on_click = lambda e: e.page.run_task(on_change_fps_click,
        e, selected_dataset, DATASETS_TYPE, change_fps_textfield_ref, thumbnails_grid_ref, update_thumbnails
    )
    rename_files_button.on_click = lambda e: e.page.run_task(on_rename_files_click,
        e, selected_dataset, DATASETS_TYPE, rename_textfield, thumbnails_grid_ref, update_thumbnails
    )

    dataset_selection_section = _build_dataset_selection_section(dataset_dropdown_control_ref.current, update_button_control)

    captioning_section = _build_captioning_section(
        caption_model_dropdown_ref.current,
        captions_checkbox_container,
        cap_command_textfield_ref.current,
        max_tokens_textfield_ref.current,
        dataset_add_captions_button_ref.current,
        dataset_delete_captions_button_ref.current,
        joy_prompt_container,
    )

    # Preprocessing section removed

    latent_test_section = _build_latent_test_section(update_thumbnails)

    batch_section = _build_batch_section(change_fps_section, rename_textfield, rename_files_button,caption_to_txt_button,
        caption_to_json_button)

    # Build dataset creation section
    dataset_creation_section = _build_dataset_creation_section(dataset_name_textfield, add_dataset_button)

    lc_content = ft.Column([
        dataset_selection_section,
        dataset_creation_section,  # Add the new dataset creation section right after selection
        captioning_section,
        latent_test_section,
        batch_section
    ], spacing=3, width=200, alignment=ft.MainAxisAlignment.START)

    bottom_app_bar = _build_bottom_status_bar()

    # Create a file picker for proper file uploads (handles both desktop and web)
    async def on_files_picked(e: ft.FilePickerResultEvent):
        # Handle selected files
        if e.files is not None and len(e.files) > 0:
            # Get the currently selected dataset
            current_dataset_name = selected_dataset.get("value")
            if not current_dataset_name:
                if e.page:
                    e.page.snack_bar = ft.SnackBar(ft.Text("Please select a dataset first"), open=True)
                    e.page.update()
                return
            
            # Get the dataset path
            base_dir, _ = dataset_utils._get_dataset_base_dir(current_dataset_name)
            dataset_path = os.path.join(base_dir, current_dataset_name)
            
            # Create dataset directory if it doesn't exist
            os.makedirs(dataset_path, exist_ok=True)
            
            successful_uploads = 0
            import shutil
            from pathlib import Path
            
            temp_uploads_dir = Path(__file__).parent.parent.parent / "temp_uploads"
            
            pending_uploads_count["value"] = len(e.files)
            for file in e.files:
                try:
                    dest_filename = file.name
                    dest_path = os.path.join(dataset_path, dest_filename)

                    if file.path:  # Desktop mode: Direct copy
                        source_path = file.path
                        shutil.copy2(source_path, dest_path)
                        successful_uploads += 1
                    else:  # Web mode: Generate upload URL directly to the selected dataset
                        # For web mode, we generate the upload URL directly to the selected dataset
                        if current_dataset_name:
                            # Generate upload URL directly to the dataset directory
                            relative_upload_path = f"{current_dataset_name}/{file.name}"
                            upload_url = e.page.get_upload_url(relative_upload_path, expires=300)  # 5-min expiry
                            file_picker.upload([ft.FilePickerUploadFile(file.name, upload_url=upload_url)])
                            
                            # In web mode, the file will be uploaded directly to the dataset folder
                            # via the upload URL, so no additional move operation is needed
                            print(f"Web upload URL generated for: {file.name} -> {relative_upload_path}")
                            
                            # The upload would happen in the background, and we can update the UI after
                            e.page.snack_bar = ft.SnackBar(ft.Text(f"Web upload initiated for {file.name}"), open=True)
                            e.page.update()
                        else:
                            print(f"Cannot upload web file {file.name}: no dataset selected")
                
                except Exception as ex:
                    print(f"Error processing {file.name}: {ex}")
                    if e.page:
                        e.page.snack_bar = ft.SnackBar(ft.Text(f"Error uploading {file.name}: {str(ex)}"), open=True)
                        e.page.update()

            # Update thumbnails after upload for desktop files
            await update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True)
            
            # Show success message for desktop files
            if e.page:
                e.page.snack_bar = ft.SnackBar(ft.Text(f"{successful_uploads} file(s) uploaded successfully"), open=True)
                e.page.update()

    async def on_file_uploaded(e: ft.FilePickerUploadEvent):
        global pending_uploads_count
        if e.file_name:
            pending_uploads_count["value"] -= 1
            if pending_uploads_count["value"] == 0:
                await update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_ref.current, force_refresh=True)
                e.page.snack_bar = ft.SnackBar(ft.Text(f"All files uploaded successfully"), open=True)
                e.page.update()
            else:
                e.page.snack_bar = ft.SnackBar(ft.Text(f"{e.file_name} uploaded. {pending_uploads_count['value']} remaining."), open=True)
                e.page.update()

    file_picker = ft.FilePicker(
        on_result=lambda e: e.page.run_task(on_files_picked, e),
        on_upload=lambda e: e.page.run_task(update_thumbnails, e.page, thumbnails_grid_ref.current, True)
    )
    if p_page:  # Use p_page instead of page
        p_page.overlay.append(file_picker)
    
    # Create upload button that triggers file picker
    async def open_file_picker_async(picker: ft.FilePicker):
        picker.pick_files(
            allow_multiple=True,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=settings.IMAGE_EXTENSIONS + settings.VIDEO_EXTENSIONS
        )

    upload_button = create_styled_button(
        "Upload Files",
        tooltip="Upload files to current dataset",
        button_style=ft.ButtonStyle(
            text_style=ft.TextStyle(size=10),
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
        
        on_click=lambda e: e.page.run_task(open_file_picker_async, file_picker)
    )
    
    # Create a simple container for the thumbnails area with upload button
    # Since true OS-level drag and drop isn't supported in Flet web, we'll use a clickable area approach
    rc_content = ft.Column([
        ft.Container(
            content=ft.Column([
                thumbnails_grid_control,
                ft.Row([upload_button], alignment=ft.MainAxisAlignment.CENTER, spacing=10)  # Add upload button below thumbnails
            ]),
            expand=True,
            border=ft.border.all(1, ft.Colors.with_opacity(0.2, ft.Colors.GREY_400)),
            border_radius=5,
            margin=ft.margin.only(top=10),
        ),
        bottom_app_bar,
    ], alignment=ft.CrossAxisAlignment.STRETCH, expand=True, spacing=10)

    lc = ft.Container(
        content=lc_content,
        padding=ft.padding.only(top=0, right=0, left=5),
    )
    rc = ft.Container(
        content=rc_content,
        padding=ft.padding.only(top=5, left=0, right=0),
        expand=True
    )

    reload_current_dataset(
        p_page,
        dataset_dropdown_control_ref.current,
        thumbnails_grid_control,
        dataset_add_captions_button_ref.current,
        dataset_delete_captions_button_ref.current
    )

    main_container = ft.Row(
        controls=[
            lc,
            ft.VerticalDivider(color=ft.Colors.GREY_500, width=1),
            rc,
        ],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.START,
        expand=True
    )

    # Expose the ABC container for external access
    main_container.abc_action_container = abc_container_ref

    return main_container, abc_container_ref, sort_controls_container_ref
