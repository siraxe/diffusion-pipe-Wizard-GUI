import flet as ft
import os
import json
import asyncio
import time
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
processed_progress_bar = ft.ProgressBar(visible=False)
processed_output_field = ft.TextField(
    label="Processed Output", text_size=10, multiline=True, read_only=True,
    visible=False, min_lines=6, max_lines=15, expand=True)
bottom_app_bar_ref = None

# Multi-selection state
selected_thumbnails_set = set() # Stores video_path of selected thumbnails
last_clicked_thumbnail_index = -1 # Stores the index of the last clicked checkbox

# Global controls (defined here but created in _create_global_controls)
bucket_size_textfield: ft.TextField = None
rename_textfield: ft.TextField = None
model_name_dropdown: ft.Dropdown = None
trigger_word_textfield: ft.TextField = None

# References to controls created in dataset_tab_layout that need external access
dataset_dropdown_control_ref = ft.Ref[ft.Dropdown]()
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

def _on_thumbnail_checkbox_change(video_path: str, is_checked: bool, thumbnail_index: int):
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

    update_thumbnails(page_ctx=ev.page, grid_control=thumbnails_grid_control)

    if dataset_delete_captions_button_control:
        pass

    if ev.page:
        ev.page.update()

def update_thumbnails(page_ctx: ft.Page | None, grid_control: ft.GridView | None, force_refresh: bool = False):
    global selected_thumbnails_set, last_clicked_thumbnail_index
    
    if not grid_control:
        return

    current_selection = selected_dataset.get("value")
    grid_control.controls.clear()
    # removed processed_map/proc indicator

    if not current_selection:
        folders_exist = get_dataset_folders() is not None and len(get_dataset_folders()) > 0
        grid_control.controls.append(ft.Text("Select a dataset to view media." if folders_exist else "No datasets found."))
    else:
        thumbnail_paths_map, video_info = get_videos_and_thumbnails(current_selection, DATASETS_TYPE["value"])
        video_files_list["value"] = list(thumbnail_paths_map.keys())
        dataset_captions = load_dataset_captions(current_selection)

        if not thumbnail_paths_map:
            grid_control.controls.append(ft.Text(f"No media found in dataset '{current_selection}'."))
        else:
            if force_refresh or (selected_dataset.get("value") != current_selection):
                selected_thumbnails_set.clear()
                last_clicked_thumbnail_index = -1

            sorted_thumbnail_items = sorted(thumbnail_paths_map.items(), key=lambda item: item[0])

            for i, (video_path, thumb_path) in enumerate(sorted_thumbnail_items):
                has_caption = any(entry.get("media_path") == os.path.basename(video_path) and entry.get("caption", "").strip() for entry in dataset_captions)
                
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

    if force_refresh and current_selection:
        dataset_type = DATASETS_TYPE["value"]
        base_dir = settings.DATASETS_DIR
        dataset_folder_path = os.path.abspath(os.path.join(base_dir, current_selection))
        
        cleanup_old_temp_thumbnails(dataset_folder_path)
        
        thumb_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, current_selection)
        if os.path.exists(thumb_dir):
            cleanup_old_temp_thumbnails(thumb_dir)

def update_dataset_dropdown(
    p_page: ft.Page | None,
    current_dataset_dropdown: ft.Dropdown,
    current_thumbnails_grid: ft.GridView,
    delete_button: ft.ElevatedButton
):
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
    if trigger_word_textfield: trigger_word_textfield.update()

    update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid)

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
        update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid)
        snack_bar_text = f"Dataset '{prev_selected_name}' reloaded."
    else:
        current_dataset_dropdown.value = None
        selected_dataset["value"] = None
        bucket_val, model_val, trigger_word_val = dataset_utils.load_dataset_config(None)
        if bucket_size_textfield: bucket_size_textfield.value = bucket_val
        if model_name_dropdown: model_name_dropdown.value = model_val
        if trigger_word_textfield: trigger_word_textfield.value = trigger_word_val or ''
        update_thumbnails(page_ctx=p_page, grid_control=current_thumbnails_grid)
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
        ))
    ])
    prefix_suffix_replace=ft.Column([
        create_textfield(label="Text",value="",expand=True, ref=affix_text_field_ref),
        ft.ResponsiveRow([
            create_styled_button("Add prefix",col=6, on_click=lambda e: e.page.run_task(apply_affix_from_textfield,
                e, "prefix", selected_dataset, DATASETS_TYPE, update_thumbnails_func, thumbnails_grid_ref, affix_text_field_ref
            )),
            create_styled_button("Add suffix",col=6, on_click=lambda e: e.page.run_task(apply_affix_from_textfield,
                e, "suffix", selected_dataset, DATASETS_TYPE, update_thumbnails_func, thumbnails_grid_ref, affix_text_field_ref
            ))
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
    return build_expansion_tile(
        title="Batch files",
        controls=[
            change_fps_section,
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
# Main GUI Layout Builder (Assembles the sections)
# ======================================================================================

def dataset_tab_layout(page=None):
    p_page = page

    if bucket_size_textfield is None:
        _create_global_controls()

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
        icon=ft.Icons.REFRESH, 
        tooltip="Update dataset list and refresh thumbnails",
        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=8)), 
        icon_size=20,
        on_click=lambda e: update_thumbnails(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True)
    )

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
        button_style=BTN_STYLE2,
    )
    dataset_delete_captions_button_ref.current = dataset_delete_captions_button_control

    dataset_add_captions_button_control = create_styled_button(
        "Add Captions",
        ref=dataset_add_captions_button_ref,
        button_style=BTN_STYLE2,
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
        button_style=BTN_STYLE2
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
        button_style=BTN_STYLE2
    )

    caption_to_json_button = create_styled_button(
        "txt > json",
        tooltip="txt > json",
        expand=True,
        on_click=lambda e: e.page.run_task(on_caption_to_json_click,
            e, selected_dataset, DATASETS_TYPE, update_thumbnails, thumbnails_grid_ref
        ),
        button_style=BTN_STYLE2
    )

    caption_to_txt_button = create_styled_button(
        "json > txt",
        tooltip="json > txt",
        expand=True,
        on_click=lambda e: e.page.run_task(on_caption_to_txt_click,
            e, selected_dataset, DATASETS_TYPE
        ),
        button_style=BTN_STYLE2
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

    lc_content = ft.Column([
        dataset_selection_section,
        captioning_section,
        latent_test_section,
        batch_section
    ], spacing=3, width=200, alignment=ft.MainAxisAlignment.START)

    bottom_app_bar = _build_bottom_status_bar()

    rc_content = ft.Column([
        thumbnails_grid_control,
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

    return main_container
