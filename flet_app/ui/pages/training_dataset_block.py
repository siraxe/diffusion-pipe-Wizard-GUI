import flet as ft
import os
import cv2
import numpy as np
import json
import base64
from .._styles import create_textfield, create_dropdown  # Import helper functions
from ..dataset_manager.dataset_utils import get_dataset_folders, _get_dataset_base_dir, get_videos_and_thumbnails  # Reuse the helper
from flet_app.settings import settings


# =====================
# Data/Utility Functions
# =====================
def load_dataset_summary(dataset):
    """
    Loads summary statistics for a dataset: number of videos, captioned, processed, and total frames.
    """
    if not dataset or dataset == "Select your dataset": # Explicitly handle the problematic string
        return {
            "Files": 0,
            "Captioned": 0,
            "Processed": 0,
            "Total frames/images": 0
        }
    
    base_dir, dataset_type = _get_dataset_base_dir(dataset)
    clean_dataset_name = dataset
    dataset_full_path = os.path.join(base_dir, clean_dataset_name)

    info_path = os.path.join(dataset_full_path, "info.json")
    captions_path = os.path.join(dataset_full_path, "captions.json")
    processed_path = os.path.join(dataset_full_path, "preprocessed_data", "processed.json")
    
    num_files = 0
    num_captioned = 0
    num_processed = 0
    total_frames_or_images = 0

    if dataset_type == "image":
        file_extensions = settings.IMAGE_EXTENSIONS
    else:
        file_extensions = settings.VIDEO_EXTENSIONS

    media_files = [f for ext in file_extensions for f in os.listdir(dataset_full_path) if f.lower().endswith(ext)]
    num_files = len(media_files)

    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            # Sum frames for videos, or count images for image datasets
            if dataset_type == "video":
                total_frames_or_images = sum(v.get("frames", 0) for v in info.values() if isinstance(v, dict))
            else: # image dataset
                total_frames_or_images = num_files # Each image is a "frame" in this context
        except Exception:
            pass
    
    if os.path.exists(captions_path):
        try:
            with open(captions_path, 'r', encoding='utf-8') as f:
                captions = json.load(f)
                num_captioned = sum(1 for entry in captions if entry.get("caption", "").strip())
        except Exception:
            pass
    
    if os.path.exists(processed_path):
        try:
            with open(processed_path, 'r', encoding='utf-8') as f:
                processed_map = json.load(f)
            num_processed = len(processed_map)
        except Exception:
            num_processed = 0
    
    return {
        "Files": num_files,
        "Captioned": num_captioned,
        "Processed": num_processed,
        "Total frames/images": total_frames_or_images
    }


def build_resolution_summary_from_info(dataset):
    """
    Builds a human-readable summary of media counts grouped by resolution
    using the dataset's info.json file.

    Example lines:
    - "4 videos - 1024x345"
    - "2 images - 304x345"
    """
    if not dataset or dataset == "Select your dataset":
        return []

    try:
        base_dir, _dataset_type = _get_dataset_base_dir(dataset)
        dataset_full_path = os.path.join(base_dir, dataset)
        info_path = os.path.join(dataset_full_path, "info.json")
    except Exception:
        return []

    if not os.path.exists(info_path):
        return []

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        return []

    if not isinstance(info, dict):
        return []

    # Group counts by (width, height) and media type (video/image)
    resolution_groups = {}
    for _name, meta in info.items():
        if not isinstance(meta, dict):
            continue
        width = meta.get("width")
        height = meta.get("height")
        if not width or not height:
            continue

        frames = meta.get("frames", 0) or 0
        fps = meta.get("fps", 0) or 0
        is_video = (frames and frames > 1) or (fps and fps > 0)

        key = (int(width), int(height))
        if key not in resolution_groups:
            resolution_groups[key] = {"videos": 0, "images": 0}
        if is_video:
            resolution_groups[key]["videos"] += 1
        else:
            resolution_groups[key]["images"] += 1

    if not resolution_groups:
        return []

    # Sort by area (width*height) descending, then width, then height
    sorted_items = sorted(
        resolution_groups.items(),
        key=lambda item: (item[0][0] * item[0][1], item[0][0], item[0][1]),
        reverse=True,
    )

    lines = []
    for (w, h), counts in sorted_items:
        if counts["videos"]:
            count = counts["videos"]
            label = "video" if count == 1 else "videos"
            lines.append(f"{count} {label} - {w}x{h}")
        if counts["images"]:
            count = counts["images"]
            label = "image" if count == 1 else "images"
            lines.append(f"{count} {label} - {w}x{h}")

    return lines

def generate_collage(thumbnails_dir, summary_path, target_w=settings.COLLAGE_WIDTH, target_h=settings.COLLAGE_HEIGHT):
    """
    Generates a collage image from all jpg thumbnails in a directory (except summary.jpg).
    """
    images = [os.path.join(thumbnails_dir, f) for f in os.listdir(thumbnails_dir)
              if f.endswith('.jpg') and f != 'summary.jpg']
    if not images:
        return False
    thumbs = [cv2.imread(img) for img in images if cv2.imread(img) is not None]
    if not thumbs:
        return False
    n = len(thumbs)
    best_cols = max(1, int(np.round(np.sqrt(n * (target_w/target_h)))))
    rows = (n + best_cols - 1) // best_cols
    scaled_thumbs = []
    for t in thumbs:
        h, w = t.shape[:2]
        scale = min(settings.THUMB_CELL_W / w, settings.THUMB_CELL_H / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(t, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_top = (settings.THUMB_CELL_H - new_h) // 2
        pad_bottom = settings.THUMB_CELL_H - new_h - pad_top
        pad_left = (settings.THUMB_CELL_W - new_w) // 2
        pad_right = settings.THUMB_CELL_W - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        scaled_thumbs.append(padded)
    grid_rows = []
    for r in range(rows):
        row_imgs = scaled_thumbs[r*best_cols:(r+1)*best_cols]
        while len(row_imgs) < best_cols:
            row_imgs.append(np.zeros((settings.THUMB_CELL_H, settings.THUMB_CELL_W, 3), dtype=np.uint8))
        grid_rows.append(np.hstack(row_imgs))
    collage = np.vstack(grid_rows)
    ch, cw = collage.shape[:2]
    if ch < target_h:
        pad_h = target_h - ch
        collage = np.pad(collage, ((pad_h//2, pad_h - pad_h//2), (0,0), (0,0)), mode='constant', constant_values=0)
    if cw < target_w:
        pad_w = target_w - cw
        collage = np.pad(collage, ((0,0), (pad_w//2, pad_w - pad_w//2), (0,0)), mode='constant', constant_values=0)
    ch, cw = collage.shape[:2]
    y0 = (ch - target_h) // 2
    x0 = (cw - target_w) // 2
    collage = collage[y0:y0+target_h, x0:x0+target_w]
    cv2.imwrite(summary_path, collage)
    return True

# =====================
# GUI-Building Functions
# =====================

# Global selected dataset state shared across training tabs
GLOBAL_TRAINING_SELECTED_DATASET = {"value": None}

def build_training_dataset_page_content(extra_right_controls=None):
    """
    Builds the main container for the training dataset selection page, including dropdown, summary, and controls.
    """
    # Share selection across instances (e.g., Config / Data Config tabs)
    selected_dataset = GLOBAL_TRAINING_SELECTED_DATASET
    selection_change_listeners = []
    num_repeats_change_listeners = []
    content_column_ref = ft.Ref[ft.Column]()
    dataset_dropdown_ref = ft.Ref[ft.Dropdown]()
    num_workers_field_ref = ft.Ref[ft.TextField]()

    def reload_current_dataset():
        col = content_column_ref.current
        if col is None:
            return
        folders = get_dataset_folders()
        # Sort dataset names A-Z by their display name
        folders = dict(sorted(folders.items(), key=lambda item: item[1].lower()))
        dataset_dropdown = dataset_dropdown_ref.current
        prev_selected = selected_dataset["value"]
        if dataset_dropdown:
            # Rebuild options with correct key/text mapping
            dropdown_options_map = {name: display_name for name, display_name in folders.items()}
            dataset_dropdown.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in dropdown_options_map.items()]
            dataset_dropdown.disabled = len(folders) == 0
            
            # prev_selected now holds the clean name (or None)
            if prev_selected and prev_selected in folders.keys(): # Check against clean names (keys)
                dataset_dropdown.value = prev_selected
                selected_dataset["value"] = prev_selected # Ensure selected_dataset also holds clean name
            else:
                # If no previous valid selection, try to select the first available dataset
                if folders:
                    first_dataset_key = list(folders.keys())[0]
                    dataset_dropdown.value = first_dataset_key
                    selected_dataset["value"] = first_dataset_key
                else:
                    dataset_dropdown.value = None # No datasets available, clear selection
                    selected_dataset["value"] = None
            dataset_dropdown.update()
        update_summary_row(force_summary_refresh=True)

    def build_controls():
        """
        Builds the top row controls: dataset dropdown, refresh button, and num workers field.
        """
        folders = get_dataset_folders()
        # Sort dataset names A-Z by their display name
        folders = dict(sorted(folders.items(), key=lambda item: item[1].lower()))
        # Prepare options for dropdown: key is clean name, text is display name
        dropdown_options_map = {name: display_name for name, display_name in folders.items()}
        dataset_dropdown = create_dropdown(
            "Select dataset",
            selected_dataset["value"], # This should store the clean name
            dropdown_options_map,
            hint_text="Select your dataset",
            fill_color=ft.Colors.with_opacity(0.18, ft.Colors.AMBER_900),
            expand=None,
            col=None,
            # Set a reasonable fixed width so it doesn't look overly wide
            # while restoring original text size/style from create_dropdown.
            
        )
        dataset_dropdown.width = 320
        dataset_dropdown_ref.current = dataset_dropdown
        dataset_dropdown.disabled = len(folders) == 0
        def on_dataset_change(e):
            selected_dataset["value"] = e.control.value if e.control.value else None
            update_summary_row()
            try:
                for cb in list(selection_change_listeners):
                    try:
                        cb(selected_dataset["value"])
                    except Exception:
                        pass
            except Exception:
                pass
        dataset_dropdown.on_change = on_dataset_change
        update_button = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Update dataset list",
            on_click=lambda e: reload_current_dataset(),
            style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=0, vertical=0)),
            icon_size=22
        )
        update_btn_container = ft.Container(update_button, alignment=ft.alignment.center_left, width=36)
        if not hasattr(build_controls, "num_workers"):
            build_controls.num_workers = 1
        if not hasattr(build_controls, "num_repeats"):
            build_controls.num_repeats = build_controls.num_workers
        def _handle_num_workers_update(e):
            current_input_value = e.control.value
            previous_valid_workers_value = getattr(build_controls, 'num_workers', 1)
            try:
                val = int(current_input_value)
                if val >= 0:
                    build_controls.num_workers = val
                    build_controls.num_repeats = val
                    # Trigger num_repeats change listeners
                    try:
                        for cb in list(num_repeats_change_listeners):
                            try:
                                cb(val)
                            except Exception:
                                pass
                    except Exception:
                        pass
                else:
                    e.control.value = str(previous_valid_workers_value)
            except ValueError:
                e.control.value = str(previous_valid_workers_value)
        num_workers_field = create_textfield(
            "num_repeats",
            str(getattr(build_controls, 'num_workers', 1)),
            expand=None
        )
        num_workers_field_ref.current = num_workers_field
        num_workers_field.width = 84
        num_workers_field.text_align = ft.TextAlign.CENTER
        num_workers_field.on_change = _handle_num_workers_update
        return ft.Row([
            dataset_dropdown,
            update_btn_container,
            num_workers_field,
        ], expand=True, spacing=0, alignment=ft.MainAxisAlignment.START)

    def build_summary_display_row():
        """
        Builds the row that displays the dataset summary and collage image.
        """
        row = ft.Row([
            ft.Container(
                key="summary_img_container",
                width=settings.COLLAGE_WIDTH,
                height=settings.COLLAGE_HEIGHT,
            ),
            ft.Column(
                key="summary_text_column",
                spacing=8,
                expand=True,
                scroll=ft.ScrollMode.ADAPTIVE,
            ),
        ], spacing=10, alignment=ft.MainAxisAlignment.START)
        return ft.Container(content=row, padding=ft.padding.only(left=30, right=8, top=8, bottom=8))

    def update_summary_row(force_summary_refresh: bool = False):
        """
        Updates the summary display row with the current dataset's summary and collage image.
        """
        page_col = content_column_ref.current
        if not page_col:
            return

        summary_img_container = None
        summary_text_column = None
        if len(page_col.controls) > 1:
            candidate = page_col.controls[1]
            if isinstance(candidate, ft.Container):
                inner = getattr(candidate, 'content', None)
            else:
                inner = candidate
            if isinstance(inner, ft.Row) and len(inner.controls) == 2:
                left = inner.controls[0]
                right = inner.controls[1]
                if isinstance(left, ft.Container) and getattr(left, 'key', None) == 'summary_img_container' and \
                   isinstance(right, ft.Column) and getattr(right, 'key', None) == 'summary_text_column':
                    summary_img_container = left
                    summary_text_column = right

        if not summary_img_container or not summary_text_column:
            return

        summary_img_container.content = None
        summary_text_column.controls.clear()

        current_selected_dataset = selected_dataset["value"]

        if not current_selected_dataset or str(current_selected_dataset).lower() == "none":
            summary_text_column.controls.append(
                ft.Text("Select a dataset", key="placeholder_select_dataset")
            )
        else:
            base_dir, dataset_type = _get_dataset_base_dir(current_selected_dataset)
            clean_dataset_name = current_selected_dataset
            thumbnails_base_dir = settings.THUMBNAILS_BASE_DIR

            thumbnails_dir = os.path.join(thumbnails_base_dir, clean_dataset_name)
            summary_path = os.path.join(thumbnails_dir, "summary.jpg")
            if not os.path.exists(thumbnails_dir):
                os.makedirs(thumbnails_dir, exist_ok=True)

            if not os.path.exists(thumbnails_dir):
                summary_text_column.controls.append(ft.Text(f"Thumbnails directory for {current_selected_dataset} not found or couldn't be created.", size=12))
            else:
                needs_summary = force_summary_refresh or not os.path.exists(summary_path)
                if needs_summary and os.path.exists(summary_path):
                    try:
                        os.remove(summary_path)
                    except Exception as e:
                        print(f"Error removing old summary image: {e}")

                if needs_summary:
                    try:
                        get_videos_and_thumbnails(clean_dataset_name, dataset_type, force_metadata_refresh=force_summary_refresh)
                    except Exception as e:
                        print(f"Error refreshing thumbnails for summary: {e}")
                    try:
                        generate_collage(thumbnails_dir, summary_path)
                    except Exception as e:
                        print(f"Error generating summary image: {e}")

                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        summary_img_container.content = ft.Image(
                            src_base64=encoded_string,
                            width=settings.COLLAGE_WIDTH - 2,
                            height=settings.COLLAGE_HEIGHT,
                            fit=ft.ImageFit.CONTAIN
                        )
                    except Exception as e:
                        print(f"Error loading summary image: {e}")
                        summary_text_column.controls.append(ft.Text(f"Error loading summary image: {e}", size=12))
                else:
                    print(f"Summary image not found at: {summary_path}")
                    summary_text_column.controls.append(ft.Text(f"Summary image not found at: {summary_path}", size=12))

                # Build and display resolution-based summary next to summary.jpg
                resolution_lines = build_resolution_summary_from_info(clean_dataset_name)
                if resolution_lines:
                    summary_text_column.controls.append(
                        ft.Text("Summary", size=13, weight=ft.FontWeight.BOLD)
                    )
                    for line in resolution_lines:
                        summary_text_column.controls.append(
                            ft.Text(line, size=12)
                        )

        if summary_img_container.page:
            summary_img_container.update()
        if summary_text_column.page:
            summary_text_column.update()

    # Build optional bottom-right Save area (moved under summary)
    bottom_right_controls = []
    if extra_right_controls:
        try:
            if isinstance(extra_right_controls, (list, tuple)):
                bottom_right_controls = list(extra_right_controls)
            else:
                bottom_right_controls = [extra_right_controls]
        except Exception:
            bottom_right_controls = []
    # Keep original size for primary button; no scaling

    bottom_actions_row = ft.Row(bottom_right_controls, alignment=ft.MainAxisAlignment.END, spacing=6)
    bottom_actions_container = ft.Container(content=bottom_actions_row, expand=True, padding=ft.padding.only(right=8))

    content_column = ft.Column(
        ref=content_column_ref,
        controls=[
            build_controls(),
            build_summary_display_row(),
            bottom_actions_container,
        ],
        scroll=ft.ScrollMode.ADAPTIVE,
        expand=True
    )
    container = ft.Container(content=content_column, expand=True, )
    def _on_mount_actions(e):
        update_summary_row()
    container.on_mount = _on_mount_actions

    # Expose selected dataset and num_workers for Save/Open
    container.get_selected_dataset = lambda: (print("Selected dataset:", selected_dataset["value"]), selected_dataset["value"])[1]
    container.get_num_workers = lambda: getattr(build_controls, 'num_workers', 1)
    container.get_num_repeats = lambda: getattr(build_controls, 'num_repeats', getattr(build_controls, 'num_workers', 1))

    # Add set_selected_dataset method
    def set_selected_dataset(dataset_name, page_ctx=None):
        dropdown = dataset_dropdown_ref.current
        folders = get_dataset_folders() # Get the clean_name: display_name map

        # dataset_name passed to this function should be the clean name
        if dataset_name is None or (dataset_name not in folders.keys()):
            selected_dataset["value"] = None
            if dropdown:
                dropdown.value = ""  # Key for "None" option
                if dropdown.page:
                    dropdown.update()
                    if page_ctx:
                        page_ctx.update()
        elif dataset_name in folders.keys(): # dataset_name is not None and is a valid clean name
            selected_dataset["value"] = dataset_name # Store the clean name
            if dropdown:
                dropdown.value = str(dataset_name) # Set dropdown value to the clean name (key)
                if dropdown.page:
                    dropdown.update()
                    if page_ctx:
                        page_ctx.update()
        
        update_summary_row()
    container.set_selected_dataset = set_selected_dataset

    # Add set_num_workers method
    _setting_num_workers = False  # Flag to prevent recursion

    def set_num_workers(num_workers, page_ctx=None):
        nonlocal _setting_num_workers
        if _setting_num_workers:  # Prevent recursion
            return

        try:
            val = int(num_workers)
            if val >= 0:
                build_controls.num_workers = val
                build_controls.num_repeats = val
                # Update textfield value in UI
                num_workers_field = num_workers_field_ref.current
                if num_workers_field:
                    # Always update the value
                    num_workers_field.value = str(val)

                    # Try to update with page context first
                    if getattr(num_workers_field, 'page', None):
                        num_workers_field.update()
                    else:
                        # Try to get page from page_ctx parameter
                        if page_ctx:
                            try:
                                # Temporarily assign page for update
                                original_page = getattr(num_workers_field, 'page', None)
                                num_workers_field.page = page_ctx
                                num_workers_field.update()
                                # Restore original page
                                if original_page:
                                    num_workers_field.page = original_page
                                # Can't delete page property, just leave it as is
                            except Exception:
                                pass

                # Trigger num_repeats change listeners (important for cross-tab sync)
                _setting_num_workers = True
                try:
                    for cb in list(num_repeats_change_listeners):
                        try:
                            cb(val)
                        except Exception:
                            pass
                except Exception:
                    pass
                finally:
                    _setting_num_workers = False
        except Exception as e:
            print(f"Error in set_num_workers: {e}")
    container.set_num_workers = set_num_workers
    container.set_num_repeats = set_num_workers

    # Allow external listeners to react to selection changes
    def add_on_selection_change(callback):
        if callable(callback):
            selection_change_listeners.append(callback)
    container.add_on_selection_change = add_on_selection_change

    # Allow external listeners to react to num_repeats changes
    def add_on_num_repeats_change(callback):
        if callable(callback):
            num_repeats_change_listeners.append(callback)
    container.add_on_num_repeats_change = add_on_num_repeats_change

    return container

# =====================
# Entry Point
# =====================
get_training_dataset_page_content = build_training_dataset_page_content
