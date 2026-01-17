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
    
    # Check captions.json first, fall back to counting .txt files
    if os.path.exists(captions_path):
        try:
            with open(captions_path, 'r', encoding='utf-8') as f:
                captions = json.load(f)
                # Count entries with non-empty caption field
                # Handle both list of dicts and dict formats
                if isinstance(captions, list):
                    num_captioned = sum(1 for entry in captions if isinstance(entry, dict) and entry.get("caption", "").strip())
                elif isinstance(captions, dict):
                    num_captioned = sum(1 for entry in captions.values() if isinstance(entry, dict) and entry.get("caption", "").strip())
        except Exception:
            pass
    else:
        # If captions.json doesn't exist, count .txt files as fallback
        from flet_app.ui.dataset_manager.dataset_utils import get_media_files
        try:
            base_dir, dataset_type = _get_dataset_base_dir(dataset)
            dataset_folder_path = os.path.join(base_dir, dataset)
            media_files = get_media_files(dataset_folder_path, dataset_type)
            num_captioned = 0
            for media_path in media_files:
                base_filename, _ = os.path.splitext(os.path.basename(media_path))
                txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")
                if os.path.exists(txt_caption_path):
                    with open(txt_caption_path, 'r', encoding='utf-8') as f:
                        caption_text = f.read().strip()
                        if caption_text:
                            num_captioned += 1
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
        base_dir, dataset_type = _get_dataset_base_dir(dataset)
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

        # Determine if video based on dataset type and frame count
        # For image datasets, always classify as images regardless of fps
        # For video datasets, only classify as video if frames > 1
        if dataset_type == "image":
            is_video = False
        else:
            # For video datasets, use frames > 1 as the primary indicator
            # fps alone is not enough (some images might have fps metadata)
            is_video = frames and frames > 1

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
                # Preserve the existing valid selection
                dataset_dropdown.value = prev_selected
                selected_dataset["value"] = prev_selected # Ensure selected_dataset also holds clean name
            elif prev_selected and prev_selected not in folders.keys():
                # Previous selection no longer exists - select first available
                if folders:
                    first_dataset_key = list(folders.keys())[0]
                    dataset_dropdown.value = first_dataset_key
                    selected_dataset["value"] = first_dataset_key
                else:
                    dataset_dropdown.value = None # No datasets available, clear selection
                    selected_dataset["value"] = None
            else:
                # Was empty before, keep it empty
                dataset_dropdown.value = None
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
                        pass

                if needs_summary:
                    try:
                        get_videos_and_thumbnails(clean_dataset_name, dataset_type, force_metadata_refresh=force_summary_refresh)
                    except Exception as e:
                        pass
                    try:
                        generate_collage(thumbnails_dir, summary_path)
                    except Exception as e:
                        pass

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
                        summary_text_column.controls.append(ft.Text(f"Error loading summary image: {e}", size=12))
                else:
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
            pass
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


def build_compact_dataset_block(label: str, initial_dataset: str = None):
    """
    Builds a compact, independent dataset block for use in training config.
    Each block has its own independent state (not shared).

    Args:
        label: Label for the dropdown (e.g., "Dataset 1", "Dataset 2", etc.)
        initial_dataset: Optional initial dataset name

    Returns:
        A container with dropdown, image, and summary text
    """
    # Independent state for this block
    selected_dataset = {"value": initial_dataset}
    frame_extraction_value = {"value": "head"}  # Default frame_extraction value
    is_ltx2_model = {"value": False}  # Track if model type is ltx-video-2
    num_repeats_value = {"value": 1}  # Default num_repeats value
    content_column_ref = ft.Ref[ft.Column]()
    dataset_dropdown_ref = ft.Ref[ft.Dropdown]()
    clear_button_ref = ft.Ref[ft.IconButton]()
    frame_extraction_dropdown_ref = ft.Ref[ft.Dropdown]()
    frame_extraction_row_ref = ft.Ref[ft.Row]()
    num_repeats_field_ref = ft.Ref[ft.TextField]()
    num_repeats_column_ref = ft.Ref[ft.Column]()

    # Smaller image size for compact display
    compact_width = int(settings.COLLAGE_WIDTH * 0.75)
    compact_height = int(settings.COLLAGE_HEIGHT * 0.75)

    def reload_current_dataset():
        col = content_column_ref.current
        if col is None:
            return
        folders = get_dataset_folders()
        folders = dict(sorted(folders.items(), key=lambda item: item[1].lower()))
        dataset_dropdown = dataset_dropdown_ref.current
        prev_selected = selected_dataset["value"]
        if dataset_dropdown:
            dropdown_options_map = {name: display_name for name, display_name in folders.items()}
            dataset_dropdown.options = [ft.dropdown.Option(key=name, text=display_name) for name, display_name in dropdown_options_map.items()]
            dataset_dropdown.disabled = len(folders) == 0

            if prev_selected and prev_selected in folders.keys():
                # Preserve the existing valid selection
                dataset_dropdown.value = prev_selected
                selected_dataset["value"] = prev_selected
            elif prev_selected and prev_selected not in folders.keys():
                # Previous selection no longer exists - select first available
                if folders:
                    first_dataset_key = list(folders.keys())[0]
                    dataset_dropdown.value = first_dataset_key
                    selected_dataset["value"] = first_dataset_key
                else:
                    dataset_dropdown.value = None
                    selected_dataset["value"] = None
            else:
                # Was empty before, keep it empty
                dataset_dropdown.value = None
                selected_dataset["value"] = None
            dataset_dropdown.update()
        update_summary_display()
        update_clear_button_visibility()

    def ensure_frame_extraction_in_dataset_toml(dataset_name):
        """Ensure frame_extraction field exists in the dataset's TOML file. Add 'head' if missing."""
        if not dataset_name:
            return
        try:
            import toml
            from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir

            base_dir, _dtype = _get_dataset_base_dir(dataset_name)
            # Config file is named after the dataset in the datasets folder (e.g., f_ghoul.toml)
            config_file = os.path.join(base_dir, f"{dataset_name}.toml")

            if not os.path.exists(config_file):
                # File doesn't exist, create it with frame_extraction
                with open(config_file, 'w') as f:
                    f.write('frame_extraction = "head"\n')
                print(f"Created {config_file} with frame_extraction = \"head\"")
                return

            # Read existing file
            with open(config_file, 'r') as f:
                content = f.read()

            # Check if frame_extraction already exists
            if 'frame_extraction' in content:
                # Already exists, no need to add
                return

            # Add frame_extraction after num_repeats line or before [[directory]]
            lines = content.split('\n')
            new_lines = []
            inserted = False

            for line in lines:
                # Insert after num_repeats line if found
                if not inserted and line.strip().startswith('num_repeats'):
                    new_lines.append(line)
                    new_lines.append('frame_extraction = "head"')
                    inserted = True
                else:
                    new_lines.append(line)

            # If still not inserted, add it before [[directory]]
            if not inserted:
                final_lines = []
                for line in new_lines:
                    if not inserted and line.strip().startswith('[[directory]]'):
                        final_lines.append('frame_extraction = "head"')
                        inserted = True
                    final_lines.append(line)
                new_lines = final_lines

            # Write back to file
            with open(config_file, 'w') as f:
                f.write('\n'.join(new_lines))
        except Exception as ex:
            pass

    def read_frame_extraction_from_dataset_toml(dataset_name):
        """Read frame_extraction value from dataset's TOML file and update UI."""
        if not dataset_name:
            return
        try:
            import toml
            from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir

            base_dir, _dtype = _get_dataset_base_dir(dataset_name)
            config_file = os.path.join(base_dir, f"{dataset_name}.toml")

            if not os.path.exists(config_file):
                # File doesn't exist, use default
                return

            # Read file and parse for frame_extraction
            with open(config_file, 'r') as f:
                content = f.read()

            # Parse frame_extraction value from TOML
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('frame_extraction'):
                    # Parse the value: frame_extraction = "head"
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        # Update internal state and UI
                        frame_extraction_value["value"] = value
                        dropdown = frame_extraction_dropdown_ref.current
                        if dropdown and dropdown.value != value:
                            dropdown.value = value
                            try:
                                dropdown.update()
                            except Exception:
                                pass
                        return
        except Exception as ex:
            pass

    def read_num_repeats_from_dataset_toml(dataset_name):
        """Read num_repeats value from dataset's TOML file and update UI."""
        if not dataset_name:
            return
        try:
            import toml
            from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir

            base_dir, _dtype = _get_dataset_base_dir(dataset_name)
            config_file = os.path.join(base_dir, f"{dataset_name}.toml")

            if not os.path.exists(config_file):
                # File doesn't exist, use default
                return

            # Read file and parse for num_repeats
            with open(config_file, 'r') as f:
                content = f.read()

            # Parse num_repeats value from TOML
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('num_repeats'):
                    # Parse the value: num_repeats = 2
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip()
                        try:
                            num_repeats_int = int(value)
                            # Update internal state and UI
                            num_repeats_value["value"] = num_repeats_int
                            dropdown = num_repeats_field_ref.current
                            if dropdown and dropdown.value != str(num_repeats_int):
                                dropdown.value = str(num_repeats_int)
                                try:
                                    dropdown.update()
                                except Exception:
                                    pass
                            return
                        except ValueError:
                            pass
        except Exception as ex:
            pass

    def update_summary_display():
        col = content_column_ref.current
        if not col:
            return

        # Find the image container and summary text column
        summary_img_container = None
        summary_text_column = None
        if len(col.controls) >= 2:
            if isinstance(col.controls[1], ft.Container):
                summary_img_container = col.controls[1]
            # The summary info is now in a Row (controls[2])
            if len(col.controls) >= 3 and isinstance(col.controls[2], ft.Row):
                summary_row = col.controls[2]
                # Get the first column from the row (summary_text_column)
                if len(summary_row.controls) >= 1 and isinstance(summary_row.controls[0], ft.Column):
                    summary_text_column = summary_row.controls[0]

        if not summary_img_container:
            return

        # Clear image content
        summary_img_container.content = None
        if summary_text_column:
            summary_text_column.controls.clear()

        current_selected = selected_dataset["value"]

        # Show/hide num_repeats based on dataset selection
        has_selection = bool(current_selected)
        num_repeats_col = num_repeats_column_ref.current
        num_repeats_field = num_repeats_field_ref.current
        if num_repeats_col:
            num_repeats_col.visible = has_selection
            try:
                if num_repeats_col.page:
                    num_repeats_col.update()
            except Exception:
                pass
        if num_repeats_field:
            num_repeats_field.visible = has_selection
            try:
                if num_repeats_field.page:
                    num_repeats_field.update()
            except Exception:
                pass

        if not current_selected or str(current_selected).lower() == "none":
            if summary_text_column:
                summary_text_column.controls.append(
                    ft.Text("No dataset selected", size=11, italic=True)
                )
        else:
            base_dir, dataset_type = _get_dataset_base_dir(current_selected)
            clean_dataset_name = current_selected
            thumbnails_base_dir = settings.THUMBNAILS_BASE_DIR

            thumbnails_dir = os.path.join(thumbnails_base_dir, clean_dataset_name)
            summary_path = os.path.join(thumbnails_dir, "summary.jpg")
            if not os.path.exists(thumbnails_dir):
                os.makedirs(thumbnails_dir, exist_ok=True)

            # Generate summary if needed
            if not os.path.exists(summary_path):
                try:
                    get_videos_and_thumbnails(clean_dataset_name, dataset_type, force_metadata_refresh=False)
                except Exception:
                    pass
                try:
                    generate_collage(thumbnails_dir, summary_path, target_w=compact_width, target_h=compact_height)
                except Exception:
                    pass

            # Load and display image
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    summary_img_container.content = ft.Image(
                        src_base64=encoded_string,
                        width=compact_width - 2,
                        height=compact_height,
                        fit=ft.ImageFit.CONTAIN
                    )
                except Exception:
                    if summary_text_column:
                        summary_text_column.controls.append(ft.Text("Error loading image", size=11))
            else:
                if summary_text_column:
                    summary_text_column.controls.append(ft.Text("No preview", size=11))

            # Build summary text
            summary_data = load_dataset_summary(clean_dataset_name)

            # Get dataset type (primary way to determine if video or image dataset)
            base_dir, dataset_type = _get_dataset_base_dir(current_selected)

            # Count files based on dataset type
            num_videos = 0
            num_images = 0

            if dataset_type == "video":
                num_videos = summary_data['Files']
            elif dataset_type == "image":
                num_images = summary_data['Files']

            if summary_text_column:
                # Add video count if > 0
                if num_videos > 0:
                    summary_text_column.controls.append(
                        ft.Text(f"videos - {num_videos}", size=11)
                    )
                # Add image count if > 0
                if num_images > 0:
                    summary_text_column.controls.append(
                        ft.Text(f"images - {num_images}", size=11)
                    )
                # Add captioned count if > 0
                captioned_count = summary_data.get('Captioned', 0)
                if captioned_count > 0:
                    summary_text_column.controls.append(
                        ft.Text(f"captioned - {captioned_count}", size=11)
                    )

        if summary_img_container.page:
            summary_img_container.update()
        if summary_text_column and summary_text_column.page:
            summary_text_column.update()

    def update_clear_button_visibility():
        """Show/hide the clear button based on whether a dataset is selected."""
        clear_btn = clear_button_ref.current
        if clear_btn:
            has_selection = bool(selected_dataset["value"])
            clear_btn.visible = has_selection
            try:
                if clear_btn.page:
                    clear_btn.update()
            except Exception:
                pass

    def build_controls():
        folders = get_dataset_folders()
        folders = dict(sorted(folders.items(), key=lambda item: item[1].lower()))
        dropdown_options_map = {name: display_name for name, display_name in folders.items()}

        dataset_dropdown = ft.Dropdown(
            label=label,
            hint_text="Select dataset",
            options=[ft.dropdown.Option(key=name, text=display_name) for name, display_name in dropdown_options_map.items()],
            value=selected_dataset["value"],
            fill_color=ft.Colors.GREY_900,
            filled=True,
            expand=True,  # Expand to fill available space (button will be fixed width)
            text_size=12,
            label_style=ft.TextStyle(size=11),
            scale=0.8,  # Scale to 80% like other dropdowns
        )
        dataset_dropdown_ref.current = dataset_dropdown
        dataset_dropdown.disabled = len(folders) == 0

        def on_dataset_change(e):
            selected_dataset["value"] = e.control.value if e.control.value else None
            update_summary_display()
            update_clear_button_visibility()
            # Update num_repeats visibility
            update_num_repeats_visibility()

            # If model type is ltx-video-2 and dataset is selected, ensure frame_extraction exists in dataset TOML
            if is_ltx2_model["value"] and selected_dataset["value"]:
                ensure_frame_extraction_in_dataset_toml(selected_dataset["value"])
                # Read frame_extraction from dataset TOML and update UI
                read_frame_extraction_from_dataset_toml(selected_dataset["value"])
            # Read num_repeats from dataset TOML and update UI
            read_num_repeats_from_dataset_toml(selected_dataset["value"])

        dataset_dropdown.on_change = on_dataset_change

        def on_clear_click(e):
            selected_dataset["value"] = None
            if dataset_dropdown:
                dataset_dropdown.value = None
                try:
                    dataset_dropdown.update()
                except Exception:
                    pass
            update_summary_display()
            update_clear_button_visibility()

        clear_button = ft.IconButton(
            icon=ft.Icons.CLOSE,
            tooltip="Clear dataset selection",
            on_click=on_clear_click,
            visible=bool(selected_dataset["value"]),
            icon_size=16,  # Slightly smaller icon
            width=32,  # Fixed width for the button
        )
        clear_button_ref.current = clear_button

        return ft.Row(
            [dataset_dropdown, clear_button],
            spacing=4,
            expand=True,
        )

    def on_num_repeats_change(e):
        """Handle num_repeats field change - update dataset config file."""
        new_value = e.control.value if e.control.value else None
        if new_value:
            try:
                # Validate and store as integer
                num_repeats_value["value"] = int(new_value)
            except ValueError:
                # Invalid input, reset to previous valid value
                field = num_repeats_field_ref.current
                if field:
                    field.value = str(num_repeats_value["value"])
                    try:
                        field.update()
                    except Exception:
                        pass
                return

            # Update the dataset's TOML config file
            current_dataset = selected_dataset.get("value")
            if current_dataset:
                try:
                    import toml
                    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir

                    base_dir, _dtype = _get_dataset_base_dir(current_dataset)
                    # Config file is named after the dataset in the datasets folder (e.g., f_ghoul.toml)
                    config_file = os.path.join(base_dir, f"{current_dataset}.toml")

                    # Read existing file content
                    content = ""
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            content = f.read()

                    # Check if num_repeats already exists in the file
                    if 'num_repeats' in content:
                        # Replace existing num_repeats line
                        lines = content.split('\n')
                        new_lines = []
                        for line in lines:
                            if line.strip().startswith('num_repeats'):
                                new_lines.append(f'num_repeats = {num_repeats_value["value"]}')
                            else:
                                new_lines.append(line)
                        content = '\n'.join(new_lines)
                    else:
                        # Add num_repeats after resolutions or before [[directory]]
                        lines = content.split('\n')
                        new_lines = []
                        inserted = False

                        for line in lines:
                            # Insert after resolutions line if found
                            if not inserted and line.strip().startswith('resolutions'):
                                new_lines.append(line)
                                new_lines.append(f'num_repeats = {num_repeats_value["value"]}')
                                inserted = True
                            else:
                                new_lines.append(line)

                        # If still not inserted, add it before [[directory]]
                        if not inserted:
                            final_lines = []
                            for line in new_lines:
                                if not inserted and line.strip().startswith('[[directory]]'):
                                    final_lines.append(f'num_repeats = {num_repeats_value["value"]}')
                                    inserted = True
                                final_lines.append(line)
                            new_lines = final_lines

                        content = '\n'.join(new_lines)

                    # Write back to file
                    with open(config_file, 'w') as f:
                        f.write(content)

                except Exception as ex:
                    pass

    def on_frame_extraction_change(e):
        """Handle frame_extraction dropdown change - update dataset config file."""
        new_value = e.control.value if e.control.value else None
        if new_value:
            frame_extraction_value["value"] = new_value

            # Update the dataset's TOML config file
            current_dataset = selected_dataset.get("value")
            if current_dataset:
                try:
                    import toml
                    from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir

                    base_dir, _dtype = _get_dataset_base_dir(current_dataset)
                    # Config file is named after the dataset in the datasets folder (e.g., f_ghoul.toml)
                    config_file = os.path.join(base_dir, f"{current_dataset}.toml")

                    # Read existing file content
                    content = ""
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            content = f.read()

                    # Check if frame_extraction already exists in the file
                    if 'frame_extraction' in content:
                        # Replace existing frame_extraction line
                        lines = content.split('\n')
                        new_lines = []
                        for line in lines:
                            if line.strip().startswith('frame_extraction'):
                                new_lines.append(f'frame_extraction = "{new_value}"')
                            else:
                                new_lines.append(line)
                        content = '\n'.join(new_lines)
                    else:
                        # Add frame_extraction after num_repeats line or before [[directory]]
                        lines = content.split('\n')
                        new_lines = []
                        inserted = False

                        for line in lines:
                            # Insert after num_repeats line if found
                            if not inserted and line.strip().startswith('num_repeats'):
                                new_lines.append(line)
                                new_lines.append(f'frame_extraction = "{new_value}"')
                                inserted = True
                            else:
                                new_lines.append(line)

                        # If still not inserted, add it before [[directory]]
                        if not inserted:
                            final_lines = []
                            for line in new_lines:
                                if not inserted and line.strip().startswith('[[directory]]'):
                                    final_lines.append(f'frame_extraction = "{new_value}"')
                                    inserted = True
                                final_lines.append(line)
                            new_lines = final_lines

                        content = '\n'.join(new_lines)

                    # Write back to file
                    with open(config_file, 'w') as f:
                        f.write(content)
                except Exception as ex:
                    pass

    content_column = ft.Column(
        ref=content_column_ref,
        controls=[
            build_controls(),
            ft.Container(
                key="summary_img_container",
                width=compact_width,
                height=compact_height,
                bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREY_800),
                border_radius=ft.border_radius.all(5),
            ),
            # Summary row with text info (left, col=7) and num_repeats (right, col=5)
            ft.Row(
                [
                    ft.Column(
                        key="summary_text_column",
                        spacing=4,
                        scroll=ft.ScrollMode.ADAPTIVE,
                        alignment=ft.MainAxisAlignment.START,
                        horizontal_alignment=ft.CrossAxisAlignment.START,
                        expand=True,
                    ),
                    ft.Column(
                        ref=num_repeats_column_ref,
                        visible=False,  # Hidden by default
                        spacing=4,
                        alignment=ft.MainAxisAlignment.END,
                        horizontal_alignment=ft.CrossAxisAlignment.END,
                        controls=[
                            ft.TextField(
                                ref=num_repeats_field_ref,
                                label="num_repeats",
                                value="1",
                                width=70,
                                height=36,  # Fix height to reduce vertical size
                                text_size=11,
                                label_style=ft.TextStyle(size=10),
                                content_padding=ft.padding.symmetric(horizontal=6, vertical=2),
                                fill_color=ft.Colors.GREY_900,
                                filled=True,
                                visible=False,  # Hidden by default
                                on_change=lambda e: on_num_repeats_change(e),
                            ),
                        ],
                    ),
                ],
                spacing=4,
                expand=True,
            ),
            # Frame extraction dropdown (hidden by default, shown only for ltx-video-2)
            ft.Row(
                ref=frame_extraction_row_ref,
                controls=[
                    ft.Dropdown(
                        ref=frame_extraction_dropdown_ref,
                        label="Frame extraction",
                        hint_text="Select frame extraction mode",
                        options=[
                            ft.dropdown.Option(key="head", text="head"),
                            ft.dropdown.Option(key="chunk", text="chunk"),
                            ft.dropdown.Option(key="slide", text="slide"),
                            ft.dropdown.Option(key="uniform", text="uniform"),
                            ft.dropdown.Option(key="full", text="full"),
                        ],
                        value="head",
                        fill_color=ft.Colors.GREY_900,
                        filled=True,
                        expand=True,
                        text_size=11,
                        label_style=ft.TextStyle(size=10),
                        scale=0.75,
                        visible=False,  # Hidden by default
                        on_change=lambda e: on_frame_extraction_change(e),
                    ),
                ],
                visible=False,  # Row is hidden by default
            ),
        ],
        spacing=8,
        horizontal_alignment=ft.CrossAxisAlignment.START,
    )

    def _on_mount_actions(e):
        update_summary_display()
        update_clear_button_visibility()
        # Also update num_repeats visibility on mount
        update_num_repeats_visibility()
    content_column.on_mount = _on_mount_actions

    def update_num_repeats_visibility():
        """Update num_repeats visibility based on dataset selection."""
        has_selection = bool(selected_dataset["value"])
        num_repeats_col = num_repeats_column_ref.current
        num_repeats_field = num_repeats_field_ref.current

        # Hide both column and field
        if num_repeats_col:
            num_repeats_col.visible = has_selection
            try:
                if num_repeats_col.page:
                    num_repeats_col.update()
            except Exception:
                pass

        if num_repeats_field:
            num_repeats_field.visible = has_selection
            try:
                if num_repeats_field.page:
                    num_repeats_field.update()
            except Exception:
                pass

    # Expose methods for external access
    container = ft.Container(content=content_column, expand=True)

    def get_selected_dataset():
        return selected_dataset["value"]

    def set_selected_dataset(dataset_name, page_ctx=None):
        dropdown = dataset_dropdown_ref.current
        folders = get_dataset_folders()

        if dataset_name is None or (dataset_name not in folders.keys()):
            selected_dataset["value"] = None
            if dropdown:
                dropdown.value = None
                if dropdown.page:
                    dropdown.update()
                if page_ctx:
                    page_ctx.update()
        elif dataset_name in folders.keys():
            selected_dataset["value"] = dataset_name
            if dropdown:
                dropdown.value = str(dataset_name)
                if dropdown.page:
                    dropdown.update()
                if page_ctx:
                    page_ctx.update()
        update_summary_display()
        update_clear_button_visibility()
        update_num_repeats_visibility()

        # If model type is ltx-video-2 and dataset is selected, ensure frame_extraction exists in dataset TOML
        if is_ltx2_model["value"] and selected_dataset["value"]:
            ensure_frame_extraction_in_dataset_toml(selected_dataset["value"])
            # Read frame_extraction from dataset TOML and update UI
            read_frame_extraction_from_dataset_toml(selected_dataset["value"])
        # Read num_repeats from dataset TOML and update UI
        read_num_repeats_from_dataset_toml(selected_dataset["value"])

    def reload_datasets():
        reload_current_dataset()

    def get_frame_extraction():
        """Get the current frame_extraction value."""
        dropdown = frame_extraction_dropdown_ref.current
        if dropdown:
            return dropdown.value
        return frame_extraction_value["value"]

    def set_frame_extraction(value, page_ctx=None):
        """Set the frame_extraction value."""
        dropdown = frame_extraction_dropdown_ref.current
        frame_extraction_value["value"] = value
        if dropdown:
            dropdown.value = value
            if dropdown.page:
                dropdown.update()
            if page_ctx:
                page_ctx.update()

    def set_frame_extraction_visible(visible: bool):
        """Show or hide the frame_extraction dropdown based on model type."""
        # Store the model type state
        is_ltx2_model["value"] = visible

        row = frame_extraction_row_ref.current
        dropdown = frame_extraction_dropdown_ref.current
        try:
            if row:
                row.visible = visible
                if row.page:
                    row.update()
            if dropdown:
                dropdown.visible = visible
                if dropdown.page:
                    dropdown.update()
        except Exception:
            pass

    container.get_selected_dataset = get_selected_dataset
    container.set_selected_dataset = set_selected_dataset
    container.reload_datasets = reload_datasets
    container.get_frame_extraction = get_frame_extraction
    container.set_frame_extraction = set_frame_extraction
    container.set_frame_extraction_visible = set_frame_extraction_visible
    container.get_num_repeats = lambda: num_repeats_value["value"]
    container.set_num_repeats = lambda val: _set_num_repeats(val)

    def _set_num_repeats(value, page_ctx=None):
        """Set num_repeats value and update UI."""
        try:
            num_repeats_int = int(value)
        except (ValueError, TypeError):
            num_repeats_int = 1
        num_repeats_value["value"] = num_repeats_int
        field = num_repeats_field_ref.current
        if field:
            field.value = str(num_repeats_int)
            try:
                field.update()
            except Exception:
                pass
            if page_ctx:
                page_ctx.update()

    return container


get_compact_dataset_block = build_compact_dataset_block
