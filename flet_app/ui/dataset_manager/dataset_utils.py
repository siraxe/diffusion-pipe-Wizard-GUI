import os
import glob
import cv2
import json
import flet as ft # Keep flet import for type hints if needed, but not for UI elements
from flet_app.settings import settings
import asyncio # Keep for async functions if any remain
import subprocess # Keep for subprocess if any remain
import shutil # Keep for shutil if any remain
import time # Keep for time if any remain

# ======================================================================================
# Data & Utility Functions (File I/O, data parsing, validation)
# ======================================================================================

def parse_bucket_string_to_list(raw_bucket_str: str) -> list[int] | None:
    raw_bucket_str = raw_bucket_str.strip()
    try:
        if raw_bucket_str.startswith('[') and raw_bucket_str.endswith(']'):
            parsed_list = json.loads(raw_bucket_str)
            if isinstance(parsed_list, list) and len(parsed_list) == 3 and all(isinstance(i, int) for i in parsed_list):
                return parsed_list
        elif 'x' in raw_bucket_str.lower():
            parts_x = raw_bucket_str.lower().split('x')
            if len(parts_x) == 3 and all(p.strip().isdigit() for p in parts_x):
                return [int(p.strip()) for p in parts_x]
    except (json.JSONDecodeError, ValueError):
        return None
    return None

def format_bucket_list_to_string(bucket_list: list) -> str:
    if isinstance(bucket_list, list) and len(bucket_list) == 3 and all(isinstance(i, (int, float)) for i in bucket_list):
        return f"[{bucket_list[0]}, {bucket_list[1]}, {bucket_list[2]}]"
    return settings.DEFAULT_BUCKET_SIZE_STR

def load_dataset_config(dataset_name: str | None) -> tuple[str, str, str]:
    bucket_to_set = settings.DEFAULT_BUCKET_SIZE_STR
    model_to_set = settings.train_def_model
    trigger_word_to_set = ''
    if dataset_name:
        base_dir, _ = _get_dataset_base_dir(dataset_name)
        dataset_info_json_path = os.path.join(base_dir, dataset_name, "info.json")
        if os.path.exists(dataset_info_json_path):
            try:
                with open(dataset_info_json_path, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict):
                        # Read from top level of loaded_data
                        bucket_res_val = loaded_data.get("bucket_resolution") # Corresponds to 'size_val'
                        model_name_val = loaded_data.get("model_name")       # Corresponds to 'type_val'
                        trigger_word_val = loaded_data.get("trigger_word")

                        # bucket_resolution can be a list or a string representation of a list
                        if isinstance(bucket_res_val, list):
                            bucket_to_set = format_bucket_list_to_string(bucket_res_val)
                        elif isinstance(bucket_res_val, str):
                            # Attempt to parse if it's a string like "[512, 512, 49]"
                            parsed_list = parse_bucket_string_to_list(bucket_res_val)
                            if parsed_list:
                                bucket_to_set = format_bucket_list_to_string(parsed_list)
                            else: # If string is not parsable, keep it as is or use default
                                bucket_to_set = bucket_res_val # Or settings.DEFAULT_BUCKET_SIZE_STR if strict parsing needed
                        
                        if isinstance(model_name_val, str):
                            model_to_set = model_name_val
                        if isinstance(trigger_word_val, str):
                            trigger_word_to_set = trigger_word_val
            except (json.JSONDecodeError, IOError):
                pass
    return bucket_to_set, model_to_set, trigger_word_to_set

def save_dataset_config(dataset_name: str, bucket_str: str, model_name: str, trigger_word: str) -> bool:
    base_dir, _ = _get_dataset_base_dir(dataset_name)
    dataset_info_json_path = os.path.join(base_dir, dataset_name, "info.json")
    
    bucket_str_val = bucket_str # Initialize with the input value

    parsed_bucket_list = parse_bucket_string_to_list(bucket_str)
    if parsed_bucket_list is None:
        # This part needs to be handled by the caller (e.g., dataset_actions.py)
        # as it involves Flet UI elements (e.page.snack_bar)
        # For now, we'll just use the default if parsing fails.
        bucket_str_val = settings.DEFAULT_BUCKET_SIZE_STR

    # Load existing config to preserve other keys
    existing_config = {}
    if os.path.exists(dataset_info_json_path):
        try:
            with open(dataset_info_json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip(): # Check if file is not empty
                    existing_config = json.loads(content)
                    if not isinstance(existing_config, dict): # Ensure it's a dict
                        existing_config = {}
                else:
                    existing_config = {} # Empty file, treat as empty dict
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing info.json for merging: {e}")
            existing_config = {} # Fallback to empty if loading fails

    # Update only the relevant keys
    existing_config["bucket_resolution"] = bucket_str_val
    existing_config["model_name"] = model_name
    existing_config["trigger_word"] = trigger_word

    try:
        os.makedirs(os.path.dirname(dataset_info_json_path), exist_ok=True)
        with open(dataset_info_json_path, "w", encoding='utf-8') as f:
            json.dump(existing_config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving dataset config: {e}")
    return False

def load_processed_map(dataset_name: str) -> dict | None:
    base_dir, _ = _get_dataset_base_dir(dataset_name)
    processed_json_path = os.path.join(base_dir, dataset_name, "preprocessed_data", "processed.json")
    if os.path.exists(processed_json_path):
        try:
            with open(processed_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_dataset_captions(dataset_name: str) -> list:
    base_dir, dataset_type = _get_dataset_base_dir(dataset_name)
    dataset_folder_path = os.path.join(base_dir, dataset_name)
    media_files = get_media_files(dataset_folder_path, dataset_type)
    
    captions_data = []
    for media_path in media_files:
        base_filename, _ = os.path.splitext(os.path.basename(media_path))
        txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")
        
        caption_text = ""
        if os.path.exists(txt_caption_path):
            with open(txt_caption_path, 'r', encoding='utf-8') as f:
                caption_text = f.read().strip()
        
        captions_data.append({
            "media_path": os.path.basename(media_path),
            "caption": caption_text
        })
        
    return captions_data

def validate_bucket_values(W_val, H_val, F_val) -> list[str]:
    errors = []
    if W_val is None or not isinstance(W_val, int) or W_val <= 0 or W_val % 32 != 0:
        errors.append(f"Width ({W_val}) must be a positive integer divisible by 32.")
    if H_val is None or not isinstance(H_val, int) or H_val <= 0 or H_val % 32 != 0:
        errors.append(f"Height ({H_val}) must be a positive integer divisible by 32.")
    # Adjusted validation for Frames based on typical dataset preprocessing requirements
    if F_val is None or not isinstance(F_val, int) or F_val <= 0:
         errors.append(f"Frames ({F_val}) must be a positive integer.")
    # Special case: Allow 1 frame for images, otherwise validate as video (4n+1 and >=5)
    if F_val is not None and F_val != 1:
        if F_val < 5 or (F_val - 1) % 4 != 0:
            errors.append(f"Frames ({F_val}) invalid (must be 1 for images or ≥5 and 4n+1 for videos).")
    return errors

def _get_dataset_base_dir(dataset_name: str) -> tuple[str, str]:
    base_dir = settings.DATASETS_DIR
    folder = os.path.join(base_dir, dataset_name)
    dtype = "video"
    try:
        if os.path.isdir(folder):
            img_exts = [e.lower().lstrip('.') for e in settings.IMAGE_EXTENSIONS]
            vid_exts = [e.lower().lstrip('.') for e in settings.VIDEO_EXTENSIONS]
            entries = os.listdir(folder)
            has_video = any(os.path.splitext(f)[1].lower().lstrip('.') in vid_exts for f in entries)
            has_image = any(os.path.splitext(f)[1].lower().lstrip('.') in img_exts for f in entries)
            if has_image and not has_video:
                dtype = "image"
            else:
                dtype = "video"
    except Exception:
        dtype = "video"
    return base_dir, dtype

#Helper to generate thumbnail for a video
def regenerate_all_thumbnails_for_dataset(dataset_name):
    # import glob # This import can be at top level
    dataset_path = os.path.join(settings.DATASETS_DIR, dataset_name)
    thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        return
    os.makedirs(thumbnails_dir, exist_ok=True)
    for ext in settings.VIDEO_EXTENSIONS:
        for video_path in glob.glob(os.path.join(dataset_path, f"*{ext}")) + glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")):
            video_name = os.path.basename(video_path)
            thumbnail_name = f"{os.path.splitext(video_name)[0]}.jpg"
            thumbnail_path = os.path.join(thumbnails_dir, thumbnail_name)
            if os.path.exists(thumbnail_path):
                try:
                    os.remove(thumbnail_path)
                except Exception:
                    pass
            generate_thumbnail(video_path, thumbnail_path)

def generate_thumbnail(video_path, thumbnail_path):
    try:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            return False
        success, image = vid.read()
        if success:
            orig_h, orig_w = image.shape[:2]
            orig_aspect_ratio = orig_w / orig_h
            crop_w, crop_h = orig_w, orig_h
            x_offset, y_offset = 0, 0
            if orig_aspect_ratio > settings.TARGET_ASPECT_RATIO:
                crop_w = int(orig_h * settings.TARGET_ASPECT_RATIO)
                x_offset = int((orig_w - crop_w) / 2)
            elif orig_aspect_ratio < settings.TARGET_ASPECT_RATIO:
                crop_h = int(orig_w / settings.TARGET_ASPECT_RATIO)
                y_offset = int((orig_h - crop_h) / 2)
            
            if crop_w <= 0 or crop_h <= 0 or x_offset < 0 or y_offset < 0 or \
               (y_offset + crop_h) > orig_h or (x_offset + crop_w) > orig_w:
                cropped_image = image
            else:
                cropped_image = image[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]

            if cropped_image.size == 0:
                vid.release()
                return False
            
            thumbnail_image = cv2.resize(cropped_image, (settings.THUMB_TARGET_W, settings.THUMB_TARGET_H), interpolation=cv2.INTER_AREA)
            os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
            cv2.imwrite(thumbnail_path, thumbnail_image)
            vid.release()
            return True
        vid.release()
    except Exception: # General exception for OpenCV/file errors
        pass 
    return False

def generate_image_thumbnail(image_path, thumbnail_path):
    """Generates a thumbnail for an image file."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False

        orig_h, orig_w = image.shape[:2]
        orig_aspect_ratio = orig_w / orig_h
        crop_w, crop_h = orig_w, orig_h
        x_offset, y_offset = 0, 0
        if orig_aspect_ratio > settings.TARGET_ASPECT_RATIO:
            crop_w = int(orig_h * settings.TARGET_ASPECT_RATIO)
            x_offset = int((orig_w - crop_w) / 2)
        elif orig_aspect_ratio < settings.TARGET_ASPECT_RATIO:
            crop_h = int(orig_w / settings.TARGET_ASPECT_RATIO)
            y_offset = int((orig_h - crop_h) / 2)

        if crop_w <= 0 or crop_h <= 0 or x_offset < 0 or y_offset < 0 or \
           (y_offset + crop_h) > orig_h or (x_offset + crop_w) > orig_w:
            cropped_image = image
        else:
            cropped_image = image[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]

        if cropped_image.size == 0:
             return False

        thumbnail_image = cv2.resize(cropped_image, (settings.THUMB_TARGET_W, settings.THUMB_TARGET_H), interpolation=cv2.INTER_AREA)
        os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
        cv2.imwrite(thumbnail_path, thumbnail_image)
        return True
    except Exception: # General exception for OpenCV/file errors
        pass
    return False

def get_dataset_folders():
    """Gets folder names from unified DATASETS_DIR.

    Returns:
        dict: A dictionary where keys are folder names and values are display names.
    """
    dataset_folders = {}
    if os.path.exists(settings.DATASETS_DIR):
        for name in os.listdir(settings.DATASETS_DIR):
            if name == "_bak":
                continue
            folder_path = os.path.join(settings.DATASETS_DIR, name)
            if os.path.isdir(folder_path):
                dataset_folders[name] = name
    return dataset_folders

def get_media_files(dataset_path, dataset_type):
    """
    Gets media files for a dataset without generating thumbnails.
    """
    if dataset_type == "image":
        media_extensions = settings.IMAGE_EXTENSIONS
    else:  # For datasets in the unified folder, support mixed media
        # Include both video and image extensions
        media_extensions = list(dict.fromkeys(settings.VIDEO_EXTENSIONS + settings.IMAGE_EXTENSIONS))

    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}") # Debugging print
        return [] # Return empty list if dataset path doesn't exist

    # List all media files with specified extensions
    media_files = []
    for ext in media_extensions:
        media_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
        media_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))

    # Ensure file paths are normalized and unique, then sort
    media_files = sorted(list(set(os.path.normpath(f) for f in media_files)))
    return media_files


def get_videos_and_thumbnails(dataset_name, dataset_type):
    """
    Gets media files and generates/retrieves thumbnails for a dataset.
    Unified handling: all datasets live under DATASETS_DIR; thumbnails under THUMBNAILS_BASE_DIR.
    """
    clean_dataset_name = dataset_name  # No special markers like '(img)'

    dataset_path = os.path.join(settings.DATASETS_DIR, clean_dataset_name)
    thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, clean_dataset_name)
    # Always consider both image and video extensions; duplicates removed while listing files
    media_extensions = list(dict.fromkeys(settings.IMAGE_EXTENSIONS + settings.VIDEO_EXTENSIONS))
    info_path = os.path.join(dataset_path, "info.json")

    os.makedirs(thumbnails_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}") # Debugging print
        return {}, {} # Return empty dictionaries if dataset path doesn't exist

    # List all media files with specified extensions
    media_files = []
    for ext in media_extensions:
        media_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
        media_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))

    # Ensure file paths are normalized and unique, then sort
    media_files = sorted(list(set(os.path.normpath(f) for f in media_files)))

    thumbnail_paths = {}
    media_info = {} # Use media_info to be general for video/image

    # Load existing info for video datasets
    if info_path and os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                loaded_info = json.load(f)
                if isinstance(loaded_info, dict):
                    media_info = loaded_info
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading info.json for {dataset_name}: {e}") # Debugging print
            media_info = {}

    info_changed = False # To track if media_info was updated and needs saving

    for media_path in media_files:
        media_name = os.path.basename(media_path)
        thumbnail_name = f"{os.path.splitext(media_name)[0]}.jpg"
        thumbnail_path = os.path.join(thumbnails_dir, thumbnail_name)

        # Determine per-file type by extension
        ext = os.path.splitext(media_name)[1].lower()
        is_image_file = ext in [e.lower() for e in settings.IMAGE_EXTENSIONS]

        # Get media dimensions/info if not already in media_info or if it's an image file
        if media_name not in media_info or is_image_file:
            try:
                if is_image_file:
                    img = cv2.imread(media_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        media_info[media_name] = {"width": width, "height": height, "frames": 1}
                        info_changed = True
                        os.makedirs(dataset_path, exist_ok=True)
                    else:
                        print(f"Could not read image file: {media_path}")
                else:
                    vid = cv2.VideoCapture(media_path)
                    if vid.isOpened():
                        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                        if width > 0 and height > 0 and frames >= 0:
                            media_info[media_name] = {"width": width, "height": height, "frames": frames}
                            info_changed = True
                    vid.release()
            except Exception as e:
                print(f"Error getting media info for {media_name}: {e}") # Debugging print
                pass # Continue even if info retrieval fails

        # Generate thumbnail if it doesn't exist
        if not os.path.exists(thumbnail_path):
            print(f"Generating thumbnail for: {media_path}") # Debugging print
            if is_image_file:
                generate_image_thumbnail(media_path, thumbnail_path)
            else:
                generate_thumbnail(media_path, thumbnail_path)

        # Add thumbnail path to the map if it exists
        if os.path.exists(thumbnail_path):
            thumbnail_paths[media_path] = thumbnail_path
        else:
             print(f"Thumbnail not found after generation attempt: {thumbnail_path}") # Debugging print


    # Save updated info.json for video datasets if changes occurred
    if info_changed and info_path:
        try:
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(media_info, f, indent=4)
        except Exception as e:
             print(f"Error saving info.json for {dataset_name}: {e}") # Debugging print
             pass # Continue even if saving fails

    return thumbnail_paths, media_info

async def apply_affix_from_textfield(e: ft.ControlEvent, affix_type: str, selected_dataset_ref, dataset_type: str, update_thumbnails_func, thumbnails_grid_ref_obj, affix_text_field_ref: ft.Ref[ft.TextField]):
    if not selected_dataset_ref["value"]:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Please select a dataset first."), open=True)
        if e.page: e.page.update()
        return

    if not affix_text_field_ref.current or not affix_text_field_ref.current.value:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Affix text cannot be empty. Please enter text in the 'Text' field."), open=True)
        if e.page: e.page.update()
        return
    
    affix_text = affix_text_field_ref.current.value.strip()
    current_dataset_name = selected_dataset_ref["value"]
    
    # Unified datasets and single base dir
    clean_dataset_name = current_dataset_name
    dataset_dir = settings.DATASETS_DIR
        
    captions_json_path = os.path.join(dataset_dir, clean_dataset_name, "captions.json")

    if not os.path.exists(captions_json_path):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No captions.json found for dataset '{current_dataset_name}'."), open=True)
        if e.page: e.page.update()
        return

    try:
        with open(captions_json_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except json.JSONDecodeError:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json for '{current_dataset_name}'. Invalid JSON."), open=True)
        if e.page: e.page.update()
        return
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json: {ex}"), open=True)
        if e.page: e.page.update()
        return

    if not isinstance(captions_data, list):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions data for '{current_dataset_name}' is not a list."), open=True)
        if e.page: e.page.update()
        return

    modified_count = 0
    for item in captions_data:
        if isinstance(item, dict) and "caption" in item and isinstance(item["caption"], str):
            if affix_type == "prefix":
                item["caption"] = f"{affix_text} {item['caption']}"
            elif affix_type == "suffix":
                item["caption"] = f"{item['caption']} {affix_text}"
            modified_count += 1
    
    if modified_count == 0 and captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No captions were modified. Check format."), open=True)
        if e.page: e.page.update()
        return
    elif modified_count == 0 and not captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions.json is empty. No captions to modify."), open=True)
        if e.page: e.page.update()
        return

    try:
        with open(captions_json_path, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, indent=4)
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Successfully {affix_type}ed to {modified_count} captions."), open=True)
        if affix_text_field_ref.current:
            affix_text_field_ref.current.value = ""
        
        if asyncio.iscoroutinefunction(update_thumbnails_func):
            await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        else:
            update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
            
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving captions: {ex}"), open=True)
    finally:
        if e.page: e.page.update()

async def find_and_replace_in_captions(e: ft.ControlEvent, selected_dataset_ref, dataset_type: str, find_text_field_ref: ft.Ref[ft.TextField], replace_text_field_ref: ft.Ref[ft.TextField], update_thumbnails_func, thumbnails_grid_ref_obj):
    if not selected_dataset_ref["value"]:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Please select a dataset first."), open=True)
        if e.page: e.page.update()
        return

    if not find_text_field_ref.current or not find_text_field_ref.current.value:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Find text cannot be empty."), open=True)
        if e.page: e.page.update()
        return
    
    find_text = find_text_field_ref.current.value
    replace_text = replace_text_field_ref.current.value if replace_text_field_ref.current and replace_text_field_ref.current.value is not None else ""
    current_dataset_name = selected_dataset_ref["value"]
    
    # Unified datasets and single base dir
    clean_dataset_name = current_dataset_name
    dataset_dir = settings.DATASETS_DIR
    captions_json_path = os.path.join(dataset_dir, clean_dataset_name, "captions.json")

    if not os.path.exists(captions_json_path):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No captions.json found for '{current_dataset_name}'."), open=True)
        if e.page: e.page.update()
        return

    try:
        with open(captions_json_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except json.JSONDecodeError:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json for '{current_dataset_name}'. Invalid JSON."), open=True)
        if e.page: e.page.update()
        return
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error reading captions.json: {ex}"), open=True)
        if e.page: e.page.update()
        return

    if not isinstance(captions_data, list):
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions data for '{current_dataset_name}' is not a list."), open=True)
        if e.page: e.page.update()
        return

    modified_count = 0
    replacements_made = 0
    for item in captions_data:
        if isinstance(item, dict) and "caption" in item and isinstance(item["caption"], str):
            original_caption = item["caption"]
            item["caption"] = original_caption.replace(find_text, replace_text)
            if original_caption != item["caption"]:
                modified_count += 1
                replacements_made += original_caption.count(find_text) if find_text else 0
    
    if modified_count == 0 and captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Text '{find_text}' not found. No changes made."), open=True)
        if e.page: e.page.update()
        return
    elif modified_count == 0 and not captions_data:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captions.json is empty. No changes made."), open=True)
        if e.page: e.page.update()
        return

    try:
        with open(captions_json_path, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, indent=4)
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Made {replacements_made} replacement(s) in {modified_count} caption(s)."), open=True)
        if find_text_field_ref.current: find_text_field_ref.current.value = ""
        if replace_text_field_ref.current: replace_text_field_ref.current.value = ""
        
        if asyncio.iscoroutinefunction(update_thumbnails_func):
            await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        else:
            update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
            
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving captions: {ex}"), open=True)
    finally:
        if e.page: e.page.update()

async def on_change_fps_click(e: ft.ControlEvent, selected_dataset_ref, dataset_type: str, change_fps_textfield_ref_obj, thumbnails_grid_ref_obj, update_thumbnails_func, settings_obj):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    if not change_fps_textfield_ref_obj or not change_fps_textfield_ref_obj.current or not change_fps_textfield_ref_obj.current.value:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: FPS textfield not available or empty."), open=True)
            e.page.update()
        return
        
    target_fps_str = change_fps_textfield_ref_obj.current.value.strip()
    try:
        target_fps_float = float(target_fps_str)
        if target_fps_float <= 0:
            raise ValueError("FPS must be positive")
    except ValueError:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Invalid FPS. Please enter a positive number."), open=True)
            e.page.update()
        return

    clean_dataset_name = current_dataset_name
    dataset_folder = os.path.join(settings_obj.DATASETS_DIR, clean_dataset_name)

    if not os.path.isdir(dataset_folder):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder '{clean_dataset_name}' not found."), open=True)
            e.page.update()
        return

    output_folder = os.path.join(dataset_folder, f"{current_dataset_name}_{target_fps_str}fps")
    os.makedirs(output_folder, exist_ok=True)

    ffmpeg_path = settings_obj.FFMPEG_PATH
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: ffmpeg path not configured or invalid in settings."), open=True)
            e.page.update()
        return
        
    processed_files = 0
    failed_files = 0
    
    video_files_to_process = []
    for ext in settings_obj.VIDEO_EXTENSIONS:
        video_files_to_process.extend(glob.glob(os.path.join(dataset_folder, f"*{ext}")))
        video_files_to_process.extend(glob.glob(os.path.join(dataset_folder, f"*{ext.upper()}")))
    
    video_files_to_process = [f for f in list(set(video_files_to_process)) if os.path.isfile(f)]


    if not video_files_to_process:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No video files found in '{current_dataset_name}'."), open=True)
            e.page.update()
        return

    for video_file in video_files_to_process:
        base_name = os.path.basename(video_file)
        output_file = os.path.join(output_folder, base_name)
        
        command = [
            ffmpeg_path, "-y",
            "-i", video_file,
            "-vf", f"fps={target_fps_float}",
            "-c:v", "libx264", # Or user-defined codec
            "-preset", "medium", # Or user-defined
            "-crf", "18", # Or user-defined
            "-c:a", "aac", # Or user-defined
            "-b:a", "128k", # Or user-defined
            output_file
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                processed_files += 1
            else:
                failed_files += 1
                print(f"ffmpeg error for {video_file}: {stderr}") # Log error
        except Exception as ex:
            failed_files += 1
            print(f"Error processing {video_file}: {ex}") # Log error

    result_message = f"FPS change: {processed_files} processed, {failed_files} failed. Output in '{output_folder}'."
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(result_message), open=True)
        # Optionally, refresh thumbnails if the operation modifies the current dataset directly
        # or if the user is expected to switch to the new dataset folder.
        # For now, we assume the user will manually select the new dataset if needed.
        # if asyncio.iscoroutinefunction(update_thumbnails_func):
        #     await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        # else:
        #     update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        e.page.update()

async def on_rename_files_click(e: ft.ControlEvent, selected_dataset_ref, dataset_type: str, rename_textfield_obj, thumbnails_grid_ref_obj, update_thumbnails_func, settings_obj):
    print("\n=== RENAME FUNCTION CALLED ===")
    print("=== Starting rename operation ===")
    print(f"[DEBUG] Event type: {type(e)}")
    print(f"[DEBUG] Selected dataset ref: {selected_dataset_ref}")
    current_dataset_name = selected_dataset_ref.get("value")
    print(f"[DEBUG] Current dataset name: {current_dataset_name}")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    if not rename_textfield_obj or not rename_textfield_obj.current or not rename_textfield_obj.current.value:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Base name for renaming not provided."), open=True)
            e.page.update()
        return

    base_name_template = rename_textfield_obj.current.value.strip()
    if not base_name_template: # Ensure it's not just whitespace
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Base name cannot be empty."), open=True)
            e.page.update()
        return

    # Unified naming; use dataset name as-is
    clean_dataset_name = current_dataset_name
    print(f"[DEBUG] Cleaned dataset name: {clean_dataset_name}")
    
    print(f"[DEBUG] Dataset type: {dataset_type}")
    
    dataset_dir = settings_obj.DATASETS_DIR
    # Process all supported media types
    file_extensions = list(dict.fromkeys(settings_obj.IMAGE_EXTENSIONS + settings_obj.VIDEO_EXTENSIONS))
    
    dataset_folder = os.path.join(dataset_dir, clean_dataset_name)
    if not os.path.isdir(dataset_folder):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder '{clean_dataset_name}' not found in {dataset_dir}."), open=True)
            e.page.update()
        return

    captions_json_path = os.path.join(dataset_folder, "captions.json")
    captions_data = []
    if os.path.exists(captions_json_path):
        try:
            with open(captions_json_path, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            if not isinstance(captions_data, list):
                captions_data = [] # Or handle error
        except (json.JSONDecodeError, Exception):
            captions_data = [] # Or handle error

    files_to_rename = []
    for ext in file_extensions:
        files_to_rename.extend(glob.glob(os.path.join(dataset_folder, f"*{ext}")))
        files_to_rename.extend(glob.glob(os.path.join(dataset_folder, f"*{ext.upper()}")))
    
    files_to_rename = sorted([f for f in list(set(files_to_rename)) if os.path.isfile(f)])

    if not files_to_rename:
        file_type = "image" if dataset_type == "image" else "video"
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No {file_type} files found in '{clean_dataset_name}' to rename."), open=True)
            e.page.update()
        return
        
    renamed_files_count = 0
    failed_files_count = 0
    updated_captions_map = {} # To store new filenames for captions

    for idx, old_file_path in enumerate(files_to_rename):
        original_basename = os.path.basename(old_file_path)
        _, ext = os.path.splitext(original_basename)
        new_basename = f"{base_name_template}_{idx+1:04d}{ext}" # e.g., myimage_0001.jpg or myvideo_0001.mp4
        new_file_path = os.path.join(dataset_folder, new_basename)

        if old_file_path == new_file_path: # Skip if name is already correct
            updated_captions_map[original_basename] = new_basename # Still need for caption update
            continue

        try:
            shutil.move(old_file_path, new_file_path)
            renamed_files_count += 1
            updated_captions_map[original_basename] = new_basename
            
            # Rename corresponding thumbnail if it exists
            old_thumb_name = f"{os.path.splitext(original_basename)[0]}.jpg"
            new_thumb_name = f"{os.path.splitext(new_basename)[0]}.jpg"
            
            old_thumb_path = os.path.join(settings_obj.THUMBNAILS_BASE_DIR, clean_dataset_name, old_thumb_name)
            new_thumb_path = os.path.join(settings_obj.THUMBNAILS_BASE_DIR, clean_dataset_name, new_thumb_name)

            if os.path.exists(old_thumb_path):
                shutil.move(old_thumb_path, new_thumb_path)

        except Exception as ex:
            failed_files_count += 1
            print(f"Error renaming {original_basename} to {new_basename}: {ex}")

    # Update captions.json
    if captions_data and updated_captions_map:
        modified_captions = False
        for item in captions_data:
            if isinstance(item, dict) and "video_filename" in item:
                if item["video_filename"] in updated_captions_map:
                    item["video_filename"] = updated_captions_map[item["video_filename"]]
                    modified_captions = True
        if modified_captions:
            try:
                with open(captions_json_path, 'w', encoding='utf-8') as f:
                    json.dump(captions_data, f, indent=4)
            except Exception as ex:
                print(f"Error updating captions.json: {ex}")
                if e.page:
                    e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Files renamed, but error updating captions.json: {ex}"), open=True)
                    # No page update here, it's part of the main finally block

    result_message = f"Renamed {renamed_files_count} files. Failed: {failed_files_count}."
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(result_message), open=True)
        if asyncio.iscoroutinefunction(update_thumbnails_func):
            await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        else:
            update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        e.page.update()
