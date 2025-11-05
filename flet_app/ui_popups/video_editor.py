import flet as ft 
import os
import shutil
from typing import Optional, List, Callable, Tuple
from flet_video.video import Video
from PIL import Image, ImageDraw
import numpy as np
import subprocess
import sys
from flet_app.ui._styles import VIDEO_PLAYER_DIALOG_WIDTH, VIDEO_PLAYER_DIALOG_HEIGHT

from . import video_player_utils as vpu


def handle_size_add(width_field: ft.TextField, height_field: ft.TextField, current_video_path: str, page: Optional[ft.Page] = None): 
    new_w_str, new_h_str = vpu.calculate_adjusted_size(
        width_field.value, height_field.value, current_video_path, 'add'
    )
    width_field.value = new_w_str
    height_field.value = new_h_str
    if page:
        width_field.update()
        height_field.update()

def handle_size_sub(width_field: ft.TextField, height_field: ft.TextField, current_video_path: str, page: Optional[ft.Page] = None): 
    new_w_str, new_h_str = vpu.calculate_adjusted_size(
        width_field.value, height_field.value, current_video_path, 'sub'
    )
    width_field.value = new_w_str
    height_field.value = new_h_str
    if page:
        width_field.update()
        height_field.update()

# Removed legacy overlay-crop path tied to old video dialog

def handle_set_closest_div32(width_field: ft.TextField, height_field: ft.TextField, current_video_path: str, page: Optional[ft.Page] = None): 
    w_str, h_str = vpu.calculate_closest_div32_dimensions(current_video_path)
    if w_str and h_str:
        width_field.value = w_str
        height_field.value = h_str
        if page:
            width_field.update()
            height_field.update()

def _generic_video_operation_ui_update(
    page: ft.Page, 
    processed_video_path: str, 
    video_list: Optional[List[str]] = None, 
    on_caption_updated_callback: Optional[Callable] = None, 
    operation_message: str = "Video operation successful."
    ):
    if page:
        page.snack_bar = ft.SnackBar(ft.Text(operation_message), open=True)
        vpu.update_video_info_json(processed_video_path)
        if on_caption_updated_callback:
            on_caption_updated_callback(processed_video_path) 
        page.update()

def on_flip_horizontal(page: ft.Page, current_video_path: str, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    success, msg, temp_output_path = vpu.flip_video_horizontal(current_video_path)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving flipped file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def on_rotate_90_video_action(page: ft.Page, current_video_path: str, direction: str, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]):
    success, msg, temp_output_path = vpu.rotate_video_90(current_video_path, direction)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving rotated file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def on_reverse(page: ft.Page, current_video_path: str, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    success, msg, temp_output_path = vpu.reverse_video(current_video_path)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving reversed file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def on_time_remap(page: ft.Page, current_video_path: str, speed_multiplier: float, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]): 
    success, msg, temp_output_path = vpu.time_remap_video_by_speed(current_video_path, speed_multiplier)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving remapped file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def handle_crop_video_click(page: ft.Page, width_field: Optional[ft.TextField], height_field: Optional[ft.TextField], current_video_path: str, video_list: Optional[List[str]] = None, on_caption_updated_callback: Optional[Callable] = None, should_update_ui: bool = True): 
    if not width_field or not width_field.value or not height_field or not height_field.value : 
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text("Width and Height must be specified."), open=True); page.update()
        return
    try:
        target_width = int(width_field.value)
        target_height = int(height_field.value)
    except ValueError:
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text("Invalid Width or Height value."), open=True); page.update()
        return

    success, msg, temp_output_path = vpu.crop_video_to_dimensions(current_video_path, target_width, target_height)
    
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            if should_update_ui:
                 _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)
            else: 
                vpu.update_video_info_json(current_video_path)
                if page: page.snack_bar = ft.SnackBar(ft.Text(f"{os.path.basename(current_video_path)}: {msg}" if msg else f"{os.path.basename(current_video_path)} cropped (batch)."), duration=2000, open=True)
        except Exception as e:
            final_msg = f"Error moving cropped file for {os.path.basename(current_video_path)}: {e}"
            if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text(final_msg), open=True)
            print(final_msg)
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        final_msg = msg if msg else f"Failed to crop {os.path.basename(current_video_path)}."
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text(final_msg), open=True)
        print(final_msg)

    if page and not should_update_ui: 
        page.update()

def handle_crop_all_videos(
    page: ft.Page, 
    current_video_path_in_dialog: Optional[str],
    width_field: Optional[ft.TextField], 
    height_field: Optional[ft.TextField], 
    video_list: Optional[List[str]], 
    on_caption_updated_callback: Optional[Callable]
): 
    if not video_list:
        if page: page.snack_bar = ft.SnackBar(ft.Text("No videos in the list to crop."), open=True); page.update()
        return
    
    if not width_field or not width_field.value or not height_field or not height_field.value:
        if page: page.snack_bar = ft.SnackBar(ft.Text("Width and Height must be specified for batch crop."), open=True); page.update()
        return

    processed_count = 0
    dialog_refreshed_for_current_video = False

    for video_path_item in video_list: 
        processed_count +=1
        # Determine if the UI (dialog) should be fully updated for this specific item
        should_refresh_dialog_for_this_item = (video_path_item == current_video_path_in_dialog)
        
        # Call handle_crop_video_click, which might reopen the dialog if should_refresh_dialog_for_this_item is True
        handle_crop_video_click(
            page, width_field, height_field, 
            video_path_item, video_list, 
            on_caption_updated_callback, 
            should_update_ui=should_refresh_dialog_for_this_item
        )
        
        if should_refresh_dialog_for_this_item:
            dialog_refreshed_for_current_video = True # Mark that the dialog was handled

    if page:
        # Display a summary snackbar
        page.snack_bar = ft.SnackBar(ft.Text(f"Batch crop attempt finished for {processed_count} videos. Review individual messages."), open=True)
        
        # Call general callback and update page only if dialog wasn't specifically refreshed for the current video
        if on_caption_updated_callback and not dialog_refreshed_for_current_video:
            on_caption_updated_callback()
        
        if not dialog_refreshed_for_current_video:
            page.update()

def cut_to_frames(page: ft.Page, current_video_path: str, start_frame: int, end_frame: int, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable], refresh_dialog_callback: Optional[Callable] = None, thumbnail_update_callback: Optional[Callable] = None):
    success, msg, temp_output_path = vpu.cut_video_by_frames(current_video_path, start_frame, end_frame)
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_video_path)
            _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, msg)

            # For cut operations, we need to specifically regenerate the thumbnail
            # because we're replacing the original file, not creating a new one
            def force_regenerate_cut_thumbnail():
                """Force regeneration of thumbnail for the cut video"""
                try:
                    # Import settings to get paths
                    from flet_app.settings import settings

                    # Extract dataset name from video path
                    # Video path format: /path/to/DATASETS_DIR/dataset_name/video.mp4
                    if current_video_path.startswith(settings.DATASETS_DIR):
                        relative_path = os.path.relpath(current_video_path, settings.DATASETS_DIR)
                        path_parts = relative_path.split(os.sep)
                        dataset_name = path_parts[0] if path_parts else None
                    else:
                        dataset_name = None
                    if dataset_name:
                        # Construct old thumbnail path
                        video_name = os.path.basename(current_video_path)
                        thumbnail_name = f"{os.path.splitext(video_name)[0]}.jpg"
                        old_thumbnail_path = os.path.join(settings.THUMBNAILS_BASE_DIR, dataset_name, thumbnail_name)

                        # Remove old thumbnail to force regeneration
                        if os.path.exists(old_thumbnail_path):
                            os.remove(old_thumbnail_path)

                        # Call the normal thumbnail update callback
                        if thumbnail_update_callback:
                            thumbnail_update_callback()

                except Exception:
                    # Fallback to normal thumbnail update
                    if thumbnail_update_callback:
                        thumbnail_update_callback()

            # Force regenerate the specific thumbnail for the cut video
            force_regenerate_cut_thumbnail()

            # Refresh the dialog to show the updated video
            if refresh_dialog_callback:
                refresh_dialog_callback()
        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving cut file: {e}"), open=True); page.update()
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()

def split_to_video(page: ft.Page, current_video_path: str, split_frame: int, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable], video_player_instance_from_dialog_state: Optional[Video], refresh_dialog_callback: Optional[Callable] = None, thumbnail_update_callback: Optional[Callable] = None):
    if split_frame <= 0:
        if page: page.snack_bar = ft.SnackBar(ft.Text("Split frame must be greater than 0."), open=True); page.update()
        return

    success, msg, temp_path1, temp_path2 = vpu.split_video_by_frame(current_video_path, split_frame)

    if success and temp_path1 and temp_path2:
        original_dir = os.path.dirname(current_video_path)
        base_name, ext = os.path.splitext(os.path.basename(current_video_path))

        final_path1 = current_video_path

        counter = 1
        final_path2_base = os.path.join(original_dir, f"{base_name}_splitP2")
        final_path2 = f"{final_path2_base}{ext}"
        while os.path.exists(final_path2):
            final_path2 = f"{final_path2_base}_{counter}{ext}"
            counter += 1
            if counter > 100:
                if page: page.snack_bar = ft.SnackBar(ft.Text("Could not find a unique name for split part 2."), open=True); page.update()
                if os.path.exists(temp_path1): os.remove(temp_path1)
                if os.path.exists(temp_path2): os.remove(temp_path2)
                return
        try:
            shutil.move(temp_path1, final_path1)
            shutil.move(temp_path2, final_path2)

            vpu.update_video_info_json(final_path1)
            vpu.update_video_info_json(final_path2)

            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Video split. Original updated, new: {os.path.basename(final_path2)}"), open=True)

            # Update thumbnails to reflect changes in dataset (new video added)
            if thumbnail_update_callback:
                thumbnail_update_callback()

            # Refresh the dialog to show the updated video (part 1)
            if refresh_dialog_callback:
                refresh_dialog_callback()

            if on_caption_updated_callback:
                on_caption_updated_callback()

            page.update()

        except Exception as e:
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error moving split files: {e}"), open=True); page.update()
            if os.path.exists(temp_path1): os.remove(temp_path1) 
            if os.path.exists(temp_path2): os.remove(temp_path2) 
    else:
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()
        if temp_path1 and os.path.exists(temp_path1): os.remove(temp_path1)
        if temp_path2 and os.path.exists(temp_path2): os.remove(temp_path2)

def cut_all_videos_to_max(
    page: ft.Page, 
    current_video_path_in_dialog: Optional[str], 
    video_list: Optional[List[str]], 
    num_to_cut_to: int, 
    on_caption_updated_callback: Optional[Callable]
): 
    if not video_list:
        if page: page.snack_bar = ft.SnackBar(ft.Text("No videos in list to cut."), open=True); page.update()
        return
    if num_to_cut_to <= 0:
        if page: page.snack_bar = ft.SnackBar(ft.Text("Number of frames to cut to must be positive."), open=True); page.update()
        return

    processed_count = 0
    failed_count = 0
    successfully_processed_paths = [] # New: track successful paths

    for video_path_item in video_list: 
        metadata = vpu.get_video_metadata(video_path_item)
        if not metadata or not metadata.get('total_frames') or metadata['total_frames'] <= num_to_cut_to :
            print(f"Skipping {os.path.basename(video_path_item)}: already short enough or no metadata.")
            continue 

        success, msg, temp_output_path = vpu.cut_video_by_frames(video_path_item, 0, num_to_cut_to) 
        if success and temp_output_path:
            try:
                # Ensure the original file exists before attempting to move over it
                if not os.path.exists(video_path_item):
                    print(f"Error: Original file {video_path_item} not found for move.")
                    failed_count += 1
                    if os.path.exists(temp_output_path): os.remove(temp_output_path)
                    continue

                shutil.move(temp_output_path, video_path_item)
                vpu.update_video_info_json(video_path_item) # Update JSON after successful move
                processed_count += 1
                successfully_processed_paths.append(video_path_item) # New: add to list
                if page: page.snack_bar = ft.SnackBar(ft.Text(f"Cut {os.path.basename(video_path_item)} to {num_to_cut_to} frames."), duration=2000, open=True); page.update()
            except Exception as e:
                failed_count += 1
                print(f"Error moving cut file for {os.path.basename(video_path_item)}: {e}")
                if os.path.exists(temp_output_path): os.remove(temp_output_path) # Clean up temp file on error
        else:
            failed_count +=1
            print(f"Failed to cut {os.path.basename(video_path_item)}: {msg}")
            if page: page.snack_bar = ft.SnackBar(ft.Text(f"Failed to cut {os.path.basename(video_path_item)}: {msg}"), open=True); page.update()
    
    if page:
        page.snack_bar = ft.SnackBar(ft.Text(f"Batch cut complete: {processed_count} succeeded, {failed_count} failed."), open=True)
        
        refreshed_current_video = False
        if current_video_path_in_dialog and current_video_path_in_dialog in successfully_processed_paths:
            # Unified popup handles current view; no legacy dialog refresh
            refreshed_current_video = True

        if on_caption_updated_callback and not refreshed_current_video:
            on_caption_updated_callback()
        
        if not refreshed_current_video: 
            page.update()

def on_clean_action_handler(page: ft.Page, current_video_path: str, overlay_coords: Tuple[int, int, int, int], video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable]):
    """
    Handles the 'clean' action by creating a mask and calling an external script.
    """
    if not current_video_path or not os.path.exists(current_video_path):
        if page: page.snack_bar = ft.SnackBar(ft.Text("Error: Invalid video path for cleaning."), open=True); page.update()
        return

    # Get video resolution
    metadata = vpu.get_video_metadata(current_video_path)
    video_width = metadata.get('width')
    video_height = metadata.get('height')

    if not video_width or not video_height:
        if page: page.snack_bar = ft.SnackBar(ft.Text(f"Error: Could not get video resolution for {os.path.basename(current_video_path)}"), open=True); page.update()
        return

    # Create a black image with video resolution
    mask_image = Image.new('L', (video_width, video_height), 0) # 'L' for black/white (grayscale)
    draw = ImageDraw.Draw(mask_image)

    # selected_area_zone is (left, top, width, height) in display coordinates
    display_area_width = VIDEO_PLAYER_DIALOG_WIDTH - 40
    display_area_height = VIDEO_PLAYER_DIALOG_HEIGHT - 40

    # Calculate video aspect ratio and display aspect ratio
    video_aspect_ratio = video_width / video_height
    display_aspect_ratio = display_area_width / display_area_height

    # Determine the actual rendered video dimensions within the player (accounting for letterbox/pillarbox)
    rendered_video_width_in_player = display_area_width
    rendered_video_height_in_player = display_area_height
    offset_x = 0
    offset_y = 0

    if video_aspect_ratio > display_aspect_ratio: # Letterboxing (video is wider than player)
        rendered_video_height_in_player = int(display_area_width / video_aspect_ratio)
        offset_y = (display_area_height - rendered_video_height_in_player) / 2
    else: # Pillarboxing (video is taller than player)
        rendered_video_width_in_player = int(display_area_height * video_aspect_ratio)
        offset_x = (display_area_width - rendered_video_width_in_player) / 2

    # Calculate scaling factors from rendered video in player to original video resolution
    scale_x = video_width / rendered_video_width_in_player
    scale_y = video_height / rendered_video_height_in_player

    # Scale the selected area coordinates, adjusting for offsets
    x1_display, y1_display, w_display, h_display = overlay_coords
    
    x1_scaled = int((x1_display - offset_x) * scale_x)
    y1_scaled = int((y1_display - offset_y) * scale_y)
    w_scaled = int(w_display * scale_x)
    h_scaled = int(h_display * scale_y)

    # Ensure coordinates are within bounds of the original video resolution
    x1_scaled = max(0, x1_scaled)
    y1_scaled = max(0, y1_scaled)
    w_scaled = min(w_scaled, video_width - x1_scaled)
    h_scaled = min(h_scaled, video_height - y1_scaled)

    x2_scaled = x1_scaled + w_scaled
    y2_scaled = y1_scaled + h_scaled

    draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], fill=255) # fill=255 for white

    # Construct mask image filename and path
    video_basename = os.path.basename(current_video_path)
    video_name_without_ext = os.path.splitext(video_basename)[0]
    mask_filename = f"{video_name_without_ext}_mask.png"
    
    # Get the directory of the current video
    video_dir = os.path.dirname(current_video_path)
    
    # Create temp_processing folder next to the video
    temp_processing_dir = os.path.join(video_dir, "temp_processing")
    os.makedirs(temp_processing_dir, exist_ok=True) # Ensure directory exists

    mask_image_path = os.path.join(temp_processing_dir, mask_filename)
    mask_image.save(mask_image_path)
    print(f"Mask image saved to: {mask_image_path}")

    script_path = os.path.join("flet_app", "modules", "minimax-remover", "run_remover_with_image_mask.py")

    # Find a Python interpreter to use: prefer venv python if present, else fallback to current
    venv_py_win = os.path.join("venv", "Scripts", "python.exe")
    venv_py_unix = os.path.join("venv", "bin", "python")
    python_exe = None
    if os.path.exists(venv_py_win):
        python_exe = venv_py_win
    elif os.path.exists(venv_py_unix):
        python_exe = venv_py_unix
    else:
        python_exe = sys.executable or "python"

    cmd = [
        python_exe,
        script_path,
        "--video_path", current_video_path,
        "--image_mask_path", mask_image_path,
    ]

    print(f"Executing command: {' '.join([str(c) for c in cmd])}")
    try:
        # Run the command without shell for cross-platform safety
        result = subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("Command execution finished.")
        # After successful execution, update UI and video info
        _generic_video_operation_ui_update(page, current_video_path, video_list, on_caption_updated_callback, "Video cleaned using external script!")
    except subprocess.CalledProcessError as e:
        msg = f"Command failed with error code {e.returncode}. See console for details."
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()
        print(msg)
    except FileNotFoundError:
        msg = "Error: 'python' command or script not found. Ensure Python is installed and in your PATH, and the script path is correct."
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()
        print(msg)
    except Exception as e:
        msg = f"An unexpected error occurred during cleaning: {e}"
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()
        print(msg)
    finally:
        # Clean up the mask image after use
        if os.path.exists(mask_image_path):
            try:
                os.remove(mask_image_path)
                print(f"Cleaned up mask file: {mask_image_path}")
            except Exception as e_del:
                print(f"Error deleting mask file {mask_image_path}: {e_del}")
