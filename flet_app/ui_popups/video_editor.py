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

def cut_to_frames(page: ft.Page, current_video_path: str, start_frame: int, end_frame: int, video_list: Optional[List[str]], on_caption_updated_callback: Optional[Callable], refresh_dialog_callback: Optional[Callable] = None, thumbnail_update_callback: Optional[Callable] = None, force_reencode: bool = False):
    success, msg, temp_output_path = vpu.cut_video_by_frames(current_video_path, start_frame, end_frame, force_reencode)
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

def split_selected_videos_into_chunks(page: ft.Page, seconds_per_chunk: int, thumbnail_update_callback: Optional[Callable] = None, force_reencode: bool = False):
    """
    Splits selected videos into chunks of specified seconds.
    Uses stream copy for fast processing and updates thumbnails afterward.
    """
    print(f"Starting chunking process with {seconds_per_chunk} seconds per chunk")

    # Import needed functions
    try:
        from flet_app.ui.dataset_manager.dataset_layout_tab import video_files_list, thumbnails_grid_ref
        from flet_app.ui.dataset_manager.dataset_actions import _get_selected_filenames
        print(f"Imported video_files_list: {video_files_list}")
    except ImportError as e:
        print(f"Import error: {e}")
        if page: page.snack_bar = ft.SnackBar(ft.Text("Error: Could not access selected videos."), open=True); page.update()
        return

    if seconds_per_chunk <= 0:
        print(f"Invalid seconds value: {seconds_per_chunk}")
        if page: page.snack_bar = ft.SnackBar(ft.Text("Please enter a valid number of seconds (> 0)."), open=True); page.update()
        return

    # Get selected video filenames from thumbnail grid
    selected_filenames = []
    if thumbnails_grid_ref and thumbnails_grid_ref.current:
        selected_filenames = _get_selected_filenames(thumbnails_grid_ref.current)
        print(f"Selected filenames: {selected_filenames}")

    if not selected_filenames:
        print("No videos selected")
        if page:
            page.snack_bar = ft.SnackBar(
                ft.Text("❌ No videos selected. Please select videos in the main dataset view before using 'Slice to'.\n\n💡 If you want to split the current video you're viewing, use the 'Split' button inside the video player dialog instead."),
                open=True,
                duration=10000  # Show for 10 seconds
            );
            page.update()
        return

    # Get selected video paths
    selected_videos = []
    videos_list = video_files_list.get("value", [])
    print(f"Available videos in list: {len(videos_list)}")

    for video_path in videos_list:
        video_name = os.path.basename(video_path)
        if video_name in selected_filenames:
            selected_videos.append(video_path)
            print(f"Found selected video: {video_name}")

    print(f"Total selected videos to process: {len(selected_videos)}")

    if not selected_videos:
        print("No valid selected videos found")
        if page: page.snack_bar = ft.SnackBar(ft.Text("No valid selected videos found."), open=True); page.update()
        return

    # Process each selected video
    total_processed = 0
    for video_path in selected_videos:
        try:
            print(f"Processing video: {video_path}")
            success, result_msg = _split_single_video_into_chunks(video_path, seconds_per_chunk, force_reencode)
            print(f"Result for {video_path}: success={success}, msg={result_msg}")

            if success:
                total_processed += 1
                if page: page.snack_bar = ft.Text(f"Chunked: {os.path.basename(video_path)} - {result_msg}"); page.update()
            else:
                if page: page.snack_bar = ft.Text(f"Failed to chunk {os.path.basename(video_path)}: {result_msg}"); page.update()
        except Exception as e:
            print(f"Error processing {os.path.basename(video_path)}: {e}")
            if page: page.snack_bar = ft.Text(f"Error processing {os.path.basename(video_path)}: {e}"); page.update()

    # Final status and thumbnail update
    if total_processed > 0:
        print(f"Successfully processed {total_processed} videos")
        if page: page.snack_bar = ft.SnackBar(ft.Text(f"Successfully chunked {total_processed} video(s)!"), open=True); page.update()

        # Update thumbnails
        if thumbnail_update_callback:
            print("Calling thumbnail update callback")
            thumbnail_update_callback()
        else:
            print("No thumbnail update callback provided")
    else:
        print("No videos were successfully chunked")
        if page: page.snack_bar = ft.SnackBar(ft.Text("No videos were successfully chunked."), open=True); page.update()

def _split_single_video_into_chunks(video_path: str, seconds_per_chunk: int, force_reencode: bool = False) -> Tuple[bool, str]:
    """
    Splits a single video into chunks of specified seconds using efficient method without re-encoding.
    Returns (success, message).
    """
    try:
        from . import video_player_utils as vpu

        # Get video metadata
        metadata = vpu.get_video_metadata(video_path)
        if not metadata or not metadata.get('fps') or not metadata.get('total_frames'):
            return False, "Could not get video metadata"

        # Calculate duration from fps and frame count
        total_duration = metadata['total_frames'] / metadata['fps']

        if total_duration <= seconds_per_chunk:
            return False, "Video is shorter than specified chunk size"

        # Calculate chunk count - always include a chunk for any remaining time
        full_chunks = int(total_duration // seconds_per_chunk)
        remainder_duration = total_duration % seconds_per_chunk
        
        if remainder_duration > 0:
            chunk_count = full_chunks + 1
        else:
            chunk_count = full_chunks
            # If the video divides evenly, we still need at least one chunk
            if chunk_count == 0:
                chunk_count = 1

        # Prepare output directory
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        base_name, ext = os.path.splitext(video_name)
        temp_dir = os.path.join(video_dir, "temp_processing")
        os.makedirs(temp_dir, exist_ok=True)

        # Split video into chunks using the efficient method without re-encoding
        chunk_files = []
        
        from .video_player_utils import _get_ffmpeg_exe_path
        ffmpeg_exe = _get_ffmpeg_exe_path()
        
        for i in range(chunk_count):
            start_time = i * seconds_per_chunk
            
            # Calculate end time for this chunk
            if i == chunk_count - 1:  # Last chunk - go to end of video
                # Calculate the actual duration for the last chunk (may include small remainder)
                actual_duration = total_duration - start_time
                command = [
                    ffmpeg_exe, "-y",
                    "-ss", str(start_time),  # Seek before input for speed (may snap to keyframe)
                    "-i", video_path,
                    "-t", str(actual_duration),
                    "-c", "copy",  # Copy both audio and video streams
                    "-avoid_negative_ts", "make_zero",
                    "-fflags", "+genpts"
                ]
            else:  # Not the last chunk - cut for seconds_per_chunk duration
                command = [
                    ffmpeg_exe, "-y",
                    "-ss", str(start_time),  # Seek before input for speed (may snap to keyframe)
                    "-i", video_path,
                    "-t", str(seconds_per_chunk),
                    "-c", "copy",  # Copy both audio and video streams
                    "-avoid_negative_ts", "make_zero",
                    "-fflags", "+genpts"
                ]
            
            # Create chunk filename with zero-padding for chronological order
            chunk_filename = f"{base_name}_chunk_{str(i+1).zfill(3)}{ext}"
            chunk_path = os.path.join(temp_dir, chunk_filename)
            command.append(chunk_path)
            
            # Execute FFmpeg command
            if i == chunk_count - 1:
                actual_duration = total_duration - start_time
                print(f"Chunk {i+1}: Cutting from {start_time:.2f}s for {actual_duration:.2f}s (to end)")
            else:
                print(f"Chunk {i+1}: Cutting from {start_time:.2f}s for {seconds_per_chunk:.2f}s")
                
            success, stdout, stderr = vpu._run_ffmpeg_process(command)
            
            if not success or not os.path.exists(chunk_path):
                error_msg = f"Failed to create chunk {i+1}: FFmpeg error"
                if stderr:
                    error_msg += f" - {stderr.strip()}"
                return False, error_msg

            chunk_files.append(chunk_path)

        # Move chunks to final directory and validate each
        final_chunk_paths = []
        for chunk_path in chunk_files:
            # Validate the chunk
            if not _validate_video_chunk(chunk_path):
                print(f"Warning: Chunk {chunk_path} failed validation, skipping...")
                continue

            final_path = os.path.join(video_dir, os.path.basename(chunk_path))
            if os.path.exists(final_path):
                # If file exists, add counter
                base, ext = os.path.splitext(final_path)
                counter = 1
                while os.path.exists(f"{base}_{counter}{ext}"):
                    counter += 1
                final_path = f"{base}_{counter}{ext}"

            shutil.move(chunk_path, final_path)
            final_chunk_paths.append(final_path)

            # Update video info for new chunk
            vpu.update_video_info_json(final_path)

        # Clean up temp directory
        try:
            # Remove all remaining files in temp directory
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                os.remove(file_path)
            os.rmdir(temp_dir)
        except:
            pass

        if not final_chunk_paths:
            return False, "No valid chunks were created"

        return True, f"Created {len(final_chunk_paths)} chunks"

    except Exception as e:
        return False, f"Error: {str(e)}"


def _validate_video_chunk(chunk_path: str) -> bool:
    """
    Validates that a video chunk is playable and has proper structure.
    Returns True if the chunk is valid, False otherwise.
    """
    if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
        return False

    try:
        import cv2
        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened():
            print(f"Video validation failed: could not open {chunk_path}")
            return False

        # Check if we can read at least one frame
        ret, frame = cap.read()
        if not ret:
            print(f"Video validation failed: could not read first frame from {chunk_path}")
            cap.release()
            return False

        # Check if we can get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        # Validate that the properties are reasonable
        if fps <= 0 or width <= 0 or height <= 0:
            print(f"Video validation failed: invalid properties for {chunk_path} (fps={fps}, w={width}, h={height})")
            return False

        # Check if frame count is reasonable for the duration (min 1 frame for very short clips)
        if frame_count <= 0:
            print(f"Video validation failed: no frames in {chunk_path}")
            return False

        # Additional check: try to read a few more frames to ensure it's not corrupted
        cap = cv2.VideoCapture(chunk_path)
        frame_count_check = 0
        for _ in range(min(10, frame_count)):  # Check up to 10 frames
            ret, _ = cap.read()
            if not ret:
                break
            frame_count_check += 1
        cap.release()

        # If we couldn't read any additional frames after the first one, it's likely corrupted
        if frame_count_check < 2 and frame_count > 10:  # If video should have more frames but we can't read them
            print(f"Video validation failed: could only read {frame_count_check} frames from {chunk_path} (expected more)")
            return False

        # Additional validation: Check if the video has proper duration based on frame count and fps
        expected_duration = frame_count / fps
        print(f"Video validation: {chunk_path} - FPS: {fps}, Frames: {frame_count}, Expected duration: {expected_duration:.2f}s")

        return True

    except Exception as e:
        print(f"Video validation error for {chunk_path}: {e}")
        return False


def concatenate_video_chunks(video_dir: str, base_name: str, ext: str, output_path: str) -> Tuple[bool, str]:
    """
    Concatenates video chunks using the efficient method without re-encoding.
    Creates a text file with file entries and uses FFmpeg concat demuxer.
    Returns (success, message).
    """
    try:
        from . import video_player_utils as vpu
        from .video_player_utils import _get_ffmpeg_exe_path
        ffmpeg_exe = _get_ffmpeg_exe_path()
        
        # Find all chunk files that match the pattern
        chunk_files = []
        for filename in sorted(os.listdir(video_dir)):
            if filename.startswith(f"{base_name}_chunk_") and filename.endswith(ext):
                chunk_path = os.path.join(video_dir, filename)
                if os.path.exists(chunk_path):
                    chunk_files.append(chunk_path)
        
        if not chunk_files:
            return False, "No chunk files found to concatenate"
        
        # Create a temporary list file for FFmpeg
        list_file_path = os.path.join(video_dir, f"{base_name}_concat_list.txt")
        with open(list_file_path, 'w', encoding='utf-8') as list_file:
            for chunk_path in chunk_files:
                # Use forward slashes or escape backslashes for FFmpeg
                escaped_path = chunk_path.replace("\\", "/")
                list_file.write(f"file '{escaped_path}'\n")
        
        # FFmpeg concat command using the concat demuxer (no re-encoding)
        concat_command = [
            ffmpeg_exe,
            "-f", "concat",
            "-safe", "0",
            "-i", list_file_path,
            "-c", "copy",  # Copy streams without re-encoding
            "-avoid_negative_ts", "make_zero",
            output_path
        ]
        
        print(f"Concatenating {len(chunk_files)} chunks to {output_path}")
        success, stdout, stderr = vpu._run_ffmpeg_process(concat_command)
        
        # Clean up the temporary list file
        if os.path.exists(list_file_path):
            os.remove(list_file_path)
        
        if success and os.path.exists(output_path):
            # Update video info for the concatenated file
            vpu.update_video_info_json(output_path)
            return True, f"Successfully concatenated {len(chunk_files)} chunks into {os.path.basename(output_path)}"
        else:
            error_msg = "Failed to concatenate chunks"
            if stderr:
                error_msg += f": {stderr.strip()}"
            return False, error_msg

    except Exception as e:
        return False, f"Error during concatenation: {str(e)}"
