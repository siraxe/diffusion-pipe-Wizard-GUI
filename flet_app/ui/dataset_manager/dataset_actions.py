# This file will contain functions related to dataset actions,
# such as captioning, preprocessing, renaming, and FPS changes.
# It will also house the event handlers for these actions.

import flet as ft
import os
import json
import asyncio
import sys
import signal
import subprocess
import shutil
import time
import re

from flet_app.settings import settings
# Import centralized video encoding settings
from flet_app.ui_popups import video_player_utils as vpu
from flet_app.ui_popups.delete_caption_dialog import show_delete_caption_dialog
from flet_app.ui.dataset_manager.dataset_utils import (
    load_dataset_config, save_dataset_config,
    load_dataset_captions, validate_bucket_values,
    _get_dataset_base_dir, get_videos_and_thumbnails, get_dataset_folders, get_media_files,
    parse_bucket_string_to_list # Add this import
)
from flet_app.ui.dataset_manager.dataset_thumb_layout import create_thumbnail_container, set_thumbnail_selection_state, update_thumbnail_caption_status
from flet_app.ui.flet_hotkeys import is_d_key_pressed_global # Import global D key state

# Global references from dataset_layout_tab.py that need to be accessed here
# These will be passed as arguments to functions or accessed via global state if necessary
# For now, I'll assume they are passed as arguments where needed.
# If a global reference is truly needed, it should be imported or managed carefully.

# Process tracking for stopping script execution
current_caption_process = {"proc": None}

def _get_selected_filenames(thumbnails_grid_control: ft.GridView) -> list[str]:
    """
    Helper function to get a list of base filenames for selected thumbnails.
    """
    selected_filenames = []
    # thumbnails_grid_control is already the GridView instance, so access controls directly
    if thumbnails_grid_control and thumbnails_grid_control.controls:
        for thumbnail_container in thumbnails_grid_control.controls:
            if isinstance(thumbnail_container, ft.Container) and \
               isinstance(thumbnail_container.content, ft.Stack):
                
                checkbox = None
                for control in thumbnail_container.content.controls:
                    if isinstance(control, ft.Checkbox):
                        checkbox = control
                        break
                
                if checkbox and checkbox.value:
                    # thumbnail_container.data holds the original video_path/image_path
                    selected_filenames.append(os.path.basename(thumbnail_container.data))
    return selected_filenames

# ======================================================================================
# Script Command Building Functions (Generate CLI commands)
# ======================================================================================

def build_caption_command(
    dataset_folder_path: str,
    output_json_path: str,
    selected_model: str,
    use_8bit: bool,
    instruction: str,
    max_new_tokens: int,
    selected_files: list[str] = None, # New parameter
    custom_model_path: str = None, # Custom model path for "custom" type
) -> str:
    # Prefer current interpreter; fallback to common venv paths or python
    python_exe = (
        sys.executable
        or (os.path.join("venv", "bin", "python") if os.path.exists(os.path.join("venv", "bin", "python")) else None)
        or (os.path.join("venv", "Scripts", "python.exe") if os.path.exists(os.path.join("venv", "Scripts", "python.exe")) else None)
        or "python3"
    )
    script_file = os.path.normpath("scripts/caption_videos.py")

    # For custom models, use qwen3_vl_8b_hf as the captioner type
    sm = (selected_model or "").lower()
    captioner_type = selected_model
    if sm == "custom":
        # For custom models, use HF backend (qwen3_vl_8b_hf or qwen3_vl_4b_hf based on preference)
        # Default to qwen3_vl_8b_hf for custom models
        captioner_type = "qwen3_vl_8b_hf"

    command = f'"{python_exe}" -u "{script_file}" "{dataset_folder_path}/" --output "{output_json_path}" --captioner-type {captioner_type}'

    if use_8bit:
        command += " --use-8bit"

    # Ensure instruction and max_new_tokens are included
    # Use json.dumps to properly escape the instruction string for the shell
    # This will add outer quotes and escape any internal quotes.
    escaped_instruction = json.dumps(instruction)
    command += f' --instruction {escaped_instruction}'
    command += f' --max-new-tokens {max_new_tokens}'

    # If using LLaVA or Qwen3, point to local curated folder under models/_misc
    # For custom models, use the provided custom path
    if sm == "custom" and custom_model_path:
        command += f' --qwen-model "{custom_model_path}"'
    elif sm == "llava_next_7b":
        command += ' --llava-model "models/_misc/LLaVA-NeXT-Video-7B-hf"'
    elif sm == "qwen3_vl_8b":
        command += ' --qwen-model "models/_misc/Qwen3-VL-8B-Instruct"'
    elif sm == "qwen3_vl_4b":
        command += ' --qwen-model "models/_misc/Qwen3-VL-4B-Instruct"'
    elif sm == "qwen3_vl_4b_hf":
        command += ' --qwen-model "models/_misc/Qwen3-VL-4B-Instruct"'
    elif sm == "qwen3_vl_8b_hf":
        command += ' --qwen-model "models/_misc/Qwen3-VL-8B-Instruct"'

    if selected_files:
        # Join selected filenames with a comma and escape for shell
        escaped_selected_files = json.dumps(",".join(selected_files))
        command += f' --selected-files {escaped_selected_files}'

    return command

def build_preprocess_command(
    input_captions_json_path: str,
    preprocess_output_dir: str,
    resolution_buckets_str: str,
    model_name_val: str,
    trigger_word: str,
) -> str:
    python_exe = (
        sys.executable
        or (os.path.join("venv", "bin", "python") if os.path.exists(os.path.join("venv", "bin", "python")) else None)
        or (os.path.join("venv", "Scripts", "python.exe") if os.path.exists(os.path.join("venv", "Scripts", "python.exe")) else None)
        or "python3"
    )
    script_file = os.path.normpath("scripts/preprocess_dataset.py")

    command = (
        f'"{python_exe}" -u "{script_file}" "{input_captions_json_path}" --output-dir "{preprocess_output_dir}" '
        f'--resolution-buckets "{resolution_buckets_str}" --caption-column "caption" '
        f'--video-column "media_path" --model-source "{model_name_val}"'
    )

    if trigger_word:
        command += f' --id-token "{trigger_word}"'

    return command

# ======================================================================================
# Async Task Runner & Process State Management (Handles running external scripts)
# ======================================================================================

async def run_dataset_script_command(
    command_str: str,
    page_ref: ft.Page,
    button_ref: ft.ElevatedButton,
    progress_bar_ref: ft.ProgressBar,
    output_field_ref: ft.TextField,
    original_button_text: str,
    set_bottom_app_bar_height_func, # Pass this function from layout_tab
    delete_button_ref=None,
    thumbnails_grid_control=None,
    on_success_callback=None,
):
    def append_output(text):
        output_field_ref.value += text
        output_field_ref.visible = True
        set_bottom_app_bar_height_func()
        try:
            output_field_ref.update()
            page_ref.update()
        except Exception:
            pass

    _progress_re = re.compile(r'^\[PROGRESS\]\s+(\d+)/(\d+)\s+(.+?)\s*\|\s*[\d.]+s\s+ETA:\s*([0-9:]+)')
    _done_re = re.compile(r'^\[DONE\]\s+(.*)')
    # Patterns for noisy HuggingFace output to skip
    _hf_noise_res = [
        re.compile(r'^Loading checkpoint shards', re.IGNORECASE),
        re.compile(r'.*torch_dtype.*deprecated', re.IGNORECASE),
        re.compile(r'^Setting `pad_token_id`', re.IGNORECASE),
        re.compile(r'^\s*\d+%\|'),  # tqdm checkpoint loading bars
    ]

    # Track last progress line to replace in-place in the output field
    _last_progress_line = [None]

    def update_progress(line_text):
        """Parse [PROGRESS] or [DONE] lines — update bar and replace last progress line."""
        m = _progress_re.match(line_text)
        if m:
            current, total, filename, eta = m.group(1), m.group(2), m.group(3), m.group(4)
            frac = int(current) / int(total) if int(total) else 0
            progress_bar_ref.value = frac

            compact = f"[{current}/{total}] {filename} | ETA: {eta}"
            lines = output_field_ref.value.splitlines()
            # Remove the previous compact progress line if present
            if _last_progress_line[0] is not None and lines and lines[-1] == _last_progress_line[0]:
                lines.pop()
            lines.append(compact)
            _last_progress_line[0] = compact
            output_field_ref.value = "\n".join(lines) + "\n"

            try:
                progress_bar_ref.update()
                output_field_ref.update()
            except Exception:
                pass
            return True

        m2 = _done_re.match(line_text)
        if m2:
            progress_bar_ref.value = 1.0
            _last_progress_line[0] = None
            lines = output_field_ref.value.splitlines()
            lines.append(m2.group(1))
            output_field_ref.value = "\n".join(lines) + "\n"
            try:
                progress_bar_ref.update()
                output_field_ref.update()
            except Exception:
                pass
            return True

        return False

    try:
        output_field_ref.value = ""
        output_field_ref.visible = True
        progress_bar_ref.visible = True
        set_bottom_app_bar_height_func()
        button_ref.disabled = True
        button_ref.text = f"{original_button_text.replace('Add', 'Adding')}..." # Dynamic button text
        try:
            page_ref.update()
        except Exception:
            pass
        # Echo the command for transparency/debug
        append_output(f"\n[Cmd] {command_str}\n")

        # Start captioning process in its own session/process group so Stop won't kill our console
        create_kwargs = {}
        try:
            if os.name == 'nt':
                create_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                # On POSIX (including WSL/Linux), start a new session
                create_kwargs["start_new_session"] = True
        except Exception:
            pass

        process = await asyncio.create_subprocess_shell(
            command_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **create_kwargs,
        )
        current_caption_process["proc"] = process

        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_text = line.decode(errors='replace').rstrip('\n\r')
            # Skip noisy HuggingFace/loader output
            if any(pat.match(line_text) for pat in _hf_noise_res):
                continue
            # Handle structured progress lines — update bar, don't spam output
            if update_progress(line_text):
                continue
            append_output(line_text + "\n")

        rc = await process.wait()
        current_caption_process["proc"] = None

        if rc == 0:
            success_message = f"Script successful: {output_field_ref.value.splitlines()[-1] if output_field_ref.value else 'No output'}..."
            page_ref.snack_bar = ft.SnackBar(content=ft.Text(success_message), open=True)
            if on_success_callback:
                on_success_callback()
        else:
            err_msg = f"Script failed with exit code {rc}\nLast output:\n{output_field_ref.value.splitlines()[-50:] if output_field_ref.value else 'N/A'}" # Show last 50 lines of error
            append_output(err_msg)
            page_ref.snack_bar = ft.SnackBar(content=ft.Text(f"Script failed (code {rc}). Check output field for details."), open=True)

    except Exception as e:
        error_trace = f"Cmd failed: {e}\n"
        append_output(error_trace)
        page_ref.snack_bar = ft.SnackBar(content=ft.Text(f"Command execution failed: {e}"), open=True)

    finally:
        current_caption_process["proc"] = None
        button_ref.text = original_button_text
        button_ref.disabled = False
        progress_bar_ref.visible = False
        set_bottom_app_bar_height_func() # Re-evaluate height in case output field is no longer visible
        # --- Restore Delete button after captioning completes or fails ---
        if delete_button_ref is not None and thumbnails_grid_control is not None:
            delete_button_ref.text = "Delete"
            # The on_click for delete needs to be re-assigned to the correct handler in dataset_actions
            # This will be handled by passing the correct function reference from dataset_layout_tab
            # For now, keep it as a placeholder or assume it's set externally.
            # delete_button_ref.on_click = lambda evt: on_delete_captions_click(evt, thumbnails_grid_control)
            delete_button_ref.tooltip = "Delete the captions.json file"
            # Only re-enable if there's a dataset selected (handled in on_dataset_dropdown_change, but good to be safe)
            # This requires selected_dataset to be passed or accessed globally.
            # For now, assume it's passed or handled by the caller.
            # delete_button_ref.disabled = not selected_dataset.get("value")
            delete_button_ref.update()

        try:
            page_ref.update()
        except Exception:
            pass

# ======================================================================================
# GUI Event Handlers (Handle user interactions)
# ======================================================================================

async def on_change_fps_click(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, change_fps_textfield_ref_obj, thumbnails_grid_ref_obj, update_thumbnails_func, change_fps_checkbox_ref_obj=None):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    if not change_fps_textfield_ref_obj.current or not change_fps_textfield_ref_obj.current.value:
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
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Invalid FPS value '{target_fps_str}'. Must be a positive number."), open=True)
            e.page.update()
        return

    base_dir, _ = _get_dataset_base_dir(current_dataset_name)
    dataset_folder_path = os.path.abspath(os.path.join(base_dir, current_dataset_name))
    if not os.path.isdir(dataset_folder_path):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error: Dataset folder not found: {dataset_folder_path}"), open=True)
            e.page.update()
        return

    # Import the proper FFmpeg path resolution function
    try:
        from flet_app.ui_popups.video_player_utils import _get_ffmpeg_exe_path
        ffmpeg_exe = _get_ffmpeg_exe_path()

        # Determine ffprobe path based on ffmpeg path
        if os.path.isabs(ffmpeg_exe):
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe" + (".exe" if os.name == "nt" else ""))
        else:
            ffprobe_exe = "ffprobe"
    except ImportError:
        # Fallback to settings if import fails
        ffmpeg_exe = settings.FFMPEG_PATH
        if os.path.isabs(ffmpeg_exe) or os.path.sep in ffmpeg_exe:
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            ffprobe_basename = "ffprobe.exe" if ffmpeg_exe.lower().endswith(".exe") else "ffprobe"
            ffprobe_exe = os.path.join(ffmpeg_dir, ffprobe_basename)
        else:
            ffprobe_exe = "ffprobe"

    video_exts = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv', '.gif']
    processed_files = 0
    failed_files = 0
    skipped_files = 0

    # Get all video files in dataset
    all_video_files_in_dataset = [f for f in os.listdir(dataset_folder_path) if os.path.splitext(f)[1].lower() in video_exts]

    # Get selected videos from thumbnails
    selected_files_from_thumbnails = _get_selected_filenames(thumbnails_grid_ref_obj.current)

    if selected_files_from_thumbnails:
        print(f"[DEBUG] Processing {len(selected_files_from_thumbnails)} selected videos for FPS change.")
        # Ensure selected files are video files to prevent processing non-video files
        video_files_to_process = sorted([f for f in selected_files_from_thumbnails if f in all_video_files_in_dataset])
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Processing {len(video_files_to_process)} selected videos in {current_dataset_name} to {target_fps_float} FPS..."), open=True)
            e.page.update()
    else:
        print("[DEBUG] No videos selected, processing all videos in dataset.")
        video_files_to_process = sorted(all_video_files_in_dataset)
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Processing all videos in {current_dataset_name} to {target_fps_float} FPS..."), open=True)
            e.page.update()

    if not video_files_to_process:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("No video files found to process."), open=True)
            e.page.update()
        return

    for video_file_name in video_files_to_process:
        input_video_path = os.path.join(dataset_folder_path, video_file_name)
        base, ext = os.path.splitext(video_file_name)
        temp_output_video_path = os.path.join(dataset_folder_path, f"{base}_tempfps{ext}")
        original_fps = None

        try:
            # Get original FPS
            ffprobe_cmd = [
                ffprobe_exe, "-v", "error", "-select_streams", "v:0", 
                "-show_entries", "stream=r_frame_rate", "-of", 
                "default=noprint_wrappers=1:nokey=1", input_video_path
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"ffprobe error for {video_file_name}: {result.stderr}")
                failed_files += 1
                continue
            
            raw_r_frame_rate = result.stdout.strip()
            if not raw_r_frame_rate:
                print(f"Could not determine r_frame_rate for {video_file_name}.")
                failed_files +=1
                continue
            
            if '/' in raw_r_frame_rate:
                num, den = map(float, raw_r_frame_rate.split('/'))
                if den == 0: # Avoid division by zero
                    print(f"Invalid r_frame_rate denominator for {video_file_name}: {raw_r_frame_rate}")
                    failed_files += 1
                    continue
                original_fps = num / den
            else:
                original_fps = float(raw_r_frame_rate)

            if abs(original_fps - target_fps_float) < 0.01: # Comparing floats
                print(f"Skipping {video_file_name}, already at target FPS ({original_fps:.2f}).")
                skipped_files += 1
                continue

            # Check if "Don't care about time" checkbox is enabled
            dont_care_about_time = False
            if change_fps_checkbox_ref_obj and change_fps_checkbox_ref_obj.current:
                dont_care_about_time = change_fps_checkbox_ref_obj.current.value

            # Change FPS using different methods based on checkbox
            if dont_care_about_time:
                # Speed up/slow down video by changing FPS without changing frame count
                # Use setpts filter to change timestamps: PTS*old_fps/new_fps
                speed_factor = original_fps / target_fps_float
                print(f"Speeding up {video_file_name}: keeping {int(original_fps)} frames, FPS {original_fps:.2f} -> {target_fps_float} (speed factor: {speed_factor:.3f})")

                # Check if video has an audio stream
                has_audio = False
                try:
                    ffprobe_audio_cmd = [
                        ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_type", "-of",
                        "default=noprint_wrappers=1:nokey=1", input_video_path
                    ]
                    result = subprocess.run(ffprobe_audio_cmd, capture_output=True, text=True, check=False)
                    has_audio = "audio" in result.stdout.lower()
                except Exception:
                    has_audio = False

                # Build audio filter chain if speed factor is not 1.0 and audio exists
                audio_filter_str = None
                if has_audio:
                    audio_filters = []
                    remaining_speed = speed_factor

                    # atempo filter has range 0.5 to 2.0, so we need to chain filters for extreme speeds
                    while abs(remaining_speed - 1.0) > 0.01:  # If not essentially 1.0
                        if remaining_speed > 2.0:
                            audio_filters.append("atempo=2.0")
                            remaining_speed /= 2.0
                        elif remaining_speed < 0.5:
                            audio_filters.append("atempo=0.5")
                            remaining_speed /= 0.5
                        else:
                            audio_filters.append(f"atempo={remaining_speed}")
                            remaining_speed = 1.0

                    audio_filter_str = ",".join(audio_filters) if audio_filters else None

                # Use -r BEFORE input to reinterpret framerate without changing frame count
                # This keeps 65 frames @ 16fps as 65 frames but reinterprets them as 24fps
                ffmpeg_cmd = [
                    ffmpeg_exe, "-y",
                    "-r", str(target_fps_float),  # Reinterpret input framerate
                    "-i", input_video_path,
                    *vpu.VideoEncodingSettings.get_cpu_encoding_flags(),  # Re-encode video
                ]

                # Add audio handling
                if has_audio:
                    if audio_filter_str:
                        # Apply tempo filter to match new video duration
                        ffmpeg_cmd.extend(["-filter:a", audio_filter_str, "-c:a", "aac", "-b:a", "128k"])
                    else:
                        # Keep audio as-is
                        ffmpeg_cmd.extend(["-c:a", "copy"])
                else:
                    # No audio, exclude it
                    ffmpeg_cmd.append("-an")

                ffmpeg_cmd.append(temp_output_video_path)
            else:
                # Normal FPS change with re-encoding (changes duration to match new FPS)
                print(f"Changing FPS for {video_file_name}: {original_fps:.2f} -> {target_fps_float}")
                ffmpeg_cmd = [
                    ffmpeg_exe, "-y", "-i", input_video_path,
                    "-r", str(target_fps_float),
                    *vpu.VideoEncodingSettings.get_cpu_encoding_flags(),
                    "-c:a", "copy",           # Preserve original audio
                    temp_output_video_path
                ]
            print(f"Running: {' '.join(ffmpeg_cmd)}")
            result_ffmpeg = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)

            if result_ffmpeg.returncode == 0:
                shutil.move(temp_output_video_path, input_video_path)
                print(f"Successfully changed FPS for {video_file_name}")
                processed_files += 1
            else:
                print(f"ffmpeg error for {video_file_name}: {result_ffmpeg.stderr}")
                failed_files += 1
                if os.path.exists(temp_output_video_path):
                    os.remove(temp_output_video_path)
        except Exception as ex:
            print(f"Error processing {video_file_name}: {ex}")
            failed_files += 1
            if os.path.exists(temp_output_video_path):
                try:
                    os.remove(temp_output_video_path)
                except OSError as ose:
                    print(f"Could not remove temp file {temp_output_video_path}: {ose}")
    
    summary_message = f"FPS change complete. Processed: {processed_files}, Skipped: {skipped_files}, Failed: {failed_files}."
    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(summary_message), open=True)
        e.page.update()

    if processed_files > 0 and thumbnails_grid_ref_obj.current and e.page:
        if asyncio.iscoroutinefunction(update_thumbnails_func):
            await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)
        else:
            update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=True)

async def on_rename_files_click(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, rename_textfield_obj, thumbnails_grid_ref_obj, update_thumbnails_func):
    print("\n=== RENAME FUNCTION CALLED (dataset_actions.py) ===")
    current_dataset_name = selected_dataset_ref.get("value")
    print(f"[DEBUG] Current dataset name: {current_dataset_name}")
    
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for renaming."), open=True)
            e.page.update()
        return
    
    if not rename_textfield_obj or not rename_textfield_obj.value:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Rename textfield not available or empty."), open=True)
            e.page.update()
        return

    base_name = rename_textfield_obj.value.strip()
    if not base_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: Enter a base name in the rename field."), open=True)
            e.page.update()
        return

    # Unified naming: no '(img)' markers
    clean_current_name = current_dataset_name
    clean_base_name = base_name
    
    print(f"[DEBUG] Clean current name: {clean_current_name}")
    print(f"[DEBUG] Clean base name: {clean_base_name}")

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE_ref["value"]
    
    print(f"[DEBUG] Dataset type: {dataset_type}")
    
    # Unified datasets under DATASETS_DIR
    source_dir = os.path.join(settings.DATASETS_DIR, clean_current_name)
    target_dir = os.path.join(settings.DATASETS_DIR, clean_base_name)
    # Rename all supported media types by default
    file_extensions = list(dict.fromkeys(settings.IMAGE_EXTENSIONS + settings.VIDEO_EXTENSIONS))
    file_type = 'media'
    
    print(f"[DEBUG] Source directory: {source_dir}")
    print(f"[DEBUG] Target directory: {target_dir}")
    
    if not os.path.exists(source_dir):
        error_msg = f"Error: Source directory not found: {source_dir}"
        print(f"[ERROR] {error_msg}")
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            e.page.update()
        return
    
    # Get existing files with appropriate extensions
    existing_files = set()
    try:
        for f in os.listdir(source_dir):
            file_ext = os.path.splitext(f)[1].lower().lstrip('.')
            if file_ext in file_extensions:
                existing_files.add(f)
        print(f"[DEBUG] Found {len(existing_files)} files to rename")
        print(f"[DEBUG] Looking for extensions: {file_extensions}")
        print(f"[DEBUG] Files in directory: {os.listdir(source_dir)[:10]}")  # Print first 10 files for debugging
    except Exception as ex:
        error_msg = f"Error reading files from {source_dir}: {str(ex)}"
        print(f"[ERROR] {error_msg}")
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            e.page.update()
        return
    
    if not existing_files:
        error_msg = f"No {file_type} files found in {source_dir}"
        print(f"[ERROR] {error_msg}")
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            e.page.update()
        return
    
    selected_files_from_thumbnails = _get_selected_filenames(thumbnails_grid_ref_obj.current)
    
    if selected_files_from_thumbnails:
        print(f"[DEBUG] Renaming {len(selected_files_from_thumbnails)} selected files.")
        # Ensure selected files are part of existing_files to prevent renaming non-existent files
        files_to_rename = sorted([f for f in selected_files_from_thumbnails if f in existing_files])
    else:
        print("[DEBUG] No thumbnails selected, renaming all existing files.")
        files_to_rename = sorted(list(existing_files))  # Sort for consistent renaming order

    if not files_to_rename:
        if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"No {file_type} files found to rename in '{clean_current_name}'."), 
                open=True
            )
            e.page.update()
        return

    

    # Prepare new names and check for collisions
    new_names = []
    for idx, old_name in enumerate(files_to_rename, 1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{base_name}_{idx:02d}{ext}"
        new_names.append(new_name)

    # Check for duplicate new names (within the batch being renamed)
    if len(set(new_names)) != len(new_names):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Naming collision detected. Aborting."), open=True)
            e.page.update()
        return

    # Handle collisions: rename conflicting existing files to temporary names first
    # These are files that would be overwritten because their names match new_names
    files_to_rename_set = set(files_to_rename)
    conflicting_files = []
    temp_new_name_map = {}  # Maps temp name -> final new name for text file handling

    for new_name in new_names:
        # Check if this new name would overwrite an existing file that's NOT in our rename batch
        if new_name in existing_files and new_name not in files_to_rename_set:
            conflicting_files.append(new_name)

    # Thumbnail directory for renaming thumbnails
    thumbnails_dir = os.path.join(settings.THUMBNAILS_BASE_DIR, clean_current_name)

    if conflicting_files:
        print(f"[DEBUG] Found {len(conflicting_files)} conflicting files: {conflicting_files}")
        # Rename conflicting files to temporary names first
        # We'll use high numbers to avoid any conflicts
        temp_offset = max(len(files_to_rename) + len(conflicting_files), 100)
        for idx, conflict_name in enumerate(conflicting_files):
            old_path = os.path.join(source_dir, conflict_name)
            ext = os.path.splitext(conflict_name)[1]
            temp_name = f"__temp_rename_{temp_offset + idx:04d}{ext}"
            temp_path = os.path.join(source_dir, temp_name)

            try:
                os.rename(old_path, temp_path)
                temp_new_name_map[temp_name] = conflict_name  # Remember what this temp file should become
                print(f"[DEBUG] Renamed conflicting file {conflict_name} to temporary {temp_name}")

                # Also rename corresponding .txt if it exists
                conflict_base = os.path.splitext(conflict_name)[0]
                old_txt = os.path.join(source_dir, f"{conflict_base}.txt")
                if os.path.exists(old_txt):
                    temp_txt = os.path.join(source_dir, f"__temp_rename_{temp_offset + idx:04d}.txt")
                    os.rename(old_txt, temp_txt)
                    print(f"[DEBUG] Renamed conflicting txt {conflict_base}.txt to temporary {os.path.basename(temp_txt)}")

                # Also rename corresponding control file to temp if it exists
                control_dir = os.path.join(source_dir, "control")
                old_control = os.path.join(control_dir, conflict_name)
                if os.path.exists(old_control):
                    temp_control = os.path.join(control_dir, f"__temp_rename_{temp_offset + idx:04d}{os.path.splitext(conflict_name)[1]}")
                    os.rename(old_control, temp_control)
                    print(f"[DEBUG] Renamed conflicting control {conflict_name} to temporary {os.path.basename(temp_control)}")
                    # Also rename control .txt if it exists
                    old_control_txt = os.path.join(control_dir, f"{conflict_base}.txt")
                    if os.path.exists(old_control_txt):
                        temp_control_txt = os.path.join(control_dir, f"__temp_rename_{temp_offset + idx:04d}.txt")
                        os.rename(old_control_txt, temp_control_txt)
                        print(f"[DEBUG] Renamed conflicting control txt {conflict_base}.txt to temporary {os.path.basename(temp_control_txt)}")

                # Also rename corresponding thumbnail to temp if it exists
                if os.path.exists(thumbnails_dir):
                    for thumb_ext in ['.jpg', '.png']:
                        old_thumb = os.path.join(thumbnails_dir, f"{conflict_base}{thumb_ext}")
                        if os.path.exists(old_thumb):
                            temp_thumb = os.path.join(thumbnails_dir, f"__temp_rename_{temp_offset + idx:04d}{thumb_ext}")
                            os.rename(old_thumb, temp_thumb)
                            print(f"[DEBUG] Renamed conflicting thumbnail {conflict_base}{thumb_ext} to temporary {os.path.basename(temp_thumb)}")
                            break

            except Exception as ex:
                error_msg = f"Failed to rename conflicting file {conflict_name}: {ex}"
                print(f"[ERROR] {error_msg}")
                if e.page:
                    e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
                    e.page.update()
                return

    # Now safe to rename files (conflicting ones are now at temp names)
    old_to_new = {}
    renaming_successful = True
    for old_name, new_name in zip(files_to_rename, new_names):
        old_path = os.path.join(source_dir, old_name)
        new_path = os.path.join(source_dir, new_name)

        old_base, old_ext = os.path.splitext(old_name)
        new_base, new_ext = os.path.splitext(new_name)

        try:
            os.rename(old_path, new_path)
            old_to_new[old_name] = new_name
            print(f"[DEBUG] Renamed {old_name} to {new_name}")

            # Check for and rename corresponding .txt file
            old_txt_path = os.path.join(source_dir, f"{old_base}.txt")
            new_txt_path = os.path.join(source_dir, f"{new_base}.txt")
            if os.path.exists(old_txt_path):
                os.rename(old_txt_path, new_txt_path)
                print(f"[DEBUG] Renamed {os.path.basename(old_txt_path)} to {os.path.basename(new_txt_path)}")

            # Check for and rename corresponding control file if it exists
            control_dir = os.path.join(source_dir, "control")
            old_control_path = os.path.join(control_dir, old_name)
            new_control_path = os.path.join(control_dir, new_name)
            if os.path.exists(old_control_path):
                os.rename(old_control_path, new_control_path)
                print(f"[DEBUG] Renamed control {old_name} to {new_name}")
                # Also rename control .txt if it exists
                old_control_txt_path = os.path.join(control_dir, f"{old_base}.txt")
                new_control_txt_path = os.path.join(control_dir, f"{new_base}.txt")
                if os.path.exists(old_control_txt_path):
                    os.rename(old_control_txt_path, new_control_txt_path)
                    print(f"[DEBUG] Renamed control txt {old_base}.txt to {new_base}.txt")

            # Rename corresponding thumbnail if it exists
            if os.path.exists(thumbnails_dir):
                # Try both .jpg and .png extensions for thumbnails
                for thumb_ext in ['.jpg', '.png']:
                    old_thumb_path = os.path.join(thumbnails_dir, f"{old_base}{thumb_ext}")
                    new_thumb_path = os.path.join(thumbnails_dir, f"{new_base}{thumb_ext}")
                    if os.path.exists(old_thumb_path):
                        os.rename(old_thumb_path, new_thumb_path)
                        print(f"[DEBUG] Renamed thumbnail {old_base}{thumb_ext} to {new_base}{thumb_ext}")
                        break  # Only rename the first matching thumbnail

        except Exception as ex:
            renaming_successful = False
            error_msg = f"Failed to rename {old_name} to {new_name}: {ex}"
            print(f"[ERROR] {error_msg}")
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
                e.page.update()
            # Decide whether to stop or continue. Stopping is safer to avoid partial renames.
            return # Stop if any rename fails

    # Finally, rename the temporary files to their final intended names
    # Find available numbers that don't collide with any existing files
    if conflicting_files:
        # Get current files in directory (after main batch rename)
        current_files = set(os.listdir(source_dir))

        # Helper function to find next available number
        def find_available_number(base_nm, extension, start_idx, existing):
            idx = start_idx
            while True:
                candidate = f"{base_nm}_{idx:02d}{extension}"
                if candidate not in existing:
                    return idx, candidate
                idx += 1

        base_idx = 1
        for temp_name, final_name_base in sorted(temp_new_name_map.items()):
            temp_path = os.path.join(source_dir, temp_name)
            ext = os.path.splitext(temp_name)[1]

            # Find an available number that doesn't collide
            base_idx, final_name = find_available_number(base_name, ext, base_idx, current_files)
            final_path = os.path.join(source_dir, final_name)

            try:
                os.rename(temp_path, final_path)
                current_files.add(final_name)  # Track this new name
                print(f"[DEBUG] Renamed temp {temp_name} to final {final_name}")

                # Also rename the corresponding .txt file
                temp_base = os.path.splitext(temp_name)[0]
                final_base = os.path.splitext(final_name)[0]
                temp_txt = os.path.join(source_dir, f"{temp_base}.txt")
                final_txt = os.path.join(source_dir, f"{final_base}.txt")
                if os.path.exists(temp_txt):
                    os.rename(temp_txt, final_txt)
                    current_files.add(f"{final_base}.txt")  # Track txt too
                    print(f"[DEBUG] Renamed temp txt {temp_base}.txt to final {final_base}.txt")

                # Also rename the corresponding control file if it exists
                control_dir = os.path.join(source_dir, "control")
                temp_control = os.path.join(control_dir, temp_name)
                final_control = os.path.join(control_dir, final_name)
                if os.path.exists(temp_control):
                    os.rename(temp_control, final_control)
                    current_files.add(final_name)  # Track control file too
                    print(f"[DEBUG] Renamed temp control {temp_name} to final {final_name}")
                    # Also rename control .txt if it exists
                    temp_control_txt = os.path.join(control_dir, f"{temp_base}.txt")
                    final_control_txt = os.path.join(control_dir, f"{final_base}.txt")
                    if os.path.exists(temp_control_txt):
                        os.rename(temp_control_txt, final_control_txt)
                        current_files.add(f"{final_base}.txt")  # Track control txt too
                        print(f"[DEBUG] Renamed temp control txt {temp_base}.txt to final {final_base}.txt")

                # Also rename the corresponding thumbnail if it exists
                if os.path.exists(thumbnails_dir):
                    for thumb_ext in ['.jpg', '.png']:
                        temp_thumb = os.path.join(thumbnails_dir, f"{temp_base}{thumb_ext}")
                        final_thumb = os.path.join(thumbnails_dir, f"{final_base}{thumb_ext}")
                        if os.path.exists(temp_thumb):
                            os.rename(temp_thumb, final_thumb)
                            print(f"[DEBUG] Renamed temp thumbnail {temp_base}{thumb_ext} to final {final_base}{thumb_ext}")
                            break

                # Track this rename for info.json updates
                old_to_new[final_name_base] = final_name
                base_idx += 1  # Move to next number for next file

            except Exception as ex:
                print(f"[ERROR] Failed to rename temp file {temp_name} to {final_name}: {ex}")
                # Non-critical, the file is safe at temp name

    

    # Update info.json if exists (rename keys, preserve values, never duplicate)
    info_path = os.path.join(source_dir, "info.json")
    print(f"[DEBUG] Checking for info.json at: {info_path}")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    info_data = json.loads(content)
                    # Ensure the loaded data is a dictionary
                    if not isinstance(info_data, dict):
                        info_data = {}
                else:
                    info_data = {}

            changed = False
            # If info_data is a dict with video filename keys, rename keys
            if isinstance(info_data, dict):
                new_info_data = {}
                # Sort the original keys to maintain some order, though info.json structure might vary
                for k in sorted(info_data.keys()):
                    v = info_data[k]
                    new_key = old_to_new.get(k, k)
                    new_info_data[new_key] = v
                    if new_key != k:
                        changed = True
                info_data = new_info_data
            # No need for recursive update for other types if structure is standardized
            # Check for cap and proc directories and set flags
            cap_dir = os.path.join(source_dir, "cap")
            proc_dir = os.path.join(source_dir, "proc")
            
            if os.path.isdir(cap_dir):
                print(f"[DEBUG] Found 'cap' directory, setting 'cap' to 'yes'")
                info_data["cap"] = "yes"
            
            if os.path.isdir(proc_dir):
                print(f"[DEBUG] Found 'proc' directory, setting 'proc' to 'yes'")
                info_data["proc"] = "yes"
            
            # Save the updated info.json
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info_data, f, indent=2, ensure_ascii=False)
                print(f"[DEBUG] Updated info.json with cap/proc settings")
        except Exception as ex:
            print(f"Error updating info.json: {ex}")
            # Non-critical, continue

    # Success feedback and UI update
    if renaming_successful:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Renamed {len(files_to_rename)} files successfully."), open=True)
            if update_thumbnails_func:
                if asyncio.iscoroutinefunction(update_thumbnails_func):
                    await update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_ref_obj.current, force_refresh=False)  # Just reload, don't regenerate
                else:
                    update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_ref_obj.current, force_refresh=False)  # Just reload, don't regenerate
            e.page.update()

async def on_bucket_or_model_change(e: ft.ControlEvent, selected_dataset_ref, bucket_size_textfield_obj, model_name_dropdown_obj, trigger_word_textfield_obj):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        return

    bucket_str_val = bucket_size_textfield_obj.value # Initialize bucket_str_val

    # Validate bucket format and update bucket_str_val if invalid
    parsed_bucket_list = parse_bucket_string_to_list(bucket_str_val)
    if parsed_bucket_list is None:
        if e.page:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Invalid Bucket Size format: '{bucket_str_val}'. Using default."),
                open=True
            )
            e.page.update()
        bucket_str_val = settings.DEFAULT_BUCKET_SIZE_STR # Use default if invalid

    # Determine model to save: if dropdown provided, use it; else default
    try:
        model_val = (model_name_dropdown_obj.value if model_name_dropdown_obj else settings.train_def_model)
    except Exception:
        model_val = settings.train_def_model
    success = save_dataset_config(current_dataset_name, bucket_str_val, model_val, trigger_word_textfield_obj.value)
    if e.page:
        if success:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Config saved for {current_dataset_name}."),
                open=True
            )
        else:
            e.page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Error saving config for {current_dataset_name}"),
                open=True
            )
        e.page.update()

async def on_add_captions_click_with_model(e: ft.ControlEvent,
                                     caption_model_dropdown: ft.Dropdown,
                                     captions_checkbox: ft.Checkbox,
                                     hf_checkbox: ft.Checkbox,
                                     custom_model_path_textfield: ft.TextField,
                                     cap_command_textfield: ft.TextField,
                                     max_tokens_textfield: ft.TextField,
                                     dataset_add_captions_button_control: ft.ElevatedButton,
                                     dataset_delete_captions_button_control: ft.ElevatedButton,
                                     thumbnails_grid_control: ft.GridView,
                                     selected_dataset_ref,
                                     DATASETS_TYPE_ref,
                                     processed_progress_bar_ref,
                                     processed_output_field_ref,
                                     set_bottom_app_bar_height_func,
                                     update_thumbnails_func):
    # If a process is running, treat as stop
    proc = current_caption_process.get("proc")
    if proc is not None and proc.returncode is None:
        stop_captioning(
            e,
            dataset_add_captions_button_control,
            dataset_delete_captions_button_control,
            thumbnails_grid_control,
            selected_dataset_ref,
            processed_progress_bar_ref,
            processed_output_field_ref,
            set_bottom_app_bar_height_func,
            update_thumbnails_func
        )
        return

    selected_model = caption_model_dropdown.value or "llava_next_7b"
    custom_model_path = None
    # If Qwen model and HF checkbox set, map to *_hf variant
    try:
        sm = (selected_model or "").lower()
        if sm in ("qwen3_vl_4b", "qwen3_vl_8b") and hf_checkbox and getattr(hf_checkbox, 'value', False):
            if sm == "qwen3_vl_4b":
                selected_model = "qwen3_vl_4b_hf"
            elif sm == "qwen3_vl_8b":
                selected_model = "qwen3_vl_8b_hf"
        # Handle custom model - store the path separately
        elif sm == "custom":
            custom_path = getattr(custom_model_path_textfield, 'value', 'models/Qwen') or 'models/Qwen'
            # Convert relative path to absolute
            from flet_app.project_root import get_project_root
            project_root = get_project_root()
            custom_path_abs = os.path.normpath(os.path.join(project_root, custom_path))
            custom_model_path = custom_path_abs
    except Exception:
        pass
    try:
        # Surface immediate feedback
        if e.page:
            e.page.snack_bar = ft.SnackBar(ft.Text(f"Starting captioning with: {selected_model}"), open=True)
            e.page.update()
    except Exception:
        pass
    current_dataset_name = selected_dataset_ref.get("value")

    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE_ref["value"]
    
    # Unified datasets under DATASETS_DIR
    clean_dataset_name = current_dataset_name
    dataset_folder_path = os.path.abspath(os.path.join(settings.DATASETS_DIR, clean_dataset_name))
    
    output_json_path = os.path.join(dataset_folder_path, "captions.json")
    # Ensure output UI becomes visible early
    try:
        processed_output_field_ref.value = "[Init] Preparing caption job...\n"
        processed_output_field_ref.visible = True
        set_bottom_app_bar_height_func()
        if processed_output_field_ref.page:
            processed_output_field_ref.update()
    except Exception:
        pass

    selected_filenames = _get_selected_filenames(thumbnails_grid_control)
    
    if selected_filenames:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Captioning {len(selected_filenames)} selected videos..."), open=True)
            e.page.update()
    else:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Captioning all videos in the dataset..."), open=True)
            e.page.update()

    # Verify local model presence to avoid online downloads
    if (selected_model or "").lower() == "llava_next_7b":
        llava_local_dir = os.path.join("models", "_misc", "LLaVA-NeXT-Video-7B-hf")
        if not os.path.isdir(llava_local_dir):
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    content=ft.Text("LLaVA model not found at models/_misc/LLaVA-NeXT-Video-7B-hf. Use Models tab → Curated → Get to download."),
                    open=True,
                )
                e.page.update()
            return
    elif (selected_model or "").lower() in ("qwen3_vl_8b", "qwen3_vl_8b_hf"):
        qwen_dir_std = os.path.join("models", "_misc", "Qwen3-VL-8B-Instruct")
        if not os.path.isdir(qwen_dir_std):
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    content=ft.Text("Qwen3-VL-8B-Instruct not found at models/_misc/Qwen3-VL-8B-Instruct. Use Models tab → Curated → Get to download."),
                    open=True,
                )
                e.page.update()
            return
    elif (selected_model or "").lower() == "qwen3_vl_4b":
        qwen_dir_4b = os.path.join("models", "_misc", "Qwen3-VL-4B-Instruct")
        if not os.path.isdir(qwen_dir_4b):
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    content=ft.Text("Qwen3-VL-4B-Instruct not found at models/_misc/Qwen3-VL-4B-Instruct. Use Models tab → Curated → Get to download."),
                    open=True,
                )
                e.page.update()
            return
    elif (selected_model or "").lower() == "qwen3_vl_4b_hf":
        qwen_dir_4b = os.path.join("models", "_misc", "Qwen3-VL-4B-Instruct")
        if not os.path.isdir(qwen_dir_4b):
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    content=ft.Text("Qwen3-VL-4B-Instruct not found at models/_misc/Qwen3-VL-4B-Instruct. Use Models tab → Curated → Get to download."),
                    open=True,
                )
                e.page.update()
            return
    elif (selected_model or "").lower() == "joycaption_llava":
        joy_dir = os.path.join("models", "_misc", "joycaption-llava")
        if not os.path.isdir(joy_dir):
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    content=ft.Text("JoyCaption not found at models/_misc/joycaption-llava. Use Models tab → Curated → Get to download."),
                    open=True,
                )
                e.page.update()
            return

    # --- Build the command string ---
    if (selected_model or "").lower() == "joycaption_llava":
        # JoyCaption: use our caption_joy.py wrapper to emit captions.json from images,
        # then reuse existing json->txt conversion.
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        # For non-image datasets (e.g., mixed/video), filter selection to images and print skipped videos
        if dataset_type != "image":
            if selected_filenames:
                selected_images = [fn for fn in selected_filenames if os.path.splitext(fn)[1].lower() in image_exts]
                skipped_non_images = [fn for fn in selected_filenames if os.path.splitext(fn)[1].lower() not in image_exts]
                if skipped_non_images:
                    try:
                        processed_output_field_ref.value += "[Info] Skipping non-image files: " + ", ".join(skipped_non_images) + "\n"
                        processed_output_field_ref.visible = True
                        set_bottom_app_bar_height_func()
                        processed_output_field_ref.update() if processed_output_field_ref.page else None
                    except Exception:
                        pass
                # If no images remain, stop with a clear message
                if not selected_images:
                    try:
                        processed_output_field_ref.value += "[Error] No images selected after filtering. Select images or choose a video-capable model.\n"
                        processed_output_field_ref.visible = True
                        set_bottom_app_bar_height_func()
                        processed_output_field_ref.update() if processed_output_field_ref.page else None
                    except Exception:
                        pass
                    if e.page:
                        e.page.snack_bar = ft.SnackBar(content=ft.Text("No images to process. Select images only or pick another model."), open=True)
                        e.page.update()
                    return
                # Override selection to images only
                selected_filenames = selected_images
            else:
                # No explicit selection: don't guess in mixed datasets
                try:
                    processed_output_field_ref.value += "[Error] Mixed dataset detected. Please select images to caption with JoyCaption.\n"
                    processed_output_field_ref.visible = True
                    set_bottom_app_bar_height_func()
                    processed_output_field_ref.update() if processed_output_field_ref.page else None
                except Exception:
                    pass
                if e.page:
                    e.page.snack_bar = ft.SnackBar(content=ft.Text("Select images to run JoyCaption on a mixed dataset."), open=True)
                    e.page.update()
                return
        python_exe = (
            sys.executable
            or (os.path.join("venv", "bin", "python") if os.path.exists(os.path.join("venv", "bin", "python")) else None)
            or (os.path.join("venv", "Scripts", "python.exe") if os.path.exists(os.path.join("venv", "Scripts", "python.exe")) else None)
            or "python3"
        )
        joy_script = os.path.normpath("scripts/caption_joy.py")
        prompt_val = cap_command_textfield.value.strip() or "Write a descriptive caption for this image in a formal tone."
        max_tokens_val = int(max_tokens_textfield.value.strip() or 128)
        joy_dir = os.path.join("models", "_misc", "joycaption-llava")
        # Selected files (after filtering), if any, pass as comma-separated list
        sel_arg = json.dumps(",".join(selected_filenames)) if selected_filenames else ""
        command = (
            f'"{python_exe}" -u "{joy_script}" "{dataset_folder_path}" '
            f'--instruction {json.dumps(prompt_val)} '
            f'--max-new-tokens {max_tokens_val} --model-path "{joy_dir}"'
        )
        if sel_arg:
            command += f" --selected-files {sel_arg}"
        # JoyCaption writes .txt files directly; just update caption status
        post_success_cb = lambda: update_thumbnail_caption_status(thumbnails_grid_control, dataset_folder_path, dataset_type)
    else:
        # Default path: use our caption_videos.py pipeline (video/image)
        command = build_caption_command(
            dataset_folder_path=dataset_folder_path,
            output_json_path=output_json_path,
            selected_model=selected_model,
            use_8bit=((selected_model or "").lower() == "llava_next_7b" and captions_checkbox.value),
            instruction=cap_command_textfield.value.strip(),
            max_new_tokens=int(max_tokens_textfield.value.strip() or 100),
            selected_files=selected_filenames, # Pass selected files
            custom_model_path=custom_model_path # Pass custom model path
        )
        # After captioning (video/mixed), convert captions.json to .txt and update caption status
        async def convert_and_update_status():
            await on_caption_to_txt_click(e, selected_dataset_ref, DATASETS_TYPE_ref, None, None)
            # Update caption status after conversion
            update_thumbnail_caption_status(thumbnails_grid_control, dataset_folder_path, dataset_type)
        post_success_cb = lambda: e.page.run_task(convert_and_update_status)
    # Echo built command
    try:
        processed_output_field_ref.value += f"[Built] {command}\n"
        processed_output_field_ref.visible = True
        set_bottom_app_bar_height_func()
        processed_output_field_ref.update() if processed_output_field_ref.page else None
    except Exception:
        pass
    # ---------------------------------------------------------------------

    # Change Delete button to Stop
    dataset_delete_captions_button_control.text = "Stop"
    dataset_delete_captions_button_control.on_click = lambda evt: stop_captioning(
        evt,
        dataset_add_captions_button_control,
        dataset_delete_captions_button_control,
        thumbnails_grid_control,
        selected_dataset_ref,
        processed_progress_bar_ref,
        processed_output_field_ref,
        set_bottom_app_bar_height_func,
        update_thumbnails_func
    )
    dataset_delete_captions_button_control.tooltip = "Stop captioning process"
    dataset_delete_captions_button_control.disabled = False
    dataset_delete_captions_button_control.update()


    if e.page:
        e.page.update()
        # Run the command asynchronously
        try:
            e.page.run_task(
                run_dataset_script_command,
                command,
                e.page,
                dataset_add_captions_button_control,
                processed_progress_bar_ref,
                processed_output_field_ref,
                "Add Captions", # Original button text
                set_bottom_app_bar_height_func,
                delete_button_ref=dataset_delete_captions_button_control,
                thumbnails_grid_control=thumbnails_grid_control,
                on_success_callback=post_success_cb
            )
        except Exception as ex_sched:
            processed_output_field_ref.value += f"[Error] Failed to schedule job: {ex_sched}\n"
            processed_output_field_ref.visible = True
            set_bottom_app_bar_height_func()
            processed_output_field_ref.update() if processed_output_field_ref.page else None
            page_err = getattr(e, 'page', None)
            if page_err:
                page_err.snack_bar = ft.SnackBar(ft.Text(f"Failed to start job: {ex_sched}"), open=True)
                page_err.update()

def on_delete_captions_click(e: ft.ControlEvent, thumbnails_grid_control: ft.GridView, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func):
    page_for_dialog = e.page
    button_control = e.control
    current_dataset_name = selected_dataset_ref.get("value")

    if not current_dataset_name:
        if page_for_dialog:
            page_for_dialog.snack_bar = ft.SnackBar(content=ft.Text("No dataset selected."), open=True)
            page_for_dialog.update()
        return

    # Disable button while dialog is open
    if button_control:
        button_control.disabled = True
    if page_for_dialog:
        page_for_dialog.update()

    try:
        # Show confirmation dialog before deleting
        show_delete_caption_dialog(
            page_for_dialog,
            current_dataset_name,
            lambda: perform_delete_captions(page_for_dialog, thumbnails_grid_control, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func)
        )
    finally:
        # Re-enable button after dialog is closed
        if button_control:
             # Only re-enable if a dataset is still selected
            button_control.disabled = not selected_dataset_ref.get("value")
        if page_for_dialog:
            page_for_dialog.update()

def perform_delete_captions(page_context: ft.Page, thumbnails_grid_control: ft.GridView, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        return

    base_dir, dataset_type = _get_dataset_base_dir(current_dataset_name)
    dataset_folder_path = os.path.join(base_dir, current_dataset_name)
    media_files = get_media_files(dataset_folder_path, dataset_type)
    
    deleted_count = 0
    for media_path in media_files:
        base_filename, _ = os.path.splitext(os.path.basename(media_path))
        txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")
        if os.path.exists(txt_caption_path):
            try:
                os.remove(txt_caption_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting caption file {txt_caption_path}: {e}")

    if page_context:
        page_context.snack_bar = ft.SnackBar(content=ft.Text(f"Deleted {deleted_count} caption file(s) for {current_dataset_name}."), open=True)
        page_context.update()
    # Just update caption status instead of full thumbnail refresh
    update_thumbnail_caption_status(thumbnails_grid_control, dataset_folder_path, dataset_type)


def stop_captioning(e: ft.ControlEvent,
                    add_button: ft.ElevatedButton,
                    delete_button: ft.ElevatedButton,
                    thumbnails_grid_control: ft.GridView,
                    selected_dataset_ref,
                    processed_progress_bar_ref,
                    processed_output_field_ref,
                    set_bottom_app_bar_height_func,
                    update_thumbnails_func):
    # Try to kill by PID file (more reliable for spawned processes)
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../scripts/caption_pid.txt')
    killed = False
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            # Try to kill the process group
            os.kill(pid, signal.SIGTERM) # Or signal.SIGKILL
            killed = True
            print(f"Sent SIGTERM to process {pid}") # Debug
        except Exception as ex:
            print(f"Error sending signal to PID {pid}: {ex}") # Debug
        # Clean up PID file regardless
        try:
            os.remove(pid_file)
            print(f"Removed PID file: {pid_file}") # Debug
        except Exception as ex:
            print(f"Error removing PID file {pid_file}: {ex}") # Debug


    # Fallback to killing the tracked process tree if PID file failed or wasn't used
    if not killed:
        proc = current_caption_process.get("proc")
        if proc is not None and proc.returncode is None:
            print("Attempting to terminate/kill process tree...") # Debug
            try:
                # Terminate process tree on Windows, terminate single process on others
                if os.name == 'nt': # Windows
                    import subprocess
                    # Use taskkill /F /T /PID to force kill process tree
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)])
                else: # Linux/macOS
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                killed = True # Mark as killed if command ran without error
                print("Process tree termination/kill command sent.") # Debug
            except Exception as ex:
                print(f"Error terminating/killing process tree: {ex}") # Debug
                try:
                    proc.kill() # Final fallback
                    killed = True # Mark as killed if kill works
                    print("Used proc.kill() as final fallback.") # Debug
                except Exception as ex_kill:
                    print(f"Error with proc.kill() fallback: {ex_kill}") # Debug
            finally:
                 current_caption_process["proc"] = None # Clear tracked process


    # Restore Delete button to its original state
    delete_button.text = "Delete"
    # Re-assign on_click to the correct handler in dataset_actions
    delete_button.on_click = lambda evt: on_delete_captions_click(evt, thumbnails_grid_control, selected_dataset_ref, processed_progress_bar_ref, processed_output_field_ref, set_bottom_app_bar_height_func, update_thumbnails_func)
    delete_button.tooltip = "Delete the captions.json file"
    delete_button.disabled = not selected_dataset_ref.get("value") # Disable if no dataset selected
    delete_button.update()

    # Restore Add Captions button state (if it was disabled)
    add_button.text = "Add Captions" # Restore original text
    add_button.disabled = False # Re-enable button
    add_button.update()

    processed_progress_bar_ref.visible = False # Hide progress bar
    if processed_output_field_ref.page:
        processed_output_field_ref.value += "\n--- Process Stopped ---\n"
        processed_output_field_ref.update()
        set_bottom_app_bar_height_func() # Adjust height after adding text

    if e.page:
        e.page.snack_bar = ft.SnackBar(content=ft.Text("Captioning process stopped."), open=True)
        e.page.update()





async def on_preprocess_dataset_click(e: ft.ControlEvent,
                                bucket_size_textfield: ft.TextField,
                                trigger_word_textfield: ft.TextField,
                                selected_dataset_ref,
                                DATASETS_TYPE_ref,
                                processed_progress_bar_ref,
                                processed_output_field_ref,
                                set_bottom_app_bar_height_func,
                                update_thumbnails_func,
                                thumbnails_grid_control: ft.GridView): # Add this argument
    await on_caption_to_json_click(e, selected_dataset_ref, DATASETS_TYPE_ref, update_thumbnails_func, thumbnails_grid_control)
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected for preprocessing."), open=True)
            e.page.update()
        return

    # Unified naming: keep dataset name as-is
    clean_dataset_name = current_dataset_name

    # Determine dataset type using the global variable
    dataset_type = DATASETS_TYPE_ref["value"]
    
    dataset_dir = settings.DATASETS_DIR
    
    input_captions_json_path = os.path.abspath(os.path.join(dataset_dir, clean_dataset_name, "captions.json"))
    preprocess_output_dir = os.path.abspath(os.path.join(dataset_dir, clean_dataset_name, "preprocessed_data"))

    # Check if captions file exists
    if not os.path.exists(input_captions_json_path):
        error_msg = f"Error: Preprocessing input file not found: {input_captions_json_path}"
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            processed_output_field_ref.value = error_msg + "\n" # Display error in output field too
            processed_output_field_ref.visible = True
            set_bottom_app_bar_height_func()
            processed_output_field_ref.update()
            e.page.update()
        return

    # Use default model from settings (model dropdown removed)
    model_name_val = settings.train_def_model.strip()
    raw_bucket_str_val = bucket_size_textfield.value.strip()
    trigger_word = trigger_word_textfield.value.strip()

    if not raw_bucket_str_val:
        error_msg = "Error: Bucket Size cannot be empty."
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
            processed_output_field_ref.value = error_msg + "\n"
            processed_output_field_ref.visible = True
            set_bottom_app_bar_height_func()
            processed_output_field_ref.update()
            e.page.update()
        return

    # Validate and parse bucket size string
    parsed_bucket_list = parse_bucket_string_to_list(raw_bucket_str_val)
    if parsed_bucket_list is None:
        error_msg = f"Error parsing Bucket Size format: '{raw_bucket_str_val}'. Expected '[W, H, F]' or 'WxHxF'."
        if e.page:
             e.page.snack_bar = ft.SnackBar(content=ft.Text(error_msg), open=True)
             processed_output_field_ref.value = error_msg + "\n"
             processed_output_field_ref.visible = True
             set_bottom_app_bar_height_func()
             processed_output_field_ref.update()
             e.page.update()
        return

    # Validate bucket values (divisible by 32, etc.)
    error_messages = validate_bucket_values(*parsed_bucket_list)
    if error_messages:
        error_msg = "Bucket Size validation errors:\n" + "\n".join(error_messages)
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Bucket Size validation failed. Check output."), open=True)
            processed_output_field_ref.value = error_msg + "\n"
            processed_output_field_ref.visible = True
            set_bottom_app_bar_height_func()
            processed_output_field_ref.update()
            e.page.update()
        return

    # Format validated values for the script
    resolution_buckets_str = f"{parsed_bucket_list[0]}x{parsed_bucket_list[1]}x{parsed_bucket_list[2]}"

    # --- Build the command string using the dedicated helper function ---
    command = build_preprocess_command(
        input_captions_json_path=input_captions_json_path,
        preprocess_output_dir=preprocess_output_dir,
        resolution_buckets_str=resolution_buckets_str,
        model_name_val=model_name_val,
        trigger_word=trigger_word,
    )
    # ---------------------------------------------------------------------

    if e.page:
        # Run the command asynchronously
        e.page.run_task(
            run_dataset_script_command,
            command,
            e.page,
            e.control, # Pass the preprocess button itself
            processed_progress_bar_ref,
            processed_output_field_ref,
            "Start Preprocess", # Original button text
            set_bottom_app_bar_height_func,
            on_success_callback=lambda: update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True) # Force refresh after preprocessing
        )

async def on_caption_to_json_click(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, update_thumbnails_func, thumbnails_grid_ref_obj):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    dataset_type = DATASETS_TYPE_ref["value"]
    base_dir, _ = _get_dataset_base_dir(current_dataset_name)
    clean_dataset_name = current_dataset_name
    dataset_folder_path = os.path.abspath(os.path.join(base_dir, clean_dataset_name))
    captions_json_path = os.path.join(dataset_folder_path, "captions.json")

    try:
        if os.path.exists(captions_json_path):
            with open(captions_json_path, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            if not isinstance(captions_data, list):
                captions_data = []
        else:
            captions_data = []

        media_files = get_media_files(dataset_folder_path, dataset_type)

        captions_dict = {os.path.basename(item['media_path']): item for item in captions_data if 'media_path' in item}

        updated_count = 0
        for media_path in media_files:
            base_filename, _ = os.path.splitext(os.path.basename(media_path))
            txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")
            neg_caption_path = os.path.join(dataset_folder_path, f"{base_filename}_neg.txt")

            caption_text = None
            neg_caption_text = None

            # Read positive caption
            if os.path.exists(txt_caption_path):
                with open(txt_caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()

            # Read negative caption
            if os.path.exists(neg_caption_path):
                with open(neg_caption_path, 'r', encoding='utf-8') as f:
                    neg_caption_text = f.read().strip()

            # Only update if we have at least a positive caption
            if caption_text:
                media_basename = os.path.basename(media_path)
                if media_basename in captions_dict:
                    captions_dict[media_basename]['caption'] = caption_text
                    # Add/update negative_caption if it exists
                    if neg_caption_text:
                        captions_dict[media_basename]['negative_caption'] = neg_caption_text
                else:
                    new_entry = {
                        "media_path": media_basename,
                        "caption": caption_text
                    }
                    # Add negative_caption if it exists
                    if neg_caption_text:
                        new_entry['negative_caption'] = neg_caption_text
                    captions_dict[media_basename] = new_entry
                updated_count += 1

        with open(captions_json_path, 'w', encoding='utf-8') as f:
            json.dump(list(captions_dict.values()), f, indent=4)

        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Updated {updated_count} captions in captions.json from .txt files."), open=True)
            if asyncio.iscoroutinefunction(update_thumbnails_func):
                await update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_ref_obj.current, force_refresh=True)
            else:
                update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_ref_obj.current, force_refresh=True)
            e.page.update()

    except Exception as ex:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"An error occurred: {ex}"), open=True)
            e.page.update()

async def on_caption_to_txt_click(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, update_thumbnails_func=None, thumbnails_grid_control=None):
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    base_dir, _ = _get_dataset_base_dir(current_dataset_name)
    clean_dataset_name = current_dataset_name
    dataset_folder_path = os.path.abspath(os.path.join(base_dir, clean_dataset_name))
    captions_json_path = os.path.join(dataset_folder_path, "captions.json")

    if not os.path.exists(captions_json_path):
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("captions.json not found."), open=True)
            e.page.update()
        return

    try:
        with open(captions_json_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)

        if not isinstance(captions_data, list):
            if e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text("captions.json is not a list."), open=True)
                e.page.update()
            return

        created_count = 0
        appended_count = 0
        for item in captions_data:
            if 'media_path' in item and 'caption' in item:
                base_filename, _ = os.path.splitext(item['media_path'])
                txt_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")

                # Check if .txt file already exists and has content
                existing_caption = ""
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        existing_caption = f.read().strip()

                new_caption = item['caption'].strip()

                # If existing caption exists, append the new one after it
                if existing_caption:
                    # Combine existing and new captions
                    combined_caption = f"{existing_caption} {new_caption}"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(combined_caption)
                    appended_count += 1
                else:
                    # No existing caption, just write the new one
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(new_caption)
                    created_count += 1

        if e.page:
            message_parts = []
            if created_count > 0:
                message_parts.append(f"created {created_count}")
            if appended_count > 0:
                message_parts.append(f"appended to {appended_count}")
            message = " and ".join(message_parts) + " .txt file(s)"
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Updated {message} from captions.json."), open=True)
            # Refresh thumbnails so the [cap - yes/no] indicator updates after video captioning
            try:
                if update_thumbnails_func and thumbnails_grid_control is not None:
                    if asyncio.iscoroutinefunction(update_thumbnails_func):
                        await update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True)
                    else:
                        update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_control, force_refresh=True)
            except Exception:
                pass
            e.page.update()
        # Remove captions.json after creating txt files
        try:
            os.remove(captions_json_path)
        except Exception:
            pass

    except Exception as ex:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"An error occurred: {ex}"), open=True)
            e.page.update()

async def apply_affix_from_textfield(e: ft.ControlEvent, affix_type: str, selected_dataset_ref, DATASETS_TYPE_ref, update_thumbnails_func, thumbnails_grid_ref_obj, affix_text_field_ref: ft.Ref[ft.TextField]):
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
    
    # Unified naming: use dataset name as-is
    clean_dataset_name = current_dataset_name
    dataset_dir = settings.DATASETS_DIR
        
    dataset_folder_path = os.path.join(dataset_dir, clean_dataset_name)

    selected_files_from_thumbnails = _get_selected_filenames(thumbnails_grid_ref_obj.current)
    
    media_files_to_process = []
    if selected_files_from_thumbnails:
        all_media_files = get_media_files(dataset_folder_path, DATASETS_TYPE_ref["value"])
        for media_file in all_media_files:
            if os.path.basename(media_file) in selected_files_from_thumbnails:
                media_files_to_process.append(media_file)
    else:
        media_files_to_process = get_media_files(dataset_folder_path, DATASETS_TYPE_ref["value"])

    if not media_files_to_process:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No media files to process."), open=True)
        if e.page: e.page.update()
        return

    modified_count = 0
    try:
        for media_path in media_files_to_process:
            base_filename, _ = os.path.splitext(os.path.basename(media_path))
            txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")
            
            caption_text = ""
            if os.path.exists(txt_caption_path):
                with open(txt_caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()

            if affix_type == "prefix":
                new_caption = f"{affix_text} {caption_text}"
            elif affix_type == "suffix":
                new_caption = f"{caption_text} {affix_text}"
            
            with open(txt_caption_path, 'w', encoding='utf-8') as f:
                f.write(new_caption)
            modified_count += 1
        
        if modified_count > 0:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Successfully {affix_type}ed to {modified_count} captions."), open=True)
            if affix_text_field_ref.current:
                affix_text_field_ref.current.value = ""
            
            if asyncio.iscoroutinefunction(update_thumbnails_func):
                await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=False)
            else:
                update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=False)
                
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error applying affix: {ex}"), open=True)
    finally:
        if e.page: e.page.update()

async def on_fill_empty_txt_click(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, update_thumbnails_func, thumbnails_grid_ref_obj):
    """Create empty .txt files for media files that don't have corresponding caption files."""
    current_dataset_name = selected_dataset_ref.get("value")
    if not current_dataset_name:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text("Error: No dataset selected."), open=True)
            e.page.update()
        return

    base_dir, dataset_type = _get_dataset_base_dir(current_dataset_name)
    clean_dataset_name = current_dataset_name
    dataset_folder_path = os.path.abspath(os.path.join(base_dir, clean_dataset_name))

    try:
        media_files = get_media_files(dataset_folder_path, dataset_type)

        created_count = 0
        for media_path in media_files:
            base_filename, _ = os.path.splitext(os.path.basename(media_path))
            txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")

            # Only create if .txt doesn't exist
            if not os.path.exists(txt_caption_path):
                with open(txt_caption_path, 'w', encoding='utf-8') as f:
                    f.write("")  # Create empty file
                created_count += 1

        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Created {created_count} empty .txt file(s)."), open=True)
            if asyncio.iscoroutinefunction(update_thumbnails_func):
                await update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_ref_obj.current, force_refresh=True)
            else:
                update_thumbnails_func(page_ctx=e.page, grid_control=thumbnails_grid_ref_obj.current, force_refresh=True)
            e.page.update()

    except Exception as ex:
        if e.page:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"An error occurred: {ex}"), open=True)
            e.page.update()


async def find_and_replace_in_captions(e: ft.ControlEvent, selected_dataset_ref, DATASETS_TYPE_ref, find_text_field_ref: ft.Ref[ft.TextField], replace_text_field_ref: ft.Ref[ft.TextField], update_thumbnails_func, thumbnails_grid_ref_obj):
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
        
    dataset_folder_path = os.path.join(dataset_dir, clean_dataset_name)

    selected_files_from_thumbnails = _get_selected_filenames(thumbnails_grid_ref_obj.current)
    
    media_files_to_process = []
    if selected_files_from_thumbnails:
        all_media_files = get_media_files(dataset_folder_path, DATASETS_TYPE_ref["value"])
        for media_file in all_media_files:
            if os.path.basename(media_file) in selected_files_from_thumbnails:
                media_files_to_process.append(media_file)
    else:
        media_files_to_process = get_media_files(dataset_folder_path, DATASETS_TYPE_ref["value"])

    if not media_files_to_process:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"No media files to process."), open=True)
        if e.page: e.page.update()
        return

    modified_count = 0
    replacements_made = 0
    try:
        for media_path in media_files_to_process:
            base_filename, _ = os.path.splitext(os.path.basename(media_path))
            txt_caption_path = os.path.join(dataset_folder_path, f"{base_filename}.txt")
            
            caption_text = ""
            if os.path.exists(txt_caption_path):
                with open(txt_caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read()

            original_caption = caption_text
            new_caption = original_caption.replace(find_text, replace_text)

            if original_caption != new_caption:
                with open(txt_caption_path, 'w', encoding='utf-8') as f:
                    f.write(new_caption)
                modified_count += 1
                replacements_made += original_caption.count(find_text) if find_text else 0
        
        if modified_count > 0:
            e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Made {replacements_made} replacement(s) in {modified_count} caption(s)."), open=True)
            if find_text_field_ref.current: find_text_field_ref.current.value = ""
            if replace_text_field_ref.current: replace_text_field_ref.current.value = ""
            
            if asyncio.iscoroutinefunction(update_thumbnails_func):
                await update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=False)
            else:
                update_thumbnails_func(e.page, thumbnails_grid_ref_obj.current, force_refresh=False)
                
    except Exception as ex:
        e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error during find and replace: {ex}"), open=True)
    finally:
        if e.page: e.page.update()
