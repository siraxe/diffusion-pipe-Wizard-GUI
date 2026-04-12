"""
Video utility functions for dataset management.
Handles Create VACE, Trim to max, and PySceneDetect operations.
"""

import flet as ft
import subprocess
import threading
import random
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_video_info(ffmpeg_exe: str, video_path: Path) -> dict:
    """Get video information using ffprobe."""
    probe_cmd = [
        ffmpeg_exe.replace('ffmpeg', 'ffprobe'),
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
        '-count_frames',
        '-of', 'json',
        str(video_path)
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_result.returncode != 0:
        raise RuntimeError("Failed to get stream info")

    video_info = json.loads(probe_result.stdout)
    stream = video_info.get('streams', [{}])[0]

    return {
        'codec': stream.get('codec_name', 'libx264'),
        'width': stream.get('width', 512),
        'height': stream.get('height', 512),
        'fps_str': stream.get('r_frame_rate', '30/1'),
        'nb_frames': int(stream.get('nb_read_frames', 0))
    }


def process_video_for_vace(
    video_path: Path,
    control_dir: Path,
    ffmpeg_exe: str,
    codec_flags: list,
    start_frames: int,
    end_frames: int,
    fill_color_hex: str
) -> dict:
    """
    Process a single video to create VACE control files.
    Creates control video with colored middle section and mask video.
    """
    try:
        video_path = Path(video_path)
        video_name = video_path.stem
        video_ext = video_path.suffix

        control_video_path = control_dir / f"{video_name}{video_ext}"
        mask_video_path = control_dir / f"{video_name}_mask{video_ext}"

        print(f"Processing {video_name}...")

        # Get video info
        info = get_video_info(ffmpeg_exe, video_path)

        if info['nb_frames'] == 0:
            return {"status": "error", "video": video_name, "error": "Cannot determine frame count"}

        codec = info['codec']
        width = info['width']
        height = info['height']
        fps_str = info['fps_str']
        nb_frames = info['nb_frames']

        # Parse and format fps for ffmpeg (keep as fraction like "25/1")
        if '/' in fps_str:
            output_fps = f"{fps_str}"
        else:
            output_fps = str(fps_str)

        print(f"Video info: {nb_frames} frames, {output_fps} fps, {width}x{height}")

        # Calculate frame ranges for VACE processing
        keep_end_frame = start_frames - 1           # Last frame to KEEP (0-indexed)
        generate_start_frame = start_frames          # First frame to COLOR (0-indexed)
        generate_end_frame = nb_frames - end_frames  # Last frame to color
        print(f"Frame ranges: keep=[0,{keep_end_frame}], colored=[{generate_start_frame},{generate_end_frame}], keep_last=[{nb_frames-end_frames},{nb_frames-1}]")

        # Create VACE filter for control video (colored middle section)
        vace_filter = f"drawbox=x=0:y=0:w=iw:h=ih:color={fill_color_hex}:t=fill:enable='between(n,{generate_start_frame},{generate_end_frame})'"

        # Create control video with colored middle frames (no audio needed for VACE)
        control_cmd = [
            ffmpeg_exe, '-y',
            '-i', str(video_path),
            '-vf', vace_filter,
            '-an',                   # Disable audio - not needed for VACE control videos
            '-c:v', codec,
            '-s', f'{width}x{height}',
            '-r', output_fps,
            *codec_flags,
            '-g', '1',              # Every frame is a keyframe - prevents motion vector artifacts
            '-tune', 'stillimage',  # Optimize for static content
            '-pix_fmt', 'yuv420p',
            str(control_video_path)
        ]

        print(f"Creating VACE control video: {' '.join(control_cmd)}")
        control_result = subprocess.run(control_cmd, capture_output=True, text=True)
        if control_result.returncode != 0:
            print(f"FFmpeg error: {control_result.stderr}")
            return {"status": "error", "video": video_name, "error": control_result.stderr[:200]}

        # Create mask video - black for keep frames (first + last), white for generate middle section
        vace_mask_filter = f"color=c=black:s={width}x{height}:r={output_fps},drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(n,{generate_start_frame},{generate_end_frame})'"

        mask_cmd = [
            ffmpeg_exe, '-y',
            '-f', 'lavfi', '-i', vace_mask_filter,
            '-c:v', codec,
            *codec_flags,
            '-pix_fmt', 'yuv420p',
            '-frames:v', str(nb_frames),  # Set frame count to match original video
            str(mask_video_path)
        ]

        print(f"Creating VACE mask video: {' '.join(mask_cmd)}")
        mask_result = subprocess.run(mask_cmd, capture_output=True, text=True)
        if mask_result.returncode != 0:
            print(f"FFmpeg error: {mask_result.stderr}")
            return {"status": "error", "video": video_name, "error": mask_result.stderr[:200]}

        return {"status": "success", "video": video_name}

    except Exception as ex:
        print(f"Error processing {video_path}: {ex}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "video": Path(video_path).name, "error": str(ex)}


def process_video_for_trim(
    video_path: Path,
    ffmpeg_exe: str,
    codec_flags: list,
    max_frames: int
) -> dict:
    """
    Process a single video to trim if needed.
    Trims videos longer than max_frames by cutting off the end.
    """
    try:
        video_path = Path(video_path)
        video_name = video_path.stem

        # Get video info
        info = get_video_info(ffmpeg_exe, video_path)

        if info['nb_frames'] == 0:
            return {"status": "error", "video": video_name, "error": "Cannot determine frame count"}

        codec = info['codec']
        width = info['width']
        height = info['height']
        fps_str = info['fps_str']
        nb_frames = info['nb_frames']

        # Parse and format fps for ffmpeg (keep as fraction like "25/1")
        if '/' in fps_str:
            output_fps = f"{fps_str}"
        else:
            output_fps = str(fps_str)

        print(f"Video info: {nb_frames} frames, {output_fps} fps, {width}x{height}")

        # Check if trimming is needed
        if nb_frames <= max_frames:
            return {"status": "skipped", "video": video_name, "reason": f"Already at or below {max_frames} frames ({nb_frames})"}

        # Calculate duration to trim to (keep only first max_frames)
        fps_num = int(fps_str.split('/')[0])
        fps_den = int(fps_str.split('/')[1]) if '/' in fps_str else 1
        fps = fps_num / fps_den
        trim_duration = max_frames / fps

        print(f"Trimming {video_name} from {nb_frames} frames to {max_frames} frames (duration: {trim_duration:.2f}s)")

        # Trim video by cutting off the end using stream copy (fast, no re-encoding)
        # Similar approach as "Slice to:" button - uses -c copy for efficiency
        temp_path = video_path.with_suffix('.tmp.mp4')  # Need .mp4 extension so FFmpeg knows format
        trim_cmd = [
            ffmpeg_exe, '-y',
            '-i', str(video_path),
            '-t', str(trim_duration),  # Duration to keep (from start)
            '-c', 'copy',              # Stream copy - no re-encoding
            '-avoid_negative_ts', 'make_zero',
            '-fflags', '+genpts',
            str(temp_path)
        ]

        print(f"Trimming video: {' '.join(trim_cmd)}")
        trim_result = subprocess.run(trim_cmd, capture_output=True, text=True)
        if trim_result.returncode != 0:
            print(f"FFmpeg stderr: {trim_result.stderr}")
            return {"status": "error", "video": video_name, "error": f"FFmpeg failed: {trim_result.stderr[:1000]}"}

        # Move temp file back to original
        temp_path.replace(video_path)

        return {"status": "success", "video": video_name, "frames_before": nb_frames, "frames_after": max_frames}

    except Exception as ex:
        print(f"Error processing {video_path}: {ex}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "video": str(video_path), "error": str(ex)}


def process_video_for_time_remap(
    video_path: Path,
    ffmpeg_exe: str,
    codec_flags: list,
    speed_multiplier: float
) -> dict:
    """
    Process a single video to change speed via time remap.
    Keeps the same FPS — drops frames to speed up, duplicates frames to slow down.
    """
    try:
        import math

        video_path = Path(video_path)
        video_name = video_path.stem

        # Get video info
        info = get_video_info(ffmpeg_exe, video_path)

        if info['nb_frames'] == 0:
            return {"status": "error", "video": video_name, "error": "Cannot determine frame count"}

        codec = info['codec']
        width = info['width']
        height = info['height']
        fps_str = info['fps_str']
        nb_frames = info['nb_frames']

        # Parse original FPS
        if '/' in fps_str:
            output_fps = f"{fps_str}"
        else:
            output_fps = str(fps_str)

        # setpts changes timestamps, then fps filter forces back to original rate
        # fps filter drops or duplicates frames to match the target rate
        pts_factor = 1.0 / speed_multiplier
        video_filter = f"setpts={pts_factor:.4f}*PTS,fps={output_fps}"

        # Audio: atempo filter (valid range 0.5-100.0, chain if needed)
        speed = speed_multiplier
        atempo_filters = []
        while speed > 100.0:
            atempo_filters.append("atempo=100.0")
            speed /= 100.0
        temp_speed = speed
        while temp_speed < 0.5:
            atempo_filters.append("atempo=0.5")
            temp_speed /= 0.5
        atempo_filters.append(f"atempo={temp_speed:.4f}")

        temp_path = video_path.with_suffix('.tmp.mp4')
        cmd = [
            ffmpeg_exe, '-y',
            '-i', str(video_path),
            '-vf', video_filter,
            '-r', output_fps,
            '-c:v', codec,
            '-s', f'{width}x{height}',
            *codec_flags,
        ]

        # Add audio filter
        audio_filter_str = ",".join(atempo_filters)
        cmd.extend(["-af", audio_filter_str, "-c:a", "aac", "-b:a", "128k"])
        cmd.append(str(temp_path))

        new_frame_count = int(nb_frames / speed_multiplier)
        print(f"Time remapping {video_name}: speed={speed_multiplier}, {nb_frames} frames -> ~{new_frame_count} frames, fps unchanged ({output_fps})")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            return {"status": "error", "video": video_name, "error": f"FFmpeg failed: {result.stderr[:1000]}"}

        # Move temp file back to original
        temp_path.replace(video_path)

        return {"status": "success", "video": video_name, "frames_before": nb_frames, "frames_after": new_frame_count}

    except Exception as ex:
        print(f"Error processing {video_path}: {ex}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "video": str(video_path), "error": str(ex)}


def run_time_remap_processing(
    selected_videos: list,
    ffmpeg_exe: str,
    codec_flags: list,
    speed_multiplier: float,
    page_ctx,
    thumbnails_grid_ref
):
    """
    Run time remap processing on selected videos in a thread.
    """
    results = []

    def run_processing():
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_video_for_time_remap, Path(v), ffmpeg_exe, codec_flags, speed_multiplier): v for v in selected_videos}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    status = result.get('status', 'unknown')
                    video_name = result.get('video', 'unknown')
                    if status == 'success':
                        print(f"Remapped: {video_name} ({result.get('frames_before', '?')} -> {result.get('frames_after', '?')} frames)")
                    elif status == 'error':
                        print(f"Error: {video_name} - {result.get('error', 'unknown error')[:100]}")
                except Exception as ex:
                    results.append({"status": "error", "video": futures[future], "error": str(ex)})

        # Final summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        print(f"Time remap complete: {success_count} remapped, {error_count} errors")

        summary = f"Done! Time remapped: {success_count}, Errors: {error_count}"
        page_ctx.snack_bar = ft.SnackBar(ft.Text(summary), open=True)
        page_ctx.update()

    thread = threading.Thread(target=run_processing, daemon=False)
    thread.start()


def run_vace_processing(
    selected_videos: list,
    control_dir: Path,
    ffmpeg_exe: str,
    codec_flags: list,
    start_frames: int,
    end_frames: int,
    fill_color_hex: str,
    page_ctx,
    thumbnails_grid_ref
):
    """
    Run VACE processing on selected videos in a thread.
    """
    results = []
    total_videos = len(selected_videos)

    def run_processing():
        nonlocal results
        print(f"Starting processing of {total_videos} videos...")
        for i, video_path in enumerate(selected_videos):
            print(f"Processing video {i+1}/{total_videos}: {video_path}")
            result = process_video_for_vace(
                Path(video_path), control_dir, ffmpeg_exe, codec_flags,
                start_frames, end_frames, fill_color_hex
            )
            print(f"Result: {result}")
            results.append(result)

            # Update progress
            page_ctx.snack_bar = ft.SnackBar(
                ft.Text(f"Processing {i+1}/{total_videos}: {result.get('video', 'Unknown')} - {result.get('status', 'unknown')}"),
                open=True
            )
            page_ctx.update()

        # Final summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')
        error_count = sum(1 for r in results if r['status'] == 'error')
        print(f"Processing complete: {success_count} success, {skipped_count} skipped, {error_count} errors")
        summary = f"VACE created: {success_count}, Skipped: {skipped_count}, Errors: {error_count}"
        page_ctx.snack_bar = ft.SnackBar(ft.Text(summary), open=True)
        page_ctx.update()

    thread = threading.Thread(target=run_processing, daemon=False)
    thread.start()


# ============================================================================
# PySceneDetect Functions
# ============================================================================

def process_video_with_pyscenedetect(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len: str = "0.6s",
    create_combos: bool = False
) -> dict:
    """
    Process a single video using PySceneDetect to split by scenes.
    Splits video into same folder with _A, _B suffixes (letters).
    With combos enabled, creates suffix/prefix combinations:
      For A,B,C,D,E: individual (A,B,C,D,E), suffixes (E,DE,CDE,BCDE), prefixes (AB,ABC,ABCD)

    Args:
        video_path: Path to the input video
        threshold: Detection threshold (lower = more sensitive, default 27.0)
        min_scene_len: Minimum scene length (default "0.6s")
        create_combos: If True, creates combo clips; otherwise just individual scenes

    Returns:
        dict with status, video name, total output count, and any error message
    """
    try:
        from flet_app.ui_popups import video_player_utils as vpu
        import shutil

        video_path = Path(video_path)
        video_name = video_path.stem
        video_ext = video_path.suffix
        video_dir = video_path.parent

        print(f"Running PySceneDetect on {video_name} (threshold={threshold}, min_scene_len={min_scene_len})...")

        # Step 1: Detect scene boundaries only (don't split yet)
        cmd_detect = [
            'scenedetect',
            '-i', str(video_path),
            'detect-content',
            '--threshold', str(threshold),
            '--min-scene-len', min_scene_len,
            'list-scenes'
        ]

        print(f"Detecting scenes: {' '.join(cmd_detect)}")
        result = subprocess.run(cmd_detect, capture_output=True, text=True)

        # Debug: Show raw output
        print(f"  PySceneDetect stdout:\n{result.stdout}")
        if result.stderr:
            print(f"  PySceneDetect stderr: {result.stderr}")

        if result.returncode != 0:
            print(f"PySceneDetect error: {result.stderr}")
            return {
                "status": "error",
                "video": video_name,
                "error": f"PySceneDetect failed: {result.stderr[:500]}"
            }

        # Parse scene list output - extract from the table format
        scenes = []
        in_table = False
        for line in result.stdout.split('\n'):
            # Detect start of table
            if '| Scene #' in line:
                in_table = True
                continue
            if in_table and '|' in line:
                parts = line.split('|')
                if len(parts) >= 4:  # Has scene number, frame, time columns
                    try:
                        scene_num_str = parts[1].strip()
                        frame_str = parts[2].strip()
                        time_str = parts[3].strip()
                        if scene_num_str.isdigit() and frame_str.isdigit():
                            scenes.append({'frame': int(frame_str), 'time': _parse_timecode(time_str)})
                    except (ValueError, IndexError):
                        continue

        # Cleanup: Remove CSV files created by PySceneDetect (do this early, before any returns)
        csv_pattern = f"{video_name}-Scenes.csv"
        csv_path = video_dir / csv_pattern
        if csv_path.exists():
            csv_path.unlink()
            print(f"  Cleaned up: {csv_pattern}")

        if not scenes:
            # No scene changes detected - skip this video
            print(f"  No scene changes detected, skipping video")
            return {
                "status": "skipped",
                "video": video_name,
                "message": "No scene changes detected"
            }

        print(f"  Parsed {len(scenes)} scene start points:")
        for i, s in enumerate(scenes):
            print(f"    Scene {i+1}: frame {s['frame']}, time {s['time']:.3f}s")

        print(f"Detected {len(scenes)} scene(s)")
        for i, s in enumerate(scenes):
            print(f"  Scene {i+1}: frame {s['frame']}, time {s['time']:.3f}s")

        # Get video duration
        metadata = vpu.get_video_metadata(str(video_path))
        if not metadata or not metadata.get('fps'):
            return {"status": "error", "video": video_name, "error": "Could not get video metadata"}

        fps = metadata['fps']
        total_frames = metadata.get('total_frames', int(metadata.get('duration', 0) * fps))
        duration = total_frames / fps

        # Add end of video as final boundary
        scenes.append({'frame': total_frames, 'time': duration})

        output_files = []
        ffmpeg_exe = vpu._get_ffmpeg_exe_path()

        def extract_segment(start_time: float, end_time: float, scene_indices: list) -> str:
            """Extract a segment from the original video."""
            # Create filename using letters (A, B, C...) instead of numbers
            def idx_to_letter(idx):
                return chr(ord('A') + idx)

            if len(scene_indices) == 1:
                seg_name = f"{video_name}_{idx_to_letter(scene_indices[0])}"
            else:
                letters = ''.join(idx_to_letter(i) for i in scene_indices)
                seg_name = f"{video_name}_{letters}"

            out_path = video_dir / f"{seg_name}{video_ext}"

            # Handle existing files
            if out_path.exists():
                base = out_path.stem
                counter = 1
                while out_path.exists():
                    out_path = video_dir / f"{base}_{counter}{video_ext}"
                    counter += 1

            # Extract segment using ffmpeg (stream copy for speed)
            cmd = [
                ffmpeg_exe, '-y',
                '-ss', str(start_time),
                '-i', str(video_path),
                '-t', str(end_time - start_time),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                str(out_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and out_path.exists():
                return str(out_path)
            return None

        def idx_to_letter(idx):
            return chr(ord('A') + idx)

        num_scenes = len(scenes) - 1  # Exclude end boundary

        # Generate individual scenes: A, B, C, D, E
        for i in range(num_scenes):
            start_time = scenes[i]['time']
            end_time = scenes[i + 1]['time']

            output_file = extract_segment(start_time, end_time, [i])
            if output_file:
                output_files.append(output_file)
                desc = idx_to_letter(i)
                print(f"  Created: {desc} ({start_time:.2f}s - {end_time:.2f}s)")

        # Generate combos only if enabled
        if create_combos and num_scenes > 1:
            # Suffix combinations (from end): E, DE, CDE, BCDE (skip ABCDE - original exists)
            for length in range(1, num_scenes):
                start = num_scenes - length
                scene_indices = list(range(start, num_scenes))
                start_time = scenes[start]['time']
                end_time = scenes[-1]['time']

                output_file = extract_segment(start_time, end_time, scene_indices)
                if output_file:
                    output_files.append(output_file)
                    desc = ''.join(idx_to_letter(i) for i in scene_indices)
                    print(f"  Created: {desc} ({start_time:.2f}s - {end_time:.2f}s)")

            # Prefix combinations (from start): AB, ABC, ABCD (skip A alone and ABCDE)
            for length in range(2, num_scenes):
                scene_indices = list(range(length))
                start_time = scenes[0]['time']
                end_time = scenes[length]['time']

                output_file = extract_segment(start_time, end_time, scene_indices)
                if output_file:
                    output_files.append(output_file)
                    desc = ''.join(idx_to_letter(i) for i in scene_indices)
                    print(f"  Created: {desc} ({start_time:.2f}s - {end_time:.2f}s)")

        return {
            "status": "success",
            "video": video_name,
            "scene_count": num_scenes,
            "total_outputs": len(output_files),
            "output_files": output_files
        }

    except Exception as ex:
        print(f"Error processing {video_path}: {ex}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "video": str(video_path),
            "error": str(ex)
        }


def _parse_timecode(time_str: str) -> float:
    """Parse timecode string to seconds."""
    time_str = time_str.strip()
    parts = time_str.split(':')

    if len(parts) == 3:  # HH:MM:SS.mmm or MM:SS.mmm
        has_hours = len(parts[0]) > 2
        if has_hours:
            hours = int(parts[0])
            minutes = int(parts[1])
        else:
            hours = 0
            minutes = int(parts[0])

        # Parse seconds with milliseconds
        sec_parts = parts[-1].split('.')
        seconds = int(sec_parts[0])
        millis = int(sec_parts[1][:3].ljust(3, '0')) if len(sec_parts) > 1 else 0

        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
    elif len(parts) == 2:  # MM:SS.mmm
        minutes = int(parts[0])
        sec_parts = parts[1].split('.')
        seconds = int(sec_parts[0])
        millis = int(sec_parts[1][:3].ljust(3, '0')) if len(sec_parts) > 1 else 0
        return minutes * 60 + seconds + millis / 1000.0
    else:
        # Just seconds
        try:
            return float(time_str)
        except ValueError:
            return 0.0


def run_pyscenedetect_processing(
    selected_videos: list,
    threshold: float = 27.0,
    min_scene_len: str = "0.6s",
    create_combos: bool = False,
    page_ctx=None,
    thumbnails_grid_ref=None
):
    """
    Run PySceneDetect processing on selected videos in a thread.

    Args:
        selected_videos: List of video paths to process
        threshold: Detection threshold (lower = more sensitive, default 27.0)
        min_scene_len: Minimum scene length (default "0.6s")
        create_combos: Create combo clips (suffix/prefix combinations), otherwise just split scenes
        page_ctx: Flet page context for UI updates
        thumbnails_grid_ref: Reference to thumbnail grid for refresh
    """
    results = []
    total_videos = len(selected_videos)

    def run_processing():
        nonlocal results
        mode_str = "with combos" if create_combos else "individual scenes only"
        print(f"Starting PySceneDetect processing of {total_videos} videos ({mode_str})...")

        for i, video_path in enumerate(selected_videos):
            print(f"Processing video {i+1}/{total_videos}: {video_path}")
            result = process_video_with_pyscenedetect(
                Path(video_path), threshold, min_scene_len, create_combos
            )
            print(f"Result: {result}")
            results.append(result)

            # Update progress
            if page_ctx:
                status_msg = result.get('video', 'Unknown')
                if result.get('status') == 'success':
                    total_outputs = result.get('total_outputs', '?')
                    status_msg += f" - created {total_outputs} output(s)"
                else:
                    status_msg += f" - {result.get('status', 'unknown')}"

                page_ctx.snack_bar = ft.SnackBar(
                    ft.Text(f"Processing {i+1}/{total_videos}: {status_msg}"),
                    open=True
                )
                page_ctx.update()

        # Final summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        total_outputs = sum(r.get('total_outputs', 0) for r in results if r['status'] == 'success')

        print(f"PySceneDetect complete: {success_count} videos -> {total_outputs} outputs, {error_count} errors")

        if page_ctx:
            summary = f"Done! Created {total_outputs} video(s) from {success_count} source(s). Errors: {error_count}"
            page_ctx.snack_bar = ft.SnackBar(ft.Text(summary), open=True)
            page_ctx.update()

    thread = threading.Thread(target=run_processing, daemon=False)
    thread.start()


def run_trim_processing(
    selected_videos: list,
    ffmpeg_exe: str,
    codec_flags: list,
    max_frames: int,
    page_ctx,
    thumbnails_grid_ref
):
    """
    Run trim processing on selected videos in a thread.
    """
    results = []

    def run_processing():
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_video_for_trim, Path(v), ffmpeg_exe, codec_flags, max_frames): v for v in selected_videos}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    status = result.get('status', 'unknown')
                    video_name = result.get('video', 'unknown')
                    if status == 'success':
                        print(f"Trimmed: {video_name} ({result.get('frames_before')} -> {result.get('frames_after')} frames)")
                    elif status == 'skipped':
                        print(f"Skipped: {video_name} - {result.get('reason')}")
                    else:
                        print(f"Error: {video_name} - {result.get('error', 'unknown error')[:100]}")
                except Exception as ex:
                    results.append({"status": "error", "video": futures[future], "error": str(ex)})

        # Final summary
        success_count = sum(1 for r in results if r['status'] == 'success')
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')
        error_count = sum(1 for r in results if r['status'] == 'error')
        print(f"Processing complete: {success_count} trimmed, {skipped_count} skipped, {error_count} errors")

        summary = f"Done! Trimmed: {success_count}, Skipped: {skipped_count}, Errors: {error_count}"
        page_ctx.snack_bar = ft.SnackBar(ft.Text(summary), open=True)
        page_ctx.update()

    thread = threading.Thread(target=run_processing, daemon=False)
    thread.start()
