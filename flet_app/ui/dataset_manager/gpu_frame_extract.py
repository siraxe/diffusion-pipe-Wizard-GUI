import os
import subprocess
import json
import time
import signal
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Global state for cancellation
_cancel_flag = threading.Event()
_active_processes = []
_active_processes_lock = threading.Lock()


def cancel_extraction():
    """Cancel any running extraction and kill active ffmpeg processes."""
    global _cancel_flag, _active_processes
    _cancel_flag.set()

    with _active_processes_lock:
        for proc in _active_processes:
            try:
                if proc.poll() is None:  # Process is still running
                    if os.name == 'nt':
                        proc.terminate()
                    else:
                        proc.terminate()
                    print(f"Terminated ffmpeg process (PID: {proc.pid})")
            except Exception as e:
                print(f"Error terminating process: {e}")
        _active_processes.clear()


def reset_cancel_flag():
    """Reset the cancellation flag for a new extraction."""
    global _cancel_flag
    _cancel_flag.clear()


def is_cancelled():
    """Check if extraction has been cancelled."""
    return _cancel_flag.is_set()


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except:
            return 0.0
    return 0.0


def extract_batch_gpu(video_path: str, timestamps: list, output_paths: list) -> int:
    if not timestamps:
        return 0

    if is_cancelled():
        return 0

    cmd = ["ffmpeg", "-y", "-hide_banner", "-hwaccel", "cuda"]

    for ts in timestamps:
        cmd.extend(["-ss", str(ts), "-i", video_path])

    for i, out_path in enumerate(output_paths):
        cmd.extend([
            "-map", f"{i}:v",
            "-frames:v", "1",
            out_path
        ])

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        # Track the process for potential cancellation
        with _active_processes_lock:
            _active_processes.append(proc)

        proc.wait()  # Wait for completion without buffering output

        # Remove from active processes
        with _active_processes_lock:
            if proc in _active_processes:
                _active_processes.remove(proc)

        return sum(1 for p in output_paths if Path(p).exists())
    except Exception as e:
        if proc:
            with _active_processes_lock:
                if proc in _active_processes:
                    _active_processes.remove(proc)
        print(f"FFmpeg Error: {e}")
        return 0


def extract_frames_gpu(video_path: str, num_frames: int, output_dir: Path, frame_count_tf=None) -> tuple[int, str, str]:
    video_name = Path(video_path).stem
    error_msg = ""
    BATCH_SIZE = 120  # Process frames in batches to avoid argument list too long

    try:
        print(f"Processing video: {video_path}")

        try:
            if frame_count_tf is not None and hasattr(frame_count_tf, 'value'):
                extracted_num_frames = int(frame_count_tf.value or "3")
            else:
                extracted_num_frames = num_frames
            if extracted_num_frames < 1:
                extracted_num_frames = 1
        except ValueError:
            extracted_num_frames = 3

        duration = get_video_duration(video_path)
        if duration <= 0:
            return 0, video_name, "Could not get video duration"

        timestamps = []
        output_names = []

        if extracted_num_frames == 1:
            timestamps = [0.0]
            output_names = [f"{video_name}_start.png"]
        elif extracted_num_frames == 2:
            timestamps = [0.0, duration - 1.0]
            output_names = [f"{video_name}_start.png", f"{video_name}_end.png"]
        elif extracted_num_frames == 3:
            timestamps = [0.0, duration / 2, duration - 1.0]
            output_names = [f"{video_name}_start.png", f"{video_name}_mid.png", f"{video_name}_end.png"]
        else:
            step = duration / extracted_num_frames
            timestamps = [i * step for i in range(extracted_num_frames)]
            output_names = [f"{video_name}_frame_{i:04d}.png" for i in range(extracted_num_frames)]

        full_output_paths = [str(output_dir / name) for name in output_names]
        total_frames = len(timestamps)

        # Process in batches to avoid "Argument list too long" error
        extracted_count = 0
        start_time = time.time()
        batch_times = []

        total_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx, i in enumerate(range(0, total_frames, BATCH_SIZE), 1):
            # Check for cancellation before each batch
            if is_cancelled():
                print(f"\n  Extraction cancelled at batch {batch_idx}/{total_batches}")
                return extracted_count, video_name, "Cancelled"

            batch_start = time.time()
            batch_ts = timestamps[i:i + BATCH_SIZE]
            batch_paths = full_output_paths[i:i + BATCH_SIZE]
            batch_extracted = extract_batch_gpu(video_path, batch_ts, batch_paths)
            extracted_count += batch_extracted
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Calculate ETA
            elapsed = time.time() - start_time
            frames_done = extracted_count
            frames_left = total_frames - frames_done
            avg_time_per_batch = sum(batch_times) / len(batch_times)
            batches_left = total_batches - batch_idx
            eta_seconds = batches_left * avg_time_per_batch

            # Format ETA
            if eta_seconds >= 60:
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = f"{int(eta_seconds)}s"

            # Print progress (overwrite previous line)
            progress_pct = (frames_done / total_frames) * 100
            print(f"\r  Progress: {frames_done}/{total_frames} frames ({progress_pct:.1f}%) | "
                  f"Remaining: {frames_left} | ETA: {eta_str}    ", end="", flush=True)

        print()  # New line after progress
        elapsed_total = time.time() - start_time
        if is_cancelled():
            print(f"Extraction cancelled - {extracted_count}/{total_frames} frames extracted")
        else:
            print(f"Extracted {extracted_count}/{total_frames} frames in {elapsed_total:.1f}s")
        return extracted_count, video_name, ""

    except Exception as ex:
        import traceback
        traceback.print_exc()
        return 0, video_name, str(ex)


def extract_frames_parallel(video_paths: list, num_frames: int, output_dir: Path,
                            max_workers: int = None, progress_callback=None,
                            frame_count_tf=None) -> dict:

    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(8, cpu_count) if cpu_count >= 8 else min(4, cpu_count)

    results = {}

    print(f"Starting GPU extraction with {max_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(extract_frames_gpu, v, num_frames, output_dir, frame_count_tf): v
            for v in video_paths
        }

        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                extracted, video_name, error = future.result()
                results[video_path] = (extracted, video_name, error)

                if progress_callback:
                    progress_callback(video_path, extracted, video_name, error)

            except Exception as ex:
                results[video_path] = (0, Path(video_path).stem, str(ex))

    return results
