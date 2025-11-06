# video_player_utils.py
import os
import json
import cv2
import subprocess
import math
import platform
import shutil
from typing import Tuple, List, Dict, Any, Optional

# Attempt to import settings for FFMPEG_PATH
try:
    from settings import settings as app_settings
except ImportError:
    # Fallback if settings module is not found or FFMPEG_PATH is not there
    class FallbackSettings:
        FFMPEG_PATH = "ffmpeg" # Default to ffmpeg in PATH
        # Define other settings if needed, or handle their absence

        def get(self, name: str, default=None):
            """Allow attribute-style settings to behave like dict lookups."""
            return getattr(self, name, default)
    app_settings = FallbackSettings()

MIN_OVERLAY_SIZE = 20 # Minimum size for the overlay during resize (used in calculations)

# Caption and Data Handling
def load_caption_for_video(video_path: str) -> Tuple[str, str, Optional[str]]:
    """
    Load caption and negative caption for a given video from their corresponding .txt files.
    Returns: (caption, negative_caption, message_string_or_none)
    """
    video_dir = os.path.dirname(video_path)
    base_filename, _ = os.path.splitext(os.path.basename(video_path))
    caption_txt_path = os.path.join(video_dir, f"{base_filename}.txt")
    neg_caption_txt_path = os.path.join(video_dir, f"{base_filename}_neg.txt")
    no_caption_text = "No captions found, add it here and press Update"
    
    caption_value = ""
    negative_caption_value = ""
    message_to_display = no_caption_text

    if os.path.exists(caption_txt_path):
        try:
            with open(caption_txt_path, 'r', encoding='utf-8') as f:
                caption_value = f.read().strip()
            message_to_display = None  # Caption found
        except Exception as e:
            print(f"Error reading caption file {caption_txt_path}: {e}")
            message_to_display = f"Error loading caption: {e}"
    
    if os.path.exists(neg_caption_txt_path):
        try:
            with open(neg_caption_txt_path, 'r', encoding='utf-8') as f:
                negative_caption_value = f.read().strip()
            if message_to_display == no_caption_text: # Only clear if no other error/caption found
                message_to_display = None
        except Exception as e:
            print(f"Error reading negative caption file {neg_caption_txt_path}: {e}")
            message_to_display = f"Error loading negative caption: {e}"
            
    return caption_value, negative_caption_value, message_to_display

def save_caption_for_video(video_path: str, new_caption: str, field_name: str = "caption") -> Tuple[bool, str]:
    """
    Save the new caption for the given video to its corresponding .txt file.
    If field_name is 'negative_caption', saves to a _neg.txt file.
    Returns: (success_bool, message_string)
    """
    video_dir = os.path.dirname(video_path)
    base_filename, _ = os.path.splitext(os.path.basename(video_path))
    
    if field_name == "caption":
        caption_txt_path = os.path.join(video_dir, f"{base_filename}.txt")
    elif field_name == "negative_caption":
        caption_txt_path = os.path.join(video_dir, f"{base_filename}_neg.txt")
    else:
        return False, "Invalid field_name for saving caption."

    try:
        with open(caption_txt_path, 'w', encoding='utf-8') as f:
            f.write(new_caption)
        
        friendly_field_name = field_name.replace('_', ' ').title()
        return True, f"{friendly_field_name} updated!"
    except Exception as ex_write:
        msg = f"Error writing caption file {caption_txt_path}: {ex_write}"
        print(msg)
        return False, f"Failed to update {friendly_field_name}: {ex_write}"

def get_next_video_path(video_list: List[str], current_video_path: str, offset: int) -> Optional[str]:
    """
    Return the next video path in the list given the current path and offset.
    Wraps around if at the end/start.
    """
    if not video_list or not current_video_path:
        return None
    try:
        current_idx = video_list.index(current_video_path)
        new_idx = (current_idx + offset) % len(video_list)
        return video_list[new_idx]
    except ValueError:
        print(f"Error: Current video {current_video_path} not in list.")
        return None

# Video Metadata

def get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata (fps, total_frames, width, height) from a video file.
    Returns a dictionary or None if an error occurs.
    """
    if not video_path or not os.path.exists(video_path):
        print(f"Video path does not exist: {video_path}")
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            print(f"Invalid metadata for video {video_path}: fps={fps}, frames={total_frames}, w={width}, h={height}")
            # Allow partial metadata if some values are valid, but log it.
            # For critical operations, the caller should check for all necessary fields.

        return {'fps': fps, 'total_frames': total_frames, 'width': width, 'height': height}
    except Exception as e:
        print(f"Error getting video metadata for {video_path}: {e}")
        return None

def update_video_info_json(video_path: str) -> bool:
    """
    Updates the info.json for a given video path with its current metadata.
    Creates info.json if it doesn't exist.
    Returns True on success, False on failure.
    """
    video_dir = os.path.dirname(video_path)
    info_json_path = os.path.join(video_dir, 'info.json')
    info_data: Dict[str, Any] = {}

    if os.path.exists(info_json_path):
        try:
            with open(info_json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                info_data = loaded_data
            else:
                print(f"Warning: info.json at {info_json_path} was not a dictionary. Reinitializing.")
        except json.JSONDecodeError:
            print(f"Warning: info.json at {info_json_path} is corrupted. Reinitializing.")
        except Exception as e:
            print(f"Error loading info.json: {e}. Reinitializing.")
    
    video_filename = os.path.basename(video_path)
    metadata = get_video_metadata(video_path)

    if metadata:
        if video_filename not in info_data:
            info_data[video_filename] = {}
        info_data[video_filename]['frames'] = metadata.get('total_frames')
        info_data[video_filename]['fps'] = metadata.get('fps')
        info_data[video_filename]['width'] = metadata.get('width')
        info_data[video_filename]['height'] = metadata.get('height')
    else:
        print(f"Could not get metadata for {video_filename} to update info.json.")
        # Optionally remove entry or leave stale, depending on desired behavior
        # For now, we'll proceed to write even if metadata is missing, to preserve other entries.

    try:
        os.makedirs(video_dir, exist_ok=True)
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error writing info.json: {e}")
        return False

# Dimension Calculations

def _closest_divisible_by(value: int, divisor: int) -> int:
    """Helper to find the closest multiple of 'divisor' to 'value', ensuring minimum of 'divisor'."""
    return max(divisor, int(round(value / divisor)) * divisor)

def calculate_adjusted_size(
    current_w_str: str, 
    current_h_str: str, 
    video_path: Optional[str], 
    adjustment_type: str, # 'add' or 'sub'
    step: int = 32
) -> Tuple[str, str]:
    """
    Adjusts width and height by a step (default 32), maintaining aspect ratio if video_path is provided.
    Returns new (width_str, height_str).
    """
    try:
        w = int(current_w_str) if current_w_str else step
        h = int(current_h_str) if current_h_str else step
    except ValueError:
        return str(step), str(step)

    aspect_ratio = None
    if video_path:
        metadata = get_video_metadata(video_path)
        if metadata and metadata['width'] > 0 and metadata['height'] > 0:
            aspect_ratio = metadata['width'] / metadata['height']

    if adjustment_type == 'add':
        w = ((w // step) + 1) * step
        if aspect_ratio:
            h = _closest_divisible_by(w / aspect_ratio, step)
        else:
            h = ((h // step) + 1) * step
    elif adjustment_type == 'sub':
        w = max(step, ((w - 1) // step) * step)
        if aspect_ratio:
            h = _closest_divisible_by(w / aspect_ratio, step)
        else:
            h = max(step, ((h - 1) // step) * step)
    
    return str(w), str(h)

def calculate_closest_div32_dimensions(video_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculates dimensions for a video that are closest to original and divisible by 32.
    Returns (width_str, height_str) or (None, None).
    """
    metadata = get_video_metadata(video_path)
    if not metadata or metadata['width'] <= 0 or metadata['height'] <= 0:
        return None, None
    
    width32 = _closest_divisible_by(metadata['width'], 32)
    height32 = _closest_divisible_by(metadata['height'], 32)
    
    return str(width32), str(height32)

# FFmpeg Utilities

_RESOLVED_FFMPEG_PATH: Optional[str] = None


def _running_in_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        release = platform.release().lower()
        version = platform.version().lower()
    except Exception:
        release = ""
        version = ""
    return "microsoft" in release or "microsoft" in version or \
        "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ


def _is_executable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _get_ffmpeg_exe_path() -> str:
    """
    Resolve the FFmpeg executable path taking into account platform differences.
    On Windows we honour the configured path (often a bundled .exe).
    On Linux/WSL we prefer a native ffmpeg binary and avoid Windows .exe files
    which are not executable in that environment.
    """
    global _RESOLVED_FFMPEG_PATH
    if _RESOLVED_FFMPEG_PATH:
        return _RESOLVED_FFMPEG_PATH

    candidates: List[str] = []

    env_override = os.getenv("FFMPEG_PATH")
    if env_override:
        candidates.append(env_override)

    configured_path = getattr(app_settings, "FFMPEG_PATH", "ffmpeg")
    if configured_path:
        candidates.append(configured_path)

    # Ensure plain "ffmpeg" is always a fallback candidate
    candidates.append("ffmpeg")

    # In WSL, explicitly check common ffmpeg locations first
    if _running_in_wsl():
        candidates.extend([
            "/usr/bin/ffmpeg",
            "/bin/ffmpeg",
            "/usr/local/bin/ffmpeg"
        ])

    seen: set[str] = set()
    for raw_candidate in candidates:
        if not raw_candidate or raw_candidate in seen:
            continue
        seen.add(raw_candidate)

        candidate = raw_candidate.replace("\\", "/")

        if os.name != "nt":
            # Skip Windows binaries when running on Linux/WSL
            if candidate.lower().endswith(".exe"):
                continue

        # Absolute path and executable?
        if os.path.isabs(candidate) and _is_executable_file(candidate):
            _RESOLVED_FFMPEG_PATH = candidate
            return _RESOLVED_FFMPEG_PATH

        # Try resolving relative paths or commands via PATH
        resolved = shutil.which(candidate)
        if resolved and (os.name == "nt" or _is_executable_file(resolved)):
            _RESOLVED_FFMPEG_PATH = resolved.replace("\\", "/")
            return _RESOLVED_FFMPEG_PATH

    # Final fallback: assume 'ffmpeg' is in PATH
    _RESOLVED_FFMPEG_PATH = "ffmpeg"

    if _running_in_wsl():
        print("Warning: Falling back to native 'ffmpeg' executable for WSL. "
              "Update settings.FFMPEG_PATH if a specific binary is required.")
    else:
        print("Warning: Could not resolve configured FFmpeg binary. Falling back to 'ffmpeg' in PATH.")

    return _RESOLVED_FFMPEG_PATH

# ============================================================================
# CENTRALIZED VIDEO ENCODING SETTINGS - SINGLE SOURCE OF TRUTH
# ============================================================================

class VideoEncodingSettings:
    """Centralized video encoding settings to maintain consistency across the application."""

    # Base quality settings (used by all encoders)
    CRF_QUALITY = 18              # EXCELLENT quality (lower is better, 18 is great)
    TARGET_BITRATE = "4M"         # Target 4 Mbps bitrate for high quality
    MAX_BITRATE = "6M"            # Maximum bitrate for streaming
    BUFFER_SIZE = "12M"           # Buffer size for rate control
    KEYFRAME_INTERVAL = 32        # Keyframe every 2 seconds at 16fps (important for web)
    PIXEL_FORMAT = "yuv420p"      # Web-compatible pixel format
    WEB_OPTIMIZATION = "+faststart" # Optimize for web streaming
    PROFILE = "high"              # High profile for better compression
    LEVEL = "4.0"                 # Compatible level for most devices

    @classmethod
    def get_base_encoding_flags(cls) -> List[str]:
        """Get the base encoding flags used by all encoders."""
        return [
            "-pix_fmt", cls.PIXEL_FORMAT,
            "-movflags", cls.WEB_OPTIMIZATION,
            "-g", str(cls.KEYFRAME_INTERVAL),
            "-b:v", cls.TARGET_BITRATE,
            "-maxrate", cls.MAX_BITRATE,
            "-bufsize", cls.BUFFER_SIZE,
            "-profile:v", cls.PROFILE,
            "-level", cls.LEVEL
        ]

    @classmethod
    def get_cpu_encoding_flags(cls) -> List[str]:
        """Get CPU (libx264) encoding flags."""
        return [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(cls.CRF_QUALITY)
        ] + cls.get_base_encoding_flags()

    @classmethod
    def get_nvenc_encoding_flags(cls) -> List[str]:
        """Get NVIDIA NVENC encoding flags."""
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-tune", "hq",
            "-rc", "vbr_hq",
            "-cq", str(cls.CRF_QUALITY)
        ] + cls.get_base_encoding_flags()

    @classmethod
    def get_amd_encoding_flags(cls) -> List[str]:
        """Get AMD AMF encoding flags."""
        return [
            "-c:v", "h264_amf",
            "-quality", "balanced",
            "-qp_i", str(cls.CRF_QUALITY),  # I-frame quality
            "-qp_p", str(cls.CRF_QUALITY)   # P-frame quality
        ] + cls.get_base_encoding_flags()

    @classmethod
    def get_intel_qsv_encoding_flags(cls) -> List[str]:
        """Get Intel QSV encoding flags."""
        return [
            "-c:v", "h264_qsv",
            "-preset", "veryfast",
            "-q", str(cls.CRF_QUALITY)
        ] + cls.get_base_encoding_flags()

# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS (use centralized settings)
# ============================================================================

def get_web_video_encoding_flags() -> List[str]:
    """
    LEGACY: Returns standardized video encoding flags for web compatibility.
    These settings ensure excellent quality and universal web browser support.
    DEPRECATED: Use VideoEncodingSettings.get_cpu_encoding_flags() instead.
    """
    return VideoEncodingSettings.get_cpu_encoding_flags()

def _get_video_codec_and_flags() -> List[str]:
    """
    Determines the best video codec and associated flags based on settings.
    Prioritizes GPU codecs if use_gpu_ffmpeg is true.
    Uses centralized VideoEncodingSettings for consistency.
    """
    if app_settings.get("use_gpu_ffmpeg", False):
        # Try NVIDIA (NVENC)
        try:
            # Check if nvenc is available by running a dummy command
            subprocess.run([_get_ffmpeg_exe_path(), "-hide_banner", "-encoders"], capture_output=True, check=True, text=True)
            if "h264_nvenc" in subprocess.run([_get_ffmpeg_exe_path(), "-encoders"], capture_output=True, text=True).stdout:
                print("Using NVIDIA (NVENC) for GPU acceleration.")
                return VideoEncodingSettings.get_nvenc_encoding_flags()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass # NVENC not found or ffmpeg not configured for it

        # Try AMD (AMF)
        try:
            if "h264_amf" in subprocess.run([_get_ffmpeg_exe_path(), "-encoders"], capture_output=True, text=True).stdout:
                print("Using AMD (AMF) for GPU acceleration.")
                return VideoEncodingSettings.get_amd_encoding_flags()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass # AMF not found

        # Try Intel (QSV)
        try:
            if "h264_qsv" in subprocess.run([_get_ffmpeg_exe_path(), "-encoders"], capture_output=True, text=True).stdout:
                print("Using Intel (QSV) for GPU acceleration.")
                return VideoEncodingSettings.get_intel_qsv_encoding_flags()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass # QSV not found

        print("No suitable GPU encoder found or configured. Falling back to CPU (libx264).")

    # Fallback to CPU (libx264) - use web-compatible settings
    return VideoEncodingSettings.get_cpu_encoding_flags()

def _run_ffmpeg_process(command: List[str]) -> Tuple[bool, str, str]:
    """
    Runs an FFmpeg command using subprocess.
    Returns: (success_bool, stdout_str, stderr_str)
    """
    try:
        print(f"Running FFmpeg command: {' '.join(command)}")
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            return True, stdout, stderr
        else:
            print(f"FFmpeg error (code {process.returncode}): {stderr.strip()}")
            return False, stdout, stderr
    except FileNotFoundError:
        msg = f"Error: FFmpeg executable not found at '{command[0]}'. Please check FFMPEG_PATH in settings."
        print(msg)
        return False, "", msg
    except Exception as e:
        msg = f"Error during FFmpeg execution: {e}"
        print(msg)
        return False, "", msg

def _get_temp_output_path(input_video_path: str, operation_suffix: str) -> str:
    """Generates a temporary output path for FFmpeg operations."""
    base_name = os.path.basename(input_video_path)
    name, ext = os.path.splitext(base_name)
    # Place temp files in a 'temp' subdirectory of the input video's directory
    # or a global temp directory if preferred. For this example, using video's dir.
    temp_dir = os.path.join(os.path.dirname(input_video_path), "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"{name}_{operation_suffix}{ext}")


def crop_video_from_overlay(
    current_video_path: str,
    overlay_x_norm: float, overlay_y_norm: float, # Normalized 0-1 relative to displayed video
    overlay_w_norm: float, overlay_h_norm: float, # Normalized 0-1 relative to displayed video
    displayed_video_w: int, displayed_video_h: int, # Actual pixel size of video as displayed
    video_orig_w: int, video_orig_h: int,
    player_content_w: int, player_content_h: int, # Dimensions of the container holding the video player
    overlay_angle_rad: float = 0.0,
) -> Tuple[bool, str, Optional[str]]:
    """
    Crops a video based on normalized overlay coordinates.
    The overlay coordinates are relative to the *displayed* video within the player.
    Handles scaling calculations to map displayed coordinates to original video coordinates
    and applies rotation when an overlay angle is provided.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    if not os.path.exists(current_video_path):
        return False, "Input video not found.", None

    # Calculate the scale factor and offsets of the displayed video within the player_content area
    # This logic mirrors how a video player might fit content (letterboxing/pillarboxing)
    video_aspect = video_orig_w / video_orig_h
    player_aspect = player_content_w / player_content_h

    actual_disp_w: float
    actual_disp_h: float
    offset_x_player: float = 0
    offset_y_player: float = 0

    if video_aspect > player_aspect: # Video is wider than player area (letterboxed if height matches)
        actual_disp_w = player_content_w
        actual_disp_h = actual_disp_w / video_aspect
        offset_y_player = (player_content_h - actual_disp_h) / 2
    else: # Video is taller than player area (pillarboxed if width matches)
        actual_disp_h = player_content_h
        actual_disp_w = actual_disp_h * video_aspect
        offset_x_player = (player_content_w - actual_disp_w) / 2
    
    # Convert normalized overlay coordinates (relative to player_content_w/h)
    # to pixel coordinates on the *actual displayed video*
    # The provided overlay_x/y/w/h are assumed to be relative to player_content_w/h
    # and the overlay is drawn on top of the (potentially letter/pillarboxed) video.

    # Crop coordinates in terms of the original video dimensions
    # First, map overlay coordinates from player_content space to displayed_video space
    # The overlay_x_norm, etc. are actually pixel values from the GestureDetector
    # which is sized to player_content_w/h.
    
    # Let's assume overlay_x_norm etc. are already pixel values from the GestureDetector (e.g., control.left)
    # These are overlay_x_px, overlay_y_px, overlay_w_px, overlay_h_px
    
    # Scale factor from original video to displayed video
    scale_to_displayed_w = actual_disp_w / video_orig_w
    scale_to_displayed_h = actual_disp_h / video_orig_h
    
    # Crop coordinates in original video pixels
    crop_orig_x = (overlay_x_norm - offset_x_player) / scale_to_displayed_w
    crop_orig_y = (overlay_y_norm - offset_y_player) / scale_to_displayed_h
    crop_orig_w = overlay_w_norm / scale_to_displayed_w
    crop_orig_h = overlay_h_norm / scale_to_displayed_h

    # Ensure crop dimensions are positive and within bounds
    crop_orig_x = max(0, crop_orig_x)
    crop_orig_y = max(0, crop_orig_y)
    crop_orig_w = min(crop_orig_w, video_orig_w - crop_orig_x)
    crop_orig_h = min(crop_orig_h, video_orig_h - crop_orig_y)

    # Ensure width and height are even and at least 2x2
    target_crop_w = math.floor(crop_orig_w / 2) * 2
    target_crop_h = math.floor(crop_orig_h / 2) * 2
    target_crop_x = math.floor(crop_orig_x / 2) * 2
    target_crop_y = math.floor(crop_orig_y / 2) * 2

    if target_crop_w < 2 or target_crop_h < 2:
        return False, f"Crop dimensions too small ({target_crop_w}x{target_crop_h}). Min 2x2.", None

    temp_output_path = _get_temp_output_path(current_video_path, "cropped_overlay")
    
    angle = overlay_angle_rad or 0.0
    use_rotation = abs(angle) > 1e-4

    vf_filters: List[str] = []
    # Pre-scaling if original video is smaller than target crop (unlikely with overlay but for consistency)
    # This part is more relevant for crop_video_to_dimensions. For overlay, we crop what's selected.
    # However, if the selected area *implies* an upscale, FFmpeg handles it with 'crop'.

    if use_rotation:
        center_x = target_crop_x + (target_crop_w / 2.0)
        center_y = target_crop_y + (target_crop_h / 2.0)
        diag = math.hypot(target_crop_w, target_crop_h)
        pre_side = max(target_crop_w, target_crop_h, math.ceil(diag))
        if pre_side % 2:
            pre_side += 1
        half_pre = pre_side / 2.0

        pad_left = max(0, math.ceil(half_pre - center_x))
        pad_right = max(0, math.ceil(center_x + half_pre - video_orig_w))
        pad_top = max(0, math.ceil(half_pre - center_y))
        pad_bottom = max(0, math.ceil(center_y + half_pre - video_orig_h))

        if pad_left or pad_right or pad_top or pad_bottom:
            pad_w = video_orig_w + pad_left + pad_right
            pad_h = video_orig_h + pad_top + pad_bottom
            vf_filters.append(f"pad={pad_w}:{pad_h}:{pad_left}:{pad_top}:color=black")
            center_x += pad_left
            center_y += pad_top
            src_w = pad_w
            src_h = pad_h
        else:
            src_w = video_orig_w
            src_h = video_orig_h

        crop_x = center_x - half_pre
        crop_y = center_y - half_pre
        crop_x = max(0.0, min(crop_x, src_w - pre_side))
        crop_y = max(0.0, min(crop_y, src_h - pre_side))

        vf_filters.append(
            f"crop={pre_side}:{pre_side}:{int(crop_x)}:{int(crop_y)}"
        )

        vf_filters.append(
            f"rotate={-angle:.10f}:ow='rotw(iw)':oh='roth(ih)':fillcolor=black"
        )

        vf_filters.append(
            f"crop={target_crop_w}:{target_crop_h}:(iw-{target_crop_w})/2:(ih-{target_crop_h})/2"
        )
    else:
        vf_filters.append(f"crop={target_crop_w}:{target_crop_h}:{target_crop_x}:{target_crop_y}")

    vf_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2") # Ensure final dimensions are even

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", ",".join(vf_filters),
        *_get_video_codec_and_flags(), # Use dynamic codec and flags
        "-c:a", "aac", "-b:a", "128k", # Or copy audio: "-c:a", "copy"
        temp_output_path
    ]

    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video cropped successfully from overlay selection!", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path) # Clean up
        return False, f"FFmpeg error during overlay crop: {stderr.strip()}", None


def crop_video_to_dimensions(
    current_video_path: str, 
    target_width: int, 
    target_height: int
) -> Tuple[bool, str, Optional[str]]:
    """
    Crops and scales a video to target_width and target_height.
    Scales video so smallest side matches target, then center crops.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    metadata = get_video_metadata(current_video_path)
    if not metadata or metadata['width'] <= 0 or metadata['height'] <= 0:
        return False, "Could not get valid original video dimensions.", None

    orig_width, orig_height = metadata['width'], metadata['height']
    
    # Scaling logic: scale so that the video covers the target area, then crop.
    scale_w_factor = target_width / orig_width
    scale_h_factor = target_height / orig_height
    scale_factor = max(scale_w_factor, scale_h_factor) # Ensure video covers target area

    scaled_width = math.ceil(orig_width * scale_factor / 2) * 2 # Ensure even
    scaled_height = math.ceil(orig_height * scale_factor / 2) * 2 # Ensure even

    temp_output_path = _get_temp_output_path(current_video_path, "cropped_dim")
    
    filters = [
        f"scale={scaled_width}:{scaled_height}:flags=lanczos",
        f"crop={target_width}:{target_height}", # Center crop from scaled video
        "pad=ceil(iw/2)*2:ceil(ih/2)*2" # Ensure final dimensions are even
    ]
    scale_crop_filter = ','.join(filters)

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", scale_crop_filter,
        *_get_video_codec_and_flags(), # Use dynamic codec and flags
        "-c:a", "aac", "-b:a", "128k", # Or copy: "-c:a", "copy"
        temp_output_path
    ]

    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video cropped successfully to dimensions!", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during dimension crop: {stderr.strip()}", None


def flip_video_horizontal(current_video_path: str) -> Tuple[bool, str, Optional[str]]:
    """Flips video horizontally. Returns (success, message, output_path_or_none)."""
    ffmpeg_exe = _get_ffmpeg_exe_path()
    temp_output_path = _get_temp_output_path(current_video_path, "flipped")
    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", "hflip",
        *_get_video_codec_and_flags(), # Use dynamic codec and flags
        "-c:a", "copy", # Copy audio stream
        temp_output_path
    ]
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video flipped horizontally.", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during flip: {stderr.strip()}", None

def rotate_video_90(current_video_path: str, direction: str) -> Tuple[bool, str, Optional[str]]:
    """
    Rotates video by 90 degrees.
    direction: 'plus' for 90 degrees clockwise, 'minus' for 90 degrees counter-clockwise.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    temp_output_path = _get_temp_output_path(current_video_path, f"rotated_{direction}")
    
    transpose_value = ""
    message = ""
    if direction == 'plus':
        transpose_value = "transpose=1" # Rotate 90 degrees clockwise
        message = "Video rotated 90 degrees clockwise."
    elif direction == 'minus':
        transpose_value = "transpose=2" # Rotate 90 degrees counter-clockwise
        message = "Video rotated 90 degrees counter-clockwise."
    else:
        return False, "Invalid rotation direction. Use 'plus' or 'minus'.", None

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", transpose_value,
        *_get_video_codec_and_flags(), # Use dynamic codec and flags
        "-c:a", "copy", # Copy audio stream
        temp_output_path
    ]
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, message, temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during rotation: {stderr.strip()}", None

def reverse_video(current_video_path: str) -> Tuple[bool, str, Optional[str]]:
    """Reverses video. Returns (success, message, output_path_or_none)."""
    ffmpeg_exe = _get_ffmpeg_exe_path()
    temp_output_path = _get_temp_output_path(current_video_path, "reversed")
    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", "reverse", 
        # Reversing audio can be complex and sometimes undesirable.
        # "-af", "areverse", # Optionally reverse audio
        *_get_video_codec_and_flags(), # Use dynamic codec and flags
        "-c:a", "copy", # Typically copy audio or omit if audio reversal is not needed / problematic
        temp_output_path
    ]
    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video reversed.", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during reverse: {stderr.strip()}", None

def time_remap_video_by_speed(current_video_path: str, speed_multiplier: float) -> Tuple[bool, str, Optional[str]]:
    """
    Remaps video timing by a speed multiplier (e.g., 0.5 for half speed, 2.0 for double speed).
    This is a proper time remap that changes the speed of the video, not just its length.
    Returns (success, message, output_path_or_none).
    """
    if speed_multiplier <= 0:
        return False, "Speed multiplier must be positive.", None

    # If speed is effectively 1.0, do nothing.
    if math.isclose(speed_multiplier, 1.0):
        return True, "Speed is 1.0, no remapping needed.", current_video_path

    ffmpeg_exe = _get_ffmpeg_exe_path()

    # Video filter: setpts changes presentation timestamps to speed up/slow down.
    # A factor < 1.0 speeds up, > 1.0 slows down.
    pts_factor = 1.0 / speed_multiplier
    video_filter = f"setpts={pts_factor:.4f}*PTS"

    # Audio filter: atempo changes audio speed. Valid range is [0.5, 100.0].
    # We need to chain filters if the desired speed is outside this range.
    speed = speed_multiplier
    atempo_filters = []
    # The UI slider is limited to 0.1-2.0, but we write robust code for wider range.
    while speed > 100.0:
        atempo_filters.append("atempo=100.0")
        speed /= 100.0
    # Chaining for slowdown must be done carefully.
    temp_speed = speed
    while temp_speed < 0.5:
        atempo_filters.append("atempo=0.5")
        temp_speed /= 0.5
    atempo_filters.append(f"atempo={temp_speed:.4f}")

    temp_output_path = _get_temp_output_path(current_video_path, "remapped")

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", video_filter,
    ]

    # Add audio filter only if it was generated and is not trivial
    if atempo_filters and not math.isclose(speed_multiplier, 1.0):
        audio_filter_str = ",".join(atempo_filters)
        command.extend(["-af", audio_filter_str])
        # Re-encoding audio is necessary when changing speed
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        # If no audio filter, just copy the audio stream
        command.extend(["-c:a", "copy"])

    # Add the rest of the command. Note: -r is removed as it can conflict with setpts.
    command.extend([
        *_get_video_codec_and_flags(),
        temp_output_path
    ])

    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, f"Video time remapped by factor {speed_multiplier}.", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False, f"FFmpeg error during time remap: {stderr.strip()}", None

def cut_video_by_frames(
    current_video_path: str,
    start_frame: int,
    end_frame: int,
    force_reencode: bool = False
) -> Tuple[bool, str, Optional[str]]:
    """
    Cuts video from start_frame to end_frame using frame-accurate cutting.
    Always uses re-encoding for precise frame cutting accuracy.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    metadata = get_video_metadata(current_video_path)
    if not metadata or not metadata.get('fps') or metadata['fps'] <= 0:
        return False, "Could not get valid FPS for cutting by frames.", None

    fps = metadata['fps']
    total_frames = metadata.get('frames', 0)

    # Validate frame range
    if start_frame > end_frame:
        return False, "Start frame must be less than or equal to end frame.", None

    # Ensure end_frame doesn't exceed video length
    if total_frames > 0 and end_frame >= total_frames:
        end_frame = total_frames - 1
        print(f"Adjusted end_frame to {end_frame} (video total frames: {total_frames})")

    # Calculate precise timing
    start_time = start_frame / fps
    # Calculate duration to include the end_frame (inclusive)
    duration = (end_frame - start_frame + 1) / fps
    frame_count = end_frame - start_frame + 1

    temp_output_path = _get_temp_output_path(current_video_path, "cut")

    # Use simpler time-based cutting with accurate duration
    command = [
        ffmpeg_exe, "-y",
        "-ss", str(start_time),   # Seek to start time
        "-i", current_video_path,
        "-t", str(duration),      # Exact duration
        "-c:a", "aac",           # Include audio with AAC encoding
        *get_web_video_encoding_flags(),  # Use standardized web-compatible encoding
        temp_output_path
    ]

    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        # Verify the output has correct frame count
        output_metadata = get_video_metadata(temp_output_path)
        output_frames = output_metadata.get('frames', 0) if output_metadata else 0

        if output_frames == frame_count or (output_frames > 0 and abs(output_frames - frame_count) <= 1):
            return True, f"Video cut from frame {start_frame} to {end_frame} ({frame_count} frames).", temp_output_path
        else:
            print(f"Frame count mismatch: expected {frame_count}, got {output_frames}")
            # Still return success if we got a reasonable number of frames
            if output_frames > 0:
                return True, f"Video cut from frame {start_frame} to {end_frame} ({output_frames} frames).", temp_output_path
            else:
                # If no frames, something went wrong
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                return False, f"Failed to extract frames correctly.", None
    else:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return False, f"FFmpeg error during cut: {stderr.strip()}", None


def split_video_by_frame(
    current_video_path: str,
    split_frame: int
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Splits a video into two parts at split_frame using frame-accurate cutting.
    The first part is from frame 0 to split_frame-1.
    The second part is from split_frame to the end.
    Returns (success, message, output_path_part1_or_none, output_path_part2_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    metadata = get_video_metadata(current_video_path)
    if not metadata or not metadata.get('fps') or metadata['fps'] <= 0 or not metadata.get('total_frames'):
        return False, "Could not get valid metadata for splitting.", None, None

    fps = metadata['fps']
    total_frames = metadata['total_frames']

    if split_frame <= 0 or split_frame >= total_frames:
        return False, "Split frame must be within the video's frame range (exclusive of 0 and total_frames).", None, None

    # Part 1: 0 to split_frame-1 (frame_count = split_frame)
    frame_count_1 = split_frame
    temp_output_path_1 = _get_temp_output_path(current_video_path, "split_part1")
    command1 = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", f"select='between(n,0,{frame_count_1-1})',setpts=PTS-STARTPTS",  # Frame-accurate selection
        "-vframes", str(frame_count_1),  # Ensure exact frame count
        "-an",                    # No audio (faster processing)
        *get_web_video_encoding_flags(),  # Use standardized web-compatible encoding
        temp_output_path_1
    ]
    success1, _, stderr1 = _run_ffmpeg_process(command1)
    if not (success1 and os.path.exists(temp_output_path_1)):
        if os.path.exists(temp_output_path_1): os.remove(temp_output_path_1)
        return False, f"FFmpeg error during split (part 1): {stderr1.strip()}", None, None

    # Part 2: split_frame to end (frame_count = total_frames - split_frame)
    frame_count_2 = total_frames - split_frame
    start_frame_2 = split_frame
    temp_output_path_2 = _get_temp_output_path(current_video_path, "split_part2")
    command2 = [
        ffmpeg_exe, "-y",
        "-ss", str(start_frame_2 / fps),  # Seek to start frame
        "-i", current_video_path,
        "-vf", f"select='between(n,0,{frame_count_2-1})',setpts=PTS-STARTPTS",  # Frame-accurate selection
        "-vframes", str(frame_count_2),  # Ensure exact frame count
        "-an",                    # No audio (faster processing)
        *get_web_video_encoding_flags(),  # Use standardized web-compatible encoding
        temp_output_path_2
    ]
    success2, _, stderr2 = _run_ffmpeg_process(command2)
    if not (success2 and os.path.exists(temp_output_path_2)):
        if os.path.exists(temp_output_path_1): os.remove(temp_output_path_1) # Clean up part 1
        if os.path.exists(temp_output_path_2): os.remove(temp_output_path_2)
        return False, f"FFmpeg error during split (part 2): {stderr2.strip()}", temp_output_path_1, None
        
    return True, f"Video split at frame {split_frame}.", temp_output_path_1, temp_output_path_2

def clean_video_from_overlay(
    current_video_path: str,
    overlay_x_norm: float, overlay_y_norm: float, # Pixel values from displayed video
    overlay_w_norm: float, overlay_h_norm: float, # Pixel values from displayed video
    displayed_video_w: int, displayed_video_h: int, # Actual pixel size of video as displayed
    video_orig_w: int, video_orig_h: int
) -> Tuple[bool, str, Optional[str]]:
    """
    "Cleans" a video by cropping based on the provided overlay coordinates,
    effectively removing black bars or unwanted padding.
    The overlay coordinates are relative to the *displayed* video within the player.
    Handles scaling calculations to map displayed coordinates to original video coordinates.
    Returns (success, message, output_path_or_none).
    """
    ffmpeg_exe = _get_ffmpeg_exe_path()
    if not os.path.exists(current_video_path):
        return False, "Input video not found for cleaning.", None

    # Calculate the scale factor and offsets of the displayed video within the player_content area
    video_aspect = video_orig_w / video_orig_h
    player_aspect = displayed_video_w / displayed_video_h # Use displayed_video_w/h as player_content_w/h for this context

    actual_disp_w: float
    actual_disp_h: float
    offset_x_player: float = 0
    offset_y_player: float = 0

    if video_aspect > player_aspect: # Video is wider than player area (letterboxed if height matches)
        actual_disp_w = displayed_video_w
        actual_disp_h = actual_disp_w / video_aspect
        offset_y_player = (displayed_video_h - actual_disp_h) / 2
    else: # Video is taller than player area (pillarboxed if width matches)
        actual_disp_h = displayed_video_h
        actual_disp_w = actual_disp_h * video_aspect
        offset_x_player = (displayed_video_w - actual_disp_w) / 2
    
    # Crop coordinates in terms of the original video dimensions
    # The overlay_x_norm, etc. are already pixel values from the GestureDetector
    
    # Scale factor from original video to displayed video
    scale_to_displayed_w = actual_disp_w / video_orig_w
    scale_to_displayed_h = actual_disp_h / video_orig_h
    
    # Crop coordinates in original video pixels
    crop_orig_x = (overlay_x_norm - offset_x_player) / scale_to_displayed_w
    crop_orig_y = (overlay_y_norm - offset_y_player) / scale_to_displayed_h
    crop_orig_w = overlay_w_norm / scale_to_displayed_w
    crop_orig_h = overlay_h_norm / scale_to_displayed_h

    # Ensure crop dimensions are positive and within bounds
    crop_orig_x = max(0, crop_orig_x)
    crop_orig_y = max(0, crop_orig_y)
    crop_orig_w = min(crop_orig_w, video_orig_w - crop_orig_x)
    crop_orig_h = min(crop_orig_h, video_orig_h - crop_orig_y)

    # Ensure width and height are divisible by 16 and at least 16x16 for encoder compatibility
    # Use _closest_divisible_by for all crop parameters
    target_crop_w = _closest_divisible_by(int(crop_orig_w), 16)
    target_crop_h = _closest_divisible_by(int(crop_orig_h), 16)
    target_crop_x = _closest_divisible_by(int(crop_orig_x), 16)
    target_crop_y = _closest_divisible_by(int(crop_orig_y), 16)

    if target_crop_w < 16 or target_crop_h < 16:
        return False, f"Clean crop dimensions too small ({target_crop_w}x{target_crop_h}). Min 16x16 for encoder.", None

    temp_output_path = _get_temp_output_path(current_video_path, "cleaned_overlay")
    
    vf_filters = []
    vf_filters.append(f"crop={target_crop_w}:{target_crop_h}:{target_crop_x}:{target_crop_y}")
    vf_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2") # Ensure final dimensions are even

    command = [
        ffmpeg_exe, "-y", "-i", current_video_path,
        "-vf", ",".join(vf_filters),
        *_get_video_codec_and_flags(), # Use dynamic codec and flags
        "-c:a", "aac", "-b:a", "128k", # Or copy audio: "-c:a", "copy"
        temp_output_path
    ]

    success, _, stderr = _run_ffmpeg_process(command)
    if success and os.path.exists(temp_output_path):
        return True, "Video cleaned (cropped) successfully from overlay selection!", temp_output_path
    else:
        if os.path.exists(temp_output_path): os.remove(temp_output_path) # Clean up
        return False, f"FFmpeg error during clean crop: {stderr.strip()}", None


def find_nearest_keyframe_before(video_path: str, target_time: float) -> Tuple[bool, float, str]:
    """
    Finds the nearest keyframe (I-frame) at or before the target time using ffprobe.
    This is much more reliable than parsing ffmpeg's stderr.
    """
    try:
        # ffprobe is generally in the same directory as ffmpeg
        ffmpeg_exe = _get_ffmpeg_exe_path()
        ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")

        # Command to get timestamps of all keyframes
        command = [
            ffprobe_exe,
            "-v", "error",
            "-skip_frame", "nokey",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_pts_time",
            "-of", "csv=p=0",
            video_path
        ]

        # Run ffprobe
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if
                                   os.name == 'nt' else 0)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return False, 0.0, f"ffprobe failed: {stderr.strip()}"

        # Parse the output to get a list of keyframe times
        # The output is a simple list of timestamps, one per line.
        keyframe_times = [float(t) for t in stdout.strip().split() if t]
        if not keyframe_times:
            # If no keyframes are found (e.g., for a single-image video), treat frame 0 as the only keyframe.
            return True, 0.0, ""

        # Find the best keyframe at or before the target time
        best_keyframe_time = 0.0
        # The list is already sorted, so we can iterate and find the last one <= target_time
        for kt in keyframe_times:
            if kt <= target_time:
                best_keyframe_time = kt
            else:
                break # Stop once we pass the target time

        return True, best_keyframe_time, ""

    except Exception as e:
        return False, 0.0, f"Error finding keyframe with ffprobe: {str(e)}"

