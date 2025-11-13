import base64
import mimetypes
import os
from typing import Optional, Tuple

import flet as ft
import cv2

from flet_app.settings import settings


def is_image_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    img_exts = [str(e).lower().lstrip('.') for e in getattr(settings, "IMAGE_EXTENSIONS", [])]
    result = ext in img_exts
    # Debug print removed
    return result


def make_image_control(path: str, width: int, height: int, page: Optional[ft.Page] = None) -> ft.Image:
    """Create an ft.Image for the given file. Use filesystem path on desktop; base64 on web."""
    src = path.replace("\\", "/")
    encoded: Optional[str] = None
    if _is_web_platform(page):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                src = None
        except Exception:
            encoded = None

    return ft.Image(
        src=src,
        src_base64=encoded,
        width=width,
        height=height,
        fit=ft.ImageFit.CONTAIN,
    )


def _is_web_platform(page: Optional[ft.Page]) -> bool:
    if page is None:
        return False
    platform = getattr(page, "platform", None)
    if isinstance(platform, str) and platform.lower() == "web":
        return True
    page_platform = getattr(ft, "PagePlatform", None)
    if page_platform is not None:
        try:
            if platform == page_platform.WEB:
                return True
        except AttributeError:
            pass
    return bool(getattr(page, "web", False))


def _build_video_media_resource(page: Optional[ft.Page], video_path: str) -> str:
    if not video_path:
        return ""

    # Normalize path separators for consistency
    normalized_path = video_path.replace("\\", "/")

    # Find the 'workspace' directory in the path
    try:
        # Split the path at the 'workspace' directory and take the part after it
        # This handles cases like '.../Dpipe/workspace/videos/file.mp4'
        # It will result in 'videos/file.mp4'
        relative_path = normalized_path.split("workspace/", 1)[1]

        # Add cache-busting timestamp to prevent browser from serving cached versions
        # after video operations like cutting/trimming
        import time
        timestamp = int(time.time())

        # Append timestamp as query parameter to force browser to reload
        if "?" in relative_path:
            return f"{relative_path}&_t={timestamp}"
        else:
            return f"{relative_path}?_t={timestamp}"

    except IndexError:
        # If 'workspace/' is not in the path, it's not a valid asset, return original path
        # The video player will likely fail, which is the correct behavior
        return video_path


def make_video_control(path: str, width: int, height: int, page: Optional[ft.Page] = None):
    """Create a simple video control. Uses filesystem path; no extra controls."""
    
    try:
        import flet_video as ftv
    except ImportError as e:
        # Fallback: show a message if video control cannot be imported
        return ft.Container(
            content=ft.Text("flet_video module not available"),
            alignment=ft.alignment.center,
            width=width,
            height=height,
        )
    
    # Use the same approach as main.py: process the path to work with asset serving
    resource = _build_video_media_resource(page, path)
    try:
        # Create video media using the same pattern as main.py (direct path argument)
        video_media = ftv.VideoMedia(resource)
    except Exception as e:
        return ft.Container(
            content=ft.Text(f"Failed to create video: {e}"),
            alignment=ft.alignment.center,
            width=width,
            height=height,
        )
    
    try:
        # Use the same settings as main.py - important: volume=100, autoplay=False
        video_player = ftv.Video(
            playlist=[video_media],
            aspect_ratio=16 / 9,
            playlist_mode=ftv.PlaylistMode.SINGLE,
            autoplay=True,  # Autoplay enabled by user request
            volume=0,       # Muted by default by user request
            width=width,
            height=height,
            expand=False,
            show_controls=False,
            fill_color=ft.Colors.BLACK,
        )
        return video_player
    except Exception as e:
        return ft.Container(
            content=ft.Text(f"Failed to create video player: {e}"),
            alignment=ft.alignment.center,
            width=width,
            height=height,
        )


def make_video_preview_control(path: str, width: int, height: int) -> ft.Image:
    """Create a static preview image from the first frame of a video.
    Falls back to a placeholder if frame extraction fails.
    """
    encoded: Optional[str] = None
    try:
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            # Keep frame in BGR for OpenCV imencode to preserve correct colors
            retval, buf = cv2.imencode('.jpg', frame)
            if retval:
                encoded = base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        encoded = None

    if encoded:
        return ft.Image(src=None, src_base64=encoded, width=width, height=height, fit=ft.ImageFit.CONTAIN)
    else:
        return ft.Image(src=None, src_base64=None, width=width, height=height, fit=ft.ImageFit.CONTAIN)


