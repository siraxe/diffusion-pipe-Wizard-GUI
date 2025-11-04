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
        return video_path
    normalized_path = video_path.replace("\\", "/")
    if _is_web_platform(page) and os.path.exists(video_path):
        try:
            mime_type, _ = mimetypes.guess_type(video_path)
            mime_type = mime_type or "video/mp4"
            with open(video_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
        except Exception:
            pass
    return normalized_path


def make_video_control(path: str, width: int, height: int, page: Optional[ft.Page] = None):
    """Create a simple video control. Uses filesystem path; no extra controls."""
    try:
        from flet_video.video import Video, VideoMedia
    except Exception:
        # Fallback: show a message if video control cannot be imported
        return ft.Container(
            content=ft.Text("Video module not available"),
            alignment=ft.alignment.center,
            width=width,
            height=height,
        )
    resource = _build_video_media_resource(page, path)
    # Debug print removed
    # On web, enforce muted playback to avoid autoplay restrictions
    volume = 0.0 if _is_web_platform(page) else (100.0 if settings.get("enable_audio", False) else 0.0)
    # Prefer built-in loop mode to ensure reliable looping on web
    return Video(
        playlist=[VideoMedia(resource=resource)],
        autoplay=True,
        width=width,
        height=height,
        expand=False,
        show_controls=False,
        playlist_mode="loop",
        volume=volume,
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
