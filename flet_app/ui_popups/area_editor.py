# area_editor.py
"""
Utility helpers for the “Area Editor” UI.

This file contains high‑level actions:
     • `toggle_area_editor`
     • `apply_crop_from_overlay`
     • `apply_clean_from_overlay`

All functions are pure side‑effect helpers (they only touch the controls you pass in
and use `page.snack_bar` for user feedback).  
Import them into your dialog like this:

    from .area_editor import (
        toggle_area_editor,
        apply_crop_from_overlay,
        apply_clean_from_overlay,
    )
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import flet as ft

def toggle_area_editor(
    page: ft.Page,
    overlay_control: ft.GestureDetector | None,
    overlay_visible: bool,
    overlay_angle: float,
) -> tuple[bool, float]:
    """
    Toggle the Area Editor overlay.

    Returns a new `(visible, angle)` pair that callers should store.
    """
    if not overlay_control:
        return overlay_visible, overlay_angle

    overlay_visible = not overlay_visible
    overlay_control.visible = overlay_visible

    # Keep visual size in sync when showing
    if overlay_visible:
        try:
            w = float(overlay_control.width or 0)
            h = float(overlay_control.height or 0)
            if w < 20 or h < 20:      # sanity check
                w, h = 200, 200
            overlay_control.width = w
            overlay_control.height = h
        except Exception:
            pass

    # Reset rotation when hidden
    if not overlay_visible and hasattr(overlay_control, "rotate"):
        try:
            overlay_control.rotate = None
        except Exception:
            pass

    page.update()
    return overlay_visible, overlay_angle


def apply_crop_from_overlay(
    page: ft.Page,
    media_path: str,
    overlay_control: ft.GestureDetector | None,
    overlay_visible: bool,
    viewer_w: int,
    viewer_h: int,
    overlay_angle: float = 0.0,
) -> bool:
    """
    Crop the image/video shown in `media_path` to the rectangle defined by
    `overlay_control`.

    The heavy lifting is done by the dedicated editor modules.

    Returns
    -------
    bool
        ``True`` if the crop succeeded and the media was replaced.
    """
    if not overlay_control or not overlay_visible:
        page.snack_bar = ft.SnackBar(ft.Text("Open Area Editor first."), open=True)
        page.update()
        return False

    left = int(overlay_control.left or 0)
    top = int(overlay_control.top or 0)
    w = int(overlay_control.width or 0)
    h = int(overlay_control.height or 0)

    if w <= 0 or h <= 0:
        page.snack_bar = ft.SnackBar(ft.Text("Invalid overlay dimensions."), open=True)
        page.update()
        return False

    # Import lazily to avoid circular imports
    from .unified_media_utils import is_image_path
    from .image_player_utils import (
        get_image_metadata,
        calculate_contained_image_dimensions,
        crop_image_from_overlay,
    )
    from .video_player_utils import get_video_metadata, crop_video_from_overlay

    if is_image_path(media_path):
        md = get_image_metadata(media_path)
        if not md:
            page.snack_bar = ft.SnackBar(ft.Text("Image metadata unavailable."), open=True)
            page.update()
            return False
        eff_w, eff_h, _, _ = calculate_contained_image_dimensions(
            md["width"], md["height"], viewer_w, viewer_h
        )
        success, msg, temp_out = crop_image_from_overlay(
            current_image_path=media_path,
            overlay_x_norm=left,
            overlay_y_norm=top,
            overlay_w_norm=w,
            overlay_h_norm=h,
            displayed_image_w=eff_w,
            displayed_image_h=eff_h,
            image_orig_w=md["width"],
            image_orig_h=md["height"],
            player_content_w=viewer_w,
            player_content_h=viewer_h,
        )
    else:
        md = get_video_metadata(media_path)
        if not md:
            page.snack_bar = ft.SnackBar(ft.Text("Video metadata unavailable."), open=True)
            page.update()
            return False
        success, msg, temp_out = crop_video_from_overlay(
            current_video_path=media_path,
            overlay_x_norm=left,
            overlay_y_norm=top,
            overlay_w_norm=w,
            overlay_h_norm=h,
            displayed_video_w=viewer_w,
            displayed_video_h=viewer_h,
            video_orig_w=md["width"],
            video_orig_h=md["height"],
            player_content_w=viewer_w,
            player_content_h=viewer_h,
            overlay_angle_rad=overlay_angle,
        )

    result = False
    if success and temp_out and os.path.exists(temp_out):
        try:
            os.replace(temp_out, media_path)
            page.snack_bar = ft.SnackBar(ft.Text(msg or "Cropped from area."), open=True)
            page.update()
            # Caller should call `refresh()` to reload the viewer
            result = True
        except Exception as exc:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error finalising crop: {exc}"), open=True)
            if os.path.exists(temp_out):
                try:
                    os.remove(temp_out)
                except Exception:
                    pass
    else:
        page.snack_bar = ft.SnackBar(ft.Text(msg or "Crop failed."), open=True)

    page.update()
    return result


def apply_clean_from_overlay(
    page: ft.Page,
    media_path: str,
    overlay_control: ft.GestureDetector | None,
    overlay_visible: bool,
    viewer_w: int,
    viewer_h: int,
) -> bool:
    """
    For videos only – remove everything outside the selected rectangle.

    The logic mirrors `apply_crop_from_overlay` but calls the “clean” helper.

    Returns
    -------
    bool
        ``True`` if the clean succeeded and the media was replaced.
    """
    if not overlay_control or not overlay_visible:
        page.snack_bar = ft.SnackBar(ft.Text("Open Area Editor first."), open=True)
        page.update()
        return False

    left = int(overlay_control.left or 0)
    top = int(overlay_control.top or 0)
    w = int(overlay_control.width or 0)
    h = int(overlay_control.height or 0)

    if w <= 0 or h <= 0:
        page.snack_bar = ft.SnackBar(ft.Text("Invalid overlay dimensions."), open=True)
        page.update()
        return False

    from .video_player_utils import get_video_metadata, clean_video_from_overlay

    md = get_video_metadata(media_path)
    if not md:
        page.snack_bar = ft.SnackBar(ft.Text("Video metadata unavailable."), open=True)
        page.update()
        return False

    success, msg, temp_out = clean_video_from_overlay(
        current_video_path=media_path,
        overlay_x_norm=left,
        overlay_y_norm=top,
        overlay_w_norm=w,
        overlay_h_norm=h,
        displayed_video_w=viewer_w,
        displayed_video_h=viewer_h,
        video_orig_w=md["width"],
        video_orig_h=md["height"],
    )

    result = False
    if success and temp_out and os.path.exists(temp_out):
        try:
            os.replace(temp_out, media_path)
            page.snack_bar = ft.SnackBar(ft.Text(msg or "Clean applied."), open=True)
            page.update()
            # Caller should call `refresh()` to reload the viewer
            result = True
        except Exception as exc:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error finalising clean: {exc}"), open=True)
            if os.path.exists(temp_out):
                try:
                    os.remove(temp_out)
                except Exception:
                    pass
    else:
        page.snack_bar = ft.SnackBar(ft.Text(msg or "Clean failed."), open=True)

    page.update()
    return result


__all__ = [
    "toggle_area_editor",
    "apply_crop_from_overlay",
    "apply_clean_from_overlay",
]
