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
    padding: int = 80,
) -> bool:
    """
    For videos only – remove objects using the minimax-remover AI model.

    Creates a mask from the selected rectangle and runs the minimax-remover
    script to remove objects from the video.

    Parameters
    ----------
    padding : int, default=80
        Padding in pixels to expand around the selected area for processing.
        The expanded region gives the AI context from surrounding pixels.

    Returns
    -------
    bool
        ``True`` if the clean was initiated successfully.
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

    # Import the AI-based clean handler from video_editor
    from .video_editor import on_clean_action_handler

    # Call the minimax-remover based clean handler
    # overlay_coords should be (left, top, width, height)
    overlay_coords = (left, top, w, h)

    # This function runs asynchronously and handles UI updates
    on_clean_action_handler(
        page=page,
        current_video_path=media_path,
        overlay_coords=overlay_coords,
        padding=padding,
        video_list=None,  # Not needed for single video processing
        on_caption_updated_callback=None,  # Not needed for this context
    )

    # Return True - the actual processing happens asynchronously
    return True


def create_mask_from_overlay(
    page: ft.Page,
    media_path: str,
    overlay_control: ft.GestureDetector | None,
    overlay_visible: bool,
    viewer_w: int,
    viewer_h: int,
) -> bool:
    """
    Create a mask from the selected area.

    Creates a black PNG image with a white rectangle in the selected area.
    Saves it to a 'mask/' subdirectory next to the media file with the same base name.

    Returns
    -------
    bool
        ``True`` if the mask was created successfully.
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

    # Import PIL for image creation
    try:
        from PIL import Image
    except ImportError:
        page.snack_bar = ft.SnackBar(ft.Text("PIL not available."), open=True)
        page.update()
        return False

    # Import for metadata
    from .unified_media_utils import is_image_path
    from .image_player_utils import get_image_metadata, calculate_contained_image_dimensions
    from .video_player_utils import get_video_metadata

    # Get media dimensions
    if is_image_path(media_path):
        md = get_image_metadata(media_path)
        if not md:
            page.snack_bar = ft.SnackBar(ft.Text("Image metadata unavailable."), open=True)
            page.update()
            return False
        eff_w, eff_h, _, _ = calculate_contained_image_dimensions(
            md["width"], md["height"], viewer_w, viewer_h
        )
        media_orig_w = md["width"]
        media_orig_h = md["height"]
    else:
        md = get_video_metadata(media_path)
        if not md:
            page.snack_bar = ft.SnackBar(ft.Text("Video metadata unavailable."), open=True)
            page.update()
            return False
        # For video, calculate actual displayed dimensions (accounting for aspect ratio)
        video_aspect = md["width"] / md["height"]
        viewer_aspect = viewer_w / viewer_h

        if video_aspect > viewer_aspect:
            # Video is wider - fits to width
            eff_w = viewer_w
            eff_h = int(viewer_w / video_aspect)
        else:
            # Video is taller - fits to height
            eff_h = viewer_h
            eff_w = int(viewer_h * video_aspect)

        media_orig_w = md["width"]
        media_orig_h = md["height"]

    # Calculate padding (centering offset)
    pad_x = (viewer_w - eff_w) / 2
    pad_y = (viewer_h - eff_h) / 2

    # Overlay coordinates are relative to container, so subtract padding to get media-relative coords
    overlay_x_relative = left - pad_x
    overlay_y_relative = top - pad_y

    # Calculate scale from displayed media to original media
    scale = media_orig_w / eff_w  # Assuming uniform scaling

    # Calculate actual rectangle in original media space
    actual_left = int(overlay_x_relative * scale)
    actual_top = int(overlay_y_relative * scale)
    actual_w = int(w * scale)
    actual_h = int(h * scale)

    # Clamp to image bounds
    actual_left = max(0, min(actual_left, media_orig_w - 1))
    actual_top = max(0, min(actual_top, media_orig_h - 1))
    actual_w = max(1, min(actual_w, media_orig_w - actual_left))
    actual_h = max(1, min(actual_h, media_orig_h - actual_top))

    # Create mask: black background, white rectangle in selected area
    mask = Image.new('L', (media_orig_w, media_orig_h), 0)  # 0 = black
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([actual_left, actual_top, actual_left + actual_w, actual_top + actual_h], fill=255)  # 255 = white

    # Determine mask file path
    media_dir = os.path.dirname(media_path)
    media_filename = os.path.basename(media_path)
    media_name, _ = os.path.splitext(media_filename)

    mask_dir = os.path.join(media_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)

    mask_path = os.path.join(mask_dir, f"{media_name}.png")

    # Save mask
    try:
        mask.save(mask_path)
        page.snack_bar = ft.SnackBar(ft.Text(f"Mask saved: {mask_path}"), open=True)
        page.update()
        return True
    except Exception as exc:
        page.snack_bar = ft.SnackBar(ft.Text(f"Error saving mask: {exc}"), open=True)
        page.update()
        return False


__all__ = [
    "toggle_area_editor",
    "apply_crop_from_overlay",
    "apply_clean_from_overlay",
    "create_mask_from_overlay",
]
