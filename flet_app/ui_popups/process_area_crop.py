from typing import Optional, Tuple

from . import image_player_utils as ipu
from . import video_player_utils as vpu


def process_area_crop(
    path: str,
    is_img: bool,
    overlay_left: int,
    overlay_top: int,
    overlay_width: int,
    overlay_height: int,
    viewer_width: int,
    viewer_height: int,
    overlay_angle: float = 0.0,
) -> Tuple[bool, str, Optional[str]]:
    """Run the crop operation for an image or video given the overlay selection.

    Overlay angles are expected in radians.
    Returns (success, message, temp_output_path_or_none)."""
    if is_img:
        metadata = ipu.get_image_metadata(path)
        if not metadata:
            return False, "Image metadata unavailable.", None

        displayed_w, displayed_h, _, _ = ipu.calculate_contained_image_dimensions(
            metadata["width"],
            metadata["height"],
            viewer_width,
            viewer_height,
        )

        return ipu.crop_image_from_overlay(
            current_image_path=path,
            overlay_x_norm=overlay_left,
            overlay_y_norm=overlay_top,
            overlay_w_norm=overlay_width,
            overlay_h_norm=overlay_height,
            displayed_image_w=displayed_w,
            displayed_image_h=displayed_h,
            image_orig_w=metadata["width"],
            image_orig_h=metadata["height"],
            player_content_w=viewer_width,
            player_content_h=viewer_height,
            overlay_angle_rad=overlay_angle,
        )

    metadata = vpu.get_video_metadata(path)
    if not metadata:
        return False, "Video metadata unavailable.", None

    return vpu.crop_video_from_overlay(
        current_video_path=path,
        overlay_x_norm=overlay_left,
        overlay_y_norm=overlay_top,
        overlay_w_norm=overlay_width,
        overlay_h_norm=overlay_height,
        displayed_video_w=viewer_width,
        displayed_video_h=viewer_height,
        video_orig_w=metadata["width"],
        video_orig_h=metadata["height"],
        player_content_w=viewer_width,
        player_content_h=viewer_height,
        overlay_angle_rad=overlay_angle,
    )
