# Image processing utilities for dataset samples

import traceback
from pathlib import Path
from loguru import logger
import flet as ft


def process_and_save_image(
    source_image_path: str,
    video_dims_tuple: tuple,
    dataset_name: str,
    target_filename: str,
    dataset_type: str = "video"
) -> str:
    base_datasets_dir = "datasets"
    dataset_sample_images_dir = Path("workspace") / base_datasets_dir / dataset_name / "sample_images"
    dataset_sample_images_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_sample_images_dir / target_filename

    try:
        from PIL import Image
        img = Image.open(source_image_path)
    except FileNotFoundError:
        logger.error(f"Source image not found: {source_image_path}")
        return None
    except Exception as e:
        logger.error(f"Error opening image {source_image_path}: {e}")
        return None

    original_width, original_height = img.size
    target_width, target_height = video_dims_tuple[0], video_dims_tuple[1]

    if original_width == 0 or original_height == 0 or target_width == 0 or target_height == 0:
        logger.warning(f"Invalid dimensions. Original: {img.size}, Target: {(target_width, target_height)}")
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(target_path)
        return str(target_path).replace('\\', '/')

    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_factor = max(width_ratio, height_ratio)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    if new_width <= 0 or new_height <= 0:
        logger.warning(f"Invalid new dimensions ({new_width}x{new_height})")
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(target_path)
        return str(target_path).replace('\\', '/')

    try:
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        img_cropped = img_resized.crop((left, top, right, bottom))

        if img_cropped.mode == 'RGBA':
            img_cropped = img_cropped.convert('RGB')

        img_cropped.save(target_path)
        result_path = str(target_path).replace('\\', '/')
        logger.info(f"Image saved and scaled to: {result_path}")
        return result_path
    except Exception as e:
        logger.error(f"Error resizing/saving image: {e}")
        logger.error(traceback.format_exc())
        return None


def save_and_scale_image(
    source_image_path: str,
    video_dims_tuple: tuple,
    dataset_name: str,
    target_filename: str,
    dataset_type: str = "video",
    page: ft.Page = None,
    target_control: str = None,
    image_display_c1=None,
    image_display_c2=None
) -> str:
    if not source_image_path or not dataset_name:
        logger.warning("save_and_scale_image: Missing source_image_path or dataset_name")
        return None

    result_path = process_and_save_image(
        source_image_path, video_dims_tuple, dataset_name, target_filename, dataset_type
    )

    if result_path and page is not None and target_control and target_control.lower() in ['c1', 'c2']:
        from .utils_top_menu import TopBarUtils
        logger.debug(f"Loading cropped image into UI for {target_control}")
        TopBarUtils._load_cropped_image_into_ui(
            page=page,
            image_path=result_path,
            target_control_key=target_control,
            image_display_c1=image_display_c1,
            image_display_c2=image_display_c2
        )

    return result_path
