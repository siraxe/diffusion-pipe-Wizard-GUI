# image_player_utils.py
import os
import json
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union

# Control Image Helper Functions
def get_original_image_path(image_path: str) -> str:
    """
    If the image is in a control folder, return the path to the original image.
    Otherwise, return the path as-is.
    """
    if "/control/" in image_path:
        # Extract the original path from control image path
        return image_path.replace("/control/", "/").replace("//", "/")
    return image_path

def get_control_image_path(image_path: str) -> Optional[str]:
    """
    Get the path to the control image for a given original image.
    Returns None if the image is already a control image or if no control exists.
    """
    if "/control/" in image_path:
        return None  # Already a control image

    image_dir = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)
    control_folder = os.path.join(image_dir, "control")
    control_image_path = os.path.join(control_folder, image_filename)

    if os.path.exists(control_image_path):
        return control_image_path
    return None

def images_have_same_dimensions(image_path1: str, image_path2: str) -> bool:
    """
    Check if two images have the same dimensions.
    Returns False if either image doesn't exist or metadata cannot be retrieved.
    """
    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        return False

    md1 = get_image_metadata(image_path1)
    md2 = get_image_metadata(image_path2)

    if not md1 or not md2:
        return False

    return md1['width'] == md2['width'] and md1['height'] == md2['height']

def get_related_image_info_for_crop(image_path: str) -> Optional[dict]:
    """
    Get information about a related image that should also be cropped.
    Returns a dict with:
        - 'path': the related image path
        - 'needs_scaling': True if dimensions are close but not exact (within 10px)
        - 'target_width': width to scale to (if needs_scaling)
        - 'target_height': height to scale to (if needs_scaling)
    Returns None if no related image exists or dimensions are too different.
    """
    is_control = "/control/" in image_path
    DIM_THRESHOLD = 10  # pixels

    if is_control:
        # Get the original image path
        original_path = get_original_image_path(image_path)
        if not os.path.exists(original_path):
            return None

        md_current = get_image_metadata(image_path)
        md_related = get_image_metadata(original_path)
        if not md_current or not md_related:
            return None

        # Check if dimensions match exactly
        if (md_current['width'] == md_related['width'] and
            md_current['height'] == md_related['height']):
            return {
                'path': original_path,
                'needs_scaling': False
            }

        # Check if dimensions are close (within threshold)
        width_diff = abs(md_current['width'] - md_related['width'])
        height_diff = abs(md_current['height'] - md_related['height'])
        if width_diff <= DIM_THRESHOLD and height_diff <= DIM_THRESHOLD:
            return {
                'path': original_path,
                'needs_scaling': True,
                'target_width': md_current['width'],
                'target_height': md_current['height']
            }
    else:
        # Get the control image path
        control_path = get_control_image_path(image_path)
        if not control_path:
            return None

        md_current = get_image_metadata(image_path)
        md_related = get_image_metadata(control_path)
        if not md_current or not md_related:
            return None

        # Check if dimensions match exactly
        if (md_current['width'] == md_related['width'] and
            md_current['height'] == md_related['height']):
            return {
                'path': control_path,
                'needs_scaling': False
            }

        # Check if dimensions are close (within threshold)
        width_diff = abs(md_current['width'] - md_related['width'])
        height_diff = abs(md_current['height'] - md_related['height'])
        if width_diff <= DIM_THRESHOLD and height_diff <= DIM_THRESHOLD:
            return {
                'path': control_path,
                'needs_scaling': True,
                'target_width': md_current['width'],
                'target_height': md_current['height']
            }

    return None

# Backwards compatibility - old function name still works
def get_related_image_path_for_crop(image_path: str) -> Optional[str]:
    """Legacy function - use get_related_image_info_for_crop instead."""
    info = get_related_image_info_for_crop(image_path)
    return info['path'] if info else None

# Caption and Data Handling
def load_caption_for_image(image_path: str) -> Tuple[str, str, Optional[str]]:
    """
    Load caption and negative caption for a given image from their corresponding .txt files.
    Returns: (caption, negative_caption, message_string_or_none)
    """
    image_dir = os.path.dirname(image_path)
    base_filename, _ = os.path.splitext(os.path.basename(image_path))
    caption_txt_path = os.path.join(image_dir, f"{base_filename}.txt")
    neg_caption_txt_path = os.path.join(image_dir, f"{base_filename}_neg.txt")
    no_caption_text = "No captions found, add it here"
    
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

def save_caption_for_image(image_path: str, new_caption: str, field_name: str = "caption") -> Tuple[bool, str]:
    """
    Save the new caption for the given image to its corresponding .txt file.
    If field_name is 'negative_caption', saves to a _neg.txt file.
    Returns: (success_bool, message_string)
    """
    image_dir = os.path.dirname(image_path)
    base_filename, _ = os.path.splitext(os.path.basename(image_path))
    
    if field_name == "caption":
        caption_txt_path = os.path.join(image_dir, f"{base_filename}.txt")
    elif field_name == "negative_caption":
        caption_txt_path = os.path.join(image_dir, f"{base_filename}_neg.txt")
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

# Image Metadata
def get_image_metadata(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata (width, height) from an image file.
    Returns a dictionary or None if an error occurs.
    """
    if not image_path or not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        return None
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not open image: {image_path}")
            return None
        
        height, width = img.shape[:2]
        return {'width': width, 'height': height}
    except Exception as e:
        print(f"Error getting image metadata for {image_path}: {e}")
        return None

def _closest_divisible_by(value: int, divisor: int) -> int:
    """Helper to find the closest multiple of 'divisor' to 'value', ensuring minimum of 'divisor'."""
    return max(divisor, int(round(value / divisor)) * divisor)

def calculate_closest_div32_dimensions(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculates dimensions for an image that are closest to original and divisible by 32.
    Returns (width_str, height_str) or (None, None).
    """
    metadata = get_image_metadata(image_path)
    if not metadata or metadata['width'] <= 0 or metadata['height'] <= 0:
        return None, None
    
    width32 = _closest_divisible_by(metadata['width'], 32)
    height32 = _closest_divisible_by(metadata['height'], 32)
    
    return str(width32), str(height32)

def calculate_contained_image_dimensions(
    image_orig_w: int, image_orig_h: int,
    target_width: int, target_height: int
) -> Tuple[int, int, int, int]:
    """
    Calculates the effective displayed width, height, and offsets of an image
    when it's 'CONTAIN'ed within a target container, maintaining aspect ratio.
    Returns (scaled_width, scaled_height, offset_x, offset_y).
    """
    if image_orig_w <= 0 or image_orig_h <= 0 or target_width <= 0 or target_height <= 0:
        return 0, 0, 0, 0 # Return zeros for invalid inputs

    target_aspect_ratio = target_width / target_height
    image_aspect_ratio = image_orig_w / image_orig_h

    if image_aspect_ratio > target_aspect_ratio:
        # Image is wider than container, constrained by width
        scaled_width = target_width
        scaled_height = int(target_width / image_aspect_ratio)
    else:
        # Image is taller than container, constrained by height
        scaled_height = target_height
        scaled_width = int(target_height * image_aspect_ratio)

    offset_x = (target_width - scaled_width) // 2
    offset_y = (target_height - scaled_height) // 2

    return scaled_width, scaled_height, offset_x, offset_y

# Image Processing Functions
def load_image(image_path: str) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Load an image and return the image array and its dimensions (h, w).
    Returns None if loading fails.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # Convert from BGR (OpenCV) to RGB
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif len(img.shape) == 3:  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img, img.shape[:2]  # Return (height, width)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(image: np.ndarray, output_path: str) -> bool:
    """Save an image to the specified path."""
    try:
        # Convert back to BGR for saving with OpenCV
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif len(image.shape) == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def crop_image(
    image_path: str,
    x: int, y: int, width: int, height: int,
    output_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Crop an image to the specified rectangle.
    Returns (success, message, output_path_or_none)
    """
    try:
        result = load_image(image_path)
        if result is None:
            return False, f"Failed to load image {image_path}", None
            
        img, _ = result
        h, w = img.shape[:2]
        
        # Ensure the crop rectangle is within image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + width), min(h, y + height)
        
        if x1 >= x2 or y1 >= y2:
            return False, "Invalid crop dimensions", None
            
        cropped = img[y1:y2, x1:x2]
        
        if output_path is None:
            name, ext = os.path.splitext(os.path.basename(image_path))
            temp_dir = os.path.join(os.path.dirname(image_path), "temp_processing")
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"{name}_cropped{ext}")
            
        if not save_image(cropped, output_path):
            return False, f"Failed to save cropped image to {output_path}", None
            
        return True, "Image cropped successfully", output_path
        
    except Exception as e:
        return False, f"Error cropping image: {e}" , None

def calculate_adjusted_size(
    current_w_str: str, 
    current_h_str: str, 
    image_path: Optional[str], 
    adjustment_type: str, # 'add' or 'sub'
    step: int = 32
) -> Tuple[str, str]:
    """
    Adjusts width and height by a step (default 32), maintaining aspect ratio if image_path is provided.
    Returns new (width_str, height_str).
    """
    try:
        w = int(current_w_str) if current_w_str else step
        h = int(current_h_str) if current_h_str else step
    except ValueError:
        return str(step), str(step)

    aspect_ratio = None
    if image_path:
        metadata = get_image_metadata(image_path)
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

def update_image_info_json(image_path: str):
    """
    Updates or creates a JSON file with image metadata (width, height).
    """
    json_path = os.path.splitext(image_path)[0] + ".json"
    metadata = get_image_metadata(image_path)
    
    if metadata:
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Error writing image info JSON for {image_path}: {e}")
    else:
        print(f"No metadata to write for {image_path}")

def crop_image_from_overlay(
    current_image_path: str,
    overlay_x_norm: float, overlay_y_norm: float,
    overlay_w_norm: float, overlay_h_norm: float,
    displayed_image_w: int, displayed_image_h: int,
    image_orig_w: int, image_orig_h: int,
    player_content_w: int, player_content_h: int,
    overlay_angle_rad: float = 0.0,
) -> Tuple[bool, str, Optional[str]]:
    """
    Crops an image based on overlay coordinates from the UI.
    Converts normalized overlay coordinates to original image pixel coordinates.
    Also crops the related control image (if exists and same dimensions).
    """
    if not os.path.exists(current_image_path):
        return False, "Image file not found.", None

    # Calculate scaling factors for the displayed image within the player content area
    # This assumes the image is scaled to fit within displayed_image_w/h while maintaining aspect ratio
    scale_w = displayed_image_w / image_orig_w
    scale_h = displayed_image_h / image_orig_h

    # Use the smaller scale factor to determine the actual displayed size of the image
    # This is crucial because the image might not fill the entire player_content_w/h
    # if its aspect ratio is different.
    actual_scale = min(scale_w, scale_h)

    # Calculate the actual width and height of the image as displayed in the UI
    actual_displayed_image_w = int(image_orig_w * actual_scale)
    actual_displayed_image_h = int(image_orig_h * actual_scale)

    # Calculate padding if the image is not filling the entire player content area
    pad_x = (player_content_w - actual_displayed_image_w) / 2
    pad_y = (player_content_h - actual_displayed_image_h) / 2

    # Convert overlay coordinates from UI pixels to displayed image pixels
    # Account for padding: overlay_x_norm and overlay_y_norm are relative to player_content_w/h
    crop_x_displayed = overlay_x_norm - pad_x
    crop_y_displayed = overlay_y_norm - pad_y
    crop_w_displayed = overlay_w_norm
    crop_h_displayed = overlay_h_norm

    # Convert displayed image pixels to original image pixels
    crop_x_orig = int(crop_x_displayed / actual_scale)
    crop_y_orig = int(crop_y_displayed / actual_scale)
    crop_w_orig = int(crop_w_displayed / actual_scale)
    crop_h_orig = int(crop_h_displayed / actual_scale)

    # Ensure dimensions are positive and divisible by 32
    crop_w_orig = max(32, (crop_w_orig // 32) * 32)
    crop_h_orig = max(32, (crop_h_orig // 32) * 32)

    # Ensure crop coordinates are within original image bounds
    crop_x_orig = max(0, crop_x_orig)
    crop_y_orig = max(0, crop_y_orig)

    # Adjust width/height if they extend beyond image boundaries
    crop_w_orig = min(crop_w_orig, image_orig_w - crop_x_orig)
    crop_h_orig = min(crop_h_orig, image_orig_h - crop_y_orig)

    # Handle rotation if angle is significant
    if abs(overlay_angle_rad) > 0.01:  # Only apply rotation if angle is more than ~0.5 degrees
        result = crop_image_with_rotation(current_image_path, crop_x_orig, crop_y_orig, crop_w_orig, crop_h_orig, overlay_angle_rad)
    else:
        result = crop_image(current_image_path, crop_x_orig, crop_y_orig, crop_w_orig, crop_h_orig)

    # If the main image crop succeeded, also crop the related control image
    if result[0]:  # success is True
        related_info = get_related_image_info_for_crop(current_image_path)
        if related_info:
            related_image_path = related_info['path']
            # First, scale the related image if needed (dimensions are close but not exact)
            if related_info.get('needs_scaling'):
                # Scale the related image to match the current image dimensions
                scale_result = load_image(related_image_path)
                if scale_result is not None:
                    related_img, _ = scale_result
                    target_w = related_info['target_width']
                    target_h = related_info['target_height']
                    # Resize using INTER_AREA for shrinking, INTER_LINEAR for growing
                    interpolation = cv2.INTER_AREA if (related_img.shape[1] > target_w or related_img.shape[0] > target_h) else cv2.INTER_LINEAR
                    resized = cv2.resize(related_img, (target_w, target_h), interpolation=interpolation)
                    # Save the scaled image
                    if save_image(resized, related_image_path):
                        pass  # Successfully scaled
            # Now crop with the same coordinates (dimensions now match)
            if abs(overlay_angle_rad) > 0.01:
                crop_image_with_rotation(related_image_path, crop_x_orig, crop_y_orig, crop_w_orig, crop_h_orig, overlay_angle_rad, related_image_path)
            else:
                crop_image(related_image_path, crop_x_orig, crop_y_orig, crop_w_orig, crop_h_orig, related_image_path)

    return result

def crop_image_with_rotation(
    image_path: str,
    x: int, y: int, width: int, height: int,
    angle_rad: float,
    output_path: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Crop an image with rotation support.
    Crops a larger area to contain the rotated rectangle, applies rotation, then final crop.
    Returns (success, message, output_path_or_none)
    """
    try:
        result = load_image(image_path)
        if result is None:
            return False, f"Failed to load image {image_path}", None
            
        img, _ = result
        h, w = img.shape[:2]
        
        # Ensure the initial crop rectangle is within image bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + width), min(h, y + height)
        
        if x1 >= x2 or y1 >= y2:
            return False, "Invalid crop dimensions", None
            
        # Calculate the bounding box needed for the rotated rectangle
        # This follows the same logic as the video crop implementation
        crop_w = x2 - x1
        crop_h = y2 - y1
        center_x = x1 + crop_w / 2
        center_y = y1 + crop_h / 2
        
        # Calculate the diagonal of the crop rectangle (minimum square needed)
        diagonal = int(np.sqrt(crop_w**2 + crop_h**2))
        
        # Calculate the larger square crop area
        half_pre = diagonal / 2
        pre_x = max(0.0, center_x - half_pre)
        pre_y = max(0.0, center_y - half_pre)
        pre_x2 = min(w, center_x + half_pre)
        pre_y2 = min(h, center_y + half_pre)
        
        # Crop the larger area first
        pre_crop = img[int(pre_y):int(pre_y2), int(pre_x):int(pre_x2)]
        
        # Create rotation matrix
        angle_deg = np.degrees(-angle_rad)  # Negative for correct direction
        pre_h, pre_w = pre_crop.shape[:2]
        center = (pre_w // 2, pre_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(pre_crop, rotation_matrix, (pre_w, pre_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Calculate final crop from the center of rotated image
        final_x = (pre_w - crop_w) // 2
        final_y = (pre_h - crop_h) // 2
        final_x2 = final_x + crop_w
        final_y2 = final_y + crop_h
        
        # Ensure final crop coordinates are valid
        final_x = max(0, final_x)
        final_y = max(0, final_y)
        final_x2 = min(pre_w, final_x2)
        final_y2 = min(pre_h, final_y2)
        
        final_cropped = rotated[final_y:final_y2, final_x:final_x2]
        
        if output_path is None:
            name, ext = os.path.splitext(os.path.basename(image_path))
            temp_dir = os.path.join(os.path.dirname(image_path), "temp_processing")
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"{name}_cropped_rotated{ext}")
            
        if not save_image(final_cropped, output_path):
            return False, f"Failed to save rotated cropped image to {output_path}", None
            
        return True, "Image cropped with rotation successfully", output_path
        
    except Exception as e:
        return False, f"Error cropping image with rotation: {e}", None


def crop_image_to_dimensions(
    image_path: str,
    target_width: int,
    target_height: int
) -> Tuple[bool, str, Optional[str]]:
    """
    Scales an image proportionally to cover the target dimensions, then center crops.
    Also crops the related control image (if exists and same dimensions).
    Returns (success, message, output_path_or_none).
    """
    metadata = get_image_metadata(image_path)
    if not metadata:
        return False, "Could not get image metadata.", None

    original_width = metadata['width']
    original_height = metadata['height']

    if target_width <= 0 or target_height <= 0:
        return False, "Target width and height must be positive.", None

    # Calculate scaling factor to ensure the image covers the target area
    scale_w_factor = target_width / original_width
    scale_h_factor = target_height / original_height
    scale_factor = max(scale_w_factor, scale_h_factor)

    # Calculate new dimensions after scaling
    scaled_width = int(original_width * scale_factor)
    scaled_height = int(original_height * scale_factor)

    # Load the original image
    result = load_image(image_path)
    if result is None:
        return False, f"Failed to load image {image_path}", None

    img, _ = result

    # Resize the image
    resized_img = cv2.resize(img, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    # Calculate crop coordinates to center the target dimensions within the scaled image
    x = (scaled_width - target_width) // 2
    y = (scaled_height - target_height) // 2

    # Ensure crop coordinates are non-negative
    x = max(0, x)
    y = max(0, y)

    # Ensure crop dimensions do not exceed scaled image dimensions
    crop_w = min(target_width, scaled_width - x)
    crop_h = min(target_height, scaled_height - y)

    # Perform the crop on the resized image
    cropped_img = resized_img[y : y + crop_h, x : x + crop_w]

    # Generate a temporary output path
    name, ext = os.path.splitext(os.path.basename(image_path))
    temp_dir = os.path.join(os.path.dirname(image_path), "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, f"{name}_scaled_cropped{ext}")

    if not save_image(cropped_img, output_path):
        return False, f"Failed to save scaled and cropped image to {output_path}", None

    # Also crop the related control image if it exists and has the same (or close) dimensions
    related_info = get_related_image_info_for_crop(image_path)
    if related_info:
        related_image_path = related_info['path']

        # First, scale the related image if needed (dimensions are close but not exact)
        if related_info.get('needs_scaling'):
            # Scale the related image to match the current image dimensions first
            scale_result = load_image(related_image_path)
            if scale_result is not None:
                related_img, _ = scale_result
                target_w = related_info['target_width']
                target_h = related_info['target_height']
                # Resize using INTER_AREA for shrinking, INTER_LINEAR for growing
                interpolation = cv2.INTER_AREA if (related_img.shape[1] > target_w or related_img.shape[0] > target_h) else cv2.INTER_LINEAR
                resized = cv2.resize(related_img, (target_w, target_h), interpolation=interpolation)
                # Save the scaled image (overwrite the original related image)
                if save_image(resized, related_image_path):
                    pass  # Successfully scaled, now dimensions match

        # Now get the (potentially updated) metadata and apply the same scale + crop
        related_metadata = get_image_metadata(related_image_path)
        if related_metadata:
            related_width = related_metadata['width']
            related_height = related_metadata['height']

            # Use the same scale factor for consistency
            related_scaled_width = int(related_width * scale_factor)
            related_scaled_height = int(related_height * scale_factor)

            # Load and resize the related image
            related_result = load_image(related_image_path)
            if related_result is not None:
                related_img, _ = related_result
                related_resized = cv2.resize(related_img, (related_scaled_width, related_scaled_height), interpolation=cv2.INTER_AREA)

                # Calculate crop coordinates for related image (same as main image)
                related_x = (related_scaled_width - target_width) // 2
                related_y = (related_scaled_height - target_height) // 2
                related_x = max(0, related_x)
                related_y = max(0, related_y)

                related_crop_w = min(target_width, related_scaled_width - related_x)
                related_crop_h = min(target_height, related_scaled_height - related_y)

                # Perform the crop
                related_cropped = related_resized[related_y : related_y + related_crop_h, related_x : related_x + related_crop_w]

                # Save the related cropped image directly to the related image path
                save_image(related_cropped, related_image_path)

    return True, "Image scaled and cropped successfully", output_path
