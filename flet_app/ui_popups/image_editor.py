import flet as ft 
import os
import shutil
from typing import Optional, List, Callable 
from PIL import Image

from . import image_player_utils as ipu
from flet_app.ui._styles import IMAGE_PLAYER_DIALOG_WIDTH, IMAGE_PLAYER_DIALOG_HEIGHT


def handle_size_add(width_field: ft.TextField, height_field: ft.TextField, current_image_path: str, page: Optional[ft.Page] = None): 
    new_w_str, new_h_str = ipu.calculate_adjusted_size(
        width_field.value, height_field.value, current_image_path, 'add'
    )
    width_field.value = new_w_str
    height_field.value = new_h_str
    if page:
        width_field.update()
        height_field.update()

def handle_size_sub(width_field: ft.TextField, height_field: ft.TextField, current_image_path: str, page: Optional[ft.Page] = None): 
    new_w_str, new_h_str = ipu.calculate_adjusted_size(
        width_field.value, height_field.value, current_image_path, 'sub'
    )
    width_field.value = new_w_str
    height_field.value = new_h_str
    if page:
        width_field.update()
        height_field.update()

# Removed legacy overlay-crop path tied to old image dialog

def handle_set_closest_div32(width_field: ft.TextField, height_field: ft.TextField, current_image_path: str, page: Optional[ft.Page] = None): 
    w_str, h_str = ipu.calculate_closest_div32_dimensions(current_image_path)
    if w_str and h_str:
        width_field.value = w_str
        height_field.value = h_str
        if page:
            width_field.update()
            height_field.update()

def _generic_image_operation_ui_update(
    page: ft.Page, 
    processed_image_path: str, 
    image_list: Optional[List[str]] = None, 
    on_caption_updated_callback: Optional[Callable] = None, 
    operation_message: str = "Image operation successful."
    ):
    if page:
        page.snack_bar = ft.SnackBar(ft.Text(operation_message), open=True)
        if on_caption_updated_callback:
            on_caption_updated_callback(processed_image_path)
        page.update()

def handle_crop_image_click(page: ft.Page, width_field: Optional[ft.TextField], height_field: Optional[ft.TextField], current_image_path: str, image_list: Optional[List[str]] = None, on_caption_updated_callback: Optional[Callable] = None, should_update_ui: bool = True): 
    if not width_field or not width_field.value or not height_field or not height_field.value : 
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text("Width and Height must be specified."), open=True); page.update()
        return
    try:
        target_width = int(width_field.value)
        target_height = int(height_field.value)
    except ValueError:
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text("Invalid Width or Height value."), open=True); page.update()
        return

    success, msg, temp_output_path = ipu.crop_image_to_dimensions(current_image_path, target_width, target_height)
    
    if success and temp_output_path:
        try:
            shutil.move(temp_output_path, current_image_path)
            if should_update_ui:
                 _generic_image_operation_ui_update(page, current_image_path, image_list, on_caption_updated_callback, msg)
            else: 
                # Removed ipu.update_image_info_json as per user request
                if page: page.snack_bar = ft.SnackBar(ft.Text(f"{os.path.basename(current_image_path)}: {msg}" if msg else f"{os.path.basename(current_image_path)} cropped (batch)."), duration=2000, open=True)
        except Exception as e:
            final_msg = f"Error moving cropped file for {os.path.basename(current_image_path)}: {e}"
            if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text(final_msg), open=True)
            print(final_msg)
            if os.path.exists(temp_output_path): os.remove(temp_output_path)
    else:
        final_msg = msg if msg else f"Failed to crop {os.path.basename(current_image_path)}."
        if page and should_update_ui: page.snack_bar = ft.SnackBar(ft.Text(final_msg), open=True)
        print(final_msg)

    if page and not should_update_ui: 
        page.update()

def handle_crop_all_images(
    page: ft.Page, 
    current_image_path_in_dialog: Optional[str],
    width_field: Optional[ft.TextField], 
    height_field: Optional[ft.TextField], 
    image_list: Optional[List[str]], 
    on_caption_updated_callback: Optional[Callable]
): 
    if not image_list:
        if page: page.snack_bar = ft.SnackBar(ft.Text("No images in the list to crop."), open=True); page.update()
        return
    
    if not width_field or not width_field.value or not height_field or not height_field.value:
        if page: page.snack_bar = ft.SnackBar(ft.Text("Width and Height must be specified for batch crop."), open=True); page.update()
        return

    processed_count = 0
    dialog_refreshed_for_current_image = False

    for image_path_item in image_list: 
        processed_count +=1
        # Determine if the UI (dialog) should be fully updated for this specific item
        should_refresh_dialog_for_this_item = (image_path_item == current_image_path_in_dialog)
        
        # Call handle_crop_image_click, which might reopen the dialog if should_refresh_dialog_for_this_item is True
        handle_crop_image_click(
            page, width_field, height_field, 
            image_path_item, image_list, 
            on_caption_updated_callback, 
            should_update_ui=should_refresh_dialog_for_this_item
        )
        
        if should_refresh_dialog_for_this_item:
            dialog_refreshed_for_current_image = True # Mark that the dialog was handled

    if page:
        # Display a summary snackbar
        page.snack_bar = ft.SnackBar(ft.Text(f"Batch crop attempt finished for {processed_count} images. Review individual messages."), open=True)
        
        # Call general callback and update page only if dialog wasn't specifically refreshed for the current image
        if on_caption_updated_callback and not dialog_refreshed_for_current_image:
            on_caption_updated_callback()
        
        if not dialog_refreshed_for_current_image:
            page.update()

def handle_flip_image(page: ft.Page, current_image_path: str, image_list: Optional[List[str]] = None, on_caption_updated_callback: Optional[Callable] = None):
    if not current_image_path or not os.path.exists(current_image_path):
        if page: page.snack_bar = ft.SnackBar(ft.Text("Error: Invalid image path for flipping."), open=True); page.update()
        return

    try:
        img = Image.open(current_image_path)
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if flipped_img.mode == 'RGBA':
            flipped_img = flipped_img.convert('RGB')
        flipped_img.save(current_image_path)
        _generic_image_operation_ui_update(page, current_image_path, image_list, on_caption_updated_callback, "Image flipped successfully.")
    except Exception as e:
        msg = f"Error flipping image: {e}"
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()
        print(msg)

def handle_rotate_image(page: ft.Page, current_image_path: str, image_list: Optional[List[str]] = None, on_caption_updated_callback: Optional[Callable] = None, degrees: int = 90):
    if not current_image_path or not os.path.exists(current_image_path):
        if page: page.snack_bar = ft.SnackBar(ft.Text("Error: Invalid image path for rotation."), open=True); page.update()
        return

    try:
        img = Image.open(current_image_path)
        rotated_img = img.rotate(degrees, expand=True)
        if rotated_img.mode == 'RGBA':
            rotated_img = rotated_img.convert('RGB')
        rotated_img.save(current_image_path)
        _generic_image_operation_ui_update(page, current_image_path, image_list, on_caption_updated_callback, f"Image rotated by {degrees} degrees successfully.")
    except Exception as e:
        msg = f"Error rotating image: {e}"
        if page: page.snack_bar = ft.SnackBar(ft.Text(msg), open=True); page.update()
        print(msg)
