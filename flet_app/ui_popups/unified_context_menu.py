import flet as ft
from typing import Callable, List, Optional, Tuple

from flet_app.ui._styles import create_styled_button, BTN_STYLE2
from . import image_editor
from . import video_player_utils as vpu
from . import image_player_utils as ipu
from PIL import Image
import shutil
import os
from . import video_editor


def get_unique_filename(directory: str, base_name: str, extension: str) -> str:
    """Generate a unique filename by adding _copy suffix if needed."""
    new_name = f"{base_name}{extension}"
    counter = 1

    while os.path.exists(os.path.join(directory, new_name)):
        new_name = f"{base_name}_copy{counter}{extension}"
        counter += 1

    return new_name


def rename_single_file(
    media_path: str,
    new_filename: str,
    page: ft.Page,
    on_refresh: Optional[Callable[[], None]] = None
) -> Tuple[bool, str]:
    """Rename the media file and its associated files (.txt, control)."""
    try:
        media_dir = os.path.dirname(media_path)
        base_filename, ext = os.path.splitext(os.path.basename(media_path))
        new_base, _ = os.path.splitext(new_filename)

        # Get unique filename if exists
        final_new_filename = get_unique_filename(media_dir, new_base, ext)
        final_new_path = os.path.join(media_dir, final_new_filename)

        # Rename the main file
        shutil.move(media_path, final_new_path)

        renamed_files = [final_new_filename]

        # Rename associated .txt file
        txt_path = os.path.join(media_dir, f"{base_filename}.txt")
        if os.path.exists(txt_path):
            new_txt_path = os.path.join(media_dir, f"{os.path.splitext(final_new_filename)[0]}.txt")
            # Handle duplicate txt filenames
            if os.path.exists(new_txt_path) and new_txt_path != txt_path:
                txt_base = os.path.splitext(final_new_filename)[0]
                new_txt_name = get_unique_filename(media_dir, txt_base, ".txt")
                new_txt_path = os.path.join(media_dir, new_txt_name)
            shutil.move(txt_path, new_txt_path)
            renamed_files.append(os.path.basename(new_txt_path))

        # Rename associated _neg.txt file
        neg_txt_path = os.path.join(media_dir, f"{base_filename}_neg.txt")
        if os.path.exists(neg_txt_path):
            new_neg_txt_path = os.path.join(media_dir, f"{os.path.splitext(final_new_filename)[0]}_neg.txt")
            if os.path.exists(new_neg_txt_path) and new_neg_txt_path != neg_txt_path:
                neg_txt_base = os.path.splitext(final_new_filename)[0]
                new_neg_txt_name = get_unique_filename(media_dir, f"{neg_txt_base}_neg", ".txt")
                new_neg_txt_path = os.path.join(media_dir, new_neg_txt_name)
            shutil.move(neg_txt_path, new_neg_txt_path)
            renamed_files.append(os.path.basename(new_neg_txt_path))

        # For images, check for control file
        if "/control/" in media_path:
            # This is a control image - rename the original image too
            original_path = ipu.get_original_image_path(media_path)
            if os.path.exists(original_path):
                orig_dir = os.path.dirname(original_path)
                orig_base = os.path.splitext(os.path.basename(original_path))[0]
                orig_ext = os.path.splitext(original_path)[1]
                new_orig_name = get_unique_filename(orig_dir, os.path.splitext(final_new_filename)[0], orig_ext)
                new_orig_path = os.path.join(orig_dir, new_orig_name)
                shutil.move(original_path, new_orig_path)
                renamed_files.append(new_orig_name)
        else:
            # This is an original image - check for and rename control image
            control_path = ipu.get_control_image_path(media_path)
            if control_path and os.path.exists(control_path):
                control_dir = os.path.dirname(control_path)
                control_base = os.path.splitext(os.path.basename(control_path))[0]
                control_ext = os.path.splitext(control_path)[1]
                new_control_name = get_unique_filename(control_dir, os.path.splitext(final_new_filename)[0], control_ext)
                new_control_path = os.path.join(control_dir, new_control_name)
                shutil.move(control_path, new_control_path)
                renamed_files.append(os.path.basename(new_control_path))

        return True, f"Renamed: {', '.join(renamed_files)}"

    except Exception as ex:
        return False, f"Rename failed: {ex}"


def build_context_menu(
    is_image: bool,
    page: ft.Page,
    media_path: str,
    media_list: List[str],
    on_close: Callable[[], None],
    on_refresh: Optional[Callable[[], None]] = None,
) -> ft.Container:
    """Builds the unified popup's context menu with styled buttons.

    Actions:
    - Flip Horizontal (image/video)
    - Rotate +90 (image/video)
    - Rotate -90 (image/video)
    - Reverse (video only; disabled for images)
    """

    def _wrap(callable_fn, *args, **kwargs):
        def _runner(ev):
            try:
                on_close()
            except Exception:
                pass
            try:
                page.run_thread(callable_fn, *args, **kwargs)
            except Exception:
                # Fallback to direct call if run_thread not available
                try:
                    callable_fn(*args, **kwargs)
                except Exception as ex:
                    print(f"[UnifiedContextMenu] action error: {ex}")
        return _runner

    if is_image:
        def flip_cb(ev):
            try:
                on_close()
                img = Image.open(media_path)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(media_path)
                page.snack_bar = ft.SnackBar(ft.Text("Image flipped."), open=True)
                page.update()
                if on_refresh:
                    on_refresh()
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Flip failed: {ex}"), open=True)
                    page.update()
                except Exception:
                    pass

        def rot_p_cb(ev):
            try:
                on_close()
                img = Image.open(media_path)
                img = img.rotate(90, expand=True)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(media_path)
                page.snack_bar = ft.SnackBar(ft.Text("Image rotated +90."), open=True)
                page.update()
                if on_refresh:
                    on_refresh()
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Rotate +90 failed: {ex}"), open=True)
                    page.update()
                except Exception:
                    pass

        def rot_m_cb(ev):
            try:
                on_close()
                img = Image.open(media_path)
                img = img.rotate(-90, expand=True)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(media_path)
                page.snack_bar = ft.SnackBar(ft.Text("Image rotated -90."), open=True)
                page.update()
                if on_refresh:
                    on_refresh()
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Rotate -90 failed: {ex}"), open=True)
                    page.update()
                except Exception:
                    pass

        rev_cb = None
    else:
        def flip_cb(ev):
            on_close()
            success, msg, temp_out = vpu.flip_video_horizontal(media_path)
            if success and temp_out and os.path.exists(temp_out):
                try:
                    shutil.move(temp_out, media_path)
                    page.snack_bar = ft.SnackBar(ft.Text(msg or "Video flipped."), open=True)
                    page.update()
                    if on_refresh:
                        on_refresh()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Finalize flip failed: {ex}"), open=True)
                    page.update()
            else:
                page.snack_bar = ft.SnackBar(ft.Text(msg or "Flip failed."), open=True)
                page.update()

        def rot_p_cb(ev):
            on_close()
            success, msg, temp_out = vpu.rotate_video_90(media_path, 'plus')
            if success and temp_out and os.path.exists(temp_out):
                try:
                    shutil.move(temp_out, media_path)
                    page.snack_bar = ft.SnackBar(ft.Text(msg or "+90 applied."), open=True)
                    page.update()
                    if on_refresh:
                        on_refresh()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Finalize rotate +90 failed: {ex}"), open=True)
                    page.update()
            else:
                page.snack_bar = ft.SnackBar(ft.Text(msg or "Rotate +90 failed."), open=True)
                page.update()

        def rot_m_cb(ev):
            on_close()
            success, msg, temp_out = vpu.rotate_video_90(media_path, 'minus')
            if success and temp_out and os.path.exists(temp_out):
                try:
                    shutil.move(temp_out, media_path)
                    page.snack_bar = ft.SnackBar(ft.Text(msg or "-90 applied."), open=True)
                    page.update()
                    if on_refresh:
                        on_refresh()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Finalize rotate -90 failed: {ex}"), open=True)
                    page.update()
            else:
                page.snack_bar = ft.SnackBar(ft.Text(msg or "Rotate -90 failed."), open=True)
                page.update()

        def rev_cb(ev):
            on_close()
            success, msg, temp_out = vpu.reverse_video(media_path)
            if success and temp_out and os.path.exists(temp_out):
                try:
                    shutil.move(temp_out, media_path)
                    page.snack_bar = ft.SnackBar(ft.Text(msg or "Reversed."), open=True)
                    page.update()
                    if on_refresh:
                        on_refresh()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Finalize reverse failed: {ex}"), open=True)
                    page.update()
            else:
                page.snack_bar = ft.SnackBar(ft.Text(msg or "Reverse failed."), open=True)
                page.update()

    # Create rename text field and button
    rename_field = ft.TextField(
        label="New name",
        hint_text="Enter new filename",
        width=200,
        height=40,
    )

    def rename_cb(ev):
        new_filename = rename_field.value.strip()
        if not new_filename:
            # Do nothing if empty
            return

        # Clear the field immediately before any async operations
        rename_field.value = ""

        base_filename, ext = os.path.splitext(os.path.basename(media_path))

        # Add extension if user didn't include it
        if not new_filename.endswith(ext):
            new_filename = new_filename + ext

        success, msg = rename_single_file(media_path, new_filename, page, on_refresh)

        # Close the context menu first (like other callbacks do)
        try:
            on_close()
        except Exception:
            pass

        page.snack_bar = ft.SnackBar(ft.Text(msg), open=True)
        page.update()

        if success and on_refresh:
            on_refresh()

    rename_single_btn = create_styled_button("Rename", on_click=rename_cb, button_style=BTN_STYLE2)
    flip_btn = create_styled_button("Flip Horizontal", on_click=flip_cb, button_style=BTN_STYLE2)
    rot_plus_btn = create_styled_button("Rotate +90", on_click=rot_p_cb, button_style=BTN_STYLE2)
    rot_minus_btn = create_styled_button("Rotate -90", on_click=rot_m_cb, button_style=BTN_STYLE2)
    reverse_btn = create_styled_button("Reverse", on_click=rev_cb, button_style=BTN_STYLE2, disabled=is_image)

    menu_col = ft.Column(
        controls=[rename_field, rename_single_btn, flip_btn, rot_plus_btn, rot_minus_btn, reverse_btn],
        spacing=4,
        tight=True,
    )

    return ft.Container(
        content=menu_col,
        bgcolor=ft.Colors.SURFACE,
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=ft.border_radius.all(8),
        padding=6,
        visible=False,
    )



