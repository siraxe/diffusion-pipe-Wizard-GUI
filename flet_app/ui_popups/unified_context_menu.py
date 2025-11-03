import flet as ft
from typing import Callable, List, Optional

from flet_app.ui._styles import create_styled_button, BTN_STYLE2
from . import image_editor
from . import video_player_utils as vpu
from . import image_player_utils as ipu
from PIL import Image
import shutil
import os
from . import video_editor


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

    flip_btn = create_styled_button("Flip Horizontal", on_click=flip_cb, button_style=BTN_STYLE2)
    rot_plus_btn = create_styled_button("Rotate +90", on_click=rot_p_cb, button_style=BTN_STYLE2)
    rot_minus_btn = create_styled_button("Rotate -90", on_click=rot_m_cb, button_style=BTN_STYLE2)
    reverse_btn = create_styled_button("Reverse", on_click=rev_cb, button_style=BTN_STYLE2, disabled=is_image)

    menu_col = ft.Column(
        controls=[flip_btn, rot_plus_btn, rot_minus_btn, reverse_btn],
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



