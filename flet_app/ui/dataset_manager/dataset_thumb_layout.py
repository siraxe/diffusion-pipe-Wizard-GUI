import base64
import flet as ft
import os
from flet_app.settings import settings
from flet_app.ui_popups.unified_popup_dialog import open_unified_popup_dialog


def _is_web_platform(page: ft.Page | None) -> bool:
    """Return True when running inside a Flet web page."""
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

def set_thumbnail_selection_state(thumbnail_container: ft.Container, is_selected: bool):
    """
    Updates the visual state (background color and checkbox opacity) of a thumbnail container.
    This function is designed to be called programmatically from outside the dataset_thumb_layout module.
    """
    checkbox = None
    if isinstance(thumbnail_container.content, ft.Stack):
        for control in thumbnail_container.content.controls:
            if isinstance(control, ft.Checkbox):
                checkbox = control
                break
    
    if checkbox:
        checkbox.value = is_selected # Set the value
        checkbox.opacity = 1 if is_selected else 0 # Update opacity based on selection
        thumbnail_container.bgcolor = ft.Colors.with_opacity(0.3, ft.Colors.BLUE_100) if is_selected else ft.Colors.TRANSPARENT
        
        if thumbnail_container.page:
            thumbnail_container.update()
            checkbox.update()

def create_thumbnail_container(
    page_ctx: ft.Page,
    video_path: str,
    thumb_path: str,
    video_info: dict,
    has_caption: bool,
    video_files_list: list,
    update_thumbnails_callback,
    grid_control: ft.GridView,
    on_checkbox_change_callback,
    thumbnail_index: int,
    is_selected_initially: bool
):
    video_name = os.path.basename(video_path)
    info = video_info.get(video_name, {})
    width, height, frames = info.get("width", "?"), info.get("height", "?"), info.get("frames", "?")
    fps = info.get("fps", "?")
    cap_val, cap_color = ("yes", ft.Colors.GREEN) if has_caption else ("no", ft.Colors.RED)

    def _handle_thumbnail_click(e_click):
        if page_ctx:
            # Use the new minimal unified popup UI with top controls only.
            open_unified_popup_dialog(
                page_ctx,
                video_path,
                video_files_list,
            )

    is_hovered = False

    checkbox = ft.Checkbox(
        value=is_selected_initially,
        check_color=ft.Colors.WHITE,
        fill_color=ft.Colors.BLUE_GREY_500,
        overlay_color=ft.Colors.TRANSPARENT,
        active_color=ft.Colors.BLUE_GREY_500,
        right=0,
        bottom=0,
        opacity=0,
        animate_opacity=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
        data=thumbnail_index
    )

    # Define the handler functions BEFORE creating thumbnail_container
    def _on_hover_container(e: ft.HoverEvent):
        nonlocal is_hovered
        is_hovered = (e.data == "true")
        # When hovered, update opacity based on hover state and current checkbox value
        checkbox.opacity = 1 if is_hovered or checkbox.value else 0
        if checkbox.page:
            checkbox.page.update()

    def _on_checkbox_change(e: ft.ControlEvent):
        # This is triggered by user click. Update visuals and then call external callback.
        set_thumbnail_selection_state(thumbnail_container, checkbox.value)
        on_checkbox_change_callback(video_path, checkbox.value, thumbnail_index)

    encoded_thumbnail = None
    if _is_web_platform(page_ctx) and os.path.exists(thumb_path):
        try:
            with open(thumb_path, "rb") as thumb_file:
                encoded_thumbnail = base64.b64encode(thumb_file.read()).decode("utf-8")
        except Exception:
            encoded_thumbnail = None

    image_control = ft.Image(
        src=None if encoded_thumbnail else thumb_path.replace("\\", "/"),
        src_base64=encoded_thumbnail,
        width=settings.THUMB_TARGET_W,
        height=settings.THUMB_TARGET_H,
        fit=ft.ImageFit.COVER,
        border_radius=ft.border_radius.all(5)
    )

    thumbnail_container = ft.Container(
        content=ft.Stack(
            [
                ft.Column([
                    image_control,
                    ft.Text(spans=[
                        ft.TextSpan("[cap - ", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                        ft.TextSpan(cap_val, style=ft.TextStyle(color=cap_color, size=10)),
                        ft.TextSpan("] - ", style=ft.TextStyle(color=ft.Colors.GREY_500, size=10)),
                        ft.TextSpan(f"{fps} fps", style=ft.TextStyle(color=ft.Colors.BLUE, size=10)),
                    ], size=10),
                    ft.Text(f"[{width}x{height} - {frames} frames]", size=10, color=ft.Colors.GREY_500),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, tight=True),
                checkbox
            ]
        ),
        data=video_path,
        on_click=_handle_thumbnail_click,
        on_hover=_on_hover_container, # Now _on_hover_container is defined
        tooltip=video_name,
        width=settings.THUMB_TARGET_W + 10,
        height=settings.THUMB_TARGET_H + 45,
        padding=5,
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=ft.border_radius.all(5),
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREEN_ACCENT_700) if is_selected_initially else ft.Colors.TRANSPARENT
    )

    # Initial visual update based on is_selected_initially
    set_thumbnail_selection_state(thumbnail_container, is_selected_initially)

    # Assign on_change to checkbox
    checkbox.on_change = _on_checkbox_change

    return thumbnail_container
