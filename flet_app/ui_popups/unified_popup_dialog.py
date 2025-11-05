import os
import platform
import tempfile
import subprocess
import threading
import time
from typing import List, Optional

import flet as ft

from flet_app.settings import settings
from flet_app.ui.flet_hotkeys import PLAY_PAUSE_KEY, NEXT_KEY, PREV_KEY
from flet_app.ui._styles import (
    IMAGE_PLAYER_DIALOG_WIDTH,
    IMAGE_PLAYER_DIALOG_HEIGHT,
    VIDEO_PLAYER_DIALOG_WIDTH,
    VIDEO_PLAYER_DIALOG_HEIGHT,
)

from .popup_dialog_base import PopupDialogBase
from .unified_media_utils import is_image_path, make_image_control, make_video_control, _is_web_platform
from .image_editor_bridge import open_in_image_editor as _bridge_open_editor, open_in_photoshop_stacked as _bridge_open_stacked
from . import image_player_utils as ipu
from . import video_player_utils as vpu
from . import image_editor
from . import video_editor
from flet_app.ui._styles import create_textfield, BTN_STYLE2
from .unified_context_menu import build_context_menu
from .area_editor import (
    apply_clean_from_overlay as area_apply_clean,
    apply_crop_from_overlay as area_apply_crop,
    toggle_area_editor as area_toggle_editor,
)

# Import for thumbnail updates - using a different approach to avoid circular imports
def update_thumbnails_callback_factory(page_context):
    """Factory function to create thumbnail update callback without circular imports"""
    try:
        # Import inside the function to avoid circular import issues
        from flet_app.ui.dataset_manager.dataset_layout_tab import update_thumbnails, thumbnails_grid_ref

        def callback():
            if update_thumbnails and thumbnails_grid_ref and thumbnails_grid_ref.current:
                try:
                    update_thumbnails(page_ctx=page_context, grid_control=thumbnails_grid_ref.current, force_refresh=True)

                    # Also explicitly update the grid control
                    if thumbnails_grid_ref.current.page:
                        thumbnails_grid_ref.current.update()

                except Exception:
                    pass  # Silently handle thumbnail update errors

        return callback
    except ImportError:

        def fallback_callback():
            pass  # Silently handle import failure

        return fallback_callback


DIALOG_WIDTH = max(IMAGE_PLAYER_DIALOG_WIDTH, VIDEO_PLAYER_DIALOG_WIDTH)
# Match legacy dialog height and leave extra space for prompts + tools (no scrollbar)
DIALOG_BASE_HEIGHT = max(IMAGE_PLAYER_DIALOG_HEIGHT, VIDEO_PLAYER_DIALOG_HEIGHT)
DIALOG_HEIGHT = DIALOG_BASE_HEIGHT + 320


def _is_wsl() -> bool:
    try:
        # Detect WSL by platform release containing 'microsoft'
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def _wsl_to_windows_path(path: str) -> str:
    try:
        out = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
        return out
    except Exception:
        return path


def _open_in_image_editor(page: ft.Page, image_path: str, mode: str = "auto"):
    # Delegate to bridge implementation
    try:
        _bridge_open_editor(page, image_path)
    except Exception as ex:
        print(f"Failed to open editor: {ex}")


def _open_in_photoshop_stacked(page: ft.Page, original_path: str, control_path: str):
    # Delegate to bridge implementation
    try:
        _bridge_open_stacked(page, original_path, control_path)
    except Exception as ex:
        print(f"Failed to open stacked images: {ex}")


    # JSX builder moved to image_editor_bridge to keep code DRY


def open_unified_popup_dialog(
    page: ft.Page,
    media_path: str,
    media_list: Optional[List[str]] = None,
):
    """
    Minimal unified popup with top UI only:
    - < and > navigation
    - Photoshop icon (enabled for images only)
    - X close (provided by base dialog)
    No bottom UI, no caption fields.
    """
    if not media_path:
        return

    items: List[str] = list(media_list) if isinstance(media_list, list) and media_list else [media_path]
    try:
        index = items.index(media_path)
    except ValueError:
        index = 0

    # Content container to swap media view
    content_container = ft.Container(width=DIALOG_WIDTH - 40)
    # Playback monitor thread state
    monitor_thread: Optional[threading.Thread] = None
    monitor_running_flag = {"run": False}

    def _stop_monitor_if_any():
        nonlocal monitor_thread
        monitor_running_flag["run"] = False
        if monitor_thread and monitor_thread.is_alive():
            try:
                monitor_thread.join(timeout=0.3)
            except Exception:
                pass
            finally:
                monitor_thread = None

    # Caption fields and timers (kept across refresh while dialog open)
    caption_tf: Optional[ft.TextField] = None
    neg_caption_tf: Optional[ft.TextField] = None
    caption_timer: Optional[threading.Timer] = None
    neg_caption_timer: Optional[threading.Timer] = None

    dialog = page.base_dialog if hasattr(page, "base_dialog") and page.base_dialog else PopupDialogBase(page, content=ft.Container())
    if not hasattr(page, "base_dialog") or not page.base_dialog:
        page.base_dialog = dialog

    # Helper functions to manage page-level hotkey flags cleanly
    def _clear_dialog_hotkey_state():
        try:
            if hasattr(page, 'video_dialog_hotkey_handler'):
                page.video_dialog_hotkey_handler = None
            if hasattr(page, 'video_dialog_open'):
                page.video_dialog_open = False
            if hasattr(page, 'image_dialog_hotkey_handler'):
                page.image_dialog_hotkey_handler = None
            if hasattr(page, 'image_dialog_open'):
                page.image_dialog_open = False
        except Exception:
            pass

    def _set_dialog_hotkey_state(is_image: bool, handler):
        # Reset both first, then set appropriate pair
        _clear_dialog_hotkey_state()
        try:
            if is_image:
                page.image_dialog_hotkey_handler = handler
                page.image_dialog_open = True
            else:
                page.video_dialog_hotkey_handler = handler
                page.video_dialog_open = True
        except Exception:
            pass

    # Overlay editor state
    overlay_visible: bool = False
    overlay_visual: Optional[ft.Container] = None
    overlay_control: Optional[ft.GestureDetector] = None
    overlay_hover_container: Optional[ft.Container] = None
    rotate_hint_badge: Optional[ft.Container] = None
    mouse_over_area: bool = False
    overlay_angle: float = 0.0
    rotation_mode: bool = False
    rotation_center = {"x": 0.0, "y": 0.0}
    rotation_offset: float = 0.0
    # Persisted overlay geometry across media switches
    overlay_saved = {"left": None, "top": None, "w": None, "h": None}

    overlay_pan_start = {"x": 0.0, "y": 0.0}
    overlay_initial = {"left": 0.0, "top": 0.0, "w": 0.0, "h": 0.0}
    overlay_interaction_mode: str = "none"  # 'pan' or resize_tl/tr/bl/br
    overlay_corner_threshold = 16
    overlay_bounds = {"w": 0, "h": 0}
    overlay_parent_stack = {"stack": None}  # set to media_stack_inner when built
    overlay_drag_active = {"val": False}
    video_restart_guard = {"pending": False}
    overlay_current_rotate = {"rot": None}
    overlay_resize_state = {"pivot_local": (0.0, 0.0), "handle_local": (0.0, 0.0), "center_x": 0.0, "center_y": 0.0}

    def _on_overlay_pan_start(e: ft.DragStartEvent):
        nonlocal overlay_control, overlay_interaction_mode
        if not overlay_control:
            return
        overlay_drag_active["val"] = True
        overlay_pan_start["x"] = e.global_x
        overlay_pan_start["y"] = e.global_y
        cw = float(overlay_control.width or 0)
        ch = float(overlay_control.height or 0)
        overlay_initial["left"] = float(overlay_control.left or 0)
        overlay_initial["top"] = float(overlay_control.top or 0)
        overlay_initial["w"] = cw
        overlay_initial["h"] = ch
        # Decide interaction: near corners -> resize, otherwise pan
        try:
            lx = float(getattr(e, 'local_x', 0) or 0)
            ly = float(getattr(e, 'local_y', 0) or 0)
        except Exception:
            lx = ly = 0.0
        # Transform mouse coordinates to account for rotation
        try:
            import math
            # Get rectangle center
            cx = cw / 2.0
            cy = ch / 2.0
            # Apply inverse rotation to get coordinates in unrotated rectangle space
            ca = math.cos(-overlay_angle)
            sa = math.sin(-overlay_angle)
            # Transform to relative coordinates
            rx = (lx - cx) * ca - (ly - cy) * sa
            ry = (lx - cx) * sa + (ly - cy) * ca
        except Exception:
            rx, ry = lx - cw/2.0, ly - ch/2.0
        t = overlay_corner_threshold
        # Check corner detection with relative coordinates
        if rx < -(cw/2.0) + t and ry < -(ch/2.0) + t:
            overlay_interaction_mode = 'resize_tl'
        elif rx > (cw/2.0) - t and ry < -(ch/2.0) + t:
            overlay_interaction_mode = 'resize_tr'
        elif rx < -(cw/2.0) + t and ry > (ch/2.0) - t:
            overlay_interaction_mode = 'resize_bl'
        elif rx > (cw/2.0) - t and ry > (ch/2.0) - t:
            overlay_interaction_mode = 'resize_br'
        else:
            overlay_interaction_mode = 'pan'
        try:
            half_w = overlay_initial["w"] / 2.0
            half_h = overlay_initial["h"] / 2.0
            center_x0 = overlay_initial["left"] + half_w
            center_y0 = overlay_initial["top"] + half_h
            if overlay_interaction_mode.startswith('resize'):
                sign_x = 1.0 if rx >= 0 else -1.0
                sign_y = 1.0 if ry >= 0 else -1.0
                handle_local = (sign_x * half_w, sign_y * half_h)
                pivot_local = (-handle_local[0], -handle_local[1])
                overlay_resize_state["pivot_local"] = pivot_local
                overlay_resize_state["handle_local"] = handle_local
            else:
                overlay_resize_state["pivot_local"] = (0.0, 0.0)
                overlay_resize_state["handle_local"] = (0.0, 0.0)
            overlay_resize_state["center_x"] = center_x0
            overlay_resize_state["center_y"] = center_y0
        except Exception:
            pass

    def _on_overlay_pan_update(e: ft.DragUpdateEvent):
        nonlocal overlay_control, overlay_visual, overlay_interaction_mode
        if not overlay_control:
            return
        dx = float(e.global_x - overlay_pan_start["x"])
        dy = float(e.global_y - overlay_pan_start["y"])
        mode = overlay_interaction_mode
        init_left = overlay_initial["left"]
        init_top = overlay_initial["top"]
        init_w = overlay_initial["w"]
        init_h = overlay_initial["h"]
        new_left = init_left
        new_top = init_top
        new_w = init_w
        new_h = init_h
        if mode == 'pan':
            new_left += dx
            new_top += dy
        elif mode.startswith('resize'):
            try:
                import math
                ca = math.cos(-overlay_angle)
                sa = math.sin(-overlay_angle)
                local_dx = dx * ca - dy * sa
                local_dy = dx * sa + dy * ca
            except Exception:
                local_dx, local_dy = dx, dy
            pivot_local_x, pivot_local_y = overlay_resize_state.get("pivot_local", (0.0, 0.0))
            handle_local_x, handle_local_y = overlay_resize_state.get("handle_local", (0.0, 0.0))
            center_x0 = overlay_resize_state.get("center_x", init_left + init_w / 2.0)
            center_y0 = overlay_resize_state.get("center_y", init_top + init_h / 2.0)
            new_handle_x = handle_local_x + local_dx
            new_handle_y = handle_local_y + local_dy
            handle_sign_x = -1.0 if handle_local_x < 0 else 1.0
            handle_sign_y = -1.0 if handle_local_y < 0 else 1.0
            min_size = 20.0
            if handle_sign_x < 0:
                width_val = pivot_local_x - new_handle_x
                if width_val < min_size:
                    width_val = min_size
                    new_handle_x = pivot_local_x - width_val
            else:
                width_val = new_handle_x - pivot_local_x
                if width_val < min_size:
                    width_val = min_size
                    new_handle_x = pivot_local_x + width_val
            if handle_sign_y < 0:
                height_val = pivot_local_y - new_handle_y
                if height_val < min_size:
                    height_val = min_size
                    new_handle_y = pivot_local_y - height_val
            else:
                height_val = new_handle_y - pivot_local_y
                if height_val < min_size:
                    height_val = min_size
                    new_handle_y = pivot_local_y + height_val
            new_w = max(min_size, abs(width_val))
            new_h = max(min_size, abs(height_val))
            new_center_local_x = (pivot_local_x + new_handle_x) / 2.0
            new_center_local_y = (pivot_local_y + new_handle_y) / 2.0
            try:
                import math
                ca_fwd = math.cos(overlay_angle)
                sa_fwd = math.sin(overlay_angle)
            except Exception:
                ca_fwd = 1.0
                sa_fwd = 0.0
            delta_cx = new_center_local_x * ca_fwd - new_center_local_y * sa_fwd
            delta_cy = new_center_local_x * sa_fwd + new_center_local_y * ca_fwd
            new_center_x = center_x0 + delta_cx
            new_center_y = center_y0 + delta_cy
            new_left = new_center_x - new_w / 2.0
            new_top = new_center_y - new_h / 2.0
        else:
            new_left += dx
            new_top += dy
        vw = float(overlay_bounds["w"]) or 0
        vh = float(overlay_bounds["h"]) or 0
        if vw > 0 and vh > 0:
            if new_left < 0:
                new_left = 0
            if new_top < 0:
                new_top = 0
            if new_left + new_w > vw:
                if mode in ('pan', 'resize_tr', 'resize_br'):
                    new_left = vw - new_w
                else:
                    new_w = max(20.0, vw - new_left)
            if new_top + new_h > vh:
                if mode in ('pan', 'resize_bl', 'resize_br'):
                    new_top = vh - new_h
                else:
                    new_h = max(20.0, vh - new_top)
        overlay_control.left = new_left
        overlay_control.top = new_top
        overlay_control.width = new_w
        overlay_control.height = new_h
        overlay_saved["left"] = new_left
        overlay_saved["top"] = new_top
        overlay_saved["w"] = new_w
        overlay_saved["h"] = new_h
        try:
            if overlay_visual is not None:
                try:
                    overlay_visual.width = new_w
                    overlay_visual.height = new_h
                except Exception:
                    pass
            if overlay_control is not None and overlay_control.page:
                overlay_control.update()
                parent = overlay_parent_stack.get("stack") if isinstance(overlay_parent_stack, dict) else None
                if parent is not None and parent.page:
                    parent.update()
        except Exception:
            pass
    def _on_overlay_pan_end(e: ft.DragEndEvent):
        nonlocal overlay_interaction_mode
        overlay_interaction_mode = 'none'
        overlay_drag_active["val"] = False

    # Context menu state inside media stack
    context_menu_ctrl: Optional[ft.Container] = None
    def _ensure_overlay(view_w: int, view_h: int):
        nonlocal overlay_visual, overlay_control, overlay_hover_container, overlay_current_rotate
        HANDLE_SIZE = 16

        def _clamp_overlay():
            if not overlay_control:
                return
            l = float(overlay_control.left or 0)
            t = float(overlay_control.top or 0)
            w = float(overlay_control.width or 0)
            h = float(overlay_control.height or 0)
            if l < 0: l = 0
            if t < 0: t = 0
            if l + w > view_w: l = max(0.0, view_w - w)
            if t + h > view_h: t = max(0.0, view_h - h)
            overlay_control.left = l
            overlay_control.top = t

        initial_rotate = None
        rotate_cls = None
        if overlay_visual is None:
            base_rect = ft.Container(
                border=ft.border.all(2, ft.Colors.RED_ACCENT_700),
                bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.WHITE),
            )

            resize_start = {"l": 0.0, "t": 0.0, "w": 0.0, "h": 0.0, "gx": 0.0, "gy": 0.0}

            def _rs_start(e: ft.DragStartEvent):
                if not overlay_control:
                    return
                resize_start["gx"] = e.global_x
                resize_start["gy"] = e.global_y
                resize_start["l"] = float(overlay_control.left or 0)
                resize_start["t"] = float(overlay_control.top or 0)
                resize_start["w"] = float(overlay_control.width or 0)
                resize_start["h"] = float(overlay_control.height or 0)

            def _apply_bounds_and_update():
                _clamp_overlay()
                try:
                    if getattr(overlay_control, 'page', None):
                        overlay_control.update()
                except Exception:
                    pass

            def _rs_update_tl(e: ft.DragUpdateEvent):
                if not overlay_control:
                    return
                dx = e.global_x - resize_start["gx"]
                dy = e.global_y - resize_start["gy"]
                try:
                    import math
                    ca = math.cos(-overlay_angle)
                    sa = math.sin(-overlay_angle)
                    rdx = dx * ca - dy * sa
                    rdy = dx * sa + dy * ca
                except Exception:
                    rdx, rdy = dx, dy
                new_l = resize_start["l"] + rdx
                new_t = resize_start["t"] + rdy
                new_w = resize_start["w"] - rdx
                new_h = resize_start["h"] - rdy
                if new_w < 20:
                    new_l = resize_start["l"] + (resize_start["w"] - 20)
                    new_w = 20
                if new_h < 20:
                    new_t = resize_start["t"] + (resize_start["h"] - 20)
                    new_h = 20
                overlay_control.left = new_l
                overlay_control.top = new_t
                overlay_control.width = new_w
                overlay_control.height = new_h
                _apply_bounds_and_update()

            def _rs_update_tr(e: ft.DragUpdateEvent):
                if not overlay_control:
                    return
                dx = e.global_x - resize_start["gx"]
                dy = e.global_y - resize_start["gy"]
                try:
                    import math
                    ca = math.cos(-overlay_angle)
                    sa = math.sin(-overlay_angle)
                    rdx = dx * ca - dy * sa
                    rdy = dx * sa + dy * ca
                except Exception:
                    rdx, rdy = dx, dy
                new_t = resize_start["t"] + rdy
                new_w = resize_start["w"] + rdx
                new_h = resize_start["h"] - rdy
                if new_w < 20:
                    new_w = 20
                if new_h < 20:
                    new_t = resize_start["t"] + (resize_start["h"] - 20)
                    new_h = 20
                overlay_control.top = new_t
                overlay_control.width = new_w
                overlay_control.height = new_h
                _apply_bounds_and_update()

            def _rs_update_bl(e: ft.DragUpdateEvent):
                if not overlay_control:
                    return
                dx = e.global_x - resize_start["gx"]
                dy = e.global_y - resize_start["gy"]
                try:
                    import math
                    ca = math.cos(-overlay_angle)
                    sa = math.sin(-overlay_angle)
                    rdx = dx * ca - dy * sa
                    rdy = dx * sa + dy * ca
                except Exception:
                    rdx, rdy = dx, dy
                new_l = resize_start["l"] + rdx
                new_w = resize_start["w"] - rdx
                new_h = resize_start["h"] + rdy
                if new_w < 20:
                    new_l = resize_start["l"] + (resize_start["w"] - 20)
                    new_w = 20
                if new_h < 20:
                    new_h = 20
                overlay_control.left = new_l
                overlay_control.width = new_w
                overlay_control.height = new_h
                _apply_bounds_and_update()

            def _rs_update_br(e: ft.DragUpdateEvent):
                if not overlay_control:
                    return
                dx = e.global_x - resize_start["gx"]
                dy = e.global_y - resize_start["gy"]
                try:
                    import math
                    ca = math.cos(-overlay_angle)
                    sa = math.sin(-overlay_angle)
                    rdx = dx * ca - dy * sa
                    rdy = dx * sa + dy * ca
                except Exception:
                    rdx, rdy = dx, dy
                new_w = resize_start["w"] + rdx
                new_h = resize_start["h"] + rdy
                if new_w < 20:
                    new_w = 20
                if new_h < 20:
                    new_h = 20
                overlay_control.width = new_w
                overlay_control.height = new_h
                _apply_bounds_and_update()

            def _make_handle(on_update):
                return ft.GestureDetector(
                    content=ft.Container(width=HANDLE_SIZE, height=HANDLE_SIZE, bgcolor=ft.Colors.RED_700, border_radius=ft.border_radius.all(2)),
                    on_pan_start=_rs_start,
                    on_pan_update=on_update,
                    drag_interval=0,
                )

            # Rotational hint badge control (text only, no dark background)
            nonlocal rotate_hint_badge
            rotate_hint_badge = ft.Container(
                content=ft.Text("⭮", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                alignment=ft.alignment.top_center,
                padding=6,
                margin=ft.margin.only(top=4),
                # No background color for a clean look
                visible=False,
            )

            overlay_visual = ft.Stack(
                [
                    base_rect,
                    ft.Container(_make_handle(_rs_update_tl), alignment=ft.alignment.top_left, width=0, height=0),
                    ft.Container(_make_handle(_rs_update_tr), alignment=ft.alignment.top_right, width=0, height=0),
                    ft.Container(_make_handle(_rs_update_bl), alignment=ft.alignment.bottom_left, width=0, height=0),
                    ft.Container(_make_handle(_rs_update_br), alignment=ft.alignment.bottom_right, width=0, height=0),
                    # Rotational hint badge placed on top of the area (shown only when mouse is outside)
                    rotate_hint_badge,
                ],
                width=min(200, view_w),
                height=min(200, view_h),
                visible=overlay_visible,
            )
            rotate_cls = None
            initial_rotate = None
            try:
                rotate_cls = getattr(getattr(ft, 'transform', None), 'Rotate', None) or getattr(ft, 'Rotate', None)
                if rotate_cls is not None:
                    initial_rotate = rotate_cls(angle=overlay_angle, alignment=ft.alignment.center)
            except Exception:
                rotate_cls = None
                initial_rotate = None
        if overlay_control is None:
            # Determine initial geometry using saved values if present
            sw = overlay_saved.get("w")
            sh = overlay_saved.get("h")
            sl = overlay_saved.get("left")
            st = overlay_saved.get("top")
            init_w = int(sw) if isinstance(sw, (int, float)) and sw and sw > 0 else int(min(200, view_w))
            init_h = int(sh) if isinstance(sh, (int, float)) and sh and sh > 0 else int(min(200, view_h))
            init_left = float(sl) if isinstance(sl, (int, float)) else (view_w - init_w) / 2
            init_top = float(st) if isinstance(st, (int, float)) else (view_h - init_h) / 2
            # Clamp to bounds
            if init_left < 0: init_left = 0
            if init_top < 0: init_top = 0
            if init_left + init_w > view_w: init_left = max(0.0, view_w - init_w)
            if init_top + init_h > view_h: init_top = max(0.0, view_h - init_h)
            try:
                overlay_visual.width = init_w
                overlay_visual.height = init_h
            except Exception:
                pass
            def _on_overlay_hover(e: ft.HoverEvent):
                nonlocal mouse_over_area
                mouse_over_area = (str(getattr(e, 'data', '')).lower() == 'true')
                try:
                    if rotate_hint_badge is not None:
                        rotate_hint_badge.visible = bool(overlay_visible and not mouse_over_area)
                        if getattr(rotate_hint_badge, 'page', None):
                            rotate_hint_badge.update()
                except Exception:
                    pass

            overlay_hover_container = ft.Container(
                content=overlay_visual,
                bgcolor=ft.Colors.TRANSPARENT,  # Transparent area for overlay
                on_hover=_on_overlay_hover,
            )
            overlay_control = ft.GestureDetector(
                content=overlay_hover_container,
                left=init_left,
                top=init_top,
                width=init_w,
                height=init_h,
                on_pan_start=_on_overlay_pan_start,
                on_pan_update=_on_overlay_pan_update,
                on_pan_end=_on_overlay_pan_end,
                drag_interval=0,
                visible=overlay_visible,
                mouse_cursor=getattr(ft.MouseCursor, 'MOVE', None),
            )
            overlay_current_rotate["rot"] = initial_rotate
            if initial_rotate is not None:
                try:
                    overlay_control.rotate = initial_rotate
                except Exception:
                    pass

    def refresh():
        nonlocal index
        nonlocal caption_tf, neg_caption_tf, caption_timer, neg_caption_timer
        nonlocal overlay_visible, overlay_visual, overlay_control, overlay_hover_container, overlay_current_rotate
        nonlocal monitor_thread
        _stop_monitor_if_any()
        path = items[index]
        title = os.path.basename(path)
        is_img = is_image_path(path)
        # Debug print removed

        # Build top controls
        prev_btn = ft.IconButton(ft.Icons.ARROW_LEFT, tooltip="Previous", on_click=lambda e: go(-1))
        next_btn = ft.IconButton(ft.Icons.ARROW_RIGHT, tooltip="Next", on_click=lambda e: go(1))
        ps_btn = ft.IconButton(
            ft.Icons.EDIT,
            tooltip="Open in editor",
            visible=is_img,
            on_click=lambda e: _open_in_image_editor(page, path),
        )
        # Placeholder switch button; visibility and handler set later after control detection
        switch_top_btn = ft.IconButton(
            ft.Icons.COMPARE,
            tooltip="Show control",
            visible=False,
        )
        prefix_controls = [prev_btn, next_btn, ps_btn, switch_top_btn]  # may be overridden for video

        # Build media view
        viewer_w = DIALOG_WIDTH - 40
        viewer_h = max(IMAGE_PLAYER_DIALOG_HEIGHT, VIDEO_PLAYER_DIALOG_HEIGHT) - 40
        overlay_bounds["w"], overlay_bounds["h"] = viewer_w, viewer_h
        # Initialize caption defaults to avoid UnboundLocalError
        cap_text = ""
        neg_text = ""

        if is_img:
            media_view = make_image_control(path, width=viewer_w, height=viewer_h, page=page)
            cap_text, neg_text, _ = ipu.load_caption_for_image(path)
        else:            # Use interactive video player again (controls hidden) to allow playback
            media_view = make_video_control(path, width=viewer_w, height=viewer_h, page=page)
            # Add play/pause control for video
            play_pause_btn = ft.IconButton(ft.Icons.PAUSE, tooltip='Play/Pause')
            def _toggle_play(e=None):
                try:
                    if media_view.is_playing():
                        media_view.pause(); play_pause_btn.icon = ft.Icons.PLAY_ARROW
                    else:
                        media_view.play(); play_pause_btn.icon = ft.Icons.PAUSE
                    if play_pause_btn.page: play_pause_btn.update()
                except Exception:
                    pass
            play_pause_btn.on_click = _toggle_play
            prefix_controls = [prev_btn, next_btn, play_pause_btn]

        # Load video captions if needed (guarded)
        if not is_img and (not cap_text and not neg_text):
            try:
                _c, _n, _ = vpu.load_caption_for_video(path)
                cap_text, neg_text = _c, _n
            except Exception:
                cap_text, neg_text = "", ""

        # Build or update caption fields
        if caption_tf is None:
            caption_tf = create_textfield(
                label="Caption",
                value=cap_text,
                expand=True,
                multiline=True,
                min_lines=3,
                max_lines=5,
            )
        else:
            caption_tf.value = cap_text

        if neg_caption_tf is None:
            neg_caption_tf = create_textfield(
                label="Negative Caption",
                value=neg_text,
                expand=True,
                multiline=True,
                min_lines=3,
                max_lines=5,
            )
        else:
            neg_caption_tf.value = neg_text

        # Debounced save handlers
        def on_caption_change(e: ft.ControlEvent):
            nonlocal caption_timer
            if caption_timer:
                caption_timer.cancel()
            def _save():
                txt = caption_tf.value.strip() if caption_tf and caption_tf.value else ""
                if is_img:
                    ipu.save_caption_for_image(path, txt, 'caption')
                else:
                    vpu.save_caption_for_video(path, txt, 'caption')
            caption_timer = threading.Timer(0.5, _save)
            caption_timer.start()

        def on_neg_caption_change(e: ft.ControlEvent):
            nonlocal neg_caption_timer
            if neg_caption_timer:
                neg_caption_timer.cancel()
            def _save():
                txt = neg_caption_tf.value.strip() if neg_caption_tf and neg_caption_tf.value else ""
                if is_img:
                    ipu.save_caption_for_image(path, txt, 'negative_caption')
                else:
                    vpu.save_caption_for_video(path, txt, 'negative_caption')
            neg_caption_timer = threading.Timer(0.5, _save)
            neg_caption_timer.start()

        caption_tf.on_change = on_caption_change
        neg_caption_tf.on_change = on_neg_caption_change

        # Build independent overlay layer the same size as media area
        # Recreate overlay instances per media switch to avoid stale attachments
        try:
            # keep saved geometry, reset controls so they get rebuilt
            overlay_visual_ref = overlay_visual
            overlay_control_ref = overlay_control
        except Exception:
            overlay_visual_ref = None; overlay_control_ref = None
        overlay_visual = None
        overlay_control = None
        _ensure_overlay(viewer_w, viewer_h)
        video_container = ft.Container(content=media_view, alignment=ft.alignment.center, width=viewer_w, height=viewer_h)
        overlay_layer_stack = ft.Stack([
            overlay_control if overlay_control else ft.Container(width=0, height=0)
        ], width=viewer_w, height=viewer_h)
        def _rotation_pan_start(e: ft.DragStartEvent):
            nonlocal rotation_mode
            nonlocal rotation_offset
            try:
                if not overlay_visible or overlay_control is None:
                    rotation_mode = False
                    return
                lx = float(getattr(e, 'local_x', 0.0))
                ly = float(getattr(e, 'local_y', 0.0))
                # Determine if start is outside the ROTATED overlay rect
                l = float(overlay_control.left or 0)
                t = float(overlay_control.top or 0)
                w = float(overlay_control.width or 0)
                h = float(overlay_control.height or 0)
                # Rotate point into unrotated rect's frame
                cx = l + w / 2.0
                cy = t + h / 2.0
                import math
                ca = math.cos(-overlay_angle)
                sa = math.sin(-overlay_angle)
                rx = (lx - cx) * ca - (ly - cy) * sa
                ry = (lx - cx) * sa + (ly - cy) * ca
                inside = (-w/2.0 <= rx <= w/2.0) and (-h/2.0 <= ry <= h/2.0)
                rotation_mode = not inside
                if rotation_mode:
                    rotation_center["x"] = cx
                    rotation_center["y"] = cy
                    # Establish offset so rotation continues from current angle without jump
                    angle0 = math.atan2(ly - cy, lx - cx)
                    rotation_offset = overlay_angle - angle0
            except Exception:
                rotation_mode = False

        def _rotation_pan_update(e: ft.DragUpdateEvent):
            nonlocal overlay_angle
            try:
                if not rotation_mode or overlay_visual is None:
                    return
                lx = float(getattr(e, 'local_x', 0.0))
                ly = float(getattr(e, 'local_y', 0.0))
                dx = lx - rotation_center["x"]
                dy = ly - rotation_center["y"]
                import math
                angle = math.atan2(dy, dx)
                overlay_angle = angle + rotation_offset
                rot_cls = getattr(getattr(ft, 'transform', None), 'Rotate', None) or getattr(ft, 'Rotate', None)
                if rot_cls is not None:
                    new_rotate = rot_cls(angle=overlay_angle, alignment=ft.alignment.center)
                    overlay_current_rotate['rot'] = new_rotate
                    if overlay_control is not None:
                        overlay_control.rotate = new_rotate
                        if getattr(overlay_control, 'page', None):
                            overlay_control.update()
                if getattr(overlay_visual, 'page', None):
                    overlay_visual.update()
            except Exception:
                pass

        def _rotation_pan_end(e: ft.DragEndEvent):
            nonlocal rotation_mode
            rotation_mode = False

        overlay_layer_container = ft.GestureDetector(
            # Keep default cursor; visual ⭮ badge indicates rotation
            mouse_cursor=getattr(ft.MouseCursor, 'BASIC', None),
            content=ft.Container(
                content=overlay_layer_stack,
                width=viewer_w,
                height=viewer_h,
                bgcolor=ft.Colors.TRANSPARENT,  # Transparent area for overlay
            ),
            on_pan_start=_rotation_pan_start,
            on_pan_update=_rotation_pan_update,
            on_pan_end=_rotation_pan_end,
            drag_interval=0,
        )
        media_stack_inner = ft.Stack([
            video_container,
            overlay_layer_container,
        ], width=viewer_w, height=viewer_h)
        try:
            overlay_parent_stack["stack"] = overlay_layer_stack
        except Exception:
            pass
        # Sync overlay visual with control and visibility when switching media
        try:
            if overlay_visual is not None and overlay_control is not None:
                try:
                    cw = float(overlay_control.width or 0)
                    ch = float(overlay_control.height or 0)
                    if cw <= 0 or ch <= 0:
                        cw = min(200, viewer_w); ch = min(200, viewer_h)
                        overlay_control.width = cw; overlay_control.height = ch
                    overlay_visual.width = cw
                    overlay_visual.height = ch
                except Exception:
                    pass
                # Preserve current visibility state
                overlay_visual.visible = bool(overlay_visible)
                overlay_control.visible = bool(overlay_visible)
                current_rot = overlay_current_rotate.get('rot') if isinstance(overlay_current_rotate, dict) else None
                if current_rot is not None:
                    try:
                        if overlay_control is not None:
                            overlay_control.rotate = current_rot
                            if getattr(overlay_control, 'page', None):
                                overlay_control.update()
                    except Exception:
                        pass
            # Nudge overlay layer to repaint independently of video
            # No dynamic cursor updates; we show a visual ⭮ badge instead
            if overlay_layer_stack.page:
                overlay_layer_stack.update()
        except Exception:
            pass
        def _open_context_menu(e: ft.ControlEvent):
            nonlocal context_menu_ctrl
            try:
                lx = getattr(e, 'local_x', None)
                ly = getattr(e, 'local_y', None)
                gx = getattr(e, 'global_x', None)
                gy = getattr(e, 'global_y', None)
                # Debug print removed
                if context_menu_ctrl is None:
                    # Build menu content to match popup UI styling
                    def _action_flip(ev):
                        _hide_context_menu()
                        if is_img:
                            page.run_thread(image_editor.handle_flip_image, page, path, items, None)
                        else:
                            page.run_thread(video_editor.on_flip_horizontal, page, path, items, None)
                    def _action_rot_plus(ev):
                        _hide_context_menu()
                        if is_img:
                            page.run_thread(image_editor.handle_rotate_image, page, path, items, None, 90)
                        else:
                            page.run_thread(video_editor.on_rotate_90_video_action, page, path, 'plus', items, None)
                    def _action_rot_minus(ev):
                        _hide_context_menu()
                        if is_img:
                            page.run_thread(image_editor.handle_rotate_image, page, path, items, None, -90)
                        else:
                            page.run_thread(video_editor.on_rotate_90_video_action, page, path, 'minus', items, None)
                    def _action_reverse(ev):
                        _hide_context_menu()
                        if not is_img:
                            page.run_thread(video_editor.on_reverse, page, path, items, None)

# migrated to unified_context_menu: flip_btn = create_styled_button("Flip Horizontal", on_click=_action_flip, button_style=BTN_STYLE2)
# migrated to unified_context_menu: rot_plus_btn = create_styled_button("Rotate +90", on_click=_action_rot_plus, button_style=BTN_STYLE2)
# migrated to unified_context_menu: rot_minus_btn = create_styled_button("Rotate -90", on_click=_action_rot_minus, button_style=BTN_STYLE2)
# migrated to unified_context_menu: reverse_btn = create_styled_button("Reverse", on_click=_action_reverse, button_style=BTN_STYLE2, disabled=is_img)

                    context_menu_ctrl = build_context_menu(
                        is_image=is_img,
                        page=page,
                        media_path=path,
                        media_list=items,
                        on_close=_hide_context_menu, on_refresh=refresh)
                    # Insert into media stack later
                # Position within media area; fall back to center
                cx = lx if isinstance(lx, (int, float)) else viewer_w/2
                cy = ly if isinstance(ly, (int, float)) else viewer_h/2
                # Clamp
                cx = max(0, min(cx, viewer_w-120))
                cy = max(0, min(cy, viewer_h-60))
                context_menu_ctrl.left = cx
                context_menu_ctrl.top = cy
                context_menu_ctrl.visible = True
                if media_stack_inner.page:
                    media_stack_inner.controls.append(context_menu_ctrl) if context_menu_ctrl not in media_stack_inner.controls else None
                    media_stack_inner.update()
            except Exception as ex:
                # Debug print removed
                pass

        def _hide_context_menu():
            nonlocal context_menu_ctrl
            try:
                if context_menu_ctrl is not None:
                    try:
                        if media_stack_inner and context_menu_ctrl in getattr(media_stack_inner, 'controls', []):
                            media_stack_inner.controls.remove(context_menu_ctrl)
                    except Exception:
                        pass
                    context_menu_ctrl.visible = False
                    if context_menu_ctrl.page:
                        context_menu_ctrl.update()
                    context_menu_ctrl = None
            except Exception as ex:
                # Debug print removed
                pass
        # Provide a programmatic way to open context menu from an icon.
        def _open_context_menu_at(cx: float, cy: float):
            class _E:
                def __init__(self, x, y):
                    self.local_x = x
                    self.local_y = y
            _open_context_menu(_E(cx, cy))

        # Remove media-area gestures entirely to avoid interfering with Area Editor.
        media_stack = media_stack_inner

        # Arrange media + caption fields
        fields_row = ft.ResponsiveRow([
            ft.Container(caption_tf, col={'md': 8, 'sm': 12}),
            ft.Container(neg_caption_tf, col={'md': 4, 'sm': 12}),
        ], spacing=10)

        # Scaling and area editor controls — LEFT COLUMN (match legacy layout)
        # Width/Height fields stacked, controls to the right
        width_field = create_textfield(label="Width", value="", col=12, keyboard_type=ft.KeyboardType.NUMBER)
        height_field = create_textfield(label="Height", value="", col=12, keyboard_type=ft.KeyboardType.NUMBER)
        # Fill with closest if available
        try:
            if is_img:
                w_str, h_str = ipu.calculate_closest_div32_dimensions(path)
            else:
                w_str, h_str = vpu.calculate_closest_div32_dimensions(path)
            if w_str and h_str:
                width_field.value = w_str
                height_field.value = h_str
        except Exception:
            pass

        # Size adjust buttons
        add_btn = ft.ElevatedButton("+", on_click=lambda e: (
            image_editor.handle_size_add(width_field, height_field, path, page) if is_img else
            video_editor.handle_size_add(width_field, height_field, path, page)
        ), style=BTN_STYLE2)
        sub_btn = ft.ElevatedButton("-", on_click=lambda e: (
            image_editor.handle_size_sub(width_field, height_field, path, page) if is_img else
            video_editor.handle_size_sub(width_field, height_field, path, page)
        ), style=BTN_STYLE2)

        closest_btn = ft.ElevatedButton("Closest", on_click=lambda e: (
            image_editor.handle_set_closest_div32(width_field, height_field, path, page) if is_img else
            video_editor.handle_set_closest_div32(width_field, height_field, path, page)
        ), style=BTN_STYLE2)

        crop_btn = ft.ElevatedButton("Crop", on_click=lambda e: (
            image_editor.handle_crop_image_click(page, width_field, height_field, path, items, None) if is_img else
            video_editor.handle_crop_video_click(page, width_field, height_field, path, items, None)
        ), style=BTN_STYLE2)

        crop_all_btn = ft.ElevatedButton("Crop All", on_click=lambda e: (
            image_editor.handle_crop_all_images(page, path, width_field, height_field, items, None) if is_img else
            video_editor.handle_crop_all_videos(page, path, width_field, height_field, items, None)
        ), style=BTN_STYLE2)

        def toggle_area_editor(e):
            nonlocal overlay_visible
            nonlocal overlay_layer_container
            overlay_visible, _ = area_toggle_editor(
                page,
                overlay_control,
                overlay_visible,
                overlay_angle,
            )
            if overlay_control:
                # Ensure sane size and position when showing
                if overlay_visible:
                    try:
                        cw = float(overlay_control.width or 0)
                        ch = float(overlay_control.height or 0)
                        if cw < 20 or ch < 20:
                            cw = min(200, overlay_bounds.get("w") or viewer_w)
                            ch = min(200, overlay_bounds.get("h") or viewer_h)
                            overlay_control.width = cw
                            overlay_control.height = ch
                        # recentre if off-bounds
                        if (overlay_control.left is None) or (overlay_control.top is None):
                            overlay_control.left = max(0, (viewer_w - cw) / 2)
                            overlay_control.top = max(0, (viewer_h - ch) / 2)
                    except Exception:
                        pass
                overlay_control.visible = overlay_visible
                try:
                    if getattr(overlay_control, 'page', None):
                        overlay_control.update()
                except Exception:
                    pass
                # Overlay is on its own layer; no z-index juggling needed
            rot_cls = getattr(getattr(ft, 'transform', None), 'Rotate', None) or getattr(ft, 'Rotate', None)
            new_rotate = None
            if rot_cls is not None:
                try:
                    new_rotate = rot_cls(angle=overlay_angle, alignment=ft.alignment.center)
                except Exception:
                    new_rotate = None
            applied_rotate = False
            if new_rotate is not None:
                overlay_current_rotate['rot'] = new_rotate
                if overlay_control is not None:
                    try:
                        overlay_control.rotate = new_rotate
                        if getattr(overlay_control, 'page', None):
                            overlay_control.update()
                    except Exception:
                        pass
                applied_rotate = True
            if not applied_rotate:
                existing_rot = overlay_current_rotate.get('rot') if isinstance(overlay_current_rotate, dict) else None
                if existing_rot is not None and overlay_control is not None:
                    try:
                        overlay_control.rotate = existing_rot
                        if getattr(overlay_control, 'page', None):
                            overlay_control.update()
                    except Exception:
                        pass
            if overlay_visual:
                overlay_visual.visible = overlay_visible
                try:
                    # Keep visual size in sync
                    try:
                        overlay_visual.width = float(overlay_control.width or 0)
                        overlay_visual.height = float(overlay_control.height or 0)
                    except Exception:
                        pass
                    if getattr(overlay_visual, 'page', None):
                        overlay_visual.update()
                except Exception:
                    pass
            # Update rotate badge visibility depending on pointer state
            try:
                if rotate_hint_badge is not None:
                    rotate_hint_badge.visible = bool(overlay_visible and not mouse_over_area)
                    if getattr(rotate_hint_badge, 'page', None):
                        rotate_hint_badge.update()
            except Exception:
                pass
            # No dynamic cursor change; we keep default and rely on ⭮ badge
            # Do NOT rebuild the entire dialog here to avoid restarting video
            try:
                parent = overlay_parent_stack.get("stack") if isinstance(overlay_parent_stack, dict) else None
                if parent is not None and parent.page:
                    parent.update()
            except Exception:
                pass

        def apply_crop_from_overlay(e):
            if area_apply_crop(page, path, overlay_control, overlay_visible, viewer_w, viewer_h, overlay_angle):
                refresh()

        def apply_clean_from_overlay(e):
            if not is_img and area_apply_clean(page, path, overlay_control, overlay_visible, viewer_w, viewer_h):
                refresh()

        area_btn = ft.ElevatedButton("Area Editor", on_click=toggle_area_editor, style=BTN_STYLE2)
        apply_crop_btn = ft.ElevatedButton("Apply Crop", on_click=apply_crop_from_overlay, style=BTN_STYLE2)
        apply_clean_btn = ft.ElevatedButton("Apply Clean", on_click=apply_clean_from_overlay, disabled=is_img, style=BTN_STYLE2, tooltip=("Disabled for images" if is_img else None))

        # First row: fields stacked on the left, +/-/Closest stacked on right
        fields_stack_col = ft.Column([width_field, height_field], spacing=2, col=8)
        adjust_stack_col = ft.Column([add_btn, sub_btn], spacing=2, col=4)

        # Left column layout
        left_col = ft.Column([
            ft.ResponsiveRow([fields_stack_col, adjust_stack_col], spacing=2, expand=True),
            ft.ResponsiveRow([
                ft.Container(crop_all_btn, col=5),
                ft.Container(crop_btn, col=3),
                ft.Container(closest_btn, col=4),
            ], spacing=1, expand=True),
            ft.Divider(thickness=1, height=4),
            ft.ResponsiveRow([ft.Container(area_btn, col=6), ft.Container(apply_crop_btn, col=6)], spacing=3, expand=True),
            ft.ResponsiveRow([ft.Container(apply_clean_btn, col=12)], spacing=3, expand=True),
        ], spacing=6, col={'md': 6, 'lg': 5, 'sm': 12})

                # Video playback slider (Start/Total/End) occupies two columns on the right for videos
        if not is_img:
            metadata = vpu.get_video_metadata(path) or {}
            original_frames = int(metadata.get('total_frames', 100) or 100)
            fps = float(metadata.get('fps', 30.0) or 30.0)

            start_value_text = ft.Text("Start: 0", size=12)
            total_frames_text = ft.Text(f"Total: {original_frames}", size=12)
            end_value_text = ft.Text(f"End: {original_frames}", size=12)

            frame_range_slider = ft.RangeSlider(
                min=0,
                max=original_frames,
                start_value=0,
                end_value=original_frames,
                divisions=original_frames if original_frames > 0 else None,
                label="{value}",
                round=0,
                expand=True,
            )

            local_video_player = media_view if hasattr(media_view, 'seek') else None

            def _on_slider_change(e_slider: ft.ControlEvent):
                try:
                    s = int(frame_range_slider.start_value or 0)
                    en = int(frame_range_slider.end_value or original_frames)
                    start_value_text.value = f"Start: {s}"
                    end_value_text.value = f"End: {en}"
                    total_frames_text.value = f"Total: {max(0, en - s)}"
                    if start_value_text.page:
                        start_value_text.update()
                    if end_value_text.page:
                        end_value_text.update()
                    if total_frames_text.page:
                        total_frames_text.update()
                except Exception:
                    pass

            def _on_slider_change_end(e_slider: ft.ControlEvent):
                try:
                    s = int(frame_range_slider.start_value or 0)
                    if local_video_player and fps > 0:
                        ms = int((s / fps) * 1000)
                        local_video_player.seek(ms)
                        local_video_player.play()
                        if local_video_player.page:
                            local_video_player.update()
                except Exception:
                    pass

            frame_range_slider.on_change = _on_slider_change
            frame_range_slider.on_change_end = _on_slider_change_end


            def _on_completed(e):
                try:
                    s = int(frame_range_slider.start_value or 0)
                    if not local_video_player or fps <= 0:
                        return
                    ms = int((s / fps) * 1000)
                    # Always loop immediately; Area Editor runs on its own layer and repaint path
                    try:
                        is_playing_now = bool(local_video_player.is_playing())
                    except Exception:
                        is_playing_now = True
                    local_video_player.seek(ms)
                    if is_playing_now:
                        local_video_player.play()
                    if local_video_player.page:
                        local_video_player.update()
                    # Overlay lives on separate layer; nothing to re-order
                except Exception:
                    pass

            try:
                if local_video_player is not None:
                    local_video_player.on_completed = _on_completed
            except Exception:
                pass


            slider_col = ft.Column(
                [
                    frame_range_slider,
                    ft.Row(
                        [start_value_text, total_frames_text, end_value_text],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=5,
                col={"md": 6, "lg": 7, "sm": 12},
            )
            # Create thumbnail update callback using factory to avoid circular imports
            update_thumbnails_callback = update_thumbnails_callback_factory(page)

            # Row 2: Split, Cut to Frames, Cut All Videos to, Num field
            num_to_cut_to = create_textfield(label="num", value=str(original_frames // 2 if original_frames > 1 else 150), keyboard_type=ft.KeyboardType.NUMBER)
            split_btn = ft.ElevatedButton("Split", on_click=lambda e: page.run_thread(video_editor.split_to_video, page, path, int(frame_range_slider.start_value or 0), items, None, local_video_player, refresh, update_thumbnails_callback), style=BTN_STYLE2)
            cut_to_frames_btn = ft.ElevatedButton("Cut to Frames", on_click=lambda e: page.run_thread(video_editor.cut_to_frames, page, path, int(frame_range_slider.start_value or 0), int(frame_range_slider.end_value or original_frames), items, None, refresh, update_thumbnails_callback), style=BTN_STYLE2)
            cut_all_btn = ft.ElevatedButton("Cut All Videos to", on_click=lambda e: page.run_thread(video_editor.cut_all_videos_to_max, page, path, items, int(num_to_cut_to.value or 0), None), style=BTN_STYLE2)
            row2 = ft.ResponsiveRow([
                ft.Container(split_btn, col={"md": 3, "lg": 3, "sm": 12}),
                ft.Container(cut_to_frames_btn, col={"md": 3, "lg": 3, "sm": 12}),
                ft.Container(cut_all_btn, col={"md": 4, "lg": 4, "sm": 12}),
                ft.Container(num_to_cut_to, col={"md": 2, "lg": 2, "sm": 12}),
            ], spacing=6, expand=True)

            # Row 3: Remap label (20%), Slider (60%), Time Remap button (20%)
            time_remap_value_text = ft.Text(str(original_frames), size=12)
            def _on_time_slider_change(e_slider: ft.ControlEvent):
                try:
                    val = float(getattr(e_slider.control, "value", 1.0) or 1.0)
                    frames = int(original_frames / val) if val > 0 else original_frames
                    time_remap_value_text.value = f"{frames}"
                    if time_remap_value_text.page:
                        time_remap_value_text.update()
                except Exception:
                    pass
            time_slider = ft.Slider(min=0.1, max=2.0, value=1.0, divisions=190, label="{value}x Speed", round=1, expand=True, on_change=_on_time_slider_change)
            time_remap_btn = ft.ElevatedButton("Time Remap", on_click=lambda e: page.run_thread(video_editor.on_time_remap, page, path, float(time_slider.value or 1.0), items, None), style=BTN_STYLE2)
            row3 = ft.ResponsiveRow([
                ft.Container(time_remap_value_text, col={"md": 2, "lg": 2, "sm": 12}, alignment=ft.alignment.center, padding=ft.padding.only(top=12, left=8)),
                ft.Container(time_slider, col={"md": 7, "lg": 7, "sm": 12}),
                ft.Container(time_remap_btn, col={"md": 3, "lg": 3, "sm": 12}, padding=ft.padding.only(top=8)),
            ], spacing=6, expand=True)

            slider_col.controls.append(row2)
            slider_col.controls.append(row3)
        else:
            # Create image info panel for images
            # Control detection is now handled in the switch state logic below

            # Initialize default values
            control_status = "No"
            control_color = ft.Colors.RED_600
            has_control = False

            # Store switching state at function level to persist across refreshes
            if not hasattr(refresh, '_switch_state'):
                refresh._switch_state = {}

            # Find the true original path (not a control image)
            true_original_path = path
            if "/control/" in path:
                # Extract the original path from control image path
                true_original_path = path.replace("/control/", "/").replace("//", "/")

            # Always use the true original image path for state management
            base_image_path = true_original_path
            # Helper to set image control source across desktop/web
            def _set_image_src(img_ctrl, target_path: str):
                try:
                    normalized = target_path.replace("\\", "/")
                    # If this Image was created with base64 (web), keep using base64
                    if hasattr(img_ctrl, 'src_base64') and img_ctrl.src_base64 is not None:
                        try:
                            with open(target_path, 'rb') as f:
                                import base64 as _b64
                                encoded = _b64.b64encode(f.read()).decode('utf-8')
                            img_ctrl.src = None
                            img_ctrl.src_base64 = encoded
                        except Exception:
                            # Fallback to file path if encoding fails
                            img_ctrl.src_base64 = None
                            img_ctrl.src = normalized
                    else:
                        # Desktop/native path switching
                        if hasattr(img_ctrl, 'src_base64'):
                            img_ctrl.src_base64 = None
                        img_ctrl.src = normalized
                    if img_ctrl.page:
                        img_ctrl.update()
                except Exception:
                    pass
            if base_image_path in refresh._switch_state:
                # If we already have state for this base image, retrieve it
                current_state = refresh._switch_state[base_image_path]
                original_path = current_state['original_path']
                control_image_path = current_state['control_image_path']
                is_showing_control = current_state['is_showing_control']
                # Preserve the original has_control status
                has_control = current_state['has_control']
                control_status = "Yes" if has_control else "No"
                control_color = ft.Colors.GREEN_600 if has_control else ft.Colors.RED_600
                # If state says we are showing control, ensure media shows it
                try:
                    if is_showing_control and os.path.exists(control_image_path):
                        _set_image_src(media_view, control_image_path)
                except Exception:
                    pass
            else:
                # Initialize state for this base image
                original_path = true_original_path
                control_image_path = os.path.join(os.path.dirname(true_original_path), "control", os.path.basename(true_original_path))
                is_showing_control = False

                # Check for control folder and matching image - only once!
                has_control = False
                try:
                    image_dir = os.path.dirname(original_path)
                    image_filename = os.path.basename(original_path)
                    control_folder = os.path.join(image_dir, "control")

                    if os.path.exists(control_folder) and os.path.isdir(control_folder):
                        control_check_path = os.path.join(control_folder, image_filename)
                        has_control = os.path.exists(control_check_path)
                except Exception:
                    has_control = False

                # Update status based on detection
                control_status = "Yes" if has_control else "No"
                control_color = ft.Colors.GREEN_600 if has_control else ft.Colors.RED_600

                # Store the original has_control status
                refresh._switch_state[base_image_path] = {
                    'is_showing_control': is_showing_control,
                    'original_path': original_path,
                    'control_image_path': control_image_path,
                    'has_control': has_control  # Store the original control status
                }

            # Legacy overlay storage (no longer used for switching). Keeping guard to avoid attribute errors
            if not hasattr(refresh, '_control_overlays'):
                refresh._control_overlays = {}

            def on_switch_click(e):
                try:
                    current_state = refresh._switch_state[base_image_path]
                    # Debug logging removed

                    if current_state['is_showing_control']:
                        # Switch back to original image (no overlay)
                        # Swapping back to base image
                        refresh._switch_state[base_image_path]['is_showing_control'] = False
                        try:
                            _set_image_src(media_view, current_state['original_path'])
                        except Exception:
                            pass
                    else:
                        # Switch to control image (replace base), not overlay
                        # Swapping to control image
                        if os.path.exists(current_state['control_image_path']):
                            refresh._switch_state[base_image_path]['is_showing_control'] = True
                            try:
                                _set_image_src(media_view, current_state['control_image_path'])
                            except Exception:
                                pass
                        else:
                            # Control image missing; nothing to swap
                            pass
                    # Update visual indicator on top switch button immediately
                    try:
                        is_showing_control_now = refresh._switch_state[base_image_path].get('is_showing_control', False)
                        switch_top_btn.icon_color = ft.Colors.AMBER if is_showing_control_now else None
                        switch_top_btn.style = None
                        switch_top_btn.tooltip = "Showing control" if is_showing_control_now else "Show control"
                        if switch_top_btn.page:
                            switch_top_btn.update()
                    except Exception:
                        pass
                except Exception as ex:
                    print(f"Error switching image: {ex}")

            # Update the top switch icon button visibility, handler and visual state
            try:
                state_entry = refresh._switch_state[base_image_path]
                has_ctrl = state_entry['has_control']
                is_showing_control = state_entry.get('is_showing_control', False)
                switch_top_btn.visible = bool(is_img and has_ctrl)
                switch_top_btn.on_click = on_switch_click
                # Visual indicator when control is active (color only) and clear any outline
                switch_top_btn.icon_color = ft.Colors.AMBER if is_showing_control else None
                switch_top_btn.style = None
                switch_top_btn.tooltip = "Showing control" if is_showing_control else "Show control"
                if switch_top_btn.page:
                    switch_top_btn.update()
            except Exception:
                pass

            # Row with control status only (no switch button here)
            info_row = ft.Row([
                ft.Text("Has control : ", size=12, color=ft.Colors.BLUE_GREY_100),
                ft.Text(control_status, size=12, weight=ft.FontWeight.BOLD, color=control_color),
                ft.Container(expand=True),  # Spacer
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, spacing=10)

            info_container = ft.Container(
                content=ft.Column([
                    ft.Container(content=info_row, padding=ft.padding.symmetric(vertical=10)),
                ], spacing=0, tight=True),
                bgcolor=ft.Colors.GREY_900,
                border=ft.border.all(1, ft.Colors.BLUE_GREY_200),
                border_radius=3,
                padding=ft.padding.all(10)
            )

            slider_col = ft.Column([info_container], spacing=6, col={"md": 6, "lg": 7, "sm": 12})

            # Ensure the Edit button opens whichever image is currently shown (control or original)
            def _current_image_to_edit() -> str:
                try:
                    st = getattr(refresh, '_switch_state', {}).get(base_image_path)
                    if st:
                        return st['control_image_path'] if st.get('is_showing_control') else st.get('original_path', path)
                except Exception:
                    pass
                return path

            try:
                if ps_btn and hasattr(ps_btn, 'on_click'):
                    def _on_edit_click(e):
                        try:
                            st = getattr(refresh, '_switch_state', {}).get(base_image_path)
                            if st and st.get('is_showing_control') and os.path.exists(st.get('control_image_path', '')) and os.path.exists(st.get('original_path', '')):
                                # Open stacked in Photoshop: original on top of control
                                _open_in_photoshop_stacked(page, st['original_path'], st['control_image_path'])
                            else:
                                _open_in_image_editor(page, _current_image_to_edit())
                        except Exception:
                            _open_in_image_editor(page, _current_image_to_edit())
                    ps_btn.on_click = _on_edit_click
                    if ps_btn.page:
                        ps_btn.update()
            except Exception:
                pass
        editing_row = ft.ResponsiveRow([left_col, slider_col], spacing=10)

        # Do not attach on_click to content container to avoid pointer button effect
        content_container.content = ft.Column([
            media_stack,
            fields_row,
            editing_row,
        ], spacing=10, tight=True)
        # Add edit icon on right side next to X; clicking opens the same dropdown
        edit_menu_btn = ft.IconButton(ft.Icons.MORE_VERT, tooltip="Edit menu")
        def _on_edit_menu_click(e):
            try:
                # Toggle: if menu is visible, hide it; otherwise open it
                if context_menu_ctrl and getattr(context_menu_ctrl, 'visible', False):
                    _hide_context_menu()
                else:
                    _open_context_menu_at(max(10, viewer_w - 140), 40)
            except Exception:
                pass
        edit_menu_btn.on_click = _on_edit_menu_click

        dialog.show_dialog(
            content=content_container,
            title=title,
            title_prefix_controls=prefix_controls,
            title_suffix_controls=[edit_menu_btn],
            new_width=DIALOG_WIDTH,
            page=page,
        )
        # Attach hotkey handler for unified popup
        def _hotkey_handler(e: ft.KeyboardEvent):
            try:
                key = getattr(e, 'key', None)
                if not key:
                    return
                # Spacebar controls video playback only
                if key == PLAY_PAUSE_KEY and (not is_img) and local_video_player is not None:
                    if local_video_player.is_playing():
                        local_video_player.pause()
                        try:
                            play_pause_btn.icon = ft.Icons.PLAY_ARROW
                            if play_pause_btn.page:
                                play_pause_btn.update()
                        except Exception:
                            pass
                    else:
                        local_video_player.play()
                        try:
                            play_pause_btn.icon = ft.Icons.PAUSE
                            if play_pause_btn.page:
                                play_pause_btn.update()
                        except Exception:
                            pass
                elif key == PREV_KEY:
                    go(-1)
                elif key == NEXT_KEY:
                    go(1)
            except Exception:
                pass
        # Register appropriate dialog hotkey handler flags per media type
        _set_dialog_hotkey_state(is_img, _hotkey_handler)

        # Start monitor to stop playback at selected end frame
        if not is_img and local_video_player is not None and fps > 0:
            monitor_running_flag["run"] = True

            def _monitor_playback():
                try:
                    while monitor_running_flag["run"] and getattr(dialog, 'open', False):
                        try:
                            end_f = int(frame_range_slider.end_value or original_frames)
                            start_f = int(frame_range_slider.start_value or 0)
                            pos_ms = local_video_player.get_current_position(wait_timeout=0.2)
                            if pos_ms is None:
                                time.sleep(0.1)
                                continue
                            end_ms = int((end_f / fps) * 1000)
                            start_ms = int((start_f / fps) * 1000)
                            if pos_ms >= end_ms:
                                local_video_player.seek(start_ms)
                                local_video_player.play()
                                if local_video_player.page:
                                    local_video_player.update()
                                time.sleep(0.05)
                        except Exception:
                            pass
                        time.sleep(0.1)
                except Exception:
                    pass
            monitor_thread = threading.Thread(target=_monitor_playback, daemon=True)
            monitor_thread.start()

    def go(delta: int):
        nonlocal index
        if not items:
            return
        index = (index + delta) % len(items)
        # Reset control toggle state across items; default to original on navigation
        try:
            if hasattr(refresh, '_switch_state'):
                refresh._switch_state.clear()
            try:
                switch_top_btn.icon_color = None
                switch_top_btn.style = None
                switch_top_btn.tooltip = "Show control"
                if switch_top_btn.page:
                    switch_top_btn.update()
            except Exception:
                pass
        except Exception:
            pass
        refresh()

    # Show dialog
    def _on_dismiss(e=None):
        try:
            _stop_monitor_if_any()
        except Exception:
            pass
        _clear_dialog_hotkey_state()

    dialog._on_dismiss_callback = _on_dismiss
    refresh()
    page.update()
