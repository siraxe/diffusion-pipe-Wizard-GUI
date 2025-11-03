import os
import platform
import subprocess
import tempfile

import flet as ft

from flet_app.settings import settings
from .unified_media_utils import _is_web_platform


def _is_wsl() -> bool:
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def _wsl_to_windows_path(path: str) -> str:
    try:
        out = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
        return out
    except Exception:
        return path


def open_in_image_editor(page: ft.Page, image_path: str):
    editor = settings.get("IMAGE_EDITOR_PATH", None)
    if not image_path or not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    system = platform.system().lower()
    is_wsl = _is_wsl()
    is_web = _is_web_platform(page)

    try:
        # WSL path: prefer cmd.exe, then wslview, then PowerShell; fallback to browser tab in web
        if is_wsl:
            win_img = _wsl_to_windows_path(image_path)
            w_cwd = "/mnt/c/Windows" if os.path.isdir("/mnt/c/Windows") else None
            try:
                if editor:
                    subprocess.Popen(["cmd.exe", "/C", "start", "", editor, win_img], cwd=w_cwd)
                else:
                    subprocess.Popen(["cmd.exe", "/C", "start", "", win_img], cwd=w_cwd)
                return
            except Exception:
                pass
            try:
                subprocess.Popen(["wslview", win_img])
                return
            except Exception:
                pass
            try:
                if editor:
                    ps_cmd = [
                        "powershell.exe", "-NoProfile", "-Command",
                        f"Start-Process -FilePath \"{editor}\" -ArgumentList @(\"{win_img}\")"
                    ]
                else:
                    ps_cmd = [
                        "powershell.exe", "-NoProfile", "-Command",
                        f"Start-Process -FilePath explorer.exe -ArgumentList @(\"{win_img}\")"
                    ]
                subprocess.Popen(ps_cmd)
                return
            except Exception:
                if is_web:
                    try:
                        import base64, mimetypes
                        mime_type, _ = mimetypes.guess_type(image_path)
                        mime_type = mime_type or "image/png"
                        with open(image_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        page.launch_url(f"data:{mime_type};base64,{b64}")
                        return
                    except Exception:
                        pass
                raise

        # Native Windows
        if system == "windows":
            if editor:
                subprocess.Popen([editor, image_path])
            else:
                subprocess.Popen(["cmd", "/c", "start", "", image_path], shell=True)
            return

        # macOS/Linux
        if editor:
            subprocess.Popen([editor, image_path])
        else:
            subprocess.Popen(["xdg-open", image_path])
    except Exception as ex:
        print(f"Failed to open editor: {ex}")


def _build_jsx_stack_script(original_path: str, control_path: str) -> str:
    """JSX: 1) open control, 2) open original, 3) copy original as top layer into control, 4) close original."""
    def esc(p: str) -> str:
        return p.replace("\\", "\\\\")

    p_orig = esc(original_path)
    p_ctrl = esc(control_path)
    return (
        "#target photoshop\n"
        "app.bringToFront();\n"
        f"var fCtrl = new File(\"{p_ctrl}\");\n"
        f"var fOrig = new File(\"{p_orig}\");\n"
        # 1) Open control first
        "app.open(fCtrl); var dCtrl = app.activeDocument;\n"
        # 2) Open original second
        "app.open(fOrig); var dOrig = app.activeDocument;\n"
        # 3) Convert original's active layer to Smart Object, then duplicate into control doc on top
        "app.activeDocument = dOrig;\n"
        "try { executeAction(stringIDToTypeID('newPlacedLayer'), undefined, DialogModes.NO); } catch(e) {}\n"
        "dOrig.activeLayer.duplicate(dCtrl, ElementPlacement.PLACEATBEGINNING);\n"
        # Set imported layer opacity to 50% in control document
        "app.activeDocument = dCtrl;\n"
        "try { dCtrl.activeLayer.opacity = 50; } catch(e) {}\n"
        # 4) Close original without saving; leave control active
        "app.activeDocument = dOrig;\n"
        "dOrig.close(SaveOptions.DONOTSAVECHANGES);\n"
        "app.activeDocument = dCtrl;\n"
    )


def open_in_photoshop_stacked(page: ft.Page, original_path: str, control_path: str):
    """Open both images in Photoshop stacked as layers (original on top). Falls back to single-image open on failure."""
    try:
        if not original_path or not control_path:
            raise RuntimeError("Missing image paths for stacking")
        if not os.path.exists(original_path) or not os.path.exists(control_path):
            raise RuntimeError("One of the images does not exist")

        system = platform.system().lower()
        is_wsl = _is_wsl()
        editor = settings.get("IMAGE_EDITOR_PATH", None)

        if is_wsl:
            win_orig = _wsl_to_windows_path(original_path)
            win_ctrl = _wsl_to_windows_path(control_path)
            temp_jsx_unix = os.path.join(tempfile.gettempdir(), "stack_photoshop.jsx")
            with open(temp_jsx_unix, "w", encoding="utf-8") as f:
                f.write(_build_jsx_stack_script(win_orig, win_ctrl))
            win_jsx = _wsl_to_windows_path(temp_jsx_unix)
            w_cwd = "/mnt/c/Windows" if os.path.isdir("/mnt/c/Windows") else None
            try:
                if editor:
                    subprocess.Popen(["cmd.exe", "/C", "start", "", editor, win_jsx], cwd=w_cwd)
                else:
                    subprocess.Popen(["cmd.exe", "/C", "start", "", win_jsx], cwd=w_cwd)
                return
            except Exception:
                pass
            try:
                subprocess.Popen(["wslview", win_jsx])
                return
            except Exception:
                pass
            try:
                ps_cmd = [
                    "powershell.exe", "-NoProfile", "-Command",
                    f"Start-Process -FilePath \"{editor or win_jsx}\" -ArgumentList @(\"{win_jsx}\")"
                ]
                subprocess.Popen(ps_cmd)
                return
            except Exception:
                pass
            open_in_image_editor(page, control_path)
            return

        if system == "windows":
            temp_jsx = os.path.join(tempfile.gettempdir(), "stack_photoshop.jsx")
            with open(temp_jsx, "w", encoding="utf-8") as f:
                f.write(_build_jsx_stack_script(original_path, control_path))
            try:
                if editor:
                    subprocess.Popen(["cmd", "/c", "start", "", editor, temp_jsx], shell=True)
                else:
                    subprocess.Popen(["cmd", "/c", "start", "", temp_jsx], shell=True)
                return
            except Exception:
                pass
            open_in_image_editor(page, control_path)
            return

        # macOS/Linux fallbacks: open two images separately
        try:
            open_in_image_editor(page, control_path)
            open_in_image_editor(page, original_path)
        except Exception:
            open_in_image_editor(page, control_path)
    except Exception as ex:
        print(f"Failed to open stacked images: {ex}")
        try:
            open_in_image_editor(page, control_path or original_path)
        except Exception:
            pass
