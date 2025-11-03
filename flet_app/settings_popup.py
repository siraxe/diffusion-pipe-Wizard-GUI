import flet as ft
import json
import os

from flet_app.ui._styles import create_textfield , create_styled_button
from flet_app.settings import settings

def create_settings_content(
    page: ft.Page,
    image_editor_path_textfield_ref: ft.Ref[ft.TextField],
    enable_audio_checkbox_ref: ft.Ref[ft.Checkbox],
    use_gpu_ffmpeg_checkbox_ref: ft.Ref[ft.Checkbox]
) -> ft.Column:
    """Creates and returns the content for the Settings dialog."""
    return ft.Column(
        controls=[
            ft.ResponsiveRow(
                [
                    ft.Text("Image Editor Path:", col=5),
                    create_textfield(label="Path", value="", col=7, expand=True, ref=image_editor_path_textfield_ref)
                ],
            ),
            ft.ResponsiveRow(
                [
                    ft.Text("Video bucket step:", col=5),
                    create_textfield(label="", value="", col=7, expand=True)
                ],
            ),
            ft.ResponsiveRow(
                [
                    ft.Text("Image bucket step:", col=5),
                    create_textfield(label="", value="", col=7, expand=True)
                ],
            ),
            ft.ResponsiveRow(
                [
                    ft.Checkbox(label="Enable audio", value=False, col=12, ref=enable_audio_checkbox_ref)
                ],
            ),
            ft.ResponsiveRow(
                [
                    ft.Checkbox(label="Use GPU FFmpeg", value=True, col=12, ref=use_gpu_ffmpeg_checkbox_ref)
                ],
            ),
            ft.ResponsiveRow(
                [
                    create_styled_button("Apply", on_click=lambda e: _save_settings(e, image_editor_path_textfield_ref, enable_audio_checkbox_ref, use_gpu_ffmpeg_checkbox_ref), col=3)
                ],
                alignment=ft.MainAxisAlignment.END,
            ),
        ],
        tight=True,
        spacing=10,
    )

def _save_settings(e, image_editor_path_textfield_ref: ft.Ref[ft.TextField], enable_audio_checkbox_ref: ft.Ref[ft.Checkbox], use_gpu_ffmpeg_checkbox_ref: ft.Ref[ft.Checkbox]):
    """Saves the settings from the textfields to settings.json."""
    settings_file_path = os.path.join(os.path.dirname(__file__), 'settings.json')
    
    try:
        with open(settings_file_path, 'r') as f:
            current_settings = json.load(f)
        
        # Update IMAGE_EDITOR_PATH
        current_settings["IMAGE_EDITOR_PATH"] = image_editor_path_textfield_ref.current.value
        current_settings["enable_audio"] = enable_audio_checkbox_ref.current.value
        current_settings["use_gpu_ffmpeg"] = use_gpu_ffmpeg_checkbox_ref.current.value
        
        with open(settings_file_path, 'w') as f:
            json.dump(current_settings, f, indent=4)
        
        # Reload settings after saving
        settings._load_settings()
        # Update the textfield with the newly loaded value
        image_editor_path_textfield_ref.current.value = settings.get("IMAGE_EDITOR_PATH", "")
        image_editor_path_textfield_ref.current.update() # Ensure the UI updates
        enable_audio_checkbox_ref.current.value = settings.get("enable_audio", False)
        enable_audio_checkbox_ref.current.update()
        use_gpu_ffmpeg_checkbox_ref.current.value = settings.get("use_gpu_ffmpeg", True)
        use_gpu_ffmpeg_checkbox_ref.current.update()
        
        print(f"Settings saved successfully to {settings_file_path}")
        # Optionally, you might want to show a confirmation to the user
        # e.g., using a SnackBar or updating a status text.
        
    except FileNotFoundError:
        print(f"Error: settings.json not found at {settings_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {settings_file_path}. Check for syntax errors.")
    except Exception as ex:
        print(f"An unexpected error occurred while saving settings: {ex}")


def open_settings_dialog(page: ft.Page):
    """Opens the Settings dialog using the PopupDialogBase."""
    image_editor_path_textfield_ref = ft.Ref[ft.TextField]()
    enable_audio_checkbox_ref = ft.Ref[ft.Checkbox]()
    use_gpu_ffmpeg_checkbox_ref = ft.Ref[ft.Checkbox]()
    settings_content = create_settings_content(page, image_editor_path_textfield_ref, enable_audio_checkbox_ref, use_gpu_ffmpeg_checkbox_ref)
    
    # Populate the textfield with the value from settings.json
    image_editor_path_textfield_ref.current.value = settings.get("IMAGE_EDITOR_PATH", "")
    enable_audio_checkbox_ref.current.value = settings.get("enable_audio", False)
    use_gpu_ffmpeg_checkbox_ref.current.value = settings.get("use_gpu_ffmpeg", True)
    
    title_str = "Settings"
    
    if hasattr(page, 'base_dialog') and page.base_dialog:
        page.base_dialog.show_dialog(content=settings_content, title=title_str)
        # Update the page after showing the dialog to ensure the textfield value is rendered
        page.update()
    else:
        print("Error: Base dialog (PopupDialogBase instance) not found on page.")
        legacy_dialog = ft.AlertDialog(
            title=ft.Text(title_str),
            content=settings_content,
            actions=[
                ft.TextButton("Close", on_click=lambda e: close_legacy_dialog(page, legacy_dialog))
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.dialog = legacy_dialog
        legacy_dialog.open = True
        page.update()

# Helper for the fallback legacy dialog
def close_legacy_dialog(page: ft.Page, dialog_instance: ft.AlertDialog):
    dialog_instance.open = False
    page.update()
