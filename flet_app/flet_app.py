import os
import sys
from pathlib import Path

# Allow running this file directly or as a module
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import flet as ft
import json
from flet_app.ui.flet_hotkeys import global_hotkey_handler
from flet_app.ui.tab_training_view import get_training_tab_content
from flet_app.ui.dataset_manager.dataset_layout_tab import dataset_tab_layout
from flet_app.ui.tab_tools_view import get_models_tab_content
from flet_app.flet_app_top_menu import create_app_menu_bar
from flet_app.ui.utils.utils_top_menu import TopBarUtils
from flet_app.ui_popups.popup_dialog_base import PopupDialogBase
from flet_app.settings import settings

# =====================
# Helper/Data Functions
# =====================

def refresh_menu_bar(page, menu_bar_column):
    """Refreshes the application menu bar."""
    new_menu_bar = create_app_menu_bar(page)
    menu_bar_column.controls.clear()
    menu_bar_column.controls.append(new_menu_bar)
    menu_bar_column.update()
    # Re-apply filename label after rebuild
    try:
        current_path = getattr(page, 'y_name', None)
        if current_path:
            TopBarUtils.update_filename_display(page, os.path.basename(str(current_path).replace('\\', '/')))
        else:
            TopBarUtils.update_filename_display(page, "")
    except Exception:
        pass

# =====================
# GUI-Building Functions
# =====================

def build_menu_bar_column(page):
    """Creates the menu bar column control."""
    app_menu_bar = create_app_menu_bar(page)
    menu_bar_column = ft.Column(
        controls=[app_menu_bar],
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        spacing=0
    )
    return menu_bar_column

def build_main_tabs(page):
    """Creates the main tab control with Training, Datasets, and Models tabs."""
    training_tab_container = get_training_tab_content(page)
    main_tabs = ft.Tabs(
        selected_index=0,
        animation_duration=100,
        tabs=[
            ft.Tab(
                text="Training",
                content=training_tab_container
            ),
            ft.Tab(
                text="Datasets",
                content=dataset_tab_layout(page)
            ),
            ft.Tab(
                text="Models",
                content=get_models_tab_content(page)
            ),
        ],
        expand=True,
    )
    return main_tabs, training_tab_container

def build_tabs_column(main_tabs):
    """Wraps the main tabs in a column that expands vertically and stretches horizontally."""
    return ft.Column(
        controls=[main_tabs],
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        spacing=0
    )

def build_base_dialog(page):
    """Creates the base popup dialog instance."""
    popup_content = ft.Text("This is a base dialog. Populate it with specific content and title.")
    return PopupDialogBase(page, content=popup_content, title="Info")

# =====================
# Main Application Entry
# =====================

def main(page: ft.Page):
    """Main entry point for the DPipe Trainer Flet application."""
    # Store project location in settings.json
    try:
        # Get the project directory (one level higher than flet_app directory)
        current_file = Path(__file__)
        project_dir = current_file.parent.parent  # One directory higher than flet_app
        project_location = str(project_dir.resolve())

        # Path to settings.json
        settings_path = current_file.parent / "settings.json"

        # Read current settings
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}

        # Update project_location
        settings['project_location'] = project_location

        # Save back to settings.json
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4)

    except Exception as e:
        print(f"Error saving project location: {e}")

    # Window setup
    page.title = "DPipe GUI Wizard"
    page.padding = 0
    page.window.center()
    page.window.height = 950
    page.window.width = 900

    # Build GUI controls
    menu_bar_column = build_menu_bar_column(page)
    main_tabs, training_tab_container = build_main_tabs(page)
    tabs_column = build_tabs_column(main_tabs)

    # Expose for Save As handler
    page.training_tab_container = training_tab_container

    # Attach menu bar refresh function
    page.refresh_menu_bar = lambda: refresh_menu_bar(page, menu_bar_column)

    # Add controls to page
    page.add(menu_bar_column)
    page.add(tabs_column)

    # Dialogs and overlays
    page.base_dialog = build_base_dialog(page)
    page.overlay.append(page.base_dialog)
    page.video_dialog_hotkey_handler = None
    page.video_dialog_open = False

    # Keyboard event handler
    page.on_keyboard_event = lambda e: global_hotkey_handler(page, e)
    page.update()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8550"))
    open_browser = os.getenv("OPEN_BROWSER", "0") == "1"

    view = ft.AppView.WEB_BROWSER if open_browser else None

    local_url = f"http://localhost:{port}"
    bound_url = f"http://localhost:{port}"
    print("Flet server starting (DPipe GUI Wizard)...")
    print(f"- Local: {local_url}")
    print(f"- Bound: {bound_url}")
    print("- Output files: http://localhost:8550/output/")
    print("Press Ctrl+C to stop.")

    # Add static file serving for output directory
    def setup_static_files(page: ft.Page):
        """Setup static file serving for output directory."""
        try:
            # Get the output directory path using project_location from settings
            project_location = settings.get("project_location")
            if not project_location:
                raise ValueError("project_location not found in settings. Run flet_app.py first to set it.")
            output_dir = Path(project_location) / "workspace" / "output"
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created output directory: {output_dir}")

            # For now, we'll print info but not mount static files
            # We'll use direct file access instead
            print(f"Output directory available: {output_dir}")

        except Exception as e:
            print(f"Failed to setup static files: {e}")

    # Create a wrapper function that includes static file setup
    def app_wrapper(page: ft.Page):
        # Setup static files first
        setup_static_files(page)
        # Then run the main app
        main(page)

    ft.app(target=app_wrapper, host=host, port=port, view=view)
