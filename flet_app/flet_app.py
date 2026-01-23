import os
import sys
from pathlib import Path

# Allow running this file directly or as a module
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

# Standard library imports
import json

# Flet imports
import flet as ft

# Local imports - UI components
from flet_app.ui.flet_hotkeys import global_hotkey_handler
from flet_app.ui.tab_training_view import get_training_tab_content
from flet_app.ui.dataset_manager.dataset_layout_tab import dataset_tab_layout, on_main_tab_change, update_sort_controls_visibility, _initialize_page_state
from flet_app.ui.tab_tools_view import get_models_tab_content
from flet_app.flet_app_top_menu import create_app_menu_bar
from flet_app.ui.utils.utils_top_menu import TopBarUtils
from flet_app.ui_popups.popup_dialog_base import PopupDialogBase
from flet_app.ui.theme_config import DPipeTheme

# Local imports - configuration
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
    # Initialize all page state for this browser tab
    _initialize_page_state(page)

    training_tab_container = get_training_tab_content(page)

    # Create dataset tab content first to get the ABC container
    dataset_tab_content, abc_container, sort_controls_container = dataset_tab_layout(page)

    # Initialize the tab state since we start on Training tab (index 0)
    page.is_in_dataset_tab = False
    update_sort_controls_visibility(page)

    main_tabs = ft.Tabs(
        selected_index=0,
        animation_duration=100,
        on_change=on_main_tab_change,
        tabs=[
            ft.Tab(
                text="Training",
                content=training_tab_container
            ),
            ft.Tab(
                text="Datasets",
                content=dataset_tab_content
            ),
            ft.Tab(
                text="Models",
                content=get_models_tab_content(page)
            ),
        ],
        expand=True,
    )
    
    # Create a container with the tabs and sort controls
    # Note: abc_container is now part of the dataset tab layout, not in the Stack
    tab_with_abc_container = ft.Stack(
        [
            main_tabs,
            sort_controls_container,
        ],
        expand=True
    )

    # Store reference to the ABC container so it can be controlled externally
    tab_with_abc_container.abc_container = abc_container

    return tab_with_abc_container, training_tab_container

def build_tabs_column(main_tabs):
    """Wraps the main tabs in a column that expands vertically and stretches horizontally."""
    # The main_tabs already contains the Stack with tabs and text
    return main_tabs

def build_base_dialog(page):
    """Creates the base popup dialog instance."""
    popup_content = ft.Text("This is a base dialog. Populate it with specific content and title.")
    return PopupDialogBase(page, content=popup_content, title="Info")

# =====================
# Main Application Entry
# =====================

def main(page: ft.Page):
    """Main entry point for the DPipe Trainer Flet application."""
    # =====================
    # Theme Setup (Browser Compatibility)
    # =====================
    DPipeTheme.apply_to_page(page, theme_mode="dark")

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

    # Store reference to ABC container so other parts of the app can access it
    page.abc_container = main_tabs.abc_container

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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8550"))
    open_browser = os.getenv("OPEN_BROWSER", "0") == "1"

    view = ft.AppView.WEB_BROWSER if open_browser else None

    # Make the workspace_dir path absolute to avoid CWD issues
    try:
        project_root = Path(__file__).resolve().parent.parent
        workspace_path = str(project_root / "workspace")
    except Exception:
        # Fallback to relative path if resolution fails
        workspace_path = "workspace"

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

    # Set up upload directory for web file uploads - use workspace/datasets as base
    upload_path = Path(__file__).parent.parent / "workspace" / "datasets"
    upload_path.mkdir(parents=True, exist_ok=True)
    
    ft.app(target=app_wrapper, host=host, port=port, view=view, assets_dir=workspace_path, upload_dir=str(upload_path))

