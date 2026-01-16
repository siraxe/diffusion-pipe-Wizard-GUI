import flet as ft
from flet_app.ui_popups.about import open_about_dialog
from flet_app.settings_popup import open_settings_dialog
from flet_app.ui.utils.utils_top_menu import TopBarUtils

# =====================
# Helper/Data Functions
# =====================

def handle_menu_item_click(page: ft.Page, e):
    """Handle clicks on menu items, delegating to the appropriate action."""
    if hasattr(e.control, 'data') and e.control.data:
        # This is a recent file menu item
        handle_open_file(page, e.control.data)
        return
    if hasattr(e.control, 'content') and hasattr(e.control.content, 'value'):
        clicked_text = e.control.content.value
        if clicked_text == "About":
            open_about_dialog(page)
        elif clicked_text == "Help":
            try:
                page.launch_url("https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md")
            except Exception:
                pass
        elif clicked_text == "Settings":
            open_settings_dialog(page)
        elif clicked_text == "Save as":
            handle_save_as(page)
        elif clicked_text == "Save":
            handle_save(page)
        elif clicked_text == "Open":
            handle_open(page)
    else:
        pass

def handle_open_file(page: ft.Page, file_path: str):
    """Open a file using TopBarUtils."""
    TopBarUtils.handle_open(page, file_path=file_path)

def handle_save(page: ft.Page):
    """Save current work using TopBarUtils."""
    TopBarUtils.handle_save(page)

def handle_save_as(page: ft.Page):
    """Save current work as a new file using TopBarUtils."""
    TopBarUtils.handle_save_as(page)

def handle_open(page: ft.Page):
    """Open a file using TopBarUtils."""
    TopBarUtils.handle_open(page)

def handle_load_default(page: ft.Page):
    """Load default settings using TopBarUtils."""
    TopBarUtils.handle_load_default(page)

# =====================
# GUI-Building Functions
# =====================

def build_file_menu(on_menu_item_click, page, text_size):
    """Build the File menu and its submenu controls."""
    return ft.SubmenuButton(
        content=ft.Container(
            content=ft.Text("File", size=text_size),
            padding=ft.padding.symmetric(horizontal=8, vertical=5),
            bgcolor=ft.Colors.TRANSPARENT
        ),
        controls=[
            ft.MenuItemButton(
                content=ft.Text("Save", size=text_size),
                on_click=on_menu_item_click
            ),
            ft.MenuItemButton(
                content=ft.Text("Save as", size=text_size),
                on_click=on_menu_item_click
            ),
            ft.MenuItemButton(
                content=ft.Text("Open", size=text_size),
                on_click=on_menu_item_click
            ),
            ft.SubmenuButton(
                content=ft.Text("Open recent", size=text_size),
                controls=TopBarUtils.get_recent_files_menu_items(on_menu_item_click, text_size=text_size)
            )
        ]
    )

def build_edit_menu(page, text_size,on_menu_item_click):
    """Build the Edit menu and its submenu controls."""
    return ft.SubmenuButton(
        content=ft.Container(
            content=ft.Text("Edit", size=text_size),
            padding=ft.padding.symmetric(horizontal=8, vertical=5),
            bgcolor=ft.Colors.TRANSPARENT
        ),
        controls=[
            ft.MenuItemButton(
                content=ft.Text("Load Default", size=text_size),
                on_click=lambda e: handle_load_default(page)
            ),
            ft.MenuItemButton(
                content=ft.Text("Settings", size=text_size),
                on_click=on_menu_item_click
            ),
        ]
    )

def build_view_menu(on_menu_item_click, text_size):
    """Build the View menu and its submenu controls."""
    return ft.SubmenuButton(
        content=ft.Container(
            content=ft.Text("View", size=text_size),
            padding=ft.padding.symmetric(horizontal=8, vertical=5),
            bgcolor=ft.Colors.TRANSPARENT
        ),
        controls=[
            ft.MenuItemButton(
                content=ft.Text("About", size=text_size),
                on_click=on_menu_item_click
            ),
            ft.MenuItemButton(
                content=ft.Text("Help", size=text_size),
                on_click=on_menu_item_click
            ),
        ]
    )

def build_menu_bar(page: ft.Page, on_menu_item_click, text_size=10):
    """Build the top application menu bar with File, Edit, and View menus."""
    # Create filename display field
    filename_display = ft.Text(
        value="",
        size=12,
        color=ft.Colors.BLUE_GREY_800,
        weight=ft.FontWeight.BOLD,
        italic=False,
    )

    # Store reference on page for updates
    page.filename_display = filename_display
    # Initialize label from current page state if available
    try:
        current_path = getattr(page, 'y_name', None)
        if current_path:
            TopBarUtils.update_filename_display(page, current_path.split('/')[-1])
        else:
            TopBarUtils.update_filename_display(page, "")
    except Exception:
        pass

    return ft.Container(
        content=ft.Row(
            [
                # Group menu and filename tightly together
                ft.Row(
                    [
                        ft.MenuBar(
                            style=ft.MenuStyle(
                                alignment=ft.alignment.top_left,
                                shape=ft.RoundedRectangleBorder(radius=0),
                                elevation=0,
                                shadow_color="transparent",
                            ),
                            controls=[
                                build_file_menu(on_menu_item_click, page, text_size),
                                build_edit_menu(page, text_size, on_menu_item_click),
                                build_view_menu(on_menu_item_click, text_size),
                            ],
                        ),
                        ft.Container(width=12),
                        ft.Text("|", size=12, color=ft.Colors.GREY_500),
                        ft.Container(width=8),
                        ft.Container(
                            content=filename_display,
                            padding=ft.padding.symmetric(horizontal=2, vertical=0),
                            tooltip="Current configuration file",
                        ),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                # Take remaining space on the right
                ft.Container(expand=True),
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
        ),
        expand=True,
        bgcolor="#1d2024",
        padding=ft.padding.symmetric(horizontal=10, vertical=5),
    )

# =====================
# Main Entry Point
# =====================

def create_app_menu_bar(page: ft.Page):
    """Create and return the application menu bar for the given page."""
    # Build and return the menu bar
    return build_menu_bar(page, lambda e: handle_menu_item_click(page, e), text_size=10)
