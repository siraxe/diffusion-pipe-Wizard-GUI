import flet as ft

def create_about_content(page: ft.Page) -> ft.Column:
    """Creates and returns the content for the About dialog."""
    return ft.Column(
        controls=[
            ft.Text("This application is based on the following repository:"),
            ft.Markdown(
                "https://github.com/tdrussell/diffusion-pipe",
                selectable=True,
                extension_set="gitHubWeb", # Keeping user's original casing
                on_tap_link=lambda e: page.launch_url(e.data),
            ),
            ft.Divider(),
            ft.Text("This is work in progress report any major issues here:"),
            ft.Markdown(
                "https://github.com/siraxe/Dpipe-Wizard-GUI",
                selectable=True,
                extension_set="gitHubWeb", # Keeping user's original casing
                on_tap_link=lambda e: page.launch_url(e.data),
            ),
            ft.Divider(),
            ft.Text("Version: 0.1.0 Alpha"), # Example version
            ft.Text("Developed with Flet and Python.")
        ],
        tight=True,
        spacing=10,
        # Add some padding inside the column if needed, 
        # otherwise PopupDialogBase will provide padding.
        # width=400 # Optional: if you want to control content width specifically
    )

def open_about_dialog(page: ft.Page):
    """Opens the About dialog using the PopupDialogBase."""
    about_content = create_about_content(page)
    title_str = "About DPipe Video Trainer UI"
    
    if hasattr(page, 'base_dialog') and page.base_dialog:
        # Call show_dialog on the PopupDialogBase instance 
        # with the new content and title.
        page.base_dialog.show_dialog(content=about_content, title=title_str)
    else:
        print("Error: Base dialog (PopupDialogBase instance) not found on page.")
        # Fallback: Show a standard Flet AlertDialog if PopupDialogBase is missing
        legacy_dialog = ft.AlertDialog(
            title=ft.Text(title_str),
            content=about_content,
            actions=[
                ft.TextButton("Close", on_click=lambda e: close_legacy_dialog(page, legacy_dialog))
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.dialog = legacy_dialog # page.dialog is the typical way to show AlertDialogs
        legacy_dialog.open = True
        page.update()

# Helper for the fallback legacy dialog
def close_legacy_dialog(page: ft.Page, dialog_instance: ft.AlertDialog):
    dialog_instance.open = False
    page.update()
