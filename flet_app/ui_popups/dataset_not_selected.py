import flet as ft
from .popup_dialog_base import PopupDialogBase
from flet_app.ui._styles import BTN_STYLE2

class DatasetNotSelectedDialog(PopupDialogBase):
    def __init__(self, page: ft.Page, question: str, on_confirm):
        # Dialog content
        content = ft.Column([
            ft.Text(question),
        ], tight=True, spacing=10)

        super().__init__(page, content=content, title="Dataset Not Selected")

        self.on_confirm = on_confirm

        # Replace the default close button handler to just hide
        self.close_button.on_click = self.hide_dialog

        # Add action buttons
        self.dialog_content_container.content = ft.Column([
            content,
            ft.Row([
                ft.ElevatedButton(
                    "Yes",
                    on_click=self._handle_confirm,
                    style=BTN_STYLE2
                ),
                ft.TextButton(
                    "Cancel",
                    on_click=self.hide_dialog,
                    style=BTN_STYLE2
                )
            ], alignment=ft.MainAxisAlignment.END, spacing=10)
        ], tight=True, spacing=20)

    def _handle_confirm(self, e):
        self.hide_dialog()
        if self.on_confirm:
            self.on_confirm()

def show_dataset_not_selected_dialog(page: ft.Page, question: str, on_confirm):
    dialog = DatasetNotSelectedDialog(page, question, on_confirm)
    page.overlay.append(dialog)
    dialog.show_dialog() 
