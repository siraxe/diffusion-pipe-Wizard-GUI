import flet as ft
from .popup_dialog_base import PopupDialogBase
from flet_app.ui._styles import BTN_STYLE2, BTN_STYLE2_RED

class DeleteCaptionDialog(PopupDialogBase):
    def __init__(self, page: ft.Page, caption_text: str, on_confirm):
        # Dialog content
        content = ft.Column([
            ft.Text("Are you sure you want to delete this caption?"),
            ft.Text(f'"{caption_text}"', italic=True, color=ft.Colors.GREY),
        ], tight=True, spacing=10)

        super().__init__(page, content=content, title="Delete Caption")

        self.on_confirm = on_confirm

        # Replace the default close button handler to just hide
        self.close_button.on_click = self.hide_dialog

        # Add action buttons
        self.dialog_content_container.content = ft.Column([
            content,
            ft.Row([
                ft.ElevatedButton(
                    "Delete",
                    on_click=self._handle_confirm,
                    style=BTN_STYLE2_RED
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

def show_delete_caption_dialog(page: ft.Page, caption_text: str, on_confirm):
    dialog = DeleteCaptionDialog(page, caption_text, on_confirm)
    page.overlay.append(dialog)
    dialog.show_dialog() 
