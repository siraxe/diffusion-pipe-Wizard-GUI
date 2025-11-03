import flet as ft
from .popup_dialog_base import PopupDialogBase
from flet_app.ui._styles import BTN_STYLE2, BTN_STYLE2_RED

class BatchCropWarningDialog(PopupDialogBase):
    def __init__(self, page: ft.Page, on_confirm):
        # Dialog content
        content = ft.Column([
            ft.Text("Are you sure you want to crop all images in the current dataset?"),
            ft.Text("This operation will apply the current crop dimensions to all images.", italic=True, color=ft.Colors.GREY),
            ft.Text("This action cannot be undone.", italic=True, color=ft.Colors.RED_ACCENT_700),
        ], tight=True, spacing=10)

        super().__init__(page, content=content, title="Batch Crop Warning")

        self.on_confirm = on_confirm

        # Replace the default close button handler to just hide
        self.close_button.on_click = self.hide_dialog

        # Add action buttons
        self.dialog_content_container.content = ft.Column([
            content,
            ft.Row([
                ft.ElevatedButton(
                    "Crop",
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

def show_batch_crop_warning_dialog(page: ft.Page, on_confirm):
    dialog = BatchCropWarningDialog(page, on_confirm)
    page.overlay.append(dialog)
    # Access the inner container that holds the actual_dialog within the stack
    # This assumes the structure: self.content (Stack) -> [background_layer, inner_container_for_actual_dialog]
    # And inner_container_for_actual_dialog is the second control in the stack
    if len(dialog.content.controls) > 1 and isinstance(dialog.content.controls[1], ft.Container):
        dialog_container_in_stack = dialog.content.controls[1]
        dialog_container_in_stack.top = None # Remove fixed top offset
        dialog_container_in_stack.left = None # Remove fixed left offset
        dialog_container_in_stack.right = None # Remove fixed right offset
        dialog_container_in_stack.alignment = ft.alignment.center # Center vertically and horizontally
        dialog_container_in_stack.expand = True # Ensure it expands to allow centering
    dialog.show_dialog()
