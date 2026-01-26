import flet as ft
import os
import shutil


class DatasetActionContainer:
    """Container for dataset actions (A, Duplicate, Delete) that appears when items are selected"""

    def __init__(self):
        self.container = None
        self.selected_thumbnails_set = None
        self.selected_dataset = None
        self.thumbnails_grid_ref = None

    def create_container(self):
        """Create the A/B/C action container"""
        self.container = ft.Container(
            content=ft.Row([
                ft.Text("A", size=14, weight=ft.FontWeight.BOLD),
                ft.TextButton(
                    text="",
                    icon=ft.Icons.CONTENT_COPY,
                    on_click=self.on_duplicate_click,
                    icon_color=ft.Colors.BLUE_600,
                    style=ft.ButtonStyle(
                        color=ft.Colors.BLUE_600,
                        padding=ft.padding.symmetric(horizontal=8, vertical=4)
                    ),
                    tooltip="Duplicate selected items"
                ),
                ft.TextButton(
                    text="",
                    icon=ft.Icons.DELETE,
                    on_click=self.on_delete_click,
                    icon_color=ft.Colors.RED_600,
                    style=ft.ButtonStyle(
                        color=ft.Colors.RED_600,
                        padding=ft.padding.symmetric(horizontal=8, vertical=4)
                    ),
                    tooltip="Delete selected items"
                ),
            ], spacing=8),
            top=10,
            right=20,  # 20px offset from the right
            padding=ft.padding.all(5),
            visible=False,  # Initially hidden
            bgcolor=ft.Colors.with_opacity(0.9, ft.Colors.SURFACE),
            border_radius=8,
            border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.OUTLINE)),
        )
        return self.container

    def set_references(self, selected_thumbnails_set, selected_dataset, thumbnails_grid_ref):
        """Set references to dataset manager state"""
        self.selected_thumbnails_set = selected_thumbnails_set
        self.selected_dataset = selected_dataset
        self.thumbnails_grid_ref = thumbnails_grid_ref

    def on_duplicate_click(self, e):
        """Handle duplicate button click"""
        try:
            if not self.selected_thumbnails_set or not self.selected_dataset:
                return

            # Get selected items
            selected_items = list(self.selected_thumbnails_set)
            if selected_items:
                current_dataset = self.selected_dataset.get("value")
                if current_dataset:
                    from .dataset_utils import _get_dataset_base_dir
                    base_dir, _ = _get_dataset_base_dir(current_dataset)
                    dataset_path = os.path.join(base_dir, current_dataset)

                    duplicated_count = 0

                    for item_path in selected_items:
                        try:
                            item_name = os.path.basename(item_path)
                            name, ext = os.path.splitext(item_name)
                            new_name = f"{name}_copy{ext}"
                            new_path = os.path.join(dataset_path, new_name)

                            # Copy the file
                            shutil.copy2(item_path, new_path)
                            duplicated_count += 1

                            # Also copy .txt file if it exists
                            txt_path = os.path.splitext(item_path)[0] + '.txt'
                            if os.path.exists(txt_path):
                                txt_new_name = f"{name}_copy.txt"
                                txt_new_path = os.path.join(dataset_path, txt_new_name)
                                shutil.copy2(txt_path, txt_new_path)

                        except Exception as copy_error:
                            print(f"Error duplicating {item_path}: {copy_error}")

                    # Show success message
                    if duplicated_count > 0:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text(f"Duplicated {duplicated_count} item(s)"),
                            open=True
                        )
                    else:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text("No items duplicated"),
                            open=True
                        )

                    # Clear selections and refresh thumbnails
                    self.selected_thumbnails_set.clear()
                    from . import dataset_layout_tab
                    dataset_layout_tab.last_clicked_thumbnail_index = -1

                    # Hide the container after operation
                    self.container.visible = False
                    self.container.update()

                    # Refresh thumbnails (do this AFTER hiding container to avoid UI conflicts)
                    if self.thumbnails_grid_ref and self.thumbnails_grid_ref.current:
                        from . import dataset_layout_tab
                        e.page.run_task(dataset_layout_tab.update_thumbnails,
                                       page_ctx=e.page,
                                       grid_control=self.thumbnails_grid_ref.current,
                                       force_refresh=True)
        except Exception as ex:
            print(f"Error in duplicate click: {ex}")
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Error: {str(ex)}"),
                    open=True
                )
                e.page.update()

    def on_delete_click(self, e):
        """Handle delete button click"""
        try:
            if not self.selected_thumbnails_set or not self.selected_dataset:
                return

            # Get selected items
            selected_items = list(self.selected_thumbnails_set)
            if selected_items:
                current_dataset = self.selected_dataset.get("value")
                if current_dataset:
                    from .dataset_utils import _get_dataset_base_dir
                    base_dir, _ = _get_dataset_base_dir(current_dataset)
                    dataset_path = os.path.join(base_dir, current_dataset)

                    deleted_count = 0

                    for item_path in selected_items:
                        try:
                            # Delete the main file
                            if os.path.exists(item_path):
                                os.remove(item_path)
                                deleted_count += 1

                            # Also delete .txt file if it exists
                            txt_path = os.path.splitext(item_path)[0] + '.txt'
                            if os.path.exists(txt_path):
                                os.remove(txt_path)

                        except Exception as delete_error:
                            print(f"Error deleting {item_path}: {delete_error}")

                    # Show success message
                    if deleted_count > 0:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text(f"Deleted {deleted_count} item(s)"),
                            open=True
                        )
                    else:
                        e.page.snack_bar = ft.SnackBar(
                            ft.Text("No items deleted"),
                            open=True
                        )

                    # Clear selections and refresh thumbnails
                    self.selected_thumbnails_set.clear()
                    from . import dataset_layout_tab
                    dataset_layout_tab.last_clicked_thumbnail_index = -1

                    # Hide the container after operation
                    self.container.visible = False
                    self.container.update()

                    # Refresh thumbnails (do this AFTER hiding container to avoid UI conflicts)
                    if self.thumbnails_grid_ref and self.thumbnails_grid_ref.current:
                        from . import dataset_layout_tab
                        e.page.run_task(dataset_layout_tab.update_thumbnails,
                                       page_ctx=e.page,
                                       grid_control=self.thumbnails_grid_ref.current,
                                       force_refresh=True)
        except Exception as ex:
            print(f"Error in delete click: {ex}")
            if e.page:
                e.page.snack_bar = ft.SnackBar(
                    ft.Text(f"Error: {str(ex)}"),
                    open=True
                )
                e.page.update()

    def update_visibility(self, show):
        """Update container visibility based on selection state"""
        if self.container:
            self.container.visible = show and len(self.selected_thumbnails_set) > 0 if self.selected_thumbnails_set else False
            if self.container.page:
                self.container.update()