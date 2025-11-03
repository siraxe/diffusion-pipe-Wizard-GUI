import flet as ft

# =====================
# Utility / Data Helper Functions
# =====================
# (Currently, this class is mostly GUI logic. Add utility/data helpers here if needed in the future.)

# =====================
# GUI-Building Functions
# =====================

def build_title_bar(title: str, on_close, prefix_controls=None) -> ft.Row:
    """Builds the title bar row for the dialog, including optional prefix controls and a close button."""
    prefix_controls_container = ft.Row(
        controls=prefix_controls if prefix_controls is not None else [],
        tight=True,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )
    title_text_control = ft.Text(
        title,
        size=20,
        weight=ft.FontWeight.BOLD,
        expand=True,
        text_align=ft.TextAlign.CENTER
    )
    close_button = ft.IconButton(ft.Icons.CLOSE, on_click=on_close)
    title_row = ft.Row(
        controls=[prefix_controls_container, title_text_control, close_button],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )
    return title_row, prefix_controls_container, title_text_control, close_button

def build_dialog_content(content: ft.Control) -> ft.Container:
    """Wraps the dialog's main content in a container for easy updating."""
    return ft.Container(content=content)

def build_actual_dialog(title_row: ft.Row, dialog_content_container: ft.Container, width: int) -> ft.Container:
    """Builds the main dialog container with title, divider, and content."""
    return ft.Container(
        content=ft.Column(
            controls=[
                title_row,
                ft.Divider(),
                dialog_content_container,
            ],
            tight=True
        ),
        width=width,
        bgcolor=ft.Colors.SURFACE,
        border_radius=ft.border_radius.all(10),
        padding=ft.padding.all(20),
        animate_offset=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=15,
            color=ft.Colors.with_opacity(0.25, ft.Colors.BLACK),
            offset=ft.Offset(0, 5),
        )
    )

def build_background_layer(on_click) -> ft.Container:
    """Builds the semi-transparent background layer for the dialog."""
    return ft.Container(
        bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
        on_click=on_click,
        expand=True,
    )

def build_dialog_stack(background_layer: ft.Container, actual_dialog: ft.Container) -> ft.Stack:
    """Builds the stack containing the background and the centered dialog.
    Centers the dialog horizontally while keeping background clickable on sides.
    """
    # Apply a top margin to the dialog itself so we don't need a full-width wrapper
    try:
        actual_dialog.margin = ft.margin.only(top=70)
    except Exception:
        pass
    return ft.Stack(
        controls=[
            background_layer,
            actual_dialog,
        ],
        expand=True,
        alignment=ft.alignment.top_center,
    )

# =====================
# Main Dialog Class
# =====================

class PopupDialogBase(ft.Container):
    """
    A reusable popup dialog base class for Flet apps, with customizable title, content, and width.
    GUI-building logic is separated into helper functions for maintainability.
    """
    DEFAULT_DIALOG_WIDTH = 500

    def __init__(self, page: ft.Page, content: ft.Control, title: str = "Dialog", on_dismiss: callable = None):
        super().__init__(expand=True)
        self.page = page
        self._initial_content = content
        self._initial_title = title
        self._on_dismiss_callback = on_dismiss

        # Build GUI controls
        self._build_gui()

        self.opacity = 0
        self.visible = False
        self.animate_opacity = ft.Animation(duration=200, curve=ft.AnimationCurve.EASE_IN)

    def _build_gui(self):
        """Initializes and builds all GUI controls for the dialog."""
        # Title bar and controls
        self.title_row, self.title_bar_prefix_controls_container, self.title_text_control, self.close_button = build_title_bar(
            self._initial_title, self.hide_dialog
        )
        # Content container
        self.dialog_content_container = build_dialog_content(self._initial_content)
        # Main dialog
        self.actual_dialog = build_actual_dialog(
            self.title_row, self.dialog_content_container, self.DEFAULT_DIALOG_WIDTH
        )
        # Background layer
        self.background_layer = build_background_layer(self.hide_dialog)
        # Stack (background + dialog)
        self.content = build_dialog_stack(self.background_layer, self.actual_dialog)

    def show_dialog(self, content: ft.Control = None, title: str = None, new_width: int = None, title_prefix_controls: list[ft.Control] = None, e=None, page: ft.Page = None):
        """
        Shows the dialog, optionally updating its content, title, width, and prefix controls.
        """
        if page is not None:
            self.page = page
        if self.page is None:
            raise RuntimeError("PopupDialogBase: 'page' is None. Cannot show dialog.")
        self.disabled = False
        self.background_layer.on_click = self.hide_dialog
        # Update title text
        if title is not None:
            self.title_text_control.value = title
        # Update prefix controls
        if title_prefix_controls is not None:
            self.title_bar_prefix_controls_container.controls = title_prefix_controls
        else:
            self.title_bar_prefix_controls_container.controls = []
        # Update content
        if content is not None:
            self.dialog_content_container.content = content
        # Update width
        self.actual_dialog.width = new_width if new_width is not None else self.DEFAULT_DIALOG_WIDTH
        self.opacity = 1
        self.actual_dialog.offset = ft.Offset(0, 0)
        self.visible = True
        self.page.open(self)

    def hide_dialog(self, e=None):
        """
        Hides the dialog and disables interaction.
        """
        self.opacity = 0
        self.actual_dialog.offset = ft.Offset(0, 0.5)
        self.visible = False
        self.disabled = True
        self.background_layer.on_click = None
        # Call the dismiss callback if it exists
        if self._on_dismiss_callback:
            self._on_dismiss_callback(e)
        self.page.close(self)

    @property
    def open(self):
        """Returns True if the dialog is visible (open)."""
        return self.visible

    @open.setter
    def open(self, value):
        """Sets the dialog's visibility and updates the page."""
        self.visible = value
        self.disabled = not value
        if self.page is not None:
            self.page.update()

# =====================
# Example Usage (for testing)
# =====================

if __name__ == "__main__":
    def main(page: ft.Page):
        page.title = "Dialog Test"
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

        example_content_1 = ft.Column([
            ft.Text("This is the INITIAL content (default width)."),
            ft.TextField(label="Field 1"),
        ])
        
        example_content_2 = ft.Column([
            ft.Text("This is REPLACED content for the dialog (wider)!"),
            ft.TextField(label="Another field (2)"),
            ft.ElevatedButton("Submit Button")
        ])

        my_dialog = PopupDialogBase(page, content=example_content_1, title="My Initial Dialog Title")
        page.overlay.append(my_dialog)

        def open_with_initial_content(e):
            my_dialog.show_dialog()

        def open_with_new_content_wider(e):
            my_dialog.show_dialog(content=example_content_2, title="My NEW Dialog Title", new_width=600)
            
        def open_with_just_new_title_default_width(e):
            my_dialog.show_dialog(title="Updated Title Only - Default Width")

        page.add(
            ft.ElevatedButton("Show Initial Dialog (Default Width)", on_click=open_with_initial_content),
            ft.ElevatedButton("Show With New Content/Title (Wider)", on_click=open_with_new_content_wider),
            ft.ElevatedButton("Show With New Title Only (Default Width)", on_click=open_with_just_new_title_default_width),
        )

    ft.app(target=main)
