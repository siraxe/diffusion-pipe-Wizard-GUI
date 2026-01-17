import os
from pathlib import Path
from datetime import datetime
import flet as ft
from loguru import logger
import traceback

try:
    import tomllib as _toml_parser  # Python 3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as _toml_parser  # Fallback for older Python if available
    except Exception:
        _toml_parser = None


def _create_open_content(page: ft.Page) -> ft.Column:
    """Create the Open dialog content"""
    logger.debug("Creating Open dialog content")

    from .utils_top_menu import TopBarUtils  # Import to access methods

    default_dir = TopBarUtils._ensure_default_config_dir()
    initial_path = getattr(page, 'y_name', None)

    # Directory path display - ALWAYS show base configs directory
    dir_display = ft.Container(
        content=ft.Text(
            str(default_dir),
            size=12,
            color=ft.Colors.BLUE_GREY_700,
            weight=ft.FontWeight.BOLD
        ),
        margin=ft.margin.only(bottom=5)
    )

    # No input field needed for Open dialog - just file selection
    status_text = ft.Text(value="", color=ft.Colors.GREEN)

    # Sorting state variables
    sort_mode = "name"  # "name" or "date"
    name_sort_ascending = True  # A-Z when True, Z-A when False
    date_sort_ascending = False  # newest first when False, oldest first when True

    def sort_files(files, mode, ascending):
        """Sort files by name or date"""
        if mode == "name":
            return sorted(files, key=lambda x: str(x[0]).lower(), reverse=not ascending)
        elif mode == "date":
            return sorted(files, key=lambda x: x[2], reverse=not ascending)
        return files

    def toggle_name_sort():
        """Toggle name sorting between A-Z and Z-A"""
        nonlocal sort_mode, name_sort_ascending
        sort_mode = "name"
        name_sort_ascending = not name_sort_ascending
        refresh_file_list()

    def toggle_date_sort():
        """Toggle date sorting between newest and oldest"""
        nonlocal sort_mode, date_sort_ascending
        sort_mode = "date"
        date_sort_ascending = not date_sort_ascending
        refresh_file_list()

    def load_existing_files():
        """Load existing .toml files from the base configs directory and subdirectories"""
        try:
            base_config_dir = default_dir
            logger.debug(f"Looking for .toml files in base directory: {base_config_dir}")

            found_files = []

            if base_config_dir.exists():
                # Find all .toml files in base directory and subdirectories
                for toml_file in base_config_dir.rglob("*.toml"):
                    rel_path = toml_file.relative_to(base_config_dir)
                    mod_time = toml_file.stat().st_mtime
                    found_files.append((rel_path, toml_file, mod_time))

                logger.debug(f"Found {len(found_files)} .toml files total")
                # Default sort by name (A-Z)
                found_files.sort(key=lambda x: str(x[0]).lower())
                return found_files
            else:
                logger.warning(f"Base config directory does not exist: {base_config_dir}")
            return []
        except Exception as e:
            logger.error(f"Error loading existing files: {e}")
            return []

    def refresh_file_list():
        """Refresh the file list display"""
        try:
            existing_files = load_existing_files()
            file_list_controls.clear()

            # Apply sorting
            if sort_mode == "name":
                sorted_files = sort_files(existing_files, "name", name_sort_ascending)
            elif sort_mode == "date":
                sorted_files = sort_files(existing_files, "date", date_sort_ascending)
            else:
                sorted_files = existing_files

            # Create sleek header row
            name_sort_text = "Name ↓" if sort_mode == "name" and not name_sort_ascending else "Name ↑" if sort_mode == "name" and name_sort_ascending else "Name"
            date_sort_text = "Date ↓" if sort_mode == "date" and not date_sort_ascending else "Date ↑" if sort_mode == "date" and date_sort_ascending else "Date"

            header_row = ft.Row([
                # Name header
                ft.Container(
                    content=ft.Row([
                        ft.Text(
                            value=name_sort_text,
                            size=13,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_300 if sort_mode == "name" else ft.Colors.GREY_300
                        ),
                        ft.Icon(
                            ft.Icons.SORT if sort_mode != "name" else (ft.Icons.ARROW_UPWARD if name_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                            size=14,
                            color=ft.Colors.BLUE_300 if sort_mode == "name" else ft.Colors.GREY_400
                        )
                    ], spacing=5),
                    width=360,  # Increased from 300 to use more space
                    height=30,
                    bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if sort_mode == "name" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                    border_radius=4,
                    padding=ft.padding.symmetric(horizontal=10, vertical=5),
                    on_click=lambda _: toggle_name_sort(),
                    ink=True
                ),
                # Date header
                ft.Container(
                    content=ft.Row([
                        ft.Text(
                            value=date_sort_text,
                            size=13,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_300 if sort_mode == "date" else ft.Colors.GREY_300
                        ),
                        ft.Icon(
                            ft.Icons.SORT if sort_mode != "date" else (ft.Icons.ARROW_UPWARD if date_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                            size=14,
                            color=ft.Colors.BLUE_300 if sort_mode == "date" else ft.Colors.GREY_400
                        )
                    ], spacing=5),
                    width=140,  # Increased from 120 to use more space
                    height=30,
                    bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if sort_mode == "date" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                    border_radius=4,
                    padding=ft.padding.symmetric(horizontal=10, vertical=5),
                    on_click=lambda _: toggle_date_sort(),
                    ink=True
                )
            ], spacing=10)
            file_list_controls.append(header_row)

            if sorted_files:
                for rel_path, full_path, mod_time in sorted_files:
                    mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")

                    file_row = ft.Row([
                        ft.TextButton(
                            text=str(rel_path).replace('\\', '/'),
                            on_click=lambda _, fp=rel_path, full=full_path: open_file(fp, full),
                            style=ft.ButtonStyle(
                                alignment=ft.alignment.center_left,
                                padding=ft.padding.symmetric(horizontal=10, vertical=5)
                            ),
                            width=360  # Increased from 300 to match header
                        ),
                        ft.Text(
                            value=mod_date,
                            size=12,
                            color=ft.Colors.GREY_600,
                            width=140,  # Increased from 120 to match header
                            text_align=ft.TextAlign.RIGHT
                        )
                    ], spacing=5)
                    file_list_controls.append(file_row)
            else:
                file_list_controls.append(
                    ft.Text("No .toml files found", italic=True, color=ft.Colors.GREY_600)
                )

            # Update the container content
            file_list_container.content = ft.ListView(
                controls=file_list_controls,
                spacing=2,
                expand=True,
                auto_scroll=False
            )
            # Ensure UI refreshes within the dialog after interactions
            try:
                if file_list_container.page:
                    file_list_container.update()
            except Exception:
                pass
            # Don't call page.update() here - PopupDialogBase will handle higher-level updates
            logger.debug("File list refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing file list: {e}")
            logger.error(traceback.format_exc())

    def open_file(rel_path: Path, full_path: Path):
        """Open the selected file"""
        try:
            toml_data = None
            raw_text = ''
            if _toml_parser:
                with open(full_path, 'rb') as f:
                    data = f.read()
                    raw_text = data.decode('utf-8', errors='ignore')
                    toml_data = _toml_parser.loads(raw_text)
            training_tab = getattr(page, 'training_tab_container', None)
            if training_tab and isinstance(toml_data, dict):
                from .config_utils import update_ui_from_toml
                update_ui_from_toml(training_tab, toml_data)
                # Detect commented timestep and set dropdown to 'None'
                try:
                    import re
                    if re.search(r"^\s*#\s*timestep_sample_method\b", raw_text, flags=re.MULTILINE):
                        def _apply(control):
                            if hasattr(control, 'controls') and control.controls:
                                for c in control.controls:
                                    _apply(c)
                            if hasattr(control, 'content') and control.content:
                                _apply(control.content)
                            label = getattr(control, 'label', None)
                            if label == 'timestep_sm' and isinstance(control, ft.Dropdown):
                                control.value = 'None'
                                if control.page:
                                    control.update()
                        _apply(training_tab.config_page_content)
                    if re.search(r"^\s*#\s*transformer_dtype\b", raw_text, flags=re.MULTILINE):
                        def _apply2(control):
                            if hasattr(control, 'controls') and control.controls:
                                for c in control.controls:
                                    _apply2(c)
                            if hasattr(control, 'content') and control.content:
                                _apply2(control.content)
                            label = getattr(control, 'label', None)
                            if label == 'transformer_dtype' and isinstance(control, ft.Dropdown):
                                control.value = 'None'
                                if control.page:
                                    control.update()
                        _apply2(training_tab.config_page_content)
                except Exception:
                    pass
            TopBarUtils.set_yaml_path_and_title(page, str(full_path))
            TopBarUtils.add_recent_file(str(full_path), page)
            page.is_default_config_loaded = False
            status_text.value = f"Successfully opened {rel_path}"
            status_text.color = ft.Colors.GREEN

            # Close the dialog after successful open
            if hasattr(page, 'base_dialog') and page.base_dialog:
                page.base_dialog.hide_dialog()

        except Exception as ex:
            logger.error(f"Error opening file {full_path}: {ex}")
            logger.error(traceback.format_exc())
            status_text.value = f"Error: {ex}"
            status_text.color = ft.Colors.RED

    # Initialize controls and container first
    file_list_controls = []

    # File list container - no height limitation for PopupDialogBase
    file_list_container = ft.Container(
        content=ft.ListView(
            controls=file_list_controls,
            spacing=2,
            expand=True,
            auto_scroll=False
        ),
        height=300,  # Fixed reasonable height, but PopupDialogBase can handle larger content
        margin=ft.margin.only(top=10),
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=8,
        padding=5,
        width=500  # Increased from 450 to use full dialog width (550px - padding)
    )

    # Now refresh the list to populate the container
    refresh_file_list()

    return ft.Column([
        dir_display,
        ft.Text("Select a configuration file to open:", size=12, color=ft.Colors.GREY_600),
        file_list_container,
        status_text,
    ], spacing=10)


def _create_save_as_content(page: ft.Page) -> ft.Column:
    """Create the Save As dialog content"""
    logger.debug("Creating Save As dialog content")

    from .utils_top_menu import TopBarUtils  # Import to access methods
    from .config_utils import build_toml_config_from_ui  # Import for TOML building
    from .ltx2_config_utils import build_ltx2_toml_from_ui  # Import LTX2 builder

    # Build TOML text from current UI
    training_tab = getattr(page, 'training_tab_container', None)
    if not training_tab:
        return ft.Column([ft.Text("Error: No training tab available")])
    try:
        # Use LTX2-specific config builder if LTX2 is selected
        if TopBarUtils._is_ltx2_selected(training_tab):
            logger.debug("LTX2 detected in Save As dialog, using LTX2-specific builder.")
            toml_text = build_ltx2_toml_from_ui(training_tab)
        else:
            toml_text = build_toml_config_from_ui(training_tab)
    except Exception as e:
        logger.error(f"Error building TOML in Save As: {e}")
        toml_text = ""
    default_dir = TopBarUtils._ensure_default_config_dir()
    initial_path = getattr(page, 'y_name', None)

    # Create config path object - ALWAYS start from base configs directory
    from pathlib import Path
    config_path = default_dir / "new_config.toml"

    # Use a simple variable for the filename
    if initial_path:
        config_filename = Path(initial_path).name
    else:
        config_filename = config_path.name

    # Directory path display - ALWAYS show base configs directory
    dir_display = ft.Container(
        content=ft.Text(
            str(default_dir),
            size=12,
            color=ft.Colors.BLUE_GREY_700,
            weight=ft.FontWeight.BOLD
        ),
        margin=ft.margin.only(bottom=5)
    )

    # Main input area with path field and Save button
    input_row = ft.Row([
        ft.TextField(
            value=config_filename,
            label="Configuration Name",
            width=400,
            expand=True,
            on_change=lambda e: update_filename(e.control.value)
        ),
        ft.ElevatedButton(
            "Save",
            icon=ft.Icons.SAVE,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.BLUE_600,
                color=ft.Colors.WHITE,
                padding=15
            ),
            on_click=lambda e: save_and_close(e)
        )
    ], spacing=10)

    status_text = ft.Text(value="", color=ft.Colors.GREEN)

    # Sorting state variables
    sort_mode = "name"  # "name" or "date"
    name_sort_ascending = True  # A-Z when True, Z-A when False
    date_sort_ascending = False  # newest first when False, oldest first when True

    def update_filename(new_name):
        """Update the filename variable"""
        nonlocal config_filename
        config_filename = new_name

    def sort_files(files, mode, ascending):
        """Sort files by name or date"""
        if mode == "name":
            return sorted(files, key=lambda x: str(x[0]).lower(), reverse=not ascending)
        elif mode == "date":
            return sorted(files, key=lambda x: x[2], reverse=not ascending)
        return files

    def toggle_name_sort():
        """Toggle name sorting between A-Z and Z-A"""
        nonlocal sort_mode, name_sort_ascending
        sort_mode = "name"
        name_sort_ascending = not name_sort_ascending
        refresh_file_list()

    def toggle_date_sort():
        """Toggle date sorting between newest and oldest"""
        nonlocal sort_mode, date_sort_ascending
        sort_mode = "date"
        date_sort_ascending = not date_sort_ascending
        refresh_file_list()

    def load_existing_files():
        """Load existing .toml files from the base configs directory and subdirectories"""
        try:
            base_config_dir = default_dir
            logger.debug(f"Looking for .toml files in base directory: {base_config_dir}")

            found_files = []

            if base_config_dir.exists():
                # Find all .toml files in base directory and subdirectories
                for toml_file in base_config_dir.rglob("*.toml"):
                    rel_path = toml_file.relative_to(base_config_dir)
                    mod_time = toml_file.stat().st_mtime
                    found_files.append((rel_path, toml_file, mod_time))

                logger.debug(f"Found {len(found_files)} .toml files total")
                # Default sort by name (A-Z)
                found_files.sort(key=lambda x: str(x[0]).lower())
                return found_files
            else:
                logger.warning(f"Base config directory does not exist: {base_config_dir}")
            return []
        except Exception as e:
            logger.error(f"Error loading existing files: {e}")
            return []

    def refresh_file_list():
        """Refresh the file list display"""
        try:
            existing_files = load_existing_files()
            file_list_controls.clear()

            # Apply sorting
            if sort_mode == "name":
                sorted_files = sort_files(existing_files, "name", name_sort_ascending)
            elif sort_mode == "date":
                sorted_files = sort_files(existing_files, "date", date_sort_ascending)
            else:
                sorted_files = existing_files

            # Create sleek header row
            name_sort_text = "Name ↓" if sort_mode == "name" and not name_sort_ascending else "Name ↑" if sort_mode == "name" and name_sort_ascending else "Name"
            date_sort_text = "Date ↓" if sort_mode == "date" and not date_sort_ascending else "Date ↑" if sort_mode == "date" and date_sort_ascending else "Date"

            header_row = ft.Row([
                # Name header
                ft.Container(
                    content=ft.Row([
                        ft.Text(
                            value=name_sort_text,
                            size=13,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_300 if sort_mode == "name" else ft.Colors.GREY_300
                        ),
                        ft.Icon(
                            ft.Icons.SORT if sort_mode != "name" else (ft.Icons.ARROW_UPWARD if name_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                            size=14,
                            color=ft.Colors.BLUE_300 if sort_mode == "name" else ft.Colors.GREY_400
                        )
                    ], spacing=5),
                    width=360,  # Increased from 300 to use more space
                    height=30,
                    bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if sort_mode == "name" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                    border_radius=4,
                    padding=ft.padding.symmetric(horizontal=10, vertical=5),
                    on_click=lambda _: toggle_name_sort(),
                    ink=True
                ),
                # Date header
                ft.Container(
                    content=ft.Row([
                        ft.Text(
                            value=date_sort_text,
                            size=13,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_300 if sort_mode == "date" else ft.Colors.GREY_300
                        ),
                        ft.Icon(
                            ft.Icons.SORT if sort_mode != "date" else (ft.Icons.ARROW_UPWARD if date_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                            size=14,
                            color=ft.Colors.BLUE_300 if sort_mode == "date" else ft.Colors.GREY_400
                        )
                    ], spacing=5),
                    width=140,  # Increased from 120 to use more space
                    height=30,
                    bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if sort_mode == "date" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                    border_radius=4,
                    padding=ft.padding.symmetric(horizontal=10, vertical=5),
                    on_click=lambda _: toggle_date_sort(),
                    ink=True
                )
            ], spacing=10)
            file_list_controls.append(header_row)

            if sorted_files:
                for rel_path, full_path, mod_time in sorted_files:
                    mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")

                    file_row = ft.Row([
                        ft.TextButton(
                            text=str(rel_path).replace('\\', '/'),
                            on_click=lambda _, fp=rel_path: on_file_click(fp),
                            style=ft.ButtonStyle(
                                alignment=ft.alignment.center_left,
                                padding=ft.padding.symmetric(horizontal=10, vertical=5)
                            ),
                            width=360  # Increased from 300 to match header
                        ),
                        ft.Text(
                            value=mod_date,
                            size=12,
                            color=ft.Colors.GREY_600,
                            width=140,  # Increased from 120 to match header
                            text_align=ft.TextAlign.RIGHT
                        )
                    ], spacing=5)
                    file_list_controls.append(file_row)
            else:
                file_list_controls.append(
                    ft.Text("No .toml files found", italic=True, color=ft.Colors.GREY_600)
                )

            # Update the container content
            file_list_container.content = ft.ListView(
                controls=file_list_controls,
                spacing=2,
                expand=True,
                auto_scroll=False
            )
            # Ensure UI refresh inside dialog
            try:
                if file_list_container.page:
                    file_list_container.update()
                    # Also force a page update to reflect sorting/saving changes immediately
                    file_list_container.page.update()
            except Exception:
                pass
            # Don't call update() elsewhere; keep dialog responsive
            logger.debug("File list refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing file list: {e}")
            logger.error(traceback.format_exc())

    def on_file_click(file_path: Path):
        """Handle file selection from the list"""
        new_name = file_path.name
        update_filename(new_name)

        # Update the input field
        for control in input_row.controls:
            if isinstance(control, ft.TextField):
                control.value = new_name
                control.update()
                break

    def save_and_close(e: ft.ControlEvent | None = None):
        file_name = config_filename.strip() if config_filename else ""

        if not file_name:
            status_text.value = "Provide a configuration name."
            status_text.color = ft.Colors.RED
            # Don't call page.update() - PopupDialogBase will handle updates
            return

        current_save_dir = Path(dir_display.content.value)
        full_path = current_save_dir / file_name
        if not str(full_path).lower().endswith('.toml'):
            full_path = Path(str(full_path) + '.toml')

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(toml_text)
            logger.info(f"Config (TOML) saved (web) to {full_path}")
            TopBarUtils.set_yaml_path_and_title(page, str(full_path))
            TopBarUtils.add_recent_file(str(full_path), page)
            page.is_default_config_loaded = False
            status_text.value = f"Successfully saved to {full_path}"
            status_text.color = ft.Colors.GREEN
            # Don't call page.update() - PopupDialogBase will handle updates

            # Refresh the file list after successful save
            refresh_file_list()
        except Exception as ex:
            logger.error(f"Error saving file (web Save As) to {full_path}: {ex}")
            logger.error(traceback.format_exc())
            status_text.value = f"Error: {ex}"
            status_text.color = ft.Colors.RED
            # Don't call page.update() - PopupDialogBase will handle updates

    # Initialize controls and container first
    file_list_controls = []

    # File list container - no height limitation for PopupDialogBase
    file_list_container = ft.Container(
        content=ft.ListView(
            controls=file_list_controls,
            spacing=2,
            expand=True,
            auto_scroll=False
        ),
        height=300,  # Fixed reasonable height, but PopupDialogBase can handle larger content
        margin=ft.margin.only(top=10),
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=8,
        padding=5,
        width=500  # Increased from 450 to use full dialog width (550px - padding)
    )

    # Now refresh the list to populate the container
    refresh_file_list()

    return ft.Column([
        dir_display,
        input_row,
        ft.Text("Existing .toml files (click to use as base):", size=12, color=ft.Colors.GREY_600),
        file_list_container,
        status_text,
    ], spacing=10)


def _handle_open_web(page: ft.Page, set_as_current: bool = True):
    logger.debug("In _handle_open_web")

    # Create the dialog content using PopupDialogBase system
    dialog_content = _create_open_content(page)

    # Use PopupDialogBase if available, otherwise fall back to overlay method
    if hasattr(page, 'base_dialog') and page.base_dialog:
        logger.debug("Using PopupDialogBase for Open dialog")
        page.base_dialog.show_dialog(
            content=dialog_content,
            title="Open Configuration",
            new_width=550  # Same width as Save As dialog
        )
    else:
        logger.debug("Using fallback overlay method for Open dialog")
        # Fallback to original overlay method if PopupDialogBase not available
        _show_open_overlay(page, dialog_content)


def _handle_save_as_web(page: ft.Page):
    logger.debug("In _handle_save_as_web")

    # Create the dialog content using PopupDialogBase system
    dialog_content = _create_save_as_content(page)

    # Use PopupDialogBase if available, otherwise fall back to overlay method
    if hasattr(page, 'base_dialog') and page.base_dialog:
        logger.debug("Using PopupDialogBase for Save As dialog")
        page.base_dialog.show_dialog(
            content=dialog_content,
            title="Save Data Config",
            new_width=550  # Slightly wider for better layout
        )
    else:
        logger.debug("Using fallback overlay method for Save As dialog")
        # Fallback to original overlay method if PopupDialogBase not available
        _show_save_as_overlay(page, dialog_content)


def _show_save_as_overlay(page: ft.Page, dialog_content: ft.Control):
    """Fallback overlay method for Save As dialog"""
    # Create a modal container as fallback
    modal_container = ft.Container(
        content=ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Save Data Config", size=18, weight=ft.FontWeight.BOLD),
                    ft.Divider(),
                    dialog_content,
                    ft.Container(height=10),
                    ft.Row([
                        ft.TextButton("Cancel", on_click=lambda _: _close_save_overlay(page)),
                    ], alignment=ft.MainAxisAlignment.END),
                ], spacing=5),
                padding=20,
                width=500
            ),
            elevation=5
        ),
        bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
        alignment=ft.alignment.center,
        expand=True,
        visible=True
    )

    overlay_stack = ft.Stack([
        ft.Container(
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            expand=True,
            on_click=lambda _: _close_save_overlay(page)
        ),
        modal_container
    ], expand=True)

    page.save_dialog_overlay = overlay_stack
    page.overlay.append(overlay_stack)
    page.update()


def _close_save_overlay(page: ft.Page):
    """Close the Save As overlay dialog"""
    if hasattr(page, 'save_dialog_overlay'):
        page.overlay.remove(page.save_dialog_overlay)
        page.update()


def _show_open_overlay(page: ft.Page, dialog_content: ft.Control):
    """Fallback overlay method for Open dialog"""
    # Create a modal container as fallback
    modal_container = ft.Container(
        content=ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Open Configuration", size=18, weight=ft.FontWeight.BOLD),
                    ft.Divider(),
                    dialog_content,
                    ft.Container(height=10),
                    ft.Row([
                        ft.TextButton("Cancel", on_click=lambda _: _close_open_overlay(page)),
                    ], alignment=ft.MainAxisAlignment.END),
                ], spacing=5),
                padding=20,
                width=500
            ),
            elevation=5
        ),
        bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
        alignment=ft.alignment.center,
        expand=True,
        visible=True
    )

    overlay_stack = ft.Stack([
        ft.Container(
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            expand=True,
            on_click=lambda _: _close_open_overlay(page)
        ),
        modal_container
    ], expand=True)

    page.open_dialog_overlay = overlay_stack
    page.overlay.append(overlay_stack)
    page.update()


def _close_open_overlay(page: ft.Page):
    """Close the Open overlay dialog"""
    if hasattr(page, 'open_dialog_overlay'):
        page.overlay.remove(page.open_dialog_overlay)
        page.update()
