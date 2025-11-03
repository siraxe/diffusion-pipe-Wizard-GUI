import os
import base64
from pathlib import Path
import flet as ft
from loguru import logger
from PIL import Image
import traceback # Import traceback for detailed error logging

# Import from our new split modules
from .config_utils import (
    build_toml_config_from_ui,
    update_ui_from_toml,
    extract_config_from_controls,
    _save_and_scale_image,
)
from .file_dialogs import (
    _create_open_content,
    _create_save_as_content,
    _handle_open_web,
    _handle_save_as_web,
    _show_save_as_overlay,
    _close_save_overlay,
    _show_open_overlay,
    _close_open_overlay,
)

from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir # Import _get_dataset_base_dir
from flet_app.ui_popups.popup_dialog_base import PopupDialogBase # Import PopupDialogBase

# Initialize logger if not already set up
try:
    logger
except NameError:
    import sys
    logger.remove() # Remove default handler
    logger.add(sys.stderr, level="DEBUG") # Add new handler with DEBUG level

# selected_image_path_c1 and selected_image_path_c2 will be imported dynamically
# within handle_save and handle_save_as to get current values.

class TopBarUtils:
    @staticmethod
    def _is_web_platform(page: ft.Page) -> bool:
        platform = getattr(page, "platform", None)
        if isinstance(platform, str):
            return platform.lower() == "web"

        page_platform = getattr(ft, "PagePlatform", None)
        if page_platform is not None:
            try:
                if platform == page_platform.WEB:
                    return True
            except AttributeError:
                pass

        return bool(getattr(page, "web", False))

    @staticmethod
    def _ensure_default_config_dir() -> Path:
        from flet_app.project_root import get_project_root
        project_root = get_project_root()
        # Navigate to workspace/configs from project root
        default_dir = project_root / "workspace" / "configs"
        default_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Default config dir resolved to: {default_dir}")
        return default_dir

    
    @staticmethod
    def _load_cropped_image_into_ui(page: ft.Page, image_path: str, target_control_key: str,
                                  image_display_c1=None, image_display_c2=None):
        """
        Load a cropped image into the UI control and update the remove button visibility.
        Args:
            page: The Flet page object
            image_path: Path to the image to load
            target_control_key: Either 'c1' or 'c2' to specify which control to update
            image_display_c1: Optional reference to the C1 image display control (can be derived from page)
            image_display_c2: Optional reference to the C2 image display control (can be derived from page)
        """

        image_display_target = None
        if target_control_key.lower() == 'c1':
            image_display_target = image_display_c1 if image_display_c1 else getattr(page, 'image_display_c1', None)
        elif target_control_key.lower() == 'c2':
            image_display_target = image_display_c2 if image_display_c2 else getattr(page, 'image_display_c2', None)

        if not image_display_target:
            logger.error(f"Target display control for {target_control_key} not found in _load_cropped_image_into_ui.")
            return

        path_exists = os.path.exists(image_path) if image_path else False

        if not image_path or not path_exists:
            logger.warning(f"Image not found at '{image_path}' in _load_cropped_image_into_ui. Clearing display for {target_control_key}.")
            image_display_target.src = None # Or a placeholder like "/images/image_placeholder.png"
            image_display_target.visible = False
            if hasattr(page, f'selected_image_path_{target_control_key.lower()}'):
                 setattr(page, f'selected_image_path_{target_control_key.lower()}', None)

            try:
                if hasattr(image_display_target, 'hide_remove_button'):
                    image_display_target.hide_remove_button()
                image_display_target.update()
            except Exception as e_clear_update:
                logger.error(f"Error updating display or hide_remove_button for {target_control_key} (path not found): {e_clear_update}")
                logger.error(traceback.format_exc())
            return

        try:
            image_path_norm = image_path.replace('\\', '/')

            if hasattr(page, f'selected_image_path_{target_control_key.lower()}'):
                setattr(page, f'selected_image_path_{target_control_key.lower()}', image_path_norm)

            image_display_target.src = image_path_norm
            image_display_target.visible = True

            # Optional: Ensure the image is visible and properly sized (Flet's fit usually handles this)
            # if hasattr(image_display_target, 'width'): image_display_target.width = 200
            # if hasattr(image_display_target, 'height'): image_display_target.height = 200

            image_display_target.update()

            if hasattr(image_display_target, 'show_remove_button'):
                image_display_target.show_remove_button()
            else:
                logger.warning(f"image_display_{target_control_key} does not have show_remove_button method (called from _load_cropped_image_into_ui).")

            if hasattr(image_display_target, 'parent') and hasattr(image_display_target.parent, 'update'):
                image_display_target.parent.update()

            if page: page.update()
            logger.info(f"Successfully loaded image for {target_control_key} from '{image_path_norm}' in _load_cropped_image_into_ui.")

        except Exception as e:
            logger.error(f"Error in _load_cropped_image_into_ui for {target_control_key} with path '{image_path}': {e}")
            logger.error(traceback.format_exc())

    @staticmethod
    def set_yaml_path_and_title(page, path, set_as_current=True):
        if set_as_current:
            page.y_name = path
        filename = os.path.basename(path) if path else None
        if filename and filename.lower() == "config_default.yaml":
            page.title = "DPipe Trainer"
            TopBarUtils.update_filename_display(page, "Default Config")
        elif filename:
            page.title = f"DPipe Trainer - {filename}"
            TopBarUtils.update_filename_display(page, filename)
        else:
            page.title = "DPipe Trainer"
            TopBarUtils.update_filename_display(page, "")
        page.update()

    @staticmethod
    def update_filename_display(page, filename):
        """Update the filename label beside the menu bar.
        Works even if the control isn't mounted yet (no page), deferring update().
        """
        try:
            if hasattr(page, 'filename_display'):
                if filename and str(filename).strip():
                    page.filename_display.value = f"📄 {filename}"
                    page.filename_display.visible = True
                    page.filename_display.italic = False
                    page.filename_display.weight = ft.FontWeight.BOLD
                    page.filename_display.color = ft.Colors.BLUE_GREY_800
                else:
                    # Do not show placeholder text when no file is loaded
                    page.filename_display.value = ""
                    page.filename_display.visible = False
                    page.filename_display.italic = False
                    page.filename_display.weight = ft.FontWeight.NORMAL
                    page.filename_display.color = ft.Colors.BLUE_GREY_700
                # Only call update if the control is mounted
                if getattr(page.filename_display, 'page', None):
                    page.filename_display.update()
        except Exception as e:
            logger.warning(f"Could not update filename display: {e}")

    @staticmethod
    def handle_save(page: ft.Page):
        logger.debug("Handle Save triggered.")
        path = getattr(page, 'y_name', None)
        is_default_loaded = getattr(page, 'is_default_config_loaded', False)

        if is_default_loaded or (path and os.path.basename(path).lower() == "config_default.yaml"):
            logger.debug("Default config loaded or path is default, redirecting to Save As.")
            TopBarUtils.handle_save_as(page)
            return

        if path:
            if not path.lower().endswith('.toml'): path += '.toml'
            training_tab = getattr(page, 'training_tab_container', None)
            if not training_tab:
                logger.error("Training tab container not found while saving TOML.")
                return
            try:
                toml_text = build_toml_config_from_ui(training_tab)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(toml_text)
                logger.info(f"Config saved to {path}")
                TopBarUtils.set_yaml_path_and_title(page, path)
                TopBarUtils.add_recent_file(path, page)
                page.is_default_config_loaded = False
            except Exception as e:
                logger.error(f"Error saving file to {path}: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.debug("No path set, redirecting to Save As.")
            TopBarUtils.handle_save_as(page)

    @staticmethod
    def handle_save_as(page: ft.Page):
        logger.debug("Handle Save As triggered.")
        is_web = TopBarUtils._is_web_platform(page)
        logger.debug(f"Is web platform: {is_web}")
        if is_web:
            logger.debug("Calling _handle_save_as_web")
            _handle_save_as_web(page)
            return

        file_picker = ft.FilePicker()
        page.overlay.append(file_picker)
        page.update()
        def on_save_result(e: ft.FilePickerResultEvent):
            if e.path:
                path = e.path
                if not path.lower().endswith('.toml'): path += '.toml'
                training_tab = getattr(page, 'training_tab_container', None)
                if not training_tab:
                    logger.error("Training tab container not found while saving TOML (Save As).")
                    return
                try:
                    toml_text = build_toml_config_from_ui(training_tab)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(toml_text)
                    logger.info(f"Config saved as to {path}")
                    TopBarUtils.set_yaml_path_and_title(page, path)
                    TopBarUtils.add_recent_file(path, page)
                    page.is_default_config_loaded = False
                except Exception as ex_save:
                    logger.error(f"Error saving file (Save As) to {path}: {ex_save}")
                    logger.error(traceback.format_exc())
            else:
                logger.debug("Save As dialog cancelled or no path selected.")
        file_picker.on_result = on_save_result
        default_name = "dpipe_config.toml"
        default_dir = str(TopBarUtils._ensure_default_config_dir())
        file_picker.save_file(
            dialog_title="Save config as TOML", file_name=default_name,
            initial_directory=default_dir, allowed_extensions=["toml"]
        )

    @staticmethod
    def handle_open(page: ft.Page, file_path=None, set_as_current=True):
        logger.debug("Handle Open triggered.")
        def _has_commented_timestep(text: str) -> bool:
            try:
                import re
                return re.search(r"^\s*#\s*timestep_sample_method\b", text, flags=re.MULTILINE) is not None
            except Exception:
                return False
        def _set_timestep_none(training_tab):
            try:
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
            except Exception:
                pass
        def _has_commented_transformer_dtype(text: str) -> bool:
            try:
                import re
                return re.search(r"^\s*#\s*transformer_dtype\b", text, flags=re.MULTILINE) is not None
            except Exception:
                return False
        def _set_transformer_dtype_none(training_tab):
            try:
                def _apply(control):
                    if hasattr(control, 'controls') and control.controls:
                        for c in control.controls:
                            _apply(c)
                    if hasattr(control, 'content') and control.content:
                        _apply(control.content)
                    label = getattr(control, 'label', None)
                    if label == 'transformer_dtype' and isinstance(control, ft.Dropdown):
                        control.value = 'None'
                        if control.page:
                            control.update()
                _apply(training_tab.config_page_content)
            except Exception:
                pass
        if file_path:
            try:
                training_tab = getattr(page, 'training_tab_container', None)
                if not training_tab:
                    logger.error("Training tab container not found in handle_open (direct path).")
                    return
                toml_data = None
                raw_text = ''
                try:
                    import tomllib as _toml_parser
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        raw_text = data.decode('utf-8', errors='ignore')
                        toml_data = _toml_parser.loads(raw_text)
                except Exception:
                    try:
                        import tomli as _toml_parser
                        with open(file_path, 'rb') as f:
                            data = f.read()
                            raw_text = data.decode('utf-8', errors='ignore')
                            toml_data = _toml_parser.loads(raw_text)
                    except Exception:
                        _toml_parser = None

                if isinstance(toml_data, dict):
                    update_ui_from_toml(training_tab, toml_data)
                    if _has_commented_timestep(raw_text):
                        _set_timestep_none(training_tab)
                    if _has_commented_transformer_dtype(raw_text):
                        _set_transformer_dtype_none(training_tab)
                TopBarUtils.set_yaml_path_and_title(page, file_path, set_as_current=set_as_current)
                TopBarUtils.add_recent_file(file_path, page)
                page.is_default_config_loaded = False
            except Exception as e:
                logger.error(f"Error opening file {file_path}: {e}")
                logger.error(traceback.format_exc())
            return

        is_web = TopBarUtils._is_web_platform(page)
        logger.debug(f"Open: Is web platform: {is_web}")
        if is_web:
            logger.debug("Calling _handle_open_web")
            _handle_open_web(page, set_as_current=set_as_current)
            return

        file_picker = ft.FilePicker()
        page.overlay.append(file_picker)
        page.file_picker_open_ref = file_picker # Store ref if needed elsewhere
        page.update()
        def on_open_result(e: ft.FilePickerResultEvent):
            if e.files and len(e.files) > 0:
                picker_file = e.files[0]
                path = picker_file.path
                try:
                    if path:
                        toml_data = None
                        raw_text = ''
                        try:
                            import tomllib as _toml_parser
                            with open(path, 'rb') as f:
                                data = f.read()
                                raw_text = data.decode('utf-8', errors='ignore')
                                toml_data = _toml_parser.loads(raw_text)
                        except Exception:
                            try:
                                import tomli as _toml_parser
                                with open(path, 'rb') as f:
                                    data = f.read()
                                    raw_text = data.decode('utf-8', errors='ignore')
                                    toml_data = _toml_parser.loads(raw_text)
                            except Exception:
                                _toml_parser = None

                        training_tab = getattr(page, 'training_tab_container', None)
                        if training_tab and isinstance(toml_data, dict):
                            update_ui_from_toml(training_tab, toml_data)
                            if _has_commented_timestep(raw_text):
                                _set_timestep_none(training_tab)
                            if _has_commented_transformer_dtype(raw_text):
                                _set_transformer_dtype_none(training_tab)
                        TopBarUtils.set_yaml_path_and_title(page, path, set_as_current=set_as_current)
                        TopBarUtils.add_recent_file(path, page)
                        page.is_default_config_loaded = False
                        logger.info(f"Opened config from dialog: {path}")
                    elif picker_file.bytes is not None:
                        display_name = picker_file.name or "uploaded.toml"
                        toml_data = None
                        try:
                            import tomllib as _toml_parser
                            raw_text = picker_file.bytes.decode('utf-8', errors='ignore')
                            toml_data = _toml_parser.loads(raw_text)
                        except Exception:
                            try:
                                import tomli as _toml_parser
                                raw_text = picker_file.bytes.decode('utf-8', errors='ignore')
                                toml_data = _toml_parser.loads(raw_text)
                            except Exception:
                                pass

                        training_tab = getattr(page, 'training_tab_container', None)
                        if training_tab and isinstance(toml_data, dict):
                            update_ui_from_toml(training_tab, toml_data)
                            try:
                                if 'raw_text' in locals() and _has_commented_timestep(raw_text):
                                    _set_timestep_none(training_tab)
                                if 'raw_text' in locals() and _has_commented_transformer_dtype(raw_text):
                                    _set_transformer_dtype_none(training_tab)
                            except Exception:
                                pass
                        TopBarUtils.set_yaml_path_and_title(page, display_name, set_as_current=False)
                        logger.info(f"Opened config from upload (desktop picker): {display_name}")
                    else:
                        logger.warning("File picker returned no path or bytes.")
                except Exception as ex_open:
                    logger.error(f"Error opening file from dialog {path}: {ex_open}")
                    logger.error(traceback.format_exc())
            else:
                logger.debug("Open file dialog cancelled or no file selected.")
        file_picker.on_result = on_open_result
        default_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "configs"))
        if not os.path.exists(default_dir): os.makedirs(default_dir)
        file_picker.pick_files(
            dialog_title="Open config TOML", initial_directory=default_dir,
            allowed_extensions=["toml"], allow_multiple=False
        )

    @staticmethod
    def handle_load_default(page: ft.Page):
        logger.debug("Handle Load Default triggered.")
        # Resolve default config at flet_app/default/config_default.toml
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ,
            'default',
            'config_default.toml'
        )
        if not os.path.exists(default_path):
            logger.error(f"Default config file not found at: {default_path}")
            # Optionally show a message to the user in the UI
            if hasattr(page, "show_snackbar"): # Assuming a snackbar method
                page.show_snackbar(ft.SnackBar(ft.Text(f"Error: Default config not found."), open=True))
            return

        TopBarUtils.handle_open(page, file_path=default_path, set_as_current=False) # set_as_current=False for default
        page.is_default_config_loaded = True
        TopBarUtils.update_filename_display(page, "Default Config")

        training_tab = getattr(page, 'training_tab_container', None)
        if training_tab:
            # Preserve current dataset selection across panel rebuild
            prev_ds = None
            try:
                old_ds_block = getattr(training_tab, 'dataset_page_content', None)
                if old_ds_block and hasattr(old_ds_block, 'get_selected_dataset'):
                    prev_ds = old_ds_block.get_selected_dataset()
            except Exception:
                prev_ds = None

            # Rebuild the Config panel and swap it in to force a clean visual state
            try:
                from flet_app.ui.pages.training_config import get_training_config_page_content
                cfg_new = get_training_config_page_content()
                training_tab.config_page_content = cfg_new
                if hasattr(cfg_new, 'dataset_block'):
                    training_tab.dataset_page_content = cfg_new.dataset_block
                    # Restore previous dataset selection if any
                    try:
                        if prev_ds and hasattr(cfg_new.dataset_block, 'set_selected_dataset'):
                            cfg_new.dataset_block.set_selected_dataset(prev_ds, page)
                    except Exception:
                        pass
                content_area = getattr(training_tab, 'content_area', None)
                if content_area is not None:
                    content_area.content = cfg_new
                if page:
                    page.update()
                # Re-apply the same TOML to the fresh controls (won't set dataset due to empty string)
                try:
                    import tomllib as _toml_parser
                    with open(default_path, 'rb') as f:
                        toml_data = _toml_parser.load(f)
                    if isinstance(toml_data, dict):
                        update_ui_from_toml(training_tab, toml_data)
                except Exception:
                    pass
                # Explicitly clear the Model Type dropdown to ensure visual reset
                try:
                    def _clear_model_type(control):
                        try:
                            if hasattr(control, 'controls') and control.controls:
                                for c in control.controls:
                                    _clear_model_type(c)
                            if hasattr(control, 'content') and control.content:
                                _clear_model_type(control.content)
                            label = getattr(control, 'label', None)
                            if label == 'Model Type' and isinstance(control, ft.Dropdown):
                                # Force-clear by temporarily inserting an empty option, selecting it, then removing
                                try:
                                    has_empty = any(getattr(opt, 'key', None) == '' for opt in (control.options or []))
                                except Exception:
                                    has_empty = False
                                try:
                                    if not has_empty:
                                        control.options = [ft.dropdown.Option('', text='')] + list(control.options or [])
                                    control.value = ''
                                    if control.page:
                                        control.update()
                                    # Now set to None and remove the temporary option to end with a cleared dropdown
                                    control.value = None
                                    if not has_empty:
                                        control.options = [opt for opt in control.options if getattr(opt, 'key', None) != '']
                                    if control.page:
                                        control.update()
                                except Exception:
                                    # Fallback simple clear
                                    control.value = None
                                    if control.page:
                                        control.update()
                        except Exception:
                            pass
                    _clear_model_type(cfg_new)
                    if page:
                        page.update()
                except Exception:
                    pass
            except Exception:
                pass

            # Emulate a quick tab switch to trigger UI refresh
            try:
                if hasattr(training_tab, 'refresh_config_panel') and callable(getattr(training_tab, 'refresh_config_panel')):
                    training_tab.refresh_config_panel()
            except Exception:
                pass
        else:
            logger.warning("Training tab container not found after loading default config.")
        page.update() # Ensure page updates after clearing dataset

    @staticmethod
    def get_recent_files_path():
        # Ensure the path is relative to this file's directory if it's intended to be bundled
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recent_files.txt')

    @staticmethod
    def load_recent_files():
        path = TopBarUtils.get_recent_files_path()
        if not os.path.exists(path): return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                files = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())] # Also check existence
            return files
        except Exception as e:
            logger.error(f"Error loading recent files from {path}: {e}")
            return []

    @staticmethod
    def save_recent_files(files):
        path = TopBarUtils.get_recent_files_path()
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                for file_path in files[:5]: # Save top 5 recent files
                    f.write(file_path + '\n')
        except Exception as e:
            logger.error(f"Error saving recent files to {path}: {e}")

    @staticmethod
    def add_recent_file(filepath, page=None):
        if not filepath or not os.path.exists(filepath): # Do not add non-existent files
            logger.warning(f"Attempted to add non-existent or empty filepath to recents: '{filepath}'")
            return
        files = TopBarUtils.load_recent_files()
        if filepath in files: files.remove(filepath)
        files.insert(0, filepath)
        TopBarUtils.save_recent_files(files)
        if page and hasattr(page, 'refresh_menu_bar'):
            page.refresh_menu_bar()

    @staticmethod
    def get_recent_files_menu_items(on_click_handler, text_size=10): # Renamed on_click to on_click_handler
        files = TopBarUtils.load_recent_files()
        if not files:
            return [ft.MenuItemButton(content=ft.Text("None", size=text_size, italic=True), disabled=True)]
        items = []
        for f_path in files:
            # Display a shorter version of the path, e.g., "configs/my_config.yaml"
            try:
                # Try to make it relative to 'workspace' or show last two parts
                workspace_dir = os.path.abspath("workspace")
                if f_path.startswith(workspace_dir):
                    display_name = os.path.relpath(f_path, os.path.dirname(workspace_dir))
                else:
                    display_name = os.path.join(os.path.basename(os.path.dirname(f_path)), os.path.basename(f_path))
            except Exception:
                display_name = os.path.basename(f_path) # Fallback

            items.append(ft.MenuItemButton(content=ft.Text(display_name, size=text_size, tooltip=f_path),
                                          on_click=on_click_handler, data=f_path))
        return items
