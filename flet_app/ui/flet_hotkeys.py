# =====================
# Hotkey Functions
# =====================

# Import lazily inside handlers to avoid circular imports

# --- Global Hotkey Logic ---
AUTO_PLAYBACK = True  # Set to True to auto-play media on open/switch
is_d_key_pressed_global = False  # Global state for 'D' key
is_shift_key_pressed_global = False  # Global state for 'Shift' modifier (for range selection)
# Track Ctrl for modifier-based actions (e.g., Ctrl+Click to open context menu)
is_ctrl_key_pressed_global = False

# --- Global Hotkey Keybindings ---
PLAY_PAUSE_KEY = " "  # Spacebar
NEXT_KEY = "]"
PREV_KEY = "["
D_KEY = "D" # Define D key

def global_hotkey_handler(page, e):
    """
    Handles global keyboard shortcuts and dialog hotkeys.
    - Esc: Close dialog
    - Ctrl+S, Ctrl+Shift+S, Ctrl+O, Ctrl+F: Menu hotkeys
    - D: Toggles global D key state for range selection
    """
    global is_d_key_pressed_global, is_shift_key_pressed_global

    # Handle D key press for global state (Flet's on_keyboard_event is keydown only)
    if hasattr(e, 'key') and e.key.upper() == D_KEY:
        is_d_key_pressed_global = True
        # This flag will be reset by the consuming UI component (e.g., dataset_layout_tab)
        # after it processes the D-modified action.
        # Do not return here, allow other handlers to process if needed.

    # Track Shift modifier globally to enable range selection in web builds.
    # Since on_keyboard_event is keydown only, the consumer resets this flag after use.
    if hasattr(e, 'shift') and bool(e.shift):
        is_shift_key_pressed_global = True
    # Also set when the Shift key itself is pressed (some platforms don't set e.shift on pure Shift keydown)
    if hasattr(e, 'key'):
        k = str(e.key).lower()
        if k in ('shift', 'shiftleft', 'shiftright'):
            is_shift_key_pressed_global = True

    # Track Ctrl modifier similarly; consumer will reset after use
    if hasattr(e, 'ctrl') and bool(e.ctrl):
        is_ctrl_key_pressed_global = True
    if hasattr(e, 'key'):
        k2 = str(e.key).lower()
        if k2 in ('control', 'ctrl', 'controlleft', 'controlright'):
            is_ctrl_key_pressed_global = True

    # If a media dialog is open and has a handler, call it
    if getattr(page, 'image_dialog_open', False) and getattr(page, 'image_dialog_hotkey_handler', None):
        page.image_dialog_hotkey_handler(e)
        return
    if getattr(page, 'video_dialog_open', False) and getattr(page, 'video_dialog_hotkey_handler', None):
        page.video_dialog_hotkey_handler(e)
        return

    # Esc key closes base dialog if open
    if hasattr(e, 'key') and e.key == 'Escape':
        if hasattr(page, 'base_dialog') and getattr(page.base_dialog, 'visible', False):
            page.base_dialog.hide_dialog()
            return

    # Global hotkeys for menu actions
    if hasattr(e, 'ctrl') and e.ctrl:
        # Ctrl+Shift+S (Save As)
        if hasattr(e, 'shift') and e.shift and hasattr(e, 'key') and e.key.lower() == 's':
            from flet_app.ui.utils.utils_top_menu import TopBarUtils
            TopBarUtils.handle_save_as(page)
        # Ctrl+S (Save)
        elif hasattr(e, 'key') and e.key.lower() == 's':
            from flet_app.ui.utils.utils_top_menu import TopBarUtils
            TopBarUtils.handle_save(page)
        # Ctrl+O (Open)
        elif hasattr(e, 'key') and e.key.lower() == 'o':
            from flet_app.ui.utils.utils_top_menu import TopBarUtils
            TopBarUtils.handle_open(page)
        # Ctrl+F (Open Base Dialog)
        elif hasattr(e, 'key') and e.key.lower() == 'f':
            if hasattr(page, 'base_dialog'):
                page.base_dialog.show_dialog()
