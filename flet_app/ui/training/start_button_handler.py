"""
Training Start Button Handler

Handles Start/Cancel button logic, process management, and training orchestration.
"""

import os
import signal
import traceback
import asyncio
import flet as ft
from loguru import logger

from .output_manager import (
    add_info_message,
    add_success_message,
    add_warning_message,
    add_error_message,
    add_action_message,
)
from musubi_ltx2 import run_musubi_ltx2_workflow
from flet_app.ui.utils.toml_to_musubi_toml import convert_toml_to_musubi_toml


# =====================
# Button State Management
# =====================

def set_button_state(main_container, text: str, page=None):
    """
    Set the Start/Cancel button text and update the page.

    Args:
        main_container: Main UI container
        text: Button text ("Start" or "Cancel")
        page: Optional Flet page to update
    """
    try:
        start_btn = getattr(main_container, 'start_btn', None)
        if start_btn is not None:
            start_btn.text = text
        if page is not None:
            page.update()
    except Exception as e:
        logger.error(f"Error setting button state: {e}")


def reset_to_start_button(main_container, training_tab_container, page=None):
    """
    Reset the button back to "Start" and clear process handles.

    Args:
        main_container: Main UI container
        training_tab_container: Training tab container
        page: Optional Flet page to update
    """
    try:
        if hasattr(main_container, 'start_btn') and main_container.start_btn is not None:
            main_container.start_btn.text = "Start"
        if training_tab_container is not None:
            training_tab_container.training_proc = None
        main_container.training_proc = None
        if page is not None:
            page.update()
    except Exception as e:
        logger.error(f"Error resetting button: {e}")


# =====================
# Process Termination
# =====================

def terminate_process(training_proc, main_container, page=None):
    """
    Terminate a running training process with proper cleanup.

    Args:
        training_proc: subprocess.Popen object to terminate
        main_container: Main UI container
        page: Optional Flet page to update
    """
    # Append cancel notice to console
    try:
        monitor_content = getattr(main_container, 'monitor_page_content', None)
        training_console_text = getattr(monitor_content, 'training_console_text', None)
        add_action_message(training_console_text, "\n[Action] Training cancelled. Terminating process...\n")
        if training_console_text is not None and training_console_text.page is not None:
            training_console_text.update()
    except Exception as e:
        logger.error(f"Error adding cancel notice: {e}")

    # Request termination (robustly)
    try:
        if os.name == 'posix':
            try:
                os.killpg(training_proc.pid, signal.SIGTERM)
            except Exception:
                training_proc.terminate()
        else:
            try:
                training_proc.send_signal(getattr(signal, 'CTRL_BREAK_EVENT', signal.SIGTERM))
            except Exception:
                training_proc.terminate()

        # Brief wait and force kill if still alive
        try:
            import time as _t
            for _ in range(30):
                if training_proc.poll() is not None:
                    break
                _t.sleep(0.1)
            if training_proc.poll() is None:
                if os.name == 'posix':
                    try:
                        os.killpg(training_proc.pid, signal.SIGKILL)
                    except Exception:
                        training_proc.kill()
                else:
                    training_proc.kill()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Error terminating process: {e}")

    # Add cancellation message
    try:
        monitor_content = getattr(main_container, 'monitor_page_content', None)
        training_console_text = getattr(monitor_content, 'training_console_text', None)
        add_action_message(training_console_text, "\n[Action] Training process terminated.\n")
        if training_console_text is not None and training_console_text.page is not None:
            training_console_text.update()
    except Exception:
        pass


# =====================
# Start Button Click Handler (Cancel logic)
# =====================

def handle_cancel_click(e, main_container):
    """
    Handle the Cancel button click - terminate running process.

    Args:
        e: Flet event
        main_container: Main UI container

    Returns:
        True if process was cancelled, False if no process was running
    """
    from musubi_ltx2 import handle_musubi_model as handle_ltx_model

    training_proc = None

    # Try to get training_proc from LTX2 specific container
    try:
        if hasattr(e.page, 'training_tab_container'):
            training_tab = e.page.training_tab_container
            # Check if this is LTX2 by checking the config
            last_config_path = getattr(training_tab, 'last_config_path', None)
            if last_config_path and os.path.exists(last_config_path):
                if handle_ltx_model(last_config_path):
                    training_proc = getattr(training_tab, 'training_proc', None)
                    # Also check main_container as backup
                    if training_proc is None:
                        training_proc = getattr(main_container, 'training_proc', None)
    except Exception:
        pass

    # Fallback to main_container if not found
    if training_proc is None:
        training_proc = getattr(main_container, 'training_proc', None)

    if training_proc is not None:
        try:
            alive = (training_proc.poll() is None)
        except Exception:
            alive = False

        if alive:
            # First, set training_proc to None so the polling loop detects cancellation immediately
            try:
                if hasattr(e.page, 'training_tab_container'):
                    e.page.training_tab_container.training_proc = None
                main_container.training_proc = None
            except Exception:
                pass

            # Then terminate the process
            terminate_process(training_proc, main_container, e.page)

            # Reset button to Start
            reset_to_start_button(main_container, None, e.page)
            return True

    return False


# =====================
# LTX2 Training Orchestration
# =====================

async def run_ltx2_training_flow(
    out_path,
    trust_cache,
    cache_only,
    resume_last,
    use_last_config,
    main_container,
    training_tab_container,
    page,
    trust_cache_checkbox
):
    """
    Orchestrate the LTX2 training flow including cache creation and training.

    Args:
        out_path: Path to config file
        trust_cache: Whether to skip cache creation
        cache_only: Whether to only create cache (no training)
        resume_last: Whether to resume from last saved state
        use_last_config: Whether to reuse existing YAML without regenerating
        main_container: Main UI container
        training_tab_container: Training tab container
        page: Flet page
        trust_cache_checkbox: Checkbox reference
    """
    from musubi_ltx2 import handle_musubi_model as handle_ltx_model

    # Get monitor components
    monitor_content = getattr(training_tab_container, 'monitor_page_content', None)
    training_console_text = getattr(monitor_content, 'training_console_text', None)
    training_cmd_text = getattr(monitor_content, 'training_cmd_text', None)
    training_cmd_container = getattr(monitor_content, 'training_cmd_container', None)

    # Prepare Training Console in Monitor tab
    try:
        if training_console_container := getattr(monitor_content, 'training_console_container', None):
            training_console_container.visible = True
        if training_console_text is not None:
            training_console_text.spans = []
            from flet_app.ui.utils.console_cleanup import cleanup_training_console
            cleanup_training_console(training_console_text)
        if page is not None:
            page.update()
    except Exception:
        pass

    # Get trust_cache and cache_only values
    trust_cache = trust_cache_checkbox.value
    cache_only = cache_only if isinstance(cache_only, bool) else False

    # Skip LTX-2 caching - go directly to musubi config creation
    # The old LTX-2 process_dataset flow has been removed

    # Now run training (for LTX model)
    # Convert TOML config to Musubi TOML and run LTX-2 training

    # Get paths
    ws_dir = os.path.dirname(out_path)
    last_data_config_path = os.path.join(ws_dir, 'last_data_config.toml')
    last_config_path = out_path  # This is the last_config.toml

    # Convert to musubi TOML format
    musubi_result = convert_toml_to_musubi_toml(last_data_config_path, last_config_path)
    musubi_config_path = musubi_result.get('output_path', os.path.join(ws_dir, 'last_data_musubi_config.toml'))

    # Check if slider config was created
    slider_config_path = musubi_result.get('slider_config_path')
    if slider_config_path:
        add_info_message(training_console_text, f"\n[Info] Slider config created at: {slider_config_path}\n")

    # TODO: Implement musubi training using musubi_config_path
    # The musubi config has been created at: musubi_config_path
    add_info_message(training_console_text, f"\n[Info] Musubi config created at: {musubi_config_path}\n")

    # Determine the workflow mode
    if cache_only:
        mode = 'cache_only'
    elif trust_cache:
        mode = 'trust_cache'
    else:
        mode = 'full'

    # Run the musubi workflow (all logic handled in musubi_ltx2.py)
    proc = run_musubi_ltx2_workflow(
        last_config_path=last_config_path,
        musubi_config_path=musubi_config_path,
        mode=mode,
        resume_last=resume_last,
        training_console_text=training_console_text,
        main_container=main_container,
        page=page,
        slider_config_path=slider_config_path
    )

    # Update UI based on result
    if proc is not None:
        # Process started - update button to Stop
        set_button_state(main_container, "Stop", page)
    else:
        # No process started - reset button
        reset_to_start_button(main_container, training_tab_container, page)
