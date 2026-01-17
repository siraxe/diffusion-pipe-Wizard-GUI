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
    start_output_streamer,
    start_ltx_output_streamer,
    start_buffered_output_streamer,
    TensorBoardLogger,
)


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
    from LTX_2 import handle_ltx_model

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
        main_container: Main UI container
        training_tab_container: Training tab container
        page: Flet page
        trust_cache_checkbox: Checkbox reference
    """
    from LTX_2 import handle_ltx_model, run_process_dataset, run_ltx_training
    from flet_app.ui.utils.toml_to_yaml import convert_toml_to_ltx2_yaml

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

    # Run process_dataset command only if trust_cache is not enabled
    if not trust_cache and not cache_only:
        # Change button to Cancel BEFORE blocking operation
        set_button_state(main_container, "Cancel", page)

        # Show status message before starting cache creation
        add_info_message(training_console_text, "\n[Status] Starting cache creation (process_dataset.py)...\n")
        if page is not None:
            page.update()

        try:
            proc, cmd_str = await run_process_dataset(out_path)
        except Exception as e:
            add_error_message(training_console_text, f"\n[Error] Failed to start cache creation: {e}\n")
            logger.error(f"Error in run_process_dataset: {e}")
            logger.error(traceback.format_exc())
            reset_to_start_button(main_container, training_tab_container, page)
            return

        # Store process handle
        try:
            training_tab_container.training_proc = proc
            main_container.training_proc = proc
        except Exception:
            pass

        # Show command used
        try:
            if training_cmd_text is not None:
                training_cmd_text.value = cmd_str
            if training_cmd_container is not None:
                training_cmd_container.visible = True
            if page is not None:
                page.update()
        except Exception:
            pass

        # Stream output to Training Console
        start_ltx_output_streamer(proc, training_console_text)

        # Wait for process to complete using polling (to allow cancellation)
        import time as _time
        exit_code = None
        try:
            while True:
                exit_code = proc.poll()
                if exit_code is not None:
                    break
                # Check if process was cancelled (training_proc set to None)
                current_proc = getattr(training_tab_container, 'training_proc', None)
                if current_proc is None:
                    # Process was cancelled, kill it and skip training
                    try:
                        proc.terminate()
                        proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    add_error_message(training_console_text, "\n[Status] Cache creation cancelled. Skipping training.\n")
                    if page is not None:
                        page.update()
                    reset_to_start_button(main_container, None, page)
                    return
                await asyncio.sleep(0.2)  # Poll every 200ms

            # If process was cancelled (terminated by signal), skip training
            if exit_code < 0:
                add_error_message(training_console_text, f"\n[Status] Cache creation cancelled. Skipping training.\n")
                if page is not None:
                    page.update()
                reset_to_start_button(main_container, training_tab_container, page)
                return
        except Exception as e:
            logger.error(f"Error waiting for process: {e}")

        # Show completion message
        add_success_message(training_console_text, "\n[Status] Cache creation complete. Starting training...\n")
        if page is not None:
            page.update()

    elif cache_only:
        # Change button to Cancel BEFORE blocking operation
        set_button_state(main_container, "Cancel", page)

        # Show status message
        add_warning_message(training_console_text, "\n[Cache Only] Starting dataset caching (process_dataset.py)...\n")
        if page is not None:
            page.update()

        # cache_only enabled: run dataset preprocessing but skip training
        try:
            proc, cmd_str = await run_process_dataset(out_path)
        except Exception as e:
            add_error_message(training_console_text, f"\n[Error] Failed to start cache creation: {e}\n")
            logger.error(f"Error in run_process_dataset (cache_only): {e}")
            logger.error(traceback.format_exc())
            reset_to_start_button(main_container, training_tab_container, page)
            return

        # Store process handle
        try:
            training_tab_container.training_proc = proc
            main_container.training_proc = proc
        except Exception:
            pass

        # Show command used
        try:
            if training_cmd_text is not None:
                training_cmd_text.value = cmd_str
            if training_cmd_container is not None:
                training_cmd_container.visible = True
            if page is not None:
                page.update()
        except Exception:
            pass

        # Stream output to Training Console
        start_ltx_output_streamer(proc, training_console_text)

        # Wait for process to complete using polling (to allow cancellation)
        exit_code = None
        try:
            while True:
                exit_code = proc.poll()
                if exit_code is not None:
                    break
                # Check if process was cancelled (training_proc set to None)
                current_proc = getattr(training_tab_container, 'training_proc', None)
                if current_proc is None:
                    # Process was cancelled, kill it
                    try:
                        proc.terminate()
                        proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    add_error_message(training_console_text, "\n[Cache Only] Dataset caching cancelled.\n")
                    if training_console_text is not None and training_console_text.page is not None:
                        training_console_text.update()
                    reset_to_start_button(main_container, None, page)
                    return
                await asyncio.sleep(0.2)  # Poll every 200ms

            # Return early since cache_only skips training
            if exit_code < 0:
                add_error_message(training_console_text, f"\n[Cache Only] Dataset caching cancelled. Exit code: {exit_code}\n")
            else:
                add_action_message(training_console_text, f"\n[Cache Only] Dataset caching complete. Exit code: {exit_code}\n")
            if training_console_text is not None and training_console_text.page is not None:
                training_console_text.update()
        except Exception as e:
            logger.error(f"Error in cache_only wait: {e}")

        reset_to_start_button(main_container, training_tab_container, page)
        return

    else:
        # trust_cache enabled: skip preprocessing and go directly to training
        try:
            if training_cmd_text is not None:
                training_cmd_text.value = "[Skipped] process_dataset.py (trust_cache enabled)"
            if training_cmd_container is not None:
                training_cmd_container.visible = True
            if page is not None:
                page.update()
        except Exception:
            pass

    # Now run training (for LTX model)
    # Convert TOML config to YAML and run LTX-2 training
    yaml_result = convert_toml_to_ltx2_yaml(out_path)
    yaml_config_path = yaml_result.get('yaml_path', os.path.join(os.getcwd(), 'diffusion-trainers/workspace/last_config.yaml'))

    # Create TensorBoard logger with output_dir from YAML config
    tb_logger = None
    try:
        import yaml as yaml_lib
        with open(yaml_config_path, 'r') as f:
            yaml_config = yaml_lib.safe_load(f)
        output_dir = yaml_config.get('output_dir', os.getcwd())
        tb_logger = TensorBoardLogger(output_dir)
    except Exception as e:
        logger.debug(f"Could not create TensorBoard logger: {e}")

    # Show status message before starting training
    add_info_message(training_console_text, "\n[Status] Starting LTX-2 training...\n")

    # Change button to Cancel BEFORE blocking operation
    set_button_state(main_container, "Cancel", page)

    # Run training
    try:
        train_proc, train_cmd_str = await run_ltx_training(yaml_config_path)
    except Exception as e:
        add_error_message(training_console_text, f"\n[Error] Failed to start training: {e}\n")
        logger.error(f"Error in run_ltx_training: {e}")
        logger.error(traceback.format_exc())
        if tb_logger is not None:
            tb_logger.close()
        reset_to_start_button(main_container, training_tab_container, page)
        return

    # Store process handle
    try:
        training_tab_container.training_proc = train_proc
        main_container.training_proc = train_proc
    except Exception:
        pass

    # Show training command used
    try:
        if training_cmd_text is not None:
            training_cmd_text.value = train_cmd_str
        if training_cmd_container is not None:
            training_cmd_container.visible = True
        if page is not None:
            page.update()
    except Exception:
        pass

    # Stream training output to Training Console with TensorBoard logging
    start_ltx_output_streamer(train_proc, training_console_text, tb_logger)
