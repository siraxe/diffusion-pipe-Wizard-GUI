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
from musubi_run import create_runner
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
            # Explicitly update the button itself
            start_btn.update()
            logger.info(f"Set button to '{text}', button now shows: '{start_btn.text if start_btn else 'N/A'}'")
        else:
            logger.warning("start_btn is None in main_container!")
        if page is not None:
            page.update()
        else:
            logger.warning("page is None in set_button_state!")
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
# Cache Execution (Modular - shared by ltx-video-2 and wan22)
# =====================

async def run_cache_commands(
    runner,
    dataset_config: str,
    main_container,
    training_tab_container,
    page,
    training_console_text,
    slider_config: str = None,
    cache_types: list = None,
):
    """
    Execute cache commands sequentially (latents, then text_encoder).
    Uses musubi_run.run_cache_async() for each cache, then chains the rest.

    Args:
        runner: MusubiRun instance
        dataset_config: Path to dataset config
        main_container: Main UI container
        training_tab_container: Training tab container
        page: Flet page
        training_console_text: Console output control
        slider_config: Path to slider config (for i2v mode detection)
        cache_types: Optional list of cache types to run (e.g., ['sample_prompts'])
                     If None, runs all available cache types
    """
    import subprocess
    import threading

    def run_cache_thread():
        try:
            cache_cmds_dict = runner.get_cache_commands(dataset_config, slider_config)
            # Dynamic cache order based on what commands are available
            # Priority: i2v_preprocess > latents > text_encoder > sample_prompts
            cache_priority = ['i2v_preprocess', 'latents', 'text_encoder', 'sample_prompts']

            # Filter by cache_types if specified
            if cache_types:
                cache_order = [ct for ct in cache_priority if ct in cache_cmds_dict and ct in cache_types]
            else:
                cache_order = [ct for ct in cache_priority if ct in cache_cmds_dict]

            # Track overall success/failure state
            all_success = True
            was_cancelled = False

            # Flag to track if current cache was cancelled
            cache_cancelled = False

            for cache_type in cache_order:

                # Print progress between caches
                pass

                # Get command(s) for this cache type
                cmd_data = cache_cmds_dict[cache_type]

                # Handle both single command (list) and multiple commands (list of lists)
                # i2v_preprocess can have multiple commands for multiple video directories
                if cmd_data and isinstance(cmd_data[0], list):
                    # List of lists - multiple commands to run sequentially
                    cmd_list = cmd_data[0]  # Use first command for display
                else:
                    cmd_list = cmd_data

                cmd_str = " ".join(cmd_list)

                # Capture current values for closures (avoid reference issues)
                current_type = cache_type

                async def show_start(ct=current_type, cs=cmd_str):
                    add_info_message(training_console_text, f"\n[Info] Running {ct.upper()} caching...\n")
                    add_info_message(training_console_text, f"\n[Command] {cs}\n")
                if page:
                    page.run_task(show_start)

                # Run the cache command using musubi_run
                proc = runner.run_cache_async(dataset_config, cache_type=cache_type, slider_config=slider_config)

                main_container.training_proc = proc
                training_tab_container.training_proc = proc

                async def update_btn():
                    set_button_state(main_container, "Stop", page)
                if page:
                    page.run_task(update_btn)

                from flet_app.ui.training.output_manager import start_ltx_output_streamer
                start_ltx_output_streamer(proc, training_console_text, main_container=main_container, page=page)
                proc.wait()

                # Process completed
                pass

                # Check if cancelled after process completes
                # The process reference might be cleared by the output manager on normal completion
                # or by user cancellation. We need to distinguish between these cases.
                if proc.returncode == 0:
                    # Process completed successfully - any clearing of training_proc is expected
                    pass
                elif getattr(main_container, 'training_proc', None) is None:
                    # Process reference was cleared but it didn't complete successfully
                    # This indicates it was cancelled by the user
                    was_cancelled = True
                    cache_cancelled = True
                    async def show_cancel(ct=current_type):
                        add_action_message(training_console_text, f"\n[Action] {ct.upper()} caching cancelled.\n")
                    if page:
                        page.run_task(show_cancel)
                    # Don't return - continue to next cache type
                    all_success = False
                    continue

                if proc.returncode != 0:
                    async def show_err(ct=current_type):
                        add_error_message(training_console_text, f"\n[Error] {ct.upper()} caching failed with exit code {proc.returncode}\n")
                    if page:
                        page.run_task(show_err)
                    # Don't return - continue to next cache type
                    all_success = False
                    continue

                async def show_success(ct=current_type):
                    add_success_message(training_console_text, f"\n[Success] {ct.upper()} caching completed.\n")
                if page:
                    page.run_task(show_success)

                # Cache completed successfully, moving to next
                pass

            # Show final status after all caches processed
            async def show_final_status():
                if was_cancelled:
                    add_warning_message(training_console_text, f"\n[Info] Caching was cancelled. Partial caches may exist.\n")
                elif all_success:
                    add_success_message(training_console_text, f"\n[Success] All caching completed for {runner.model_type}.\n")
                else:
                    add_warning_message(training_console_text, f"\n[Warning] Caching completed with errors for {runner.model_type}.\n")
                reset_to_start_button(main_container, training_tab_container, page)
            if page:
                page.run_task(show_final_status)

        except Exception as e:
            error_msg = str(e)  # Capture error message before async context
            async def show_err(err_msg=error_msg):
                add_error_message(training_console_text, f"\n[Error] Failed to run cache commands: {err_msg}\n")
            if page:
                page.run_task(show_err)
            logger.error(f"Failed to run cache commands: {e}")

            async def reset_btn():
                reset_to_start_button(main_container, training_tab_container, page)
            if page:
                page.run_task(reset_btn)

    # Start the cache thread and wait for it to complete without blocking event loop
    cache_thread = threading.Thread(target=run_cache_thread, daemon=False)
    cache_thread.start()

    # Wait for thread to complete while allowing event loop to process UI updates
    while cache_thread.is_alive():
        await asyncio.sleep(0.1)  # Yield control every 100ms to allow UI updates


# =====================
# Start Button Click Handler (Cancel logic)
# =====================

def handle_cancel_click(e, main_container):
    training_proc = None

    # Try to get training_proc from training_tab_container
    if hasattr(e.page, 'training_tab_container'):
        training_proc = getattr(e.page.training_tab_container, 'training_proc', None)

    # Fallback to main_container
    if training_proc is None:
        training_proc = getattr(main_container, 'training_proc', None)

    if training_proc is not None:
        try:
            alive = (training_proc.poll() is None)
        except Exception:
            alive = False

        if alive:
            # Set training_proc to None first
            if hasattr(e.page, 'training_tab_container'):
                e.page.training_tab_container.training_proc = None
            main_container.training_proc = None

            terminate_process(training_proc, main_container, e.page)
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
    trust_cache_checkbox,
    reset_optimizer=False,
    reset_optimizer_params=False
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

    # Run using musubi_run wrapper
    runner = create_runner(last_config_path)
    dataset_config = musubi_config_path

    # Check if training is supported for this model
    if not runner.run_handler:
        add_info_message(training_console_text, f"\n[Info] Training not implemented for {runner.model_type} yet\n")
        add_info_message(training_console_text, f"\n[Info] Running cache commands for {runner.model_type}...\n")

        # Run cache commands using the shared function
        await run_cache_commands(runner, dataset_config, main_container, training_tab_container, page, training_console_text, slider_config_path)
        return

    # Find resume path if needed
    resume_path = None
    if resume_last:
        from musubi_utils import find_last_state_directory
        output_dir = runner.get_config().get('model', {}).get('output_dir', 'output/ltx2_lora')
        output_name = runner.get_config().get('model', {}).get('name', '')
        resume_path = find_last_state_directory(output_dir, output_name)
        if resume_path:
            add_info_message(training_console_text, f"\n[Resume] Found state: {resume_path}\n")
        else:
            add_warning_message(training_console_text, f"\n[Resume] No state found\n")

    # Run cache commands before training
    # In trust_cache mode: skip latents/text_encoder but still run sample_prompts caching
    # In cache_only mode: run all caching then stop
    # In full mode: run all caching then train

    if mode == 'trust_cache':
        add_info_message(training_console_text, f"\n[Info] Trust-cache mode: Skipping latents/text_encoder caching\n")
        # Only run sample_prompts caching in trust_cache mode
        add_info_message(training_console_text, f"\n[Info] Running sample prompts caching...\n")
        if training_console_text.page:
            training_console_text.update()

        # Run only sample_prompts cache
        await run_cache_commands(runner, dataset_config, main_container, training_tab_container, page, training_console_text, slider_config_path, cache_types=['sample_prompts'])
    elif mode == 'cache_only':
        add_info_message(training_console_text, f"\n[Info] Cache-only mode: Running all cache commands\n")
        if training_console_text.page:
            training_console_text.update()

        # Run all cache commands
        await run_cache_commands(runner, dataset_config, main_container, training_tab_container, page, training_console_text, slider_config_path)
        return  # Stop after caching
    else:  # full mode
        add_info_message(training_console_text, f"\n[Info] Running cache commands before training...\n")
        if training_console_text.page:
            training_console_text.update()

        # Run all cache commands
        await run_cache_commands(runner, dataset_config, main_container, training_tab_container, page, training_console_text, slider_config_path)

    # Build training command
    cmd = runner.get_training_command(dataset_config, slider_config_path, resume_path, reset_optimizer, reset_optimizer_params)

    # Print the training command for reference (sorted and formatted)
    add_info_message(training_console_text, f"\n[Info] Training command:\n")

    # Keep accelerate launch and script path at start, sort the rest
    base_cmd = []
    flags = []
    i = 0
    # Skip 'accelerate', 'launch', and --num_cpu_threads_per_process + its value
    while i < len(cmd):
        base_cmd.append(cmd[i])
        if cmd[i] == '--num_cpu_threads_per_process' and i + 1 < len(cmd):
            base_cmd.append(cmd[i + 1])
            i += 2
        elif cmd[i].endswith('.py'):
            # Script path - include it, then rest are flags
            i += 1
            break
        else:
            i += 1

    # Remaining items are flags
    while i < len(cmd):
        flags.append(cmd[i])
        i += 1

    # Sort flags alphabetically (group --flag with its value(s))
    def sort_flags(flags_list):
        flag_pairs = []
        i = 0
        while i < len(flags_list):
            item = flags_list[i]
            if item.startswith('--') and i + 1 < len(flags_list) and not flags_list[i + 1].startswith('--'):
                # Check if this is --optimizer_args with multiple quoted values
                if item == '--optimizer_args':
                    # Collect all following quoted strings (values that don't start with --)
                    values = []
                    j = i + 1
                    while j < len(flags_list) and not flags_list[j].startswith('--'):
                        values.append(flags_list[j])
                        j += 1
                    flag_pairs.append((item, values))
                    i = j
                else:
                    # Single value
                    flag_pairs.append((item, [flags_list[i + 1]]))
                    i += 2
            else:
                flag_pairs.append((item, []))
                i += 1
        flag_pairs.sort(key=lambda x: x[0])
        result = []
        for flag, values in flag_pairs:
            result.append(flag)
            result.extend(values)
        return result

    sorted_flags = sort_flags(flags)
    formatted_cmd = base_cmd + sorted_flags

    # Print each flag on new line (flag and value(s) together)
    add_info_message(training_console_text, f"{formatted_cmd[0]} {formatted_cmd[1]} \\\n")
    i = 2
    while i < len(formatted_cmd):
        # Check if this is a flag with values
        if formatted_cmd[i].startswith('--') and i + 1 < len(formatted_cmd) and not formatted_cmd[i + 1].startswith('--'):
            # Collect all values for this flag
            flag_line = f"    {formatted_cmd[i]}"
            i += 1
            while i < len(formatted_cmd) and not formatted_cmd[i].startswith('--'):
                flag_line += f" {formatted_cmd[i]}"
                i += 1
            # Check if this is the last line
            if i >= len(formatted_cmd):
                add_info_message(training_console_text, f"{flag_line}\n")
            else:
                add_info_message(training_console_text, f"{flag_line} \\\n")
        else:
            # Standalone flag or value
            if i == len(formatted_cmd) - 1:
                add_info_message(training_console_text, f"    {formatted_cmd[i]}\n")
            else:
                add_info_message(training_console_text, f"    {formatted_cmd[i]} \\\n")
            i += 1
    if training_console_text.page:
        training_console_text.update()

    # Execute with proper process handling
    import subprocess
    try:
        # Create process group for clean termination
        creation_flags = {}
        if os.name == 'posix':
            creation_flags['preexec_fn'] = os.setsid
        else:
            creation_flags['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=runner.project_root,
            **creation_flags
        )

        # Store process reference
        main_container.training_proc = proc
        training_tab_container.training_proc = proc

        # Start output streaming
        try:
            from flet_app.ui.training.output_manager import start_ltx_output_streamer
            start_ltx_output_streamer(proc, training_console_text, main_container=main_container, page=page)
        except ImportError:
            logger.error("Could not import output streamer")

        set_button_state(main_container, "Stop", page)

        # Debug: verify button state was set
        start_btn = getattr(main_container, 'start_btn', None)
        logger.info(f"Set button to Stop. Button text: {start_btn.text if start_btn else 'None'}")

        return
    except Exception as e:
        add_error_message(training_console_text, f"\n[Error] Failed to start training: {e}\n")
        logger.error(f"Failed to start training: {e}")
        reset_to_start_button(main_container, training_tab_container, page)
