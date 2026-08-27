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
# Cache Execution Constants & Helpers
# =====================

# Priority order for cache types
CACHE_PRIORITY = ['i2v_preprocess', 'latents', 'vace_latents', 'latents_negative', 'text_encoder', 'sample_prompts']

# Human-readable names for cache types
CACHE_DISPLAY_NAMES = {
    'i2v_preprocess': 'I2V Preprocess',
    'latents': 'Latents',
    'vace_latents': 'VACE Latents',
    'latents_negative': 'Control Images (Negative)',
    'text_encoder': 'Text Encoder',
    'sample_prompts': 'Sample Prompts'
}


def get_cache_display_name(cache_type: str, context: str = 'running') -> str:
    if cache_type == 'latents_negative':
        return 'Control images' if context != 'running' else 'Control images (negative)'
    return CACHE_DISPLAY_NAMES.get(cache_type, cache_type.replace('_', ' ').title())


def run_on_page_if_available(page, coro):
    if page:
        page.run_task(coro)


# =====================
# Cache Execution Helper Functions
# =====================

def _get_command_string(cmd_data) -> str:
    if cmd_data and isinstance(cmd_data[0], list):
        cmd_list = cmd_data[0]
    else:
        cmd_list = cmd_data
    return " ".join(cmd_list)


def _show_cache_summary(runner, training_console_text, page, cache_order: list):
    from flet_app.ui.training.output_manager import add_warning_message

    display_names = [CACHE_DISPLAY_NAMES.get(ct, ct.replace('_', ' ').title()) for ct in cache_order]

    # Check t_type dropdown value (replaces ic_lora/vace_lora checkboxes)
    config = runner.get_config()
    training_strategy = config.get('training_strategy', {})
    t_type = training_strategy.get('t_type', 'none') or 'none'
    ic_lora_enabled = (t_type == 'ic_lora')
    vace_enabled = (t_type == 'vace_lora')

    mode_suffixes = []
    if ic_lora_enabled:
        mode_suffixes.append('IC-LoRA')
    if vace_enabled:
        # Determine if VACE is using LoRA adapters or full training
        vace_config = training_strategy.get('vace', {})
        vace_lora_explicit = vace_config.get('lora', None)
        if vace_lora_explicit is not None:
            vace_is_lora = str(vace_lora_explicit).lower() in ('true', '1', 'yes')
        else:
            vace_is_lora = False  # Default to full VACE training
        mode_suffixes.append('VACE-LoRA' if vace_is_lora else 'VACE')

    summary_line = f"\n[Cache Summary] Will run: {', '.join(display_names)}"
    if mode_suffixes:
        summary_line += f" ({' + '.join(mode_suffixes)})"
    summary_line += "\n"

    async def show(_txt=training_console_text):
        add_warning_message(_txt, summary_line)
    run_on_page_if_available(page, show)


def _show_cache_starting(cache_type: str, cmd_str: str, training_console_text, page):
    display_name = get_cache_display_name(cache_type, 'running')

    async def show(_txt=training_console_text):
        add_info_message(_txt, f"\n[Info] Running {display_name} caching...\n")
        add_info_message(_txt, f"\n[Command] {cmd_str}\n")
    run_on_page_if_available(page, show)


def _handle_cache_result(cache_type: str, proc, main_container,
                        training_console_text, page) -> bool:
    from flet_app.ui.training.output_manager import add_action_message

    # Check if process reference was cleared (indicates user cancellation)
    if proc.returncode != 0 and getattr(main_container, 'training_proc', None) is None:
        display_name = get_cache_display_name(cache_type, 'cancelled')

        async def show(_txt=training_console_text):
            add_action_message(_txt,
                            f"\n[Action] {display_name} caching cancelled.\n")
        run_on_page_if_available(page, show)
        return True
    return False


def _show_cache_failed(cache_type: str, exit_code: int, training_console_text, page):
    from flet_app.ui.training.output_manager import add_error_message
    display_name = get_cache_display_name(cache_type, 'completed')

    async def show(_txt=training_console_text):
        add_error_message(_txt,
                        f"\n[Error] {display_name} caching failed with exit code {exit_code}\n")
    run_on_page_if_available(page, show)


def _show_cache_completed(cache_type: str, training_console_text, page):
    from flet_app.ui.training.output_manager import add_success_message
    display_name = get_cache_display_name(cache_type, 'completed')

    async def show(_txt=training_console_text):
        add_success_message(_txt,
                          f"\n[Success] {display_name} caching completed.\n")
    run_on_page_if_available(page, show)


def _show_final_status(was_cancelled: bool, all_success: bool, model_type: str,
                      main_container, training_tab_container, page,
                      training_console_text,
                      reset_button_on_complete: bool):
    from flet_app.ui.training.output_manager import add_warning_message, add_success_message

    async def show(_txt=training_console_text):
        if was_cancelled:
            add_warning_message(_txt,
                             f"\n[Info] Caching was cancelled. Partial caches may exist.\n")
        elif all_success:
            add_success_message(_txt,
                             f"\n[Success] All caching completed for {model_type}.\n")
        else:
            add_warning_message(_txt,
                             f"\n[Warning] Caching completed with errors for {model_type}.\n")

        if reset_button_on_complete:
            reset_to_start_button(main_container, training_tab_container, page)
    run_on_page_if_available(page, show)


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
    reset_button_on_complete: bool = True,
):
    import threading

    def run_cache_thread():
        try:
            # Get available cache commands
            cache_cmds_dict = runner.get_cache_commands(dataset_config, slider_config)

            # Determine which caches to run (respecting priority order)
            if cache_types:
                cache_order = [ct for ct in CACHE_PRIORITY if ct in cache_cmds_dict and ct in cache_types]
            else:
                cache_order = [ct for ct in CACHE_PRIORITY if ct in cache_cmds_dict]

            # Track overall success/failure state
            all_success = True
            was_cancelled = False

            # Show cache summary before starting
            _show_cache_summary(runner, training_console_text, page, cache_order)

            # Process each cache type in order
            for cache_type in cache_order:
                # Get command string for display
                cmd_str = _get_command_string(cache_cmds_dict[cache_type])

                # Show what's starting
                _show_cache_starting(cache_type, cmd_str, training_console_text, page)

                # Run the cache command
                proc = runner.run_cache_async(dataset_config, cache_type=cache_type, slider_config=slider_config)

                # Store process reference for cancellation
                main_container.training_proc = proc
                training_tab_container.training_proc = proc

                # Update button to "Stop"
                async def update_btn(_mc=main_container, _pg=page):
                    set_button_state(_mc, "Stop", _pg)
                run_on_page_if_available(page, update_btn)

                # Stream output and wait for completion
                from flet_app.ui.training.output_manager import start_ltx_output_streamer
                start_ltx_output_streamer(proc, training_console_text, main_container=main_container, page=page)
                proc.wait()

                # Handle result (success/failure/cancellation)
                was_cancelled = _handle_cache_result(
                    cache_type, proc, main_container,
                    training_console_text, page
                )
                if was_cancelled:
                    all_success = False
                    continue

                if proc.returncode != 0:
                    _show_cache_failed(cache_type, proc.returncode, training_console_text, page)
                    all_success = False
                    continue

                # Success case
                _show_cache_completed(cache_type, training_console_text, page)

            # Show final status after all caches processed
            _show_final_status(was_cancelled, all_success, runner.model_type,
                            main_container, training_tab_container, page,
                            training_console_text,
                            reset_button_on_complete)


        except Exception as e:
            import traceback
            error_detail = f"{e}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Failed to run cache commands: {error_detail}")

            async def show_err(err=e, _txt=training_console_text, _mc=main_container, _ttc=training_tab_container, _pg=page):
                from flet_app.ui.training.output_manager import add_error_message
                add_error_message(_txt,
                               f"\n[Error] Failed to run cache commands: {err}\n")
                if reset_button_on_complete:
                    reset_to_start_button(_mc, _ttc, _pg)
            run_on_page_if_available(page, show_err)

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

    # Check if txt_slider config was created (H3 mode = txt_slider)
    txt_slider_config_path = musubi_result.get('txt_slider_config_path')
    if txt_slider_config_path:
        add_info_message(training_console_text, f"\n[Info] Txt slider config created at: {txt_slider_config_path}\n")
        add_info_message(training_console_text, "\nslider training started\n")
        print("slider training started", flush=True)

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
        cfg = runner.get_config()
        model_cfg = cfg.get('model', {})
        # output_dir / output_name are written at top level by toml_builder;
        # fall back to [model] for older configs.
        output_dir = cfg.get('output_dir') or model_cfg.get('output_dir') or 'output/ltx2_lora'
        output_name = cfg.get('output_name') or model_cfg.get('output_name') or model_cfg.get('name') or ''
        resume_path = find_last_state_directory(output_dir, output_name)
        if resume_path:
            add_info_message(training_console_text, f"\n[Resume] Found state: {resume_path}\n")
        else:
            add_warning_message(training_console_text, f"\n[Resume] No state found\n")

    # H3 txt slider training needs no dataset caching — prompts/latents
    # come from the slider TOML itself.
    if txt_slider_config_path:
        add_info_message(training_console_text, "\n[Info] Txt slider mode: Skipping cache commands...\n")
        if training_console_text.page:
            training_console_text.update()
    else:
        match mode:
            case 'trust_cache':
                add_info_message(training_console_text, f"\n[Info] Trust-cache mode: Skipping latents/text_encoder caching\n")
                add_info_message(training_console_text, f"\n[Info] Running sample prompts caching...\n")
                if training_console_text.page:
                    training_console_text.update()
                await run_cache_commands(runner, dataset_config, main_container, training_tab_container, page, training_console_text, slider_config_path, cache_types=['sample_prompts'], reset_button_on_complete=False)

            case 'cache_only':
                add_info_message(training_console_text, f"\n[Info] Cache-only mode: Running all cache commands\n")
                if training_console_text.page:
                    training_console_text.update()
                await run_cache_commands(runner, dataset_config, main_container, training_tab_container, page, training_console_text, slider_config_path, reset_button_on_complete=True)
                return  # Stop after caching

            case 'full':
                add_info_message(training_console_text, f"\n[Info] Running cache commands before training...\n")
                if training_console_text.page:
                    training_console_text.update()
                await run_cache_commands(runner, dataset_config, main_container, training_tab_container, page, training_console_text, slider_config_path, reset_button_on_complete=False)

    # Detect VACE mode: check t_type dropdown value
    # ltx2_run.py will auto-detect and use the correct training script/flags
    vace_dataset_config = None
    config = runner.get_config()
    training_strategy = config.get('training_strategy', {})
    t_type = training_strategy.get('t_type', 'none') or 'none'
    vace_enabled = (t_type == 'vace_lora')
    if vace_enabled and dataset_config:
        vace_config_path = dataset_config.replace('.toml', '_vace.toml')
        if os.path.exists(vace_config_path):
            vace_dataset_config = vace_config_path
            add_info_message(training_console_text, f"\n[Info] VACE mode detected - using {os.path.basename(vace_config_path)}\n")

    # Build training command (use VACE config if detected; txt slider
    # config takes precedence over the regular slider config)
    cmd = runner.get_training_command(
        vace_dataset_config or dataset_config,
        txt_slider_config_path or slider_config_path, resume_path, reset_optimizer, reset_optimizer_params
    )

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
