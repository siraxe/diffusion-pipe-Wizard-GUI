"""
Training UI Components

Contains modular components for the training tab, including:
- output_manager: Console rendering, streaming, status messages
- start_button_handler: Start/Cancel button logic, process management

This module separates concerns from the main tab_training_view.py file.
"""

from .output_manager import (
    cleanup_training_console,
    format_output_line,
    add_status_message,
    add_info_message,
    add_success_message,
    add_warning_message,
    add_error_message,
    add_action_message,
    start_output_streamer,
    start_ltx_output_streamer,
    start_buffered_output_streamer,
    scroll_to_bottom_if_needed,
    force_scroll_to_bottom,
    StatusColors,
    TensorBoardLogger,
)

from .start_button_handler import (
    set_button_state,
    reset_to_start_button,
    terminate_process,
    handle_cancel_click,
    run_ltx2_training_flow,
)

__all__ = [
    'cleanup_training_console',
    'format_output_line',
    'add_status_message',
    'add_info_message',
    'add_success_message',
    'add_warning_message',
    'add_error_message',
    'add_action_message',
    'start_output_streamer',
    'start_ltx_output_streamer',
    'start_buffered_output_streamer',
    'scroll_to_bottom_if_needed',
    'force_scroll_to_bottom',
    'StatusColors',
    'TensorBoardLogger',
    'set_button_state',
    'reset_to_start_button',
    'terminate_process',
    'handle_cancel_click',
    'run_ltx2_training_flow',
]
