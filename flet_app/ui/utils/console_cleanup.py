"""
Training Console Cleanup Utilities

Shared utilities for managing training console text cleanup.
This prevents circular imports between training modules.
"""

import flet as ft

# Training console cleanup configuration
MAX_CONSOLE_LINES = 200  # Maximum number of lines before cleanup
CLEANUP_RATIO = 0.5  # Remove this fraction of lines when cleaning up (remove 50% of oldest)

def cleanup_training_console(training_console_text):
    """
    Removes the oldest portion of training console text when it exceeds size limits.
    Assumes newest entries are at the top of the list (prepended), so it
    trims from the bottom when over threshold.

    Args:
        training_console_text: The flet Text control containing console spans

    Returns:
        bool: True if cleanup was performed, False otherwise
    """
    try:
        if training_console_text is None or training_console_text.spans is None:
            return False

        current_spans = list(training_console_text.spans)
        current_line_count = sum(span.text.count('\n') for span in current_spans if hasattr(span, 'text'))

        # Debug: Log line count periodically
        if hasattr(cleanup_training_console, 'call_count'):
            cleanup_training_console.call_count += 1
        else:
            cleanup_training_console.call_count = 1

        if cleanup_training_console.call_count % 10 == 0:
            print(f"[Cleanup Debug] Current line count: {current_line_count}, threshold: {MAX_CONSOLE_LINES}")

        # Only trigger cleanup if we exceed the maximum line count
        if current_line_count > MAX_CONSOLE_LINES:
            # Calculate how many lines to keep (keep newest portion at the top)
            lines_to_keep = int(current_line_count * (1 - CLEANUP_RATIO))

            # Find the cutoff point scanning from the top (newest-first)
            line_count = 0
            cutoff_index = len(current_spans) - 1  # default to keep all

            for i, span in enumerate(current_spans):
                if hasattr(span, 'text'):
                    line_count += span.text.count('\n')
                    if line_count >= lines_to_keep:
                        cutoff_index = i
                        break

            # Keep only the newest portion at the top; drop bottom (older) lines
            new_spans = current_spans[: cutoff_index + 1]

            # Add a cleanup notification at the top
            cleanup_notice = ft.TextSpan(
                f"\n[System] Console cleaned: removed {current_line_count - lines_to_keep} old lines\n",
                ft.TextStyle(color=ft.Colors.with_opacity(0.6, ft.Colors.GREY))
            )
            new_spans.insert(0, cleanup_notice)

            # Update the console
            training_console_text.spans = new_spans
            print(f"[Cleanup] Removed {current_line_count - lines_to_keep} old lines, kept {lines_to_keep} lines")
            return True

        return False

    except Exception as e:
        print(f"Error during console cleanup: {e}")
        return False
