"""
Training Output Manager

Handles console output, streaming, status messages, and cleanup for the training view.
"""

import flet as ft
import re
from threading import Thread
import asyncio
import time
from loguru import logger
from pathlib import Path
from threading import Lock
import os


# =====================
# TensorBoard Logger
# =====================

class TensorBoardLogger:
    """
    Parses training output and logs metrics to TensorBoard format.
    Creates event files in output_dir/.tensorboard/ that can be viewed with tensorboard.
    """

    # Pattern to match training step logs like:
    # INFO     Step 161/2000 - Loss: 0.7078, LR: 1.00e-04, Time/Step: 0.94s, Total Time: 1h 24m
    STEP_PATTERN = re.compile(
        r'Step\s+(\d+)/\d+\s+-\s+Loss:\s+([\d.]+),\s+LR:\s+([\d.e-]+)'
    )

    def __init__(self, output_dir: str | Path):
        """
        Initialize TensorBoard logger.

        Args:
            output_dir: Directory where checkpoints/.tensorboard folder will be created
        """
        self._output_dir = Path(output_dir)
        self._tb_dir = self._output_dir / "checkpoints" / ".tensorboard"
        self._tb_dir.mkdir(parents=True, exist_ok=True)

        # Try to import tensorboard logger
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self._tb_dir))
            logger.info(f"TensorBoard logging enabled: {self._tb_dir}")
        except ImportError:
            logger.warning("tensorboard not installed - TensorBoard logging disabled")

        self._lock = Lock()
        self._closed = False

    def parse_and_log(self, line: str) -> None:
        """
        Parse a line of training output and log metrics to TensorBoard if found.

        Args:
            line: Raw output line from training
        """
        if self._writer is None or self._closed:
            return

        match = self.STEP_PATTERN.search(line)
        if match:
            try:
                step = int(match.group(1))
                loss = float(match.group(2))
                lr = float(match.group(3))

                with self._lock:
                    if not self._closed:
                        self._writer.add_scalar('train/loss', loss, step)
                        self._writer.add_scalar('train/learning_rate', lr, step)
                        self._writer.flush()
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse metrics from line: {e}")

    def close(self) -> None:
        """Close the TensorBoard writer."""
        with self._lock:
            if not self._closed and self._writer is not None:
                self._writer.close()
                self._closed = True
                logger.debug("TensorBoard logger closed")


# =====================
# Configuration
# =====================

MAX_CONSOLE_LINES = 450  # Maximum number of lines before cleanup
CLEANUP_LINES = 400  # Keep this many lines after cleanup (remove oldest lines)
AUTO_SCROLL_INTERVAL = 0.5  # Seconds between auto-scrolls (throttled)


# =====================
# Scroll State
# =====================

_last_scroll_time = 0
_scroll_lock = Thread.Lock() if hasattr(Thread, 'Lock') else None


# =====================
# Console Cleanup
# =====================

def cleanup_training_console(training_console_text):
    """
    Removes the oldest portion of training console text when it exceeds size limits.
    Assumes newest entries are at the bottom (appended), so it trims from the top.

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

        # Only trigger cleanup if we exceed the maximum line count
        if current_line_count > MAX_CONSOLE_LINES:
            # Find the cutoff point scanning from the bottom (newest-first)
            # Keep CLEANUP_LINES most recent lines
            line_count = 0
            cutoff_index = 0  # default to keep all

            for i in range(len(current_spans) - 1, -1, -1):
                span = current_spans[i]
                if hasattr(span, 'text'):
                    line_count += span.text.count('\n')
                    if line_count >= CLEANUP_LINES:
                        cutoff_index = i
                        break

            # Keep only the newest portion at the bottom; drop top (older) lines
            new_spans = current_spans[cutoff_index:]

            # Add a cleanup notification at the top
            cleanup_notice = ft.TextSpan(
                f"\n[System] Console cleaned: removed {current_line_count - CLEANUP_LINES} old lines (kept {CLEANUP_LINES})\n",
                ft.TextStyle(color=ft.Colors.with_opacity(0.6, ft.Colors.GREY))
            )
            new_spans.insert(0, cleanup_notice)

            # Update the console
            training_console_text.spans = new_spans
            logger.debug(f"[Cleanup] Removed {current_line_count - CLEANUP_LINES} old lines, kept {CLEANUP_LINES} lines")
            return True

        return False

    except Exception as e:
        logger.error(f"Error during console cleanup: {e}")
        return False


# =====================
# Spans Formatting
# =====================

def format_output_line(text: str):
    """
    Format a line of output as styled spans for the console.

    Args:
        text: Raw text line from process output

    Returns:
        list[ft.TextSpan]: List of styled text spans
    """
    spans_line = []
    try:
        s = text.rstrip("\n")
        parts = re.split(r"(\[[^\]]*\])", s)
        bracket_count = 0
        for part in parts:
            if part is None or part == "":
                continue
            if part.startswith("[") and part.endswith("]"):
                bracket_count += 1
                if bracket_count <= 2:
                    color = ft.Colors.with_opacity(0.3, ft.Colors.WHITE)
                else:
                    color = ft.Colors.WHITE
                spans_line.append(ft.TextSpan(part, style=ft.TextStyle(color=color)))
            else:
                spans_line.append(ft.TextSpan(part, style=ft.TextStyle(color=ft.Colors.WHITE)))
        spans_line.append(ft.TextSpan("\n"))
    except Exception:
        spans_line.append(ft.TextSpan(text, style=ft.TextStyle(color=ft.Colors.WHITE)))
    return spans_line


# =====================
# Status Messages
# =====================

class StatusColors:
    """Color constants for status messages."""
    INFO = ft.Colors.with_opacity(0.7, ft.Colors.CYAN)
    SUCCESS = ft.Colors.with_opacity(0.7, ft.Colors.GREEN)
    WARNING = ft.Colors.with_opacity(0.7, ft.Colors.YELLOW)
    ERROR = ft.Colors.with_opacity(0.7, ft.Colors.RED)
    WHITE = ft.Colors.WHITE


def add_status_message(training_console_text, message: str, color=None):
    """
    Add a status message to the training console.

    Args:
        training_console_text: The flet Text control containing console spans
        message: Status message to add
        color: Optional color override (defaults to StatusColors.INFO)
    """
    try:
        if training_console_text is None:
            return
        spans = list((training_console_text.spans or []))
        if color is None:
            color = StatusColors.INFO
        spans.append(ft.TextSpan(message, style=ft.TextStyle(color=color)))
        training_console_text.spans = spans
    except Exception as e:
        logger.error(f"Error adding status message: {e}")


def add_info_message(training_console_text, message: str):
    """Add an info status message."""
    add_status_message(training_console_text, message, StatusColors.INFO)


def add_success_message(training_console_text, message: str):
    """Add a success status message."""
    add_status_message(training_console_text, message, StatusColors.SUCCESS)


def add_warning_message(training_console_text, message: str):
    """Add a warning status message."""
    add_status_message(training_console_text, message, StatusColors.WARNING)


def add_error_message(training_console_text, message: str):
    """Add an error status message."""
    add_status_message(training_console_text, message, StatusColors.ERROR)


def add_action_message(training_console_text, message: str):
    """Add an action message (white text)."""
    add_status_message(training_console_text, message, StatusColors.WHITE)


# =====================
# Auto-Scroll (Throttled)
# =====================

def scroll_to_bottom_if_needed(training_console_text):
    """
    Scroll the console to bottom, but only if enough time has passed since last scroll.
    This prevents excessive scrolling during rapid output.

    Args:
        training_console_text: The flet Text control for console output

    Returns:
        bool: True if scroll was performed, False otherwise
    """
    global _last_scroll_time

    try:
        if training_console_text is None:
            return False

        current_time = time.time()

        # Check if enough time has passed since last scroll
        if current_time - _last_scroll_time >= AUTO_SCROLL_INTERVAL:
            # Update last scroll time
            if _scroll_lock is not None:
                with _scroll_lock:
                    _last_scroll_time = current_time
            else:
                _last_scroll_time = current_time

            # Find the parent scrollable container and scroll it
            # In Flet, scroll is typically on a parent Column with scroll=True
            parent = None
            if hasattr(training_console_text, 'parent'):
                parent = training_console_text.parent

            # Try to scroll the parent if it's a scrollable Column
            if parent is not None and hasattr(parent, 'scroll_to'):
                try:
                    parent.scroll_to(offset=-1, duration=50)
                    return True
                except Exception:
                    pass

            # Fallback: try page-level scroll
            if hasattr(training_console_text, 'page') and training_console_text.page is not None:
                try:
                    # Try to find a scrollable control in the page
                    page = training_console_text.page
                    if hasattr(page, 'controls'):
                        def find_scrollable(control):
                            if hasattr(control, 'scroll_to'):
                                return control
                            if hasattr(control, 'controls'):
                                for c in control.controls:
                                    result = find_scrollable(c)
                                    if result:
                                        return result
                            return None

                        scrollable = find_scrollable(page)
                        if scrollable is not None:
                            scrollable.scroll_to(offset=-1, duration=50)
                            return True
                except Exception:
                    pass

        return False

    except Exception as e:
        logger.error(f"Error scrolling to bottom: {e}")
        return False


def force_scroll_to_bottom(training_console_text):
    """
    Force scroll to bottom immediately, bypassing the throttle.
    Use this for important messages that should always be visible.

    Args:
        training_console_text: The flet Text control for console output
    """
    global _last_scroll_time

    try:
        if training_console_text is None:
            return

        # Update last scroll time
        if _scroll_lock is not None:
            with _scroll_lock:
                _last_scroll_time = time.time()
        else:
            _last_scroll_time = time.time()

        # Find the parent scrollable container and scroll it
        parent = None
        if hasattr(training_console_text, 'parent'):
            parent = training_console_text.parent

        if parent is not None and hasattr(parent, 'scroll_to'):
            try:
                parent.scroll_to(offset=-1, duration=50)
                return
            except Exception:
                pass

        # Fallback: try page-level scroll
        if hasattr(training_console_text, 'page') and training_console_text.page is not None:
            try:
                page = training_console_text.page
                if hasattr(page, 'controls'):
                    def find_scrollable(control):
                        if hasattr(control, 'scroll_to'):
                            return control
                        if hasattr(control, 'controls'):
                            for c in control.controls:
                                result = find_scrollable(c)
                                if result:
                                    return result
                        return None

                    scrollable = find_scrollable(page)
                    if scrollable is not None:
                        scrollable.scroll_to(offset=-1, duration=50)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Error force scrolling to bottom: {e}")


# =====================
# Output Streaming
# =====================

def start_output_streamer(proc, training_console_text, training_tab_container, main_container):
    """
    Start a background thread to stream process output to the console.
    Handles normal append order (newest at bottom).

    Args:
        proc: subprocess.Popen object with stdout
        training_console_text: The flet Text control for console output
        training_tab_container: Container for training state
        main_container: Main container for button state
    """
    def _reader():
        try:
            if proc.stdout is None:
                return

            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                spans = format_output_line(line)
                if training_console_text is not None:
                    training_console_text.spans = list((training_console_text.spans or [])) + spans
                    training_console_text.update()
                    # Throttled scroll to bottom (every ~3 seconds)
                    scroll_to_bottom_if_needed(training_console_text)
        except Exception as e:
            logger.error(f"Error in output streamer: {e}")
        finally:
            # Process completion handling
            try:
                rc = proc.wait(timeout=2)
            except Exception:
                rc = -1
            try:
                spans = list((training_console_text.spans or [])) if training_console_text is not None else []
                spans.append(ft.TextSpan(f"\n[Done] Exit code: {rc}\n", style=ft.TextStyle(color=ft.Colors.WHITE)))
                if training_console_text is not None:
                    training_console_text.spans = spans
            except Exception:
                pass
            # Reset Start button and clear proc handle
            try:
                if hasattr(main_container, 'start_btn') and main_container.start_btn is not None:
                    main_container.start_btn.text = "Start"
                training_tab_container.training_proc = None
                main_container.training_proc = None
            except Exception:
                pass
            try:
                if training_console_text.page is not None:
                    training_console_text.page.update()
            except Exception:
                pass

    reader_thread = Thread(target=_reader, daemon=True)
    reader_thread.start()
    return reader_thread


def start_ltx_output_streamer(proc, training_console_text, tb_logger: TensorBoardLogger = None):
    """
    Start a background thread to stream LTX training output to the console.
    Simpler version that doesn't handle button state.

    Args:
        proc: subprocess.Popen object with stdout
        training_console_text: The flet Text control for console output
        tb_logger: Optional TensorBoardLogger for parsing metrics
    """
    def _reader():
        line_count = 0
        try:
            if proc.stdout is None:
                return

            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                # Parse and log to TensorBoard if logger provided
                if tb_logger is not None:
                    tb_logger.parse_and_log(line)
                spans = format_output_line(line)
                if training_console_text is not None and training_console_text.page is not None:
                    training_console_text.spans = list((training_console_text.spans or [])) + spans
                    training_console_text.update()

                    # Periodic cleanup (every 50 lines) and scroll
                    line_count += 1
                    if line_count % 50 == 0:
                        try:
                            cleanup_training_console(training_console_text)
                        except Exception:
                            pass

                    # Throttled scroll to bottom (every 0.5 seconds)
                    scroll_to_bottom_if_needed(training_console_text)
        except Exception as e:
            logger.error(f"Error in LTX output streamer: {e}")
        finally:
            # Close TensorBoard logger when process ends
            if tb_logger is not None:
                tb_logger.close()

    reader_thread = Thread(target=_reader, daemon=True)
    reader_thread.start()
    return reader_thread


# =====================
# Buffered Streaming (for batched output)
# =====================

def start_buffered_output_streamer(proc, training_console_text, monitor_content, page):
    """
    Start an async buffered output streamer that batches output for better performance.
    Uses normal append order (newest at bottom).

    Args:
        proc: subprocess.Popen object with stdout
        training_console_text: The flet Text control for console output
        monitor_content: Monitor page content containing animation wrapper
        page: Flet page for run_task
    """
    import time as _t

    _buffer = []  # pending spans batches (each element is a list[TextSpan])
    _flush_running = {"v": False}

    async def _flush_buffer_periodically():
        _flush_running["v"] = True
        try:
            anim_wrap = getattr(monitor_content, 'training_console_anim', None)
            while True:
                try:
                    if not _buffer:
                        # If process likely ended and buffer empty, stop
                        if proc.poll() is not None:
                            break
                        await asyncio.sleep(0.08)
                        continue
                    # Drain buffer into a single batch
                    drained = []
                    while _buffer:
                        drained.extend(_buffer.pop(0))
                    # Apply once: append drained to existing spans (normal order)
                    spans_current = list((training_console_text.spans or [])) if training_console_text is not None else []
                    new_total = spans_current + drained
                    if training_console_text is not None:
                        training_console_text.spans = new_total

                    # One smooth, scaled animation per batch
                    try:
                        if anim_wrap is not None:
                            lines_added = 1
                            try:
                                lines_added = max(1, sum(s.text.count('\n') for s in drained if hasattr(s, 'text')))
                            except Exception:
                                pass
                            top_pad = min(72, max(12, lines_added * 12))
                            anim_wrap.padding = ft.padding.only(top=top_pad)
                            anim_wrap.update()
                            # small delay to allow implicit animate to kick in
                            await asyncio.sleep(0.016)
                            anim_wrap.padding = ft.padding.only(top=0)
                            anim_wrap.update()
                    except Exception:
                        pass

                    # Cleanup every batch to keep memory stable
                    try:
                        cleanup_performed = cleanup_training_console(training_console_text)
                        if cleanup_performed:
                            logger.debug("[AutoCleanup] Console cleaned post-batch")
                    except Exception:
                        pass

                    try:
                        training_console_text.update()
                    except Exception:
                        pass

                    # Throttled scroll to bottom (every ~3 seconds)
                    scroll_to_bottom_if_needed(training_console_text)

                    # Frame spacing between batches
                    await asyncio.sleep(0.1)
                except Exception:
                    await asyncio.sleep(0.1)
                    continue
        finally:
            _flush_running["v"] = False

    def _read_to_buffer():
        for line in proc.stdout:
            try:
                # Pre-split line into styled spans
                new_line_spans = format_output_line(line) or []
                # Enqueue to buffer for batched append
                _buffer.append(new_line_spans)
                # Start flusher if not running
                if page is not None and not _flush_running["v"]:
                    # Pass coroutine function to run_task (do not call it here)
                    page.run_task(_flush_buffer_periodically)
            except Exception as e:
                logger.error(f"Error buffering output: {e}")

    reader_thread = Thread(target=_read_to_buffer, daemon=True)
    reader_thread.start()
    return reader_thread
