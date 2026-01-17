"""
Training Output Manager

Handles console output, streaming, status messages, and cleanup for the training view.
Refactored for performance (buffered streaming) and thread safety.
"""

import flet as ft
import re
import time
import asyncio
from threading import Thread, Lock
from pathlib import Path
from loguru import logger
import os

# =====================
# Configuration
# =====================

MAX_CONSOLE_LINES = 450
CLEANUP_LINES = 400
AUTO_SCROLL_INTERVAL = 0.5  # Seconds between auto-scrolls
# Pre-compile regex for performance
ANSI_BRACKET_PATTERN = re.compile(r"(\[[^\]]*\])")


# =====================
# TensorBoard Logger
# =====================

class TensorBoardLogger:
    """
    Parses training output and logs metrics to TensorBoard format.
    Creates event files in output_dir/.tensorboard/ that can be viewed with tensorboard.
    """

    # Updated pattern: Case insensitive, handles various spacing
    STEP_PATTERN = re.compile(
        r'Step\s+(\d+)/\d+\s+-\s+Loss:\s+([\d.]+),\s+LR:\s+([\d.e-]+)',
        re.IGNORECASE
    )

    def __init__(self, output_dir: str | Path):
        self._output_dir = Path(output_dir)
        checkpoints_dir = self._output_dir / "checkpoints"
        tb_dir = checkpoints_dir / ".tensorboard"

        tb_dir.mkdir(parents=True, exist_ok=True)

        # Versioning logic
        version = 1
        existing_dirs = []
        for item in tb_dir.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                try:
                    v_str = item.name[1:]
                    existing_dirs.append(int(v_str))
                except (ValueError, IndexError):
                    pass

        if existing_dirs:
            version = max(existing_dirs) + 1

        self._tb_dir = tb_dir / f"v{version}"
        self._tb_dir.mkdir(parents=True, exist_ok=False)

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
                        # Flush occasionally, not every step, to save IO (handled by TB usually, but explicit is okay)
                        # self._writer.flush() 
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse metrics: {e}")

    def close(self) -> None:
        with self._lock:
            if not self._closed and self._writer is not None:
                self._writer.close()
                self._closed = True
                logger.debug("TensorBoard logger closed")


# =====================
# Console Controller
# =====================

class ConsoleController:
    """
    Manages the UI updates for the console, including appending text, 
    scrolling, and cleaning up old lines.
    """
    def __init__(self, text_control: ft.Text, scroll_control=None, anim_control=None):
        """
        Args:
            text_control: The Flet Text control holding the spans.
            scroll_control: The scrollable container (Column/ListView). If None, tries to find parent.
            anim_control: Optional container wrapper for animation effects.
        """
        self.text_control = text_control
        self.scroll_control = scroll_control
        self.anim_control = anim_control
        
        # State
        self._last_scroll_time = 0
        self._scroll_lock = Lock()

    def append_spans(self, new_spans: list[ft.TextSpan], trigger_anim: bool = False):
        """Appends spans to the text control safely."""
        if self.text_control is None:
            return

        try:
            # 1. Get current spans
            current_spans = list(self.text_control.spans or [])

            # 2. Append new
            updated_spans = current_spans + new_spans

            # 3. Apply to control
            self.text_control.spans = updated_spans

            # 4. Handle Animation (if provided)
            if trigger_anim and self.anim_control:
                self._trigger_animation(new_spans)

            # 5. Cleanup if too large (Optimization: don't check every single frame)
            if len(updated_spans) > MAX_CONSOLE_LINES * 1.5:  # heuristic check
                self.cleanup()

            # Only update if the control has been added to the page
            if self.text_control.page is not None:
                self.text_control.update()

            # 6. Scroll
            self.scroll_to_bottom()

        except Exception as e:
            logger.error(f"Error updating console UI: {e}")

    def _trigger_animation(self, added_spans):
        """Applies a subtle top-padding bounce effect."""
        try:
            lines_added = max(1, sum(s.text.count('\n') for s in added_spans if hasattr(s, 'text')))
            top_pad = min(72, max(12, lines_added * 12))

            self.anim_control.padding = ft.padding.only(top=top_pad)
            if self.anim_control.page is not None:
                self.anim_control.update()
            # The async sleep to reset this must be handled by the caller or an asyncio task
            # For simplicity in this synchronous method, we set it, but resetting needs a task.
            # In the LogStreamer below, we handle the timing.
        except Exception:
            pass

    def cleanup(self):
        """Removes old lines from the top."""
        if not self.text_control or not self.text_control.spans:
            return

        spans = self.text_control.spans
        total_lines = sum(span.text.count('\n') for span in spans if hasattr(span, 'text'))

        if total_lines > MAX_CONSOLE_LINES:
            lines_to_keep = CLEANUP_LINES
            current_lines = 0
            cutoff_index = 0

            # Scan backwards
            for i in range(len(spans) - 1, -1, -1):
                span = spans[i]
                if hasattr(span, 'text'):
                    current_lines += span.text.count('\n')
                    if current_lines >= lines_to_keep:
                        cutoff_index = i
                        break
            
            # Slice
            new_spans = spans[cutoff_index:]
            
            # Add notice
            removed_count = total_lines - current_lines
            notice = ft.TextSpan(
                f"\n[System] Console cleaned: removed {removed_count} old lines.\n",
                ft.TextStyle(color=ft.Colors.with_opacity(0.6, ft.Colors.GREY))
            )
            new_spans.insert(0, notice)
            
            self.text_control.spans = new_spans
            logger.debug(f"[Cleanup] Removed {removed_count} lines")

    def scroll_to_bottom(self, force=False):
        """Scrolls to bottom, throttled unless forced."""
        current_time = time.time()
        
        should_scroll = force
        if not should_scroll:
            with self._scroll_lock:
                if current_time - self._last_scroll_time >= AUTO_SCROLL_INTERVAL:
                    self._last_scroll_time = current_time
                    should_scroll = True
        
        if should_scroll:
            self._do_scroll()

    def _do_scroll(self):
        # 1. Try explicit control
        if self.scroll_control and hasattr(self.scroll_control, 'scroll_to'):
            try:
                self.scroll_control.scroll_to(offset=-1, duration=50)
                return
            except Exception:
                pass
        
        # 2. Try parent (Legacy fallback)
        try:
            if hasattr(self.text_control, 'parent') and hasattr(self.text_control.parent, 'scroll_to'):
                self.text_control.parent.scroll_to(offset=-1, duration=50)
        except Exception:
            pass


# =====================
# Log Streamer (Unified)
# =====================

class LogStreamer:
    """
    Unified class to stream process output to the console.
    Handles buffering, TensorBoard logging, and callbacks.
    """
    def __init__(self,
                 proc,
                 console_controller: ConsoleController,
                 page: ft.Page,
                 tb_logger: TensorBoardLogger = None,
                 on_complete = None,
                 completion_message: str = None,
                 main_container = None):

        self.proc = proc
        self.controller = console_controller
        self.page = page
        self.tb_logger = tb_logger
        self.on_complete = on_complete
        self.completion_message = completion_message
        self.main_container = main_container

        self._buffer = []
        self._running = False
        self._reader_thread = None

    def start(self):
        """Starts the reader thread and the async flusher task."""
        self._running = True
        
        # Start background thread to read stdout
        self._reader_thread = Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        
        # Start async task on the UI thread to flush buffer
        if self.page:
            self.page.run_task(self._flush_buffer_loop)

    def _read_stdout(self):
        """Background thread: reads line by line, parses TB, buffers text."""
        try:
            if self.proc.stdout:
                for line in iter(self.proc.stdout.readline, ''):
                    if not line:
                        break
                    
                    # Log to TensorBoard
                    if self.tb_logger:
                        self.tb_logger.parse_and_log(line)
                    
                    # Format and Buffer
                    spans = format_output_line(line)
                    self._buffer.append(spans)
                    
        except Exception as e:
            logger.error(f"Streamer read error: {e}")
        finally:
            self._running = False
            # Close TB Logger
            if self.tb_logger:
                self.tb_logger.close()

    async def _flush_buffer_loop(self):
        """Async UI Task: periodically flushes buffer to UI."""
        while self._running or self._buffer:
            try:
                if not self._buffer:
                    # If process finished and buffer empty, exit loop
                    if not self._running: 
                        break
                    await asyncio.sleep(0.1)
                    continue

                # Atomic grab of current buffer content
                chunk_of_span_lists = []
                while self._buffer:
                    chunk_of_span_lists.append(self._buffer.pop(0))
                
                # Flatten list of lists
                flat_spans = [span for sublist in chunk_of_span_lists for span in sublist]
                
                # Update UI via Controller
                if flat_spans:
                    trigger_anim = (self.controller.anim_control is not None)
                    self.controller.append_spans(flat_spans, trigger_anim=trigger_anim)

                    # If animating, handle the "reset" of the padding
                    if trigger_anim and self.controller.anim_control:
                        await asyncio.sleep(0.05)
                        self.controller.anim_control.padding = ft.padding.only(top=0)
                        if self.controller.anim_control.page is not None:
                            self.controller.anim_control.update()

                await asyncio.sleep(0.1) # 10 FPS update rate
                
            except Exception as e:
                logger.error(f"Streamer flush error: {e}")
                await asyncio.sleep(0.5)

        # === Cleanup & Completion ===
        self._handle_completion()

    def _handle_completion(self):
        """Final cleanups after stream ends."""
        # 1. Add completion message
        msgs = []
        if self.completion_message:
            msgs.extend(format_output_line(self.completion_message))
        else:
            # Check exit code
            rc = self.proc.poll()
            if rc is not None:
                color = ft.Colors.GREEN if rc == 0 else ft.Colors.RED
                msgs.append(ft.TextSpan(f"\n[Process Ended] Exit Code: {rc}\n", ft.TextStyle(color=color)))

        if msgs:
            self.controller.append_spans(msgs)
            self.controller.scroll_to_bottom(force=True)

        # 2. Check if process was cancelled by user before running callback
        # If training_proc is None, it means user cancelled (set in handle_cancel_click)
        # Or if exit code is negative, it means terminated by signal (SIGTERM=-15, SIGKILL=-9)
        was_cancelled = False
        rc = self.proc.poll()
        if rc is not None:
            # Negative exit code means signal termination (user cancelled)
            if rc < 0:
                was_cancelled = True
            # Also check if training_proc was cleared (user cancelled)
            elif self.main_container:
                training_proc = getattr(self.main_container, 'training_proc', None)
                if training_proc is None:
                    was_cancelled = True
                # Also check the training_tab_container
                elif hasattr(self.main_container, 'page'):
                    tab_container = getattr(self.main_container.page, 'training_tab_container', None)
                    if tab_container and getattr(tab_container, 'training_proc', None) is None:
                        was_cancelled = True

        # 3. Run Callback only if not cancelled
        if self.on_complete and not was_cancelled:
            try:
                # If callback is a coroutine, we can't await it easily here if this isn't async
                # strictly speaking, but run_task wraps us.
                # For safety, assume sync callback or fire-and-forget.
                if asyncio.iscoroutinefunction(self.on_complete):
                    asyncio.create_task(self.on_complete())
                else:
                    self.on_complete()
            except Exception as e:
                logger.error(f"Completion callback error: {e}")


# =====================
# Utilities
# =====================

def format_output_line(text: str):
    """
    Format a line of output as styled spans for the console.
    """
    spans_line = []
    try:
        s = text.rstrip("\n")

        # Special case: Green for Audio Mode cross-attention message
        if "[Audio Mode] Using real video latents" in s:
            return [ft.TextSpan(s + "\n", style=ft.TextStyle(color=StatusColors.SUCCESS))]

        # Use pre-compiled regex
        parts = ANSI_BRACKET_PATTERN.split(s)
        bracket_count = 0
        for part in parts:
            if not part:
                continue
            if part.startswith("[") and part.endswith("]"):
                bracket_count += 1
                # Dim the first couple of brackets (timestamps/log levels usually)
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
# Status Message Helpers
# =====================

class StatusColors:
    INFO = ft.Colors.with_opacity(0.7, ft.Colors.CYAN)
    SUCCESS = ft.Colors.with_opacity(0.7, ft.Colors.GREEN)
    WARNING = ft.Colors.with_opacity(0.7, ft.Colors.YELLOW)
    ERROR = ft.Colors.with_opacity(0.7, ft.Colors.RED)
    WHITE = ft.Colors.WHITE

def add_status_message(training_console_text, message: str, color=None):
    """Direct helper for simple status messages."""
    try:
        if training_console_text is None: return
        if color is None: color = StatusColors.INFO

        spans = list(training_console_text.spans or [])
        spans.append(ft.TextSpan(message, style=ft.TextStyle(color=color)))
        training_console_text.spans = spans
        # Only update if the control has been added to the page
        if training_console_text.page is not None:
            training_console_text.update()

        # Try to scroll
        try:
            if training_console_text.parent:
                training_console_text.parent.scroll_to(offset=-1, duration=50)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Error adding status: {e}")

def add_info_message(t, m): add_status_message(t, m, StatusColors.INFO)
def add_success_message(t, m): add_status_message(t, m, StatusColors.SUCCESS)
def add_warning_message(t, m): add_status_message(t, m, StatusColors.WARNING)
def add_error_message(t, m): add_status_message(t, m, StatusColors.ERROR)
def add_action_message(t, m): add_status_message(t, m, StatusColors.WHITE)
def force_scroll_to_bottom(t): 
    try: 
        if t.parent: t.parent.scroll_to(offset=-1, duration=50) 
    except: pass


# =====================
# Legacy Wrappers
# =====================
# These functions maintain compatibility with existing code while using the new system.

def start_output_streamer(proc, training_console_text, training_tab_container, main_container):
    """Legacy wrapper for standard streaming."""
    page = training_console_text.page
    
    # Define completion callback to reset buttons
    def on_done():
        try:
            if hasattr(main_container, 'start_btn') and main_container.start_btn:
                main_container.start_btn.text = "Start"
                main_container.start_btn.update()
            if hasattr(training_tab_container, 'training_proc'):
                training_tab_container.training_proc = None
            if hasattr(main_container, 'training_proc'):
                main_container.training_proc = None
            page.update()
        except Exception:
            pass

    # Try to find scrollable parent
    scroll_parent = getattr(training_console_text, 'parent', None)

    ctrl = ConsoleController(training_console_text, scroll_control=scroll_parent)
    streamer = LogStreamer(proc, ctrl, page, on_complete=on_done)
    streamer.start()
    return streamer._reader_thread # Return thread to satisfy legacy signature


def start_ltx_output_streamer(proc, training_console_text, tb_logger=None, main_container=None, page=None, on_complete_callback=None, skip_button_reset=False, completion_message=None):
    """Legacy wrapper for LTX streaming (with TensorBoard)."""

    def on_done():
        if on_complete_callback:
            try:
                on_complete_callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")

        if not skip_button_reset and main_container and page:
            try:
                # Attempt to import dynamically to avoid circular imports
                from flet_app.ui.training.start_button_handler import reset_to_start_button
                # Try to get tab container from page
                tab_container = getattr(page, 'training_tab_container', None)
                reset_to_start_button(main_container, tab_container, page)
            except Exception:
                pass

    scroll_parent = getattr(training_console_text, 'parent', None)
    ctrl = ConsoleController(training_console_text, scroll_control=scroll_parent)

    streamer = LogStreamer(
        proc,
        ctrl,
        page,
        tb_logger=tb_logger,
        on_complete=on_done,
        completion_message=completion_message,
        main_container=main_container
    )
    streamer.start()
    return streamer._reader_thread


def start_buffered_output_streamer(proc, training_console_text, monitor_content, page):
    """Legacy wrapper for buffered streaming with animation."""
    
    # Attempt to find the animation wrapper used in original code
    anim_wrap = getattr(monitor_content, 'training_console_anim', None)
    scroll_parent = getattr(training_console_text, 'parent', None)

    ctrl = ConsoleController(
        training_console_text, 
        scroll_control=scroll_parent, 
        anim_control=anim_wrap
    )
    
    streamer = LogStreamer(proc, ctrl, page)
    streamer.start()
    return streamer._reader_thread