import flet as ft
from .._styles import create_textfield, add_section_title
from ..utils.console_cleanup import cleanup_training_console
import subprocess
import socket
import time
import shutil
import os
import re
from pathlib import Path
from flet_app.settings import settings


def get_training_monitor_page_content():
    """Build the Monitor page with Monitoring and Tensorboard sections."""

    # Reference to store console text for manual cleanup
    console_ref = {"training_console_text": None}

    def clean_console_manually():
        """Manual cleanup function for the Clean button - clears all console content."""
        try:
            console_text = console_ref.get("training_console_text")
            if console_text and console_text.spans:
                # Count current lines before clearing
                current_lines = sum(span.text.count('\n') for span in console_text.spans if hasattr(span, 'text'))

                if current_lines > 0:  # Only clean if there's content
                    # Clear all content completely
                    console_text.spans = []
                    console_text.update()
                    print(f"[Console] Manual full cleanup: cleared {current_lines} lines")
                else:
                    print("[Console] Manual cleanup: no content to clear")
            else:
                print("[Console] Manual cleanup: no console content available")
        except Exception as e:
            print(f"[Console] Manual cleanup error: {e}")

    # Store the cleanup function in the console_ref for access later
    console_ref["clean_console_manually"] = clean_console_manually

    # Monitoring column (displayed on the right) is defined after Tensorboard configuration so it can re-use shared helpers.

    # Tensorboard column (displayed on the left, Save As popup style)
    # Scan workspace/output two levels deep and list items as parent\child with mtime
    from pathlib import Path
    import os
    def _resolve_output_base_dir() -> Path:
        candidates = []
        # Use project_location from settings
        project_location = settings.get("project_location")
        if project_location:
            candidates.append(Path(project_location) / "workspace" / "output")
            # Fallback for any backslash normalization issues
            candidates.append(Path(project_location.replace("\\", "/")) / "workspace" / "output")
        # From this file (flet_app/ui/pages/ -> project root -> workspace/output)
        try:
            from flet_app.project_root import get_project_root
            proj_root = get_project_root()
            candidates.append(proj_root / "workspace" / "output")
        except Exception:
            pass
        # From cwd
        try:
            candidates.append(Path(os.getcwd()) / "workspace" / "output")
        except Exception:
            pass
        for c in candidates:
            try:
                if c.exists():
                    return c
            except Exception:
                continue
        return candidates[0]

    base_output_dir = _resolve_output_base_dir()

    def _scan_runs(base_dir: Path):
        items = []
        try:
            if not base_dir.exists():
                return []
            for parent in base_dir.iterdir():
                if not parent.is_dir():
                    continue
                for child in parent.iterdir():
                    if not child.is_dir():
                        continue
                    disp = f"{parent.name}\\{child.name}"
                    try:
                        ts = child.stat().st_mtime
                    except Exception:
                        ts = 0.0
                    # Format date as YYYY-MM-DD HH:MM
                    from datetime import datetime
                    date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                    items.append((disp, date_str, str(child)))
        except Exception:
            pass
        # Default sort: newest first
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    items_data = _scan_runs(base_output_dir)

    sort_mode = {"value": "date"}  # 'name' or 'date'
    sort_asc = {"name": True, "date": False}
    selected = {"full": None, "display": ""}

    LOSS_TAG_CANDIDATES = [
        "epoch_loss",
        "loss",
        "train/loss",
        "train_loss",
        "loss/total",
        "train/epoch_loss",
    ]

    _tb_event_module_cache = {"module": None, "tried": False}

    def _get_event_module():
        cache = _tb_event_module_cache
        if cache["tried"]:
            return cache["module"]
        cache["tried"] = True
        try:
            from tensorboard.backend.event_processing import event_accumulator as ea_module  # type: ignore
        except Exception:  # pragma: no cover - tensorboard optional
            ea_module = None
        cache["module"] = ea_module
        return ea_module

    def _summarize_loss(log_dir: Path) -> tuple[str, str]:
        ea_module = _get_event_module()
        if ea_module is None:
            return "", "tensorboard not installed"
        try:
            accumulator = ea_module.EventAccumulator(
                str(log_dir),
                size_guidance={ea_module.SCALARS: 0},
            )
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", []) or []
            if not scalar_tags:
                return "", "no scalar tags"
            tag = next((t for t in LOSS_TAG_CANDIDATES if t in scalar_tags), None)
            if tag is None:
                for t in scalar_tags:
                    if "loss" in t.lower():
                        tag = t
                        break
            if tag is None:
                return "", f"no loss tag (found: {', '.join(scalar_tags[:5])})"
            scalars = accumulator.Scalars(tag) or []
            if not scalars:
                return "", "no scalar data"
            last = scalars[-1]
            value = getattr(last, "value", None)
            step = getattr(last, "step", None)
            if value is None:
                return "", "last scalar missing value"
            try:
                value_display = f"{float(value):.4g}"
            except Exception:
                value_display = str(value)
            summary = f"loss: {value_display}"
            if step is not None:
                summary += f" (step {step})"
            return summary, ""
        except Exception as exc:
            return "", f"error: {exc.__class__.__name__}"

    def _describe_event_dir(log_dir: Path) -> None:
        try:
            event_files = sorted(log_dir.glob("events.out.tfevents*"))
            if not event_files:
                return
            ea_module = _get_event_module()
            if ea_module is None:
                return
            accumulator = ea_module.EventAccumulator(
                str(log_dir),
                size_guidance={ea_module.SCALARS: 0},
            )
            accumulator.Reload()
            tags = accumulator.Tags() or {}
            scalar_tags = tags.get("scalars", []) or []
            for tag in scalar_tags[:5]:
                scalars = accumulator.Scalars(tag) or []
                if not scalars:
                    continue
                last = scalars[-1]
                step = getattr(last, 'step', None)
                value = getattr(last, 'value', None)
            if len(scalar_tags) > 5:
                pass
            collected = _collect_epoch_losses(log_dir)
            mapping = collected.get("by_step") if isinstance(collected, dict) else {}
            if isinstance(mapping, dict) and mapping:
                preview = sorted(mapping.items())[:5]
                def _fmt_val(val):
                    try:
                        return f"{float(val):.4g}"
                    except Exception:
                        return str(val)
                preview_str = ", ".join(f"step {step}: {_fmt_val(value)}" for step, value in preview)
                tag_name = collected.get("tag") if isinstance(collected, dict) else ""
                tag_display = _shorten_tag(str(tag_name)) if tag_name else "loss"
                if len(mapping) > len(preview):
                    pass
            else:
                debug_msg = collected.get("debug") if isinstance(collected, dict) else None
                if debug_msg:
                    pass
        except Exception as exc:
            pass

    def _collect_epoch_losses(run_dir: Path | str) -> dict[str, object]:
        result: dict[str, object] = {"by_step": {}, "tag": "", "debug": ""}
        try:
            run_path = Path(run_dir)
        except Exception:
            run_path = Path(str(run_dir))
        ea_module = _get_event_module()
        if ea_module is None:
            result["debug"] = "tensorboard not installed"
            return result
        try:
            accumulator = ea_module.EventAccumulator(
                str(run_path),
                size_guidance={ea_module.SCALARS: 0},
            )
            accumulator.Reload()
            tags = accumulator.Tags() or {}
            scalar_tags = tags.get("scalars", []) or []
            if not scalar_tags:
                result["debug"] = "no scalar tags"
                return result
            preferred_order = ["train/epoch_loss", "epoch_loss"] + [t for t in LOSS_TAG_CANDIDATES if t not in {"train/epoch_loss", "epoch_loss"}]
            tag = next((t for t in preferred_order if t in scalar_tags), None)
            if tag is None:
                tag = next((t for t in scalar_tags if "loss" in t.lower()), None)
            if tag is None:
                preview = ", ".join(scalar_tags[:5])
                result["debug"] = f"no loss tag (found: {preview})"
                return result
            scalars = accumulator.Scalars(tag) or []
            if not scalars:
                result["debug"] = f"{tag}: no scalar data"
                return result
            mapping: dict[int, float] = {}
            for scalar in scalars:
                step = getattr(scalar, "step", None)
                value = getattr(scalar, "value", None)
                if step is None or value is None:
                    continue
                try:
                    mapping[int(step)] = float(value)
                except Exception:
                    continue
            if not mapping:
                result["debug"] = f"{tag}: no numeric steps"
                return result
            result["by_step"] = mapping
            result["tag"] = tag
        except Exception as exc:
            result["debug"] = f"error: {exc.__class__.__name__}"
        return result

    epoch_loss_cache = {"by_step": {}, "tag": "", "debug": ""}

    def _parse_epoch_step(name: str) -> int | None:
        import re
        match = re.search(r"epo(?:ch|ck)[_-]?(\d+)", name.lower())
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _shorten_tag(tag: str) -> str:
        if not tag:
            return "loss"
        if "/" in tag:
            return tag.split("/")[-1]
        return tag

    def _scan_subfolders(parent_dir: Path | str) -> list[tuple[str, str, str, str, str]]:
        items: list[tuple[str, str, str, str, str]] = []
        try:
            if not parent_dir:
                return []
            p = Path(parent_dir)
            if not p.exists() or not p.is_dir():
                return []
            for child in p.iterdir():
                if not child.is_dir():
                    continue
                try:
                    ts = child.stat().st_mtime
                except Exception:
                    ts = 0.0
                from datetime import datetime
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                epoch_step = _parse_epoch_step(child.name)
                loss_summary = ""
                loss_debug = ""
                mapping = epoch_loss_cache.get("by_step", {})
                tag_name = epoch_loss_cache.get("tag", "")
                if epoch_step is not None and isinstance(mapping, dict) and mapping:
                    if epoch_step in mapping:
                        value = mapping[epoch_step]
                        try:
                            value_display = f"{float(value):.4g}"
                        except Exception:
                            value_display = str(value)
                        short_tag = _shorten_tag(str(tag_name))
                        loss_summary = f"{short_tag}: {value_display} (step {epoch_step})"
                    else:
                        loss_debug = f"no scalar for step {epoch_step}"
                if not loss_summary:
                    alt_summary, alt_debug = _summarize_loss(child)
                    if alt_summary:
                        loss_summary = alt_summary
                        loss_debug = alt_debug
                    elif not loss_debug:
                        cache_debug = epoch_loss_cache.get("debug")
                        if cache_debug:
                            loss_debug = str(cache_debug)
                        else:
                            loss_debug = "no loss data"
                items.append((child.name, date_str, str(child), loss_summary, loss_debug))
        except Exception:
            pass
        items.sort(key=lambda x: x[0].lower())
        return items

    monitor_sort_mode = {"value": "name"}
    monitor_sort_asc = {"name": True, "date": False, "loss": False}
    monitor_items_state = {"items": [("(no folder selected)", "", "", "", "")]}
    monitor_selected = {"full": "", "name": "", "loss_value": ""}
    _pending_download_file = None

    def handle_save_dialog_result(e: ft.FilePickerResultEvent):
        """Handle the result of the save file dialog."""
        global _pending_download_file
        if e.path and _pending_download_file:
            try:
                # Copy the file to the selected location
                shutil.copy2(_pending_download_file, e.path)
                print(f"[Download] SUCCESS: File saved to {e.path}")
                _pending_download_file = None
            except Exception as copy_error:
                print(f"[Download] ERROR: Failed to copy file: {copy_error}")
        else:
            print(f"[Download] Save dialog cancelled or failed. Path: {e.path}")

    monitor_list_view = ft.ListView(spacing=2, expand=True, auto_scroll=False, first_item_prototype=True)
    # monitor_selected_path is no longer needed since we removed the path display
    monitor_refresh_ctrl = {"btn": None}

    # Create FilePicker for download functionality
    download_file_picker = ft.FilePicker(
        on_result=handle_save_dialog_result
    )

    def make_monitor_tile(name: str, date_str: str, full_path: str, loss_summary: str, loss_debug: str) -> ft.Container:
        # Extract just the number and step from loss_summary, removing "epoch loss:" prefix
        loss_display = ""
        if loss_summary:
            # Extract just the number and (step num) part, removing any prefix like "epoch loss:" or "loss:"
            import re
            match = re.search(r'([\d.]+)\s*\(step\s*(\d+)\)', loss_summary)
            if match:
                value, step = match.groups()
                loss_display = f"{value} ({step})"
            else:
                # Fallback: try to extract just a number if no step found
                match = re.search(r'([\d.]+)', loss_summary)
                if match:
                    loss_display = match.group(1)
        elif loss_debug:
            loss_display = f"unavailable ({loss_debug})"

        bg_default = ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY_50)
        bg_selected = ft.Colors.with_opacity(0.18, ft.Colors.LIGHT_BLUE_100)
        is_selected = monitor_selected["full"] == full_path

        def on_click(e):
            # Check if clicking the same item that's already selected (deselect)
            if monitor_selected["full"] == full_path and full_path != "(no folder selected)":
                # Deselect the item
                monitor_selected["full"] = ""
                monitor_selected["name"] = ""
                monitor_selected["loss_value"] = ""

                # Clear visual selection
                tile.bgcolor = bg_default
                if tile.page:
                    tile.update()

                # Disable buttons
                update_button_states(False)
                return

            # Clear previous selection
            try:
                for ctrl in monitor_list_view.controls:
                    if isinstance(ctrl, ft.Container):
                        ctrl.bgcolor = bg_default
                        if ctrl.page:
                            ctrl.update()
            except Exception:
                pass

            # Select new tile
            monitor_selected["full"] = full_path
            monitor_selected["name"] = name
            # Extract and store loss value (just the number, no step info)
            if loss_summary:
                import re
                match = re.search(r'([\d.]+)', loss_summary)
                if match:
                    monitor_selected["loss_value"] = match.group(1)
                else:
                    monitor_selected["loss_value"] = ""
            else:
                monitor_selected["loss_value"] = ""

            # Update visual selection
            tile.bgcolor = bg_selected
            if tile.page:
                tile.update()

            # Update button states
            if full_path and full_path != "(no folder selected)":
                update_button_states(True)
            else:
                update_button_states(False)

        tile = ft.Container(
            content=ft.Row([
                # Folder name - 33% width, left aligned
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.FOLDER, size=16, color=ft.Colors.BLUE_GREY_400),
                        ft.Text(name, size=12, color=ft.Colors.BLUE_GREY_500, weight=ft.FontWeight.BOLD),
                    ], spacing=8),
                    expand=True,  # Use expand for percentage-based width
                    alignment=ft.alignment.center_left,
                ),
                # Epoch loss - 33% width, left aligned
                ft.Container(
                    content=ft.Text(
                        loss_display,
                        size=12,
                        color=ft.Colors.BLUE_GREY_700 if loss_summary else ft.Colors.BLUE_GREY_600,
                        italic=True if not loss_summary else False,
                        weight=ft.FontWeight.NORMAL
                    ),
                    expand=True,  # Use expand for percentage-based width
                    alignment=ft.alignment.center_left,
                ),
                # Date - 33% width, right aligned
                ft.Container(
                    content=ft.Text(date_str, size=12, color=ft.Colors.BLUE_GREY_600),
                    expand=True,  # Use expand for percentage-based width
                    alignment=ft.alignment.center_right,
                ),
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.symmetric(horizontal=8, vertical=8),
            border_radius=6,
            bgcolor=bg_selected if is_selected else bg_default,
            tooltip=full_path or None,
            on_click=on_click,
        )
        return tile

    def refresh_monitor_list():
        data = list(monitor_items_state.get("items", []))
        if monitor_sort_mode["value"] == "name":
            data.sort(key=lambda x: x[0].lower() if x[0] else "", reverse=not monitor_sort_asc["name"])
        elif monitor_sort_mode["value"] == "loss":
            # Sort by loss value (extract numeric part for proper sorting)
            def extract_loss_value(loss_str):
                import re
                match = re.search(r'([\d.]+)', loss_str)
                return float(match.group(1)) if match else float('inf')
            data.sort(key=lambda x: extract_loss_value(x[3]), reverse=not monitor_sort_asc["loss"])
        else:  # date
            data.sort(key=lambda x: x[1], reverse=not monitor_sort_asc["date"])
        monitor_list_view.controls = [make_monitor_tile(n, d, full, loss, dbg) for (n, d, full, loss, dbg) in data]
        if monitor_list_view.page:
            monitor_list_view.update()

    def update_monitoring_from_selection(rescan: bool = True):
        path_value = selected.get("full")
        if not path_value:
            monitor_selected["full"] = ""
            monitor_selected["name"] = ""
            monitor_items_state["items"] = [("(no folder selected)", "", "", "", "")]
            update_button_states(False)
            btn = monitor_refresh_ctrl.get("btn")
            if btn:
                btn.disabled = True
                if btn.page:
                    btn.update()
            refresh_monitor_list()
            return
        if rescan:
            epoch_loss_cache["by_step"] = {}
            epoch_loss_cache["tag"] = ""
            epoch_loss_cache["debug"] = ""
            collected = _collect_epoch_losses(path_value)
            if isinstance(collected, dict):
                epoch_loss_cache.update(collected)
            items = _scan_subfolders(path_value)
            monitor_items_state["items"] = items or [("(no subfolders found)", "", "", "", "")]
        btn = monitor_refresh_ctrl.get("btn")
        if btn:
            btn.disabled = False
            if btn.page:
                btn.update()
        refresh_monitor_list()

    def monitor_on_name_click(_: ft.ControlEvent | None = None):
        monitor_sort_mode["value"] = "name"
        monitor_sort_asc["name"] = not monitor_sort_asc["name"]
        refresh_monitor_list()

    def monitor_on_date_click(_: ft.ControlEvent | None = None):
        monitor_sort_mode["value"] = "date"
        monitor_sort_asc["date"] = not monitor_sort_asc["date"]
        refresh_monitor_list()

    def monitor_on_loss_click(_: ft.ControlEvent | None = None):
        monitor_sort_mode["value"] = "loss"
        monitor_sort_asc["loss"] = not monitor_sort_asc["loss"]
        refresh_monitor_list()

    def find_adapter_model(folder_path: Path) -> Path | None:
        """Find adapter_model.safetensors file in the folder and its subdirectories."""
        try:
            print(f"[find_adapter_model] Searching in: {folder_path}")
            # Search for adapter_model.safetensors in the folder and subdirectories
            found_files = []
            for file_path in folder_path.rglob("adapter_model.safetensors"):
                if file_path.is_file():
                    found_files.append(file_path)
                    print(f"[find_adapter_model] Found file: {file_path}")

            if found_files:
                print(f"[find_adapter_model] Returning first file: {found_files[0]}")
                return found_files[0]
            else:
                print(f"[find_adapter_model] No files found")
                return None
        except Exception as e:
            print(f"[find_adapter_model] ERROR: {e}")
            return None

    def generate_download_name(source_path: Path, loss_value: str = "") -> str:
        """Generate download filename from the source path pattern."""
        try:
            print(f"[generate_download_name] Processing path: {source_path}")
            # Extract path parts: output/q_test/20251101_02-52-21/epoch30/adapter_model.safetensors
            parts = source_path.parts
            print(f"[generate_download_name] Path parts: {parts}")

            # Find the folder structure pattern
            # Look for date pattern (YYYYMMDD_HH-MM-SS) and epoch folder
            date_pattern = None
            epoch_num = None
            project_name = None

            for i, part in enumerate(parts):
                print(f"[generate_download_name] Processing part {i}: {part}")

                # Check for date pattern (YYYYMMDD_HH-MM-SS)
                if re.match(r'\d{8}_\d{2}-\d{2}-\d{2}', part):
                    date_pattern = part
                    print(f"[generate_download_name] Found date pattern: {date_pattern}")
                    # Extract date portion (YYYYMMDD) and convert to MMDD
                    date_part = part.split('_')[0]  # Get YYYYMMDD
                    mmdd = date_part[4:]  # Get MMDD from YYYYMMDD
                    # Get project name from previous part if available
                    if i > 0:
                        project_name = parts[i-1]
                        print(f"[generate_download_name] Found project name: {project_name}")
                    continue

                # Check for epoch pattern
                if part.startswith('epoch'):
                    epoch_part = part[5:]  # Remove 'epoch' prefix
                    if epoch_part.isdigit():
                        epoch_num = epoch_part
                        print(f"[generate_download_name] Found epoch number: {epoch_num}")

            # Construct the new filename
            if project_name and date_pattern and epoch_num:
                date_part = date_pattern.split('_')[0]  # YYYYMMDD
                mmdd = date_part[4:]  # MMDD
                base_name = f"{project_name}_{mmdd}_ep{epoch_num}"
                # Add loss suffix if provided
                if loss_value:
                    base_name += f"_{loss_value}"
                result = f"{base_name}.safetensors"
                print(f"[generate_download_name] Generated name: {result}")
                return result
            else:
                # Fallback: use original filename with optional loss suffix
                base_name = source_path.stem
                if loss_value:
                    base_name += f"_{loss_value}"
                result = f"{base_name}{source_path.suffix}"
                print(f"[generate_download_name] Pattern not matched, using modified original: {result}")
                print(f"[generate_download_name] project_name: {project_name}, date_pattern: {date_pattern}, epoch_num: {epoch_num}")
                return result

        except Exception as e:
            # Fallback to original filename
            print(f"[generate_download_name] ERROR: {e}")
            return source_path.name

    def monitor_on_download_click(_: ft.ControlEvent | None = None):
        selected_path = monitor_selected["full"]
        selected_name = monitor_selected["name"]
        print(f"[Download] Clicked - selected_path: {selected_path}, selected_name: {selected_name}")

        if selected_path and selected_path != "(no folder selected)":
            try:
                folder_path = Path(selected_path)
                print(f"[Download] Folder path: {folder_path}")
                print(f"[Download] Folder exists: {folder_path.exists()}, is_dir: {folder_path.is_dir()}")

                # Find the adapter_model.safetensors file
                adapter_file = find_adapter_model(folder_path)
                print(f"[Download] Found adapter file: {adapter_file}")

                if not adapter_file:
                    print(f"[Download] ERROR: adapter_model.safetensors not found in {folder_path}")
                    return

                # Generate the download filename with optional loss suffix
                loss_value = monitor_selected.get("loss_value", "") if monitor_include_loss_checkbox.value else ""
                download_name = generate_download_name(adapter_file, loss_value)
                print(f"[Download] Generated download name: {download_name} (loss included: {bool(loss_value)})")

                # Start simple HTTP server for file download
                try:
                    import threading
                    import http.server
                    import socketserver
                    import subprocess
                    from urllib.parse import quote

                    def start_file_server():
                        """Start HTTP server to serve the file with download headers."""
                        try:
                            # Get WSL IP address for Windows access
                            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=5)
                            wsl_ip = result.stdout.strip().split()[0] if result.returncode == 0 else 'localhost'

                            # Create HTTP server that serves the file
                            class DownloadHandler(http.server.SimpleHTTPRequestHandler):
                                def do_GET(self):
                                    if self.path.startswith('/download'):
                                        try:
                                            # Extract filename from URL
                                            filename = self.path[10:]  # Remove '/download/'
                                            if filename == download_name:
                                                # Serve the file with download headers
                                                self.send_response(200)
                                                self.send_header('Content-Type', 'application/octet-stream')
                                                self.send_header('Content-Disposition', f'attachment; filename="{download_name}"')
                                                self.send_header('Content-Length', str(os.path.getsize(adapter_file)))
                                                self.end_headers()

                                                with open(adapter_file, 'rb') as f:
                                                    self.wfile.write(f.read())
                                                return
                                        except Exception as e:
                                            self.send_error(500, f"Error serving file: {e}")
                                    else:
                                        self.send_error(404, "File not found")

                                def log_message(self, format, *args):
                                    # Suppress server logging
                                    pass

                            # Start server on a random port
                            with socketserver.TCPServer(("0.0.0.0", 0), DownloadHandler) as httpd:
                                port = httpd.server_address[1]
                                server_url = f"http://{wsl_ip}:{port}/download/{quote(download_name)}"

                                print(f"[Download] Server started: {server_url}")

                                # Open browser after short delay using same method as Monitor button
                                def open_browser():
                                    import time
                                    time.sleep(0.5)
                                    try:
                                        # Use same method as Monitor button - launch_url through content.page
                                        if hasattr(content, 'page') and content.page:
                                            content.page.launch_url(server_url)
                                            print(f"[Download] Browser opened with URL: {server_url}")
                                        else:
                                            # Fallback to Windows browser if page context not available
                                            subprocess.run(['cmd.exe', '/c', 'start', server_url], check=False)
                                            print(f"[Download] Fallback browser opened with URL: {server_url}")
                                    except Exception as e:
                                        print(f"[Download] Browser opening failed: {e}")
                                        # Final fallback to webbrowser
                                        import webbrowser
                                        webbrowser.open(server_url)
                                        print(f"[Download] Final fallback browser opened with URL: {server_url}")

                                threading.Thread(target=open_browser, daemon=True).start()

                                # Serve the file (handle one request then shutdown)
                                httpd.handle_request()
                                httpd.server_close()
                                print(f"[Download] Server shut down")

                        except Exception as e:
                            print(f"[Download] Server error: {e}")

                    # Start server in separate thread
                    server_thread = threading.Thread(target=start_file_server, daemon=True)
                    server_thread.start()

                    print(f"[Download] Server starting for {download_name}")

                except Exception as download_error:
                    print(f"[Download] HTTP server failed: {download_error}")
                    import traceback
                    print(f"[Download] Traceback: {traceback.format_exc()}")

            except Exception as e:
                # Handle download errors
                print(f"[Download] ERROR: {e}")
                import traceback
                print(f"[Download] Traceback: {traceback.format_exc()}")
                pass
        else:
            print(f"[Download] ERROR: No valid selection - path: {selected_path}")

    def monitor_on_delete_click(_: ft.ControlEvent | None = None):
        selected_path = monitor_selected["full"]
        selected_name = monitor_selected["name"]
        if selected_path and selected_path != "(no folder selected)":
            try:
                # Confirm deletion
                folder_path = Path(selected_path)
                if folder_path.exists() and folder_path.is_dir():
                    # Delete the entire folder and its contents
                    shutil.rmtree(folder_path)

                    # Clear selection and disable buttons
                    monitor_selected["full"] = ""
                    monitor_selected["name"] = ""
                    update_button_states(False)

                    # Refresh the monitor list
                    update_monitoring_from_selection(rescan=True)

                    # Clear any visual selection in the list
                    try:
                        for ctrl in monitor_list_view.controls:
                            if isinstance(ctrl, ft.Container):
                                ctrl.bgcolor = ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY_50)
                                if ctrl.page:
                                    ctrl.update()
                    except Exception:
                        pass

            except Exception as e:
                # Handle deletion errors (permissions, folder not found, etc.)
                print(f"Error deleting folder {selected_path}: {e}")
                pass

    def update_button_states(is_enabled: bool):
        """Update button visual states based on selection"""
        try:
            if is_enabled:
                # Enable buttons with bright colors (icon and text only)
                monitor_download_btn.disabled = False
                monitor_download_btn.icon_color = ft.Colors.BLUE_600
                monitor_download_btn.style.color = ft.Colors.BLUE_600

                monitor_include_loss_checkbox.disabled = False

                monitor_delete_btn.disabled = False
                monitor_delete_btn.icon_color = ft.Colors.RED_600
                monitor_delete_btn.style.color = ft.Colors.RED_600
            else:
                # Disable buttons with dimmed colors (icon and text only)
                monitor_download_btn.disabled = True
                monitor_download_btn.icon_color = ft.Colors.BLUE_GREY_400
                monitor_download_btn.style.color = ft.Colors.BLUE_GREY_400

                monitor_include_loss_checkbox.disabled = True

                monitor_delete_btn.disabled = True
                monitor_delete_btn.icon_color = ft.Colors.BLUE_GREY_400
                monitor_delete_btn.style.color = ft.Colors.BLUE_GREY_400

            if monitor_download_btn.page:
                monitor_download_btn.update()
            if monitor_include_loss_checkbox.page:
                monitor_include_loss_checkbox.update()
            if monitor_delete_btn.page:
                monitor_delete_btn.update()
        except Exception:
            pass

    monitor_header_row = ft.Container(
        content=ft.Row([
            # Folder name - 33% width, left aligned
            ft.Container(
                content=ft.TextButton(text="Name", on_click=monitor_on_name_click),
                expand=True,
                alignment=ft.alignment.center_left,
            ),
            # Epoch loss - 33% width, left aligned
            ft.Container(
                content=ft.TextButton(text="Epoch Loss", on_click=monitor_on_loss_click),
                expand=True,
                alignment=ft.alignment.center_left,
            ),
            # Date - 33% width, right aligned
            ft.Container(
                content=ft.TextButton(text="Date", on_click=monitor_on_date_click),
                expand=True,
                alignment=ft.alignment.center_right,
            ),
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.padding.symmetric(horizontal=8, vertical=6),
        border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.15, ft.Colors.OUTLINE))),
    )

    def monitor_refresh(_: ft.ControlEvent | None = None):
        update_monitoring_from_selection(rescan=True)

    monitor_refresh_btn = ft.IconButton(icon=ft.Icons.REFRESH, tooltip="Refresh", on_click=monitor_refresh, icon_color=ft.Colors.BLUE_GREY_600, disabled=True)
    monitor_refresh_ctrl["btn"] = monitor_refresh_btn

    monitor_download_btn = ft.TextButton(
        text="Download",
        icon=ft.Icons.DOWNLOAD,
        on_click=monitor_on_download_click,
        disabled=True,
        icon_color=ft.Colors.BLUE_GREY_400,
        style=ft.ButtonStyle(
            color=ft.Colors.BLUE_GREY_400,
        ),
    )

    # Add +_loss checkbox for including loss in filename
    monitor_include_loss_checkbox = ft.Checkbox(
        label="+_loss",
        value=False,
        disabled=True
    )
    monitor_delete_btn = ft.TextButton(
        text="Delete",
        icon=ft.Icons.DELETE,
        on_click=monitor_on_delete_click,
        disabled=True,
        icon_color=ft.Colors.BLUE_GREY_400,
        style=ft.ButtonStyle(
            color=ft.Colors.BLUE_GREY_400,
        ),
    )

    monitor_header_with_refresh = ft.Row([monitor_header_row, monitor_refresh_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    monitor_list_container = ft.Container(
        content=ft.Column([
            monitor_header_with_refresh,
            ft.Container(
                content=monitor_list_view,
                expand=True,
                clip_behavior=ft.ClipBehavior.HARD_EDGE
            )
        ], spacing=0, expand=True),
        height=300,
        margin=ft.margin.only(top=10),
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=8,
        padding=5,
    )

    monitor_footer = ft.Row([
        ft.Container(expand=True),  # Spacer to push buttons to the right
        ft.Row([
            monitor_include_loss_checkbox,
            monitor_download_btn,
            monitor_delete_btn,
        ], spacing=8),
    ], alignment=ft.MainAxisAlignment.END, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    monitoring_col = ft.Column([
        *add_section_title("Monitoring"),
        monitor_list_container,
        monitor_footer,
    ], spacing=6)

    update_monitoring_from_selection(rescan=False)

    def make_tile(name: str, date_str: str, full_path: str) -> ft.Container:
        bg_default = ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY_50)
        bg_selected = ft.Colors.with_opacity(0.18, ft.Colors.LIGHT_BLUE_100)
        is_sel = selected["full"] == full_path
        tile = ft.Container(
            content=ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.FOLDER, size=16, color=ft.Colors.BLUE_GREY_400),
                    ft.Row([
                        # Split name: project name (bold, lighter color) + date part (regular, darker color)
                        ft.Text(
                            name.split('\\')[0] if '\\' in name else name,
                            size=12,
                            color=ft.Colors.BLUE_GREY_500,
                            weight=ft.FontWeight.BOLD
                        ),
                        ft.Text(
                            '\\' + name.split('\\')[1] if '\\' in name else '',
                            size=12,
                            color=ft.Colors.BLUE_GREY_700
                        ),
                    ], spacing=0),
                ], spacing=8),
                ft.Container(expand=True),
                ft.Text(date_str, size=12, color=ft.Colors.BLUE_GREY_600),
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.symmetric(horizontal=8, vertical=8),
            border_radius=6,
            bgcolor=(bg_selected if is_sel else bg_default),
            tooltip=full_path or None,
        )

        def on_click(_: ft.ControlEvent | None = None):
            # Toggle selection if clicking the same tile
            if selected["full"] == full_path:
                selected["full"] = None
                selected["display"] = ""
                try:
                    for ctrl in list_view.controls:
                        if isinstance(ctrl, ft.Container):
                            ctrl.bgcolor = bg_default
                            if ctrl.page:
                                ctrl.update()
                except Exception:
                    pass
                try:
                    monitor_btn.disabled = True
                    tensorboard_delete_btn.disabled = True
                    tensorboard_delete_btn.icon_color = ft.Colors.BLUE_GREY_400
                    tensorboard_delete_btn.style.color = ft.Colors.BLUE_GREY_400
                    if monitor_btn.page:
                        monitor_btn.update()
                    if tensorboard_delete_btn.page:
                        tensorboard_delete_btn.update()
                except Exception:
                    pass
                update_monitoring_from_selection(rescan=False)
                return

            # Select new tile and unselect others
            selected["full"] = full_path
            selected["display"] = name
            try:
                _describe_event_dir(Path(full_path))
            except Exception:
                pass
            try:
                for ctrl in list_view.controls:
                    if isinstance(ctrl, ft.Container):
                        ctrl.bgcolor = bg_default
                        if ctrl.page:
                            ctrl.update()
                tile.bgcolor = bg_selected
                if tile.page:
                    tile.update()
            except Exception:
                pass
            try:
                monitor_btn.disabled = False
                tensorboard_delete_btn.disabled = False
                tensorboard_delete_btn.icon_color = ft.Colors.RED_600
                tensorboard_delete_btn.style.color = ft.Colors.RED_600
                if monitor_btn.page:
                    monitor_btn.update()
                if tensorboard_delete_btn.page:
                    tensorboard_delete_btn.update()
            except Exception:
                pass
            update_monitoring_from_selection(rescan=True)

        tile.on_click = on_click
        return tile

    list_view = ft.ListView(spacing=2, expand=True, auto_scroll=False, first_item_prototype=True)

    # Left info: "Started at:" + clickable link (hidden initially)
    left_info_prefix = ft.Text(value="Started at:", size=12, color=ft.Colors.BLUE_GREY_700, visible=False)
    def _open_tb(e):
        try:
            if e and getattr(e, 'page', None):
                # Open the current link text if visible
                if left_info_link and left_info_link.text:
                    e.page.launch_url(left_info_link.text)
        except Exception:
            pass
    left_info_link = ft.TextButton(text="", on_click=_open_tb, visible=False)
    left_info_container = ft.Row([left_info_prefix, left_info_link], spacing=6, visible=False)

    # Tensorboard process state
    tb_state = {"proc": None, "port": None}

    def _is_proc_running(proc: subprocess.Popen | None) -> bool:
        try:
            return (proc is not None) and (proc.poll() is None)
        except Exception:
            return False

    def _stop_tb_proc():
        proc = tb_state.get("proc")
        if _is_proc_running(proc):
            try:
                proc.terminate()
            except Exception:
                pass
            # brief wait, then kill if still alive
            try:
                for _ in range(10):
                    if not _is_proc_running(proc):
                        break
                    time.sleep(0.1)
                if _is_proc_running(proc):
                    proc.kill()
            except Exception:
                pass
        tb_state["proc"] = None
        tb_state["port"] = None

    def _find_free_port(start_port: int = 6006, limit: int = 50) -> int:
        for p in range(start_port, start_port + limit):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", p))
                s.close()
                return p
            except Exception:
                try:
                    s.close()
                except Exception:
                    pass
                continue
        return start_port
    def on_monitor_click(_: ft.ControlEvent | None = None):
        # Start/restart tensorboard for selected folder
        try:
            if hasattr(content, 'page') and content.page and selected["full"]:
                setattr(content.page, 'tensorboard_selected_run', selected["full"])
                # Stop any previous TB
                _stop_tb_proc()
                # Pick a free port and launch
                port = _find_free_port(6006, 100)
                logdir = str(selected["full"])
                cmd = ["tensorboard", "--logdir", logdir, "--host", "127.0.0.1", "--port", str(port)]
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    tb_state["proc"] = proc
                    tb_state["port"] = port
                except Exception:
                    tb_state["proc"] = None
                    tb_state["port"] = None
                # Update link
                url = f"http://127.0.0.1:{tb_state['port'] or 6006}/"
                left_info_link.text = url
                left_info_prefix.visible = True
                left_info_link.visible = True
                left_info_container.visible = True
                if left_info_link.page:
                    left_info_link.update()
                if left_info_prefix.page:
                    left_info_prefix.update()
                if left_info_container.page:
                    left_info_container.update()
                # Auto-open URL
                try:
                    content.page.launch_url(url)
                except Exception:
                    pass
        except Exception:
            pass

    def on_tensorboard_delete_click(_: ft.ControlEvent | None = None):
        selected_path = selected["full"]
        selected_name = selected["display"]
        if selected_path and selected_path != "":
            try:
                # Confirm deletion
                folder_path = Path(selected_path)
                if folder_path.exists() and folder_path.is_dir():
                    # Delete the entire folder and its contents
                    shutil.rmtree(folder_path)

                    # Clear selection
                    selected["full"] = None
                    selected["display"] = ""

                    # Disable monitor button and delete button
                    monitor_btn.disabled = True
                    tensorboard_delete_btn.disabled = True
                    tensorboard_delete_btn.icon_color = ft.Colors.BLUE_GREY_400
                    tensorboard_delete_btn.style.color = ft.Colors.BLUE_GREY_400
                    if monitor_btn.page:
                        monitor_btn.update()
                    if tensorboard_delete_btn.page:
                        tensorboard_delete_btn.update()

                    # Clear tensorboard link
                    left_info_link.text = ""
                    left_info_prefix.visible = False
                    left_info_link.visible = False
                    left_info_container.visible = False
                    if left_info_link.page:
                        left_info_link.update()
                    if left_info_prefix.page:
                        left_info_prefix.update()
                    if left_info_container.page:
                        left_info_container.update()

                    # Refresh the tensorboard list
                    nonlocal items_data
                    items_data = _scan_runs(base_output_dir)
                    refresh_list()

                    # Clear any visual selection in the list
                    try:
                        for ctrl in list_view.controls:
                            if isinstance(ctrl, ft.Container):
                                ctrl.bgcolor = ft.Colors.with_opacity(0.04, ft.Colors.BLUE_GREY_50)
                                if ctrl.page:
                                    ctrl.update()
                    except Exception:
                        pass

            except Exception as e:
                # Handle deletion errors (permissions, folder not found, etc.)
                print(f"Error deleting tensorboard folder {selected_path}: {e}")
                pass

    monitor_btn = ft.ElevatedButton(
        "Monitor",
        icon=ft.Icons.MONITOR_HEART,
        disabled=True,
        on_click=on_monitor_click,
    )

    tensorboard_delete_btn = ft.TextButton(
        text="Delete",
        icon=ft.Icons.DELETE,
        on_click=on_tensorboard_delete_click,
        disabled=True,
        icon_color=ft.Colors.BLUE_GREY_400,
        style=ft.ButtonStyle(
            color=ft.Colors.BLUE_GREY_400,
        ),
    )

    def refresh_list():
        data = list(items_data)
        if sort_mode["value"] == "name":
            data.sort(key=lambda x: x[0].lower(), reverse=not sort_asc["name"])
        else:
            # date string is YYYY-MM-DD HH:MM so lexicographic works
            data.sort(key=lambda x: x[1], reverse=not sort_asc["date"])
        list_view.controls = [make_tile(n, d, full) for (n, d, full) in data]
        if list_view.page:
            list_view.update()

    # Header row inside the bordered container with sorting handlers
    def on_name_click(_: ft.ControlEvent | None = None):
        sort_mode["value"] = "name"
        sort_asc["name"] = not sort_asc["name"]
        refresh_list()

    def on_date_click(_: ft.ControlEvent | None = None):
        sort_mode["value"] = "date"
        sort_asc["date"] = not sort_asc["date"]
        refresh_list()

    header_row = ft.Container(
        content=ft.Row([
            ft.TextButton(text="Name", on_click=on_name_click),
            ft.Container(expand=True),
            ft.TextButton(text="Date", on_click=on_date_click),
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=ft.padding.symmetric(horizontal=8, vertical=6),
        border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.15, ft.Colors.OUTLINE))),
    )

    # Add a refresh button (manual rescan)
    def on_refresh(_: ft.ControlEvent | None = None):
        nonlocal items_data
        items_data = _scan_runs(base_output_dir)
        refresh_list()
        update_monitoring_from_selection(rescan=True)

    # Initial fill
    if not items_data:
        # If empty, still show placeholder entry to indicate scan ran
        items_data = [("(no runs found)", "", "")]
    refresh_list()

    refresh_btn = ft.IconButton(icon=ft.Icons.REFRESH, tooltip="Refresh", on_click=on_refresh, icon_color=ft.Colors.BLUE_GREY_600)
    header_with_refresh = ft.Row([header_row, refresh_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    tensorboard_list_container = ft.Container(
        content=ft.Column([
            header_with_refresh,
            ft.Container(
                content=list_view,
                expand=True,
                clip_behavior=ft.ClipBehavior.HARD_EDGE
            )
        ], spacing=0, expand=True),
        height=300,
        margin=ft.margin.only(top=10),
        border=ft.border.all(1, ft.Colors.OUTLINE),
        border_radius=8,
        padding=5,
    )

    tensorboard_col = ft.Column([
        *add_section_title("Tensorboard"),
        tensorboard_list_container,
        ft.Row([
            ft.Container(content=left_info_container, expand=True),
            ft.Row([
                monitor_btn,
                tensorboard_delete_btn,
            ], spacing=8),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
    ], spacing=6)

    # Bottom: Training Console (header visible, console hidden by default)
    # Style matches the console used in Tools -> Download Models
    # Hidden command area (appears when training starts)
    training_cmd_text = ft.Text(value="", selectable=True, color=ft.Colors.WHITE)
    training_cmd_container = ft.Container(
        content=training_cmd_text,
        visible=False,
        margin=ft.margin.only(top=8),
        bgcolor="#0f0f0f",
        padding=10,
        border_radius=6,
        expand=False,
    )

    training_console_text = ft.Text(spans=[], selectable=True)
    # Animated wrapper to allow smooth padding/size transitions when prepending
    training_console_anim = ft.Container(
        content=training_console_text,
        padding=ft.padding.only(top=0),
        # Ease-in-out for balanced acceleration/deceleration
        animate=ft.Animation(700, ft.AnimationCurve.EASE_IN_OUT),
    )
    # Set console reference for manual cleanup
    console_ref["training_console_text"] = training_console_text
    training_console_list = ft.ListView(
        controls=[training_console_anim],
        expand=True,
        auto_scroll=False,
        spacing=0,
        padding=0,
        height=320,
    )
    training_console_container = ft.Container(
        content=training_console_list,
        visible=False,
        margin=ft.margin.only(top=8),
        bgcolor="#0f0f0f",
        padding=10,
        border_radius=6,
        expand=True,
    )

    content = ft.Container(
        content=ft.Column([
            ft.ResponsiveRow([
                ft.Column([tensorboard_col], col=6),
                ft.Column([monitoring_col], col=6),
            ], spacing=12),
            # Full-width training console area
            # Training Console header with Clean button
            ft.Row([
                ft.Text("Training Console", size=16, weight=ft.FontWeight.BOLD, expand=True),
                ft.ElevatedButton(
                    "Clean",
                    icon=ft.Icons.CLEANING_SERVICES,
                    on_click=lambda e: console_ref["clean_console_manually"](),
                    style=ft.ButtonStyle(
                        bgcolor=ft.Colors.BLUE_GREY_700,
                        color=ft.Colors.WHITE,
                        padding=ft.padding.symmetric(horizontal=12, vertical=8)
                    ),
                    tooltip="Clear all console content"
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            # Command used (hidden until training starts)
            training_cmd_container,
            training_console_container,
        ], spacing=8, scroll=ft.ScrollMode.AUTO),
        padding=ft.padding.all(5),
        expand=True,
    )

    # Expose console controls for future use by caller
    setattr(content, "training_console_container", training_console_container)
    setattr(content, "training_console_text", training_console_text)
    setattr(content, "training_console_anim", training_console_anim)
    setattr(content, "training_console_list", training_console_list)
    setattr(content, "training_cmd_container", training_cmd_container)
    setattr(content, "training_cmd_text", training_cmd_text)

    return content
