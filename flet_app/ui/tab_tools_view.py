import flet as ft
import sys
import os
import subprocess
from pathlib import Path
from flet_app.project_root import get_project_root
from threading import Thread
from ._styles import create_textfield, add_section_title

def get_models_tab_content(page: ft.Page):
    """Entrypoint for the Models tab content. Builds and returns the tab's UI container."""
    page_controls = []

    # Download Models Section
    page_controls.extend(add_section_title("Download Models"))

    # Download options container (compact, single row)
    # Fields: Model URL/Repo, Max Connections, Download button — all in one row
    url_field = create_textfield(
        label="Model URL/Repo",
        value="",
        hint_text="https://...",
        expand=True,
    )

    max_conn_field = create_textfield(
        label="Max Connections",
        value="4",
        hint_text="e.g. 4",
        expand=0,
        width=160,
        keyboard_type=ft.KeyboardType.NUMBER,
    )

    # Match styling to the "Refresh" button used in the Models list header
    download_btn = ft.ElevatedButton(
        "Download Model",
        icon=ft.Icons.DOWNLOAD,
        width=150,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=3)
        ),
    )

    # Console area (hidden until download starts)
    # Use Text with spans (RichText not available in this Flet version)
    console_rich = ft.Text(spans=[], selectable=True)
    console_list = ft.ListView(
        controls=[console_rich],
        expand=True,
        auto_scroll=False,
        spacing=0,
        padding=0,
        height=200,
    )
    console_container = ft.Container(
        content=console_list,
        visible=False,
        margin=ft.margin.only(top=8),
        bgcolor="#0f0f0f",
        padding=10,
        border_radius=6,
        expand=True,
    )

    current_proc = None
    cancel_requested = False

    def on_download_click(e):
        nonlocal current_proc, cancel_requested
        # If a job is running, treat as Cancel
        if current_proc is not None and current_proc.poll() is None:
            cancel_requested = True
            # Inject cancel message above the 'Completed' section immediately
            try:
                spans = list(console_rich.spans or [])
                insert_at = None
                for i, sp in enumerate(spans):
                    if isinstance(sp, ft.TextSpan) and isinstance(sp.text, str) and "Completed (" in sp.text:
                        insert_at = i
                        break
                notice_span = ft.TextSpan("\n  [Action] Cancel requested. Terminating...\n", style=ft.TextStyle(color=ft.Colors.WHITE))
                if insert_at is not None:
                    spans = spans[:insert_at] + [notice_span] + spans[insert_at:]
                else:
                    spans.append(notice_span)
                console_rich.spans = spans
                console_rich.update()
            except Exception:
                pass
            try:
                current_proc.terminate()
            except Exception:
                pass
            # Fallback kill after short delay handled in reader end
            return
        src = (url_field.value or "").strip()
        if not src:
            page.snack_bar = ft.SnackBar(ft.Text("Provide a Hugging Face URL or ORG/REPO"))
            page.snack_bar.open = True
            page.update()
            return

        # Prepare console
        # Clear console
        console_rich.spans = []
        console_container.visible = True
        download_btn.text = "Cancel"
        page.update()

        # Find script path and start subprocess
        script_path = Path(__file__).resolve().parent / "print_hf_links.py"
        py = sys.executable or "python"
        max_conn = (max_conn_field.value or "4").strip()
        if not max_conn.isdigit():
            max_conn = "4"
        try:
            project_root = str(get_project_root())
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            proc = subprocess.Popen(
                [py, str(script_path), src, max_conn],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=project_root,
                env=env,
            )
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Failed to start job: {ex}"))
            page.snack_bar.open = True
            download_btn.text = "Download Model"
            page.update()
            return

        current_proc = proc
        cancel_requested = False

        def _reader():
            nonlocal current_proc
            snapshot_mode = False
            snapshot_buf: list[str] = []
            SNAP_START = "===SNAPSHOT_START==="
            SNAP_END = "===SNAPSHOT_END==="
            cancel_line_added = False

            def _render_snapshot_to_rich(snapshot_text: str):
                # Build spans; color the bar's filled cells green and empty cells grey
                spans: list[ft.TextSpan] = []
                lines = snapshot_text.splitlines()
                # If cancel was requested, inject a notice above the 'Completed' section
                nonlocal cancel_line_added
                if cancel_requested and not cancel_line_added:
                    try:
                        idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Completed ("))
                        # Insert a blank line and the notice above Completed
                        lines.insert(idx, "  [Action] Cancel requested. Terminating...")
                        lines.insert(idx, "")
                        cancel_line_added = True
                    except StopIteration:
                        # If 'Completed' not found, append at end
                        lines.append("")
                        lines.append("  [Action] Cancel requested. Terminating...")
                        cancel_line_added = True
                for ln in lines:
                    # Detect a bar line like: "  [▓▓░░...]  40% • ..."
                    import re
                    m = re.search(r"^(\s*)\[([▓░]{25})\](.*)$", ln)
                    if m:
                        prefix = m.group(1)
                        bar = m.group(2)
                        rest = m.group(3)
                        if prefix:
                            spans.append(ft.TextSpan(prefix, style=ft.TextStyle(color=ft.Colors.WHITE)) )
                        spans.append(ft.TextSpan("[", style=ft.TextStyle(color=ft.Colors.GREY_600)))
                        for ch in bar:
                            if ch == "▓":
                                spans.append(ft.TextSpan(ch, style=ft.TextStyle(color=ft.Colors.GREEN_ACCENT_700)))
                            else:
                                spans.append(ft.TextSpan(ch, style=ft.TextStyle(color=ft.Colors.GREY_700)))
                        spans.append(ft.TextSpan("]", style=ft.TextStyle(color=ft.Colors.GREY_600)))
                        if rest:
                            spans.append(ft.TextSpan(rest, style=ft.TextStyle(color=ft.Colors.WHITE)))
                        spans.append(ft.TextSpan("\n"))
                    else:
                        spans.append(ft.TextSpan(ln + "\n", style=ft.TextStyle(color=ft.Colors.WHITE)))
                console_rich.spans = spans
                try:
                    console_rich.update()
                except Exception:
                    pass
            try:
                if proc.stdout:
                    for line in proc.stdout:
                        if line.strip() == SNAP_START:
                            snapshot_mode = True
                            snapshot_buf.clear()
                            continue
                        if line.strip() == SNAP_END:
                            # Replace entire console with snapshot (rich render)
                            _render_snapshot_to_rich("".join(snapshot_buf))
                            snapshot_mode = False
                            snapshot_buf.clear()
                            continue
                        if snapshot_mode:
                            snapshot_buf.append(line)
                        else:
                            # Append plain line to rich console
                            _render_snapshot_to_rich(line)
            finally:
                try:
                    rc = proc.wait(timeout=2)
                except Exception:
                    # If still alive after terminate, kill
                    try:
                        proc.kill()
                        rc = proc.wait(timeout=2)
                    except Exception:
                        rc = -1
                # Append completion line
                console_rich.spans = list(console_rich.spans or []) + [
                    ft.TextSpan(f"\n[Done] Exit code: {rc}\n", style=ft.TextStyle(color=ft.Colors.WHITE))
                ]
                download_btn.text = "Download Model"
                
                try:
                    page.update()
                except Exception:
                    pass
                # Clear current proc reference
                current_proc = None

        Thread(target=_reader, daemon=True).start()

    download_btn.on_click = on_download_click

    download_container = ft.Container(
        content=ft.Column([
            ft.Text("Provide model URL/repo and max connections", size=14),
            ft.Row([
                url_field,
                max_conn_field,
                download_btn,
            ], spacing=10, alignment=ft.MainAxisAlignment.START),
            console_container,
        ], spacing=6),
        padding=15,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_300),
        border_radius=8,
        bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.WHITE)
    )

    page_controls.append(download_container)

    # Separator
    page_controls.append(ft.Divider(height=30, color=ft.Colors.with_opacity(0.2, ft.Colors.ON_SURFACE)))

    # Models on Disk Section
    page_controls.extend(add_section_title("Models on Disk"))

    # Sorting state and helpers (reuse logic style from Open/Save dialogs)
    sort_mode = "name"  # "name" | "size" | "date"
    name_sort_ascending = True
    size_sort_ascending = False
    date_sort_ascending = False

    def sort_entries(entries, mode, ascending):
        if mode == "name":
            return sorted(entries, key=lambda x: str(x[0]).lower(), reverse=not ascending)
        if mode == "size":
            return sorted(entries, key=lambda x: x[3], reverse=not ascending)
        if mode == "date":
            return sorted(entries, key=lambda x: x[2], reverse=not ascending)
        return entries

    # Paths and loader
    from pathlib import Path
    def get_models_root() -> Path:
        # This file is flet_app/ui/tab_tools_view.py -> parents[1] is flet_app/, parents[2] is repo root
        # Requirement: go up one dir from flet_app.py (flet_app/), then into 'models'
        return get_project_root() / "models"

    def _dir_size_bytes(path: Path) -> int:
        try:
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except Exception:
                        pass
            return total
        except Exception:
            return 0

    def load_models():
        try:
            base_dir = get_models_root()
            items = []
            if base_dir.exists() and base_dir.is_dir():
                for child in base_dir.iterdir():
                    if child.is_dir():
                        # Skip special folders
                        if child.name == "_misc":
                            continue
                        rel_name = child.name
                        mod_time = child.stat().st_mtime
                        size_bytes = _dir_size_bytes(child)
                        items.append((rel_name, child, mod_time, size_bytes))
                # default sort A-Z by name
                items.sort(key=lambda x: str(x[0]).lower())
            return items
        except Exception:
            return []

    # UI containers
    list_controls: list[ft.Control] = []

    list_container = ft.Container(
        content=ft.ListView(
            controls=list_controls,
            spacing=2,
            expand=True,
            auto_scroll=False,
        ),
        height=260,
        expand=True,
        margin=ft.margin.only(top=10),
        border=ft.border.all(1, ft.Colors.BLUE_GREY_200),
        border_radius=6,
        padding=8,
    )

    # Actions
    def toggle_name_sort(_=None):
        nonlocal sort_mode, name_sort_ascending
        sort_mode = "name"
        name_sort_ascending = not name_sort_ascending
        refresh_list()

    def toggle_size_sort(_=None):
        nonlocal sort_mode, size_sort_ascending
        sort_mode = "size"
        size_sort_ascending = not size_sort_ascending
        refresh_list()

    def toggle_date_sort(_=None):
        nonlocal sort_mode, date_sort_ascending
        sort_mode = "date"
        date_sort_ascending = not date_sort_ascending
        refresh_list()

    def refresh_list(_=None):
        from datetime import datetime
        entries = load_models()
        if sort_mode == "name":
            entries = sort_entries(entries, "name", name_sort_ascending)
        elif sort_mode == "size":
            entries = sort_entries(entries, "size", size_sort_ascending)
        elif sort_mode == "date":
            entries = sort_entries(entries, "date", date_sort_ascending)

        list_controls.clear()

        # Header: Name | Date | Refresh (same row)
        name_sort_text = (
            "Name ↓" if sort_mode == "name" and not name_sort_ascending
            else "Name ↑" if sort_mode == "name" and name_sort_ascending
            else "Name"
        )
        size_sort_text = (
            "Size ↓" if sort_mode == "size" and not size_sort_ascending
            else "Size ↑" if sort_mode == "size" and size_sort_ascending
            else "Size"
        )
        date_sort_text = (
            "Date ↓" if sort_mode == "date" and not date_sort_ascending
            else "Date ↑" if sort_mode == "date" and date_sort_ascending
            else "Date"
        )

        header = ft.Row([
            ft.Container(
                content=ft.Row([
                    ft.Text(
                        value=name_sort_text,
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_300 if sort_mode == "name" else ft.Colors.GREY_300,
                    ),
                    ft.Icon(
                        ft.Icons.SORT if sort_mode != "name" else (ft.Icons.ARROW_UPWARD if name_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                        size=14,
                        color=ft.Colors.BLUE_300 if sort_mode == "name" else ft.Colors.GREY_400,
                    ),
                ], spacing=5),
                expand=True,
                height=30,
                bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if sort_mode == "name" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                border_radius=4,
                padding=ft.padding.all(0),
                on_click=toggle_name_sort,
            ),
            ft.Container(
                content=ft.Row([
                    ft.Text(
                        value=size_sort_text,
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_300 if sort_mode == "size" else ft.Colors.GREY_300,
                    ),
                    ft.Icon(
                        ft.Icons.SORT if sort_mode != "size" else (ft.Icons.ARROW_UPWARD if size_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                        size=14,
                        color=ft.Colors.BLUE_300 if sort_mode == "size" else ft.Colors.GREY_400,
                    ),
                ], spacing=5, alignment=ft.MainAxisAlignment.END),
                width=90,
                height=30,
                bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if sort_mode == "size" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                border_radius=4,
                padding=ft.padding.all(0),
                on_click=toggle_size_sort,
            ),
            ft.Container(
                content=ft.Row([
                    ft.Text(
                        value=date_sort_text,
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_300 if sort_mode == "date" else ft.Colors.GREY_300,
                    ),
                    ft.Icon(
                        ft.Icons.SORT if sort_mode != "date" else (ft.Icons.ARROW_UPWARD if date_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                        size=14,
                        color=ft.Colors.BLUE_300 if sort_mode == "date" else ft.Colors.GREY_400,
                    ),
                ], spacing=5, alignment=ft.MainAxisAlignment.END),
                width=160,
                height=30,
                bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if sort_mode == "date" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                border_radius=4,
                padding=ft.padding.all(0),
                on_click=toggle_date_sort,
            ),
        ], alignment=ft.MainAxisAlignment.START, spacing=10)

        list_controls.append(header)

        if entries:
            def _format_size(n: int) -> str:
                units = ["B", "KB", "MB", "GB", "TB"]
                size = float(n)
                i = 0
                while size >= 1024 and i < len(units) - 1:
                    size /= 1024
                    i += 1
                return f"{int(size)} {units[i]}" if i == 0 else f"{size:.1f} {units[i]}"

            for name, path, mtime, size_bytes in entries:
                mod_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                size_text = _format_size(size_bytes)
                row = ft.Row([
                    ft.Text(value=str(name), expand=True),
                    ft.Text(value=size_text, width=90, text_align=ft.TextAlign.RIGHT, color=ft.Colors.GREY_600),
                    ft.Text(value=mod_date, width=160, text_align=ft.TextAlign.RIGHT, color=ft.Colors.GREY_600),
                ], spacing=10)
                list_controls.append(row)
        else:
            list_controls.append(ft.Text("No model folders found in 'models'", italic=True, color=ft.Colors.GREY_600))

        list_container.content = ft.ListView(
            controls=list_controls,
            spacing=2,
            expand=True,
            auto_scroll=False,
        )
        try:
            if list_container.page:
                list_container.update()
        except Exception:
            pass

    # Build models container with top-right refresh button
    refresh_btn_top = ft.ElevatedButton(
        "Refresh",
        icon=ft.Icons.REFRESH,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=3)),
        on_click=refresh_list,
    )

    title_row = ft.Row([
        ft.Text("Manage models stored on your local disk", size=14),
        refresh_btn_top,
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    models_container = ft.Container(
        content=ft.Column([
            title_row,
            list_container,
        ], spacing=5, expand=True),
        padding=15,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_300),
        border_radius=8,
        bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
        expand=60,
    )

    # Initial population for left panel
    refresh_list()

    # ----- Curated Models Panel (right column) -----
    curated_sort_mode = "name"  # "name" | "size"
    curated_name_sort_ascending = True
    curated_size_sort_ascending = False

    def curated_sort_entries(entries, mode, ascending):
        if mode == "name":
            return sorted(entries, key=lambda x: str(x[0]).lower(), reverse=not ascending)
        if mode == "size":
            return sorted(entries, key=lambda x: x[2], reverse=not ascending)
        return entries

    curated_list_controls: list[ft.Control] = []
    curated_list_container = ft.Container(
        content=ft.ListView(
            controls=curated_list_controls,
            spacing=2,
            expand=True,
            auto_scroll=False,
        ),
        height=260,
        expand=True,
        margin=ft.margin.only(top=10),
        border=ft.border.all(1, ft.Colors.BLUE_GREY_200),
        border_radius=6,
        padding=8,
    )

    def _format_size(n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(n)
        i = 0
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        return f"{int(size)} {units[i]}" if i == 0 else f"{size:.1f} {units[i]}"

    def curated_refresh_list(_=None):
        base_dir = get_models_root()
        targets = [
            ("minimax-remover", base_dir / "_misc" / "minimax-remover"),
            ("LLaVA-NeXT-Video-7B-hf", base_dir / "_misc" / "LLaVA-NeXT-Video-7B-hf"),
            ("Qwen3-VL-8B-Instruct", base_dir / "_misc" / "Qwen3-VL-8B-Instruct"),
            ("Qwen3-VL-4B-Instruct", base_dir / "_misc" / "Qwen3-VL-4B-Instruct"),
            ("joycaption-llava", base_dir / "_misc" / "joycaption-llava"),
        ]
        rows = []  # (name, exists, size_bytes)
        for name, path in targets:
            exists = path.exists() and path.is_dir()
            size_bytes = _dir_size_bytes(path) if exists else 0
            rows.append((name, exists, size_bytes))

        # Sort
        if curated_sort_mode == "name":
            rows = curated_sort_entries([(n, e, s) for (n, e, s) in rows], "name", curated_name_sort_ascending)
        elif curated_sort_mode == "size":
            rows = curated_sort_entries([(n, e, s) for (n, e, s) in rows], "size", curated_size_sort_ascending)

        curated_list_controls.clear()

        name_sort_text = (
            "Name ↓" if curated_sort_mode == "name" and not curated_name_sort_ascending
            else "Name ↑" if curated_sort_mode == "name" and curated_name_sort_ascending
            else "Name"
        )
        size_sort_text = (
            "Size ↓" if curated_sort_mode == "size" and not curated_size_sort_ascending
            else "Size ↑" if curated_sort_mode == "size" and curated_size_sort_ascending
            else "Size"
        )

        def toggle_curated_name_sort(_=None):
            nonlocal curated_sort_mode, curated_name_sort_ascending
            curated_sort_mode = "name"
            curated_name_sort_ascending = not curated_name_sort_ascending
            curated_refresh_list()

        def toggle_curated_size_sort(_=None):
            nonlocal curated_sort_mode, curated_size_sort_ascending
            curated_sort_mode = "size"
            curated_size_sort_ascending = not curated_size_sort_ascending
            curated_refresh_list()

        header = ft.Row([
            ft.Container(
                content=ft.Row([
                    ft.Text(
                        value=name_sort_text,
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_300 if curated_sort_mode == "name" else ft.Colors.GREY_300,
                    ),
                    ft.Icon(
                        ft.Icons.SORT if curated_sort_mode != "name" else (ft.Icons.ARROW_UPWARD if curated_name_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                        size=14,
                        color=ft.Colors.BLUE_300 if curated_sort_mode == "name" else ft.Colors.GREY_400,
                    ),
                ], spacing=5),
                expand=True,
                height=30,
                bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if curated_sort_mode == "name" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                border_radius=4,
                padding=ft.padding.symmetric(horizontal=10, vertical=5),
                on_click=toggle_curated_name_sort,
            ),
            ft.Container(
                content=ft.Row([
                    ft.Text(
                        value=size_sort_text,
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.BLUE_300 if curated_sort_mode == "size" else ft.Colors.GREY_300,
                    ),
                    ft.Icon(
                        ft.Icons.SORT if curated_sort_mode != "size" else (ft.Icons.ARROW_UPWARD if curated_size_sort_ascending else ft.Icons.ARROW_DOWNWARD),
                        size=14,
                        color=ft.Colors.BLUE_300 if curated_sort_mode == "size" else ft.Colors.GREY_400,
                    ),
                ], spacing=5, alignment=ft.MainAxisAlignment.END),
                width=90,
                height=30,
                bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.BLUE_900) if curated_sort_mode == "size" else ft.Colors.with_opacity(0.2, ft.Colors.GREY_800),
                border_radius=4,
                padding=ft.padding.all(0),
                on_click=toggle_curated_size_sort,
            ),
            ft.Container(
                content=ft.Text("Exists", size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300),
                width=60,
                height=30,
                alignment=ft.alignment.center,
            ),
        ], alignment=ft.MainAxisAlignment.START, spacing=10)

        curated_list_controls.append(header)

        def start_curated_download(model_name: str):
            nonlocal current_proc, cancel_requested
            # Do not start if another job is running
            if current_proc is not None and current_proc.poll() is None:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text("Another job is running. Cancel it first."))
                    page.snack_bar.open = True
                    page.update()
                except Exception:
                    pass
                return

            # Show console
            console_container.visible = True
            try:
                page.update()
            except Exception:
                pass

            # Determine source based on the curated model name
            src = None
            if model_name == "LLaVA-NeXT-Video-7B-hf":
                src = "llava-hf/LLaVA-NeXT-Video-7B-hf"
            elif model_name == "Qwen3-VL-8B-Instruct":
                src = "Qwen/Qwen3-VL-8B-Instruct"
            elif model_name == "Qwen3-VL-4B-Instruct":
                src = "Qwen/Qwen3-VL-4B-Instruct"
            elif model_name == "joycaption-llava":
                src = "fancyfeast/llama-joycaption-beta-one-hf-llava"
            elif model_name == "minimax-remover":
                # Not downloadable via HF here; show a message and return
                try:
                    page.snack_bar = ft.SnackBar(ft.Text("No direct download configured for minimax-remover."))
                    page.snack_bar.open = True
                    page.update()
                except Exception:
                    pass
                return

            if not src:
                return

            # Launch the existing print_hf_links.py to stream progress snapshots
            from pathlib import Path as _Path
            py = sys.executable or "python"
            script_path = _Path(__file__).resolve().parent / "print_hf_links.py"
            # Direct into models/_misc/<repo_name>
            subfolder = None
            if model_name == "LLaVA-NeXT-Video-7B-hf":
                subfolder = "_misc/LLaVA-NeXT-Video-7B-hf"
            elif model_name == "Qwen3-VL-8B-Instruct":
                subfolder = "_misc/Qwen3-VL-8B-Instruct"
            elif model_name == "Qwen3-VL-4B-Instruct":
                subfolder = "_misc/Qwen3-VL-4B-Instruct"
            elif model_name == "joycaption-llava":
                subfolder = "_misc/joycaption-llava"
            cmd = [str(py), str(script_path), src]
            if subfolder:
                # Pass both max connections and target subfolder
                cmd += ["4", subfolder]

            try:
                project_root = str(get_project_root())
                env = os.environ.copy()
                env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=project_root,
                    env=env,
                )
            except Exception as ex:
                try:
                    page.snack_bar = ft.SnackBar(ft.Text(f"Failed to start job: {ex}"))
                    page.snack_bar.open = True
                    page.update()
                except Exception:
                    pass
                return

            current_proc = proc
            cancel_requested = False

            def _reader2():
                nonlocal current_proc
                # Snapshot parsing like the main downloader
                SNAP_START = "===SNAPSHOT_START==="
                SNAP_END = "===SNAPSHOT_END==="
                snapshot_mode = False
                snapshot_buf: list[str] = []

                def _render_snapshot_to_rich(snapshot_text: str):
                    import re as _re
                    spans: list[ft.TextSpan] = []
                    lines = snapshot_text.splitlines()
                    for ln in lines:
                        # Match lines starting with an optional prefix, a 25-cell progress bar, and the rest
                        m = _re.search(r"^(\s*)\[([▓░]{25})\](.*)$", ln)
                        if m:
                            prefix = m.group(1)
                            bar = m.group(2)
                            rest = m.group(3)
                            if prefix:
                                spans.append(ft.TextSpan(prefix, style=ft.TextStyle(color=ft.Colors.WHITE)))
                            spans.append(ft.TextSpan("[", style=ft.TextStyle(color=ft.Colors.GREY_600)))
                            for ch in bar:
                                spans.append(ft.TextSpan(ch, style=ft.TextStyle(color=(ft.Colors.GREEN_ACCENT_700 if ch == "▓" else ft.Colors.GREY_700))))
                            spans.append(ft.TextSpan("]", style=ft.TextStyle(color=ft.Colors.GREY_600)))
                            if rest:
                                spans.append(ft.TextSpan(rest, style=ft.TextStyle(color=ft.Colors.WHITE)))
                            spans.append(ft.TextSpan("\n"))
                        else:
                            spans.append(ft.TextSpan(ln + "\n", style=ft.TextStyle(color=ft.Colors.WHITE)))
                    console_rich.spans = spans
                    try:
                        console_rich.update()
                    except Exception:
                        pass

                try:
                    if proc.stdout:
                        for line in proc.stdout:
                            if line.strip() == SNAP_START:
                                snapshot_mode = True
                                snapshot_buf.clear()
                                continue
                            if line.strip() == SNAP_END:
                                _render_snapshot_to_rich("".join(snapshot_buf))
                                snapshot_mode = False
                                snapshot_buf.clear()
                                continue
                            if snapshot_mode:
                                snapshot_buf.append(line)
                            else:
                                # Fallback: append raw line
                                spans = list(console_rich.spans or [])
                                spans.append(ft.TextSpan(line, style=ft.TextStyle(color=ft.Colors.WHITE)))
                                console_rich.spans = spans
                                try:
                                    console_rich.update()
                                except Exception:
                                    pass
                finally:
                    try:
                        rc = proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                            rc = proc.wait(timeout=2)
                        except Exception:
                            rc = -1
                    console_rich.spans = list(console_rich.spans or []) + [ft.TextSpan(f"\n[Done] Exit code: {rc}\n", style=ft.TextStyle(color=ft.Colors.WHITE))]
                    try:
                        page.update()
                    except Exception:
                        pass
                    current_proc = None

            Thread(target=_reader2, daemon=True).start()

        for name, exists, size_bytes in rows:
            size_text = _format_size(size_bytes)

            if exists:
                # Right-click on "Yes" to re-download missing parts (gap-fill)
                exists_control = ft.Container(
                    content=ft.GestureDetector(
                        content=ft.Text(
                            value="Yes",
                            color=ft.Colors.GREEN,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        on_secondary_tap=(lambda e, m=name: start_curated_download(m)),
                        mouse_cursor=ft.MouseCursor.CLICK,
                    ),
                    width=60,
                    alignment=ft.alignment.center,
                    tooltip="Right-click to re-download missing files",
                )
            else:
                exists_control = ft.Container(
                    content=ft.TextButton(
                        "Get",
                        style=ft.ButtonStyle(
                            color={ft.ControlState.DEFAULT: ft.Colors.RED},
                            padding=0,
                        ),
                        on_click=(lambda e, m=name: start_curated_download(m)),
                    ),
                    width=60,
                    alignment=ft.alignment.center,
                )

            row = ft.Row([
                ft.Text(value=str(name), expand=True),
                ft.Text(value=size_text, width=90, text_align=ft.TextAlign.RIGHT, color=ft.Colors.GREY_600),
                exists_control,
            ], spacing=10)
            curated_list_controls.append(row)

        curated_list_container.content = ft.ListView(
            controls=curated_list_controls,
            spacing=2,
            expand=True,
            auto_scroll=False,
        )
        try:
            if curated_list_container.page:
                curated_list_container.update()
        except Exception:
            pass

    curated_refresh_btn_top = ft.ElevatedButton(
        "Refresh",
        icon=ft.Icons.REFRESH,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=3)),
        on_click=curated_refresh_list,
    )
    curated_title_row = ft.Row([
        ft.Text("Curated models (quick check)", size=14),
        curated_refresh_btn_top,
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

    curated_container = ft.Container(
        content=ft.Column([
            curated_title_row,
            curated_list_container,
        ], spacing=5, expand=True),
        padding=15,
        border=ft.border.all(1, ft.Colors.BLUE_GREY_300),
        border_radius=8,
        bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
        expand=40,
    )

    # Initial population for right panel
    curated_refresh_list()

    # Place the two panels side by side
    panels_row = ft.Row([
        models_container,
        curated_container,
    ], spacing=10)

    page_controls.append(panels_row)

    return ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True,
        padding=ft.padding.all(10)
    )
