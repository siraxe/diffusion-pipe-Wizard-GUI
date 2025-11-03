import sys
import os
import shutil
import subprocess
from typing import Optional, List
import re
import time

# When executed as a script, the script directory is on sys.path,
# so we can import the helper directly.
from model_repo_fetch import prepare_hf_file_entries


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: print_hf_links.py <HF_URL_OR_ORG/REPO> [max_connections] [target_subfolder]", flush=True)
        return 2
    src = argv[1].strip()
    max_conn: Optional[int] = None
    target_subfolder: Optional[str] = None
    if len(argv) >= 3:
        try:
            max_conn = int(argv[2])
        except Exception:
            # If not an int, treat it as subfolder and leave max_conn as None
            max_conn = None
            target_subfolder = argv[2].strip() or None
    if len(argv) >= 4:
        target_subfolder = argv[3].strip() or None

    print(f"[Job] Source: {src}", flush=True)
    if max_conn is not None:
        print(f"[Job] Max connections: {max_conn}", flush=True)

    try:
        tree_url, target_dir, entries = prepare_hf_file_entries(src, target_subfolder)
    except Exception as ex:
        print(f"[Error] {ex}", flush=True)
        return 1

    total = len(entries)
    # Snapshot protocol to keep UI clean: print snapshots that overwrite in the UI
    SNAP_START = "===SNAPSHOT_START==="
    SNAP_END = "===SNAPSHOT_END==="

    def render_snapshot(completed: List[str], current: Optional[str], pending: List[str], progress_line: Optional[str] = None, overall_speed: Optional[str] = None) -> str:
        lines: List[str] = []
        lines.append(f"Tree: {tree_url}")
        lines.append(f"Target: {target_dir}")
        lines.append(f"Files: {total}")
        lines.append("")
        # Downloading section always on top, fixed position
        lines.append("Downloading:")
        lines.append(f"  ⇣ {current}" if current else "  -")
        if progress_line:
            lines.append(f"  {progress_line}")
        # Overall speed not shown per request
        lines.append("")
        # Completed next
        lines.append(f"Completed ({len(completed)}/{total}):")
        for p in completed:
            lines.append(f"  ✓ {p}")
        # Pending last
        if pending:
            lines.append("")
            lines.append(f"Pending ({len(pending)}):")
            for p in pending:
                lines.append(f"  • {p}")
        return "\n".join(lines)

    def print_snapshot(completed: List[str], current: Optional[str], pending: List[str], progress_line: Optional[str] = None, overall_speed: Optional[str] = None) -> None:
        print(SNAP_START, flush=True)
        print(render_snapshot(completed, current, pending, progress_line, overall_speed), flush=True)
        print(SNAP_END, flush=True)

    # Start aria2c download after printing list
    aria = shutil.which("aria2c")
    if aria is None:
        print("[Warn] aria2c not found in PATH. Skipping downloads.", flush=True)
        # Show a snapshot anyway
        completed: List[str] = []
        pending: List[str] = [rel for rel, _ in entries]
        print_snapshot(completed, None, pending)
        return 0

    # Ensure all subdirectories exist first and filter out already-present files
    filtered_entries = []
    for rel, url in entries:
        subdir = os.path.join(str(target_dir), os.path.dirname(rel))
        if subdir and not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(str(target_dir), rel)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            # Skip existing non-empty file (gap-fill only missing)
            continue
        filtered_entries.append((rel, url))

    total = len(filtered_entries)
    print(f"[Job] Starting aria2c downloads ({total} files)...", flush=True)
    # Use per-file invocation to preserve relative paths
    connections = max(1, int(max_conn)) if max_conn is not None else 4
    completed: List[str] = []
    pending: List[str] = [rel for rel, _ in filtered_entries]
    current: Optional[str] = None

    # Initial snapshot
    print_snapshot(completed, None, pending)

    for idx, (rel, url) in enumerate(filtered_entries, start=1):
        out_dir = os.path.join(str(target_dir), os.path.dirname(rel))
        out_name = os.path.basename(rel)
        current = rel
        if rel in pending:
            pending.remove(rel)
        # progress line state
        progress_line: Optional[str] = None
        print_snapshot(completed, current, pending, progress_line, None)
        cmd = [
            aria,
            "-x", str(connections),
            "-s", str(connections),
            "--summary-interval=1",
            "--console-log-level=notice",
            "--continue=true",
            "--auto-file-renaming=false",
            "-d", out_dir if out_dir else str(target_dir),
            "-o", out_name,
            url,
        ]
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
                last_snap = 0.0
                last_speed_txt: Optional[str] = None
                if p.stdout:
                    for line in p.stdout:
                        # Parse progress from aria2c line if available
                        try:
                            # Try percent
                            m_pct = re.search(r"(\d{1,3})%", line)
                            percent = int(m_pct.group(1)) if m_pct else None
                            # Try sizes: cur/total with units like MiB/GiB/KiB/B
                            m_sz = re.search(r"([0-9]+(?:\.[0-9]+)?)([KMG]i?)?B/([0-9]+(?:\.[0-9]+)?)([KMG]i?)?B", line)
                            cur_txt = tot_txt = None
                            if m_sz:
                                cur_txt = f"{m_sz.group(1)}{m_sz.group(2) or ''}B"
                                tot_txt = f"{m_sz.group(3)}{m_sz.group(4) or ''}B"
                            # Try speed (several formats)
                            speed_txt = None
                            # Format 1: in parentheses e.g. "(12.3MiB/s)"
                            m_sp = re.search(r"\(([0-9.]+[KMG]i?B/s)\)", line)
                            if m_sp:
                                speed_txt = m_sp.group(1)
                            else:
                                # Format 2: aria2 notice style e.g. "DL:12.3MiB"
                                m_sp2 = re.search(r"DL:([0-9.]+[KMG]i?B)(?:/s)?", line)
                                if m_sp2:
                                    # Normalize to /s suffix for display
                                    val = m_sp2.group(1)
                                    speed_txt = val if val.endswith("/s") else f"{val}/s"
                                else:
                                    # Format 3: bare token like "12.3MiB/s"
                                    m_sp3 = re.search(r"\s([0-9.]+[KMG]i?B/s)\s", line)
                                    if m_sp3:
                                        speed_txt = m_sp3.group(1)
                            # Try ETA
                            m_eta = re.search(r"ETA:?\s*([0-9hms:]+)", line)
                            eta_txt = m_eta.group(1) if m_eta else None

                            if percent is not None:
                                # 25-cell bar, each cell represents ~4%
                                bar_len = 25
                                filled = max(0, min(bar_len, percent // 4))
                                bar = f"[{'▓'*filled}{'░'*(bar_len-filled)}] {percent}%"
                            else:
                                bar = "[░░░░░░░░░░░░░░░░░░░░░░░░░] --%"

                            extra = []
                            if cur_txt and tot_txt:
                                extra.append(f"{cur_txt}/{tot_txt}")
                            if speed_txt:
                                extra.append(speed_txt)
                                last_speed_txt = speed_txt
                            if eta_txt:
                                extra.append(f"ETA {eta_txt}")
                            progress_line = f"{bar}  {' • '.join(extra) if extra else ''}".rstrip()

                            now = time.time()
                            if now - last_snap >= 3.0:
                                print_snapshot(completed, current, pending, progress_line, last_speed_txt)
                                last_snap = now
                        except Exception:
                            # Ignore parse errors
                            pass
                rc = p.wait()
            if rc != 0:
                print(f"[aria2c] Non-zero exit ({rc}) for {rel}", flush=True)
        except Exception as ex:
            print(f"[aria2c] Failed for {rel}: {ex}", flush=True)
        finally:
            completed.append(rel)
            current = None
            # Final snapshot update for this file, clear progress line
            print_snapshot(completed, current, pending, None, None)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
