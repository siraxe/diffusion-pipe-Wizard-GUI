"""
Process Cleanup Utility

Kills existing training processes before starting new ones to prevent zombie processes.
"""

import os
import signal
import subprocess
from loguru import logger


def kill_existing_training_processes() -> int:
    """
    Kill any existing training processes to prevent zombies.

    Searches for and terminates processes running:
    - diffusion-trainers/diffusion-pipe/train.py
    - diffusion-trainers/LTX-2/packages/ltx-trainer/scripts/train.py
    - deepspeed with train.py
    - process_dataset.py

    Returns:
        Number of processes killed.
    """
    killed_count = 0

    # Try using psutil first (more reliable)
    try:
        import psutil

        current_pid = os.getpid()

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                if proc.info['pid'] == current_pid:
                    continue

                cmdline = proc.info.get('cmdline', [])
                if not cmdline:
                    continue

                cmdline_str = ' '.join(cmdline)

                # Kill training-related processes
                kill_patterns = [
                    'diffusion-pipe/train.py',
                    'ltx-trainer/scripts/train.py',
                    'process_dataset.py',
                ]

                if any(pattern in cmdline_str for pattern in kill_patterns):
                    _terminate_process(proc.info['pid'], cmdline_str)
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    except ImportError:
        # Fallback without psutil - use pgrep/pkill on Unix
        killed_count = _kill_without_psutil()

    if killed_count > 0:
        logger.info(f"[Cleanup] Killed {killed_count} existing training process(es)")
        # Give processes time to terminate
        import time
        time.sleep(1)

    return killed_count


def _terminate_process(pid: int, cmdline_str: str):
    """Terminate a single process with fallback methods."""
    try:
        if os.name == 'posix':
            # Try to kill process group first
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Fallback to single process
                os.kill(pid, signal.SIGTERM)
        else:
            # Windows
            import psutil
            proc = psutil.Process(pid)
            proc.terminate()

        logger.debug(f"[Cleanup] Killed training process PID {pid}: {cmdline_str[:60]}...")
    except Exception as e:
        # Force kill as last resort
        try:
            if os.name == 'posix':
                os.kill(pid, signal.SIGKILL)
            else:
                import psutil
                proc = psutil.Process(pid)
                proc.kill()
        except Exception:
            pass


def _kill_without_psutil() -> int:
    """Fallback cleanup method without psutil."""
    killed_count = 0

    if os.name != 'posix':
        return killed_count  # Only Unix support without psutil

    # Try pgrep/pkill
    try:
        # Pattern for finding training processes
        patterns = [
            'diffusion-pipe/train.py',
            'ltx-trainer/scripts/train.py',
            'process_dataset.py',
        ]

        for pattern in patterns:
            try:
                # Find PIDs matching the pattern
                result = subprocess.run(
                    ['pgrep', '-f', pattern],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    for pid_str in pids:
                        if pid_str:
                            try:
                                pid = int(pid_str)
                                os.kill(pid, signal.SIGTERM)
                                killed_count += 1
                                logger.debug(f"[Cleanup] Killed training process PID {pid}")
                            except (ProcessLookupError, PermissionError, ValueError):
                                pass
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    except Exception as e:
        logger.debug(f"[Cleanup] Fallback method error: {e}")

    return killed_count
