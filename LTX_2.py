"""LTX-2 model handler - processes ltx-video-2 model configurations"""
import toml
import os
import subprocess
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor


def _build_ltx2_pythonpath(current_dir: str, existing_pythonpath: str = '') -> str:
    """Build PYTHONPATH with LTX-2 package src directories."""
    ltx2_base = os.path.join(current_dir, "diffusion-trainers/LTX-2/packages")
    paths = [
        os.path.join(ltx2_base, "ltx-core/src"),
        os.path.join(ltx2_base, "ltx-pipelines/src"),
        os.path.join(ltx2_base, "ltx-trainer/src"),
    ]
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    return ":".join(paths)


def _get_popen_kwargs(cwd: str, env: dict) -> dict:
    """Configure subprocess.Popen kwargs with platform-specific settings."""
    kwargs = {
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    try:
        if os.name == 'posix':
            kwargs["preexec_fn"] = os.setsid
        elif os.name == 'nt':
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    except Exception:
        pass
    return kwargs


def _build_resolution_buckets(resolutions: list, frame_buckets: list) -> str:
    """Build resolution-buckets string from resolutions and frame buckets."""
    buckets = []
    if resolutions and frame_buckets:
        for res in resolutions:
            for fb in frame_buckets:
                buckets.append(f"{res}x{res}x{fb}")
    return ";".join(buckets) if buckets else "512x512x33"


def _extract_config_data(result: dict) -> tuple:
    """Extract and return configuration data from conversion result."""
    return (
        result.get('frame_buckets', [1, 33, 76]),
        result.get('resolutions', []),
        result.get('preprocessed_data_root'),
        result.get('model_path'),
        result.get('text_encoder_path'),
        result.get('yaml_path'),
        result.get('num_repeats', 1),
    )


def _print_config_info(yaml_path: str, preprocessed_data_root: str, frame_buckets: list, resolutions: list):
    """Print configuration information."""
    print(f"LTX2 - Converted config to: {yaml_path}")
    print(f"\ndata:")
    print(f"  preprocessed_data_root: {preprocessed_data_root}")
    print(f"\nDataset configuration:")
    print(f"  frame_buckets: {frame_buckets if frame_buckets else 'Not found'}")
    print(f"  resolutions: {resolutions if resolutions else 'Not found'}")


def _print_command(cmd_str: str, title: str = "Command"):
    """Print formatted command."""
    print("\n" + "="*80)
    print(f"{title}:")
    print("="*80)
    print(cmd_str)
    print("="*80 + "\n")


async def run_process_dataset(config_path: str):
    """Run process_dataset.py for LTX-2 model. Returns process immediately for streaming."""
    def _run():
        from flet_app.ui.utils.toml_to_yaml import convert_toml_to_ltx2_yaml
        result = convert_toml_to_ltx2_yaml(config_path)

        frame_buckets, resolutions, preprocessed_data_root, model_path, text_encoder_path, yaml_path, num_repeats = _extract_config_data(result)
        _print_config_info(yaml_path, preprocessed_data_root, frame_buckets, resolutions)

        dataset_json = os.path.join(preprocessed_data_root, 'captions.json')
        resolution_buckets_str = _build_resolution_buckets(resolutions, frame_buckets)
        current_dir = os.getcwd()
        script_path = os.path.join(current_dir, "diffusion-trainers/LTX-2/packages/ltx-trainer/scripts/process_dataset.py")

        # Add -u flag for unbuffered output
        cmd = [
            "python", "-u",
            script_path,
            dataset_json,
            "--resolution-buckets",
            resolution_buckets_str,
            "--model-path",
            model_path,
            "--text-encoder-path",
            text_encoder_path,
            "--batch-size",
            str(num_repeats),
        ]
        cmd_str = " ".join(cmd)
        _print_command(cmd_str, "Process Dataset Command")
        print("LTX2")

        env = os.environ.copy()
        env['PYTHONPATH'] = _build_ltx2_pythonpath(current_dir, env.get('PYTHONPATH', ''))
        env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
        popen_kwargs = _get_popen_kwargs(current_dir, env)

        # Create process but DON'T wait - let caller handle streaming
        proc = subprocess.Popen(cmd, **popen_kwargs)
        gc.collect()
        return proc, cmd_str

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        proc, cmd_str = await loop.run_in_executor(pool, _run)
    return proc, cmd_str


async def run_ltx_training(yaml_config_path: str, cache_only: bool = False):
    """Run LTX-2 training script with Python."""
    def _run():
        import sys
        current_dir = os.getcwd()
        python_exe = sys.executable or "python"
        script_path = os.path.join(current_dir, "diffusion-trainers/LTX-2/packages/ltx-trainer/scripts/train.py")

        # Add -u flag for unbuffered output and --disable-progress-bars for text logging
        cmd = [python_exe, "-u", script_path, yaml_config_path, "--disable-progress-bars"]
        cmd_str = " ".join(cmd)
        _print_command(cmd_str, "LTX-2 Training Command")

        env = os.environ.copy()
        env['PYTHONPATH'] = _build_ltx2_pythonpath(current_dir, env.get('PYTHONPATH', ''))
        env['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
        popen_kwargs = _get_popen_kwargs(current_dir, env)

        # Use PIPE for stdout to enable streaming output
        popen_kwargs['stdout'] = subprocess.PIPE
        popen_kwargs['stderr'] = subprocess.STDOUT

        proc = subprocess.Popen(cmd, **popen_kwargs)
        gc.collect()
        return proc, cmd_str

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        proc, cmd_str = await loop.run_in_executor(pool, _run)
    return proc, cmd_str


def _is_ltx2_in_content(content: str) -> bool:
    """Check if ltx-video-2 is present in file content."""
    return 'ltx-video-2' in content.lower()


def _is_ltx2_in_toml(content: str) -> bool:
    """Check if model type is ltx-video-2 via TOML parsing."""
    try:
        config = toml.loads(content)
        model_type = config.get('model', {}).get('type', '')
        return str(model_type).strip().lower() == 'ltx-video-2'
    except Exception as e:
        print(f"[DEBUG] TOML parsing failed: {e}")
        return False


def _read_config_file(config_path: str) -> str:
    """Read configuration file content."""
    with open(config_path, 'r') as f:
        return f.read()


def handle_ltx_model(config_path):
    """Check if model type is ltx-video-2."""
    try:
        content = _read_config_file(config_path)

        if _is_ltx2_in_content(content):
            print(f"[DEBUG] Detected LTX-2 model in config")
            return True

        if _is_ltx2_in_toml(content):
            print(f"[DEBUG] Detected LTX-2 model via TOML parsing")
            return True

        return False
    except Exception as e:
        print(f"[DEBUG] Could not check model type: {e}")
        return False
