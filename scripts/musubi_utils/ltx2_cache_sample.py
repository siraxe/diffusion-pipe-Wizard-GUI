#!/usr/bin/env python3
"""
LTX-Video-2 Sample Prompt Caching Wrapper

This script wraps the standard LTX2 caching commands with:
1. Sample prompts file generation from inline config
2. Hash-based validation to avoid redundant caching
3. Proper streaming output for Flet console integration

Usage:
    python ltx2_cache_sample.py --config workspace/last_config.toml --dataset_config <path>
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

# Try importing toml, provide helpful error if not available
try:
    import toml
except ImportError:
    print("ERROR: toml module required. Install with: pip install toml")
    sys.exit(1)

from loguru import logger

# Configure loguru to output to stderr (captured by subprocess) with simple format
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<level>{message}</level>", level="INFO", colorize=False)


def find_project_root() -> Path:
    """Auto-detect the project root directory."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "flet_app").exists() or (parent / "diffusion-trainers").exists():
            return parent
    return Path.cwd()


def compute_sample_cache_hash(validation: dict, project_root: Path) -> str:
    """Compute a hash of the sampling configuration to detect changes."""
    hasher = hashlib.sha256()

    hash_fields = [
        'prompts', 'negative_prompt', 'video_dims', 'sample_steps',
        'guidance_scale', 'seed', 'start_images', 'interval'
    ]

    for field in hash_fields:
        value = validation.get(field)
        if value is not None:
            if field == 'start_images' and value:
                img_path = Path(value)
                if not img_path.is_absolute():
                    img_path = project_root / img_path
                if img_path.exists():
                    with open(img_path, 'rb') as f:
                        hasher.update(f.read())
                    hasher.update(str(img_path).encode())
                else:
                    hasher.update(str(img_path).encode())
            else:
                hasher.update(str(value).encode())

    return hasher.hexdigest()


def log_print(msg: str):
    """Print to stderr with immediate flush for real-time streaming."""
    print(msg, file=sys.stderr, flush=True)


def should_rebuild_sample_cache(validation: dict, output_dir: str, project_root: Path) -> bool:
    """Check if sample cache needs to be rebuilt based on hash comparison."""
    sample_dir = Path(output_dir) / 'sample'
    cache_path = sample_dir / 'sample_prompts_cache.pt'
    hash_file = sample_dir / '.sample_cache_hash'

    if not cache_path.exists():
        log_print("Sample cache does not exist, will create it")
        return True

    if not hash_file.exists():
        log_print("Hash file does not exist, will create cache")
        return True

    try:
        current_hash = compute_sample_cache_hash(validation, project_root)
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()

        if current_hash != stored_hash:
            log_print("Sample config changed, cache will be rebuilt")
            return True
        else:
            log_print("Sample cache is valid (hash matches)")
            return False
    except Exception as e:
        logger.warning(f"Error checking cache hash: {e}, will rebuild")
        return True


def save_sample_cache_hash(validation: dict, output_dir: str, project_root: Path) -> None:
    """Save the hash of the current sampling configuration."""
    sample_dir = Path(output_dir) / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)

    hash_file = sample_dir / '.sample_cache_hash'
    current_hash = compute_sample_cache_hash(validation, project_root)

    with open(hash_file, 'w') as f:
        f.write(current_hash)


def generate_sample_prompts_file(validation: dict, output_dir: str) -> str:
    """
    Generate sample_prompts.txt from inline config.

    Returns:
        Path to the generated sample_prompts.txt file
    """
    sample_dir = Path(output_dir) / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_prompts_path = sample_dir / 'sample_prompts.txt'

    inline_prompts = validation.get('prompts')
    if not inline_prompts:
        raise ValueError("No 'prompts' found in validation config")

    prompt_line = inline_prompts

    # Parse video_dims to get width, height, frames
    video_dims = validation.get('video_dims', '768, 512, 45')
    if video_dims:
        try:
            dims = [d.strip() for d in str(video_dims).split(',')]
            if len(dims) >= 3:
                width = round(int(dims[0]) / 32) * 32
                height = round(int(dims[1]) / 32) * 32
                frames = max(round((int(dims[2]) - 1) / 8) * 8 + 1, 9)
                prompt_line += f" --w {width} --h {height} --f {frames}"
        except Exception:
            pass

    # Add sample steps
    if validation.get('sample_steps'):
        prompt_line += f" --s {validation['sample_steps']}"

    # Add guidance scale
    if validation.get('guidance_scale'):
        prompt_line += f" --g {validation['guidance_scale']}"

    # Add seed
    if validation.get('seed') is not None:
        prompt_line += f" --d {validation['seed']}"

    # Add negative prompt
    negative = validation.get('negative_prompt')
    if negative:
        prompt_line += f" --n {negative}"

    # Add image path for i2v if provided
    start_images = validation.get('start_images')
    if start_images:
        img_path = Path(start_images)
        if not img_path.is_absolute():
            resolved = find_project_root() / img_path
            if not resolved.exists():
                resolved = Path.cwd() / img_path
            img_path = resolved.resolve()
        else:
            img_path = img_path.resolve()
        prompt_line += f" --i {img_path}"

    with open(sample_prompts_path, 'w') as f:
        f.write(prompt_line + "\n")

    return str(sample_prompts_path)


def parse_bool(value) -> bool:
    """Convert various input types to boolean."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'on')


def run_cache_sample_prompts(
    config: dict,
    dataset_config: str,
    project_root: Path,
    output_dir: str = None,
    force: bool = False
) -> bool:
    """
    Run sample prompts caching with hash validation.

    Returns True if caching was run, False if skipped.
    """
    model = config.get('model', {})
    training_strategy = config.get('training_strategy', {})
    acceleration = config.get('acceleration', {})
    validation = config.get('validation', {})

    # Check if caching is enabled
    if not parse_bool(validation.get('cache_te', True)):
        return False

    # Check if sampling is enabled
    sample_interval = validation.get('interval', -1)
    sample_at_first = parse_bool(validation.get('sample_at_first', False))

    if sample_interval < 0 and not sample_at_first:
        return False

    if sample_interval < 1:
        return False

    output_dir = output_dir or model.get('output_dir', 'output/ltx2_lora')

    # Step 1: Generate sample_prompts.txt
    try:
        generate_sample_prompts_file(validation, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate sample_prompts.txt: {e}")
        return False

    # Step 2: Check hash validation
    if not force and not should_rebuild_sample_cache(validation, output_dir, project_root):
        log_print("Sample cache is valid, skipping caching")
        return False

    # Step 3: Build caching command
    musubi_root = project_root / "diffusion-trainers" / "musubi-tuner"
    cache_script = str(musubi_root / "ltx2_cache_text_encoder_outputs.py")

    ltx2_checkpoint = model.get('model_path', '')
    gemma_root = model.get('text_encoder_path', '')
    ltx2_mode = training_strategy.get('ltx_mode', 'video')
    mixed_precision = acceleration.get('mixed_precision_mode', 'bf16')
    gemma_8bit = parse_bool(acceleration.get('8_bit_te', True))

    sample_dir = Path(output_dir) / 'sample'
    sample_prompts_path = str(sample_dir / 'sample_prompts.txt')
    sample_prompts_cache = str(sample_dir / 'sample_prompts_cache.pt')

    cmd = [
        sys.executable,
        cache_script,
        "--dataset_config", dataset_config,
        "--ltx2_checkpoint", ltx2_checkpoint,
        "--gemma_root", gemma_root,
        "--device", "cuda",
        "--mixed_precision", mixed_precision,
        "--ltx2_mode", ltx2_mode,
        "--batch_size", "1",
        "--precache_sample_prompts",
        "--sample_prompts", sample_prompts_path,
        "--sample_prompts_cache", sample_prompts_cache
    ]

    if gemma_8bit:
        cmd.append("--gemma_load_in_8bit")

    if validation.get('start_images'):
        cmd.append("--cache_i2v")

    # Step 4: Run caching with streaming output
    log_print("=" * 60)
    log_print("Caching sample prompts and images...")
    log_print("This will take a few minutes but speeds up sampling during training")
    log_print("=" * 60)

    # Set up environment with PYTHONPATH for musubi_tuner imports
    musubi_src = str(musubi_root / "src")
    env = os.environ.copy()
    env['PYTHONPATH'] = musubi_src + ':' + env.get('PYTHONPATH', '')
    # Force unbuffered output
    env['PYTHONUNBUFFERED'] = '1'

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            env=env,
        )

        for line in process.stdout:
            if line.rstrip():
                # Print directly to stderr for immediate streaming
                print(line.rstrip(), file=sys.stderr, flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        save_sample_cache_hash(validation, output_dir, project_root)

        log_print("=" * 60)
        log_print("Sample caching completed successfully!")
        log_print("=" * 60)

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Sample caching failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during sample caching: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cache LTX-2 sample prompts with hash validation")
    parser.add_argument("--config", required=True, help="Path to main config file")
    parser.add_argument("--dataset_config", help="Path to dataset config file")
    parser.add_argument("--project_root", default=None, help="Project root directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force rebuild cache")

    args = parser.parse_args()

    # Detect project root
    project_root = Path(args.project_root) if args.project_root else find_project_root()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    with open(config_path, 'r') as f:
        config = toml.load(f)

    # Use dataset_config from args or config
    dataset_config = args.dataset_config or config.get('data', {}).get('preprocessed_data_root', '')

    if not dataset_config:
        log_print("ERROR: Dataset config must be specified via --dataset_config or in config")
        sys.exit(1)

    # Resolve dataset_config path
    if not Path(dataset_config).is_absolute():
        dataset_config = str(project_root / dataset_config)

    # Run caching
    result = run_cache_sample_prompts(
        config=config,
        dataset_config=dataset_config,
        project_root=project_root,
        output_dir=args.output_dir,
        force=args.force
    )

    if not result:
        log_print("No caching performed (cache valid or caching disabled)")
        sys.exit(0)


if __name__ == "__main__":
    main()
