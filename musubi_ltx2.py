"""Musubi training handler - refactored interface for flet app integration."""
import os
import subprocess
import shlex
import toml
import flet as ft
from loguru import logger
import safetensors.torch
import torch

# Import musubi utilities
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from musubi_utils import find_last_state_directory

# =============================================================================
# == 1. Constants & Defaults ==================================================
# =============================================================================

DEFAULT_LTX2_CHECKPOINT = '/home/user/Dpipe/models/ltx2/ltx-2-19b-dev.safetensors'
DEFAULT_GEMMA_ROOT = '/home/user/Dpipe/models/gemma-3'
DEFAULT_OUTPUT_DIR = '/home/user/workspace/output/ltx2_lora'

def get_config_data(config_path: str) -> dict:
    """Safely loads and returns TOML configuration data."""
    if not config_path or not os.path.exists(config_path):
        logger.warning(f"Config path missing or invalid: {config_path}")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return toml.load(f)
    except Exception as e:
        logger.error(f"Failed to load config at {config_path}: {e}")
        return {}

def resolve_project_root() -> str:
    """Determines the project root directory."""
    try:
        from flet_app.project_root import get_project_root
        return str(get_project_root())
    except ImportError:
        return os.getcwd()

def is_musubi_model(config_path: str) -> bool:
    """Checks if the configuration target is a Musubi (ltx-video-2) model."""
    config = get_config_data(config_path)
    model_type = config.get('model', {}).get('type', '').lower()
    return 'ltx-video-2' in model_type

def parse_bool(value) -> bool:
    """Converts various input types (str, int, bool) to a boolean."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'on')

# =============================================================================
# == 2. LoRA Format Detection & Conversion =====================================
# =============================================================================

def get_lora_rank(file_path: str) -> int:
    """
    Detect the rank of a LoRA checkpoint.

    Returns the rank (dimension of lora_down/lora_A), or 0 if unable to detect.
    """
    try:
        state_dict = safetensors.torch.load_file(file_path)
        if not state_dict:
            return 0

        # Look for lora_down or lora_A weight to determine rank
        for key in state_dict.keys():
            # Training format: lora_unet_model_*.lora_down.weight
            if key.endswith('.lora_down.weight'):
                return state_dict[key].shape[0]
            # ComfyUI format: diffusion_model.*.lora_A.weight
            elif key.endswith('.lora_A.weight'):
                return state_dict[key].shape[0]

        return 0
    except Exception as e:
        logger.warning(f"Error detecting LoRA rank for {file_path}: {e}")
        return 0


def is_comfy_format_lora(file_path: str) -> bool:
    """
    Check if a LoRA file is in ComfyUI format by examining the keys.

    ComfyUI format keys start with 'diffusion_model.'
    Training format keys start with 'lora_unet_model_'
    """
    try:
        state_dict = safetensors.torch.load_file(file_path)
        if not state_dict:
            return False

        # Check first few keys
        for key in list(state_dict.keys())[:5]:
            if key.startswith('diffusion_model.'):
                return True
            if key.startswith('lora_unet_model_'):
                return False

        # If no clear prefix, check for other ComfyUI patterns
        for key in state_dict.keys():
            if '.lora_A.' in key or '.lora_B.' in key:
                return True

        return False
    except Exception as e:
        logger.warning(f"Error checking LoRA format for {file_path}: {e}")
        return False


def convert_comfy_to_training_with_rank(file_path: str, target_rank: int, console: ft.Text = None, page: ft.Page = None) -> str:
    """
    Convert ComfyUI format LoRA to training format with optional rank conversion.

    Creates a new file with _rank{target_rank} suffix instead of overwriting.
    Runs in a subprocess to avoid blocking the UI.

    Returns the path to the new file, or None if failed.
    """
    import sys
    import subprocess
    from pathlib import Path

    try:
        # Determine output path (with rank suffix)
        input_file = Path(file_path)
        output_path = input_file.parent / f"{input_file.stem}_rank{target_rank}{input_file.suffix}"

        # Path to conversion script
        sys_path = os.path.join(resolve_project_root(), 'scripts')
        convert_script = os.path.join(sys_path, 'convert_comfy_to_training_lora.py')

        msg = f"[Converting] Converting ComfyUI LoRA to training format (rank {target_rank})...\n"
        msg += f"[Converting] This may take 10-30 seconds depending on model size...\n"
        _safe_append(console, msg)
        update_ui(page)
        logger.info(f"Converting ComfyUI LoRA with rank conversion: {file_path} -> rank {target_rank}")

        # Build command
        cmd = [
            sys.executable,
            convert_script,
            file_path,
            '--target_rank', str(target_rank),
            '-o', str(output_path)
        ]

        # Run the conversion in subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )

        if result.returncode == 0:
            msg = f"[Success] Created converted checkpoint: {output_path}\n"
            _safe_append(console, msg, color='yellow')
            update_ui(page)
            logger.info(f"Successfully created converted checkpoint: {output_path}")
            return str(output_path)
        else:
            error_msg = f"[Error] Conversion failed: {result.stderr}\n"
            _safe_append(console, error_msg, color='red')
            update_ui(page)
            logger.error(f"ComfyUI conversion failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        error_msg = f"[Error] Conversion timed out after 3 minutes\n"
        _safe_append(console, error_msg, color='red')
        update_ui(page)
        logger.error("ComfyUI conversion timed out")
        return None
    except Exception as e:
        error_msg = f"[Error] Failed to convert LoRA: {e}\n"
        _safe_append(console, error_msg, color='red')
        update_ui(page)
        logger.error(f"Failed to convert ComfyUI LoRA {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def rerank_training_format_lora(file_path: str, target_rank: int, console: ft.Text = None, page: ft.Page = None) -> str:
    """
    Rerank a training format LoRA checkpoint to a different rank.

    Creates a new file with _rank{target_rank} suffix.

    Returns the path to the new file, or None if failed.
    """
    import sys
    from pathlib import Path
    import subprocess

    try:
        # Determine output path
        input_file = Path(file_path)
        output_path = input_file.parent / f"{input_file.stem}_rank{target_rank}{input_file.suffix}"

        # Use the dedicated reranking script
        sys_path = os.path.join(resolve_project_root(), 'scripts')
        rerank_script = os.path.join(sys_path, 'rerank_lora.py')

        msg = f"[Reranking] Converting rank -> {target_rank} (this may take 10-30 seconds)...\n"
        msg += f"[Reranking] Using GPU for faster SVD computation\n"
        _safe_append(console, msg)
        update_ui(page)
        logger.info(f"Reranking checkpoint: {file_path} -> rank {target_rank}")

        # Build command
        cmd = [
            sys.executable,
            rerank_script,
            file_path,
            '--target_rank', str(target_rank),
            '-o', str(output_path),
            '--device', 'cuda'
        ]

        # Run the reranking
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )

        if result.returncode == 0:
            msg = f"[Success] Created reranked checkpoint: {output_path}\n"
            _safe_append(console, msg, color='yellow')
            update_ui(page)
            logger.info(f"Successfully reranked checkpoint: {output_path}")
            return str(output_path)
        else:
            error_msg = f"[Error] Reranking failed: {result.stderr}\n"
            _safe_append(console, error_msg, color='red')
            update_ui(page)
            logger.error(f"Reranking failed: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        error_msg = f"[Error] Reranking timed out after 3 minutes\n"
        _safe_append(console, error_msg, color='red')
        update_ui(page)
        logger.error("Reranking timed out")
        return None
    except Exception as e:
        error_msg = f"[Error] Failed to rerank LoRA: {e}\n"
        _safe_append(console, error_msg, color='red')
        update_ui(page)
        logger.error(f"Failed to rerank LoRA {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_comfy_to_training_inplace(file_path: str, console: ft.Text = None, page: ft.Page = None) -> bool:
    """
    Convert a ComfyUI format LoRA to training format, overwriting the original file.

    Returns True if conversion was successful, False otherwise.
    """
    import tempfile
    import shutil

    try:
        # Create a temporary file for conversion
        temp_fd, temp_path = tempfile.mkstemp(suffix='.safetensors')
        os.close(temp_fd)

        # Import the conversion script as a module
        sys_path = os.path.join(resolve_project_root(), 'scripts')
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)

        from convert_comfy_to_training_lora import convert_comfy_to_training

        # Perform conversion to temp file
        msg = f"[Converting] Detected ComfyUI format LoRA: {file_path}\n"
        msg += f"[Converting] Converting to training format...\n"
        _safe_append(console, msg)
        update_ui(page)
        logger.info(f"Converting ComfyUI LoRA to training format: {file_path}")

        convert_comfy_to_training(file_path, temp_path, alpha=None, verbose=False)

        # Replace original file with converted version
        shutil.move(temp_path, file_path)

        msg = f"[Success] LoRA converted and original file overwritten: {file_path}\n"
        _safe_append(console, msg, color='yellow')
        update_ui(page)
        logger.info(f"Successfully converted and replaced: {file_path}")

        return True

    except Exception as e:
        # Clean up temp file if it exists
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

        error_msg = f"[Error] Failed to convert LoRA: {e}\n"
        _safe_append(console, error_msg, color='red')
        update_ui(page)
        logger.error(f"Failed to convert ComfyUI LoRA {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# == 3. Utilities (Data & UI) ================================================
# =============================================================================

def _safe_append(console: ft.Text, text: str, color: str = None):
    """Updates Flet Text control safely with new spans or plain text."""
    if console is None:
        return
    
    console.value = (console.value or "") + text
    if console.spans is None:
        console.spans = []
    
    if color:
        console.spans.append(ft.TextSpan(text, style=ft.TextStyle(color=color)))

def update_ui(page: ft.Page):
    """Trigger a page update if the page exists."""
    if page:
        page.update()

# =============================================================================
# == 4. Argument & Command Builders ===========================================
# =============================================================================

def _build_ltx2_train_args(cfg: dict, musubi_config_path: str, slider_config_path: str = None, console: ft.Text = None, page: ft.Page = None, resume_last: bool = False) -> list:
    """
    Centralized logic to build training arguments.
    Used by both the execution handler and the command display generator.

    Args:
        cfg: Configuration dictionary
        musubi_config_path: Path to musubi config (used for normal training)
        slider_config_path: Path to slider config (used for slider training)
        console: Optional console for logging
        page: Optional page for UI updates
        resume_last: Whether to resume from last saved state
    """
    # Sub-section extraction with safety defaults
    m = cfg.get('model', {})
    o = cfg.get('optimization', {})
    a = cfg.get('acceleration', {})
    s = cfg.get('training_strategy', {})
    c = cfg.get('checkpoints', {})
    l = cfg.get('lora', {})
    f = cfg.get('flow_matching', {})

    # Read use_mask directly from last_config.toml
    enable_mask = parse_bool(s.get('use_mask', False))

    # Check if slider training is enabled
    slider_enabled = parse_bool(s.get('slider', False))
    use_slider = slider_enabled and slider_config_path and os.path.exists(slider_config_path)

    # Choose script and config based on slider mode
    if use_slider:
        script_path = os.path.join(resolve_project_root(), 'diffusion-trainers', 'musubi-tuner', 'ltx2_train_slider.py')
        config_flag = '--slider_config'
        config_path = slider_config_path
    else:
        script_path = os.path.join(resolve_project_root(), 'diffusion-trainers', 'musubi-tuner', 'ltx2_train_network.py')
        config_flag = '--dataset_config'
        config_path = musubi_config_path

    args = [
        script_path,
        '--mixed_precision', a.get('mixed_precision_mode', 'bf16'),
        config_flag, config_path,
        '--gemma_root', m.get('text_encoder_path', DEFAULT_GEMMA_ROOT),
        '--ltx2_checkpoint', m.get('model_path', DEFAULT_LTX2_CHECKPOINT),
        '--flash_attn',
    ]

    # output_name from model config (added via TOML save)
    output_name = m.get('name', None)
    if output_name:
        args.extend(['--output_name', str(output_name)])

    # Boolean flags
    if parse_bool(o.get('enable_gradient_checkpointing', True)):
        args.append('--gradient_checkpointing')
    if parse_bool(a.get('fp8_base', True)):
        args.append('--fp8_base')
    if parse_bool(a.get('fp8_scaled', True)):
        args.append('--fp8_scaled')
    if parse_bool(a.get('8_bit_te', False)):
        args.append('--gemma_load_in_8bit')
    if parse_bool(a.get('attn_chunking', False)):
        args.extend(['--split_attn_target', 'video', '--split_attn_mode', 'query', '--split_attn_chunk_size', '512'])
    if parse_bool(c.get('save_state', False)):
        args.append('--save_state')
    if enable_mask:
        args.append('--ltx2_enable_mask')

    # Resume from last state if requested
    if resume_last:
        output_dir = m.get('output_dir', DEFAULT_OUTPUT_DIR)
        # Get output_name - try model.name first, then fall back to extracted name
        resume_output_name = m.get('name', output_name)

        # If no name set, search for ANY state directory in output_dir
        if not resume_output_name:
            # Find any state directory (search without name filter)
            if os.path.exists(output_dir):
                state_dirs = []
                for entry in os.listdir(output_dir):
                    entry_path = os.path.join(output_dir, entry)
                    if os.path.isdir(entry_path) and entry.endswith("-state"):
                        mtime = os.path.getmtime(entry_path)
                        state_dirs.append((mtime, entry_path))
                if state_dirs:
                    # Sort by modification time, get most recent
                    state_dirs.sort(key=lambda x: x[0], reverse=True)
                    last_state_dir = state_dirs[0][1]
                else:
                    last_state_dir = None
            else:
                last_state_dir = None
        else:
            # Use name-based search
            last_state_dir = find_last_state_directory(output_dir, resume_output_name)

        if last_state_dir:
            args.extend(['--resume', last_state_dir])
            msg = f"[Resume] Found last state directory: {last_state_dir}\n"
            if console:
                _safe_append(console, msg, color='green')
            logger.info(f"Resuming from state: {last_state_dir}")
        else:
            msg = f"[Resume] No state directory found in {output_dir}\n"
            if console:
                _safe_append(console, msg, color='yellow')
            logger.warning(f"No state directory found for resuming")

    # Training Interval args
    ckpt_mode = c.get('mode', 'steps')
    interval = c.get('interval', 50)
    steps = o.get('max_steps', 200)
    keep_last_n = c.get('keep_last_n', -1)

    if ckpt_mode == 'epochs':
        args.extend(['--save_every_n_epochs', str(interval)])
        if keep_last_n > 0:
            args.extend(['--save_last_n_epochs', str(keep_last_n)])
        args.extend(['--max_train_epochs', str(steps)])
    else:
        args.extend(['--save_every_n_steps', str(interval)])
        if keep_last_n > 0:
            args.extend(['--save_last_n_steps', str(keep_last_n)])
        args.extend(['--max_train_steps', str(steps)])

    # Network args - network_dim and network_alpha are ALWAYS needed for LoRA/LoKR rank
    args.extend([
        '--network_dim', str(l.get('rank', 32)),
        '--network_alpha', str(l.get('alpha', 32)),
    ])

    network_dropout = l.get('dropout', 0.0)
    if network_dropout > 0:
        args.extend(['--network_dropout', str(network_dropout)])

    # Optimizer and scheduler args
    args.extend([
        '--gradient_accumulation_steps', str(o.get('gradient_accumulation_steps', 4)),
        '--learning_rate', str(o.get('learning_rate', 0.001)),
        '--optimizer_type', o.get('optimizer_type', 'AdamW'),
        '--lr_scheduler', o.get('scheduler_type', 'constant'),
        '--timestep_sampling', f.get('timestep_sampling_mode', 'shifted_logit_normal'),
    ])

    # Add lr_warmup_steps if constant_with_warmup scheduler is selected
    if o.get('scheduler_type') == 'constant_with_warmup':
        lr_warmup_steps = o.get('lr_warmup_steps', 50)
        args.extend(['--lr_warmup_steps', str(lr_warmup_steps)])

    # Max grad norm and block swap
    max_grad_norm = o.get('max_grad_norm', 1.0)
    if max_grad_norm > 0:
        args.extend(['--max_grad_norm', str(max_grad_norm)])

    blocks_to_swap = o.get('blocks_to_swap', 0)
    if blocks_to_swap > 0:
        args.extend(['--blocks_to_swap', str(blocks_to_swap)])

    # Handle optimizer-specific parameters (pass optimizer_args as-is, training script parses internally)
    optimizer_args_str = o.get('optimizer_args', None)
    if optimizer_args_str:
        args.extend(['--optimizer_args', optimizer_args_str])

    # Output args

    # Output args
    output_name = m.get('output_name', 'ltx2_lora')
    output_dir = m.get('output_dir', DEFAULT_OUTPUT_DIR)
    args.extend([
        '--output_name', output_name,
        '--output_dir', output_dir,
        '--log_with', 'tensorboard',
        '--logging_dir', os.path.join(output_dir, '.tensorboard'),
    ])

    # Training strategy args
    ltx_mode = s.get('ltx_mode', 'video')
    args.extend([
        '--ltx2_mode', ltx_mode,
        '--ltx2_first_frame_conditioning_p', str(s.get('first_frame_conditioning_p', 0.1)),
    ])

    # Add lora_target_preset based on ltx_mode for both lora and lokr
    # This controls which layers are targeted for adapter training
    if ltx_mode == 'audio':
        args.extend(['--lora_target_preset', 'audio'])
    elif ltx_mode == 'video' or ltx_mode == 'av':
        # Use 't2v' (default) for video/av - targets all attention across modalities
        # Could use 'v2v' to also include feed-forward layers
        pass  # uses default t2v preset
    # Note: 'full' preset would target all linear layers (not needed for now)

    if parse_bool(s.get('separate_audio_buckets', False)):
        args.append('--separate_audio_buckets')

    # Load checkpoint from init_from_existing (formerly load_checkpoint)
    init_from_existing = l.get('init_from_existing', '')
    if init_from_existing and str(init_from_existing).lower() not in ('', 'null', 'none'):
        original_path = init_from_existing
        # Convert to absolute path if relative
        if not os.path.isabs(init_from_existing):
            # Resolve relative to project root
            project_root = resolve_project_root()
            init_from_existing = os.path.abspath(os.path.join(project_root, init_from_existing))
            logger.info(f"Converted relative checkpoint path: {original_path} -> {init_from_existing}")

        # Check if the file exists
        if os.path.exists(init_from_existing):
            logger.info(f"Checkpoint file exists: {init_from_existing}")

            # Get target rank from config
            target_rank = l.get('rank', 32)

            # Check if conversion is needed (ComfyUI format OR rank mismatch)
            needs_conversion = False
            conversion_reason = ""
            is_comfy = False

            # Check 1: ComfyUI format
            if is_comfy_format_lora(init_from_existing):
                needs_conversion = True
                conversion_reason = "ComfyUI format"
                is_comfy = True
            # Check 2: Rank mismatch (only for training format files)
            else:
                checkpoint_rank = get_lora_rank(init_from_existing)
                if checkpoint_rank > 0 and checkpoint_rank != target_rank:
                    needs_conversion = True
                    conversion_reason = f"rank mismatch (checkpoint: {checkpoint_rank}, config: {target_rank})"
                else:
                    logger.info(f"Checkpoint rank {checkpoint_rank} matches config rank {target_rank}")

            # Convert if needed
            if needs_conversion:
                if is_comfy:
                    # ComfyUI format: convert to training format with target rank
                    converted_path = convert_comfy_to_training_with_rank(init_from_existing, target_rank, console, page)
                else:
                    # Training format but rank mismatch: just rerank
                    converted_path = rerank_training_format_lora(init_from_existing, target_rank, console, page)

                if converted_path:
                    init_from_existing = converted_path
                    logger.info(f"Using converted checkpoint: {init_from_existing}")
            else:
                logger.info(f"Checkpoint is already in correct format and rank")
        else:
            logger.warning(f"Checkpoint file does not exist: {init_from_existing}")
            # Still add to args - let the training script handle the error
        args.extend(['--network_weights', init_from_existing])

    # Network module - use networks.lora_ltx2 for both lora and lokr
    # Note: lycoris.kohya doesn't support LTX-2's transformer architecture (finds 0 modules)
    # networks.lora_ltx2 is LTX2-specific but only supports standard LoRA (not true LoKR)
    training_mode = m.get('training_mode', 'lora')
    args.extend(['--network_module', 'networks.lora_ltx2'])

    if training_mode == 'lokr':
        # Update lycoris config file with current rank/alpha settings from UI
        # (for reference/future use, but networks.lora_ltx2 doesn't read it)
        lycoris_config_path = os.path.join(resolve_project_root(), 'workspace', 'ltx2_lycoris_config.toml')
        lokr_factor = l.get('rank', 32)
        lokr_conv_dim = l.get('alpha', 32)
        lokr_norm = l.get('init_lokr_norm', 0.001)

        try:
            lycoris_config_content = f"""# LyCORIS config for LTX2 LoKR training
# Auto-generated from flet_app settings
# NOTE: networks.lora_ltx2 doesn't use this file (for reference only)
[network]
base_algo = "lokr"
base_factor = {lokr_factor}

[network.modules.BasicAVTransformerBlock]
algo = "lokr"
factor = {lokr_factor}
conv_dim = {lokr_conv_dim}

[network.init]
lokr_norm = {lokr_norm}
"""
            with open(lycoris_config_path, 'w') as f:
                f.write(lycoris_config_content)
            logger.info(f"Updated lycoris config: {lycoris_config_path} with factor={lokr_factor}, conv_dim={lokr_conv_dim}")
        except Exception as e:
            logger.warning(f"Failed to update lycoris config: {e}")

        # conv_dim for Conv2D layers (if any Conv2D modules are targeted)
        args.extend(['--network_args', f"conv_dim={lokr_conv_dim}"])
        logger.info(f"LoKR mode: Using rank={lokr_factor}, alpha={lokr_conv_dim} via standard LoRA")
        logger.warning("Note: True LoKR is not currently supported by networks.lora_ltx2 - using standard LoRA")

    # Validation/Sampling args (from validation section)
    v = cfg.get('validation', {})
    sample_interval = v.get('interval', '-1')
    sample_at_first = parse_bool(v.get('sample_at_first', 'false'))

    # Only add sampling flags if sampling is enabled (interval != -1 OR sample_at_first is true)
    sampling_enabled = (sample_interval and str(sample_interval) != '-1') or sample_at_first

    if sampling_enabled:
        if sample_at_first:
            args.append('--sample_at_first')

        if sample_interval and str(sample_interval) != '-1':
            # Use checkpoint mode to determine steps vs epochs
            ckpt_mode = c.get('mode', 'steps')
            if ckpt_mode == 'epochs':
                args.extend(['--sample_every_n_epochs', str(sample_interval)])
            else:
                args.extend(['--sample_every_n_steps', str(sample_interval)])

        # Video dimensions: width, height, frames
        video_dims = v.get('video_dims', '768, 512, 45')
        if video_dims and str(video_dims) != '768, 512, 45':
            # Parse "width, height, frames" format
            try:
                dims = [d.strip() for d in str(video_dims).split(',')]
                if len(dims) >= 3:
                    width = int(dims[0])
                    height = int(dims[1])
                    frames = int(dims[2])

                    # Adjust width and height to be divisible by 32
                    width = round(width / 32) * 32
                    height = round(height / 32) * 32

                    # Adjust frames to be n * 8 + 1 (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89...)
                    # Formula: frames = (frames - 1) / 8, round, then * 8 + 1
                    frames = round((frames - 1) / 8) * 8 + 1
                    frames = max(frames, 9)  # Minimum 9 frames

                    args.extend(['--width', str(width), '--height', str(height), '--sample_num_frames', str(frames)])
            except:
                pass  # Use defaults if parsing fails

        # Generate audio flag
        if parse_bool(v.get('generate_audio', 'false')):
            args.append('--sample_merge_audio')

        # Sampling optimization flags
        # Enable offloading if s_offload is true (saves ~8GB VRAM)
        if parse_bool(v.get('s_offload', True)):
            args.append('--sample_with_offloading')

        # Tiled VAE for spatial processing (saves VRAM)
        # Note: temporal tiling must be divisible by 8, or disabled (0)
        if parse_bool(v.get('tiled_vae', True)):
            args.append('--sample_tiled_vae')
            args.extend(['--sample_vae_tile_size', '512'])  # Spatial tile size
            args.extend(['--sample_vae_tile_overlap', '64'])  # Spatial overlap
            args.extend(['--sample_vae_temporal_tile_size', '16'])  # 16 frames (divisible by 8)
            args.extend(['--sample_vae_temporal_tile_overlap', '8'])  # Temporal overlap

        # Prompts and start_images (save to file, pass path to training)
        prompts = v.get('prompts', '')
        negative_prompt = v.get('negative_prompt', '')
        start_images = v.get('start_images', 'none')

        if prompts or negative_prompt or start_images:
            # Build sample prompts content in musubi format
            # Format: prompt --n negative_prompt --g guidance --s steps --i image_path (all on one line)
            prompt_parts = []

            # Main prompt
            if prompts:
                prompt_parts.append(prompts.strip())

            # Negative prompt with --n prefix
            if negative_prompt:
                prompt_parts.append(f"--n {negative_prompt.strip()}")

            # Sampling parameters from config
            sample_steps = v.get('sample_steps', '30')
            guidance_scale = v.get('guidance_scale', '4.0')
            seed = v.get('seed', '42')

            # Guidance scale - using --g prefix
            prompt_parts.append(f"--g {guidance_scale}")

            # Inference steps - using --s prefix
            prompt_parts.append(f"--s {sample_steps}")

            # Seed - using --d prefix
            prompt_parts.append(f"--d {seed}")

            # Start images with --i prefix (if not "none" AND file exists)
            if start_images and str(start_images).strip().lower() != 'none':
                img_path = start_images.strip()
                # Check if the image exists (relative to project root or absolute path)
                if os.path.isabs(img_path):
                    img_exists = os.path.exists(img_path)
                else:
                    # Relative path - check from project root
                    img_exists = os.path.exists(os.path.join(resolve_project_root(), img_path))
                if img_exists:
                    prompt_parts.append(f"--i {img_path}")

            # Only save if we have actual content
            if not prompt_parts:
                # Nothing to save
                pass
            else:
                # Join all parts with space
                sample_prompts_content = " ".join(prompt_parts)

                # Save to prompts file in output/sample directory
                output_dir = m.get('output_dir', DEFAULT_OUTPUT_DIR)
                sample_dir = os.path.join(output_dir, 'sample')
                os.makedirs(sample_dir, exist_ok=True)
                sample_prompts_path = os.path.join(sample_dir, 'sample_prompts.txt')

                try:
                    with open(sample_prompts_path, 'w') as f:
                        f.write(sample_prompts_content)
                    args.extend(['--sample_prompts', sample_prompts_path])

                    # Check if sample prompts cache exists, if so use it
                    cache_path = os.path.join(sample_dir, 'sample_prompts_cache.pt')
                    if os.path.exists(cache_path):
                        args.extend(['--use_precached_sample_prompts', '--sample_prompts_cache', cache_path])
                        logger.info(f"Using pre-cached sample prompts: {cache_path}")

                    # Check if sample latents cache exists (I2V images)
                    latents_cache_path = os.path.join(sample_dir, 'sample_latents_cache.pt')
                    if os.path.exists(latents_cache_path):
                        args.extend(['--sample_latents_cache', latents_cache_path])
                        logger.info(f"Using pre-cached sample latents (I2V): {latents_cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save sample prompts: {e}")

    # Preservation & Regularization flags (from acceleration section)
    # blank_preservation
    if parse_bool(a.get('blank_preservation', False)):
        args.append('--blank_preservation')
        blank_preservation_args = a.get('blank_preservation_args', 'multiplier=0.5')
        args.extend(['--blank_preservation_args', blank_preservation_args])

    # dop (Differential Output Preservation)
    if parse_bool(a.get('dop', False)):
        args.append('--dop')
        dop_args = a.get('dop_args', 'class=woman multiplier=1.0')
        args.extend(['--dop_args', dop_args])

    # prior_divergence
    if parse_bool(a.get('prior_divergence', False)):
        args.append('--prior_divergence')
        prior_divergence_args = a.get('prior_divergence_args', 'multiplier=0.1')
        args.extend(['--prior_divergence_args', prior_divergence_args])

    # CREPA (Cross-frame Representation Alignment)
    crepa_val = a.get('crepa', False)
    logger.info(f"[CREPA DEBUG] crepa value from config: {crepa_val} (type: {type(crepa_val)})")
    if parse_bool(crepa_val):
        args.append('--crepa')
        crepa_mode = a.get('crepa_mode', 'backbone')
        crepa_args = a.get('crepa_args', 'student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2')
        # Prepend mode to args and split into separate items for nargs='*'
        crepa_args_with_mode = f"mode={crepa_mode} {crepa_args}"
        args.append('--crepa_args')
        args.extend(crepa_args_with_mode.split())
        logger.info(f"[CREPA DEBUG] Added --crepa with args: {crepa_args_with_mode}")

    # Block-targeted optimizer grouping (full fine-tuning)
    freeze_early = a.get('freeze_early_blocks', 0)
    if freeze_early and int(freeze_early) > 0:
        args.extend(['--freeze_early_blocks', str(freeze_early)])

    freeze_block_indices = a.get('freeze_block_indices', None)
    if freeze_block_indices:
        args.extend(['--freeze_block_indices', str(freeze_block_indices)])

    block_lr_scales = a.get('block_lr_scales', None)
    if block_lr_scales:
        args.extend(['--block_lr_scales', str(block_lr_scales)])

    non_block_lr_scale = a.get('non_block_lr_scale', 1.0)
    if non_block_lr_scale and float(non_block_lr_scale) != 1.0:
        args.extend(['--non_block_lr_scale', str(non_block_lr_scale)])

    # Attention geometry protection
    attn_geom_scale = a.get('attn_geometry_lr_scale', 1.0)
    if attn_geom_scale and float(attn_geom_scale) != 1.0:
        args.extend(['--attn_geometry_lr_scale', str(attn_geom_scale)])

    if parse_bool(a.get('freeze_attn_geometry', False)):
        args.append('--freeze_attn_geometry')

    return args

def build_cache_commands(config_path: str, musubi_config_path: str) -> dict:
    """Constructs command strings for latents and text encoder caching."""
    root = resolve_project_root()
    cfg = get_config_data(config_path)

    # Extract params
    model_cfg = cfg.get('model', {})
    strat_cfg = cfg.get('training_strategy', {})
    accel_cfg = cfg.get('acceleration', {})

    # Check for slider dataset (handles multiple config formats)
    dataset_cfg = get_config_data(musubi_config_path)
    is_slider = False
    positive_dir = None
    control_dir = None

    positive_dir = _get_image_directory_from_config(dataset_cfg)
    if positive_dir:
        is_slider, control_dir = _detect_slider_dataset(positive_dir)

    ctx = {
        "dataset": musubi_config_path or 'last_data_musubi_config.toml',
        "ckpt": model_cfg.get('model_path', DEFAULT_LTX2_CHECKPOINT),
        "gemma": model_cfg.get('text_encoder_path', DEFAULT_GEMMA_ROOT),
        "mode": strat_cfg.get('ltx_mode', 'video'),
        "prec": accel_cfg.get('mixed_precision_mode', 'bf16'),
        "gemma_8bit": parse_bool(accel_cfg.get('8_bit_te', True)),
        "latents_script": os.path.join(root, 'diffusion-trainers/musubi-tuner/ltx2_cache_latents.py'),
        "te_script": os.path.join(root, 'diffusion-trainers/musubi-tuner/ltx2_cache_text_encoder_outputs.py')
    }

    if is_slider and positive_dir and control_dir:
        # Slider dataset - two separate configs
        latents_cmd = (
            f"# Slider Dataset: Two separate configs will be created\n"
            f"# 1. Cache positive images:\n"
            f"python {ctx['latents_script']} --dataset_config {ctx['dataset']} "
            f"--ltx2_checkpoint {ctx['ckpt']} --device cuda --vae_dtype bf16 --ltx2_mode {ctx['mode']} --batch_size 1\n\n"
            f"# 2. Cache control images (last_data_musubi_neg_config.toml will be created):\n"
            f"python {ctx['latents_script']} --dataset_config last_data_musubi_neg_config.toml "
            f"--ltx2_checkpoint {ctx['ckpt']} --device cuda --vae_dtype bf16 --ltx2_mode {ctx['mode']} --batch_size 1"
        )
    else:
        latents_cmd = (
            f"python {ctx['latents_script']} --dataset_config {ctx['dataset']} "
            f"--ltx2_checkpoint {ctx['ckpt']} --device cuda --vae_dtype bf16 --ltx2_mode {ctx['mode']} --batch_size 1"
        )

    te_cmd = (
        f"python {ctx['te_script']} --dataset_config {ctx['dataset']} "
        f"--ltx2_checkpoint {ctx['ckpt']} --gemma_root {ctx['gemma']} "
        f"{'--gemma_load_in_8bit ' if ctx['gemma_8bit'] else ''}"
        f"--device cuda --mixed_precision {ctx['prec']} --ltx2_mode {ctx['mode']} --batch_size 1"
    )

    return {"latents": latents_cmd, "text_encoder": te_cmd, "is_slider": is_slider}

# =============================================================================
# == 5. Core Workflow Handlers (Main Entry Points) ============================
# =============================================================================

def run_musubi_ltx2_workflow(
    last_config_path: str,
    musubi_config_path: str,
    mode: str,
    resume_last: bool = False,
    training_console_text=None,
    main_container=None,
    page=None,
    slider_config_path: str = None
):
    """Orchestrates the caching and training flow."""
    
    # Normalize mode to handle legacy/mismatched strings
    mode_map = {
        'cache_only': 'cache_only',
        'trust_cache': 'trust_cache',
        'full': 'full',
        'full_training': 'full' 
    }
    mode = mode_map.get(mode, mode)

    # 1. UI Notification
    msg = {
        "cache_only": "[TODO] Cache only - caching files...",
        "trust_cache": "\n[TODO] Trust cache enabled - starting training...\n",
        "full": "\n[TODO] Start training - creating cache then training...\n"
    }.get(mode, f"\n[Error] Unknown mode: {mode}\n")
    
    _safe_append(training_console_text, msg)
    
    if mode not in ["cache_only", "trust_cache", "full"]:
        logger.error(f"Invalid workflow mode: {mode}")
        update_ui(page)
        return None

    # 2. Command Display
    cmds = build_cache_commands(last_config_path, musubi_config_path)
    if mode == 'cache_only':
        _safe_append(training_console_text, f"\n[Commands]\n{cmds['latents']}\n\n{cmds['text_encoder']}\n")
    
    update_ui(page)

    # 3. Execution Logic
    if mode == 'cache_only':
        return start_chained_caching(last_config_path, musubi_config_path, training_console_text, main_container, page)

    if mode == 'full':
        return start_full_workflow(last_config_path, musubi_config_path, training_console_text, main_container, page, slider_config_path, resume_last)

    if mode == 'trust_cache':
        # Cache sample prompts (with I2V latents if enabled) before training (assumes training dataset TE cache exists)
        cfg = get_config_data(last_config_path)
        v = cfg.get('validation', {})

        # Create sample_prompts.txt file BEFORE cache workflow
        create_sample_prompts_file(last_config_path, musubi_config_path)

        # Regenerate slider config for trust_cache mode to pick up latest settings (like batch_size)
        try:
            slider_config_path = _create_slider_config(musubi_config_path, None, None)
            _safe_append(training_console_text, f"[Slider Config] Updated: {slider_config_path}\n")
        except Exception as e:
            logger.error(f"Failed to update slider config: {e}")
            _safe_append(training_console_text, f"[Warning] Failed to update slider config: {e}\n")

        def on_cache_done():
            train_cmd = ltx2_musubi_trainer_start(last_config_path, musubi_config_path, slider_config_path, resume_last)
            _safe_append(training_console_text, f"\n[Training Command]\n{train_cmd}\n")
            update_ui(page)
            proc = run_ltx2_training(last_config_path, musubi_config_path, slider_config_path, training_console_text, main_container, page, resume_last)
            # Update button state based on whether training started successfully
            if main_container is not None and proc is not None:
                from flet_app.ui.training.start_button_handler import set_button_state
                set_button_state(main_container, "Stop", page)

        if parse_bool(v.get('cache_te', True)) or parse_bool(v.get('cache_i2v', True)):
            _safe_append(training_console_text, "\n[Pre-check] Checking sample prompts cache...\n")
            update_ui(page)
            return run_ltx2_cache_sample_prompts(
                last_config_path, musubi_config_path,
                training_console_text, main_container, page,
                on_complete_callback=on_cache_done
            )
        else:
            # Skip cache, go directly to training
            on_cache_done()
            return None

    return None

def start_chained_caching(last_config_path, musubi_config_path, console, container, page):
    """Starts latents caching with a callback to start TE caching upon success."""

    def on_latents_done():
        _safe_append(console, "\n[Action] Starting text encoder caching...\n")
        update_ui(page)
        run_ltx2_cache_text_encoder(last_config_path, musubi_config_path, console, container, page)

    # Detect if this is a slider dataset for the message
    dataset_cfg = get_config_data(musubi_config_path)
    is_slider = False
    positive_dir = _get_image_directory_from_config(dataset_cfg)
    if positive_dir:
        is_slider, _ = _detect_slider_dataset(positive_dir)

    if is_slider:
        _safe_append(console, "\n[Action] Starting slider latents caching (positive + control)...\n")
    else:
        _safe_append(console, "\n[Action] Starting latents caching...\n")
    update_ui(page)

    return run_ltx2_cache_latents(
        last_config_path, musubi_config_path, console, container, page,
        on_complete_callback=on_latents_done, skip_button_reset=True
    )

def create_sample_prompts_file(last_config_path, musubi_config_path):
    """Create sample_prompts.txt file before caching to ensure it exists for cache generation."""
    cfg = get_config_data(last_config_path)
    v = cfg.get('validation', {})
    m = cfg.get('model', {})

    prompts = v.get('prompts', '')
    negative_prompt = v.get('negative_prompt', '')
    start_images = v.get('start_images', 'none')

    if not (prompts or negative_prompt or start_images):
        return None

    # Build sample prompts content in musubi format
    prompt_parts = []

    if prompts:
        prompt_parts.append(prompts.strip())

    if negative_prompt:
        prompt_parts.append(f"--n {negative_prompt.strip()}")

    sample_steps = v.get('sample_steps', '30')
    guidance_scale = v.get('guidance_scale', '4.0')
    seed = v.get('seed', '42')

    # Get video_dims string - the encoder will validate and adjust to be divisible by 32 and n*8+1
    # Use the same video_dims that's used for training args (validated in _build_ltx2_train_args)
    video_dims = v.get('video_dims', '768, 512, 45')

    prompt_parts.append(f"--g {guidance_scale}")
    prompt_parts.append(f"--s {sample_steps}")
    prompt_parts.append(f"--d {seed}")
    prompt_parts.append(f"--video_dims {video_dims}")

    # Start images with --i prefix
    if start_images and str(start_images).strip().lower() != 'none':
        img_path = start_images.strip()
        if os.path.isabs(img_path):
            img_exists = os.path.exists(img_path)
        else:
            img_exists = os.path.exists(os.path.join(resolve_project_root(), img_path))
        if img_exists:
            prompt_parts.append(f"--i {img_path}")

    if not prompt_parts:
        return None

    # Save to prompts file
    output_dir = m.get('output_dir', DEFAULT_OUTPUT_DIR)
    sample_dir = os.path.join(output_dir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)
    sample_prompts_path = os.path.join(sample_dir, 'sample_prompts.txt')

    sample_prompts_content = " ".join(prompt_parts)
    with open(sample_prompts_path, 'w') as f:
        f.write(sample_prompts_content)

    logger.info(f"Created sample prompts file: {sample_prompts_path}")
    return sample_prompts_path

def start_full_workflow(last_config_path, musubi_config_path, console, container, page, slider_config_path=None, resume_last=False):
    """Starts caching (latents -> TE -> sample prompts -> sample latents) with a callback to start training upon success."""

    # Create sample_prompts.txt file BEFORE cache workflow so it exists for cache generation
    create_sample_prompts_file(last_config_path, musubi_config_path)

    def on_latents_done():
        _safe_append(console, "\n[Action] Starting text encoder caching...\n")
        update_ui(page)

        def on_te_done():
            # Text encoder done - check if we need to cache sample prompts
            cfg = get_config_data(last_config_path)
            v = cfg.get('validation', {})

            def on_sample_prompts_cache_done():
                # Sample prompts cache done (including I2V latents if cache_i2v was enabled)
                # Now start training
                train_cmd = ltx2_musubi_trainer_start(last_config_path, musubi_config_path, slider_config_path, resume_last)
                _safe_append(console, f"\n[Training Command]\n{train_cmd}\n")
                update_ui(page)
                proc = run_ltx2_training(last_config_path, musubi_config_path, slider_config_path, console, container, page, resume_last)
                # Update button state based on whether training started successfully
                if container is not None and proc is not None:
                    from flet_app.ui.training.start_button_handler import set_button_state
                    set_button_state(container, "Stop", page)

            if parse_bool(v.get('cache_te', True)) or parse_bool(v.get('cache_i2v', True)):
                _safe_append(console, "\n[Pre-check] Checking sample prompts cache...\n")
                update_ui(page)
                return run_ltx2_cache_sample_prompts(
                    last_config_path, musubi_config_path,
                    console, container, page,
                    on_complete_callback=on_sample_prompts_cache_done
                )
            else:
                on_sample_prompts_cache_done()
                return None

        run_ltx2_cache_text_encoder(last_config_path, musubi_config_path, console, container, page,
                                     on_complete_callback=on_te_done, skip_button_reset=True)

    # Detect if this is a slider dataset for the message
    dataset_cfg = get_config_data(musubi_config_path)
    is_slider = False
    positive_dir = _get_image_directory_from_config(dataset_cfg)
    if positive_dir:
        is_slider, _ = _detect_slider_dataset(positive_dir)

    if is_slider:
        _safe_append(console, "\n[Action] Starting slider latents caching (positive + control)...\n")
    else:
        _safe_append(console, "\n[Action] Starting latents caching...\n")
    update_ui(page)

    return run_ltx2_cache_latents(
        last_config_path, musubi_config_path, console, container, page,
        on_complete_callback=on_latents_done, skip_button_reset=True
    )

# =============================================================================
# == 6. Process Execution Wrapper =============================================
# =============================================================================

class ProcessWrapper:
    """
    Centralized subprocess wrapper for LTX2 training/caching processes.

    Handles:
    - Process creation with proper environment
    - Process group creation for clean termination
    - Output streaming to UI console
    - Process reference storage
    """

    def __init__(self, root=None):
        """Initialize wrapper with project root."""
        self.root = root or resolve_project_root()

    def _build_env(self, env_extra=None) -> dict:
        """Build environment variables for subprocess."""
        env = os.environ.copy()
        if env_extra:
            current_path = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = env_extra + (os.pathsep + current_path if current_path else '')
        return env

    def _get_creation_flags(self) -> dict:
        """Get platform-specific process creation flags."""
        if os.name == 'posix':
            return {'preexec_fn': os.setsid}
        else:
            return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}

    def execute(self, cmd_list, console, container, page, env_extra=None, **streamer_kwargs):
        """
        Execute a command in a subprocess with output streaming.

        Args:
            cmd_list: Command list to execute
            console: Flet Text control for output
            container: Main container for process reference
            page: Flet page for UI updates
            env_extra: Extra paths to add to PYTHONPATH
            **streamer_kwargs: Additional arguments for output streamer

        Returns:
            subprocess.Popen object or None if failed
        """
        env = self._build_env(env_extra)
        creation_kwargs = self._get_creation_flags()

        try:
            proc = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.root,
                env=env,
                **creation_kwargs
            )

            # Store process reference
            if container:
                container.training_proc = proc

            # Start output streaming
            if console:
                try:
                    from flet_app.ui.training.output_manager import start_ltx_output_streamer
                    start_ltx_output_streamer(proc, console, main_container=container, page=page, **streamer_kwargs)
                except ImportError:
                    logger.error("Could not import output streamer. Check flet_app path.")

            return proc

        except Exception as e:
            logger.error(f"Process failed: {e}")
            _safe_append(console, f"\n[Error] Failed to launch process: {e}\n")
            update_ui(page)
            return None


# Global process wrapper instance
_process_wrapper = ProcessWrapper()

def _execute_subproc(cmd_list, console, container, page, env_extra=None, **streamer_kwargs):
    """
    Internal helper to dry up subprocess creation and streaming.
    Delegates to ProcessWrapper for actual execution.
    """
    return _process_wrapper.execute(cmd_list, console, container, page, env_extra, **streamer_kwargs)

# =============================================================================
# == 7. Dataset & Caching Helpers ==============================================
# =============================================================================

def _get_image_files(directory: str) -> set:
    """Get all image filenames (without extension) from a directory."""
    if not os.path.exists(directory):
        return set()
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
    files = set()
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            files.add(name)
    return files

def _ensure_captions_for_control(control_dir: str, console=None, page=None) -> int:
    """
    Create empty .txt caption files for images in the control directory
    if they do not already exist. Returns count of created files.
    """
    if not os.path.exists(control_dir):
        return 0

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
    created_count = 0

    for filename in os.listdir(control_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            txt_path = os.path.join(control_dir, f"{name}.txt")
            if not os.path.exists(txt_path):
                # Create empty caption file
                with open(txt_path, 'w') as f:
                    f.write("")  # Empty caption
                created_count += 1

    if created_count > 0:
        logger.info(f"Created {created_count} empty caption files in control directory")
        if console:
            _safe_append(console, f"[Setup] Created {created_count} empty caption files for control images\n")
            update_ui(page)

    return created_count

def _create_neg_config(musubi_config_path: str, control_dir: str) -> str:
    """
    Create a separate config file for the control/negative dataset.
    Supports multiple datasets by creating corresponding negative datasets.
    """
    cfg_data = get_config_data(musubi_config_path)

    # Get the first dataset entry to copy resolution settings
    resolution = [512, 512]
    if 'datasets' in cfg_data and len(cfg_data['datasets']) > 0:
        first_ds = cfg_data['datasets'][0]
        if 'resolution' in first_ds:
            resolution = first_ds['resolution']
    elif 'directory' in cfg_data and len(cfg_data['directory']) > 0:
        first_dir = cfg_data['directory'][0]
        if 'resolution' in first_dir:
            resolution = first_dir['resolution']

    # Support multiple datasets - create a negative dataset for each positive one
    neg_datasets = []

    if 'datasets' in cfg_data and len(cfg_data['datasets']) > 0:
        # For each positive dataset, create a corresponding negative dataset
        for i, ds in enumerate(cfg_data['datasets']):
            pos_cache_dir = ds.get('cache_directory', '')
            pos_dir = ds.get('image_directory', ds.get('video_directory', ''))

            # Determine the corresponding negative directory
            # Priority:
            # 1. Check for a 'control' subdirectory within the dataset directory itself
            # 2. Check for a 'control' subdirectory in the parent directory
            # 3. Use the provided control_dir as fallback (for legacy single-dataset case)

            neg_dir = None

            if pos_dir:
                # First, check if there's a 'control' subdirectory within the dataset directory
                # This handles cases like: anatomy/ and anatomy/control/
                potential_control = os.path.join(pos_dir, 'control')
                if os.path.exists(potential_control) and os.path.isdir(potential_control):
                    neg_dir = potential_control
                else:
                    # Second, check if the dataset directory IS a subdirectory of a parent that has 'control'
                    # This handles cases like: datasets/anatomy/ and datasets/control/
                    parent_dir = os.path.dirname(pos_dir)
                    potential_control = os.path.join(parent_dir, 'control')
                    if os.path.exists(potential_control) and os.path.isdir(potential_control):
                        neg_dir = potential_control
                    elif parent_dir != pos_dir:  # Not at root, try one more level up
                        grandparent_dir = os.path.dirname(parent_dir)
                        potential_control = os.path.join(grandparent_dir, 'control')
                        if os.path.exists(potential_control) and os.path.isdir(potential_control):
                            neg_dir = potential_control

            # If no control directory found through directory structure, use provided control_dir
            if neg_dir is None:
                neg_dir = control_dir

            if neg_dir is None:
                logger.warning(f"Could not determine control directory for dataset {i}, skipping")
                continue

            neg_ds = {
                'resolution': ds.get('resolution', resolution),
                'num_repeats': ds.get('num_repeats', 1),
                'cache_directory': os.path.join(neg_dir, 'cache_musubi'),
            }

            # Copy the directory type (image or video)
            if 'image_directory' in ds:
                neg_ds['image_directory'] = neg_dir
            elif 'video_directory' in ds:
                neg_ds['video_directory'] = neg_dir
                if 'target_frames' in ds:
                    neg_ds['target_frames'] = ds['target_frames']
                if 'frame_extraction' in ds:
                    neg_ds['frame_extraction'] = ds['frame_extraction']

            neg_datasets.append(neg_ds)
    else:
        # Legacy single dataset support
        neg_datasets.append({
            'resolution': resolution,
            'image_directory': control_dir,
            'cache_directory': os.path.join(control_dir, 'cache_musubi'),
            'num_repeats': 1
        })

    neg_config = {
        'datasets': neg_datasets,
        'general': {
            'batch_size': 1,
            'enable_bucket': True,
            'bucket_no_upscale': False
        }
    }

    # Copy batch_size and enable_bucket from original config if they exist
    if 'general' in cfg_data:
        original_general = cfg_data['general']
        if 'batch_size' in original_general:
            neg_config['general']['batch_size'] = original_general['batch_size']
        if 'enable_bucket' in original_general:
            neg_config['general']['enable_bucket'] = original_general['enable_bucket']
        if 'bucket_no_upscale' in original_general:
            neg_config['general']['bucket_no_upscale'] = original_general['bucket_no_upscale']

    # Create neg config file next to original
    config_dir = os.path.dirname(musubi_config_path)
    neg_config_path = os.path.join(config_dir, 'last_data_musubi_neg_config.toml')

    with open(neg_config_path, 'w') as f:
        toml.dump(neg_config, f)

    logger.info(f"Created neg config with {len(neg_datasets)} dataset(s): {neg_config_path}")
    return neg_config_path

def _get_image_directory_from_config(dataset_cfg: dict) -> str | None:
    """Extract the image directory path from various config formats."""
    # Format 1: [[directory]] with 'path'
    if 'directory' in dataset_cfg and len(dataset_cfg['directory']) > 0:
        first_dir = dataset_cfg['directory'][0]
        if isinstance(first_dir, dict):
            if 'path' in first_dir:
                return first_dir['path']
            if 'image_directory' in first_dir:
                return first_dir['image_directory']

    # Format 2: [[datasets]] with 'image_directory'
    if 'datasets' in dataset_cfg and len(dataset_cfg['datasets']) > 0:
        first_ds = dataset_cfg['datasets'][0]
        if isinstance(first_ds, dict):
            if 'image_directory' in first_ds:
                return first_ds['image_directory']
            if 'path' in first_ds:
                return first_ds['path']

    # Format 3: top-level 'image_directory'
    if 'image_directory' in dataset_cfg:
        return dataset_cfg['image_directory']

    # Format 4: top-level 'path'
    if 'path' in dataset_cfg:
        return dataset_cfg['path']

    return None

def _detect_slider_dataset(dataset_dir: str) -> tuple[bool, str | None]:
    """
    Check if dataset directory has a 'control' subdirectory with matching images.

    Returns (is_slider, control_dir_path)
    """
    if not os.path.exists(dataset_dir):
        return False, None

    control_dir = os.path.join(dataset_dir, 'control')
    if not os.path.exists(control_dir) or not os.path.isdir(control_dir):
        return False, None

    positive_files = _get_image_files(dataset_dir)
    control_files = _get_image_files(control_dir)

    if not positive_files or not control_files:
        return False, None

    # Check for matching filenames (at least some overlap)
    matching_files = positive_files & control_files
    if len(matching_files) == 0:
        return False, None

    logger.info(f"Detected slider dataset: {len(matching_files)} matching image pairs")
    return True, control_dir


def _find_all_control_dirs(musubi_config_path: str) -> list:
    """
    Find all control directories for a multi-dataset musubi config.

    Returns a list of control directory paths that have matching image files.
    """
    dataset_cfg = get_config_data(musubi_config_path)
    control_dirs = []

    if 'datasets' in dataset_cfg and len(dataset_cfg['datasets']) > 0:
        for ds in dataset_cfg['datasets']:
            pos_dir = ds.get('image_directory', ds.get('video_directory', ''))
            if not pos_dir:
                continue

            # Check for control subdirectory within the dataset directory
            potential_control = os.path.join(pos_dir, 'control')
            if os.path.exists(potential_control) and os.path.isdir(potential_control):
                # Verify there are matching images
                pos_files = _get_image_files(pos_dir)
                control_files = _get_image_files(potential_control)
                if pos_files and control_files and (pos_files & control_files):
                    control_dirs.append(potential_control)

    return control_dirs

def _create_slider_config(musubi_config_path: str, positive_dir: str, control_dir: str) -> str:
    """
    Create a slider config file next to the musubi dataset config.

    The config contains paths to the cache directories that will be created
    during caching. Supports multiple datasets.

    Args:
        musubi_config_path: Path to the musubi config file
        positive_dir: Primary positive directory (legacy/single dataset)
        control_dir: Primary control/negative directory (legacy/single dataset)

    Returns:
        Path to the created slider config file
    """
    cfg_data = get_config_data(musubi_config_path)

    # Cache directories are inside each image directory
    # Support multiple datasets
    pos_cache_dirs = []
    neg_cache_dirs = []
    text_cache_dirs = []

    if 'datasets' in cfg_data and len(cfg_data['datasets']) > 0:
        # Multi-dataset support
        for ds in cfg_data['datasets']:
            pos_dir = ds.get('image_directory', ds.get('video_directory', ''))
            pos_cache_dir = ds.get('cache_directory', os.path.join(pos_dir, 'cache'))

            # Determine the corresponding negative directory
            # Priority:
            # 1. Check for a 'control' subdirectory within the dataset directory itself
            # 2. Check for a 'control' subdirectory in the parent directory
            # 3. Use the provided control_dir as fallback (for legacy single-dataset case)

            neg_dir = None

            if pos_dir:
                # First, check if there's a 'control' subdirectory within the dataset directory
                potential_control = os.path.join(pos_dir, 'control')
                if os.path.exists(potential_control) and os.path.isdir(potential_control):
                    neg_dir = potential_control
                else:
                    # Second, check if the dataset directory IS a subdirectory of a parent that has 'control'
                    parent_dir = os.path.dirname(pos_dir)
                    potential_control = os.path.join(parent_dir, 'control')
                    if os.path.exists(potential_control) and os.path.isdir(potential_control):
                        neg_dir = potential_control
                    elif parent_dir != pos_dir:  # Not at root, try one more level up
                        grandparent_dir = os.path.dirname(parent_dir)
                        potential_control = os.path.join(grandparent_dir, 'control')
                        if os.path.exists(potential_control) and os.path.isdir(potential_control):
                            neg_dir = potential_control

            # If no control directory found through directory structure, use provided control_dir
            if neg_dir is None:
                neg_dir = control_dir

            # Skip if no valid control directory found
            if neg_dir is None:
                logger.warning(f"Could not determine control directory for dataset with pos_dir={pos_dir}, skipping from slider config")
                continue

            # Use the same cache directory name as the positive cache
            cache_dir_name = os.path.basename(pos_cache_dir)
            neg_cache_dir = os.path.join(neg_dir, cache_dir_name)

            pos_cache_dirs.append(pos_cache_dir)
            neg_cache_dirs.append(neg_cache_dir)
            text_cache_dirs.append(pos_cache_dir)  # Text cache shared with positive
    else:
        # No datasets found, create minimal config from provided directories
        pos_cache_dir = os.path.join(positive_dir, 'cache_musubi')
        ctrl_cache_dir = os.path.join(control_dir, 'cache_musubi')

        pos_cache_dirs = [pos_cache_dir]
        neg_cache_dirs = [ctrl_cache_dir]
        text_cache_dirs = [pos_cache_dir]

    # Get batch_size from original config (default to 1 if not specified)
    batch_size = 1
    if 'general' in cfg_data and 'batch_size' in cfg_data['general']:
        batch_size = cfg_data['general']['batch_size']

    # Preserve existing sample_slider_range if slider config already exists
    sample_slider_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
    config_dir = os.path.dirname(musubi_config_path)
    existing_slider_config_path = os.path.join(config_dir, 'last_data_musubi_slider_config.toml')
    if os.path.exists(existing_slider_config_path):
        try:
            with open(existing_slider_config_path, 'r') as f:
                existing_slider_config = toml.load(f)
                existing_range = existing_slider_config.get('sample_slider_range')
                if existing_range is not None:
                    sample_slider_range = existing_range
        except Exception:
            pass  # Fall back to default

    slider_config = {
        'mode': 'reference',
        'pos_cache_dirs': pos_cache_dirs,
        'neg_cache_dirs': neg_cache_dirs,
        'text_cache_dirs': text_cache_dirs,
        'sample_slider_range': sample_slider_range,
        'batch_size': batch_size,
    }

    # Create config file next to musubi_config_path
    config_dir = os.path.dirname(musubi_config_path)
    slider_config_path = os.path.join(config_dir, 'last_data_musubi_slider_config.toml')

    with open(slider_config_path, 'w') as f:
        toml.dump(slider_config, f)

    logger.info(f"Created slider config with {len(pos_cache_dirs)} dataset(s): {slider_config_path}")
    return slider_config_path

def run_ltx2_cache_latents(last_config_path, musubi_config_path, console, container, page, **kwargs):
    cfg = get_config_data(last_config_path)
    model_path = cfg.get('model', {}).get('model_path', DEFAULT_LTX2_CHECKPOINT)
    ltx_mode = cfg.get('training_strategy', {}).get('ltx_mode', 'video')

    # Check if this is a slider dataset
    dataset_cfg = get_config_data(musubi_config_path)
    is_slider = False
    positive_dir = None
    control_dir = None

    # Get the image directory from dataset config (handles multiple formats)
    positive_dir = _get_image_directory_from_config(dataset_cfg)
    if positive_dir:
        is_slider, control_dir = _detect_slider_dataset(positive_dir)

    if is_slider and positive_dir and control_dir:
        _safe_append(console, "\n[Slider Dataset] Detected positive/control image pairs\n")
        update_ui(page)

        # 1. Find ALL control directories for multi-dataset support
        all_control_dirs = _find_all_control_dirs(musubi_config_path)
        if len(all_control_dirs) > 1:
            _safe_append(console, f"[Slider Dataset] Found {len(all_control_dirs)} dataset(s) with control directories\n")
        update_ui(page)

        # 2. Create separate neg config for all control directories (caption files not required for control images)
        neg_config_path = _create_neg_config(musubi_config_path, control_dir)
        _safe_append(console, f"[Config] Created: {neg_config_path}\n")
        _safe_append(console, f"[Info] Control images don't require caption files for slider training\n")
        update_ui(page)

        # 3. Cache positive dataset first
        script = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/ltx2_cache_latents.py')
        cmd_pos = ['python', script, '--dataset_config', musubi_config_path, '--ltx2_checkpoint', model_path,
                   '--device', 'cuda', '--vae_dtype', 'bf16', '--ltx2_mode', ltx_mode]

        _safe_append(console, f"[Caching] Caching positive images from {len(all_control_dirs)} dataset(s)\n")
        update_ui(page)

        musubi_src = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/src')
        on_complete = kwargs.get('on_complete_callback')
        skip_reset = kwargs.get('skip_button_reset', False)

        def on_positive_done():
            # 4. Cache control/negative dataset(s)
            cmd_neg = ['python', script, '--dataset_config', neg_config_path, '--ltx2_checkpoint', model_path,
                       '--device', 'cuda', '--vae_dtype', 'bf16', '--ltx2_mode', ltx_mode]

            _safe_append(console, f"[Caching] Caching control images from {len(all_control_dirs)} dataset(s)\n")
            update_ui(page)

            def on_negative_done():
                # 5. Create slider config file
                try:
                    slider_config_path = _create_slider_config(musubi_config_path, positive_dir, control_dir)
                    _safe_append(console, f"[Slider Config] Created: {slider_config_path}\n")
                    update_ui(page)
                except Exception as e:
                    logger.error(f"Failed to create slider config: {e}")
                    _safe_append(console, f"[Warning] Failed to create slider config: {e}\n")
                    update_ui(page)

                if on_complete:
                    on_complete()

            _execute_subproc(cmd_neg, console, container, page, env_extra=musubi_src,
                           on_complete_callback=on_negative_done, skip_button_reset=skip_reset)

        return _execute_subproc(cmd_pos, console, container, page, env_extra=musubi_src,
                               on_complete_callback=on_positive_done, skip_button_reset=True)

    # Standard single-dataset caching
    script = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/ltx2_cache_latents.py')
    cmd = ['python', script, '--dataset_config', musubi_config_path, '--ltx2_checkpoint', model_path,
           '--device', 'cuda', '--vae_dtype', 'bf16', '--ltx2_mode', ltx_mode]

    musubi_src = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/src')
    return _execute_subproc(cmd, console, container, page, env_extra=musubi_src, **kwargs)

def run_ltx2_cache_text_encoder(last_config_path, musubi_config_path, console, container, page, **kwargs):
    """Cache text encoder outputs for training dataset captions."""
    cfg = get_config_data(last_config_path)
    accel = cfg.get('acceleration', {})
    strat_cfg = cfg.get('training_strategy', {})

    script = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/ltx2_cache_text_encoder_outputs.py')
    cmd = [
        'python', script, '--dataset_config', musubi_config_path,
        '--ltx2_checkpoint', cfg.get('model', {}).get('model_path', DEFAULT_LTX2_CHECKPOINT),
        '--gemma_root', cfg.get('model', {}).get('text_encoder_path', DEFAULT_GEMMA_ROOT),
        '--device', 'cuda', '--mixed_precision', accel.get('mixed_precision_mode', 'bf16'),
        '--ltx2_mode', strat_cfg.get('ltx_mode', 'video'), '--batch_size', '1'
    ]

    if parse_bool(accel.get('8_bit_te', True)):
        cmd.append('--gemma_load_in_8bit')

    musubi_src = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/src')
    return _execute_subproc(cmd, console, container, page, env_extra=musubi_src, **kwargs)


def _get_prompts_hash(cfg: dict) -> str:
    """
    Compute a hash of the prompts configuration for cache validation.
    Includes prompts, negative_prompt, start_images, and sampling parameters that affect encoding.
    """
    import hashlib
    v = cfg.get('validation', {})
    # Include all fields that affect the encoded output
    hash_data = {
        'prompts': v.get('prompts', ''),
        'negative_prompt': v.get('negative_prompt', ''),
        'start_images': v.get('start_images', 'none'),
        'sample_steps': v.get('sample_steps', '30'),
        'guidance_scale': v.get('guidance_scale', '4.0'),
    }
    # Create a stable string representation
    hash_str = str(sorted(hash_data.items()))
    return hashlib.sha256(hash_str.encode()).hexdigest()


def run_ltx2_cache_sample_prompts(
    last_config_path: str,
    musubi_config_path: str,
    console: ft.Text = None,
    container=None,
    page: ft.Page = None,
    on_complete_callback=None,
    skip_button_reset=False,
) -> bool:
    """
    Generate pre-encoded sample prompts cache to avoid loading Gemma during sampling.

    Runs in a separate subprocess with output streaming to avoid blocking UI.

    Returns True if process was started successfully, False otherwise.
    """
    cfg = get_config_data(last_config_path)
    v = cfg.get('validation', {})
    m = cfg.get('model', {})
    accel = cfg.get('acceleration', {})

    # Check if sampling is enabled and prompts are configured
    prompts = v.get('prompts', '')
    sample_interval = v.get('interval', '-1')
    sample_at_first = parse_bool(v.get('sample_at_first', 'false'))
    sampling_enabled = (sample_interval and str(sample_interval) != '-1') or sample_at_first

    if not sampling_enabled or not prompts:
        logger.info("Sampling not enabled or no prompts configured, skipping sample prompt cache")
        if on_complete_callback:
            on_complete_callback()
        return True

    # Check if cache already exists
    output_dir = m.get('output_dir', DEFAULT_OUTPUT_DIR)
    sample_dir = os.path.join(output_dir, 'sample')
    sample_prompts_path = os.path.join(sample_dir, 'sample_prompts.txt')

    if not os.path.exists(sample_prompts_path):
        logger.info(f"Sample prompts file not found: {sample_prompts_path}, skipping cache")
        if on_complete_callback:
            on_complete_callback()
        return False

    # Cache file path (same directory as sample_prompts.txt)
    cache_path = os.path.join(sample_dir, 'sample_prompts_cache.pt')
    hash_path = os.path.join(sample_dir, 'sample_prompts_hash.txt')

    # Check if cache is valid (exists and prompts haven't changed)
    current_hash = _get_prompts_hash(cfg)
    cache_valid = False

    if os.path.exists(cache_path) and os.path.exists(hash_path):
        try:
            with open(hash_path, 'r') as f:
                stored_hash = f.read().strip()
            if stored_hash == current_hash:
                cache_valid = True
                msg = f"[Cache] Using valid sample prompts cache: {cache_path}\n"
                _safe_append(console, msg, color='green')
                update_ui(page)
                logger.info(f"Sample prompts cache is valid (hash matches): {cache_path}")
            else:
                msg = f"[Cache] Prompts changed, regenerating cache...\n"
                _safe_append(console, msg, color='yellow')
                update_ui(page)
                logger.info(f"Prompts hash changed, regenerating cache (old: {stored_hash[:8]}..., new: {current_hash[:8]}...)")
                # Delete old cache and hash
                os.remove(cache_path)
                os.remove(hash_path)
        except Exception as e:
            logger.warning(f"Error reading cache hash, regenerating: {e}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(hash_path):
                os.remove(hash_path)

    if cache_valid:
        if on_complete_callback:
            on_complete_callback()
        return True

    # Create sample directory and write prompts hash file
    os.makedirs(sample_dir, exist_ok=True)
    with open(hash_path, 'w') as f:
        f.write(current_hash)

    # Build the caching command
    script = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/ltx2_cache_text_encoder_outputs.py')
    cmd = [
        'python', script,
        '--dataset_config', musubi_config_path,
        '--ltx2_checkpoint', m.get('model_path', DEFAULT_LTX2_CHECKPOINT),
        '--gemma_root', m.get('text_encoder_path', DEFAULT_GEMMA_ROOT),
        '--device', 'cuda',
        '--mixed_precision', accel.get('mixed_precision_mode', 'bf16'),
        '--ltx2_mode', cfg.get('training_strategy', {}).get('ltx_mode', 'video'),
        '--batch_size', '1',
        '--precache_sample_prompts',
        '--sample_prompts', sample_prompts_path,
        '--sample_prompts_cache', cache_path,
    ]

    if parse_bool(accel.get('8_bit_te', True)):
        cmd.append('--gemma_load_in_8bit')

    # Add cache_i2v flag if enabled in validation config
    if parse_bool(v.get('cache_i2v', True)):
        cmd.append('--cache_i2v')

    msg = f"\n[Cache] Generating sample prompts cache...\n"
    msg += f"[Cache] This will encode prompts once to avoid loading Gemma during sampling.\n"
    _safe_append(console, msg)
    update_ui(page)

    # Run in separate subprocess with output streaming
    musubi_src = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/src')
    return _execute_subproc(
        cmd, console, container, page,
        env_extra=musubi_src,
        on_complete_callback=on_complete_callback,
        skip_button_reset=skip_button_reset,
        completion_message=f"[Success] Sample prompts cache created: {cache_path}\n"
    )

def run_ltx2_cache_sample_latents(
    last_config_path: str,
    musubi_config_path: str,
    console: ft.Text = None,
    container=None,
    page: ft.Page = None,
    on_complete_callback=None,
    skip_button_reset=False,
) -> bool:
    """
    I2V latents caching is now handled by run_ltx2_cache_sample_prompts with --cache_i2v flag.
    This function is kept for compatibility but skips the caching step.
    """
    # I2V latents are now cached in sample_prompts_cache.pt when --cache_i2v is enabled
    logger.info("I2V latents caching is now handled by sample prompts cache with --cache_i2v")
    if on_complete_callback:
        on_complete_callback()
    return True


# =============================================================================
# == 8. Training Operations ===================================================
# =============================================================================

def run_ltx2_training(last_config_path, musubi_config_path, slider_config_path=None, console=None, container=None, page=None, resume_last=False, **kwargs):
    """Executes the LTX2 training script via accelerate launch using centralized arg builder."""
    cfg = get_config_data(last_config_path)

    # Base accelerate command
    cmd = ['accelerate', 'launch', '--num_cpu_threads_per_process', '4']

    # Append generated args (pass slider_config_path for slider training)
    script_args = _build_ltx2_train_args(cfg, musubi_config_path, slider_config_path, console, page, resume_last)
    cmd.extend(script_args)

    musubi_src = os.path.join(resolve_project_root(), 'diffusion-trainers/musubi-tuner/src')
    return _execute_subproc(cmd, console, container, page, env_extra=musubi_src, completion_message="[Info] Training completed.\n", **kwargs)

def ltx2_musubi_trainer_start(last_config_path, musubi_config_path, slider_config_path=None, output_name=None, resume_last=False):
    """Constructs the full accelerate training command string for display."""
    cfg = get_config_data(last_config_path)

    # Use the same builder as the execution function to ensure consistency
    script_args = _build_ltx2_train_args(cfg, musubi_config_path, slider_config_path, resume_last=resume_last)

    # Prepend accelerate launch
    full_cmd = ['accelerate', 'launch', '--num_cpu_threads_per_process', '4'] + script_args

    # Format with each flag on a new line for readability
    formatted_lines = []
    i = 0
    while i < len(full_cmd):
        arg = full_cmd[i]
        # If this is a flag that takes a value, include both on the same line
        if i + 1 < len(full_cmd) and not full_cmd[i + 1].startswith('-'):
            formatted_lines.append(f"  {arg} {full_cmd[i + 1]} \\")
            i += 2
        else:
            formatted_lines.append(f"  {arg} \\")
            i += 1

    # Remove trailing \ from last line
    if formatted_lines:
        formatted_lines[-1] = formatted_lines[-1].rstrip(' \\')

    return "\\\n".join(formatted_lines)

# =============================================================================
# == 9. Legacy Compatibility Stubs ===========================================
# =============================================================================

def handle_musubi_model(p): return is_musubi_model(p)
def handle_musubi_training_mode(c=False, t=False): return "cache_only" if c else "trust_cache" if t else "full"