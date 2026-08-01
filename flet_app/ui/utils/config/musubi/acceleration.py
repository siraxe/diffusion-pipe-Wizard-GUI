"""
Musubi acceleration section handling for training UI.

Handles precision fields and acceleration section building/loading.
"""

from typing import Any, Dict, List


# =============================================================================
# Default Values
# =============================================================================

def get_musubi_precision_defaults() -> Dict[str, Any]:
    """Get default values for musubi precision fields."""
    return {
        'mixed_precision_mode': 'bf16',
        'fp8_base': True,
        'fp8_scaled': True,
        'attn_chunking': False,
        'flash_attn': True,
    }


# =============================================================================
# Population Functions
# =============================================================================

def populate_musubi_acceleration_section(toml_data: Dict, label_vals: Dict, to_bool_func) -> None:
    """Populate musubi-specific acceleration fields from TOML."""
    acceleration = toml_data.get('acceleration', {}) or {}
    if not isinstance(acceleration, dict):
        return

    if 'mixed_precision_mode' in acceleration:
        label_vals['mixed_precision_mode'] = acceleration.get('mixed_precision_mode')
    if 'fp8_base' in acceleration:
        label_vals['fp8_base'] = to_bool_func(acceleration.get('fp8_base', True))
    if 'fp8_scaled' in acceleration:
        label_vals['fp8_scaled'] = to_bool_func(acceleration.get('fp8_scaled', True))
    if 'attn_chunking' in acceleration:
        label_vals['attn_chunking'] = to_bool_func(acceleration.get('attn_chunking', False))
    if 'flash_attn' in acceleration:
        label_vals['flash_attn'] = to_bool_func(acceleration.get('flash_attn', True))

    # Advanced options checkboxes and their args
    if 'blank_preservation' in acceleration:
        label_vals['blank_preservation'] = to_bool_func(acceleration.get('blank_preservation', False))
    if 'blank_preservation_args' in acceleration:
        label_vals['blank_preservation_args'] = acceleration.get('blank_preservation_args', 'multiplier=0.5')
    if 'dop' in acceleration:
        label_vals['dop'] = to_bool_func(acceleration.get('dop', False))
    if 'dop_args' in acceleration:
        label_vals['dop_args'] = acceleration.get('dop_args', 'class=woman multiplier=1.0')
    if 'prior_divergence' in acceleration:
        label_vals['prior_divergence'] = to_bool_func(acceleration.get('prior_divergence', False))
    if 'prior_divergence_args' in acceleration:
        label_vals['prior_divergence_args'] = acceleration.get('prior_divergence_args', 'multiplier=0.1')
    if 'crepa' in acceleration:
        label_vals['crepa'] = to_bool_func(acceleration.get('crepa', False))
    if 'crepa_mode' in acceleration:
        label_vals['crepa_mode'] = acceleration.get('crepa_mode', 'backbone')
    if 'crepa_args' in acceleration:
        label_vals['crepa_args'] = acceleration.get('crepa_args', 'student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2')
    # self_flow and cts_lambda
    if 'self_flow' in acceleration:
        label_vals['self_flow'] = to_bool_func(acceleration.get('self_flow', False))
    if 'self_flow_args' in acceleration:
        label_vals['self_flow_args'] = acceleration.get('self_flow_args', 'teacher_mode=base student_block_ratio=0.3 teacher_block_ratio=0.7 lambda_self_flow=0.1')
    if 'cts_lambda' in acceleration:
        label_vals['cts_lambda'] = to_bool_func(acceleration.get('cts_lambda', False))
    if 'cts_lambda_args' in acceleration:
        label_vals['cts_lambda_args'] = acceleration.get('cts_lambda_args', 'video_driven=0.3 audio_driven=0.1')
    if 'forward_xm' in acceleration:
        label_vals['forward_xm'] = to_bool_func(acceleration.get('forward_xm', False))
    if 'forward_xm_args' in acceleration:
        label_vals['forward_xm_args'] = acceleration.get('forward_xm_args', '2')


# =============================================================================
# TOML Building
# =============================================================================

def append_musubi_acceleration_section(lines: List[str], _get_func) -> bool:
    """
    Append [acceleration] section for musubi models to TOML lines.

    Handles musubi-specific precision fields:
    - mixed_precision_mode
    - fp8_base
    - fp8_scaled
    - attn_chunking
    - flash_attn

    Returns True if the section was appended.
    """
    mixed_precision_mode = _get_func('mixed_precision_mode', 'bf16')
    fp8_base = _get_func('fp8_base', True)
    fp8_scaled = _get_func('fp8_scaled', True)
    attn_chunking = _get_func('attn_chunking', False)
    flash_attn = _get_func('flash_attn', True)

    lines.append("")
    lines.append("[acceleration]")
    lines.append(f"mixed_precision_mode = '{mixed_precision_mode}'")

    lines.append(f"fp8_base = {'true' if fp8_base else 'false'}")
    lines.append(f"fp8_scaled = {'true' if fp8_scaled else 'false'}")
    lines.append(f"attn_chunking = {'true' if attn_chunking else 'false'}")
    lines.append(f"flash_attn = {'true' if flash_attn else 'false'}")

    return True
