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

    Returns True if the section was appended.
    """
    mixed_precision_mode = _get_func('mixed_precision_mode', 'bf16')
    fp8_base = _get_func('fp8_base', True)
    fp8_scaled = _get_func('fp8_scaled', True)
    attn_chunking = _get_func('attn_chunking', False)

    lines.append("")
    lines.append("[acceleration]")
    lines.append(f"mixed_precision_mode = '{mixed_precision_mode}'")

    lines.append(f"fp8_base = {'true' if fp8_base else 'false'}")
    lines.append(f"fp8_scaled = {'true' if fp8_scaled else 'false'}")
    lines.append(f"attn_chunking = {'true' if attn_chunking else 'false'}")

    return True
