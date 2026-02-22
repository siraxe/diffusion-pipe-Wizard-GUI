"""
Musubi-specific configuration utilities for training UI.

This module acts as a compatibility wrapper around the refactored musubi package.
All functionality has been moved to the config/musubi/ subpackage for better organization.

Musubi-specific fields:
- optimizer_type_m: Dropdown optimizer selection (AdamW, AdamW8bit, Adafactor, Prodigy, Automagic)
- optimizer_args: Text field for optimizer arguments (shown when Automagic is selected)
- mixed_precision_mode: Dropdown for precision mode (no, fp16, bf16)
- fp8_base: Checkbox for FP8 base training
- fp8_scaled: Checkbox for FP8 scaled training
- attn_chunking: Checkbox for attention chunking
"""

# Re-export all from the musubi package
from .config.musubi import (
    # Optimizer
    MUSUBI_OPTIMIZER_TYPE_MAP,
    MUSUBI_OPTIMIZER_TYPE_MAP_REVERSE,
    get_musubi_optimizer_type_for_toml,
    get_musubi_optimizer_type_for_ui,
    is_automagic_optimizer,
    populate_musubi_optimization_section,
    get_automagic_optimizer_args_default,
    build_musubi_optimizer_args_line,
    # Visibility
    set_musubi_field_visibility,
    set_musubi_field_value,
    set_optimizer_args_visibility,
    set_musubi_precision_fields_visibility,
    trigger_musubi_optimizer_change,
    # Acceleration
    get_musubi_precision_defaults,
    populate_musubi_acceleration_section,
    append_musubi_acceleration_section,
    # Model
    populate_musubi_model_section,
    append_musubi_model_section,
    # Main
    update_musubi_ui_from_toml,
)


# Legacy quote function (for backward compatibility)
def quote(value):
    """Quote a string value for TOML output."""
    if value is None:
        return "''"
    return f"'{value}'"


__all__ = [
    # Optimizer
    'MUSUBI_OPTIMIZER_TYPE_MAP',
    'MUSUBI_OPTIMIZER_TYPE_MAP_REVERSE',
    'get_musubi_optimizer_type_for_toml',
    'get_musubi_optimizer_type_for_ui',
    'is_automagic_optimizer',
    'populate_musubi_optimization_section',
    'get_automagic_optimizer_args_default',
    'build_musubi_optimizer_args_line',
    # Visibility
    'set_musubi_field_visibility',
    'set_musubi_field_value',
    'set_optimizer_args_visibility',
    'set_musubi_precision_fields_visibility',
    'trigger_musubi_optimizer_change',
    # Acceleration
    'get_musubi_precision_defaults',
    'populate_musubi_acceleration_section',
    'append_musubi_acceleration_section',
    # Model
    'populate_musubi_model_section',
    'append_musubi_model_section',
    # Main
    'update_musubi_ui_from_toml',
    # Legacy
    'quote',
]
