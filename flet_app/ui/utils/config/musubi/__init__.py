"""
Musubi configuration utilities for training UI.

This package handles configuration for all musubi trainer models (LTX2, WAN22, etc.)
to avoid code duplication across different model-specific config utilities.

Musubi-specific fields:
- optimizer_type_m: Dropdown optimizer selection (AdamW, AdamW8bit, Adafactor, Prodigy, Automagic)
- optimizer_args: Text field for optimizer arguments (shown when Automagic is selected)
- mixed_precision_mode: Dropdown for precision mode (no, fp16, bf16)
- fp8_base: Checkbox for FP8 base training
- fp8_scaled: Checkbox for FP8 scaled training
- attn_chunking: Checkbox for attention chunking
"""

from typing import Any, Dict

# Optimizer utilities
from .optimizer import (
    MUSUBI_OPTIMIZER_TYPE_MAP,
    MUSUBI_OPTIMIZER_TYPE_MAP_REVERSE,
    get_musubi_optimizer_type_for_toml,
    get_musubi_optimizer_type_for_ui,
    is_automagic_optimizer,
    populate_musubi_optimization_section,
    get_automagic_optimizer_args_default,
    build_musubi_optimizer_args_line,
)

# Visibility management
from .visibility import (
    set_musubi_field_visibility,
    set_musubi_field_value,
    set_optimizer_args_visibility,
    set_musubi_precision_fields_visibility,
    trigger_musubi_optimizer_change,
)

# Acceleration handling
from .acceleration import (
    get_musubi_precision_defaults,
    populate_musubi_acceleration_section,
    append_musubi_acceleration_section,
)

# Model handling
from .model import (
    populate_musubi_model_section,
    append_musubi_model_section,
)


# =============================================================================
# Main UI Update Function
# =============================================================================

def update_musubi_ui_from_toml(training_tab_container: Any, toml_data: Dict) -> None:
    """
    Update musubi-specific UI fields from TOML data.

    This function handles all musubi trainer models (LTX2, WAN22, etc.)
    to ensure consistent behavior for optimizer_args visibility and other
    musubi-specific fields.

    Args:
        training_tab_container: The training tab container
        toml_data: The loaded TOML data dictionary
    """
    try:
        page = getattr(training_tab_container, 'page', None)

        # Populate model section fields (wan_task, etc.)
        musubi_section = getattr(training_tab_container, 'musubi_custom_section', None)
        if musubi_section:
            model_fields = {}
            populate_musubi_model_section(toml_data, model_fields)
            if model_fields:
                for field_name, field_value in model_fields.items():
                    set_musubi_field_value(musubi_section, field_name, field_value, page)

        # Get optimization section
        optimization = toml_data.get('optimization', {}) or {}

        # Get optimizer type and trigger optimizer change for optimizer_args visibility
        optimizer_type = optimization.get('optimizer_type', '')
        if optimizer_type:
            mapped_type = get_musubi_optimizer_type_for_ui(optimizer_type)
            trigger_musubi_optimizer_change(training_tab_container, mapped_type, page)

        # Sync factor field visibility based on adapter (lokr shows, lora hides)
        # This should happen regardless of optimization section
        try:
            from flet_app.ui.pages.training_config import sync_dependent_field_visibility
            sync_dependent_field_visibility()
        except Exception:
            pass

    except Exception:
        pass  # Silently fail to avoid breaking config loading


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
]
