"""
Configuration utilities for Flet training UI.

This package provides:
- TOML building from UI controls
- TOML loading into UI controls
- Formatting and path utilities
- Constants and type helpers
"""

# Main API functions
from .toml_builder import (
    build_toml_config_from_ui,
    extract_config_from_controls,
)

from .toml_loader import (
    update_ui_from_toml,
    apply_values_recursive,
    apply_all_values,
)

# Formatting utilities
from .toml_formatting import (
    quote,
    to_bool,
    toml_bool,
    normalize_slashes,
    is_absolute_path,
    expand_model_path,
    collapse_model_path,
    collapse_path_to_relative,
    resolve_output_dir,
    resolve_path_if_relative,
)

# Constants and helpers
from .constants import (
    MUSUBI_MODEL_TYPES,
    LTX_MODEL_TYPES,
    WAN_MODEL_TYPES,
    Trainers,
    is_musubi_trainer,
    is_musubi_model,
    is_ltx_model,
    is_wan_model,
    ALWAYS_INCLUDE_FIELDS,
    DEFAULTS,
)

__all__ = [
    # Builder
    'build_toml_config_from_ui',
    'extract_config_from_controls',
    # Loader
    'update_ui_from_toml',
    'apply_values_recursive',
    'apply_all_values',
    # Formatting
    'quote',
    'to_bool',
    'toml_bool',
    'normalize_slashes',
    'is_absolute_path',
    'expand_model_path',
    'collapse_model_path',
    'collapse_path_to_relative',
    'resolve_output_dir',
    'resolve_path_if_relative',
    # Constants
    'MUSUBI_MODEL_TYPES',
    'LTX_MODEL_TYPES',
    'WAN_MODEL_TYPES',
    'Trainers',
    'is_musubi_trainer',
    'is_musubi_model',
    'is_ltx_model',
    'is_wan_model',
    'ALWAYS_INCLUDE_FIELDS',
    'DEFAULTS',
]
