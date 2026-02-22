"""
Configuration utilities for training UI.

This module acts as a compatibility wrapper around the refactored config package.
All functionality has been moved to the config/ subpackage for better organization.

Main functions:
- build_toml_config_from_ui: Build TOML from UI controls
- update_ui_from_toml: Load TOML into UI controls
"""

# Re-export main API from the config package
from .config import (
    # Main functions
    build_toml_config_from_ui,
    update_ui_from_toml,
    extract_config_from_controls,
    apply_values_recursive,
    apply_all_values,
    # Formatting utilities
    quote,
    to_bool,
    toml_bool,
    normalize_slashes,
    is_absolute_path,
    expand_model_path,
    collapse_model_path,
    collapse_path_to_relative,
    # Constants
    ALWAYS_INCLUDE_FIELDS,
    DEFAULTS,
    Trainers,
    is_musubi_trainer,
    is_musubi_model,
    is_ltx_model,
)

# Legacy aliases for backward compatibility
_normalize_slashes = normalize_slashes
_is_absolute_path = is_absolute_path
_quote = quote
_to_bool = to_bool
_collapse_path_to_relative = collapse_path_to_relative

# Image processing (kept here for backward compatibility)
def _process_and_save_image(source_image_path, video_dims_tuple, dataset_name, target_filename, dataset_type="video"):
    """Process and save image - delegates to image_utils module."""
    from .image_utils import process_and_save_image
    return process_and_save_image(source_image_path, video_dims_tuple, dataset_name, target_filename, dataset_type)


def _save_and_scale_image(source_image_path: str, video_dims_tuple: tuple, dataset_name: str,
                          target_filename: str, dataset_type: str = "video",
                          page=None, target_control: str = None,
                          image_display_c1=None, image_display_c2=None):
    """Save and scale image - delegates to image_utils module."""
    from .image_utils import save_and_scale_image
    return save_and_scale_image(
        source_image_path, video_dims_tuple, dataset_name, target_filename, dataset_type,
        page, target_control, image_display_c1, image_display_c2
    )


__all__ = [
    # Main API
    'build_toml_config_from_ui',
    'update_ui_from_toml',
    'extract_config_from_controls',
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
    # Image processing
    '_process_and_save_image',
    '_save_and_scale_image',
    # Legacy aliases
    '_normalize_slashes',
    '_is_absolute_path',
    '_quote',
    '_to_bool',
    '_collapse_path_to_relative',
]
