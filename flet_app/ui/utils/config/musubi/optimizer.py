"""
Musubi optimizer handling for training UI.

Handles optimizer type mapping, optimizer_args building, and defaults.
"""

from typing import Dict, Any, Optional


# =============================================================================
# Optimizer Type Mapping
# =============================================================================

MUSUBI_OPTIMIZER_TYPE_MAP = {
    'adamw': 'AdamW',
    'adamw8bit': 'AdamW8bit',
    'adafactor': 'Adafactor',
    'prodigy': 'Prodigy',
    'automagic': 'Automagic',
}

MUSUBI_OPTIMIZER_TYPE_MAP_REVERSE = {v: k for k, v in MUSUBI_OPTIMIZER_TYPE_MAP.items()}


def get_musubi_optimizer_type_for_toml(ui_value: str) -> str:
    """Convert UI dropdown value to TOML storage format (lowercase)."""
    return MUSUBI_OPTIMIZER_TYPE_MAP_REVERSE.get(
        str(ui_value).strip(), str(ui_value).lower()
    )


def get_musubi_optimizer_type_for_ui(toml_value: str) -> str:
    """Convert TOML storage format to UI dropdown value (capitalized)."""
    return MUSUBI_OPTIMIZER_TYPE_MAP.get(
        str(toml_value).strip().lower(), toml_value
    )


def is_automagic_optimizer(optimizer_type: str) -> bool:
    """Check if the optimizer type is Automagic."""
    return str(optimizer_type).strip().lower() in ('automagic', 'Automagic')


# =============================================================================
# Population Functions
# =============================================================================

def populate_musubi_optimization_section(toml_data: Dict, label_vals: Dict) -> None:
    """
    Populate musubi-specific optimization fields from TOML.

    Sets both 'optimizer_type' and 'optimizer_type_m' in label_vals.
    """
    optimization = toml_data.get('optimization', {}) or {}
    if not isinstance(optimization, dict):
        return

    # Handle optimizer_type
    if 'optimizer_type' in optimization:
        opt_type = optimization.get('optimizer_type')
        mapped_type = get_musubi_optimizer_type_for_ui(opt_type)
        label_vals['optimizer_type'] = mapped_type
        label_vals['optimizer_type_m'] = mapped_type

    # Load all optimization fields
    for k in ('learning_rate', 'lr', 'max_steps', 'batch_size',
              'max_grad_norm', 'blocks_to_swap', 'scheduler_type', 'optimizer_args',
              'enable_gradient_checkpointing'):
        if k in optimization:
            label_vals[k] = optimization.get(k)

    # Handle gradient_accumulation_steps -> grad_accum_steps
    if 'gradient_accumulation_steps' in optimization:
        label_vals['gradient_accumulation_steps'] = optimization.get('gradient_accumulation_steps')
        label_vals['grad_accum_steps'] = optimization.get('gradient_accumulation_steps')


# =============================================================================
# Default Values
# =============================================================================

def get_automagic_optimizer_args_default() -> str:
    """Get default optimizer_args value for Automagic optimizer."""
    return 'min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6, eps=(1e-30; 1e-3), clip_threshold=1.0, beta2=0.999, weight_decay=0.0, do_paramiter_swapping=False, paramiter_swapping_factor=0.1'


# =============================================================================
# TOML Building
# =============================================================================

def build_musubi_optimizer_args_line(optimizer_type: str, optimizer_args_value: str = None) -> Optional[str]:
    """
    Build the optimizer_args line for musubi TOML if Automagic is selected.

    Args:
        optimizer_type: The optimizer type value
        optimizer_args_value: The optimizer_args value from UI (if any)

    Returns:
        The optimizer_args line, or None if not Automagic
    """
    if not is_automagic_optimizer(optimizer_type):
        return None

    args_value = optimizer_args_value or get_automagic_optimizer_args_default()
    return f"optimizer_args = '{args_value}'"
