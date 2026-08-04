"""
Configuration constants for training UI.

Centralizes model types, trainer types, field mappings, and other
configuration-related constants to avoid magic strings throughout the codebase.
"""

from typing import Set, Dict, Any

# =============================================================================
# Model Type Groupings
# =============================================================================

MUSUBI_MODEL_TYPES: Set[str] = {'_wan22', 'ltx-video-2', 'ltx2', 'wan22', 'wan', 'minimaxh3'}
LTX_MODEL_TYPES: Set[str] = {'ltx-video', 'ltx', 'ltx-video-2', 'ltx2'}
WAN_MODEL_TYPES: Set[str] = {'_wan22', 'wan22', 'wan'}

# Models that don't use diffusers_path/transformer_path
SKIP_PATH_MODELS: Set[str] = {'sdxl', 'ltx-video', 'ltx', 'ltx-video-2', 'ltx2', 'minimaxh3'}

# =============================================================================
# Trainer Types
# =============================================================================

class Trainers:
    """Trainer type constants."""
    DIFFUSION_PIPE = 'diffusion-pipe'
    MUSUBI = 'musubi'


def is_musubi_trainer(trainer: str) -> bool:
    """Check if the trainer is musubi."""
    return str(trainer).lower() == Trainers.MUSUBI


def is_musubi_model(model_type: str) -> bool:
    """Check if the model type is a musubi model."""
    mt_lower = str(model_type).lower().strip()
    return mt_lower in MUSUBI_MODEL_TYPES or any(m in mt_lower for m in MUSUBI_MODEL_TYPES)


def is_ltx_model(model_type: str) -> bool:
    """Check if the model type is an LTX model."""
    mt_lower = str(model_type).lower().strip()
    return mt_lower in LTX_MODEL_TYPES or any(m in mt_lower for m in LTX_MODEL_TYPES)


def is_wan_model(model_type: str) -> bool:
    """Check if the model type is a WAN model."""
    mt_lower = str(model_type).lower().strip()
    return mt_lower in WAN_MODEL_TYPES or any(m in mt_lower for m in WAN_MODEL_TYPES)


# =============================================================================
# Field Mappings
# =============================================================================

# Internal UI label -> TOML key mappings
FIELD_TO_TOML_MAPPINGS: Dict[str, str] = {
    'grad_accum_steps': 'gradient_accumulation_steps',
    'H3 mode': 'h3_training_mode',
}

# TOML key -> Internal UI label mappings
TOML_TO_FIELD_MAPPINGS: Dict[str, str] = {
    v: k for k, v in FIELD_TO_TOML_MAPPINGS.items()
}


def get_toml_key(ui_label: str) -> str:
    """Convert UI label to TOML key."""
    return FIELD_TO_TOML_MAPPINGS.get(ui_label, ui_label)


def get_ui_label(toml_key: str) -> str:
    """Convert TOML key to UI label."""
    return TOML_TO_FIELD_MAPPINGS.get(toml_key, toml_key)


# =============================================================================
# Fields That Should Always Be Included
# =============================================================================

# Fields to extract even when hidden (optimizer-related)
ALWAYS_INCLUDE_FIELDS: Set[str] = {
    'optimizer_type', 'optimizer_type_m', 'grad_accum_steps', 'lr', 'betas',
    'weight_decay', 'eps', 'beta3', 'd0', 'd_coef', 'schedulefree_c',
    'optimizer_args', 'factor', 'h3_training_mode'
}


# =============================================================================
# Default Values
# =============================================================================

DEFAULTS: Dict[str, Any] = {
    # Training
    'epochs': None,
    'micro_batch_size_per_gpu': None,
    'pipeline_stages': None,
    'gradient_accumulation_steps': 1,
    'gradient_clipping': None,
    'warmup_steps': None,
    'lr_scheduler': 'constant',
    'activation_checkpointing': 'unsloth',

    # Eval
    'eval_every_n_epochs': 1,
    'eval_before_first_step': True,
    'eval_micro_batch_size_per_gpu': 1,
    'eval_gradient_accumulation_steps': 1,

    # Misc
    'save_every_n_epochs': 5,
    'checkpoint_every_n_minutes': 10,
    'partition_method': 'parameters',
    'save_dtype': 'bfloat16',
    'caching_batch_size': 1,
    'steps_per_print': 1,
    'video_clip_mode': 'single_beginning',

    # Model
    'dtype': 'bfloat16',
    'transformer_dtype': 'float8',
    'timestep_sample_method': 'logit_normal',

    # Adapter
    'adapter': 'lora',
    'a_rank': 32,
    'a_dtype': 'bfloat16',

    # Block swap
    'blocks_swap': 0,
    'disable_bsfe': True,

    # Optimizer
    'lr': 2e-5,
    'betas': '[0.9, 0.99]',
    'weight_decay': 0.01,
    'eps': 1e-8,

    # Prodigy
    'd0': 1e-6,
    'd_coef': 1.0,
    'schedulefree_c': 0.0,

    # Automagic
    'min_lr': 1e-7,
    'max_lr': 1e-3,
    'lr_bump': 1e-6,
    'clip_threshold': 1.0,
    'do_paramiter_swapping': False,
    'paramiter_swapping_factor': 0.1,
}
