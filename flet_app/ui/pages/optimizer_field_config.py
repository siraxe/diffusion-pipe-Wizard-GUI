"""
Centralized optimizer field configuration for training UI.
Maps optimizer types to their field visibility and default values.
"""

# Centralized optimizer configuration: defines visibility and defaults for each optimizer
OPTIMIZER_CONFIG = {
    "adamw_optimi": {
        "aliases": ["adamw_optimi"],
        "show_prodigy_row": False,
        "show_automagic_row": False,
        "defaults": {
            "lr": "2e-5",
            "betas": "[0.9, 0.99]",
            "weight_decay": "0.01",
            "eps": "1e-8",
        },
    },
    "prodigyplusschedulefree": {
        "aliases": ["prodigyplusschedulefree", "prodigyplus", "ppsf"],
        "show_prodigy_row": True,
        "show_automagic_row": False,
        "defaults": {
            "lr": "1.0",
            "betas": "[0.9, 0.99]",
            "weight_decay": "0.0",
            "eps": "1e-8",
            "beta3": "None",
            "d0": "1e-6",
            "d_coef": "1.0",
            "schedulefree_c": "0.0",
        },
    },
    "automagic": {
        "aliases": ["automagic"],
        "show_prodigy_row": False,
        "show_automagic_row": True,
        "defaults": {
            "lr": "1e-6",
            "betas": "[0.9, 0.99]",
            "weight_decay": "0.0",
            "eps": "[1e-30, 0.001]",
            "min_lr": "1e-7",
            "max_lr": "1e-3",
            "lr_bump": "1e-6",
            "clip_threshold": "1.0",
            "do_paramiter_swapping": "False",
            "paramiter_swapping_factor": "0.1",
        },
    },
    "adamw8bit": {
        "aliases": ["adamw8bit"],
        "show_prodigy_row": False,
        "show_automagic_row": False,
        "defaults": {
            "lr": "2e-5",
            "betas": "[0.9, 0.99]",
            "weight_decay": "0.01",
            "eps": "1e-8",
        },
    },
    "adamw8bitkahan": {
        "aliases": ["adamw8bitkahan", "adamwkahan"],
        "show_prodigy_row": False,
        "show_automagic_row": False,
        "defaults": {
            "lr": "2e-5",
            "betas": "[0.9, 0.99]",
            "weight_decay": "0.01",
            "eps": "1e-8",
        },
    },
    "adamw": {
        "aliases": ["adamw"],
        "show_prodigy_row": False,
        "show_automagic_row": False,
        "defaults": {
            "lr": "2e-5",
            "betas": "[0.9, 0.99]",
            "weight_decay": "0.01",
            "eps": "1e-8",
        },
    },
    "stableadamw": {
        "aliases": ["stableadamw", "stableadam"],
        "show_prodigy_row": False,
        "show_automagic_row": False,
        "defaults": {
            "lr": "2e-5",
            "betas": "[0.9, 0.99]",
            "weight_decay": "0.01",
            "eps": "1e-8",
        },
    },
}


def normalize_optimizer_name(optimizer_name):
    """Normalize an optimizer name to lowercase for comparison."""
    return str(optimizer_name).strip().lower() if optimizer_name else ""


def get_optimizer_key(normalized_name):
    """Get the config key for a normalized optimizer name by checking aliases."""
    for key, config in OPTIMIZER_CONFIG.items():
        if normalized_name in config.get("aliases", []):
            return key
    return None


def get_optimizer_config(optimizer_name):
    """Get the full config dict for an optimizer name."""
    normalized = normalize_optimizer_name(optimizer_name)
    key = get_optimizer_key(normalized)
    return OPTIMIZER_CONFIG.get(key) if key else None


def get_field_visibility(optimizer_name):
    """Get the show_prodigy_row value for an optimizer."""
    config = get_optimizer_config(optimizer_name)
    return config.get("show_prodigy_row", False) if config else False


def get_automagic_visibility(optimizer_name):
    """Get the show_automagic_row value for an optimizer."""
    config = get_optimizer_config(optimizer_name)
    return config.get("show_automagic_row", False) if config else False


def get_field_defaults(optimizer_name):
    """Get the defaults dict for an optimizer."""
    config = get_optimizer_config(optimizer_name)
    return config.get("defaults", {}) if config else {}


# Default values for standard fields (used when optimizer not in config)
DEFAULT_STANDARD_VALUES = {
    "lr": "2e-5",
    "betas": "[0.9, 0.99]",
    "weight_decay": "0.01",
    "eps": "1e-8",
}

# Default values for prodigy-specific fields
DEFAULT_PRODIGY_VALUES = {
    "beta3": "None",
    "d0": "1e-6",
    "d_coef": "1.0",
    "schedulefree_c": "0.0",
}


def get_standard_field_default(optimizer_name, field_name):
    """Get the default value for a standard field (lr, betas, weight_decay, eps)."""
    config = get_optimizer_config(optimizer_name)
    if config:
        defaults = config.get("defaults", {})
        if field_name in defaults:
            return defaults[field_name]
    return DEFAULT_STANDARD_VALUES.get(field_name, "")


def get_prodigy_field_default(optimizer_name, field_name):
    """Get the default value for a prodigy field (beta3, d0, d_coef, schedulefree_c)."""
    config = get_optimizer_config(optimizer_name)
    if config and config.get("show_prodigy_row", False):
        defaults = config.get("defaults", {})
        if field_name in defaults:
            return defaults[field_name]
    return DEFAULT_PRODIGY_VALUES.get(field_name, "")


# Default values for automagic-specific fields
DEFAULT_AUTOMAGIC_VALUES = {
    "min_lr": "1e-7",
    "max_lr": "1e-3",
    "lr_bump": "1e-6",
    "clip_threshold": "1.0",
    "do_paramiter_swapping": "False",
    "paramiter_swapping_factor": "0.1",
}


def get_automagic_field_default(optimizer_name, field_name):
    """Get the default value for an automagic field (min_lr, max_lr, lr_bump, clip_threshold, do_paramiter_swapping, paramiter_swapping_factor)."""
    config = get_optimizer_config(optimizer_name)
    if config and config.get("show_automagic_row", False):
        defaults = config.get("defaults", {})
        if field_name in defaults:
            return defaults[field_name]
    return DEFAULT_AUTOMAGIC_VALUES.get(field_name, "")


# List of all optimizer types for the dropdown
OPTIMIZER_OPTIONS = {
    "adamw_optimi": "AdamW Optimi",
    "prodigyplusschedulefree": "ProdigyPlusScheduleFree",
    "automagic": "Automagic",
    "adamw8bit": "AdamW 8-bit",
    "adamw8bitkahan": "AdamW 8-bit Kahan",
    "adamw": "AdamW",
    "stableadamw": "StableAdamW",
}
