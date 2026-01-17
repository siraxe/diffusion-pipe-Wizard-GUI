"""Musubi utility functions for training and state management."""
import re
import ast


def parse_optimizer_args_list(optimizer_args_list: list[str] | None) -> dict:
    """
    Parse optimizer arguments from argparse (list of strings) into a dictionary.

    Handles both space-separated and comma-separated arguments:
    - Space-separated: ['min_lr=1e-7', 'max_lr=1e-3', 'eps=(1e-30; 1e-3)']
    - Comma-separated (single string in list): ['min_lr=1e-7, max_lr=1e-3, eps=(1e-30; 1e-3)']

    Note: Use semicolons in tuple values (e.g., eps=(1e-30; 1e-3)) to avoid
    conflicts with comma-based argument separation. Semicolons are replaced
    with commas during parsing.

    Args:
        optimizer_args_list: List of optimizer arg strings from argparse

    Returns:
        dict of parsed key=value pairs
    """
    if not optimizer_args_list:
        return {}

    # Join all args and split by comma to handle both formats
    combined = " ".join(optimizer_args_list)
    return parse_optimizer_args(combined)


def build_optimizer_flags(optimizer_type: str, optimizer_args_dict: dict) -> list:
    """
    Build CLI flags for optimizer parameters.

    For Automagic optimizer: converts dict to individual --optimizer_* flags
    For other optimizers: returns empty list (caller handles --optimizer_args)

    Args:
        optimizer_type: Type of optimizer (e.g., 'Automagic', 'AdamW')
        optimizer_args_dict: Parsed dict from parse_optimizer_args()

    Returns:
        List of CLI flag arguments like ['--optimizer_min_lr', '1e-7', ...]
    """
    flags = []

    if optimizer_type.lower() != 'automagic':
        return flags

    # Automagic-specific parameters with their CLI flag names
    param_mapping = {
        'min_lr': '--optimizer_min_lr',
        'max_lr': '--optimizer_max_lr',
        'lr_bump': '--optimizer_lr_bump',
        'eps': '--optimizer_eps',
        'clip_threshold': '--optimizer_clip_threshold',
        'beta2': '--optimizer_beta2',
        'weight_decay': '--optimizer_weight_decay',
        'do_paramiter_swapping': '--optimizer_do_paramiter_swapping',
        'paramiter_swapping_factor': '--optimizer_paramiter_swapping_factor',
    }

    for key, flag in param_mapping.items():
        if key in optimizer_args_dict:
            value = optimizer_args_dict[key]
            # Handle tuple values like eps=(1e-30, 1e-3)
            if isinstance(value, tuple):
                flags.extend([flag, str(value)])
            else:
                flags.extend([flag, str(value)])

    return flags


def parse_optimizer_args(optimizer_args: str) -> dict:
    """
    Parse comma-separated optimizer arguments into a dictionary.

    Supports formats like:
    - min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6
    - eps=(1e-30; 1e-3), clip_threshold=1.0
    - beta2=0.999, weight_decay=0.0
    - do_paramiter_swapping=False, paramiter_swapping_factor=0.1

    Note: Use semicolons in tuple values (e.g., eps=(1e-30; 1e-3)) to avoid
    conflicts with comma-based argument separation. Semicolons are replaced
    with commas during parsing.

    Returns:
        dict of parsed key=value pairs
    """
    if not optimizer_args or not optimizer_args.strip():
        return {}

    result = {}
    # Parse with parenthesis awareness - handle tuples like eps=(1e-30, 1e-3)
    pairs = []
    current = ""
    paren_depth = 0

    for char in optimizer_args:
        if char == '(':
            paren_depth += 1
            current += char
        elif char == ')':
            paren_depth -= 1
            current += char
        elif char == ',' and paren_depth == 0:
            if current.strip():
                pairs.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        pairs.append(current.strip())

    for pair in pairs:
        if '=' not in pair:
            logger.warning(f"Invalid optimizer arg format (missing '='): {pair}")
            continue

        # Split on first '=' only
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Replace semicolons with commas in tuple values (e.g., eps=(1e-30; 1e-3))
        # This allows safe comma-separated parsing without breaking tuples
        value = value.replace(';', ',')

        # Try to parse as Python literal for numbers/bools/strings/tuples
        try:
            result[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If parsing fails, keep as string
            result[key] = value

    return result
import os
from loguru import logger


def find_last_state_directory(output_dir: str, output_name: str) -> str | None:
    """
    Find the most recent state directory for resuming training.

    Searches for state directories in the musubi-tuner format:
    - {output_name}-{epoch_number:06d}-state (epoch states)
    - {output_name}-step{step_number:08d}-state (stepwise states)
    - {output_name}-state (final state)

    Returns the path to the most recent state directory, or None if none found.
    """
    if not os.path.exists(output_dir):
        return None

    state_dirs = []

    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Check if this is a state directory (must end with "-state")
        if not entry.endswith("-state"):
            continue

        # Extract step/epoch number for sorting
        # Format: {name}-{number:06d}-state (epoch) or {name}-step{number:08d}-state (step) or {name}-state (last)
        if entry == f"{output_name}-state":
            # "last" state gets highest priority
            step_or_epoch = float('inf')
        elif entry.startswith(f"{output_name}-step") and entry.endswith("-state"):
            # Step state: extract step number
            # Format: {output_name}-step{step_no:08d}-state
            try:
                step_str = entry[len(f"{output_name}-step"):].rstrip("-state")
                step_or_epoch = int(step_str)
            except ValueError:
                continue
        elif entry.startswith(f"{output_name}-") and entry.endswith("-state"):
            # Epoch state: extract epoch number
            # Format: {output_name}-{epoch_no:06d}-state
            try:
                epoch_str = entry[len(f"{output_name}-"):].rstrip("-state")
                step_or_epoch = int(epoch_str) * 1000000  # Epochs get priority over steps
            except ValueError:
                continue
        else:
            continue

        # Get modification time for tiebreaking
        mtime = os.path.getmtime(entry_path)
        state_dirs.append((step_or_epoch, mtime, entry_path))

    if not state_dirs:
        return None

    # Sort by step/epoch (descending), then by mtime (descending)
    state_dirs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    latest_state_dir = state_dirs[0][2]
    logger.info(f"Found latest state directory: {latest_state_dir}")
    return latest_state_dir
