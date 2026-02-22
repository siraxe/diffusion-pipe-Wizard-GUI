"""
Musubi model section handling for training UI.

Handles model-specific fields like wan_task.
"""

from typing import Dict, Any, List


# =============================================================================
# Population Functions
# =============================================================================

def populate_musubi_model_section(toml_data: Dict, label_vals: Dict) -> None:
    """
    Populate musubi-specific model fields from TOML.

    Handles WAN-specific fields like wan_task (i2v-A14B vs t2v-A14B).
    """
    model = toml_data.get('model', {}) or {}
    if not isinstance(model, dict):
        return

    if 'wan_task' in model:
        label_vals['wan_task'] = model.get('wan_task')


# =============================================================================
# TOML Building
# =============================================================================

def append_musubi_model_section(lines: List[str], model: Dict, _get_func) -> bool:
    """
    Append WAN-specific model fields to TOML lines.

    Returns True if fields were appended (for WAN models), False otherwise.
    """
    model_type = model.get('type', '')
    if 'wan' in str(model_type).lower():
        wan_task = _get_func('wan_task', '')
        if wan_task:
            lines.append(f"wan_task = '{wan_task}'")
            return True
    return False
