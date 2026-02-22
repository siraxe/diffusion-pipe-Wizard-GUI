"""
Musubi field visibility management for training UI.

Handles showing/hiding musubi-specific fields based on model/optimizer selection.
"""

from typing import Any, Dict

import flet as ft

from .optimizer import is_automagic_optimizer, get_musubi_optimizer_type_for_ui


# =============================================================================
# Field Visibility
# =============================================================================

def set_musubi_field_visibility(control: Any, field_name: str, visible: bool) -> None:
    """
    Recursively find a musubi field and set its visibility.

    Args:
        control: The control to search (can be a container with controls)
        field_name: The field label to find (e.g., 'optimizer_args', 'mixed_precision_mode')
        visible: Whether the field should be visible
    """
    if hasattr(control, 'controls') and control.controls:
        for c in control.controls:
            set_musubi_field_visibility(c, field_name, visible)

    if hasattr(control, 'content') and control.content:
        set_musubi_field_visibility(control.content, field_name, visible)

    ctrl_label = getattr(control, 'label', None)
    ctrl_data = getattr(control, 'data', None)

    if ctrl_label == field_name or ctrl_data == field_name:
        control.visible = visible
        if hasattr(control, 'page') and control.page:
            control.page.update()


def set_musubi_field_value(control: Any, field_name: str, value: Any, page=None) -> None:
    """
    Recursively find a musubi field and set its value.

    Args:
        control: The control to search (can be a container with controls)
        field_name: The field label to find (e.g., 'wan_task', 'optimizer_args')
        value: The value to set
        page: Optional Flet page to update
    """
    if hasattr(control, 'controls') and control.controls:
        for c in control.controls:
            set_musubi_field_value(c, field_name, value, page)

    if hasattr(control, 'content') and control.content:
        set_musubi_field_value(control.content, field_name, value, page)

    ctrl_label = getattr(control, 'label', None)
    ctrl_data = getattr(control, 'data', None)

    if ctrl_label == field_name or ctrl_data == field_name:
        control.value = value
        if hasattr(control, 'page') and control.page:
            control.page.update()
        elif page:
            page.update()


def set_optimizer_args_visibility(container: Any, optimizer_type_value: str) -> None:
    """
    Show/hide optimizer_args field based on optimizer type.

    The optimizer_args field is only visible when Automagic is selected.
    """
    is_automagic = is_automagic_optimizer(optimizer_type_value)
    set_musubi_field_visibility(container, 'optimizer_args', is_automagic)


def set_musubi_precision_fields_visibility(container: Any, visible: bool) -> None:
    """
    Show/hide musubi precision fields (mixed_precision_mode, fp8_base, fp8_scaled, attn_chunking).
    """
    for field_name in ['mixed_precision_mode', 'fp8_base', 'fp8_scaled', 'attn_chunking']:
        set_musubi_field_visibility(container, field_name, visible)


# =============================================================================
# Event Triggering
# =============================================================================

def trigger_musubi_optimizer_change(training_tab_container, optimizer_type_value: str, page=None) -> None:
    """
    Trigger the optimizer change event in the musubi section to update optimizer_args visibility.

    This properly simulates the user changing the optimizer dropdown.
    """
    try:
        from flet_app.ui.pages.training_config import musubi_custom_section_ref
        if not musubi_custom_section_ref or not musubi_custom_section_ref.current:
            return

        musubi_section = musubi_custom_section_ref.current

        def find_and_trigger(control):
            if hasattr(control, 'controls') and control.controls:
                for c in control.controls:
                    if find_and_trigger(c):
                        return True

            if hasattr(control, 'content') and control.content:
                if find_and_trigger(control.content):
                    return True

            ctrl_label = getattr(control, 'label', None)
            if ctrl_label == 'optimizer_type_m' and isinstance(control, ft.Dropdown):
                control.value = optimizer_type_value
                if hasattr(control, 'on_change') and control.on_change:
                    try:
                        class _E:
                            pass
                        e = _E()
                        setattr(e, 'control', control)
                        setattr(e, 'data', optimizer_type_value)
                        setattr(e, 'page', page)
                        control.on_change(e)
                        return True
                    except Exception:
                        pass
                if hasattr(control, 'page') and control.page:
                    control.page.update()
                return True
            return False

        find_and_trigger(musubi_section)
    except Exception:
        pass
