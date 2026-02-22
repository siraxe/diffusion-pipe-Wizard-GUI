# Recursive control traversal utilities for Flet UI

from typing import Any, Callable, List, Optional, Dict, Set
import flet as ft


class ControlWalker:
    # Fields to always include even when hidden
    ALWAYS_INCLUDE_FIELDS: Set[str] = {
        'optimizer_type', 'optimizer_type_m', 'grad_accum_steps', 'lr', 'betas',
        'weight_decay', 'eps', 'beta3', 'd0', 'd_coef', 'schedulefree_c',
        'optimizer_args', 'factor'
    }

    @staticmethod
    def find_by_label(container: Any, label: str) -> Optional[ft.Control]:
        result = None

        def _search(ctrl):
            nonlocal result
            if result is not None:
                return
            if hasattr(ctrl, 'controls') and ctrl.controls:
                for c in ctrl.controls:
                    _search(c)
            if hasattr(ctrl, 'content') and ctrl.content:
                _search(ctrl.content)
            ctrl_label = getattr(ctrl, 'label', None)
            ctrl_data = getattr(ctrl, 'data', None)
            if ctrl_label == label or ctrl_data == label:
                result = ctrl

        _search(container)
        return result

    @staticmethod
    def find_all(container: Any, predicate: Callable) -> List[ft.Control]:
        results = []

        def _search(ctrl):
            if hasattr(ctrl, 'controls') and ctrl.controls:
                for c in ctrl.controls:
                    _search(c)
            if hasattr(ctrl, 'content') and ctrl.content:
                _search(ctrl.content)
            if predicate(ctrl):
                results.append(ctrl)

        _search(container)
        return results

    @staticmethod
    def apply_to_all(container: Any, action: Callable) -> None:
        def _traverse(ctrl):
            action(ctrl)
            if hasattr(ctrl, 'controls') and ctrl.controls:
                for c in ctrl.controls:
                    _traverse(c)
            if hasattr(ctrl, 'content') and ctrl.content:
                _traverse(ctrl.content)

        _traverse(container)

    @staticmethod
    def set_field_visibility(container: Any, field_name: str, visible: bool) -> None:
        def _set_visible(ctrl):
            if hasattr(ctrl, 'controls') and ctrl.controls:
                for c in ctrl.controls:
                    _set_visible(c)
            if hasattr(ctrl, 'content') and ctrl.content:
                _set_visible(ctrl.content)
            ctrl_label = getattr(ctrl, 'label', None)
            ctrl_data = getattr(ctrl, 'data', None)
            if ctrl_label == field_name or ctrl_data == field_name:
                ctrl.visible = visible
                if hasattr(ctrl, 'page') and ctrl.page:
                    ctrl.page.update()

        _set_visible(container)

    @staticmethod
    def set_field_value(container: Any, field_name: str, value: Any, page=None) -> None:
        def _set_value(ctrl):
            if hasattr(ctrl, 'controls') and ctrl.controls:
                for c in ctrl.controls:
                    _set_value(c)
            if hasattr(ctrl, 'content') and ctrl.content:
                _set_value(ctrl.content)
            ctrl_label = getattr(ctrl, 'label', None)
            ctrl_data = getattr(ctrl, 'data', None)
            if ctrl_label == field_name or ctrl_data == field_name:
                ctrl.value = value
                if hasattr(ctrl, 'page') and ctrl.page:
                    ctrl.page.update()
                elif page:
                    page.update()

        _set_value(container)

    @staticmethod
    def extract_config(control: Any) -> Dict:
        result = {}
        field_mappings = {'grad_accum_steps': 'gradient_accumulation_steps'}

        def _extract(child):
            if hasattr(child, 'controls') and child.controls:
                for sub_child in child.controls:
                    _extract(sub_child)
            elif hasattr(child, 'content') and child.content:
                _extract(child.content)
            elif isinstance(child, ft.TextField):
                label = getattr(child, 'label', None)
                if label in ControlWalker.ALWAYS_INCLUDE_FIELDS:
                    result[label] = child.value
                elif getattr(child, 'visible', True):
                    result[label] = child.value
            elif isinstance(child, ft.Dropdown):
                label = getattr(child, 'label', None)
                result_key = field_mappings.get(label, label)
                current_visible = getattr(child, 'visible', True)
                if label in ControlWalker.ALWAYS_INCLUDE_FIELDS:
                    result[result_key] = child.value
                elif current_visible:
                    result[result_key] = child.value
            elif isinstance(child, ft.Checkbox):
                if getattr(child, 'visible', True):
                    key = getattr(child, 'data', None) or child.label
                    if child.label in ['8_bit_te']:
                        key = child.label
                    result[key] = child.value

        if control:
            _extract(control)
        return result

    @staticmethod
    def apply_values(control: Any, label_vals: Dict, page, skip_fields: Set[str] = None) -> None:
        from .config.toml_formatting import to_bool

        if skip_fields is None:
            skip_fields = set()

        def _apply(ctrl):
            try:
                if hasattr(ctrl, 'controls') and ctrl.controls:
                    for c in ctrl.controls:
                        _apply(c)
                elif hasattr(ctrl, 'content') and ctrl.content:
                    _apply(ctrl.content)

                label = getattr(ctrl, 'label', None)
                if not label or label in skip_fields:
                    return

                if label in label_vals:
                    val = label_vals[label]
                    if isinstance(ctrl, ft.TextField):
                        ctrl.value = str(val) if val is not None else ''
                        if ctrl.page:
                            ctrl.update()
                    elif isinstance(ctrl, ft.Dropdown):
                        if val is None or str(val).strip() == '':
                            ctrl.value = None
                        else:
                            ctrl.value = str(val)
                        if ctrl.page:
                            ctrl.update()
                    elif isinstance(ctrl, ft.Checkbox):
                        ctrl.value = to_bool(val) if val is not None else False
                        if ctrl.page:
                            ctrl.update()
            except Exception:
                pass

        _apply(control)
