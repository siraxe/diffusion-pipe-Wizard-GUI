# Config Utils Refactoring Plan

## Progress

| Task | Status | File |
|------|--------|------|
| Create config/constants.py | ✅ Done | `config/constants.py` |
| Create config/toml_formatting.py | ✅ Done | `config/toml_formatting.py` |
| Create ui/utils/control_walker.py | ✅ Done | `ui/utils/control_walker.py` |
| Create ui/utils/image_utils.py | ✅ Done | `ui/utils/image_utils.py` |
| Create config/toml_builder.py | ✅ Done | `config/toml_builder.py` |
| Create config/toml_loader.py | ✅ Done | `config/toml_loader.py` |
| Create config/__init__.py | ✅ Done | `config/__init__.py` |
| Update config_utils.py as wrapper | ✅ Done | `config_utils.py` |
| Create config/musubi/__init__.py | ✅ Done | `config/musubi/__init__.py` |
| Create config/musubi/optimizer.py | ✅ Done | `config/musubi/optimizer.py` |
| Create config/musubi/visibility.py | ✅ Done | `config/musubi/visibility.py` |
| Create config/musubi/acceleration.py | ✅ Done | `config/musubi/acceleration.py` |
| Create config/musubi/model.py | ✅ Done | `config/musubi/model.py` |
| Refactor config_utils_musubi.py | ✅ Done | `config_utils_musubi.py` |
| Create config/sections/*.py | ⏸️ Deferred | Optional further cleanup |

## Final File Structure

```
flet_app/ui/utils/
├── config/
│   ├── __init__.py           ✅ Re-exports main API
│   ├── constants.py          ✅ Types, trainers, field mappings
│   ├── toml_formatting.py    ✅ quote(), to_bool(), path utils
│   ├── toml_builder.py       ✅ Build TOML from UI
│   ├── toml_loader.py        ✅ Load TOML into UI
│   ├── plan.md
│   └── musubi/
│       ├── __init__.py       ✅ Re-exports musubi API
│       ├── optimizer.py      ✅ Optimizer mapping, args
│       ├── visibility.py     ✅ Field visibility management
│       ├── acceleration.py   ✅ Precision fields, acceleration
│       └── model.py          ✅ Model-specific fields (wan_task)
├── control_walker.py         ✅ Control traversal utilities
├── image_utils.py            ✅ Image processing
├── config_utils.py           ✅ Wrapper (backward compatible)
└── config_utils_musubi.py    ✅ Wrapper (backward compatible)
```

## What Each File Contains

### config/constants.py ✅
- Model type sets (MUSUBI_MODEL_TYPES, LTX_MODEL_TYPES, WAN_MODEL_TYPES)
- Trainer types (Trainers class)
- Helper functions: is_musubi_trainer(), is_musubi_model(), is_ltx_model()
- Field mappings (FIELD_TO_TOML_MAPPINGS)
- ALWAYS_INCLUDE_FIELDS set
- DEFAULTS dict

### config/toml_formatting.py ✅
- quote(), to_bool(), toml_bool()
- normalize_slashes(), is_absolute_path()
- expand_model_path(), collapse_model_path()
- resolve_output_dir(), resolve_path_if_relative()

### ui/utils/control_walker.py ✅
- ControlWalker class with static methods
- find_by_label(), find_all(), apply_to_all()
- set_field_visibility(), set_field_value()
- extract_config(), apply_values()

### ui/utils/image_utils.py ✅
- process_and_save_image()
- save_and_scale_image()
- Image resizing/cropping logic

### config/toml_builder.py ✅
- build_toml_config_from_ui()
- build_training_section()
- build_eval_section()
- build_misc_section()
- build_model_section()
- build_optimizer_section()
- build_adapter_section()
- build_monitoring_section()
- extract_config_from_controls()

### config/toml_loader.py ✅
- update_ui_from_toml()
- populate_model_section()
- populate_optimizer_section()
- populate_adapter_section()
- populate_lora_section()
- populate_training_strategy_section()
- populate_acceleration_section()
- populate_checkpoints_section()
- populate_monitoring_section()
- populate_data_section()
- populate_validation_section()
- apply_values_recursive()
- apply_all_values()
- handle_dataset_selection()
- handle_dataset_list_selection()

### config/__init__.py ✅
- Re-exports all main API functions
- Re-exports formatting utilities
- Re-exports constants

### config_utils.py ✅
- Backward-compatible wrapper
- Delegates to config/ package
- Provides legacy aliases (_normalize_slashes, etc.)
- Delegates image processing to image_utils.py

### config/musubi/optimizer.py ✅
- MUSUBI_OPTIMIZER_TYPE_MAP, MUSUBI_OPTIMIZER_TYPE_MAP_REVERSE
- get_musubi_optimizer_type_for_toml(), get_musubi_optimizer_type_for_ui()
- is_automagic_optimizer()
- populate_musubi_optimization_section()
- get_automagic_optimizer_args_default()
- build_musubi_optimizer_args_line()

### config/musubi/visibility.py ✅
- set_musubi_field_visibility()
- set_musubi_field_value()
- set_optimizer_args_visibility()
- set_musubi_precision_fields_visibility()
- trigger_musubi_optimizer_change()

### config/musubi/acceleration.py ✅
- get_musubi_precision_defaults()
- populate_musubi_acceleration_section()
- append_musubi_acceleration_section()

### config/musubi/model.py ✅
- populate_musubi_model_section()
- append_musubi_model_section()

### config/musubi/__init__.py ✅
- Re-exports all musubi functions
- update_musubi_ui_from_toml() main function

### config_utils_musubi.py ✅
- Backward-compatible wrapper
- Delegates to config/musubi/ package
- Provides legacy quote() function
