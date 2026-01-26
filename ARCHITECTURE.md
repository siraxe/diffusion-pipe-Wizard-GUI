# DPipe Architecture Map

*Generated: 2026-01-26*

---

## 1. Major Subsystems

| Subsystem | Location | Responsibility |
|-----------|----------|----------------|
| **Flet UI Layer** | `flet_app/` | Main application interface with tabbed navigation |
| **Configuration System** | `flet_app/settings.py`, `ui/utils/` | App settings, TOML parsing, model configs |
| **Dataset Management** | `flet_app/ui/dataset_manager/` | Browse, edit, caption datasets |
| **Training Pipeline** | `flet_app/ui/tab_training_view.py`, `ui/pages/` | Training configuration & monitoring |
| **Popup System** | `flet_app/ui_popups/` | Dialogs, media editors, context menus |
| **Training Engine** | `diffusion-trainers/diffusion-pipe/`, `LTX-2/` | Backend training scripts |
| **Module System** | `flet_app/modules/` | Independent modules (MiniMax, JoyCaption) |

---

## 2. Key Entry Points

```
flet_app.py (main)
├── settings.py (config singleton)
├── theme_config.py (UI theming)
└── Tabs:
    ├── tab_training_view.py → Training
    ├── dataset_manager/ → Datasets
    └── tab_tools_view.py → Models/Tools
```

---

## 3. Shared Services (Reusability: GOOD)

| Service | Location | Notes |
|---------|----------|-------|
| **Config Singleton** | `settings.py` | Well-designed, reuse |
| **Project Root** | `project_root.py` | Centralized path resolution |
| **Theme Config** | `ui/theme_config.py` | Browser-compatible |
| **Popup Base** | `ui_popups/popup_dialog_base.py` | Solid foundation |

---

## 4. Duplication-Sensitive Areas (Needs Consolidation)

| Area | Where Duplicated | Risk Level |
|------|------------------|------------|
| **Subprocess patterns** | `image_editor_bridge.py`, `tab_tools_view.py`, `tab_training_view.py`, `process_cleanup.py` | 🔴 HIGH |
| **TOML config parsing** | `config_utils.py`, scattered across page modules | 🟡 MEDIUM |
| **Async handling** | Dataset modules, training view | 🟡 MEDIUM |
| **UI text fields** | `_styles.py`, repeated in pages | 🟢 LOW |
| **Section creation** | Multiple page modules | 🟢 LOW |

---

## 5. Missing Abstractions (Should Create)

```
RECOMMENDED SERVICE LAYER:
├── DatasetService     (centralize dataset operations)
├── TrainingService    (abstract training execution)
├── ModelService       (model download/management)
├── ProcessManager     (unified subprocess handling)
└── AsyncUtils         (coroutine helpers)
```

---

## 6. Reuse vs Reimplement Guide

| Category | Decision | Rationale |
|----------|----------|-----------|
| `settings.py` config | **Reuse** | Clean singleton pattern |
| `theme_config.py` | **Reuse** | Good browser compatibility |
| Popup system | **Reimplement** | Too much duplication |
| Process management | **Reimplement** | Scattered, inconsistent |
| Config validation | **Reimplement** | Duplicated logic |
| Async patterns | **Reimplement** | Needs centralization |

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        FLET UI LAYER                         │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐              │
│  │ Training │  │  Datasets  │  │   Models   │              │
│  │   Tab    │  │    Tab     │  │    Tab     │              │
│  └────┬─────┘  └─────┬──────┘  └─────┬──────┘              │
│       │              │                │                     │
│  ┌────▼──────────────▼────────────────▼────┐               │
│  │     Configuration System (TOML)         │               │
│  │  - settings.py  - config_utils.py       │               │
│  └──────────────────┬──────────────────────┘               │
│                     │                                       │
└─────────────────────┼───────────────────────────────────────┘
                      │
                      ▼ (Subprocess Calls)
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING ENGINE                         │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │ diffusion-pipe/  │        │     LTX-2/       │          │
│  │  - train.py      │        │  - src/ltx_...  │          │
│  │  - models/       │        │                  │          │
│  │  - configs/      │        │                  │          │
│  └──────────────────┘        └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘

                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   SHARED UTILITIES                          │
│  - project_root.py  - theme_config.py                       │
│  - process_cleanup.py  - console_cleanup.py                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Refactoring Priorities

| Priority | Task | Impact |
|----------|------|--------|
| 🔴 **HIGH** | Create `ProcessManager` service | Eliminates subprocess duplication |
| 🔴 **HIGH** | Add service layer (Dataset/Training/Model) | Better separation of concerns |
| 🟡 **MEDIUM** | Centralize async utilities | Consistent async handling |
| 🟡 **MEDIUM** | Abstract config validation | DRY principle |
| 🟢 **LOW** | Standardize UI component creation | Minor duplication cleanup |

---

## 9. Component Details

### UI Layer Structure

```
flet_app/
├── flet_app.py              # Main entry point
├── settings.py              # Configuration singleton
├── project_root.py          # Path resolution
├── theme_config.py          # Theme configuration
├── ui/
│   ├── tab_training_view.py    # Training tab
│   ├── tab_tools_view.py       # Models/Tools tab
│   ├── dataset_manager/        # Dataset management
│   │   ├── dataset_layout_tab.py
│   │   ├── dataset_actions.py
│   │   ├── dataset_utils.py
│   │   └── dataset_controls.py
│   ├── pages/                 # Configuration pages
│   │   ├── training_config.py
│   │   ├── training_data_config.py
│   │   ├── training_monitor.py
│   │   └── model_field_config.py
│   ├── _styles.py             # UI styling
│   └── utils/                 # Utilities
│       ├── config_utils.py
│       ├── process_cleanup.py
│       └── console_cleanup.py
└── ui_popups/               # Popup dialogs
    ├── popup_dialog_base.py
    ├── image_editor.py
    ├── video_editor.py
    └── unified_context_menu.py
```

### Training Engine Structure

```
diffusion-trainers/
├── diffusion-pipe/
│   ├── train.py             # Main training script
│   ├── models/              # Model implementations
│   ├── utils/               # Training utilities
│   └── configs/             # Training configurations
└── LTX-2/
    └── src/ltx_trainer/     # LTX-2 training implementation
```

### Module System

```
flet_app/modules/
├── minimax-remover/         # MiniMax image processing
└── joycaption/              # Caption generation
```

---

## 10. Data Flow

### Training Flow
1. User configures training via UI tabs
2. Configuration saved to TOML via `config_utils.py`
3. Training launched via subprocess to `diffusion-pipe/train.py`
4. Progress monitored back through UI

### Dataset Management Flow
1. Browse datasets via `dataset_manager/`
2. Select dataset/view thumbnails
3. Apply operations (crop, caption, etc.)
4. Changes persisted via `dataset_actions.py`

### Model Download Flow
1. User enters model URL in Models tab
2. Download executed via subprocess
3. Model stored in models directory
4. Available for training configuration

---

## 11. Dependencies

```
flet_app.py
    ├── settings.py (config)
    ├── theme_config.py (theming)
    ├── tab_training_view.py
    ├── dataset_manager/
    └── tab_tools_view.py

UI Components
    ├── utils/ (config, process, console)
    ├── pages/ (config pages)
    └── _styles.py (styling)

Popup System
    └── popup_dialog_base.py (base class)
        ├── image_editor.py
        ├── video_editor.py
        └── unified_context_menu.py
```

---

*Note: The `workspace/` folder is intentionally excluded as it is used for file storage, not application code.*
