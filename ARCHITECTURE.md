# DPipe Architecture Map

*Generated: 2026-02-05*

---

## 1. Major Subsystems

| Subsystem | Location | Responsibility |
|-----------|----------|----------------|
| **Flet UI Layer** | `flet_app/` | Main application interface with tabbed navigation |
| **Configuration System** | `flet_app/settings.py`, `ui/utils/` | App settings, TOML parsing, model configs |
| **Training Pipeline** | `flet_app/ui/training/`, `tab_training_view.py` | Training configuration, monitoring, workflow |
| **Dataset Management** | `flet_app/ui/dataset_manager/` | Browse, edit, caption datasets |
| **Menu System** | `flet_app_top_menu.py`, `ui/utils/utils_top_menu.py` | Application navigation menu |
| **Popup System** | `flet_app/ui_popups/` | Dialogs, media editors, context menus |
| **Training Engine** | `diffusion-trainers/`, `musubi_ltx2.py` | Backend training scripts (Musubi, Diffusion Pipe) |
| **Module System** | `flet_app/modules/` | Independent modules (MiniMax, JoyCaption) |
| **Utilities** | `scripts/` | Standalone tools for LoRA conversion, captioning |

---

## 2. Key Entry Points

```
flet_app.py (main)
├── settings.py (config singleton)
├── theme_config.py (UI theming)
├── flet_app_top_menu.py (menu)
└── Tabs:
    ├── tab_training_view.py → Training
    │   └── ui/training/
    │       ├── start_button_handler.py (execution)
    │       ├── output_manager.py (console streaming)
    │       └── training_dataset_block.py (dataset config)
    ├── dataset_manager/ → Datasets
    │   ├── dataset_layout_tab.py (browser)
    │   ├── data_config_panel.py (config)
    │   └── dataset_thumb_layout.py (thumbnails)
    └── tab_tools_view.py → Models/Tools
```

**External Entry Points:**
- `musubi_ltx2.py` - LTX-2 training workflow interface
- `scripts/convert_comfy_to_training_lora.py` - LoRA format conversion

---

## 3. Shared Services (Reusability: GOOD)

| Service | Location | Notes |
|---------|----------|-------|
| **Config Singleton** | `settings.py` | Well-designed singleton pattern with WSL path handling |
| **Project Root** | `project_root.py` | Centralized path resolution |
| **Theme Config** | `ui/theme_config.py` | Browser-compatible dark/light themes |
| **Popup Base** | `ui_popups/popup_dialog_base.py` | Solid foundation for modal dialogs |
| **Output Manager** | `ui/training/output_manager.py` | Console rendering and streaming |

---

## 4. Duplication-Sensitive Areas (Needs Consolidation)

| Area | Where Duplicated | Risk Level |
|------|------------------|------------|
| **Subprocess patterns** | 13+ files across training, dataset, tools | 🔴 HIGH |
| **TOML config parsing** | `config_utils.py`, scattered across pages | 🟡 MEDIUM |
| **Async handling** | Dataset modules, training view | 🟡 MEDIUM |
| **page.update() calls** | 100+ occurrences throughout UI | 🟡 MEDIUM |
| **UI text fields** | `_styles.py`, repeated in pages | 🟢 LOW |
| **Section creation** | Multiple page modules | 🟢 LOW |

---

## 5. Missing Abstractions (Should Create)

```
RECOMMENDED SERVICE LAYER:
├── ProcessManager     (unified subprocess handling with streaming)
├── DatasetService     (centralize dataset operations)
├── TrainingService    (abstract training execution & monitoring)
├── ModelService       (model download/management)
├── ConfigValidator    (centralized config validation)
└── AsyncUtils         (coroutine helpers, update batching)
```

---

## 6. Reuse vs Reimplement Guide

| Category | Decision | Rationale |
|----------|----------|-----------|
| `settings.py` config | **Reuse** | Clean singleton pattern |
| `theme_config.py` | **Reuse** | Good browser compatibility |
| `output_manager.py` | **Reuse** | Well-designed streaming |
| Popup system | **Reimplement** | Too much duplication |
| Process management | **Reimplement** | Scattered, inconsistent |
| Config validation | **Reimplement** | Duplicated logic |
| Async patterns | **Reimplement** | Needs centralization |

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLET UI LAYER                             │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐                  │
│  │ Training │  │  Datasets  │  │   Models   │                  │
│  │   Tab    │  │    Tab     │  │    Tab     │                  │
│  └────┬─────┘  └─────┬──────┘  └─────┬──────┘                  │
│       │              │                │                         │
│  ┌────▼──────────────▼────────────────▼────┐                    │
│  │     Configuration System (TOML)         │                    │
│  │  - settings.py  - config_utils.py       │                    │
│  └──────────────────┬──────────────────────┘                    │
│                     │                                             │
│  ┌──────────────────▼──────────────────────┐                    │
│  │         Training Pipeline System        │                    │
│  │  - start_button_handler.py              │                    │
│  │  - output_manager.py                    │                    │
│  │  - training_dataset_block.py            │                    │
│  └──────────────────┬──────────────────────┘                    │
│                     │                                             │
└─────────────────────┼─────────────────────────────────────────────┘
                      │
                      ▼ (Subprocess Calls)
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING ENGINE                             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Musubi Tuner    │  │ Diffusion    │  │   LTX-2/         │  │
│  │  (LTX-2 focused) │  │   Pipe       │  │   (legacy)       │  │
│  │  - ltx2_train_   │  │  - Flux      │  │                  │  │
│  │    network.py    │  │  - Kandinsky │  │                  │  │
│  │  - ltx2_cache_   │  │  - Qwen      │  │                  │  │
│  │    latents.py    │  │              │  │                  │  │
│  └──────────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODULES & UTILITIES                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │ MiniMax      │  │ JoyCaption   │  │ Scripts/         │      │
│  │ Remover      │  │              │  │ - LoRA convert   │      │
│  │              │  │              │  │ - Rerank         │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Refactoring Priorities

| Priority | Task | Impact |
|----------|------|--------|
| 🔴 **HIGH** | Create `ProcessManager` service | Eliminates subprocess duplication across 13+ files |
| 🔴 **HIGH** | Add service layer (Dataset/Training/Model) | Better separation of concerns |
| 🟡 **MEDIUM** | Centralize async utilities | Consistent async handling |
| 🟡 **MEDIUM** | Abstract config validation | DRY principle |
| 🟡 **MEDIUM** | Batch page.update() calls | Performance improvement |
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
├── flet_app_top_menu.py     # Application menu
├── settings_popup.py        # Settings dialog
├── ui/
│   ├── __init__.py
│   ├── tab_training_view.py    # Training tab (orchestrator)
│   ├── tab_tools_view.py       # Models/Tools tab
│   ├── dataset_manager/        # Dataset management
│   │   ├── dataset_layout_tab.py
│   │   ├── dataset_actions.py
│   │   ├── dataset_utils.py
│   │   ├── dataset_controls.py
│   │   ├── data_config_panel.py
│   │   └── dataset_thumb_layout.py
│   ├── training/               # Training pipeline (NEW)
│   │   ├── start_button_handler.py
│   │   ├── output_manager.py
│   │   └── training_dataset_block.py
│   ├── pages/                 # Configuration pages
│   │   ├── training_config.py
│   │   ├── training_data_config.py
│   │   ├── training_monitor.py
│   │   ├── model_field_config.py
│   │   ├── video_config.py
│   │   ├── network_config.py
│   │   └── optimizer_config.py
│   ├── _styles.py             # UI styling
│   └── utils/                 # Utilities
│       ├── config_utils.py
│       ├── process_cleanup.py
│       ├── console_cleanup.py
│       ├── file_dialogs.py
│       └── utils_top_menu.py
├── ui_popups/               # Popup dialogs (11 files)
│   ├── popup_dialog_base.py
│   ├── unified_popup_dialog.py
│   ├── image_editor.py
│   ├── video_editor.py
│   ├── area_editor.py
│   ├── unified_context_menu.py
│   └── ... (specialized dialogs)
└── modules/                 # External modules
    ├── minimax-remover/     # MiniMax object removal
    └── joycaption/          # Caption generation
```

### Training Engine Structure

```
diffusion-trainers/
└── diffusion-pipe/
    └── musubi_tuner/
        ├── ltx2_train_network.py         # Core training
        ├── ltx2_cache_latents.py         # Latents caching
        ├── ltx2_cache_text_encoder_outputs.py  # TE caching
        └── ltx2_validate.py              # Validation
```

**External Integration:**
- `musubi_ltx2.py` (project root) - Bridge between Flet UI and Musubi training

### Scripts & Utilities

```
scripts/
├── convert_comfy_to_training_lora.py  # LoRA format conversion
├── rerank_lora.py                     # LoRA rank adjustment
├── caption_llava.py                   # LLaVA captioning
├── caption_qwen.py                    # Qwen captioning
├── caption_joy.py                     # JoyCaption
├── split_scenes.py                    # Video scene splitting
├── deep_video_analysis.py             # Video analysis
└── analyze_video_chunks.py            # Chunk processing
```

---

## 10. Data Flow

### Training Flow (Musubi LTX-2)
1. User configures training via UI tabs (`tab_training_view.py`)
2. Configuration validated and saved to TOML
3. User clicks start → `start_button_handler.py` invoked
4. `musubi_ltx2.py` workflow orchestrates:
   - Cache latents (`ltx2_cache_latents.py`)
   - Train network (`ltx2_train_network.py`)
   - Validate (`ltx2_validate.py`)
5. Output streamed via `output_manager.py` to UI console

### Dataset Management Flow
1. Browse datasets via `dataset_manager/`
2. Select dataset/view thumbnails (`dataset_thumb_layout.py`)
3. Configure via `data_config_panel.py`
4. Apply operations (crop, caption, etc.) via `dataset_actions.py`
5. Changes persisted to dataset metadata

### Model/LoRA Flow
1. User enters model URL in Models tab
2. Download executed via subprocess
3. Model stored in models directory
4. LoRA conversion available via `scripts/convert_comfy_to_training_lora.py`

---

## 11. Dependencies

```
flet_app.py
    ├── settings.py (config)
    ├── theme_config.py (theming)
    ├── flet_app_top_menu.py (menu)
    ├── tab_training_view.py
    │   └── ui/training/ (pipeline)
    ├── dataset_manager/
    └── tab_tools_view.py

UI Components
    ├── utils/ (config, process, console)
    ├── pages/ (config pages)
    └── _styles.py (styling)

Popup System
    └── popup_dialog_base.py (base class)
        ├── unified_popup_dialog.py
        ├── image_editor.py
        ├── video_editor.py
        └── ... (specialized dialogs)
```

### Key External Dependencies
- **flet** - Primary UI framework
- **loguru** - Logging
- **PIL** - Image processing
- **tomli/tomllib** - TOML parsing
- **safetensors** - Model weights

---

## 12. Recent Changes (Since 2026-01-26)

| Change | Impact |
|--------|--------|
| New `ui/training/` module | Better training pipeline organization |
| Musubi LTX-2 integration | Dedicated LTX-2 training workflow |
| `output_manager.py` | Unified console output streaming |
| `start_button_handler.py` | Separated training execution logic |
| `dataset_thumb_layout.py` | Improved thumbnail grid layout |

---

*Note: Directories `dp_env/`, `models/`, `workspace/`, and `.git/` are excluded as they contain environment files, model data, workspace storage, and version control respectively.*
