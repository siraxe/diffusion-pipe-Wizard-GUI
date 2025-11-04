# LongCat Patch Module

Minimal patch to add LongCat-Video support to diffusion-pipe.

## Files Added

This patch includes only the essential files needed for LongCat-Video support:

- `train.py`: Modified training script with longcat model support (lines 349-351)
- `models/longcat_video.py`: LongCat-Video model implementation
- `examples/longcat.toml`: Example configuration file

## Usage

```bash
cd C:\Users\E\Desktop\QQ\flet_app\modules\longcat_patch
python train.py --config examples/longcat.toml
```

## Requirements

- LongCat-Video submodule (automatically initialized by apply_longcat_patch.sh)
- Base diffusion-pipe installation

## Automatic Setup

Use the patch application script to automatically handle all dependencies:

```bash
cd C:\Users\E\Desktop\QQ
bash apply_longcat_patch.sh
```

This will:
1. Copy all necessary files to diffusion-pipe
2. Update .gitmodules to include LongCat-Video
3. Initialize the LongCat-Video submodule automatically

## Model Type

Use `type = 'longcat'` in your configuration to enable LongCat-Video training.