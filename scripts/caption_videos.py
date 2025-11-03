#!/usr/bin/env python3

"""
Auto-caption videos using vision-language models.

This script provides a command-line interface for generating captions for videos using
a vision-language model. It supports processing individual videos or entire directories,
customizing the captioning model, and saving the results to various formats.

The paths to videos in the generated dataset/captions file will be RELATIVE to the
directory where the output file is stored. This makes the dataset more portable and
easier to use in different environments.

Basic usage:
    # Caption a single video
    caption_videos.py video.mp4 --output captions.txt

    # Caption all videos in a directory
    caption_videos.py videos_dir/ --output captions.csv

    # Caption with custom instruction
    caption_videos.py video.mp4 --instruction "Describe what happens in this video in detail."

Advanced usage:
    # Use specific captioner type and device
    caption_videos.py videos_dir/ --captioner-type llava_next_7b --device cuda:0

    # Process videos with specific extensions and save as JSON
    caption_videos.py videos_dir/ --extensions mp4,mov,avi --output captions.json
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path to resolve imports like 'flet_app'
# Assumes the script is run from within the project directory structure
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # Start one level up from 'scripts'

# Try to walk up to find a known project root file (e.g., pyproject.toml).
# If not found, fall back to the immediate parent that contains this script.
_probe = project_root
while not (_probe / "pyproject.toml").exists() and _probe != _probe.parent:
    _probe = _probe.parent

# If we didn't actually find a pyproject.toml, keep the original parent as root.
if not (_probe / "pyproject.toml").exists():
    _probe = project_root

project_root = _probe

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import csv
import json
from enum import Enum
from flet_app.settings import settings

# --- Write PID to file for external process tracking ---
try:
    pid_file = os.path.join(os.path.dirname(__file__), 'caption_pid.txt')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
except Exception:
    pass

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from transformers.utils.logging import disable_progress_bar
from dataclasses import dataclass
from typing import Protocol
from PIL import Image
import numpy as np
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration  # type: ignore
except Exception as _ex:
    AutoProcessor = None  # type: ignore
    LlavaForConditionalGeneration = None  # type: ignore
try:
    # Local video captioner wrappers
    from scripts.caption_llava import LlavaNextVideoCaptioner  # type: ignore
except Exception:
    LlavaNextVideoCaptioner = None  # type: ignore
try:
    from scripts.caption_qwen import Qwen3VLCaptioner  # type: ignore
except Exception:
    Qwen3VLCaptioner = None  # type: ignore

# Lightweight HF backend for Qwen3-VL-4B
try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except Exception:
    Qwen3VLForConditionalGeneration = None  # type: ignore

DEFAULT_VLM_CAPTION_INSTRUCTION = (
    "Shortly describe the content of this video in one or two sentences."
)


class MediaCaptioningModel(Protocol):
    def caption(self, path: Path, fps: int, clean_caption: bool, max_new_tokens: int) -> str: ...


@dataclass
class CaptionerType:
    LLaVA_NEXT_7B: str = "llava_next_7b"


def _load_single_frame(path: Path, fps: int) -> Image.Image:
    # Try to extract a representative frame (middle frame) from video; fallback to open as image
    try:
        from decord import VideoReader  # type: ignore
        vr = VideoReader(str(path))
        idx = len(vr) // 2 if len(vr) > 0 else 0
        frame = vr[idx].asnumpy()
        if frame.ndim == 3 and frame.shape[2] == 3:
            img = Image.fromarray(frame[:, :, ::-1])  # decord is BGR; convert to RGB
        else:
            img = Image.fromarray(frame)
        return img
    except Exception:
        # Try OpenCV as a fallback
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                target = max(0, total // 2)
                if target:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame)
        except Exception:
            pass
        # Not a video or cannot decode — try as image
        return Image.open(str(path)).convert("RGB")


class LlavaNextCaptioner:
    def __init__(self, model_id_or_path: str, device: str = "cuda", instruction: str = DEFAULT_VLM_CAPTION_INSTRUCTION):
        if AutoProcessor is None or LlavaForConditionalGeneration is None:
            raise RuntimeError("transformers with LLaVA support is required")
        self.processor = AutoProcessor.from_pretrained(model_id_or_path, local_files_only=True)
        # Choose dtype based on cuda availability
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id_or_path,
            local_files_only=True,
            torch_dtype=torch_dtype or "auto",
            device_map=0 if device.startswith("cuda") else "cpu",
        )
        self.model.eval()
        self.device = device
        self.instruction = instruction

    def caption(self, path: Path, fps: int, clean_caption: bool, max_new_tokens: int) -> str:
        img = _load_single_frame(path, fps)
        # Build chat template with a single image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "image"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.model.device)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        out = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        # Extract assistant reply if present
        if "ASSISTANT:" in out:
            out = out.split("ASSISTANT:", 1)[-1].strip()
        return out.strip()


class Qwen3VLHFLocalCaptioner(MediaCaptioningModel):
    def __init__(self, model_id_or_path: str, *, device: str = "cuda", instruction: str = DEFAULT_VLM_CAPTION_INSTRUCTION) -> None:
        if AutoProcessor is None or Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("transformers>=4.44 is required for Qwen3-VL HF backend")
        # dtype="auto", device_map="auto" minimize VRAM
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id_or_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)
        self.instruction = instruction

    def caption(self, path: Path, fps: int, clean_caption: bool, max_new_tokens: int) -> str:
        p = str(path)
        ext = path.suffix.lower()
        is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}
        content = []
        if is_video:
            content.append({"type": "video", "video": p})
        else:
            content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": self.instruction})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return (output_text[0] if output_text else "").strip()


def create_captioner(
    captioner_type: str,
    device: str,
    use_8bit: bool,
    vlm_instruction: str,
    llava_model_id_or_path: str,
    qwen_model_id_or_path: str,
) -> MediaCaptioningModel:
    ctype = (captioner_type or "").lower().strip()
    # Prefer project root from settings if available
    settings_project_root = settings.get("PROJECT_ROOT", None)
    pr = Path(settings_project_root) if settings_project_root else project_root
    if ctype in (getattr(CaptionerType, 'LLaVA_NEXT_7B', 'llava_next_7b'), "llava", "llava_next", "llava_next_7b"):
        model_path = llava_model_id_or_path
        # Default to curated local folder if given an org/repo-like string
        if os.path.sep not in model_path and "/" in model_path:
            # Fallback to curated local
            alt = pr / "models" / "_misc" / "LLaVA-NeXT-Video-7B-hf"
            model_path = str(alt)
        
        if LlavaNextVideoCaptioner is None:
            raise RuntimeError("LlavaNextVideoCaptioner not available; ensure scripts/caption_llava.py is importable")
        return LlavaNextVideoCaptioner(
            model_path,
            device=device,
            instruction=vlm_instruction,
            use_4bit=use_8bit,
            use_flash_attention_2=True,
        )
    if ctype in ("qwen3_vl_8b", "qwen3_vl_4b", "qwen3_vl_4b_hf", "qwen3_vl_8b_hf", "qwen3", "qwen3_vl"):
        model_path = qwen_model_id_or_path
        if os.path.sep not in model_path and "/" in model_path:
            # Choose a sensible curated default based on variant
            if ctype == "qwen3_vl_4b":
                alt = pr / "models" / "_misc" / "Qwen3-VL-4B-Instruct"
            else:
                alt = pr / "models" / "_misc" / "Qwen3-VL-8B-Instruct"
            model_path = str(alt)
        if ctype in ("qwen3_vl_4b_hf", "qwen3_vl_8b_hf"):
            # Lightweight HF Transformers backend for 4B
            return Qwen3VLHFLocalCaptioner(
                model_path,
                device=device,
                instruction=vlm_instruction,
            )
        else:
            if Qwen3VLCaptioner is None:
                raise RuntimeError("Qwen3VLCaptioner not available; ensure vLLM and qwen_vl_utils are installed")
            return Qwen3VLCaptioner(
                model_path,
                device=device,
                instruction=vlm_instruction,
            )
    # Unknown type
    raise RuntimeError(f"Unknown captioner_type: {captioner_type}")


console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Auto-caption videos using vision-language models.",
)

disable_progress_bar()


class OutputFormat(str, Enum):
    """Available output formats for captions."""

    TXT = "txt"  # Separate files for captions and video paths, one caption / video path per line
    CSV = "csv"  # CSV file with video path and caption columns
    JSON = "json"  # JSON file with video paths as keys and captions as values
    JSONL = "jsonl"  # JSON Lines file with one JSON object per line


def caption_media(
    input_path: Path,
    output_path: Path,
    captioner: MediaCaptioningModel,
    extensions: list[str],
    recursive: bool,
    fps: int,
    clean_caption: bool,
    output_format: OutputFormat,
    override: bool,
    max_new_tokens: int,
    selected_files: list[str] = None, # New parameter
) -> None:
    """Caption videos and images using the provided captioning model.
    Args:
        input_path: Path to input video file or directory
        output_path: Path to output caption file
        captioner: Video captioning model
        extensions: List of video file extensions to include
        recursive: Whether to search subdirectories recursively
        fps: Frames per second to sample from videos
        clean_caption: Whether to clean up captions
        output_format: Format to save the captions in
        override: Whether to override existing captions
        selected_files: List of specific filenames to caption. If provided, only these files will be processed.
    """

    # Get list of all media files in the input path
    all_media_files = _get_media_files(input_path, extensions, recursive)

    if not all_media_files:
        console.print("[bold yellow]No media files found to process.[/]")
        return

    console.print(f"Found [bold]{len(all_media_files)}[/] media files in total.")

    # Filter media files based on selected_files if provided
    if selected_files:
        selected_basenames = {Path(f).name for f in selected_files}
        media_files = [f for f in all_media_files if f.name in selected_basenames]
        if not media_files:
            console.print("[bold yellow]No selected media files found in the input path.[/]")
            return
        console.print(f"Processing [bold]{len(media_files)}[/] selected media files.")
        # If specific files are selected, we always override their captions
        override = True
    else:
        media_files = all_media_files
        console.print(f"Processing [bold]{len(media_files)}[/] media files.")


    # Get the base directory for relative paths (the directory containing the output file)
    base_dir = output_path.parent.resolve()
    console.print(f"Using [bold blue]{base_dir}[/] as base directory for relative paths")

    # Load existing captions if the output file exists
    existing_captions = _load_existing_captions(output_path, output_format)

    # Convert existing captions keys to absolute paths for comparison
    existing_captions_abs = {}
    for rel_path, caption in existing_captions.items():
        abs_path = str((base_dir / rel_path).resolve())
        existing_captions_abs[abs_path] = caption

    # Filter out media that already have captions if not overriding
    media_to_process = []
    skipped_media = []

    for media_file in media_files:
        media_path_str = str(media_file.resolve())
        if not override and media_path_str in existing_captions_abs:
            skipped_media.append(media_file)
        else:
            media_to_process.append(media_file)

    if skipped_media:
        console.print(f"[bold yellow]Skipping [bold]{len(skipped_media)}[/] media that already have captions.[/]")

    if not media_to_process:
        console.print("[bold yellow]No media to process. All media already have captions.[/]")
        if not selected_files: # Only suggest override if not already processing selected files
            console.print("[bold yellow]Use --override to recaption all media.[/]")
        return

    console.print(f"Actually processing [bold]{len(media_to_process)}[/] media.")

    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    # Start with existing captions
    captions = existing_captions.copy()

    with progress:
        task = progress.add_task("Generating captions", total=len(media_to_process))

        for media_file in media_to_process:
            # Update progress description to show current file
            progress.update(task, description=f"Captioning [bold blue]{media_file.name}[/]")

            try:
                # Generate caption for the media
                caption = captioner.caption(
                    path=media_file,
                    fps=fps,
                    clean_caption=clean_caption,
                    max_new_tokens=max_new_tokens,
                )

                # Convert absolute path to relative path (relative to the output file's directory)
                rel_path = str(media_file.resolve().relative_to(base_dir))
                # Store the caption with the relative path as key
                captions[rel_path] = caption
                print(f"Added caption for {media_file.name}")

            except Exception as e:
                console.print(f"[bold red]Error captioning [bold blue]{media_file}[/]: {e}[/]")

            # Advance progress bar
            progress.advance(task)

    # Save captions to file
    _save_captions(captions, output_path, output_format)

    # Print summary
    processed_media_count = len(media_to_process)
    console.print(
        f"[bold green]✓[/] Captioned [bold]{processed_media_count}[/] media successfully.",
    )


def _get_media_files(
    input_path: Path,
    extensions: list[str] = settings.MEDIA_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Get all media files from the input path."""
    input_path = Path(input_path)
    # Normalize extensions to lowercase without dots
    extensions = [ext.lower().lstrip(".") for ext in extensions]

    if input_path.is_file():
        # If input is a file, check if it has a valid extension
        if input_path.suffix.lstrip(".").lower() in extensions:
            return [input_path]
        else:
            typer.echo(f"Warning: {input_path} is not a recognized media file. Skipping.")
            return []
    elif input_path.is_dir():
        # If input is a directory, find all media files
        media_files = []

        # Define the glob pattern based on whether we're searching recursively
        glob_pattern = "**/*" if recursive else "*"

        # Find all files with the specified extensions
        for ext in extensions:
            media_files.extend(input_path.glob(f"{glob_pattern}.{ext}"))

        return sorted(media_files)
    else:
        typer.echo(f"Error: {input_path} does not exist.")
        raise typer.Exit(code=1)


def _save_captions(
    captions: dict[str, str],
    output_path: Path,
    format_type: OutputFormat,
) -> None:
    """Save captions to a file in the specified format.

    Args:
        captions: Dictionary mapping media paths to captions
        output_path: Path to save the output file
        format_type: Format to save the captions in
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Saving captions...[/]")

    match format_type:
        case OutputFormat.TXT:
            # Create two separate files for captions and media paths
            captions_file = output_path.with_stem(f"{output_path.stem}_captions")
            paths_file = output_path.with_stem(f"{output_path.stem}_paths")

            with captions_file.open("w", encoding="utf-8") as f:
                for caption in captions.values():
                    f.write(f"{caption}\n")

            with paths_file.open("w", encoding="utf-8") as f:
                for media_path in captions:
                    f.write(f"{media_path}\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{captions_file}[/]")
            console.print(f"[bold green]✓[/] Media paths saved to [cyan]{paths_file}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print(f"  caption_column='{captions_file.name}'")
            console.print(f"  video_column='{paths_file.name}'")

        case OutputFormat.CSV:
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["caption", "media_path"])
                for media_path, caption in captions.items():
                    writer.writerow([caption, media_path])

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case OutputFormat.JSON:
            # Format as list of dictionaries with caption and media_path keys
            json_data = [{"caption": caption, "media_path": media_path} for media_path, caption in captions.items()]

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case OutputFormat.JSONL:
            with output_path.open("w", encoding="utf-8") as f:
                for media_path, caption in captions.items():
                    f.write(json.dumps({"caption": caption, "media_path": media_path}, ensure_ascii=False) + "\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case _:
            raise ValueError(f"Unsupported output format: {format_type}")


def _load_existing_captions(  # noqa: PLR0912
    output_path: Path,
    format_type: OutputFormat,
) -> dict[str, str]:
    """Load existing captions from a file.

    Args:
        output_path: Path to the captions file
        format_type: Format of the captions file

    Returns:
        Dictionary mapping media paths to captions, or empty dict if file doesn't exist
    """
    if not output_path.exists():
        return {}

    console.print(f"[bold blue]Loading existing captions from [cyan]{output_path}[/]...[/]")

    existing_captions = {}

    try:
        match format_type:
            case OutputFormat.TXT:
                # For TXT format, we have two separate files
                captions_file = output_path.with_stem(f"{output_path.stem}_captions")
                paths_file = output_path.with_stem(f"{output_path.stem}_paths")

                if captions_file.exists() and paths_file.exists():
                    captions = captions_file.read_text(encoding="utf-8").splitlines()
                    paths = paths_file.read_text(encoding="utf-8").splitlines()

                    if len(captions) == len(paths):
                        existing_captions = dict(zip(paths, captions, strict=False))

            case OutputFormat.CSV:
                with output_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    # Skip header
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            caption, media_path = row[0], row[1]
                            existing_captions[media_path] = caption

            case OutputFormat.JSON:
                with output_path.open("r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    for item in json_data:
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case OutputFormat.JSONL:
                with output_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case _:
                raise ValueError(f"Unsupported output format: {format_type}")

        console.print(f"[bold green]✓[/] Loaded [bold]{len(existing_captions)}[/] existing captions")
        return existing_captions

    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not load existing captions: {e}[/]")
        return {}


@app.command()
def main(  # noqa: PLR0913
    input_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to input video/image file or directory containing media files",
        exists=True,
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Path to output file for captions. Format determined by file extension.",
    ),
    captioner_type: str = typer.Option(  # noqa: B008
        "llava_next_7b",
        "--captioner-type",
        "-c",
        help="Type of captioner to use. Valid values: 'llava_next_7b', 'qwen3_vl_8b', 'qwen3_vl_4b', 'qwen3_vl_4b_hf', 'qwen3_vl_8b_hf'",
        case_sensitive=False,
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use for inference (e.g., 'cuda', 'cuda:0', 'cpu')",
    ),
    use_8bit: bool = typer.Option(
        False,
        "--use-8bit",
        help="Whether to use 8-bit precision for the captioning model",
    ),
    instruction: str = typer.Option(
        DEFAULT_VLM_CAPTION_INSTRUCTION,
        "--instruction",
        "-i",
        help="Instruction to give to the captioning model",
    ),
    extensions: str = typer.Option(
        ",".join(settings.MEDIA_EXTENSIONS),
        "--extensions",
        "-e",
        help="Comma-separated list of media file extensions to process",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search for media files in subdirectories recursively",
    ),
    max_new_tokens: int = typer.Option(
        100,
        "--max-new-tokens",
        help="Maximum new tokens to generate for the caption.",
    ),
    fps: int = typer.Option(
        3,
        "--fps",
        "-f",
        help="Frames per second to sample from videos (ignored for images)",
    ),
    llava_model_id_or_path: str = typer.Option(
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "--llava-model",
        help="LLaVA model ID or local path (e.g., models/LLaVA-NeXT-Video-7B-hf or llava-hf/LLaVA-NeXT-Video-7B-hf)."
    ),
    qwen_model_id_or_path: str = typer.Option(
        "Qwen/Qwen3-VL-8B-Instruct",
        "--qwen-model",
        help="Qwen3 model ID or local path (e.g., models/_misc/Qwen3-VL-8B-Instruct or Qwen/Qwen3-VL-8B-Instruct)."
    ),
    clean_caption: bool = typer.Option(
        True,
        "--clean-caption",
        help="Whether to clean up captions by removing common VLM patterns",
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Whether to override existing captions for media",
    ),
    selected_files: str = typer.Option(
        None,
        "--selected-files",
        help="Comma-separated list of specific filenames (basename only) to caption. If provided, only these files will be processed and their captions will be overwritten.",
    ),
) -> None:
    """Auto-caption videos and images using vision-language models.

    This tool can process individual video/image files or directories of media files and generate
    captions using a vision-language model. The captions can be saved in various formats.
    """

    # Determine device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Parse extensions
    ext_list = [ext.strip() for ext in extensions.split(",")]

    # Parse selected files if provided
    selected_files_list = [f.strip() for f in selected_files.split(",")] if selected_files else None

    # Determine output path and format
    if output is None:
        output_format = OutputFormat.JSON
        if input_path.is_file():  # noqa: SIM108
            # Default to a JSON file with the same name as the input media
            output = input_path.with_suffix(".captions.json")
        else:
            # Default to a JSON file in the input directory
            output = input_path / "captions.json"
    else:
        # Determine format from file extension
        output_format = OutputFormat(Path(output).suffix.lstrip(".").lower())

    # Ensure output path is absolute
    output = Path(output).resolve()
    console.print(f"Output will be saved to [bold blue]{output}[/]")

    # Initialize captioning model
    with console.status("Loading captioning model...", spinner="dots"):
        captioner = create_captioner(
            captioner_type=captioner_type,
            device=device,
            use_8bit=use_8bit,
            vlm_instruction=instruction,
            llava_model_id_or_path=llava_model_id_or_path,
            qwen_model_id_or_path=qwen_model_id_or_path,
        )
        console.print("[bold green]✓[/] Captioning model loaded successfully")

    # Caption media files
    caption_media(
        input_path=input_path,
        output_path=output,
        captioner=captioner,
        extensions=ext_list,
        recursive=recursive,
        fps=fps,
        clean_caption=clean_caption,
        output_format=output_format,
        override=override,
        max_new_tokens=max_new_tokens,
        selected_files=selected_files_list, # Pass the parsed list
    )


if __name__ == "__main__":
    app()
