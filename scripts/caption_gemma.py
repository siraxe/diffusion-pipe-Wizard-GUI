#!/usr/bin/env python3

"""
Caption images in a folder using Gemma-4-E4B (GGUF + mmproj via llama-cpp-python),
writing .txt caption files next to each image.

Example:
  caption_gemma.py /path/to/images \
      --instruction "Describe this image in detail." \
      --max-new-tokens 256 \
      --model-path models/text_encoders/gemma-4-E4B/Gemma-4-E4B-Q8_K_P.gguf \
      --mmproj-path models/text_encoders/gemma-4-E4B/mmproj-Gemma-4-E4B-f16.gguf
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import List

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
except ImportError:
    print("[ERROR] llama-cpp-python not installed or incompatible.", flush=True)
    raise RuntimeError(
        "llama-cpp-python is required. Install with:\n"
        "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
    )


def _list_images(dir_path: Path, selected_files: List[str] | None) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    paths: List[Path] = []
    if selected_files:
        sset = set(selected_files)
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix.lower() in exts and p.name in sset:
                paths.append(p)
    else:
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    paths.sort()
    return paths


def _image_to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _caption_single(llm: Llama, image_path: Path, instruction: str, max_new_tokens: int) -> str:
    data_url = _image_to_data_url(image_path)
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        max_tokens=max_new_tokens,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"].strip()


def main() -> int:
    print("[STATUS] Starting Gemma-4-E4B caption script...", flush=True)
    sys.stdout.flush()
    import argparse

    p = argparse.ArgumentParser(description="Caption images with Gemma-4-E4B (GGUF) and write .txt files")
    p.add_argument("input_dir", type=str, help="Folder with images")
    p.add_argument("--output", type=str, required=False, help="(Ignored) Kept for compatibility")
    p.add_argument("--instruction", type=str, required=True, help="Prompt/instruction text")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--model-path", type=str, default="models/text_encoders/gemma-4-E4B/Gemma-4-E4B-Q8_K_P.gguf")
    p.add_argument("--mmproj-path", type=str, default="models/text_encoders/gemma-4-E4B/mmproj-Gemma-4-E4B-f16.gguf")
    p.add_argument("--selected-files", type=str, default="", help="Comma-separated base filenames to process")

    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise RuntimeError(f"Input dir not found: {in_dir}")

    model_file = Path(args.model_path)
    mmproj_file = Path(args.mmproj_path)
    if not model_file.is_file():
        raise RuntimeError(f"Model file not found: {model_file}")
    if not mmproj_file.is_file():
        raise RuntimeError(f"mmproj file not found: {mmproj_file}")

    selected: List[str] | None = None
    if args.selected_files:
        selected = [s for s in (args.selected_files.split(",") if args.selected_files else []) if s]

    images = _list_images(in_dir, selected)
    if not images:
        print("No images found to caption.")
        return 0

    total = len(images)
    print(f"[STATUS] Found {total} image(s) to caption. Loading model...", flush=True)
    print(f"[STATUS] Model: {model_file}", flush=True)
    print(f"[STATUS] mmproj: {mmproj_file}", flush=True)
    sys.stdout.flush()

    t_load = time.time()
    try:
        chat_handler = Gemma4ChatHandler(
            clip_model_path=str(mmproj_file),
            verbose=False,
        )
    except Exception:
        raise
    try:
        llm = Llama(
            model_path=str(model_file),
            chat_handler=chat_handler,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", flush=True)
        raise
    load_sec = time.time() - t_load
    print(f"[STATUS] Model loaded in {load_sec:.1f}s. Starting captioning...", flush=True)

    wrote = 0
    t0 = time.time()
    for idx, img_path in enumerate(images, 1):
        rel = os.path.relpath(str(img_path), str(in_dir))
        t_start = time.time()
        try:
            cap = _caption_single(llm, img_path, args.instruction, args.max_new_tokens)
            txt_path = img_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as ftxt:
                ftxt.write(cap)
            wrote += 1
        except Exception as ex:
            print(f"[ERROR] {rel}: {ex}", flush=True)

        elapsed = time.time() - t_start
        avg = (time.time() - t0) / idx
        remaining = avg * (total - idx)
        eta_min, eta_sec = divmod(int(remaining), 60)
        print(f"[PROGRESS] {idx}/{total} {rel} | {elapsed:.1f}s ETA: {eta_min:d}:{eta_sec:02d}", flush=True)

    print(f"[DONE] Wrote {wrote}/{total} caption text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
