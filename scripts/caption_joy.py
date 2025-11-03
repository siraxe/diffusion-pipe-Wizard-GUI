#!/usr/bin/env python3

"""
Caption images in a folder using JoyCaption (LLaVA HF variant), writing captions.json
compatible with the existing dataset pipeline, which can then be converted to .txt files.

Example:
  caption_joy.py /path/to/images --output /path/to/images/captions.json \
      --instruction "Write a descriptive caption for this image in a formal tone." \
      --max-new-tokens 120 --model-path models/_misc/joycaption-llava
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
import torchvision.transforms.functional as TVF  # type: ignore

try:
    from transformers import AutoTokenizer, LlavaForConditionalGeneration  # type: ignore
except Exception as _ex:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    LlavaForConditionalGeneration = None  # type: ignore


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


def _caption_single(
    model, tokenizer, image_path: Path, instruction: str, max_new_tokens: int
) -> str:
    # Build conversation (aligns with JoyCaption script defaults)
    convo = [
        {
            "role": "system",
            "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
        },
        {"role": "user", "content": instruction},
    ]
    convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    convo_tokens = tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

    # Load and preprocess image to 384x384 RGB and to tensor in [-1, 1]
    img = Image.open(str(image_path))
    if img.size != (384, 384):
        img = img.resize((384, 384), Image.LANCZOS)
    img = img.convert("RGB")
    pixel_values = TVF.pil_to_tensor(img).to(dtype=torch.float32)  # [C,H,W] in [0,255]
    pixel_values = pixel_values / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # -> [-1,1]
    pixel_values = pixel_values.unsqueeze(0)  # [1,C,H,W]

    image_token_id = getattr(model.config, "image_token_index", None)
    image_seq_length = getattr(model.config, "image_seq_length", None)
    if image_token_id is None or image_seq_length is None:
        raise RuntimeError("JoyCaption model config missing image_token_index/image_seq_length")

    # Expand image token
    input_tokens: List[int] = []
    for tok in convo_tokens:
        if tok == image_token_id:
            input_tokens.extend([image_token_id] * int(image_seq_length))
        else:
            input_tokens.append(int(tok))

    input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    # Devices/dtypes
    vision_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
    vision_device = model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
    language_device = model.language_model.get_input_embeddings().weight.device

    pixel_values = pixel_values.to(vision_device, dtype=vision_dtype, non_blocking=True)
    input_ids = input_ids.to(language_device, non_blocking=True)
    attention_mask = attention_mask.to(language_device, non_blocking=True)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            suppress_tokens=None,
            use_cache=True,
        )

    # Trim off prompts per JoyCaption convention
    eoh = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ids_list = out_ids[0].tolist()
    # Remove up to last end_of_header occurrences
    while True:
        try:
            i = ids_list.index(eoh)
        except ValueError:
            break
        ids_list = ids_list[i + 1 :]
    try:
        j = ids_list.index(eot)
        ids_list = ids_list[:j]
    except ValueError:
        pass

    text = tokenizer.decode(ids_list, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text.strip()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Caption images with JoyCaption and write .txt files next to images")
    p.add_argument("input_dir", type=str, help="Folder with images")
    # --output kept optional for backward-compatibility; ignored when writing .txt files
    p.add_argument("--output", type=str, required=False, help="(Ignored) Previously used for captions.json output")
    p.add_argument("--instruction", type=str, required=True, help="Prompt/instruction text")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--model-path", type=str, default="models/_misc/joycaption-llava")
    p.add_argument("--selected-files", type=str, default="", help="Comma-separated base filenames to process")

    args = p.parse_args()

    if AutoTokenizer is None or LlavaForConditionalGeneration is None:
        raise RuntimeError("transformers with LLaVA support is required for JoyCaption")

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise RuntimeError(f"Input dir not found: {in_dir}")

    selected: List[str] | None = None
    if args.selected_files:
        selected = [s for s in (args.selected_files.split(",") if args.selected_files else []) if s]

    # Collect images
    images = _list_images(in_dir, selected)
    if not images:
        print("No images found to caption.")
        return 0

    # Load tokenizer/model
    device_map = 0 if torch.cuda.is_available() else "cpu"
    llm_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=llm_dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()

    wrote = 0
    for img_path in images:
        rel = os.path.relpath(str(img_path), str(in_dir))
        try:
            cap = _caption_single(model, tok, img_path, args.instruction, args.max_new_tokens)
            txt_path = img_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as ftxt:
                ftxt.write(cap)
            wrote += 1
            print(f"[OK] {rel}")
        except Exception as ex:
            print(f"[ERR] {rel}: {ex}")
    print(f"Wrote {wrote} caption text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
