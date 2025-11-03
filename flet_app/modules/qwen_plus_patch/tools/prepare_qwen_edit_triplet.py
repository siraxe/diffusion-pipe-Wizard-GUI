#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image, ImageOps


def load_rgba(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def letterbox_to_height(img: Image.Image, height: int, pad_color=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    scale = height / h
    new_w = max(1, int(round(w * scale)))
    img_resized = img.resize((new_w, height), Image.LANCZOS)
    if new_w != height:
        pad_left = (height - new_w) // 2
        pad_right = height - new_w - pad_left
        img_resized = ImageOps.expand(
            img_resized,
            border=(pad_left, 0, pad_right, 0),
            fill=pad_color,
        )
    return img_resized


def make_composite(
    left_img: Image.Image,
    right_img: Image.Image,
    height: int = 1024,
    sep: int = 0,
    pad_color=(0, 0, 0),
) -> Image.Image:
    left = letterbox_to_height(left_img, height, pad_color)
    right = letterbox_to_height(right_img, height, pad_color)
    width = left.width + (sep if sep > 0 else 0) + right.width
    canvas = Image.new("RGB", (width, height), pad_color)
    x = 0
    canvas.paste(left, (x, 0))
    x += left.width
    if sep > 0:
        x += sep
    canvas.paste(right, (x, 0))
    return canvas


def sort_key(path: Path):
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def format_base(path: Path) -> str:
    stem = path.stem
    try:
        return f"{int(stem):04d}"
    except ValueError:
        return stem


def main(src_root: str, dst_root: str, height: int = 1024) -> None:
    src = Path(src_root)
    dst = Path(dst_root)
    out_images = dst / "train" / "images"
    out_control = dst / "train" / "control"
    out_images.mkdir(parents=True, exist_ok=True)
    out_control.mkdir(parents=True, exist_ok=True)

    input_dir = src / "input"
    ref_dir = src / "breast"
    target_dir = src / "output"

    inputs = sorted(input_dir.glob("*.png"), key=sort_key)
    refs = sorted(ref_dir.glob("*.png"), key=sort_key)
    outs = sorted(target_dir.glob("*.png"), key=sort_key)
    if not (len(inputs) == len(refs) == len(outs) and len(inputs) > 0):
        raise ValueError("Mismatched counts.")

    for pin, pref, pout in zip(inputs, refs, outs):
        left_src = Image.open(pin).convert("RGB")
        right_ref = load_rgba(pref)
        left_tgt = Image.open(pout).convert("RGB")

        control = make_composite(left_src, right_ref, height, sep=0)
        target = make_composite(left_tgt, right_ref, height, sep=0)

        base = format_base(pin)

        control.save(out_control / f"{base}.png")
        target.save(out_images / f"{base}.png")
        (out_images / f"{base}.txt").write_text(" ")

    print(f"Done. Wrote composites to: {dst.resolve()}")


if __name__ == "__main__":
    src_root = sys.argv[1]
    dst_root = sys.argv[2]
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    main(src_root, dst_root, height=height)
