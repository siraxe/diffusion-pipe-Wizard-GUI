import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='diffusers')

import torch
from diffusers.utils import export_to_video
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline
from PIL import Image
import numpy as np
import argparse
import os

random_seed = 42
device = torch.device("cuda:0")

script_dir = os.path.dirname(__file__)
# Resolve model directory relative to project root (one level above flet_app):
# <project_root>/models/_misc/minimax-remover
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
model_root = os.path.join(project_root, "models", "_misc", "minimax-remover")

def _load_local_or_fail(loader, path, name, **kwargs):
    try:
        return loader(path, local_files_only=True, **kwargs)
    except Exception as e:
        raise FileNotFoundError(
            f"Missing local {name} at: {path}.\n"
            f"Place the required files (e.g., config.json and weights) under this directory.\n"
            f"No Hugging Face download is attempted (offline-only)."
        ) from e

vae = _load_local_or_fail(AutoencoderKLWan.from_pretrained, os.path.join(model_root, "vae"), "VAE", torch_dtype=torch.float16)
transformer = _load_local_or_fail(Transformer3DModel.from_pretrained, os.path.join(model_root, "transformer"), "Transformer", torch_dtype=torch.float16)
scheduler = _load_local_or_fail(UniPCMultistepScheduler.from_pretrained, os.path.join(model_root, "scheduler"), "Scheduler")

pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
pipe.to(device)

# the iterations is the hyperparameter for mask dilation
def inference(pixel_values, masks, video_length, height, width, fps, video_out_path, iterations=6):
    video = pipe(
        images=pixel_values,
        masks=masks,
        num_frames=video_length,
        height=height,
        width=width,
        num_inference_steps=12,
        generator=torch.Generator(device=device).manual_seed(random_seed),
        iterations=iterations
    ).frames[0]
    export_to_video(video, video_out_path, fps=fps)

def load_video(video_path):
    vr = VideoReader(video_path)
    video_length = len(vr)
    height, width, _ = vr.get_batch([0]).shape[1:]
    fps = vr.get_avg_fps()
    images = vr.get_batch(list(range(video_length))).asnumpy()
    images = torch.from_numpy(images)/127.5 - 1.0
    return images, video_length, height, width, fps

def load_image_mask(image_path, video_length, height, width):
    mask_image = Image.open(image_path).convert("L") # Open as grayscale
    mask_image = mask_image.resize((width, height))
    mask_np = np.array(mask_image)
    
    # Normalize mask to 0-1 and expand dimensions to (H, W, 1)
    mask_np = mask_np / 255.0
    mask_np = np.expand_dims(mask_np, axis=-1)

    # Replicate the single image mask for each frame of the video
    masks = np.tile(mask_np, (video_length, 1, 1, 1))
    masks = torch.from_numpy(masks)
    return masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Minimax Remover with an image mask.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--image_mask_path", type=str, required=True, help="Path to the input image mask file.")
    
    args = parser.parse_args()

    video_path = args.video_path
    image_mask_path = args.image_mask_path

    # Generate output video path
    video_dir, video_filename = os.path.split(video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    video_out_path = os.path.join(video_dir, f"{video_name}_clean{video_ext}")

    images, video_length, height, width, fps = load_video(video_path)
    masks = load_image_mask(image_mask_path, video_length, height, width)

    inference(images, masks, video_length, height, width, fps, video_out_path)
