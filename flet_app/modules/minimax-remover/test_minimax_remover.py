import os
import torch
from diffusers.utils import export_to_video
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

random_seed = 42
video_length = 81
device = torch.device("cuda:0")

script_dir = os.path.dirname(__file__)
# Resolve model directory relative to project root (one level above flet_app):
# <project_root>/models/_misc/minimax-remover
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
model_dir = os.path.join(project_root, "models", "_misc", "minimax-remover")

def _load_local_or_fail(loader, path, name, **kwargs):
    try:
        return loader(path, local_files_only=True, **kwargs)
    except Exception as e:
        raise FileNotFoundError(
            f"Missing local {name} at: {path}.\n"
            f"Place the required files (e.g., config.json and weights) under this directory.\n"
            f"No Hugging Face download is attempted (offline-only)."
        ) from e

vae = _load_local_or_fail(AutoencoderKLWan.from_pretrained, os.path.join(model_dir, "vae"), "VAE", torch_dtype=torch.float16)
transformer = _load_local_or_fail(Transformer3DModel.from_pretrained, os.path.join(model_dir, "transformer"), "Transformer", torch_dtype=torch.float16)
scheduler = _load_local_or_fail(UniPCMultistepScheduler.from_pretrained, os.path.join(model_dir, "scheduler"), "Scheduler")

pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
pipe.to(device)

# the iterations is the hyperparameter for mask dilation
def inference(pixel_values, masks, iterations=6):
    video = pipe(
        images=pixel_values,
        masks=masks,
        num_frames=video_length,
        height=544,
        width=960,
        num_inference_steps=12,
        generator=torch.Generator(device=device).manual_seed(random_seed),
        iterations=iterations
    ).frames[0]
    export_to_video(video, "./output.mp4")

def load_video(video_path):
    vr = VideoReader(video_path)
    images = vr.get_batch(list(range(video_length))).asnumpy()
    images = torch.from_numpy(images)/127.5 - 1.0
    return images

def load_mask(mask_path):
    vr = VideoReader(mask_path)
    masks = vr.get_batch(list(range(video_length))).asnumpy()
    masks = torch.from_numpy(masks)
    masks = masks[:, :, :, :1]
    masks[masks > 20] = 255
    masks[masks < 255] = 0
    masks = masks / 255.0
    return masks

# Example test assets (update paths as needed if available)
video_path = os.path.join(script_dir, "sample_video.mp4")
mask_path = os.path.join(script_dir, "sample_mask.mp4")

images = load_video(video_path)
masks = load_mask(mask_path)

inference(images, masks)
