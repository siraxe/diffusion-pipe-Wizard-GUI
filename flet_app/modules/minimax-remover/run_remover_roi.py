"""
Minimax-Remover ROI-based Processing
Process only a selected region of the video instead of the whole video.
Much faster and preserves original video dimensions.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='diffusers')

import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline
from PIL import Image, ImageDraw
import numpy as np
import argparse
import os
from einops import rearrange
import cv2

random_seed = 42
device = torch.device("cuda:0")

script_dir = os.path.dirname(__file__)
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

# Load models once at module level
vae = _load_local_or_fail(AutoencoderKLWan.from_pretrained, os.path.join(model_root, "vae"), "VAE", torch_dtype=torch.float16)
transformer = _load_local_or_fail(Transformer3DModel.from_pretrained, os.path.join(model_root, "transformer"), "Transformer", torch_dtype=torch.float16)
scheduler = _load_local_or_fail(UniPCMultistepScheduler.from_pretrained, os.path.join(model_root, "scheduler"), "Scheduler")

pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
pipe.to(device)


def calculate_expanded_roi(
    selected_left: int, selected_top: int, selected_w: int, selected_h: int,
    video_width: int, video_height: int,
    padding: int = 80
) -> tuple:
    """
    Calculate expanded ROI that:
    1. Expands from selected area by padding pixels
    2. Stays within video bounds
    3. Is divisible by 32 (Wan VAE requires spatial dimensions divisible by 32 for proper encoding)

    Returns: (roi_left, roi_top, roi_w, roi_h, mask_left_in_roi, mask_top_in_roi, mask_w, mask_h)
    """
    # Wan VAE spatial downsample factor is 8, but we need dimensions divisible by 32
    # due to how the encoder handles padding with kernel_size=3, stride=2
    SPATIAL_DIVISOR = 32

    # Expand with padding
    expanded_left = max(0, selected_left - padding)
    expanded_top = max(0, selected_top - padding)
    expanded_right = min(video_width, selected_left + selected_w + padding)
    expanded_bottom = min(video_height, selected_top + selected_h + padding)

    roi_w = expanded_right - expanded_left
    roi_h = expanded_bottom - expanded_top

    # Make divisible by 32 for Wan VAE compatibility
    roi_w = ((roi_w + SPATIAL_DIVISOR - 1) // SPATIAL_DIVISOR) * SPATIAL_DIVISOR
    roi_h = ((roi_h + SPATIAL_DIVISOR - 1) // SPATIAL_DIVISOR) * SPATIAL_DIVISOR

    # Adjust left/top if we expanded beyond bounds to maintain divisibility
    if expanded_left + roi_w > video_width:
        expanded_left = max(0, video_width - roi_w)
    if expanded_top + roi_h > video_height:
        expanded_top = max(0, video_height - roi_h)

    # Recalculate in case bounds adjustment changed things
    roi_w = min(roi_w, video_width - expanded_left)
    roi_h = min(roi_h, video_height - expanded_top)

    # Ensure still divisible by 32 after bounds adjustment
    roi_w = (roi_w // SPATIAL_DIVISOR) * SPATIAL_DIVISOR
    roi_h = (roi_h // SPATIAL_DIVISOR) * SPATIAL_DIVISOR

    # Minimum size check
    min_size = SPATIAL_DIVISOR
    if roi_w < min_size or roi_h < min_size:
        print(f"Warning: ROI {roi_w}x{roi_h} is too small, using minimum {min_size}x{min_size}")
        roi_w = max(roi_w, min_size)
        roi_h = max(roi_h, min_size)

    # Calculate where the original selected area is within the ROI
    mask_left_in_roi = selected_left - expanded_left
    mask_top_in_roi = selected_top - expanded_top

    return expanded_left, expanded_top, roi_w, roi_h, mask_left_in_roi, mask_top_in_roi, selected_w, selected_h


def load_video_roi(video_path, roi_left, roi_top, roi_w, roi_h, target_length=None):
    """
    Load video and extract only the ROI region from each frame.

    Args:
        target_length: If provided, extend/truncate frames to match this length

    Returns:
        roi_images: Tensor of shape (video_length, roi_h, roi_w, 3)
        video_length, original_height, original_width, fps
    """
    vr = VideoReader(video_path)
    video_length = len(vr)
    full_height, full_width, _ = vr.get_batch([0]).shape[1:]
    fps = vr.get_avg_fps()

    # Load all frames
    images = vr.get_batch(list(range(video_length))).asnumpy()
    images = torch.from_numpy(images) / 127.5 - 1.0

    original_video_length = video_length

    # Adjust frame count for temporal compatibility - EXTEND instead of truncate
    temporal_scale = 4
    if video_length % temporal_scale != 0:
        # Extend to next multiple of temporal_scale by duplicating frames
        adjusted_length = ((video_length + temporal_scale - 1) // temporal_scale) * temporal_scale
        frames_to_add = adjusted_length - video_length

        if frames_to_add > 0:
            # Duplicate the last few frames to extend
            last_frames = images[-frames_to_add:] if frames_to_add <= video_length else images
            # If we need more frames than we have, cycle through the video
            if frames_to_add > video_length:
                cycles_needed = frames_to_add // video_length
                remainder = frames_to_add % video_length
                extra = torch.cat([images] * cycles_needed + [images[:remainder]], dim=0)
            else:
                extra = last_frames

            images = torch.cat([images, extra], dim=0)
            video_length = adjusted_length
            print(f"  Extended from {original_video_length} to {video_length} frames for temporal compatibility (duplicated last {frames_to_add} frame(s))")

    # If target_length is specified, match it (for compositing)
    if target_length is not None and video_length != target_length:
        if video_length < target_length:
            # Need to extend more - duplicate last frames
            frames_to_add = target_length - video_length
            last_frames = images[-frames_to_add:] if frames_to_add <= video_length else images
            if frames_to_add > video_length:
                cycles_needed = frames_to_add // video_length
                remainder = frames_to_add % video_length
                extra = torch.cat([images] * cycles_needed + [images[:remainder]], dim=0)
            else:
                extra = last_frames
            images = torch.cat([images, extra], dim=0)
            video_length = target_length
            print(f"  Extended to {video_length} frames to match original video")
        else:
            # Truncate if we somehow have more (shouldn't happen with above logic)
            images = images[:target_length]
            video_length = target_length
            print(f"  Truncated to {video_length} frames to match original video")

    # Extract ROI from each frame
    roi_images = images[:, roi_top:roi_top+roi_h, roi_left:roi_left+roi_w, :]
    print(f"  Extracted ROI: {roi_w}x{roi_h} from position ({roi_left}, {roi_top})")

    return roi_images, video_length, full_height, full_width, fps, original_video_length


def create_roi_mask(video_length, roi_w, roi_h, mask_left_in_roi, mask_top_in_roi, mask_w, mask_h):
    """
    Create a mask tensor for the ROI video.

    The mask is black (0) everywhere except the selected area which is white (1).
    """
    # Create single frame mask
    mask_frame = np.zeros((roi_h, roi_w), dtype=np.float32)

    # Set the selected area to white (1.0)
    mask_right = mask_left_in_roi + mask_w
    mask_bottom = mask_top_in_roi + mask_h

    # Clamp to ROI bounds
    mask_left_clamped = max(0, mask_left_in_roi)
    mask_top_clamped = max(0, mask_top_in_roi)
    mask_right_clamped = min(roi_w, mask_right)
    mask_bottom_clamped = min(roi_h, mask_bottom)

    mask_frame[mask_top_clamped:mask_bottom_clamped, mask_left_clamped:mask_right_clamped] = 1.0

    # Expand to match video length
    masks = np.tile(mask_frame, (video_length, 1, 1))
    masks = torch.from_numpy(masks).unsqueeze(-1)  # (F, H, W, 1)

    print(f"  Created mask: white area at ({mask_left_in_roi}, {mask_top_in_roi}) size {mask_w}x{mask_h}")

    return masks


def inference_roi(pixel_values, masks, video_length, height, width, fps, video_out_path, iterations=1):
    """
    Run inference on the ROI video.

    Args:
        iterations: Mask dilation iterations.
                    - 0 = No dilation, mask stays exactly as drawn
                    - 1 = Minimal dilation (1 pixel expansion)
                    - Higher values = More expansion around masked area
    """
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
    print(f"  ROI video saved to: {video_out_path}")


def composite_roi_to_original(
    original_video_path: str,
    roi_video_path: str,
    output_video_path: str,
    roi_left: int, roi_top: int, roi_w: int, roi_h: int
):
    """
    Composite the processed ROI video back onto the original video.

    Reads the ROI video and pastes each frame onto the corresponding
    frame of the original video at the correct position.
    """
    import subprocess

    # Use FFmpeg to composite
    # Overlay the ROI video onto the original at the specified position
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video_path,
        "-i", roi_video_path,
        "-filter_complex",
        f"[1:v]scale={roi_w}:{roi_h}[scaled];[0:v][scaled]overlay={roi_left}:{roi_top}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        output_video_path
    ]

    print(f"  Compositing with FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr}")
        # Fallback: use opencv to do frame-by-frame composition
        print("  Falling back to OpenCV composition...")

        original = VideoReader(original_video_path)
        roi = VideoReader(roi_video_path)

        original_length = len(original)
        roi_length = len(roi)

        # Get FPS
        fps = original.get_avg_fps()

        # Read all frames
        original_frames = original.get_batch(list(range(original_length))).asnumpy()
        roi_frames = roi.get_batch(list(range(min(roi_length, original_length)))).asnumpy()

        # Composite ROI onto original
        for i in range(min(original_length, roi_length)):
            frame = original_frames[i].copy()
            roi_frame = roi_frames[i]

            # Resize ROI frame if needed
            if roi_frame.shape[1] != roi_w or roi_frame.shape[0] != roi_h:
                roi_frame = cv2.resize(roi_frame, (roi_w, roi_h))

            # Paste ROI onto original
            frame[roi_top:roi_top+roi_h, roi_left:roi_left+roi_w] = roi_frame
            original_frames[i] = frame

        # Write output using imageio or similar
        # For simplicity, use ffmpeg to write frames
        temp_dir = os.path.join(os.path.dirname(output_video_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)

        for i, frame in enumerate(original_frames):
            cv2.imwrite(os.path.join(temp_dir, f"frame_{i:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Use ffmpeg to combine frames
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        subprocess.run(cmd, capture_output=True)

        # Clean up temp frames
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"  Composited video saved to: {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Minimax Remover with ROI-based processing.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--selected_area", type=str, required=True,
                        help="Selected area as 'left,top,width,height' (in original video pixels).")
    parser.add_argument("--padding", type=int, default=80,
                        help="Padding to expand around selected area (default: 80).")

    args = parser.parse_args()

    # Parse selected area
    try:
        selected_left, selected_top, selected_w, selected_h = map(int, args.selected_area.split(','))
    except:
        print("Error: selected_area must be in format 'left,top,width,height'")
        exit(1)

    video_path = args.video_path
    padding = args.padding

    print(f"Processing video: {video_path}")
    print(f"Selected area: {selected_left}, {selected_top}, {selected_w}, {selected_h}")
    print(f"Padding: {padding}")

    # First, get original video dimensions
    vr = VideoReader(video_path)
    video_length_original = len(vr)
    video_height, video_width, _ = vr.get_batch([0]).shape[1:]
    fps = vr.get_avg_fps()

    print(f"Original video: {video_width}x{video_height}, {video_length_original} frames, {fps} fps")

    # Calculate expanded ROI
    roi_left, roi_top, roi_w, roi_h, mask_left_in_roi, mask_top_in_roi, mask_w, mask_h = calculate_expanded_roi(
        selected_left, selected_top, selected_w, selected_h,
        video_width, video_height, padding
    )

    print(f"Expanded ROI: ({roi_left}, {roi_top}) size {roi_w}x{roi_h}")
    print(f"Mask within ROI: ({mask_left_in_roi}, {mask_top_in_roi}) size {mask_w}x{mask_h}")

    # Load only the ROI from the video, matching original frame count
    roi_images, roi_video_length, full_h, full_w, fps, original_roi_length = load_video_roi(
        video_path, roi_left, roi_top, roi_w, roi_h, target_length=video_length_original
    )

    # Create mask for the ROI (use the processed video length)
    roi_masks = create_roi_mask(
        roi_video_length, roi_w, roi_h,
        mask_left_in_roi, mask_top_in_roi, mask_w, mask_h
    )

    # Generate paths
    video_dir, video_filename = os.path.split(video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    roi_video_path = os.path.join(video_dir, f"{video_name}_roi_clean{video_ext}")
    final_output_path = os.path.join(video_dir, f"{video_name}_clean{video_ext}")

    # Run inference on the small ROI video
    print(f"\nRunning inference on ROI ({roi_w}x{roi_h}, {roi_video_length} frames)...")
    inference_roi(roi_images, roi_masks, roi_video_length, roi_h, roi_w, fps, roi_video_path)

    # Composite the processed ROI back onto the original video
    print(f"\nCompositing processed ROI back to original video...")
    composite_roi_to_original(
        video_path, roi_video_path, final_output_path,
        roi_left, roi_top, roi_w, roi_h
    )

    # Clean up the temporary ROI video
    if os.path.exists(roi_video_path):
        os.remove(roi_video_path)
        print(f"Cleaned up temporary ROI video")

    print(f"\nDone! Final output: {final_output_path}")
    print(f"The cleaned video has the same dimensions as the original: {video_width}x{video_height}")
