import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

try:
    import av  # type: ignore
except Exception:
    av = None  # type: ignore

try:
    from transformers import (
        LlavaNextVideoProcessor,
        LlavaNextVideoForConditionalGeneration,
    )  # type: ignore
except Exception as _ex:  # pragma: no cover
    LlavaNextVideoProcessor = None  # type: ignore
    LlavaNextVideoForConditionalGeneration = None  # type: ignore


def _read_video_pyav(container, indices: List[int]) -> np.ndarray:
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames]) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)


def _sample_video_frames(video_path: str, num_frames: int = 8) -> np.ndarray:
    if av is not None:
        try:
            container = av.open(video_path)
            total_frames = max(1, container.streams.video[0].frames)
            idx = np.linspace(0, total_frames - 1, num=num_frames).astype(int).tolist()
            clip = _read_video_pyav(container, idx)
            container.close()
            return clip
        except Exception:
            pass
    # Fallback to decord
    try:
        from decord import VideoReader  # type: ignore
        vr = VideoReader(video_path)
        total = len(vr)
        idx = np.linspace(0, max(0, total - 1), num=num_frames).astype(int).tolist()
        batch = vr.get_batch(idx).asnumpy()
        return batch[:, :, :, ::-1]  # BGR->RGB
    except Exception:
        pass
    # Last resort: OpenCV
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        frames = []
        for i in np.linspace(0, max(0, total - 1), num=num_frames).astype(int).tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)
    except Exception:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)


class LlavaNextVideoCaptioner:
    def __init__(
        self,
        model_id_or_path: str,
        *,
        device: str = "cuda",
        instruction: str = "Describe the content factually and concisely.",
        use_4bit: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        if LlavaNextVideoProcessor is None or LlavaNextVideoForConditionalGeneration is None:
            raise RuntimeError("transformers >= 4.42.0 with LLaVA-Next support is required")

        # Prepare kwargs based on environment and flags
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs = dict(
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=("flash_attention_2" if use_flash_attention_2 else None),
        )
        if use_4bit:
            kwargs.update(load_in_4bit=True, device_map="auto")
        else:
            kwargs.update(dtype=dtype)

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id_or_path,
            **kwargs,
        )
        if not use_4bit:
            self.model = self.model.to(0 if device.startswith("cuda") else "cpu")

        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id_or_path, local_files_only=True)
        self.device = device
        self.instruction = instruction

    def _build_prompt(self, kind: str) -> str:
        content_type = {"video": "video", "image": "image"}.get(kind, "video")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": content_type},
                ],
            },
        ]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def caption(self, path: Path, fps: int, clean_caption: bool, max_new_tokens: int) -> str:
        p = str(path)
        # Try as image first
        try:
            img = Image.open(p).convert("RGB")
            prompt = self._build_prompt("image")
            inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.model.device, torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            txt = self.processor.decode(output[0], skip_special_tokens=True)
            # Clean any prompt remnants
            if "ASSISTANT:" in txt:
                txt = txt.split("ASSISTANT:", 1)[-1]
            if "USER:" in txt:
                txt = txt.split("USER:")[-1]
            return txt.strip()
        except Exception:
            pass

        # Fallback: treat as video
        clip = _sample_video_frames(p, num_frames=max(1, min(16, (fps or 8))))
        prompt = self._build_prompt("video")
        inputs = self.processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        txt = self.processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT:" in txt:
            txt = txt.split("ASSISTANT:", 1)[-1]
        if "USER:" in txt:
            txt = txt.split("USER:")[-1]
        return txt.strip()
