import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

try:
    from transformers import AutoProcessor  # type: ignore
except Exception as _ex:  # pragma: no cover
    AutoProcessor = None  # type: ignore

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:
    process_vision_info = None  # type: ignore

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


def _ensure_deps():
    if AutoProcessor is None:
        raise RuntimeError("transformers is required for Qwen3-VL captioning")
    if process_vision_info is None:
        raise RuntimeError("qwen_vl_utils>=0.0.14 is required for Qwen3-VL captioning")
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM is required for Qwen3-VL captioning")


class Qwen3VLCaptioner:
    def __init__(
        self,
        model_id_or_path: str,
        *,
        device: str = "cuda",
        instruction: str = "Shortly describe the content of this video in one or two sentences.",
        max_tokens_default: int = 100,
    ) -> None:
        _ensure_deps()
        # vLLM launcher config
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # Force PyTorch SDPA backend to avoid flash-attn import issues on WSL / mismatched builds
        # Valid values include: TORCH_SDPA, FLASH_ATTN, TRITON_ATTN, etc. (vLLM 0.11.x)
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
        # For older vLLM builds that still consult this flag
        os.environ.setdefault("VLLM_USE_FLASH_ATTENTION", "0")
        # Disable torch.compile in constrained environments to save memory
        os.environ.setdefault("VLLM_TORCH_COMPILE", "0")
        self.model_path = model_id_or_path
        # Prefer the fast tokenizer to avoid missing slow-vocab files
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id_or_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception as ex:
            raise FileNotFoundError(
                "Failed to load Qwen3 processor locally. Ensure tokenizer.json and tokenizer_config.json exist in the model folder, "
                f"and that the path is correct: {model_id_or_path}"
            ) from ex
        # Initialize vLLM engine
        tp = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        self.llm = LLM(
            model=model_id_or_path,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            # Reduce KV cache size and batch token budget to fit smaller GPUs
            max_model_len=2048,
            max_num_batched_tokens=512,
            seed=0,
        )
        self.instruction = instruction
        self.max_tokens_default = max_tokens_default

    def _prepare_inputs(self, messages):
        # Apply chat template to produce the prompt with special tokens
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Prepare multi-modal data for vLLM
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=getattr(self.processor, "image_processor", getattr(self.processor, "visual", None)).patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def caption(self, path: Path, fps: int, clean_caption: bool, max_new_tokens: int) -> str:
        # Build messages: prefer video if file is video-like, else image
        p = str(path)
        ext = path.suffix.lower()
        is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}

        content: List[dict] = []
        if is_video:
            content.append({"type": "video", "video": p})
        else:
            content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": self.instruction})

        messages = [{"role": "user", "content": content}]
        inputs = [self._prepare_inputs(messages)]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=int(max_new_tokens or self.max_tokens_default),
            top_k=-1,
            stop_token_ids=[],
        )

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        # vLLM returns a list; each item has .outputs[0].text
        text_out = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        # Clean any prompt remnants if present
        if "ASSISTANT:" in text_out:
            text_out = text_out.split("ASSISTANT:", 1)[-1]
        if "USER:" in text_out:
            text_out = text_out.split("USER:")[-1]
        return text_out.strip()
