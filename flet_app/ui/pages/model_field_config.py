"""
Centralized model field configuration for training UI.
Maps model names to their field visibility and default values.
"""

# Centralized model configuration: defines visibility, aliases, and defaults for each model
MODEL_CONFIG = {
    "sdxl": {
        "aliases": ["sdxl"],
        "show_fields": {
            "model_path": True,
            "load_checkpoint": False,
            "diffusers_path": False,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": True,
            "min_t": False,
            "max_t": False,
            "max_seq_len": False,
            "flux_shift": False,
            "bypass_g_emb": False,
            "lumina_shift": False,
            "float8_e5m2": False,
            "longcat_float8": False,
            "max_llama3_seq_len": False,
            "hidream_4bit": False,
            "hidream_tdtype": False,
            "v_pred": True,
            "d_est_loss": True,
            "min_snr_gamma": True,
            "unet_lr": True,
            "te1_lr": True,
            "te2_lr": True,
        },
        "defaults": {
            "model_path": "models/sdxl/sd_xl_base_1.0_0.9vae.safetensors",
            "v_pred": True,
            "d_est_loss": True,
            "min_snr_gamma": "5",
            "unet_lr": "4e-5",
            "te1_lr": "2e-5",
            "te2_lr": "2e-5",
        },
        "timestep_sm": "None",
        "transformer_dtype": "None",
    },
    "ltx-video-2": {
        "aliases": ["ltx-video-2", "ltx2"],
        "show_fields": {
            "model_path": True,
            "load_checkpoint": True,
            "diffusers_path": False,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": True,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": True,
            "min_t": False,
            "max_t": False,
            "max_seq_len": False,
            "flux_shift": False,
            "bypass_g_emb": False,
            "lumina_shift": False,
            "float8_e5m2": False,
            "longcat_float8": False,
            "llama3_path_field": False,
            "max_llama3_seq_len": False,
            "hidream_4bit": False,
            "hidream_tdtype": False,
            "v_pred": False,
            "d_est_loss": False,
            "min_snr_gamma": False,
            "unet_lr": False,
            "te1_lr": False,
            "te2_lr": False,
            "audio_attn": True,
            "audio_ff": True,
            "video_attn": True,
            "video_ff": True,
            "cross_modal_attn": True,
            "all_modules": True,
            "rank": True,
            "alpha": True,
            "dropout": True,
            "first_frame_conditioning_p_ltx2": True,
            "dtype": False,
            "transformer_dtype": False,
            "timestep_sm": False,
            "mixed_precision_mode": True,
            "quantization": True,
            "load_text_encoder_in_8bit": True,
        },
        "defaults": {
            "model_path": "models/ltx2/ltx-2-19b-dev.safetensors",
            "load_checkpoint": "",
            "text_encoder_path": "models/text_encoders/gemma3",
            "all_modules": False,
            "video_attn": True,
            "video_ff": False,
            "audio_attn": False,
            "audio_ff": False,
            "cross_modal_attn": False,
            "rank": "32",
            "alpha": "32",
            "dropout": "0.0",
            "first_frame_conditioning_p_ltx2": "0.5",
            "mixed_precision_mode": "bf16",
            "quantization": "int8-quanto",
            "load_text_encoder_in_8bit": False,
        },
        "timestep_sm": "logit_normal",
        "transformer_dtype": "float8",
    },
    "ltx-video": {
        "aliases": ["ltx-video", "ltx"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": True,
            "with_audio": False,
        },
        "defaults": {
            "diffusers_path": "models/LTX-Video",
            "single_file_path": "models/LTX-Video/ltx-video-2b-v0.9.1.safetensors",
        },
        "timestep_sm": "logit_normal",
    },
    "wan22": {
        "aliases": ["wan22"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": True,
            "ckpt_path_wan22": True,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "min_t": True,
            "max_t": True,
            "max_seq_len": False,
            "flux_shift": False,
            "bypass_g_emb": False,
            "lumina_shift": False,
            "float8_e5m2": False,
            "longcat_float8": False,
            "max_llama3_seq_len": False,
            "hidream_4bit": False,
            "hidream_tdtype": False,
            "v_pred": False,
            "d_est_loss": False,
            "min_snr_gamma": False,
            "unet_lr": False,
            "te1_lr": False,
            "te2_lr": False,
        },
        "defaults": {
            "ckpt_path_wan22": "models/Wan2.2-I2V-A14B",
            "transformer_path": "models/Wan2.2-I2V-A14B/high_noise_model",
            "llm_path": "",
        },
        "timestep_sm": "logit_normal",
        "transformer_dtype": "float8",
    },
    "wan": {
        "aliases": ["wan"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": True,
            "ckpt_path_wan22": True,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
        },
        "defaults": {
            "ckpt_path_wan22": "models/Wan2.1-T2V-1.3B",
            "transformer_path": "models/wan/wan2.1_t2v_1.3B_bf16.safetensors",
            "llm_path": "models/wan/wrapper/umt5-xxl-enc-bf16.safetensors",
        },
        "timestep_sm": "logit_normal",
    },
    "auraflow": {
        "aliases": ["auraflow"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": True,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "max_seq_len": True,
        },
        "defaults": {
            "transformer_path": "models/auraflow/pony-v7-base.safetensors",
            "text_encoder_path": "models/auraflow/umt5_auraflow.fp16.safetensors",
            "vae_path": "models/auraflow/sdxl_vae.safetensors",
        },
        "timestep_sm": "logit_normal",
    },
    "chroma": {
        "aliases": ["chroma"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": False,
            "transformer_path_full": True,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "flux_shift": True,
        },
        "defaults": {
            "diffusers_path": "models/chroma/FLUX.1-dev",
            "transformer_path_full": "models/chroma/Chroma1-HD.safetensors",
            "flux_shift": True,
        },
    },
    "flux": {
        "aliases": ["flux"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "flux_shift": True,
            "bypass_g_emb": True,
        },
        "defaults": {
            "diffusers_path": "models/FLUX.1-dev",
            "transformer_path": "models/flux-dev-single-files/flux1-kontext-dev.safetensors",
            "flux_shift": True,
        },
    },
    "flux2": {
        "aliases": ["flux2"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "flux2_diffusion_model": True,
        },
        "defaults": {
            "flux2_diffusion_model": "models/FLUX.2-dev/flux2-dev.safetensors",
            "flux2_vae": "models/vae/flux2-vae.safetensors",
            "flux2_text_encoders": "models/text_encoders/mistral_3_small_flux2_fp8.safetensors",
            "flux2_shift": "3",
        },
        "timestep_sm": "logit_normal",
    },
    "flux2_klein_4b": {
        "aliases": ["flux2_klein_4b"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "flux2_diffusion_model": True,
        },
        "defaults": {
            "flux2_diffusion_model": "models/FLUX.2-klein-4B/flux-2-klein-base-4b.safetensors",
            "flux2_vae": "models/vae/flux2-vae.safetensors",
            "flux2_text_encoders": "models/text_encoders/qwen_3_4b.safetensors",
            "flux2_shift": "3",
        },
        "timestep_sm": "logit_normal",
    },
    "flux2_klein_9b": {
        "aliases": ["flux2_klein_9b"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "flux2_diffusion_model": True,
        },
        "defaults": {
            "flux2_diffusion_model": "models/FLUX.2-klein-9B/flux-2-klein-base-9b.safetensors",
            "flux2_vae": "models/vae/flux2-vae.safetensors",
            "flux2_text_encoders": "models/text_encoders/qwen_3_8b.safetensors",
            "flux2_shift": "3",
        },
        "timestep_sm": "logit_normal",
    },
    "sd3": {
        "aliases": ["sd3"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "flux_shift": True,
        },
        "defaults": {
            "diffusers_path": "models/stable-diffusion-3.5-medium",
            "flux_shift": True,
        },
    },
    "lumina": {
        "aliases": ["lumina", "lumina_2"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": True,
            "llm_path": True,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
        },
        "defaults": {
            "transformer_path": "models/lumina2/lumina_2_model_bf16.safetensors",
            "llm_path": "models/lumina2/gemma_2_2b_fp16.safetensors",
            "vae_path": "models/lumina2/flux_vae.safetensors",
            "lumina_shift": True,
        },
    },
    "z_image": {
        "aliases": ["z_image"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "z_image_diffusion_model": True,
        },
        "defaults": {
            "z_image_diffusion_model": "models/z_image_turbo/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            "z_image_vae": "models/z_image_turbo/split_files/vae/ae.safetensors",
            "z_image_text_encoders": "models/z_image_turbo/split_files/text_encoders/qwen_3_4b.safetensors",
            "z_image_merge_adapters": "models/z_image_turbo/zimage_turbo_training_adapter_v2.safetensors",
            "z_image_diffusion_model_dtype_fp8": False,
        },
    },
    "qwen_image": {
        "aliases": ["qwen_image"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
        },
        "defaults": {
            "diffusers_path": "models/Qwen-Image",
        },
        "timestep_sm": "logit_normal",
    },
    "qwen_image_plus": {
        "aliases": ["qwen_image_plus"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
        },
        "defaults": {
            "diffusers_path": "models/Qwen-Image-Edit-2509",
        },
        "timestep_sm": "logit_normal",
    },
    "cosmos": {
        "aliases": ["cosmos"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": True,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
        },
        "defaults": {
            "transformer_path": "models/cosmos/cosmos-1.0-diffusion-7b-text2world.pt",
            "text_encoder_path": "models/cosmos/oldt5_xxl_fp16.safetensors",
            "vae_path": "models/cosmos/cosmos_cv8x8x8_1.0.safetensors",
        },
    },
    "cosmos_predict2": {
        "aliases": ["cosmos_predict2"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": True,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": True,
            "single_file_path": False,
            "with_audio": False,
            "float8_e5m2": True,
        },
        "defaults": {
            "transformer_path": "models/Cosmos-Predict2-2B-Text2Image.pt",
            "t5_path": "models/oldt5_xxl_fp16.safetensors",
            "vae_path": "models/wan_2.1_vae.safetensors",
        },
    },
    "hunyuan-video": {
        "aliases": ["hunyuan-video"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": True,
            "llm_path": True,
            "ckpt_path_wan22": True,
            "clip_path": True,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
        },
        "defaults": {
            "ckpt_path_wan22": "models/HunyuanVideo/ckpts",
            "transformer_path": "models/hunyuan/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors",
            "vae_path": "models/hunyuan/hunyuan_video_vae_bf16.safetensors",
        },
        "timestep_sm": "logit_normal",
    },
    "hunyuan_image": {
        "aliases": ["hunyuan_image"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": True,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": True,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
        },
        "defaults": {
            "transformer_path": "models/hunyuan/hunyuanimage2.1.safetensors",
            "vae_path": "models/hunyuan/hunyuan_image_2.1_vae_fp16.safetensors",
            "text_encoder_path": "models/qwen_2.5_vl_7b.safetensors",
        },
    },
    "hidream": {
        "aliases": ["hidream"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": True,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "max_llama3_seq_len": True,
            "hidream_4bit": True,
            "hidream_tdtype": True,
            "flux_shift": True,
        },
        "defaults": {
            "diffusers_path": "models/HiDream-I1-Full",
            "llama3_path": "models/Meta-Llama-3.1-8B-Instruct",
            "max_llama3_seq_len": "128",
        },
    },
    "longcat": {
        "aliases": ["longcat"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": False,
            "transformer_path": True,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": False,
            "llm_path": False,
            "ckpt_path_wan22": True,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "longcat_float8": True,
        },
        "defaults": {
            "transformer_path": "models/wan/LongCat_TI2V_comfy_bf16.safetensors",
        },
        "timestep_sm": "logit_normal",
    },
    "omnigen2": {
        "aliases": ["omnigen2"],
        "show_fields": {
            "checkpoint_path": False,
            "diffusers_path": True,
            "transformer_path": False,
            "transformer_path_full": False,
            "text_encoder_path": False,
            "vae_path": True,
            "llm_path": False,
            "ckpt_path_wan22": False,
            "clip_path": False,
            "llama3_path": False,
            "byt5_path": False,
            "t5_path": False,
            "single_file_path": False,
            "with_audio": False,
            "flux_shift": True,
        },
        "defaults": {
            "diffusers_path": "models/OmniGen2",
            "flux_shift": True,
        },
    },
}


def normalize_model_name(model_name):
    """Normalize a model name to lowercase for comparison."""
    return str(model_name).strip().lower() if model_name else ""


def get_model_key(normalized_name):
    """Get the config key for a normalized model name by checking aliases."""
    for key, config in MODEL_CONFIG.items():
        if normalized_name in config.get("aliases", []):
            return key
    return None


def get_model_config(model_name):
    """Get the full config dict for a model name."""
    normalized = normalize_model_name(model_name)
    key = get_model_key(normalized)
    return MODEL_CONFIG.get(key) if key else None


def get_field_visibility(model_name):
    """Get the show_fields dict for a model."""
    config = get_model_config(model_name)
    return config.get("show_fields", {}) if config else {}


def get_field_defaults(model_name):
    """Get the defaults dict for a model."""
    config = get_model_config(model_name)
    return config.get("defaults", {}) if config else {}


# Default field visibility values (used to ensure all models have complete field definitions)
DEFAULT_FIELD_VISIBILITY = {
    # Path fields
    "model_path": False,
    "load_checkpoint": False,
    "diffusers_path": False,
    "transformer_path": False,
    "transformer_path_full": False,
    "text_encoder_path": False,
    "vae_path": False,
    "llm_path": False,
    "ckpt_path_wan22": False,
    "clip_path": False,
    "llama3_path": False,
    "byt5_path": False,
    "t5_path": False,
    "single_file_path": False,
    "with_audio": False,
    # Special fields
    "min_t": False,
    "max_t": False,
    "max_seq_len": False,
    "flux_shift": False,
    "bypass_g_emb": False,
    "lumina_shift": False,
    "float8_e5m2": False,
    "longcat_float8": False,
    "max_llama3_seq_len": False,
    "hidream_4bit": False,
    "hidream_tdtype": False,
    # SDXL-specific  fields
    "v_pred": False,
    "d_est_loss": False,
    "min_snr_gamma": False,
    "unet_lr": False,
    "te1_lr": False,
    "te2_lr": False,
    # LTX2-specific module checkboxes
    "audio_attn": False,
    "audio_ff": False,
    "video_attn": False,
    "video_ff": False,
    "cross_modal_attn": False,
    "all_modules": False,
    # LTX2-specific adapter fields
    "rank": False,
    "alpha": False,
    "dropout": False,
    "first_frame_conditioning_p_ltx2": False,
    # LTX2-specific precision fields
    "mixed_precision_mode": False,
    "quantization": False,
    "load_text_encoder_in_8bit": False,
    # dtype and sampling fields (visible by default, hidden for LTX2)
    "dtype": True,
    "transformer_dtype": True,
    "timestep_sm": True,
    # Model-specific fields
    "flux2_diffusion_model": False,
    "z_image_diffusion_model": False,
}

def get_complete_field_visibility(model_name):
    """Get field visibility dict with defaults for any missing fields."""
    config = get_model_config(model_name)
    if not config:
        return DEFAULT_FIELD_VISIBILITY.copy()

    result = DEFAULT_FIELD_VISIBILITY.copy()
    result.update(config.get("show_fields", {}))
    return result


def get_timestep_sm_default(model_name):
    """Get the default timestep_sm value for a model."""
    config = get_model_config(model_name)
    if config:
        return config.get("timestep_sm", None)
    return None


def get_transformer_dtype_default(model_name):
    """Get the default transformer_dtype value for a model."""
    config = get_model_config(model_name)
    if config:
        return config.get("transformer_dtype", None)
    return None


# List of all field ref names to reset on model switch
RESETTABLE_FIELDS = [
    "diffusers_path",
    "transformer_path",
    "transformer_path_full",
    "llm_path",
    "text_encoder_path",
    "vae_path",
    "ckpt_path_wan22",
    "clip_path",
    "llama3_path",
    "max_llama3_seq_len",
    "byt5_path",
    "single_file_path",
    "t5_path",
    "model_path",
    "load_checkpoint",
    # Flux2-specific fields
    "flux2_diffusion_model",
    "flux2_vae",
    "flux2_text_encoders",
    "flux2_shift",
    # Z_image-specific fields
    "z_image_diffusion_model",
    "z_image_vae",
    "z_image_text_encoders",
    "z_image_merge_adapters",
    # LTX2-specific adapter fields
    "rank",
    "alpha",
    "dropout",
    "first_frame_conditioning_p_ltx2",
]

# Fields with special reset values
SPECIAL_RESET_VALUES = {
    "hidream_4bit": True,
    "hidream_tdtype": False,
}

def get_field_visibility(model_name, field_name):
    """Get visibility for a specific field for a given model.

    Args:
        model_name: The model name (can be normalized or raw)
        field_name: The field name to check visibility for

    Returns:
        Boolean indicating if the field should be visible
    """
    config = get_model_config(model_name)
    if not config:
        return DEFAULT_FIELD_VISIBILITY.get(field_name, False)

    show_fields = config.get("show_fields", {})
    return show_fields.get(field_name, DEFAULT_FIELD_VISIBILITY.get(field_name, False))


def reset_all_model_fields(field_refs_dict):
    """Reset all model-dependent field values to empty/default.

    Args:
        field_refs_dict: Dict mapping field names to their Ref objects
    """
    try:
        # Reset regular fields to empty string
        for field_name in RESETTABLE_FIELDS:
            ref = field_refs_dict.get(field_name)
            if ref and ref.current:
                ref.current.value = ""

        # Reset special fields to their special values
        for field_name, value in SPECIAL_RESET_VALUES.items():
            ref = field_refs_dict.get(field_name)
            if ref and ref.current:
                ref.current.value = value
    except Exception:
        pass
