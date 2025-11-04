import flet as ft
from contextlib import contextmanager
# import yaml # Removed as hardcoded config data is reduced
from .._styles import create_textfield, create_dropdown, add_section_title # Import helper functions
from .training_dataset_block import get_training_dataset_page_content
from flet_app.ui.utils.utils_top_menu import TopBarUtils
from flet_app.settings import settings

# Global references to access from outside the function
model_type_dropdown_ref = ft.Ref[ft.Dropdown]()
min_t_field_ref = ft.Ref[ft.TextField]()
max_t_field_ref = ft.Ref[ft.TextField]()
max_seq_len_field_ref = ft.Ref[ft.TextField]()
flux_shift_checkbox_ref = ft.Ref[ft.Checkbox]()
bypass_g_emb_checkbox_ref = ft.Ref[ft.Checkbox]()
ffc_p_field_ref = ft.Ref[ft.TextField]()
lumina_shift_checkbox_ref = ft.Ref[ft.Checkbox]()
v_pred_checkbox_ref = ft.Ref[ft.Checkbox]()
d_est_loss_checkbox_ref = ft.Ref[ft.Checkbox]()
min_snr_gamma_field_ref = ft.Ref[ft.TextField]()
unet_lr_field_ref = ft.Ref[ft.TextField]()
te1_lr_field_ref = ft.Ref[ft.TextField]()
te2_lr_field_ref = ft.Ref[ft.TextField]()
checkpoint_path_field_ref = ft.Ref[ft.TextField]()
diffusers_path_field_ref = ft.Ref[ft.TextField]()
transformer_path_field_ref = ft.Ref[ft.TextField]()
transformer_path_full_ref = ft.Ref[ft.TextField]()
text_encoder_path_field_ref = ft.Ref[ft.TextField]()
vae_path_field_ref = ft.Ref[ft.TextField]()
ckpt_path_wan22_field_ref = ft.Ref[ft.TextField]()
llm_path_field_ref = ft.Ref[ft.TextField]()
float8_e5m2_checkbox_ref = ft.Ref[ft.Checkbox]()
clip_path_field_ref = ft.Ref[ft.TextField]()
llama3_path_field_ref = ft.Ref[ft.TextField]()
max_llama3_seq_len_field_ref = ft.Ref[ft.TextField]()
hidream_4bit_checkbox_ref = ft.Ref[ft.Checkbox]()
hidream_tdtype_checkbox_ref = ft.Ref[ft.Checkbox]()
byt5_path_field_ref = ft.Ref[ft.TextField]()
single_file_path_field_ref = ft.Ref[ft.TextField]()
t5_path_field_ref = ft.Ref[ft.TextField]()
checkpoint_row_ref = ft.Ref[ft.ResponsiveRow]()
ckpt_path_row_ref = ft.Ref[ft.ResponsiveRow]()
diffusers_row_ref = ft.Ref[ft.ResponsiveRow]()
single_file_row_ref = ft.Ref[ft.ResponsiveRow]()
transformer_full_row_ref = ft.Ref[ft.ResponsiveRow]()
byt5_row_ref = ft.Ref[ft.ResponsiveRow]()
t5_row_ref = ft.Ref[ft.ResponsiveRow]()
llama3_row_ref = ft.Ref[ft.ResponsiveRow]()
clip_row_ref = ft.Ref[ft.ResponsiveRow]()
timestep_sm_dropdown_ref = ft.Ref[ft.Dropdown]()
transformer_dtype_dropdown_ref = ft.Ref[ft.Dropdown]()

_suppress_model_defaults = False

@contextmanager
def suppress_model_defaults():
    """Disable auto-default population temporarily (e.g., when loading from existing configs)."""
    global _suppress_model_defaults
    prev = _suppress_model_defaults
    _suppress_model_defaults = True
    try:
        yield
    finally:
        _suppress_model_defaults = prev

def get_training_config_page_content():
    """Generates Flet controls with hardcoded configuration values, grouped by section."""

    def on_model_type_change(e):
        """Handle model type dropdown change to show/hide model-specific fields"""
        sel = model_type_dropdown_ref.current.value if model_type_dropdown_ref.current else None
        is_wan22 = (str(sel).strip().lower() == "wan22")
        is_wan = (str(sel).strip().lower() == "wan")
        is_auraflow = (str(sel).strip().lower() == "auraflow")
        is_chroma = (str(sel).strip().lower() == "chroma")
        is_flux = (str(sel).strip().lower() == "flux")
        is_sd3 = (str(sel).strip().lower() == "sd3")
        is_ltx = (str(sel).strip().lower() in ("ltx-video", "ltx"))
        is_lumina = (str(sel).strip().lower() in ("lumina", "lumina_2"))
        is_sdxl = (str(sel).strip().lower() == "sdxl")
        sel_norm = str(sel).strip().lower() if sel is not None else ""
        is_qwen_plus = (sel_norm == "qwen_image_plus")
        is_qwen_image = (sel_norm == "qwen_image")
        is_qwen_any = is_qwen_plus or is_qwen_image
        is_cosmos = (sel_norm == "cosmos")
        is_cosmos_p2 = (sel_norm == "cosmos_predict2")
        is_omnigen2 = (sel_norm == "omnigen2")
        is_hunyuan_video = (sel_norm == "hunyuan-video")
        is_hunyuan_image = (sel_norm == "hunyuan_image")
        is_hidream = (sel_norm == "hidream")
        skip_defaults = _suppress_model_defaults

        # Reset a core list of model-dependent path fields on any model switch.
        # Presets for specific models (below) will then repopulate as needed.
        if not skip_defaults:
            try:
                if diffusers_path_field_ref.current:
                    diffusers_path_field_ref.current.value = ""
                if transformer_path_field_ref.current:
                    transformer_path_field_ref.current.value = ""
                if 'transformer_path_full_ref' in globals() or 'transformer_path_full_ref' in locals():
                    if transformer_path_full_ref.current:
                        transformer_path_full_ref.current.value = ""
                if llm_path_field_ref.current:
                    llm_path_field_ref.current.value = ""
                if text_encoder_path_field_ref.current:
                    text_encoder_path_field_ref.current.value = ""
                if vae_path_field_ref.current:
                    vae_path_field_ref.current.value = ""
                if ckpt_path_wan22_field_ref.current:
                    ckpt_path_wan22_field_ref.current.value = ""
                if clip_path_field_ref.current:
                    clip_path_field_ref.current.value = ""
                if llama3_path_field_ref.current:
                    llama3_path_field_ref.current.value = ""
                if max_llama3_seq_len_field_ref.current:
                    max_llama3_seq_len_field_ref.current.value = ""
                if hidream_4bit_checkbox_ref.current:
                    hidream_4bit_checkbox_ref.current.value = True
                if hidream_tdtype_checkbox_ref.current:
                    hidream_tdtype_checkbox_ref.current.value = False
                if byt5_path_field_ref.current:
                    byt5_path_field_ref.current.value = ""
                if single_file_path_field_ref.current:
                    single_file_path_field_ref.current.value = ""
                if t5_path_field_ref.current:
                    t5_path_field_ref.current.value = ""
            except Exception:
                # Be resilient to any missing/hidden controls
                pass
        if min_t_field_ref.current:
            min_t_field_ref.current.visible = is_wan22
        if max_t_field_ref.current:
            max_t_field_ref.current.visible = is_wan22
        if max_seq_len_field_ref.current:
            max_seq_len_field_ref.current.visible = is_auraflow
        if flux_shift_checkbox_ref.current:
            flux_shift_checkbox_ref.current.visible = (is_chroma or is_flux or is_sd3 or is_omnigen2 or is_hidream)

        # Set default timestep_sm per model selection
        if not skip_defaults:
            try:
                if timestep_sm_dropdown_ref.current:
                    models_logit = {"ltx-video", "ltx", "hunyuan-video", "wan", "qwen_image", "qwen_image_plus", "auraflow"}
                    if sel_norm in models_logit:
                        timestep_sm_dropdown_ref.current.value = "logit_normal"
                    else:
                        timestep_sm_dropdown_ref.current.value = "None"
                    if timestep_sm_dropdown_ref.current.page:
                        timestep_sm_dropdown_ref.current.update()
            except Exception:
                pass

        # Set default transformer_dtype per model selection
        if not skip_defaults:
            try:
                if transformer_dtype_dropdown_ref.current:
                    models_none = {"sdxl", "lumina_2", "omnigen2"}
                    if sel_norm in models_none:
                        transformer_dtype_dropdown_ref.current.value = "None"
                    else:
                        transformer_dtype_dropdown_ref.current.value = "float8"
                    if transformer_dtype_dropdown_ref.current.page:
                        transformer_dtype_dropdown_ref.current.update()
            except Exception:
                pass
        if bypass_g_emb_checkbox_ref.current:
            bypass_g_emb_checkbox_ref.current.visible = is_flux
        if ffc_p_field_ref.current:
            ffc_p_field_ref.current.visible = is_ltx
        if single_file_path_field_ref.current:
            single_file_path_field_ref.current.visible = is_ltx
        if lumina_shift_checkbox_ref.current:
            lumina_shift_checkbox_ref.current.visible = is_lumina
        if float8_e5m2_checkbox_ref.current:
            float8_e5m2_checkbox_ref.current.visible = is_cosmos_p2
        if llama3_path_field_ref.current:
            llama3_path_field_ref.current.visible = is_hidream
        if max_llama3_seq_len_field_ref.current:
            max_llama3_seq_len_field_ref.current.visible = is_hidream
        if hidream_4bit_checkbox_ref.current:
            hidream_4bit_checkbox_ref.current.visible = is_hidream
        if hidream_tdtype_checkbox_ref.current:
            hidream_tdtype_checkbox_ref.current.visible = is_hidream
        if byt5_path_field_ref.current:
            byt5_path_field_ref.current.visible = (sel_norm == 'hunyuan_image')
        if t5_path_field_ref.current:
            t5_path_field_ref.current.visible = is_cosmos_p2
        # SDXL-specific controls
        if v_pred_checkbox_ref.current:
            v_pred_checkbox_ref.current.visible = is_sdxl
        if d_est_loss_checkbox_ref.current:
            d_est_loss_checkbox_ref.current.visible = is_sdxl
        if min_snr_gamma_field_ref.current:
            min_snr_gamma_field_ref.current.visible = is_sdxl
        if unet_lr_field_ref.current:
            unet_lr_field_ref.current.visible = is_sdxl
        if te1_lr_field_ref.current:
            te1_lr_field_ref.current.visible = is_sdxl
        if te2_lr_field_ref.current:
            te2_lr_field_ref.current.visible = is_sdxl
        if checkpoint_path_field_ref.current:
            checkpoint_path_field_ref.current.visible = is_sdxl
            if is_sdxl and not skip_defaults:
                try:
                    if not checkpoint_path_field_ref.current.value or checkpoint_path_field_ref.current.value.strip() == '':
                        checkpoint_path_field_ref.current.value = 'models/sdxl/sd_xl_base_1.0_0.9vae.safetensors'
                except Exception:
                    pass
        # Keep rows collapsed by updating their containers as well
        try:
            if checkpoint_row_ref.current:
                checkpoint_row_ref.current.visible = is_sdxl
            if ckpt_path_row_ref.current:
                ckpt_path_row_ref.current.visible = (is_wan22 or is_hunyuan_video or is_wan)
            if diffusers_row_ref.current:
                diffusers_row_ref.current.visible = (
                    (not is_sdxl)
                    and (not is_wan22)
                    and (not is_wan)
                    and (not is_auraflow)
                    and (not is_cosmos)
                    and (not is_cosmos_p2)
                    and (not is_hunyuan_video)
                    and (not is_lumina)
                )
            if single_file_row_ref.current:
                single_file_row_ref.current.visible = is_ltx
            if transformer_full_row_ref.current:
                transformer_full_row_ref.current.visible = is_chroma
            if byt5_row_ref.current:
                byt5_row_ref.current.visible = (sel_norm == 'hunyuan_image')
            if t5_row_ref.current:
                t5_row_ref.current.visible = is_cosmos_p2
            if llama3_row_ref.current:
                llama3_row_ref.current.visible = is_hidream
            if clip_row_ref.current:
                clip_row_ref.current.visible = is_hunyuan_video
        except Exception:
            pass
        # Ensure all model-specific toggles are hidden for Cosmos
        if sel_norm == "cosmos":
            try:
                if flux_shift_checkbox_ref.current:
                    flux_shift_checkbox_ref.current.visible = False
                if bypass_g_emb_checkbox_ref.current:
                    bypass_g_emb_checkbox_ref.current.visible = False
                if ffc_p_field_ref.current:
                    ffc_p_field_ref.current.visible = False
                if lumina_shift_checkbox_ref.current:
                    lumina_shift_checkbox_ref.current.visible = False
                if v_pred_checkbox_ref.current:
                    v_pred_checkbox_ref.current.visible = False
                if d_est_loss_checkbox_ref.current:
                    d_est_loss_checkbox_ref.current.visible = False
                if min_snr_gamma_field_ref.current:
                    min_snr_gamma_field_ref.current.visible = False
                if unet_lr_field_ref.current:
                    unet_lr_field_ref.current.visible = False
                if te1_lr_field_ref.current:
                    te1_lr_field_ref.current.visible = False
                if te2_lr_field_ref.current:
                    te2_lr_field_ref.current.visible = False
                if checkpoint_path_field_ref.current:
                    checkpoint_path_field_ref.current.visible = False
                if max_seq_len_field_ref.current:
                    max_seq_len_field_ref.current.visible = False
            except Exception:
                pass
        # wan22-specific ckpt_path
        if ckpt_path_wan22_field_ref.current:
            ckpt_path_wan22_field_ref.current.visible = is_wan22 or is_hunyuan_video or is_wan
        if clip_path_field_ref.current:
            clip_path_field_ref.current.visible = is_hunyuan_video
        # Hide base path fields for SDXL
        if diffusers_path_field_ref.current:
            diffusers_path_field_ref.current.visible = (
                (not is_sdxl)
                and (not is_wan22)
                and (not is_wan)
                and (not is_auraflow)
                and (not is_cosmos)
                and (not is_cosmos_p2)
                and (not is_hunyuan_video)
                and (not is_hunyuan_image)
                and (not is_lumina)
            )
        if transformer_path_field_ref.current:
            # Show standard transformer path for most models (Qwen included), hide for Chroma/SDXL
            transformer_path_field_ref.current.visible = (not is_sdxl) and (not is_chroma) and (not is_omnigen2) and (not is_hidream) and (not is_ltx) and (not is_sd3)
        if 'transformer_path_full_ref' in globals() or 'transformer_path_full_ref' in locals():
            try:
                if transformer_path_full_ref.current:
                    # Only Chroma uses the full transformer single-file path UI
                    transformer_path_full_ref.current.visible = is_chroma
            except Exception:
                pass
        if llm_path_field_ref.current:
            llm_path_field_ref.current.visible = (
                (not is_sdxl)
                and (not is_auraflow)
                and (not is_chroma)
                and (not is_qwen_any)
                and (not is_cosmos)
                and (not is_cosmos_p2)
                and (not is_omnigen2)
                and (not is_flux)
                and (not is_hidream)
                and (not is_hunyuan_image)
                and (not is_ltx)
                and (not is_sd3)
            )
        if text_encoder_path_field_ref.current:
            text_encoder_path_field_ref.current.visible = (
                (not is_sdxl)
                and (not is_wan22)
                and (not is_wan)
                and (not is_chroma)
                and (not is_qwen_plus)
                and (not is_omnigen2)
                and (not is_flux)
                and (not is_hunyuan_video)
                and (not is_hidream)
                and (not is_lumina)
                and (not is_ltx)
                and (not is_cosmos_p2)
                and (not is_sd3)
            )
        if vae_path_field_ref.current:
            vae_path_field_ref.current.visible = (
                (not is_sdxl)
                and (not is_wan22)
                and (not is_wan)
                and (not is_chroma)
                and (not is_qwen_plus)
                and (not is_omnigen2)
                and (not is_flux)
                and (not is_hidream)
                and (not is_ltx)
                and (not is_sd3)
            )
        if is_auraflow and not skip_defaults:
            try:
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/auraflow/pony-v7-base.safetensors'
                if text_encoder_path_field_ref.current:
                    if not text_encoder_path_field_ref.current.value or text_encoder_path_field_ref.current.value.strip() == '':
                        text_encoder_path_field_ref.current.value = 'models/auraflow/umt5_auraflow.fp16.safetensors'
                if vae_path_field_ref.current:
                    if not vae_path_field_ref.current.value or vae_path_field_ref.current.value.strip() == '':
                        vae_path_field_ref.current.value = 'models/auraflow/sdxl_vae.safetensors'
            except Exception:
                pass

        # Set wan22 defaults when selected
        if is_wan22 and not skip_defaults:
            try:
                if ckpt_path_wan22_field_ref.current:
                    if not ckpt_path_wan22_field_ref.current.value or ckpt_path_wan22_field_ref.current.value.strip() == '':
                        ckpt_path_wan22_field_ref.current.value = 'models/Wan2.2-T2V-A14B'
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/wan2.2_t2v_low_noise_14B_fp16.safetensors'
                if llm_path_field_ref.current:
                    if not llm_path_field_ref.current.value or llm_path_field_ref.current.value.strip() == '':
                        llm_path_field_ref.current.value = 'models/umt5_xxl_fp16.safetensors'
            except Exception:
                pass
        # Set wan (2.1) defaults when selected
        if is_wan and not skip_defaults:
            try:
                if ckpt_path_wan22_field_ref.current:
                    if not ckpt_path_wan22_field_ref.current.value or ckpt_path_wan22_field_ref.current.value.strip() == '':
                        ckpt_path_wan22_field_ref.current.value = 'models/Wan2.1-T2V-1.3B'
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/wan/wan2.1_t2v_1.3B_bf16.safetensors'
                if llm_path_field_ref.current:
                    if not llm_path_field_ref.current.value or llm_path_field_ref.current.value.strip() == '':
                        llm_path_field_ref.current.value = 'models/wan/wrapper/umt5-xxl-enc-bf16.safetensors'
            except Exception:
                pass
        # Set chroma defaults when selected
        if is_chroma and not skip_defaults:
            try:
                if diffusers_path_field_ref.current:
                    if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                        diffusers_path_field_ref.current.value = 'models/chroma/FLUX.1-dev'
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/chroma/Chroma1-HD.safetensors'
                if 'transformer_path_full_ref' in globals() or 'transformer_path_full_ref' in locals():
                    if transformer_path_full_ref.current:
                        if not transformer_path_full_ref.current.value or transformer_path_full_ref.current.value.strip() == '':
                            if not transformer_path_full_ref.current.value or transformer_path_full_ref.current.value.strip() == '':
                                transformer_path_full_ref.current.value = 'models/chroma/Chroma1-HD.safetensors'
            except Exception:
                pass
        # Set LTX-Video defaults when selected
        if is_ltx and not skip_defaults:
            try:
                if diffusers_path_field_ref.current:
                    if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                        diffusers_path_field_ref.current.value = 'models/LTX-Video'
                if single_file_path_field_ref.current:
                    if not single_file_path_field_ref.current.value or single_file_path_field_ref.current.value.strip() == '':
                        single_file_path_field_ref.current.value = 'models/LTX-Video/ltx-video-2b-v0.9.1.safetensors'
            except Exception:
                pass
        # Set SD3 defaults when selected
        if is_sd3 and not skip_defaults:
            try:
                if diffusers_path_field_ref.current:
                    if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                        diffusers_path_field_ref.current.value = 'models/stable-diffusion-3.5-medium'
                if flux_shift_checkbox_ref.current:
                    flux_shift_checkbox_ref.current.value = True
            except Exception:
                pass
        # Set lumina_2 defaults when selected
        if sel_norm == 'lumina_2' and not skip_defaults:
            try:
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/lumina2/lumina_2_model_bf16.safetensors'
                if llm_path_field_ref.current:
                    if not llm_path_field_ref.current.value or llm_path_field_ref.current.value.strip() == '':
                        llm_path_field_ref.current.value = 'models/lumina2/gemma_2_2b_fp16.safetensors'
                if vae_path_field_ref.current:
                    if not vae_path_field_ref.current.value or vae_path_field_ref.current.value.strip() == '':
                        vae_path_field_ref.current.value = 'models/lumina2/flux_vae.safetensors'
                if lumina_shift_checkbox_ref.current:
                    lumina_shift_checkbox_ref.current.value = True
            except Exception:
                pass
        # Set Qwen Image defaults when selected
        # qwen_image and qwen_image_plus share many fields but have different defaults for
        # diffusers_path and transformer_path per request.
        if is_qwen_any and not skip_defaults:
            try:
                if is_qwen_plus:
                    if diffusers_path_field_ref.current:
                        if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                            diffusers_path_field_ref.current.value = 'models/Qwen-Image-Edit-2509'
                    if transformer_path_field_ref.current:
                        if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                            transformer_path_field_ref.current.value = ''
                else:
                    if diffusers_path_field_ref.current:
                        if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                            diffusers_path_field_ref.current.value = 'models/Qwen-Image'
                    if transformer_path_field_ref.current:
                        if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                            transformer_path_field_ref.current.value = 'models/qwen/qwen_image_bf16.safetensors'
                if 'transformer_path_full_ref' in globals() or 'transformer_path_full_ref' in locals():
                    if transformer_path_full_ref.current:
                        if not transformer_path_full_ref.current.value or transformer_path_full_ref.current.value.strip() == '':
                            transformer_path_full_ref.current.value = ''
                if text_encoder_path_field_ref.current:
                    if not text_encoder_path_field_ref.current.value or text_encoder_path_field_ref.current.value.strip() == '':
                        text_encoder_path_field_ref.current.value = 'models/qwen/qwen_2.5_vl_7b.safetensors'
                if vae_path_field_ref.current:
                    if not vae_path_field_ref.current.value or vae_path_field_ref.current.value.strip() == '':
                        vae_path_field_ref.current.value = 'models/qwen/diffusion_pytorch_model.safetensors'
            except Exception:
                pass
        # Set Hunyuan-Video defaults when selected
        if is_hunyuan_video and not skip_defaults:
            try:
                if ckpt_path_wan22_field_ref.current:
                    if not ckpt_path_wan22_field_ref.current.value or ckpt_path_wan22_field_ref.current.value.strip() == '':
                        ckpt_path_wan22_field_ref.current.value = 'models/HunyuanVideo/ckpts'
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/hunyuan/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors'
                if vae_path_field_ref.current:
                    if not vae_path_field_ref.current.value or vae_path_field_ref.current.value.strip() == '':
                        vae_path_field_ref.current.value = 'models/hunyuan/hunyuan_video_vae_bf16.safetensors'
                if llm_path_field_ref.current:
                    if not llm_path_field_ref.current.value or llm_path_field_ref.current.value.strip() == '':
                        llm_path_field_ref.current.value = 'models/hunyuan/llava-llama-3-8b-text-encoder-tokenizer'
                if clip_path_field_ref.current:
                    if not clip_path_field_ref.current.value or clip_path_field_ref.current.value.strip() == '':
                        clip_path_field_ref.current.value = 'models/hunyuan/clip-vit-large-patch14'
            except Exception:
                pass
        # Set Hunyuan Image defaults when selected
        if is_hunyuan_image and not skip_defaults:
            try:
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/hunyuan/hunyuanimage2.1.safetensors'
                if vae_path_field_ref.current:
                    if not vae_path_field_ref.current.value or vae_path_field_ref.current.value.strip() == '':
                        vae_path_field_ref.current.value = 'models/hunyuan/hunyuan_image_2.1_vae_fp16.safetensors'
                if text_encoder_path_field_ref.current:
                    if not text_encoder_path_field_ref.current.value or text_encoder_path_field_ref.current.value.strip() == '':
                        text_encoder_path_field_ref.current.value = 'models/qwen_2.5_vl_7b.safetensors'
                if byt5_path_field_ref.current:
                    if not byt5_path_field_ref.current.value or byt5_path_field_ref.current.value.strip() == '':
                        byt5_path_field_ref.current.value = 'models/byt5_small_glyphxl_fp16.safetensors'
            except Exception:
                pass
        # Set Flux defaults when selected
        if is_flux and not skip_defaults:
            try:
                if diffusers_path_field_ref.current:
                    if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                        diffusers_path_field_ref.current.value = 'models/FLUX.1-dev'
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/flux-dev-single-files/flux1-kontext-dev.safetensors'
                if flux_shift_checkbox_ref.current:
                    flux_shift_checkbox_ref.current.value = True
            except Exception:
                pass
        # Set HiDream defaults when selected
        if is_hidream and not skip_defaults:
            try:
                if diffusers_path_field_ref.current:
                    if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                        diffusers_path_field_ref.current.value = 'models/HiDream-I1-Full'
                if llama3_path_field_ref.current:
                    if not llama3_path_field_ref.current.value or llama3_path_field_ref.current.value.strip() == '':
                        llama3_path_field_ref.current.value = 'models/Meta-Llama-3.1-8B-Instruct'
                if max_llama3_seq_len_field_ref.current:
                    if not max_llama3_seq_len_field_ref.current.value or max_llama3_seq_len_field_ref.current.value.strip() == '':
                        max_llama3_seq_len_field_ref.current.value = '128'
                if flux_shift_checkbox_ref.current:
                    flux_shift_checkbox_ref.current.value = True
                if hidream_4bit_checkbox_ref.current:
                    hidream_4bit_checkbox_ref.current.value = True
                if hidream_tdtype_checkbox_ref.current:
                    hidream_tdtype_checkbox_ref.current.value = False
            except Exception:
                pass
        # Set OmniGen2 defaults when selected
        if is_omnigen2 and not skip_defaults:
            try:
                if diffusers_path_field_ref.current:
                    if not diffusers_path_field_ref.current.value or diffusers_path_field_ref.current.value.strip() == '':
                        diffusers_path_field_ref.current.value = 'models/OmniGen2'
                if flux_shift_checkbox_ref.current:
                    flux_shift_checkbox_ref.current.value = True
            except Exception:
                pass
        # Set Cosmos Predict2 defaults when selected
        if is_cosmos_p2 and not skip_defaults:
            try:
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/Cosmos-Predict2-2B-Text2Image.pt'
                if t5_path_field_ref.current:
                    if not t5_path_field_ref.current.value or t5_path_field_ref.current.value.strip() == '':
                        t5_path_field_ref.current.value = 'models/oldt5_xxl_fp16.safetensors'
                if vae_path_field_ref.current:
                    if not vae_path_field_ref.current.value or vae_path_field_ref.current.value.strip() == '':
                        vae_path_field_ref.current.value = 'models/wan_2.1_vae.safetensors'
                if float8_e5m2_checkbox_ref.current:
                    float8_e5m2_checkbox_ref.current.value = False
            except Exception:
                pass
        # Set Cosmos defaults when selected
        if is_cosmos and not skip_defaults:
            try:
                if transformer_path_field_ref.current:
                    if not transformer_path_field_ref.current.value or transformer_path_field_ref.current.value.strip() == '':
                        transformer_path_field_ref.current.value = 'models/cosmos/cosmos-1.0-diffusion-7b-text2world.pt'
                if text_encoder_path_field_ref.current:
                    if not text_encoder_path_field_ref.current.value or text_encoder_path_field_ref.current.value.strip() == '':
                        text_encoder_path_field_ref.current.value = 'models/cosmos/oldt5_xxl_fp16.safetensors'
                if vae_path_field_ref.current:
                    if not vae_path_field_ref.current.value or vae_path_field_ref.current.value.strip() == '':
                        vae_path_field_ref.current.value = 'models/cosmos/cosmos_cv8x8x8_1.0.safetensors'
            except Exception:
                pass
        if e and getattr(e, 'page', None):
            e.page.update()

    page_controls = []

    # --- Model Configuration & Dataset Selection (Side by Side) ---
    # Add Save Configuration button to the dataset row (styled like Monitor)
    save_cfg_btn = ft.ElevatedButton(
        "Save Data Config",
        icon=ft.Icons.SAVE,
        on_click=lambda e: TopBarUtils.handle_save_as(e.page),
    )
    dataset_block = get_training_dataset_page_content(extra_right_controls=[save_cfg_btn])
    page_controls.append(
        ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Model Configuration"),
                ft.Container(
                    content=ft.Column([
                    ft.ResponsiveRow(controls=[
                        create_dropdown(
                            "Model Type",
                            settings.train_def_model,
                            settings.dpipe_model_dict,
                            hint_text="Select model or specify path below", col=4, expand=True,
                            fill_color=ft.Colors.with_opacity(0.18, ft.Colors.AMBER_900),
                            on_change=on_model_type_change, ref=model_type_dropdown_ref
                        ),
                        create_dropdown(
                            "dtype",
                            "bfloat16",
                            {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"},
                            col=3, expand=True, scale=0.8
                        ),
                        create_dropdown(
                            "transformer_dtype",
                            "float8",
                            {"float8": "float8", "None": "None"},
                            col=3, expand=True, scale=0.8, ref=transformer_dtype_dropdown_ref
                        ),
                        create_dropdown(
                            "timestep_sm",
                            "logit_normal",
                            {"logit_normal": "logit_normal", "uniform": "uniform", "None": "None"},
                            col=2, expand=True, scale=0.8, ref=timestep_sm_dropdown_ref
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "checkpoint_path",
                                "models/sdxl/sd_xl_base_1.0_0.9vae.safetensors",
                                col=12, expand=True, ref=checkpoint_path_field_ref,
                                visible=(settings.train_def_model == "sdxl")
                            ),
                        ],
                        ref=checkpoint_row_ref,
                        visible=(settings.train_def_model == "sdxl")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "ckpt_path",
                                "models/Wan2.2-T2V-A14B",
                                col=12, expand=True, ref=ckpt_path_wan22_field_ref,
                                visible=(
                                    settings.train_def_model == "wan22"
                                    or settings.train_def_model == "wan"
                                    or settings.train_def_model == "hunyuan-video"
                                )
                            ),
                        ],
                        ref=ckpt_path_row_ref,
                        visible=(
                            settings.train_def_model == "wan22"
                            or settings.train_def_model == "wan"
                            or settings.train_def_model == "hunyuan-video"
                        )
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "diffusers_path", "models/Qwen-Image", col=12, expand=True,
                                ref=diffusers_path_field_ref, visible=(
                                    settings.train_def_model != "sdxl"
                                    and settings.train_def_model != "wan22"
                                    and settings.train_def_model != "wan"
                                    and settings.train_def_model != "auraflow"
                                    and settings.train_def_model != "cosmos"
                                    and settings.train_def_model != "cosmos_predict2"
                                    and settings.train_def_model != "hunyuan-video"
                                    and settings.train_def_model != "lumina_2"
                                )
                            ),
                        ], spacing=2,
                        ref=diffusers_row_ref,
                        visible=(
                            settings.train_def_model != "sdxl"
                            and settings.train_def_model != "wan22"
                            and settings.train_def_model != "wan"
                            and settings.train_def_model != "auraflow"
                            and settings.train_def_model != "cosmos"
                            and settings.train_def_model != "cosmos_predict2"
                            and settings.train_def_model != "hunyuan-video"
                            and settings.train_def_model != "lumina_2"
                        )
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "single_file_path", "", col=12, expand=True,
                                ref=single_file_path_field_ref, visible=(settings.train_def_model == "ltx-video")
                            ),
                        ], spacing=2,
                        ref=single_file_row_ref,
                        visible=(settings.train_def_model == "ltx-video")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "transformer_path", "", col=12, expand=True,
                                ref=transformer_path_full_ref, visible=(settings.train_def_model == "chroma")
                            ),
                        ], spacing=2,
                        ref=transformer_full_row_ref,
                        visible=(settings.train_def_model == "chroma")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "byt5_path", "", col=12, expand=True,
                                ref=byt5_path_field_ref, visible=(settings.train_def_model == "hunyuan_image")
                            ),
                        ], spacing=2,
                        ref=byt5_row_ref,
                        visible=(settings.train_def_model == "hunyuan_image")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "t5_path", "", col=12, expand=True,
                                ref=t5_path_field_ref, visible=(settings.train_def_model == "cosmos_predict2")
                            ),
                        ], spacing=2,
                        ref=t5_row_ref,
                        visible=(settings.train_def_model == "cosmos_predict2")
                    ),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "llama3_path", "", col=12, expand=True,
                            ref=llama3_path_field_ref, visible=(settings.train_def_model == "hidream")
                        ),
                    ], spacing=2, ref=llama3_row_ref, visible=(settings.train_def_model == "hidream")),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "clip_path", "", col=12, expand=True,
                            ref=clip_path_field_ref, visible=(settings.train_def_model == "hunyuan-video")
                        ),
                    ], spacing=2, ref=clip_row_ref, visible=(settings.train_def_model == "hunyuan-video")),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "transformer_path", "", col=6, expand=True,
                            ref=transformer_path_field_ref, visible=(settings.train_def_model != "sdxl" and settings.train_def_model != "chroma" and settings.train_def_model != "omnigen2")
                        ),
                        ft.Checkbox(
                            label="float8_e5m2",
                            value=False,
                            scale=0.8,
                            ref=float8_e5m2_checkbox_ref,
                            visible=(settings.train_def_model == "cosmos_predict2"),
                            col=6,
                        ),
                        create_textfield(
                            "llm_path", "", col=6, expand=True,
                            ref=llm_path_field_ref, visible=(
                                settings.train_def_model != "sdxl"
                                and settings.train_def_model != "auraflow"
                                and settings.train_def_model != "chroma"
                                and settings.train_def_model != "qwen_image"
                                and settings.train_def_model != "qwen_image_plus"
                                and settings.train_def_model != "cosmos"
                                and settings.train_def_model != "cosmos_predict2"
                                and settings.train_def_model != "omnigen2"
                                and settings.train_def_model != "flux"
                                and settings.train_def_model != "hidream"
                            )
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        create_textfield("text_encoder_path", "", col=6, expand=True, ref=text_encoder_path_field_ref, visible=(
                            settings.train_def_model != "sdxl"
                            and settings.train_def_model != "wan22"
                            and settings.train_def_model != "wan"
                            and settings.train_def_model != "chroma"
                            and settings.train_def_model != "omnigen2"
                            and settings.train_def_model != "flux"
                            and settings.train_def_model != "hunyuan-video"
                            and settings.train_def_model != "hidream"
                            and settings.train_def_model != "lumina_2"
                            and settings.train_def_model != "cosmos_predict2"
                        )),
                        create_textfield("vae_path", "", col=6, expand=True, ref=vae_path_field_ref, visible=(
                            settings.train_def_model != "sdxl"
                            and settings.train_def_model != "wan22"
                            and settings.train_def_model != "wan"
                            and settings.train_def_model != "chroma"
                            and settings.train_def_model != "omnigen2"
                            and settings.train_def_model != "flux"
                        )),
                ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "max_llama3_sequence_length", 128, col=6, expand=True,
                            ref=max_llama3_seq_len_field_ref, visible=(settings.train_def_model == "hidream")
                        ),
                        ft.Checkbox(
                            label="llama3_4bit",
                            value=True,
                            scale=0.8,
                            ref=hidream_4bit_checkbox_ref,
                            visible=(settings.train_def_model == "hidream"),
                            col=3,
                        ),
                        ft.Checkbox(
                            label="t_dtype_nf4",
                            value=False,
                            scale=0.8,
                            ref=hidream_tdtype_checkbox_ref,
                            visible=(settings.train_def_model == "hidream"),
                            col=3,
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        create_dropdown(
                            "adapter",
                            "lora",
                            {
                                "lora": "lora"
                            }, col=3, expand=False, scale=0.8,
                        ),
                        ft.Column([
                            ft.ResponsiveRow(controls=[
                                create_textfield("min_t", 0, expand=True, col=6, scale=0.8, ref=min_t_field_ref, visible=(settings.train_def_model == "wan22")),
                                create_textfield("max_t", 0.875, expand=True, col=6, scale=0.8, ref=max_t_field_ref, visible=(settings.train_def_model == "wan22")),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                create_textfield(
                                    "max_sequence_length", 768, expand=True, col=12, scale=0.8,
                                    ref=max_seq_len_field_ref, visible=(settings.train_def_model == "auraflow")
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                create_textfield(
                                    "first_frame_conditioning_p", 1.0, expand=True, col=12, scale=0.8,
                                    ref=ffc_p_field_ref, visible=(settings.train_def_model == "ltx-video")
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                ft.Checkbox(
                                    label="flux_shift",
                                    value=True,
                                    scale=0.8,
                                    ref=flux_shift_checkbox_ref,
                                    visible=(settings.train_def_model == "chroma" or settings.train_def_model == "flux" or settings.train_def_model == "sd3" or settings.train_def_model == "omnigen2"),
                                    col=6,
                                ),
                                ft.Checkbox(
                                    label="bypass_g_emb",
                                    value=True,
                                    scale=0.8,
                                    ref=bypass_g_emb_checkbox_ref,
                                    visible=(settings.train_def_model == "flux"),
                                    col=6,
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                ft.Checkbox(
                                    label="lumina_shift",
                                    value=True,
                                    scale=0.8,
                                    ref=lumina_shift_checkbox_ref,
                                    visible=(settings.train_def_model == "lumina" or settings.train_def_model == "lumina_2"),
                                    col=12,
                                ),
                            ], spacing=2),
                            # SDXL rows
                            ft.ResponsiveRow(controls=[
                                ft.Checkbox(
                                    label="v_pred",
                                    value=True,
                                    scale=0.8,
                                    ref=v_pred_checkbox_ref,
                                    visible=(settings.train_def_model == "sdxl"),
                                    col=6,
                                ),
                                ft.Checkbox(
                                    label="d_est_loss",
                                    value=True,
                                    scale=0.8,
                                    ref=d_est_loss_checkbox_ref,
                                    visible=(settings.train_def_model == "sdxl"),
                                    col=6,
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                create_textfield(
                                    "min_snr_gamma", 5, expand=True, col=6, scale=0.8,
                                    ref=min_snr_gamma_field_ref, visible=(settings.train_def_model == "sdxl")
                                ),
                                create_textfield(
                                    "unet_lr", 4e-5, expand=True, col=6, scale=0.8,
                                    ref=unet_lr_field_ref, visible=(settings.train_def_model == "sdxl")
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                create_textfield(
                                    "text_encoder_1_lr", 2e-5, expand=True, col=6, scale=0.8,
                                    ref=te1_lr_field_ref, visible=(settings.train_def_model == "sdxl")
                                ),
                                create_textfield(
                                    "text_encoder_2_lr", 2e-5, expand=True, col=6, scale=0.8,
                                    ref=te2_lr_field_ref, visible=(settings.train_def_model == "sdxl")
                                ),
                            ], spacing=2),
                        ], col=9, spacing=2, alignment=ft.MainAxisAlignment.START),
                    ], spacing=2, vertical_alignment=ft.CrossAxisAlignment.START),
                    # Adapter details row
                    ft.ResponsiveRow(controls=[
                        create_textfield("a_rank", 32, col=3, expand=True),
                        create_textfield("a_dtype", "bfloat16", col=3, expand=True),
                        create_textfield("blocks_swap", 10, col=3, expand=True),
                        create_textfield("disable_bsfe", "true", col=3, expand=True),
                    ], spacing=2),
                ]),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                    bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
                )
            ], col=6), # Model Configuration column set to 6
            ft.Column([
                *add_section_title("Dataset Selection"),
                ft.Container(
                    content=dataset_block,
                    padding=ft.padding.only(left=0, right=10, top=10, bottom=10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6), # Dataset Selection column set to 6
        ], spacing=20, vertical_alignment=ft.CrossAxisAlignment.START)
    )
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))

    # --- Training & Misc Settings (Two Columns) ---
    page_controls.append(
        ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Training settings"),
                # Two sub-columns within Training settings
                ft.Container(
                    content=ft.ResponsiveRow(controls=[
                        ft.Column([
                            create_textfield("epochs", 1000, expand=True),
                            create_textfield("pipeline_stages", 1, expand=True),
                            create_textfield("gradient_clipping", 1.0, expand=True),
                        ], col=6, spacing=6),
                        ft.Column([
                            create_textfield("warmup_steps", 0, expand=True),
                            create_textfield("micro_batch_size_per_gpu", 1, expand=True, fill_color="#232A2C"),
                            create_textfield("gradient_accumulation_steps", 1, expand=True, fill_color="#232A2C"),
                            ft.ResponsiveRow(controls=[
                                create_dropdown(
                                    "lr_scheduler",
                                    "constant",
                                    {"constant": "constant", "linear": "linear"},
                                    expand=True, col=6, scale=0.8
                                ),
                                create_dropdown(
                                    "activation_checkpointing",
                                    "unsloth",
                                    {"unsloth": "unsloth", "false": "off"},
                                    expand=True, col=6, scale=0.8
                                ),
                            ], spacing=2),
                            # (intentionally blank; toggles are in left column)
                        ], col=6, spacing=6),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),

            ft.Column([
                *add_section_title("Misc settings"),
                ft.Container(
                    content=ft.Column([
                        ft.ResponsiveRow(controls=[
                            create_textfield("save_every_n_epochs", 5, col=4, expand=True),
                            create_textfield("checkpoint_every_n_minutes", 10, col=4, expand=True),
                            create_textfield("caching_batch_size", 1, col=4, expand=True),
                        ], spacing=4),
                        ft.ResponsiveRow(controls=[
                            create_dropdown(
                                "save_dtype",
                                "bfloat16",
                                {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"},
                                col=4, expand=True, scale=0.8
                            ),
                            create_dropdown(
                                "partition_method",
                                "parameters",
                                {"parameters": "parameters", "uniform": "uniform", "manual": "manual"},
                                col=4, expand=True, scale=0.8
                            ),
                            create_dropdown(
                                "video_clip_mode",
                                "single_beginning",
                                {"single_beginning": "single_beginning", "single_middle": "single_middle", "multiple_overlapping": "multiple_overlapping"},
                                col=4, expand=True, scale=0.8
                            ),
                        ], spacing=4),
                        ft.ResponsiveRow(controls=[
                            create_textfield("steps_per_print", 1, col=4, expand=True),
                        ], spacing=4),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START)
    )

    # --- Eval & Optimizer Settings (Two Columns) ---
    page_controls.append(
        ft.ResponsiveRow([
            # Eval settings column
            ft.Column([
                *add_section_title("Eval settings"),
                ft.Container(
                    content=ft.ResponsiveRow(controls=[
                        ft.Column([
                            create_textfield("eval_every_n_epochs", 1, expand=True),
                            ft.Checkbox(label="eval_before_first_step", value=True, scale=0.8),
                            
                        ], col=6, spacing=6),
                        ft.Column([
                            create_textfield("eval_micro_batch_size_per_gpu", 1, expand=True, fill_color="#232A2C"),
                            create_textfield("eval_gradient_accumulation_steps", 1, expand=True, fill_color="#232A2C"),
                        ], col=6, spacing=6),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),

            # Optimizer settings column
            ft.Column([
                *add_section_title("Optimizer"),
                ft.Container(
                    content=ft.Column([
                        ft.ResponsiveRow(controls=[
                            create_textfield("optimizer_type", "adamw_optimi", col=12, expand=True),
                        ], spacing=4),
                        ft.ResponsiveRow(controls=[
                            ft.Column([
                                create_textfield("lr", 2e-5, expand=True),
                                create_textfield("betas", "[0.9, 0.99]", expand=True),
                            ], col=6, spacing=6),
                            ft.Column([
                                create_textfield("weight_decay", 0.01, expand=True),
                                create_textfield("eps", 1e-8, expand=True),
                            ], col=6, spacing=6),
                        ], spacing=6),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
                # Monitoring (WandB) section under Optimizer
                *add_section_title("Monitoring"),
                ft.Container(
                    content=ft.Column([
                        ft.ResponsiveRow(controls=[
                            ft.Checkbox(label="enable_wandb", value=False, scale=0.85),
                        ], spacing=4),
                        ft.ResponsiveRow(controls=[
                            create_textfield("wandb_api_key", "", col=12, expand=True),
                        ], spacing=4),
                        ft.ResponsiveRow(controls=[
                            create_textfield("wandb_tracker_name", "", col=12, expand=True),
                        ], spacing=4),
                        ft.ResponsiveRow(controls=[
                            create_textfield("wandb_run_name", "", col=12, expand=True),
                        ], spacing=4),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START)
    )

    container = ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8, # Slightly reduced spacing between controls/sections
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True, # Allow container to take full height
        padding=ft.padding.all(5)
    )
    container.dataset_block = dataset_block
    return container

def update_wan_fields_visibility(is_wan22: bool, min_t_value=None, max_t_value=None):
    """Update visibility and values of min_t and max_t fields from external calls"""
    try:
        if min_t_field_ref.current:
            min_t_field_ref.current.visible = is_wan22
            if min_t_value is not None:
                min_t_field_ref.current.value = str(min_t_value)
        if max_t_field_ref.current:
            max_t_field_ref.current.visible = is_wan22
            if max_t_value is not None:
                max_t_field_ref.current.value = str(max_t_value)

        # Update the page if we can access it
        if min_t_field_ref.current and min_t_field_ref.current.page:
            min_t_field_ref.current.page.update()
    except Exception:
        pass

def update_auraflow_fields_visibility(is_auraflow: bool, max_sequence_length_value=None):
    """Update visibility and value of max_sequence_length for auraflow from external calls"""
    try:
        if max_seq_len_field_ref.current:
            max_seq_len_field_ref.current.visible = is_auraflow
            if max_sequence_length_value is not None:
                max_seq_len_field_ref.current.value = str(max_sequence_length_value)
        if max_seq_len_field_ref.current and max_seq_len_field_ref.current.page:
            max_seq_len_field_ref.current.page.update()
    except Exception:
        pass

def update_chroma_fields_visibility(is_chroma: bool, flux_shift_value=None):
    """Update visibility and value of flux_shift for chroma from external calls"""
    try:
        if flux_shift_checkbox_ref.current:
            flux_shift_checkbox_ref.current.visible = is_chroma
            if flux_shift_value is not None:
                # Accept bool-y strings and bools
                sval = str(flux_shift_value).strip().lower()
                flux_shift_checkbox_ref.current.value = (flux_shift_value is True) or (sval in ['1','true','yes','on'])
        if flux_shift_checkbox_ref.current and flux_shift_checkbox_ref.current.page:
            flux_shift_checkbox_ref.current.page.update()
    except Exception:
        pass

def update_flux_fields_visibility(is_flux: bool, flux_shift_value=None, bypass_g_emb_value=None):
    """Update visibility and values of flux-specific fields (flux_shift, bypass_g_emb)."""
    try:
        if flux_shift_checkbox_ref.current:
            flux_shift_checkbox_ref.current.visible = is_flux or flux_shift_checkbox_ref.current.visible
            if flux_shift_value is not None:
                sval = str(flux_shift_value).strip().lower()
                flux_shift_checkbox_ref.current.value = (flux_shift_value is True) or (sval in ['1','true','yes','on'])
        if bypass_g_emb_checkbox_ref.current:
            bypass_g_emb_checkbox_ref.current.visible = is_flux
            if bypass_g_emb_value is not None:
                sval2 = str(bypass_g_emb_value).strip().lower()
                bypass_g_emb_checkbox_ref.current.value = (bypass_g_emb_value is True) or (sval2 in ['1','true','yes','on'])
        if (
            (flux_shift_checkbox_ref.current and flux_shift_checkbox_ref.current.page)
            or (bypass_g_emb_checkbox_ref.current and bypass_g_emb_checkbox_ref.current.page)
        ):
            # Update page once
            page_obj = None
            if flux_shift_checkbox_ref.current and flux_shift_checkbox_ref.current.page:
                page_obj = flux_shift_checkbox_ref.current.page
            if not page_obj and bypass_g_emb_checkbox_ref.current and bypass_g_emb_checkbox_ref.current.page:
                page_obj = bypass_g_emb_checkbox_ref.current.page
            if page_obj:
                page_obj.update()
    except Exception:
        pass

def update_ltx_fields_visibility(is_ltx: bool, ffc_p_value=None):
    """Update visibility and value for ltx-video specific fields."""
    try:
        if ffc_p_field_ref.current:
            ffc_p_field_ref.current.visible = is_ltx
            if ffc_p_value is not None:
                ffc_p_field_ref.current.value = str(ffc_p_value)
        if ffc_p_field_ref.current and ffc_p_field_ref.current.page:
            ffc_p_field_ref.current.page.update()
    except Exception:
        pass

def update_lumina_fields_visibility(is_lumina: bool, lumina_shift_value=None):
    """Update visibility and value of lumina_shift for lumina/lumina_2 from external calls"""
    try:
        if lumina_shift_checkbox_ref.current:
            lumina_shift_checkbox_ref.current.visible = is_lumina
            if lumina_shift_value is not None:
                sval = str(lumina_shift_value).strip().lower()
                lumina_shift_checkbox_ref.current.value = (lumina_shift_value is True) or (sval in ['1','true','yes','on'])
        if lumina_shift_checkbox_ref.current and lumina_shift_checkbox_ref.current.page:
            lumina_shift_checkbox_ref.current.page.update()
    except Exception:
        pass

def update_sdxl_fields_visibility(
    is_sdxl: bool,
    v_pred_value=None,
    d_est_loss_value=None,
    min_snr_gamma_value=None,
    unet_lr_value=None,
    te1_lr_value=None,
    te2_lr_value=None,
    checkpoint_path_value=None,
):
    """Update visibility and values for SDXL-specific fields."""
    try:
        if v_pred_checkbox_ref.current:
            v_pred_checkbox_ref.current.visible = is_sdxl
            if v_pred_value is not None:
                sval = str(v_pred_value).strip().lower()
                v_pred_checkbox_ref.current.value = (v_pred_value is True) or (sval in ['1','true','yes','on'])
        if d_est_loss_checkbox_ref.current:
            d_est_loss_checkbox_ref.current.visible = is_sdxl
            if d_est_loss_value is not None:
                sval = str(d_est_loss_value).strip().lower()
                d_est_loss_checkbox_ref.current.value = (d_est_loss_value is True) or (sval in ['1','true','yes','on'])
        if min_snr_gamma_field_ref.current:
            min_snr_gamma_field_ref.current.visible = is_sdxl
            if min_snr_gamma_value is not None:
                min_snr_gamma_field_ref.current.value = str(min_snr_gamma_value)
        if unet_lr_field_ref.current:
            unet_lr_field_ref.current.visible = is_sdxl
            if unet_lr_value is not None:
                unet_lr_field_ref.current.value = str(unet_lr_value)
        if te1_lr_field_ref.current:
            te1_lr_field_ref.current.visible = is_sdxl
            if te1_lr_value is not None:
                te1_lr_field_ref.current.value = str(te1_lr_value)
        if te2_lr_field_ref.current:
            te2_lr_field_ref.current.visible = is_sdxl
            if te2_lr_value is not None:
                te2_lr_field_ref.current.value = str(te2_lr_value)
        if checkpoint_path_field_ref.current:
            checkpoint_path_field_ref.current.visible = is_sdxl
            if checkpoint_path_value is not None:
                checkpoint_path_field_ref.current.value = str(checkpoint_path_value)
        # Update page once if possible
        page_obj = None
        for r in [v_pred_checkbox_ref, d_est_loss_checkbox_ref, min_snr_gamma_field_ref, unet_lr_field_ref, te1_lr_field_ref, te2_lr_field_ref, checkpoint_path_field_ref]:
            if r.current and r.current.page:
                page_obj = r.current.page
                break
        if page_obj:
            page_obj.update()
    except Exception:
        pass

def update_wan22_ckpt_visibility(is_wan22: bool, ckpt_value=None):
    """Update visibility and value for wan22 ckpt_path field."""
    try:
        if ckpt_path_wan22_field_ref.current:
            ckpt_path_wan22_field_ref.current.visible = is_wan22
            if ckpt_value is not None:
                ckpt_path_wan22_field_ref.current.value = str(ckpt_value)
        if ckpt_path_wan22_field_ref.current and ckpt_path_wan22_field_ref.current.page:
            ckpt_path_wan22_field_ref.current.page.update()
    except Exception:
        pass
