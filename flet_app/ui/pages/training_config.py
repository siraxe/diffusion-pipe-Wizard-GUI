import flet as ft
from contextlib import contextmanager
# import yaml # Removed as hardcoded config data is reduced
from .._styles import create_textfield, create_dropdown, add_section_title # Import helper functions
from .training_dataset_block import get_compact_dataset_block
from flet_app.ui.utils.utils_top_menu import TopBarUtils
from flet_app.settings import settings
from . import model_field_config as mfc
from . import optimizer_field_config as ofc
from .training_musubi import get_musubi_training_settings

# Global references to access from outside the function
trainer_dropdown_ref = ft.Ref[ft.Dropdown]()
model_type_dropdown_ref = ft.Ref[ft.Dropdown]()
min_t_field_ref = ft.Ref[ft.TextField]()
max_t_field_ref = ft.Ref[ft.TextField]()
max_seq_len_field_ref = ft.Ref[ft.TextField]()
flux_shift_checkbox_ref = ft.Ref[ft.Checkbox]()
bypass_g_emb_checkbox_ref = ft.Ref[ft.Checkbox]()
lumina_shift_checkbox_ref = ft.Ref[ft.Checkbox]()
v_pred_checkbox_ref = ft.Ref[ft.Checkbox]()
d_est_loss_checkbox_ref = ft.Ref[ft.Checkbox]()
min_snr_gamma_field_ref = ft.Ref[ft.TextField]()
unet_lr_field_ref = ft.Ref[ft.TextField]()
te1_lr_field_ref = ft.Ref[ft.TextField]()
te2_lr_field_ref = ft.Ref[ft.TextField]()
llm_adapter_lr_field_ref = ft.Ref[ft.TextField]()
model_path_field_ref = ft.Ref[ft.TextField]()
rank_field_ref = ft.Ref[ft.TextField]()
alpha_field_ref = ft.Ref[ft.TextField]()
factor_field_ref = ft.Ref[ft.TextField]()
network_dropout_field_ref = ft.Ref[ft.TextField]()
caption_dropout_rate_field_ref = ft.Ref[ft.TextField]()
first_frame_conditioning_p_ltx2_field_ref = ft.Ref[ft.TextField]()
a_rank_field_ref = ft.Ref[ft.TextField]()
a_dtype_field_ref = ft.Ref[ft.TextField]()
blocks_swap_field_ref = ft.Ref[ft.TextField]()
disable_bsfe_field_ref = ft.Ref[ft.TextField]()
diffusers_path_field_ref = ft.Ref[ft.TextField]()
transformer_path_field_ref = ft.Ref[ft.TextField]()
transformer_path_full_ref = ft.Ref[ft.TextField]()
text_encoder_path_field_ref = ft.Ref[ft.TextField]()
vae_path_field_ref = ft.Ref[ft.TextField]()
vae_audio_path_field_ref = ft.Ref[ft.TextField]()
tokenizer_path_field_ref = ft.Ref[ft.TextField]()
ckpt_path_field_ref = ft.Ref[ft.TextField]()
llm_path_field_ref = ft.Ref[ft.TextField]()
float8_e5m2_checkbox_ref = ft.Ref[ft.Checkbox]()
longcat_float8_checkbox_ref = ft.Ref[ft.Checkbox]()
clip_path_field_ref = ft.Ref[ft.TextField]()
llama3_path_field_ref = ft.Ref[ft.TextField]()
max_llama3_seq_len_field_ref = ft.Ref[ft.TextField]()
hidream_4bit_checkbox_ref = ft.Ref[ft.Checkbox]()
hidream_tdtype_checkbox_ref = ft.Ref[ft.Checkbox]()
byt5_path_field_ref = ft.Ref[ft.TextField]()
single_file_path_field_ref = ft.Ref[ft.TextField]()
first_frame_conditioning_p_field_ref = ft.Ref[ft.TextField]()
t5_path_field_ref = ft.Ref[ft.TextField]()
ltx_mode_dropdown_ref = ft.Ref[ft.Dropdown]()
target_fps_field_ref = ft.Ref[ft.TextField]()
ltx_2_3_checkbox_ref = ft.Ref[ft.Checkbox]()
wan_mode_dropdown_ref = ft.Ref[ft.Dropdown]()
wan_task_dropdown_ref = ft.Ref[ft.Dropdown]()
separate_audio_buckets_checkbox_ref = ft.Ref[ft.Checkbox]()
gradient_checkpointing_checkbox_ref = ft.Ref[ft.Checkbox]()
t_type_dropdown_ref = ft.Ref[ft.Dropdown]()
flash_attn_checkbox_ref = ft.Ref[ft.Checkbox]()
reference_downscale_field_ref = ft.Ref[ft.TextField]()
use_mask_checkbox_ref = ft.Ref[ft.Checkbox]()
checkpoint_row_ref = ft.Ref[ft.ResponsiveRow]()
ckpt_path_row_ref = ft.Ref[ft.ResponsiveRow]()
diffusers_row_ref = ft.Ref[ft.ResponsiveRow]()
single_file_row_ref = ft.Ref[ft.ResponsiveRow]()
first_frame_conditioning_p_row_ref = ft.Ref[ft.ResponsiveRow]()
transformer_full_row_ref = ft.Ref[ft.ResponsiveRow]()
byt5_row_ref = ft.Ref[ft.ResponsiveRow]()
t5_row_ref = ft.Ref[ft.ResponsiveRow]()
llama3_row_ref = ft.Ref[ft.ResponsiveRow]()
clip_row_ref = ft.Ref[ft.ResponsiveRow]()
text_encoder_row_ref = ft.Ref[ft.ResponsiveRow]()
dtype_dropdown_ref = ft.Ref[ft.Dropdown]()
timestep_sm_dropdown_ref = ft.Ref[ft.Dropdown]()
transformer_dtype_dropdown_ref = ft.Ref[ft.Dropdown]()
network_network_dropout_field_ref = ft.Ref[ft.TextField]()
# Musubi-specific precision fields
mixed_precision_mode_dropdown_ref = ft.Ref[ft.Dropdown]()
fp8_base_checkbox_ref = ft.Ref[ft.Checkbox]()
fp8_scaled_checkbox_ref = ft.Ref[ft.Checkbox]()
load_text_encoder_in_8bit_checkbox_ref = ft.Ref[ft.Checkbox]()
nf4_te_checkbox_ref = ft.Ref[ft.Checkbox]()
attn_chunking_checkbox_ref = ft.Ref[ft.Checkbox]()
# Preservation & regularization fields
blank_preservation_checkbox_ref = ft.Ref[ft.Checkbox]()
blank_preservation_args_field_ref = ft.Ref[ft.TextField]()
dop_checkbox_ref = ft.Ref[ft.Checkbox]()
dop_args_field_ref = ft.Ref[ft.TextField]()
prior_divergence_checkbox_ref = ft.Ref[ft.Checkbox]()
prior_divergence_args_field_ref = ft.Ref[ft.TextField]()
# CREPA fields
crepa_checkbox_ref = ft.Ref[ft.Checkbox]()
crepa_mode_dropdown_ref = ft.Ref[ft.Dropdown]()
crepa_args_field_ref = ft.Ref[ft.TextField]()
# self_flow and cts_lambda fields
self_flow_checkbox_ref = ft.Ref[ft.Checkbox]()
self_flow_args_field_ref = ft.Ref[ft.TextField]()
cts_lambda_checkbox_ref = ft.Ref[ft.Checkbox]()
cts_lambda_args_field_ref = ft.Ref[ft.TextField]()
forward_xm_checkbox_ref = ft.Ref[ft.Checkbox]()
forward_xm_args_field_ref = ft.Ref[ft.TextField]()
extra_flags_field_ref = ft.Ref[ft.TextField]()
audio_lr_rate_ref = ft.Ref[ft.TextField]()
sample_slider_range_field_ref = ft.Ref[ft.TextField]()
i2v_type_dropdown_ref = ft.Ref[ft.Dropdown]()
sample_each_field_ref = ft.Ref[ft.TextField]()
adapter_dropdown_ref = ft.Ref[ft.Dropdown]()
# z_image specific fields
z_image_diffusion_model_field_ref = ft.Ref[ft.TextField]()
z_image_vae_field_ref = ft.Ref[ft.TextField]()
z_image_text_encoders_field_ref = ft.Ref[ft.TextField]()
z_image_merge_adapters_field_ref = ft.Ref[ft.TextField]()
z_image_diffusion_model_dtype_checkbox_ref = ft.Ref[ft.Checkbox]()
z_image_row_ref = ft.Ref[ft.ResponsiveRow]()
# flux2 specific fields
flux2_diffusion_model_field_ref = ft.Ref[ft.TextField]()
flux2_vae_field_ref = ft.Ref[ft.TextField]()
flux2_text_encoders_field_ref = ft.Ref[ft.TextField]()
flux2_shift_field_ref = ft.Ref[ft.TextField]()
flux2_row_ref = ft.Ref[ft.ResponsiveRow]()

# Optimizer fields
optimizer_type_dropdown_ref = ft.Ref[ft.Dropdown]()
lr_field_ref = ft.Ref[ft.TextField]()
betas_field_ref = ft.Ref[ft.TextField]()
weight_decay_field_ref = ft.Ref[ft.TextField]()
eps_field_ref = ft.Ref[ft.TextField]()
beta3_field_ref = ft.Ref[ft.TextField]()
d0_field_ref = ft.Ref[ft.TextField]()
d_coef_field_ref = ft.Ref[ft.TextField]()
schedulefree_c_field_ref = ft.Ref[ft.TextField]()
prodigy_row_ref = ft.Ref[ft.ResponsiveRow]()
# Automagic optimizer fields
min_lr_field_ref = ft.Ref[ft.TextField]()
max_lr_field_ref = ft.Ref[ft.TextField]()
lr_bump_field_ref = ft.Ref[ft.TextField]()
clip_threshold_field_ref = ft.Ref[ft.TextField]()
do_paramiter_swapping_field_ref = ft.Ref[ft.Checkbox]()
paramiter_swapping_factor_field_ref = ft.Ref[ft.TextField]()
automagic_row_ref = ft.Ref[ft.ResponsiveRow]()

# Section visibility refs for conditional UI
standard_training_section_ref = ft.Ref[ft.ResponsiveRow]()
standard_eval_optimizer_section_ref = ft.Ref[ft.ResponsiveRow]()
musubi_custom_section_ref = ft.Ref[ft.Container]()

_suppress_model_defaults = False

def _convert_bool_value(value):
    """Convert various boolean representations to actual bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    sval = str(value).strip().lower()
    return sval in ['1', 'true', 'yes', 'on']

def _update_field_refs_visibility(field_ref_mapping, is_visible, field_values=None):
    """
    Generic helper to update multiple field refs' visibility and optionally their values.

    Args:
        field_ref_mapping: Dict mapping field names to Ref objects
        is_visible: Boolean indicating if fields should be visible
        field_values: Optional dict mapping field names to values to set
    """
    if field_values is None:
        field_values = {}

    page_obj = None
    try:
        for field_name, ref in field_ref_mapping.items():
            if ref and ref.current:
                ref.current.visible = is_visible
                # Set value if provided and field supports it
                if field_name in field_values and field_values[field_name] is not None:
                    value = field_values[field_name]
                    # Handle checkboxes (boolean values)
                    if isinstance(ref.current, ft.Checkbox):
                        ref.current.value = _convert_bool_value(value)
                    else:
                        ref.current.value = value
                # Keep track of last valid page for batch update
                if ref.current.page:
                    page_obj = ref.current.page
    except Exception:
        pass

    # Update page once at the end
    if page_obj:
        try:
            page_obj.update()
        except Exception:
            pass

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

def _should_show_field(field_name, model_name=None):
    """Determine if a field should be visible for a given model.

    Uses model_field_config.py as the single source of truth.
    Falls back to settings.train_def_model if model_name not provided.
    """
    if model_name is None:
        model_name = settings.train_def_model
    return mfc.get_field_visibility(model_name, field_name)


def _should_show_factor_field():
    """Factor field only shows when adapter is 'lokr'."""
    try:
        if adapter_dropdown_ref and adapter_dropdown_ref.current:
            return adapter_dropdown_ref.current.value == "lokr"
    except Exception:
        pass
    return False


def _on_t_type_change(e):
    """Handle t_type dropdown change - show/hide dependent fields."""
    sync_dependent_field_visibility()


def _on_adapter_change(e):
    """Handle adapter dropdown change - show/hide factor field for lokr."""
    sync_dependent_field_visibility()


def _on_ltx_mode_change(e):
    """Handle ltx_mode dropdown change - show/hide audio_lr field for av mode."""
    if audio_lr_rate_ref and audio_lr_rate_ref.current:
        ltx_mode = e.control.value if e.control else None
        # Show audio_lr field only when ltx_mode is "av"
        should_show = ltx_mode == "av"
        audio_lr_rate_ref.current.visible = should_show
        if audio_lr_rate_ref.current.page:
            audio_lr_rate_ref.current.page.update()


def _on_8bit_te_change(e):
    """8_bit_te and nf4_te are mutually exclusive quantization modes for H3."""
    if e.control.value and nf4_te_checkbox_ref and nf4_te_checkbox_ref.current:
        nf4_te_checkbox_ref.current.value = False
        if nf4_te_checkbox_ref.current.page:
            nf4_te_checkbox_ref.current.page.update()


def _on_nf4_te_change(e):
    """8_bit_te and nf4_te are mutually exclusive quantization modes for H3."""
    if e.control.value and load_text_encoder_in_8bit_checkbox_ref and load_text_encoder_in_8bit_checkbox_ref.current:
        load_text_encoder_in_8bit_checkbox_ref.current.value = False
        if load_text_encoder_in_8bit_checkbox_ref.current.page:
            load_text_encoder_in_8bit_checkbox_ref.current.page.update()


def sync_dependent_field_visibility():
    """
    Centralized function to sync visibility of all dependent fields based on their checkbox states.
    Called from: on_change handlers, model type changes, and after TOML loading.
    """
    # Handle sample_slider_range, i2v_type, sample_each visibility (depends on t_type dropdown + model type)
    try:
        t_type = t_type_dropdown_ref.current.value if t_type_dropdown_ref and t_type_dropdown_ref.current else "none"
        slider_active = t_type == "slider"
        # Get the currently selected model from the dropdown
        current_model = None
        if model_type_dropdown_ref and model_type_dropdown_ref.current:
            current_model = model_type_dropdown_ref.current.value
        # Only show slider-related fields if slider is selected AND it's visible for this model
        is_visible_for_model = _should_show_field("sample_slider_range", current_model)
        should_show_slider_fields = is_visible_for_model and slider_active

        # sample_slider_range
        if sample_slider_range_field_ref and sample_slider_range_field_ref.current:
            sample_slider_range_field_ref.current.visible = should_show_slider_fields
            if sample_slider_range_field_ref.current.page:
                sample_slider_range_field_ref.current.update()

        # i2v_type
        if i2v_type_dropdown_ref and i2v_type_dropdown_ref.current:
            i2v_type_dropdown_ref.current.visible = should_show_slider_fields
            if i2v_type_dropdown_ref.current.page:
                i2v_type_dropdown_ref.current.update()

        # sample_each
        if sample_each_field_ref and sample_each_field_ref.current:
            sample_each_field_ref.current.visible = should_show_slider_fields
            if sample_each_field_ref.current.page:
                sample_each_field_ref.current.update()
    except Exception:
        pass

    # Handle preservation & regularization args field visibility
    try:
        # blank_preservation_args
        blank_preservation_checked = blank_preservation_checkbox_ref.current.value if blank_preservation_checkbox_ref and blank_preservation_checkbox_ref.current else False
        if blank_preservation_args_field_ref and blank_preservation_args_field_ref.current:
            blank_preservation_args_field_ref.current.visible = blank_preservation_checked
            if blank_preservation_args_field_ref.current.page:
                blank_preservation_args_field_ref.current.update()

        # dop_args
        dop_checked = dop_checkbox_ref.current.value if dop_checkbox_ref and dop_checkbox_ref.current else False
        if dop_args_field_ref and dop_args_field_ref.current:
            dop_args_field_ref.current.visible = dop_checked
            if dop_args_field_ref.current.page:
                dop_args_field_ref.current.update()

        # prior_divergence_args
        prior_divergence_checked = prior_divergence_checkbox_ref.current.value if prior_divergence_checkbox_ref and prior_divergence_checkbox_ref.current else False
        if prior_divergence_args_field_ref and prior_divergence_args_field_ref.current:
            prior_divergence_args_field_ref.current.visible = prior_divergence_checked
            if prior_divergence_args_field_ref.current.page:
                prior_divergence_args_field_ref.current.update()

        # self_flow_args
        self_flow_checked = self_flow_checkbox_ref.current.value if self_flow_checkbox_ref and self_flow_checkbox_ref.current else False
        if self_flow_args_field_ref and self_flow_args_field_ref.current:
            self_flow_args_field_ref.current.visible = self_flow_checked
            if self_flow_args_field_ref.current.page:
                self_flow_args_field_ref.current.update()

        # cts_lambda_args
        cts_lambda_checked = cts_lambda_checkbox_ref.current.value if cts_lambda_checkbox_ref and cts_lambda_checkbox_ref.current else False
        if cts_lambda_args_field_ref and cts_lambda_args_field_ref.current:
            cts_lambda_args_field_ref.current.visible = cts_lambda_checked
            if cts_lambda_args_field_ref.current.page:
                cts_lambda_args_field_ref.current.update()

        # forward_xm_args
        forward_xm_checked = forward_xm_checkbox_ref.current.value if forward_xm_checkbox_ref and forward_xm_checkbox_ref.current else False
        if forward_xm_args_field_ref and forward_xm_args_field_ref.current:
            forward_xm_args_field_ref.current.visible = forward_xm_checked
            if forward_xm_args_field_ref.current.page:
                forward_xm_args_field_ref.current.update()

        # CREPA mode and args
        crepa_checked = crepa_checkbox_ref.current.value if crepa_checkbox_ref and crepa_checkbox_ref.current else False
        if crepa_mode_dropdown_ref and crepa_mode_dropdown_ref.current:
            crepa_mode_dropdown_ref.current.visible = crepa_checked
            if crepa_mode_dropdown_ref.current.page:
                crepa_mode_dropdown_ref.current.update()
        if crepa_args_field_ref and crepa_args_field_ref.current:
            crepa_args_field_ref.current.visible = crepa_checked
            if crepa_args_field_ref.current.page:
                crepa_args_field_ref.current.update()

        # Factor field (only for lokr)
        if factor_field_ref and factor_field_ref.current:
            factor_field_ref.current.visible = _should_show_factor_field()
            if factor_field_ref.current.page:
                factor_field_ref.current.update()

        # audio_lr field (only for av ltx_mode)
        ltx_mode = ltx_mode_dropdown_ref.current.value if ltx_mode_dropdown_ref and ltx_mode_dropdown_ref.current else None
        if audio_lr_rate_ref and audio_lr_rate_ref.current:
            audio_lr_rate_ref.current.visible = ltx_mode == "av"
            if audio_lr_rate_ref.current.page:
                audio_lr_rate_ref.current.update()

        # reference_downscale field (only when t_type is ic_lora)
        t_type = t_type_dropdown_ref.current.value if t_type_dropdown_ref and t_type_dropdown_ref.current else "none"
        if reference_downscale_field_ref and reference_downscale_field_ref.current:
            reference_downscale_field_ref.current.visible = (t_type == "ic_lora")
            if reference_downscale_field_ref.current.page:
                reference_downscale_field_ref.current.update()
    except Exception:
        pass


def get_training_config_page_content():
    """Generates Flet controls with hardcoded configuration values, grouped by section."""

    def _safe_set_value(ref, value):
        """Helper to safely set a Ref's value with minimal try/except clutter."""
        if ref and ref.current and value is not None:
            try:
                ref.current.value = value
                if ref.current.page:
                    ref.current.update()
            except Exception:
                pass

    def _apply_field_visibility(sel_norm):
        """Apply visibility rules from model config to all field refs based on selected model."""
        # Get complete field visibility for this model (includes defaults for missing fields)
        show_fields = mfc.get_complete_field_visibility(sel_norm)

        # Map all field names to their field refs
        field_mapping = {
            # Path fields
            "model_path": model_path_field_ref,
            "diffusers_path": diffusers_path_field_ref,
            "transformer_path": transformer_path_field_ref,
            "transformer_path_full": transformer_path_full_ref,
            "text_encoder_path": text_encoder_path_field_ref,
            "vae_path": vae_path_field_ref,
            "vae_audio_path": vae_audio_path_field_ref,
            "tokenizer_path": tokenizer_path_field_ref,
            "llm_path": llm_path_field_ref,
            "ckpt_path": ckpt_path_field_ref,
            "clip_path": clip_path_field_ref,
            "llama3_path": llama3_path_field_ref,
            "byt5_path": byt5_path_field_ref,
            "t5_path": t5_path_field_ref,
            "single_file_path": single_file_path_field_ref,
            "first_frame_conditioning_p": first_frame_conditioning_p_field_ref,
            # Special fields
            "min_t": min_t_field_ref,
            "max_t": max_t_field_ref,
            "max_seq_len": max_seq_len_field_ref,
            "flux_shift": flux_shift_checkbox_ref,
            "bypass_g_emb": bypass_g_emb_checkbox_ref,
            "lumina_shift": lumina_shift_checkbox_ref,
            "float8_e5m2": float8_e5m2_checkbox_ref,
            "longcat_float8": longcat_float8_checkbox_ref,
            "max_llama3_seq_len": max_llama3_seq_len_field_ref,
            "hidream_4bit": hidream_4bit_checkbox_ref,
            "hidream_tdtype": hidream_tdtype_checkbox_ref,
            # SDXL-specific fields
            "v_pred": v_pred_checkbox_ref,
            "d_est_loss": d_est_loss_checkbox_ref,
            "min_snr_gamma": min_snr_gamma_field_ref,
            "unet_lr": unet_lr_field_ref,
            "te1_lr": te1_lr_field_ref,
            "te2_lr": te2_lr_field_ref,
            "llm_adapter_lr": llm_adapter_lr_field_ref,
            # Musubi-specific fields (ltx_mode dropdown)
            "ltx_mode": ltx_mode_dropdown_ref,
            "target_fps": target_fps_field_ref,
            "wan_mode": wan_mode_dropdown_ref,
            "wan_task": wan_task_dropdown_ref,
            "separate_audio_buckets": separate_audio_buckets_checkbox_ref,
            "gradient_checkpointing": gradient_checkpointing_checkbox_ref,
            "ltx_2_3": ltx_2_3_checkbox_ref,
            "8_bit_te": load_text_encoder_in_8bit_checkbox_ref,
            "nf4_te": nf4_te_checkbox_ref,
            "t_type": t_type_dropdown_ref,
            "flash_attn": flash_attn_checkbox_ref,
            "use_mask": use_mask_checkbox_ref,
            "sample_slider_range": sample_slider_range_field_ref,
            "i2v_type": i2v_type_dropdown_ref,
            "sample_each": sample_each_field_ref,
            "ref_downscale": reference_downscale_field_ref,
            # Musubi-specific adapter fields
            "rank": rank_field_ref,
            "alpha": alpha_field_ref,
            "factor": factor_field_ref,
            "network_dropout": network_dropout_field_ref,
            "caption_dropout_rate": caption_dropout_rate_field_ref,
            "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
            # dtype, transformer_dtype, timestep_sm
            "dtype": dtype_dropdown_ref,
            "transformer_dtype": transformer_dtype_dropdown_ref,
            "timestep_sm": timestep_sm_dropdown_ref,
            # Musubi-specific precision fields
            "mixed_precision_mode": mixed_precision_mode_dropdown_ref,
            "fp8_base": fp8_base_checkbox_ref,
            "fp8_scaled": fp8_scaled_checkbox_ref,
            "load_text_encoder_in_8bit": load_text_encoder_in_8bit_checkbox_ref,
            "attn_chunking": attn_chunking_checkbox_ref,
            # Preservation & regularization
            "blank_preservation": blank_preservation_checkbox_ref,
            "dop": dop_checkbox_ref,
            "prior_divergence": prior_divergence_checkbox_ref,
            # CREPA
            "crepa": crepa_checkbox_ref,
            "crepa_mode": crepa_mode_dropdown_ref,
            "crepa_args": crepa_args_field_ref,
            # self_flow and cts_lambda
            "self_flow": self_flow_checkbox_ref,
            "cts_lambda": cts_lambda_checkbox_ref,
            "forward_xm": forward_xm_checkbox_ref,
            # Flux2-specific fields
            "vae": flux2_vae_field_ref,
            "text_encoders": flux2_text_encoders_field_ref,
            "shift": flux2_shift_field_ref,
        }

        try:
            for field_name, is_visible in show_fields.items():
                ref = field_mapping.get(field_name)
                if ref and ref.current:
                    ref.current.visible = is_visible
        except Exception:
            pass

        return show_fields

    def _apply_model_defaults(model_config):
        """Apply default values from model config to field refs."""
        defaults = model_config.get("defaults", {})

        # Comprehensive mapping of all available field refs
        field_mapping = {
            # Path fields
            "model_path": model_path_field_ref,
            "diffusers_path": diffusers_path_field_ref,
            "transformer_path": transformer_path_field_ref,
            "transformer_path_full": transformer_path_full_ref,
            "text_encoder_path": text_encoder_path_field_ref,
            "vae_path": vae_path_field_ref,
            "vae_audio_path": vae_audio_path_field_ref,
            "tokenizer_path": tokenizer_path_field_ref,
            "llm_path": llm_path_field_ref,
            "ckpt_path": ckpt_path_field_ref,
            "clip_path": clip_path_field_ref,
            "llama3_path": llama3_path_field_ref,
            "byt5_path": byt5_path_field_ref,
            "t5_path": t5_path_field_ref,
            "single_file_path": single_file_path_field_ref,
            "first_frame_conditioning_p": first_frame_conditioning_p_field_ref,
            # Text fields
            "min_t": min_t_field_ref,
            "max_t": max_t_field_ref,
            "max_seq_len": max_seq_len_field_ref,
            "min_snr_gamma": min_snr_gamma_field_ref,
            "unet_lr": unet_lr_field_ref,
            "te1_lr": te1_lr_field_ref,
            "te2_lr": te2_lr_field_ref,
            "llm_adapter_lr": llm_adapter_lr_field_ref,
            "max_llama3_seq_len": max_llama3_seq_len_field_ref,
            # Musubi-specific fields
            "target_fps": target_fps_field_ref,
            "sample_slider_range": sample_slider_range_field_ref,
            "i2v_type": i2v_type_dropdown_ref,
            "sample_each": sample_each_field_ref,
            # Musubi-specific adapter fields
            "rank": rank_field_ref,
            "alpha": alpha_field_ref,
            "factor": factor_field_ref,
            "network_dropout": network_dropout_field_ref,
            "caption_dropout_rate": caption_dropout_rate_field_ref,
            "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
            "ref_downscale": reference_downscale_field_ref,
            # Model-specific fields
            "z_image_diffusion_model": z_image_diffusion_model_field_ref,
            "z_image_vae": z_image_vae_field_ref,
            "z_image_text_encoders": z_image_text_encoders_field_ref,
            "z_image_merge_adapters": z_image_merge_adapters_field_ref,
            "diffusion_model": flux2_diffusion_model_field_ref,
            "vae": flux2_vae_field_ref,
            "text_encoders": flux2_text_encoders_field_ref,
            "shift": flux2_shift_field_ref,
        }

        # Boolean field mapping
        bool_field_mapping = {
            "flux_shift": flux_shift_checkbox_ref,
            "lumina_shift": lumina_shift_checkbox_ref,
            "bypass_g_emb": bypass_g_emb_checkbox_ref,
            "v_pred": v_pred_checkbox_ref,
            "d_est_loss": d_est_loss_checkbox_ref,
            "float8_e5m2": float8_e5m2_checkbox_ref,
            "longcat_float8": longcat_float8_checkbox_ref,
            "hidream_4bit": hidream_4bit_checkbox_ref,
            "hidream_tdtype": hidream_tdtype_checkbox_ref,
            # Z-image specific
            "z_image_diffusion_model_dtype_fp8": z_image_diffusion_model_dtype_checkbox_ref,
            "attn_chunking": attn_chunking_checkbox_ref,
            # LTX2 specific - t_type dropdown handles slider/ic_lora/vace_lora selection
            "flash_attn": flash_attn_checkbox_ref,
            "use_mask": use_mask_checkbox_ref,
            # Preservation & regularization args
            "blank_preservation_args": blank_preservation_args_field_ref,
            "dop_args": dop_args_field_ref,
            "prior_divergence_args": prior_divergence_args_field_ref,
            # CREPA
            "crepa_mode": crepa_mode_dropdown_ref,
            "crepa_args": crepa_args_field_ref,
            # self_flow and cts_lambda
            "self_flow_args": self_flow_args_field_ref,
            "cts_lambda_args": cts_lambda_args_field_ref,
            "forward_xm_args": forward_xm_args_field_ref,
        }

        try:
            for field_name, value in defaults.items():
                if field_name in field_mapping:
                    ref = field_mapping[field_name]
                    if ref and ref.current:
                        # Only set default if field is empty
                        current_val = ref.current.value
                        is_empty = not current_val or (isinstance(current_val, str) and current_val.strip() == '')
                        if is_empty:
                            ref.current.value = str(value) if value is not None else ""
                elif field_name in bool_field_mapping:
                    ref = bool_field_mapping[field_name]
                    if ref and ref.current:
                        ref.current.value = _convert_bool_value(value)
        except Exception:
            pass

    def on_trainer_change(e):
        """Handle trainer dropdown change to filter model type options"""
        trainer = trainer_dropdown_ref.current.value if trainer_dropdown_ref.current else None
        if not trainer or not model_type_dropdown_ref.current:
            return

        if trainer == "musubi":
            # Show ltx-video-2 and _wan22 models for musubi trainer
            musubi_models = {k: v for k, v in settings.dpipe_model_dict.items()
                          if "ltx-video-2" in k.lower()}
            # Add _wan22 as a separate entry for musubi trainer
            musubi_models["_wan22"] = "_wan22"
            # Add minimaxH3 as a separate entry for musubi trainer
            musubi_models["minimaxH3"] = "minimaxH3"
            if musubi_models:
                # Update options to show only musubi models
                model_type_dropdown_ref.current.options = [
                    ft.dropdown.Option(key=k, text=v) for k, v in musubi_models.items()
                ]
                # Preserve current value if valid for musubi, otherwise select first
                current_model = model_type_dropdown_ref.current.value
                if current_model and current_model in musubi_models:
                    model_type_dropdown_ref.current.value = current_model
                else:
                    model_type_dropdown_ref.current.value = list(musubi_models.keys())[0]
                # Trigger model type change to update UI
                if model_type_dropdown_ref.current.on_change:
                    model_type_dropdown_ref.current.on_change(e)
        else:  # diffusion-pipe
            # Show all models except ltx-video-2, _wan22, and minimaxH3 (those are musubi-only)
            dpipe_models = {k: v for k, v in settings.dpipe_model_dict.items()
                          if "ltx-video-2" not in k.lower() and k != "_wan22" and k != "minimaxH3"}
            # Update options to show diffusion-pipe models
            model_type_dropdown_ref.current.options = [
                ft.dropdown.Option(key=k, text=v) for k, v in dpipe_models.items()
            ]
            # Preserve current value if valid for diffusion-pipe, otherwise select first
            current_model = model_type_dropdown_ref.current.value
            if current_model and current_model in dpipe_models:
                model_type_dropdown_ref.current.value = current_model
            else:
                model_type_dropdown_ref.current.value = list(dpipe_models.keys())[0]
            # Trigger model type change to update UI
            if model_type_dropdown_ref.current.on_change:
                model_type_dropdown_ref.current.on_change(e)

        if model_type_dropdown_ref.current.page:
            model_type_dropdown_ref.current.update()

    def on_model_type_change(e, from_toml_load=False):
        """Handle model type dropdown change to show/hide model-specific fields

        Args:
            e: Event object
            from_toml_load: If True, skip value updates (values already loaded from TOML)
        """
        sel = model_type_dropdown_ref.current.value if model_type_dropdown_ref.current else None
        if not sel:
            return

        # 1. Normalize and Prep
        sel_norm = mfc.normalize_model_name(sel)
        model_key = mfc.get_model_key(sel_norm)
        skip_defaults = _suppress_model_defaults or from_toml_load

        # 2. Apply Field Visibility
        # We capture the visibility dict to determine if Rows should be hidden
        vis_config = _apply_field_visibility(sel_norm)

        # 3. Determine UI mode based on trainer (trainer takes precedence)
        trainer = trainer_dropdown_ref.current.value if trainer_dropdown_ref.current else None
        uses_musubi_ui = (trainer == "musubi")

        # Update musubi custom section visibility
        if musubi_custom_section_ref and musubi_custom_section_ref.current:
            musubi_custom_section_ref.current.visible = uses_musubi_ui
            if musubi_custom_section_ref.current.page:
                musubi_custom_section_ref.current.page.update()

        # Ensure model_path is visible for SDXL and Musubi ltx-video-2 (but not _wan22)
        if sel_norm == "sdxl" or (uses_musubi_ui and sel_norm not in ["_wan22"]):
            vis_config["model_path"] = True
            if model_path_field_ref and model_path_field_ref.current:
                model_path_field_ref.current.visible = True

        # Force load_text_encoder_in_8bit visible for _wan22 + musubi
        if sel_norm == "_wan22" and uses_musubi_ui:
            vis_config["load_text_encoder_in_8bit"] = True
            if load_text_encoder_in_8bit_checkbox_ref and load_text_encoder_in_8bit_checkbox_ref.current:
                load_text_encoder_in_8bit_checkbox_ref.current.visible = True
            if text_encoder_row_ref and text_encoder_row_ref.current:
                text_encoder_row_ref.current.visible = True
                if text_encoder_row_ref.current.page:
                    text_encoder_row_ref.current.update()

        # Show wan_mode and wan_task for _wan22 (they're hidden by default for wan22)
        # These fields are in the main config area
        if sel_norm == "_wan22":
            if wan_mode_dropdown_ref and wan_mode_dropdown_ref.current:
                wan_mode_dropdown_ref.current.visible = True
                if wan_mode_dropdown_ref.current.page:
                    wan_mode_dropdown_ref.current.page.update()
            if wan_task_dropdown_ref and wan_task_dropdown_ref.current:
                wan_task_dropdown_ref.current.visible = True
                if wan_task_dropdown_ref.current.page:
                    wan_task_dropdown_ref.current.page.update()

        # Hide dtype, transformer_dtype, and timestep_sm for all musubi trainer types
        # These are replaced by mixed_precision_mode in the musubi UI
        if uses_musubi_ui:
            vis_config["dtype"] = False
            vis_config["transformer_dtype"] = False
            vis_config["timestep_sm"] = False
            for ref in [dtype_dropdown_ref, transformer_dtype_dropdown_ref, timestep_sm_dropdown_ref]:
                if ref and ref.current:
                    ref.current.visible = False
                    if ref.current.page:
                        ref.current.update()

            # Show musubi-specific precision fields for all musubi trainer types
            vis_config["mixed_precision_mode"] = True
            vis_config["fp8_base"] = True
            vis_config["fp8_scaled"] = True
            vis_config["attn_chunking"] = True
            for ref in [mixed_precision_mode_dropdown_ref, fp8_base_checkbox_ref, fp8_scaled_checkbox_ref, attn_chunking_checkbox_ref]:
                if ref and ref.current:
                    ref.current.visible = True
                    if ref.current.page:
                        ref.current.update()

        # 3. Dynamic Row Visibility
        # Map Rows to the "Main Field" they contain. If the field is visible, the row is visible.
        row_triggers = {
            checkpoint_row_ref: "model_path",
            ckpt_path_row_ref: "ckpt_path",
            diffusers_row_ref: "diffusers_path",
            single_file_row_ref: "single_file_path",
            first_frame_conditioning_p_row_ref: "first_frame_conditioning_p",
            transformer_full_row_ref: "transformer_path_full",
            byt5_row_ref: "byt5_path",
            t5_row_ref: "t5_path",
            llama3_row_ref: "llama3_path",
            clip_row_ref: "clip_path",
            text_encoder_row_ref: "text_encoder_path",
            flux2_row_ref: "diffusion_model",
            z_image_row_ref: "z_image_diffusion_model",
        }

        try:
            for row_ref, trigger_field in row_triggers.items():
                if row_ref and row_ref.current:
                    # Default to False if the field isn't in config
                    should_show = vis_config.get(trigger_field, False)
                    row_ref.current.visible = should_show
                    if row_ref.current.page:
                        row_ref.current.update()
        except Exception:
            pass

        # 4. Reset Paths and Fields (Clears incompatible fields) - MUST happen BEFORE applying defaults
        if not skip_defaults:
            field_refs_to_reset = {
                "diffusers_path": diffusers_path_field_ref,
                "transformer_path": transformer_path_field_ref,
                "transformer_path_full": transformer_path_full_ref,
                "llm_path": llm_path_field_ref,
                "text_encoder_path": text_encoder_path_field_ref,
                "vae_path": vae_path_field_ref,
            "vae_audio_path": vae_audio_path_field_ref,
            "tokenizer_path": tokenizer_path_field_ref,
                "ckpt_path": ckpt_path_field_ref,
                "clip_path": clip_path_field_ref,
                "llama3_path": llama3_path_field_ref,
                "max_llama3_seq_len": max_llama3_seq_len_field_ref,
                "byt5_path": byt5_path_field_ref,
                "single_file_path": single_file_path_field_ref,
                "first_frame_conditioning_p": first_frame_conditioning_p_field_ref,
                "t5_path": t5_path_field_ref,
                "model_path": model_path_field_ref,
                "llm_adapter_lr": llm_adapter_lr_field_ref,
                "hidream_4bit": hidream_4bit_checkbox_ref,
                "hidream_tdtype": hidream_tdtype_checkbox_ref,
                # Musubi-specific adapter fields
                "rank": rank_field_ref,
                "alpha": alpha_field_ref,
                "network_dropout": network_dropout_field_ref,
                "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
                # Flux2-specific fields
                "diffusion_model": flux2_diffusion_model_field_ref,
                "vae": flux2_vae_field_ref,
                "text_encoders": flux2_text_encoders_field_ref,
                "shift": flux2_shift_field_ref,
                # Z_image-specific fields
                "z_image_diffusion_model": z_image_diffusion_model_field_ref,
                "z_image_vae": z_image_vae_field_ref,
                "z_image_text_encoders": z_image_text_encoders_field_ref,
                "z_image_merge_adapters": z_image_merge_adapters_field_ref,
            }
            mfc.reset_all_model_fields(field_refs_to_reset)

        # 5. Apply Defaults (Values)
        if model_key and mfc.MODEL_CONFIG.get(model_key) and not skip_defaults:
            _apply_model_defaults(mfc.MODEL_CONFIG[model_key])

        # 6. Apply Dropdown Defaults (Timestep / Dtype)
        if not skip_defaults:
            _safe_set_value(timestep_sm_dropdown_ref, mfc.get_timestep_sm_default(sel_norm))
            _safe_set_value(transformer_dtype_dropdown_ref, mfc.get_transformer_dtype_default(sel_norm))

        # 6.5. Apply Musubi-specific precision defaults (for all musubi trainer types)
        if uses_musubi_ui and not skip_defaults:
            _safe_set_value(mixed_precision_mode_dropdown_ref, "bf16")
            _safe_set_value(fp8_base_checkbox_ref, True)
            _safe_set_value(fp8_scaled_checkbox_ref, True)
            _safe_set_value(attn_chunking_checkbox_ref, False)

        # 7. Handle model-specific defaults not yet in config (model-specific field overrides)
        # Note: Most defaults are now in model_field_config.py and applied in step 5

        # 8. Dynamic Row Visibility (handled earlier)

        # 9. Conditional UI Swap for Musubi Trainer
        # Toggle standard training sections (hide for musubi trainer)
        if standard_training_section_ref.current:
            standard_training_section_ref.current.visible = not uses_musubi_ui
            if standard_training_section_ref.current.page:
                standard_training_section_ref.current.update()
        if standard_eval_optimizer_section_ref.current:
            standard_eval_optimizer_section_ref.current.visible = not uses_musubi_ui
            if standard_eval_optimizer_section_ref.current.page:
                standard_eval_optimizer_section_ref.current.update()

        # Toggle Musubi custom section (show only for musubi trainer)
        if musubi_custom_section_ref.current:
            show_musubi = uses_musubi_ui
            musubi_custom_section_ref.current.visible = show_musubi
            # Force update the musubi section
            if musubi_custom_section_ref.current.page:
                musubi_custom_section_ref.current.update()

        # 10. Update adapter field visibility (hide old ones for Musubi, show new ones only for Musubi)
        musubi_adapter_fields = {
            "rank": rank_field_ref,
            "alpha": alpha_field_ref,
            "factor": factor_field_ref,
            "network_dropout": network_dropout_field_ref,
            "caption_dropout_rate": caption_dropout_rate_field_ref,
            "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
        }
        old_adapter_fields = {
            "a_rank": a_rank_field_ref,
            "a_dtype": a_dtype_field_ref,
            "blocks_swap": blocks_swap_field_ref,
            "disable_bsfe": disable_bsfe_field_ref,
        }
        try:
            # Show new Musubi fields only for Musubi trainer
            # But hide dropout and first_frame_conditioning for wan/wan22
            for field_name, ref in musubi_adapter_fields.items():
                if ref and ref.current:
                    should_hide = (field_name in ["network_dropout", "caption_dropout_rate", "first_frame_conditioning_p_ltx2"] and sel_norm in ["wan", "wan22"])
                    ref.current.visible = uses_musubi_ui and not should_hide
                    if ref.current.page:
                        ref.current.update()
            # Hide old adapter fields for Musubi trainer
            for field_name, ref in old_adapter_fields.items():
                if ref and ref.current:
                    ref.current.visible = not uses_musubi_ui
                    if ref.current.page:
                        ref.current.update()
        except Exception:
            pass

        # 11. Show/hide frame_extraction dropdowns for dataset blocks (only for Musubi trainer)
        try:
            for ds_block in [dataset_1_block, dataset_2_block, dataset_3_block]:
                if hasattr(ds_block, 'set_frame_extraction_visible'):
                    ds_block.set_frame_extraction_visible(uses_musubi_ui)
        except Exception:
            pass

        # 11.5. Update adapter dropdown options based on trainer type
        # For Musubi: show both "lora" and "lokr"
        # For other trainers: show only "lora"
        try:
            if adapter_dropdown_ref and adapter_dropdown_ref.current:
                if uses_musubi_ui:
                    # Musubi supports both lora and lokr
                    adapter_dropdown_ref.current.options = [
                        ft.dropdown_option("lora"),
                        ft.dropdown_option("lokr"),
                    ]
                else:
                    # Other models only support lora
                    adapter_dropdown_ref.current.options = [
                        ft.dropdown_option("lora"),
                    ]
                    # Reset to lora if currently on lokr
                    if adapter_dropdown_ref.current.value == "lokr":
                        adapter_dropdown_ref.current.value = "lora"
                if adapter_dropdown_ref.current.page:
                    adapter_dropdown_ref.current.update()
        except Exception:
            pass

        # 12. Sync dependent field visibility (sample_slider_range, preservation args)
        sync_dependent_field_visibility()

        # 13. Update page
        if e and getattr(e, 'page', None):
            e.page.update()

    page_controls = []

    # --- Model Configuration & Dataset Selection (Side by Side) ---

    # Create 3 independent dataset blocks for Dataset 1, Dataset 2, Dataset 3
    dataset_1_block = get_compact_dataset_block("Dataset 1")
    dataset_2_block = get_compact_dataset_block("Dataset 2")
    dataset_3_block = get_compact_dataset_block("Dataset 3")

    # Refresh button for all datasets
    def refresh_all_datasets(e):
        for ds_block in [dataset_1_block, dataset_2_block, dataset_3_block]:
            if hasattr(ds_block, 'reload_datasets'):
                try:
                    ds_block.reload_datasets()
                except Exception:
                    pass

    refresh_button = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh dataset lists",
        on_click=refresh_all_datasets,
        style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=0, vertical=0)),
        icon_size=20
    )

    page_controls.append(
        ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Model Configuration"),
                ft.Container(
                    content=ft.Column([
                    ft.ResponsiveRow(controls=[
                        create_dropdown(
                            "Trainer",
                            "diffusion-pipe",
                            {"diffusion-pipe": "diffusion-pipe", "musubi": "musubi"},
                            col=3, expand=True,fill_color=ft.Colors.with_opacity(0.18, ft.Colors.AMBER_900),
                            on_change=on_trainer_change, ref=trainer_dropdown_ref
                        ),
                        create_dropdown(
                            "Model Type",
                            settings.train_def_model,
                            settings.dpipe_model_dict,
                            hint_text="Select model or specify path below", col=3, expand=True,
                            fill_color=ft.Colors.with_opacity(0.18, ft.Colors.AMBER_900),
                            on_change=on_model_type_change, ref=model_type_dropdown_ref
                        ),
                        # Non-Musubi fields
                        create_dropdown(
                            "dtype",
                            "bfloat16",
                            {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"},
                            col=2, expand=True, scale=0.8, ref=dtype_dropdown_ref,
                            visible=_should_show_field("dtype")
                        ),
                        create_dropdown(
                            "transformer_dtype",
                            "float8",
                            {"float8": "float8", "None": "None"},
                            col=2, expand=True, scale=0.8, ref=transformer_dtype_dropdown_ref,
                            visible=_should_show_field("transformer_dtype")
                        ),
                        create_dropdown(
                            "timestep_sm",
                            "logit_normal",
                            {"logit_normal": "logit_normal", "uniform": "uniform", "None": "None"},
                            col=2, expand=True, scale=0.8, ref=timestep_sm_dropdown_ref,
                            visible=_should_show_field("timestep_sm")
                        ),
                        # Musubi-specific precision fields
                        create_dropdown(
                            "mixed_precision_mode",
                            "bf16",
                            {"no": "no", "fp16": "fp16", "bf16": "bf16"},
                            col=2, expand=True, scale=0.8, ref=mixed_precision_mode_dropdown_ref,
                            visible=_should_show_field("mixed_precision_mode")
                        ),
                        ft.Checkbox(
                            label="fp8_base",
                            value=True,
                            scale=0.8,
                            ref=fp8_base_checkbox_ref,
                            visible=_should_show_field("fp8_base"),
                            data="fp8_base",
                            col=2,
                        ),
                        ft.Checkbox(
                            label="fp8_scaled",
                            value=True,
                            scale=0.8,
                            ref=fp8_scaled_checkbox_ref,
                            visible=_should_show_field("fp8_scaled"),
                            data="fp8_scaled",
                            col=2,
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "model_path",
                                "",
                                col=12, expand=True, ref=model_path_field_ref,
                                visible=_should_show_field("model_path")
                            ),
                        ],
                        ref=checkpoint_row_ref,
                        visible=_should_show_field("model_path")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "ckpt_path",
                                "models/Wan2.2-T2V-A14B",
                                col=12, expand=True, ref=ckpt_path_field_ref,
                                visible=_should_show_field("ckpt_path")
                            ),
                        ],
                        ref=ckpt_path_row_ref,
                        visible=_should_show_field("ckpt_path")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield("text_encoder_path", "", col=6, expand=True, ref=text_encoder_path_field_ref, visible=_should_show_field("text_encoder_path")),
                            create_textfield("tokenizer_path", "", col=6, expand=True, ref=tokenizer_path_field_ref, visible=_should_show_field("tokenizer_path")),
                        ], spacing=2,
                        ref=text_encoder_row_ref,
                        visible=True
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "diffusers_path", "models/Qwen-Image", col=12, expand=True,
                                ref=diffusers_path_field_ref, visible=_should_show_field("diffusers_path")
                            ),
                        ], spacing=2,
                        ref=diffusers_row_ref,
                        visible=_should_show_field("diffusers_path")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "single_file_path", "", col=12, expand=True,
                                ref=single_file_path_field_ref, visible=_should_show_field("single_file_path")
                            ),
                        ], spacing=2,
                        ref=single_file_row_ref,
                        visible=_should_show_field("single_file_path")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "first_frame_conditioning_p", "0.0", col=12, expand=True,
                                ref=first_frame_conditioning_p_field_ref, visible=_should_show_field("first_frame_conditioning_p")
                            ),
                        ], spacing=2,
                        ref=first_frame_conditioning_p_row_ref,
                        visible=_should_show_field("first_frame_conditioning_p")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "transformer_path", "", col=12, expand=True,
                                ref=transformer_path_full_ref, visible=_should_show_field("transformer_path_full")
                            ),
                        ], spacing=2,
                        ref=transformer_full_row_ref,
                        visible=_should_show_field("transformer_path_full")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "byt5_path", "", col=12, expand=True,
                                ref=byt5_path_field_ref, visible=_should_show_field("byt5_path")
                            ),
                        ], spacing=2,
                        ref=byt5_row_ref,
                        visible=_should_show_field("byt5_path")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "t5_path", "", col=12, expand=True,
                                ref=t5_path_field_ref, visible=_should_show_field("t5_path")
                            ),
                        ], spacing=2,
                        ref=t5_row_ref,
                        visible=_should_show_field("t5_path")
                    ),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "llama3_path", "", col=12, expand=True,
                            ref=llama3_path_field_ref, visible=_should_show_field("llama3_path")
                        ),
                    ], spacing=2, ref=llama3_row_ref, visible=_should_show_field("llama3_path")),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "clip_path", "", col=12, expand=True,
                            ref=clip_path_field_ref, visible=_should_show_field("clip_path")
                        ),
                    ], spacing=2, ref=clip_row_ref, visible=_should_show_field("clip_path")),
                    # z_image specific fields
                    ft.ResponsiveRow(
                        controls=[
                            ft.Column([
                                create_textfield(
                                    "diffusion_model", "models/z_image_turbo/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
                                    col=12, expand=True, ref=z_image_diffusion_model_field_ref
                                ),
                                create_textfield(
                                    "vae", "models/z_image_turbo/split_files/vae/ae.safetensors",
                                    col=12, expand=True, ref=z_image_vae_field_ref
                                ),
                            ], col=6, spacing=2),
                            ft.Column([
                                create_textfield(
                                    "text_encoders", "models/z_image_turbo/split_files/text_encoders/qwen_3_4b.safetensors",
                                    col=12, expand=True, ref=z_image_text_encoders_field_ref
                                ),
                                create_textfield(
                                    "merge_adapters", "models/z_image_turbo/zimage_turbo_training_adapter_v2.safetensors",
                                    col=12, expand=True, ref=z_image_merge_adapters_field_ref
                                ),
                                ft.Checkbox(
                                    label="diffusion_model_dtype_fp8",
                                    value=False,
                                    scale=0.8,
                                    ref=z_image_diffusion_model_dtype_checkbox_ref,
                                    data="diffusion_model_dtype",  # actual key for config
                                ),
                            ], col=6, spacing=2),
                        ],
                        spacing=2,
                        ref=z_image_row_ref,
                        visible=_should_show_field("z_image_diffusion_model")
                    ),
                    # flux2 specific fields
                    ft.ResponsiveRow(
                        controls=[
                            ft.Column([
                                create_textfield(
                                    "diffusion_model", "",
                                    col=12, expand=True, ref=flux2_diffusion_model_field_ref
                                ),
                                create_textfield(
                                    "vae", "models/vae/flux2-vae.safetensors",
                                    col=12, expand=True, ref=flux2_vae_field_ref
                                ),
                            ], col=6, spacing=2),
                            ft.Column([
                                create_textfield(
                                    "text_encoders", "models/text_encoders/mistral_3_small_flux2_fp8.safetensors",
                                    col=12, expand=True, ref=flux2_text_encoders_field_ref
                                ),
                                create_textfield(
                                    "shift", "3",
                                    col=12, expand=True, ref=flux2_shift_field_ref
                                ),
                            ], col=6, spacing=2),
                        ],
                        spacing=2,
                        ref=flux2_row_ref,
                        visible=_should_show_field("diffusion_model")
                    ),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "transformer_path", "", col=6, expand=True,
                            ref=transformer_path_field_ref, visible=_should_show_field("transformer_path")
                        ),
                        ft.Checkbox(
                            label="float8_e5m2",
                            value=False,
                            scale=0.8,
                            ref=float8_e5m2_checkbox_ref,
                            visible=_should_show_field("float8_e5m2"),
                            col=6,
                        ),
                        ft.Checkbox(
                            label="float8 t_dtype",
                            value=False,
                            scale=0.8,
                            ref=longcat_float8_checkbox_ref,
                            visible=_should_show_field("longcat_float8"),
                            col=6,
                        ),
                        create_textfield(
                            "llm_path", "", col=6, expand=True,
                            ref=llm_path_field_ref, visible=_should_show_field("llm_path")
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        create_textfield("vae_path", "", col=6, expand=True, ref=vae_path_field_ref, visible=_should_show_field("vae_path")),
                        create_textfield("vae_audio_path", "", col=6, expand=True, ref=vae_audio_path_field_ref, visible=_should_show_field("vae_audio_path")),
                ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        create_textfield("llm_adapter_lr", "", col=12, expand=True, ref=llm_adapter_lr_field_ref, visible=_should_show_field("llm_adapter_lr")),
                ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        create_textfield(
                            "max_llama3_sequence_length", 128, col=6, expand=True,
                            ref=max_llama3_seq_len_field_ref, visible=_should_show_field("max_llama3_seq_len")
                        ),
                        ft.Checkbox(
                            label="llama3_4bit",
                            value=True,
                            scale=0.8,
                            ref=hidream_4bit_checkbox_ref,
                            visible=_should_show_field("hidream_4bit"),
                            col=3,
                        ),
                        ft.Checkbox(
                            label="t_dtype_nf4",
                            value=False,
                            scale=0.8,
                            ref=hidream_tdtype_checkbox_ref,
                            visible=_should_show_field("hidream_tdtype"),
                            col=3,
                        ),
                    ], spacing=2),
                    # Musubi checkboxes row 1: ltx_2_3, sab, gradient_checkpointing, 8_bit_te
                    ft.ResponsiveRow(controls=[
                        ft.Checkbox(
                            label="ltx_2_3",
                            value=False,
                            scale=0.8,
                            adaptive=True,
                            data="ltx_2_3",
                            ref=ltx_2_3_checkbox_ref,
                            visible=_should_show_field("ltx_2_3"),
                            col=2,
                        ),
                        ft.Checkbox(
                            label="separate_audio_buckets",
                            value=True,
                            scale=0.8,
                            tooltip="Separate audio buckets",
                            ref=separate_audio_buckets_checkbox_ref,
                            visible=_should_show_field("separate_audio_buckets"),
                            data="separate_audio_buckets",
                            col=3.5,
                        ),
                        ft.Checkbox(
                            label="gradient_checkpointing",
                            value=True,
                            scale=0.8,
                            ref=gradient_checkpointing_checkbox_ref,
                            visible=_should_show_field("gradient_checkpointing"),
                            data="gradient_checkpointing",
                            col=3.5,
                        ),
                        ft.Checkbox(
                            label="8_bit_te",
                            value=True,
                            scale=0.8,
                            ref=load_text_encoder_in_8bit_checkbox_ref,
                            visible=True,
                            data="8_bit_te",
                            on_change=_on_8bit_te_change,
                            col=3,
                        ),
                        ft.Checkbox(
                            label="nf4_te",
                            value=False,
                            scale=0.8,
                            ref=nf4_te_checkbox_ref,
                            visible=_should_show_field("nf4_te"),
                            data="nf4_te",
                            on_change=_on_nf4_te_change,
                            col=3,
                        ),
                    ], spacing=2),
                    # Musubi row: t_type dropdown, flash_attn checkbox, use_mask checkbox
                    ft.ResponsiveRow(controls=[
                        create_dropdown(
                            "t_type",
                            "none",
                            {"none": "none", "slider": "slider", "ic_lora": "ic_lora", "vace_lora": "vace_lora"},
                            col=2.5,
                            expand=False,
                            scale=0.8,
                            ref=t_type_dropdown_ref,
                            visible=_should_show_field("t_type"),
                            on_change=_on_t_type_change
                        ),
                        ft.Checkbox(
                            label="flash_attn",
                            value=True,
                            scale=0.8,
                            ref=flash_attn_checkbox_ref,
                            visible=_should_show_field("flash_attn"),
                            data="flash_attn",
                            col=3,
                        ),
                        ft.Checkbox(
                            label="use_mask",
                            value=False,
                            scale=0.8,
                            ref=use_mask_checkbox_ref,
                            visible=_should_show_field("use_mask"),
                            data="use_mask",
                            col=3,
                        ),
                    ], spacing=2),
                    # Adapter row with sample_slider_range
                    ft.ResponsiveRow(controls=[
                        create_dropdown(
                            "adapter",
                            "lora",
                            {
                                "lora": "lora",
                                "lokr": "lokr",
                                "full": "full"
                            }, col=1.8, expand=False, scale=0.8, ref=adapter_dropdown_ref, on_change=_on_adapter_change,
                        ),
                        create_dropdown(
                            "ltx_mode",
                            "video",
                            {"video": "video", "av": "av", "audio": "audio"},
                            col=1.7, expand=True, scale=0.8, ref=ltx_mode_dropdown_ref,
                            visible=_should_show_field("ltx_mode"),
                            on_change=_on_ltx_mode_change
                        ),
                        create_textfield(
                            "target_fps", "25",
                            hint_text="Fps to cache and train at",
                            expand=True,
                            col=1.5,
                            scale=0.8,
                            ref=target_fps_field_ref,
                            visible=_should_show_field("target_fps"),
                        ),
                        # Wan2.2 mode dropdown
                        create_dropdown(
                            "wan_mode",
                            "high",
                            {"high": "high", "low": "low", "both": "both"},
                            col=2.0, expand=True, scale=0.8, ref=wan_mode_dropdown_ref,
                            visible=_should_show_field("wan_mode")
                        ),
                        create_dropdown(
                            "wan_task",
                            "i2v-A14B",
                            {"i2v-A14B": "i2v-A14B","t2v-A14B": "t2v-A14B", },
                            col=2.5, expand=True, scale=0.8, ref=wan_task_dropdown_ref,
                            visible=_should_show_field("wan_task")
                        ),
                        #slider
                        create_textfield(
                            "sample_slider_range", "0.0, 1.0, 2.0",
                            hint_text="Slider sample range",
                            expand=True, col=2, scale=0.8,
                            ref=sample_slider_range_field_ref,
                            visible=False  # Invisible by default (only visible when slider checkbox is checked)
                        ),
                        create_dropdown(
                            "i2v_type",
                            "jump",
                            {"jump": "jump", "freeze": "freeze", "fade": "fade", "reverse": "reverse"},
                            col=1.5, expand=True, scale=0.8, ref=i2v_type_dropdown_ref,
                            visible=False  # Invisible by default (only visible when slider checkbox is checked)
                        ),
                        create_textfield(
                            "sample_each", "3",
                            hint_text="Sample each",
                            expand=True, col=1.5, scale=0.8,
                            ref=sample_each_field_ref,
                            visible=False  # Invisible by default (only visible when slider checkbox is checked)
                        ),
                        create_textfield(
                            "ref_downscale", 1,
                            hint_text="1=same, 2=half res",
                            tooltip="Spatial downscale factor for IC-LoRA references: 1=same resolution, 2=half resolution",
                            expand=True, col=1.5, scale=0.8,
                            ref=reference_downscale_field_ref,
                            visible=False  # Only visible when ic_lora is checked
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        ft.Column([
                            ft.ResponsiveRow(controls=[
                                create_textfield("min_t", 0.9, hint_text="HIGH = 0.9 , LOW t2v = 0.0 ,i2v = 0.0", expand=True, col=6, scale=0.8, ref=min_t_field_ref, visible=_should_show_field("min_t")),
                                create_textfield("max_t", 1.000, hint_text="HIGH = 1.0 , LOW t2v = 0.875 ,i2v = 0.900 ,", expand=True, col=6, scale=0.8, ref=max_t_field_ref, visible=_should_show_field("max_t")),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                create_textfield(
                                    "max_sequence_length", 768, expand=True, col=12, scale=0.8,
                                    ref=max_seq_len_field_ref, visible=_should_show_field("max_seq_len")
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                ft.Checkbox(
                                    label="flux_shift",
                                    value=True,
                                    scale=0.8,
                                    ref=flux_shift_checkbox_ref,
                                    visible=_should_show_field("flux_shift"),
                                    col=6,
                                ),
                                ft.Checkbox(
                                    label="bypass_g_emb",
                                    value=True,
                                    scale=0.8,
                                    ref=bypass_g_emb_checkbox_ref,
                                    visible=_should_show_field("bypass_g_emb"),
                                    col=6,
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                ft.Checkbox(
                                    label="lumina_shift",
                                    value=True,
                                    scale=0.8,
                                    ref=lumina_shift_checkbox_ref,
                                    visible=_should_show_field("lumina_shift"),
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
                                    visible=_should_show_field("v_pred"),
                                    col=6,
                                ),
                                ft.Checkbox(
                                    label="d_est_loss",
                                    value=True,
                                    scale=0.8,
                                    ref=d_est_loss_checkbox_ref,
                                    visible=_should_show_field("d_est_loss"),
                                    col=6,
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                create_textfield(
                                    "min_snr_gamma", 5, expand=True, col=6, scale=0.8,
                                    ref=min_snr_gamma_field_ref, visible=_should_show_field("min_snr_gamma")
                                ),
                                create_textfield(
                                    "unet_lr", 4e-5, expand=True, col=6, scale=0.8,
                                    ref=unet_lr_field_ref, visible=_should_show_field("unet_lr")
                                ),
                            ], spacing=2),
                            ft.ResponsiveRow(controls=[
                                create_textfield(
                                    "text_encoder_1_lr", 2e-5, expand=True, col=6, scale=0.8,
                                    ref=te1_lr_field_ref, visible=_should_show_field("te1_lr")
                                ),
                                create_textfield(
                                    "text_encoder_2_lr", 2e-5, expand=True, col=6, scale=0.8,
                                    ref=te2_lr_field_ref, visible=_should_show_field("te2_lr")
                                ),
                            ], spacing=2),
                        ], col=9, spacing=2, alignment=ft.MainAxisAlignment.START),
                    ], spacing=2, vertical_alignment=ft.CrossAxisAlignment.START),
                    # Adapter details row
                    ft.ResponsiveRow(controls=[
                        create_textfield("a_rank", 32, col=3, expand=True, ref=a_rank_field_ref, visible=not _should_show_field("rank")),
                        create_textfield("a_dtype", "bfloat16", col=3, expand=True, ref=a_dtype_field_ref, visible=not _should_show_field("rank")),
                        create_textfield("blocks_swap", 0, col=3, expand=True, ref=blocks_swap_field_ref, visible=not _should_show_field("rank")),
                        create_textfield("disable_bsfe", "true", col=3, expand=True, ref=disable_bsfe_field_ref, visible=not _should_show_field("rank")),
                    ], spacing=2),
                    # Musubi-specific adapter row
                    ft.ResponsiveRow(controls=[
                        create_textfield("rank", 32, col=1.5, expand=True, ref=rank_field_ref, visible=_should_show_field("rank")),
                        create_textfield("alpha", 32, col=1.5, expand=True, ref=alpha_field_ref, visible=_should_show_field("alpha")),
                        create_textfield("factor", 4, col=2, expand=True, ref=factor_field_ref, visible=_should_show_factor_field()),
                        create_textfield("network_dropout", 0.0, col=2, expand=True, ref=network_dropout_field_ref, visible=_should_show_field("network_dropout")),
                        create_textfield("caption_dropout_rate", 0.0, col=2, expand=True, ref=caption_dropout_rate_field_ref, visible=_should_show_field("caption_dropout_rate")),
                        create_textfield("first_frame_conditioning_p", 0.5, col=3, expand=True, ref=first_frame_conditioning_p_ltx2_field_ref, visible=_should_show_field("first_frame_conditioning_p_ltx2")),
                    ], spacing=2),
                ]),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                    bgcolor=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
                )
            ], col=6), # Model Configuration column set to 6
            ft.Column([
                # Dataset Selection title with refresh button
                ft.Row(
                    [
                        ft.Text("Dataset Selection", size=16, weight=ft.FontWeight.BOLD),
                        refresh_button,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Divider(height=5, thickness=1),
                # 3 dataset columns side by side
                ft.Row(
                    [
                        ft.Container(
                            content=dataset_1_block,
                            expand=True,
                            padding=ft.padding.all(5),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                            border_radius=ft.border_radius.all(10),
                        ),
                        ft.Container(
                            content=dataset_2_block,
                            expand=True,
                            padding=ft.padding.all(5),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                            border_radius=ft.border_radius.all(10),
                        ),
                        ft.Container(
                            content=dataset_3_block,
                            expand=True,
                            padding=ft.padding.all(5),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                            border_radius=ft.border_radius.all(10),
                        ),
                    ],
                    spacing=10,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            ], col=6),
        ], spacing=20, vertical_alignment=ft.CrossAxisAlignment.START)
    )
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))

    # --- Training & Misc Settings (Two Columns) ---
    standard_training_section = ft.ResponsiveRow([
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
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START, ref=standard_training_section_ref)

    # Determine initial visibility based on default trainer (not model)
    default_trainer = trainer_dropdown_ref.current.value if trainer_dropdown_ref.current else None
    uses_musubi_ui = (default_trainer == "musubi")
    standard_training_section.visible = not uses_musubi_ui
    page_controls.append(standard_training_section)

    # --- Eval & Optimizer Settings (Two Columns) ---
    standard_eval_optimizer_section = ft.ResponsiveRow([
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
                            create_dropdown(
                                "optimizer_type",
                                "adamw_optimi",
                                ofc.OPTIMIZER_OPTIONS,
                                col=12,
                                expand=True,
                                ref=optimizer_type_dropdown_ref,
                                on_change=lambda e: on_optimizer_type_change(e),
                            ),
                        ], spacing=4),
                        ft.ResponsiveRow(controls=[
                            create_textfield("lr", 2e-5, col=3, expand=True, ref=lr_field_ref),
                            create_textfield("betas", "[0.9, 0.99]", col=3, expand=True, ref=betas_field_ref),
                            create_textfield("weight_decay", 0.01, col=3, expand=True, ref=weight_decay_field_ref),
                            create_textfield("eps", 1e-8, col=3, expand=True, ref=eps_field_ref),
                        ], spacing=4),
                        # Prodigy-specific row (hidden by default)
                        ft.ResponsiveRow(controls=[
                            create_textfield("beta3", "None", col=3, expand=True, ref=beta3_field_ref),
                            create_textfield("d0", "1e-6", col=3, expand=True, ref=d0_field_ref),
                            create_textfield("d_coef", "1.0", col=3, expand=True, ref=d_coef_field_ref),
                            create_textfield("schedulefree_c", "0.0", col=3, expand=True, ref=schedulefree_c_field_ref),
                        ], spacing=4, ref=prodigy_row_ref, visible=False),
                        # Automagic-specific row (hidden by default)
                        ft.ResponsiveRow(controls=[
                            create_textfield("min_lr", "1e-7", col=2, expand=True, ref=min_lr_field_ref),
                            create_textfield("max_lr", "1e-3", col=2, expand=True, ref=max_lr_field_ref),
                            create_textfield("lr_bump", "1e-6", col=2, expand=True, ref=lr_bump_field_ref),
                            create_textfield("clip_threshold", "1.0", col=2, expand=True, ref=clip_threshold_field_ref),
                            ft.Checkbox(label="do_paramiter_swapping", value=False, col=2, ref=do_paramiter_swapping_field_ref),
                            create_textfield("paramiter_swapping_factor", "0.1", col=2, expand=True, ref=paramiter_swapping_factor_field_ref),
                        ], spacing=4, ref=automagic_row_ref, visible=False),
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
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START, ref=standard_eval_optimizer_section_ref)

    standard_eval_optimizer_section.visible = not uses_musubi_ui
    page_controls.append(standard_eval_optimizer_section)

    # --- Musubi Custom Training Settings ---
    musubi_custom_section = get_musubi_training_settings(
        ref=musubi_custom_section_ref,
        attn_chunking_ref=attn_chunking_checkbox_ref,
        blank_preservation_ref=blank_preservation_checkbox_ref,
        blank_preservation_args_ref=blank_preservation_args_field_ref,
        dop_ref=dop_checkbox_ref,
        dop_args_ref=dop_args_field_ref,
        prior_divergence_ref=prior_divergence_checkbox_ref,
        prior_divergence_args_ref=prior_divergence_args_field_ref,
        crepa_ref=crepa_checkbox_ref,
        crepa_mode_ref=crepa_mode_dropdown_ref,
        crepa_args_ref=crepa_args_field_ref,
        self_flow_ref=self_flow_checkbox_ref,
        self_flow_args_ref=self_flow_args_field_ref,
        cts_lambda_ref=cts_lambda_checkbox_ref,
        cts_lambda_args_ref=cts_lambda_args_field_ref,
        forward_xm_ref=forward_xm_checkbox_ref,
        forward_xm_args_ref=forward_xm_args_field_ref,
        extra_flags_ref=extra_flags_field_ref,
        audio_lr_rate_ref=audio_lr_rate_ref,
        sync_visibility_func=sync_dependent_field_visibility,
    )
    musubi_custom_section.visible = uses_musubi_ui
    page_controls.append(musubi_custom_section)

    container = ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8, # Slightly reduced spacing between controls/sections
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True, # Allow container to take full height
        padding=ft.padding.all(5)
    )
    # Expose the 3 independent dataset blocks (only Dataset 1 is checked for selected dataset)
    container.dataset_1_block = dataset_1_block
    container.dataset_2_block = dataset_2_block
    container.dataset_3_block = dataset_3_block
    # For backward compatibility, also expose as dataset_block (points to dataset_1)
    container.dataset_block = dataset_1_block

    # Initialize model type options based on default trainer (diffusion-pipe)
    # Filter out ltx-video-2 and wan22 from initial options since default trainer is diffusion-pipe
    if trainer_dropdown_ref and trainer_dropdown_ref.current and model_type_dropdown_ref and model_type_dropdown_ref.current:
        default_trainer = trainer_dropdown_ref.current.value
        if default_trainer == "musubi":
            # Show ltx-video-2, wan22, and minimaxH3 models
            musubi_models = {k: v for k, v in settings.dpipe_model_dict.items()
                          if "ltx-video-2" in k.lower() or "wan22" in k.lower()}
            musubi_models["minimaxH3"] = "minimaxH3"
            if musubi_models:
                model_type_dropdown_ref.current.options = [
                    ft.dropdown.Option(key=k, text=v) for k, v in musubi_models.items()
                ]
                first_model = list(musubi_models.keys())[0]
                model_type_dropdown_ref.current.value = first_model
        else:
            # Show all models except ltx-video-2 (wan22 works with both trainers)
            dpipe_models = {k: v for k, v in settings.dpipe_model_dict.items()
                          if "ltx-video-2" not in k.lower()}
            if dpipe_models:
                model_type_dropdown_ref.current.options = [
                    ft.dropdown.Option(key=k, text=v) for k, v in dpipe_models.items()
                ]
                # Keep the default model from settings if it's in the filtered list
                default_model = settings.train_def_model
                if default_model in dpipe_models:
                    model_type_dropdown_ref.current.value = default_model
                else:
                    model_type_dropdown_ref.current.value = list(dpipe_models.keys())[0]

    return container

def update_wan_fields_visibility(is_wan22: bool, min_t_value=None, max_t_value=None):
    """Update visibility and values of min_t and max_t fields from external calls"""
    field_refs = {
        "min_t": min_t_field_ref,
        "max_t": max_t_field_ref,
    }
    field_values = {}
    if min_t_value is not None:
        field_values["min_t"] = str(min_t_value)
    if max_t_value is not None:
        field_values["max_t"] = str(max_t_value)

    _update_field_refs_visibility(field_refs, is_wan22, field_values)

def update_auraflow_fields_visibility(is_auraflow: bool, max_sequence_length_value=None):
    """Update visibility and value of max_sequence_length for auraflow from external calls"""
    field_refs = {"max_seq_len": max_seq_len_field_ref}
    field_values = {}
    if max_sequence_length_value is not None:
        field_values["max_seq_len"] = str(max_sequence_length_value)

    _update_field_refs_visibility(field_refs, is_auraflow, field_values)

def update_chroma_fields_visibility(is_chroma: bool, flux_shift_value=None):
    """Update visibility and value of flux_shift for chroma from external calls"""
    field_refs = {"flux_shift": flux_shift_checkbox_ref}
    field_values = {}
    if flux_shift_value is not None:
        field_values["flux_shift"] = flux_shift_value

    _update_field_refs_visibility(field_refs, is_chroma, field_values)

def update_flux_fields_visibility(is_flux: bool, flux_shift_value=None, bypass_g_emb_value=None):
    """Update visibility and values of flux-specific fields (flux_shift, bypass_g_emb)."""
    field_refs = {
        "flux_shift": flux_shift_checkbox_ref,
        "bypass_g_emb": bypass_g_emb_checkbox_ref,
    }
    field_values = {}
    if flux_shift_value is not None:
        field_values["flux_shift"] = flux_shift_value
    if bypass_g_emb_value is not None:
        field_values["bypass_g_emb"] = bypass_g_emb_value

    _update_field_refs_visibility(field_refs, is_flux, field_values)

def update_lumina_fields_visibility(is_lumina: bool, lumina_shift_value=None):
    """Update visibility and value of lumina_shift for lumina/lumina_2 from external calls"""
    field_refs = {"lumina_shift": lumina_shift_checkbox_ref}
    field_values = {}
    if lumina_shift_value is not None:
        field_values["lumina_shift"] = lumina_shift_value

    _update_field_refs_visibility(field_refs, is_lumina, field_values)

def update_anima_fields_visibility(is_anima: bool, llm_adapter_lr_value=None):
    """Update visibility and value of llm_adapter_lr for anima from external calls"""
    field_refs = {"llm_adapter_lr": llm_adapter_lr_field_ref}
    field_values = {}
    if llm_adapter_lr_value is not None:
        field_values["llm_adapter_lr"] = str(llm_adapter_lr_value)

    _update_field_refs_visibility(field_refs, is_anima, field_values)

def update_sdxl_fields_visibility(
    is_sdxl: bool,
    v_pred_value=None,
    d_est_loss_value=None,
    min_snr_gamma_value=None,
    unet_lr_value=None,
    te1_lr_value=None,
    te2_lr_value=None,
    model_path_value=None,
    is_ltx2=False,  # For model_path visibility (shared by SDXL and LTX2/Musubi models)
):
    """Update visibility and values for SDXL-specific fields."""
    field_refs = {
        "v_pred": v_pred_checkbox_ref,
        "d_est_loss": d_est_loss_checkbox_ref,
        "min_snr_gamma": min_snr_gamma_field_ref,
        "unet_lr": unet_lr_field_ref,
        "te1_lr": te1_lr_field_ref,
        "te2_lr": te2_lr_field_ref,
        "model_path": model_path_field_ref,
    }
    field_values = {}
    if v_pred_value is not None:
        field_values["v_pred"] = v_pred_value
    if d_est_loss_value is not None:
        field_values["d_est_loss"] = d_est_loss_value
    if min_snr_gamma_value is not None:
        field_values["min_snr_gamma"] = str(min_snr_gamma_value)
    if unet_lr_value is not None:
        field_values["unet_lr"] = str(unet_lr_value)
    if te1_lr_value is not None:
        field_values["te1_lr"] = str(te1_lr_value)
    if te2_lr_value is not None:
        field_values["te2_lr"] = str(te2_lr_value)
    if model_path_value is not None:
        field_values["model_path"] = str(model_path_value)

    # model_path field is used by both SDXL and LTX2/Musubi models, so handle special case
    page_obj = None
    try:
        for field_name, ref in field_refs.items():
            if ref and ref.current:
                if field_name == "model_path":
                    ref.current.visible = (is_sdxl or is_ltx2)
                else:
                    ref.current.visible = is_sdxl

                # Set value if provided
                if field_name in field_values and field_values[field_name] is not None:
                    value = field_values[field_name]
                    # Handle checkboxes
                    if isinstance(ref.current, ft.Checkbox):
                        ref.current.value = _convert_bool_value(value)
                    else:
                        ref.current.value = value
                # Keep track of last valid page for batch update
                if ref.current.page:
                    page_obj = ref.current.page
    except Exception:
        pass

    # Update page once at the end
    if page_obj:
        try:
            page_obj.update()
        except Exception:
            pass

def update_wan22_ckpt_visibility(is_wan22: bool, ckpt_value=None):
    """Update visibility and value for wan22 ckpt_path field."""
    field_refs = {"ckpt_path": ckpt_path_field_ref}
    field_values = {}
    if ckpt_value is not None:
        field_values["ckpt_path"] = str(ckpt_value)

    _update_field_refs_visibility(field_refs, is_wan22, field_values)

def update_longcat_ckpt_visibility(is_longcat: bool, ckpt_value=None):
    """Update visibility and value for longcat ckpt_path field."""
    field_refs = {"ckpt_path": ckpt_path_field_ref}
    field_values = {}
    if ckpt_value is not None:
        field_values["ckpt_path"] = str(ckpt_value)

    _update_field_refs_visibility(field_refs, is_longcat, field_values)

def update_z_image_fields_visibility(is_z_image: bool, diffusion_model_value=None, vae_value=None, text_encoders_value=None, merge_adapters_value=None):
    """Update visibility and values for z_image specific fields."""
    field_refs = {
        "z_image_diffusion_model": z_image_diffusion_model_field_ref,
        "z_image_vae": z_image_vae_field_ref,
        "z_image_text_encoders": z_image_text_encoders_field_ref,
        "z_image_merge_adapters": z_image_merge_adapters_field_ref,
    }
    field_values = {}
    if diffusion_model_value is not None:
        field_values["z_image_diffusion_model"] = str(diffusion_model_value)
    if vae_value is not None:
        field_values["z_image_vae"] = str(vae_value)
    if text_encoders_value is not None:
        field_values["z_image_text_encoders"] = str(text_encoders_value)
    if merge_adapters_value is not None:
        field_values["z_image_merge_adapters"] = str(merge_adapters_value)

    _update_field_refs_visibility(field_refs, is_z_image, field_values)

    # Also update the row visibility
    if z_image_row_ref and z_image_row_ref.current:
        z_image_row_ref.current.visible = is_z_image
        if z_image_row_ref.current.page:
            z_image_row_ref.current.update()

def update_flux2_fields_visibility(is_flux2: bool, diffusion_model_value=None, vae_value=None, text_encoders_value=None, shift_value=None):
    """Update visibility and values for flux2 specific fields."""
    field_refs = {
        "diffusion_model": flux2_diffusion_model_field_ref,
        "vae": flux2_vae_field_ref,
        "text_encoders": flux2_text_encoders_field_ref,
        "shift": flux2_shift_field_ref,
    }
    field_values = {}
    if diffusion_model_value is not None:
        field_values["diffusion_model"] = str(diffusion_model_value)
    if vae_value is not None:
        field_values["vae"] = str(vae_value)
    if text_encoders_value is not None:
        field_values["text_encoders"] = str(text_encoders_value)
    if shift_value is not None:
        field_values["shift"] = str(shift_value)

    _update_field_refs_visibility(field_refs, is_flux2, field_values)

    # Also update the row visibility
    if flux2_row_ref and flux2_row_ref.current:
        flux2_row_ref.current.visible = is_flux2
        if flux2_row_ref.current.page:
            flux2_row_ref.current.update()

def update_musubi_fields_visibility(
    is_musubi: bool,
    ltx_mode_value=None,
    separate_audio_buckets_value=None,
    gradient_checkpointing_value=None,
    t_type_value=None
):
    """Update visibility and values for Musubi-specific fields."""
    field_refs = {
        "ltx_mode": ltx_mode_dropdown_ref,
        "separate_audio_buckets": separate_audio_buckets_checkbox_ref,
        "gradient_checkpointing": gradient_checkpointing_checkbox_ref,
        "8_bit_te": load_text_encoder_in_8bit_checkbox_ref,
        "nf4_te": nf4_te_checkbox_ref,
        "t_type": t_type_dropdown_ref,
        "flash_attn": flash_attn_checkbox_ref,
        "use_mask": use_mask_checkbox_ref,
    }
    field_values = {}
    if ltx_mode_value is not None:
        field_values["ltx_mode"] = ltx_mode_value
    if separate_audio_buckets_value is not None:
        field_values["separate_audio_buckets"] = separate_audio_buckets_value
    if gradient_checkpointing_value is not None:
        field_values["gradient_checkpointing"] = gradient_checkpointing_value
    if t_type_value is not None:
        field_values["t_type"] = t_type_value

    _update_field_refs_visibility(field_refs, is_musubi, field_values)

    # separate_audio_buckets is always visible for musubi (ignored if ltx_mode is not 'av')
    try:
        if separate_audio_buckets_checkbox_ref and separate_audio_buckets_checkbox_ref.current:
            separate_audio_buckets_checkbox_ref.current.visible = is_musubi
            if separate_audio_buckets_checkbox_ref.current.page:
                separate_audio_buckets_checkbox_ref.current.page.update()
    except Exception:
        pass


def on_optimizer_type_change(e, from_toml_load=False):
    """Handle optimizer type dropdown changes.

    Args:
        e: Event object
        from_toml_load: If True, skip value updates (values already loaded from TOML)
    """
    if not e.control or not e.data:
        return

    optimizer_type = e.data

    # Explicitly update the dropdown's value to ensure it's set correctly
    if e.control and hasattr(e.control, 'value'):
        e.control.value = optimizer_type

    show_prodigy = ofc.get_field_visibility(optimizer_type)
    prodigy_row = prodigy_row_ref.current

    if prodigy_row:
        prodigy_row.visible = show_prodigy
        prodigy_row.update()

    show_automagic = ofc.get_automagic_visibility(optimizer_type)
    automagic_row = automagic_row_ref.current

    if automagic_row:
        automagic_row.visible = show_automagic
        automagic_row.update()

    # Only update field values when user manually changes dropdown,
    # NOT when loading from TOML (values already set correctly)
    if from_toml_load:
        return

    # Update field values based on optimizer config defaults
    # Only update if field is still at initial UI default (not modified by user or TOML)
    def _update_if_at_default(ref, default_val, initial_defaults):
        if ref and ref.current:
            current = ref.current.value
            # Only update if current value is one of the initial UI defaults
            if str(current) in initial_defaults:
                ref.current.value = str(default_val)
                ref.current.update()

    # Initial UI defaults (what the UI shows before any changes)
    initial_lr_defaults = {"2e-5", "1.0", "", "None"}
    initial_betas_defaults = {"[0.9, 0.99]", "", "None"}
    initial_wd_defaults = {"0.01", "0.0", "", "None"}
    initial_eps_defaults = {"1e-8", "[1e-30, 0.001]", "", "None"}

    initial_prodigy_defaults = {"None", "", "1e-6", "1.0", "0.0"}
    initial_automagic_defaults = {"1e-7", "1e-3", "1e-6", "1.0", "False", "0.1", "", "None"}

    # Update standard fields only if at initial default
    _update_if_at_default(lr_field_ref, ofc.get_standard_field_default(optimizer_type, "lr"), initial_lr_defaults)
    _update_if_at_default(betas_field_ref, ofc.get_standard_field_default(optimizer_type, "betas"), initial_betas_defaults)
    _update_if_at_default(weight_decay_field_ref, ofc.get_standard_field_default(optimizer_type, "weight_decay"), initial_wd_defaults)
    _update_if_at_default(eps_field_ref, ofc.get_standard_field_default(optimizer_type, "eps"), initial_eps_defaults)

    # Update prodigy fields if prodigy is selected and at initial default
    if show_prodigy:
        _update_if_at_default(beta3_field_ref, ofc.get_prodigy_field_default(optimizer_type, "beta3"), initial_prodigy_defaults)
        _update_if_at_default(d0_field_ref, ofc.get_prodigy_field_default(optimizer_type, "d0"), initial_prodigy_defaults)
        _update_if_at_default(d_coef_field_ref, ofc.get_prodigy_field_default(optimizer_type, "d_coef"), initial_prodigy_defaults)
        _update_if_at_default(schedulefree_c_field_ref, ofc.get_prodigy_field_default(optimizer_type, "schedulefree_c"), initial_prodigy_defaults)

    # Update automagic fields if automagic is selected and at initial default
    if show_automagic:
        _update_if_at_default(min_lr_field_ref, ofc.get_automagic_field_default(optimizer_type, "min_lr"), initial_automagic_defaults)
        _update_if_at_default(max_lr_field_ref, ofc.get_automagic_field_default(optimizer_type, "max_lr"), initial_automagic_defaults)
        _update_if_at_default(lr_bump_field_ref, ofc.get_automagic_field_default(optimizer_type, "lr_bump"), initial_automagic_defaults)
        _update_if_at_default(clip_threshold_field_ref, ofc.get_automagic_field_default(optimizer_type, "clip_threshold"), initial_automagic_defaults)
        # Checkbox needs special handling
        if do_paramiter_swapping_field_ref and do_paramiter_swapping_field_ref.current:
            current = do_paramiter_swapping_field_ref.current.value
            if current in [False, None] or str(current) in initial_automagic_defaults:
                default_val = ofc.get_automagic_field_default(optimizer_type, "do_paramiter_swapping")
                do_paramiter_swapping_field_ref.current.value = _convert_bool_value(default_val)
                do_paramiter_swapping_field_ref.current.update()
        _update_if_at_default(paramiter_swapping_factor_field_ref, ofc.get_automagic_field_default(optimizer_type, "paramiter_swapping_factor"), initial_automagic_defaults)

    page = e.control.page if hasattr(e.control, 'page') else None
    if page:
        page.update()
