import flet as ft
from contextlib import contextmanager
# import yaml # Removed as hardcoded config data is reduced
from .._styles import create_textfield, create_dropdown, add_section_title # Import helper functions
from .training_dataset_block import get_training_dataset_page_content
from flet_app.ui.utils.utils_top_menu import TopBarUtils
from flet_app.settings import settings
from . import model_field_config as mfc
from .training_ltx2 import get_ltx2_training_settings

# Global references to access from outside the function
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
model_path_field_ref = ft.Ref[ft.TextField]()
load_checkpoint_field_ref = ft.Ref[ft.TextField]()
rank_field_ref = ft.Ref[ft.TextField]()
alpha_field_ref = ft.Ref[ft.TextField]()
dropout_field_ref = ft.Ref[ft.TextField]()
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
ckpt_path_wan22_field_ref = ft.Ref[ft.TextField]()
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
t5_path_field_ref = ft.Ref[ft.TextField]()
all_modules_checkbox_ref = ft.Ref[ft.Checkbox]()
video_attn_checkbox_ref = ft.Ref[ft.Checkbox]()
video_ff_checkbox_ref = ft.Ref[ft.Checkbox]()
audio_attn_checkbox_ref = ft.Ref[ft.Checkbox]()
audio_ff_checkbox_ref = ft.Ref[ft.Checkbox]()
cross_modal_attn_checkbox_ref = ft.Ref[ft.Checkbox]()
with_audio_checkbox_ref = ft.Ref[ft.Checkbox]()
checkpoint_row_ref = ft.Ref[ft.ResponsiveRow]()
ckpt_path_row_ref = ft.Ref[ft.ResponsiveRow]()
diffusers_row_ref = ft.Ref[ft.ResponsiveRow]()
single_file_row_ref = ft.Ref[ft.ResponsiveRow]()
transformer_full_row_ref = ft.Ref[ft.ResponsiveRow]()
byt5_row_ref = ft.Ref[ft.ResponsiveRow]()
t5_row_ref = ft.Ref[ft.ResponsiveRow]()
llama3_row_ref = ft.Ref[ft.ResponsiveRow]()
clip_row_ref = ft.Ref[ft.ResponsiveRow]()
text_encoder_row_ref = ft.Ref[ft.ResponsiveRow]()
dtype_dropdown_ref = ft.Ref[ft.Dropdown]()
timestep_sm_dropdown_ref = ft.Ref[ft.Dropdown]()
transformer_dtype_dropdown_ref = ft.Ref[ft.Dropdown]()
# LTX2-specific precision fields
mixed_precision_mode_dropdown_ref = ft.Ref[ft.Dropdown]()
quantization_dropdown_ref = ft.Ref[ft.Dropdown]()
load_text_encoder_in_8bit_checkbox_ref = ft.Ref[ft.Checkbox]()
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

# Section visibility refs for conditional UI
standard_training_section_ref = ft.Ref[ft.ResponsiveRow]()
standard_eval_optimizer_section_ref = ft.Ref[ft.ResponsiveRow]()
ltx2_custom_section_ref = ft.Ref[ft.Container]()

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
            "load_checkpoint": load_checkpoint_field_ref,
            "diffusers_path": diffusers_path_field_ref,
            "transformer_path": transformer_path_field_ref,
            "transformer_path_full": transformer_path_full_ref,
            "text_encoder_path": text_encoder_path_field_ref,
            "vae_path": vae_path_field_ref,
            "llm_path": llm_path_field_ref,
            "ckpt_path_wan22": ckpt_path_wan22_field_ref,
            "clip_path": clip_path_field_ref,
            "llama3_path": llama3_path_field_ref,
            "byt5_path": byt5_path_field_ref,
            "t5_path": t5_path_field_ref,
            "single_file_path": single_file_path_field_ref,
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
            "with_audio": with_audio_checkbox_ref,
            # SDXL-specific fields
            "v_pred": v_pred_checkbox_ref,
            "d_est_loss": d_est_loss_checkbox_ref,
            "min_snr_gamma": min_snr_gamma_field_ref,
            "unet_lr": unet_lr_field_ref,
            "te1_lr": te1_lr_field_ref,
            "te2_lr": te2_lr_field_ref,
            # LTX2-specific module checkboxes
            "audio_attn": audio_attn_checkbox_ref,
            "audio_ff": audio_ff_checkbox_ref,
            "video_attn": video_attn_checkbox_ref,
            "video_ff": video_ff_checkbox_ref,
            "cross_modal_attn": cross_modal_attn_checkbox_ref,
            "all_modules": all_modules_checkbox_ref,
            # LTX2-specific adapter fields
            "rank": rank_field_ref,
            "alpha": alpha_field_ref,
            "dropout": dropout_field_ref,
            "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
            # dtype, transformer_dtype, timestep_sm
            "dtype": dtype_dropdown_ref,
            "transformer_dtype": transformer_dtype_dropdown_ref,
            "timestep_sm": timestep_sm_dropdown_ref,
            # LTX2-specific precision fields
            "mixed_precision_mode": mixed_precision_mode_dropdown_ref,
            "quantization": quantization_dropdown_ref,
            "load_text_encoder_in_8bit": load_text_encoder_in_8bit_checkbox_ref,
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
            "load_checkpoint": load_checkpoint_field_ref,
            "diffusers_path": diffusers_path_field_ref,
            "transformer_path": transformer_path_field_ref,
            "transformer_path_full": transformer_path_full_ref,
            "text_encoder_path": text_encoder_path_field_ref,
            "vae_path": vae_path_field_ref,
            "llm_path": llm_path_field_ref,
            "ckpt_path_wan22": ckpt_path_wan22_field_ref,
            "clip_path": clip_path_field_ref,
            "llama3_path": llama3_path_field_ref,
            "byt5_path": byt5_path_field_ref,
            "t5_path": t5_path_field_ref,
            "single_file_path": single_file_path_field_ref,
            # Text fields
            "min_t": min_t_field_ref,
            "max_t": max_t_field_ref,
            "max_seq_len": max_seq_len_field_ref,
            "min_snr_gamma": min_snr_gamma_field_ref,
            "unet_lr": unet_lr_field_ref,
            "te1_lr": te1_lr_field_ref,
            "te2_lr": te2_lr_field_ref,
            "max_llama3_seq_len": max_llama3_seq_len_field_ref,
            # LTX2-specific adapter fields
            "rank": rank_field_ref,
            "alpha": alpha_field_ref,
            "dropout": dropout_field_ref,
            "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
            # Model-specific fields
            "z_image_diffusion_model": z_image_diffusion_model_field_ref,
            "z_image_vae": z_image_vae_field_ref,
            "z_image_text_encoders": z_image_text_encoders_field_ref,
            "z_image_merge_adapters": z_image_merge_adapters_field_ref,
            "flux2_diffusion_model": flux2_diffusion_model_field_ref,
            "flux2_vae": flux2_vae_field_ref,
            "flux2_text_encoders": flux2_text_encoders_field_ref,
            "flux2_shift": flux2_shift_field_ref,
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
            "with_audio": with_audio_checkbox_ref,
            # LTX2-specific module checkboxes
            "all_modules": all_modules_checkbox_ref,
            "video_attn": video_attn_checkbox_ref,
            "video_ff": video_ff_checkbox_ref,
            "audio_attn": audio_attn_checkbox_ref,
            "audio_ff": audio_ff_checkbox_ref,
            "cross_modal_attn": cross_modal_attn_checkbox_ref,
            # Z-image specific
            "z_image_diffusion_model_dtype_fp8": z_image_diffusion_model_dtype_checkbox_ref,
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

    def on_model_type_change(e):
        """Handle model type dropdown change to show/hide model-specific fields"""
        sel = model_type_dropdown_ref.current.value if model_type_dropdown_ref.current else None
        print(f"DEBUG on_model_type_change: sel={sel}")
        if not sel:
            return

        # 1. Normalize and Prep
        sel_norm = mfc.normalize_model_name(sel)
        model_key = mfc.get_model_key(sel_norm)
        skip_defaults = _suppress_model_defaults
        print(f"DEBUG on_model_type_change: sel_norm={sel_norm}, model_key={model_key}")

        # 2. Apply Field Visibility
        # We capture the visibility dict to determine if Rows should be hidden
        vis_config = _apply_field_visibility(sel_norm)

        # Ensure model_path and load_checkpoint are visible for both SDXL and LTX2
        is_ltx2 = sel_norm in ("ltx-video-2", "ltx2")
        if sel_norm == "sdxl" or is_ltx2:
            vis_config["model_path"] = True
            vis_config["load_checkpoint"] = True
            if model_path_field_ref and model_path_field_ref.current:
                model_path_field_ref.current.visible = True
            if load_checkpoint_field_ref and load_checkpoint_field_ref.current:
                load_checkpoint_field_ref.current.visible = True

        # 3. Dynamic Row Visibility
        # Map Rows to the "Main Field" they contain. If the field is visible, the row is visible.
        row_triggers = {
            checkpoint_row_ref: "model_path",
            ckpt_path_row_ref: "ckpt_path_wan22",
            diffusers_row_ref: "diffusers_path",
            single_file_row_ref: "single_file_path",
            transformer_full_row_ref: "transformer_path_full",
            byt5_row_ref: "byt5_path",
            t5_row_ref: "t5_path",
            llama3_row_ref: "llama3_path",
            clip_row_ref: "clip_path",
            text_encoder_row_ref: "text_encoder_path",
            flux2_row_ref: "flux2_diffusion_model",
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
                "ckpt_path_wan22": ckpt_path_wan22_field_ref,
                "clip_path": clip_path_field_ref,
                "llama3_path": llama3_path_field_ref,
                "max_llama3_seq_len": max_llama3_seq_len_field_ref,
                "byt5_path": byt5_path_field_ref,
                "single_file_path": single_file_path_field_ref,
                "t5_path": t5_path_field_ref,
                "model_path": model_path_field_ref,
                "load_checkpoint": load_checkpoint_field_ref,
                "hidream_4bit": hidream_4bit_checkbox_ref,
                "hidream_tdtype": hidream_tdtype_checkbox_ref,
                # LTX2-specific adapter fields
                "rank": rank_field_ref,
                "alpha": alpha_field_ref,
                "dropout": dropout_field_ref,
                "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
                # Flux2-specific fields
                "flux2_diffusion_model": flux2_diffusion_model_field_ref,
                "flux2_vae": flux2_vae_field_ref,
                "flux2_text_encoders": flux2_text_encoders_field_ref,
                "flux2_shift": flux2_shift_field_ref,
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

        # 7. Handle model-specific defaults not yet in config (model-specific field overrides)
        # Note: Most defaults are now in model_field_config.py and applied in step 5

        # 8. Dynamic Row Visibility (handled earlier)

        # 9. Conditional UI Swap for LTX2
        # Toggle standard training sections
        if standard_training_section_ref.current:
            standard_training_section_ref.current.visible = not is_ltx2
            print(f"DEBUG: Set standard_training_section visible={not is_ltx2}")
            if standard_training_section_ref.current.page:
                standard_training_section_ref.current.update()
        if standard_eval_optimizer_section_ref.current:
            standard_eval_optimizer_section_ref.current.visible = not is_ltx2
            print(f"DEBUG: Set standard_eval_optimizer_section visible={not is_ltx2}")
            if standard_eval_optimizer_section_ref.current.page:
                standard_eval_optimizer_section_ref.current.update()

        # Toggle LTX2 custom section
        if ltx2_custom_section_ref.current:
            ltx2_custom_section_ref.current.visible = is_ltx2
            print(f"DEBUG: Set ltx2_custom_section visible={is_ltx2}, is_ltx2={is_ltx2}, sel_norm={sel_norm}")
            # Force update the ltx2 section
            if ltx2_custom_section_ref.current.page:
                ltx2_custom_section_ref.current.update()
        else:
            print(f"DEBUG: ltx2_custom_section_ref.current is None!")

        # 10. Update adapter field visibility (hide old ones for LTX2, show new ones only for LTX2)
        ltx2_adapter_fields = {
            "rank": rank_field_ref,
            "alpha": alpha_field_ref,
            "dropout": dropout_field_ref,
            "first_frame_conditioning_p_ltx2": first_frame_conditioning_p_ltx2_field_ref,
        }
        old_adapter_fields = {
            "a_rank": a_rank_field_ref,
            "a_dtype": a_dtype_field_ref,
            "blocks_swap": blocks_swap_field_ref,
            "disable_bsfe": disable_bsfe_field_ref,
        }
        try:
            # Show new LTX2 fields only for LTX2
            for field_name, ref in ltx2_adapter_fields.items():
                if ref and ref.current:
                    ref.current.visible = is_ltx2
                    if ref.current.page:
                        ref.current.update()
            # Hide old adapter fields for LTX2
            for field_name, ref in old_adapter_fields.items():
                if ref and ref.current:
                    ref.current.visible = not is_ltx2
                    if ref.current.page:
                        ref.current.update()
        except Exception:
            pass

        # 11. Update page
        if e and getattr(e, 'page', None):
            e.page.update()
        else:
            print("DEBUG: Page update - e or e.page is None")

    page_controls = []

    # --- Model Configuration & Dataset Selection (Side by Side) ---
    def _on_save_data_config_click(e: ft.ControlEvent):
        try:
            ds_block = dataset_block
            selected = None
            if ds_block and hasattr(ds_block, 'get_selected_dataset'):
                selected = ds_block.get_selected_dataset()
            if not selected:
                if e and e.page:
                    e.page.snack_bar = ft.SnackBar(content=ft.Text("Select a dataset before saving."), open=True)
                    e.page.update()
                return

            from flet_app.ui.dataset_manager.dataset_utils import _get_dataset_base_dir
            import os
            import json
            try:
                import tomllib as _toml_reader  # Python 3.11+
            except Exception:  # pragma: no cover
                _toml_reader = None

            base_dir, dataset_type = _get_dataset_base_dir(selected)
            clean_dataset_name = str(selected)
            dataset_full_path = os.path.join(base_dir, clean_dataset_name)
            parent_dir = os.path.dirname(dataset_full_path)
            out_toml_path = os.path.join(parent_dir, f"{clean_dataset_name}.toml")

            # Build TOML string with basic data config structure
            toml_lines = []
            toml_lines.append("resolutions = [512]")
            toml_lines.append("")
            toml_lines.append("enable_ar_bucket = true")
            toml_lines.append("")
            toml_lines.append("# Min and max aspect ratios, given as width/height ratio.")
            toml_lines.append("min_ar = 0.5")
            toml_lines.append("max_ar = 2.0")
            toml_lines.append("ar_buckets = [[448, 576]]")
            toml_lines.append("")
            toml_lines.append("# Total number of aspect ratio buckets, evenly spaced (in log space) between min_ar and max_ar.")
            toml_lines.append("num_ar_buckets = 9")
            toml_lines.append("num_repeats = 1")
            toml_lines.append("")
            toml_lines.append("# Frame buckets (commented out by default)")
            toml_lines.append("# frame_buckets = [1, 33]")
            toml_lines.append("")
            toml_lines.append("[[directory]]")
            toml_lines.append("# The target images go in here. These are the images that the model will learn to produce.")
            # Use the selected dataset path
            dir_path_val = dataset_full_path
            # Escape backslashes minimally
            toml_lines.append(f"path = '{dir_path_val}'")

            os.makedirs(parent_dir, exist_ok=True)
            with open(out_toml_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(toml_lines) + "\n")

            if e and e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Saved: {out_toml_path}"), open=True)
                e.page.update()
        except Exception as ex:
            if e and e.page:
                e.page.snack_bar = ft.SnackBar(content=ft.Text(f"Error saving: {ex}"), open=True)
                e.page.update()

    # Add Save Configuration button to the dataset row (styled like Monitor)
    save_cfg_btn = ft.ElevatedButton(
        "Save Data Config",
        icon=ft.Icons.SAVE,
        on_click=_on_save_data_config_click,
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
                        # Non-LTX2 fields
                        create_dropdown(
                            "dtype",
                            "bfloat16",
                            {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"},
                            col=3, expand=True, scale=0.8, ref=dtype_dropdown_ref,
                            visible=_should_show_field("dtype")
                        ),
                        create_dropdown(
                            "transformer_dtype",
                            "float8",
                            {"float8": "float8", "None": "None"},
                            col=3, expand=True, scale=0.8, ref=transformer_dtype_dropdown_ref,
                            visible=_should_show_field("transformer_dtype")
                        ),
                        create_dropdown(
                            "timestep_sm",
                            "logit_normal",
                            {"logit_normal": "logit_normal", "uniform": "uniform", "None": "None"},
                            col=2, expand=True, scale=0.8, ref=timestep_sm_dropdown_ref,
                            visible=_should_show_field("timestep_sm")
                        ),
                        # LTX2-specific precision fields
                        create_dropdown(
                            "mixed_precision_mode",
                            "bf16",
                            {"no": "no", "fp16": "fp16", "bf16": "bf16"},
                            col=4, expand=True, scale=0.8, ref=mixed_precision_mode_dropdown_ref,
                            visible=_should_show_field("mixed_precision_mode")
                        ),
                        create_dropdown(
                            "quantization",
                            "int8-quanto",
                            {"null": "null", "int8-quanto": "int8-quanto", "int4-quanto": "int4-quanto", "int2-quanto": "int2-quanto", "fp8-quanto": "fp8-quanto", "fp8uz-quanto": "fp8uz-quanto"},
                            col=4, expand=True, scale=0.8, ref=quantization_dropdown_ref,
                            visible=_should_show_field("quantization")
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield(
                                "model_path",
                                "",
                                col=6, expand=True, ref=model_path_field_ref,
                                visible=_should_show_field("model_path")
                            ),
                            ft.Checkbox(
                                label="with_audio",
                                value=True,
                                scale=0.8,
                                ref=with_audio_checkbox_ref,
                                visible=_should_show_field("with_audio"),
                                col=6,
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
                                col=12, expand=True, ref=ckpt_path_wan22_field_ref,
                                visible=_should_show_field("ckpt_path_wan22")
                            ),
                        ],
                        ref=ckpt_path_row_ref,
                        visible=_should_show_field("ckpt_path_wan22")
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            create_textfield("text_encoder_path", "", col=6, expand=True, ref=text_encoder_path_field_ref, visible=_should_show_field("text_encoder_path")),
                            create_textfield("load_checkpoint", "", col=6, expand=True, ref=load_checkpoint_field_ref, visible=_should_show_field("load_checkpoint")),
                        ], spacing=2,
                        ref=text_encoder_row_ref,
                        visible=_should_show_field("text_encoder_path") or _should_show_field("load_checkpoint")
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
                        visible=_should_show_field("flux2_diffusion_model")
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
                    # LTX2-specific checkboxes
                    ft.ResponsiveRow(controls=[
                        ft.Checkbox(
                            label="all_modules",
                            value=False,
                            scale=0.8,
                            ref=all_modules_checkbox_ref,
                            visible=_should_show_field("all_modules"),
                            col=4,
                        ),
                        ft.Checkbox(
                            label="video_attn",
                            value=True,
                            scale=0.8,
                            ref=video_attn_checkbox_ref,
                            visible=_should_show_field("video_attn"),
                            col=4,
                        ),
                        ft.Checkbox(
                            label="video_ff",
                            value=False,
                            scale=0.8,
                            ref=video_ff_checkbox_ref,
                            visible=_should_show_field("video_ff"),
                            col=4,
                        ),
                    ], spacing=2),
                    ft.ResponsiveRow(controls=[
                        ft.Checkbox(
                            label="audio_attn",
                            value=False,
                            scale=0.8,
                            ref=audio_attn_checkbox_ref,
                            visible=_should_show_field("audio_attn"),
                            col=4,
                        ),
                        ft.Checkbox(
                            label="audio_ff",
                            value=False,
                            scale=0.8,
                            ref=audio_ff_checkbox_ref,
                            visible=_should_show_field("audio_ff"),
                            col=4,
                        ),
                        ft.Checkbox(
                            label="cross_modal_attn",
                            value=False,
                            scale=0.8,
                            ref=cross_modal_attn_checkbox_ref,
                            visible=_should_show_field("cross_modal_attn"),
                            col=4,
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
                        ft.Checkbox(
                            label="load_text_encoder_in_8bit",
                            value=False,
                            scale=0.8,
                            ref=load_text_encoder_in_8bit_checkbox_ref,
                            visible=_should_show_field("load_text_encoder_in_8bit"),
                            col=3,
                        ),
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
                        create_textfield("blocks_swap", 10, col=3, expand=True, ref=blocks_swap_field_ref, visible=not _should_show_field("rank")),
                        create_textfield("disable_bsfe", "true", col=3, expand=True, ref=disable_bsfe_field_ref, visible=not _should_show_field("rank")),
                    ], spacing=2),
                    # LTX2 specific adapter row
                    ft.ResponsiveRow(controls=[
                        create_textfield("rank", 32, col=3, expand=True, ref=rank_field_ref, visible=_should_show_field("rank")),
                        create_textfield("alpha", 32, col=3, expand=True, ref=alpha_field_ref, visible=_should_show_field("alpha")),
                        create_textfield("dropout", 0.0, col=3, expand=True, ref=dropout_field_ref, visible=_should_show_field("dropout")),
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

    # Determine initial visibility based on default model
    is_default_ltx2 = settings.train_def_model.lower() in ("ltx-video-2", "ltx2")
    standard_training_section.visible = not is_default_ltx2
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
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START, ref=standard_eval_optimizer_section_ref)

    standard_eval_optimizer_section.visible = not is_default_ltx2
    page_controls.append(standard_eval_optimizer_section)

    # --- LTX2 Custom Training Settings ---
    ltx2_custom_section = get_ltx2_training_settings(ref=ltx2_custom_section_ref)
    ltx2_custom_section.visible = is_default_ltx2
    page_controls.append(ltx2_custom_section)

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
    container.save_data_config_button = save_cfg_btn
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

def update_sdxl_fields_visibility(
    is_sdxl: bool,
    v_pred_value=None,
    d_est_loss_value=None,
    min_snr_gamma_value=None,
    unet_lr_value=None,
    te1_lr_value=None,
    te2_lr_value=None,
    model_path_value=None,
    is_ltx2=False,  # Added parameter to handle LTX2 model as well
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

    # model_path field is used by both SDXL and LTX2 models, so handle special case
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
    field_refs = {"ckpt_path_wan22": ckpt_path_wan22_field_ref}
    field_values = {}
    if ckpt_value is not None:
        field_values["ckpt_path_wan22"] = str(ckpt_value)

    _update_field_refs_visibility(field_refs, is_wan22, field_values)

def update_longcat_ckpt_visibility(is_longcat: bool, ckpt_value=None):
    """Update visibility and value for longcat ckpt_path field."""
    field_refs = {"ckpt_path_wan22": ckpt_path_wan22_field_ref}
    field_values = {}
    if ckpt_value is not None:
        field_values["ckpt_path_wan22"] = str(ckpt_value)

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

def update_flux2_fields_visibility(is_flux2: bool, diffusion_model_value=None, vae_value=None, text_encoders_value=None, shift_value=None):
    """Update visibility and values for flux2 specific fields."""
    field_refs = {
        "flux2_diffusion_model": flux2_diffusion_model_field_ref,
        "flux2_vae": flux2_vae_field_ref,
        "flux2_text_encoders": flux2_text_encoders_field_ref,
        "flux2_shift": flux2_shift_field_ref,
    }
    field_values = {}
    if diffusion_model_value is not None:
        field_values["flux2_diffusion_model"] = str(diffusion_model_value)
    if vae_value is not None:
        field_values["flux2_vae"] = str(vae_value)
    if text_encoders_value is not None:
        field_values["flux2_text_encoders"] = str(text_encoders_value)
    if shift_value is not None:
        field_values["flux2_shift"] = str(shift_value)

    _update_field_refs_visibility(field_refs, is_flux2, field_values)

def update_ltx2_fields_visibility(
    is_ltx2: bool,
    all_modules_value=None,
    video_attn_value=None,
    video_ff_value=None,
    audio_attn_value=None,
    audio_ff_value=None,
    cross_modal_attn_value=None
):
    """Update visibility and values for LTX2-specific fields."""
    field_refs = {
        "all_modules": all_modules_checkbox_ref,
        "video_attn": video_attn_checkbox_ref,
        "video_ff": video_ff_checkbox_ref,
        "audio_attn": audio_attn_checkbox_ref,
        "audio_ff": audio_ff_checkbox_ref,
        "cross_modal_attn": cross_modal_attn_checkbox_ref,
    }
    field_values = {}
    if all_modules_value is not None:
        field_values["all_modules"] = all_modules_value
    if video_attn_value is not None:
        field_values["video_attn"] = video_attn_value
    if video_ff_value is not None:
        field_values["video_ff"] = video_ff_value
    if audio_attn_value is not None:
        field_values["audio_attn"] = audio_attn_value
    if audio_ff_value is not None:
        field_values["audio_ff"] = audio_ff_value
    if cross_modal_attn_value is not None:
        field_values["cross_modal_attn"] = cross_modal_attn_value

    _update_field_refs_visibility(field_refs, is_ltx2, field_values)
