import flet as ft


def append_model_specific_lines(lines, get_value, model_type: str):
    """Append model-specific [model] lines to TOML output.
    Includes optional paths and per-model parameters.
    - get_value: callable(name, default) -> value
    - model_type: normalized string value of Model Type
    """
    mt = (str(model_type) if model_type is not None else '').strip().lower()

    # Import the path expansion function
    try:
        from .config_utils import expand_model_path
    except ImportError:
        # Fallback if import fails
        def expand_model_path(path): return path

    # Optional additional model paths (expand to absolute paths)
    te_path = get_value('text_encoder_path', None)
    if te_path and str(te_path).strip():
        expanded_te_path = expand_model_path(str(te_path))
        lines.append(f"text_encoder_path   = '{expanded_te_path}'")
    vae_path = get_value('vae_path', None)
    if vae_path and str(vae_path).strip():
        expanded_vae_path = expand_model_path(str(vae_path))
        lines.append(f"vae_path   = '{expanded_vae_path}'")
    llm_path = get_value('llm_path', None)
    if llm_path and str(llm_path).strip():
        expanded_llm_path = expand_model_path(str(llm_path))
        lines.append(f"llm_path   = '{expanded_llm_path}'")
    byt5_path = get_value('byt5_path', None)
    if byt5_path and str(byt5_path).strip():
        expanded_byt5_path = expand_model_path(str(byt5_path))
        lines.append(f"byt5_path   = '{expanded_byt5_path}'")
    llama3_path = get_value('llama3_path', None)
    if llama3_path and str(llama3_path).strip():
        expanded_llama3_path = expand_model_path(str(llama3_path))
        lines.append(f"llama3_path   = '{expanded_llama3_path}'")
    clip_path = get_value('clip_path', None)
    if clip_path and str(clip_path).strip():
        expanded_clip_path = expand_model_path(str(clip_path))
        lines.append(f"clip_path   = '{expanded_clip_path}'")

    def _has(v):
        return v is not None and str(v).strip() != ''

    # wan22: save as wan in TOML, but include extras only if provided
    if mt == 'wan22':
        min_t_val = get_value('min_t', None)
        max_t_val = get_value('max_t', None)
        if _has(min_t_val):
            lines.append(f"min_t = {min_t_val}")
        if _has(max_t_val):
            lines.append(f"max_t = {max_t_val}")
        ckpt = get_value('ckpt_path', None)
        if _has(ckpt):
            expanded_ckpt_path = expand_model_path(str(ckpt))
            lines.append(f"ckpt_path = '{expanded_ckpt_path}'")
    # hunyuan-video: include ckpt_path when provided
    if mt == 'hunyuan-video':
        ckpt = get_value('ckpt_path', None)
        if _has(ckpt):
            expanded_ckpt_path = expand_model_path(str(ckpt))
            lines.append(f"ckpt_path = '{expanded_ckpt_path}'")
    # longcat: include ckpt_path when provided
    if mt == 'longcat':
        ckpt = get_value('ckpt_path', None)
        if _has(ckpt):
            expanded_ckpt_path = expand_model_path(str(ckpt))
            lines.append(f"ckpt_path = '{expanded_ckpt_path}'")

    # auraflow
    if mt == 'auraflow':
        msl = get_value('max_sequence_length', None)
        if _has(msl):
            lines.append(f"max_sequence_length = {msl}")

    # ltx-video
    if mt in ('ltx-video', 'ltx'):
        ffc = get_value('first_frame_conditioning_p', None)
        if _has(ffc):
            lines.append(f"first_frame_conditioning_p = {ffc}")

    # chroma
    if mt == 'chroma':
        fs = get_value('flux_shift', True)
        fs_bool = str(fs).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(fs, bool) else fs
        lines.append(f"flux_shift = {'true' if fs_bool else 'false'}")

    # flux
    if mt == 'flux':
        fs = get_value('flux_shift', True)
        fs_bool = str(fs).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(fs, bool) else fs
        lines.append(f"flux_shift = {'true' if fs_bool else 'false'}")
        bge = get_value('bypass_g_emb', True)
        bge_bool = str(bge).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(bge, bool) else bge
        lines.append(f"bypass_guidance_embedding = {'true' if bge_bool else 'false'}")

    # hidream
    if mt == 'hidream':
        fs = get_value('flux_shift', True)
        fs_bool = str(fs).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(fs, bool) else fs
        lines.append(f"flux_shift = {'true' if fs_bool else 'false'}")
        l4b = get_value('llama3_4bit', True)
        l4b_bool = str(l4b).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(l4b, bool) else l4b
        lines.append(f"llama3_4bit = {'true' if l4b_bool else 'false'}")
        mlsl = get_value('max_llama3_sequence_length', None)
        if _has(mlsl):
            lines.append(f"max_llama3_sequence_length = {mlsl}")

    # sd3
    if mt == 'sd3':
        fs = get_value('flux_shift', True)
        fs_bool = str(fs).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(fs, bool) else fs
        lines.append(f"flux_shift = {'true' if fs_bool else 'false'}")

    # lumina / lumina_2
    if mt in ('lumina', 'lumina_2'):
        ls = get_value('lumina_shift', True)
        ls_bool = str(ls).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(ls, bool) else ls
        lines.append(f"lumina_shift = {'true' if ls_bool else 'false'}")

    # sdxl
    if mt == 'sdxl':
        vp = get_value('v_pred', None)
        vp_bool = str(vp).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(vp, bool) else vp
        if vp is not None:
            lines.append(f"v_pred = {'true' if vp_bool else 'false'}")
        deloss = get_value('d_est_loss', None)
        deloss_bool = str(deloss).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(deloss, bool) else deloss
        if deloss is not None:
            lines.append(f"debiased_estimation_loss = {'true' if deloss_bool else 'false'}")
        mng = get_value('min_snr_gamma', None)
        if _has(mng):
            lines.append(f"min_snr_gamma = {mng}")
        ulr = get_value('unet_lr', None)
        if _has(ulr):
            lines.append(f"unet_lr = {ulr}")
        te1 = get_value('text_encoder_1_lr', None)
        if _has(te1):
            lines.append(f"text_encoder_1_lr = {te1}")
        te2 = get_value('text_encoder_2_lr', None)
        if _has(te2):
            lines.append(f"text_encoder_2_lr = {te2}")


def populate_label_vals_from_model(model_dict: dict, label_vals: dict) -> str:
    """Populate label_vals from [model] dict and return normalized model_type string.
    Handles conversion wan->wan22 when min/max present and propagates model-specific fields.
    """
    model_type = model_dict.get('type') if isinstance(model_dict, dict) else None
    mt = (str(model_type) if model_type is not None else '').strip()
    mt_lower = mt.lower()

    # Import the path collapse function
    try:
        from .config_utils import collapse_model_path
    except ImportError:
        # Fallback if import fails
        def collapse_model_path(path): return path

    if mt_lower == 'wan' and isinstance(model_dict, dict) and ('min_t' in model_dict or 'max_t' in model_dict):
        mt = 'wan22'
        mt_lower = 'wan22'

    # Copy common optional paths (collapse to relative paths for UI)
    if isinstance(model_dict, dict):
        if 'diffusers_path' in model_dict:
            label_vals['diffusers_path'] = collapse_model_path(model_dict.get('diffusers_path'))
        if 'transformer_path' in model_dict:
            label_vals['transformer_path'] = collapse_model_path(model_dict.get('transformer_path'))
        if 'text_encoder_path' in model_dict:
            label_vals['text_encoder_path'] = collapse_model_path(model_dict.get('text_encoder_path'))
        if 'vae_path' in model_dict:
            label_vals['vae_path'] = collapse_model_path(model_dict.get('vae_path'))
        if 'llm_path' in model_dict:
            label_vals['llm_path'] = collapse_model_path(model_dict.get('llm_path'))

    # Per-model specifics
    if mt_lower == 'wan22':
        if 'min_t' in model_dict:
            label_vals['min_t'] = model_dict.get('min_t')
        if 'max_t' in model_dict:
            label_vals['max_t'] = model_dict.get('max_t')
        if 'ckpt_path' in model_dict:
            label_vals['ckpt_path'] = collapse_model_path(model_dict.get('ckpt_path'))
    elif mt_lower == 'auraflow':
        if 'max_sequence_length' in model_dict:
            label_vals['max_sequence_length'] = model_dict.get('max_sequence_length')
    elif mt_lower == 'chroma':
        if 'flux_shift' in model_dict:
            label_vals['flux_shift'] = model_dict.get('flux_shift')
    elif mt_lower == 'flux':
        if 'flux_shift' in model_dict:
            label_vals['flux_shift'] = model_dict.get('flux_shift')
        if 'bypass_guidance_embedding' in model_dict:
            label_vals['bypass_g_emb'] = model_dict.get('bypass_guidance_embedding')
    elif mt_lower == 'sd3':
        if 'flux_shift' in model_dict:
            label_vals['flux_shift'] = model_dict.get('flux_shift')
    elif mt_lower in ('ltx-video', 'ltx'):
        if 'first_frame_conditioning_p' in model_dict:
            label_vals['first_frame_conditioning_p'] = model_dict.get('first_frame_conditioning_p')
    elif mt_lower in ('lumina', 'lumina_2'):
        if 'lumina_shift' in model_dict:
            label_vals['lumina_shift'] = model_dict.get('lumina_shift')
    elif mt_lower == 'sdxl':
        if 'v_pred' in model_dict:
            label_vals['v_pred'] = model_dict.get('v_pred')
        if 'debiased_estimation_loss' in model_dict:
            label_vals['d_est_loss'] = model_dict.get('debiased_estimation_loss')
        if 'checkpoint_path' in model_dict:
            label_vals['checkpoint_path'] = collapse_model_path(model_dict.get('checkpoint_path'))
        for k in ('min_snr_gamma', 'unet_lr', 'text_encoder_1_lr', 'text_encoder_2_lr'):
            if k in model_dict:
                label_vals[k] = model_dict.get(k)
    elif mt_lower == 'longcat':
        if 'ckpt_path' in model_dict:
            label_vals['ckpt_path'] = collapse_model_path(model_dict.get('ckpt_path'))

    return mt


def postprocess_visibility_after_apply(label_vals: dict, page: ft.Page, model_type_dropdown_ref):
    """After values are applied to controls, ensure visibility for model-specific fields
    and set values where provided.
    """
    try:
        from flet_app.ui.pages.training_config import (
            update_wan_fields_visibility,
            update_auraflow_fields_visibility,
            update_chroma_fields_visibility,
            update_flux_fields_visibility,
            update_ltx_fields_visibility,
        )
    except Exception:
        return

    def _bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ('1', 'true', 'yes', 'on')

    is_wan22 = is_auraflow = is_chroma = is_flux = is_sd3 = is_ltx = is_lumina = is_sdxl = is_longcat = False
    try:
        mt = str(label_vals.get('Model Type', '')).strip().lower()
        is_wan22 = (mt == 'wan22')
        is_auraflow = (mt == 'auraflow')
        is_chroma = (mt == 'chroma')
        is_flux = (mt == 'flux')
        is_sd3 = (mt == 'sd3')
        is_ltx = (mt in ('ltx-video', 'ltx'))
        is_lumina = (mt in ('lumina', 'lumina_2'))
        is_sdxl = (mt == 'sdxl')
        is_longcat = (mt == 'longcat')
    except Exception:
        pass

    try:
        if getattr(model_type_dropdown_ref, 'current', None) and getattr(model_type_dropdown_ref.current, 'value', None):
            curv = str(model_type_dropdown_ref.current.value).strip().lower()
            is_wan22 = is_wan22 or (curv == 'wan22')
            is_auraflow = is_auraflow or (curv == 'auraflow')
            is_chroma = is_chroma or (curv == 'chroma')
            is_flux = is_flux or (curv == 'flux')
            is_sd3 = is_sd3 or (curv == 'sd3')
            is_ltx = is_ltx or (curv in ('ltx-video', 'ltx'))
            is_lumina = is_lumina or (curv in ('lumina', 'lumina_2'))
            is_sdxl = is_sdxl or (curv == 'sdxl')
            is_longcat = is_longcat or (curv == 'longcat')
    except Exception:
        pass

    # Apply visibility + any provided values
    update_wan_fields_visibility(is_wan22, label_vals.get('min_t'), label_vals.get('max_t'))
    try:
        from flet_app.ui.pages.training_config import update_wan22_ckpt_visibility, update_longcat_ckpt_visibility
        update_wan22_ckpt_visibility(is_wan22, label_vals.get('ckpt_path'))
        update_longcat_ckpt_visibility(is_longcat, label_vals.get('ckpt_path'))
    except Exception:
        pass
    update_auraflow_fields_visibility(is_auraflow, label_vals.get('max_sequence_length'))
    update_chroma_fields_visibility(is_chroma or is_sd3, label_vals.get('flux_shift'))
    update_flux_fields_visibility(is_flux, label_vals.get('flux_shift'), label_vals.get('bypass_g_emb'))
    update_ltx_fields_visibility(is_ltx, label_vals.get('first_frame_conditioning_p'))
    try:
        from flet_app.ui.pages.training_config import update_lumina_fields_visibility
        update_lumina_fields_visibility(is_lumina, label_vals.get('lumina_shift'))
    except Exception:
        pass
    try:
        from flet_app.ui.pages.training_config import update_sdxl_fields_visibility
        update_sdxl_fields_visibility(
            is_sdxl,
            label_vals.get('v_pred'),
            label_vals.get('d_est_loss'),
            label_vals.get('min_snr_gamma'),
            label_vals.get('unet_lr'),
            label_vals.get('text_encoder_1_lr'),
            label_vals.get('text_encoder_2_lr'),
            label_vals.get('checkpoint_path'),
        )
    except Exception:
        pass
