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

    # ltx-video (but not ltx-video-2 which uses the new format)
    if mt in ('ltx-video', 'ltx'):
        ffc = get_value('first_frame_conditioning_p', None)
        if _has(ffc):
            lines.append(f"first_frame_conditioning_p = {ffc}")

    # Also add checkpoint_path and diffusers_path for LTX models
    if mt in ('ltx-video', 'ltx', 'ltx-video-2'):
        checkpoint_path = get_value('checkpoint_path', None)
        if checkpoint_path and str(checkpoint_path).strip():
            expanded_checkpoint_path = expand_model_path(str(checkpoint_path))
            lines.append(f"checkpoint_path = '{expanded_checkpoint_path}'")

        diffusers_path = get_value('diffusers_path', None)
        if diffusers_path and str(diffusers_path).strip():
            expanded_diffusers_path = expand_model_path(str(diffusers_path))
            lines.append(f"diffusers_path = '{expanded_diffusers_path}'")

        # Add LTX2-specific adapter targets for LoRA
        def _to_bool(val):
            if isinstance(val, bool):
                return val
            s = str(val).strip().lower()
            return s in ('1', 'true', 'yes', 'on')

        if mt == 'ltx-video-2':
            # Add adapter.lora_targets section for LTX2
            all_modules = get_value('all_modules', False)
            video_attn = get_value('video_attn', True)
            video_ff = get_value('video_ff', False)
            audio_attn = get_value('audio_attn', False)
            audio_ff = get_value('audio_ff', False)
            cross_modal_attn = get_value('cross_modal_attn', False)

            lines.append("")
            lines.append("[adapter.lora_targets]")
            lines.append(f"all_modules = {'true' if _to_bool(all_modules) else 'false'}")
            lines.append(f"video_attn = {'true' if _to_bool(video_attn) else 'false'}")
            lines.append(f"video_ff = {'true' if _to_bool(video_ff) else 'false'}")
            lines.append(f"audio_attn = {'true' if _to_bool(audio_attn) else 'false'}")
            lines.append(f"audio_ff = {'true' if _to_bool(audio_ff) else 'false'}")
            lines.append(f"cross_modal_attn = {'true' if _to_bool(cross_modal_attn) else 'false'}")

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

    # flux2 and klein variants
    if mt in ('flux2', 'flux2_klein_4b', 'flux2_klein_9b'):
        # diffusion_model, vae, text_encoders, shift
        diffusion_model = get_value('diffusion_model', None)
        if diffusion_model and str(diffusion_model).strip():
            expanded_dm_path = expand_model_path(str(diffusion_model))
            lines.append(f"diffusion_model = '{expanded_dm_path}'")
        vae = get_value('vae', None)
        if vae and str(vae).strip():
            expanded_vae = expand_model_path(str(vae))
            lines.append(f"vae = '{expanded_vae}'")
        text_encoders = get_value('text_encoders', None)
        if text_encoders and str(text_encoders).strip():
            expanded_te = expand_model_path(str(text_encoders))
            # Save text_encoders as a list with path and type, like z_image does
            lines.append(f"text_encoders = [")
            lines.append(f"    {{path = '{expanded_te}', type = 'flux2'}}")
            lines.append(f"]")
        shift = get_value('shift', None)
        if _has(shift):
            lines.append(f"shift = {shift}")

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

    # z_image - uses custom field names
    if mt == 'z_image':
        diffusion_model = get_value('diffusion_model', None)
        if diffusion_model and str(diffusion_model).strip():
            expanded_dm_path = expand_model_path(str(diffusion_model))
            lines.append(f"diffusion_model = '{expanded_dm_path}'")
        # diffusion_model_dtype checkbox - always add the line, comment if unchecked
        dm_dtype = get_value('diffusion_model_dtype', False)
        dm_dtype_bool = str(dm_dtype).strip().lower() in ('1', 'true', 'yes', 'on') if not isinstance(dm_dtype, bool) else dm_dtype
        if dm_dtype_bool:
            lines.append(f"diffusion_model_dtype = 'float8'")
        else:
            lines.append(f"#diffusion_model_dtype = 'float8'")
        vae = get_value('vae', None)
        if vae and str(vae).strip():
            expanded_vae = expand_model_path(str(vae))
            lines.append(f"vae = '{expanded_vae}'")
        text_encoders = get_value('text_encoders', None)
        if text_encoders and str(text_encoders).strip():
            expanded_te = expand_model_path(str(text_encoders))
            # Save text_encoders as a list with path and type
            lines.append(f"text_encoders = [")
            lines.append(f"    {{path = '{expanded_te}', type = 'lumina2'}}")
            lines.append(f"]")
        merge_adapters = get_value('merge_adapters', None)
        if merge_adapters and str(merge_adapters).strip():
            expanded_ma = expand_model_path(str(merge_adapters))
            # Save merge_adapters as a list in TOML
            lines.append(f"merge_adapters = ['{expanded_ma}']")

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

    # Add with_audio for both SDXL and LTX2 models
    def _to_bool(val):
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ('1', 'true', 'yes', 'on')

    if mt in ('sdxl', 'ltx-video-2'):
        with_audio = get_value('with_audio', True)
        lines.append(f"with_audio = {'true' if _to_bool(with_audio) else 'false'}")


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

    # Handle wan22 detection for both legacy and new configs
    if mt_lower == 'wan22':
        # New configs with explicit type = 'wan22'
        mt = 'wan22'
        mt_lower = 'wan22'
    elif mt_lower == 'wan' and isinstance(model_dict, dict) and ('min_t' in model_dict or 'max_t' in model_dict):
        # Legacy configs: convert 'wan' to 'wan22' when min/max_t present
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
    elif mt_lower in ('flux2', 'flux2_klein_4b', 'flux2_klein_9b'):
        # flux2 uses custom field names similar to z_image
        if 'diffusion_model' in model_dict:
            label_vals['diffusion_model'] = collapse_model_path(model_dict.get('diffusion_model'))
        if 'vae' in model_dict:
            label_vals['vae'] = collapse_model_path(model_dict.get('vae'))
        if 'text_encoders' in model_dict:
            # text_encoders is stored as a list with {path, type}, extract path for UI (like z_image)
            text_encoders_val = model_dict.get('text_encoders')
            if isinstance(text_encoders_val, list) and len(text_encoders_val) > 0:
                first_encoder = text_encoders_val[0]
                if isinstance(first_encoder, dict) and 'path' in first_encoder:
                    label_vals['text_encoders'] = collapse_model_path(first_encoder['path'])
                elif isinstance(first_encoder, str):
                    # Fallback if it's a simple string in the list
                    label_vals['text_encoders'] = collapse_model_path(first_encoder)
            elif isinstance(text_encoders_val, str):
                # Fallback if it's a string (for backwards compatibility)
                label_vals['text_encoders'] = collapse_model_path(text_encoders_val)
        if 'shift' in model_dict:
            label_vals['shift'] = model_dict.get('shift')
    elif mt_lower == 'sd3':
        if 'flux_shift' in model_dict:
            label_vals['flux_shift'] = model_dict.get('flux_shift')
    elif mt_lower in ('ltx-video', 'ltx', 'ltx-video-2'):
        # Skip old first_frame_conditioning_p for LTX2 (uses new format)
        if mt_lower != 'ltx-video-2' and 'first_frame_conditioning_p' in model_dict:
            label_vals['first_frame_conditioning_p'] = model_dict.get('first_frame_conditioning_p')
        if 'checkpoint_path' in model_dict:
            label_vals['checkpoint_path'] = collapse_model_path(model_dict.get('checkpoint_path'))
        if 'diffusers_path' in model_dict:
            label_vals['diffusers_path'] = collapse_model_path(model_dict.get('diffusers_path'))

        # Handle LTX2-specific adapter.lora_targets fields
        if mt_lower == 'ltx-video-2' and 'adapter' in model_dict:
            adapter_dict = model_dict['adapter']
            if isinstance(adapter_dict, dict) and 'lora_targets' in adapter_dict:
                lora_targets = adapter_dict['lora_targets']
                if isinstance(lora_targets, dict):
                    if 'all_modules' in lora_targets:
                        label_vals['all_modules'] = lora_targets['all_modules']
                    if 'video_attn' in lora_targets:
                        label_vals['video_attn'] = lora_targets['video_attn']
                    if 'video_ff' in lora_targets:
                        label_vals['video_ff'] = lora_targets['video_ff']
                    if 'audio_attn' in lora_targets:
                        label_vals['audio_attn'] = lora_targets['audio_attn']
                    if 'audio_ff' in lora_targets:
                        label_vals['audio_ff'] = lora_targets['audio_ff']
                    if 'cross_modal_attn' in lora_targets:
                        label_vals['cross_modal_attn'] = lora_targets['cross_modal_attn']
    elif mt_lower in ('lumina', 'lumina_2'):
        if 'lumina_shift' in model_dict:
            label_vals['lumina_shift'] = model_dict.get('lumina_shift')
    elif mt_lower == 'z_image':
        # z_image uses custom field names
        if 'diffusion_model' in model_dict:
            label_vals['diffusion_model'] = collapse_model_path(model_dict.get('diffusion_model'))
        if 'diffusion_model_dtype' in model_dict:
            # Check if diffusion_model_dtype is set to 'float8'
            dm_dtype_val = model_dict.get('diffusion_model_dtype')
            label_vals['diffusion_model_dtype'] = (str(dm_dtype_val).strip().lower() == 'float8')
        if 'vae' in model_dict:
            label_vals['vae'] = collapse_model_path(model_dict.get('vae'))
        if 'text_encoders' in model_dict:
            # text_encoders is stored as a list with {path, type}, extract path for UI
            text_encoders_val = model_dict.get('text_encoders')
            if isinstance(text_encoders_val, list) and len(text_encoders_val) > 0:
                first_encoder = text_encoders_val[0]
                if isinstance(first_encoder, dict) and 'path' in first_encoder:
                    label_vals['text_encoders'] = collapse_model_path(first_encoder['path'])
                elif isinstance(first_encoder, str):
                    # Fallback if it's a simple string in the list
                    label_vals['text_encoders'] = collapse_model_path(first_encoder)
            elif isinstance(text_encoders_val, str):
                # Fallback if it's a string (for backwards compatibility)
                label_vals['text_encoders'] = collapse_model_path(text_encoders_val)
        if 'merge_adapters' in model_dict:
            # merge_adapters is stored as a list in TOML, extract first item for UI
            merge_adapters_val = model_dict.get('merge_adapters')
            if isinstance(merge_adapters_val, list) and len(merge_adapters_val) > 0:
                label_vals['merge_adapters'] = collapse_model_path(merge_adapters_val[0])
            elif isinstance(merge_adapters_val, str):
                # Fallback if it's a string (for backwards compatibility)
                label_vals['merge_adapters'] = collapse_model_path(merge_adapters_val)
    elif mt_lower in ('sdxl', 'ltx-video-2'):
        if 'v_pred' in model_dict:
            label_vals['v_pred'] = model_dict.get('v_pred')
        if 'debiased_estimation_loss' in model_dict:
            label_vals['d_est_loss'] = model_dict.get('debiased_estimation_loss')
        if 'checkpoint_path' in model_dict:
            label_vals['checkpoint_path'] = collapse_model_path(model_dict.get('checkpoint_path'))
        if 'with_audio' in model_dict:
            label_vals['with_audio'] = model_dict.get('with_audio')
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
        )
    except Exception:
        return

    def _bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ('1', 'true', 'yes', 'on')

    is_wan22 = is_auraflow = is_chroma = is_flux = is_flux2 = is_sd3 = is_ltx = is_ltx2 = is_lumina = is_sdxl = is_longcat = is_hunyuan_video = is_wan = is_z_image = False
    try:
        mt = str(label_vals.get('Model Type', '')).strip().lower()
        is_wan22 = (mt == 'wan22')
        is_auraflow = (mt == 'auraflow')
        is_chroma = (mt == 'chroma')
        is_flux = (mt == 'flux')
        is_flux2 = (mt in ('flux2', 'flux2_klein_4b', 'flux2_klein_9b'))
        is_sd3 = (mt == 'sd3')
        is_ltx = (mt in ('ltx-video', 'ltx', 'ltx-video-2'))
        is_ltx2 = (mt == 'ltx-video-2')
        is_lumina = (mt in ('lumina', 'lumina_2'))
        is_z_image = (mt == 'z_image')
        is_sdxl = (mt == 'sdxl')
        is_longcat = (mt == 'longcat')
        is_hunyuan_video = (mt == 'hunyuan-video')
        is_wan = (mt == 'wan')
    except Exception:
        pass

    try:
        if getattr(model_type_dropdown_ref, 'current', None) and getattr(model_type_dropdown_ref.current, 'value', None):
            curv = str(model_type_dropdown_ref.current.value).strip().lower()
            is_wan22 = is_wan22 or (curv == 'wan22')
            is_auraflow = is_auraflow or (curv == 'auraflow')
            is_chroma = is_chroma or (curv == 'chroma')
            is_flux = is_flux or (curv == 'flux')
            is_flux2 = is_flux2 or (curv in ('flux2', 'flux2_klein_4b', 'flux2_klein_9b'))
            is_sd3 = is_sd3 or (curv == 'sd3')
            is_ltx = is_ltx or (curv in ('ltx-video', 'ltx', 'ltx-video-2'))
            is_ltx2 = is_ltx2 or (curv == 'ltx-video-2')
            is_lumina = is_lumina or (curv in ('lumina', 'lumina_2'))
            is_z_image = is_z_image or (curv == 'z_image')
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

        # Also control the ckpt_path_row_ref visibility (same logic as on_model_type_change)
        from flet_app.ui.pages.training_config import ckpt_path_row_ref, ckpt_path_wan22_field_ref
        if ckpt_path_row_ref.current:
            should_be_visible = (is_wan22 or is_longcat or is_wan or is_hunyuan_video)
            ckpt_path_row_ref.current.visible = should_be_visible

            # Also ensure field-level visibility is set
            if ckpt_path_wan22_field_ref.current:
                ckpt_path_wan22_field_ref.current.visible = should_be_visible

            if ckpt_path_row_ref.current.page:
                ckpt_path_row_ref.current.page.update()
    except Exception:
        pass
    update_auraflow_fields_visibility(is_auraflow, label_vals.get('max_sequence_length'))
    update_chroma_fields_visibility(is_chroma or is_sd3, label_vals.get('flux_shift'))
    update_flux_fields_visibility(is_flux, label_vals.get('flux_shift'), label_vals.get('bypass_g_emb'))
    try:
        from flet_app.ui.pages.training_config import update_ltx2_fields_visibility
        update_ltx2_fields_visibility(
            is_ltx2,
            label_vals.get('all_modules'),
            label_vals.get('video_attn'),
            label_vals.get('video_ff'),
            label_vals.get('audio_attn'),
            label_vals.get('audio_ff'),
            label_vals.get('cross_modal_attn')
        )
    except Exception:
        pass
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
            is_ltx2=is_ltx2,  # Pass the LTX2 flag to handle checkpoint_path visibility
        )
    except Exception:
        pass

    # Update with_audio checkbox if it exists (only if value is explicitly provided)
    try:
        from flet_app.ui.pages.training_config import with_audio_checkbox_ref
        if with_audio_checkbox_ref.current:
            if 'with_audio' in label_vals:
                with_audio_checkbox_ref.current.value = label_vals['with_audio']
                if with_audio_checkbox_ref.current.page:
                    with_audio_checkbox_ref.current.update()
    except Exception:
        pass
    try:
        from flet_app.ui.pages.training_config import update_z_image_fields_visibility
        update_z_image_fields_visibility(
            is_z_image,
            label_vals.get('diffusion_model'),
            label_vals.get('vae'),
            label_vals.get('text_encoders'),
            label_vals.get('merge_adapters'),
        )
    except Exception:
        pass
    try:
        from flet_app.ui.pages.training_config import update_flux2_fields_visibility
        update_flux2_fields_visibility(
            is_flux2,
            label_vals.get('diffusion_model'),
            label_vals.get('vae'),
            label_vals.get('text_encoders'),
            label_vals.get('shift'),
        )
    except Exception:
        pass