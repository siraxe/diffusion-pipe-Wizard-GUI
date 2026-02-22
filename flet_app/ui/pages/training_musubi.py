import flet as ft
from .._styles import add_section_title, create_textfield, create_dropdown, create_checkbox


def get_musubi_training_settings(
    ref=None,
    attn_chunking_ref=None,
    blank_preservation_ref=None,
    blank_preservation_args_ref=None,
    dop_ref=None,
    dop_args_ref=None,
    prior_divergence_ref=None,
    prior_divergence_args_ref=None,
    crepa_ref=None,
    crepa_mode_ref=None,
    crepa_args_ref=None,
    sync_visibility_func=None,
):
    """
    Get Musubi trainer-specific training settings UI.
    Returns a container with Musubi-specific training options.

    Args:
        ref: Optional Flet Ref to attach to the container
    """

    page_controls = []

    # Create refs for learning_rate, optimizer, and scheduler dropdowns
    learning_rate_ref = ft.Ref[ft.TextField]()
    optimizer_ref = ft.Ref[ft.Dropdown]()
    scheduler_ref = ft.Ref[ft.Dropdown]()
    lr_warmup_steps_ref = ft.Ref[ft.TextField]()
    optimizer_args_ref = ft.Ref[ft.TextField]()

    def on_optimizer_change(e):
        """Update learning rate when optimizer changes."""
        # NOTE: Disabled automatic learning rate changes to preserve loaded TOML values
        # Users can manually adjust the learning rate as needed

        # Show/hide optimizer_args field based on optimizer selection
        if optimizer_ref.current and optimizer_args_ref.current:
            if optimizer_ref.current.value == "Automagic":
                optimizer_args_ref.current.visible = True
            else:
                optimizer_args_ref.current.visible = False
            optimizer_args_ref.current.update()

    def on_scheduler_change(e):
        """Show/hide lr_warmup_steps field based on scheduler selection."""
        if scheduler_ref.current and lr_warmup_steps_ref.current:
            if scheduler_ref.current.value == "constant_with_warmup":
                lr_warmup_steps_ref.current.visible = True
            else:
                lr_warmup_steps_ref.current.visible = False
            lr_warmup_steps_ref.current.update()

    # --- Musubi Optimization & Checkpoints Settings (Two Columns) ---
    musubi_settings_section = ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Optimization"),
                ft.Container(
                    content=ft.Column(controls=[
                        # Row 1: batch_size, grad_accum_steps, max_grad_norm, blocks_to_swap
                        ft.ResponsiveRow(controls=[
                            create_textfield("batch_size", 1, col=3, expand=True),
                            create_textfield("grad_accum_steps", 1, col=3, expand=True),
                            create_textfield("max_grad_norm", 1.0, col=3, expand=True),
                            create_textfield("blocks_to_swap", 0, col=3, expand=True),
                        ], spacing=6),
                        # Row 2: learning_rate, optimizer, scheduler_type, timestep_sampling
                        ft.ResponsiveRow(controls=[
                            create_textfield("learning_rate", 0.0001, col=3, expand=True, ref=learning_rate_ref),
                            create_dropdown(
                                "optimizer_type_m",
                                "AdamW",
                                {
                                    "AdamW": "AdamW",
                                    "AdamW8bit": "AdamW8bit",
                                    "Adafactor": "Adafactor",
                                    "Prodigy": "Prodigy",
                                    "Automagic": "Automagic",
                                },
                                col=3, expand=True, scale=0.8,
                                on_change=on_optimizer_change, ref=optimizer_ref
                            ),
                            create_dropdown(
                                "scheduler_type",
                                "constant",
                                {
                                    "constant": "constant",
                                    "constant_with_warmup": "constant_with_warmup",
                                    "linear": "linear",
                                    "cosine": "cosine",
                                    "cosine_with_restarts": "cosine_with_restarts",
                                    "polynomial": "polynomial",
                                    "adafactor": "adafactor",
                                    "rex": "rex",
                                },
                                col=3, expand=True, scale=0.8,
                                on_change=on_scheduler_change, ref=scheduler_ref
                            ),
                            create_dropdown(
                                "timestep_sm_m",
                                "shifted_logit_normal",
                                {
                                    "shifted_logit_normal": "shifted_logit_normal",
                                    "sigma": "sigma",
                                    "uniform": "uniform",
                                    "sigmoid": "sigmoid",
                                    "logsnr": "logsnr",
                                },
                                col=3, expand=True, scale=0.8
                            ),
                        ], spacing=6),
                        # Row 3: optimizer_args - only visible when Automagic is selected
                        ft.Container(
                            content=ft.TextField(
                                label="optimizer_args",
                                value="min_lr=1e-7, max_lr=1e-3, lr_bump=1e-6, eps=(1e-30; 1e-3), clip_threshold=1.0, beta2=0.999, weight_decay=0.0, do_paramiter_swapping=False, paramiter_swapping_factor=0.1",
                                ref=optimizer_args_ref,
                                data="optimizer_args",
                                visible=False,
                                fill_color=ft.Colors.GREY_900,
                                border_color=ft.Colors.BLUE_GREY_200,
                                focused_border_color=ft.Colors.BLUE_GREY_700,
                                content_padding=ft.padding.symmetric(vertical=10, horizontal=10),
                                text_size=12,
                                label_style=ft.TextStyle(size=12),
                                text_style=ft.TextStyle(size=12),
                                dense=True,
                            ),
                            expand=True,
                        ),
                        # Row 4: lr_warmup_steps - only visible when constant_with_warmup is selected
                        ft.ResponsiveRow(controls=[
                            create_textfield("lr_warmup_steps", 50, col=3, expand=True, ref=lr_warmup_steps_ref, visible=False),
                        ], spacing=6),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
                # Validation section
                *add_section_title("Validation"),
                ft.Container(
                    content=ft.Column(controls=[
                        # Row 1: sample_at_first, sample_every_n_interval, video_dims, generate_audio
                        ft.ResponsiveRow(controls=[
                            create_dropdown(
                                "sample_at_first",
                                "false",
                                {"false": "false", "true": "true"},
                                col=3, expand=True
                            ),
                            create_textfield("sample_every_n_interval", "-1", col=3, expand=True),
                            create_textfield("video_dims", "768, 512, 45", col=3, expand=True),
                            ft.Container(
                                content=ft.Checkbox(
                                    label="Audio",
                                    value=False,
                                    data="generate_audio",
                                    scale=0.8,
                                ),
                                col=3, expand=True,
                                alignment=ft.alignment.center
                            ),
                        ], spacing=6),
                        # Row 2: sample_steps, guidance_scale, seed, tiled_vae, s_offload, cache_te, cache_i2v
                        ft.ResponsiveRow(controls=[
                            create_textfield("sample_steps", "30", col=1.5, expand=True),
                            create_textfield("guidance_scale", "4.0", col=1.5, expand=True),
                            create_textfield("seed", "42", col=1.5, expand=True),
                            create_checkbox("Tiled VAE", True, "tiled_vae", col=1.5, expand=True, scale=0.8),
                            create_checkbox("S-Offload", True, "s_offload", col=2, expand=True, scale=0.8),
                            create_checkbox("Cache TE", True, "cache_te", col=2, expand=True, scale=0.8),
                            create_checkbox("Cache I2V", True, "cache_i2v", col=2, expand=True, scale=0.8),
                        ], spacing=6),
                        # Prompts: prompts, negative_prompt, start_images
                        create_textfield("prompts", "Two women with long brown hair dancing on the dance floor", expand=True),
                        create_textfield("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted", expand=True),
                        create_textfield("start_images", "none", expand=True),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),

            ft.Column([
                # Custom title row with dropdowns
                ft.Container(
                    content=ft.Row([
                        ft.Text("Checkpoints", weight=ft.FontWeight.BOLD, size=16, expand=True),
                        create_checkbox("Save State", False, "save_state", scale=0.85),
                        create_dropdown(
                            "checkpoint_mode",
                            "steps",
                            {"steps": "steps", "epochs": "epochs"},
                            width=130
                        ),
                        create_dropdown(
                            "convert_to_comfy",
                            "true",
                            {"true": "true", "false": "false"},
                            width=130
                        ),
                    ]),
                    margin=ft.margin.only(bottom=-5)
                ),
                ft.Divider(height=5, thickness=1),
                ft.Container(
                    content=ft.ResponsiveRow(controls=[
                        create_textfield("max_steps", 2000, col=3, expand=True),
                        create_textfield("interval", 50, col=3, expand=True),
                        create_textfield("keep_last_n", -1, col=3, expand=True),
                        create_dropdown(
                            "precision",
                            "bfloat16",
                            {"bfloat16": "bfloat16", "float32": "float32"},
                            col=3, expand=True
                        ),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
                ft.ExpansionTile(
                    title=ft.Text("Advanced options", size=12),
                    bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.GREY_800),
                    collapsed_bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_700),
                    controls=[
                        ft.Divider(height=1),
                        # Row 1: Checkboxes (5 columns)
                        ft.ResponsiveRow(controls=[
                            ft.Container(
                                content=ft.Checkbox(
                                    label="attn_chunking",
                                    value=False,
                                    scale=0.8,
                                    ref=attn_chunking_ref,
                                    data="attn_chunking",
                                    tooltip="Experimental: Splits attention computation into 512-token chunks to reduce VRAM usage.",
                                ),
                                col=2.4, expand=True,
                            ),
                            ft.Container(
                                content=ft.Checkbox(
                                    label="blank_preservation",
                                    value=False,
                                    scale=0.8,
                                    ref=blank_preservation_ref,
                                    data="blank_preservation",
                                    on_change=lambda e: sync_visibility_func() if sync_visibility_func else None,
                                    tooltip="Prevents the LoRA from altering the model's blank-prompt output (CFG baseline). +2 fwd, +1 back. Recommended: 0.5 - 1.0",
                                ),
                                col=2.4, expand=True,
                            ),
                            ft.Container(
                                content=ft.Checkbox(
                                    label="dop",
                                    value=False,
                                    scale=0.8,
                                    ref=dop_ref,
                                    data="dop",
                                    on_change=lambda e: sync_visibility_func() if sync_visibility_func else None,
                                    tooltip="Prevents altering class-prompt output, scoping LoRA effect to trigger word only. +2 fwd, +1 back. Recommended: 0.5 - 1.0",
                                ),
                                col=2.4, expand=True,
                            ),
                            ft.Container(
                                content=ft.Checkbox(
                                    label="prior_divergence",
                                    value=False,
                                    scale=0.8,
                                    ref=prior_divergence_ref,
                                    data="prior_divergence",
                                    on_change=lambda e: sync_visibility_func() if sync_visibility_func else None,
                                    tooltip="Encourages LoRA to differ from base model on training prompts, preventing weak/timid LoRAs. +1 fwd, 0 back. Recommended: 0.05 - 0.1",
                                ),
                                col=2.4, expand=True,
                            ),
                            ft.Container(
                                content=ft.Checkbox(
                                    label="CREPA",
                                    value=False,
                                    scale=0.8,
                                    ref=crepa_ref,
                                    data="crepa",
                                    on_change=lambda e: sync_visibility_func() if sync_visibility_func else None,
                                    tooltip="Cross-frame Representation Alignment - temporal consistency via feature alignment. No extra forward passes. Based on arXiv 2506.09229",
                                ),
                                col=2.4, expand=True,
                            ),
                        ], spacing=6),
                        # CREPA row: mode dropdown + args (hidden by default)
                        ft.ResponsiveRow(controls=[
                            ft.Container(
                                content=ft.Dropdown(
                                    label="crepa_mode",
                                    value="backbone",
                                    options=[
                                        ft.dropdown.Option("backbone"),
                                        ft.dropdown.Option("dino"),
                                    ],
                                    scale=0.8,
                                    ref=crepa_mode_ref,
                                    data="crepa_mode",
                                    visible=False,
                                    expand=True,
                                    on_change=lambda e: (
                                        setattr(crepa_args_ref.current, 'value',
                                            "student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2"
                                            if e.control.value == "backbone"
                                            else "dino_model=dinov2_vitb14 student_block_idx=16 lambda_crepa=0.1"
                                        ),
                                        crepa_args_ref.current.update() if crepa_args_ref.current else None
                                    ),
                                ),
                                col=3,
                            ),
                            ft.Container(
                                content=ft.TextField(
                                    label="crepa_args",
                                    value="student_block_idx=16 teacher_block_idx=32 lambda_crepa=0.1 tau=1.0 num_neighbors=2",
                                    scale=0.8,
                                    ref=crepa_args_ref,
                                    data="crepa_args",
                                    visible=False,
                                    expand=True,
                                ),
                                col=9,
                            ),
                        ], spacing=2),
                        # Row 2: Args fields (3 columns, hidden by default)
                        ft.ResponsiveRow(controls=[
                            ft.Container(
                                content=ft.TextField(
                                    label="blank_preservation_args",
                                    value="multiplier=0.5",
                                    scale=0.8,
                                    ref=blank_preservation_args_ref,
                                    data="blank_preservation_args",
                                    visible=False,
                                ),
                                col=4, expand=True,
                            ),
                            ft.Container(
                                content=ft.TextField(
                                    label="dop_args",
                                    value="class=woman multiplier=1.0",
                                    scale=0.8,
                                    ref=dop_args_ref,
                                    data="dop_args",
                                    visible=False,
                                ),
                                col=4, expand=True,
                            ),
                            ft.Container(
                                content=ft.TextField(
                                    label="prior_divergence_args",
                                    value="multiplier=0.1",
                                    scale=0.8,
                                    ref=prior_divergence_args_ref,
                                    data="prior_divergence_args",
                                    visible=False,
                                ),
                                col=4, expand=True,
                            ),
                        ], spacing=6),
                        ft.Divider(height=1),
                    ],
                    initially_expanded=False,
                    collapsed_shape=ft.RoundedRectangleBorder(radius=10),
                    shape=ft.RoundedRectangleBorder(radius=10),
                ),
            ], col=6),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START)

    page_controls.append(musubi_settings_section)
    page_controls.append(ft.Divider(height=5, color=ft.Colors.TRANSPARENT))

    container = ft.Container(
        content=ft.Column(
            controls=page_controls,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True,
        padding=ft.padding.all(5),
        ref=ref  # Attach the ref if provided
    )

    return container
