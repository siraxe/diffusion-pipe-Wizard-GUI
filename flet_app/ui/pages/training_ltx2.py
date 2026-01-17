import flet as ft
from .._styles import add_section_title, create_textfield, create_dropdown


def get_ltx2_training_settings(ref=None):
    """
    Get LTX2-specific training settings UI.
    Returns a container with LTX2-specific training options.

    Args:
        ref: Optional Flet Ref to attach to the container
    """

    page_controls = []

    # --- LTX2 Optimization & Checkpoints Settings (Two Columns) ---
    ltx2_settings_section = ft.ResponsiveRow([
            ft.Column([
                *add_section_title("Optimization"),
                ft.Container(
                    content=ft.ResponsiveRow(controls=[
                        ft.Column([
                            create_textfield("steps", 2000, expand=True),
                            create_textfield("batch_size", 1, expand=True),
                            create_textfield("gradient_accumulation_steps", 1, expand=True),
                        ], col=6, spacing=6),
                        ft.Column([
                            create_textfield("learning_rate", 0.0001, expand=True),
                            create_textfield("max_grad_norm", 1.0, expand=True),
                            ft.ResponsiveRow(controls=[
                                create_dropdown(
                                    "optimizer_type",
                                    "adamw8bit",
                                    {"adamw8bit": "adamw8bit", "adamw": "adamw"},
                                    expand=True, col=6, scale=0.8
                                ),
                                create_dropdown(
                                    "scheduler_type",
                                    "constant",
                                    {"constant": "constant", "linear": "linear", "cosine": "cosine", "cosine_with_restarts": "cosine_with_restarts", "polynomial": "polynomial"},
                                    expand=True, col=6, scale=0.8
                                ),
                            ], spacing=2),
                        ], col=6, spacing=6),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),

            ft.Column([
                *add_section_title("Checkpoints"),
                ft.Container(
                    content=ft.ResponsiveRow(controls=[
                        create_textfield("interval", 50, col=4, expand=True),
                        create_textfield("keep_last_n", -1, col=4, expand=True),
                        create_dropdown(
                            "precision",
                            "bfloat16",
                            {"bfloat16": "bfloat16", "float32": "float32"},
                            col=4, expand=True
                        ),
                    ], spacing=6),
                    padding=ft.padding.all(10),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                    border_radius=ft.border_radius.all(10),
                ),
            ], col=6),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START)

    # --- Validation Settings ---
    validation_section = ft.ResponsiveRow([
        ft.Column([
            *add_section_title("Validation"),
            ft.Container(
                content=ft.Column(controls=[
                    # Row 1:
                    ft.ResponsiveRow(controls=[
                        create_dropdown(
                            "skip_initial_validation",
                            "false",
                            {"false": "false", "true": "true"},
                            col=4, expand=True
                        ),
                        create_dropdown(
                            "generate_audio",
                            "false",
                            {"false": "false", "true": "true"},
                            col=4, expand=True
                        ),
                        create_textfield("validation_interval", "none", col=4, expand=True),
                    ], spacing=6),
                    # Three separate rows: prompts, negative_prompt, images
                    create_textfield("prompts", "Two woman with long brown hair", expand=True),
                    create_textfield("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted", expand=True),
                    create_textfield("images", "none", expand=True),
                    # Row 2: video_dims, videos_per_prompt, guidance_scale
                    ft.ResponsiveRow(controls=[
                        create_textfield("video_dims", "640, 416, 89", col=4, expand=True),
                        create_textfield("videos_per_prompt", 1, col=4, expand=True),
                        create_textfield("guidance_scale", 4.0, col=4, expand=True),
                    ], spacing=6),
                    # Row 3: frame_rate, seed, inference_steps
                    ft.ResponsiveRow(controls=[
                        create_textfield("frame_rate", 25, col=4, expand=True),
                        create_textfield("seed", 42, col=4, expand=True),
                        create_textfield("inference_steps", 30, col=4, expand=True),
                    ], spacing=6),
                ], spacing=6),
                padding=ft.padding.all(10),
                border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.GREY_600)),
                border_radius=ft.border_radius.all(10),
            ),
            ], col=6),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START)

    page_controls.append(ltx2_settings_section)
    page_controls.append(validation_section)
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
