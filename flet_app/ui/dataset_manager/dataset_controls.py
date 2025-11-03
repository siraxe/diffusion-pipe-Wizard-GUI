# This file will contain helper functions for building Flet UI controls.

import flet as ft

# Constant for ExpansionTile border radius
EXPANSION_TILE_BORDER_RADIUS = 10
EXPANSION_TILE_HEADER_BG_COLOR = ft.CupertinoColors.with_opacity(0.08, ft.CupertinoColors.ACTIVE_BLUE)
EXPANSION_TILE_INSIDE_BG_COLOR = ft.Colors.TRANSPARENT

def build_expansion_tile(
    title: str,
    controls: list[ft.Control],
    initially_expanded: bool = False,
):
    return ft.ExpansionTile(
        title=ft.Text(title, size=12),
        bgcolor=EXPANSION_TILE_INSIDE_BG_COLOR,
        collapsed_bgcolor=EXPANSION_TILE_HEADER_BG_COLOR,
        controls=[ft.Divider(), ft.Column(controls, spacing=10) ,ft.Divider()], # Wrap controls in a Column
        initially_expanded=initially_expanded,
        collapsed_shape=ft.RoundedRectangleBorder(radius=EXPANSION_TILE_BORDER_RADIUS),
        shape=ft.RoundedRectangleBorder(radius=EXPANSION_TILE_BORDER_RADIUS),
        enable_feedback=False,
    )
