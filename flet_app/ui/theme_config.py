"""
Centralized theme configuration for DPipe Flet app.
Ensures consistent colors and styling across all browsers (Firefox, Chrome, Safari, etc.).
"""

import flet as ft


class DPipeTheme:
    """Centralized theme configuration for DPipe application."""

    # =====================
    # Core Color Palette
    # =====================

    # Primary colors
    PRIMARY = ft.Colors.BLUE_600
    SECONDARY = ft.Colors.BLUE_GREY_600

    # Status colors
    SUCCESS = ft.Colors.GREEN_600
    ERROR = ft.Colors.RED_600
    WARNING = ft.Colors.AMBER_600
    INFO = ft.Colors.LIGHT_BLUE_600

    # Background colors
    BACKGROUND_DARK = "#0f0f0f"  # Very dark grey, matches your existing terminal-like theme
    SURFACE_DARK = ft.Colors.SURFACE

    # Text colors
    TEXT_PRIMARY = ft.Colors.BLUE_GREY_700
    TEXT_SECONDARY = ft.Colors.BLUE_GREY_500

    # =====================
    # Theme Configuration
    # =====================

    # Dark theme (primary theme for DPipe)
    DARK_THEME = ft.Theme(
        color_scheme_seed=PRIMARY,
        use_material3=True,
    )

    # Light theme (for future light mode support)
    LIGHT_THEME = ft.Theme(
        color_scheme_seed=PRIMARY,
        use_material3=True,
    )

    @staticmethod
    def apply_to_page(page: ft.Page, theme_mode: str = "dark"):
        """
        Apply DPipe theme to a Flet page.

        Args:
            page: Flet Page object
            theme_mode: "dark" or "light"

        This ensures consistent appearance across all browsers including Firefox.
        """
        # Set dark theme as primary
        page.theme = DPipeTheme.DARK_THEME

        # Explicitly set background color (fixes Firefox white background issue)
        page.bgcolor = DPipeTheme.BACKGROUND_DARK

        # Force dark mode explicitly
        page.theme_mode = ft.ThemeMode.DARK

        # Update page to apply changes
        page.update()
