"""_render_mathtext_rgba() rasterizes axis/colorbar titles via
matplotlib's Agg backend — no display needed, so it's safe to exercise
directly in CI."""

import numpy as np

from astrovox.viewer import _MATHTEXT_CACHE, _render_mathtext_rgba


def test_returns_a_non_empty_rgba_array():
    _MATHTEXT_CACHE.clear()
    rgba = _render_mathtext_rgba("Intensity", color=(1.0, 1.0, 1.0), fontsize=14)
    assert rgba.ndim == 3
    assert rgba.shape[2] == 4
    assert rgba.shape[0] > 0 and rgba.shape[1] > 0
    assert rgba.dtype == np.uint8


def test_larger_fontsize_produces_a_larger_image():
    _MATHTEXT_CACHE.clear()
    small = _render_mathtext_rgba("Intensity", fontsize=10)
    large = _render_mathtext_rgba("Intensity", fontsize=30)
    assert large.shape[0] > small.shape[0]


def test_result_is_cached_by_text_color_fontsize_and_weight():
    _MATHTEXT_CACHE.clear()
    _render_mathtext_rgba("Cached", color=(1.0, 1.0, 1.0), fontsize=12, fontweight="bold")
    key = ("Cached", (1.0, 1.0, 1.0), 12, "bold")
    assert key in _MATHTEXT_CACHE


def test_malformed_mathtext_falls_back_instead_of_raising():
    _MATHTEXT_CACHE.clear()
    # An unmatched "$" is invalid mathtext — must degrade to literal
    # text rather than crash the app.
    rgba = _render_mathtext_rgba("half-typed $unit", fontsize=12)
    assert rgba.shape[0] > 0 and rgba.shape[1] > 0


def test_bold_and_normal_weight_are_rendered_independently():
    _MATHTEXT_CACHE.clear()
    bold = _render_mathtext_rgba("Weight", fontsize=14, fontweight="bold")
    normal = _render_mathtext_rgba("Weight", fontsize=14, fontweight="normal")
    # Different glyphs -> different pixel content (bold strokes are
    # wider), even if by coincidence the canvas sizes matched.
    assert bold.shape != normal.shape or not np.array_equal(bold, normal)
