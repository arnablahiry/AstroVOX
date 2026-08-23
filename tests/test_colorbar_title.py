"""_compose_colorbar_title() builds the main viewer's colorbar title —
bold(quantity name) on one line, [unit] beneath it — pure string logic,
independent of PyQt5/Qt actually being available."""

from astrovox.gui import _compose_colorbar_title


def test_name_and_unit_render_bold_name_then_bracketed_unit_on_next_line():
    title = _compose_colorbar_title("Intensity", "Jy/beam")
    assert title == "$\\mathbf{Intensity}$\n[Jy/beam]"


def test_spaces_in_the_name_are_escaped_for_math_mode():
    # Bare spaces are collapsed inside $...$ math mode — this is exactly
    # the "TotalMatterDensity" (no spaces) bug this escaping fixes.
    title = _compose_colorbar_title("Total Matter Density", "Msun/h")
    assert "Total\\ Matter\\ Density" in title
    assert "Total Matter Density" not in title


def test_dollar_signs_in_the_name_are_escaped():
    title = _compose_colorbar_title("$weird$", "unit")
    assert r"\$weird\$" in title


def test_unit_only_when_name_is_blank():
    assert _compose_colorbar_title("", "Jy/beam") == "[Jy/beam]"


def test_name_only_when_unit_is_blank():
    assert _compose_colorbar_title("Intensity", "") == "$\\mathbf{Intensity}$"


def test_blank_name_and_unit_returns_empty_string():
    assert _compose_colorbar_title("", "") == ""


def test_surrounding_whitespace_is_stripped():
    assert _compose_colorbar_title("  Intensity  ", "  Jy/beam  ") == "$\\mathbf{Intensity}$\n[Jy/beam]"
