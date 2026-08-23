"""_format_ticks_with_abbreviation() renders sexagesimal RA/Dec tick
labels the way astropy's own WCSAxes ticks do: only the leading
component that actually changed from the previous tick gets spelled
out in full; a neighbour differing only in seconds shows just that."""

from astrovox.viewer import _format_ticks_with_abbreviation

RA_UNITS = ("h", "m", "s")
DEC_UNITS = ("°", "'", '"')


def test_first_tick_is_always_spelled_out_in_full():
    components = [(1, 22, 16, 9.5)]
    assert _format_ticks_with_abbreviation(components, RA_UNITS) == ["22h16m09.5s"]


def test_neighbour_differing_only_in_seconds_is_abbreviated():
    components = [(1, 22, 16, 9.5), (1, 22, 16, 9.1)]
    result = _format_ticks_with_abbreviation(components, RA_UNITS)
    assert result[0] == "22h16m09.5s"
    assert result[1] == "09.1s"


def test_neighbour_differing_in_minutes_shows_minutes_and_seconds():
    components = [(1, 22, 16, 9.5), (1, 22, 17, 0.0)]
    result = _format_ticks_with_abbreviation(components, RA_UNITS)
    assert result[1] == "17m00.0s"


def test_neighbour_differing_in_hours_is_spelled_out_again():
    components = [(1, 22, 16, 9.5), (1, 23, 0, 0.0)]
    result = _format_ticks_with_abbreviation(components, RA_UNITS)
    assert result[1] == "23h00m00.0s"


def test_dec_ticks_carry_an_explicit_sign_ra_does_not():
    dec_components = [(1, 36, 50, 30.0)]
    assert _format_ticks_with_abbreviation(dec_components, DEC_UNITS) == ['+36°50\'30.0"']

    neg_dec_components = [(-1, 36, 50, 30.0)]
    assert _format_ticks_with_abbreviation(neg_dec_components, DEC_UNITS) == ['-36°50\'30.0"']

    ra_components = [(1, 22, 16, 9.5)]
    assert _format_ticks_with_abbreviation(ra_components, RA_UNITS) == ["22h16m09.5s"]


def test_sign_flip_forces_a_full_re_spell_even_if_magnitudes_match():
    components = [(1, 36, 50, 30.0), (-1, 36, 50, 30.0)]
    result = _format_ticks_with_abbreviation(components, DEC_UNITS)
    assert result[1] == '-36°50\'30.0"'
