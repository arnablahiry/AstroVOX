"""_nice_ticks() is the "nice number" tick locator behind the custom
RA/Dec tick overlay — no VTK/Qt involved, so it's cheap to pin down
directly."""

from astrovox.viewer import _nice_ticks


def test_degenerate_span_returns_single_tick():
    assert _nice_ticks(5.0, 5.0) == [5.0]
    assert _nice_ticks(5.0, 4.0) == [5.0]


def test_ticks_are_sorted_and_within_or_bracketing_the_span():
    ticks = _nice_ticks(0.0, 100.0)
    assert ticks == sorted(ticks)
    assert len(ticks) >= 2
    assert ticks[0] >= 0.0 - 1e-9
    assert ticks[-1] <= 100.0 + 1e-9


def test_tick_count_is_close_to_target():
    for target in (4, 8, 12):
        ticks = _nice_ticks(0.0, 1000.0, target=target)
        # "nice number" stepping can't always land exactly on target,
        # but it should never be wildly off.
        assert abs(len(ticks) - target) <= 3


def test_step_size_is_a_round_1_2_2_5_or_5_multiple_of_a_power_of_ten():
    ticks = _nice_ticks(0.0, 47.0, target=6)
    step = round(ticks[1] - ticks[0], 10)
    assert step > 0
    for exponent in range(-6, 7):
        mag = 10.0 ** exponent
        for base in (1, 2, 2.5, 5):
            if abs(step - base * mag) < 1e-9:
                return
    raise AssertionError(f"step {step} isn't a nice 1/2/2.5/5 x 10^n value")


def test_handles_small_and_large_magnitudes():
    small = _nice_ticks(1e-5, 3e-5)
    assert len(small) >= 2
    large = _nice_ticks(1e8, 5e8)
    assert len(large) >= 2


def test_handles_negative_ranges():
    ticks = _nice_ticks(-50.0, 50.0)
    assert ticks[0] <= 0 <= ticks[-1]
