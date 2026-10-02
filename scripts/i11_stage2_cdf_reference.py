"""Independent 20 ms CDF reference for the uncalibrated I11-2 test hazard."""

import math

SAMPLE_RATE = 48_000
STEP_SAMPLES = 960
POINTS = 201
MIDPOINTS_PER_STEP = 256


def hazard_cdf(elapsed_seconds, period_seconds, c0=-2.0, c10=4.0):
    """Integrate the registered phase hazard with a fixed midpoint rule."""
    if (
        elapsed_seconds is None
        or period_seconds is None
        or not math.isfinite(elapsed_seconds)
        or not math.isfinite(period_seconds)
        or elapsed_seconds < 0.0
        or period_seconds <= 0.0
    ):
        return None
    values = [0.0]
    accumulated = 0.0
    dt = STEP_SAMPLES / SAMPLE_RATE / MIDPOINTS_PER_STEP
    for step in range(POINTS - 1):
        start = elapsed_seconds + step * STEP_SAMPLES / SAMPLE_RATE
        interval = 0.0
        for substep in range(MIDPOINTS_PER_STEP):
            elapsed = start + (substep + 0.5) * dt
            z = c0 + c10 * math.cos(math.tau * (elapsed / period_seconds % 1.0))
            interval += (max(z, 0.0) + math.log1p(math.exp(-abs(z)))) * dt
        accumulated += interval
        values.append(-math.expm1(-accumulated))
    return tuple(values)


def hazard_cdf_for_forecast(elapsed_seconds, reset_unknown, period_seconds):
    """Do not turn an uncertain elapsed interval into a point CDF."""
    if (
        reset_unknown
        or elapsed_seconds is None
        or len(elapsed_seconds) != 2
        or elapsed_seconds[0] != elapsed_seconds[1]
    ):
        return None
    return hazard_cdf(elapsed_seconds[0], period_seconds)


def periodic_cdf(elapsed_seconds, period_seconds):
    """Use the next peak time as a step, matching the I6 periodic rule."""
    if (
        elapsed_seconds is None
        or period_seconds is None
        or not math.isfinite(elapsed_seconds)
        or not math.isfinite(period_seconds)
        or elapsed_seconds < 0.0
        or period_seconds <= 0.0
    ):
        return None
    wait = period_seconds - elapsed_seconds % period_seconds
    return tuple(float(i * STEP_SAMPLES / SAMPLE_RATE >= wait) for i in range(POINTS))


def cdf_at(values, sample_offset):
    """Interpolate only inside the registered four-second horizon."""
    if values is None or len(values) != POINTS or not 0 <= sample_offset <= 192_000:
        return None
    index, remainder = divmod(sample_offset, STEP_SAMPLES)
    if remainder == 0:
        return values[index]
    return values[index] + (values[index + 1] - values[index]) * remainder / STEP_SAMPLES


def arrival_window_probability(values, at_offset, width_samples):
    """A candidate past the CDF horizon makes this group ineligible."""
    if width_samples < 0 or at_offset < 0 or at_offset + width_samples > 192_000:
        return None
    before = cdf_at(values, max(0, at_offset - width_samples))
    after = cdf_at(values, at_offset + width_samples)
    if before is None or after is None:
        return None
    return after - before


def active_group_window(values, issued_at, horizon_end, now, at, width_samples):
    """Apply the strict consumption deadline before evaluating a window."""
    if (
        issued_at > now
        or now >= horizon_end
        or horizon_end != issued_at + 192_000
        or at < issued_at
    ):
        return None
    return arrival_window_probability(values, at - issued_at, width_samples)
