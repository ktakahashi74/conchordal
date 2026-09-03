/// Clamp into `[0,1]`, saturating infinities and mapping NaN to `0.0`.
/// This is the single 0..1 clamp for the whole tree — do not add another.
#[inline]
pub fn sanitize01(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else if x.is_infinite() {
        if x.is_sign_positive() { 1.0 } else { 0.0 }
    } else {
        0.0
    }
}

/// Pass `x` through when finite, otherwise substitute `fallback`.
#[inline]
pub fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Clamp to non-negative, mapping non-finite input to `0.0`.
#[inline]
pub fn sanitize_nonnegative_finite(x: f32) -> f32 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

/// Unnormalized Gaussian weight `exp(-d^2 / (2*sigma^2))`: `1.0` at `d == 0`, no
/// area normalization. Reach for this instead of spelling the exponent out again.
/// `sigma` must be strictly positive; callers clamp it, because the distance unit
/// (octaves, semitones, cents, bins, field score) dictates the floor.
///
/// Exception: `core::roughness_kernel` keeps its own `(-desq / (2*s*s)).exp()` form.
/// That spelling is not bit-identical to this one, and the kernel's tests pin its
/// output against locally recomputed values.
#[inline]
pub fn unit_gaussian(d: f32, sigma: f32) -> f32 {
    (-0.5 * (d / sigma).powi(2)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize01_saturates_infinities_and_clamps_range() {
        assert_eq!(sanitize01(f32::NAN), 0.0);
        assert_eq!(sanitize01(f32::INFINITY), 1.0);
        assert_eq!(sanitize01(f32::NEG_INFINITY), 0.0);
        assert_eq!(sanitize01(-1.0), 0.0);
        assert_eq!(sanitize01(2.0), 1.0);
        assert_eq!(sanitize01(0.3), 0.3);
    }

    #[test]
    fn sanitize_nonnegative_finite_handles_nonfinite_and_negative() {
        assert_eq!(sanitize_nonnegative_finite(f32::NAN), 0.0);
        assert_eq!(sanitize_nonnegative_finite(f32::INFINITY), 0.0);
        assert_eq!(sanitize_nonnegative_finite(-1.0), 0.0);
        assert_eq!(sanitize_nonnegative_finite(3.0), 3.0);
    }

    #[test]
    fn unit_gaussian_peaks_at_zero_and_matches_closed_form() {
        assert_eq!(unit_gaussian(0.0, 2.0), 1.0);
        // One sigma out is exp(-0.5); symmetric in d.
        let expected = (-0.5f32).exp();
        assert!((unit_gaussian(2.0, 2.0) - expected).abs() < 1e-6);
        assert!((unit_gaussian(-2.0, 2.0) - expected).abs() < 1e-6);
        // Same curve whichever way the exponent is spelled.
        let d = 37.0f32;
        let sigma = 60.0f32;
        let long_form = (-(d * d) / (2.0 * sigma * sigma)).exp();
        assert!((unit_gaussian(d, sigma) - long_form).abs() < 1e-6);
    }

    #[test]
    fn finite_or_substitutes_only_nonfinite() {
        assert_eq!(finite_or(f32::NAN, 0.25), 0.25);
        assert_eq!(finite_or(f32::INFINITY, 0.25), 0.25);
        assert_eq!(finite_or(-1.0, 0.25), -1.0);
        assert_eq!(finite_or(3.0, 0.25), 3.0);
    }
}
