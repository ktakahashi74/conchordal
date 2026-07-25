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
    fn finite_or_substitutes_only_nonfinite() {
        assert_eq!(finite_or(f32::NAN, 0.25), 0.25);
        assert_eq!(finite_or(f32::INFINITY, 0.25), 0.25);
        assert_eq!(finite_or(-1.0, 0.25), -1.0);
        assert_eq!(finite_or(3.0, 0.25), 3.0);
    }
}
