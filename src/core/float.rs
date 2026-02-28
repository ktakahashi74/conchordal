#[inline]
pub fn clamp01_finite(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
pub fn sanitize_nonnegative_finite(x: f32, fallback: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else if fallback.is_finite() {
        fallback.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp01_finite_handles_nonfinite_and_range() {
        assert_eq!(clamp01_finite(f32::NAN), 0.0);
        assert_eq!(clamp01_finite(f32::INFINITY), 0.0);
        assert_eq!(clamp01_finite(-1.0), 0.0);
        assert_eq!(clamp01_finite(2.0), 1.0);
        assert_eq!(clamp01_finite(0.3), 0.3);
    }

    #[test]
    fn sanitize_nonnegative_finite_handles_nonfinite_negative_and_positive() {
        assert_eq!(sanitize_nonnegative_finite(f32::NAN, 0.25), 0.25);
        assert_eq!(sanitize_nonnegative_finite(-1.0, 0.25), 0.0);
        assert_eq!(sanitize_nonnegative_finite(3.0, 0.25), 3.0);
    }
}
