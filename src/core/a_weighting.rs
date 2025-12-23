//! A-weighting helpers for amplitude and power domains.

use crate::core::utils::a_weighting_gain;

/// A-weighting power gain (amplitude gain squared).
pub fn a_weighting_gain_pow(f_hz: f32) -> f32 {
    let g = a_weighting_gain(f_hz);
    g * g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::a_weighting_gain;

    #[test]
    fn a_weighting_gain_pow_matches_square() {
        let f = 1000.0;
        let g = a_weighting_gain(f);
        let g_pow = a_weighting_gain_pow(f);
        assert!((g_pow - g * g).abs() < 1e-6);
    }

    #[test]
    fn a_weighting_gain_pow_is_finite_and_near_unity_at_1khz() {
        let g_pow = a_weighting_gain_pow(1000.0);
        assert!(g_pow.is_finite());
        assert!((g_pow - 1.0).abs() < 0.05, "g_pow={g_pow}");
    }
}
