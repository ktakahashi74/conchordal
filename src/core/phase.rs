use std::f32::consts::{PI, TAU};

#[inline]
pub fn wrap_0_tau(x: f32) -> f32 {
    x.rem_euclid(TAU)
}

/// Normalize to the range [-PI, PI).
#[inline]
pub fn wrap_pm_pi(x: f32) -> f32 {
    (x + PI).rem_euclid(TAU) - PI
}

#[inline]
pub fn angle_diff_pm_pi(a: f32, b: f32) -> f32 {
    wrap_pm_pi(a - b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_diff_is_wrapped() {
        let pairs = [
            (0.0, 0.0),
            (TAU, 0.0),
            (PI, -PI),
            (0.25 * PI, -0.25 * PI),
            (3.0 * PI, PI),
        ];
        for (a, b) in pairs {
            let d = angle_diff_pm_pi(a, b);
            assert!(d >= -PI && d < PI, "angle_diff out of range: {d}");
            let d2 = angle_diff_pm_pi(a + TAU, b);
            assert!((d - d2).abs() < 1e-5, "angle_diff periodicity failed");
        }
    }

    #[test]
    fn wrap_0_tau_in_range() {
        let values = [-10.0 * TAU, -TAU, -PI, -0.1, 0.0, PI, TAU, 3.5 * TAU];
        for v in values {
            let w = wrap_0_tau(v);
            assert!(w >= 0.0 && w < TAU, "wrap_0_tau out of range: {w}");
        }
    }

    #[test]
    fn wrap_pm_pi_in_range() {
        let values = [-10.0 * TAU, -TAU, -PI, -0.1, 0.0, PI, TAU, 3.5 * TAU];
        for v in values {
            let w = wrap_pm_pi(v);
            assert!(w >= -PI && w < PI, "wrap_pm_pi out of range: {w}");
        }
    }
}
