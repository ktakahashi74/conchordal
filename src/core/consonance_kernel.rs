use crate::core::psycho_state::sanitize01;

#[derive(Clone, Copy, Debug)]
pub struct ConsonanceKernel {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}

impl Default for ConsonanceKernel {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: -1.35,
            c: 1.0,
            d: 0.0,
        }
    }
}

impl ConsonanceKernel {
    #[inline]
    pub fn score(&self, h01: f32, r01: f32) -> f32 {
        let h01 = sanitize01(h01);
        let r01 = sanitize01(r01);
        let a = sanitize_finite(self.a);
        let b = sanitize_finite(self.b);
        let c = sanitize_finite(self.c);
        let d = sanitize_finite(self.d);
        let score = a * h01 + b * r01 + c * h01 * r01 + d;
        sanitize_finite(score)
    }

    /// Density preset kernel: H * (1 - rho * R) = 1*H + 0*R + (-rho)*(H*R) + 0.
    #[inline]
    pub fn density_with_rho(rho: f32) -> Self {
        const RHO_MAX: f32 = 1_000_000.0;
        let rho = if rho.is_finite() {
            rho.clamp(0.0, RHO_MAX)
        } else {
            1.0
        };
        Self {
            a: 1.0,
            b: 0.0,
            c: -rho,
            d: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConsonanceRepresentationParams {
    pub beta: f32,
    pub theta: f32,
}

impl Default for ConsonanceRepresentationParams {
    fn default() -> Self {
        Self {
            beta: 2.0,
            theta: 0.0,
        }
    }
}

impl ConsonanceRepresentationParams {
    #[inline]
    pub fn level(&self, score: f32) -> f32 {
        let beta = if self.beta.is_finite() {
            self.beta.max(0.0)
        } else {
            0.0
        };
        let theta = sanitize_finite(self.theta);
        let score = sanitize_finite(score);
        sigmoid01_stable(beta * (score - theta))
    }

    #[inline]
    pub fn energy(&self, score: f32) -> f32 {
        -sanitize_finite(score)
    }
}

pub fn compose_consonance_field_level_scan(
    h01_scan: &[f32],
    r01_scan: &[f32],
    kernel: &ConsonanceKernel,
    repr: &ConsonanceRepresentationParams,
    out: &mut [f32],
) {
    debug_assert_eq!(h01_scan.len(), r01_scan.len());
    debug_assert_eq!(out.len(), h01_scan.len());
    let len = out.len().min(h01_scan.len()).min(r01_scan.len());
    for i in 0..len {
        let score = kernel.score(h01_scan[i], r01_scan[i]);
        out[i] = repr.level(score);
    }
}

#[inline]
pub fn sigmoid01_stable(x: f32) -> f32 {
    if x.is_nan() {
        return 0.0;
    }
    let x = x.clamp(-80.0, 80.0);
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sanitize_finite(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_score_matches_bilinear_formula() {
        let kernel = ConsonanceKernel {
            a: 0.8,
            b: -0.4,
            c: 1.25,
            d: 0.1,
        };
        let h = 0.6;
        let r = 0.2;
        let expected = 0.8 * h - 0.4 * r + 1.25 * h * r + 0.1;
        let got = kernel.score(h, r);
        assert_eq!(got.to_bits(), expected.to_bits());
    }

    #[test]
    fn level_is_bounded_and_monotonic() {
        let repr = ConsonanceRepresentationParams {
            beta: 3.0,
            theta: -0.2,
        };
        let xs = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let ys = xs.map(|x| repr.level(x));
        for y in ys {
            assert!((0.0..=1.0).contains(&y), "level out of range: {y}");
        }
        for i in 1..xs.len() {
            assert!(
                ys[i] >= ys[i - 1],
                "not monotonic: {} < {}",
                ys[i],
                ys[i - 1]
            );
        }
    }

    #[test]
    fn level_handles_nan_and_inf_inputs() {
        let repr = ConsonanceRepresentationParams::default();
        let y_nan = repr.level(f32::NAN);
        let y_pos = repr.level(f32::INFINITY);
        let y_neg = repr.level(f32::NEG_INFINITY);
        for y in [y_nan, y_pos, y_neg] {
            assert!(y.is_finite(), "level should be finite: {y}");
            assert!((0.0..=1.0).contains(&y), "level out of range: {y}");
        }
    }

    #[test]
    fn energy_is_negative_score() {
        let repr = ConsonanceRepresentationParams::default();
        let score = 0.37;
        let energy = repr.energy(score);
        assert!((energy + score).abs() < 1e-6);
    }

    #[test]
    fn energy_non_finite_input_maps_to_zero() {
        let repr = ConsonanceRepresentationParams::default();
        assert_eq!(repr.energy(f32::NAN), 0.0);
        assert_eq!(repr.energy(f32::INFINITY), 0.0);
        assert_eq!(repr.energy(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn density_with_rho_builds_expected_coefficients() {
        let k1 = ConsonanceKernel::density_with_rho(1.0);
        assert_eq!(k1.a, 1.0);
        assert_eq!(k1.b, 0.0);
        assert_eq!(k1.c, -1.0);
        assert_eq!(k1.d, 0.0);

        let k0 = ConsonanceKernel::density_with_rho(0.0);
        assert_eq!(k0.a, 1.0);
        assert_eq!(k0.b, 0.0);
        assert_eq!(k0.c, 0.0);
        assert_eq!(k0.d, 0.0);

        let k2 = ConsonanceKernel::density_with_rho(2.0);
        assert_eq!(k2.c, -2.0);
    }

    #[test]
    fn density_with_rho_sanitizes_negative_and_non_finite() {
        let k_neg = ConsonanceKernel::density_with_rho(-1.0);
        assert_eq!(k_neg.c, 0.0, "negative rho must clamp to 0");

        let k_nan = ConsonanceKernel::density_with_rho(f32::NAN);
        assert_eq!(k_nan.c, -1.0, "non-finite rho must fall back to 1.0");

        let k_inf = ConsonanceKernel::density_with_rho(f32::INFINITY);
        assert_eq!(k_inf.c, -1.0, "non-finite rho must fall back to 1.0");
    }

    #[test]
    fn density_with_rho_clamps_extreme_positive_values() {
        let k = ConsonanceKernel::density_with_rho(1e30);
        assert_eq!(k.c, -1_000_000.0);
    }
}
