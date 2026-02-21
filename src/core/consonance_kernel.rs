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
}

#[derive(Clone, Copy, Debug)]
pub struct ConsonanceRepresentationParams {
    pub beta: f32,
    pub theta: f32,
    pub temperature: f32,
    pub epsilon: f32,
}

impl Default for ConsonanceRepresentationParams {
    fn default() -> Self {
        Self {
            beta: 2.0,
            theta: 0.0,
            temperature: 1.0,
            epsilon: 1e-6,
        }
    }
}

impl ConsonanceRepresentationParams {
    #[inline]
    pub fn level01(&self, score: f32) -> f32 {
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
    pub fn weight(&self, score: f32) -> f32 {
        let score = sanitize_finite(score);
        let temperature = if self.temperature.is_finite() {
            self.temperature.max(1e-6)
        } else {
            1e-6
        };
        let epsilon = if self.epsilon.is_finite() {
            self.epsilon.max(0.0)
        } else {
            0.0
        };
        let exp_arg = (score / temperature).clamp(-80.0, 80.0);
        let v = exp_arg.exp() + epsilon;
        if v.is_finite() { v.max(0.0) } else { epsilon }
    }

    #[inline]
    pub fn energy(&self, score: f32) -> f32 {
        -sanitize_finite(score)
    }

    pub fn density(&self, weights: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; weights.len()];
        self.normalize_density(weights, &mut out);
        out
    }

    pub fn normalize_density(&self, weights: &[f32], out: &mut [f32]) {
        debug_assert_eq!(weights.len(), out.len());
        if weights.is_empty() || out.is_empty() {
            return;
        }
        let len = weights.len().min(out.len());
        let mut total = 0.0f32;
        for i in 0..len {
            let w = weights[i];
            let sanitized = if w.is_finite() { w.max(0.0) } else { 0.0 };
            out[i] = sanitized;
            total += sanitized;
        }

        if total > 0.0 && total.is_finite() {
            let inv = 1.0 / total;
            for v in out.iter_mut().take(len) {
                *v *= inv;
            }
        } else {
            let uniform = 1.0 / len as f32;
            for v in out.iter_mut().take(len) {
                *v = uniform;
            }
        }
    }
}

pub fn compose_consonance_level01_scan(
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
        out[i] = repr.level01(score);
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
fn sanitize01(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else if x.is_infinite() {
        if x.is_sign_positive() { 1.0 } else { 0.0 }
    } else {
        0.0
    }
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
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn level01_is_bounded_and_monotonic() {
        let repr = ConsonanceRepresentationParams {
            beta: 3.0,
            theta: -0.2,
            temperature: 1.0,
            epsilon: 1e-6,
        };
        let xs = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let ys = xs.map(|x| repr.level01(x));
        for y in ys {
            assert!((0.0..=1.0).contains(&y), "level01 out of range: {y}");
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
    fn level01_handles_nan_and_inf_inputs() {
        let repr = ConsonanceRepresentationParams::default();
        let y_nan = repr.level01(f32::NAN);
        let y_pos = repr.level01(f32::INFINITY);
        let y_neg = repr.level01(f32::NEG_INFINITY);
        for y in [y_nan, y_pos, y_neg] {
            assert!(y.is_finite(), "level01 should be finite: {y}");
            assert!((0.0..=1.0).contains(&y), "level01 out of range: {y}");
        }
    }

    #[test]
    fn weight_is_nonnegative_and_finite_for_extremes() {
        let repr_tiny = ConsonanceRepresentationParams {
            beta: 2.0,
            theta: 0.0,
            temperature: 1e-9,
            epsilon: 5e-4,
        };
        let repr_huge = ConsonanceRepresentationParams {
            beta: 2.0,
            theta: 0.0,
            temperature: 1e9,
            epsilon: 1e-6,
        };
        let repr_zero = ConsonanceRepresentationParams {
            beta: 2.0,
            theta: 0.0,
            temperature: 0.0,
            epsilon: 1e-6,
        };
        let repr_negative = ConsonanceRepresentationParams {
            beta: 2.0,
            theta: 0.0,
            temperature: -7.0,
            epsilon: 1e-6,
        };
        for repr in [repr_tiny, repr_huge, repr_zero, repr_negative] {
            for score in [-1e6f32, -10.0, 0.0, 10.0, 1e6, f32::NAN, f32::INFINITY] {
                let w = repr.weight(score);
                assert!(w >= 0.0, "weight<0 for score={score}");
                assert!(
                    w.is_finite(),
                    "weight should be finite for score={score}: {w}"
                );
            }
        }
        assert!(
            repr_tiny.weight(-1e6) >= repr_tiny.epsilon,
            "epsilon should be an effective floor for very negative scores"
        );
    }

    #[test]
    fn normalize_density_sums_to_one_and_fallbacks() {
        let repr = ConsonanceRepresentationParams::default();

        let density = repr.density(&[1.0, 2.0, 3.0]);
        let sum: f32 = density.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");

        let zeros = repr.density(&[0.0, 0.0, 0.0, 0.0]);
        for &v in &zeros {
            assert!((v - 0.25).abs() < 1e-6, "fallback not uniform: {v}");
        }

        let nan_inf = repr.density(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        let sum: f32 = nan_inf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        for &v in &nan_inf {
            assert!((v - (1.0 / 3.0)).abs() < 1e-6, "fallback not uniform: {v}");
        }
    }

    #[test]
    fn energy_is_negative_score() {
        let repr = ConsonanceRepresentationParams::default();
        let score = 0.37;
        let energy = repr.energy(score);
        assert!(
            (energy + score).abs() < 1e-6,
            "energy={energy} score={score}"
        );
    }

    #[test]
    fn energy_non_finite_input_maps_to_zero() {
        let repr = ConsonanceRepresentationParams::default();
        assert_eq!(repr.energy(f32::NAN), 0.0);
        assert_eq!(repr.energy(f32::INFINITY), 0.0);
        assert_eq!(repr.energy(f32::NEG_INFINITY), 0.0);
    }
}
