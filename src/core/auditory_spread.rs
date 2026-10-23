//! Addendum-5 lag coherence and Gaussian deposition, test-only.
use super::*;
use rustfft::num_complex::Complex64;

#[derive(Clone)]
pub(super) struct SpreadHistory {
    pub lag: Vec<usize>,
    offsets: Vec<usize>,
    ring: Vec<Complex32>,
    cursors: Vec<usize>,
    correlation: Vec<Complex64>,
    energy: Vec<[f64; 2]>,
    pub coherence: Vec<f64>,
    pub sigma_hz: Vec<f64>,
}

impl SpreadHistory {
    pub fn new(centers: &[f32], fs: f32, beta: f64) -> Self {
        let lag: Vec<_> = centers
            .iter()
            .map(|&f| {
                (f64::from(fs) / (4.0 * beta * 1.0186 * (24.7 + f64::from(f) / 9.265))).round()
                    as usize
            })
            .collect();
        assert!(lag.iter().all(|&l| l > 0));
        let mut offsets = Vec::with_capacity(lag.len());
        let mut samples = 0;
        for &l in &lag {
            offsets.push(samples);
            samples += l;
        }
        let n = centers.len();
        Self {
            lag,
            offsets,
            ring: vec![Complex32::default(); samples],
            cursors: vec![0; n],
            correlation: vec![Complex64::default(); n],
            energy: vec![[0.0; 2]; n],
            coherence: vec![1.0; n],
            sigma_hz: vec![0.0; n],
        }
    }

    pub fn start_hop(&mut self) {
        self.correlation.fill(Complex64::default());
        self.energy.fill([0.0; 2]);
    }

    pub fn observe(&mut self, output: &[Complex32]) {
        for (c, &z) in output.iter().enumerate() {
            let index = self.offsets[c] + self.cursors[c];
            let past = self.ring[index];
            self.ring[index] = z;
            self.cursors[c] = (self.cursors[c] + 1) % self.lag[c];
            // The filter remains f32; f64 avoids cancellation in acos near kappa=1.
            let z = Complex64::new(f64::from(z.re), f64::from(z.im));
            let past = Complex64::new(f64::from(past.re), f64::from(past.im));
            self.correlation[c] += z * past.conj();
            self.energy[c][0] += z.norm_sqr();
            self.energy[c][1] += past.norm_sqr();
        }
    }

    pub fn finish(&mut self, fs: f32) {
        for c in 0..self.lag.len() {
            let denominator = (self.energy[c][0] * self.energy[c][1]).sqrt();
            let k = if denominator > 0.0 {
                (self.correlation[c].norm() / denominator).clamp(0.0, 1.0)
            } else {
                1.0
            };
            self.coherence[c] = k;
            self.sigma_hz[c] =
                k.acos() * f64::from(fs) / (std::f64::consts::TAU * self.lag[c] as f64);
        }
    }

    pub fn clear(&mut self) {
        self.ring.fill(Complex32::default());
        self.cursors.fill(0);
        self.start_hop();
        self.coherence.fill(1.0);
        self.sigma_hz.fill(0.0);
    }
}

// Integrate N(mean,sigma^2) against the same linear log-frequency hats as the
// point path. This is a continuous zero-width limit, including virtual end bins.
pub(super) fn gaussian_weights(
    space: &Log2Space,
    mean: f64,
    sigma: f64,
    mut add: impl FnMut(usize, f32),
) {
    assert!(sigma.is_finite() && sigma > 0.0);
    let step = f64::from(space.step()) * std::f64::consts::LN_2;
    let first = f64::from(space.centers_log2[0]) * std::f64::consts::LN_2;
    let left = (mean - 8.0 * sigma).max(f64::MIN_POSITIVE);
    let right = mean + 8.0 * sigma;
    if right <= 0.0 {
        return;
    }
    let n = space.n_bins() as i32;
    let low = (((left.ln() - first) / step).floor() as i32).max(-1);
    let high = (((right.ln() - first) / step).floor() as i32).min(n - 1);
    for i in low..=high {
        let log_lo = first + f64::from(i) * step;
        let lo = log_lo.exp();
        let hi = (log_lo + step).exp();
        let a = (lo - mean) / sigma;
        let b = (hi - mean) / sigma;
        let pa = (-0.5 * a * a).exp() / std::f64::consts::TAU.sqrt();
        let pb = (-0.5 * b * b).exp() / std::f64::consts::TAU.sqrt();
        let sqrt2 = std::f64::consts::SQRT_2;
        // Same-side erfc differences retain the small tail probabilities.
        let probability = if a >= 0.0 {
            0.5 * (libm::erfc(a / sqrt2) - libm::erfc(b / sqrt2))
        } else if b <= 0.0 {
            0.5 * (libm::erfc(-b / sqrt2) - libm::erfc(-a / sqrt2))
        } else {
            1.0 - 0.5 * libm::erfc(-a / sqrt2) - 0.5 * libm::erfc(b / sqrt2)
        };
        if probability <= 0.0 {
            continue;
        }
        let mut moments = [probability, pa - pb, 0.0, 0.0, 0.0];
        for k in 2..=4 {
            moments[k] = (k - 1) as f64 * moments[k - 2] + a.powi(k as i32 - 1) * pa
                - b.powi(k as i32 - 1) * pb;
        }
        let mid = 0.5 * (lo + hi);
        let offset = (mean - mid) / mid;
        let scale = sigma / mid;
        let mut powers = [0.0; 5];
        powers[0] = probability;
        const CHOOSE: [[f64; 5]; 5] = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 3.0, 1.0, 0.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ];
        for k in 1..=4 {
            for j in 0..=k {
                powers[k] +=
                    CHOOSE[k][j] * offset.powi((k - j) as i32) * scale.powi(j as i32) * moments[j];
            }
        }
        // Across one 12.5-cent bin, the omitted log-series remainder is <1.3e-13.
        let integrated_log = mid.ln() * probability + powers[1] - 0.5 * powers[2] + powers[3] / 3.0
            - 0.25 * powers[4];
        let upper = ((integrated_log - log_lo * probability) / step).clamp(0.0, probability);
        for (bin, weight) in [(i, probability - upper), (i + 1, upper)] {
            if (0..n).contains(&bin) {
                add(bin as usize, weight as f32);
            }
        }
    }
}

#[test]
fn lag_history_matches_direct_current_hop_sum() {
    let centers = [50.0, 1000.0];
    let mut history = SpreadHistory::new(&centers, 48000.0, 1.0 / 3.0);
    assert!(history.lag[0] > 512);
    let trace: Vec<_> = (0..4096)
        .map(|n| {
            centers.map(|f| {
                let t = n as f32 / 48000.0;
                Complex32::from_polar(1.0 + 0.6 * (TAU * 23.0 * t).sin(), TAU * f * t)
                    + Complex32::from_polar(0.5, TAU * (f + 15.0) * t + 0.3)
            })
        })
        .collect();
    for (n, z) in trace.iter().enumerate() {
        if n % 512 == 0 {
            history.start_hop();
        }
        history.observe(z);
    }
    history.finish(48000.0);
    for c in 0..centers.len() {
        let mut s = Complex64::default();
        let mut energy = [0.0; 2];
        for n in 3584..4096 {
            let now = trace[n][c];
            let past = trace[n - history.lag[c]][c];
            let z = Complex64::new(f64::from(now.re), f64::from(now.im));
            let p = Complex64::new(f64::from(past.re), f64::from(past.im));
            s += z * p.conj();
            energy[0] += z.norm_sqr();
            energy[1] += p.norm_sqr();
        }
        let expected = s.norm() / (energy[0] * energy[1]).sqrt();
        assert_eq!(history.coherence[c], expected);
        assert!(history.sigma_hz[c] > 0.0);
    }
    let mut copy = history.clone();
    history.observe(&trace[0]);
    copy.observe(&trace[0]);
    assert_eq!(history.ring, copy.ring);
    history.clear();
    history.finish(48000.0);
    assert_eq!(history.coherence, vec![1.0; 2]);
    assert_eq!(history.sigma_hz, vec![0.0; 2]);
}

#[test]
fn gaussian_matches_independent_quadrature_and_point_limit() {
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut worst = 0.0_f64;
    for mean in [100.0_f64, 1000.0, 4000.0] {
        for sigma in [mean * 1e-7, 0.01, 0.5, 4.0, mean * 0.1] {
            let mut actual = vec![0.0_f64; space.n_bins()];
            gaussian_weights(&space, mean, sigma, |i, w| actual[i] += f64::from(w));
            let mut reference = vec![0.0_f64; space.n_bins()];
            // Midpoint quadrature in standard-normal coordinates is independent
            // of the analytic truncated-moment integration used above.
            let dt = 16.0 / 65536.0;
            for k in 0..65536 {
                let t = -8.0 + (k as f64 + 0.5) * dt;
                let f = mean + sigma * t;
                if f <= 0.0 {
                    continue;
                }
                let pos = (f.log2() - f64::from(space.centers_log2[0])) / f64::from(space.step());
                let i = pos.floor() as i32;
                let fraction = pos - f64::from(i);
                let probability = (-0.5 * t * t).exp() / std::f64::consts::TAU.sqrt() * dt;
                for (j, w) in [(i, 1.0 - fraction), (i + 1, fraction)] {
                    if (0..space.n_bins() as i32).contains(&j) {
                        reference[j as usize] += probability * w;
                    }
                }
            }
            let error = actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            worst = worst.max(error);
            // The retained PCM gate's relative density budget protects consumers.
            assert!(
                error <= 1e-3,
                "Gaussian quadrature mean={mean}, sigma={sigma}, L1={error}"
            );
            assert!(actual.iter().all(|&w| w >= 0.0));
            let a: f64 = actual.iter().sum();
            let b: f64 = reference.iter().sum();
            assert!(
                (a - b).abs() <= 16.0 * f64::from(f32::EPSILON),
                "Gaussian mass conservation"
            );
            if sigma == mean * 1e-7 {
                let pos = ((mean as f32).log2() - space.centers_log2[0]) / space.step();
                let i = pos.floor() as usize;
                let fraction = pos - i as f32;
                let mut point = vec![0.0_f64; space.n_bins()];
                point[i] = f64::from(1.0 - fraction);
                point[i + 1] = f64::from(fraction);
                let error = actual
                    .iter()
                    .zip(point)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                assert!(
                    error <= 1e-3,
                    "continuous point limit within the existing density gate"
                );
            }
        }
    }
    println!("AUD_M1_GAUSSIAN_QUADRATURE max_relative_mass_l1={worst}");
}
