//! Hop boxcars and bounded critical-band power queries.
use super::*;
use fearless_simd::{dispatch, f64x8};
use rustfft::num_complex::Complex64;

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct HopSum {
    energy: f32,
    correlation: Complex32,
}

#[derive(Clone)]
pub(super) struct StatisticsWindow {
    pub counts: Vec<usize>,
    offsets: Vec<usize>,
    cursors: Vec<usize>,
    ring: Vec<HopSum>,
    samples: usize,
}

impl StatisticsWindow {
    pub fn new(centers: &[f32], fs: f32, samples: usize) -> Self {
        let counts: Vec<_> = centers
            .iter()
            .map(|&f| {
                (f64::from(fs) / (samples as f64 * 0.75 * (24.7 + f64::from(f) / 9.265)))
                    .ceil()
                    .max(1.0) as usize
            })
            .collect();
        let mut offsets = Vec::with_capacity(counts.len());
        let mut total = 0;
        for &n in &counts {
            offsets.push(total);
            total += n;
        }
        Self {
            cursors: vec![0; counts.len()],
            counts,
            offsets,
            ring: vec![HopSum::default(); total],
            samples,
        }
    }

    pub fn push(&mut self, energy: &mut [f32], correlation: &mut [Complex32], samples: usize) {
        assert_eq!(
            samples, self.samples,
            "boxcar requires the registered hop geometry"
        );
        for c in 0..self.counts.len() {
            self.ring[self.offsets[c] + self.cursors[c]] = HopSum {
                energy: energy[c],
                correlation: correlation[c],
            };
            self.cursors[c] = (self.cursors[c] + 1) % self.counts[c];
            let mut e = 0.0_f64;
            let mut s = Complex64::default();
            for value in &self.ring[self.offsets[c]..self.offsets[c] + self.counts[c]] {
                e += f64::from(value.energy);
                s += Complex64::new(
                    f64::from(value.correlation.re),
                    f64::from(value.correlation.im),
                );
            }
            energy[c] = e as f32;
            correlation[c] = Complex32::new(s.re as f32, s.im as f32);
        }
    }

    pub fn clear(&mut self) {
        self.ring.fill(HopSum::default());
        self.cursors.fill(0);
    }
}

#[derive(Clone)]
pub(super) struct ErbContext {
    first: f64,
    step: f64,
    centers: Vec<f64>,
    moments: Vec<[f64; 7]>,
    // The first `occupied_len` entries are the ascending buckets that received a deposit
    // this hop; all other buckets are zero. Sized once per channel so clones never grow.
    occupied: Vec<usize>,
    occupied_len: usize,
    fs: f32,
}

impl ErbContext {
    pub fn new(fs: f32, channels: usize) -> Self {
        let first = erbn(1.0);
        let step = 1.0 / 32.0;
        let bins = ((erbn(f64::from(fs / 2.0 - 1.0)) - first) / step).floor() as usize + 1;
        Self {
            first,
            step,
            centers: (0..bins)
                .map(|i| inv_erbn(first + (i as f64 + 0.5) * step))
                .collect(),
            moments: vec![[0.0; 7]; bins],
            occupied: vec![0; channels],
            occupied_len: 0,
            fs,
        }
    }

    pub fn prepare(&mut self, frequencies: &[f32], power: &[f32]) {
        for &i in &self.occupied[..self.occupied_len] {
            self.moments[i] = [0.0; 7];
        }
        let mut len = 0;
        for (&f, &d) in frequencies.iter().zip(power) {
            let f = f64::from(f.clamp(1.0, self.fs / 2.0 - 1.0));
            let i =
                (((erbn(f) - self.first) / self.step).floor() as usize).min(self.moments.len() - 1);
            self.occupied[len] = i;
            len += 1;
            let delta = f - self.centers[i];
            let mut value = f64::from(d);
            for m in &mut self.moments[i] {
                *m += value;
                value *= delta;
            }
        }
        let occupied = &mut self.occupied[..len];
        occupied.sort_unstable();
        self.occupied_len = 0;
        for k in 0..len {
            if k == 0 || occupied[k] != occupied[self.occupied_len - 1] {
                occupied[self.occupied_len] = occupied[k];
                self.occupied_len += 1;
            }
        }
    }

    /// Critical-band power at every frequency, one query per lane. Each lane visits its
    /// occupied buckets in the ascending order of a full scan and repeats the scalar
    /// f64 operations of the per-query reference (`power_at`, a test-only full scan), so
    /// the result is bit-identical. Exhausted lanes add zero terms, which leave a sum
    /// that started at +0 unchanged.
    pub fn powers_into(&self, frequencies: &[f32], out: &mut [f32]) {
        assert_eq!(frequencies.len(), out.len());
        dispatch!(Level::new(), simd => self.powers_lanes(simd, frequencies, out));
    }

    #[inline(always)]
    pub(super) fn powers_lanes<S: Simd>(&self, simd: S, frequencies: &[f32], out: &mut [f32]) {
        let splat = |value: f64| f64x8::splat(simd, value);
        for (queries, out) in frequencies.chunks(8).zip(out.chunks_mut(8)) {
            let (mut f, mut b) = ([1.0; 8], [1.0; 8]);
            let mut scales = [[1.0; 8]; 7];
            let mut ranges = [(0, 0); 8];
            for (k, &frequency) in queries.iter().enumerate() {
                f[k] = f64::from(frequency.clamp(1.0, self.fs / 2.0 - 1.0));
                let e = erbn(f[k]);
                let low = (((e - 4.0 - self.first) / self.step).floor() as isize).max(0) as usize;
                let high = (((e + 4.0 - self.first) / self.step).floor() as usize)
                    .min(self.moments.len() - 1);
                b[k] = 1.0186 * (24.7 + f[k] / 9.265);
                for n in 1..7 {
                    scales[n][k] = scales[n - 1][k] / b[k];
                }
                let occupied = &self.occupied[..self.occupied_len];
                ranges[k] = (
                    occupied.partition_point(|&i| i < low),
                    occupied.partition_point(|&i| i <= high),
                );
            }
            let steps = ranges
                .iter()
                .map(|&(start, end)| end - start)
                .max()
                .unwrap_or(0);
            let fv = f64x8::load_array(simd, f);
            let bv = f64x8::load_array(simd, b);
            let sv = scales.map(|lanes| f64x8::load_array(simd, lanes));
            let mut power = splat(0.0);
            for j in 0..steps {
                // Exhausted or empty lanes sit at their own frequency with zero moments.
                let mut center = f;
                let mut moments = [[0.0; 8]; 7];
                for (k, &(start, end)) in ranges.iter().enumerate() {
                    if start + j < end {
                        let i = self.occupied[start + j];
                        if self.moments[i][0] != 0.0 {
                            center[k] = self.centers[i];
                            for n in 0..7 {
                                moments[n][k] = self.moments[i][n];
                            }
                        }
                    }
                }
                let q = (f64x8::load_array(simd, center) - fv) / bv;
                let a = splat(1.0) + q * q;
                let mut coefficients = [splat(0.0); 7];
                coefficients[0] = splat(1.0);
                for n in 0..6 {
                    let previous = if n > 0 {
                        splat((n + 7) as f64) * coefficients[n - 1]
                    } else {
                        splat(0.0)
                    };
                    coefficients[n + 1] =
                        -(splat(2.0) * q * splat((n + 4) as f64) * coefficients[n] + previous)
                            / (a * splat((n + 1) as f64));
                }
                // `powi(4)` lowers to these two squarings.
                let a2 = a * a;
                let a4 = a2 * a2;
                for n in 0..7 {
                    power += coefficients[n] * f64x8::load_array(simd, moments[n]) * sv[n] / a4;
                }
            }
            for (out, power) in out.iter_mut().zip(power.to_array()) {
                *out = power.max(0.0) as f32;
            }
        }
    }

    #[cfg(test)]
    pub fn power_at(&self, frequency: f32) -> f32 {
        let f = f64::from(frequency.clamp(1.0, self.fs / 2.0 - 1.0));
        let e = erbn(f);
        // Boundary buckets are retained whole, so every omitted deposit is
        // strictly more than four ERB-number units away.
        let low = (((e - 4.0 - self.first) / self.step).floor() as isize).max(0) as usize;
        let high =
            (((e + 4.0 - self.first) / self.step).floor() as usize).min(self.moments.len() - 1);
        let b = 1.0186 * (24.7 + f / 9.265);
        let mut power = 0.0_f64;
        for i in low..=high {
            if self.moments[i][0] == 0.0 {
                continue;
            }
            let q = (self.centers[i] - f) / b;
            let a = 1.0 + q * q;
            let mut coefficients = [0.0; 7];
            coefficients[0] = 1.0;
            for n in 0..6 {
                coefficients[n + 1] = -(2.0 * q * (n + 4) as f64 * coefficients[n]
                    + if n > 0 {
                        (n + 7) as f64 * coefficients[n - 1]
                    } else {
                        0.0
                    })
                    / (a * (n + 1) as f64);
            }
            let mut scale = 1.0;
            for (coefficient, moment) in coefficients.iter().zip(&self.moments[i]) {
                power += coefficient * moment * scale / a.powi(4);
                scale /= b;
            }
        }
        power.max(0.0) as f32
    }

    #[cfg(test)]
    pub fn omitted_weight_bound(frequency: f32) -> f64 {
        let f = f64::from(frequency);
        let b = 1.0186 * (24.7 + f / 9.265);
        let lower = inv_erbn(erbn(f) - 4.0);
        (1.0 + ((lower - f) / b).powi(2)).powi(-4)
    }
}

#[cfg(test)]
#[test]
fn erb_context_matches_direct_sum_and_isolated_power() {
    let mut context = ErbContext::new(48000.0, 137);
    let frequencies: Vec<_> = (0..137)
        .map(|i| inv_erbn(erbn(1.0) + i as f64 / 136.0 * (erbn(23999.0) - erbn(1.0))) as f32)
        .collect();
    let power: Vec<_> = (0..137).map(|i| 0.01 / (1 + i % 7) as f32).collect();
    context.prepare(&frequencies, &power);
    let total = power.iter().map(|&d| f64::from(d)).sum::<f64>();
    let mut maximum = 0.0_f64;
    for &x in &frequencies {
        let b = 1.0186 * (24.7 + f64::from(x) / 9.265);
        let exact = frequencies
            .iter()
            .zip(&power)
            .map(|(&y, &d)| {
                f64::from(d) * (1.0 + ((f64::from(y) - f64::from(x)) / b).powi(2)).powi(-4)
            })
            .sum::<f64>();
        let error = (f64::from(context.power_at(x)) - exact).abs();
        maximum = maximum.max(error / total);
        // The sixth-order moment tail is <3e-8 times total power. f32 output
        // rounding adds one epsilon; omitted deposits use the analytic tail bound.
        assert!(
            error / total <= ErbContext::omitted_weight_bound(x) + 3e-8 + f64::from(f32::EPSILON)
        );
    }
    for f in [55.0, 110.0, 1000.0, 8000.0] {
        context.prepare(&[f; 5], &[0.001; 5]);
        assert!(
            (context.power_at(f) / 0.005 - 1.0).abs() <= 2.0 * f32::EPSILON,
            "isolated component power"
        );
    }
    println!("AUD_M1_ERB_CONTEXT direct_sum_max_error_over_total_power={maximum}");
}

#[cfg(test)]
#[test]
fn boxcar_matches_explicit_hop_history_and_body_skip() {
    let (kernel, params) = crate::runtime::body_metabolism_test_core();
    for beta in [0.25, 1.0 / 3.0] {
        for erb in [false, true] {
            let plain = ShadowFrontEnd::new(
                kernel.space().clone(),
                params.fs,
                beta,
                false,
                kernel.nfft(),
            );
            let mut every = plain.clone().with_statistics(kernel.hop(), erb);
            let counts = every.bank.statistics.as_ref().unwrap().counts.clone();
            let mut raw = plain;
            let mut sparse = every.clone();
            let mut history = Vec::new();
            for frame in 0..24 {
                let pcm: Vec<_> = (0..kernel.hop())
                    .map(|i| {
                        let t = (frame * kernel.hop() + i) as f64 / f64::from(params.fs);
                        (0.1 * (std::f64::consts::TAU * 110.0 * t).cos()
                            + 0.05 * (std::f64::consts::TAU * 150.0 * t + 0.7).cos())
                            as f32
                    })
                    .collect();
                raw.process(&pcm, &params);
                every.process(&pcm, &params);
                history.push((raw.bank.energy.clone(), raw.bank.correlation.clone()));
                for (c, &count) in counts.iter().enumerate() {
                    let first = history.len().saturating_sub(count);
                    let e = history[first..]
                        .iter()
                        .map(|h| f64::from(h.0[c]) * (kernel.hop() as f64 / 2.0))
                        .sum::<f64>() as f32;
                    let s = history[first..]
                        .iter()
                        .map(|h| Complex64::new(f64::from(h.1[c].re), f64::from(h.1[c].im)))
                        .sum::<Complex64>();
                    // Normalizing raw hop means and summing raw energy differ only
                    // by the documented f32 scaling of each stored hop.
                    let expected = e * 2.0 / (kernel.hop() * count) as f32;
                    assert!(
                        (every.bank.energy[c] - expected).abs()
                            <= 4.0 * f32::EPSILON * expected.abs().max(f32::MIN_POSITIVE)
                    );
                    assert_eq!(
                        every.bank.correlation[c],
                        Complex32::new(s.re as f32, s.im as f32)
                    );
                }
                if frame % 4 == 3 {
                    sparse.process_body_gap(&pcm, 4, &params);
                    assert_eq!(every.bank.state, sparse.bank.state);
                    assert_eq!(
                        every.bank.statistics.as_ref().unwrap().ring,
                        sparse.bank.statistics.as_ref().unwrap().ring
                    );
                    assert_eq!(every.bank.correlation, sparse.bank.correlation);
                    assert_eq!(every.channel_power, sparse.channel_power);
                    assert_eq!(every.channel_mass, sparse.channel_mass);
                } else {
                    sparse.skip(&pcm);
                }
            }
            let mut clone = every.clone();
            let silence = vec![0.0; kernel.hop()];
            every.process(&silence, &params);
            clone.process(&silence, &params);
            assert_eq!(every.density_scan, clone.density_scan);
            clone.reset();
            assert!(
                clone
                    .bank
                    .statistics
                    .as_ref()
                    .unwrap()
                    .ring
                    .iter()
                    .all(|h| *h == HopSum::default())
            );
            assert!(!clone.process(&silence, &params));
        }
    }
}
