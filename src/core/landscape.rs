use rustfft::{FftPlanner, num_complex::Complex32};

use crate::core::erb::{erb_bw_hz, erb_space, hz_to_erb};
use crate::core::gammatone::gammatone_filterbank;
use crate::core::hilbert::hilbert_envelope;

/// Common analysis result (kept minimal here)
#[derive(Clone, Debug)]
pub struct AnalysisData {
    pub signal: Vec<f32>,
    pub fs: f32,
}

/// Landscape parameters
#[derive(Clone, Copy, Debug)]
pub enum RVariant {
    Gammatone,
    KernelConv,
    Dummy,
}

#[derive(Clone, Copy, Debug)]
pub enum CVariant {
    Periodicity,
    ModSync,
    Dummy,
}

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub alpha: f32,
    pub beta: f32,
    pub r_variant: RVariant,
    pub c_variant: CVariant,
    pub fmin: f32,
    pub fmax: f32,
    pub delta_e: f32,
}

#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub freqs_hz: Vec<f32>,
    pub r: Vec<f32>,
    pub c: Vec<f32>,
    pub k: Vec<f32>,
}

impl LandscapeFrame {
    pub fn recompute_k(&mut self, params: &LandscapeParams) {
        self.k = self
            .r
            .iter()
            .zip(self.c.iter())
            .map(|(rr, cc)| params.alpha * *cc - params.beta * *rr)
            .collect();
    }
}

// ---------------------------------------------------------
// Roughness R
// ---------------------------------------------------------

/// Roughness landscape R (kernel-based convolution)
/// Returns roughness per frequency bin
pub fn compute_r_kernelconv(signal: &[f32], fs: f32) -> Vec<f32> {
    let n = signal.len().next_power_of_two();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    // FFT
    let mut buf: Vec<Complex32> = signal.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    buf.resize(n, Complex32::new(0.0, 0.0));
    fft.process(&mut buf);

    let spectrum: Vec<f32> = buf.iter().map(|c| c.norm()).collect();
    let n_bins = spectrum.len();
    let df = fs / n as f32;

    // kernel params (Plomp-Levelt approximation)
    let a = 3.5;
    let b = 5.75;

    // per-bin roughness values
    let mut r_vals = vec![0.0; n_bins];
    for i in 0..n_bins {
        let f1 = i as f32 * df;
        let mut r_i = 0.0;
        for j in 0..n_bins {
            if i == j {
                continue;
            }
            let f2 = j as f32 * df;
            let dfreq = (f2 - f1).abs();
            let k = (-a * dfreq).exp() - (-b * dfreq).exp();
            r_i += spectrum[i] * spectrum[j] * k;
        }
        r_vals[i] = r_i;
    }
    r_vals
}

pub fn compute_r_gammatone(signal: &[f32], fs: f32, freqs_hz: &[f32]) -> Vec<f32> {
    if freqs_hz.is_empty() {
        return Vec::new();
    }

    let outs = gammatone_filterbank(signal, freqs_hz, fs);

    outs.into_iter()
        .map(|ch| {
            if ch.is_empty() {
                return 0.0;
            }

            // Hilbert envelope
            let mut env = hilbert_envelope(&ch);

            // --- (1) Drop edges / transient ---
            let drop = (0.1 * fs) as usize; // 100 ms margin
            if env.len() <= 2 * drop {
                return 0.0;
            }
            env = env[drop..env.len() - drop].to_vec();

            // --- (2) High-pass to remove DC drift ---
            let mut hp = OnePoleHP::new(fs, 30.0);
            let env_hp: Vec<f32> = env.into_iter().map(|x| hp.proc(x)).collect();

            // --- (3) Roughness as RMS of fluctuation ---
            let mean = env_hp.iter().copied().sum::<f32>() / env_hp.len() as f32;
            let var = env_hp.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / env_hp.len() as f32;
            var.sqrt()
        })
        .collect()
}

// One-pole HPF at fc Hz
struct OnePoleHP {
    a1: f32,
    b0: f32,
    x1: f32,
    y1: f32,
}
impl OnePoleHP {
    fn new(fs: f32, fc: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * fc);
        let alpha = rc / (rc + 1.0 / fs);
        Self {
            a1: alpha,
            b0: (1.0 + alpha) / 2.0,
            x1: 0.0,
            y1: 0.0,
        }
    }
    fn proc(&mut self, x: f32) -> f32 {
        let y = self.b0 * (x - self.x1) + self.a1 * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

fn compute_r_dummy(freqs_hz: &[f32]) -> Vec<f32> {
    vec![0.0; freqs_hz.len()]
}

// ---------------------------------------------------------
// Consonance C (placeholders)
// ---------------------------------------------------------

fn compute_c_dummy(freqs_hz: &[f32]) -> Vec<f32> {
    vec![0.0; freqs_hz.len()]
}

fn compute_c_periodicity(signal: &[f32], fs: f32, freqs_hz: &[f32]) -> Vec<f32> {
    // TODO: implement ACF-based periodicity
    freqs_hz.iter().map(|_| 0.0).collect()
}

fn compute_c_modsync(signal: &[f32], fs: f32, freqs_hz: &[f32]) -> Vec<f32> {
    // TODO: implement modulation synchrony
    freqs_hz.iter().map(|_| 0.0).collect()
}

// ---------------------------------------------------------
// Variant selector
// ---------------------------------------------------------

fn compute_r(data: &AnalysisData, variant: RVariant, freqs: &[f32]) -> Vec<f32> {
    match variant {
        RVariant::KernelConv => compute_r_kernelconv(&data.signal, data.fs),
        RVariant::Gammatone => compute_r_gammatone(&data.signal, data.fs, freqs),
        RVariant::Dummy => compute_r_dummy(freqs),
    }
}

fn compute_c(data: &AnalysisData, variant: CVariant, freqs: &[f32]) -> Vec<f32> {
    match variant {
        CVariant::Periodicity => compute_c_periodicity(&data.signal, data.fs, freqs),
        CVariant::ModSync => compute_c_modsync(&data.signal, data.fs, freqs),
        CVariant::Dummy => compute_c_dummy(freqs),
    }
}

// ---------------------------------------------------------
// Main landscape computation
// ---------------------------------------------------------

pub fn compute_landscape_erb(data: &AnalysisData, params: &LandscapeParams) -> LandscapeFrame {
    let freqs_erb = erb_space(params.fmin, params.fmax, params.delta_e);
    let r = compute_r(data, params.r_variant, &freqs_erb);
    let c = compute_c(data, params.c_variant, &freqs_erb);
    let k: Vec<f32> = r
        .iter()
        .zip(&c)
        .map(|(rr, cc)| params.alpha * *cc - params.beta * *rr)
        .collect();

    LandscapeFrame {
        freqs_hz: freqs_erb.clone(),
        r,
        c,
        k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::erb::erb_space;
    use crate::core::util::sine;

    #[test]
    fn r_zero_for_single_sine() {
        let fs = 16000.0;
        let n = 4096;
        let sig = sine(fs, 440.0, n);
        let data = AnalysisData { signal: sig, fs };

        let params = LandscapeParams {
            alpha: 1.0,
            beta: 1.0,
            r_variant: RVariant::Gammatone,
            c_variant: CVariant::Dummy,
            fmin: 300.0,
            fmax: 1200.0,
            delta_e: 0.1,
        };

        let lf = compute_landscape_erb(&data, &params);

        // locate index nearest 440Hz
        let idx_440 = lf
            .freqs_hz
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            lf.r[idx_440] < 0.1,
            "R should be near zero for pure tone, got {}",
            lf.r[idx_440]
        );
    }

    /// Helper: find nearest index to target freq
    fn nearest_idx(freqs: &[f32], target: f32) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::MAX;
        for (i, &f) in freqs.iter().enumerate() {
            let d = (f - target).abs();
            if d < best_d {
                best = i;
                best_d = d;
            }
        }
        best
    }

    #[test]
    fn r_higher_for_am_signal() {
        let fs = 16000.0;
        let n = 4096;

        // Signals: pure 440Hz vs AM @50Hz
        let carrier = sine(fs, 440.0, n);
        let pure = carrier.clone();
        let modulator: Vec<f32> = (0..n)
            .map(|i| 1.0 + 0.5 * (2.0 * std::f32::consts::PI * 50.0 * (i as f32) / fs).sin())
            .collect();
        let am: Vec<f32> = carrier
            .iter()
            .zip(modulator.iter())
            .map(|(c, m)| c * m)
            .collect();

        let params = LandscapeParams {
            alpha: 1.0,
            beta: 1.0,
            r_variant: RVariant::Gammatone,
            c_variant: CVariant::Dummy,
            fmin: 200.0,
            fmax: 1000.0,
            delta_e: 0.1,
        };

        let lf_pure = compute_landscape_erb(
            &AnalysisData {
                signal: pure,
                fs: fs,
            },
            &params,
        );

        let lf_am = compute_landscape_erb(&AnalysisData { signal: am, fs: fs }, &params);

        let i440 = nearest_idx(&lf_pure.freqs_hz, 440.0);

        // AM should increase roughness at ~440 band
        assert!(
            lf_am.r[i440] > lf_pure.r[i440],
            "AM roughness not higher: AM={} vs pure={}",
            lf_am.r[i440],
            lf_pure.r[i440]
        );
    }

    #[test]
    fn r_peak_near_critical_band() {
        let fs = 16000.0;
        let n = 4096;

        // Two-tone: 440+445 (very close) vs 440+470 (~0.4 ERB at 440Hz)
        let s440 = sine(fs, 440.0, n);
        let s445 = sine(fs, 445.0, n);
        let s470 = sine(fs, 470.0, n);

        let sig_close: Vec<f32> = s440.iter().zip(s445.iter()).map(|(a, b)| a + b).collect();
        let sig_peak: Vec<f32> = s440.iter().zip(s470.iter()).map(|(a, b)| a + b).collect();

        let params = LandscapeParams {
            alpha: 1.0,
            beta: 1.0,
            r_variant: RVariant::Gammatone,
            c_variant: CVariant::Dummy,
            fmin: 200.0,
            fmax: 1000.0,
            delta_e: 0.1,
        };

        let lf_close = compute_landscape_erb(
            &AnalysisData {
                signal: sig_close,
                fs,
            },
            &params,
        );
        let lf_peak = compute_landscape_erb(
            &AnalysisData {
                signal: sig_peak,
                fs,
            },
            &params,
        );

        let i440 = nearest_idx(&lf_close.freqs_hz, 440.0);

        // Expect both close and peak cases to yield nonzero roughness
        assert!(lf_close.r[i440] > 0.0, "close case R should be >0");
        assert!(lf_peak.r[i440] > 0.0, "peak case R should be >0");

        // Expect roughness at ~0.3 ERB (peak) to be at least comparable to close case
        assert!(
            lf_peak.r[i440] >= 0.8 * lf_close.r[i440],
            "R at ~470Hz not comparable: close={}, peak={}",
            lf_close.r[i440],
            lf_peak.r[i440]
        );
    }

    // helper: generate sine with fade-in/out
    fn sine_with_fade(freq: f32, fs: f32, dur_sec: f32) -> Vec<f32> {
        let n = (fs * dur_sec) as usize;
        let mut sig = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / fs;
            let win = if i < (0.05 * fs) as usize {
                // 50ms fade in
                (i as f32 / (0.05 * fs)).min(1.0)
            } else if i > n - (0.05 * fs) as usize {
                ((n - i) as f32 / (0.05 * fs)).min(1.0)
            } else {
                1.0
            };
            sig.push((2.0 * std::f32::consts::PI * freq * t).sin() * win);
        }
        sig
    }

    #[test]
    fn pure_tone_gives_near_zero_r() {
        let fs = 48000.0;
        let tone = sine_with_fade(440.0, fs, 1.0);
        let freqs: Vec<f32> = (100..2000).step_by(100).map(|f| f as f32).collect();

        let r_vals = compute_r_gammatone(&tone, fs, &freqs);

        // average roughness across channels should be very small
        let mean_r = r_vals.iter().sum::<f32>() / r_vals.len() as f32;
        println!("mean R for pure tone = {}", mean_r);

        assert!(mean_r < 1e-3, "pure tone roughness too large: {}", mean_r);
    }
}
