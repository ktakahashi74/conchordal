//! cochlea.rs — Stateful cochlear front-end for continuous landscape computation.
//! Keeps all filter/envelope states across blocks (no per-block transients).

use rustfft::num_complex::Complex32;
use std::f32::consts::PI;

use crate::core::erb::erb_bw_hz; // ERB bandwidth helper

// ============================== helpers ==============================

#[inline]
fn a_from_bw(fs: f32, bw_hz: f32) -> f32 {
    // Continuous pole at -2π*BW [rad/s] -> discrete 1-pole factor a.
    (-2.0 * PI * bw_hz / fs).exp()
}

// gate factor: 0 if mean < floor, ramps to 1 by 5*floor
fn smooth_gate(mean: f32, floor: f32) -> f32 {
    let t = (mean / floor - 1.0) / 4.0; // normalized 0..1
    t.clamp(0.0, 1.0)
}

// ---------------- Oscillator for heterodyne (stable multiplier) -----

#[derive(Clone, Copy, Debug)]
struct Osc {
    rot: Complex32, // e^{j w}
    osc: Complex32, // current phasor
}
impl Osc {
    fn new(fs: f32, fc: f32) -> Self {
        let w = 2.0 * PI * fc / fs;
        Self {
            rot: Complex32::new(w.cos(), w.sin()),
            osc: Complex32::new(1.0, 0.0),
        }
    }
    #[inline]
    fn reset_phase(&mut self) {
        self.osc = Complex32::new(1.0, 0.0);
    }
    #[inline]
    fn mix_down(&mut self, x: f32) -> Complex32 {
        // z = x * e^{-j w n}
        let z = self.osc.conj() * Complex32::new(x, 0.0);
        self.osc *= self.rot;
        z
    }
}

// ---------------- Complex one-pole LP (baseband analytic) -----------

#[derive(Clone, Copy, Debug)]
struct OnePoleLP {
    a: f32,       // 0 < a < 1
    y: Complex32, // state
}
impl OnePoleLP {
    fn new(a: f32) -> Self {
        Self {
            a,
            y: Complex32::new(0.0, 0.0),
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.y = Complex32::new(0.0, 0.0);
    }
    #[inline]
    fn process(&mut self, x: Complex32) -> Complex32 {
        self.y = self.a * self.y + (1.0 - self.a) * x;
        self.y
    }
}

// ---------------- Envelope filters (stateful IIR) -------------------

#[derive(Clone, Copy, Debug)]
struct HP1 {
    a0: f32,
    a1: f32,
    b1: f32,
    x1: f32,
    y1: f32,
}
impl HP1 {
    fn new(fs: f32, fc: f32) -> Self {
        let w = (PI * fc / fs).tan();
        let a0 = 1.0 / (1.0 + w);
        Self {
            a0,
            a1: -a0,
            b1: (1.0 - w) * a0,
            x1: 0.0,
            y1: 0.0,
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.a0 * x + self.a1 * self.x1 + self.b1 * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

#[derive(Clone, Copy, Debug)]
struct LP2 {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}
impl LP2 {
    fn new(fs: f32, fc: f32) -> Self {
        let w0 = (PI * fc / fs).tan();
        let k = w0;
        let a0 = (k * k + (2.0_f32).sqrt() * k + 1.0);
        let b0 = k * k / a0;
        let b1 = 2.0 * k * k / a0;
        let b2 = k * k / a0;
        let a1 = 2.0 * (k * k - 1.0) / a0;
        let a2 = (k * k - (2.0_f32).sqrt() * k + 1.0) / a0;
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

// ---------------- EWMA mean & RMS (low-latency meters) --------------

#[derive(Clone, Copy, Debug)]
struct EwMean {
    a: f32,
    m: f32,
}
impl EwMean {
    fn new(tc_samples: f32) -> Self {
        // alpha = 1 - exp(-1/tau)
        let a = 1.0 - (-1.0 / tc_samples).exp();
        Self { a, m: 0.0 }
    }
    #[inline]
    fn reset(&mut self) {
        self.m = 0.0;
    }
    #[inline]
    fn update(&mut self, x: f32) -> f32 {
        self.m += self.a * (x - self.m);
        self.m
    }
}

#[derive(Clone, Copy, Debug)]
struct EwRms {
    a: f32,
    p: f32,
}
impl EwRms {
    fn new(tc_samples: f32) -> Self {
        let a = 1.0 - (-1.0 / tc_samples).exp();
        Self { a, p: 0.0 }
    }
    #[inline]
    fn reset(&mut self) {
        self.p = 0.0;
    }
    #[inline]
    fn update(&mut self, x: f32) -> f32 {
        self.p += self.a * (x * x - self.p);
        self.p.sqrt()
    }
}

// =========================== one channel ============================

/// Stateful cochlear channel.
/// Path: heterodyne @ cf -> complex LP^4 (ERB) -> |.| envelope ->
/// mean-normalize -> HP(20 Hz) -> optional LP(300 Hz) -> EW RMS.
/// Output is a *running roughness* sample per input sample.
// --- 追加: フィールド
pub struct CochlearChan {
    // heterodyne & ERB LP
    osc: Osc,
    lp1: OnePoleLP,
    lp2: OnePoleLP,
    lp3: OnePoleLP,
    lp4: OnePoleLP,
    // envelope path
    mean: EwMean,
    hp: HP1,
    lp: Option<LP2>,
    rms: EwRms,
    // params
    pub cf: f32,
    // --- new: guard params
    mean_floor: f32, // abs floor to avoid blow-up when channel energy ~0
    e_clip: f32,     // safety limiter on modulation index
}

impl CochlearChan {
    pub fn new(fs: f32, cf: f32, use_lp_300hz: bool) -> Self {
        let bw = 1.019 * erb_bw_hz(cf);
        let a = a_from_bw(fs, bw);
        Self {
            osc: Osc::new(fs, cf),
            lp1: OnePoleLP::new(a),
            lp2: OnePoleLP::new(a),
            lp3: OnePoleLP::new(a),
            lp4: OnePoleLP::new(a),
            mean: EwMean::new(0.050 * fs), // ~50 ms
            hp: HP1::new(fs, 20.0),
            lp: if use_lp_300hz {
                Some(LP2::new(fs, 300.0))
            } else {
                None
            },
            rms: EwRms::new(0.150 * fs), // ~150 ms
            cf,
            // --- heuristics: tuned for input in [-1,1]
            mean_floor: 0.05, // envelope mean below this => treat as silence
            e_clip: 6.0,      // cap extreme ratios in edge cases
        }
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        // baseband analytic
        let z = self.osc.mix_down(x);
        let y1 = self.lp1.process(z);
        let y2 = self.lp2.process(y1);
        let y3 = self.lp3.process(y2);
        let y4 = self.lp4.process(y3);

        // envelope and modulation index
        let env = y4.norm();
        let mean = self.mean.update(env);

        // gate based on mean energy
        let gate = if mean <= self.mean_floor {
            0.0
        } else if mean >= 5.0 * self.mean_floor {
            1.0
        } else {
            (mean / self.mean_floor - 1.0) / 4.0
        };

        if gate == 0.0 {
            // silence channel
            let r = self.rms.update(0.0);
            let weight = 1.0 / (1.0 + (self.cf / 1500.0).powi(2));
            return r * weight;
        }

        // normalized envelope fluctuation
        let mut e = env / mean - 1.0;

        // modulation filtering
        e = self.hp.process(e);
        if let Some(lp) = self.lp.as_mut() {
            e = lp.process(e);
        }

        // safety limiter
        e = e.clamp(-self.e_clip, self.e_clip);

        // apply gate
        e *= gate;

        // running RMS + phase-locking roll-off
        let r = self.rms.update(e);
        let weight = 1.0 / (1.0 + (self.cf / 1500.0).powi(2));
        r * weight
    }

    /// Reset all internal states (use only at stream start).
    pub fn reset(&mut self) {
        self.osc.reset_phase();
        self.lp1.reset();
        self.lp2.reset();
        self.lp3.reset();
        self.lp4.reset();
        self.mean.reset();
        self.hp.reset();
        if let Some(l) = self.lp.as_mut() {
            l.reset();
        }
        self.rms.reset();
    }

    /// Process a block, returning one roughness sample per input sample.
    pub fn process_block(&mut self, x: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for &xi in x {
            out.push(self.process_sample(xi));
        }
        out
    }
}

// ============================== bank ================================

/// Bank of stateful cochlear channels.
/// Keep this object alive and feed consecutive blocks without resetting.
pub struct Cochlea {
    pub fs: f32,
    pub chans: Vec<CochlearChan>,
}

impl Cochlea {
    /// Build a bank. If `use_lp_300hz` is true, enable 300 Hz LP in modulation path.
    pub fn new(fs: f32, freqs_hz: &[f32], use_lp_300hz: bool) -> Self {
        let chans = freqs_hz
            .iter()
            .map(|&cf| CochlearChan::new(fs, cf, use_lp_300hz))
            .collect();
        Self { fs, chans }
    }

    /// Reset all channels (use only at stream start).
    pub fn reset(&mut self) {
        for ch in &mut self.chans {
            ch.reset();
        }
    }

    /// Process a block. Returns [channels][samples].
    pub fn process_block(&mut self, x: &[f32]) -> Vec<Vec<f32>> {
        self.chans
            .iter_mut()
            .map(|ch| ch.process_block(x))
            .collect()
    }

    /// Process a block and return per-channel *windowed mean* roughness (simple downmix).
    pub fn process_block_mean(&mut self, x: &[f32]) -> Vec<f32> {
        self.chans
            .iter_mut()
            .map(|ch| {
                let y = ch.process_block(x);
                if y.is_empty() {
                    0.0
                } else {
                    y.iter().sum::<f32>() / y.len() as f32
                }
            })
            .collect()
    }
}

// ============================== tests ===============================

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(fs: f32, f: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * f * (i as f32) / fs).sin())
            .collect()
    }

    #[test]
    fn pure_tone_yields_low_roughness_after_warmup() {
        let fs = 16000.0;
        let f0 = 440.0;
        let x = sine(fs, f0, 8192); // ~0.5 s

        let freqs = vec![300.0, 380.0, 420.0, 440.0, 470.0, 520.0];
        let mut coch = Cochlea::new(fs, &freqs, true);
        coch.reset();

        // process in small blocks to mimic streaming
        let mut last_mean = vec![0.0; freqs.len()];
        for chunk in x.chunks(256) {
            last_mean = coch.process_block_mean(chunk);
        }
        // warmup consumed: all channels should be ~0 (allow tiny residuals)
        for (i, r) in last_mean.iter().enumerate() {
            assert!(*r < 0.01, "ch {} roughness too high: {}", i, r);
        }
    }

    #[test]
    fn two_tone_nearby_shows_nonzero_roughness() {
        let fs = 16000.0;
        let n = 8192;
        let x: Vec<f32> = sine(fs, 440.0, n)
            .iter()
            .zip(sine(fs, 445.0, n).iter())
            .map(|(a, b)| a + b)
            .collect();

        let freqs = vec![430.0, 440.0, 450.0];
        let mut coch = Cochlea::new(fs, &freqs, true);
        coch.reset();

        let mut last = vec![0.0; freqs.len()];
        for chunk in x.chunks(256) {
            last = coch.process_block_mean(chunk);
        }
        assert!(last.iter().cloned().fold(0.0, f32::max) > 0.01);
    }
}
