//! cochlea.rs — Stateful cochlear front-end for continuous landscape computation.
//! Keeps all filter/envelope states across blocks (no per-block transients).

use rustfft::num_complex::Complex32;

use ringbuf::StaticRb;
use ringbuf::traits::{Consumer, RingBuffer};

use std::f32::consts::PI;
use wide::f32x4;

use crate::core::erb::{ErbSpace, erb_bw_hz}; // ERB bandwidth helper

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

// ---------------- EW mean of unit phasor (PLV tracker) --------------
#[derive(Clone, Copy, Debug)]
struct EwPhasor {
    a: f32,       // smoothing alpha
    m: Complex32, // complex mean of unit phasors
}
impl EwPhasor {
    fn new(tc_samples: f32) -> Self {
        // alpha = 1 - exp(-1/tau)
        let a = 1.0 - (-1.0 / tc_samples).exp();
        Self {
            a,
            m: Complex32::new(0.0, 0.0),
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.m = Complex32::new(0.0, 0.0);
    }
    #[inline]
    fn update_unit_opt(&mut self, u: Option<Complex32>) -> f32 {
        // If None, decay toward 0 (no evidence / gated silence).
        let target = u.unwrap_or(Complex32::new(0.0, 0.0));
        self.m = self.m + self.a * (target - self.m);
        // PLV in [0,1]
        (self.m.re * self.m.re + self.m.im * self.m.im).sqrt()
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
    plv: EwPhasor,
    // params
    pub cf: f32,
    // --- new: guard params
    mean_floor: f32, // abs floor to avoid blow-up when channel energy ~0
    e_clip: f32,     // safety limiter on modulation index
    pub complex_out: Complex32,
    pub complex_buf: StaticRb<Complex32, 512>,
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
            rms: EwRms::new(0.150 * fs),    // ~150 ms
            plv: EwPhasor::new(0.100 * fs), // ~100 ms for phase stability
            cf,
            // --- heuristics: tuned for input in [-1,1]
            mean_floor: 0.05, // envelope mean below this => treat as silence
            e_clip: 6.0,      // cap extreme ratios in edge cases
            complex_out: Complex32::new(0.0, 0.0),
            complex_buf: StaticRb::<Complex32, 512>::default(),
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

    /// Phase-locking-based consonance (C_pl) per sample.
    /// Uses unit phasor of complex baseband (post-ERB) and EW mean magnitude.
    #[inline]
    pub fn process_sample_c_pl(&mut self, x: f32) -> f32 {
        // Front-end to complex baseband
        let z = self.osc.mix_down(x);
        self.complex_out = z;
        let y1 = self.lp1.process(z);
        let y2 = self.lp2.process(y1);
        let y3 = self.lp3.process(y2);
        let y4 = self.lp4.process(y3);

        let _ = self.complex_buf.push_overwrite(y4);
        self.complex_out = y4;

        let env = y4.norm();
        let mean = self.mean.update(env);

        // energy gate (same policy as other paths)
        let gate = if mean <= self.mean_floor {
            0.0
        } else if mean >= 5.0 * self.mean_floor {
            1.0
        } else {
            (mean / self.mean_floor - 1.0) / 4.0
        };

        // unit phasor when reliable, otherwise decay
        let plv = if gate == 0.0 || env <= 1e-20 {
            self.plv.update_unit_opt(None)
        } else {
            let inv = 1.0 / env.max(1e-20);
            let u = Complex32::new(y4.re * inv, y4.im * inv);
            self.plv.update_unit_opt(Some(u))
        };

        // phase-locking roll-off to reduce HF dominance
        let weight = 1.0 / (1.0 + (self.cf / 1500.0).powi(2));
        plv * weight
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
        self.plv.reset();
        self.complex_buf.clear();
        self.complex_out = Complex32::new(0.0, 0.0);
    }

    /// Process a block, returning one roughness sample per input sample.
    pub fn process_block(&mut self, x: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for &xi in x {
            out.push(self.process_sample(xi));
        }
        out
    }

    /// Per-block C_pl mean (downmixed to one value per channel).
    pub fn process_block_c_pl_mean(&mut self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0;
        for &xi in x {
            acc += self.process_sample_c_pl(xi);
        }
        acc / x.len() as f32
    }
}

// ============================== bank ================================

/// Bank of stateful cochlear channels.
/// Keep this object alive and feed consecutive blocks without resetting.
pub struct Cochlea {
    pub fs: f32,
    pub erb_space: ErbSpace,
    use_lp_300hz: bool,
    pub chans: Vec<CochlearChan>,
    pub plv_pairs: Vec<Vec<EwPhasor>>,
}

impl Cochlea {
    /// Build a bank. If `use_lp_300hz` is true, enable 300 Hz LP in modulation path.
    pub fn new(fs: f32, erb_space: ErbSpace, use_lp_300hz: bool) -> Self {
        let chans = erb_space
            .freqs_hz()
            .iter()
            .map(|&cf| CochlearChan::new(fs, cf, use_lp_300hz))
            .collect();
        let n = erb_space.len();
        let plv_pairs = (0..n)
            .map(|_| (0..n).map(|_| EwPhasor::new(0.100 * fs)).collect())
            .collect();
        Self {
            fs,
            erb_space,
            use_lp_300hz,
            chans,
            plv_pairs,
        }
    }

    /// Reset all channels (use only at stream start).
    pub fn reset(&mut self) {
        for ch in &mut self.chans {
            ch.reset();
        }
    }

    pub fn n_ch(&self) -> usize {
        self.chans.len()
    }

    /// Process a block. Returns [channels][samples].
    pub fn process_block(&mut self, x: &[f32]) -> Vec<Vec<f32>> {
        self.chans
            .iter_mut()
            .map(|ch| ch.process_block(x))
            .collect()
    }

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

    /// Process a block and return per-channel C_pl (phase-locking consonance) means.
    pub fn process_block_c_pl(&mut self, x: &[f32]) -> Vec<f32> {
        self.chans
            .iter_mut()
            .map(|ch| ch.process_block_c_pl_mean(x))
            .collect()
    }

    pub fn process_block_all(&mut self, x: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let ch = self.chans.len();
        if ch == 0 || x.is_empty() {
            return (vec![], vec![vec![]]);
        }

        // --- per-sample update (C_pl accumulation) ---
        let mut acc_c = vec![0.0f32; ch];
        for &sample in x {
            for (ci, ch) in self.chans.iter_mut().enumerate() {
                acc_c[ci] += ch.process_sample_c_pl(sample);
            }
        }
        for v in acc_c.iter_mut() {
            *v /= x.len() as f32;
        }

        // --- instantaneous PLV matrix snapshot ---
        let plv = self.current_plv_matrix();

        (acc_c, plv)
    }

    /// Unified cochlear step: update all channels once and return envelope & PLV matrix.
    /// This replaces separate process_block_mean() and process_block_all() calls.
    pub fn process_block_core(&mut self, x: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let n_ch = self.chans.len();
        if n_ch == 0 || x.is_empty() {
            return (vec![], vec![vec![]]);
        }

        // Envelope per channel (mean amplitude after 50ms EW)
        let mut env_vec = vec![0.0f32; n_ch];
        for (ci, ch) in self.chans.iter_mut().enumerate() {
            let mut acc = 0.0;
            for &xi in x {
                // process_sample_c_pl updates both complex & mean state
                let z = ch.osc.mix_down(xi);
                let y1 = ch.lp1.process(z);
                let y2 = ch.lp2.process(y1);
                let y3 = ch.lp3.process(y2);
                let y4 = ch.lp4.process(y3);
                let env = y4.norm();
                acc += env;
                ch.mean.update(env);
                let _ = ch.complex_buf.push_overwrite(y4);
                ch.complex_out = y4;
            }
            env_vec[ci] = ch.complex_out.norm(); //acc / x.len() as f32;
        }

        // After updating all channels, build PLV matrix
        let plv_mat = self.current_plv_matrix();
        (env_vec, plv_mat)
    }

    /// Compute phase-locking-based consonance vector from PLV matrix.
    /// Simple per-channel mean of PLV magnitudes (excluding self).
    pub fn compute_c_from_plv(&self, plv: &[Vec<f32>]) -> Vec<f32> {
        let n = plv.len();
        if n == 0 {
            return vec![];
        }
        let mut c = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += plv[i][j];
                }
            }
            c[i] = sum / (n - 1).max(1) as f32;
        }
        c
    }

    /// Return instantaneous PLV matrix across all cochlear channels.
    /// Uses each channel's current EwPhasor (phase mean) or internal buffer.
    pub fn current_plv_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.chans.len();
        let mut mat = vec![vec![0.0; n]; n];
        if n == 0 {
            return mat;
        }

        for i in 0..n {
            for j in i..n {
                // simple PLV estimate using complex buffer
                let buf_i = self.chans[i]
                    .complex_buf
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let buf_j = self.chans[j]
                    .complex_buf
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let v = Self::compute_plv_fast(&buf_i, &buf_j);
                mat[i][j] = v;
                mat[j][i] = v;
            }
        }
        mat
    }

    /// Fast Phase-Locking Value (PLV) between two complex signals.
    /// Avoids atan2; uses normalized complex dot products.
    /// Returns PLV in [0, 1].
    pub fn compute_plv_fast(sig_a: &[Complex32], sig_b: &[Complex32]) -> f32 {
        use wide::f32x4;

        let n = sig_a.len().min(sig_b.len());
        if n == 0 {
            return 0.0;
        }

        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        let chunk = 4;
        let end = n - (n % chunk);
        let mut i = 0;

        // --- SIMD loop (4 samples at once) ---
        while i < end {
            let a_re = f32x4::from([
                sig_a[i].re,
                sig_a[i + 1].re,
                sig_a[i + 2].re,
                sig_a[i + 3].re,
            ]);
            let a_im = f32x4::from([
                sig_a[i].im,
                sig_a[i + 1].im,
                sig_a[i + 2].im,
                sig_a[i + 3].im,
            ]);
            let b_re = f32x4::from([
                sig_b[i].re,
                sig_b[i + 1].re,
                sig_b[i + 2].re,
                sig_b[i + 3].re,
            ]);
            let b_im = f32x4::from([
                sig_b[i].im,
                sig_b[i + 1].im,
                sig_b[i + 2].im,
                sig_b[i + 3].im,
            ]);

            // normalize magnitudes
            let denom = ((a_re * a_re + a_im * a_im) * (b_re * b_re + b_im * b_im))
                .sqrt()
                .max(f32x4::splat(1e-12));

            let a_re_n = a_re / denom;
            let a_im_n = a_im / denom;

            // complex product: (a/|a|)*(b*/|b|)
            let re_term = a_re_n * b_re + a_im_n * b_im;
            let im_term = a_im_n * b_re - a_re_n * b_im;

            sum_re += re_term.reduce_add();
            sum_im += im_term.reduce_add();

            i += chunk;
        }

        // --- scalar tail ---
        for k in end..n {
            let a = sig_a[k];
            let b = sig_b[k];
            let denom = (a.norm_sqr() * b.norm_sqr()).sqrt().max(1e-12);
            let re_i = a.re / denom;
            let im_i = a.im / denom;
            sum_re += re_i * b.re + im_i * b.im;
            sum_im += im_i * b.re - re_i * b.im;
        }

        // PLV magnitude
        let norm = (sum_re * sum_re + sum_im * sum_im).sqrt();
        (norm / n as f32).clamp(0.0, 1.0)
    }

    /// Return per-channel latest C_pl values (EwPhasor magnitude).
    pub fn current_c_pl_vector(&self) -> Vec<f32> {
        self.chans
            .iter()
            .map(|ch| {
                let m = ch.plv.m;
                (m.re * m.re + m.im * m.im).sqrt()
            })
            .collect()
    }

    /// Return per-channel roughness RMS estimates.
    pub fn current_r_vector(&self) -> Vec<f32> {
        self.chans.iter().map(|ch| ch.rms.p.sqrt()).collect()
    }

    /// Return per-channel current envelope power (mean amplitude after EW integration).
    /// Used as input to potential-R computation.
    pub fn current_envelope_levels(&self) -> Vec<f32> {
        self.chans.iter().map(|ch| ch.mean.m.max(0.0)).collect()
    }

    fn update_pairwise_plv_once(&mut self) {
        let n = self.chans.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let cf_i = self.chans[i].cf;
                let cf_j = self.chans[j].cf;
                if (cf_i - cf_j).abs() > 1.5 * erb_bw_hz(cf_i) {
                    continue; // skip far bands
                }
                let a = self.chans[i].complex_out;
                let b = self.chans[j].complex_out;
                if a.norm_sqr() > 1e-12 && b.norm_sqr() > 1e-12 {
                    let da = a / a.norm();
                    let db = b / b.norm();
                    let u = da * db.conj();
                    self.plv_pairs[i][j].update_unit_opt(Some(u));
                } else {
                    self.plv_pairs[i][j].update_unit_opt(None);
                }
            }
        }
    }
}

// ============================== tests ===============================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::erb::ErbSpace;
    use std::f32::consts::PI;

    fn sine(fs: f32, f: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * f * (i as f32) / fs).sin())
            .collect()
    }

    // #[test]
    // fn pure_tone_yields_low_roughness_after_warmup() {
    //     let fs = 16000.0;
    //     let f0 = 440.0;
    //     let x = sine(fs, f0, 8192); // ~0.5 s

    //     // ERB範囲を狭く取って、テスト用チャンネルを再現
    //     let erb_space = ErbSpace::new(300.0, 520.0, 0.4);
    //     let mut coch = Cochlea::new(fs, erb_space, true);
    //     coch.reset();

    //     let n_ch = coch.n_ch();
    //     let mut last_mean = vec![0.0; n_ch];
    //     for chunk in x.chunks(256) {
    //         last_mean = coch.process_block_mean(chunk);
    //     }

    //     for (i, r) in last_mean.iter().enumerate() {
    //         assert!(*r < 0.01, "ch {} roughness too high: {}", i, r);
    //     }
    // }

    #[test]
    fn two_tone_nearby_shows_nonzero_roughness() {
        let fs = 16000.0;
        let n = 8192;
        let x: Vec<f32> = sine(fs, 440.0, n)
            .iter()
            .zip(sine(fs, 445.0, n).iter())
            .map(|(a, b)| a + b)
            .collect();

        let erb_space = ErbSpace::new(430.0, 450.0, 0.25);
        let mut coch = Cochlea::new(fs, erb_space, true);
        coch.reset();

        let n_ch = coch.n_ch();
        let mut last = vec![0.0; n_ch];
        for chunk in x.chunks(256) {
            last = coch.process_block_mean(chunk);
        }

        assert!(
            last.iter().cloned().fold(0.0, f32::max) > 0.01,
            "Roughness unexpectedly low for beating tones"
        );
    }

    #[test]
    fn c_pl_high_for_single_tone_lower_for_beats() {
        let fs = 16000.0;
        let n = 8192;
        let single = sine(fs, 440.0, n);
        let beats: Vec<f32> = single
            .iter()
            .zip(sine(fs, 445.0, n).iter())
            .map(|(a, b)| a + b)
            .collect();

        let erb_space = ErbSpace::new(440.0, 440.0, 0.1); // 単一chでもERB扱いで生成可
        let mut coch = Cochlea::new(fs, erb_space.clone(), true);

        coch.reset();
        let mut c_single = 0.0;
        for chunk in single.chunks(256) {
            c_single = coch.process_block_c_pl(chunk)[0];
        }

        coch.reset();
        let mut c_beats = 0.0;
        for chunk in beats.chunks(256) {
            c_beats = coch.process_block_c_pl(chunk)[0];
        }

        assert!(
            c_single > c_beats,
            "C_pl single {} <= beats {}",
            c_single,
            c_beats
        );
    }
}
