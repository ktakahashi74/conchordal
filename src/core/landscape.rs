//! core/landscape.rs — Landscape computed on log2(NSGT) domain.
//! Each frame integrates time–frequency features from analytic bands
//! (NSGT output) to compute potential roughness (R) and consonance (C).

use rustfft::num_complex::Complex32;

use crate::core::consonance_kernel::ConsonanceKernel;
use crate::core::nsgt::{BandCoeffs, NsgtLog2}; // NSGT log2-domain transform
use crate::core::roughness_kernel::{RoughnessKernel, potential_r_from_log2_spectrum};

#[derive(Clone, Copy, Debug)]
pub enum RVariant {
    NsgtKernel, // NSGT + Δlog2 kernel convolution
    Dummy,
}

#[derive(Clone, Copy, Debug)]
pub enum CVariant {
    NsgtPhaseLock, // PLV-like consonance via phase correlation
    Dummy,
}

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub hop_s: f32, // hop size [s] (e.g. 0.01)
    pub max_hist_cols: usize,
    pub alpha: f32,
    pub beta: f32,
    pub r_variant: RVariant,
    pub c_variant: CVariant,
    pub roughness_kernel: RoughnessKernel,
    pub consonance_kernel: ConsonanceKernel,
}

#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub fs: f32,
    pub freqs_hz: Vec<f32>,
    pub r_last: Vec<f32>,
    pub c_last: Vec<f32>,
    pub k_last: Vec<f32>,
    pub amp_last: Vec<f32>,
    pub r_hist: Vec<Vec<f32>>,
    pub k_hist: Vec<Vec<f32>>,
}

pub struct Landscape {
    nsgt: NsgtLog2,
    params: LandscapeParams,
    last_r: Vec<f32>,
    last_c: Vec<f32>,
    last_k: Vec<f32>,
    amp_last: Vec<f32>,
    analytic_last: Option<Vec<Complex32>>,
}

impl Landscape {
    pub fn new(params: LandscapeParams, nsgt: NsgtLog2) -> Self {
        let n_ch = nsgt.freqs_hz().len();
        Self {
            nsgt,
            params,
            last_r: vec![0.0; n_ch],
            last_c: vec![0.0; n_ch],
            last_k: vec![0.0; n_ch],
            amp_last: vec![0.0; n_ch],
            analytic_last: None,
        }
    }

    /// Process one hop frame: NSGT update + R/C/K compute.
    pub fn process_frame(&mut self, x: &[f32]) {
        // --- 1. NSGT analysis (single pass) ---
        // 各バンドに必ず値を持たせる：係数が無ければ 0 を入れる
        let bands = self.nsgt.analyze(x);

        // 代表複素数（中心フレーム or 0）
        let analytic_flat: Vec<Complex32> = bands
            .iter()
            .map(|b| {
                b.coeffs
                    .get(b.coeffs.len().saturating_sub(1) / 2)
                    .copied()
                    .unwrap_or(Complex32::new(0.0, 0.0))
            })
            .collect();
        self.analytic_last = Some(analytic_flat);

        // 包絡（平均振幅 or 0）
        let amp: Vec<f32> = bands
            .iter()
            .map(|b| {
                if b.coeffs.is_empty() {
                    0.0
                } else {
                    b.coeffs.iter().map(|z| z.norm()).sum::<f32>() / b.coeffs.len() as f32
                }
            })
            .collect();
        self.amp_last = amp.clone();

        // --- 2. Roughness (potential R) ---
        match self.params.r_variant {
            RVariant::NsgtKernel => {
                // (self.last_r, _) = potential_r_from_log2_spectrum(
                //     &self.nsgt.freqs_hz(),
                //     &self.params.roughness_kernel.params,
                //     1.0,
                //     0.0,
                //     false,
                // );
            }
            RVariant::Dummy => self.last_r.fill(0.0),
        }

        // --- 3. Consonance (potential C) ---
        match self.params.c_variant {
            CVariant::NsgtPhaseLock => {
                self.last_c.fill(0.0); // Placeholder for actual implementation
                //     self.last_c = self.params.consonance_kernel.compute_from_nsgt(
                //         self.analytic_last.as_ref().unwrap(),
                //         self.nsgt.freqs_hz(),
                //     );
            }
            CVariant::Dummy => self.last_c.fill(0.0),
        }

        // --- 4. Combine to landscape K = αC − βR ---
        self.last_k = self
            .last_r
            .iter()
            .zip(&self.last_c)
            .map(|(r, c)| self.params.alpha * *c - self.params.beta * *r)
            .collect();
    }

    fn push_hist(bufs: &mut [Vec<f32>], new: &[f32], max_len: usize) {
        for (i, &v) in new.iter().enumerate() {
            if bufs[i].len() == max_len {
                bufs[i].remove(0);
            }
            bufs[i].push(v);
        }
    }

    pub fn snapshot(&self, mut prev: Option<LandscapeFrame>) -> LandscapeFrame {
        let n_ch = self.nsgt.freqs_hz().len();
        let mut out = prev.unwrap_or_default();

        if out.freqs_hz.len() != n_ch {
            out.freqs_hz = self.nsgt.freqs_hz().to_vec();
            out.fs = self.params.fs;
            out.r_last = vec![0.0; n_ch];
            out.c_last = vec![0.0; n_ch];
            out.k_last = vec![0.0; n_ch];
            out.amp_last = vec![0.0; n_ch];
            out.r_hist = vec![Vec::with_capacity(self.params.max_hist_cols); n_ch];
            out.k_hist = vec![Vec::with_capacity(self.params.max_hist_cols); n_ch];
        }

        out.r_last.clone_from(&self.last_r);
        out.c_last.clone_from(&self.last_c);
        out.k_last.clone_from(&self.last_k);
        out.amp_last.clone_from(&self.amp_last);

        Self::push_hist(&mut out.r_hist, &self.last_r, self.params.max_hist_cols);
        Self::push_hist(&mut out.k_hist, &self.last_k, self.params.max_hist_cols);

        out
    }

    pub fn params(&self) -> &LandscapeParams {
        &self.params
    }

    pub fn nsgt(&self) -> &NsgtLog2 {
        &self.nsgt
    }
}
