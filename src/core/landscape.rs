//! core/landscape.rs — Landscape computed on log2(NSGT-RT) domain.
//! Each incoming frame updates the potential roughness (R) and consonance (C)
//! using a real-time NSGT analyzer.

use crate::core::consonance_kernel::ConsonanceKernel;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::RoughnessKernel;

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub max_hist_cols: usize,
    pub gamma: f32,
    pub alpha: f32,
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
    pub amps_last: Vec<f32>,
}

pub struct Landscape {
    nsgt_rt: RtNsgtKernelLog2,
    params: LandscapeParams,
    last_r: Vec<f32>,
    last_c: Vec<f32>,
    last_k: Vec<f32>,
    amps_last: Vec<f32>,
}

impl Landscape {
    pub fn new(params: LandscapeParams, nsgt_rt: RtNsgtKernelLog2) -> Self {
        let n_ch = nsgt_rt.space().n_bins();
        Self {
            nsgt_rt,
            params,
            last_r: vec![0.0; n_ch],
            last_c: vec![0.0; n_ch],
            last_k: vec![0.0; n_ch],
            amps_last: vec![0.0; n_ch],
        }
    }
    /// Process one streaming frame (e.g. hop-size worth of samples)
    pub fn process_frame(&mut self, x_frame: &[f32]) -> LandscapeFrame {
        let fs = self.params.fs;

        // === 1. NSGT-RT解析（ストリーム入力）===
        let envelope: Vec<f32> = self.nsgt_rt.process_frame(x_frame);
        self.amps_last.clone_from(&envelope);

        // 周波数空間
        let space = self.nsgt_rt.space();

        // === 3. potential R ===
        let (r, _r_total) = self.params.roughness_kernel.potential_r_from_log2_spectrum(
            &envelope,
            space,
            self.params.gamma,
            self.params.alpha,
        );

        // === 4. potential C（位相相関）===
        let (c, _norm) = self
            .params
            .consonance_kernel
            .potential_c_from_log2_spectrum(&envelope, space, self.params.gamma);

        // === 5. potential K（RとCの統合指標）===
        // 例: K = C - α * R
        let k: Vec<f32> = r
            .iter()
            .zip(&c)
            .map(|(ri, ci)| ci - self.params.alpha * ri)
            .collect();

        // 内部状態を更新
        self.last_r = r;
        self.last_c = c;
        self.last_k = k;

        LandscapeFrame {
            fs,
            freqs_hz: space.centers_hz.clone(),
            r_last: self.last_r.clone(),
            c_last: self.last_c.clone(),
            k_last: self.last_k.clone(),
            amps_last: self.amps_last.clone(),
        }
    }

    pub fn snapshot(&self) -> LandscapeFrame {
        LandscapeFrame {
            fs: self.params.fs,
            freqs_hz: self.nsgt_rt.space().centers_hz.clone(),
            r_last: self.last_r.clone(),
            c_last: self.last_c.clone(),
            k_last: self.last_k.clone(),
            amps_last: self.amps_last.clone(),
        }
    }

    pub fn params(&self) -> &LandscapeParams {
        &self.params
    }
}
