// core/landscape.rs — Landscape computed by stateful cochlea front-end.

use rustfft::num_complex::Complex32;

use crate::core::cochlea::Cochlea;
use crate::core::erb::ErbSpace;
//use crate::core::roughness::compute_potential_r_from_signal;

use crate::core::fft::hilbert;
use crate::core::roughness_kernel::{KernelParams, potential_r_from_analytic};

#[derive(Clone, Copy, Debug)]
pub enum RVariant {
    KernelConv,
    Cochlea,
    CochleaPotential,
    Dummy,
}

#[derive(Clone, Copy, Debug)]
pub enum CVariant {
    CochleaPl,
    Dummy,
}

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub use_lp_300hz: bool,
    pub max_hist_cols: usize,
    pub alpha: f32,
    pub beta: f32,
    pub r_variant: RVariant,
    pub c_variant: CVariant,
    pub kernel_params: KernelParams,
}

#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub fs: f32,
    pub freqs_hz: Vec<f32>,
    pub r_last: Vec<f32>,
    pub c_last: Vec<f32>,
    pub k_last: Vec<f32>,
    pub env_last: Vec<f32>,
    pub r_hist: Vec<Vec<f32>>,
    pub k_hist: Vec<Vec<f32>>,
    pub plv_last: Option<Vec<Vec<f32>>>,
}

pub struct Landscape {
    cochlea: Cochlea,
    params: LandscapeParams,
    last_r: Vec<f32>,
    last_c: Vec<f32>,
    last_k: Vec<f32>,
    env_last: Vec<f32>,
    last_plv: Option<Vec<Vec<f32>>>,
    analytic_last: Option<Vec<Complex32>>,
}

impl Landscape {
    pub fn new(params: LandscapeParams, erb_space: ErbSpace) -> Self {
        let ch = erb_space.len();
        let mut cochlea = Cochlea::new(params.fs, erb_space, params.use_lp_300hz);
        cochlea.reset();
        Self {
            cochlea,
            params,
            last_r: vec![0.0; ch],
            last_c: vec![0.0; ch],
            last_k: vec![0.0; ch],
            env_last: vec![0.0; ch],
            last_plv: Some(vec![vec![0.0; ch]; ch]),
            analytic_last: None,
        }
    }

    /// Process one block: cochlea update + R/C/K compute.
    pub fn process_block(&mut self, x: &[f32]) {
        let analytic = hilbert(x);
        self.analytic_last = Some(analytic.clone());

        // --- R ---
        match self.params.r_variant {
            RVariant::Cochlea => {
                //self.last_r = self.cochlea.compute_r_from_env(&env_vec);
            }
            RVariant::CochleaPotential => {
                // let e_ch = self.cochlea.current_envelope_levels(); // envelope mean per channel
                // let erb_space = &self.cochlea.erb_space;
                // let freqs = erb_space.freqs_hz();

                //                self.last_r = compute_potential_r(&e_ch, freqs, erb_space, &PotRParams::default());
            }
            RVariant::KernelConv => {
                (self.last_r, _) = potential_r_from_analytic(
                    &analytic,
                    self.params.fs,
                    &self.params.kernel_params,
                    0.5,
                    0.0, // salience parameter
                );
            }
            RVariant::Dummy => self.last_r.fill(0.0),
        }

        // --- C ---
        match self.params.c_variant {
            CVariant::CochleaPl => {
                //                self.last_c = self.cochlea.compute_c_from_plv(&plv_mat);
            }
            CVariant::Dummy => {
                self.last_c.fill(0.0);
            }
        }

        // PLV スナップショットは共通で最新を保存
        //      self.last_plv = Some(plv_mat);

        // --- K ---
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
        let ch = self.cochlea.n_ch();
        let mut out = prev.unwrap_or_default();

        if out.freqs_hz.len() != ch {
            out.freqs_hz = self.cochlea.erb_space.freqs_hz().to_vec();
            out.fs = self.params.fs;
            out.r_last = vec![0.0; ch];
            out.c_last = vec![0.0; ch];
            out.k_last = vec![0.0; ch];
            out.env_last = vec![0.0; ch];
            out.r_hist = vec![Vec::with_capacity(self.params.max_hist_cols); ch];
            out.k_hist = vec![Vec::with_capacity(self.params.max_hist_cols); ch];
        }

        out.r_last.clone_from(&self.last_r);
        out.c_last.clone_from(&self.last_c);
        out.k_last.clone_from(&self.last_k);
        out.env_last.clone_from(&self.env_last);

        Self::push_hist(&mut out.r_hist, &self.last_r, self.params.max_hist_cols);
        Self::push_hist(&mut out.k_hist, &self.last_k, self.params.max_hist_cols);

        if let Some(plv) = &self.last_plv {
            out.plv_last = Some(plv.clone());
        }
        out
    }

    pub fn params(&self) -> &LandscapeParams {
        &self.params
    }
}
