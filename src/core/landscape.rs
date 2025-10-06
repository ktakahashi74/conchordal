//! core/landscape.rs — Landscape computed by stateful cochlea front-end.

use crate::core::cochlea::Cochlea;
use crate::core::erb::ErbSpace;

/// Which variant of roughness to compute (only Cochlea supported here).
#[derive(Clone, Copy, Debug)]
pub enum RVariant {
    KernelConv,
    Cochlea,
    Dummy,
}

#[derive(Clone, Copy, Debug)]
pub enum CVariant {
    CochleaPl,
    Dummy,
}

/// Parameters for the landscape.
#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub use_lp_300hz: bool,
    pub max_hist_cols: usize,
    pub alpha: f32,
    pub beta: f32,
    pub r_variant: RVariant,
    pub c_variant: CVariant,
}

/// One frame of the landscape for UI.
#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub fs: f32,
    pub freqs_hz: Vec<f32>,
    pub r_last: Vec<f32>,
    pub c_last: Vec<f32>,
    pub k_last: Vec<f32>,
    /// History buffer [channels][time cols].
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
    last_plv: Option<Vec<Vec<f32>>>,
}

impl Landscape {
    /// Build landscape from params; keeps cochlea states across blocks.
    pub fn new(params: LandscapeParams, erb_space: ErbSpace) -> Self {
        let ch = erb_space.n_bins();
        let mut cochlea = Cochlea::new(params.fs, erb_space, params.use_lp_300hz);
        cochlea.reset();
        Self {
            cochlea,
            params,
            last_r: vec![0.0; ch],
            last_c: vec![0.0; ch],
            last_k: vec![0.0; ch],
            last_plv: Some(vec![vec![0.0; ch]; ch]),
        }
    }

    /// Process one block: cochlea update + R/C/K compute.
    pub fn process_block(&mut self, x: &[f32]) {
        // Unified cochlea step (envelope + PLV)o
        let (env_vec, plv_mat) = self.cochlea.process_block_core(x);

        // --- R ---
        match self.params.r_variant {
            RVariant::Cochlea => {
                self.last_r = self.cochlea.compute_r_from_env(&env_vec);
            }
            RVariant::KernelConv => {
                let (env_vec, plv_mat) = self.cochlea.process_block_core(x);
                use crate::core::roughness_kernel::{KernelParams, compute_r_kernelconv};
                let erb_step = 0.01;
                let kp = KernelParams::default();
                self.last_r = compute_r_kernelconv(&env_vec, &self.cochlea.erb_space, &kp);
                self.last_plv = Some(plv_mat);
            }
            RVariant::Dummy => self.last_r.fill(0.0),
        }

        // --- C ---
        match self.params.c_variant {
            CVariant::CochleaPl => {
                self.last_c = self.cochlea.compute_c_from_plv(&plv_mat);
                self.last_plv = Some(plv_mat);
            }
            CVariant::Dummy => {
                self.last_c.fill(0.0);
                let ch = self.last_c.len();
                self.last_plv = Some(vec![vec![0.0; ch]; ch]);
            }
        }

        // --- K ---
        self.last_k = self
            .last_r
            .iter()
            .zip(&self.last_c)
            .map(|(r, c)| self.params.alpha * *c - self.params.beta * *r)
            .collect();
    }

    /// Helper: append new column to history buffers.
    fn push_hist(bufs: &mut [Vec<f32>], new: &[f32], max_len: usize) {
        for (i, &v) in new.iter().enumerate() {
            if bufs[i].len() == max_len {
                bufs[i].remove(0);
            }
            bufs[i].push(v);
        }
    }

    /// Extract a snapshot for UI.
    pub fn snapshot(&self, mut prev: Option<LandscapeFrame>) -> LandscapeFrame {
        let ch = self.cochlea.n_ch();
        let mut out = prev.unwrap_or_default();

        if out.freqs_hz.len() != ch {
            out.freqs_hz = self.cochlea.erb_space.freqs_hz().to_vec();
            out.fs = self.params.fs;
            out.r_last = vec![0.0; ch];
            out.c_last = vec![0.0; ch];
            out.k_last = vec![0.0; ch];
            out.r_hist = vec![Vec::with_capacity(self.params.max_hist_cols); ch];
            out.k_hist = vec![Vec::with_capacity(self.params.max_hist_cols); ch];
        }

        out.r_last.clone_from(&self.last_r);
        out.c_last.clone_from(&self.last_c);
        out.k_last.clone_from(&self.last_k);

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
