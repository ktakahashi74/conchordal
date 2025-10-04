//! core/landscape.rs — Landscape computed by stateful cochlea front-end.

use crate::core::cochlea::Cochlea;

/// Which variant of roughness to compute (only Cochlea supported here).
#[derive(Clone, Copy, Debug)]
pub enum RVariant {
    Cochlea,
    Dummy,
}

#[derive(Clone, Copy, Debug)]
pub enum CVariant {
    Dummy,
}

/// Parameters for the landscape.
#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub fs: f32,
    pub freqs_hz: Vec<f32>,
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
}

pub struct Landscape {
    cochlea: Cochlea,
    params: LandscapeParams,
    last_r: Vec<f32>,
    last_c: Vec<f32>,
    last_k: Vec<f32>,
}

impl Landscape {
    /// Build landscape from params; keeps cochlea states across blocks.
    pub fn new(params: LandscapeParams) -> Self {
        let mut cochlea = Cochlea::new(params.fs, &params.freqs_hz, params.use_lp_300hz);
        cochlea.reset();
        let ch = params.freqs_hz.len();
        Self {
            cochlea,
            params,
            last_r: vec![0.0; ch],
            last_c: vec![0.0; ch],
            last_k: vec![0.0; ch],
        }
    }

    /// Process one audio block and update R/C/K.
    pub fn process_block(&mut self, x: &[f32]) {
        match self.params.r_variant {
            RVariant::Cochlea => {
                self.last_r = self.cochlea.process_block_mean(x);
            }
            RVariant::Dummy => {
                self.last_r.fill(0.0);
            }
        }

        // C is not yet implemented (dummy).
        self.last_c = vec![0.0; self.params.freqs_hz.len()];

        // Compute K = α*C - β*R
        self.last_k = self
            .last_r
            .iter()
            .zip(&self.last_c)
            .map(|(r, c)| self.params.alpha * *c - self.params.beta * *r)
            .collect();
    }

    /// Extract a snapshot for UI.
    pub fn snapshot(&self, mut prev: Option<LandscapeFrame>) -> LandscapeFrame {
        let ch = self.params.freqs_hz.len();
        let mut out = prev.unwrap_or_default();

        if out.freqs_hz.len() != ch {
            out.freqs_hz = self.params.freqs_hz.clone();
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

        // Append history
        for (ci, &v) in self.last_r.iter().enumerate() {
            if out.r_hist[ci].len() == self.params.max_hist_cols {
                out.r_hist[ci].remove(0);
            }
            out.r_hist[ci].push(v);
        }
        for (ci, &v) in self.last_k.iter().enumerate() {
            if out.k_hist[ci].len() == self.params.max_hist_cols {
                out.k_hist[ci].remove(0);
            }
            out.k_hist[ci].push(v);
        }

        out
    }

    pub fn params(&self) -> &LandscapeParams {
        &self.params
    }

    pub fn retune(&mut self, new_freqs_hz: Vec<f32>) {
        self.params.freqs_hz = new_freqs_hz;
        self.cochlea = Cochlea::new(
            self.params.fs,
            &self.params.freqs_hz,
            self.params.use_lp_300hz,
        );
        self.cochlea.reset();
        let ch = self.params.freqs_hz.len();
        self.last_r = vec![0.0; ch];
        self.last_c = vec![0.0; ch];
        self.last_k = vec![0.0; ch];
    }
}
