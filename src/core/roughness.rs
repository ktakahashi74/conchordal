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
    Dummy,
}

#[derive(Clone, Copy, Debug)]
pub enum CVariant {
    Dummy,
    FromHilbert,
}

#[derive(Clone, Debug)]
pub struct LandscapeParams {
    pub alpha: f32,
    pub beta: f32,
    pub r_variant: RVariant,
    pub c_variant: CVariant,
}

#[derive(Clone, Debug, Default)]
pub struct LandscapeFrame {
    pub freqs_hz: Vec<f32>,
    pub r: Vec<f32>,
    pub c: Vec<f32>,
    pub k: Vec<f32>,
}

impl LandscapeFrame {
    pub fn recompute_k(&mut self, p: &LandscapeParams) {
        self.k = self
            .r
            .iter()
            .zip(self.c.iter())
            .map(|(rr, cc)| p.alpha * *cc - p.beta * *rr)
            .collect();
    }
}

// ---------------------------------------------------------
// Roughness R (gammatone-based)
// ---------------------------------------------------------

fn compute_r_gammatone(signal: &[f32], fs: f32, freqs_hz: &[f32]) -> Vec<f32> {
    let outs = gammatone_filterbank(signal, freqs_hz, fs);
    outs.into_iter()
        .map(|ch| {
            let env = hilbert_envelope(&ch);
            let mean = env.iter().sum::<f32>() / env.len() as f32;
            if mean <= 1e-9 {
                return 0.0;
            }
            let var = env.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / env.len() as f32;
            (var.sqrt()) / mean
        })
        .collect()
}

fn compute_r_dummy(freqs_hz: &[f32]) -> Vec<f32> {
    vec![0.0; freqs_hz.len()]
}

// ---------------------------------------------------------
// Consonance C (dummy / hilbert-based placeholder)
// ---------------------------------------------------------

fn compute_c_dummy(freqs_hz: &[f32]) -> Vec<f32> {
    vec![0.0; freqs_hz.len()]
}

fn compute_c_hilbert(signal: &[f32], fs: f32, freqs_hz: &[f32]) -> Vec<f32> {
    let outs = gammatone_filterbank(signal, freqs_hz, fs);
    outs.into_iter()
        .map(|ch| {
            let env = hilbert_envelope(&ch);
            let mean = env.iter().sum::<f32>() / env.len() as f32;
            mean * 0.1 // dummy scaling
        })
        .collect()
}

// ---------------------------------------------------------
// Variant selector
// ---------------------------------------------------------

fn compute_r(data: &AnalysisData, variant: RVariant, freqs: &[f32]) -> Vec<f32> {
    match variant {
        RVariant::Gammatone => compute_r_gammatone(&data.signal, data.fs, freqs),
        RVariant::Dummy => compute_r_dummy(freqs),
    }
}

fn compute_c(data: &AnalysisData, variant: CVariant, freqs: &[f32]) -> Vec<f32> {
    match variant {
        CVariant::Dummy => compute_c_dummy(freqs),
        CVariant::FromHilbert => compute_c_hilbert(&data.signal, data.fs, freqs),
    }
}

// ---------------------------------------------------------
// Main landscape computation
// ---------------------------------------------------------

pub fn compute_landscape(data: &AnalysisData, p: &LandscapeParams, freqs: &[f32]) -> LandscapeFrame {
    let r = compute_r(data, p.r_variant, freqs);
    let c = compute_c(data, p.c_variant, freqs);
    let k: Vec<f32> = r
        .iter()
        .zip(&c)
        .map(|(rr, cc)| p.alpha * *cc - p.beta * *rr)
        .collect();

    LandscapeFrame {
        freqs_hz: freqs.to_vec(),
        r,
        c,
        k,
    }
}


