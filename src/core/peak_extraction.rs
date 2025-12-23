//! core/peak_extraction.rs — Extract delta-like peaks from ERB-density spectra.

use crate::core::log2space::Log2Space;
use crate::core::roughness_kernel::erb_grid;

#[derive(Clone, Debug)]
pub struct Peak {
    pub u_erb: f32,
    pub mass: f32,
    pub bin_idx: usize,
}

#[derive(Clone, Debug)]
pub struct PeakExtractConfig {
    pub max_peaks: Option<usize>,
    pub min_rel_db: f32,
    pub min_prominence_db: f32,
    pub min_sep_erb: f32,
}

impl Default for PeakExtractConfig {
    fn default() -> Self {
        Self {
            max_peaks: None,
            min_rel_db: -60.0,
            min_prominence_db: 3.0,
            min_sep_erb: 0.10,
        }
    }
}

fn local_prominence_db(amps: &[f32], idx: usize) -> f32 {
    let peak = amps[idx];
    if peak <= 0.0 {
        return f32::NEG_INFINITY;
    }

    let mut min_val = peak;
    for j in (0..idx).rev() {
        min_val = min_val.min(amps[j]);
        if amps[j] > peak {
            break;
        }
    }
    let left_min = min_val;

    min_val = peak;
    for j in (idx + 1)..amps.len() {
        min_val = min_val.min(amps[j]);
        if amps[j] > peak {
            break;
        }
    }
    let right_min = min_val;

    let base = left_min.max(right_min).max(1e-12);
    20.0 * (peak / base).log10()
}

pub fn extract_peaks_density(
    amps_density: &[f32],
    space: &Log2Space,
    cfg: &PeakExtractConfig,
) -> Vec<Peak> {
    if amps_density.is_empty() || space.centers_hz.is_empty() {
        return vec![];
    }
    assert_eq!(amps_density.len(), space.centers_hz.len());

    let (erb, du) = erb_grid(space);
    let max_amp = amps_density.iter().cloned().fold(0.0f32, f32::max);
    if max_amp <= 0.0 {
        return vec![];
    }

    let min_abs = max_amp * 10.0f32.powf(cfg.min_rel_db / 20.0);

    #[derive(Clone, Copy, Debug)]
    struct Candidate {
        idx: usize,
        u_erb: f32,
        amp: f32,
    }

    let mut candidates = Vec::new();
    if amps_density.len() >= 3 {
        for i in 1..(amps_density.len() - 1) {
            let a = amps_density[i];
            if a < min_abs {
                continue;
            }
            if a > amps_density[i - 1] && a >= amps_density[i + 1] {
                let prom_db = local_prominence_db(amps_density, i);
                if prom_db >= cfg.min_prominence_db {
                    candidates.push(Candidate {
                        idx: i,
                        u_erb: erb[i],
                        amp: a,
                    });
                }
            }
        }
    }

    if candidates.is_empty() {
        return vec![];
    }

    candidates.sort_by(|a, b| b.amp.partial_cmp(&a.amp).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<Candidate> = Vec::new();
    for cand in candidates {
        if selected
            .iter()
            .any(|p| (p.u_erb - cand.u_erb).abs() < cfg.min_sep_erb)
        {
            continue;
        }
        selected.push(cand);
        if let Some(max_peaks) = cfg.max_peaks {
            if selected.len() >= max_peaks {
                break;
            }
        }
    }

    if selected.is_empty() {
        return vec![];
    }

    let mut mass_sum = vec![0.0f32; selected.len()];
    let mut u_weighted = vec![0.0f32; selected.len()];

    for (j, &amp) in amps_density.iter().enumerate() {
        if amp < min_abs {
            continue;
        }
        let u = erb[j];
        let mut best = 0usize;
        let mut best_d = (u - selected[0].u_erb).abs();
        for (k, sel) in selected.iter().enumerate().skip(1) {
            let d = (u - sel.u_erb).abs();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        let mass = amp * du[j];
        mass_sum[best] += mass;
        u_weighted[best] += u * mass;
    }

    let mut peaks = Vec::with_capacity(selected.len());
    for (k, sel) in selected.iter().enumerate() {
        let mass = mass_sum[k];
        if mass <= 0.0 {
            continue;
        }
        let u_erb = u_weighted[k] / mass;
        peaks.push(Peak {
            u_erb,
            mass,
            bin_idx: sel.idx,
        });
    }

    peaks
}

pub fn peaks_to_delta_density(peaks: &[Peak], du: &[f32], len: usize) -> Vec<f32> {
    let mut density = vec![0.0f32; len];
    if du.len() < len {
        return density;
    }

    for peak in peaks {
        if peak.bin_idx >= len {
            continue;
        }
        let step = du[peak.bin_idx];
        if step <= 0.0 {
            continue;
        }
        density[peak.bin_idx] += peak.mass / step;
    }

    density
}
