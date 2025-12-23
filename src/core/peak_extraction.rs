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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;
    use crate::core::roughness_kernel::erb_grid;

    #[test]
    fn small_peaks_are_dropped_and_mass_is_conserved() {
        let space = Log2Space::new(100.0, 2000.0, 64);
        let (_erb, du) = erb_grid(&space);
        let mut density = vec![0.0f32; space.centers_hz.len()];

        let main = density.len() / 2;
        density[main] = 1.0;
        density[main - 2] = 0.05;
        density[main + 2] = 0.08;
        density[main - 3] = 0.01;
        density[main + 3] = 0.02;

        let cfg = PeakExtractConfig {
            max_peaks: None,
            min_rel_db: -20.0,
            min_prominence_db: 0.0,
            min_sep_erb: 0.1,
        };
        let peaks = extract_peaks_density(&density, &space, &cfg);
        assert_eq!(peaks.len(), 1);

        let max_amp = density.iter().cloned().fold(0.0f32, f32::max);
        let min_abs = max_amp * 10.0f32.powf(cfg.min_rel_db / 20.0);
        let expected_mass: f32 = density
            .iter()
            .zip(du.iter())
            .filter(|&(&a, _)| a >= min_abs)
            .map(|(&a, &d)| a * d)
            .sum();
        let diff = (peaks[0].mass - expected_mass).abs();
        assert!(diff < 1e-6, "mass diff {}", diff);
    }

    #[test]
    fn compress_after_aggregate_removes_split_advantage() {
        let space = Log2Space::new(200.0, 2000.0, 48);
        let (_erb, du) = erb_grid(&space);
        let exp = 0.23f32;
        let ref_power = 1.0f32;
        let cfg = PeakExtractConfig {
            max_peaks: None,
            min_rel_db: -120.0,
            min_prominence_db: 0.0,
            min_sep_erb: 0.2,
        };

        let total_mass = 1.0f32;
        let idx = space.centers_hz.len() / 2;
        let idx2 = idx + 1;

        let mut density_single = vec![0.0f32; space.centers_hz.len()];
        density_single[idx] = total_mass / du[idx].max(1e-12);

        let mut density_split = vec![0.0f32; space.centers_hz.len()];
        density_split[idx] = (total_mass * 0.5) / du[idx].max(1e-12);
        density_split[idx2] = (total_mass * 0.5) / du[idx2].max(1e-12);

        let new_single = compress_total_after_aggregate(
            &density_single,
            &space,
            exp,
            ref_power,
            &cfg,
        );
        let new_split = compress_total_after_aggregate(
            &density_split,
            &space,
            exp,
            ref_power,
            &cfg,
        );
        assert!(
            (new_single - new_split).abs() < 1e-6,
            "aggregate mismatch: {} vs {}",
            new_single,
            new_split
        );

        let old_single = compress_total_per_bin(&density_single, &du, exp, ref_power);
        let old_split = compress_total_per_bin(&density_split, &du, exp, ref_power);
        assert!(
            old_split > old_single * 1.01,
            "split should be larger: {} vs {}",
            old_split,
            old_single
        );
    }

    fn compress_total_after_aggregate(
        density: &[f32],
        space: &Log2Space,
        exp: f32,
        ref_power: f32,
        cfg: &PeakExtractConfig,
    ) -> f32 {
        let peaks = extract_peaks_density(density, space, cfg);
        peaks
            .iter()
            .map(|p| (p.mass / ref_power).powf(exp))
            .sum::<f32>()
    }

    fn compress_total_per_bin(density: &[f32], du: &[f32], exp: f32, ref_power: f32) -> f32 {
        density
            .iter()
            .zip(du.iter())
            .map(|(&a, &d)| (a * d / ref_power).powf(exp))
            .sum::<f32>()
    }
}
