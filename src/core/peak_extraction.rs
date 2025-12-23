//! core/peak_extraction.rs — Extract delta-like peaks from ERB-density spectra.
//! density: per-ERB (or per-u) power density; mass: sum(density * du).

use crate::core::db;
use crate::core::density;
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
    /// Power-domain threshold (10*log10), relative to max density.
    pub min_rel_db_power: f32,
    /// Power-domain prominence (10*log10) for local peak selection.
    pub min_prominence_db_power: f32,
    /// Mass-domain threshold (10*log10), relative to max peak mass.
    /// Mass is Σ density * du (area), using power-domain dB.
    pub min_rel_mass_db_power: f32,
    /// Optional mass-domain threshold relative to total mass.
    /// Mass is Σ density * du (area), using power-domain dB semantics.
    pub min_mass_fraction: Option<f32>,
    pub min_sep_erb: f32,
}

impl Default for PeakExtractConfig {
    fn default() -> Self {
        Self {
            max_peaks: None,
            min_rel_db_power: -35.0,
            min_prominence_db_power: 10.0,
            min_rel_mass_db_power: -35.0,
            min_mass_fraction: None,
            min_sep_erb: 0.30,
        }
    }
}

impl PeakExtractConfig {
    pub fn strict() -> Self {
        Self {
            min_mass_fraction: Some(0.05),
            ..Self::default()
        }
    }

    pub fn normal() -> Self {
        Self {
            max_peaks: None,
            min_rel_db_power: -60.0,
            min_prominence_db_power: 3.0,
            min_rel_mass_db_power: -70.0,
            min_mass_fraction: None,
            min_sep_erb: 0.10,
        }
    }
}

fn local_prominence_db_power(power_density: &[f32], idx: usize) -> f32 {
    let peak = power_density[idx];
    if peak <= 0.0 {
        return f32::NEG_INFINITY;
    }

    let mut min_val = peak;
    for j in (0..idx).rev() {
        min_val = min_val.min(power_density[j]);
        if power_density[j] > peak {
            break;
        }
    }
    let left_min = min_val;

    min_val = peak;
    for j in (idx + 1)..power_density.len() {
        min_val = min_val.min(power_density[j]);
        if power_density[j] > peak {
            break;
        }
    }
    let right_min = min_val;

    let base = left_min.max(right_min).max(db::EPS_POWER);
    db::power_to_db(peak / base)
}

/// Extract peaks from an ERB power density. Total mass is conserved for the
/// masked spectrum (power >= min_abs), not the original input.
pub fn extract_peaks_density(
    power_density: &[f32],
    space: &Log2Space,
    cfg: &PeakExtractConfig,
) -> Vec<Peak> {
    if power_density.is_empty() || space.centers_hz.is_empty() {
        return vec![];
    }
    assert_eq!(power_density.len(), space.centers_hz.len());

    let (erb, du) = erb_grid(space);
    let max_power = power_density.iter().cloned().fold(0.0f32, f32::max);
    if max_power <= 0.0 {
        return vec![];
    }

    let min_abs = max_power * db::db_to_power_ratio(cfg.min_rel_db_power);

    #[derive(Clone, Copy, Debug)]
    struct Candidate {
        idx: usize,
        u_erb: f32,
        power: f32,
    }

    let mut candidates = Vec::new();
    if power_density.len() >= 3 {
        for i in 1..(power_density.len() - 1) {
            let a = power_density[i];
            if a < min_abs {
                continue;
            }
            if a > power_density[i - 1] && a >= power_density[i + 1] {
                let prom_db = local_prominence_db_power(power_density, i);
                if prom_db >= cfg.min_prominence_db_power {
                    candidates.push(Candidate {
                        idx: i,
                        u_erb: erb[i],
                        power: a,
                    });
                }
            }
        }
    }

    if candidates.is_empty() {
        return vec![];
    }

    candidates.sort_by(|a, b| {
        b.power
            .partial_cmp(&a.power)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

    for (j, &power) in power_density.iter().enumerate() {
        if power < min_abs {
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
        let mass = power * du[j];
        mass_sum[best] += mass;
        u_weighted[best] += u * mass;
    }

    let max_mass = mass_sum.iter().cloned().fold(0.0f32, f32::max);
    let total_mass: f32 = mass_sum.iter().sum();
    let min_mass_rel = max_mass * db::db_to_power_ratio(cfg.min_rel_mass_db_power);
    let min_mass_total = cfg.min_mass_fraction.map(|frac| total_mass * frac);
    let mut keep: Vec<bool> = mass_sum
        .iter()
        .map(|&mass| {
            let mut remove = mass < min_mass_rel;
            if let Some(min_total) = min_mass_total {
                remove |= mass < min_total;
            }
            !remove
        })
        .collect();

    if keep.iter().all(|&k| !k) && max_mass > 0.0 {
        if let Some((idx, _)) = mass_sum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            keep[idx] = true;
        }
    }

    if keep.iter().any(|&k| k) {
        for i in 0..mass_sum.len() {
            if keep[i] || mass_sum[i] <= 0.0 {
                continue;
            }
            let u_i = selected[i].u_erb;
            let mut best = None;
            let mut best_d = f32::MAX;
            for (k, &is_keep) in keep.iter().enumerate() {
                if !is_keep {
                    continue;
                }
                let d = (selected[k].u_erb - u_i).abs();
                if d < best_d {
                    best_d = d;
                    best = Some(k);
                }
            }
            if let Some(k) = best {
                mass_sum[k] += mass_sum[i];
                u_weighted[k] += u_weighted[i];
                mass_sum[i] = 0.0;
                u_weighted[i] = 0.0;
            }
        }
    }

    let mut peaks = Vec::with_capacity(selected.len());
    for (k, sel) in selected.iter().enumerate() {
        let mass = mass_sum[k];
        if mass <= 0.0 {
            continue;
        }
        let u_erb = u_weighted[k] / mass;
        let mut bin_idx = 0usize;
        let mut best = f32::MAX;
        for (i, &u) in erb.iter().enumerate() {
            let d = (u - u_erb).abs();
            if d < best {
                best = d;
                bin_idx = i;
            }
        }
        peaks.push(Peak {
            u_erb,
            mass,
            bin_idx,
        });
    }

    peaks
}

pub fn peaks_to_delta_density(peaks: &[Peak], du: &[f32], len: usize) -> Vec<f32> {
    let peak_masses: Vec<(usize, f32)> = peaks.iter().map(|p| (p.bin_idx, p.mass)).collect();
    density::peaks_mass_to_delta_density(len, &peak_masses, du)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::density;
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
            min_rel_db_power: -10.0,
            min_prominence_db_power: 0.0,
            min_rel_mass_db_power: -50.0,
            min_mass_fraction: None,
            min_sep_erb: 0.1,
        };
        let peaks = extract_peaks_density(&density, &space, &cfg);
        assert_eq!(peaks.len(), 1);

        let max_amp = density.iter().cloned().fold(0.0f32, f32::max);
        let min_abs = max_amp * db::db_to_power_ratio(cfg.min_rel_db_power);
        let mut masked = vec![0.0f32; density.len()];
        for (i, &a) in density.iter().enumerate() {
            if a >= min_abs {
                masked[i] = a;
            }
        }
        let expected_mass = density::density_to_mass(&masked, &du);
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
            min_rel_db_power: -120.0,
            min_prominence_db_power: 0.0,
            min_rel_mass_db_power: -70.0,
            min_mass_fraction: None,
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

        let new_single =
            compress_total_after_aggregate(&density_single, &space, exp, ref_power, &cfg);
        let new_split =
            compress_total_after_aggregate(&density_split, &space, exp, ref_power, &cfg);
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

    #[test]
    fn side_lobe_peak_dropped_by_mass_filter() {
        let space = Log2Space::new(80.0, 8000.0, 128);
        let (erb, du) = erb_grid(&space);
        let mut density = vec![0.0f32; erb.len()];

        let center = erb.len() / 2;
        for i in (center - 3)..=(center + 3) {
            let x = (i as f32 - center as f32) / 1.5;
            density[i] = (-0.5 * x * x).exp();
        }
        let side = center + 18;
        density[side] = 0.01;

        let cfg = PeakExtractConfig::strict();
        let peaks = extract_peaks_density(&density, &space, &cfg);

        assert!(peaks.len() >= 1);
        assert!(
            peaks
                .iter()
                .any(|p| (p.bin_idx as isize - center as isize).abs() <= 2),
            "main peak should remain"
        );
        assert!(
            peaks.iter().all(|p| p.bin_idx != side),
            "side-lobe peak should be removed"
        );
    }

    #[test]
    fn strict_reduces_small_spike_count_vs_normal() {
        let space = Log2Space::new(80.0, 8000.0, 128);
        let (erb, _du) = erb_grid(&space);
        let mut density = vec![0.0f32; erb.len()];

        let center = erb.len() / 2;
        for i in (center - 4)..=(center + 4) {
            let x = (i as f32 - center as f32) / 2.0;
            density[i] = (-0.5 * x * x).exp();
        }
        for offset in [-30isize, -18, -9, 9, 18, 30] {
            let idx = (center as isize + offset) as usize;
            density[idx] = 0.01;
        }

        let normal = extract_peaks_density(&density, &space, &PeakExtractConfig::normal());
        let strict = extract_peaks_density(&density, &space, &PeakExtractConfig::strict());
        assert!(normal.len() >= 2, "expected multiple normal peaks");
        assert!(
            strict.len() < normal.len(),
            "expected strict to reduce peaks: normal={} strict={}",
            normal.len(),
            strict.len()
        );
    }

    #[test]
    fn mass_is_conserved_after_mass_filter() {
        let space = Log2Space::new(80.0, 8000.0, 128);
        let (_erb, du) = erb_grid(&space);
        let mut density = vec![0.0f32; space.centers_hz.len()];

        let center = density.len() / 2;
        for i in (center - 4)..=(center + 4) {
            let x = (i as f32 - center as f32) / 2.0;
            density[i] = (-0.5 * x * x).exp();
        }
        let side = center + 20;
        density[side] = 0.4;

        let cfg = PeakExtractConfig {
            min_rel_db_power: -120.0,
            min_prominence_db_power: 0.0,
            min_rel_mass_db_power: -6.0,
            min_mass_fraction: Some(0.05),
            ..PeakExtractConfig::normal()
        };
        let peaks = extract_peaks_density(&density, &space, &cfg);
        let total_mass_in = density::density_to_mass(&density, &du);
        let total_mass_out: f32 = peaks.iter().map(|p| p.mass).sum();
        let diff = (total_mass_in - total_mass_out).abs();
        assert!(diff < 1e-5, "mass diff {}", diff);
    }

    #[test]
    fn mass_filter_keeps_at_least_one_peak() {
        let space = Log2Space::new(200.0, 2000.0, 48);
        let (_erb, _du) = erb_grid(&space);
        let mut density = vec![0.0f32; space.centers_hz.len()];
        let center = density.len() / 2;
        density[center] = 0.1;
        density[center + 4] = 0.08;

        let cfg = PeakExtractConfig {
            max_peaks: None,
            min_rel_db_power: -120.0,
            min_prominence_db_power: 0.0,
            min_rel_mass_db_power: 3.0,
            min_mass_fraction: Some(0.9),
            min_sep_erb: 0.2,
        };
        let peaks = extract_peaks_density(&density, &space, &cfg);
        assert!(!peaks.is_empty(), "expected at least one peak");
    }
}
