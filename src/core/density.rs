//! Density/mass helpers for ERB (or other u-axis) representations.
//! density: per-u (e.g., per ERB); mass: integral of density over du.

/// Integrate a density over du to produce total mass.
pub fn density_to_mass(density: &[f32], du: &[f32]) -> f32 {
    density
        .iter()
        .zip(du.iter())
        .map(|(d, du)| d * du)
        .sum()
}

/// Convert peak masses into a delta density vector.
/// peaks: (idx, mass)
pub fn peaks_mass_to_delta_density(len: usize, peaks: &[(usize, f32)], du: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    if du.len() < len {
        return out;
    }
    for &(idx, mass) in peaks {
        if idx >= len {
            continue;
        }
        let step = du[idx];
        debug_assert!(step > 0.0, "du must be positive");
        if step <= 0.0 {
            continue;
        }
        out[idx] += mass / step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn peaks_mass_round_trip_preserves_total_mass() {
        let len = 64;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let du: Vec<f32> = (0..len).map(|_| rng.random_range(0.01..0.2)).collect();

        let mut peaks = Vec::new();
        let mut total_mass = 0.0f32;
        for _ in 0..20 {
            let idx = rng.random_range(0..len);
            let mass = rng.random_range(0.1..2.0);
            peaks.push((idx, mass));
            total_mass += mass;
        }

        let delta_density = peaks_mass_to_delta_density(len, &peaks, &du);
        let mass2 = density_to_mass(&delta_density, &du);
        let diff = (mass2 - total_mass).abs();
        assert!(diff < 1e-5, "mass diff {diff}");
    }
}
