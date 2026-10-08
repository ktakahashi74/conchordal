//! Offline reference for the body-weighted consonance fitness rule.

use crate::core::consonance_kernel::ConsonanceRepresentationParams;
use crate::core::log2space::Log2Space;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BodyFitness {
    pub score: f32,
    pub level: f32,
    pub in_band_mass: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unsupported {
    InvalidInput,
    NoInBandMass,
}

/// Integrate effective C over a representative body's subjective-intensity
/// density. All three scans must use the same Log2Space bins. `du_scan` is the
/// ERB width of each bin, and the caller retains its original source and epoch.
pub(crate) fn evaluate(
    space: &Log2Space,
    subjective_intensity_scan: &[f32],
    du_scan: &[f32],
    c_score_eff_scan: &[f32],
    repr: &ConsonanceRepresentationParams,
) -> Result<BodyFitness, Unsupported> {
    space.assert_scan_len_named(subjective_intensity_scan, "subjective_intensity_scan");
    space.assert_scan_len_named(du_scan, "du_scan");
    space.assert_scan_len_named(c_score_eff_scan, "c_score_eff_scan");

    if !repr.beta.is_finite() || !repr.theta.is_finite() {
        return Err(Unsupported::InvalidInput);
    }
    let mut mass = 0.0f64;
    let mut weighted_score = 0.0f64;
    for ((&density, &width), &score) in subjective_intensity_scan
        .iter()
        .zip(du_scan)
        .zip(c_score_eff_scan)
    {
        if !density.is_finite()
            || density < 0.0
            || !width.is_finite()
            || width <= 0.0
            || !score.is_finite()
        {
            return Err(Unsupported::InvalidInput);
        }
        let bin_mass = f64::from(density) * f64::from(width);
        mass += bin_mass;
        weighted_score += bin_mass * f64::from(score);
    }
    if mass <= 0.0 {
        return Err(Unsupported::NoInBandMass);
    }
    let score = (weighted_score / mass) as f32;
    let in_band_mass = mass as f32;
    if !score.is_finite() || !in_band_mass.is_finite() {
        return Err(Unsupported::InvalidInput);
    }
    if in_band_mass <= 0.0 {
        return Err(Unsupported::NoInBandMass);
    }
    Ok(BodyFitness {
        score,
        level: repr.level(score),
        in_band_mass,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::consonance_kernel::ConsonanceKernel;
    use crate::core::psycho_state::{normalize_density, roughness_ratio_to_state01};
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel, erb_grid};

    fn space() -> Log2Space {
        Log2Space::new(100.0, 2_000.0, 12)
    }

    #[test]
    fn one_bin_matches_point_score_including_a_valid_zero() {
        let space = space();
        let n = space.n_bins();
        let mut density = vec![0.0; n];
        let du = vec![0.2; n];
        let mut c = vec![-1.0; n];
        let bin = space.index_of_freq(440.0).unwrap();
        density[bin] = 5.0;
        c[bin] = 0.0;
        let repr = ConsonanceRepresentationParams::default();
        let fitness = evaluate(&space, &density, &du, &c, &repr).unwrap();
        assert_eq!(fitness.score, 0.0);
        assert_eq!(fitness.level, repr.level(0.0));
        assert_eq!(fitness.in_band_mass, 1.0);
    }

    #[test]
    fn unequal_bin_widths_weight_equal_masses_equally() {
        let space = space();
        let n = space.n_bins();
        let mut density = vec![0.0; n];
        let mut du = vec![1.0; n];
        let mut c = vec![0.0; n];
        density[2] = 4.0;
        du[2] = 0.25;
        c[2] = -1.0;
        density[12] = 0.5;
        du[12] = 2.0;
        c[12] = 3.0;
        let fitness = evaluate(&space, &density, &du, &c, &Default::default()).unwrap();
        assert_eq!(fitness.score, 1.0);
        assert_eq!(fitness.in_band_mass, 2.0);
    }

    #[test]
    fn same_fundamental_different_upper_partial_gets_different_fitness() {
        let space = space();
        let n = space.n_bins();
        let du = vec![1.0; n];
        let fundamental = space.index_of_freq(220.0).unwrap();
        let third = space.index_of_freq(660.0).unwrap();
        let fourth = space.index_of_freq(880.0).unwrap();
        let mut c = vec![0.0; n];
        c[fundamental] = 0.2;
        c[third] = -1.0;
        c[fourth] = 1.0;
        let mut colliding = vec![0.0; n];
        colliding[fundamental] = 1.0;
        colliding[third] = 1.0;
        let mut clear = vec![0.0; n];
        clear[fundamental] = 1.0;
        clear[fourth] = 1.0;
        let repr = ConsonanceRepresentationParams::default();
        let collision = evaluate(&space, &colliding, &du, &c, &repr).unwrap();
        let no_collision = evaluate(&space, &clear, &du, &c, &repr).unwrap();
        assert_eq!(collision.score, -0.4);
        assert_eq!(no_collision.score, 0.6);
        assert!(collision.level < no_collision.level);
    }

    #[test]
    fn positive_density_scale_changes_mass_but_not_fitness() {
        let space = space();
        let n = space.n_bins();
        let du = vec![0.5; n];
        let mut density = vec![0.0; n];
        let mut c = vec![0.0; n];
        density[3] = 2.0;
        density[7] = 4.0;
        c[3] = -0.5;
        c[7] = 1.0;
        let repr = ConsonanceRepresentationParams::default();
        let first = evaluate(&space, &density, &du, &c, &repr).unwrap();
        let scaled: Vec<_> = density.iter().map(|value| value * 17.0).collect();
        let second = evaluate(&space, &scaled, &du, &c, &repr).unwrap();
        assert!((first.score - second.score).abs() < 1e-6);
        assert!((first.level - second.level).abs() < 1e-6);
        assert_eq!(second.in_band_mass, first.in_band_mass * 17.0);
    }

    #[test]
    fn sigmoid_is_applied_after_score_average() {
        let space = space();
        let n = space.n_bins();
        let mut density = vec![0.0; n];
        let du = vec![1.0; n];
        let mut c = vec![0.0; n];
        density[2] = 1.0;
        density[3] = 1.0;
        c[2] = 0.0;
        c[3] = 2.0;
        let repr = ConsonanceRepresentationParams {
            beta: 2.0,
            theta: 0.3,
        };
        let fitness = evaluate(&space, &density, &du, &c, &repr).unwrap();
        assert_eq!(fitness.score, 1.0);
        assert_eq!(fitness.level, repr.level(1.0));
        assert!((fitness.level - (repr.level(0.0) + repr.level(2.0)) * 0.5).abs() > 0.1);
    }

    #[test]
    fn body_integral_of_raw_roughness_matches_sparse_partial_pair_sum() {
        let space = Log2Space::new(300.0, 2_000.0, 24);
        let (erb, du) = erb_grid(&space);
        let n = space.n_bins();
        let mut body = vec![0.0; n];
        let mut environment = vec![0.0; n];
        let body_bins = [(30, 0.25f32), (37, 0.75)];
        let environment_bins = [(32, 1.0f32), (41, 0.4)];
        for &(bin, mass) in &body_bins {
            body[bin] = mass / du[bin];
        }
        for &(bin, mass) in &environment_bins {
            environment[bin] = mass / du[bin];
        }
        let eps = 1e-12;
        let (environment_density, environment_mass) = normalize_density(&environment, &du, eps);
        assert!((environment_mass - 1.4).abs() < 1e-6);

        let params = KernelParams {
            mix_tail: 0.3,
            ..Default::default()
        };
        let kernel = RoughnessKernel::new(params, 0.005);
        let (raw_r_scan, _) = kernel.potential_r_from_log2_spectrum_density_with_grid(
            &environment_density,
            &space,
            &erb,
            &du,
        );
        let integrated = evaluate(
            &space,
            &body,
            &du,
            &raw_r_scan,
            &ConsonanceRepresentationParams::default(),
        )
        .unwrap();

        let sample_lut = |d: f32| -> f32 {
            let pos = d / kernel.erb_step + kernel.hw as f32;
            if pos < 0.0 || pos >= (kernel.lut.len() - 1) as f32 {
                return 0.0;
            }
            let left = pos.floor() as usize;
            let frac = pos - left as f32;
            kernel.lut[left] * (1.0 - frac) + kernel.lut[left + 1] * frac
        };
        let body_mass = body_bins.iter().map(|(_, mass)| mass).sum::<f32>();
        let mut pair_sum = 0.0f64;
        for &(probe, _) in &body_bins {
            for &(masker, _) in &environment_bins {
                let body_weight = f64::from(body[probe] * du[probe] / body_mass);
                let environment_weight = f64::from(environment_density[masker] * du[masker]);
                pair_sum += body_weight
                    * environment_weight
                    * f64::from(sample_lut(erb[probe] - erb[masker]));
            }
        }
        let pair_sum = pair_sum as f32;
        assert!(pair_sum > 0.0);
        assert!(
            (integrated.score - pair_sum).abs() <= 1e-6 + 5e-4 * pair_sum.abs(),
            "integrated={} pair_sum={pair_sum}",
            integrated.score
        );
        let d = erb[32] - erb[30];
        assert!(sample_lut(d) > sample_lut(-d));

        // Independent continuous formula; do not read the production LUT values.
        let continuous = |d: f64| {
            let u = d.abs() / f64::from(params.sethares_kappa_erb);
            let core = f64::from(params.sethares_gain)
                * ((-f64::from(params.sethares_b) * u).exp()
                    - (-f64::from(params.sethares_c) * u).exp());
            let asymmetry = 1.0
                + d.signum()
                    * f64::from(params.mix_tail)
                    * (-d.abs() / f64::from(params.tau_erb)).exp();
            let suppression = (1.0
                - (-d * d / (2.0 * f64::from(params.suppress_sigma_erb).powi(2))).exp())
            .powf(f64::from(params.suppress_pow));
            core * asymmetry * suppression
        };
        let normalization = (-(kernel.hw as i32)..=kernel.hw as i32)
            .map(|bin| continuous(f64::from(bin) * f64::from(kernel.erb_step)))
            .sum::<f64>()
            * f64::from(kernel.erb_step);
        let mut continuous_pairs = 0.0;
        for &(probe, probe_mass) in &body_bins {
            for &(masker, masker_mass) in &environment_bins {
                continuous_pairs += f64::from(probe_mass / body_mass)
                    * f64::from(masker_mass / (environment_mass + eps))
                    * continuous(f64::from(erb[probe]) - f64::from(erb[masker]))
                    / normalization;
            }
        }
        assert!(
            (f64::from(integrated.score) - continuous_pairs).abs()
                <= 1e-6 + 5e-4 * continuous_pairs.abs(),
            "integrated={} continuous_pairs={continuous_pairs}",
            integrated.score
        );
    }

    #[test]
    fn averaging_raw_roughness_then_saturating_changes_the_c_rule() {
        let space = space();
        let n = space.n_bins();
        let du = vec![1.0; n];
        let mut body = vec![0.0; n];
        let mut c = vec![0.0; n];
        let first = 2;
        let second = 3;
        body[first] = 1.0;
        body[second] = 1.0;
        let raw_r = [0.0, 4.0];
        let h = [1.0, 0.0];
        let kernel = ConsonanceKernel::default();
        c[first] = kernel.score(h[0], roughness_ratio_to_state01(raw_r[0], 1.0));
        c[second] = kernel.score(h[1], roughness_ratio_to_state01(raw_r[1], 1.0));
        let body_mean_c = evaluate(&space, &body, &du, &c, &Default::default())
            .unwrap()
            .score;
        let raw_r_mean_then_c = kernel.score(
            (h[0] + h[1]) * 0.5,
            roughness_ratio_to_state01((raw_r[0] + raw_r[1]) * 0.5, 1.0),
        );
        assert!((body_mean_c - raw_r_mean_then_c).abs() > 0.01);
    }

    #[test]
    fn missing_or_invalid_mass_is_never_a_zero_score() {
        let space = space();
        let n = space.n_bins();
        let zero = vec![0.0; n];
        let du = vec![1.0; n];
        let c = vec![0.0; n];
        let repr = ConsonanceRepresentationParams::default();
        assert_eq!(
            evaluate(&space, &zero, &du, &c, &repr),
            Err(Unsupported::NoInBandMass)
        );
        let mut bad = zero.clone();
        bad[0] = f32::NAN;
        assert_eq!(
            evaluate(&space, &bad, &du, &c, &repr),
            Err(Unsupported::InvalidInput)
        );
        bad[0] = -1.0;
        assert_eq!(
            evaluate(&space, &bad, &du, &c, &repr),
            Err(Unsupported::InvalidInput)
        );
        let mut bad_score = c.clone();
        bad_score[0] = f32::INFINITY;
        assert_eq!(
            evaluate(&space, &zero, &du, &bad_score, &repr),
            Err(Unsupported::InvalidInput)
        );
        let mut bad_du = du.clone();
        bad_du[0] = 0.0;
        assert_eq!(
            evaluate(&space, &zero, &bad_du, &c, &repr),
            Err(Unsupported::InvalidInput)
        );
    }

    #[test]
    fn scan_boundaries_assert_in_release_too() {
        let space = space();
        let n = space.n_bins();
        let density = vec![0.0; n];
        let du = vec![1.0; n];
        let c = vec![0.0; n];
        let repr = ConsonanceRepresentationParams::default();
        assert!(
            std::panic::catch_unwind(|| { evaluate(&space, &density[..n - 1], &du, &c, &repr) })
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| { evaluate(&space, &density, &du[..n - 1], &c, &repr) })
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| { evaluate(&space, &density, &du, &c[..n - 1], &repr) })
                .is_err()
        );
    }
}
