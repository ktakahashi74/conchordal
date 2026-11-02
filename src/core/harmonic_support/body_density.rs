//! Nominal single-body partial density for reading auditory fields.
//! Compression applies within the self-report, never to a hypothetical mixture.

use crate::core::log2space::Log2Space;
use crate::core::roughness_kernel::erb_grid;
use crate::core::utils::a_weighting_gain_pow;

/// Convert peak sinusoidal amplitudes to the selected front end's steady S4-S6
/// law. The result is a body self-report, not a prediction of future perception.
pub(crate) fn project(
    amplitudes: &[f32],
    fundamental: f32,
    space: &Log2Space,
    ref_power: f32,
    loudness_exp: f32,
) -> Vec<(isize, f32)> {
    space.assert_scan_len_named(amplitudes, "body_spectral_amplitudes");
    let (_, du) = erb_grid(space);
    let components: Vec<_> = amplitudes
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| (a.is_finite() && a > 0.).then_some((i, 0.5 * a * a)))
        .collect();
    let origin = space.nearest_index(fundamental) as isize;
    components
        .iter()
        .filter_map(|&(i, power)| {
            let f = space.freq_of_index(i);
            let bandwidth = 1.0186 * (24.7 + f64::from(f) / 9.265);
            let local = components
                .iter()
                .map(|&(j, p)| {
                    f64::from(p)
                        * (1.
                            + ((f64::from(space.freq_of_index(j)) - f64::from(f)) / bandwidth)
                                .powi(2))
                        .powi(-4)
                })
                .sum::<f64>() as f32;
            let gate = (1. - 1e-6 * du[i] / local).max(0.);
            let density = gate * power / local
                * (a_weighting_gain_pow(f) * local / ref_power.max(1e-12))
                    .powf(loudness_exp.max(0.01))
                / du[i];
            (density.is_finite() && density > 0.).then_some((i as isize - origin, density))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_and_critical_band_compression() {
        let space = Log2Space::new(220., 880., 120);
        let mut amplitudes = vec![0.; space.n_bins()];
        assert!(project(&amplitudes, 440., &space, 1., 0.3).is_empty());
        let i = space.nearest_index(440.);
        amplitudes[i] = 0.04;
        let one = project(&amplitudes, 440., &space, 1., 0.3);
        amplitudes[i + 1] = 0.04;
        let two = project(&amplitudes, 440., &space, 1., 0.3);
        assert!(two[0].1 < one[0].1, "shared critical-band compression");
        assert!(
            two.iter()
                .all(|&(_, value)| value.is_finite() && value > 0.)
        );
    }
}
