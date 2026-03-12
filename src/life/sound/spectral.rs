use crate::core::log2space::Log2Space;
use crate::life::scenario::{HarmonicMode, TimbreGenotype};

const DARKEST_SPECTRAL_SLOPE: f32 = 1.2;
const BRIGHTEST_SPECTRAL_SLOPE: f32 = 0.2;

pub(crate) fn spectral_slope_from_brightness(brightness: f32) -> f32 {
    let brightness = brightness.clamp(0.0, 1.0);
    DARKEST_SPECTRAL_SLOPE + (BRIGHTEST_SPECTRAL_SLOPE - DARKEST_SPECTRAL_SLOPE) * brightness
}

pub(crate) fn brightness_from_spectral_slope(spectral_slope: f32) -> f32 {
    let span = DARKEST_SPECTRAL_SLOPE - BRIGHTEST_SPECTRAL_SLOPE;
    if span <= 0.0 {
        return 0.0;
    }
    ((DARKEST_SPECTRAL_SLOPE - spectral_slope) / span).clamp(0.0, 1.0)
}

pub(crate) fn harmonic_ratio(genotype: &TimbreGenotype, harmonic_index_1: usize) -> f32 {
    let kf = harmonic_index_1.max(1) as f32;
    let base = match genotype.mode {
        HarmonicMode::Harmonic => kf,
        HarmonicMode::Metallic => kf.powf(1.4),
    };
    let stretch = 1.0 + genotype.stiffness * kf * kf;
    (base * stretch).max(0.1)
}

pub(crate) fn harmonic_gain(
    genotype: &TimbreGenotype,
    harmonic_index_1: usize,
    energy: f32,
) -> f32 {
    let kf = harmonic_index_1.max(1) as f32;
    let spectral_slope = genotype.spectral_slope.max(0.0);
    let mut amp = 1.0 / kf.powf(spectral_slope.max(1e-6));
    if harmonic_index_1.max(1).is_multiple_of(2) {
        amp *= 1.0 - genotype.comb.clamp(0.0, 1.0);
    }
    let damping = genotype.damping.max(0.0);
    if damping > 0.0 {
        let energy = energy.clamp(0.0, 1.0);
        amp *= energy.powf(damping * kf);
    }
    amp
}

pub(crate) fn add_log2_energy(amps: &mut [f32], space: &Log2Space, freq_hz: f32, energy: f32) {
    if !freq_hz.is_finite() || energy == 0.0 {
        return;
    }
    if freq_hz < space.fmin || freq_hz > space.fmax {
        return;
    }
    let log_f = freq_hz.log2();
    let base = space.centers_log2[0];
    let step = space.step();
    let pos = (log_f - base) / step;
    let idx_base = pos.floor();
    let idx = idx_base as isize;
    if idx < 0 {
        return;
    }
    let idx = idx as usize;
    let frac = pos - idx_base;
    if idx + 1 < amps.len() {
        amps[idx] += energy * (1.0 - frac);
        amps[idx + 1] += energy * frac;
    } else if idx < amps.len() {
        amps[idx] += energy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_log2_energy_reference(amps: &mut [f32], space: &Log2Space, freq_hz: f32, energy: f32) {
        if !freq_hz.is_finite() || energy == 0.0 {
            return;
        }
        if freq_hz < space.fmin || freq_hz > space.fmax {
            return;
        }
        let log_f = freq_hz.log2();
        let base = space.centers_log2[0];
        let step = space.step();
        let pos = (log_f - base) / step;
        let idx_base = pos.floor();
        let idx = idx_base as isize;
        if idx < 0 {
            return;
        }
        let idx = idx as usize;
        let frac = pos - idx_base;
        if idx + 1 < amps.len() {
            amps[idx] += energy * (1.0 - frac);
            amps[idx + 1] += energy * frac;
        } else if idx < amps.len() {
            amps[idx] += energy;
        }
    }

    fn harmonic_ratio_reference(genotype: &TimbreGenotype, k: usize) -> f32 {
        let kf = k.max(1) as f32;
        let base = match genotype.mode {
            HarmonicMode::Harmonic => kf,
            HarmonicMode::Metallic => kf.powf(1.4),
        };
        let stretch = 1.0 + genotype.stiffness * kf * kf;
        (base * stretch).max(0.1)
    }

    fn harmonic_gain_reference(genotype: &TimbreGenotype, k: usize, energy: f32) -> f32 {
        let kf = k.max(1) as f32;
        let spectral_slope = genotype.spectral_slope.max(0.0);
        let mut amp = 1.0 / kf.powf(spectral_slope.max(1e-6));
        if k.max(1).is_multiple_of(2) {
            amp *= 1.0 - genotype.comb.clamp(0.0, 1.0);
        }
        let damping = genotype.damping.max(0.0);
        if damping > 0.0 {
            let energy = energy.clamp(0.0, 1.0);
            amp *= energy.powf(damping * kf);
        }
        amp
    }

    #[test]
    fn add_log2_energy_matches_reference() {
        let space = Log2Space::new(55.0, 1760.0, 24);
        let mut got = vec![0.0f32; space.n_bins()];
        let mut want = vec![0.0f32; space.n_bins()];

        let cases = [
            (55.0, 0.25),
            (110.0, 1.0),
            (220.25, 0.75),
            (440.0, -0.5),
            (1760.0, 0.125),
            (10.0, 1.0),
            (5000.0, 0.8),
            (f32::NAN, 1.0),
            (440.0, 0.0),
        ];
        for (freq_hz, energy) in cases {
            add_log2_energy(&mut got, &space, freq_hz, energy);
            add_log2_energy_reference(&mut want, &space, freq_hz, energy);
        }

        for (lhs, rhs) in got.iter().zip(want.iter()) {
            assert!((lhs - rhs).abs() <= 1e-6);
        }
    }

    #[test]
    fn harmonic_ratio_matches_reference_for_modes() {
        let mut genotype = TimbreGenotype::default();
        genotype.stiffness = 0.013;
        for mode in [HarmonicMode::Harmonic, HarmonicMode::Metallic] {
            genotype.mode = mode;
            for k in 1..=32 {
                let got = harmonic_ratio(&genotype, k);
                let want = harmonic_ratio_reference(&genotype, k);
                assert!((got - want).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn brightness_maps_to_expected_spectral_slope_range() {
        for (brightness, spectral_slope) in [(0.0, 1.2), (0.6, 0.6), (1.0, 0.2)] {
            let got_slope = spectral_slope_from_brightness(brightness);
            assert!((got_slope - spectral_slope).abs() <= 1e-6);
            let got_brightness = brightness_from_spectral_slope(spectral_slope);
            assert!((got_brightness - brightness).abs() <= 1e-6);
        }
    }

    #[test]
    fn brighter_public_control_retains_more_upper_harmonics() {
        let dark = TimbreGenotype {
            spectral_slope: spectral_slope_from_brightness(0.0),
            ..TimbreGenotype::default()
        };
        let bright = TimbreGenotype {
            spectral_slope: spectral_slope_from_brightness(1.0),
            ..TimbreGenotype::default()
        };

        let dark_gain = harmonic_gain(&dark, 8, 1.0);
        let bright_gain = harmonic_gain(&bright, 8, 1.0);
        assert!(bright_gain > dark_gain);
    }

    #[test]
    fn harmonic_gain_matches_reference_curve() {
        let genotype = TimbreGenotype {
            spectral_slope: 0.83,
            comb: 0.37,
            damping: 0.42,
            ..TimbreGenotype::default()
        };
        for k in 1..=32 {
            for energy in [0.0, 0.1, 0.5, 1.0, 1.7] {
                let got = harmonic_gain(&genotype, k, energy);
                let want = harmonic_gain_reference(&genotype, k, energy);
                assert!((got - want).abs() <= 1e-6);
            }
        }
    }
}
