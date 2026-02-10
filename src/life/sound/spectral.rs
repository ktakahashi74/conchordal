use crate::core::log2space::Log2Space;
use crate::life::scenario::{HarmonicMode, TimbreGenotype};

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
    let slope = genotype.brightness.max(0.0);
    let mut amp = 1.0 / kf.powf(slope.max(1e-6));
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
        let slope = genotype.brightness.max(0.0);
        let mut amp = 1.0 / kf.powf(slope.max(1e-6));
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
    fn harmonic_gain_matches_reference_curve() {
        let genotype = TimbreGenotype {
            brightness: 0.83,
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
