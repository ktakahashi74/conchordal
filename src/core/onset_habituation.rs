//! Audio-only attack history with sample-counted recovery after front-end reset.

use crate::core::erb::hz_to_erb;
use crate::core::habituation::HabituationParams;
use crate::core::log2space::Log2Space;
use crate::core::onset::REFRACTORY_SEC;

const POWER_FLOOR: f32 = 1e-6 * 0.1875;

#[derive(Clone, Debug, Default)]
struct Band {
    previous_energy: Option<f32>,
    baseline: f32,
    variance: f32,
    previous_drive: f32,
    refractory_left: f32,
    attack_at: Option<u64>,
    warmup_samples: usize,
}

impl Band {
    fn process(&mut self, energy: f32, dt: f32, end: u64) {
        let energy = energy.max(POWER_FLOOR);
        let drive = self.previous_energy.map_or(0.0, |previous| {
            // A difference of logs also avoids overflow in the power ratio.
            (10.0 * (energy.log10() - previous.log10())).max(0.0)
        });
        self.previous_energy = Some(energy);
        let a = (-dt).exp();
        self.baseline = a * self.baseline + (1.0 - a) * drive;
        let dev = drive - self.baseline;
        self.variance = a * self.variance + (1.0 - a) * dev * dev;
        let threshold = (self.baseline + 2.0 * self.variance.max(1e-6).sqrt()).max(1.0);
        self.refractory_left = (self.refractory_left - dt).max(0.0);
        if self.refractory_left <= 0.0 && self.previous_drive < threshold && drive >= threshold {
            self.attack_at = Some(end);
            self.refractory_left = REFRACTORY_SEC;
        }
        self.previous_drive = drive;
    }
}

#[derive(Clone)]
pub(crate) struct OnsetHabituation {
    bands: Vec<Band>,
    weights: Vec<Vec<(usize, f32)>>,
    drive: Vec<f32>,
    pub(crate) state: Vec<f32>,
    params: HabituationParams,
    window_samples: f64,
    fs: f32,
    warmup_samples: usize,
    pub(crate) settling_samples: Vec<usize>,
}

impl OnsetHabituation {
    pub(crate) fn new(
        space: &Log2Space,
        centers: &[f32],
        params: HabituationParams,
        period: f64,
        fs: f32,
    ) -> Self {
        let weights = centers
            .iter()
            .map(|&center| {
                let center = hz_to_erb(center);
                space
                    .centers_hz
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &hz)| {
                        let distance = hz_to_erb(hz) - center;
                        (distance.abs() <= 1.5)
                            .then(|| (i, crate::core::float::unit_gaussian(distance, 0.5)))
                    })
                    .collect()
            })
            .collect();
        Self {
            bands: vec![Band::default(); centers.len()],
            weights,
            drive: vec![0.0; space.n_bins()],
            state: vec![0.0; space.n_bins()],
            params,
            window_samples: period * f64::from(fs),
            fs,
            warmup_samples: 0,
            settling_samples: vec![0; centers.len()],
        }
    }

    pub(crate) fn process(&mut self, energies: &[f32], samples: usize, end: u64) {
        assert_eq!(energies.len(), self.bands.len());
        let dt = samples as f32 / self.fs;
        let warming = self.warmup_samples > 0;
        self.warmup_samples = self.warmup_samples.saturating_sub(samples);
        self.drive.fill(0.0);
        for ((band, &energy), weights) in self.bands.iter_mut().zip(energies).zip(&self.weights) {
            if band.warmup_samples > 0 {
                band.warmup_samples = band.warmup_samples.saturating_sub(samples);
                continue;
            }
            band.process(energy, dt, end);
            if band
                .attack_at
                .is_some_and(|at| (end.saturating_sub(at) as f64) < self.window_samples)
            {
                for &(bin, weight) in weights {
                    self.drive[bin] = self.drive[bin].max(weight);
                }
            }
        }
        if self.params.enabled && !warming {
            let ae = (-dt * std::f32::consts::LN_10 / self.params.satiation_sec).exp();
            let ar = (-dt * std::f32::consts::LN_10 / self.params.recovery_sec).exp();
            for (state, &drive) in self.state.iter_mut().zip(&self.drive) {
                let a = if drive > *state { ae } else { ar };
                *state = a * *state + (1.0 - a) * drive;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn attacks_at(&self, end: u64) -> impl Iterator<Item = usize> + '_ {
        self.bands
            .iter()
            .enumerate()
            .filter_map(move |(c, band)| (band.attack_at == Some(end)).then_some(c))
    }

    pub(crate) fn known(&self) -> bool {
        self.warmup_samples == 0 && self.bands.iter().all(|band| band.previous_energy.is_some())
    }

    pub(crate) fn missing(&mut self, samples: u64) {
        let dt = samples as f32 / self.fs;
        for band in &mut self.bands {
            band.previous_energy = None;
            band.attack_at = None;
            band.refractory_left = (band.refractory_left - dt).max(0.0);
        }
    }

    pub(crate) fn restart_frontend(
        &mut self,
        window_samples: usize,
        band_hops: &[usize],
        hop: usize,
    ) {
        assert_eq!(band_hops.len(), self.bands.len());
        self.warmup_samples = window_samples;
        for ((band, &hops), &settling) in self
            .bands
            .iter_mut()
            .zip(band_hops)
            .zip(&self.settling_samples)
        {
            band.warmup_samples = window_samples.max(hops.saturating_mul(hop)).max(settling);
            // Hold the shared scan until every band can supply fresh evidence.
            self.warmup_samples = self.warmup_samples.max(band.warmup_samples);
            band.previous_energy = None;
            band.attack_at = None;
        }
    }

    pub(crate) fn reset(&mut self) {
        self.bands.fill(Band::default());
        self.warmup_samples = 0;
        self.state.fill(0.0);
        self.drive.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::auditory_frontend::AuditoryFrontEnd;

    #[test]
    fn restart_freezes_statistics_and_primes_each_band_after_its_existing_window() {
        let space = Log2Space::new(1000.0, 4000.0, 96);
        let mut observer = OnsetHabituation::new(
            &space,
            &[1000.0, 1200.0],
            HabituationParams::default(),
            0.5,
            48000.0,
        );
        observer.process(&[0.0, 0.0], 512, 512);
        observer.process(&[0.1, 0.1], 512, 1024);
        observer.restart_frontend(3 * 512, &[1, 5], 512);
        let before = observer.clone();
        for frame in 0..3 {
            observer.process(&[100.0, 100.0], 512, (frame + 3) * 512);
            assert_eq!(observer.state, before.state);
            for (band, saved) in observer.bands.iter().zip(&before.bands) {
                assert_eq!(band.baseline, saved.baseline);
                assert_eq!(band.variance, saved.variance);
                assert_eq!(band.refractory_left, saved.refractory_left);
                assert_eq!(band.previous_energy, None);
                assert_eq!(band.attack_at, None);
            }
        }
        for frame in 3..6 {
            observer.process(&[100.0, 100.0], 512, (frame + 3) * 512);
            assert_eq!(observer.bands[0].previous_drive, 0.0);
            assert_eq!(observer.bands[0].attack_at, None);
            if frame < 5 {
                assert_eq!(observer.state, before.state);
                assert_eq!(observer.bands[1].previous_energy, None);
                assert_eq!(observer.bands[1].baseline, before.bands[1].baseline);
                assert_eq!(observer.bands[1].variance, before.bands[1].variance);
                assert_eq!(
                    observer.bands[1].refractory_left,
                    before.bands[1].refractory_left
                );
            } else {
                assert_eq!(observer.bands[1].previous_energy, Some(100.0));
                assert_eq!(observer.bands[1].previous_drive, 0.0);
                assert_eq!(observer.bands[1].attack_at, None);
            }
        }
    }

    #[test]
    fn relaxation_and_gaussian_support() {
        let space = Log2Space::new(1000.0, 4000.0, 96);
        let mut observer = OnsetHabituation::new(
            &space,
            &[1000.0],
            HabituationParams::default(),
            0.5,
            48000.0,
        );
        // A drive held at one for exactly five seconds, independent of detection.
        for end in 1..=500 {
            observer.bands[0].attack_at = Some(end * 480);
            observer.process(&[1.0], 480, end * 480);
        }
        assert!((observer.state[0] - 0.9).abs() < 1e-5);
        assert_eq!(*observer.state.last().unwrap(), 0.0);
        let initial = observer.state[0];
        observer.bands[0].attack_at = None;
        for end in 501..=1300 {
            observer.process(&[1.0], 480, end * 480);
        }
        assert!((observer.state[0] - 0.1 * initial).abs() < 1e-5);
    }

    #[test]
    fn pcm_repeated_attacks_and_stationary_tone() {
        let (_, params) = crate::runtime::body_metabolism_test_core();
        let fs = params.fs;
        let hop = 512;
        let space = Log2Space::new(55.0, 8000.0, 96);
        for repeated in [true, false] {
            let mut frontend = AuditoryFrontEnd::selected(space.clone(), fs, hop, 16384);
            let mut observer = OnsetHabituation::new(
                &space,
                frontend.band_energy().0,
                HabituationParams::default(),
                0.5,
                fs,
            );
            let center = frontend
                .band_energy()
                .0
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - 440.0).abs().total_cmp(&(*b - 440.0).abs()))
                .unwrap()
                .0;
            let mut detections = Vec::new();
            let mut late_attacks = Vec::new();
            let mut reference = vec![0.0_f64; space.n_bins()];
            let mut audio = vec![0.0; hop];
            let mut max_error = 0.0_f64;
            for frame in 0..(12.0 * fs / hop as f32) as u64 {
                for (i, sample) in audio.iter_mut().enumerate() {
                    let t = (frame * hop as u64 + i as u64) as f64 / f64::from(fs);
                    let elapsed = t - 0.5;
                    let envelope = if elapsed < 0.0 {
                        0.0
                    } else if repeated {
                        (-elapsed.rem_euclid(0.5) / 0.025).exp()
                    } else {
                        1.0
                    };
                    *sample = (0.1 * envelope * (std::f64::consts::TAU * 440.0 * t).sin()) as f32;
                }
                frontend.process(&audio, &params);
                let end = (frame + 1) * hop as u64;
                observer.process(frontend.band_energy().1, hop, end);
                if observer.bands[center].attack_at == Some(end) {
                    detections.push(end as f64 / f64::from(fs));
                }
                if !repeated && end as f64 / f64::from(fs) > 1.0 {
                    for (channel, band) in observer.bands.iter().enumerate() {
                        if band.attack_at == Some(end) {
                            late_attacks.push((channel, end));
                        }
                    }
                }
                for ((expected, &drive), &actual) in reference
                    .iter_mut()
                    .zip(&observer.drive)
                    .zip(&observer.state)
                {
                    let tau = if f64::from(drive) > *expected {
                        5.0
                    } else {
                        8.0
                    } / std::f64::consts::LN_10;
                    let a = (-(hop as f64 / f64::from(fs)) / tau).exp();
                    *expected = a * *expected + (1.0 - a) * f64::from(drive);
                    max_error = max_error.max((*expected - f64::from(actual)).abs());
                }
            }
            println!(
                "repeated={repeated} center_hz={} detections={detections:?} late_attacks={late_attacks:?} max_error={max_error}",
                frontend.band_energy().0[center]
            );
            assert!(
                max_error < 2e-5,
                "f32 recurrence compared with f64 reference"
            );
            if repeated {
                for attack in 1..23 {
                    let start = 0.5 + attack as f64 * 0.5;
                    assert!(
                        detections.iter().any(|&t| t >= start && t < start + 0.5),
                        "missed attack at {start}"
                    );
                }
            } else {
                assert!(
                    late_attacks.is_empty(),
                    "stationary pure tone must not retrigger after P"
                );
            }
        }
    }

    #[test]
    fn missing_preserves_state_and_discards_cross_gap_energy() {
        let space = Log2Space::new(1000.0, 4000.0, 96);
        let mut observer = OnsetHabituation::new(
            &space,
            &[1000.0],
            HabituationParams::default(),
            0.5,
            48000.0,
        );
        observer.process(&[0.0], 512, 512);
        observer.process(&[0.1], 512, 1024);
        let before = observer.state.clone();
        let baseline = observer.bands[0].baseline;
        observer.missing(480000);
        assert_eq!(observer.state, before);
        assert_eq!(observer.bands[0].baseline, baseline);
        observer.process(&[1.0], 512, 481536);
        assert_eq!(observer.bands[0].attack_at, None);
        assert_eq!(observer.bands[0].previous_drive, 0.0);
        observer.reset();
        assert!(observer.state.iter().all(|&s| s == 0.0));
    }
}
