//! A Voice generation's synthesis clock. Indexing its LFO and octave noise by
//! age lets overlapping tones, silent gaps and read-only forecasts share one
//! trajectory without locking, advancing a reader, or replaying the gap.
use super::oscillator_bank::splitmix64_next;

pub(crate) const PERIOD: u64 = 8;
const OCTAVES: u32 = 10;
const PINK_GAIN: f64 = 0.00004;

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MotionClock {
    pub(crate) birth: u64,
    pub(crate) seed: u64,
}

impl MotionClock {
    fn noise_point(self, octave: u32, block: u64) -> f64 {
        let mut address = self.seed ^ (u64::from(octave) << 56) ^ block;
        (splitmix64_next(&mut address) >> 40) as f64 / 16777216.
    }

    /// Equal-power octave bands give approximately 1/f power density. Each
    /// band's held value is the difference of adjacent independent points;
    /// this suppresses DC and gives an exact telescoping integral per band.
    pub(crate) fn offset(self, tick: u64, fs: f32, rate: f32, depth: f32, jitter: f32) -> f32 {
        let block = tick.saturating_sub(self.birth) / PERIOD;
        let angle = std::f64::consts::TAU * f64::from(rate) * PERIOD as f64 / f64::from(fs);
        let mut pink = 0.;
        for octave in 0..OCTAVES {
            let index = block >> octave;
            pink += self.noise_point(octave, index + 1) - self.noise_point(octave, index);
        }
        (depth as f64 * ((block + 1) as f64 * angle).sin() + jitter as f64 * PINK_GAIN * pink)
            as f32
    }

    /// Sum of held fractional offsets over [birth, end), in samples.
    #[cfg(any(test, feature = "research"))]
    pub(super) fn integral(self, end: u64, fs: f32, depth: f32, jitter: f32) -> f64 {
        let age = end.saturating_sub(self.birth);
        let n = (age / PERIOD) as f64;
        let angle = std::f64::consts::TAU * 5. * PERIOD as f64 / f64::from(fs);
        let half = angle * 0.5;
        let sine_sum = if half.sin().abs() < f64::EPSILON {
            0.
        } else {
            (n * half).sin() * ((n + 1.) * half).sin() / half.sin()
        };
        let vibrato = PERIOD as f64 * sine_sum + (age % PERIOD) as f64 * ((n + 1.) * angle).sin();
        let mut pink = 0.;
        for octave in 0..OCTAVES {
            let width = PERIOD << octave;
            let index = age / width;
            let point = self.noise_point(octave, index);
            pink += width as f64 * (point - self.noise_point(octave, 0))
                + (age % width) as f64 * (self.noise_point(octave, index + 1) - point);
        }
        depth as f64 * vibrato + jitter as f64 * PINK_GAIN * pink
    }
}
