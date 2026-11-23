use super::*;

impl MotionClock {
    /// Sum of held fractional offsets over [birth, end), in samples.
    pub(in crate::life::sound) fn integral(
        self,
        end: u64,
        fs: f32,
        depth: f32,
        jitter: f32,
    ) -> f64 {
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
