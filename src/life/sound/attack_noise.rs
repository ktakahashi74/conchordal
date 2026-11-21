use super::oscillator_bank::splitmix64_next;
pub(crate) const NOISE_LANES: usize = 8;
pub(crate) const ATTACK_TAU: f32 = 0.015;

/// Stratified random-phase multisine. The same compact description supplies
/// the rotating render lanes and a random-access forecast of the short burst.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Burst {
    pub onset: u64,
    pub seed: u64,
    pub fs: f32,
    pub low: f32,
    pub high: f32,
    pub rms: f32,
}

impl Burst {
    // At 18 time constants the burst is below -156 dB.
    fn samples(self) -> usize {
        (18. * self.fs * ATTACK_TAU).ceil() as usize
    }

    pub(crate) fn end(self) -> u64 {
        self.onset.saturating_add(self.samples() as u64)
    }

    fn component(self, lane: usize) -> (f64, f64) {
        let mut key =
            self.seed ^ self.onset.rotate_left(29) ^ (lane as u64).wrapping_mul(0xd6e8feb86659fd93);
        let mut uniform = || (splitmix64_next(&mut key) >> 11) as f64 / (1u64 << 53) as f64;
        let hz = f64::from(self.low)
            + f64::from(self.high - self.low) * (lane as f64 + uniform()) / NOISE_LANES as f64;
        (
            std::f64::consts::TAU * hz / f64::from(self.fs),
            std::f64::consts::TAU * uniform(),
        )
    }

    pub(crate) fn point(self, lane: usize, tick: u64) -> [f64; 4] {
        let (omega, phase) = self.component(lane);
        let age = (i128::from(tick) - i128::from(self.onset)) as f64;
        let decay = -1. / f64::from(self.fs * ATTACK_TAU);
        let amplitude =
            f64::from(self.rms) * (2. / NOISE_LANES as f64).sqrt() * (age * decay).exp();
        let (sin, cos) = (phase + age * omega).sin_cos();
        [amplitude * cos, amplitude * sin, omega, decay]
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct NoiseRender {
    pub burst: Option<Burst>,
    lanes: [[f32; 4]; NOISE_LANES],
    remaining: usize,
    pub end_tick: u64,
}

impl NoiseRender {
    pub(crate) fn start(&mut self, burst: Burst) {
        self.burst = Some(burst);
        self.end_tick = burst.end();
        let decay = (-1. / (burst.fs * ATTACK_TAU)).exp();
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            let [x, y, omega, _] = burst.point(i, burst.onset);
            let (s, c) = omega.sin_cos();
            *lane = [x as f32, y as f32, c as f32 * decay, s as f32 * decay];
        }
        self.remaining = burst.samples();
    }

    pub(crate) fn sample(&mut self) -> f32 {
        if self.remaining == 0 {
            self.burst = None;
            return 0.;
        }
        self.remaining -= 1;
        let mut sum = 0.;
        for [x, y, c, s] in &mut self.lanes {
            sum += *y;
            (*x, *y) = (*x * *c - *y * *s, *x * *s + *y * *c);
        }
        sum
    }

    pub(crate) fn project(&self, amps: &mut [f32], space: &crate::core::log2space::Log2Space) {
        if let Some(burst) = self.burst {
            let age = (burst.samples() - self.remaining).saturating_sub(1);
            let tick = burst.onset.saturating_add(age as u64);
            for i in 0..NOISE_LANES {
                let [x, y, omega, _] = burst.point(i, tick);
                let hz = (omega * f64::from(burst.fs) / std::f64::consts::TAU) as f32;
                super::spectral::add_log2_energy(amps, space, hz, x.hypot(y) as f32);
            }
        }
    }
}
