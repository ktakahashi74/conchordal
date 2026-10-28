//! A unit-variance process with a 1/f spectrum, shared by Rain's impact rate and a
//! Population's participation. Design: `docs/design-notes/rain.md` (Sway).

use rand::{RngExt, rngs::SmallRng};
use std::f32::consts::TAU;

/// One AR(1) per octave from 0.4 s to 102 s. Equal variances at octave spacing give a
/// 1/f spectrum between about 0.0016 and 0.4 Hz. The fastest stays below the pulse range
/// (about 0.5 Hz and up), so the sway is heard as changing activity, not as a beat.
const TAU_MIN_SEC: f32 = 0.4;
const POLES: usize = 9;

#[derive(Clone, Debug)]
pub(crate) struct Sway {
    state: [f32; POLES],
}

impl Sway {
    /// Start from the stationary distribution, so the sway is present at once.
    pub(crate) fn new(rng: &mut SmallRng) -> Self {
        Self {
            state: std::array::from_fn(|_| normal(rng)),
        }
    }

    /// Advance by `dt` seconds and return the unit-variance value.
    pub(crate) fn advance(&mut self, dt: f32, rng: &mut SmallRng) -> f32 {
        let mut g = 0.0f32;
        for (k, x) in self.state.iter_mut().enumerate() {
            let a = (-dt / (TAU_MIN_SEC * (1u32 << k) as f32)).exp();
            let n = normal(rng);
            *x = a * *x + (1.0 - a * a).sqrt() * n;
            g += *x;
        }
        g / (POLES as f32).sqrt()
    }
}

fn normal(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}
