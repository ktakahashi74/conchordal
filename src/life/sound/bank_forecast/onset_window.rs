use super::*;

impl BankForecast {
    /// Slowly varying gain of one lane relative to `lane_at` at `reference`: the damped
    /// partial gain and the excitation gain of an oscillator. A resonator's decay stays in
    /// the carrier, so its ratio is one.
    pub(crate) fn gain_ratio(&self, lane: usize, tick: u64, reference: u64) -> Option<f64> {
        if matches!(self.response, Response::Resonator { .. }) {
            return Some(1.);
        }
        let magnitude = |at| {
            self.lane_at(lane, at)
                .map(|[x, y, ..]: [f64; 4]| x.hypot(y))
        };
        let base = magnitude(reference)?;
        (base > 0.).then_some(magnitude(tick)? / base)
    }

    /// `[x, y, omega, log_decay]` of one lane, where `y` is its unit-envelope output.
    pub(crate) fn lane_at(&self, lane: usize, tick: u64) -> Option<[f64; 4]> {
        // A span that contains the onset is referenced before it; gains stay at the onset.
        let steps = (i128::from(tick) - i128::from(self.first_sample)) as f64 + 1.;
        let phase_tick = tick;
        let tick = tick.max(self.first_sample);
        let lane = self.lanes[..self.len].get(lane)?;
        let [x, y] = lane.state.map(f64::from);
        let result = match self.response {
            Response::Oscillator {
                drive_env,
                drive_decay,
                spectral_env,
                spectral_decay,
                spectral_floor,
                refresh_at,
                refresh_period,
                motion,
            } => {
                let [c, s] = lane.step.map(f64::from);
                let omega = match motion {
                    Some((_, fs, _)) => std::f64::consts::TAU * c * s / f64::from(fs),
                    None => s.atan2(c),
                };
                let mask = if tick < refresh_at {
                    f64::from(lane.held)
                } else {
                    // The renderer holds each damped gain until its next pitch refresh.
                    let refreshed = tick - (tick - refresh_at) % refresh_period.max(1);
                    let env = f64::from(spectral_env)
                        * f64::from(spectral_decay).powf((refreshed - self.first_sample) as f64);
                    let floor = f64::from(spectral_floor);
                    let energy = (floor + (1. - floor) * env).clamp(floor, 1.);
                    f64::from(lane.gain) * energy.powf(f64::from(lane.damping))
                };
                let drive = f64::from(drive_env)
                    * f64::from(drive_decay).powf((tick - self.first_sample) as f64);
                // Oscillator radius is maintained by the renderer, unlike modal decay.
                let gain = mask * (1. + 0.25 * drive);
                let (phase, omega) = if let Some((clock, fs, amount)) = motion {
                    let depth = amount * 0.02;
                    let offset_samples =
                        clock.integral(phase_tick.saturating_add(1), fs, depth, amount)
                            - clock.integral(self.first_sample, fs, depth, amount);
                    (
                        omega * (steps + offset_samples),
                        omega * (1. + f64::from(clock.offset(phase_tick, fs, 5., depth, amount))),
                    )
                } else {
                    (omega * steps, omega)
                };
                let (sin, cos) = phase.sin_cos();
                [
                    (x * cos - y * sin) * gain,
                    (x * sin + y * cos) * gain,
                    omega,
                    0.,
                ]
            }
            Response::Resonator { .. } => {
                let [r, e] = lane.step.map(f64::from);
                // The magic-circle update is `r` times a unit-determinant rotation.
                let cosine = 1. - r * e * e / 2.;
                let sine = (1. - cosine * cosine).sqrt();
                if r <= 0. || sine <= 0. {
                    return None;
                }
                let omega = sine.atan2(cosine);
                let quadrature = (r * e * x - r * e * e / 2. * y) / sine;
                let (sin, cos) = (omega * steps).sin_cos();
                let gain = r.powf(steps) * f64::from(lane.gain);
                [
                    (quadrature * cos - y * sin) * gain,
                    (y * cos + quadrature * sin) * gain,
                    omega,
                    r.ln(),
                ]
            }
        };
        result.iter().all(|v| v.is_finite()).then_some(result)
    }
}
