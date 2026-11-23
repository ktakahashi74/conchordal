use super::*;

impl ControlForecast {
    /// Includes the target sample's control step, without stepping a DSP backend.
    /// Ticks where the amplitude target steps or a sequence gate closes.
    pub(crate) fn steps(self) -> [Option<u64>; 5] {
        let mut out = [None; 5];
        if let Some(updates) = self.amplitude_updates {
            for (slot, update) in out.iter_mut().zip(&updates.events[..updates.len]) {
                *slot = Some(update.at_sample.max(self.issued_at));
            }
        }
        if let AmplitudeModel::SeqGate {
            timer,
            duration_sec,
        } = self.model
        {
            let dt = f64::from(self.sample_dt);
            let (origin, timer) = match self.kick_at {
                Some(at) => (at.max(self.issued_at), 0.),
                None => (self.issued_at, f64::from(timer)),
            };
            // The renderer advances its timer before the comparison, so the step that
            // reaches the duration is already closed.
            let left = (f64::from(duration_sec) - timer) / dt;
            if left.is_finite() && left > 0. {
                out[4] = Some(origin.saturating_add((left.ceil() as u64).saturating_sub(1)));
            }
        }
        out
    }

    /// Span bound while `[from, to)` overlaps an amplitude-smoothing transient: the same
    /// `2 lambda h <= 0.1` rule as the envelope decay, over seven time constants after the
    /// issue or a target step.
    pub(crate) fn smoothing_span(self, [from, to]: [u64; 2]) -> Option<u64> {
        let alpha = f64::from(self.amplitude_smoothing?.alpha);
        if !(alpha > 0. && alpha < 1.) {
            return None;
        }
        let lambda = -(1. - alpha).ln();
        let settle = (7. / lambda).ceil() as u64;
        std::iter::once(Some(self.issued_at))
            .chain(self.steps().into_iter().take(4))
            .flatten()
            .any(|at| from < at.saturating_add(settle) && to > at)
            .then(|| (0.1 / (2. * lambda)) as u64)
    }

    /// The modulator's linear attack `[origin, end)`; its gain changes slope at `end`.
    pub(crate) fn attack_span(self) -> Option<[u64; 2]> {
        let AmplitudeModel::EntrainPulse {
            attack_step,
            state,
            env_level,
            ..
        } = self.model
        else {
            return None;
        };
        let step = f64::from(attack_step * self.sample_dt);
        if !step.is_finite() || step <= 0. {
            return None;
        }
        let (origin, level) = match self.kick_at {
            Some(at) => (at.max(self.issued_at), 0.),
            None if state == RenderModulatorStateKind::Attack => {
                (self.issued_at, f64::from(env_level))
            }
            None => return None,
        };
        Some([
            origin,
            origin.saturating_add(((1. - level) / step).ceil().max(1.) as u64),
        ])
    }
}
