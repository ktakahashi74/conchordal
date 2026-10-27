//! Known authored interval, not evidence of a perceived phrase or closure.

use super::control::{PhonationGate, PitchApplyMode, PitchMode, VoiceControl};
use super::{Action, DurationSpec, PhonationTiming, RhythmRelation, Scenario};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Expression {
    pub start: f64,
    pub end: f64,
    pub arch: f64,
    pub ritard: f64,
    pub position: (usize, usize),
}

impl Expression {
    // Friberg & Sundberg (1999), equations 2–4, q=3. Rationalization
    // retains the neutral limit without subtracting nearly equal numbers.
    pub(crate) fn integral(self, x: f64) -> f64 {
        let k = (1.0 - 0.7 * self.ritard).powi(3) - 1.0;
        let c = (1.0 + k * x.clamp(0.0, 1.0)).cbrt();
        1.5 * x * (c + 1.0) / (c * c + c + 1.0)
    }

    pub(crate) fn performance(self, score: f64) -> f64 {
        if self.ritard == 0.0 || score <= self.start {
            return score;
        }
        let length = self.end - self.start;
        if score >= self.end {
            return score + length * (self.integral(1.0) - 1.0);
        }
        self.start + length * self.integral((score - self.start) / length)
    }

    pub(crate) fn score(self, time: f64) -> f64 {
        if self.ritard == 0.0 || time <= self.start {
            return time;
        }
        let length = self.end - self.start;
        let finish = self.performance(self.end);
        if time >= finish {
            return self.end + time - finish;
        }
        let tau = (time - self.start) / length;
        let k = (1.0 - 0.7 * self.ritard).powi(3) - 1.0;
        let z = (1.0 + 2.0 * k * tau / 3.0).max(0.0).sqrt();
        self.start + length * (2.0 * tau * (z * z + z + 1.0) / (3.0 * (z + 1.0)))
    }

    pub(crate) fn gain(self, time: f64) -> f32 {
        if self.arch == 0.0 || time <= self.start || time >= self.performance(self.end) {
            return 1.0;
        }
        let x = ((self.score(time) - self.start) / (self.end - self.start)).clamp(0.0, 1.0);
        // Todd (1992), single layer: m=1, E=0, U=-.5, K=100, delta=.5.
        let intensity = 2.0 * x.min(1.0 - x);
        // The unit floor and weight are author rules, not Todd's intensity.
        (1.0 + self.arch * intensity).sqrt() as f32
    }

    fn supports(control: &VoiceControl) -> bool {
        if control.phonation.rest_periods.is_some()
            || control.phonation.spec.is_sustained_participation()
        {
            return false;
        }
        match control.phonation.spec.timing {
            PhonationTiming::Once => {
                matches!(control.phonation.spec.duration, DurationSpec::WhileAlive)
                    && control.phonation.gate == PhonationGate::Immediate
                    && !(control.pitch.mode == PitchMode::Free
                        && control.pitch.pitch_apply_mode == PitchApplyMode::Rearticulate)
            }
            PhonationTiming::Coupled(timing) => {
                timing.relation == RhythmRelation::Synchronized
                    && matches!(
                        control.phonation.spec.duration,
                        DurationSpec::Gates(_) | DurationSpec::Field(_)
                    )
            }
            _ => false,
        }
    }

    pub(crate) fn validate(self, scenario: &Scenario) -> Result<(), &'static str> {
        if !self.start.is_finite()
            || !self.end.is_finite()
            || self.start < 0.0
            || self.end <= self.start
        {
            return Err("requires a finite positive interval");
        }
        let mut events: Vec<_> = scenario.events.iter().collect();
        events.sort_by(|a, b| a.time.total_cmp(&b.time).then(a.order.cmp(&b.order)));
        let mut populations = BTreeMap::new();
        for event in events {
            for action in &event.actions {
                if f64::from(event.time) > self.end && !matches!(action, Action::Finish) {
                    return Err(
                        "requires the final section, with all populations released by its end",
                    );
                }
                match action {
                    Action::PlaceFlora { .. } => {
                        return Err("Flora is outside the initial supported scene");
                    }
                    Action::Spawn {
                        population_id,
                        spec,
                        ..
                    } => {
                        if !Self::supports(&spec.control) {
                            return Err(
                                "requires immediate Once without free rearticulation or non-sustained synchronized metric voices",
                            );
                        }
                        populations.insert(*population_id, (spec.control.clone(), false));
                    }
                    Action::UpdatePopulation {
                        population_id,
                        patch,
                    } => {
                        if let Some((control, _)) = populations.get_mut(population_id) {
                            if f64::from(event.time) >= self.start
                                && patch.pitch_apply_mode.is_some()
                            {
                                return Err("pitch application cannot change inside the interval");
                            }
                            control.apply_update(patch);
                            if !Self::supports(control) {
                                return Err("a population patch produces an unsupported voice");
                            }
                        }
                    }
                    Action::ReleasePopulation { population_id, .. } => {
                        if let Some((_, released)) = populations.get_mut(population_id) {
                            *released = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        if populations.values().any(|(_, released)| !released) {
            return Err("every population needs an explicit or scoped release");
        }
        Ok(())
    }
}

impl Scenario {
    /// Called once with the actual device clock, before any runtime is started.
    pub(crate) fn prepare_expression(&mut self, fs: f32, hop: usize) {
        let Some(expression) = self.expression else {
            return;
        };
        let mut reason = expression.validate(self).err();
        let delivery = |time: f32| {
            let mut n = (f64::from(time) * f64::from(fs) / hop as f64).ceil() as u64;
            while n > 0 && ((n - 1) * hop as u64) as f32 / fs >= time {
                n -= 1;
            }
            while (n * hop as u64) as f32 / fs < time {
                n += 1;
            }
            n
        };
        for event in &self.events {
            for action in &event.actions {
                if let Action::Spawn {
                    population_id,
                    spec,
                    ..
                } = action
                    && matches!(spec.control.phonation.spec.timing, PhonationTiming::Once)
                {
                    for release in &self.events {
                        if release.actions.iter().any(|a| matches!(a, Action::ReleasePopulation {population_id: id, ..} if id == population_id))
                            && delivery(expression.performance(f64::from(event.time)) as f32) >= delivery(expression.performance(f64::from(release.time)) as f32)
                        { reason = Some("Once onset and release must occupy distinct delivery hops"); }
                    }
                }
            }
        }
        let mut finish = expression.performance(f64::from(self.duration_sec));
        // Finish already contains physical release tails. Recover the score
        // cursor saved by the compiler instead of stretching those tails.
        if let Some(cursor) = self.expression_cursor {
            finish = expression.performance(f64::from(cursor));
        }
        for event in &self.events {
            for action in &event.actions {
                if let Action::ReleasePopulation { fade_sec, .. } = action {
                    finish = finish
                        .max(expression.performance(f64::from(event.time)) + f64::from(*fade_sec));
                }
            }
        }
        if !finish.is_finite() || finish > 600.0 {
            reason = Some("transformed scene exceeds the initial 600 physical second support");
        }
        if let Some(reason) = reason {
            tracing::warn!(
                line = expression.position.0,
                column = expression.position.1,
                "expressive_section ignored: {reason}"
            );
            self.expression = None;
            return;
        }
        for event in &mut self.events {
            event.time = if event.actions.iter().any(|a| matches!(a, Action::Finish)) {
                finish as f32
            } else {
                expression.performance(f64::from(event.time)) as f32
            };
        }
        for marker in &mut self.scene_markers {
            marker.time = expression.performance(f64::from(marker.time)) as f32;
        }
        self.duration_sec = finish as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_matches_independent_decimal_reference_and_inverse() {
        let rows: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/research/pe1_reference.json")).unwrap();
        for row in rows.as_array().unwrap() {
            let number = |key: &str| row[key].as_str().unwrap().parse::<f64>().unwrap();
            let e = Expression {
                start: 17.0,
                end: 317.0,
                arch: number("weight").min(1.0),
                ritard: number("weight").min(1.0),
                position: (0, 0),
            };
            let x = number("x");
            let time = e.performance(e.start + 300.0 * x);
            assert!(
                (time - number("mapped_seconds")).abs() * 48000.0 < 0.25,
                "{row}"
            );
            assert!((e.integral(x) - number("F")).abs() * 300.0 * 48000.0 < 0.25);
            assert!((e.score(time) - (e.start + 300.0 * number("inverse"))).abs() * 48000.0 < 0.25);
            assert!(
                e.gain(time)
                    .to_bits()
                    .abs_diff((number("gain") as f32).to_bits())
                    <= 1
            );
            assert!(time.is_finite());
            let reference_sample = number("mapped_seconds") * 48000.0;
            if (reference_sample.fract() - 0.5).abs() >= 0.25 {
                assert_eq!(
                    (time * 48000.0).round() as u64,
                    row["mapped_sample"].as_u64().unwrap()
                );
            }
        }
        let e = Expression {
            start: 0.0,
            end: 1.0,
            arch: 1.0,
            ritard: 1.0,
            position: (0, 0),
        };
        let parts: f64 = (0..4)
            .map(|i| e.integral((i + 1) as f64 / 4.) - e.integral(i as f64 / 4.))
            .sum();
        assert!((parts - 2.73 / 1.946).abs() < 1e-14);
        for w in [f64::from_bits(1), f64::EPSILON] {
            let tiny = Expression { ritard: w, ..e };
            assert!(tiny.score(tiny.performance(0.7)).is_finite());
            assert!((tiny.score(tiny.performance(0.7)) - 0.7).abs() < f64::EPSILON);
        }
    }
}
