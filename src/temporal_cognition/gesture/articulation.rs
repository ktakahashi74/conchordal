//! Actual articulation state and one immutable acoustic interval, without path weights.

use super::{
    Evidence, RawDescriptor, Run, State, TemporalGestureConfig, rate_features, rates, ridge,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition) struct Articulation {
    pub state: State,
    pub entered: u64,
    pub run: Run,
}

pub(in crate::temporal_cognition) struct Input {
    pub observed: bool,
    pub low: bool,
    pub dt: f64,
    start: u64,
    end: u64,
    rate: u32,
    evidence: Option<Evidence>,
    features: [f64; 11],
}

impl Input {
    pub fn new(
        group: ridge::Handle,
        raw: Option<&RawDescriptor>,
        motion: Option<f64>,
        [start, end]: [u64; 2],
        rate: u32,
        config: &TemporalGestureConfig,
    ) -> Result<Self, &'static str> {
        if rate == 0
            || end <= start
            || motion.is_some_and(|v| !v.is_finite())
            || raw.is_some_and(|r| {
                r.group != group
                    || r.start != start
                    || r.end != end
                    || r.source_start > r.start
                    || r.source_end < r.end
                    || r.source_end > r.available_end
                    || r.available_end > end
                    || r.known_samples > end - start
                    || r.values.iter().flatten().any(|v| !v.is_finite())
            })
        {
            return Err("invalid articulation input clock, owner or support");
        }
        let observed = raw.is_some_and(|r| r.known_samples > 0 && r.values[2].is_some());
        Ok(Self {
            observed,
            low: observed && raw.unwrap().values[2].unwrap().exp2() <= 0.01 * config.rms_reference,
            dt: (end - start) as f64 / f64::from(rate),
            start,
            end,
            rate,
            evidence: raw.filter(|_| observed).map(|r| Evidence {
                start: r.start,
                end: r.end,
                source_start: r.source_start,
                source_end: r.source_end,
                available: r.available_end,
            }),
            features: rate_features(
                [
                    raw.and_then(|r| r.values[3]),
                    raw.and_then(|r| r.values[4]),
                    raw.and_then(|r| r.values[5]),
                    motion,
                ],
                config,
            ),
        })
    }

    pub fn rates(
        &self,
        parent: Option<Articulation>,
        config: &TemporalGestureConfig,
    ) -> Result<[f64; 4], &'static str> {
        let Some(parent) = parent else {
            return Ok([0.; 4]);
        };
        let elapsed = self
            .start
            .checked_sub(parent.entered)
            .ok_or("future articulation parent")?;
        if parent.run.observed_end > self.start
            || [parent.run.attack, parent.run.release, parent.run.gap]
                .iter()
                .flatten()
                .any(|e| e.end > self.start || e.available > self.start)
        {
            return Err("future articulation parent evidence");
        }
        rates(
            parent.state,
            &self.features,
            (elapsed as f64 / f64::from(self.rate)).ln_1p(),
            config,
        )
    }

    pub fn child(&self, parent: Option<Articulation>, state: State) -> Articulation {
        let Some(parent) = parent else {
            return Articulation {
                state,
                entered: self.end,
                run: Run {
                    attack: (state == State::Attack).then_some(self.evidence).flatten(),
                    observed_end: if state != State::Gap && self.observed {
                        self.end
                    } else {
                        0
                    },
                    censored: state != State::Attack || !self.observed,
                    ..Run::default()
                },
            };
        };
        let changed = state != parent.state;
        let mut run = parent.run;
        if changed && parent.state == State::Gap && state != State::Gap {
            run = Run {
                censored: true,
                ..Run::default()
            };
        }
        if changed && state == State::Attack {
            run = Run {
                attack: self.evidence,
                censored: !self.observed,
                ..Run::default()
            };
        }
        if !self.observed {
            run.censored = true;
        }
        if changed && state == State::Release {
            run.release = self.evidence;
        }
        if changed && state == State::Gap && run.release.is_some() {
            run.gap = self.evidence;
        }
        if state != State::Gap && self.observed {
            run.observed_end = self.end;
        }
        Articulation {
            state,
            entered: if changed { self.end } else { parent.entered },
            run,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_and_parent_reject_foreign_acausal_and_invalid_supported_values() {
        let group = ridge::Handle {
            bus: 0,
            epoch: 1,
            generation: 2,
        };
        let config = TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        };
        let raw = RawDescriptor {
            group,
            start: 100,
            end: 200,
            known_samples: 100,
            source_start: 0,
            source_end: 200,
            available_end: 200,
            values: [Some(0.); 10],
        };
        for case in 0..8 {
            let mut invalid = raw;
            match case {
                0 => invalid.group.bus = 1,
                1 => invalid.group.epoch = 2,
                2 => invalid.start = 101,
                3 => invalid.source_start = 101,
                4 => invalid.source_end = 199,
                5 => invalid.available_end = 201,
                6 => invalid.known_samples = 101,
                _ => invalid.values[3] = Some(f64::NAN),
            }
            assert!(Input::new(group, Some(&invalid), None, [100, 200], 1000, &config).is_err());
        }
        assert!(Input::new(group, Some(&raw), Some(f64::NAN), [100, 200], 1000, &config).is_err());
        assert!(Input::new(group, Some(&raw), None, [100, 200], 0, &config).is_err());
        let input = Input::new(group, Some(&raw), None, [100, 200], 1000, &config).unwrap();
        let mut parent = Articulation {
            state: State::Attack,
            entered: 101,
            run: Run::default(),
        };
        assert!(input.rates(Some(parent), &config).is_err());
        parent.entered = 0;
        parent.run.observed_end = 101;
        assert!(input.rates(Some(parent), &config).is_err());
        parent.run.observed_end = 100;
        parent.run.attack = Some(Evidence {
            start: 0,
            end: 100,
            source_start: 0,
            source_end: 100,
            available: 101,
        });
        assert!(input.rates(Some(parent), &config).is_err());
        parent.run.attack.as_mut().unwrap().available = 100;
        assert!(input.rates(Some(parent), &config).is_ok());
    }
}
