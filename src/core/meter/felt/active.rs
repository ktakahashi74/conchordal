use super::*;

#[derive(Debug, Default)]
pub(super) struct FeltMeter {
    pub(super) felt_beat: Option<Box<crate::core::felt_beat::FeltBeat>>,
    pub(super) felt_period_guard: bool,
    pub(super) felt_input: Option<Box<crate::core::felt_input::FeltInput>>,
}

impl FeltMeter {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_onset(
        &mut self,
        dt: f32,
        frequency: f32,
        onset: Onset,
        seeded_now: bool,
        stability: f32,
        hopf: f32,
        drive_known: bool,
    ) {
        if let Some(felt) = self.felt_beat.as_mut() {
            if !drive_known {
                felt.invalidate();
            }
            if let Some(input) = self.felt_input.as_mut() {
                let frame = crate::core::felt_input::Frame {
                    dt,
                    frequency,
                    fired: onset.fired,
                    frac: onset.frac,
                    seeded: seeded_now,
                    stability,
                    hopf,
                    ..Default::default()
                };
                let released = if drive_known {
                    input.push(frame)
                } else {
                    input.reset();
                    None
                };
                if let Some(frame) = released {
                    felt.process(
                        frame.dt,
                        frame.frequency,
                        Onset {
                            fired: frame.fired,
                            frac: frame.frac,
                        },
                        frame.seeded,
                        frame.stability,
                        frame.hopf,
                    );
                } else if input.missing {
                    felt.invalidate();
                }
            } else {
                felt.process(dt, frequency, onset, seeded_now, stability, hopf);
            }
        }
    }

    pub(super) fn blocks_period_update(&self, period: f32, beat_omega: f32) -> bool {
        // The felt half-peak explains odd half-cycle multiples when the author guard is active.
        self.felt_period_guard
            && self.felt_beat.as_ref().is_some_and(|felt| {
                let state = felt.state();
                !state.unknown && state.p_beat >= 0.5
            })
            && {
                let current_period = TAU / beat_omega;
                (1..=8).step_by(2).any(|k| {
                    let expected = 0.5 * k as f32 * current_period;
                    (period - expected).abs() <= ioi_window(expected)
                })
            }
    }
}

impl MeterNetwork {
    pub(crate) fn enable_felt_beat(&mut self, params: crate::config::FeltBeatParams) {
        self.felt.felt_beat = Some(Box::new(crate::core::felt_beat::FeltBeat::new(params)));
    }

    pub(crate) fn enable_felt_input(&mut self, fs: f32, hop: usize) {
        if self.felt.felt_beat.is_some() {
            self.felt.felt_input = Some(Box::new(crate::core::felt_input::FeltInput::new(fs, hop)));
        }
    }

    pub(crate) fn needs_felt_bands(&self) -> bool {
        self.felt.felt_input.is_some()
    }

    pub(crate) fn set_felt_bands(&mut self, end: u64, bands: Option<[f32; 3]>) {
        if let Some(input) = self.felt.felt_input.as_mut() {
            input.set_bands(end, bands);
        }
    }

    pub(crate) fn enable_felt_period_guard(&mut self) {
        self.felt.felt_period_guard = true;
    }

    pub(crate) fn felt_beat(&self) -> Option<crate::core::felt_beat::FeltBeatState> {
        self.felt.felt_beat.as_ref().map(|f| {
            let mut state = f.state();
            if let Some(input) = self.felt.felt_input.as_ref() {
                input.project(&mut state, self.beat_phi);
            }
            state.meter_frequency_hz = self.last.beat.freq_hz;
            state.meter_confidence = self.last.beat.confidence;
            state
        })
    }
}
