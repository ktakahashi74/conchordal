//! Sustained-level confirmation of acoustic events for felt-beat inference only.
use crate::core::felt_beat::FeltBeatState;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Decision {
    pub candidate_end: u64,
    pub frac: f32,
    pub complete: bool,
    pub accepted: bool,
    pub increment_db: Option<f64>,
    pub from_silence: bool,
    pub before: [f64; 3],
    pub after: [f64; 3],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct InputState {
    pub rule: &'static str,
    pub window_sec: f32,
    pub window_hops: usize,
    pub delay_samples: u64,
    pub posterior_end: Option<u64>,
    pub decision: Option<Decision>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Frame {
    pub end: u64,
    pub dt: f32,
    pub frequency: f32,
    pub fired: bool,
    pub frac: f32,
    pub seeded: bool,
    pub stability: f32,
    pub hopf: f32,
}

#[derive(Debug)]
pub(crate) struct FeltInput {
    window: usize,
    hop: u64,
    energy: Vec<[f64; 3]>,
    frames: Vec<Frame>,
    cursor: usize,
    count: usize,
    before: [f64; 3],
    after: [f64; 3],
    ahead_cycles: f64,
    pending: Option<(u64, [f32; 3])>,
    current_end: u64,
    posterior_end: Option<u64>,
    decision: Option<Decision>,
    pub(crate) missing: bool,
}

impl FeltInput {
    pub(crate) fn new(fs: f32, hop: usize) -> Self {
        // Round upward: the detector's refractory interval ends only after
        // whole-hop advances have covered REFRACTORY_SEC. Never shorten it.
        let window = (f64::from(crate::core::onset::REFRACTORY_SEC) / (hop as f64 / f64::from(fs)))
            .ceil() as usize;
        let window = window.max(1);
        Self {
            window,
            hop: hop as u64,
            energy: vec![[0.0; 3]; 2 * window],
            frames: vec![Frame::default(); window],
            cursor: 0,
            count: 0,
            before: [0.0; 3],
            after: [0.0; 3],
            ahead_cycles: 0.0,
            pending: None,
            current_end: 0,
            posterior_end: None,
            decision: None,
            missing: true,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.energy.fill([0.0; 3]);
        self.frames.fill(Frame::default());
        self.cursor = 0;
        self.count = 0;
        self.before = [0.0; 3];
        self.after = [0.0; 3];
        self.ahead_cycles = 0.0;
        self.pending = None;
        self.current_end = 0;
        self.posterior_end = None;
        self.decision = None;
        self.missing = true;
    }

    pub(crate) fn set_bands(&mut self, end: u64, bands: Option<[f32; 3]>) {
        self.pending = bands.map(|b| (end, b));
    }

    pub(crate) fn push(&mut self, mut frame: Frame) -> Option<Frame> {
        let Some((end, bands)) = self
            .pending
            .take()
            .filter(|(_, b)| b.iter().all(|v| v.is_finite() && *v >= 0.0))
        else {
            self.reset();
            return None;
        };
        if self.count > 0 && end != self.current_end + self.hop {
            // A clock gap must not republish the previous coordinate's posterior.
            self.reset();
            return None;
        }
        self.missing = false;
        self.current_end = end;
        frame.end = end;
        self.decision = None;
        let n = self.window;
        let mid = (self.cursor + n) % (2 * n);
        for (b, &value) in bands.iter().enumerate() {
            // f64 rolling sums retain small band powers after long decays;
            // no extra energy floor is introduced.
            self.before[b] += self.energy[mid][b] - self.energy[self.cursor][b];
            self.after[b] += f64::from(value) - self.energy[mid][b];
        }
        self.energy[self.cursor] = bands.map(f64::from);
        self.cursor = (self.cursor + 1) % (2 * n);
        self.count = (self.count + 1).min(2 * n);
        let slot = (self.cursor + n - 1) % n;
        self.frames[slot] = frame;
        self.ahead_cycles += f64::from(frame.frequency) * f64::from(frame.dt);
        if self.count < n {
            return None;
        }
        let mut released = self.frames[self.cursor % n];
        self.ahead_cycles -= f64::from(released.frequency) * f64::from(released.dt);
        self.posterior_end = Some(released.end);
        if released.fired {
            let complete = self.count >= 2 * n;
            let before = self.before.map(|v| v.max(0.0) / n as f64);
            let after = self.after.map(|v| v.max(0.0) / n as f64);
            let base: f64 = before.iter().sum();
            let rise: f64 = after
                .iter()
                .zip(before)
                .map(|(b, a)| (b - a).max(0.0))
                .sum();
            // Zero-to-positive power is an unbounded dB rise, not an epsilon.
            let from_silence = base == 0.0 && rise > 0.0;
            let increment_db = if base > 0.0 {
                Some(10.0 * (1.0 + rise / base).log10())
            } else if from_silence {
                None
            } else {
                Some(0.0)
            };
            let accepted = complete && (from_silence || increment_db.is_some_and(|db| db >= 1.0));
            self.decision = Some(Decision {
                candidate_end: released.end,
                frac: released.frac,
                complete,
                accepted,
                increment_db,
                from_silence,
                before,
                after,
            });
            released.fired = accepted;
        }
        Some(released)
    }

    pub(crate) fn project(&self, state: &mut FeltBeatState, current_hopf: f32) {
        state.phase = crate::core::phase::wrap_pm_pi(
            state.phase + (std::f64::consts::TAU * self.ahead_cycles) as f32,
        );
        state.hopf_phase_difference = crate::core::phase::wrap_pm_pi(state.phase - current_hopf);
        state.input = Some(InputState {
            rule: "Author rule: read only onsets with a sustained level rise of at least 1 dB across 60 ms.",
            window_sec: crate::core::onset::REFRACTORY_SEC,
            window_hops: self.window,
            delay_samples: self
                .current_end
                .saturating_sub(self.posterior_end.unwrap_or(self.current_end)),
            posterior_end: self.posterior_end,
            decision: self.decision,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::onset::OnsetDetector;
    use crate::core::stream::dorsal::DorsalStream;

    #[test]
    fn refractory_seconds_choose_whole_hops_without_shortening() {
        for (fs, hop, expected) in [
            (48000., 512, 6),
            (48000., 480, 6),
            (44100., 512, 6),
            (32000., 256, 8),
            (48000., 1024, 3),
            (48000., 256, 12),
        ] {
            let gate = FeltInput::new(fs, hop);
            assert_eq!(gate.window, expected);
            let dt = hop as f64 / f64::from(fs);
            assert!(gate.window as f64 * dt >= f64::from(crate::core::onset::REFRACTORY_SEC));
            assert!((gate.window - 1) as f64 * dt < f64::from(crate::core::onset::REFRACTORY_SEC));
        }
    }

    #[test]
    fn missing_bands_and_clock_gaps_discard_the_old_queue() {
        let mut input = FeltInput::new(48000.0, 512);
        for i in 1..=12 {
            input.set_bands(i * 512, Some([1.0; 3]));
            input.push(Frame {
                dt: 512.0 / 48000.0,
                frequency: 2.0,
                ..Default::default()
            });
        }
        assert!(input.posterior_end.is_some());
        input.set_bands(13 * 512, None);
        assert!(input.push(Frame::default()).is_none());
        assert!(input.missing && input.posterior_end.is_none());
        for i in 14..=19 {
            input.set_bands(i * 512, Some([1.0; 3]));
            input.push(Frame {
                dt: 512.0 / 48000.0,
                frequency: 2.0,
                ..Default::default()
            });
        }
        assert_eq!(input.posterior_end, Some(14 * 512));
        input.set_bands(21 * 512, Some([1.0; 3]));
        assert!(input.push(Frame::default()).is_none());
        assert!(input.missing && input.posterior_end.is_none());
    }

    #[test]
    fn steady_flora_partials_make_raw_candidates_but_no_confirmed_events() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/research/felt_input_steady.json"))
                .unwrap();
        let (fs, hop) = (48000.0_f32, 512_usize);
        let dt = hop as f32 / fs;
        for case in fixtures["cases"].as_array().unwrap() {
            let tones: Vec<(f64, f64)> = case["tones"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| {
                    (
                        t[0].as_f64().unwrap(),
                        (2.0 * t[1].as_f64().unwrap()).sqrt(),
                    )
                })
                .collect();
            let (mut all_raw, mut all_accepted) = (0, 0);
            for phases in case["phases"].as_array().unwrap() {
                let phases: Vec<f64> = phases
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let mut dorsal = DorsalStream::new(fs);
                let mut detector = OnsetDetector::default();
                let mut input = FeltInput::new(fs, hop);
                let mut pcm = vec![0.0; hop];
                let (mut raw, mut accepted) = (0, 0);
                for i in 0..(10.0 * f64::from(fs) / hop as f64) as usize {
                    for (j, x) in pcm.iter_mut().enumerate() {
                        let t = (i * hop + j) as f64 / f64::from(fs);
                        *x = tones
                            .iter()
                            .zip(&phases)
                            .map(|(&(f, a), &p)| a * (std::f64::consts::TAU * f * t + p).sin())
                            .sum::<f64>() as f32;
                    }
                    dorsal.process(&pcm);
                    let m = dorsal.last_metrics();
                    let onset =
                        detector.process(dt, (500.0 * m.flux.max(0.0)).tanh().clamp(0.0, 1.0));
                    let end = ((i + 1) * hop) as u64;
                    if onset.fired && i * hop >= 2 * fs as usize {
                        raw += 1;
                    }
                    input.set_bands(end, Some([m.e_low, m.e_mid, m.e_high]));
                    input.push(Frame {
                        dt,
                        frequency: 2.0,
                        fired: onset.fired,
                        frac: onset.frac,
                        ..Default::default()
                    });
                    if let Some(d) = input.decision
                        && d.accepted
                        && d.candidate_end - hop as u64 >= 2 * fs as u64
                    {
                        accepted += 1;
                    }
                }
                assert!(
                    raw > 0,
                    "{} must expose the existing detector",
                    case["name"]
                );
                assert_eq!(accepted, 0, "{}", case["name"]);
                all_raw += raw;
                all_accepted += accepted;
            }
            println!(
                "felt input steady {}: phases=20 raw={all_raw} accepted={all_accepted}",
                case["name"]
            );
        }
    }

    #[test]
    fn isolated_clicks_keep_original_events_and_exact_delayed_posterior() {
        use crate::core::meter::MeterNetwork;
        let (fs, hop) = (48000.0_f32, 512_usize);
        let mut dorsal = DorsalStream::new(fs);
        let mut old = MeterNetwork::new();
        old.enable_felt_beat(Default::default());
        let mut new = MeterNetwork::new();
        new.enable_felt_beat(Default::default());
        new.enable_felt_input(fs, hop);
        let delay = FeltInput::new(fs, hop).window - 1;
        assert_eq!(delay, 5);
        let mut history = Vec::new();
        let mut raw = Vec::new();
        let mut passed = Vec::new();
        let mut pcm = vec![0.0; hop];
        for i in 0..(12.0 * f64::from(fs) / hop as f64) as usize {
            for (j, x) in pcm.iter_mut().enumerate() {
                let t = (i * hop + j) as f64 / f64::from(fs);
                let p = (t - 1.0).rem_euclid(0.5);
                *x = if t >= 1.0 && p < 0.04 {
                    (0.8 * (-p / 0.01).exp() * (std::f64::consts::TAU * 3000.0 * t).sin()) as f32
                } else {
                    0.0
                };
            }
            dorsal.process(&pcm);
            let m = dorsal.last_metrics();
            let drive = (500.0 * m.flux.max(0.0)).tanh().clamp(0.0, 1.0);
            let end = ((i + 1) * hop) as u64;
            old.process_at((i * hop) as u64, hop, fs, Some(drive));
            let a = old.felt_beat().unwrap();
            if let Some(frac) = a.onset_frac {
                raw.push((end, frac.to_bits()));
            }
            history.push(a);
            new.set_felt_bands(end, Some([m.e_low, m.e_mid, m.e_high]));
            new.process_at((i * hop) as u64, hop, fs, Some(drive));
            let b = new.felt_beat().unwrap();
            if let Some(d) = b.input.unwrap().decision {
                assert!(d.accepted);
                passed.push((d.candidate_end, d.frac.to_bits()));
            }
            if i >= delay {
                let past = history[i - delay];
                assert_eq!(
                    b.p_beat.to_bits(),
                    past.p_beat.to_bits(),
                    "P at source hop {}",
                    i - delay
                );
                assert_eq!(
                    b.c.to_bits(),
                    past.c.to_bits(),
                    "c at source hop {}",
                    i - delay
                );
                assert_eq!(
                    b.m_plus.to_bits(),
                    past.m_plus.to_bits(),
                    "m_plus at source hop {}",
                    i - delay
                );
                assert_eq!(
                    b.input.unwrap().posterior_end,
                    Some(((i - delay + 1) * hop) as u64)
                );
                assert_eq!(b.input.unwrap().delay_samples, (delay * hop) as u64);
            }
        }
        assert!(!raw.is_empty());
        assert_eq!(raw, passed);
        println!(
            "felt input clicks: raw={} passed={} exact P/c/m_plus delay={delay} hops",
            raw.len(),
            passed.len()
        );
    }

    #[test]
    #[ignore = "requires FB-11 saved-input JSON and output paths in FB11_REPLAY_INPUT/OUTPUT"]
    fn saved_input_matches_fb10_sustained_reference() {
        let input = std::fs::File::open(std::env::var("FB11_REPLAY_INPUT").unwrap()).unwrap();
        let cases: serde_json::Value = serde_json::from_reader(input).unwrap();
        let mut out = Vec::new();
        for case in cases["cases"].as_array().unwrap() {
            let mut gate = FeltInput::new(48000.0, 512);
            let mut decisions = Vec::new();
            let mut kept = 0;
            for (i, row) in case["rows"].as_array().unwrap().iter().enumerate() {
                let bands = std::array::from_fn(|b| row[b].as_f64().unwrap() as f32);
                gate.set_bands(((i + 1) * 512) as u64, Some(bands));
                gate.push(Frame {
                    dt: 512.0 / 48000.0,
                    frequency: 2.0,
                    fired: !row[3].is_null(),
                    frac: row[3].as_f64().unwrap_or(0.0) as f32,
                    ..Default::default()
                });
                if let Some(d) = gate.decision {
                    let t = (d.candidate_end - 512) as f64 / 48000.0
                        + f64::from(d.frac) * 512.0 / 48000.0;
                    if (12.0..100.0).contains(&t) && d.accepted {
                        kept += 1;
                    }
                    decisions.push(d);
                }
            }
            println!("felt input saved {}: body_kept={kept}", case["name"]);
            assert_eq!(kept, case["expected_kept_body"].as_u64().unwrap());
            out.push(
                serde_json::json!({"name":case["name"],"body_kept":kept,"decisions":decisions}),
            );
        }
        std::fs::write(
            std::env::var("FB11_REPLAY_OUTPUT").unwrap(),
            serde_json::to_string_pretty(&out).unwrap(),
        )
        .unwrap();
    }
}
