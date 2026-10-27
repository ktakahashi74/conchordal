//! LBDM candidate equations and an explicitly causal, author-rule adapter.
use super::own_sequence::{ExecutionKind, ExecutionRecord};

/// Renderer reservation, carried by one ToneSpec; never an SQ-1 learning event.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PhraseTone {
    pub(crate) reservation: Option<Reservation>,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Reservation {
    pub voice: u64,
    pub generation: u32,
    pub tone: u64,
    pub confirming_seq: u64,
    pub confirming_tone: u64,
    pub known: u64,
    pub expires: u64,
    pub gain: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Receipt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reservation: Option<Reservation>,
    pub reaction_applied: bool,
    pub interval_supported: bool,
    pub record: ExecutionRecord,
    pub origin: &'static str,
    pub gain: f32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
struct Note {
    id: u64,
    seq: u64,
    at: u64,
    hz: f64,
    off: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Boundary {
    pub voice: u64,
    pub generation: u32,
    window: [u64; 6],
    left: Note,
    right: Note,
    pub known: u64,
    pub confirming_seq: u64,
    pub confirming_tone: u64,
    pub published_sequence: u64,
    pub latest_on_id: Option<u64>,
    pub support: &'static str,
    pub expires: u64,
    pub score: f64,
    caps: [f64; 3],
    model: &'static str,
    publication: &'static str,
}

/// Finite-column completion: truncate, interior strengths, component maxima.
/// Endpoints have no profile, and hence cannot be used as peak neighbors.
fn profile(input: &[[f64; 3]], caps: [f64; 3], output: &mut [f64]) -> bool {
    assert_eq!(output.len(), input.len().saturating_sub(2));
    output.fill(0.0);
    if !caps.iter().all(|x| x.is_finite() && *x > 0.)
        || !input.iter().flatten().all(|x| x.is_finite() && *x >= 0.)
    {
        return false;
    }
    for k in 0..3 {
        let scale = input
            .iter()
            .skip(1)
            .take(output.len())
            .map(|row| row[k].min(caps[k]))
            .fold(0.0_f64, f64::max);
        if scale == 0. {
            continue;
        }
        let strength = |i: usize| {
            let [a, b, c] = [input[i][k], input[i + 1][k], input[i + 2][k]].map(|x| x.min(caps[k]));
            let change = |x: f64, y: f64| {
                // Scaling avoids overflow while preserving the source ratio.
                let m = x.max(y);
                if m == 0. {
                    0.
                } else {
                    ((x / m) - (y / m)).abs() / ((x / m) + (y / m))
                }
            };
            // A common interior scale cancels in maximum normalization.
            (b / scale) * (change(a, b) + change(b, c))
        };
        let max = (0..output.len()).map(&strength).fold(0.0_f64, f64::max);
        if max > 0. {
            for (i, value) in output.iter_mut().enumerate() {
                *value += [0.25, 0.5, 0.25][k] * (strength(i) / max);
            }
        }
    }
    true
}

pub(crate) struct PhraseBoundary {
    voice: u64,
    generation: u32,
    fs: f64,
    weight: f64,
    caps: [f64; 3],
    notes: [Note; 6],
    len: usize,
    last_seq: u64,
    last_on: Option<u64>,
    pending: Option<Boundary>,
    pub selected: Option<Reservation>,
    pub trace: Vec<Boundary>,
    pub trace_enabled: bool,
}

impl PhraseBoundary {
    pub fn new(voice: u64, generation: u32, fs: f32, weight: f64, caps: [f64; 3]) -> Self {
        Self {
            voice,
            generation,
            fs: fs as f64,
            weight,
            caps,
            notes: [Note::default(); 6],
            len: 0,
            last_seq: 0,
            last_on: None,
            pending: None,
            selected: None,
            trace: Vec::new(),
            trace_enabled: false,
        }
    }

    pub fn lose_support(&mut self) {
        self.len = 0;
        self.pending = None;
        self.selected = None;
    }

    pub fn observe(&mut self, receipt: &Receipt, now: u64) {
        let r = &receipt.record;
        if r.voice_id != self.voice || r.generation != self.generation || r.available_at > now {
            return;
        }
        if r.execution_sequence <= self.last_seq {
            return;
        }
        if r.execution_sequence != self.last_seq + 1 {
            self.lose_support();
        }
        self.last_seq = r.execution_sequence;
        match r.kind {
            ExecutionKind::Missing | ExecutionKind::PitchUpdate => self.lose_support(),
            ExecutionKind::Off => {
                if self.len > 0 && self.notes[self.len - 1].id == r.tone_id {
                    self.notes[self.len - 1].off.get_or_insert(r.sample);
                }
            }
            ExecutionKind::On => {
                self.last_on = Some(r.tone_id);
                self.pending = None;
                let Some(hz) = r
                    .frequency_hz
                    .filter(|x| receipt.interval_supported && x.is_finite() && *x > 0.)
                else {
                    self.lose_support();
                    return;
                };
                if self.len > 0 {
                    let previous = self.notes[self.len - 1];
                    if previous.at >= r.sample || previous.off.is_none_or(|off| off > r.sample) {
                        self.lose_support();
                    }
                }
                if self.len == 6 {
                    self.notes.rotate_left(1);
                    self.len = 5;
                }
                self.notes[self.len] = Note {
                    id: r.tone_id,
                    seq: r.execution_sequence,
                    at: r.sample,
                    hz: hz as f64,
                    off: None,
                };
                self.len += 1;
                if self.len != 6 {
                    return;
                }
                let mut intervals = [[0.; 3]; 5];
                for (i, x) in intervals.iter_mut().enumerate() {
                    let a = self.notes[i];
                    let b = self.notes[i + 1];
                    *x = [
                        12. * (b.hz / a.hz).log2().abs(),
                        (b.at - a.at) as f64 / self.fs,
                        (b.at - a.off.expect("supported interval")) as f64 / self.fs,
                    ];
                }
                let mut p = [0.; 3];
                if !profile(&intervals, self.caps, &mut p)
                    || !(p[1] > 0. && p[1] > p[0] && p[1] > p[2])
                {
                    return;
                }
                let Some(expires) = now.checked_add(self.notes[5].at - self.notes[4].at) else {
                    return;
                };
                let boundary = Boundary {
                    voice: self.voice,
                    generation: self.generation,
                    window: self.notes.map(|n| n.id),
                    left: self.notes[2],
                    right: self.notes[3],
                    known: now,
                    confirming_seq: r.execution_sequence,
                    confirming_tone: r.tone_id,
                    published_sequence: r.execution_sequence,
                    latest_on_id: Some(r.tone_id),
                    support: "supported",
                    expires,
                    score: p[1],
                    caps: self.caps,
                    model: "LBDM-2001/local-6-v1/renderer-executed",
                    publication: "final-local",
                };
                if self.trace_enabled {
                    self.trace.push(boundary.clone());
                }
                self.pending = Some(boundary);
            }
        }
    }

    pub fn finish_publication(&mut self, now: u64) {
        for b in self
            .trace
            .iter_mut()
            .chain(self.pending.iter_mut())
            .filter(|b| b.known == now)
        {
            b.published_sequence = self.last_seq;
            b.latest_on_id = self.last_on;
        }
    }

    /// Called at the first existing opportunity, including a vetoed opportunity.
    pub fn opportunity(&mut self, at: u64) -> Option<Reservation> {
        let b = self.pending.take()?;
        if at < b.known || at >= b.expires {
            return None;
        }
        // B<=1, so even the largest finite weight cannot overflow w*B.
        let gain = ((1. / self.weight.max(1.))
            / ((1. / self.weight.max(1.)) + (self.weight / self.weight.max(1.)) * b.score))
            as f32;
        Some(Reservation {
            voice: self.voice,
            generation: self.generation,
            tone: 0,
            confirming_seq: b.confirming_seq,
            confirming_tone: b.confirming_tone,
            known: b.known,
            expires: b.expires,
            gain,
        })
    }

    pub fn tone(&mut self, id: u64) -> PhraseTone {
        let reservation = self.selected.filter(|r| r.tone == id);
        if reservation.is_some() {
            self.selected = None;
        }
        PhraseTone { reservation }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_reference_equations_caps_and_completions() {
        // Independent direct source formula, with the declared endpoint completion.
        let reference = |x: &[[f64; 3]], caps: [f64; 3]| {
            let mut result = vec![0.; x.len().saturating_sub(2)];
            for k in 0..3 {
                let clipped: Vec<_> = x.iter().map(|row| row[k].min(caps[k])).collect();
                let delta = |a: f64, b: f64| {
                    if a + b == 0. {
                        0.
                    } else {
                        (a - b).abs() / (a + b)
                    }
                };
                let raw: Vec<_> = clipped
                    .windows(3)
                    .map(|w| w[1] * (delta(w[0], w[1]) + delta(w[1], w[2])))
                    .collect();
                let maximum = raw.iter().copied().fold(0_f64, f64::max);
                if maximum > 0. {
                    for (r, v) in result.iter_mut().zip(raw) {
                        *r += [0.25, 0.5, 0.25][k] * v / maximum;
                    }
                }
            }
            result
        };
        for x in [
            vec![[0.; 3]; 5],
            [1., 1., 4., 1., 1.].map(|p| [p, 1., 0.]).to_vec(),
            vec![
                [1., 1., 1.],
                [1., 2., 0.],
                [40., 1., 4.],
                [40., 5., 0.],
                [1., 1., 1.],
                [1., 1., 1.],
            ],
            vec![[1., 1., 1.]; 2],
        ] {
            let mut got = vec![0.; x.len().saturating_sub(2)];
            assert!(profile(&x, [12., 4., 4.], &mut got));
            for (a, b) in got.iter().zip(reference(&x, [12., 4., 4.])) {
                assert!((a - b).abs() < 1e-14);
            }
        }
        let mut got = [0.; 3];
        profile(
            &[1., 1., 4., 1., 1.].map(|p| [p, 1., 0.]),
            [12., 4., 4.],
            &mut got,
        );
        assert_eq!(got, [0.03125, 0.25, 0.03125]);
        assert!(!profile(&[[f64::NAN, 0., 0.]; 5], [12., 4., 4.], &mut got));
        assert!(!profile(&[[0.; 3]; 5], [12., 0., 4.], &mut got));
    }

    fn history(pitches: [f32; 6], weight: f64, reporting: bool) -> PhraseBoundary {
        let mut p = PhraseBoundary::new(2, 7, 1000., weight, [12., 4., 4.]);
        p.trace_enabled = reporting;
        let mut seq = 0;
        for (i, hz) in pitches.into_iter().enumerate() {
            let at = i as u64 * 1000;
            if i > 0 {
                seq += 1;
                p.observe(
                    &Receipt {
                        reservation: None,
                        reaction_applied: false,
                        interval_supported: true,
                        record: ExecutionRecord {
                            voice_id: 2,
                            generation: 7,
                            tone_id: i as u64,
                            kind: ExecutionKind::Off,
                            sample: at,
                            available_at: at + 16,
                            execution_sequence: seq,
                            frequency_hz: None,
                            strength: None,
                        },
                        origin: "command_off",
                        gain: 1.,
                    },
                    at + 16,
                );
            }
            seq += 1;
            p.observe(
                &Receipt {
                    reservation: None,
                    reaction_applied: false,
                    interval_supported: true,
                    record: ExecutionRecord {
                        voice_id: 2,
                        generation: 7,
                        tone_id: i as u64 + 1,
                        kind: ExecutionKind::On,
                        sample: at,
                        available_at: at + 16,
                        execution_sequence: seq,
                        frequency_hz: Some(hz),
                        strength: None,
                    },
                    origin: "renderer_on",
                    gain: 1.,
                },
                at + 16,
            );
        }
        p
    }
    fn pitches() -> [f32; 6] {
        [0., 1., 2., 6., 7., 8.].map(|n| 220. * 2_f32.powf(n / 12.))
    }

    #[test]
    fn local_boundary_is_causal_one_shot_and_neutral_without_support() {
        let mut quiet = history(pitches(), 1., false);
        assert!(quiet.trace.is_empty());
        assert_eq!(quiet.opportunity(6000).unwrap().gain, 0.8);
        let mut p = history(pitches(), 1., true);
        assert_eq!(p.trace.len(), 1);
        let b = p.pending.clone().unwrap();
        assert_eq!(
            (b.left.id, b.right.id, b.known, b.expires),
            (3, 4, 5016, 6016)
        );
        assert_eq!(b.score, 0.25);
        let r = p.opportunity(6000).unwrap();
        assert_eq!(r.gain, 0.8);
        assert!(p.opportunity(6001).is_none());
        for weight in [0., f64::MAX] {
            let r = history(pitches(), weight, true).opportunity(5016).unwrap();
            assert!(r.gain.is_finite());
            if weight == 0. {
                assert_eq!(r.gain, 1.);
            }
        }
        assert!(history([220.; 6], 1., true).pending.is_none());
        assert!(history(pitches(), 1., true).opportunity(6016).is_none());
        // Replay one identical prefix before each different future.
        for kind in [
            ExecutionKind::PitchUpdate,
            ExecutionKind::Missing,
            ExecutionKind::On,
        ] {
            let mut p = history(pitches(), 1., true);
            p.finish_publication(5016);
            let saved = serde_json::to_value(&p.trace).unwrap();
            p.observe(
                &Receipt {
                    reservation: None,
                    reaction_applied: false,
                    interval_supported: true,
                    record: ExecutionRecord {
                        voice_id: 2,
                        generation: 7,
                        tone_id: 7,
                        kind,
                        sample: 5500,
                        available_at: 5516,
                        execution_sequence: 12,
                        frequency_hz: Some(880.),
                        strength: None,
                    },
                    origin: "future",
                    gain: 1.,
                },
                5516,
            );
            p.finish_publication(5516);
            assert_eq!(serde_json::to_value(&p.trace).unwrap(), saved);
            assert!(p.pending.is_none());
        }
    }

    #[test]
    fn duplicate_generation_future_and_missing_receipts_never_retarget() {
        let mut p = history(pitches(), 1., true);
        let mut receipt = Receipt {
            reservation: None,
            reaction_applied: false,
            interval_supported: true,
            record: ExecutionRecord {
                voice_id: 2,
                generation: 8,
                tone_id: 7,
                kind: ExecutionKind::Missing,
                sample: 5020,
                available_at: 5032,
                execution_sequence: 12,
                frequency_hz: None,
                strength: None,
            },
            origin: "gap",
            gain: 1.,
        };
        p.observe(&receipt, 5032);
        assert!(p.pending.is_some());
        receipt.record.generation = 7;
        p.observe(&receipt, 5031);
        assert!(p.pending.is_some());
        receipt.record.execution_sequence = 11;
        p.observe(&receipt, 5032);
        assert!(p.pending.is_some());
        receipt.record.execution_sequence = 13;
        p.observe(&receipt, 5032);
        assert!(p.pending.is_none());
        assert_eq!(p.len, 0);
    }
}
