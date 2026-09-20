//! Issue-time private participation forecasts and chronologically sealed own-audio credit.

use super::action_observation::{ActionKind, Outcome};
use crate::{
    config::TemporalPrivateTraceConfig,
    temporal_cognition::{private_trace as trace, reference_inventory::Context},
};

const VOICES: usize = 64;
const PENDING: usize = super::action_observation::CAPACITY;
const N: usize = trace::REFERENCES;

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Stats {
    pub timing_previews: u64,
    pub timing_queries: u64,
    pub timing_rate_limited: u64,
    pub issued: u64,
    pub completed: u64,
    pub learned: u64,
    pub assigned: f64,
    pub unassigned: f64,
    pub unsupported: u64,
    pub capacity_dropped: u64,
    pub output_dropped: u64,
    pub retired_traces: u64,
    pub evicted_traces: u64,
    pub replaced_voices: u64,
    pub pending: usize,
    pub queued: usize,
    pub ignored_deliveries: u64,
    pub errors: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub(crate) struct Fit {
    pub candidate_mass: Option<f64>,
    pub body_default_mass: Option<f64>,
    pub difference: Option<f64>,
    pub paired_support: f64,
    pub unassigned: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct TimingCandidate {
    pub event_sample: u64,
    pub fits: [Fit; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct TimingPreview {
    pub version: u8,
    pub body_generation: u32,
    pub timing_model: &'static str,
    pub body_default_event_sample: u64,
    pub candidates: [Option<TimingCandidate>; 13],
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct Origin {
    pub command_id: Option<u64>,
    pub source_id: u64,
    pub source_generation: u32,
    pub epoch: u64,
    pub bus: u8,
    pub issued_at_sample: u64,
    pub inventory_at_sample: Option<u64>,
}

pub(crate) struct Frozen {
    pub origin: Origin,
    pub head: trace::Head,
    pub intrinsic_period_sec: Option<f64>,
    references: [trace::Reference; N],
    count: usize,
    predictions: [[Option<[f64; 33]>; N]; 2],
}

impl Frozen {
    pub fn fit(&self, rate: u32, candidate: u64, body_default: u64) -> [Fit; 2] {
        paired_fits(
            &self.references[..self.count],
            &self.predictions,
            rate,
            candidate,
            body_default,
        )
    }
}

fn paired_fits(
    references: &[trace::Reference],
    predictions: &[[Option<[f64; 33]>; N]; 2],
    rate: u32,
    candidate: u64,
    body_default: u64,
) -> [Fit; 2] {
    std::array::from_fn(|model| {
        let mut value = trace::PairedLookup::default();
        for (reference, masses) in references.iter().zip(&predictions[model]) {
            let Some(masses) = masses else { continue };
            let next = trace::paired_lookup(
                reference,
                masses,
                candidate as f64 / f64::from(rate),
                body_default as f64 / f64::from(rate),
            );
            value.candidate += next.candidate;
            value.body_default += next.body_default;
            value.support += next.support;
        }
        Fit {
            candidate_mass: (value.support > 0.).then_some(value.candidate),
            body_default_mass: (value.support > 0.).then_some(value.body_default),
            difference: (value.support > 0.).then_some(value.candidate - value.body_default),
            paired_support: value.support.min(1.),
            unassigned: (1. - value.support).max(0.),
        }
    })
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Forecast {
    #[serde(serialize_with = "crate::config::fixed_array::serialize")]
    pub probabilities: [f64; 33],
    pub prior_used: bool,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Entry {
    pub reference: trace::Reference,
    pub forecasts: [Forecast; 2],
    #[serde(serialize_with = "crate::config::fixed_array::serialize")]
    pub target: [f64; 33],
    pub credit: f64,
    pub log_loss: [Option<f64>; 2],
    pub zero_probability: [bool; 2],
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Record {
    pub timing_preview: Option<TimingPreview>,
    pub command_id: u64,
    pub source_id: u64,
    pub source_generation: u32,
    pub epoch: u64,
    pub bus: u8,
    pub head: trace::Head,
    pub issued_at_sample: u64,
    pub inventory_at_sample: Option<u64>,
    pub intrinsic_period_sec: Option<f64>,
    pub observed_interval_sec: Option<[f64; 2]>,
    pub available_at_sample: u64,
    pub sealed_at_sample: u64,
    pub known_fraction: f64,
    pub applied: bool,
    pub unassigned: f64,
    pub error: Option<&'static str>,
    pub entries: [Option<Entry>; N],
}

struct Voice {
    owner: Option<(u64, u32)>,
    last_issue: u64,
    last_preview: Option<u64>,
    models: [trace::Bank; 2],
}

struct Pending {
    preview: Option<TimingPreview>,
    id: u64,
    source: (u64, u32),
    slot: usize,
    epoch: u64,
    issued: u64,
    inventory_at: Option<u64>,
    period: Option<f64>,
    head: trace::Head,
    references: [trace::Reference; N],
    count: usize,
    predictions: [[Option<[f64; 33]>; N]; 2],
    result: Option<Outcome>,
}

pub(crate) struct Bank {
    rate: u32,
    config: TemporalPrivateTraceConfig,
    epoch: Option<u64>,
    context: Context,
    voices: Vec<Voice>,
    pending: Vec<Option<Pending>>,
    ready: Vec<Record>,
    pub stats: Stats,
}

impl Bank {
    pub fn new(rate: u32, config: TemporalPrivateTraceConfig) -> Self {
        Self {
            rate,
            config,
            epoch: None,
            context: Context::default(),
            voices: (0..VOICES)
                .map(|_| Voice {
                    owner: None,
                    last_issue: 0,
                    last_preview: None,
                    models: std::array::from_fn(|i| {
                        trace::Bank::new(
                            0,
                            trace::Parameters {
                                tau: config.tau_sec,
                                kappa: config.kappa,
                                strength_max: config.strength_max,
                                interference: i == 0,
                            },
                            N,
                        )
                        .expect("validated private trace parameters")
                    }),
                })
                .collect(),
            pending: (0..PENDING).map(|_| None).collect(),
            ready: Vec::with_capacity(PENDING),
            stats: Stats::default(),
        }
    }

    pub fn context(&mut self, context: Context, now: u64) {
        if context.inventory.as_ref().is_some_and(|i| {
            i.available_at_sample > now
                || self.context.inventory.as_ref().is_some_and(|old| {
                    i.epoch < old.epoch
                        || (i.epoch == old.epoch && i.available_at_sample < old.available_at_sample)
                })
        }) {
            return;
        }
        if let Some(inventory) = context.inventory.as_ref().filter(|i| i.bus == 0) {
            if self.epoch != Some(inventory.epoch) {
                for v in &mut self.voices {
                    v.last_preview = None;
                    self.stats.retired_traces += v.models[0].keys().count() as u64;
                    for m in &mut v.models {
                        m.reset(inventory.epoch);
                    }
                }
                self.epoch = Some(inventory.epoch);
            }
            for v in &mut self.voices {
                for (i, m) in v.models.iter_mut().enumerate() {
                    let n = m.retain(|key| {
                        context
                            .retained
                            .binary_search(&(key.episode, key.generation))
                            .is_ok()
                    });
                    if i == 0 {
                        self.stats.retired_traces += n as u64;
                    }
                }
            }
        }
        self.context = context;
    }

    pub fn issue(&mut self, outcome: &Outcome, period: Option<f64>) {
        if outcome.command_status != "accepted" || outcome.action == ActionKind::Shutdown {
            return;
        }
        let Some(pending_slot) = self.pending.iter().position(Option::is_none) else {
            self.stats.capacity_dropped += 1;
            return;
        };
        let source = (outcome.source_id, outcome.source_generation);
        let slot = self
            .voices
            .iter()
            .position(|v| v.owner == Some(source))
            .or_else(|| {
                self.voices
                    .iter()
                    .position(|v| v.owner.is_some_and(|o| o.0 == source.0))
            })
            .or_else(|| self.voices.iter().position(|v| v.owner.is_none()))
            .unwrap_or_else(|| {
                self.voices
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, v)| v.last_issue)
                    .unwrap()
                    .0
            });
        let v = &mut self.voices[slot];
        if v.owner != Some(source) {
            self.stats.replaced_voices += u64::from(v.owner.is_some());
            v.owner = Some(source);
            v.last_preview = None;
            for m in &mut v.models {
                m.reset(self.epoch.unwrap_or(0));
            }
        }
        v.last_issue = outcome.command_id;
        let head = if outcome.action == ActionKind::Onset {
            trace::Head::Onset
        } else {
            trace::Head::Release
        };
        let mut pending = Pending {
            preview: None,
            id: outcome.command_id,
            source,
            slot,
            epoch: self.epoch.unwrap_or(0),
            issued: outcome.issued_at_sample,
            inventory_at: None,
            period,
            head,
            references: std::array::from_fn(|i| trace::Reference {
                key: trace::Key {
                    epoch: 0,
                    episode: i as u64,
                    generation: 0,
                    family: trace::Family::Nonperiodic,
                },
                weight: 0.,
                anchors: [trace::Anchor::default(); 7],
            }),
            count: 0,
            predictions: [[None; N]; 2],
            result: None,
        };
        if let Some(inventory) = self.context.inventory.as_ref().filter(|i| {
            i.bus == 0
                && i.available_at_sample <= outcome.issued_at_sample
                && outcome.issued_at_sample - i.available_at_sample
                    <= u64::from(self.rate).div_ceil(10)
        }) {
            pending.inventory_at = Some(inventory.available_at_sample);
            for r in inventory.references.iter().flatten() {
                if self
                    .context
                    .retained
                    .binary_search(&(r.key.episode, r.key.generation))
                    .is_err()
                {
                    continue;
                }
                let index = pending.count;
                pending.references[index] = trace::Reference {
                    key: r.key,
                    weight: r.weight,
                    anchors: r.anchors.map(|a| {
                        a.map_or_else(trace::Anchor::default, |a| {
                            let period =
                                a.period_sec.or(period).filter(|p| p.is_finite() && *p > 0.);
                            trace::Anchor {
                                group: a.group,
                                interval: period.map(|_| a.interval_sec),
                                period: period.unwrap_or(1.),
                                weight: a.weight,
                            }
                        })
                    }),
                };
                for (j, m) in v.models.iter().enumerate() {
                    pending.predictions[j][index] = m.probabilities(r.key, head);
                }
                pending.count += 1;
            }
        }
        self.pending[pending_slot] = Some(pending);
        self.stats.issued += 1;
        self.stats.pending += 1;
    }

    /// Read only the issue-time distributions; hypothetical event times never enter observe().
    pub fn preview(&mut self, outcome: &Outcome) {
        let Some(forecast) = outcome
            .prediction
            .as_ref()
            .filter(|_| outcome.buses[0].routed)
        else {
            return;
        };
        let Some(p) = self.pending.iter_mut().flatten().find(|p| {
            p.id == outcome.command_id
                && p.source == (outcome.source_id, outcome.source_generation)
                && p.preview.is_none()
        }) else {
            return;
        };
        let v = &mut self.voices[p.slot];
        if v.owner != Some(p.source) || self.epoch.unwrap_or(0) != p.epoch {
            return;
        }
        if v.last_preview.is_some_and(|last| {
            outcome.issued_at_sample < last.saturating_add(u64::from(self.rate).div_ceil(20))
        }) {
            self.stats.timing_rate_limited += 1;
            return;
        }
        let default = match outcome.action {
            ActionKind::Onset => outcome
                .scheduled_action_sample
                .and_then(|t| t.checked_add(1)),
            ActionKind::Release => Some(forecast.envelope.release_end),
            ActionKind::Shutdown => None,
        };
        let Some(default) = default.filter(|t| *t >= p.issued && *t != u64::MAX) else {
            return;
        };
        let span = p
            .period
            .filter(|p| p.is_finite() && *p > 0.)
            .map(|period| ((2. * period).min(4.) * f64::from(self.rate)).round() as u64);
        let (times, count) = super::action_candidates::times(p.issued, span, default, [None; 3])
            .expect("validated default event time");
        assert!(
            count <= 13,
            "no calibrated quantiles in the private event diagnostic"
        );
        let mut preview = TimingPreview {
            version: 1,
            body_generation: forecast.body_generation,
            timing_model: "fixed_renderer_end_or_onset_plus_one_sample",
            body_default_event_sample: default,
            candidates: [None; 13],
        };
        for (row, at) in preview.candidates.iter_mut().zip(&times[..count]) {
            let fits = paired_fits(
                &p.references[..p.count],
                &p.predictions,
                self.rate,
                *at,
                default,
            );
            *row = Some(TimingCandidate {
                event_sample: *at,
                fits,
            });
        }
        p.preview = Some(preview);
        v.last_preview = Some(p.issued);
        self.stats.timing_previews += 1;
        self.stats.timing_queries += count as u64;
    }

    pub(crate) fn freeze_onset(
        &self,
        command_id: u64,
        source: (u64, u32),
        issued: u64,
    ) -> Option<Frozen> {
        let p = self.pending.iter().flatten().find(|p| {
            p.id == command_id
                && p.source == source
                && p.issued == issued
                && p.head == trace::Head::Onset
        })?;
        if self.voices[p.slot].owner != Some(source) || self.epoch.unwrap_or(0) != p.epoch {
            return None;
        }
        Some(Frozen {
            origin: Origin {
                command_id: Some(command_id),
                source_id: source.0,
                source_generation: source.1,
                epoch: p.epoch,
                bus: 0,
                issued_at_sample: issued,
                inventory_at_sample: p.inventory_at,
            },
            head: p.head,
            intrinsic_period_sec: p.period,
            references: p.references,
            count: p.count,
            predictions: p.predictions,
        })
    }

    /// A current release-head query creates neither a pending command nor a trace.
    pub(crate) fn freeze_release(
        &self,
        source: (u64, u32),
        issued: u64,
        period: Option<f64>,
    ) -> Option<Frozen> {
        let voice = self.voices.iter().find(|v| v.owner == Some(source))?;
        let epoch = self.epoch?;
        let mut frozen = Frozen {
            origin: Origin {
                command_id: None,
                source_id: source.0,
                source_generation: source.1,
                epoch,
                bus: 0,
                issued_at_sample: issued,
                inventory_at_sample: None,
            },
            head: trace::Head::Release,
            intrinsic_period_sec: period,
            references: std::array::from_fn(|i| trace::Reference {
                key: trace::Key {
                    epoch: 0,
                    episode: i as u64,
                    generation: 0,
                    family: trace::Family::Nonperiodic,
                },
                weight: 0.,
                anchors: [trace::Anchor::default(); 7],
            }),
            count: 0,
            predictions: [[None; N]; 2],
        };
        if let Some(inventory) = self.context.inventory.as_ref().filter(|i| {
            i.bus == 0
                && i.epoch == epoch
                && i.available_at_sample <= issued
                && issued - i.available_at_sample <= u64::from(self.rate).div_ceil(10)
        }) {
            frozen.origin.inventory_at_sample = Some(inventory.available_at_sample);
            for r in inventory.references.iter().flatten() {
                if self
                    .context
                    .retained
                    .binary_search(&(r.key.episode, r.key.generation))
                    .is_err()
                {
                    continue;
                }
                let index = frozen.count;
                frozen.references[index] = trace::Reference {
                    key: r.key,
                    weight: r.weight,
                    anchors: r.anchors.map(|a| {
                        a.map_or_else(trace::Anchor::default, |a| {
                            let period =
                                a.period_sec.or(period).filter(|p| p.is_finite() && *p > 0.);
                            trace::Anchor {
                                group: a.group,
                                interval: period.map(|_| a.interval_sec),
                                period: period.unwrap_or(1.),
                                weight: a.weight,
                            }
                        })
                    }),
                };
                for (j, model) in voice.models.iter().enumerate() {
                    frozen.predictions[j][index] = model.probabilities(r.key, trace::Head::Release);
                }
                frozen.count += 1;
            }
        }
        Some(frozen)
    }

    pub fn complete(&mut self, outcome: Outcome) {
        let Some(p) = self
            .pending
            .iter_mut()
            .flatten()
            .find(|p| p.id == outcome.command_id && p.result.is_none())
        else {
            self.stats.ignored_deliveries += 1;
            return;
        };
        p.result = Some(outcome);
        self.stats.queued += 1;
    }

    pub fn seal(&mut self, watermark: u64, now: u64) {
        loop {
            let next = self
                .pending
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    let p = p.as_ref()?;
                    let o = p.result.as_ref()?;
                    let end = self.interval(o).map_or(0, |t| t + 1);
                    (end <= watermark).then_some((i, (end, p.head, p.id)))
                })
                .min_by_key(|(_, order)| *order);
            let Some((i, _)) = next else { break };
            let p = self.pending[i].take().unwrap();
            self.stats.pending -= 1;
            self.stats.queued -= 1;
            let o = p.result.unwrap();
            let interval = self.interval(&o).map(|t| {
                [
                    t as f64 / f64::from(self.rate),
                    (t + 1) as f64 / f64::from(self.rate),
                ]
            });
            let known = o
                .scheduled_action_sample
                .zip(o.window_end_sample)
                .filter(|(a, b)| a < b)
                .map_or(0., |(a, b)| {
                    (o.observed_samples as f64 / (b - a) as f64).clamp(0., 1.)
                });
            let mut record = Record {
                timing_preview: p.preview,
                command_id: p.id,
                source_id: p.source.0,
                source_generation: p.source.1,
                epoch: p.epoch,
                bus: 0,
                head: p.head,
                issued_at_sample: p.issued,
                inventory_at_sample: p.inventory_at,
                intrinsic_period_sec: p.period,
                observed_interval_sec: interval,
                available_at_sample: o.available_at_sample,
                sealed_at_sample: now,
                known_fraction: known,
                applied: false,
                unassigned: 1.,
                error: None,
                entries: [None; N],
            };
            let context = self.context.inventory.as_ref().filter(|s| {
                s.bus == 0
                    && s.epoch == p.epoch
                    && s.available_at_sample <= now
                    && now - s.available_at_sample <= u64::from(self.rate).div_ceil(10)
            });
            let v = &mut self.voices[p.slot];
            let supported = context.is_some() && v.owner == Some(p.source);
            let mut retained = [p.references[0].key; 3 * N];
            let mut n = 0;
            if supported {
                for key in v.models[0]
                    .keys()
                    .chain(v.models[1].keys())
                    .chain(p.references[..p.count].iter().map(|r| r.key))
                {
                    if key.epoch == p.epoch
                        && self
                            .context
                            .retained
                            .binary_search(&(key.episode, key.generation))
                            .is_ok()
                        && !retained[..n].contains(&key)
                    {
                        retained[n] = key;
                        n += 1;
                    }
                }
            }
            let mut credits = [0.; N];
            if supported && interval.is_some() {
                for (j, m) in v.models.iter_mut().enumerate() {
                    match m.observe(trace::Outcome {
                        id: p.id,
                        head: p.head,
                        interval,
                        references: &p.references[..p.count],
                        observed_fraction: known,
                        retained: &retained[..n],
                        confirmed: true,
                    }) {
                        Ok(receipt) => {
                            if j == 0 {
                                credits = receipt.credits;
                                record.applied = receipt.applied;
                                record.unassigned = receipt.unassigned;
                                self.stats.retired_traces +=
                                    receipt.removed.iter().flatten().count() as u64;
                                self.stats.evicted_traces +=
                                    receipt.evicted.iter().flatten().count() as u64;
                            }
                        }
                        Err(e) => {
                            record.error = Some(e);
                            self.stats.errors += 1;
                        }
                    }
                }
            }
            for (i, r) in p.references[..p.count].iter().enumerate() {
                let bins = if r.key.family == trace::Family::Periodic {
                    32
                } else {
                    33
                };
                let forecasts = std::array::from_fn(|j| Forecast {
                    probabilities: p.predictions[j][i].unwrap_or_else(|| {
                        std::array::from_fn(|b| if b < bins { 1. / bins as f64 } else { 0. })
                    }),
                    prior_used: p.predictions[j][i].is_none(),
                });
                let (mut target, _, coverage) = interval.map_or(([0.; 33], [0.; 33], 0.), |t| {
                    trace::timing(t, r, self.config.tau_sec)
                });
                if coverage > 0. {
                    for t in &mut target {
                        *t /= coverage;
                    }
                }
                let mut zero_probability = [false; 2];
                let log_loss = std::array::from_fn(|j| {
                    if credits[i] <= 0. {
                        return None;
                    }
                    let mut loss = 0.;
                    for (&t, &q) in target.iter().zip(&forecasts[j].probabilities) {
                        if t > 0. {
                            if q == 0. {
                                zero_probability[j] = true;
                                return None;
                            }
                            loss -= t * q.ln();
                        }
                    }
                    Some(loss)
                });
                record.entries[i] = Some(Entry {
                    reference: *r,
                    forecasts,
                    target,
                    credit: credits[i],
                    log_loss,
                    zero_probability,
                });
            }
            self.stats.completed += 1;
            self.stats.learned += u64::from(credits.iter().any(|c| *c > 0.));
            self.stats.unsupported += u64::from(!supported || interval.is_none());
            self.stats.assigned += 1. - record.unassigned;
            self.stats.unassigned += record.unassigned;
            if self.ready.len() < self.ready.capacity() {
                self.ready.push(record);
            } else {
                self.stats.output_dropped += 1;
            }
        }
    }

    fn interval(&self, o: &Outcome) -> Option<u64> {
        if o.command_status != "accepted"
            || !o.buses[0].routed
            || o.buses[0].status == "invalid_sample"
        {
            return None;
        }
        let activity = o.buses[0].first_activity_sample?;
        let t = match o.action {
            ActionKind::Onset => activity,
            ActionKind::Release => o.renderer_end_sample?,
            ActionKind::Shutdown => return None,
        };
        (t >= o.scheduled_action_sample?
            && t < o.window_end_sample?
            && t < o.available_at_sample
            && t < o.contiguous_observed_end_sample?)
            .then_some(t)
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Record> + '_ {
        self.ready.drain(..)
    }
}

#[cfg(test)]
mod tests;
