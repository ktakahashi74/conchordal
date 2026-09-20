//! Acoustic-generation ownership from conserved energy through grouping proposals.

#[cfg(test)]
use super::Config;
use super::{Frontend, Observation, Output};
#[cfg(test)]
use crate::core::log2space::Log2Space;
use crate::temporal_cognition::{
    accents::{
        Ledger, Summary,
        periods::{self, groupings},
    },
    arrival, auditory_timing,
    ridge::Handle,
};

#[derive(Clone, Copy)]
struct Settings {
    forecast: Option<crate::config::TemporalPeriodConfig>,
    capacity: usize,
    window_samples: u64,
    rebuild_admissions: u64,
    peak_separation: f64,
    grouping: groupings::Controls,
}

struct Slot {
    owner: Option<Handle>,
    arrival: Option<arrival::Engine>,
    estimator: periods::Estimator,
    grouping: groupings::Inventory,
}

#[derive(Clone, Copy)]
struct GroupFrame {
    ledger: Summary,
    acoustic_eligible: bool,
    association_known: bool,
    period: periods::View,
    period_source: Option<[u64; 3]>,
    grouping: Option<groupings::View>,
    forecast: Option<arrival::Forecast>,
}

struct Frame {
    end_sample: u64,
    received_at: u64,
    evidence_groups: [Option<GroupFrame>; 7],
    residual: Summary,
    next_owners: [Option<Handle>; 7],
    #[cfg(test)]
    newborn: [Option<Summary>; 7],
}

pub(crate) struct Recurrence {
    settings: Settings,
    frontend: Frontend,
    slots: Box<[Slot; 7]>,
    residual: Ledger,
    frame: Box<Option<Frame>>,
    received_at: u64,
    failure: Option<&'static str>,
    timing: auditory_timing::Stream,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct GroupSnapshot {
    pub ledger: Summary,
    pub active: bool,
    pub acoustic_eligible: bool,
    pub association_known: bool,
    pub peaks: [Option<periods::Peak>; 8],
    pub period_source: Option<[u64; 3]>,
    pub grouping: Option<groupings::Snapshot>,
    pub forecast: Option<arrival::Forecast>,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub end_sample: u64,
    pub received_at: u64,
    pub groups: [Option<GroupSnapshot>; 7],
    pub residual: Summary,
    pub timing: Option<auditory_timing::Snapshot>,
}

impl Recurrence {
    #[cfg(test)]
    fn new(space: Log2Space, config: Config, settings: Settings) -> Result<Self, &'static str> {
        Self::with_frontend(Frontend::new(space, config)?, settings)
    }

    pub(crate) fn configured(
        frontend: Frontend,
        forecast: crate::config::TemporalPeriodConfig,
    ) -> Result<Self, &'static str> {
        let rate = frontend.config.group.sample_rate;
        Self::with_frontend(
            frontend,
            Settings {
                forecast: Some(forecast),
                capacity: 128,
                window_samples: 32 * u64::from(rate),
                rebuild_admissions: 1024,
                peak_separation: 1. / 24.,
                grouping: groupings::Controls {
                    tolerance: 0.1,
                    integers_234_only: false,
                    strict_integer: false,
                    one_skip_words: false,
                },
            },
        )
    }

    fn with_frontend(frontend: Frontend, settings: Settings) -> Result<Self, &'static str> {
        let config = frontend.config;
        let mut slots = Vec::with_capacity(7);
        for generation in 2..9 {
            let placeholder = Handle {
                bus: config.group.bus,
                epoch: config.group.epoch,
                generation,
            };
            slots.push(Slot {
                owner: None,
                arrival: settings.forecast.map(arrival::Engine::new).transpose()?,
                estimator: periods::Estimator::new(
                    placeholder,
                    settings.capacity,
                    config.group.sample_rate,
                    settings.window_samples,
                    settings.rebuild_admissions,
                    settings.peak_separation,
                )?,
                grouping: groupings::Inventory::new(
                    placeholder,
                    config.epoch_start,
                    settings.grouping,
                )?,
            });
        }
        let mut residual = Ledger::new(
            Handle {
                bus: config.group.bus,
                epoch: config.group.epoch,
                generation: 1,
            },
            settings.capacity,
            settings.window_samples,
        )?;
        residual.reset(
            Handle {
                bus: config.group.bus,
                epoch: config.group.epoch,
                generation: 1,
            },
            config.epoch_start,
        )?;
        Ok(Self {
            settings,
            frontend,
            slots: slots
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            residual,
            frame: Box::new(None),
            received_at: config.epoch_start,
            failure: None,
            timing: auditory_timing::Stream::new(
                config.group.sample_rate,
                config.group.hop,
                config.epoch_start,
            ),
        })
    }

    pub(in crate::temporal_cognition) fn advance(
        &mut self,
        end: u64,
        received_at: u64,
        observation: Option<Observation<'_>>,
    ) -> Result<Output, &'static str> {
        if self.failure.is_some() {
            return Err("recurrence epoch has failed");
        }
        let result = self.process(end, received_at, observation);
        if let Err(error) = result {
            self.failure = Some(error);
            *self.frame = None;
        }
        result
    }

    fn process(
        &mut self,
        end: u64,
        received_at: u64,
        observation: Option<Observation<'_>>,
    ) -> Result<Output, &'static str> {
        if received_at < self.received_at
            || end > received_at
            || observation.is_some_and(|o| o.available_end > received_at)
        {
            return Err("unavailable acoustic input or backward receive clock");
        }
        let acoustic = self.frontend.advance(end, observation)?;
        self.consume_acoustic(end, received_at, observation, &acoustic)?;
        Ok(acoustic)
    }

    // Diagnostic scratch must not overlap the frontend's nested proposal stack.
    #[inline(never)]
    fn consume_acoustic(
        &mut self,
        end: u64,
        received_at: u64,
        observation: Option<Observation<'_>>,
        acoustic: &Output,
    ) -> Result<(), &'static str> {
        let mut evidence_groups = [None; 7];
        let mut timing = [None; 7];
        let assignment_total: f64 = acoustic
            .groups
            .assignment
            .rows
            .iter()
            .flatten()
            .flat_map(|r| r.weights)
            .sum();
        // Consume the old assignment before any cache can be rebound to a newborn.
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let Some(owner) = slot.owner else { continue };
            assert_eq!(acoustic.groups.assignment.group_handles[index], Some(owner));
            let mut delivered = false;
            let mut delivered_accent = None;
            if let Some(update) = acoustic.features[index].as_ref() {
                assert_eq!(update.raw.group, owner);
                if let Some(accent) = update.detector.as_ref().and_then(|d| d.accent) {
                    delivered = slot.estimator.deliver(accent, received_at)?.is_some();
                    delivered_accent = delivered.then_some(accent);
                }
            }
            if !delivered {
                slot.estimator.advance(received_at)?;
            }
            let refreshed = slot.grouping.refresh(&slot.estimator)?;
            let forecast = if let Some(engine) = &mut slot.arrival {
                let raw = acoustic.features[index]
                    .as_ref()
                    .map(|u| &u.raw)
                    .filter(|_| self.frontend.energy_eligible[index] && observation.is_some());
                let accent = acoustic.features[index]
                    .as_ref()
                    .and_then(|u| u.detector.and_then(|d| d.accent));
                let view = slot.estimator.view();
                let peak = view.peaks.iter().flatten().copied().max_by(|a, b| {
                    a.support
                        .total_cmp(&b.support)
                        .then(b.period_seconds.total_cmp(&a.period_seconds))
                });
                let support = slot.estimator.source_support().unwrap_or([0; 3]);
                let ctx = arrival::Context {
                    peak,
                    word: peak.and_then(|p| {
                        slot.grouping
                            .snapshot()
                            .and_then(|v| v.word_indicator(p.bin, slot.estimator.source_support()))
                    }),
                    source_start: observation.map_or(support[0], |o| {
                        if support[2] == 0 {
                            o.source_start
                        } else {
                            support[0].min(o.source_start)
                        }
                    }),
                    source_end: observation.map_or(support[1], |o| support[1].max(o.source_end)),
                    available: support[2].max(observation.map_or(0, |o| o.available_end)),
                };
                engine.advance(
                    raw,
                    accent,
                    ctx,
                    received_at,
                    self.frontend.config.group.sample_rate,
                )?
            } else {
                None
            };
            let known = observation.is_some()
                && self.frontend.energy_eligible[index]
                && self.frontend.shape_supported
                && acoustic.features[index].is_some_and(|u| {
                    u.raw.group == owner
                        && u.raw.end == end
                        && u.raw.start == end - self.frontend.config.group.hop
                        && u.raw.known_samples == self.frontend.config.group.hop
                        && u.raw.source_start <= u.raw.start
                        && u.raw.source_end >= end
                        && u.raw.source_end <= received_at
                        && u.raw.available_end <= received_at
                });
            let alpha = if known && assignment_total > 0. {
                acoustic
                    .groups
                    .assignment
                    .rows
                    .iter()
                    .flatten()
                    .map(|r| r.weights[index])
                    .sum::<f64>()
                    / assignment_total
            } else {
                0.
            };
            let _ = (known, alpha, refreshed);
            evidence_groups[index] = Some(GroupFrame {
                ledger: slot.estimator.ledger_summary(),
                acoustic_eligible: self.frontend.energy_eligible[index],
                association_known: observation.is_some()
                    && self.frontend.energy_eligible[index]
                    && self.frontend.shape_supported,
                period: slot.estimator.view(),
                period_source: slot.estimator.source_support(),
                grouping: slot.grouping.snapshot(),
                forecast,
            });
            timing[index] = Some(auditory_timing::Input {
                group: owner,
                known,
                accent: delivered_accent,
                peaks: slot.estimator.view().peaks,
            });
        }
        if let Some(accent) = acoustic.features[7]
            .as_ref()
            .and_then(|u| u.detector.as_ref())
            .and_then(|d| d.accent)
        {
            self.residual.deliver(accent, received_at)?;
        }
        self.residual.advance(received_at)?;
        let next_owners = self.frontend.lifecycle.retained_handles();
        self.timing.advance(end, timing, next_owners);
        let mut newborn = [None; 7];
        for (index, (slot, &next)) in self.slots.iter_mut().zip(&next_owners).enumerate() {
            if slot.owner == next {
                continue;
            }
            if let Some(handle) = next {
                assert!(
                    acoustic
                        .groups
                        .admissions
                        .iter()
                        .flatten()
                        .any(|a| a.children.contains(&Some(handle)))
                );
                slot.estimator.reset(handle, received_at)?;
                slot.arrival = self
                    .settings
                    .forecast
                    .map(arrival::Engine::new)
                    .transpose()?;
                slot.grouping.reset(handle, received_at)?;
                newborn[index] = Some(slot.estimator.ledger_summary());
            }
            slot.owner = next;
        }
        *self.frame = Some(Frame {
            end_sample: end,
            received_at,
            evidence_groups,
            residual: self.residual.summary(),
            next_owners,
            #[cfg(test)]
            newborn,
        });
        self.received_at = received_at;
        Ok(())
    }

    pub(in crate::temporal_cognition) fn acoustic_snapshot(&self, out: &Output) -> super::Snapshot {
        self.frontend.snapshot(out)
    }

    pub(in crate::temporal_cognition) fn arrival_issues(
        &self,
        cut: u64,
    ) -> [Option<arrival::Frozen>; 7] {
        std::array::from_fn(|i| {
            let frame = self.frame.as_ref().as_ref()?;
            if self.failure.is_some() || frame.received_at != cut {
                return None;
            }
            let owner = self.slots[i].owner?;
            if frame.next_owners[i] != Some(owner)
                || frame.evidence_groups[i]?.ledger.group != owner
            {
                return None;
            }
            self.slots[i].arrival.as_ref()?.freeze(
                owner,
                cut,
                self.frontend.config.group.sample_rate,
            )
        })
    }

    pub(crate) fn diagnostics(&self) -> Option<Snapshot> {
        let frame = self.frame.as_ref().as_ref()?;
        let timing = Some(self.timing.snapshot()).filter(|s| s.window[1] == frame.end_sample);
        let snapshot = Snapshot {
            end_sample: frame.end_sample,
            received_at: frame.received_at,
            residual: frame.residual,
            // Finishing at a later delivery clock does not invent acoustic coverage.
            timing,
            groups: std::array::from_fn(|i| {
                frame.evidence_groups[i].map(|g| {
                    let active = frame.next_owners[i] == Some(g.ledger.group);
                    let forecast = g.forecast.filter(|f| {
                        active
                            && self.settings.forecast.is_some_and(|c| {
                                f.valid_for(g.ledger.group, c.model, frame.received_at)
                            })
                    });
                    GroupSnapshot {
                        ledger: g.ledger,
                        active,
                        acoustic_eligible: g.acoustic_eligible,
                        association_known: g.association_known,
                        peaks: g.period.peaks,
                        period_source: g.period_source,
                        grouping: g.grouping.map(|v| v.diagnostics()),
                        forecast,
                    }
                })
            }),
        };
        Some(snapshot)
    }

    pub(crate) fn finish(&mut self, cut: u64) -> Result<(), &'static str> {
        if cut < self.received_at {
            return Err("backward recurrence finish");
        }
        if cut == self.received_at {
            return Ok(());
        }
        let Some(frame) = self.frame.as_mut() else {
            return Ok(());
        };
        for (index, slot) in self.slots.iter_mut().enumerate() {
            slot.estimator.advance(cut)?;
            slot.grouping.refresh(&slot.estimator)?;
            if let Some(group) = frame.evidence_groups[index]
                .as_mut()
                .filter(|g| Some(g.ledger.group) == slot.owner)
            {
                group.ledger = slot.estimator.ledger_summary();
                group.period = slot.estimator.view();
                group.period_source = slot.estimator.source_support();
                group.grouping = slot.grouping.snapshot();
                group.association_known = false;
                let peak = group.period.peaks.iter().flatten().copied().max_by(|a, b| {
                    a.support
                        .total_cmp(&b.support)
                        .then(b.period_seconds.total_cmp(&a.period_seconds))
                });
                let support = slot.estimator.source_support().unwrap_or([0; 3]);
                let context = arrival::Context {
                    peak,
                    word: peak.and_then(|p| {
                        group
                            .grouping
                            .and_then(|v| v.word_indicator(p.bin, slot.estimator.source_support()))
                    }),
                    source_start: support[0],
                    source_end: support[1],
                    available: support[2],
                };
                group.forecast = slot
                    .arrival
                    .as_mut()
                    .map(|a| {
                        a.advance(
                            None,
                            None,
                            context,
                            cut,
                            self.frontend.config.group.sample_rate,
                        )
                    })
                    .transpose()?
                    .flatten();
            }
        }
        self.residual.advance(cut)?;
        frame.residual = self.residual.summary();
        frame.received_at = cut;
        self.received_at = cut;
        Ok(())
    }

    #[cfg(test)]
    fn snapshot(&self) -> Option<&Frame> {
        self.frame.as_ref().as_ref()
    }
}

#[cfg(test)]
mod tests;
