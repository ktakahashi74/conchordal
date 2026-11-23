use super::temporal_ports::TemporalObservation;
use super::*;
use crate::life::arrival_cost::ArrivalSources;
use crate::life::onset_footprint::Worker;
use crate::life::voice::Voice;
use crate::temporal_cognition::capture::BodyCapture;

#[derive(Default)]
pub(super) struct OnsetConfig {
    config: Option<crate::config::TemporalOnsetComparisonConfig>,
    deterministic: bool,
}

impl OnsetConfig {
    pub(super) fn from_config(config: &AppConfig, deterministic: bool) -> Self {
        Self {
            config: config.temporal_onset_comparison,
            deterministic,
        }
    }

    pub(super) fn prepare(&self) -> OnsetRuntime {
        OnsetRuntime {
            #[cfg(feature = "native")]
            footprint_worker: self.config.map(|_| {
                let mut worker = crate::life::onset_footprint::Worker::new();
                if self.deterministic {
                    worker.deliver_footprints_deterministically();
                }
                worker
            }),
            #[cfg(not(feature = "native"))]
            footprint_worker: None,
            participation_decisions: Vec::new(),
        }
    }

    pub(super) fn configure(&self, voices: &mut [Voice], trace: bool) -> bool {
        if self.config.is_some() {
            for voice in voices {
                voice.phonation_engine.set_onset_comparison(self.config);
                voice.phonation_engine.set_decision_trace(trace);
            }
            true
        } else {
            false
        }
    }

    pub(super) fn arrival<'a>(
        &self,
        observation: &'a TemporalObservation,
        capture: &'a BodyCapture,
    ) -> ArrivalSources<'a> {
        ArrivalSources {
            snapshot: self
                .config
                .filter(|c| c.arrival)
                .map(|_| &observation.temporal_frames[0]),
            body: observation.body_snapshot.as_deref(),
            capture: capture.as_ref(),
        }
    }
}

pub(super) struct OnsetRuntime {
    footprint_worker: Option<Worker>,
    /// Traced decisions wait here so their encoding is charged to the report phase.
    participation_decisions: Vec<(
        u64,
        crate::life::temporal_participation::ParticipationDecision,
    )>,
}

impl OnsetRuntime {
    #[cfg(test)]
    pub(super) fn footprint_worker(&self) -> Option<&Worker> {
        self.footprint_worker.as_ref()
    }

    pub(super) fn request(&mut self, voices: &mut [Voice], fs: f32, now: Tick, phase3: bool) {
        let Some(worker) = self.footprint_worker.as_mut() else {
            return;
        };
        worker.prepare(
            voices
                .iter()
                .map(|voice| (voice.id(), voice.metadata.generation, voice.body_snapshot())),
        );
        for voice in voices {
            if !voice.is_alive() {
                continue;
            }
            let Some(mut recipe) = voice.footprint_recipe(fs) else {
                continue;
            };
            recipe.renderer_phase3 = phase3;
            let Some(body_generation) =
                worker.body_generation((voice.id(), voice.metadata.generation))
            else {
                continue;
            };
            let identity =
                crate::life::onset_footprint::Identity::new(voice.id(), body_generation, &recipe);
            let resent = voice
                .phonation_engine
                .footprint_hop(identity, now, &recipe, |request| {
                    worker.request_footprint(request)
                });
            if resent {
                worker.stats.footprint_resent += 1;
            }
        }
    }

    #[cfg(feature = "native")]
    pub(super) fn route(
        &mut self,
        voices: &mut [Voice],
        reporter: &mut Option<JsonlReporter>,
        now: Tick,
    ) {
        if let Some(worker) = self.footprint_worker.as_mut() {
            route_body_footprints(voices, worker, reporter, now);
        }
    }

    #[cfg(feature = "native")]
    pub(super) fn finish(
        &mut self,
        voices: &mut [Voice],
        reporter: &mut Option<JsonlReporter>,
        now: Tick,
    ) {
        if let Some(worker) = self.footprint_worker.as_mut() {
            worker.finish();
            route_body_footprints(voices, worker, reporter, now);
            report_try(reporter, "final body footprint worker", |writer| {
                writer.write_body_footprint_worker(&worker.stats)
            });
        }
    }

    pub(super) fn collect(&mut self, id: u64, voice: &mut Voice) {
        self.participation_decisions.extend(
            voice
                .phonation_engine
                .drain_participation_decisions()
                .map(|decision| (id, decision)),
        );
    }

    pub(super) fn emit(&mut self, reporter: &mut Option<JsonlReporter>, sample_rate: u32) {
        for (voice_id, decision) in self.participation_decisions.drain(..) {
            report_try(reporter, "participation decision", |writer| {
                writer.write_participation_decision(voice_id, sample_rate, &decision)
            });
        }
    }
}

#[cfg(feature = "native")]
pub(super) fn route_body_footprints(
    voices: &mut [crate::life::voice::Voice],
    worker: &mut crate::life::onset_footprint::Worker,
    reporter: &mut Option<JsonlReporter>,
    now: Tick,
) {
    let mut superseded = 0;
    for record in worker.drain_footprints(now) {
        let discarded = voices
            .iter_mut()
            .find(|voice| voice.id() == record.identity.source_id)
            .is_some_and(|voice| voice.phonation_engine.footprint_receive(&record));
        superseded += u64::from(discarded);
        report_try(reporter, "body footprint", |writer| {
            writer.write_body_footprint(&record, discarded)
        });
    }
    worker.stats.footprint_superseded += superseded;
    let mut released = 0;
    for identity in worker.drain_released_footprints() {
        if let Some(voice) = voices
            .iter_mut()
            .find(|voice| voice.id() == identity.source_id)
            && voice.phonation_engine.footprint_release(identity)
        {
            released += 1;
        }
    }
    worker.stats.footprint_released += released;
}
