//! Representative-onset body footprint: a declared approximation frozen from the recipe
//! alone, not a prediction of the tone that will actually sound.

use super::energy::project_window;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::phonation_engine::OnsetKick;
use crate::life::schedule_renderer::modal_phase_seed;
use crate::life::self_prediction::ToneEnergy;
use crate::life::sound::{
    AutonomousPulseSpec, BodyKind, BodySnapshot, RenderModulatorSpec, RenderModulatorStateKind,
    Tone, ToneAdsr,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Registered representative values (i11-onset-comparison §4.2).
const KICK_STRENGTH: f32 = 1.0;
/// The footprint is normalized to its peak bin, so the amplitude is a pure scale. A fixed
/// value keeps the live gain (gate, vitality, release) out of the identity.
const AMP: f32 = 1.0;
const CAP_SECONDS: f32 = 4.0;
const HOP: usize = 64;

/// Pins the live envelope state out of the recipe. The representative kick restarts an
/// entrained envelope from zero and the prediction never reads the autonomous pulse phase,
/// so these values cannot reach the footprint; left live, they change the identity every
/// hop. `DroneSway.phase` survives a kick and stays a generation input.
pub(crate) fn representative_modulator(spec: RenderModulatorSpec) -> RenderModulatorSpec {
    match spec {
        RenderModulatorSpec::EntrainPulse {
            attack_step,
            decay_rate,
            sustain_level,
            alpha_gain,
            beta_gain,
            autonomous_pulse,
            ..
        } => RenderModulatorSpec::EntrainPulse {
            attack_step,
            decay_rate,
            sustain_level,
            initial_state: RenderModulatorStateKind::Idle,
            initial_env_level: 0.,
            alpha_gain,
            beta_gain,
            autonomous_pulse: autonomous_pulse.map(|pulse| AutonomousPulseSpec {
                phase_0_1: 0.,
                ..pulse
            }),
        },
        spec => spec,
    }
}

/// Generation inputs of the representative tone. Everything here is hashed into `Identity`.
#[derive(Clone, Debug)]
pub(crate) struct Recipe {
    pub body: BodySnapshot,
    pub freq_hz: f32,
    pub hold: Tick,
    pub adsr: Option<ToneAdsr>,
    pub modulator: RenderModulatorSpec,
    pub smoothing_tau_sec: f32,
    pub fs: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct Identity {
    pub source_id: u64,
    pub body_generation: u32,
    pub recipe_hash: [u8; 32],
}

impl Identity {
    pub(crate) fn new(source_id: u64, body_generation: u32, recipe: &Recipe) -> Self {
        let mut hasher = Sha256::new();
        let body = &recipe.body;
        hasher.update([match body.kind {
            BodyKind::Sine => 0u8,
            BodyKind::Harmonic => 1,
            BodyKind::Modal => 2,
        }]);
        for value in [
            body.amp_scale,
            body.brightness,
            body.inharmonic,
            body.spread,
            body.motion,
        ] {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update((body.unison as u64).to_le_bytes());
        match body.ratios.as_deref() {
            None => hasher.update([0u8]),
            Some(ratios) => {
                hasher.update([1u8]);
                hasher.update((ratios.len() as u64).to_le_bytes());
                for ratio in ratios {
                    hasher.update(ratio.to_bits().to_le_bytes());
                }
            }
        }
        hasher.update(AMP.to_bits().to_le_bytes());
        hasher.update(recipe.freq_hz.to_bits().to_le_bytes());
        hasher.update(recipe.hold.to_le_bytes());
        match recipe.adsr {
            None => hasher.update([0u8]),
            Some(adsr) => {
                hasher.update([1u8]);
                for value in [
                    adsr.attack_sec,
                    adsr.decay_sec,
                    adsr.sustain_level,
                    adsr.release_sec,
                ] {
                    hasher.update(value.to_bits().to_le_bytes());
                }
            }
        }
        hash_modulator(&mut hasher, &recipe.modulator);
        hasher.update(recipe.smoothing_tau_sec.to_bits().to_le_bytes());
        hasher.update(KICK_STRENGTH.to_bits().to_le_bytes());
        hasher.update(modal_phase_seed(source_id, 0, 0).to_le_bytes());
        hasher.update(recipe.fs.to_bits().to_le_bytes());
        Self {
            source_id,
            body_generation,
            recipe_hash: hasher.finalize().into(),
        }
    }
}

/// `RenderModulatorSpec` carries no release-build `Serialize`, so the encoding is spelled out.
fn hash_modulator(hasher: &mut Sha256, spec: &RenderModulatorSpec) {
    match spec {
        RenderModulatorSpec::EntrainPulse {
            attack_step,
            decay_rate,
            sustain_level,
            initial_state,
            initial_env_level,
            alpha_gain,
            beta_gain,
            autonomous_pulse,
        } => {
            hasher.update([0u8]);
            for value in [
                attack_step,
                decay_rate,
                sustain_level,
                initial_env_level,
                alpha_gain,
                beta_gain,
            ] {
                hasher.update(value.to_bits().to_le_bytes());
            }
            hasher.update([match initial_state {
                crate::life::sound::RenderModulatorStateKind::Idle => 0u8,
                crate::life::sound::RenderModulatorStateKind::Attack => 1,
                crate::life::sound::RenderModulatorStateKind::Decay => 2,
            }]);
            match autonomous_pulse {
                None => hasher.update([0u8]),
                Some(pulse) => {
                    hasher.update([1u8, u8::from(pulse.retrigger)]);
                    for value in [
                        pulse.rate_hz,
                        pulse.phase_0_1,
                        pulse.env_open_threshold,
                        pulse.mag_threshold,
                        pulse.alpha_threshold,
                    ] {
                        hasher.update(value.to_bits().to_le_bytes());
                    }
                }
            }
        }
        RenderModulatorSpec::SeqGate { duration_sec } => {
            hasher.update([1u8]);
            hasher.update(duration_sec.to_bits().to_le_bytes());
        }
        RenderModulatorSpec::DroneSway { phase, sway_rate } => {
            hasher.update([2u8]);
            hasher.update(phase.to_bits().to_le_bytes());
            hasher.update(sway_rate.to_bits().to_le_bytes());
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Request {
    pub identity: Identity,
    pub requested_at: u64,
    pub recipe: Recipe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum State {
    Body,
    BodySilent,
    Unsupported(&'static str),
}

/// The values stood in for quantities that only exist after the selection.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Representative {
    pub amp: f32,
    pub kick_strength: f32,
    pub seed: u64,
    pub hold_samples: u64,
    pub rhythms_default: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Record {
    pub identity: Identity,
    pub requested_at: u64,
    pub computed_at: u64,
    pub received_at: Option<u64>,
    pub d_samples: u64,
    pub truncated: bool,
    pub state: State,
    pub energies: [f64; 16],
    pub power: [f64; 16],
    pub representative: Representative,
}

pub(crate) fn compute(request: &Request) -> Record {
    let started = std::time::Instant::now();
    let recipe = &request.recipe;
    let seed = modal_phase_seed(request.identity.source_id, 0, 0);
    let cap = (CAP_SECONDS * recipe.fs) as u64;
    let hold = recipe.hold.min(cap);
    let mut record = Record {
        identity: request.identity,
        requested_at: request.requested_at,
        computed_at: request.requested_at,
        received_at: None,
        d_samples: 0,
        truncated: false,
        state: State::Unsupported("tone"),
        energies: [0.; 16],
        power: [0.; 16],
        representative: Representative {
            amp: AMP,
            kick_strength: KICK_STRENGTH,
            seed,
            hold_samples: hold,
            rhythms_default: true,
        },
    };
    // The worker has no hop clock; the finish time is the request plus the measured cost.
    let finish = |mut record: Record| {
        record.computed_at =
            request.requested_at + (started.elapsed().as_secs_f64() * f64::from(recipe.fs)) as u64;
        record
    };

    let Some(mut tone) = Tone::from_parts(
        Timebase {
            fs: recipe.fs,
            hop: HOP,
        },
        0,
        hold,
        recipe.freq_hz,
        AMP,
        Some(recipe.body.clone()),
        Some(recipe.modulator.clone()),
        recipe.adsr,
    ) else {
        return finish(record);
    };
    tone.set_smoothing_tau_sec(recipe.smoothing_tau_sec);
    tone.seed_modal_phases(seed);
    tone.schedule_planned_kick(OnsetKick {
        strength: KICK_STRENGTH,
    });
    tone.arm_onset_trigger(KICK_STRENGTH);

    let (_, amplitude, envelope) = tone.prediction_parameters(None);
    let frozen = ToneEnergy {
        amplitude,
        envelope,
        control: Some(tone.prediction_control(0, &NeuralRhythms::default())),
        scheduled_release: None,
        sine: tone.prediction_sine(0),
        bank: tone.prediction_bank(0),
    };
    record.d_samples = envelope.release_end.min(cap);
    record.truncated = envelope.release_end > cap;

    record.state = State::Unsupported("window");
    let Some(window) = project_window(
        &[],
        Some(([true, false], frozen)),
        0,
        (None, None),
        0,
        [0, record.d_samples],
        true,
    ) else {
        return finish(record);
    };
    record.state = State::Unsupported("coherent");
    let mut energies = [0.; 16];
    for (out, coherent) in energies.iter_mut().zip(window.coherent_energies) {
        let Some(value) = coherent else {
            return finish(record);
        };
        *out = value;
    }
    record.energies = energies;
    let peak = energies.iter().copied().fold(0., f64::max);
    if peak > 0. {
        record.state = State::Body;
        for (power, energy) in record.power.iter_mut().zip(energies) {
            *power = energy / peak;
        }
    } else {
        record.state = State::BodySilent;
    }
    finish(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::action_candidates::energy::{ScheduledRequest, Worker};
    use crate::temporal_cognition::body::VOICES;
    use std::sync::Arc;

    const FS: f32 = 48000.;

    fn recipe(kind: BodyKind, hold: Tick) -> Recipe {
        Recipe {
            body: BodySnapshot {
                kind,
                amp_scale: 1.,
                brightness: 0.6,
                inharmonic: 0.,
                spread: 0.,
                unison: 1,
                motion: 0.,
                ratios: matches!(kind, BodyKind::Modal)
                    .then(|| (1..=16).map(|k| k as f32).collect::<Arc<[f32]>>()),
            },
            freq_hz: 293.,
            hold,
            adsr: Some(ToneAdsr {
                attack_sec: 0.001,
                decay_sec: 0.,
                sustain_level: 1.,
                release_sec: 0.1,
            }),
            modulator: RenderModulatorSpec::SeqGate { duration_sec: 10. },
            smoothing_tau_sec: 0.002,
            fs: FS,
        }
    }

    fn request(recipe: Recipe) -> Request {
        Request {
            identity: Identity::new(3, 1, &recipe),
            requested_at: 8192,
            recipe,
        }
    }

    #[test]
    fn every_body_kind_yields_a_normalized_sixteen_bin_footprint() {
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            let hold = (0.5 * FS) as Tick;
            let record = compute(&request(recipe(kind, hold)));
            assert_eq!(record.state, State::Body, "{kind:?}");
            assert!(!record.truncated && record.d_samples > hold, "{kind:?}");
            assert!(record.computed_at >= record.requested_at);
            assert_eq!(record.received_at, None);
            assert_eq!(
                record.representative,
                Representative {
                    amp: 1.,
                    kick_strength: 1.,
                    seed: modal_phase_seed(3, 0, 0),
                    hold_samples: hold,
                    rhythms_default: true,
                }
            );
            let peak = record.power.iter().copied().fold(0., f64::max);
            assert!((peak - 1.).abs() < 1e-12, "{kind:?} peak={peak}");
            for power in record.power {
                assert!((0. ..=1.).contains(&power), "{kind:?} power={power}");
            }
        }
    }

    #[test]
    fn a_held_recipe_is_truncated_at_four_seconds() {
        let record = compute(&request(recipe(BodyKind::Harmonic, Tick::MAX)));
        assert_eq!(record.state, State::Body);
        assert!(record.truncated);
        assert_eq!(record.d_samples, 4 * FS as u64);
        assert_eq!(record.representative.hold_samples, 4 * FS as u64);
    }

    #[test]
    fn the_recipe_hash_covers_every_generation_input() {
        let base = recipe(BodyKind::Harmonic, 24000);
        let identity = Identity::new(3, 1, &base);
        assert_eq!(
            identity,
            Identity::new(3, 1, &recipe(BodyKind::Harmonic, 24000))
        );
        let mut variants = vec![
            Recipe {
                adsr: None,
                ..base.clone()
            },
            Recipe {
                smoothing_tau_sec: 0.003,
                ..base.clone()
            },
            Recipe {
                hold: 24001,
                ..base.clone()
            },
            Recipe {
                freq_hz: 294.,
                ..base.clone()
            },
            Recipe {
                modulator: RenderModulatorSpec::SeqGate { duration_sec: 11. },
                ..base.clone()
            },
            Recipe {
                modulator: RenderModulatorSpec::DroneSway {
                    phase: 0.,
                    sway_rate: 1.,
                },
                ..base.clone()
            },
            Recipe {
                fs: 44100.,
                ..base.clone()
            },
        ];
        let mut brighter = base.clone();
        brighter.body.brightness = 0.7;
        variants.push(brighter);
        let mut ratios = base.clone();
        ratios.body.ratios = Some((1..=8).map(|k| k as f32).collect());
        variants.push(ratios);
        for variant in variants {
            assert_ne!(identity, Identity::new(3, 1, &variant));
        }
        assert_ne!(identity, Identity::new(4, 1, &base));
        assert_ne!(identity, Identity::new(3, 2, &base));
    }

    #[test]
    fn the_live_envelope_state_moves_neither_the_footprint_nor_the_pinned_identity() {
        let entrain =
            |initial_state, initial_env_level, phase_0_1| RenderModulatorSpec::EntrainPulse {
                attack_step: 40.,
                decay_rate: 3.,
                sustain_level: 0.5,
                initial_state,
                initial_env_level,
                alpha_gain: 0.3,
                beta_gain: 0.2,
                autonomous_pulse: Some(AutonomousPulseSpec {
                    rate_hz: 2.,
                    phase_0_1,
                    retrigger: true,
                    env_open_threshold: 0.1,
                    mag_threshold: 0.1,
                    alpha_threshold: 0.1,
                }),
            };
        for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
            let with = |modulator| Recipe {
                modulator,
                ..recipe(kind, 24000)
            };
            let rest = with(entrain(RenderModulatorStateKind::Idle, 0., 0.));
            let reference = compute(&request(rest.clone()));
            assert_eq!(reference.state, State::Body, "{kind:?}");
            let pinned = Identity::new(3, 1, &with(representative_modulator(rest.modulator)));
            for (state, level, phase) in [
                (RenderModulatorStateKind::Attack, 0.9, 0.3),
                (RenderModulatorStateKind::Decay, 0.37, 0.8),
            ] {
                let live = with(entrain(state, level, phase));
                // The representative kick overwrites the live state, bit for bit.
                let record = compute(&request(live.clone()));
                assert_eq!(record.state, reference.state, "{kind:?}");
                assert_eq!(record.d_samples, reference.d_samples, "{kind:?}");
                assert_eq!(
                    record.energies.map(f64::to_bits),
                    reference.energies.map(f64::to_bits),
                    "{kind:?}"
                );
                assert_ne!(Identity::new(3, 1, &live), pinned, "{kind:?}");
                assert_eq!(
                    Identity::new(3, 1, &with(representative_modulator(live.modulator))),
                    pinned,
                    "{kind:?}"
                );
            }
        }
        // A sway keeps its phase through a kick, so the phase stays in the identity.
        let sway = |phase| {
            representative_modulator(RenderModulatorSpec::DroneSway {
                phase,
                sway_rate: 1.,
            })
        };
        let base = recipe(BodyKind::Sine, 24000);
        assert_ne!(
            Identity::new(
                3,
                1,
                &Recipe {
                    modulator: sway(0.),
                    ..base.clone()
                }
            ),
            Identity::new(
                3,
                1,
                &Recipe {
                    modulator: sway(1.),
                    ..base
                }
            )
        );
    }

    #[test]
    fn a_silent_recipe_and_a_moving_body_are_unsupported() {
        let mut silent = recipe(BodyKind::Sine, 24000);
        silent.body.amp_scale = 0.;
        assert_eq!(compute(&request(silent)).state, State::Unsupported("tone"));
        let mut moving = recipe(BodyKind::Harmonic, 24000);
        moving.body.motion = 0.5;
        let record = compute(&request(moving));
        assert_eq!(record.state, State::Unsupported("coherent"));
        assert_eq!(record.power, [0.; 16]);
    }

    #[test]
    fn the_sine_footprint_matches_the_rendered_bin_energies() {
        let recipe = recipe(BodyKind::Sine, (0.25 * FS) as Tick);
        let record = compute(&request(recipe.clone()));
        assert_eq!(record.state, State::Body);
        let mut tone = Tone::from_parts(
            Timebase { fs: FS, hop: HOP },
            0,
            recipe.hold,
            recipe.freq_hz,
            AMP,
            Some(recipe.body.clone()),
            Some(recipe.modulator.clone()),
            recipe.adsr,
        )
        .unwrap();
        tone.set_smoothing_tau_sec(recipe.smoothing_tau_sec);
        tone.seed_modal_phases(modal_phase_seed(3, 0, 0));
        tone.schedule_planned_kick(OnsetKick { strength: 1. });
        tone.arm_onset_trigger(1.);
        let rhythms = NeuralRhythms::default();
        let width = record.d_samples;
        let mut rendered = [0.; 16];
        for tick in 0..width {
            tone.kick_planned_if_due(tick);
            let sample = f64::from(tone.render_tick(tick, FS, 1. / FS, &rhythms));
            let bin = ((u128::from(tick) * 16 / u128::from(width)) as usize).min(15);
            rendered[bin] += sample * sample;
        }
        for k in 0..16 {
            let [left, right] =
                [k, k + 1].map(|edge| (u128::from(width) * edge as u128).div_ceil(16) as u64);
            let actual = rendered[k] / (right - left) as f64;
            let error = (record.energies[k] / actual - 1.).abs();
            assert!(error < 1e-3, "bin {k} relative_error={error}");
        }
    }

    fn scheduled_packet(worker: &mut Worker) -> bool {
        let mut tone = Tone::from_parts(
            Timebase { fs: FS, hop: HOP },
            0,
            200_000,
            293.,
            0.2,
            Some(recipe(BodyKind::Sine, 24000).body),
            Some(RenderModulatorSpec::SeqGate { duration_sec: 10. }),
            recipe(BodyKind::Sine, 24000).adsr,
        )
        .unwrap();
        tone.seed_modal_phases(11);
        tone.arm_onset_trigger(1.);
        let (_, amplitude, envelope) = tone.prediction_parameters(None);
        let frozen = ToneEnergy {
            amplitude,
            envelope,
            control: Some(tone.prediction_control(800, &NeuralRhythms::default())),
            scheduled_release: None,
            sine: tone.prediction_sine(800),
            bank: tone.prediction_bank(800),
        };
        let Some(mut packet) = worker.acquire() else {
            return false;
        };
        packet.scheduled = Some(ScheduledRequest {
            source_id: 2,
            source_generation: 7,
            body_generation: 3,
            issued_at: 800,
            sample_rate: 48000,
            hop: 400,
            period: Some(4000),
        });
        packet.retained.push((1, [true, true], frozen));
        worker.submit(packet, true);
        true
    }

    fn poll_until(worker: &mut Worker, done: impl Fn(&Worker) -> bool) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while std::time::Instant::now() < deadline {
            worker.poll();
            if done(worker) {
                return true;
            }
            std::thread::yield_now();
        }
        false
    }

    #[test]
    fn a_full_request_queue_refuses_footprints_only_past_its_capacity() {
        let mut worker = Worker::new();
        let (mut accepted, mut before_refusal) = (0_u64, None);
        // The worker consumes while the queue fills, so only the first refusal is a bound.
        for _ in 0..VOICES * 2 {
            if worker.request_footprint(request(recipe(BodyKind::Sine, 24000))) {
                accepted += 1;
            } else if before_refusal.is_none() {
                before_refusal = Some(accepted);
            }
        }
        assert!(
            before_refusal.is_some_and(|n| n >= VOICES as u64),
            "{before_refusal:?}"
        );
        assert_eq!(worker.stats.footprint_requested, accepted);
        assert_eq!(
            worker.stats.footprint_dropped,
            (VOICES * 2) as u64 - accepted
        );
        worker.finish();
        assert_eq!(worker.stats.footprint_completed, accepted);
        let expected = accepted - worker.stats.footprint_output_dropped;
        let records: Vec<_> = worker.drain_footprints(12345).collect();
        assert_eq!(records.len() as u64, expected);
        assert!(records.iter().all(|r| r.received_at == Some(12345)));
        assert!(records.iter().all(|r| r.state == State::Body));
    }

    #[test]
    fn footprint_requests_are_taken_before_a_candidate_backlog() {
        let mut worker = Worker::new();
        let submitted = (0..VOICES)
            .filter(|_| scheduled_packet(&mut worker))
            .count() as u64;
        for _ in 0..4 {
            assert!(worker.request_footprint(request(recipe(BodyKind::Sine, 24000))));
        }
        assert!(poll_until(&mut worker, |w| w.stats.footprint_completed == 4));
        assert!(
            worker.stats.completed < submitted,
            "footprints waited for the candidate backlog: {}",
            worker.stats.completed
        );
        worker.finish();
        assert_eq!(worker.stats.completed, submitted);
        assert_eq!(worker.drain_footprints(7).count(), 4);
    }
}
