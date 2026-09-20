//! Offline seven-class ground truth for I10 projection checks; never a live predictor.

use clap::Parser;
use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::Timebase;
use conchordal::life::action_candidates::{BodyState, Class};
use conchordal::life::phonation_engine::{OnsetKick, ToneCmd, ToneUpdate};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, ToneAdsr};
use conchordal::life::voice::{PhonationBatch, ToneSpec};
use conchordal::scenario::control::Routing;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const FS: u32 = 48_000;
const HOP: usize = 480;
const ISSUE: u64 = 14_400;
const HORIZON: usize = 192_000;
const PERIOD: u64 = 9_600;
const SOURCE: u64 = 20260918;

#[derive(Parser)]
struct Args {
    output: PathBuf,
    #[arg(long, default_value_t = HOP)]
    hop_samples: usize,
}

#[derive(Clone, Copy)]
struct Recipe {
    kind: BodyKind,
    routing: Routing,
    sway: bool,
}

impl Recipe {
    fn note(self, source: u64, id: u64, onset: u64) -> PhonationBatch {
        let second = id == 2;
        PhonationBatch {
            body_policy: None,
            body_opportunity: None,
            intrinsic_period_sec: Some(PERIOD as f64 / FS as f64),
            source_id: source,
            source_generation: 0,
            routing: self.routing,
            cmds: vec![
                ToneCmd::On {
                    tone_id: id,
                    kick: OnsetKick { strength: 1. },
                },
                ToneCmd::Update {
                    tone_id: id,
                    at_tick: Some(onset),
                    update: ToneUpdate {
                        target_freq_hz: None,
                        target_amp: None,
                        continuous_drive: Some(if self.kind == BodyKind::Modal {
                            0.005
                        } else {
                            0.002
                        }),
                    },
                },
            ],
            tones: vec![ToneSpec {
                opportunity: None,
                tone_id: id,
                onset,
                hold_ticks: Some(if id == 3 {
                    33_600
                } else if second {
                    60_000
                } else {
                    48_000
                }),
                freq_hz: if second { 330. } else { 220. },
                amp: if second { 0.07 } else { 0.1 },
                smoothing_tau_sec: 0.03,
                body: BodySnapshot {
                    kind: self.kind,
                    amp_scale: 1.,
                    brightness: 0.4,
                    inharmonic: 0.,
                    spread: 0.05,
                    unison: 1,
                    motion: 0.1,
                    ratios: None,
                },
                render_modulator: if self.sway {
                    RenderModulatorSpec::DroneSway {
                        phase: 0.37,
                        sway_rate: 0.7,
                    }
                } else {
                    RenderModulatorSpec::SeqGate { duration_sec: 3. }
                },
                adsr: Some(ToneAdsr {
                    attack_sec: 0.03,
                    decay_sec: 0.1,
                    sustain_level: 0.65,
                    release_sec: 0.8,
                }),
            }],
            onsets: Vec::new(),
        }
    }

    fn prefix(
        self,
        active: bool,
        external: bool,
        time: Timebase,
        issue: u64,
    ) -> (ScheduleRenderer, NeuralRhythms, [Vec<f32>; 2]) {
        let mut renderer = ScheduleRenderer::new(time);
        let mut audio = [
            Vec::with_capacity(issue as usize),
            Vec::with_capacity(issue as usize),
        ];
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.freq_hz = 2.;
        rhythms.delta.freq_hz = 1.;
        for now in (0..issue).step_by(time.hop) {
            let mut commands = Vec::new();
            if active && now == 0 {
                commands.push(self.note(SOURCE, 1, 0));
            }
            if active && now <= 5760 && 5760 < now + time.hop as u64 {
                commands.push(self.note(SOURCE, 2, 5760));
            }
            if active && now == issue - 2 * time.hop as u64 {
                commands.push(PhonationBatch {
                    source_id: SOURCE,
                    cmds: vec![ToneCmd::Update {
                        tone_id: 1,
                        at_tick: Some(issue + 2400),
                        update: ToneUpdate {
                            target_freq_hz: Some(226.),
                            target_amp: Some(0.09),
                            continuous_drive: None,
                        },
                    }],
                    ..PhonationBatch::default()
                });
            }
            if external && now == 0 {
                let mut other = self.note(SOURCE + 1, 1, 0);
                other.routing = Routing {
                    to_habitat: true,
                    to_presentation: true,
                };
                other.tones[0].freq_hz = 997.;
                other.tones[0].amp = 0.4;
                commands.push(other);
            }
            let frame = renderer.render(&commands, now, &rhythms);
            audio[0].extend_from_slice(frame.habitat);
            audio[1].extend_from_slice(frame.presentation);
            for _ in 0..time.hop {
                rhythms.advance_in_place(1. / FS as f32);
            }
        }
        (renderer, rhythms, audio)
    }
}

fn save(path: &Path, samples: &[f32]) {
    let mut out = BufWriter::new(File::create(path).unwrap());
    for sample in samples {
        out.write_all(&sample.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();
}

fn main() {
    let args = Args::parse();
    assert!(
        [480, 512].contains(&args.hop_samples),
        "registered hops are 480 and 512"
    );
    let time = Timebase {
        fs: FS as f32,
        hop: args.hop_samples,
    };
    let issue = ISSUE.div_ceil(time.hop as u64) * time.hop as u64;
    assert_eq!(HORIZON % time.hop, 0);
    fs::create_dir(&args.output).expect("use a fresh output directory");
    let mut cases = Vec::new();
    let mut compared = 0;
    let mut ineligible = 0;
    for (body, kind) in [
        ("sine", BodyKind::Sine),
        ("harmonic", BodyKind::Harmonic),
        ("modal", BodyKind::Modal),
    ] {
        for (route, routing) in [
            (
                "both",
                Routing {
                    to_habitat: true,
                    to_presentation: true,
                },
            ),
            (
                "habitat",
                Routing {
                    to_habitat: true,
                    to_presentation: false,
                },
            ),
            (
                "presentation",
                Routing {
                    to_habitat: false,
                    to_presentation: true,
                },
            ),
        ] {
            for active in [false, true] {
                for sway in [false, true] {
                    let recipe = Recipe {
                        kind,
                        routing,
                        sway,
                    };
                    let name = format!(
                        "{body}-{route}-{}-{}",
                        if active { "two_tones" } else { "silent" },
                        if sway { "sway" } else { "gate" }
                    );
                    let directory = args.output.join(&name);
                    fs::create_dir(&directory).unwrap();
                    let (mixed, issue_rhythms, _) = recipe.prefix(active, true, time, issue);
                    let mut prefix: Option<[Vec<f32>; 2]> = None;
                    let mut natural: Option<[Vec<f32>; 2]> = None;
                    let mut branches = Vec::new();
                    for (class, kind, offsets) in [
                        ("wait", Class::Wait, &[6000_u64, 24000][..]),
                        ("skip", Class::Skip, &[0][..]),
                        ("continue", Class::Continue, &[0][..]),
                        ("onset_now", Class::OnsetNow, &[0][..]),
                        ("delayed_onset", Class::DelayedOnset, &[6000, 24000][..]),
                        ("release", Class::Release, &[0, 6000, 24000][..]),
                        ("gap", Class::Gap, &[0, 6000, 24000][..]),
                    ] {
                        for &offset in offsets {
                            let at = issue + offset;
                            let input = kind.input(
                                issue,
                                at,
                                Some(PERIOD),
                                FS,
                                BodyState {
                                    permits_action: Some(true),
                                    active_at_candidate: Some(active),
                                    pending_opportunity: true,
                                    due_unconsumed: true,
                                },
                            );
                            let eligible = input.is_some();
                            let mut entry = json!({"class":class, "candidate_sample":at, "eligible":eligible,
                                "eligibility_scope":"registered bodily state; no ecological or calibrated context selection",
                                "reconsider_at":input.and_then(|p| p.reconsider_at),
                                "consumes_due_opportunity":input.is_some_and(|p| p.consumes_due_opportunity),
                                "withhold_until":input.and_then(|p| p.withhold_until)});
                            if !eligible {
                                ineligible += 1;
                                branches.push(entry);
                                continue;
                            }
                            let input = input.unwrap();
                            let commands = if let Some(at) = input.excitation_at {
                                vec![recipe.note(SOURCE, 3, at)]
                            } else if let Some(at) = input.release_at {
                                vec![PhonationBatch {
                                    source_id: SOURCE,
                                    source_generation: 0,
                                    routing,
                                    cmds: vec![
                                        ToneCmd::Off {
                                            tone_id: 1,
                                            off_tick: at,
                                        },
                                        ToneCmd::Off {
                                            tone_id: 2,
                                            off_tick: at,
                                        },
                                    ],
                                    ..PhonationBatch::default()
                                }]
                            } else {
                                Vec::new()
                            };
                            let mut fork = mixed.fork_source(SOURCE);
                            let (mut direct, mut rhythms, observed_prefix) =
                                recipe.prefix(active, false, time, issue);
                            if let Some(prefix) = &prefix {
                                assert_eq!(
                                    &observed_prefix, prefix,
                                    "{name}: branch-independent private prefix"
                                );
                            } else {
                                let routed = usize::from(!routing.to_habitat);
                                save(&directory.join("prefix.f32le"), &observed_prefix[routed]);
                                prefix = Some(observed_prefix);
                            }
                            assert_eq!(rhythms.theta.phase, issue_rhythms.theta.phase);
                            let mut audio =
                                [Vec::with_capacity(HORIZON), Vec::with_capacity(HORIZON)];
                            for delta in (0..HORIZON).step_by(time.hop) {
                                let batch = if delta == 0 { commands.as_slice() } else { &[] };
                                let a = fork.render(batch, issue + delta as u64, &rhythms);
                                let b = direct.render(batch, issue + delta as u64, &rhythms);
                                assert_eq!(
                                    a.habitat, b.habitat,
                                    "{name}/{class}/{offset}: owned habitat"
                                );
                                assert_eq!(
                                    a.presentation, b.presentation,
                                    "{name}/{class}/{offset}: owned presentation"
                                );
                                audio[0].extend_from_slice(a.habitat);
                                audio[1].extend_from_slice(a.presentation);
                                for _ in 0..time.hop {
                                    rhythms.advance_in_place(1. / FS as f32);
                                }
                            }
                            let routed = usize::from(!routing.to_habitat);
                            assert!(audio.iter().flatten().all(|v| v.is_finite()));
                            if !routing.to_habitat {
                                assert!(audio[0].iter().all(|v| *v == 0.));
                            }
                            if !routing.to_presentation {
                                assert!(audio[1].iter().all(|v| *v == 0.));
                            }
                            if matches!(class, "wait" | "skip" | "continue") {
                                if let Some(natural) = &natural {
                                    assert_eq!(&audio, natural);
                                } else {
                                    natural = Some(audio.clone());
                                }
                            }
                            if class == "delayed_onset" {
                                for bus in 0..2 {
                                    assert_eq!(
                                        &audio[bus][..offset as usize],
                                        &natural.as_ref().unwrap()[bus][..offset as usize]
                                    );
                                }
                            }
                            if active && matches!(class, "release" | "gap") {
                                assert!(
                                    audio[routed][offset as usize..(offset + PERIOD) as usize]
                                        .iter()
                                        .any(|v| v.abs() > 1e-6),
                                    "release tail must survive abstention"
                                );
                                assert_ne!(
                                    audio,
                                    *natural.as_ref().unwrap(),
                                    "release must affect actual sound"
                                );
                            }
                            if !active && !matches!(class, "onset_now" | "delayed_onset") {
                                assert!(audio.iter().flatten().all(|v| *v == 0.));
                            }
                            if matches!(class, "onset_now" | "delayed_onset") {
                                assert_ne!(
                                    audio,
                                    *natural.as_ref().unwrap(),
                                    "excitation must affect actual sound"
                                );
                            }
                            let file = format!("{class}-{offset}.f32le");
                            save(&directory.join(&file), &audio[routed]);
                            if routing.to_habitat && routing.to_presentation {
                                assert_eq!(audio[0], audio[1]);
                            }
                            entry["pcm"] = json!([
                                routing.to_habitat.then_some(&file),
                                routing.to_presentation.then_some(&file)
                            ]);
                            entry["energy_per_hop"] = json!(audio.map(|samples| {
                                samples
                                    .chunks_exact(time.hop)
                                    .map(|window| {
                                        window.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()
                                            / time.hop as f64
                                    })
                                    .collect::<Vec<_>>()
                            }));
                            entry["snapshot_matches_independent_replay"] = json!(true);
                            entry["prefix_matches_independent_replay"] = json!(true);
                            compared += 1;
                            branches.push(entry);
                        }
                    }
                    let prefix = prefix.unwrap();
                    if routing.to_habitat && routing.to_presentation {
                        assert_eq!(prefix[0], prefix[1]);
                    }
                    if !routing.to_habitat {
                        assert!(prefix[0].iter().all(|v| *v == 0.));
                    }
                    if !routing.to_presentation {
                        assert!(prefix[1].iter().all(|v| *v == 0.));
                    }
                    let prefix_energy = prefix.map(|samples| {
                        samples
                            .chunks_exact(time.hop)
                            .map(|window| {
                                window.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()
                                    / time.hop as f64
                            })
                            .collect::<Vec<_>>()
                    });
                    fs::write(directory.join("profiles.json"), serde_json::to_vec_pretty(&json!({
                        "body":body, "routing":route, "active_tones":if active { 2 } else { 0 },
                        "modulator":if sway { "drone_sway" } else { "seq_gate" },
                        "prefix_pcm":[routing.to_habitat.then_some("prefix.f32le"), routing.to_presentation.then_some("prefix.f32le")],
                        "prefix_energy_per_hop":prefix_energy, "branches":branches
                    })).unwrap()).unwrap();
                    cases.push(name.clone());
                    println!("{name}: cumulative {compared} realized, {ineligible} ineligible");
                }
            }
        }
    }
    fs::write(args.output.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "schema":"i10-actual-action-profiles-v2", "sample_rate":FS, "hop_samples":time.hop,
        "requested_issue_sample":ISSUE, "prefix_samples":issue,
        "prefix_support":[0,issue], "future_support":[issue,issue+HORIZON as u64],
        "energy_grid":"mean square over each complete physical hop; both prefix and future",
        "issue_sample":issue, "horizon_samples":HORIZON, "intrinsic_period_samples":PERIOD,
        "source_id":SOURCE, "pcm_format":"mono f32 little-endian; habitat then presentation; null path means known routed silence",
        "pending_intrinsic_opportunity_sample":issue,
        "initial_rhythms":{"theta_hz":2.,"delta_hz":1.,"other_fields":"NeuralRhythms::default"},
        "modulators":{"seq_gate_duration_sec":3.,"drone_sway_phase":0.37,"drone_sway_rate":0.7},
        "continuous_drive":{"sine":0.002,"harmonic":0.002,"modal":0.005},
        "external_prefix":{"source_id":SOURCE+1,"hz":997.,"amp":0.4,"routing":"both","other_parameters":"same recipe as tone 1"},
        "body_recipe":{"amp_scale":1.,"brightness":0.4,"inharmonic":0.,"spread":0.05,"unison":1,"motion":0.1,"ratios":null},
        "adsr":[0.03,0.1,0.65,0.8], "smoothing_tau_sec":0.03,
        "prefix_tones":[{"id":1,"onset":0,"hold":48000,"hz":220.,"amp":0.1}, {"id":2,"onset":5760,"hold":60000,"hz":330.,"amp":0.07}],
        "new_onset":{"id":3,"hold":33600,"hz":220.,"amp":0.1},
        "queued_update":{"tone":1,"at":issue+2400,"hz":226.,"amp":0.09},
        "gap_has_no_new_excitation_before_frozen_endpoint":true,
        "no_extra_future_events_after_each_registered_action":true,
        "cases":cases,"realized_branches":compared,"ineligible_branches":ineligible,
        "claim":"offline class semantics, owned-state replay and counterfactual PCM only; no live table, trace update, calibration or context-effect acceptance"
    })).unwrap()).unwrap();
}
