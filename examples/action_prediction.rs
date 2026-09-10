//! Offline owned-state forecasts and matched action/wait acoustic branches.
//! Run with a fresh output directory; no instrument audio-export path is added.

use clap::Parser;
use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::Timebase;
use conchordal::life::phonation_engine::{OnsetKick, ToneCmd, ToneUpdate};
use conchordal::life::schedule_renderer::ScheduleRenderer;
use conchordal::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, ToneAdsr};
use conchordal::life::voice::{PhonationBatch, ToneSpec};
use conchordal::scenario::control::Routing;
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser)]
struct Args {
    output: PathBuf,
    #[arg(long, num_args = 1.., default_values = ["184757", "184771", "184777"])]
    seeds: Vec<u64>,
    #[arg(long, default_value_t = 2)]
    future_hops: usize,
    #[arg(long, default_value_t = 15)]
    warmup_hops: u64,
}

#[allow(clippy::too_many_arguments)]
fn note(
    source_id: u64,
    tone_id: u64,
    onset: u64,
    fs: u32,
    frequency: f32,
    kind: BodyKind,
    habitat: bool,
    amp: f32,
) -> PhonationBatch {
    PhonationBatch {
        source_id,
        routing: Routing {
            to_presentation: source_id < 1_000_000,
            to_habitat: habitat,
        },
        cmds: vec![
            ToneCmd::On {
                tone_id,
                kick: OnsetKick { strength: 1.0 },
            },
            ToneCmd::Update {
                tone_id,
                at_tick: Some(onset),
                update: ToneUpdate {
                    target_freq_hz: None,
                    target_amp: None,
                    continuous_drive: Some(if kind == BodyKind::Modal {
                        0.005
                    } else {
                        0.002
                    }),
                },
            },
        ],
        tones: vec![ToneSpec {
            tone_id,
            onset,
            hold_ticks: Some((fs as f64 * 0.155) as u64),
            freq_hz: frequency,
            amp,
            smoothing_tau_sec: 0.03,
            body: BodySnapshot {
                kind,
                amp_scale: 1.0,
                brightness: 0.4,
                inharmonic: 0.0,
                spread: 0.05,
                unison: 1,
                motion: 0.1,
                ratios: Some(Arc::from(if kind == BodyKind::Modal {
                    [1.0, 1.37, 2.09, 3.7]
                } else {
                    [1.0, 2.0, 3.02, 4.03]
                })),
            },
            render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 0.3 },
            adsr: Some(ToneAdsr {
                attack_sec: 0.01,
                decay_sec: 0.12,
                sustain_level: 0.5,
                release_sec: 0.035,
            }),
        }],
        onsets: Vec::new(),
    }
}

fn write_audio(path: &Path, audio: &[f32]) {
    let mut file = fs::File::create(path).unwrap();
    for sample in audio {
        file.write_all(&sample.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args = Args::parse();
    assert!(args.seeds.iter().all(|seed| *seed < 1_000_000));
    assert!(args.future_hops >= 2);
    assert!(args.warmup_hops >= 3);
    let output = &args.output;
    fs::create_dir(output).expect("output directory must not already exist");
    let mut manifest = Vec::new();
    for fs in [24_000_u32, 48_000] {
        let hop = fs as usize / 100;
        let tb = Timebase { fs: fs as f32, hop };
        let issue = args.warmup_hops * hop as u64;
        for &seed in &args.seeds {
            for (body, kind) in [
                ("sine", BodyKind::Sine),
                ("harmonic", BodyKind::Harmonic),
                ("modal", BodyKind::Modal),
            ] {
                for habitat in [true, false] {
                    for (environment, other_hold_ticks, unseen) in [
                        ("held", issue + 15 * hop as u64, false),
                        ("release", issue + hop as u64 / 2, false),
                        ("release_and_onset", issue + hop as u64 / 2, true),
                    ] {
                        let route = if habitat { "habitat" } else { "presentation" };
                        let name = format!("{body}-{route}-{environment}-{fs}-{seed}");
                        let directory = output.join(&name);
                        fs::create_dir(&directory).unwrap();
                        let f = 200.0 + (seed % 100) as f32;
                        let mut actor = note(seed, 1, 0, fs, f, kind, habitat, 0.07);
                        actor.tones[0].hold_ticks = Some(issue + hop as u64 / 2);
                        actor.tones[0].render_modulator = RenderModulatorSpec::SeqGate {
                            duration_sec: (issue + 15 * hop as u64) as f32 / fs as f32,
                        };
                        let mut other = note(
                            1_000_000 + seed,
                            1,
                            0,
                            fs,
                            f * 1.31,
                            BodyKind::Harmonic,
                            true,
                            0.045,
                        );
                        other.tones[0].hold_ticks = Some(other_hold_ticks);
                        other.tones[0].render_modulator = actor.tones[0].render_modulator.clone();
                        let mut waiting = ScheduleRenderer::new(tb);
                        let mut sounding = ScheduleRenderer::new(tb);
                        let mut known_own = ScheduleRenderer::new(tb);
                        let mut past_mix = Vec::new();
                        let mut past_own = Vec::new();
                        let mut rhythms = NeuralRhythms::default();
                        for h in 0..args.warmup_hops {
                            let now = h * hop as u64;
                            let own = if h == 0 {
                                vec![actor.clone()]
                            } else if h == args.warmup_hops - 2 {
                                vec![PhonationBatch {
                                    source_id: seed,
                                    cmds: vec![ToneCmd::Update {
                                        tone_id: 1,
                                        at_tick: Some(issue + hop as u64 / 2),
                                        update: ToneUpdate {
                                            target_freq_hz: Some(f * 1.03),
                                            target_amp: Some(0.065),
                                            continuous_drive: None,
                                        },
                                    }],
                                    ..PhonationBatch::default()
                                }]
                            } else {
                                Vec::new()
                            };
                            let mut all = own.clone();
                            if h == 0 {
                                all.push(other.clone());
                            }
                            let a = waiting.render(&all, now, &rhythms).habitat;
                            let b = sounding.render(&all, now, &rhythms).habitat;
                            assert_eq!(
                                a, b,
                                "both realized branches must have identical histories"
                            );
                            past_mix.extend_from_slice(a);
                            past_own
                                .extend_from_slice(known_own.render(&own, now, &rhythms).habitat);
                            for _ in 0..hop {
                                rhythms.advance_in_place(1.0 / fs as f32);
                            }
                        }
                        let mut own_wait = waiting.fork_source(seed);
                        let mut own_sound = waiting.fork_source(seed);
                        let mut unknown = waiting.fork_source(u64::MAX);
                        let action = note(
                            seed,
                            2,
                            issue + hop as u64 / 3,
                            fs,
                            f * 1.07,
                            kind,
                            habitat,
                            0.10,
                        );
                        let new_other = note(
                            1_000_000 + seed,
                            2,
                            issue + hop as u64 / 2,
                            fs,
                            f * 0.73,
                            BodyKind::Sine,
                            true,
                            0.12,
                        );
                        let mut predicted_wait = Vec::new();
                        let mut predicted_sound = Vec::new();
                        let mut actual_wait = Vec::new();
                        let mut actual_sound = Vec::new();
                        let mut actual_own_wait = Vec::new();
                        let mut own_presentation_wait = Vec::new();
                        // Only owned snapshots and the selected action enter these forecasts.
                        // Other voices' future commands are used only by the realized branches.
                        for h in 0..args.future_hops {
                            let now = issue + (h * hop) as u64;
                            let commanded = if h == 0 {
                                vec![action.clone()]
                            } else {
                                Vec::new()
                            };
                            let a = own_wait.render(&[], now, &rhythms);
                            predicted_wait.extend_from_slice(a.habitat);
                            own_presentation_wait.extend_from_slice(a.presentation);
                            predicted_sound.extend_from_slice(
                                own_sound.render(&commanded, now, &rhythms).habitat,
                            );
                            actual_own_wait
                                .extend_from_slice(known_own.render(&[], now, &rhythms).habitat);
                            assert!(
                                unknown
                                    .render(&[], now, &rhythms)
                                    .habitat
                                    .iter()
                                    .all(|x| *x == 0.0)
                            );
                            let environment = if h == 0 && unseen {
                                vec![new_other.clone()]
                            } else {
                                Vec::new()
                            };
                            actual_wait.extend_from_slice(
                                waiting.render(&environment, now, &rhythms).habitat,
                            );
                            let mut acted = commanded;
                            acted.extend(environment);
                            actual_sound
                                .extend_from_slice(sounding.render(&acted, now, &rhythms).habitat);
                            for _ in 0..hop {
                                rhythms.advance_in_place(1.0 / fs as f32);
                            }
                        }
                        assert_eq!(
                            predicted_wait, actual_own_wait,
                            "owned continuation must preserve renderer state"
                        );
                        assert!(
                            own_presentation_wait.iter().any(|x| x.abs() > 1e-6),
                            "waiting retains earlier ringing"
                        );
                        if !habitat {
                            assert_eq!(actual_wait, actual_sound);
                        }
                        for (key, audio) in [
                            ("past_mix", &past_mix),
                            ("past_own", &past_own),
                            ("own_wait", &predicted_wait),
                            ("own_sound", &predicted_sound),
                            ("truth_wait", &actual_wait),
                            ("truth_sound", &actual_sound),
                            ("own_presentation_wait", &own_presentation_wait),
                        ] {
                            write_audio(&directory.join(format!("{key}.f32le")), audio);
                        }
                        manifest.push(json!({"name":name,"fs":fs,"seed":seed,"body":body,
                            "route":route,"external_condition":environment,"issue_sample":issue,
                            "external_release_sample":other.tones[0].hold_ticks,
                            "onset_sample":issue+hop as u64/3,"forecast_samples":args.future_hops*hop,
                            "known_own_state_matches":true,"own_snapshot_excludes_other_voices":true,
                            "future_external_state_supplied_to_forecast":false}));
                        fs::write(
                            output.join("inputs.json"),
                            serde_json::to_string_pretty(&manifest).unwrap(),
                        )
                        .unwrap();
                        println!("{name}");
                    }
                }
            }
        }
    }
}
