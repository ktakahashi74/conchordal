//! Offline research capture through the ordered native renderer and its proposal replay.
use super::*;
use serde_json::json;
#[cfg(feature = "research")]
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[path = "room_oracle.rs"]
pub(crate) mod oracle;
#[path = "room_shadow.rs"]
mod room_shadow;
pub(crate) use room_shadow::{observe_dry, room_fields};

#[cfg(feature = "research")]
fn hash_samples(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    for x in samples {
        hash.update(x.to_le_bytes());
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(feature = "research")]
fn meter(network: &MeterNetwork) -> serde_json::Value {
    let beat = network.state().beat;
    json!([
        beat.freq_hz,
        beat.phase,
        beat.confidence,
        network.felt_beat().filter(|s| !s.unknown).map(|s| s.p_beat)
    ])
}

#[test]
#[ignore = "explicit offline paired room performance acquisition"]
#[cfg(feature = "research")]
fn acquire() {
    let var = |key| std::env::var(key).unwrap();
    let script = PathBuf::from(var("ROOM_SCORE"));
    let out = PathBuf::from(var("ROOM_OUTPUT"));
    let seed: u64 = var("ROOM_SEED").parse().unwrap();
    let mut config = AppConfig::load_or_default(Some(&var("ROOM_CONFIG"))).unwrap();
    crate::life::modal::register_modal();
    let stop = Arc::new(AtomicBool::new(false));
    let end = Arc::new(AtomicBool::new(false));
    let (label, scenario, score) = load_scenario(&script, Some(seed), None, &stop, &end).unwrap();
    resolve_effective_room(&mut config, &scenario, &script.to_string_lossy()).unwrap();
    let paired = std::env::var_os("ROOM_RESCORE").is_some();
    let report = if paired || oracle::active() {
        PathBuf::from("/dev/null")
    } else {
        out.join("native.jsonl")
    };
    let mut reporter = JsonlReporter::create(report.to_str().unwrap()).unwrap();
    reporter.write_meta(scenario.seed).unwrap();
    reporter
        .write_scene_markers(&scenario.scene_markers)
        .unwrap();
    let mut hops = BufWriter::new(File::create(out.join("hops.jsonl")).unwrap());
    let mut choices = BufWriter::new(File::create(out.join("actions.jsonl")).unwrap());
    let fixed = std::env::var_os("ROOM_FIXED").is_some();
    let mut stems = std::env::var_os("ROOM_STEMS")
        .map(|_| BufWriter::new(File::create(out.join("buses.f32le")).unwrap()));
    let probe: OfflineBodyProbe = Box::new(move |s, tick, n, pcm| {
        if oracle::mode() == "census" {
            let live = s
                .pop
                .voices
                .iter()
                .map(|v| (v.id(), v.metadata.generation))
                .collect();
            let metabolism = s
                .pop
                .body_metabolism
                .as_ref()
                .map_or(&[][..], |b| b.pcm.as_slice());
            let counts = s.schedule_renderer.oracle_readers(&live, metabolism, tick);
            writeln!(
                hops,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                tick, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
            )
            .unwrap();
            return;
        }
        if oracle::active() {
            for b in &s.phonation_batches_buf[..n] {
                for t in &b.tones {
                    if t.amp > 0. {
                        writeln!(
                            choices,
                            "on\t{}\t{}\t{}\t{}\t{}",
                            t.onset,
                            b.source_id,
                            b.source_generation,
                            t.tone_id,
                            f64::from(t.freq_hz)
                        )
                        .unwrap();
                    }
                }
                for cmd in &b.cmds {
                    if let crate::life::phonation_engine::ToneCmd::Off { tone_id, off_tick } = cmd {
                        writeln!(
                            choices,
                            "off\t{off_tick}\t{}\t{}\t{tone_id}\t0",
                            b.source_id, b.source_generation
                        )
                        .unwrap();
                    }
                }
            }
            return;
        }
        if paired {
            return;
        }
        let land = &s.current_landscape;
        let density = &land.subjective_intensity;
        let h = s.listener.hab_ecology.state();
        let (i, &peak) = density
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let drive = |x: f32| {
            let raw = if peak > 0. { x.max(0.) / peak } else { 0. };
            raw / (raw + s.lparams.habituation.ref_drive)
        };
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let max_h = h.iter().copied().fold(0.0, f32::max);
        let voices: Vec<_> = s
            .pop
            .voices
            .iter()
            .map(|v| {
                assert_eq!(v.effective_control.pitch.tail_roughness_aversion, 0.);
                assert_eq!(v.effective_control.phonation.syncopate, 0.);
                (
                    v.id(),
                    v.metadata.generation,
                    v.target_pitch_log2(),
                    fixed.then(|| v.pitch_ctl.rng_state_for_test()),
                )
            })
            .collect();
        let sources: Vec<_> = s
            .schedule_renderer
            .loo_pcm
            .iter()
            .map(|src| {
                let peak_ext = pcm[0]
                    .iter()
                    .zip(&src.habitat)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                (
                    format!("{:?}", src.kind),
                    src.id,
                    src.generation,
                    peak_ext,
                    fixed.then(|| hash_samples(&src.habitat)),
                )
            })
            .collect();
        let flux = s.dorsal.last_metrics().flux;
        writeln!(hops,"{}",json!({
            "tick":tick,"frame":s.listener.last_analysis_frame,
            "density":[peak,i,drive(peak),density.iter().map(|&x|drive(x)).sum::<f32>() / density.len() as f32],
            "h":[h[i],max_h,mean(h)],"prod":meter(&s.temporal.prod_meter),
            "audio":s.temporal.flora_meter.as_ref().map(meter),"flux":flux,
            "voices":voices,"sources":sources,"end":s.lifecycle.scenario_end_tick,
            "hash":fixed.then(|| [hash_samples(pcm[0]),hash_samples(h),hash_samples(density),
                hash_samples(&land.consonance_field_score_eff)]),
            "flora_rng":fixed.then(|| s.pop.flora.iter().map(|f|(f.id,f.rng_state_for_test())).collect::<Vec<_>>())
        })).unwrap();
        for b in &s.phonation_batches_buf[..n] {
            if !b.cmds.is_empty() || !b.tones.is_empty() || !b.onsets.is_empty() {
                writeln!(
                    choices,
                    "{}",
                    json!({"tick":tick,"id":b.source_id,"generation":b.source_generation,
                    "route":b.routing,"cmds":b.cmds,"tones":b.tones,"onsets":b.onsets})
                )
                .unwrap();
            }
        }
        if let Some(w) = &mut stems {
            for (&hab, &pres) in pcm[0].iter().zip(pcm[1]) {
                w.write_all(&hab.to_le_bytes()).unwrap();
                w.write_all(&pres.to_le_bytes()).unwrap();
            }
        }
    });
    let wiring = wire_runtime(
        &config,
        config.audio.sample_rate,
        label,
        scenario,
        stop,
        WiringOptions {
            score,
            end_request: end,
            duration_sec: None,
            offline_body_probe: Some(probe),
            raw_t2_delay: Duration::ZERO,
            ui_tx: None,
            listener_forced: true,
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: None,
            reporter: Some(reporter),
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter: None,
            underrun_frames: None,
            reserve_runtime_ids_through: 0,
            profile: None,
            telemetry: false,
            audio_counters: None,
        },
    )
    .unwrap();
    join_thread("worker", wiring.worker_handle)
        .unwrap()
        .unwrap();
    join_thread("analysis", wiring.analysis_handle).unwrap();
    oracle::save_rejections(&out);
    if let Some(listener) = wiring.listener_analysis_handle {
        join_thread("listener", listener).unwrap();
    }
}

// Run in separate processes with the same binary/seed to compare the full
// stereo WAV, decision trace and report against the DR-27 diagnostic capacity.
#[test]
#[ignore = "explicit finite strict room capacity comparison"]
fn render_reservation() {
    let out = PathBuf::from(std::env::var_os("ROOM_OUTPUT").unwrap());
    let score = std::env::var("ROOM_SCORE").unwrap();
    let mut config =
        AppConfig::load_or_default(Some(&std::env::var("ROOM_CONFIG").unwrap())).unwrap();
    let seed = std::env::var("ROOM_SEED").unwrap().parse().unwrap();
    if std::env::var_os("ROOM_REFERENCE_64").is_some() {
        wiring::ROOM_OWNER_OVERRIDE.set(Some(64));
    }
    let stop = Arc::new(AtomicBool::new(false));
    let end = Arc::new(AtomicBool::new(false));
    crate::life::modal::register_modal();
    let (label, mut scenario, score) =
        load_scenario(Path::new(&score), Some(seed), None, &stop, &end).unwrap();
    let default_site = std::env::var_os("ROOM_DEFAULT_SITE").is_some();
    if default_site {
        crate::scenario::Site::register(&mut scenario.sites, Some(crate::scenario::Site::DEFAULT));
    }
    let resolved = resolve_effective_room(&mut config, &scenario, &label).unwrap();
    eprintln!(
        "output channels={} gain={}",
        config.output_channels(),
        resolved.output_gain
    );
    let time = config.analysis.hop_size;
    let kernel = crate::core::room::RoomKernel::new(&crate::core::room::york_response(), time);
    eprintln!("bytes per stream = {}", kernel.stream_bytes());
    let mut actions = BufWriter::new(File::create(out.join("actions.jsonl")).unwrap());
    let probe: OfflineBodyProbe = Box::new(move |s, tick, n, _| {
        let voices: Vec<_> = s
            .pop
            .voices
            .iter()
            .map(|v| {
                (
                    v.id(),
                    v.metadata.generation,
                    v.is_alive(),
                    v.target_pitch_log2(),
                    v.pitch_ctl.rng_state_for_test(),
                )
            })
            .collect();
        let notes: Vec<_> = s.phonation_batches_buf[..n]
            .iter()
            .map(|b| json!([b.source_id, b.source_generation, b.cmds, b.tones, b.onsets]))
            .collect();
        writeln!(actions, "{}", json!([tick, voices, notes])).unwrap();
    });
    let mut reporter = JsonlReporter::create(out.join("report.jsonl").to_str().unwrap()).unwrap();
    reporter.write_meta(seed).unwrap();
    reporter
        .write_scene_markers(&scenario.scene_markers)
        .unwrap();
    let (wav, pcm) = bounded(8);
    let wiring = wire_runtime(
        &config,
        config.audio.sample_rate,
        label,
        scenario,
        stop,
        WiringOptions {
            score,
            end_request: end,
            duration_sec: None,
            offline_body_probe: Some(probe),
            #[cfg(test)]
            raw_t2_delay: Duration::ZERO,
            ui_tx: None,
            listener_forced: true,
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: Some(wav),
            reporter: Some(reporter),
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter: None,
            underrun_frames: None,
            reserve_runtime_ids_through: 0,
            profile: None,
            telemetry: false,
            audio_counters: None,
        },
    )
    .unwrap();
    let mut writer = hound::WavWriter::create(
        out.join("audio.wav"),
        hound::WavSpec {
            channels: config.output_channels() as u16,
            sample_rate: config.audio.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    let channels = config.output_channels();
    let mut guards = default_site.then(|| {
        [0f32, 9.].map(|db| {
            let gain = 10f32.powf(db / 20.);
            let meter = Arc::new(crate::audio::limiter::LimiterMeter::default());
            let limiter = crate::audio::limiter::Limiter::new(
                wiring::resolve_limiter_mode(&config),
                config.audio.sample_rate,
                channels,
            )
            .with_input_gain(gain)
            .with_meter(meter.clone());
            (gain, limiter, meter, [0f32; 2], [0f32; 2], None, None)
        })
    });
    let mut frame = 0;
    let mut post = Vec::new();
    for chunk in pcm {
        for &sample in chunk.iter() {
            writer.write_sample(sample).unwrap();
        }
        if let Some(guards) = &mut guards {
            for (gain, limiter, _, pre_peak, post_peak, first, last) in guards {
                post.clear();
                post.extend_from_slice(&chunk);
                limiter.process_interleaved(&mut post, channels);
                for (i, (&raw, &y)) in chunk.iter().zip(&post).enumerate() {
                    let ch = i % channels;
                    let x = raw * *gain;
                    pre_peak[ch] = pre_peak[ch].max(x.abs());
                    post_peak[ch] = post_peak[ch].max(y.abs());
                    if (x - y).abs() > 1e-7 {
                        first.get_or_insert(frame + i / channels);
                        *last = Some(frame + i / channels);
                    }
                }
            }
        }
        frame += chunk.len() / channels;
    }
    if let Some(guards) = guards {
        for (gain, _, meter, pre, post, first, last) in guards {
            let stats = meter.take_snapshot().unwrap_or_default();
            eprintln!(
                "SCORE gain={gain} frames={frame} pre={pre:?} post={post:?} over={} reduction_db={} first={first:?} last={last:?}",
                stats.num_over, stats.max_reduction_db
            );
        }
    }
    writer.finalize().unwrap();
    join_runtime(
        Some(wiring.worker_handle),
        [
            ("analysis", Some(wiring.analysis_handle)),
            ("listener", wiring.listener_analysis_handle),
        ],
    )
    .unwrap();
    wiring::ROOM_OWNER_OVERRIDE.set(None);
}
