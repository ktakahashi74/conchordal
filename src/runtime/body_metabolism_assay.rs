//! B-7 deterministic, bounded offline rendering on actual Voice and PCM paths.

use super::*;
use serde_json::json;
use std::io::Write;

#[test]
#[ignore = "B-7 explicit dynamic diagnostic, no real-time acceptance"]
fn b7_dynamic_etude_render() {
    let path = std::env::var("B7_SCENARIO").unwrap();
    let directory = std::path::PathBuf::from(std::env::var("B7_DIRECTORY").unwrap());
    let seed = std::env::var("B7_SEED")
        .map(|value| value.parse::<u64>().unwrap())
        .unwrap_or(42);
    let config = AppConfig {
        body_metabolism: Some(crate::config::BodyMetabolismConfig {
            enabled: true,
            updates_per_hop: 2,
            observation_frames: 8,
            representative_hold_sec: 1.,
        }),
        ..Default::default()
    };
    crate::life::modal::register_modal();
    let scenario = compile_scenario_from_script(
        std::path::Path::new(&path),
        &render_compile_args(&path, Some(seed)),
        &config,
    )
    .unwrap();
    assert_eq!(scenario.seed, seed);
    let mut states =
        std::io::BufWriter::new(std::fs::File::create(directory.join("states.jsonl")).unwrap());
    let mut wav = hound::WavWriter::create(
        directory.join("habitat.wav"),
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    let probe: OfflineBodyProbe = Box::new(move |state, now, _, pcm| {
        for &sample in pcm[0] {
            assert!(sample.is_finite());
            wav.write_sample(sample).unwrap();
        }
        let voices: Vec<_> = state
            .pop
            .voices
            .iter()
            .map(|voice| {
                let energy = match &voice.articulation.core {
                    crate::life::voice::AnyArticulationCore::Entrain(core) => Some(core.energy),
                    _ => None,
                };
                json!({"id":voice.id(),"generation":voice.metadata.generation,"energy":energy,
                "alive":voice.is_alive(),"freq_hz":voice.body.base_freq_hz(),
                "target_bits":voice.pitch_ctl.target_pitch_log2().to_bits()})
            })
            .collect();
        writeln!(
            states,
            "{}",
            json!({"sample":now,"voices":voices,
            "metabolic_counts":crate::life::body_metabolism::assay::take_costs(),
            "body_evaluations":state.pop.body_metabolism.as_ref().unwrap().evaluation_count()})
        )
        .unwrap();
        states.flush().unwrap();
    });
    let mut reporter =
        JsonlReporter::create(directory.join("native.jsonl").to_str().unwrap()).unwrap();
    reporter.write_meta(scenario.seed).unwrap();
    let wiring = wire_runtime(
        &config,
        48_000,
        "B-7 etude".into(),
        scenario,
        Arc::new(AtomicBool::new(false)),
        WiringOptions {
            offline_body_probe: Some(probe),
            ui_channel_capacity: 1,
            listener_forced: false,
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
            audio_counters: None,
        },
    )
    .unwrap();
    join_thread("worker", wiring.worker_handle).unwrap();
    join_thread("analysis", wiring.analysis_handle).unwrap();
    if let Some(handle) = wiring.listener_analysis_handle {
        join_thread("listener", handle).unwrap();
    }
    if let Some(rx) = wiring.report_error_rx {
        if let Ok(error) = rx.try_recv() {
            panic!("{error}");
        }
    }
}
