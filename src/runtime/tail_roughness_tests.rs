//! Exact audio, event and random-state regression for the neutral composer word.
use super::*;
use std::io::{BufRead, BufReader, BufWriter, Write};

const SCRIPT: &str = r#"
seed(1128749181177961461);
output_gain(9.0);
let p = place(sine().sustain().seek_consonance().pitch_apply_mode("rearticulate")
    .leave_self_out(true) WEIGHT .temperature(0.2).amp(0.1).adsr(0.02,0.1,0.8,1.5), at(440));
let q = place(sine().sustain().anchor().amp(0.12), at(466));
wait(3); q.amp(0.2); wait(3); release(q); wait(2); release(p); wait(2);
"#;
const CONFIG: &str = include_str!("../../tests/fixtures/articulation_persistence/scene.toml");

fn render(directory: &Path, script: &str, mut config: AppConfig, native: bool) {
    std::fs::create_dir_all(directory).unwrap();
    let script_path = directory.join("input.rhai");
    std::fs::write(&script_path, script).unwrap();
    config.render_prototype = native;
    config.validate().unwrap();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let end_request = Arc::new(AtomicBool::new(false));
    let (label, scenario, output_gain, score) = load_scenario(
        &script_path,
        Some(1128749181177961461),
        &config.audio,
        None,
        &stop_flag,
        &end_request,
    )
    .unwrap();
    let mut trace = BufWriter::new(std::fs::File::create(directory.join("rng.jsonl")).unwrap());
    let probe: OfflineBodyProbe = Box::new(move |state, start, _, _| {
        for voice in &state.pop.voices {
            writeln!(
                trace,
                "{}",
                serde_json::json!({
                    "sample": start, "id": voice.id(),
                    "pitch_rng": voice.pitch_ctl.rng_state_for_test(),
                    "articulation": format!("{:?}", voice.articulation),
                    "sway_rng": format!("{:?}", voice.phonation_engine.sway_rng),
                })
            )
            .unwrap();
        }
        trace.flush().unwrap();
    });
    let mut reporter =
        JsonlReporter::create(directory.join("native.jsonl").to_str().unwrap()).unwrap();
    reporter.write_meta(scenario.seed).unwrap();
    reporter
        .write_scene_markers(&scenario.scene_markers)
        .unwrap();
    let guard_meter = Some(Arc::new(LimiterMeter::default()));
    let (wav_tx, wav_rx) = bounded::<Arc<[f32]>>(16);
    let wav = crate::audio::writer::WavOutput::run(
        wav_rx,
        directory.join("audio.wav").to_string_lossy().into_owned(),
        config.audio.sample_rate,
        resolve_limiter_mode(&config),
        guard_meter.clone(),
        output_gain,
    );
    let wiring = wire_runtime(
        &config,
        config.audio.sample_rate,
        label,
        scenario,
        stop_flag,
        WiringOptions {
            score,
            end_request,
            duration_sec: None,
            offline_body_probe: Some(probe),
            raw_t2_delay: std::time::Duration::ZERO,
            ui_tx: None,
            listener_forced: true,
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: Some(wav_tx),
            reporter: Some(reporter),
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter,
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
    join_thread("wav", wav).unwrap().unwrap();
    join_thread("analysis", wiring.analysis_handle).unwrap();
    if let Some(listener) = wiring.listener_analysis_handle {
        join_thread("listener", listener).unwrap();
    }
}

#[test]
fn tail_roughness_zero_preserves_whole_wav_events_and_rng() {
    let root = std::env::temp_dir().join(format!("conchordal-tail-zero-{}", std::process::id()));
    let _cleanup = super::test_output_directory::TestOutputDirectory::new(root.clone());
    for native in [false, true] {
        let a = root.join(format!("{native}-default"));
        let b = root.join(format!("{native}-zero"));
        render(
            &a,
            &SCRIPT.replace("WEIGHT", ""),
            toml::from_str(CONFIG).unwrap(),
            native,
        );
        render(
            &b,
            &SCRIPT.replace("WEIGHT", ".tail_roughness_aversion(0)"),
            toml::from_str(CONFIG).unwrap(),
            native,
        );
        for file in ["audio.wav", "rng.jsonl"] {
            assert_eq!(
                std::fs::read(a.join(file)).unwrap(),
                std::fs::read(b.join(file)).unwrap(),
                "{file}, native={native}"
            );
        }
        let rows = |dir: &Path| {
            BufReader::new(std::fs::File::open(dir.join("native.jsonl")).unwrap())
                .lines()
                .map(|l| serde_json::from_str::<serde_json::Value>(&l.unwrap()).unwrap())
                .filter(|r| r["type"] != "hop_timing")
                .collect::<Vec<_>>()
        };
        let left = rows(&a);
        assert!(!left.is_empty());
        assert_eq!(left, rows(&b), "native={native}");
    }
}

#[test]
#[ignore = "cross-revision acquisition; explicit output directory required"]
fn tail_roughness_reference_acquisition() {
    let output = std::env::var("VR1_REFERENCE_OUTPUT").expect("VR1_REFERENCE_OUTPUT");
    let script = std::env::var("VR1_REFERENCE_SCRIPT").map_or_else(
        |_| SCRIPT.replace("WEIGHT", ""),
        |path| std::fs::read_to_string(path).unwrap(),
    );
    let config = std::env::var("VR1_REFERENCE_CONFIG").map_or_else(
        |_| CONFIG.to_string(),
        |path| std::fs::read_to_string(path).unwrap(),
    );
    let config: AppConfig = toml::from_str(&config).unwrap();
    let native = config.render_prototype;
    render(Path::new(&output), &script, config, native);
}
