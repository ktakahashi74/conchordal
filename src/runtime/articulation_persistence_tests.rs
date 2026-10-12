//! Offline wiring checks; the extra policy observations exist only in test builds.

use super::*;
use crate::life::phonation_engine::PhonationClock;
use crate::life::shape_policy::Shape;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Mutex;

const ONE: &str = include_str!("../../tests/fixtures/articulation_persistence/one.rhai");
const PEER: &str = include_str!("../../tests/fixtures/articulation_persistence/peer.rhai");
const CONFIG: &str = include_str!("../../tests/fixtures/articulation_persistence/scene.toml");

#[derive(Default)]
struct Observation {
    onsets: usize,
    connect_samples: u64,
    through: u64,
    connect: bool,
}

fn render_case(
    directory: &Path,
    script: &str,
    native: bool,
    fixed_base: bool,
    measured_sec: f32,
    tie: Option<(f32, bool)>,
) {
    std::fs::create_dir_all(directory).unwrap();
    let script_path = directory.join("input.rhai");
    std::fs::write(&script_path, script).unwrap();
    let mut config: AppConfig = toml::from_str(CONFIG).unwrap();
    config.render_prototype = native;
    config.validate().unwrap();
    let mut scenario = compile_scenario_from_script(
        &script_path,
        &render_compile_args(script_path.to_str().unwrap(), None),
        &config,
    )
    .unwrap();
    for action in scenario
        .events
        .iter_mut()
        .flat_map(|event| &mut event.actions)
    {
        if let crate::scenario::Action::Spawn { spec, .. } = action
            && (spec.control.supports_articulation_persistence()
                || (tie.is_some() && spec.control.supports_tie()))
        {
            if let Some((p, force)) = tie {
                spec.control.phonation.tie = p;
                spec.control.phonation.force_tie_attack = force;
            }
            spec.control.phonation.force_articulation_base = fixed_base;
            spec.control.phonation.observe_articulation = true;
        }
    }
    let observed = Arc::new(Mutex::new(BTreeMap::<u64, Observation>::new()));
    let counts = observed.clone();
    let end_sample = (measured_sec * config.audio.sample_rate as f32) as u64;
    let probe: OfflineBodyProbe = Box::new(move |state, start, batches, pcm| {
        assert!(pcm.iter().all(|bus| bus.iter().all(|s| s.is_finite())));
        let until = (start + pcm[0].len() as u64).min(end_sample);
        let mut observed = counts.lock().unwrap();
        for voice in &mut state.pop.voices {
            if !voice.effective_control.supports_articulation_persistence()
                && !(tie.is_some() && voice.effective_control.supports_tie())
            {
                continue;
            }
            let item = observed.entry(voice.id()).or_default();
            if let PhonationClock::Participation(clock) = &mut voice.phonation_engine.clock
                && let Some(shape) = clock.shape.as_mut()
                && let Some(changes) = shape.observations.as_mut()
            {
                for (at, after) in changes.drain(..) {
                    if at > until {
                        continue;
                    }
                    assert!(at >= item.through);
                    if item.connect {
                        item.connect_samples += at - item.through;
                    }
                    item.through = at;
                    item.connect = after == Shape::Connect;
                }
            }
            if until > item.through {
                if item.connect {
                    item.connect_samples += until - item.through;
                }
                item.through = until;
            }
        }
        for batch in &state.phonation_batches_buf[..batches] {
            if let Some(item) = observed.get_mut(&batch.source_id) {
                item.onsets += batch
                    .onsets
                    .iter()
                    .filter(|on| on.onset_tick < end_sample)
                    .count();
            }
        }
    });
    let mut reporter =
        JsonlReporter::create(directory.join("native.jsonl").to_str().unwrap()).unwrap();
    reporter.write_meta(scenario.seed).unwrap();
    reporter
        .write_scene_markers(&scenario.scene_markers)
        .unwrap();
    let guard_mode = resolve_limiter_mode(&config);
    let guard_meter = Some(Arc::new(LimiterMeter::default()));
    let (wav_tx, wav_rx) = bounded::<Arc<[f32]>>(16);
    let wav_handle = crate::audio::writer::WavOutput::run(
        wav_rx,
        directory.join("audio.wav").to_string_lossy().into_owned(),
        config.audio.sample_rate,
        guard_mode,
        guard_meter.clone(),
    );
    let wiring = wire_runtime(
        &config,
        config.audio.sample_rate,
        "articulation regression".into(),
        scenario,
        Arc::new(AtomicBool::new(false)),
        WiringOptions {
            offline_body_probe: Some(probe),
            ui_channel_capacity: 1,
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
            audio_counters: None,
        },
    )
    .unwrap();
    join_thread("worker", wiring.worker_handle).unwrap();
    join_thread("wav", wav_handle).unwrap();
    join_thread("analysis", wiring.analysis_handle).unwrap();
    if let Some(handle) = wiring.listener_analysis_handle {
        join_thread("listener", handle).unwrap();
    }
    if let Some(rx) = wiring.report_error_rx {
        assert!(rx.try_recv().is_err());
    }
    let observations = observed.lock().unwrap();
    assert!(!observations.is_empty());
    assert!(
        observations
            .values()
            .all(|v| v.onsets > 0 && v.through == end_sample)
    );
    let rows: Vec<_> = observations.iter().map(|(id,v)|serde_json::json!({"voice":id,"onsets":v.onsets,"connect_samples":v.connect_samples,"samples":end_sample,"connect_fraction":v.connect_samples as f64/end_sample as f64})).collect();
    std::fs::write(
        directory.join("observations.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
}

fn output_directory(label: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "conchordal-articulation-{label}-{}-{stamp}",
        std::process::id()
    ))
}

fn assert_outputs_equal(a: &Path, b: &Path) {
    assert_eq!(
        std::fs::read(a.join("audio.wav")).unwrap(),
        std::fs::read(b.join("audio.wav")).unwrap(),
        "WAV mismatch: {a:?} / {b:?}"
    );
    let rows = |path: &Path| {
        BufReader::new(std::fs::File::open(path.join("native.jsonl")).unwrap())
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(&line.unwrap()).unwrap())
            .filter(|row| row["type"] != "hop_timing")
    };
    let mut left = rows(a);
    let mut right = rows(b);
    let mut count = 0;
    loop {
        match (left.next(), right.next()) {
            (None, None) => break,
            (Some(a), Some(b)) => assert_eq!(a, b, "native row {count}"),
            _ => panic!("native report lengths differ at row {count}"),
        }
        count += 1;
    }
    assert!(count > 0);
    println!(
        "articulation identical WAV and {count} native rows: {} / {}",
        a.display(),
        b.display()
    );
}

#[test]
fn articulation_persistence_fixed_base_one_full_audio_and_release_report_match() {
    let directory = output_directory("fixed-one");
    render_case(&directory.join("direct"), ONE, false, false, 96., None);
    render_case(&directory.join("fixed"), ONE, false, true, 96., None);
    assert_outputs_equal(&directory.join("direct"), &directory.join("fixed"));
}

#[test]
fn articulation_persistence_fixed_base_peer_full_audio_and_release_report_match() {
    let directory = output_directory("fixed-peer");
    render_case(&directory.join("direct"), PEER, false, false, 96., None);
    render_case(&directory.join("fixed"), PEER, false, true, 96., None);
    assert_outputs_equal(&directory.join("direct"), &directory.join("fixed"));
}

#[test]
fn articulation_persistence_all_bodies_both_renderers_flow_repeat() {
    let directory = output_directory("bodies");
    for native in [false, true] {
        for body in ["sine", "harmonic", "modal"] {
            let script = ONE
                .replace("harmonic()", &format!("{body}()"))
                .replace("entrained()", "flow()")
                .replace(
                    "rearticulate_above(50)",
                    "rearticulate_above(50).articulation_persistence(0.75)",
                )
                .replace("wait(96.0)", "wait(8.0)")
                .replace("wait(2.0)", "wait(0.5)");
            let a = directory.join(format!("{body}-{native}-a"));
            let b = directory.join(format!("{body}-{native}-b"));
            render_case(&a, &script, native, false, 8., None);
            render_case(&b, &script, native, false, 8., None);
            assert_outputs_equal(&a, &b);
        }
    }
}

#[test]
#[ignore = "full offline articulation audition acquisition"]
fn articulation_persistence_audition_materials() {
    let directory = output_directory("audition");
    let changed = |source: &str, p: &str| {
        source.replace(
            "rearticulate_above(50)",
            &format!("rearticulate_above(50).articulation_persistence({p})"),
        )
    };
    let cases = [
        ("one-default", ONE.to_owned(), false),
        ("one-p05", changed(ONE, "0.5"), false),
        ("one-p075", changed(ONE, "0.75"), false),
        ("one-p0875", changed(ONE, "0.875"), false),
        ("peer-p075", changed(PEER, "0.75"), false),
        (
            "flow-p075",
            changed(ONE, "0.75").replace("entrained()", "flow()"),
            false,
        ),
        (
            "modal-p075",
            changed(ONE, "0.75").replace("harmonic()", "modal()"),
            true,
        ),
        (
            "rest-p075",
            changed(ONE, "0.75").replace(
                "rearticulate_above(50)",
                "cycles(2).rest(1.0).rearticulate_above(50)",
            ),
            false,
        ),
    ];
    for (name, script, native) in cases {
        render_case(&directory.join(name), &script, native, false, 96., None);
    }
    println!("articulation audition directory: {}", directory.display());
}

#[test]
fn tie_forced_attack_and_explicit_zero_match_one_and_peer_complete_output() {
    let directory = output_directory("tie-fixed");
    for (label, script) in [("one", ONE), ("peer", PEER)] {
        let base = directory.join(format!("{label}-base"));
        let zero = directory.join(format!("{label}-zero"));
        let forced = directory.join(format!("{label}-forced"));
        render_case(&base, script, false, false, 96., None);
        render_case(&zero, script, false, false, 96., Some((0., false)));
        render_case(&forced, script, false, false, 96., Some((0., true)));
        assert_outputs_equal(&base, &zero);
        assert_outputs_equal(&base, &forced);
    }
}

#[test]
fn tie_all_bodies_renderers_flow_rest_persistence_and_repeat() {
    let directory = output_directory("tie-bodies");
    for native in [false, true] {
        for body in ["sine", "harmonic", "modal"] {
            for extra in ["", ".cycles(2).rest(1).articulation_persistence(0.75)"] {
                let script = ONE
                    .replace("harmonic()", &format!("{body}()"))
                    .replace("entrained()", "flow().intrinsic_rate(2)")
                    .replace(
                        "rearticulate_above(50)",
                        &format!("rearticulate_above(50){extra}"),
                    )
                    .replace("wait(96.0)", "wait(8.0)")
                    .replace("wait(2.0)", "wait(0.5)");
                let suffix = if extra.is_empty() { "base" } else { "composed" };
                let a = directory.join(format!("{body}-{native}-{suffix}-a"));
                let b = directory.join(format!("{body}-{native}-{suffix}-b"));
                render_case(&a, &script, native, false, 8., Some((0.5, false)));
                render_case(&b, &script, native, false, 8., Some((0.5, false)));
                assert_outputs_equal(&a, &b);
            }
        }
    }
}
