//! Actual Voice, PCM, and production-meter closed-loop regression.

use super::*;
use std::collections::BTreeMap;
use std::sync::Mutex;

fn check_metric_population(seed: u64, count: usize, body: &str) {
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    config.analysis.hop_size = 512;
    let duration = 24.0;
    // One complete confirmation history at the end of the immutable fixture.
    let observation_start = duration - (2.0 / 0.5 + 0.06);
    let script = format!(
        "seed({seed}); let voice = {body}().amp(0.045).metric().entrainment(0.95).cycles(1).anchor().adsr(0.015, 0.08, 0.4, 0.12); \
         let group = place(voice, line(220.0, 745.0).count({count})); wait({duration}); release(group); wait(1.0);"
    );
    let path = std::env::temp_dir().join(format!(
        "conchordal-meter-{}-{seed}-{count}-{body}.rhai",
        std::process::id()
    ));
    std::fs::write(&path, script).unwrap();
    let scenario = compile_scenario_from_script(
        &path,
        &render_compile_args(path.to_str().unwrap(), None),
        &config,
    )
    .unwrap();
    std::fs::remove_file(path).unwrap();
    let observed = Arc::new(Mutex::new(BTreeMap::<u64, usize>::new()));
    let counts = observed.clone();
    let body = body.to_owned();
    let probe: OfflineBodyProbe = Box::new(move |state, start_tick, batches, pcm| {
        assert!(pcm.iter().all(|bus| bus.iter().all(|s| s.is_finite())));
        for batch in &state.phonation_batches_buf[..batches] {
            for onset in &batch.onsets {
                let time = onset.onset_tick as f64 / config.audio.sample_rate as f64;
                if time < observation_start || time >= duration {
                    continue;
                }
                // The landscape holds the previous hop's meter, exactly as the Voice reads it.
                let beat = state.prod_meter.state().beat;
                assert!(
                    beat.confidence > 0.0,
                    "an unconfirmed meter cannot certify synchrony"
                );
                let delta = state.current_landscape.rhythm.delta;
                assert!(
                    0.95 * delta.alpha > 0.0,
                    "shared-beat coupling must be active"
                );
                let offset =
                    (onset.onset_tick - start_tick) as f32 / config.audio.sample_rate as f32;
                let phase = wrap_pm_pi(beat.phase + std::f32::consts::TAU * beat.freq_hz * offset);
                let error = phase.abs() / (std::f32::consts::TAU * beat.freq_hz);
                let window = (0.2 / beat.freq_hz).min(0.06);
                assert!(
                    error <= window,
                    "{body} seed {seed}, {count} Voices: onset error {error} exceeds participation window {window}"
                );
                *counts.lock().unwrap().entry(batch.source_id).or_default() += 1;
            }
        }
    });
    let wiring = wire_runtime(
        &config,
        config.audio.sample_rate,
        "meter recurrence regression".into(),
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
            reporter: None,
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
    assert_eq!(observed.lock().unwrap().len(), count);
}

#[test]
fn two_metric_voices_confirm_and_follow_the_acoustic_beat_without_a_prior() {
    check_metric_population(1, 2, "sine");
}

#[test]
fn eight_metric_voices_follow_the_acoustic_beat_without_a_prior() {
    check_metric_population(21, 8, "sine");
}

#[test]
#[ignore = "complete 18-condition offline meter recurrence assay"]
fn all_registered_metric_populations_follow_the_confirmed_acoustic_beat() {
    for seed in [21, 1, 20260918] {
        for count in [2, 4, 8] {
            for body in ["sine", "harmonic"] {
                check_metric_population(seed, count, body);
            }
        }
    }
}
