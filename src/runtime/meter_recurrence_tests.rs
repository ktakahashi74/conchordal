//! Actual Voice, PCM, and production-meter closed-loop regression.

use super::*;
use std::collections::BTreeMap;
use std::sync::Mutex;

fn check_metric_population(
    seed: u64,
    count: usize,
    body: &str,
    basin: Option<(f32, f32)>,
    stability: f32,
    amplitude: f32,
) {
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    config.analysis.hop_size = 512;
    let duration = 24.0;
    // One complete confirmation history at the end of the immutable fixture.
    let observation_start = duration - (2.0 / 0.5 + 0.06);
    let shaping = basin.map_or(String::new(), |(lo, hi)| {
        format!("meter_stability({stability}); temporal_basin({lo}, {hi});")
    });
    let script = if count == 0 {
        format!("{shaping} wait({duration});")
    } else {
        format!(
            "{shaping} seed({seed}); let voice = {body}().amp({amplitude}).metric().entrainment(0.95).cycles(1).anchor().adsr(0.015, 0.08, 0.4, 0.12); \
         let group = place(voice, line(220.0, 745.0).count({count})); wait({duration}); release(group); wait(1.0);"
        )
    };
    let path = std::env::temp_dir().join(format!(
        "conchordal-meter-{}-{seed}-{count}-{body}-{basin:?}-{stability}-{amplitude}.rhai",
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
    let observed = Arc::new(Mutex::new(BTreeMap::<u64, (usize, u64)>::new()));
    let counts = observed.clone();
    let probe_body = body.to_owned();
    let probe: OfflineBodyProbe = Box::new(move |state, start_tick, batches, pcm| {
        assert!(pcm.iter().all(|bus| bus.iter().all(|s| s.is_finite())));
        if amplitude == 0.0 {
            assert!(pcm.iter().all(|bus| bus.iter().all(|&s| s == 0.0)));
            let meter = state.prod_meter.state();
            assert_eq!(meter.beat.confidence, 0.0);
            assert_eq!(meter.subdivision.confidence, 0.0);
            assert_eq!(meter.measure.confidence, 0.0);
            assert_eq!(meter.subdivision_ratio, 0);
            assert_eq!(meter.measure_ratio, 0);
            assert_eq!(state.current_landscape.rhythm.delta.alpha, 0.0);
        }
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
                    "{probe_body} seed {seed}, {count} Voices: onset error {error} exceeds participation window {window}"
                );
                let mut observed = counts.lock().unwrap();
                let previous = observed.get(&batch.source_id).copied();
                if let Some((lo, hi)) = basin {
                    assert!(
                        (lo..=hi).contains(&beat.freq_hz),
                        "confirmed beat must stay in the hard basin"
                    );
                    if let Some((_, last_tick)) = previous {
                        let rate = f64::from(config.audio.sample_rate)
                            / (onset.onset_tick - last_tick) as f64;
                        assert!(
                            (f64::from(lo)..=f64::from(hi)).contains(&rate),
                            "{probe_body} seed {seed}, {count} Voices: actual onset rate {rate} outside basin {lo}..{hi}"
                        );
                    }
                }
                observed.insert(
                    batch.source_id,
                    (previous.map_or(1, |(n, _)| n + 1), onset.onset_tick),
                );
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
    let observed = observed.lock().unwrap();
    if amplitude == 0.0 {
        assert!(observed.is_empty());
    } else {
        assert_eq!(observed.len(), count);
        if basin.is_some() {
            assert!(
                observed.values().all(|&(n, _)| n >= 2),
                "each Voice must supply a complete onset interval"
            );
        }
    }
    let onsets: usize = observed.values().map(|&(n, _)| n).sum();
    println!(
        "metric case: seed={seed} count={count} body={body} basin={basin:?} stability={stability} amplitude={amplitude}; Voices={} onsets={onsets}",
        observed.len()
    );
}

#[test]
fn two_metric_voices_confirm_and_follow_the_acoustic_beat_without_a_prior() {
    check_metric_population(1, 2, "sine", None, 0.0, 0.045);
}

#[test]
fn eight_metric_voices_follow_the_acoustic_beat_without_a_prior() {
    check_metric_population(21, 8, "sine", None, 0.0, 0.045);
}

#[test]
#[ignore = "complete 18-condition offline meter recurrence assay"]
fn all_registered_metric_populations_follow_the_confirmed_acoustic_beat() {
    for seed in [21, 1, 20260918] {
        for count in [2, 4, 8] {
            for body in ["sine", "harmonic"] {
                check_metric_population(seed, count, body, None, 0.0, 0.045);
            }
        }
    }
}

#[test]
fn two_metric_voices_follow_the_high_basin_previously_left_unconfirmed() {
    check_metric_population(21, 2, "sine", Some((2.8, 3.4)), 0.85, 0.045);
}

#[test]
fn eight_metric_voices_follow_the_high_basin_with_zero_stability() {
    check_metric_population(21, 8, "sine", Some((2.8, 3.4)), 0.0, 0.045);
}

#[test]
fn a_basin_without_radiated_sound_cannot_confirm_a_beat() {
    for count in [0, 8] {
        check_metric_population(21, count, "sine", Some((2.8, 3.4)), 0.85, 0.0);
    }
}

#[test]
#[ignore = "complete 18-condition high-basin and low-basin offline assay"]
fn all_registered_metric_populations_follow_the_confirmed_basin_beat() {
    for seed in [21, 1, 20260918] {
        for count in [2, 4, 8] {
            for body in ["sine", "harmonic"] {
                check_metric_population(seed, count, body, Some((2.8, 3.4)), 0.85, 0.045);
            }
        }
    }
    check_metric_population(21, 8, "sine", Some((1.2, 1.6)), 0.85, 0.045);
}
