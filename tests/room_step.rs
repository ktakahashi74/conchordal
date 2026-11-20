use conchordal::{
    StepEngine, StepStatus,
    core::room::{york_response, york_right_response},
};

#[test]
fn room_step_output_gain_includes_scene_gain() {
    let script = "output_gain(2.0); wait(0.1);";
    for (section, default_db) in [
        ("", 4.5),
        ("[room]\nmode='off'", 4.5),
        ("[room]\nmode='presentation'", -10.5),
        ("[room]\nmode='both'", -10.5),
    ] {
        for (audio, db) in [("", default_db), ("[audio]\noutput_gain_db=4.5", 4.5)] {
            let config = format!("{section}\n{audio}");
            let engine = StepEngine::new(script, &config, Some(1), 48000).unwrap();
            assert_eq!(engine.output_gain(), 10.0_f32.powf((db + 2.0) / 20.0));
        }
    }
}

#[test]
fn room_step_configuration_uses_the_host_rate() {
    let script = "wait(0.1);";
    for (section, rate) in [
        ("", 44100),
        ("[room]\nmode='off'", 44100),
        ("[room]\nmode='presentation'", 48000),
        ("[room]\nmode='both'\nself_return='strict'", 48000),
    ] {
        let config = format!("[audio]\nsample_rate=44100\n{section}");
        StepEngine::new(script, &config, Some(1), rate).unwrap();
    }
    for (section, expected) in [
        ("", "missing field"),
        ("mode='wet'", "wet"),
        ("mode='off'\nprofile=1", "profile"),
        ("mode='both'\nself_return='echo'", "echo"),
        ("mode='off'\nself_return='strict'", "room.self_return"),
        (
            "mode='presentation'",
            "requires 48000 Hz (runtime rate 44100 Hz)",
        ),
    ] {
        let error = StepEngine::new(script, &format!("[room]\n{section}"), Some(1), 44100)
            .err()
            .unwrap();
        assert!(error.message.contains(expected), "{error}");
    }
}

#[test]
fn room_step_presentation_matches_direct_convolution() {
    let script = "wait(0.05); place(sine().amp(0.03).sustain().anchor(), at(220)); wait(0.5);";
    let mut dry = StepEngine::new(script, "", Some(17), 48000).unwrap();
    let mut wet = StepEngine::new(script, "[room]\nmode='presentation'", Some(17), 48000).unwrap();
    let ir = [york_response(), york_right_response()];
    let mut history = Vec::new();
    let mut x = vec![0.; dry.hop_len()];
    let mut y = vec![0.; 2 * x.len()];
    let mut first_difference = None;
    let mut max_error = 0.0f64;
    for hop in 0..12 {
        assert_eq!(dry.step(&mut x), StepStatus::Running);
        assert_eq!(wet.step(&mut y), StepStatus::Running);
        if first_difference.is_none() && x.iter().ne(y.iter().step_by(2)) {
            first_difference = Some(hop);
        }
        for (&input, pair) in x.iter().zip(y.as_chunks::<2>().0) {
            history.push(input);
            for (&output, filter) in pair.iter().zip(&ir) {
                let reference: f64 = history
                    .iter()
                    .rev()
                    .zip(filter)
                    .map(|(&x, &h)| f64::from(x) * f64::from(h))
                    .sum();
                assert!(output.is_finite());
                max_error = max_error.max((f64::from(output) - reference).abs());
            }
        }
    }
    assert!(first_difference.is_some_and(|hop| hop > 0));
    assert!(max_error < 1e-5, "{max_error}");
    println!("D/W first divergent hop {first_difference:?}, max raw PCM error {max_error:e}");
}

#[test]
fn wet_step_returns_the_last_supported_samples_with_finished() {
    use conchordal::viewdata::PlaybackState;

    // No habitat input: the two wet modes hear the same ecology.
    let script = "let v=place(sine().amp(0.05).anchor().sustain().adsr(0,0,1,0.04).send(presentation_bus),at(330)); wait(0.08); release(v); wait(0.3);";
    let ir = [york_response(), york_right_response()];
    let mut dry = StepEngine::new(script, "", Some(4), 48000).unwrap();
    let hop = dry.hop_len();
    let mut block = vec![0.; hop];
    let mut input = Vec::new();
    while !dry.finished() {
        assert!(!matches!(dry.step(&mut block), StepStatus::Error(_)));
        input.extend_from_slice(&block);
    }
    let s = input.iter().rposition(|&x| x != 0.).unwrap();
    let end = s + ir[0].len();
    block.resize(2 * hop, 0.);
    let finish_hop = input.len() / hop - 1;
    for mode in ["presentation", "both"] {
        let mut wet =
            StepEngine::new(script, &format!("[room]\nmode='{mode}'"), Some(4), 48000).unwrap();
        for i in 0..end.div_ceil(hop) {
            let final_hop = (i + 1) * hop >= end;
            let status = wet.step(&mut block);
            assert_eq!(
                status,
                if final_hop {
                    StepStatus::Finished
                } else {
                    StepStatus::Running
                }
            );
            if [0, finish_hop, finish_hop + 1].contains(&i) || final_hop {
                let state = wet.ui_frame().unwrap().meta.playback_state;
                assert_eq!(
                    state,
                    if final_hop {
                        PlaybackState::Finished
                    } else if i >= finish_hop {
                        PlaybackState::Ending
                    } else {
                        PlaybackState::Playing
                    }
                );
            }
            if i == finish_hop + 2 {
                wet.request_end();
            }
            if final_hop {
                for (j, &y) in block.iter().enumerate() {
                    let t = i * hop + j / 2;
                    if t >= end {
                        assert_eq!(y.to_bits(), 0);
                    } else {
                        let expected: f64 = input[..=s]
                            .iter()
                            .enumerate()
                            .filter_map(|(k, &x)| {
                                Some(f64::from(x) * f64::from(*ir[j % 2].get(t - k)?))
                            })
                            .sum();
                        assert!((f64::from(y) - expected).abs() < 1e-7);
                    }
                }
                assert!(block.iter().any(|&x| x != 0.));
            }
        }
        assert_eq!(wet.step(&mut block), StepStatus::Finished);
        assert!(block.iter().all(|&x| x == 0.));
    }
}

#[test]
fn silent_room_finishes_on_the_scheduled_hop() {
    for mode in ["presentation", "both"] {
        let mut engine = StepEngine::new(
            "wait(1024.0 / 48000.0);",
            &format!("[room]\nmode='{mode}'"),
            Some(4),
            48000,
        )
        .unwrap();
        let mut pcm = vec![1.; 2 * engine.hop_len()];
        for expected in [
            StepStatus::Running,
            StepStatus::Running,
            StepStatus::Finished,
        ] {
            assert_eq!(engine.step(&mut pcm), expected);
            assert!(pcm.iter().all(|&x| x == 0.));
        }
    }
}

#[test]
fn habitat_only_release_waits_for_exactly_the_enabled_bus() {
    // The gain ramp reaches zero on its last sample; the prior sample still drives H.
    let script = "place(rain().objects(4).rate(100).amp(0.1).fade(0.25).send(habitat_bus), consonance(220,2000)); wait(0.5);";
    let release_end = 24_000usize.next_multiple_of(512) + 12_000;
    let room_end = release_end - 2 + york_response().len();
    for (mode, end) in [("presentation", release_end), ("both", room_end)] {
        let mut engine =
            StepEngine::new(script, &format!("[room]\nmode='{mode}'"), Some(21), 48000).unwrap();
        let hop = engine.hop_len();
        let mut block = vec![0.; 2 * hop];
        for i in 0..end.div_ceil(hop) {
            let expected = if (i + 1) * hop >= end {
                StepStatus::Finished
            } else {
                StepStatus::Running
            };
            assert_eq!(engine.step(&mut block), expected, "{mode} hop {i}");
            assert!(block.iter().all(|&x| x == 0.));
        }
    }
}
