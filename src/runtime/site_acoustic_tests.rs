//! Finite-score acquisition. The reference receives the actual fixed-pitch dry stem.
//! Run explicitly because it renders each amplitude, placement, room and output gain.
use super::*;
use crate::config::RoomMode;
use crate::core::room::{RoomKernel, york_response, york_right_response};
use crate::core::temporal_expectation::AcousticTemporalExpectation;
use std::io::{BufRead, BufReader};

fn timing_free() {
    let status = std::process::Command::new("bash")
        .args(["scripts/timing_lock.sh", "check"])
        .output()
        .unwrap();
    assert!(status.status.success() && status.stdout == b"free\n");
}

// Exercise the product WAV sink with exactly the score seen by the independent reader.
fn check_wav(
    folder: &std::path::Path,
    script: &str,
    expected: &[i16],
    channels: usize,
    complete: bool,
) {
    let stem = format!("site-wav-{}", std::process::id());
    let source = folder.join(format!("{stem}.rhai"));
    let output = folder.join(format!("{stem}.wav"));
    std::fs::write(&source, script).unwrap();
    timing_free();
    wiring::run_render(
        source.to_str().unwrap(),
        output.to_str().unwrap().into(),
        AppConfig::default(),
        Arc::new(AtomicBool::new(false)),
        Some(21),
        None,
        0,
        None,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let mut reader = hound::WavReader::open(&output).unwrap();
    assert_eq!(reader.spec().channels as usize, channels);
    if complete {
        assert_eq!(reader.len() as usize, expected.len());
    }
    assert!(
        reader
            .samples::<i16>()
            .take(expected.len())
            .map(Result::unwrap)
            .eq(expected.iter().copied())
    );
    drop(reader);
    for file in [source, output] {
        std::fs::remove_file(file).unwrap();
    }
}

#[test]
#[ignore = "site acceptance matrix; set SITE_ASSAY_DIR to a temporary directory"]
fn site_score_acoustic_readers() {
    let folder = std::path::PathBuf::from(std::env::var_os("SITE_ASSAY_DIR").unwrap());
    for mode in [RoomMode::Off, RoomMode::Presentation, RoomMode::Both] {
        for distance in [1., 12., 23.5] {
            for amp in [0.03, 0.04] {
                let mut baseline = None;
                for db in [0, 9] {
                    timing_free();
                    // Calibrate the gain of this fixed seed and envelope from its measured dry peak.
                    let reference_peaks = [
                        0.029976847,
                        0.02997494,
                        0.029983578,
                        0.029972,
                        0.030020302,
                        0.030009804,
                        0.029994212,
                        0.030012,
                        0.029995902,
                        0.029976794,
                        0.029987762,
                        0.030000001,
                        0.029997902,
                        0.030029047,
                        0.02999907,
                        0.030002363,
                        0.029979706,
                    ];
                    let gains =
                        reference_peaks.map(|peak| (0.03 * 0.03 / 0.034632806) * amp / peak);
                    let script = format!(
                        r#"output_gain({db}); room("{}");
                    let gains = {gains:?};
                    let v = sine().brain("seq").anchor().sustain().adsr(0.02,0,1,0.002);
                    place(rain().amp(0).objects(0).follow_meter(1),consonance(200,900));
                    wait(0.5);
                    let a = place(v.amp(gains[0]),at(375).site(0,{distance})); wait(0.4);release(a);
                    wait(7.8);
                    for n in 0..16 {{
                        let hz = if n % 2 == 0 {{375}} else {{500}};
                        let b = place(v.amp(gains[n+1]),at(hz).site(0,{distance})); wait(0.2); release(b); wait(0.3);
                    }}
                    wait(0.1);"#,
                        match mode {
                            RoomMode::Off => "off",
                            RoomMode::Presentation => "presentation",
                            RoomMode::Both => "both",
                        }
                    );
                    let mut step = StepEngine::new(&script, "", Some(21), 48000).unwrap();
                    let site = u32::from(distance != 12.);
                    let report_path = folder.join(format!("listener-{}.jsonl", std::process::id()));
                    step.runtime.state.reporter =
                        Some(JsonlReporter::create(report_path.to_str().unwrap()).unwrap());
                    assert!(
                        step.runtime
                            .state
                            .listener
                            .listener_twin
                            .enable_contour_reporting(48000.)
                    );
                    let mut listener = ListenerTwin::with_sample_rate(48000., Default::default());
                    assert!(listener.enable_contour_reporting(48000.));
                    let mut meter = MeterNetwork::new();
                    let mut dorsal = DorsalStream::new(48000.);
                    let mut temporal = AcousticTemporalExpectation::new(48000).unwrap();
                    let mut reflection = [york_response(), york_right_response()].map(|mut ir| {
                        ir[0] = 0.;
                        RoomKernel::new(&ir, 512)
                            .stream_pool(1, 1)
                            .unwrap()
                            .pop()
                            .unwrap()
                    });
                    let mut out = vec![0.; 512 * step.output_channels()];
                    let mut wet = [[0.; 512]; 2];
                    let mut guard = crate::audio::limiter::Limiter::new(
                        step.limiter_mode(),
                        48000,
                        step.output_channels(),
                    )
                    .with_input_gain(step.output_gain());
                    let mut wav = Vec::new();
                    let mut episodes = Vec::new();
                    let mut rms_energy = 0.;
                    let mut first_crossing = None;
                    let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
                    let (mut fired, mut bands, mut confidence, mut peak, mut flux, mut attention) =
                        (Vec::new(), 0usize, 0f32, 0f32, 0f32, 0f32);
                    let mut peaks = [0f32; 17];
                    for hop in 0..1565u64 {
                        assert_eq!(step.step(&mut out), StepStatus::Running);
                        let state = &mut step.runtime.state;
                        let dry = state
                            .schedule_renderer
                            .sites
                            .as_ref()
                            .unwrap()
                            .dry_habitat(site);
                        let event = if hop * 512 < 417600 {
                            0
                        } else {
                            1 + ((hop * 512 - 417600) / 24000) as usize
                        };
                        peaks[event] = dry.iter().fold(peaks[event], |m, s| m.max(s.abs()));
                        peak = dry.iter().fold(peak, |m, s| m.max(s.abs()));
                        let send: [f32; 512] = std::array::from_fn(|i| dry[i] * (1f32 / 23.5));
                        if mode != RoomMode::Off {
                            for (r, w) in reflection.iter_mut().zip(&mut wet) {
                                r.process(&send, w);
                            }
                        }
                        let direct = |i: usize| (1f64 / distance) as f32 * dry[i];
                        let h: [f32; 512] = std::array::from_fn(|i| {
                            12. * (direct(i)
                                + if mode == RoomMode::Both {
                                    wet[0][i]
                                } else {
                                    0.
                                })
                        });
                        let left: [f32; 512] =
                            std::array::from_fn(|i| 12. * (direct(i) + wet[0][i]));
                        assert_eq!(state.schedule_renderer.habitat_output(), h);
                        for (i, &sample) in left.iter().enumerate() {
                            let end = hop * 512 + i as u64 + 1;
                            rms_energy += f64::from(sample).powi(2);
                            if end.is_multiple_of(480) {
                                if (rms_energy / 480.).sqrt() >= 0.005 {
                                    first_crossing.get_or_insert(end - 480);
                                }
                                rms_energy = 0.;
                            }
                        }
                        for (i, frame) in out
                            .chunks_exact(1 + usize::from(mode != RoomMode::Off))
                            .enumerate()
                        {
                            assert_eq!(frame[0], left[i]);
                            if frame.len() == 2 {
                                assert_eq!(frame[1], 12. * (direct(i) + wet[1][i]));
                            }
                        }
                        use std::hash::Hash;
                        for s in &out {
                            s.to_bits().hash(&mut fingerprint);
                        }
                        let channels = 1 + usize::from(mode != RoomMode::Off);
                        guard.process_interleaved(&mut out, channels);
                        wav.extend(
                            out.iter()
                                .map(|x| (x.clamp(-1., 1.) * f32::from(i16::MAX)) as i16),
                        );
                        let now = hop * 512;
                        dorsal.process(&h);
                        let drive = (500. * dorsal.last_metrics().flux.max(0.)).tanh();
                        let expected = meter.process_at(now, 512, 48000., Some(drive)).beat;
                        let actual = state.temporal.flora_rhythm.state.beat;
                        assert_eq!(
                            (actual.phase, actual.freq_hz, actual.confidence),
                            (expected.phase, expected.freq_hz, expected.confidence)
                        );
                        assert_eq!(state.temporal.flora_rhythm.flux, drive);
                        assert_eq!(
                            state.temporal.flora_meter.as_ref().unwrap().onset_age(),
                            meter.onset_age()
                        );
                        if meter.onset_age().is_some_and(|age| age < 512. / 48000.) {
                            fired.push(hop);
                        }
                        let l = listener.observe_presentation_audio(now, hop, &left);
                        let a = state.listener.latest_listener_fast_state.unwrap();
                        assert_eq!(
                            (
                                a.attention_level,
                                a.attention_metrics.flux,
                                a.meter_state.beat.confidence,
                                a.meter_state.beat.phase,
                                a.meter_state.beat.freq_hz
                            ),
                            (
                                l.attention_level,
                                l.attention_metrics.flux,
                                l.meter_state.beat.confidence,
                                l.meter_state.beat.phase,
                                l.meter_state.beat.freq_hz
                            )
                        );
                        confidence = confidence.max(l.meter_state.beat.confidence);
                        flux = flux.max(dorsal.last_metrics().flux);
                        attention = attention.max(l.attention_level);
                        listener.observe_contour_audio(now, &left, |e| {
                            if e.kind == "periodic_episode" {
                                episodes.push((
                                    hop,
                                    e.periodic_frequency_hz.unwrap(),
                                    (e.onset_sec.unwrap() * 48000.).round() as u64,
                                ));
                            }
                        });
                        temporal.process(now, &h, |o| {
                            bands += usize::from(o.band != 0);
                        });
                        if hop % 20 == 0 {
                            let expected = temporal.forecast().unwrap();
                            let actual = state
                                .temporal
                                .temporal_expectation
                                .as_mut()
                                .unwrap()
                                .forecast()
                                .unwrap();
                            assert_eq!(
                                actual.period_reference(0.5),
                                expected.period_reference(0.5)
                            );
                            assert_eq!(
                                actual.band_energy_at((now + 1024) as f64),
                                expected.band_energy_at((now + 1024) as f64)
                            );
                        }
                    }
                    check_wav(&folder, &script, &wav, step.output_channels(), false);
                    drop(step.runtime.state.reporter.take());
                    let mut observed = Vec::new();
                    for line in BufReader::new(std::fs::File::open(&report_path).unwrap()).lines() {
                        let line = line.unwrap();
                        if line.contains("\"listener_contour\"")
                            && line.contains("\"periodic_episode\"")
                        {
                            let row: serde_json::Value = serde_json::from_str(&line).unwrap();
                            observed.push((
                                row["generated_frame_id"].as_u64().unwrap(),
                                row["periodic_frequency_hz"].as_f64().unwrap() as f32,
                                (row["onset_sec"].as_f64().unwrap() * 48000.).round() as u64,
                            ));
                        }
                    }
                    std::fs::remove_file(report_path).unwrap();
                    assert_eq!(observed, episodes);
                    assert_eq!(first_crossing, episodes.first().map(|e| e.2));
                    println!("STEMS {peaks:?}");
                    println!(
                        "SITE_SCORE {mode:?} d={distance} amp={amp} db={db} peak={peak} fired={fired:?} bands={bands} flux={flux} attention={attention} confidence={confidence} isolated={:?}",
                        episodes.first()
                    );
                    assert!(
                        peaks.iter().all(|p| (f64::from(*p) - amp).abs() < 0.00001),
                        "dry stem peaks {peaks:?}, expected {amp}"
                    );
                    assert!(
                        fired.len() > 2
                            && bands > 2
                            && confidence > 0.
                            && episodes.first().is_some_and(|(hop, _, _)| *hop < 300)
                    );
                    let signature = (
                        std::hash::Hasher::finish(&fingerprint),
                        fired,
                        bands,
                        episodes,
                    );
                    if let Some(before) = &baseline {
                        assert_eq!(before, &signature);
                    } else {
                        baseline = Some(signature);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "finite offline output acceptance; set SITE_ASSAY_DIR"]
fn site_score_near_far_and_full_piece_output() {
    use crate::audio::limiter::{Limiter, LimiterMode};
    let folder = std::path::PathBuf::from(std::env::var_os("SITE_ASSAY_DIR").unwrap());
    let piece = include_str!("../../samples/demo/line-moss-response.rhai")
        .replace("random(98.0, 1568.0)", "random(98.0, 1568.0).site(0,12)");
    let pair = r#"let tone = sine().brain("seq").anchor().sustain().amp(0.04).adsr(0.02,0,1,0.002);
        let far = place(tone,at(375).site(0,23.5)); wait(0.128);
        let near = place(tone,at(375).site(0,1)); wait(0.512); release(near);
        wait(0.192);release(far);wait(0.1);"#;
    // The renderer seeds carrier phase by owner and onset. At hop 767 these two
    // carriers differ by 0.0036 rad. Check the actual dry samples below.
    let chord = r#"wait(8.181333333333333); let tone=sine().brain("seq").anchor().sustain().adsr(0.02,0,1,0.002);
        let a=place(tone.amp(NEAR),at(375).site(0,1));
        let b=place(tone.amp(FAR),at(375).site(0,23.5));
        wait(0.3);release(a);wait(0.1);release(b);wait(0.1);"#;
    timing_free();
    let mut calibration = StepEngine::new(
        &chord.replace("NEAR", "0.04").replace("FAR", "0.04"),
        "",
        Some(21),
        48000,
    )
    .unwrap();
    let mut peaks = [0f32; 2];
    let mut dot = [0f64; 3];
    for _ in 0..795 {
        assert_eq!(calibration.step(&mut [0.; 512]), StepStatus::Running);
        let sites = calibration
            .runtime
            .state
            .schedule_renderer
            .sites
            .as_ref()
            .unwrap();
        for (&a, &b) in sites.dry_habitat(1).iter().zip(sites.dry_habitat(2)) {
            let (a, b) = (f64::from(a), f64::from(b));
            dot[0] += a * b;
            dot[1] += a * a;
            dot[2] += b * b;
        }
        for (id, peak) in peaks.iter_mut().enumerate() {
            *peak = sites
                .dry_habitat(id as u32 + 1)
                .iter()
                .fold(*peak, |m, x| m.max(x.abs()));
        }
    }
    assert!(
        dot[0] / (dot[1] * dot[2]).sqrt() > 0.99998,
        "carrier phase correlation: {dot:?}"
    );
    let [near, far] = peaks.map(|p| (0.04_f64.powi(2) / f64::from(p)).to_string());
    let chord = chord.replace("NEAR", &near).replace("FAR", &far);
    for (name, score) in [
        ("unison", chord.as_str()),
        ("pair", pair),
        ("piece", piece.as_str()),
    ] {
        for mode in ["off", "presentation", "both"] {
            let score = format!("room(\"{mode}\");\n{score}");
            timing_free();
            let mut step = StepEngine::new(&score, "", Some(21), 48000).unwrap();
            let channels = step.output_channels();
            let mut pcm = vec![0.; 512 * channels];
            let mut guards = [0., 9.].map(|db| {
                Limiter::new(LimiterMode::default(), 48000, channels)
                    .with_input_gain(10f32.powf(db / 20.))
            });
            let mut wav = [Vec::new(), Vec::new()];
            let mut levels = [[[0f32; 2]; 2]; 2];
            let mut over = [0u64; 2];
            let mut reductions = [0f32; 2];
            let mut changed = [(None, None); 2];
            let mut hop = 0u64;
            while !step.finished() {
                match step.step(&mut pcm) {
                    StepStatus::Running | StepStatus::Finished => (),
                    status => panic!("{status:?}"),
                }
                for (g, guard) in guards.iter_mut().enumerate() {
                    let gain = if g == 0 { 1. } else { 10f32.powf(9. / 20.) };
                    let mut post = pcm.clone();
                    guard.process_interleaved(&mut post, channels);
                    let stats = guard.stats();
                    reductions[g] = reductions[g].max(stats.max_reduction_db);
                    over[g] += stats.num_over;
                    for (i, (&raw, &out)) in pcm.iter().zip(&post).enumerate() {
                        let ch = i % channels;
                        levels[g][0][ch] = levels[g][0][ch].max((raw * gain).abs());
                        levels[g][1][ch] = levels[g][1][ch].max(out.abs());
                        if (raw * gain - out).abs() > 1e-7 {
                            let n = hop * 512 + (i / channels) as u64;
                            changed[g].0.get_or_insert(n);
                            changed[g].1 = Some(n);
                        }
                        wav[g].push((out.clamp(-1., 1.) * f32::from(i16::MAX)) as i16);
                    }
                }
                hop += 1;
            }
            for (g, db) in [0, 9].into_iter().enumerate() {
                check_wav(
                    &folder,
                    &format!("{score}\noutput_gain({db});"),
                    &wav[g],
                    channels,
                    true,
                );
                println!(
                    "SITE_LIMIT {name} {mode} db={db} hops={hop} pre_post={:?} over={} max_reduction_db={} first_last={:?}",
                    levels[g], over[g], reductions[g], changed[g]
                );
            }
            if name == "unison" && mode == "off" {
                assert!(
                    (levels[0][0][0] - 0.04 * (12. + 12. / 23.5)).abs() < 0.0001,
                    "{levels:?}"
                );
            }
            if name != "piece" && mode == "off" {
                assert_eq!(over[0], 0);
                assert!(over[1] > 0 && reductions[1] > 0. && levels[1][1][0] <= 0.98);
            }
        }
    }
}
