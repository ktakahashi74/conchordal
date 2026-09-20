//! Full-clock acoustic replay probe; this is not a roomy-reference acceptance gate.

use super::*;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use std::io::Write;

#[test]
#[ignore = "explicit I9 long-form acoustic capacity probe"]
fn thirty_minute_acoustic_capacity_probe() {
    let material = crate::temporal_cognition::long_form::Material::load();
    let bus: u8 = std::env::var("CONCHORDAL_I9_BUS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    assert!(bus <= 1);
    let seconds: u64 = std::env::var("CONCHORDAL_I9_SECONDS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1800);
    let capacity: usize = std::env::var("CONCHORDAL_I9_EPISODES")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(256);
    let candidates: Option<usize> = std::env::var("CONCHORDAL_I9_CANDIDATES")
        .ok()
        .map(|s| s.parse().unwrap());
    assert!(seconds > 0 && seconds <= 1800);
    let audio_path = std::env::var_os("CONCHORDAL_I9_WAV");
    let mut reader = audio_path.as_ref().map(|path| {
        assert!(
            std::env::var_os("CONCHORDAL_I9_MATERIAL").is_some(),
            "WAV replay requires frozen material bookkeeping"
        );
        let reader = hound::WavReader::open(path).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (1, 48000, 16)
        );
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        assert_eq!(
            reader.duration() % 512,
            0,
            "registered render must end on a complete hop"
        );
        reader
    });
    let frames = reader.as_ref().map_or(seconds * 48000 / 512, |r| {
        let total = u64::from(r.duration()) / 512;
        if std::env::var_os("CONCHORDAL_I9_SECONDS").is_some() {
            total.min(seconds * 48000 / 512)
        } else {
            total
        }
    });
    let mut pcm = reader.as_mut().map(|r| r.samples::<i16>());
    let directory = std::path::PathBuf::from(
        std::env::var("CONCHORDAL_I9_OUTPUT")
            .unwrap_or_else(|_| "target/temporal-dcc/i9-probe".into()),
    );
    std::fs::create_dir_all(&directory).unwrap();
    let mut log =
        std::io::BufWriter::new(std::fs::File::create(directory.join("profile.jsonl")).unwrap());
    let space = Log2Space::new(100., 6400., 24);
    let kernel = NsgtKernelLog2::new(
        NsgtLog2Config {
            fs: 48000.,
            overlap: 0.75,
            nfft_override: Some(2048),
            ..Default::default()
        },
        space.clone(),
        None,
        PowerMode::Coherent,
    );
    let mut analysis = RtNsgtKernelLog2::new(kernel);
    let section_config = section::tests::runtime_config();
    let options = Options {
        groove: None,
        deterministic: true,
        ridge: Some(TemporalRidgeConfig {
            means: [0.; 3],
            deviations: [0.05, 4., 1.],
        }),
        acoustic: Some(TemporalAcousticConfig {
            group_means: [0.; 3],
            group_deviations: [0.05, 4., 1.],
            accent_means: [0.; 2],
            accent_deviations: [1.; 2],
            group_retirement_sec: 2.,
            inactive_energy_max: 1e-8,
            correlation_window_sec: 0.25,
            min_pairs: 8,
            min_coverage: 0.9,
            persistence_hops: 3,
        }),
        memory: Some(crate::config::TemporalMemoryConfig {
            retention: None,
            candidates,
            scales: [1.; 10],
            span_hops: 64,
            episodes: capacity,
            query_cadence_ms: 100,
            deadline_ms: 200,
        }),
        gesture: Some(crate::config::TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        }),
        period: Some(crate::config::TemporalPeriodConfig {
            model: crate::config::ArrivalModel::Hazard,
            coefficients: [0.; 18],
            means: [0.; 8],
            deviations: [1.; 8],
            horizon_sec: 0.1,
        }),
        phrase: Some(phrase::tests::config()),
        section: Some(section_config),
        whole: Some(whole::tests::config()),
        body_prototypes: None,
        action_profiles: None,
    };
    let mut tap = Tap::spawn(bus, 48000, 512, 2048, space, options);
    let output = Arc::clone(&tap.snapshot);
    let mut samples = [0_f32; 512];
    let mut checksum = 14695981039346656037_u64;
    let start = Instant::now();
    let mut last_profile = 0;
    let mut latest_query = [0_u64; 7];
    let mut return_queries = [0_u64; 3];
    let mut retained_early_matches = [0_u64; 3];
    let mut peak_episodes = 0;
    for frame in 0..frames {
        for (i, sample) in samples.iter_mut().enumerate() {
            if let Some(pcm) = pcm.as_mut() {
                *sample =
                    f32::from(pcm.next().expect("registered audio ended early").unwrap()) / 32768.;
                checksum = (checksum ^ u64::from(sample.to_bits())).wrapping_mul(1099511628211);
                continue;
            }
            let t = (frame * 512 + i as u64) as f64 / 48000.;
            let occurrence = [0., 144., 804., 1740.]
                .into_iter()
                .find(|a| *a <= t && t < *a + 8.);
            let (local, frequency, amplitude) = if let Some(begin) = occurrence {
                let local = t - begin;
                let pulse = (std::f64::consts::TAU * local / 0.51).sin();
                (local, 500., 0.025 + 0.018 * pulse)
            } else {
                let cell = ((t / 11.).floor() as u64).max(1);
                let local = t % 11.;
                let frequency = 170. + ((cell * 73) % 1100) as f64;
                let amp = if local > 8. {
                    0.
                } else {
                    0.018
                        + 0.014
                            * (std::f64::consts::TAU * local / (0.31 + 0.017 * (cell % 13) as f64))
                                .sin()
                };
                (local, frequency, amp)
            };
            *sample = (amplitude
                * ((std::f64::consts::TAU * frequency * local).sin()
                    + 0.17 * (std::f64::consts::TAU * 2.71 * frequency * local).sin()))
                as f32;
            checksum = (checksum ^ u64::from(sample.to_bits())).wrapping_mul(1099511628211);
        }
        let energy = samples.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 512.;
        tap.observe(frame, analysis.process_hop(&samples), energy);
        let until = Instant::now() + std::time::Duration::from_secs(20);
        let state = loop {
            let state = output.lock().unwrap();
            if state.frame_id == Some(frame) {
                break state;
            }
            drop(state);
            assert!(
                Instant::now() < until,
                "observer stopped before frame {frame}"
            );
            std::thread::sleep(std::time::Duration::from_micros(100));
        };
        let error = state
            .memory_error
            .or(state.phrase_error)
            .or(state.section_error)
            .or(state.whole_error);
        if error.is_some() || state.acoustic_failed || state.ridge_failed {
            writeln!(log,"{}",serde_json::json!({"frame":frame,"error":error,"state":state.state,"acoustic_failed":state.acoustic_failed})).unwrap();
            log.flush().unwrap();
            panic!("long replay failed at frame {frame}: {error:?}");
        }
        if let Some(memory) = state.memory {
            peak_episodes = peak_episodes.max(memory.stored_episodes);
            for (slot, q) in memory
                .group_queries
                .iter()
                .enumerate()
                .filter_map(|(i, q)| q.map(|q| (i, q)))
            {
                if q.query_id == latest_query[slot] {
                    continue;
                }
                latest_query[slot] = q.query_id;
                let t = q.support_end_sample as f64 / 48000.;
                for (j, [begin, until]) in material.returns.into_iter().enumerate() {
                    if begin <= t && t < until {
                        return_queries[j] += 1;
                        if q.best.is_some_and(|m| {
                            !m.ambiguous
                                && m.cost <= 1.
                                && m.support_end_sample < material.returns[0][0] as u64 * 48000
                        }) {
                            retained_early_matches[j] += 1;
                        }
                    }
                }
            }
        }
        let elapsed = (frame + 1) * 512 / 48000;
        if elapsed >= last_profile + 10 || frame + 1 == frames {
            let section = state.section;
            let phrase_mass: f64 = state.phrase.as_ref().map_or(0., |p| {
                p.groups
                    .iter()
                    .flatten()
                    .flat_map(|g| g.candidates.iter().flatten())
                    .map(|c| c.mass)
                    .sum()
            });
            let section_mass: f64 = section.as_ref().map_or(0., |s| {
                s.groups
                    .iter()
                    .flatten()
                    .flat_map(|g| g.candidates.iter().flatten())
                    .map(|c| c.mass)
                    .sum()
            });
            writeln!(log,"{}",serde_json::json!({"bus":bus,"seconds":elapsed,"wall_seconds":start.elapsed().as_secs_f64(),"memory":state.memory,
                "phrase_mass":phrase_mass,"section_mass":section_mass,"whole":state.whole,
                "commitments":section.map(|s|s.committed_phrases),"commitment_losses":section.map(|s|s.commitment_losses),
                "return_queries":return_queries,"early_best_matches":retained_early_matches})).unwrap();
            log.flush().unwrap();
            last_profile = elapsed;
        }
    }
    let before_eof = output.lock().unwrap().whole;
    drop(tap);
    let state = output.lock().unwrap();
    assert_eq!(
        serde_json::to_value(before_eof).unwrap(),
        serde_json::to_value(state.whole).unwrap(),
        "EOF changed the last heard whole-piece graph or scoring snapshot"
    );
    let result = serde_json::json!({"status":"probe_only_not_roomy_reference_acceptance","bus":bus,"seconds":seconds,
        "frames":frames,"audio_seconds":frames as f64*512./48000.,"audio_path":audio_path.map(std::path::PathBuf::from),"material":material,
        "sample_rate":48000,"hop":512,"nfft":2048,"bank_capacity":capacity,
        "candidate_capacity":candidates.unwrap_or(crate::temporal_cognition::memory::CANDIDATES),
        "pcm_word_checksum":format!("{checksum:016x}"),"peak_episodes":peak_episodes,
        "memory":state.memory,"return_queries":return_queries,"early_best_matches":retained_early_matches,
        "whole":state.whole,"wall_seconds":start.elapsed().as_secs_f64()});
    std::fs::write(
        directory.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("I9_PROBE {result}");
    assert_eq!(state.support_end_sample, frames * 512);
    assert_eq!(state.received_frames, frames);
    assert!(!state.memory_failed && state.section_error.is_none());
}
