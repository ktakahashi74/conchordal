//! Offline research harness; compiled only into the library test executable.

use super::{MeterNetwork, MeterShaping, TAU, wrap_pm_pi};
use crate::core::stream::dorsal::DorsalStream;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug, PartialEq)]
struct Observation {
    time_sec: f64,
    flux: f32,
    band_energy: [f32; 3],
    beat_hz: f32,
    beat_phase: f32,
    beat_confidence: f32,
    subdivision_ratio: u8,
    subdivision_confidence: f32,
    measure_ratio: u8,
    measure_confidence: f32,
    onset: Option<(f64, f32)>,
}

fn observe_audio(
    audio: &[f32],
    sample_rate: u32,
    hop: usize,
    shaping: MeterShaping,
) -> Vec<Observation> {
    assert!(sample_rate > 0 && hop > 0);
    let mut dorsal = DorsalStream::new(sample_rate as f32);
    let mut meter = MeterNetwork::new();
    meter.set_shaping(shaping);
    let mut rows = Vec::with_capacity(audio.len().div_ceil(hop));
    for (index, chunk) in audio.chunks(hop).enumerate() {
        dorsal.process(chunk);
        let metrics = dorsal.last_metrics();
        let drive = (metrics.flux.max(0.0) * 500.0).tanh().clamp(0.0, 1.0);
        let dt = chunk.len() as f32 / sample_rate as f32;
        // Inspect the same detector without changing the live analysis state.
        let onset = meter.onset_detector.clone().process(dt, drive);
        let phase_before = meter.beat_phi;
        let cycles_before = meter.beat_cycles;
        let state = meter.process(dt, drive);
        let start_sec = (index * hop) as f64 / sample_rate as f64;
        rows.push(Observation {
            time_sec: (index * hop + chunk.len()) as f64 / sample_rate as f64,
            flux: metrics.flux,
            band_energy: [metrics.e_low, metrics.e_mid, metrics.e_high],
            beat_hz: state.beat.freq_hz,
            beat_phase: state.beat.phase,
            beat_confidence: state.beat.confidence,
            subdivision_ratio: state.subdivision_ratio,
            subdivision_confidence: state.subdivision.confidence,
            measure_ratio: state.measure_ratio,
            measure_confidence: state.measure.confidence,
            onset: onset.fired.then(|| {
                (
                    start_sec + dt as f64 * onset.frac as f64,
                    wrap_pm_pi(
                        phase_before + TAU * (meter.beat_cycles - cycles_before) * onset.frac,
                    ),
                )
            }),
        });
    }
    rows
}

#[test]
fn audio_observations_are_causal_and_keep_silence_uncertain() {
    let prefix = vec![0.0; 4096];
    let mut changed_future = prefix.clone();
    changed_future.extend((0..4096).map(|i| (i as f32 * 0.21).sin() * 0.4));
    let short = observe_audio(&prefix, 48_000, 512, MeterShaping::default());
    let long = observe_audio(&changed_future, 48_000, 512, MeterShaping::default());
    assert_eq!(short, long[..short.len()]);
    assert!(
        short
            .iter()
            .all(|r| r.onset.is_none() && r.beat_confidence == 0.0)
    );
    assert!(long[short.len()..].iter().any(|r| r.onset.is_some()));
}

#[test]
fn acoustic_pulses_reach_the_meter_without_event_labels() {
    let fs = 48_000;
    let mut audio = vec![0.0f32; fs * 12];
    for start in (fs / 2..audio.len() - 2400).step_by(fs / 2) {
        for j in 0..2400 {
            let t = j as f32 / fs as f32;
            audio[start + j] = 0.4 * (TAU * 750.0 * t).sin() * (-t / 0.012).exp();
        }
    }
    let rows = observe_audio(&audio, fs as u32, 512, MeterShaping::default());
    let onsets = rows.iter().filter(|r| r.onset.is_some()).count();
    assert!(
        (22..=24).contains(&onsets),
        "detected {onsets} acoustic onsets"
    );
    let last = rows.last().unwrap();
    assert!((last.beat_hz - 2.0).abs() < 0.3);
    assert!(last.beat_confidence > 0.6);
}

#[test]
#[ignore = "offline audio-only rhythm campaign; set CONCHORDAL_RHYTHM_ASSAY_DIR"]
fn export_audio_perception_assay() -> anyhow::Result<()> {
    let root = std::env::var_os("CONCHORDAL_RHYTHM_ASSAY_DIR")
        .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_RHYTHM_ASSAY_DIR is required"))?;
    let mut cases = fs::read_dir(Path::new(&root))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    cases.retain(|p| p.is_dir() && p.join("audio.wav").is_file());
    cases.sort();
    anyhow::ensure!(!cases.is_empty(), "no audio.wav inputs");
    for case in cases {
        let mut reader = hound::WavReader::open(case.join("audio.wav"))?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.channels == 1
                && spec.bits_per_sample == 16
                && spec.sample_format == hound::SampleFormat::Int,
            "assay inputs must be mono PCM16"
        );
        let audio = reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?;
        anyhow::ensure!(!audio.is_empty(), "empty audio input");
        // The same priors apply to every input; no case labels enter the model.
        for (label, shaping) in [
            ("unshaped", MeterShaping::default()),
            (
                "sample08_prior",
                MeterShaping {
                    stability: 0.45,
                    basin_hz: Some((1.6, 2.0)),
                },
            ),
        ] {
            let rows = observe_audio(&audio, spec.sample_rate, 512, shaping);
            let path = case.join(format!("{label}.csv"));
            let mut out =
                BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
            writeln!(
                out,
                "time_sec,flux,e_low,e_mid,e_high,beat_hz,beat_phase,beat_confidence,subdivision_ratio,subdivision_confidence,measure_ratio,measure_confidence,onset_time_sec,onset_phase"
            )?;
            for row in rows {
                let onset = match row.onset {
                    Some((time, phase)) => format!("{time:.9},{phase:.9}"),
                    None => ",".to_string(),
                };
                writeln!(
                    out,
                    "{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{:.9},{},{:.9},{}",
                    row.time_sec,
                    row.flux,
                    row.band_energy[0],
                    row.band_energy[1],
                    row.band_energy[2],
                    row.beat_hz,
                    row.beat_phase,
                    row.beat_confidence,
                    row.subdivision_ratio,
                    row.subdivision_confidence,
                    row.measure_ratio,
                    row.measure_confidence,
                    onset
                )?;
            }
            out.flush()?;
        }
        println!("observed {}", case.display());
    }
    Ok(())
}
