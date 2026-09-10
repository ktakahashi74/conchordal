//! Offline port comparison and acoustic observer export, unavailable to the instrument.

use super::{AcousticTemporalExpectation, TemporalExpectation};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

#[test]
#[ignore = "offline conditional energy forecasts; set CONCHORDAL_HISTORY_PREDICTION_PLAN"]
fn export_history_prediction_assay() -> anyhow::Result<()> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        input: std::path::PathBuf,
        output: std::path::PathBuf,
    }
    let plan = std::env::var_os("CONCHORDAL_HISTORY_PREDICTION_PLAN")
        .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_HISTORY_PREDICTION_PLAN is required"))?;
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(plan)?)?;
    anyhow::ensure!(!cases.is_empty(), "no prediction inputs");
    for case in cases {
        let mut reader = hound::WavReader::open(&case.input)?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.channels == 1
                && spec.bits_per_sample == 16
                && spec.sample_format == hound::SampleFormat::Int,
            "expected mono PCM16"
        );
        let audio = reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?;
        let mut observer = AcousticTemporalExpectation::new(spec.sample_rate)
            .ok_or_else(|| anyhow::anyhow!("unsupported observation rate"))?;
        let window = observer.window.len();
        let mut own = super::OwnSoundHistory::new(&observer, 0);
        let silence = vec![0.0; window];
        let mut matches = Vec::with_capacity(7);
        let mut out = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&case.output)?,
        );
        for (index, chunk) in audio.chunks(window).enumerate() {
            matches.clear();
            own.process((index * window) as u64, &silence[..chunk.len()], &silence[..chunk.len()], chunk, |start, window, matched| {
                matches.push(serde_json::json!({"target_start_frame":start,"window_frames":window,"matched":matched}));
            });
            observer.process((index * window) as u64, chunk, |_| {});
            if chunk.len() != window {
                continue;
            }
            let energy = observer.energy.history
                [((observer.energy.step_index - 1) % super::HISTORY as u64) as usize];
            let history = observer.energy.temporal.snapshot();
            let query = if (index + 1) % 10 == 0 {
                let shared = observer.forecast().unwrap();
                let candidate = own.external.history_prediction.forecast(energy, &history);
                let (mixed, weight) = own.external.history_prediction.issue_comparison(
                    observer.energy.step_index,
                    shared.band_energy,
                    candidate,
                    Some(((index + 1) * window) as u64),
                );
                Some(serde_json::json!({"shared":shared.band_energy.to_vec(),
                    "history":candidate.to_vec(), "mixed":mixed.to_vec(), "weight":weight.to_vec()}))
            } else {
                None
            };
            serde_json::to_writer(
                &mut out,
                &serde_json::json!({
                    "step":observer.energy.step_index,"end_frame":(index+1)*window,
                    "sample_rate":spec.sample_rate,"window_frames":window,"energy":energy,
                    "observed_history":history,"query":query,
                    "completed_prediction_errors":own.external.history_prediction.take_errors(),
                    "prediction_matches":matches
                }),
            )?;
            writeln!(out)?;
        }
        out.flush()?;
        println!(
            "conditional energy {}: {} complete windows",
            case.input.display(),
            audio.len() / window
        );
    }
    Ok(())
}

#[test]
#[ignore = "offline continuous history port comparison; set CONCHORDAL_HISTORY_ASSAY_PLAN"]
fn export_continuous_history_assay() -> anyhow::Result<()> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        input: std::path::PathBuf,
        output: std::path::PathBuf,
    }
    let plan = std::env::var_os("CONCHORDAL_HISTORY_ASSAY_PLAN")
        .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_HISTORY_ASSAY_PLAN is required"))?;
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(plan)?)?;
    anyhow::ensure!(!cases.is_empty(), "no history assay inputs");
    for case in cases {
        let mut reader = hound::WavReader::open(&case.input)?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.channels == 1
                && spec.bits_per_sample == 16
                && spec.sample_format == hound::SampleFormat::Int,
            "expected mono PCM16"
        );
        let audio = reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?;
        let mut observer = AcousticTemporalExpectation::new(spec.sample_rate)
            .ok_or_else(|| anyhow::anyhow!("unsupported observation rate"))?;
        let window = observer.window.len();
        let mut out = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&case.output)?,
        );
        for (index, chunk) in audio.chunks(window).enumerate() {
            observer.process((index * window) as u64, chunk, |_| {});
            if chunk.len() != window {
                continue;
            }
            let energy = observer.energy.history
                [((observer.energy.step_index - 1) % super::HISTORY as u64) as usize];
            serde_json::to_writer(
                &mut out,
                &serde_json::json!({
                    "start_frame":index * window,"end_frame":(index+1)*window,"sample_rate":spec.sample_rate,
                    "step_sec":observer.energy.step_sec,"band_energy":energy,
                    "history":observer.energy.temporal.snapshot()
                }),
            )?;
            writeln!(out)?;
        }
        out.flush()?;
        println!(
            "history {}: {} complete windows",
            case.input.display(),
            audio.len() / window
        );
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    step_sec: f32,
    bands: Vec<u8>,
}

#[test]
#[ignore = "offline recurrence comparison; set CONCHORDAL_TEMPORAL_ASSAY_DIR"]
fn export_temporal_assay() -> anyhow::Result<()> {
    let root = std::env::var_os("CONCHORDAL_TEMPORAL_ASSAY_DIR")
        .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_TEMPORAL_ASSAY_DIR is required"))?;
    let mut cases = fs::read_dir(Path::new(&root))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    cases.retain(|p| {
        p.is_dir() && (p.join("input.json").is_file() || p.join("audio.wav").is_file())
    });
    cases.sort();
    anyhow::ensure!(!cases.is_empty(), "no inputs");
    for case in cases {
        for (input_name, output_name) in [
            ("input.json", "recurrence.csv"),
            ("audio.wav", "acoustic_recurrence.csv"),
        ] {
            let input = case.join(input_name);
            if !input.is_file() {
                continue;
            }
            let mut out = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(case.join(output_name))?,
            );
            writeln!(
                out,
                "time_sec,observed_band,event_probability,null_event_probability,recurrence_weight,best_period_sec,recent_onsets,gain_bits"
            )?;
            let mut rows = Vec::new();
            if input_name == "input.json" {
                let input: Input = serde_json::from_slice(&fs::read(input)?)?;
                let mut model = TemporalExpectation::new(input.step_sec);
                for (tick, band) in input.bands.into_iter().enumerate() {
                    let (prediction, gain) = model.observe(band);
                    rows.push((
                        (tick + 1) as f64 * input.step_sec as f64,
                        band,
                        prediction,
                        gain,
                    ));
                }
            } else {
                let mut reader = hound::WavReader::open(input)?;
                let spec = reader.spec();
                anyhow::ensure!(
                    spec.channels == 1
                        && spec.bits_per_sample == 16
                        && spec.sample_format == hound::SampleFormat::Int,
                    "expected mono PCM16"
                );
                let audio = reader
                    .samples::<i16>()
                    .map(|s| s.map(|v| v as f32 / 32768.0))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut observer = AcousticTemporalExpectation::new(spec.sample_rate)
                    .ok_or_else(|| anyhow::anyhow!("unsupported observation rate"))?;
                observer.process(0, &audio, |row| {
                    rows.push((
                        row.end_frame as f64 / spec.sample_rate as f64,
                        row.band,
                        row.prediction,
                        row.gain_bits,
                    ));
                });
            }
            for (time, band, prediction, gain) in rows {
                let period = prediction
                    .leading_period_sec
                    .map(|p| format!("{p:.9}"))
                    .unwrap_or_default();
                writeln!(
                    out,
                    "{time:.9},{band},{:.9},{:.9},{:.9},{period},{:.9},{gain:.9}",
                    1.0 - prediction.probability[0],
                    1.0 - prediction.null_probability[0],
                    prediction.recurrence_weight,
                    prediction.recent_onsets
                )?;
            }
            out.flush()?;
        }
        println!("observed {}", case.display());
    }
    Ok(())
}

#[test]
#[ignore = "offline delayed energy forecasts; set CONCHORDAL_ENERGY_ASSAY_PLAN"]
fn export_energy_forecast_assay() -> anyhow::Result<()> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        input: std::path::PathBuf,
        output: std::path::PathBuf,
    }
    let plan = std::env::var_os("CONCHORDAL_ENERGY_ASSAY_PLAN")
        .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_ENERGY_ASSAY_PLAN is required"))?;
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(plan)?)?;
    anyhow::ensure!(!cases.is_empty(), "no cases");
    for case in cases {
        let mut reader = hound::WavReader::open(&case.input)?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.channels == 1
                && spec.bits_per_sample == 16
                && spec.sample_format == hound::SampleFormat::Int,
            "expected mono PCM16"
        );
        let audio = reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?;
        let mut observer = AcousticTemporalExpectation::new(spec.sample_rate)
            .ok_or_else(|| anyhow::anyhow!("unsupported observation rate"))?;
        let window = observer.window.len();
        let mut out = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&case.output)?,
        );
        writeln!(
            out,
            "issued_frame,target_start_frame,target_end_frame,lead_sec,band,recurrence,persistence,predicted,observed,weight"
        )?;
        let mut elapsed = std::time::Duration::ZERO;
        let mut count = 0;
        for (step, audio) in audio.chunks_exact(window).enumerate() {
            // Copy the original predictions before the target audio is observed.
            let mut pending = Vec::new();
            for lead in [0, 1, 4, 10, 20, 50, 100, 200] {
                let Some(issued_step) = step.checked_sub(lead * super::FORECAST_STRIDE) else {
                    continue;
                };
                let issued =
                    &observer.energy_learning.pending[issued_step % super::ENERGY_PENDING_LEN];
                if issued.step == Some(issued_step as u64) {
                    pending.push((
                        issued_step,
                        lead,
                        issued.recurrence[lead],
                        issued.persistence,
                        issued.weight[lead],
                    ));
                }
            }
            let mut observed = None;
            let start = std::time::Instant::now();
            observer.process((step * window) as u64, audio, |row| observed = Some(row));
            elapsed += start.elapsed();
            let observed = observed.expect("one complete observation window");
            for (issued_step, lead, recurrence, persistence, weight) in pending {
                for band in 0..3 {
                    let predicted =
                        weight[band] * recurrence[band] + (1.0 - weight[band]) * persistence[band];
                    writeln!(
                        out,
                        "{},{},{},{:.9},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                        issued_step * window,
                        observed.start_frame,
                        observed.end_frame,
                        (lead * super::FORECAST_STRIDE * window) as f64 / spec.sample_rate as f64,
                        band,
                        recurrence[band],
                        persistence[band],
                        predicted,
                        observed.band_energy[band],
                        weight[band]
                    )?;
                    count += 1;
                }
            }
        }
        out.flush()?;
        println!(
            "energy forecast {}: rows={count} observer_sec={:.6} audio_sec={:.6} pending_bytes={}",
            case.input.display(),
            elapsed.as_secs_f64(),
            audio.len() as f64 / spec.sample_rate as f64,
            std::mem::size_of::<super::IssuedEnergyForecast>() * super::ENERGY_PENDING_LEN
        );
    }
    Ok(())
}
