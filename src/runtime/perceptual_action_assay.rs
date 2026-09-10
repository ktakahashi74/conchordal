//! Offline action consequences through the native, stateful auditory analysis.
//! Compiled only into tests; no instrument export path or public API is added.

use super::build_analysis_runtime_core;
use crate::config::AppConfig;
use crate::core::landscape::LandscapeParams;
use crate::core::stream::analysis::AnalysisStream;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    config: PathBuf,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    sample_rate: u32,
    issued_sample: usize,
    bus: String,
    past: PathBuf,
    output: PathBuf,
    branches: Vec<Branch>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Branch {
    id: String,
    origin: String,
    audio: PathBuf,
}

#[derive(Debug, PartialEq, Serialize)]
struct Observation {
    end_sample: usize,
    fft_start_sample: usize,
    startup_zero_samples: usize,
    nsgt_power_scan: Vec<f32>,
    intensity_scan: Vec<f32>,
    r_state01_scan: Vec<f32>,
    h_state01_scan: Vec<f32>,
    c_field_level_scan: Vec<f32>,
    r_state01_scalar: f32,
    loudness_mass: f32,
}

fn observe(
    stream: &mut AnalysisStream,
    params: &LandscapeParams,
    hop: usize,
    start: usize,
    audio: &[f32],
) -> Vec<Observation> {
    audio
        .chunks_exact(hop)
        .enumerate()
        .map(|(index, chunk)| {
            let mut frame = stream.process(chunk);
            frame.recompute_consonance(params);
            let end_sample = start + (index + 1) * hop;
            Observation {
                end_sample,
                fft_start_sample: end_sample.saturating_sub(stream.window_samples()),
                startup_zero_samples: stream.window_samples().saturating_sub(end_sample),
                nsgt_power_scan: frame.nsgt_power,
                intensity_scan: frame.subjective_intensity,
                r_state01_scan: frame.roughness01,
                h_state01_scan: frame.harmonicity01,
                c_field_level_scan: frame.consonance_field_level,
                r_state01_scalar: frame.roughness01_scalar,
                loudness_mass: frame.loudness_mass,
            }
        })
        .collect()
}

fn read_audio(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len().is_multiple_of(4), "incomplete f32 sample");
    let audio: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    anyhow::ensure!(audio.iter().all(|x| x.is_finite()), "non-finite audio");
    Ok(audio)
}

#[test]
#[ignore = "offline perceptual consequences; requires CONCHORDAL_PERCEPTUAL_ACTION_PLAN"]
fn export_perceptual_action_assay() -> anyhow::Result<()> {
    let path = std::env::var_os("CONCHORDAL_PERCEPTUAL_ACTION_PLAN")
        .ok_or_else(|| anyhow::anyhow!("CONCHORDAL_PERCEPTUAL_ACTION_PLAN is required"))?;
    let plan: Plan = serde_json::from_slice(&fs::read(path)?)?;
    // Reading a research plan must never create a missing config file.
    let config: AppConfig = toml::from_str(&fs::read_to_string(plan.config)?)?;
    anyhow::ensure!(!plan.cases.is_empty(), "no cases");
    for case in plan.cases {
        anyhow::ensure!(
            matches!(case.bus.as_str(), "habitat" | "presentation"),
            "unknown bus"
        );
        let mut config = config.clone();
        config.audio.sample_rate = case.sample_rate;
        config.validate()?;
        anyhow::ensure!(
            !config.psychoacoustics.habituation.enabled,
            "assay excludes habituation state"
        );
        let core = build_analysis_runtime_core(&config, case.sample_rate);
        let past = read_audio(&case.past)?;
        anyhow::ensure!(
            past.len() == case.issued_sample,
            "history must run from sample zero through issue"
        );
        let mut observed = AnalysisStream::new(core.lparams.clone(), core.nsgt);
        for chunk in past.chunks_exact(core.hop) {
            observed.process(chunk);
        }
        let processed = past.len() / core.hop * core.hop;
        let pending = &past[processed..];
        let mut before = observed.last().clone();
        before.recompute_consonance(&core.lparams);
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(case.output)?,
        );
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "type": "contract", "id": case.id, "sample_rate": case.sample_rate,
                "bus": case.bus, "issued_sample": case.issued_sample,
                "hop_samples": core.hop, "window_samples": observed.window_samples(),
                "processed_samples": processed, "pending_samples": pending.len(),
            "frequency_hz": observed.last().space.centers_hz,
            "observed_r_state01_scan": before.roughness01,
            "observed_h_state01_scan": before.harmonicity01,
            "observed_c_field_level_scan": before.consonance_field_level,
            "scope": "Native analysis with copied observed state and pending PCM. Complete hops only; final snapshot exported after all updates. FFT support is not per-band latency or cognitive duration. Habituation and listener feedback excluded. Predictive branches never update observed state."
            }),
        )?;
        writeln!(writer)?;
        for branch in case.branches {
            anyhow::ensure!(
                matches!(branch.origin.as_str(), "predictive" | "perceptual"),
                "unknown origin"
            );
            let future = read_audio(&branch.audio)?;
            let mut joined = pending.to_vec();
            joined.extend_from_slice(&future);
            let rows = observe(
                &mut observed.clone(),
                &core.lparams,
                core.hop,
                processed,
                &joined,
            );
            anyhow::ensure!(
                rows.iter().all(|row| {
                    row.nsgt_power_scan
                        .iter()
                        .chain(&row.intensity_scan)
                        .chain(&row.r_state01_scan)
                        .chain(&row.h_state01_scan)
                        .chain(&row.c_field_level_scan)
                        .chain([&row.r_state01_scalar, &row.loudness_mass])
                        .all(|x| x.is_finite())
                }),
                "non-finite perceptual result"
            );
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "type": "branch", "id": branch.id, "origin": branch.origin,
                    "target_end_sample": case.issued_sample + future.len(),
                "trailing_samples": joined.len() % core.hop,
                "processed_end_samples": rows.iter().map(|row| row.end_sample).collect::<Vec<_>>(),
                "observation": rows.last()
                }),
            )?;
            writeln!(writer)?;
        }
        writer.flush()?;
        println!("observed {}", case.id);
    }
    Ok(())
}

#[test]
fn perceptual_forks_preserve_history_pending_pcm_and_branch_independence() {
    for fs in [24_000, 48_000] {
        let mut config = AppConfig::default();
        config.analysis.nfft = 2048;
        config.analysis.hop_size = 128;
        let core = build_analysis_runtime_core(&config, fs);
        let past: Vec<_> = (0..4173)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.07 * (std::f32::consts::TAU * (400.0 * t + 90.0 * t * t)).sin()
            })
            .collect();
        let mut observed = AnalysisStream::new(core.lparams.clone(), core.nsgt.clone());
        for chunk in past.chunks_exact(core.hop) {
            observed.process(chunk);
        }
        let processed = past.len() / core.hop * core.hop;
        let before = observed.last().clone();
        let mut branches = Vec::new();
        for sign in [1.0, -1.0] {
            let future: Vec<_> = (0..379)
                .map(|i| {
                    sign * 0.06
                        * (std::f32::consts::TAU * 415.0 * (i + past.len()) as f32 / fs as f32)
                            .sin()
                })
                .collect();
            let mut joined = past[processed..].to_vec();
            joined.extend_from_slice(&future);
            let forked = observe(
                &mut observed.clone(),
                &core.lparams,
                core.hop,
                processed,
                &joined,
            );
            let mut complete = past.clone();
            complete.extend_from_slice(&future);
            let mut replay = AnalysisStream::new(core.lparams.clone(), core.nsgt.clone());
            let expected = observe(&mut replay, &core.lparams, core.hop, 0, &complete);
            assert_eq!(forked, expected[processed / core.hop..]);
            assert_eq!(
                forked.last().unwrap().end_sample,
                complete.len() / core.hop * core.hop
            );
            branches.push(forked);
        }
        assert_ne!(branches[0], branches[1]);
        assert_eq!(observed.last().nsgt_power, before.nsgt_power);
        assert_eq!(
            observed.last().subjective_intensity,
            before.subjective_intensity
        );
        let short = &past[processed..];
        assert!(
            observe(
                &mut observed.clone(),
                &core.lparams,
                core.hop,
                processed,
                short
            )
            .is_empty()
        );
        let continuation = vec![0.0; core.hop];
        let mut untouched = AnalysisStream::new(core.lparams.clone(), core.nsgt);
        for chunk in past.chunks_exact(core.hop) {
            untouched.process(chunk);
        }
        assert_eq!(
            observe(
                &mut observed,
                &core.lparams,
                core.hop,
                processed,
                &continuation
            ),
            observe(
                &mut untouched,
                &core.lparams,
                core.hop,
                processed,
                &continuation
            )
        );
    }
}

#[test]
fn perceptual_distribution_cannot_be_replaced_by_its_mean_waveform() {
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    config.analysis.hop_size = 128;
    let core = build_analysis_runtime_core(&config, 24_000);
    let audio: Vec<_> = (0..4096)
        .map(|i| {
            let t = i as f32 / 24_000.0;
            0.07 * ((std::f32::consts::TAU * 440.0 * t).sin()
                + (std::f32::consts::TAU * 470.0 * t).sin())
        })
        .collect();
    let opposite: Vec<_> = audio.iter().map(|x| -*x).collect();
    let mut stream = AnalysisStream::new(core.lparams.clone(), core.nsgt);
    let positive = observe(&mut stream.clone(), &core.lparams, core.hop, 0, &audio);
    let negative = observe(&mut stream.clone(), &core.lparams, core.hop, 0, &opposite);
    let mean = observe(
        &mut stream,
        &core.lparams,
        core.hop,
        0,
        &vec![0.0; audio.len()],
    );
    assert_eq!(positive, negative);
    let signal = positive.last().unwrap();
    let silence = mean.last().unwrap();
    assert!(signal.loudness_mass > 0.0);
    assert!(signal.r_state01_scalar > 0.0);
    assert!(signal.h_state01_scan.iter().any(|x| *x > 0.0));
    assert_eq!(silence.loudness_mass, 0.0);
    assert_eq!(silence.r_state01_scalar, 0.0);
    assert!(silence.h_state01_scan.iter().all(|x| *x == 0.0));
}
