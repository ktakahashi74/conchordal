//! Opt-in registration A-2 measurements. No acceptance thresholds.

use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, BufWriter},
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;
use crate::core::stream::analysis::AnalysisStream;

mod comparison;
pub(super) mod controls;
mod exploratory;
mod external;

const HOP: usize = 512;
const FS: u32 = 48_000;
const BOOTSTRAP_SEED: u64 = 20261020;
const RESAMPLES: usize = 10_000;
const VERSION_NAMES: [&str; 6] = ["R-S", "R-O", "N-S", "N-O", "N-P-S", "N-P-O"];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target/info-terrain-20261020")
}

fn results() -> Result<PathBuf> {
    let path = root().join("results");
    fs::create_dir_all(&path)?;
    Ok(path)
}

pub(super) fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer_pretty(BufWriter::new(fs::File::create(path)?), value)?;
    Ok(())
}

fn provenance() -> Result<Value> {
    let describe = Command::new("git")
        .args(["describe", "--always", "--dirty", "--tags"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()?;
    ensure!(describe.status.success(), "git describe failed");
    let registration = fs::read(root().join("registration-a.ja.md"))?;
    let revision = fs::read(root().join("registration-a2.ja.md"))?;
    ensure!(
        String::from_utf8_lossy(&registration)
            .starts_with("# 事前登録：予測精度と驚きの観測（機能 A）草案 第5稿"),
        "harness is pinned to registration draft 5"
    );
    ensure!(
        String::from_utf8_lossy(&revision).starts_with(
            "# 事前登録の改訂：予測精度と驚きの観測 A-2（知覚の床と拡張版の起動）草案 第3稿"
        ),
        "harness is pinned to revision A-2 draft 3"
    );
    let mut sources = BTreeMap::new();
    for relative in [
        "src/core/info_terrain.rs",
        "src/core/analysis_worker.rs",
        "src/core/spectral_history.rs",
        "src/core/stream/analysis.rs",
        "src/core/nsgt_kernel.rs",
        "src/core/nsgt_rt.rs",
        "src/runtime/mod.rs",
        "src/core/info_terrain/info_terrain_eval/mod.rs",
        "src/core/info_terrain/info_terrain_eval/controls.rs",
        "src/core/info_terrain/info_terrain_eval/external.rs",
        "src/core/info_terrain/info_terrain_eval/comparison.rs",
        "Cargo.lock",
    ] {
        sources.insert(
            relative,
            format!("{:x}", Sha256::digest(fs::read(relative)?)),
        );
    }
    let algorithm = json!({

            "window_sec": 1.0, "window_frames": "floor(1 / hop_sec), including current",
            "min_probability": 1e-12, "surprise_cap_nat": SURPRISE_CAP,
            "amplitude_floor_ratio": 1e-4, "precision_reference": 1.0,
            "rls_dimension": DIM, "rls_lambda": LAMBDA, "rls_initial_p_diagonal": 100.0,
            "rls_initial_weights": Predictor::default().w, "rls_p_trace_reset_above": 1e6,
            "rls_symmetrize_each_update": true,
            "amplitude_ema_tau_sec": 1.0, "amplitude_ema_initial": "history mean at first eligible full-window frame; retry until eligible; init frame missing without RLS/residual update",
            "residual_ema_tau_sec": 1.0, "residual_mean_initial": 0.0,
            "residual_variance_initial": amplitude_delta().powi(2),
            "perceptual_delta_db": DELTA_L_DB, "amplitude_delta": amplitude_delta(),
            "variance_floor": amplitude_delta().powi(2), "variance_floor_evaluation_only": true,
            "age_coverage_threshold": 0.5, "uncovered_normalized_feature": 1.0,
            "history_ages_sec": crate::core::temporal_history::HISTORY_AGES_SEC,
            "block_sec": 1, "queue_capacity_blocks": 64,
            "nsgt_frequency_hz": [55, 8000], "nsgt_bins_per_octave": 96,
            "nsgt_hop_samples": HOP, "runtime_config": crate::config::AppConfig::default()
    });
    let controls = json!({
            "controls_sample_rate": FS, "controls_duration_sec": 20,
            "evaluation_window_sec": [2, 20], "sine_peak_dbfs": -12,
            "reference_rms": 10_f64.powf(-12.0 / 20.0) / 2_f64.sqrt(),
            "gain_pair_db": -30, "gain_absolute_tolerance": 1e-6,
            "gain_relative_tolerance": 1e-4,
            "noise_rng": "splitmix64; upper 53 bits, open (0,1): (bits+0.5)/2^53",
            "white_seed": 1, "pink_seed": 2,
            "gaussian": "Box-Muller, cosine then sine for each pair",
            "pink_filter": {"b0_pole": 0.99765, "b0_input": 0.0990460,
                "b1_pole": 0.96300, "b1_input": 0.2965164,
                "b2_pole": 0.57000, "b2_input": 1.0526913, "direct": 0.1848,
                "initial_state": [0,0,0]},
            "noise_normalization": "whole 20 s RMS, after pink filtering",
            "segment_sec": 0.5, "segment_phase_radians": 0,
            "boundary_sec": 0.010, "boundary_half_width_sec": 0.005,
            "alternating_hz": [440, 660], "replacement_hz": 550,
            "replacement_starts_sec": [6.5,10.5,14.5,18.5],
            "flora_seed": 5, "sample_seed": 11,
            "flora_configurations": [["sine",true],["sine",false],["resonant",true]]
    });
    let external = json!({
            "external_sample_rate": 44100, "stft_window_ms": 100,
            "stft_step_ms": 20, "stft_fft_samples": 8192,
            "stft_window_samples": 4410, "stft_step_samples": 882,
            "mel_bands": 20, "mel_scale": "HTK: 2595 log10(1 + f/700)",
            "mel_range": "0 to Nyquist; 22 equally spaced mel edges",
            "stft_window": "symmetric Hann: 0.5 - 0.5 cos(2 pi n/(L-1))",
            "stft_power": "sum triangle(f_k) * |FFT_k|^2, k=0..N/2; no doubling or normalization",
            "stft_frames": "start sample 0; drop incomplete 100 ms frame",
            "S_startup": "at least 3 frames including current; short window allowed",
            "O_startup": "wait for full window; history-mean silence mask",
            "O_perceptual_floor": {"N":amplitude_delta(),"R_and_N_P":10_f64.powf(DELTA_L_DB/10.0)-1.0},
            "S_perceptual_floor": null, "S_values": "unchanged, reused from results-a1",
            "sd": "population, f64 two-pass centered sum of squares",
            "primary_aggregation": "history-only positive maxima; uniform when sum zero; NA terms dropped without renormalization; all NA -> NA",
            "auxiliary_aggregation": "window maxima including current; same NA policy; uniform when sum zero (registered OSF rule)",
            "bootstrap_unit": "prototype, first file-stem field",
            "bootstrap_resamples": RESAMPLES, "bootstrap_rng": "splitmix64, unbiased rejection sampling",
            "bootstrap_seed": BOOTSTRAP_SEED, "ci_percentiles": [0.025, 0.975],
            "ci_interpolation": "linear, index p*(n-1), R type 7",
            "contrasts": ["R-S minus N-S", "N-S minus N-P-S", "R-S minus R-O"]
    });
    Ok(json!({
        "git_describe": String::from_utf8(describe.stdout)?.trim(),
        "source_sha256": sources,
        "registration_draft": 5,
        "registration_sha256": format!("{:x}", Sha256::digest(registration)),
        "revision": "A-2", "revision_draft": 3,
        "revision_sha256": format!("{:x}", Sha256::digest(revision)),
        "token_erratum": "second underscore-separated field of file stem; prototype is first",
        "registered_values": {"function_A":algorithm,"controls":controls,"external":external},
        "limitations": [
            "Rust R input approximates melfcc; no pre-emphasis, resampling, or filterbank library; symmetric Hann and discrete unnormalized power triangles specified above",
            "MP3 leading silence and encoding delay retained",
            "Paper use of MP3 versus original source audio unconfirmed",
            "fit uncertainty (lwr/upr) not included in bootstrap confidence intervals",
            "OSF auxiliary follows the registered per-hop weighting rule, not the bundled script's final-window weight reuse or sample SD",
            "Only complete 512-sample NSGT hops; trailing samples reported, no synthetic padding"
        ]
    }))
}

pub(super) fn read_wav(path: &Path) -> Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(spec.channels > 0, "no WAV channels");
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2_f64.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|x| x.map(|v| (f64::from(v) / scale) as f32))
                .collect::<std::result::Result<_, _>>()?
        }
    };
    ensure!(
        interleaved.len().is_multiple_of(spec.channels as usize),
        "partial WAV channel frame"
    );
    let mono = interleaved
        .chunks_exact(spec.channels as usize)
        .map(|channels| {
            (channels.iter().map(|x| f64::from(*x)).sum::<f64>() / channels.len() as f64) as f32
        })
        .collect();
    Ok((spec.sample_rate, mono))
}

fn precision(terrain: &InfoTerrain, v: usize) -> Option<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (&p, &m) in terrain.versions[v]
        .precision_scan
        .iter()
        .zip(&terrain.mean_scan)
    {
        if p.is_finite() {
            numerator += p * m;
            denominator += m;
        }
    }
    (denominator > 0.0).then(|| numerator / denominator)
}

#[derive(Serialize)]
struct Evaluation {
    sample_rate: u32,
    input_samples: usize,
    trailing_samples: usize,
    evaluation_window_sec: [f64; 2],
    window_summary: Option<Block>,
    blocks: Vec<Block>,
    trace: Vec<Value>,
    gain_check: Option<Value>,
}

// Mirrors the contiguous habitat worker loop, including its initial warmup=0.
// The normal worker-equivalence test below pins this order to analysis_worker::run.
fn evaluate(
    samples: &[f32],
    fs: u32,
    stream: AnalysisStream,
    gain_factor: Option<f64>,
) -> Result<Evaluation> {
    let check_gain = gain_factor.is_some();
    let space = stream.last().space.clone();
    let mut streams = vec![stream];
    if check_gain {
        streams.push(streams[0].clone());
    }
    let mut histories: Vec<_> = streams
        .iter()
        .map(|s| SpectralHistory::new(space.clone(), f64::from(fs), s.window_samples()))
        .collect();
    let (observer, output) = Observer::new(space.clone(), fs, HOP, "habitat", true, true);
    let mut observers = vec![observer];
    let gain_output = if check_gain {
        let (observer, output) = Observer::new(space.clone(), fs, HOP, "habitat", true, true);
        observers.push(observer);
        Some(output)
    } else {
        None
    };
    let mut window = BlockAccumulator::new(space.n_bins(), true, samples.len() / HOP);
    let mut blocks = Vec::new();
    let mut trace = Vec::new();
    let mut mask_mismatches = 0_u64;
    let mut tolerance_failures = 0_u64;
    let mut maximum_tolerance_fraction = 0_f64;
    let gain = gain_factor.unwrap_or(1.0);
    let mut violations = Vec::new();
    let mut evaluation_mask_mismatches = 0_u64;
    let mut evaluation_tolerance_failures = 0_u64;
    let mut maximum_amplitude_relative_error = 0_f64;
    for (frame, audio) in samples.chunks_exact(HOP).enumerate() {
        for lane in 0..streams.len() {
            let scaled;
            let audio = if lane == 0 {
                audio
            } else {
                scaled = audio
                    .iter()
                    .map(|a| (f64::from(*a) * gain) as f32)
                    .collect::<Vec<_>>();
                &scaled
            };
            let landscape = streams[lane].process(audio);
            histories[lane].advance_to((frame * HOP) as u64);
            observers[lane].observe(frame as u64, &landscape.nsgt_power, &histories[lane], false);
            histories[lane].observe(
                (frame * HOP) as u64,
                ((frame + 1) * HOP) as u64,
                &landscape.nsgt_power,
            );
        }
        let time = (frame * HOP) as f64 / f64::from(fs);
        if check_gain {
            let lane_amplitudes = |lane: usize, bin: usize| {
                let terrain = &observers[lane].terrain;
                let latest = (terrain.cursor + terrain.history_len - 1) % terrain.history_len;
                terrain.history[latest * space.n_bins() + bin]
            };
            for b in 0..space.n_bins() {
                let ideal = lane_amplitudes(0, b) * gain;
                if ideal > 0.0 {
                    maximum_amplitude_relative_error = maximum_amplitude_relative_error
                        .max((lane_amplitudes(1, b) / ideal - 1.0).abs());
                }
            }
            for v in 0..2 {
                let a = &observers[0].terrain.versions[v];
                let b = &observers[1].terrain.versions[v];
                for (indicator, xs, ys) in [
                    ("surprise", &a.surprise_scan, &b.surprise_scan),
                    ("precision", &a.precision_scan, &b.precision_scan),
                ] {
                    for (bin, (&x, &y)) in xs.iter().zip(ys).enumerate() {
                        if x.is_finite() != y.is_finite() {
                            mask_mismatches += 1;
                            evaluation_mask_mismatches += u64::from((2.0..20.0).contains(&time));
                        }
                        if x.is_finite() && y.is_finite() {
                            let fraction = (x - y).abs() / (1e-6 + 1e-4 * x.abs().max(y.abs()));
                            maximum_tolerance_fraction = maximum_tolerance_fraction.max(fraction);
                            tolerance_failures += u64::from(fraction > 1.0);
                            if fraction > 1.0 {
                                evaluation_tolerance_failures +=
                                    u64::from((2.0..20.0).contains(&time));
                                violations.push(json!({"frame_id":frame,"time_sec":time,"bin":bin,
                                    "frequency_hz":space.freq_of_index(bin),"version":if v==0 {"original"} else {"extended"},
                                    "indicator":indicator,"values":[x,y],"absolute_difference":(x-y).abs(),
                                    "allowed_difference":1e-6+1e-4*x.abs().max(y.abs()),"tolerance_fraction":fraction,
                                    "current_amplitudes":[lane_amplitudes(0,bin),lane_amplitudes(1,bin)],
                                    "post_update_residual_variances":[observers[0].terrain.predictors[bin].residual_variance, observers[1].terrain.predictors[bin].residual_variance],
                                    "post_update_weights":[observers[0].terrain.predictors[bin].w, observers[1].terrain.predictors[bin].w]}));
                            }
                        }
                    }
                }
            }
        }
        if (2.0..20.0).contains(&time) {
            window.add(&observers[0].terrain, frame as u64, 0, false);
        }
        let terrain = &observers[0].terrain;
        trace.push(
            json!({"frame_id": frame, "start_sample": frame*HOP, "time_sec": time,
            "surprise": terrain.versions.each_ref().map(|v| [
                integrate(&space, &v.surprise_scan, &terrain.soundgen_weight_scan),
                integrate(&space, &v.surprise_scan, &terrain.osf_weight_scan)]),
            "precision": [precision(terrain, 0), precision(terrain, 1)]}),
        );
        blocks.extend(output.rx.try_iter());
        if let Some(output) = &gain_output {
            for _ in output.rx.try_iter() {}
        }
    }
    let window_summary = window.finish(&Arc::new(space), "habitat", HOP, true);
    drop(observers);
    blocks.extend(output.rx.iter());
    ensure!(output.dropped.load(Ordering::Relaxed) == 0, "lost blocks");
    if let Some(output) = gain_output {
        for _ in output.rx.iter() {}
        ensure!(
            output.dropped.load(Ordering::Relaxed) == 0,
            "lost gain blocks"
        );
    }
    Ok(Evaluation {
        sample_rate: fs, input_samples: samples.len(), trailing_samples: samples.len()%HOP,
        evaluation_window_sec: [2.0,20.0], window_summary, blocks, trace,
        gain_check: check_gain.then(|| json!({"mask_mismatches": mask_mismatches,
            "tolerance_failures": tolerance_failures, "maximum_tolerance_fraction": maximum_tolerance_fraction,
            "gain_factor":gain,"maximum_nsgt_amplitude_relative_error":maximum_amplitude_relative_error,
            "evaluation_window":{"mask_mismatches":evaluation_mask_mismatches,"tolerance_failures":evaluation_tolerance_failures},
            "violations":violations,"scope": "every bin and hop, both indicators and versions, including startup"})),
    })
}

#[test]
fn replay_matches_habitat_worker_blocks() -> Result<()> {
    let stream = crate::runtime::info_terrain_eval_stream(FS);
    let audio: Vec<_> = (0..HOP * 200)
        .map(|i| (i as f32 * 0.1).sin() * 0.1)
        .collect();
    let evaluation = evaluate(&audio, FS, stream.clone(), None)?;
    let (observer, output) =
        Observer::new(stream.last().space.clone(), FS, HOP, "habitat", true, true);
    let (hop_tx, hop_rx) = crossbeam_channel::unbounded();
    let (result_tx, _result_rx) = crossbeam_channel::unbounded();
    let (_update_tx, update_rx) = crossbeam_channel::unbounded();
    for (frame, chunk) in audio.chunks_exact(HOP).enumerate() {
        hop_tx.send((frame as u64, Arc::from(chunk)))?;
    }
    drop(hop_tx);
    crate::core::analysis_worker::run(
        stream,
        hop_rx,
        result_tx,
        update_rx,
        crate::core::analysis_worker::AnalysisDelivery::Latest,
        None,
        Some(observer),
    );
    let blocks: Vec<_> = output.rx.iter().collect();
    assert_eq!(
        serde_json::to_value(&evaluation.blocks)?,
        serde_json::to_value(blocks)?
    );
    Ok(())
}

fn info_terrain_eval_wav() -> Result<()> {
    let input = if let Some(path) = std::env::var_os("CONCHORDAL_INFO_TERRAIN_WAV") {
        PathBuf::from(path)
    } else {
        let mut paths = fs::read_dir(root().join("anikin-stimuli-wav"))?
            .map(|p| p.map(|p| p.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.retain(|p| p.extension().is_some_and(|e| e == "wav"));
        paths.sort();
        paths.into_iter().next().context("no prepared WAVs")?
    };
    let (fs, samples) = read_wav(&input)?;
    let evaluation = evaluate(
        &samples,
        fs,
        crate::runtime::info_terrain_eval_stream(fs),
        None,
    )?;
    write_json(
        &results()?.join("wav.json"),
        &json!({
            "provenance": provenance()?, "input": input, "evaluation": evaluation,
            "wav_sha256": format!("{:x}", Sha256::digest(fs::read(&input)?))
        }),
    )?;
    fs::write(
        results()?.join("wav.md"),
        format!(
            "# WAV replay\n\nInput: `{}`; {} Hz; {} samples; {} complete hops; {} trailing samples pending.\n\nFull one-second aggregates and per-hop traces: `wav.json`. Optional [2,20) s summary is null for files ending before the evaluation window.\n",
            input.display(),
            fs,
            samples.len(),
            samples.len() / HOP,
            samples.len() % HOP
        ),
    )?;
    Ok(())
}

fn info_terrain_eval_renders() -> Result<()> {
    let output = results()?.join("renders");
    fs::create_dir_all(&output)?;
    let build = Command::new("cargo")
        .args([
            "build",
            "--offline",
            "--release",
            "--bin",
            "conchordal-render",
        ])
        .status()?;
    ensure!(build.success(), "renderer build failed");
    let binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/conchordal-render");
    let mut cases = Vec::new();
    for (name, tone, residual) in [
        ("flora_sine_residual", "sine", true),
        ("flora_sine_only", "sine", false),
        ("flora_resonant_residual", "resonant", true),
    ] {
        let source = fs::read_to_string("samples/13_flora.rhai")?;
        ensure!(
            source.matches(".fade(2)").count() == 1,
            "Flora variant substitution ambiguous"
        );
        cases.push((
            name.to_string(),
            source.replace(
                ".fade(2)",
                &format!(".fade(2).tone(\"{tone}\").residual({residual})"),
            ),
            5,
        ));
    }
    for name in ["03_gravity", "09_rain", "12_emergence_and_resolution"] {
        let mut script = fs::read_to_string(format!("samples/{name}.rhai"))?;
        if name == "03_gravity" {
            script.push_str("\nwait(6.0);\n");
        }
        cases.push((name.to_string(), script, 11));
    }
    let mut summaries = Vec::new();
    for (name, script, seed) in cases {
        let dir = output.join(&name);
        fs::create_dir_all(&dir)?;
        fs::write(dir.join("scenario.rhai"), script)?;
        let mut config = crate::config::AppConfig::default();
        config.info_terrain.enabled = true;
        config.info_terrain.trace = true;
        fs::write(dir.join("config.toml"), toml::to_string_pretty(&config)?)?;
        let arguments = vec![
            dir.join("scenario.rhai").display().to_string(),
            "--config".into(),
            dir.join("config.toml").display().to_string(),
            "--seed".into(),
            seed.to_string(),
            "--output".into(),
            dir.join("audio.wav").display().to_string(),
            "--report".into(),
            dir.join("report.jsonl").display().to_string(),
        ];
        let run = Command::new(&binary)
            .args(&arguments)
            .env("RUST_LOG", "warn")
            .stdout(fs::File::create(dir.join("stdout.log"))?)
            .stderr(fs::File::create(dir.join("stderr.log"))?)
            .status()?;
        ensure!(run.success(), "render failed: {name}");
        let mut blocks = Vec::new();
        let mut terminal = None;
        for line in BufReader::new(fs::File::open(dir.join("report.jsonl"))?).lines() {
            let record: Value = serde_json::from_str(&line?)?;
            if record["type"] == "info_terrain_block" && record["stream"] == "habitat" {
                ensure!(record["input_gap"] == false, "habitat gap");
                ensure!(
                    record["hop_start"].as_u64()
                        == Some(
                            blocks
                                .last()
                                .map_or(0, |v: &Value| v["hop_end_exclusive"].as_u64().unwrap())
                        ),
                    "habitat hop gap"
                );
                ensure!(
                    record["trace"].as_array().context("trace absent")?.len() as u64
                        == record["hops"].as_u64().unwrap(),
                    "incomplete trace"
                );
                blocks.push(record);
            } else if record["type"] == "info_terrain_summary" {
                terminal = Some(record);
            }
        }
        ensure!(!blocks.is_empty(), "no habitat blocks");
        ensure!(
            terminal.as_ref().is_some_and(|r| r["dropped_blocks"] == 0),
            "missing terminal summary or dropped blocks"
        );
        let mut versions = BTreeMap::new();
        // Block means must be combined with their indicator-specific defined-hop counts.
        for version in ["original", "extended"] {
            let mut surprise = Moments::default();
            let mut precision = Moments::default();
            for block in &blocks {
                let start = block["start_sample"].as_u64().unwrap() as f64 / f64::from(FS);
                if !(2.0..20.0).contains(&start) {
                    continue;
                }
                let v = &block[version];
                if let Some(mean) = v["soundgen"]["mean"].as_f64() {
                    let count = v["soundgen"]["defined_hops"].as_u64().unwrap();
                    surprise.sum += mean * count as f64;
                    surprise.count += count;
                }
                if let Some(mean) = v["precision_mean"].as_f64() {
                    let count = v["precision_defined_hops"].as_u64().unwrap();
                    precision.sum += mean * count as f64;
                    precision.count += count;
                }
            }
            versions.insert(version, json!({"mean_surprise": surprise.mean(), "mean_precision": precision.mean(),
                "surprise_defined_hops": surprise.count, "precision_defined_hops": precision.count}));
        }
        write_json(&dir.join("habitat.json"), &blocks)?;
        summaries.push(
            json!({"stimulus": name,"seed": seed,"argv": arguments,"evaluation_window_sec": [2,20], "terminal_extension_sec": if name=="03_gravity" {6} else {0},
            "versions": versions,"habitat_blocks": blocks.len(),"terminal": terminal}),
        );
        println!("render {name}: {}", versions["original"]);
    }
    write_json(
        &results()?.join("renders.json"),
        &json!({"provenance": provenance()?,"cases":summaries}),
    )?;
    let mut markdown = String::from(
        "# Render measurements\n\nEvaluation: defined habitat hops starting in [2,20) s. Gravity receives a 6 s terminal wait to reach 20 s; other scripts render their full duration, with the excess excluded.\n\n| Stimulus | Original S / precision | Extended S / precision |\n|---|---:|---:|\n",
    );
    for row in &summaries {
        markdown.push_str(&format!(
            "| {} | {} / {} | {} / {} |\n",
            row["stimulus"].as_str().unwrap(),
            row["versions"]["original"]["mean_surprise"],
            row["versions"]["original"]["mean_precision"],
            row["versions"]["extended"]["mean_surprise"],
            row["versions"]["extended"]["mean_precision"]
        ));
    }
    fs::write(results()?.join("renders.md"), markdown)?;
    Ok(())
}

#[test]
#[ignore = "registration A-2: controls, renders, external comparison, WAV replay, then paired A-1/A-2 report"]
fn info_terrain_eval_registered_suite() -> Result<()> {
    controls::info_terrain_eval_controls()?;
    info_terrain_eval_renders()?;
    external::info_terrain_eval_external()?;
    info_terrain_eval_wav()?;
    comparison::write_report()
}

pub(super) struct SplitMix64(pub(super) u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub(super) fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
    fn index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = self.next();
            if x >= threshold {
                return (x % n) as usize;
            }
        }
    }
}
