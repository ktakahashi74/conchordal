//! Opt-in offline campaigns. Measurements are not acceptance when margins are absent.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;
use serde_json::{Value, json};

const CONFIGURATIONS: &[(&str, &str, bool)] = &[
    ("sine_residual", "sine", true),
    ("sine_only", "sine", false),
    ("resonant_residual", "resonant", true),
];
const PROTOCOL: &str = include_str!("fixtures/flora_mixing_protocol.json");

#[derive(Deserialize)]
struct Protocol {
    tuning_seed: u64,
    evaluation_seeds: [u64; 3],
    margins: Option<[f64; 2]>,
    measurement_window_sec: [usize; 2],
}

fn render(sample: &str, name: &str, tone: &str, residual: bool, campaign: &str) -> PathBuf {
    let output = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/flora-assays")
        .join(campaign)
        .join(sample)
        .join(name);
    fs::create_dir_all(&output).unwrap();
    let source = fs::read_to_string(format!("samples/research/{sample}.rhai")).unwrap();
    let script = source.replace(
        ".fade(2)",
        &format!(".fade(2).tone(\"{tone}\").residual({residual})"),
    );
    fs::write(output.join("scenario.rhai"), script).unwrap();
    fs::write(
        output.join("config.toml"),
        "[audio]\nsample_rate = 48000\n[analysis]\nnfft = 2048\nhop_size = 512\n",
    )
    .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg(output.join("scenario.rhai"))
        .arg("--config")
        .arg(output.join("config.toml"))
        .arg("-o")
        .arg(output.join("audio.wav"))
        .arg("--report")
        .arg(output.join("report.jsonl"))
        .output()
        .expect("run offline renderer");
    fs::write(output.join("render.log"), &run.stderr).unwrap();
    assert!(run.status.success(), "render failed: {}", output.display());
    println!("rendered {}", output.display());
    output
}

fn per_second(output: &Path) -> Vec<Value> {
    let mut previous: BTreeMap<u64, Value> = BTreeMap::new();
    let mut records = Vec::new();
    let mut writer = BufWriter::new(fs::File::create(output.join("per_second.jsonl")).unwrap());
    for line in fs::read_to_string(output.join("report.jsonl"))
        .unwrap()
        .lines()
    {
        let mut row: Value = serde_json::from_str(line).unwrap();
        if row["type"] != "flora" {
            continue;
        }
        let id = row["flora_id"].as_u64().unwrap();
        let age = row["age_sec"].as_f64().unwrap();
        let old = previous.get(&id);
        let elapsed = age - old.map_or(0.0, |v| v["age_sec"].as_f64().unwrap());
        for key in ["births", "retirements", "flapping"] {
            let count = row[key].as_u64().unwrap();
            let prior = old.map_or(0, |v| v[key].as_u64().unwrap());
            assert!(count >= prior, "nonmonotone Flora counter");
            row[format!("{key}_in_second")] = json!(count - prior);
        }
        for (energy, power) in [
            ("tonal_energy", "tonal_power"),
            ("residual_energy", "residual_power"),
        ] {
            let prior = old.map_or(0.0, |v| v[energy].as_f64().unwrap());
            row[power] = json!((row[energy].as_f64().unwrap() - prior) / elapsed);
        }
        let density = row["density_scan"].as_array().unwrap();
        assert_eq!(density.len(), row["frequency_hz"].as_array().unwrap().len());
        let total: f64 = density.iter().map(|v| v.as_f64().unwrap()).sum();
        assert!((total - 1.0).abs() < 1e-5, "latent PMF mass {total}");
        serde_json::to_writer(&mut writer, &row).unwrap();
        writeln!(writer).unwrap();
        previous.insert(id, row.clone());
        records.push(row);
    }
    writer.flush().unwrap();
    assert!(!records.is_empty(), "missing Flora telemetry");
    records
}

fn fractions(points: impl Iterator<Item = (f64, f64)>, f0: f64, extent: [f64; 2]) -> [f64; 2] {
    let erb = |f: f64| 21.4 * (1.0 + 0.00437 * f).log10();
    let mut total = 0.0;
    let mut near = 0.0;
    let mut lattice = 0.0;
    for (hz, mass) in points {
        if hz < extent[0] || hz > extent[1] {
            continue;
        }
        assert!(mass.is_finite() && mass >= 0.0);
        total += mass;
        if (1..=16).any(|n| (erb(hz) - erb(f0 * n as f64)).abs() <= 0.5) {
            near += mass;
        }
        let multiple = (hz / (f0 / 2.0)).round().max(1.0);
        if (1200.0 * (hz / (multiple * f0 / 2.0)).log2()).abs() <= 25.0 {
            lattice += mass;
        }
    }
    assert!(total > 0.0, "silent or empty measurement extent");
    [near / total, lattice / total]
}

fn pass(values: [f64; 2], uniform: [f64; 2], margins: [f64; 2]) -> bool {
    uniform[0] - values[0] > margins[0] && values[1] - uniform[1] > margins[1]
}

fn mixing(campaign: &str, evaluation: bool) {
    let protocol: Protocol = serde_json::from_str(PROTOCOL).unwrap();
    assert!(!protocol.evaluation_seeds.contains(&protocol.tuning_seed));
    if let Some(margins) = protocol.margins {
        assert!(margins.iter().all(|x| x.is_finite() && *x >= 0.0));
    }
    // Freeze the complete registration before launching even the first render.
    let root = Path::new("target/flora-assays").join(campaign);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("protocol.json"), PROTOCOL).unwrap();
    let samples: Vec<_> = if evaluation {
        (1..=3)
            .map(|n| format!("flora_mixing_evaluation_{n}"))
            .collect()
    } else {
        vec!["flora_mixing_tuning".into()]
    };
    let f0 = if evaluation { 146.83 } else { 110.0 };
    let extent = [f0, if evaluation { 2349.3 } else { 1760.0 }];
    let [begin, end] = protocol.measurement_window_sec;
    let mut results = Vec::new();
    for (i, sample) in samples.iter().enumerate() {
        let seed = if evaluation {
            protocol.evaluation_seeds[i]
        } else {
            protocol.tuning_seed
        };
        let source = fs::read_to_string(format!("samples/research/{sample}.rhai")).unwrap();
        assert!(source.contains(&format!("seed({seed});")));
        for &(name, tone, residual) in CONFIGURATIONS {
            let output = render(sample, name, tone, residual, campaign);
            let rows = per_second(&output);
            let hz: Vec<_> = rows[0]["frequency_hz"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let uniform = fractions(hz.iter().map(|&f| (f, 1.0)), f0, extent);
            let selected: Vec<_> = rows
                .iter()
                .filter(|row| {
                    let second = row["time_sec"].as_f64().unwrap().floor() as usize;
                    second > begin && second <= end
                })
                .collect();
            assert_eq!(selected.len(), end - begin);
            let mut mass = vec![0.0; hz.len()];
            for row in selected {
                for (sum, value) in mass.iter_mut().zip(row["density_scan"].as_array().unwrap()) {
                    *sum += value.as_f64().unwrap();
                }
            }
            let latent = fractions(hz.iter().copied().zip(mass), f0, extent);
            let reader = hound::WavReader::open(output.join("audio.wav")).unwrap();
            let fs = reader.spec().sample_rate as usize;
            let pcm: Vec<_> = reader
                .into_samples::<i16>()
                .map(|s| s.unwrap() as f64 / 32768.0)
                .collect();
            assert!(pcm.len() >= end * fs);
            let nfft = 65536;
            let fft = FftPlanner::new().plan_fft_forward(nfft);
            let mut spectrum = vec![0.0; nfft / 2 + 1];
            let mut per_second_fft =
                BufWriter::new(fs::File::create(output.join("spectrum_per_second.jsonl")).unwrap());
            for second in begin..end {
                let mut bins = vec![Complex::new(0.0, 0.0); nfft];
                for n in 0..fs {
                    let window = 0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / fs as f64).cos();
                    bins[n].re = pcm[second * fs + n] * window;
                }
                fft.process(&mut bins);
                let powers: Vec<_> = bins[..spectrum.len()]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .collect();
                for (sum, power) in spectrum.iter_mut().zip(&powers) {
                    *sum += power;
                }
                let values = fractions(
                    powers
                        .into_iter()
                        .enumerate()
                        .map(|(k, p)| (k as f64 * fs as f64 / nfft as f64, p)),
                    f0,
                    extent,
                );
                serde_json::to_writer(&mut per_second_fft, &json!({"second": second, "near_partials": values[0], "common_root_lattice": values[1]})).unwrap();
                writeln!(per_second_fft).unwrap();
            }
            per_second_fft.flush().unwrap();
            let rendered = fractions(
                spectrum
                    .into_iter()
                    .enumerate()
                    .map(|(k, p)| (k as f64 * fs as f64 / nfft as f64, p)),
                f0,
                extent,
            );
            let accepted = protocol
                .margins
                .map(|m| pass(latent, uniform, m) && pass(rendered, uniform, m));
            let row = json!({"seed": seed, "configuration": name, "uniform": uniform,
                "latent": latent, "rendered": rendered, "accepted": accepted,
                "fraction_order": ["within_0.5_ERB_of_drone_partials", "within_25_cents_of_f0_over_2_lattice"]});
            println!("{row}");
            results.push(row);
        }
    }
    fs::write(
        root.join("mixing.json"),
        serde_json::to_string_pretty(&results).unwrap(),
    )
    .unwrap();
    if evaluation && protocol.margins.is_some() {
        assert!(
            results.iter().all(|r| r["accepted"] == true),
            "mixing margins failed; see {}",
            root.display()
        );
    }
}

#[test]
fn mixing_masks_and_pass_rule_distinguish_avoidance_from_lattice_alignment() {
    assert_eq!(
        fractions([(110.0, 1.0)].into_iter(), 110.0, [110.0, 1760.0]),
        [1.0, 1.0]
    );
    assert_eq!(
        fractions([(165.0, 1.0)].into_iter(), 110.0, [110.0, 1760.0]),
        [0.0, 1.0]
    );
    assert!(pass([0.1, 0.8], [0.5, 0.2], [0.2, 0.3]));
    assert!(!pass([0.4, 0.8], [0.5, 0.2], [0.2, 0.3]));
    assert!(!pass([0.1, 0.3], [0.5, 0.2], [0.2, 0.3]));
    assert!(!pass([0.25, 0.5], [0.5, 0.25], [0.25, 0.25]));
}

#[test]
#[ignore = "nine offline renders; writes target/flora-assays/closed-loop"]
fn closed_loop() {
    for sample in ["flora_lone", "flora_low", "flora_overlap"] {
        for &(name, tone, residual) in CONFIGURATIONS {
            let output = render(sample, name, tone, residual, "closed-loop");
            let records = per_second(&output);
            assert!(records.len() >= if sample == "flora_overlap" { 64 } else { 32 });
        }
    }
}

#[test]
#[ignore = "tuning only: 110 Hz, seed 41101; writes target/flora-assays/tuning"]
fn mixing_tuning() {
    mixing("tuning", false);
}

#[test]
#[ignore = "held-out measurement only: margins unregistered; writes target/flora-assays/evaluation"]
fn mixing_evaluation() {
    mixing("evaluation", true);
}
