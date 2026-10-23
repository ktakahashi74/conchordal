//! Opt-in offline campaigns. Measurements are not acceptance when margins are absent.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use conchordal::config::AppConfig;
use conchordal::scenario::{Action, FloraTone};
use conchordal::scripting::ScriptHost;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CONFIGURATIONS: &[(&str, &str, bool)] = &[
    ("sine_residual", "sine", true),
    ("sine_only", "sine", false),
    ("resonant_residual", "resonant", true),
];
const PROTOCOL: &str = include_str!("fixtures/flora_mixing_protocol.json");
// All current campaigns use the default front end and habituation.
fn output_root() -> PathBuf {
    std::env::var_os("FLORA_ASSAY_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/flora-assays"))
}

#[derive(Deserialize)]
struct Protocol {
    tuning_seed: u64,
    evaluation_seeds: [u64; 3],
    margins: Option<[f64; 2]>,
    measurement_window_sec: [usize; 2],
}

fn render(sample: &str, name: &str, tone: &str, residual: bool, campaign: &str) -> PathBuf {
    let output = output_root().join(campaign).join(sample).join(name);
    fs::create_dir_all(&output).unwrap();
    let source = fs::read_to_string(format!("samples/research/{sample}.rhai")).unwrap();
    assert!(
        source.contains(".fade(2).residual(true)"),
        "missing renderer substitution point"
    );
    let script = source.replace(
        ".fade(2).residual(true)",
        &format!(".fade(2).tone(\"{tone}\").residual({residual})"),
    );
    fs::write(output.join("scenario.rhai"), script).unwrap();
    fs::write(
        output.join("config.toml"),
        "[audio]\nsample_rate = 48000\n[analysis]\nnfft = 16384\nhop_size = 512\n",
    )
    .unwrap();
    let config =
        AppConfig::load_or_default(Some(output.join("config.toml").to_str().unwrap())).unwrap();
    assert_eq!(config.analysis.nfft, 16384);
    assert_eq!(config.analysis.hop_size, 512);
    assert!(config.psychoacoustics.habituation.enabled);
    fs::write(
        output.join("resolved-config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let (scenario, warnings) =
        ScriptHost::load_script_with_warnings(output.join("scenario.rhai").to_str().unwrap(), None)
            .unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let settings: Vec<_> = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .filter_map(|action| {
            if let Action::PlaceFlora { spec, .. } = action {
                assert_eq!(
                    spec.tone,
                    if tone == "sine" {
                        FloraTone::Sine
                    } else {
                        FloraTone::Resonant
                    }
                );
                assert_eq!(spec.residual, residual);
                Some(
                    json!({"tone": tone, "residual": spec.residual, "amp": spec.amp,
                "rate": spec.rate, "renewal_sec": spec.renewal_sec, "fade_sec": spec.fade_sec}),
                )
            } else {
                None
            }
        })
        .collect();
    assert!(!settings.is_empty(), "missing Flora placement");
    fs::write(
        output.join("resolved-scenario.json"),
        serde_json::to_vec_pretty(&json!({"seed": scenario.seed, "flora": settings})).unwrap(),
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

#[test]
#[ignore = "three six-second real-producer renders before the full campaigns"]
fn producer_compatibility() {
    let mut results = Vec::new();
    for &(name, tone, residual) in CONFIGURATIONS {
        let output = render(
            "flora_producer_compatibility",
            name,
            tone,
            residual,
            "producer-compatibility",
        );
        let rows = per_second(&output);
        assert!(rows.len() >= 4);
        assert!(rows.windows(2).all(|pair| {
            pair[1]["time_sec"].as_f64().unwrap().floor()
                > pair[0]["time_sec"].as_f64().unwrap().floor()
        }));
        assert!(
            rows.iter()
                .any(|r| r["tonal_power"].as_f64().unwrap() > 0.0)
        );
        assert_eq!(
            rows.iter()
                .any(|r| r["residual_power"].as_f64().unwrap() > 0.0),
            residual
        );
        for row in &rows {
            for partial in row["rendered_partials"].as_array().unwrap() {
                assert_eq!(partial.as_array().unwrap().len(), 3);
                assert!(partial[2].is_boolean());
                if !residual {
                    assert_eq!(partial[2], true);
                }
            }
            let partials: Vec<_> = row["rendered_partials"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
                .collect();
            let grid: Vec<_> = (0..=500).map(|i| i as f64 * 0.05).collect();
            if let Some(spectrum) = rendered_spectrum(&partials, &grid) {
                assert_eq!(js_distance(&spectrum, &spectrum), 0.0);
            }
        }
        let hash: String = Sha256::digest(fs::read(output.join("audio.wav")).unwrap())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert!(results.iter().all(|r: &Value| r["wav_sha256"] != hash));
        results.push(json!({"configuration": name, "records": rows.len(), "wav_sha256": hash,
            "resolved_config": output.join("resolved-config.json"), "resolved_scenario": output.join("resolved-scenario.json")}));
    }
    fs::write(
        output_root().join("producer-compatibility/summary.json"),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .unwrap();
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
        // Lifecycle snapshots must not overweight seconds with more turnover.
        if old.is_some_and(|prior| {
            prior["time_sec"].as_f64().unwrap().floor() == row["time_sec"].as_f64().unwrap().floor()
        }) {
            continue;
        }
        let elapsed = age - old.map_or(0.0, |v| v["age_sec"].as_f64().unwrap());
        row["interval_sec"] = json!(elapsed);
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
        let biomass = row["biomass_mass_scan"].as_array().unwrap();
        assert_eq!(biomass.len(), row["frequency_hz"].as_array().unwrap().len());
        let total: f64 = biomass.iter().map(|v| v.as_f64().unwrap()).sum();
        assert!(total.is_finite() && total >= 0.0, "biomass total {total}");
        let reported = row["total_biomass_mass"].as_f64().unwrap();
        assert!((total - reported).abs() <= 1e-5 * total.max(1.0));
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

#[cfg(test)]
fn pass(values: [f64; 2], uniform: [f64; 2], margins: [f64; 2]) -> bool {
    uniform[0] - values[0] > margins[0] && values[1] - uniform[1] > margins[1]
}

fn mixing(campaign: &str, evaluation: bool) {
    let protocol: Protocol = serde_json::from_str(PROTOCOL).unwrap();
    assert!(!protocol.evaluation_seeds.contains(&protocol.tuning_seed));
    assert!(
        protocol.margins.is_none(),
        "measurement-only protocol; acceptance requires a separate registration"
    );
    // Freeze the complete registration before launching even the first render.
    let root = output_root().join(campaign);
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
                for (sum, value) in mass
                    .iter_mut()
                    .zip(row["biomass_mass_scan"].as_array().unwrap())
                {
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
            let row = json!({"seed": seed, "configuration": name, "uniform": uniform,
                "latent": latent, "rendered": rendered, "accepted": null, "status": "measurement_only",
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

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0;
    for value in values {
        assert!(value.is_finite(), "non-finite metric: {value}");
        sum += value;
        count += 1;
    }
    (count > 0).then(|| {
        let result = sum / count as f64;
        assert!(result.is_finite());
        result
    })
}

fn rendered_spectrum(partials: &[(f64, f64)], grid: &[f64]) -> Option<Vec<f64>> {
    let mut spectrum = vec![0.0; grid.len()];
    let total: f64 = partials.iter().map(|&(_, power)| power).sum();
    assert!(total.is_finite());
    if total == 0.0 {
        return None;
    }
    for &(hz, power) in partials {
        assert!(hz.is_finite() && hz > 0.0 && power.is_finite() && power >= 0.0);
        let center = 21.4 * (1.0 + 0.00437 * hz).log10();
        let gaussian = |erb: f64| (-0.5 * ((erb - center) / 0.25).powi(2)).exp();
        let weight: f64 = grid.iter().map(|&erb| gaussian(erb)).sum();
        for (bin, &erb) in spectrum.iter_mut().zip(grid) {
            *bin += power / total * gaussian(erb) / weight;
        }
    }
    let sum: f64 = spectrum.iter().sum();
    for bin in &mut spectrum {
        *bin /= sum;
        assert!(bin.is_finite());
    }
    Some(spectrum)
}

fn js_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let divergence: f64 = a
        .iter()
        .zip(b)
        .map(|(&p, &q)| {
            let m = (p + q) / 2.0;
            // Entropy differences avoid ratios overflowing in Gaussian tails.
            let entropy = |x: f64| if x > 0.0 { -x * x.log2() } else { 0.0 };
            entropy(m) - (entropy(p) + entropy(q)) / 2.0
        })
        .sum();
    let distance = divergence.max(0.0).sqrt();
    assert!(divergence.is_finite() && distance.is_finite());
    distance
}

#[test]
fn succession_distance_uses_base_two_and_skips_silence() {
    assert_eq!(js_distance(&[1.0, 0.0], &[1.0, 0.0]), 0.0);
    assert_eq!(js_distance(&[1.0, 0.0], &[0.0, 1.0]), 1.0);
    assert!((js_distance(&[1.0, 0.0], &[0.5, 0.5]) - 0.5579230452841438).abs() < 1e-12);
    assert_eq!(js_distance(&[1.0, f64::from_bits(1)], &[0.0, 1.0]), 1.0);
    assert_eq!(js_distance(&[1.0, f64::from_bits(1)], &[1.0, 0.0]), 0.0);
    assert!(rendered_spectrum(&[], &[1.0, 1.05]).is_none());
    assert!(rendered_spectrum(&[(110.0, 0.0)], &[1.0, 1.05]).is_none());
}

#[test]
fn succession_spectrum_preserves_edge_power_and_erb_width() {
    let erb = |hz: f64| 21.4 * (1.0 + 0.00437 * hz).log10();
    let grid: Vec<_> = (0..=200).map(|i| erb(110.0) + i as f64 * 0.05).collect();
    let middle_hz = (10.0_f64.powf((grid[0] + 5.0) / 21.4) - 1.0) / 0.00437;
    let spectrum = rendered_spectrum(&[(110.0, 1.0), (middle_hz, 1.0)], &grid).unwrap();
    assert!((spectrum.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((spectrum[..50].iter().sum::<f64>() - 0.5).abs() < 1e-12);
    let centered = rendered_spectrum(&[(middle_hz, 3.0)], &grid).unwrap();
    let variance: f64 = centered
        .iter()
        .zip(&grid)
        .map(|(&p, &x)| p * (x - erb(middle_hz)).powi(2))
        .sum();
    assert!((variance - 0.25_f64.powi(2)).abs() < 1e-12);
}

#[test]
#[ignore = "24 offline renders; writes target/flora-assays/succession"]
fn succession() {
    let root = output_root().join("succession");
    fs::create_dir_all(&root).unwrap();
    let mut results = Vec::new();
    let mut markdown = String::from(
        "# Flora succession measurements\n\n\
         Samples: `flora_succession_<sample>.rhai`. Configurations: sin+r = sine/residual, sin = sine only, res+r = resonant/residual. No pass thresholds.\n\n\
         Main window [22, release) s; release = 102 s for drone/mute, 88 s otherwise. JS uses non-silent endpoints after age 20 s. Rec excludes seconds without an eligible predecessor. Lattice: drone = 110 Hz [22,42), 110/146.83 Hz [57,102); others = drone lattice over the main window. Mute level cells list [22,42), [42,62), [67,102). Powers are lane proxies; combined rows pool partials without cross terms.\n\n\
         | Sample | Cfg | Flora | N | JS 1/5/20 | Rec | PR | Lattice | Lanes mean/max; flap/s | T/R power | Biomass min/max; full max | Silent |\n\
         |---|---|---|---:|---|---:|---:|---|---|---|---|---|\n",
    );
    for sample in [
        "drone",
        "mute",
        "population",
        "habitat_only",
        "presentation_only",
        "rate8",
        "low",
        "overlap",
    ] {
        let release = if matches!(sample, "drone" | "mute") {
            102
        } else {
            88
        };
        let f0 = if sample == "low" { 55.0 } else { 110.0 };
        let lattice_roots = if sample == "drone" {
            vec![110.0, 146.83]
        } else {
            vec![f0]
        };
        for &(configuration, tone, residual) in CONFIGURATIONS {
            let name = format!("flora_succession_{sample}");
            let output = render(&name, configuration, tone, residual, "succession");
            let mut groups = BTreeMap::<String, Vec<Value>>::new();
            for row in per_second(&output) {
                groups
                    .entry(row["flora_id"].to_string())
                    .or_default()
                    .push(row);
            }
            if sample == "overlap" {
                let mut combined = BTreeMap::<u64, Value>::new();
                for rows in groups.values() {
                    for row in rows {
                        let second = row["time_sec"].as_f64().unwrap().floor() as u64;
                        if let Some(sum) = combined.get_mut(&second) {
                            for key in [
                                "births_in_second",
                                "retirements_in_second",
                                "flapping_in_second",
                                "tonal_power",
                                "residual_power",
                                "total_biomass_mass",
                            ] {
                                sum[key] =
                                    json!(sum[key].as_f64().unwrap() + row[key].as_f64().unwrap());
                            }
                            sum["rendered_partials"].as_array_mut().unwrap().extend(
                                row["rendered_partials"].as_array().unwrap().iter().cloned(),
                            );
                        } else {
                            combined.insert(second, row.clone());
                        }
                    }
                }
                let rows: Vec<_> = combined
                    .into_values()
                    .map(|mut row| {
                        row["extent_hz"] = json!([110.0, 1760.0]);
                        row
                    })
                    .collect();
                groups.insert("combined".into(), rows);
            }
            for (id, rows) in groups {
                assert!(
                    rows.len() >= 80,
                    "{name}/{configuration}/{id}: fewer than 80 records"
                );
                let extent = [
                    rows[0]["extent_hz"][0].as_f64().unwrap(),
                    rows[0]["extent_hz"][1].as_f64().unwrap(),
                ];
                let erb = |hz: f64| 21.4 * (1.0 + 0.00437 * hz).log10();
                let grid: Vec<_> = (0..)
                    .map(|i| erb(extent[0]) + i as f64 * 0.05)
                    .take_while(|&x| x <= erb(extent[1]))
                    .collect();
                let mut seconds = Vec::new();
                for row in &rows {
                    let number = |key: &str| row[key].as_f64().unwrap();
                    let partials: Vec<_> = row["rendered_partials"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
                        .collect();
                    let power: f64 = partials.iter().map(|p| p.1).sum();
                    let participation = (power > 0.0)
                        .then(|| power * power / partials.iter().map(|p| p.1 * p.1).sum::<f64>());
                    let lattice: Vec<_> = lattice_roots
                        .iter()
                        .map(|&root| {
                            let fraction = (power > 0.0)
                                .then(|| fractions(partials.iter().copied(), root, extent)[1]);
                            json!({"f0_hz": root, "fraction": fraction})
                        })
                        .collect();
                    let interval = number("interval_sec");
                    let lanes =
                        (number("births_in_second") + number("retirements_in_second")) / interval;
                    let flapping = number("flapping_in_second") / interval;
                    for value in [
                        lanes,
                        flapping,
                        power,
                        number("tonal_power"),
                        number("residual_power"),
                        number("total_biomass_mass"),
                    ]
                    .into_iter()
                    .chain(participation)
                    {
                        assert!(
                            value.is_finite(),
                            "{name}/{configuration}/{id}: non-finite metric"
                        );
                    }
                    seconds.push(json!({
                        "second": number("time_sec").floor() as u64, "time_sec": number("time_sec"),
                        "age_sec": number("age_sec"), "interval_sec": interval,
                        "spectrum_pmf": rendered_spectrum(&partials, &grid), "rendered_power": power,
                        "participation": participation, "lattice": lattice,
                        "births_retirements_per_sec": lanes, "flapping_per_sec": flapping,
                        "tonal_power": number("tonal_power"), "residual_power": number("residual_power"),
                        "total_biomass_mass": number("total_biomass_mass")
                    }));
                }
                let active: Vec<_> = seconds
                    .iter()
                    .filter(|s| {
                        s["age_sec"].as_f64().unwrap() >= 20.0
                            && s["second"].as_u64().unwrap() < release
                    })
                    .collect();
                let sounding: BTreeMap<_, Vec<f64>> = active
                    .iter()
                    .filter_map(|s| {
                        s["spectrum_pmf"].as_array().map(|v| {
                            (
                                s["second"].as_u64().unwrap(),
                                v.iter().map(|p| p.as_f64().unwrap()).collect(),
                            )
                        })
                    })
                    .collect();
                let mut change = Vec::new();
                for lag in [1, 5, 20] {
                    let distances: Vec<_> = sounding
                        .iter()
                        .filter_map(|(&second, spectrum)| {
                            second
                                .checked_sub(lag)
                                .and_then(|earlier| sounding.get(&earlier))
                                .map(|earlier| js_distance(spectrum, earlier))
                        })
                        .collect();
                    change.push(json!({"lag_sec": lag, "mean_js_distance": mean(distances.iter().copied()), "pairs": distances.len()}));
                }
                let recurrent: Vec<_> = sounding
                    .iter()
                    .filter_map(|(&second, spectrum)| {
                        let nearest = sounding
                            .iter()
                            .take_while(|(earlier, _)| **earlier + 5 <= second)
                            .map(|(_, earlier)| js_distance(spectrum, earlier))
                            .reduce(f64::min);
                        nearest.map(|d| if d <= 0.1 { 1.0 } else { 0.0 })
                    })
                    .collect();
                let recurrence = mean(recurrent.iter().copied());
                let participation = mean(active.iter().filter_map(|s| s["participation"].as_f64()));
                let windows = if sample == "drone" {
                    vec![(110.0, 22, 42), (110.0, 57, release), (146.83, 57, release)]
                } else {
                    vec![(f0, 22, release)]
                };
                let lattice: Vec<_> = windows.into_iter().map(|(root, begin, end)| {
                    let value = mean(seconds.iter().filter(|s| {
                        let t = s["second"].as_u64().unwrap(); t >= begin && t < end
                    }).filter_map(|s| s["lattice"].as_array().unwrap().iter()
                        .find(|entry| entry["f0_hz"].as_f64().unwrap() == root).unwrap()["fraction"].as_f64()));
                    json!({"f0_hz": root, "window_sec": [begin, end], "mean_power_fraction": value})
                }).collect();
                let intervals: Vec<_> = seconds
                    .iter()
                    .filter(|s| {
                        let t = s["second"].as_u64().unwrap();
                        t > 22 && t <= release
                    })
                    .collect();
                let lanes = mean(
                    intervals
                        .iter()
                        .map(|s| s["births_retirements_per_sec"].as_f64().unwrap()),
                );
                let lanes_max = intervals
                    .iter()
                    .map(|s| s["births_retirements_per_sec"].as_f64().unwrap())
                    .reduce(f64::max);
                let flapping = mean(
                    intervals
                        .iter()
                        .map(|s| s["flapping_per_sec"].as_f64().unwrap()),
                );
                let level = |begin: u64, end: u64| {
                    let snapshots: Vec<_> = seconds
                        .iter()
                        .filter(|s| {
                            let t = s["second"].as_u64().unwrap();
                            t >= begin && t < end
                        })
                        .collect();
                    let intervals: Vec<_> = seconds
                        .iter()
                        .filter(|s| {
                            let t = s["second"].as_u64().unwrap();
                            t > begin && t <= end
                        })
                        .collect();
                    json!({"window_sec": [begin, end], "seconds": snapshots.len(),
                        "tonal_power": mean(intervals.iter().map(|s| s["tonal_power"].as_f64().unwrap())),
                        "residual_power": mean(intervals.iter().map(|s| s["residual_power"].as_f64().unwrap())),
                        "biomass_min": snapshots.iter().map(|s| s["total_biomass_mass"].as_f64().unwrap()).reduce(f64::min),
                        "biomass_max": snapshots.iter().map(|s| s["total_biomass_mass"].as_f64().unwrap()).reduce(f64::max),
                        "silent_fraction": mean(snapshots.iter().map(|s| if s["spectrum_pmf"].is_null() { 1.0 } else { 0.0 }))})
                };
                let maximum_total_biomass = seconds
                    .iter()
                    .map(|r| r["total_biomass_mass"].as_f64().unwrap())
                    .reduce(f64::max)
                    .unwrap();
                let main_level = level(22, release);
                let mute_levels: Vec<_> = if sample == "mute" {
                    [(22, 42), (42, 62), (67, 102)]
                        .into_iter()
                        .map(|(begin, end)| level(begin, end))
                        .collect()
                } else {
                    Vec::new()
                };
                let displayed_levels = if sample == "mute" {
                    mute_levels.as_slice()
                } else {
                    std::slice::from_ref(&main_level)
                };
                let fmt =
                    |value: &Value| value.as_f64().map_or("—".into(), |v| format!("{v:.3}"));
                let changes = change
                    .iter()
                    .map(|v| fmt(&v["mean_js_distance"]))
                    .collect::<Vec<_>>()
                    .join("/");
                let lattices = lattice
                    .iter()
                    .map(|v| fmt(&v["mean_power_fraction"]))
                    .collect::<Vec<_>>()
                    .join("/");
                let powers = displayed_levels
                    .iter()
                    .map(|v| {
                        format!(
                            "{:.2e}/{:.2e}",
                            v["tonal_power"].as_f64().unwrap(),
                            v["residual_power"].as_f64().unwrap()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                let biomass = displayed_levels
                    .iter()
                    .map(|v| format!("{}/{}", fmt(&v["biomass_min"]), fmt(&v["biomass_max"])))
                    .collect::<Vec<_>>()
                    .join("; ");
                let silent = displayed_levels
                    .iter()
                    .map(|v| fmt(&v["silent_fraction"]))
                    .collect::<Vec<_>>()
                    .join("; ");
                let cfg = match configuration {
                    "sine_residual" => "sin+r",
                    "sine_only" => "sin",
                    _ => "res+r",
                };
                markdown.push_str(&format!("| {sample} | {cfg} | {id} | {} | {changes} | {} | {} | {lattices} | {}/{}; {} | {powers} | {biomass}; {maximum_total_biomass:.3} | {silent} |\n",
                    rows.len(), fmt(&json!(recurrence)), fmt(&json!(participation)), fmt(&json!(lanes)), fmt(&json!(lanes_max)), fmt(&json!(flapping))));
                results.push(json!({"sample": name, "configuration": configuration,
                    "flora_id": if id == "combined" { json!(id) } else { json!(id.parse::<u64>().unwrap()) },
                    "records": rows.len(), "measurement_window_sec": [22, release],
                    "change": change, "recurrence": recurrence, "recurrence_eligible_seconds": recurrent.len(),
                    "participation": participation, "lattice": lattice,
                    "lanes_mean_per_sec": lanes, "lanes_max_per_sec": lanes_max, "flapping_mean_per_sec": flapping,
                    "level": main_level, "maximum_total_biomass": maximum_total_biomass, "mute_levels": mute_levels, "spectrum_grid_erb": grid, "per_second": seconds}));
            }
        }
    }
    let mut writer = BufWriter::new(fs::File::create(root.join("summary.json")).unwrap());
    serde_json::to_writer(&mut writer, &json!({"warmup_sec": 20, "grid_step_erb": 0.05, "gaussian_sigma_erb": 0.25, "recurrence_min_lag_sec": 5, "recurrence_js_distance": 0.1, "runs": results})).unwrap();
    writer.flush().unwrap();
    fs::write(root.join("summary.md"), markdown).unwrap();
}
