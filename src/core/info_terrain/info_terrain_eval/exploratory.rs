//! Exploratory A4 measurements; unregistered and never used by function A.

use super::*;
use crate::core::{erb::hz_to_erb, landscape::Landscape};
use std::collections::VecDeque;

const MEASURES: [&str; 14] = [
    "D1a_raw",
    "D1a_floor",
    "D1b",
    "D2a",
    "D2b",
    "D3a",
    "D3b",
    "D3c",
    "D4a_mean",
    "D4a_median",
    "D4b_mean",
    "D4b_median",
    "A2_original_surprise",
    "A2_original_precision",
];

struct WindowStats {
    mean: Vec<f64>,
    variance: Vec<f64>,
    maximum_mean: f64,
}

fn bayesian_change(
    mean: f64,
    variance: f64,
    prior_mean: f64,
    prior_variance: f64,
    len: usize,
) -> f64 {
    let ratio = variance / prior_variance;
    let argument =
        (mean - prior_mean).powi(2) / (2.0 * prior_variance) + (ratio - 1.0 - ratio.ln()) / 2.0;
    if argument.is_finite() {
        argument.max(1e-12).ln() + 2.0 * (len as f64).ln()
    } else {
        f64::NAN
    }
}

fn spectral_measures(landscape: &Landscape) -> ([f64; 3], [f64; 2]) {
    let space = &landscape.space;
    space.assert_scan_len_named(&landscape.nsgt_power, "nsgt_power");
    space.assert_scan_len_named(&landscape.harmonicity01, "harmonicity01");
    space.assert_scan_len_named(&landscape.consonance_field_level, "consonance_field_level");
    let mut total = 0.0;
    let mut log_power = 0.0;
    let mut harmonicity = 0.0;
    let mut consonance = 0.0;
    let mut count = 0;
    for b in 0..space.n_bins() {
        if (100.0..=4000.0).contains(&space.freq_of_index(b)) {
            let power = f64::from(landscape.nsgt_power[b]);
            total += power;
            log_power += power.ln();
            harmonicity += power * f64::from(landscape.harmonicity01[b]);
            consonance += power * f64::from(landscape.consonance_field_level[b]);
            count += 1;
        }
    }
    let organization = if total > 0.0 {
        [
            (log_power / count as f64).exp() / (total / count as f64),
            harmonicity / total,
            consonance / total,
        ]
    } else {
        [f64::NAN; 3]
    };
    let maximum = landscape.nsgt_power.iter().copied().fold(0.0_f32, f32::max);
    let mut candidates: Vec<_> = (1..space.n_bins() - 1)
        .filter(|&b| {
            let p = landscape.nsgt_power[b];
            (80.0..=4000.0).contains(&space.freq_of_index(b))
                && p > 0.0
                && p >= maximum * 1e-3
                && p > landscape.nsgt_power[b - 1]
                && p >= landscape.nsgt_power[b + 1]
        })
        .collect();
    candidates.sort_by(|&a, &b| {
        landscape.nsgt_power[b]
            .total_cmp(&landscape.nsgt_power[a])
            .then(a.cmp(&b))
    });
    let mut peaks = Vec::new();
    for b in candidates {
        let erb = hz_to_erb(space.freq_of_index(b));
        if peaks
            .iter()
            .all(|&other| (erb - hz_to_erb(space.freq_of_index(other))).abs() >= 1.0)
        {
            peaks.push(b);
        }
    }
    let mut groups = Vec::<f32>::new();
    for &b in &peaks {
        let hz = space.freq_of_index(b);
        if let Some(lowest) = groups.iter_mut().find(|lowest| {
            (1..=12).any(|multiple| (hz / (**lowest * multiple as f32) - 1.0).abs() <= 0.03)
        }) {
            *lowest = lowest.min(hz);
        } else {
            groups.push(hz);
        }
    }
    (organization, [peaks.len() as f64, groups.len() as f64])
}

fn measure(path: &Path) -> Result<Value> {
    let (fs, samples) = read_wav(path)?;
    ensure!(
        samples.len() as f64 / f64::from(fs) >= 20.0,
        "exploratory: input shorter than 20 s"
    );
    let mut stream = crate::runtime::info_terrain_eval_stream(fs);
    let space = stream.last().space.clone();
    let n = space.n_bins();
    let bins: Vec<_> = (0..n)
        .filter(|&b| (100.0..=4000.0).contains(&space.freq_of_index(b)))
        .collect();
    let mut history = SpectralHistory::new(space.clone(), f64::from(fs), stream.window_samples());
    let (mut observer, output) = Observer::new(space.clone(), fs, HOP, "habitat", false, true);
    let window_len = observer.terrain.history_len + 1;
    let lag = (f64::from(fs) / HOP as f64).round() as usize;
    let mut windows = VecDeque::<WindowStats>::new();
    let mut moments = [Moments::default(); 10];
    let mut peaks = [Vec::new(), Vec::new()];
    let mut reference = [Moments::default(); 2];
    let mut hops = 0_u64;
    let mut constant_bin_hops = 0_u64;
    let mut nonfinite_bin_hops = [0_u64; 2];
    for (frame, audio) in samples.chunks_exact(HOP).enumerate() {
        let time = (frame * HOP) as f64 / f64::from(fs);
        if time >= 20.0 {
            break;
        }
        let mut landscape = stream.process(audio);
        // The runtime completes these derived views after receiving analysis.
        landscape.recompute_consonance(stream.landscape_params());
        history.advance_to((frame * HOP) as u64);
        // Keep only the displaced row and pre-update residual state. All A-2
        // history means, maxima, masks and indicator scans come from the observer.
        let displaced = observer.terrain.history
            [observer.terrain.cursor * n..(observer.terrain.cursor + 1) * n]
            .to_vec();
        let replaced_row = observer.terrain.cursor;
        let residuals: Vec<_> = bins
            .iter()
            .map(|&b| {
                let predictor = &observer.terrain.predictors[b];
                (predictor.residual_variance, predictor.amplitude_ema)
            })
            .collect();
        observer.observe(frame as u64, &landscape.nsgt_power, &history, false);
        history.observe(
            (frame * HOP) as u64,
            ((frame + 1) * HOP) as u64,
            &landscape.nsgt_power,
        );
        for _ in output.rx.try_iter() {}
        let terrain = &observer.terrain;
        if terrain.mean_scan.iter().all(|x| !x.is_finite()) {
            continue;
        }
        let mut current = WindowStats {
            mean: Vec::with_capacity(bins.len()),
            variance: Vec::with_capacity(bins.len()),
            maximum_mean: 0.0,
        };
        // A-2 does not retain its variance scan. Reuse its exact mean, then
        // recover centered population variance from the unchanged history rows.
        let mut changes = [vec![f64::NAN; n], vec![f64::NAN; n]];
        let mut weights = vec![0.0; n];
        for &b in &bins {
            let prior_mean = terrain.mean_scan[b];
            let mut squared = 0.0;
            let mut minimum = f64::INFINITY;
            for t in 0..terrain.history_len {
                let a = if t == replaced_row {
                    displaced[b]
                } else {
                    terrain.history[t * n + b]
                };
                squared += (a - prior_mean).powi(2);
                minimum = minimum.min(a);
            }
            let variance = squared / terrain.history_len as f64;
            let amplitude = f64::from(landscape.nsgt_power[b]).sqrt();
            let mean = prior_mean + (amplitude - prior_mean) / window_len as f64;
            let variance_x =
                (squared + (amplitude - prior_mean) * (amplitude - mean)) / window_len as f64;
            current.mean.push(mean);
            current.variance.push(variance_x);
            current.maximum_mean = current.maximum_mean.max(mean);
            weights[b] = terrain.soundgen_weight_scan[b];
            let constant = terrain.soundgen_weight_scan[b] == minimum;
            if (2.0..20.0).contains(&time) {
                constant_bin_hops += u64::from(constant);
            }
            for (v, values) in changes.iter_mut().enumerate() {
                let value = if constant {
                    f64::NAN
                } else {
                    let denominator = if v == 0 {
                        variance
                    } else {
                        variance.max((amplitude_delta() * prior_mean).powi(2))
                    };
                    bayesian_change(mean, variance_x, prior_mean, denominator, window_len)
                };
                if (2.0..20.0).contains(&time) {
                    nonfinite_bin_hops[v] += u64::from(!constant && !value.is_finite());
                }
                values[b] = value;
            }
        }
        windows.push_back(current);
        if windows.len() > lag + 1 {
            windows.pop_front();
        }
        if !(2.0..20.0).contains(&time) {
            continue;
        }
        hops += 1;
        for v in 0..2 {
            moments[v].add(integrate(&space, &changes[v], &weights));
        }
        let current = windows.back().unwrap();
        let mut r2_sum = 0.0;
        let mut r2_weight = 0.0;
        let mut kl_sum = 0.0;
        let mut kl_weight = 0.0;
        for (i, &b) in bins.iter().enumerate() {
            let mean = current.mean[i];
            let variance = current.variance[i].max((amplitude_delta() * mean).powi(2));
            if terrain.versions[1].precision_scan[b].is_finite()
                && mean > 0.0
                && mean >= 1e-4 * current.maximum_mean
                && let Some(scale) = residuals[i].1
            {
                let residual_variance = residuals[i].0 * scale.powi(2);
                let r2 = (1.0 - residual_variance / variance).clamp(0.0, 1.0);
                r2_sum += mean * r2;
                r2_weight += mean;
            }
            if windows.len() == lag + 1 {
                let previous = &windows[0];
                let old_mean = previous.mean[i];
                if mean > 0.0
                    && old_mean > 0.0
                    && mean >= 1e-4 * current.maximum_mean
                    && old_mean >= 1e-4 * previous.maximum_mean
                {
                    let old_variance =
                        previous.variance[i].max((amplitude_delta() * old_mean).powi(2));
                    let kl = 0.25
                        * (variance / old_variance + old_variance / variance - 2.0
                            + (mean - old_mean).powi(2) * (1.0 / variance + 1.0 / old_variance));
                    kl_sum += mean * kl;
                    kl_weight += mean;
                }
            }
        }
        if kl_weight > 0.0 {
            moments[2].add(kl_sum / kl_weight);
        }
        moments[3].add(precision(terrain, 1).unwrap_or(f64::NAN));
        if r2_weight > 0.0 {
            moments[4].add(r2_sum / r2_weight);
        }
        let (organization, complexity) = spectral_measures(&landscape);
        for (moment, value) in moments[5..8].iter_mut().zip(organization) {
            moment.add(value);
        }
        for i in 0..2 {
            moments[8 + i].add(complexity[i]);
            peaks[i].push(complexity[i]);
        }
        reference[0].add(integrate(
            &space,
            &terrain.versions[0].surprise_scan,
            &terrain.soundgen_weight_scan,
        ));
        reference[1].add(precision(terrain, 0).unwrap_or(f64::NAN));
    }
    drop(observer);
    for _ in output.rx.iter() {}
    ensure!(
        output.dropped.load(Ordering::Relaxed) == 0,
        "exploratory: lost observer blocks"
    );
    let mut means = BTreeMap::new();
    let mut missing = BTreeMap::new();
    for (name, moment) in MEASURES[..8].iter().zip(&moments[..8]) {
        means.insert(*name, moment.mean());
        missing.insert(*name, hops - moment.count);
    }
    for i in 0..2 {
        peaks[i].sort_by(f64::total_cmp);
        let len = peaks[i].len();
        let median = (len > 0).then(|| (peaks[i][(len - 1) / 2] + peaks[i][len / 2]) / 2.0);
        means.insert(MEASURES[8 + 2 * i], moments[8 + i].mean());
        means.insert(MEASURES[9 + 2 * i], median);
        missing.insert(MEASURES[8 + 2 * i], hops - moments[8 + i].count);
        missing.insert(MEASURES[9 + 2 * i], hops - moments[8 + i].count);
    }
    Ok(
        json!({"status":"exploratory", "sample_rate":fs, "input_samples":samples.len(),
        "trailing_samples":samples.len()%HOP, "evaluation_hops":hops,
        "window_frames":window_len, "window_duration_sec":window_len as f64*HOP as f64/f64::from(fs),
        "D1b_lag_hops":lag,"D1b_lag_sec":lag as f64*HOP as f64/f64::from(fs),
        "analysis_frequency_hz":[space.freq_of_index(0),space.freq_of_index(n-1)],
        "analysis_bins":n,"measurement_bins":bins.len(),"means":means,"missing_hops":missing,
        "D1a_constant_history_bin_hops":constant_bin_hops,
        "D1a_nonconstant_nonfinite_bin_hops_raw_floor":nonfinite_bin_hops,
        "replayed_A2_original":{"mean_surprise":reference[0].mean(),"mean_precision":reference[1].mean()}}),
    )
}

#[test]
fn exploratory_numeric_checks() {
    // x1=[1,3], x=[1,3,5]: population variances 1 and 8/3.
    let expected =
        (0.5 + (8.0_f64 / 3.0 - 1.0 - (8.0_f64 / 3.0).ln()) / 2.0).ln() + 2.0 * 3.0_f64.ln();
    assert!((bayesian_change(3.0, 8.0 / 3.0, 2.0, 1.0, 3) - expected).abs() < 1e-14);
    assert_eq!(
        bayesian_change(2.0, 1.0, 2.0, 1.0, 3),
        1e-12_f64.ln() + 2.0 * 3.0_f64.ln()
    );
    assert!(bayesian_change(1.0, 0.0, 1.0, 0.0, 3).is_nan());
    let space = crate::core::log2space::Log2Space::new(55.0, 8000.0, 96);
    let mut landscape = Landscape::new(space);
    let (organization, counts) = spectral_measures(&landscape);
    assert!(organization.iter().all(|x| x.is_nan()));
    assert_eq!(counts, [0.0, 0.0]);
    landscape.nsgt_power.fill(1.0);
    landscape.harmonicity01.fill(0.25);
    landscape.consonance_field_level.fill(0.75);
    assert_eq!(
        spectral_measures(&landscape),
        ([1.0, 0.25, 0.75], [0.0, 0.0])
    );
    landscape.nsgt_power.fill(0.0);
    for (frequency, power) in [(220.0, 1.0), (225.0, 0.9), (440.0, 0.7), (660.0, 0.5)] {
        let b = landscape.space.nearest_index(frequency);
        landscape.nsgt_power[b] = power;
    }
    assert_eq!(spectral_measures(&landscape).1, [3.0, 1.0]);

    let mut stream = crate::runtime::info_terrain_eval_stream(FS);
    for frame in 0..32 {
        let audio: Vec<_> = (0..HOP)
            .map(|i| {
                (std::f32::consts::TAU * 440.0 * (frame * HOP + i) as f32 / FS as f32).sin() * 0.1
            })
            .collect();
        let mut landscape = stream.process(&audio);
        landscape.recompute_consonance(stream.landscape_params());
        if frame == 31 {
            let (organization, _) = spectral_measures(&landscape);
            assert!(organization[1] > 0.0 && organization[2] > 0.0);
        }
    }
}

#[test]
#[ignore = "exploratory A4: unregistered WAV replay; writes only exploratory_dimensions outputs"]
fn info_terrain_eval_exploratory() -> Result<()> {
    let directory = results()?;
    let mut cases = Vec::new();
    let mut provenance = provenance()?;
    for relative in [
        "src/core/info_terrain/info_terrain_eval/exploratory.rs",
        "src/core/landscape.rs",
        "src/core/erb.rs",
        "src/audio/writer.rs",
    ] {
        provenance["source_sha256"][relative] =
            json!(format!("{:x}", Sha256::digest(fs::read(relative)?)));
    }
    provenance["status"] = json!("exploratory");
    for group in ["controls", "renders"] {
        let summary_path = directory.join(format!("{group}.json"));
        let archived: Value = serde_json::from_reader(fs::File::open(&summary_path)?)?;
        let summary_hash = format!("{:x}", Sha256::digest(fs::read(&summary_path)?));
        ensure!(
            archived["provenance"]["revision"] == "A-2",
            "exploratory: expected A-2 references"
        );
        ensure!(
            archived["provenance"]["source_sha256"]["src/core/info_terrain.rs"]
                == provenance["source_sha256"]["src/core/info_terrain.rs"],
            "exploratory: function A changed since registered measurement"
        );
        for case in archived["cases"]
            .as_array()
            .context("exploratory: no registered cases")?
        {
            let name = case["stimulus"]
                .as_str()
                .context("exploratory: unnamed stimulus")?;
            let path = directory.join(if group == "controls" {
                format!("controls/{name}.wav")
            } else {
                format!("renders/{name}/audio.wav")
            });
            let wav_hash = format!("{:x}", Sha256::digest(fs::read(&path)?));
            if group == "controls" {
                ensure!(
                    case["wav_sha256"] == wav_hash,
                    "exploratory: changed control WAV"
                );
            }
            let mut row = measure(&path)?;
            let original = if group == "controls" {
                &case["original"]
            } else {
                &case["versions"]["original"]
            };
            row["means"]["A2_original_surprise"] = original["mean_surprise"].clone();
            row["means"]["A2_original_precision"] = original["mean_precision"].clone();
            row["archived_A2_extended"] = if group == "controls" {
                case["extended"].clone()
            } else {
                case["versions"]["extended"].clone()
            };
            row["stimulus"] = json!(name);
            row["group"] = json!(group);
            row["wav"] = json!(path.strip_prefix(env!("CARGO_MANIFEST_DIR"))?);
            row["wav_sha256"] = json!(wav_hash);
            row["reference_summary"] =
                json!(summary_path.strip_prefix(env!("CARGO_MANIFEST_DIR"))?);
            row["reference_summary_sha256"] = json!(summary_hash);
            println!("exploratory {name}: {}", row["means"]);
            cases.push(row);
        }
    }
    ensure!(
        cases.len() == 12,
        "exploratory: expected six controls and six renders"
    );
    let conventions = json!({
        "status":"exploratory", "registration":"UNREGISTERED; only to inform a later pre-registration",
        "evaluation_window_sec":"[2,20); complete hops selected by start sample; no padding",
        "input":"existing controls and render WAVs; native sample rate, mono read_wav, default habitat analysis path, hop 512; renders are PCM16 presentation WAVs, not internal habitat samples",
        "frequency_hz":{"D1a_D1b_D2b_D3":[100,4000],"D2a_and_archived_A2_references":[55,8000],"D4":[80,4000]},
        "window":"floor(fs/512) frames including current; D1a history excludes current; D1b compares full windows separated by round(fs/512) hops; actual durations and lag recorded per row",
        "D1a":"soundgen 3.0.0 Eq.5 log Gaussian KL + 2 ln(window_frames); population variances; kl_arg floored at 1e-12; constant history -> NA; no silence mask; history positive maxima normalized across in-band bins; NA terms dropped without renormalization; all NA -> NA",
        "D1a_floor":"only var_x1 replaced with max(var_x1,(delta*mean_x1)^2), including in var_ratio; current variance unchanged; constant history still NA",
        "D1b":"0.5*(KL(current||previous)+KL(previous||current)); each population variance floored at its own (delta*mean)^2; both windows positive and >=1e-4 of their in-band maximum mean; current-window mean amplitude weights, renormalized over eligible bins",
        "D2a":"existing A-2 full-band per-hop extended precision aggregation, recomputed on WAV replay; archived extended values retained separately",
        "D2b":"clamp(1-pre_update_residual_variance*pre_update_amplitude_ema^2 / max(current_window_population_variance,(delta*current_window_mean)^2),0,1); residual variance is raw, before its evaluation-only floor; require defined A-2 extended precision and positive non-silent current window; current-window mean amplitude weights",
        "D3a":"geometric mean NSGT power / arithmetic mean; exact zero bin -> zero flatness; all-zero band -> NA; no epsilon",
        "D3b_D3c":"after AnalysisStream::process, call Landscape::recompute_consonance with that stream's actual LandscapeParams, as runtime merge does; NSGT-power-weighted harmonicity01 and base consonance_field_level; zero band power -> NA",
        "D4a":"local power maxima: > left and >= right (plateau left edge); threshold >=1e-3 of full-hop maximum; descending-power suppression within 1 ERB-rate; equal-power ties lower bin first; silence -> 0",
        "D4b":"heuristic, not a stream-segregation model; descending-power peaks; first eligible group with +/-3% of integer multiple 1..12 of its lowest member; otherwise new group; update lowest member when joined; group order is creation order",
        "A2_original_references":"copied from existing registered A-2 summary files, full band; render references used internal habitat and block selection, distinct from this PCM16 WAV replay; not overwritten",
        "summary":"finite-hop arithmetic means; D4 also ordinary time medians; null for all-missing; missing hop and D1a degenerate-bin counts retained",
        "ranks":"descending numerical value for every table column; exact ties share competition rank; names break display ties; NA excluded and listed separately; no quality direction implied",
        "soundgen_semantics_local_source":"target/info-terrain-20261020/reg-review-1.log:1927-1943,2021-2036,2120-2193",
        "delta_db":DELTA_L_DB,"delta":amplitude_delta(),"composite_index":null
    });
    let mut markdown = String::from(
        "# exploratory A4 dimensions\n\nUNREGISTERED exploratory measurement, only to inform a later pre-registration. All values below are exploratory. No composite index.\n\n",
    );
    markdown.push_str("| exploratory stimulus |");
    for name in MEASURES {
        markdown.push_str(&format!(" {name} |"));
    }
    markdown.push_str("\n|---|");
    for _ in MEASURES {
        markdown.push_str("---:|");
    }
    markdown.push('\n');
    for row in &cases {
        markdown.push_str(&format!("| {} |", row["stimulus"].as_str().unwrap()));
        for name in MEASURES {
            markdown.push_str(&format!(
                " {} |",
                row["means"][name]
                    .as_f64()
                    .map_or_else(|| "NA".into(), |x| format!("{x:.2e}"))
            ));
        }
        markdown.push('\n');
    }
    let mut ranks = BTreeMap::new();
    markdown.push_str(
        "\nExploratory ranks, descending numerical value; exact ties share rank; NA unranked.\n\n",
    );
    for name in MEASURES {
        let mut order: Vec<_> = cases
            .iter()
            .filter_map(|row| {
                row["means"][name]
                    .as_f64()
                    .map(|x| (row["stimulus"].as_str().unwrap(), x))
            })
            .collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
        let mut rank = 0;
        let mut ranked = Vec::new();
        let mut labels = Vec::new();
        for (i, &(stimulus, value)) in order.iter().enumerate() {
            if i == 0 || value != order[i - 1].1 {
                rank = i + 1;
            }
            ranked.push(json!({"rank":rank,"stimulus":stimulus,"value":value}));
            labels.push(format!("{rank}: {stimulus}"));
        }
        let missing: Vec<_> = cases
            .iter()
            .filter(|row| row["means"][name].is_null())
            .map(|row| row["stimulus"].clone())
            .collect();
        markdown.push_str(&format!(
            "- exploratory {name}: {}; NA: {}\n",
            labels.join("; "),
            json!(missing)
        ));
        ranks.insert(
            name,
            json!({"status":"exploratory","descending":ranked,"missing":missing}),
        );
    }
    markdown.push_str("\nExploratory conventions and provenance (the registered values describe the inherited harness, not an A4 registration):\n\n```json\n");
    markdown.push_str(&serde_json::to_string_pretty(&conventions)?);
    markdown.push_str("\n```\n\nFull exploratory provenance, per-stimulus input hashes, actual window durations, missing counts and archived references: `exploratory_dimensions.json`.\n\n```json\n");
    markdown.push_str(&serde_json::to_string_pretty(&provenance)?);
    markdown.push_str("\n```\n");
    write_json(
        &directory.join("exploratory_dimensions.json"),
        &json!({"status":"exploratory","provenance":provenance,"conventions":conventions,"columns":MEASURES,"cases":cases,"ranks":ranks}),
    )?;
    fs::write(directory.join("exploratory_dimensions.md"), markdown)?;
    Ok(())
}
