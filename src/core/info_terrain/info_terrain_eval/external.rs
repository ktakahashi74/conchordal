use super::*;
use rustfft::{FftPlanner, num_complex::Complex};

fn weighted(values: &[f64], weights: &[f64]) -> Option<f64> {
    if values.iter().all(|x| !x.is_finite()) {
        return None;
    }
    let total: f64 = weights.iter().sum();
    Some(
        values
            .iter()
            .zip(weights)
            .filter(|(x, _)| x.is_finite())
            .map(|(x, w)| {
                x * if total == 0.0 {
                    1.0 / values.len() as f64
                } else {
                    w / total
                }
            })
            .sum(),
    )
}

// All input representations share these temporal and aggregation rules.
// S is the local source snapshot in reg-review-1.log, lines 1927-1943,
// 2021-2036 and 2120-2159. O adds full-window/silence rules and the A-2 floor.
fn series(frames: &[Vec<f64>], window: usize, delta: f64) -> [Vec<[Option<f64>; 2]>; 2] {
    let mut result: [Vec<_>; 2] = std::array::from_fn(|_| vec![[None; 2]; frames.len()]);
    if frames.is_empty() {
        return result;
    }
    let bands = frames[0].len();
    assert!(window >= 3);
    assert!(frames.iter().all(|f| f.len() == bands));
    let mut mean = vec![0.0; bands];
    let mut low = vec![0.0; bands];
    let mut high = vec![0.0; bands];
    let mut variance = vec![0.0; bands];
    let mut values = vec![f64::NAN; bands];
    let mut weights = vec![0.0; bands];
    let mut osf = vec![0.0; bands];
    for t in 2..frames.len() {
        let start = (t + 1).saturating_sub(window);
        let history = &frames[start..t];
        mean.fill(0.0);
        low.fill(f64::INFINITY);
        high.fill(f64::NEG_INFINITY);
        variance.fill(0.0);
        for row in history {
            for b in 0..bands {
                mean[b] += row[b];
                low[b] = low[b].min(row[b]);
                high[b] = high[b].max(row[b]);
            }
        }
        for m in &mut mean {
            *m /= history.len() as f64;
        }
        for row in history {
            for b in 0..bands {
                variance[b] += (row[b] - mean[b]).powi(2);
            }
        }
        for b in 0..bands {
            variance[b] /= history.len() as f64;
            weights[b] = if high[b].is_finite() {
                high[b].max(0.0)
            } else {
                0.0
            };
            osf[b] = high[b].max(frames[t][b]);
            let z = (frames[t][b] - mean[b]) / variance[b].sqrt();
            values[b] =
                if high[b] - low[b] > 0.0 && frames[t][b].is_finite() && variance[b].is_finite() {
                    if z.is_finite() {
                        (0.5 * z * z).min(SURPRISE_CAP)
                    } else {
                        SURPRISE_CAP
                    }
                } else {
                    f64::NAN
                };
        }
        result[0][t] = [weighted(&values, &weights), weighted(&values, &osf)];
        if t + 1 >= window {
            let maximum = mean.iter().copied().fold(0.0, f64::max);
            for b in 0..bands {
                if maximum == 0.0 || !mean[b].is_finite() || mean[b] < 1e-4 * maximum {
                    values[b] = f64::NAN;
                } else if values[b].is_finite() {
                    let sd_eff = variance[b].sqrt().max(delta * mean[b]);
                    let z = (frames[t][b] - mean[b]) / sd_eff;
                    values[b] = (0.5 * z * z).min(SURPRISE_CAP);
                }
            }
            result[1][t] = [weighted(&values, &weights), weighted(&values, &osf)];
        }
    }
    result
}

fn mel_frames(samples: &[f32], fs: u32) -> Vec<Vec<f64>> {
    const FFT: usize = 8192;
    let length = (0.1 * f64::from(fs)).round() as usize;
    let step = (0.02 * f64::from(fs)).round() as usize;
    let mel_max = 2595.0 * (1.0 + f64::from(fs) / 2.0 / 700.0).log10();
    let edges: Vec<_> = (0..22)
        .map(|b| 700.0 * (10_f64.powf(mel_max * b as f64 / 21.0 / 2595.0) - 1.0))
        .collect();
    let filters: Vec<Vec<(usize, f64)>> = (0..20)
        .map(|b| {
            (0..=FFT / 2)
                .filter_map(|k| {
                    let hz = k as f64 * f64::from(fs) / FFT as f64;
                    let weight = if hz <= edges[b + 1] {
                        (hz - edges[b]) / (edges[b + 1] - edges[b])
                    } else {
                        (edges[b + 2] - hz) / (edges[b + 2] - edges[b + 1])
                    };
                    (weight > 0.0).then_some((k, weight))
                })
                .collect()
        })
        .collect();
    let window: Vec<_> = (0..length)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (length - 1) as f64).cos())
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(FFT);
    let mut buffer = vec![Complex::new(0.0, 0.0); FFT];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let mut frames = Vec::new();
    if samples.len() < length {
        return frames;
    }
    for start in (0..=samples.len() - length).step_by(step) {
        buffer.fill(Complex::new(0.0, 0.0));
        for i in 0..length {
            buffer[i].re = f64::from(samples[start + i]) * window[i];
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        frames.push(
            filters
                .iter()
                .map(|filter| filter.iter().map(|&(k, w)| w * buffer[k].norm_sqr()).sum())
                .collect(),
        );
    }
    frames
}

#[derive(Serialize, serde::Deserialize)]
pub(super) struct Stimulus {
    file: String,
    token: String,
    prototype: String,
    unpredictability: Option<f64>,
    // Version order VERSION_NAMES; aggregation order primary, auxiliary.
    means: [[Option<f64>; 2]; 6],
    defined_frames: [[usize; 2]; 6],
    exclusions: [Vec<String>; 6],
}

fn stimulus(
    path: &Path,
    template: &AnalysisStream,
    ratings: &BTreeMap<String, f64>,
) -> Result<Stimulus> {
    let stem = path
        .file_stem()
        .unwrap()
        .to_str()
        .context("non-UTF8 file stem")?;
    let fields: Vec<_> = stem.split('_').collect();
    ensure!(fields.len() >= 2, "invalid stimulus filename");
    let (fs, samples) = read_wav(path)?;
    ensure!(fs == 44100, "external stimulus must remain at 44.1 kHz");
    let mut stream = template.clone();
    let mut amplitude = Vec::new();
    let mut power = Vec::new();
    for audio in samples.chunks_exact(HOP) {
        let frame = stream.process(audio);
        frame
            .space
            .assert_scan_len_named(&frame.nsgt_power, "external_nsgt_power_scan");
        amplitude.push(
            frame
                .nsgt_power
                .iter()
                .map(|p| f64::from(*p).sqrt())
                .collect(),
        );
        power.push(frame.nsgt_power.iter().map(|p| f64::from(*p)).collect());
    }
    let mel = mel_frames(&samples, fs);
    let mut traces = Vec::new();
    let mut means = [[None; 2]; 6];
    let mut defined_frames = [[0; 2]; 6];
    let mut frame_counts = [0; 6];
    let archived: Value = serde_json::from_reader(fs::File::open(
        root()
            .join("results-a1/external")
            .join(format!("{stem}.json")),
    )?)?;
    ensure!(
        archived["wav_sha256"] == format!("{:x}", Sha256::digest(fs::read(path)?)),
        "external input differs from A-1: {stem}"
    );
    for (input, frames) in [&mel, &amplitude, &power].into_iter().enumerate() {
        let hop = if input == 0 { 882 } else { HOP };
        let window = (1.0 / (hop as f64 / f64::from(fs))).floor() as usize;
        let delta = if input == 1 {
            amplitude_delta()
        } else {
            10_f64.powf(DELTA_L_DB / 10.0) - 1.0
        };
        let versions = series(frames, window, delta);
        for (rule, mut trace) in versions.into_iter().enumerate() {
            let v = input * 2 + rule;
            if rule == 0 {
                trace = serde_json::from_value(
                    archived["traces"][v]["trace_primary_auxiliary"].clone(),
                )?;
            }
            frame_counts[v] = trace.len();
            for aggregation in 0..2 {
                let values: Vec<_> = trace.iter().filter_map(|f| f[aggregation]).collect();
                defined_frames[v][aggregation] = values.len();
                means[v][aggregation] =
                    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
            }
            if rule == 0 {
                means[v] = serde_json::from_value(archived["stimulus"]["means"][v].clone())?;
                defined_frames[v] =
                    serde_json::from_value(archived["stimulus"]["defined_frames"][v].clone())?;
            }
            traces.push(
                json!({"version":VERSION_NAMES[v],"hop_samples":hop,"window_frames":window,
                "trace_primary_auxiliary":trace}),
            );
        }
    }
    let unpredictability = ratings.get(fields[1]).copied();
    let exclusions = std::array::from_fn(|v| {
        let mut reasons = Vec::new();
        if unpredictability.is_none() {
            reasons.push("token_join_failure".to_string());
        }
        for (aggregation, label) in ["primary", "auxiliary"].into_iter().enumerate() {
            if means[v][aggregation].is_none() {
                reasons.push(format!("all_missing_{label}"));
            }
        }
        reasons
    });
    let stimulus = Stimulus {
        file: path.file_name().unwrap().to_str().unwrap().to_string(),
        token: fields[1].to_string(),
        prototype: fields[0].to_string(),
        unpredictability,
        means,
        defined_frames,
        exclusions,
    };
    write_json(
        &results()?.join("external").join(format!("{stem}.json")),
        &json!({"stimulus":stimulus,"traces":traces,
        "sample_rate":fs,"samples":samples.len(),"trailing_nsgt_samples":samples.len()%HOP,"frame_counts":frame_counts,
        "wav_sha256":format!("{:x}",Sha256::digest(fs::read(path)?))}),
    )?;
    Ok(stimulus)
}

// Sufficient statistics preserve duplicate draws of a prototype and avoid
// allocating copied stimuli during the 10,000 paired cluster resamples.
#[derive(Clone, Copy, Default)]
struct Correlation {
    n: f64,
    x: f64,
    y: f64,
    xx: f64,
    yy: f64,
    xy: f64,
}

impl Correlation {
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.x += x;
        self.y += y;
        self.xx += x * x;
        self.yy += y * y;
        self.xy += x * y;
    }
    fn merge(&mut self, b: Self) {
        self.n += b.n;
        self.x += b.x;
        self.y += b.y;
        self.xx += b.xx;
        self.yy += b.yy;
        self.xy += b.xy;
    }
    fn r(self) -> Option<f64> {
        let xx = self.xx - self.x * self.x / self.n;
        let yy = self.yy - self.y * self.y / self.n;
        let xy = self.xy - self.x * self.y / self.n;
        (self.n >= 2.0 && xx > 0.0 && yy > 0.0).then(|| xy / (xx * yy).sqrt())
    }
}

fn interval(values: &mut [f64]) -> Option<[f64; 2]> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some([0.025, 0.975].map(|p| {
        let index = p * (values.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        values[lower] + (values[upper] - values[lower]) * (index - lower as f64)
    }))
}

fn bootstrap(stimuli: &[Stimulus], names: [&str; 6], contrasts: [(usize, usize); 3]) -> Value {
    let common: Vec<_> = stimuli
        .iter()
        .filter(|s| s.unpredictability.is_some() && s.means.iter().flatten().all(Option::is_some))
        .collect();
    let mut groups: BTreeMap<&str, [[Correlation; 6]; 2]> = BTreeMap::new();
    for stimulus in &common {
        let group = groups.entry(&stimulus.prototype).or_default();
        for aggregation in 0..2 {
            for v in 0..6 {
                group[aggregation][v].add(
                    stimulus.means[v][aggregation].unwrap(),
                    stimulus.unpredictability.unwrap(),
                );
            }
        }
    }
    let mut totals = [[Correlation::default(); 6]; 2];
    for group in groups.values() {
        for a in 0..2 {
            for v in 0..6 {
                totals[a][v].merge(group[a][v]);
            }
        }
    }
    let groups: Vec<_> = groups.into_iter().collect();
    let mut draws: [[Vec<f64>; 6]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::with_capacity(RESAMPLES)));
    let mut differences: [[Vec<f64>; 3]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::with_capacity(RESAMPLES)));
    let mut invalid = [[0_usize; 6]; 2];
    let mut invalid_contrasts = [[0_usize; 3]; 2];
    let mut rng = SplitMix64(BOOTSTRAP_SEED);
    let mut resample_hash = Sha256::new();
    if !groups.is_empty() {
        for _ in 0..RESAMPLES {
            let mut moments = [[Correlation::default(); 6]; 2];
            for _ in 0..groups.len() {
                let idx = rng.index(groups.len());
                resample_hash.update((idx as u64).to_le_bytes());
                let group = &groups[idx].1;
                for a in 0..2 {
                    for v in 0..6 {
                        moments[a][v].merge(group[a][v]);
                    }
                }
            }
            for a in 0..2 {
                let rs = moments[a].map(Correlation::r);
                for v in 0..6 {
                    if let Some(r) = rs[v] {
                        draws[a][v].push(r);
                    } else {
                        invalid[a][v] += 1;
                    }
                }
                for (c, &(left, right)) in contrasts.iter().enumerate() {
                    if let Some((l, r)) = rs[left].zip(rs[right]) {
                        differences[a][c].push(l - r);
                    } else {
                        invalid_contrasts[a][c] += 1;
                    }
                }
            }
        }
    }
    let aggregations:Vec<_>=(0..2).map(|a| {
        let versions:Vec<_>=(0..6).map(|v|json!({"version":names[v],"r":totals[a][v].r(),"ci95":interval(&mut draws[a][v]),"undefined_bootstrap_resamples":invalid[a][v]})).collect();
        let contrasts:Vec<_>=contrasts.iter().enumerate().map(|(c,&(left,right))|json!({"contrast":format!("{} minus {}",names[left],names[right]),
            "difference":totals[a][left].r().zip(totals[a][right].r()).map(|(l,r)|l-r),"ci95":interval(&mut differences[a][c]),"undefined_bootstrap_resamples":invalid_contrasts[a][c]})).collect();
        json!({"aggregation":if a==0 {"soundgen_primary"} else {"osf_auxiliary"},"versions":versions,"contrasts":contrasts})
    }).collect();
    json!({"common_count":common.len(),"common_files":common.iter().map(|s|&s.file).collect::<Vec<_>>(),
        "prototype_count":groups.len(),"prototypes":groups.iter().map(|(name,g)|json!({"prototype":name,"stimuli":g[0][0].n as usize})).collect::<Vec<_>>(),
        "resample_indices_sha256_le_u64":format!("{:x}",resample_hash.finalize()),
        "bootstrap_seed":BOOTSTRAP_SEED,"resamples":RESAMPLES,"aggregations":aggregations})
}

pub(super) fn revision_statistics(a1: &Value, a2: &Value) -> Result<Value> {
    let old: Vec<Stimulus> = serde_json::from_value(a1["stimuli"].clone())?;
    let new: Vec<Stimulus> = serde_json::from_value(a2["stimuli"].clone())?;
    let old: BTreeMap<_, _> = old.into_iter().map(|s| (s.file.clone(), s)).collect();
    ensure!(old.len() == new.len(), "external input set changed");
    let mut paired = Vec::new();
    for current in new {
        let previous = old.get(&current.file).context("A-1 stimulus absent")?;
        ensure!(
            current.token == previous.token
                && current.prototype == previous.prototype
                && current.unpredictability == previous.unpredictability,
            "external pairing changed"
        );
        for v in [0, 2, 4] {
            ensure!(current.means[v] == previous.means[v], "rule S changed");
        }
        let indices = [3, 1, 5];
        paired.push(Stimulus {
            means: std::array::from_fn(|i| {
                if i % 2 == 0 {
                    current.means[indices[i / 2]]
                } else {
                    previous.means[indices[i / 2]]
                }
            }),
            defined_frames: [[0; 2]; 6],
            exclusions: std::array::from_fn(|_| Vec::new()),
            ..current
        });
    }
    let statistics = bootstrap(
        &paired,
        [
            "N-O A-2",
            "N-O A-1",
            "R-O A-2",
            "R-O A-1",
            "N-P-O A-2",
            "N-P-O A-1",
        ],
        [(0, 1), (2, 3), (4, 5)],
    );
    for source in [a1, a2] {
        ensure!(
            statistics["common_files"] == source["statistics"]["common_files"],
            "common set changed"
        );
        ensure!(
            statistics["prototypes"] == source["statistics"]["prototypes"],
            "prototype order changed"
        );
        ensure!(
            statistics["resample_indices_sha256_le_u64"]
                == source["statistics"]["resample_indices_sha256_le_u64"],
            "shared resamples changed"
        );
    }
    Ok(statistics)
}

#[test]
fn only_rule_o_uses_the_floor_in_the_supplied_input_units() {
    let frames = vec![vec![0.99], vec![1.0], vec![1.01], vec![1.1]];
    let amplitude = series(&frames, 4, amplitude_delta());
    let power_delta = 10_f64.powf(DELTA_L_DB / 10.0) - 1.0;
    let power = series(&frames, 4, power_delta);
    assert_eq!(amplitude[0], power[0]);
    for (trace, delta) in [(&amplitude, amplitude_delta()), (&power, power_delta)] {
        let expected = 0.5 * (0.1 / delta).powi(2);
        assert!((trace[1][3][0].unwrap() - expected).abs() < 1e-12);
        assert!(trace[1][3][0].unwrap() < trace[0][3][0].unwrap());
    }
}

#[test]
fn stft_frame_grid_drops_partial_frames_and_uses_power() {
    let mut audio = vec![0.0_f32; 4410 + 882];
    audio[0] = 1.0;
    let frames = mel_frames(&audio, 44100);
    assert_eq!(frames.len(), 2);
    assert!(frames.iter().flatten().all(|x| *x == 0.0));
    audio[2000] = 1.0;
    let frames = mel_frames(&audio, 44100);
    let partial = mel_frames(&audio[..audio.len() - 1], 44100);
    assert_eq!(partial.len(), 1);
    assert_eq!(partial[0], frames[0]);
    assert!(frames.iter().flatten().all(|x| *x > 0.0));
    for x in &mut audio {
        *x *= 2.0;
    }
    let doubled = mel_frames(&audio, 44100);
    for (a, b) in frames.iter().flatten().zip(doubled.iter().flatten()) {
        assert!((4.0 * a - b).abs() < 1e-10 * b);
    }
}

#[test]
fn soundgen_short_sequences_match_hand_calculation() {
    // At t=2: [1,3] -> mean 2, population variance 1, current 5 -> S=4.5.
    // Other band has constant history: NA. Its weight remains in denominator.
    let frames = vec![
        vec![1.0, 4.0],
        vec![3.0, 4.0],
        vec![5.0, 4.0],
        vec![1000.0, 4.0],
    ];
    let output = series(&frames, 4, amplitude_delta());
    assert_eq!(output[0][0], [None; 2]);
    assert_eq!(output[0][1], [None; 2]);
    assert!((output[0][2][0].unwrap() - 4.5 * 3.0 / 7.0).abs() < 1e-12);
    assert!((output[0][2][1].unwrap() - 4.5 * 5.0 / 9.0).abs() < 1e-12);
    assert_eq!(output[1][2], [None; 2]);
    assert!((output[0][3][0].unwrap() - SURPRISE_CAP * 5.0 / 9.0).abs() < 1e-12);
    assert_eq!(output[0][3], output[1][3]);
    assert!(
        series(&vec![vec![1.0]; 4], 4, amplitude_delta())
            .iter()
            .flatten()
            .all(|v| *v == [None; 2])
    );
    // Positive-max clipping, zero-sum uniform fallback, no NA renormalization.
    let negatives = vec![vec![-3.0, -4.0], vec![-1.0, -4.0], vec![-2.0, -4.0]];
    assert_eq!(series(&negatives, 3, amplitude_delta())[0][2][0], Some(0.0));
    assert_eq!(weighted(&[4.0, f64::NAN], &[0.0, 0.0]), Some(2.0));
    assert_eq!(weighted(&[4.0, f64::NAN], &[1.0, 3.0]), Some(1.0));
    assert_eq!(weighted(&[f64::NAN, f64::NAN], &[1.0, 3.0]), None);
}

#[test]
fn registered_comparison_matches_a2_original() {
    let space = Log2Space::new(100.0, 200.0, 1);
    let mut terrain = InfoTerrain::new(space, 0.25);
    let powers = [
        [1.0_f32, 0.0],
        [9.0, 0.0],
        [25.0, 0.0],
        [49.0, 0.0],
        [9.0, 0.0],
    ];
    let frames: Vec<_> = powers
        .iter()
        .map(|row| row.iter().map(|x| f64::from(*x).sqrt()).collect())
        .collect();
    let expected = series(&frames, 4, amplitude_delta());
    for (t, power) in powers.iter().enumerate() {
        terrain.observe(power, &[[0.0; 8]; 2], [0.0; 8], false);
        for (a, weights) in [&terrain.soundgen_weight_scan, &terrain.osf_weight_scan]
            .into_iter()
            .enumerate()
        {
            let actual = integrate(&terrain.space, &terrain.versions[0].surprise_scan, weights);
            assert_eq!(actual.is_finite(), expected[1][t][a].is_some());
            if let Some(value) = expected[1][t][a] {
                assert!((actual - value).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn pearson_and_paired_cluster_bootstrap_preserve_identical_versions() {
    let stimuli: Vec<_> = (0..8)
        .map(|i| Stimulus {
            file: format!("p{}_t{i}.wav", i / 2),
            token: format!("t{i}"),
            prototype: format!("p{}", i / 2),
            unpredictability: Some(i as f64),
            means: [[Some(i as f64), Some(i as f64)]; 6],
            defined_frames: [[3; 2]; 6],
            exclusions: std::array::from_fn(|_| vec![]),
        })
        .collect();
    let output = bootstrap(&stimuli, VERSION_NAMES, [(0, 2), (2, 4), (0, 1)]);
    assert_eq!(output["common_count"], 8);
    for aggregation in output["aggregations"].as_array().unwrap() {
        for version in aggregation["versions"].as_array().unwrap() {
            assert_eq!(version["r"], 1.0);
            assert_eq!(version["ci95"], json!([1.0, 1.0]));
        }
        for contrast in aggregation["contrasts"].as_array().unwrap() {
            assert_eq!(contrast["difference"], 0.0);
            assert_eq!(contrast["ci95"], json!([0.0, 0.0]));
        }
    }
}

pub(super) fn info_terrain_eval_external() -> Result<()> {
    let output = results()?;
    fs::create_dir_all(output.join("external"))?;
    let mut ratings = BTreeMap::new();
    let rating_text = fs::read_to_string(root().join("fit_per_sound.csv"))?;
    let mut lines = rating_text.lines();
    let header: Vec<_> = lines
        .next()
        .context("empty ratings")?
        .split(',')
        .map(|x| x.trim_matches('"'))
        .collect();
    let token = header
        .iter()
        .position(|x| *x == "token")
        .context("token column absent")?;
    let fit = header
        .iter()
        .position(|x| *x == "fit")
        .context("fit column absent")?;
    for line in lines {
        let fields: Vec<_> = line.split(',').map(|x| x.trim_matches('"')).collect();
        let value: f64 = fields[fit].parse()?;
        ensure!(value.is_finite(), "nonfinite fit");
        ensure!(
            ratings
                .insert(fields[token].to_string(), 1.0 - value)
                .is_none(),
            "duplicate ratings token"
        );
    }
    let mut paths = fs::read_dir(root().join("anikin-stimuli-wav"))?
        .map(|p| p.map(|p| p.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|p| p.extension().is_some_and(|e| e == "wav"));
    paths.sort();
    ensure!(paths.len() == 303, "expected 303 prepared WAV files");
    let stream = crate::runtime::info_terrain_eval_stream(44100);
    let mut stimuli = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        stimuli.push(stimulus(path, &stream, &ratings)?);
        if (i + 1) % 10 == 0 {
            println!("external: {}/{}", i + 1, paths.len());
        }
    }
    let statistics = bootstrap(&stimuli, VERSION_NAMES, [(0, 2), (2, 4), (0, 1)]);
    let exclusions:Vec<_>=(0..6).map(|v|json!({"version":VERSION_NAMES[v],"excluded":stimuli.iter().filter(|s|!s.exclusions[v].is_empty()).map(|s|json!({"file":s.file,"token":s.token,"reasons":s.exclusions[v]})).collect::<Vec<_>>()})).collect();
    let payload = json!({"provenance":provenance()?,"ratings_sha256":format!("{:x}",Sha256::digest(rating_text.as_bytes())),
        "input_count":stimuli.len(),"ratings_count":ratings.len(),"version_order":VERSION_NAMES,"aggregation_order":["soundgen_primary","osf_auxiliary"],
        "stimuli":stimuli,"exclusions":exclusions,"statistics":statistics});
    write_json(&output.join("external.json"), &payload)?;
    let mut markdown = format!(
        "# External comparison\n\nRevision A-2 draft 3; {} prepared WAVs; {} stimuli in the common set; {} prototypes. Bootstrap: {} shared cluster resamples, splitmix64 seed {}. S reused from A-1; O uses input-unit perceptual floors.\n\n",
        stimuli.len(),
        statistics["common_count"],
        statistics["prototype_count"],
        RESAMPLES,
        BOOTSTRAP_SEED
    );
    for aggregation in statistics["aggregations"].as_array().unwrap() {
        markdown.push_str(&format!(
            "{}\n\n| Version | r | 95% CI |\n|---|---:|---|\n",
            aggregation["aggregation"].as_str().unwrap()
        ));
        for row in aggregation["versions"].as_array().unwrap() {
            markdown.push_str(&format!(
                "| {} | {} | {} |\n",
                row["version"].as_str().unwrap(),
                row["r"],
                row["ci95"]
            ));
        }
        markdown.push('\n');
        for row in aggregation["contrasts"].as_array().unwrap() {
            markdown.push_str(&format!(
                "- {}: {}, 95% CI {}\n",
                row["contrast"].as_str().unwrap(),
                row["difference"],
                row["ci95"]
            ));
        }
        markdown.push('\n');
    }
    markdown.push_str("Exclusions (per version):\n\n");
    for row in &exclusions {
        markdown.push_str(&format!(
            "- {}: {}\n",
            row["version"].as_str().unwrap(),
            row["excluded"]
        ));
    }
    markdown.push_str("\nR is the registered Rust STFT/HTK-mel approximation, with symmetric Hann and raw one-sided band power. Leading MP3 silence retained. Fit uncertainty excluded from bootstrap. OSF auxiliary uses the registered per-hop maxima and population SD, rather than reproducing the bundled final-window weight reuse.\n");
    fs::write(output.join("external.md"), markdown)?;
    println!("external statistics: {}", statistics["aggregations"]);
    Ok(())
}
