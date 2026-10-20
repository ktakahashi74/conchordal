//! Ignored, two-stage registration A-3 measurements. Thresholds never tune inference.
use super::*;
use crate::{
    config::AppConfig,
    core::{
        info_terrain::{
            InfoTerrain, Observer,
            info_terrain_eval::{controls, read_wav, write_json},
        },
        spectral_history::SpectralHistory,
    },
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

const FS: u32 = 48_000;
const HOP: usize = 512;
const HOPS: usize = 20 * FS as usize / HOP;
fn directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target/info-terrain-20261020/results-a3")
}
fn git(args: &[&str]) -> Result<Vec<u8>> {
    let out = Command::new("git").args(args).output()?;
    ensure!(out.status.success(), "git {args:?} failed");
    Ok(out.stdout)
}
fn effective_config() -> Result<AppConfig> {
    let mut config: AppConfig = toml::from_str(&fs::read_to_string("config.toml")?)?;
    config.info_terrain.enabled = true;
    config.info_terrain.efe = true;
    config.validate()?;
    ensure!(
        config.audio.sample_rate == FS && config.analysis.hop_size == HOP,
        "registration requires 48 kHz / hop 512"
    );
    Ok(config)
}
fn provenance(config: &AppConfig) -> Result<()> {
    let dir = directory().join("provenance");
    fs::create_dir_all(&dir)?;
    fs::write(
        dir.join("diff-head.binary.patch"),
        git(&["diff", "HEAD", "--binary"])?,
    )?;
    fs::write(
        dir.join("effective-config.toml"),
        toml::to_string_pretty(config)?,
    )?;
    fs::copy("config.toml", dir.join("repository-config.toml"))?;
    let mut untracked = BTreeMap::new();
    for bytes in git(&["ls-files", "--others", "--exclude-standard", "-z"])?
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
    {
        let path = Path::new(std::str::from_utf8(bytes)?);
        if !["src", "tests", "samples", "rhai-defs"]
            .iter()
            .any(|p| path.starts_with(p))
        {
            continue;
        }
        let data = fs::read(path)?;
        let target = dir.join("untracked").join(path);
        fs::create_dir_all(target.parent().unwrap())?;
        fs::write(target, &data)?;
        untracked.insert(
            path.display().to_string(),
            format!("{:x}", Sha256::digest(&data)),
        );
    }
    let mut registrations = BTreeMap::new();
    for (name, prefix) in [
        (
            "registration-a.ja.md",
            "# 事前登録：予測精度と驚きの観測（機能 A）草案 第5稿",
        ),
        (
            "registration-a2.ja.md",
            "# 事前登録の改訂：予測精度と驚きの観測 A-2（知覚の床と拡張版の起動）草案 第3稿",
        ),
        (
            "registration-a3.ja.md",
            "# 事前登録：EFE に着想を得た受動的な合成値 A-3 草案 第6稿",
        ),
    ] {
        let data = fs::read(directory().parent().unwrap().join(name))?;
        ensure!(
            std::str::from_utf8(&data)?.starts_with(prefix),
            "registration revision mismatch: {name}"
        );
        fs::create_dir_all(dir.join("registrations"))?;
        fs::write(dir.join("registrations").join(name), &data)?;
        registrations.insert(name, format!("{:x}", Sha256::digest(data)));
    }
    write_json(
        &dir.join("manifest.json"),
        &json!({"commit":String::from_utf8(git(&["rev-parse","HEAD"])?)?.trim(),
        "git_describe":String::from_utf8(git(&["describe","--always","--dirty","--tags"])?)?.trim(),
        "untracked_used_files_sha256":untracked,"registrations_sha256":registrations,
        "rng":"splitmix64; exactly representable open (0,1) 52-bit midpoints; Box-Muller cosine then cached sine; Marsaglia-Tsang gamma including shape<1 correction",
        "inference_seed_domains":"seed xor stream*0xd2b74407b1ce6e93; first splitmix seed for resampling, then each particle v/s transition; seed assignments do not depend on particle count; slot RNGs not resampled",
        "stage1_series":"x0=ln(0.1); o0=x0+sqrt(s)*normal0; thereafter normal(2t-1) for x, normal(2t) for o; inherited A-1 Box-Muller stimulus generator",
        "stage1_analysis":"effective repository analysis settings; full inherited A-2 observer supplies current-excluded M and missing order; target cell only",
        "initial_habituation":"zero per bin, separate reference-only state per stream; reset on gap",
        "metric_order":["A","N","S_x","R","V"],"lambda_v":0.01,"lambda_s":0.01,"s_floor":s_floor(),"reference_beta":4.0}),
    )
}
fn float_wav(path: &Path, audio: &[f32]) -> Result<()> {
    let mut w = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: FS,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for &x in audio {
        w.write_sample(x)?;
    }
    w.finalize()?;
    Ok(())
}
fn series(seed: u64, v: f64, s: f64) -> Vec<f64> {
    let noise = controls::gaussian(seed, 2 * HOPS - 1);
    let mut x = 0.1_f64.ln();
    (0..HOPS)
        .map(|t| {
            if t > 0 {
                x += v.sqrt() * noise[2 * t - 1];
            }
            x + s.sqrt() * noise[2 * t]
        })
        .collect()
}
fn analyzed_observations(
    seed: u64,
    observations: &[f64],
    config: &AppConfig,
) -> Result<Vec<Option<f64>>> {
    let dir = directory().join("stage1-inputs");
    fs::create_dir_all(&dir)?;
    let mut stream = crate::runtime::info_terrain_efe_eval_stream(config, FS);
    let space = stream.last().space.clone();
    let terrain = Terrain::new(space.clone(), 0);
    let bin = space.nearest_index(1000.0);
    let cell = terrain
        .cells
        .iter()
        .find(|c| c.bins.contains(&bin))
        .context("1000 Hz cell absent")?;
    let hz = f64::from(crate::core::erb::erb_to_hz(
        (cell.lo_erb + cell.hi_erb) / 2.0,
    ));
    let audio: Vec<f32> = (0..20 * FS as usize)
        .map(|i| {
            let position = i as f64 / HOP as f64 - 0.5;
            let lo = position.floor() as isize;
            let f = position - position.floor();
            let amp = |k: isize| observations[k.clamp(0, HOPS as isize - 1) as usize].exp();
            (((1.0 - f) * amp(lo) + f * amp(lo + 1))
                * (std::f64::consts::TAU * hz * i as f64 / FS as f64).sin()) as f32
        })
        .collect();
    let path = dir.join(format!("seed-{seed}.wav"));
    float_wav(&path, &audio)?;
    let (fs, replay) = read_wav(&path)?;
    ensure!(fs == FS && replay == audio, "WAV replay mismatch");
    let mut a2 = InfoTerrain::new(space.clone(), HOP as f64 / FS as f64);
    let ages = vec![[0.0; 8]; space.n_bins()];
    let mut result = Vec::new();
    for audio in replay.chunks_exact(HOP) {
        let frame = stream.process(audio);
        a2.observe(&frame.nsgt_power, &ages, [0.0; 8], false);
        let maximum = a2
            .mean_scan
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .max_by(f64::total_cmp);
        let energy = cell
            .bins
            .clone()
            .map(|b| f64::from(frame.nsgt_power[b]))
            .sum::<f64>();
        let o = maximum.filter(|m| *m > 0.0).and_then(|m| {
            let epsilon = (1e-4 * m).powi(2);
            (energy >= epsilon && energy.is_finite()).then(|| 0.5 * (energy + epsilon).ln())
        });
        result.push(o);
    }
    write_json(
        &dir.join(format!("seed-{seed}.json")),
        &json!({"seed":seed,"frequency_hz":hz,"cell_bins":[cell.bins.start,cell.bins.end],
        "cell_erb":[cell.lo_erb,cell.hi_erb],"normalization":false,"wav_sha256":format!("{:x}",Sha256::digest(fs::read(path)?)),
        "direct_observations":observations,"analysis_observations":result}),
    )?;
    Ok(result)
}
#[derive(Serialize)]
struct Recovery {
    variant: &'static str,
    path: &'static str,
    seed: u64,
    truth_v: f64,
    truth_s: f64,
    median_v: Option<f64>,
    median_s: Option<f64>,
    log10_error_v: Option<f64>,
    log10_error_s: Option<f64>,
    value_v_pass: bool,
    value_s_pass: bool,
    valid_hops: usize,
    evaluation_hops: usize,
    valid_rate: f64,
    valid_rate_pass: bool,
}
fn median(mut xs: Vec<f64>) -> Option<f64> {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    (n > 0).then(|| (xs[(n - 1) / 2] + xs[n / 2]) / 2.0)
}
fn recover(
    variant: &'static str,
    path: &'static str,
    seed: u64,
    truth: (f64, f64),
    len: usize,
    initial: (f64, f64),
    input: &[Option<f64>],
) -> Result<Recovery> {
    let (v, s) = truth;
    let mut filter = Filter::new(seed + 50_000, 0, len, initial);
    let mut vs = Vec::new();
    let mut ss = Vec::new();
    let mut hops = 0;
    let dir = directory().join("stage1-traces");
    fs::create_dir_all(&dir)?;
    let mut trace = BufWriter::new(fs::File::create(
        dir.join(format!("{variant}-{path}-{seed}.jsonl")),
    )?);
    for (i, &o) in input.iter().enumerate() {
        let row = o.map_or_else(
            || CellHop::missing("startup_or_silence"),
            |o| filter.observe(o),
        );
        serde_json::to_writer(
            &mut trace,
            &json!({"frame_id":i,"time_sec":i as f64*HOP as f64/FS as f64,"cell":row}),
        )?;
        writeln!(trace)?;
        if i * HOP >= 10 * FS as usize {
            hops += 1;
            if let Some(m) = row.metrics {
                vs.push(m.v);
                ss.push(m.s);
            }
        }
    }
    trace.flush()?;
    let valid = vs.len();
    let mv = median(vs);
    let ms = median(ss);
    let ev = mv.map(|x| x.log10() - v.log10());
    let es = ms.map(|x| x.log10() - s.log10());
    let rate = valid as f64 / hops as f64;
    Ok(Recovery {
        variant,
        path,
        seed,
        truth_v: v,
        truth_s: s,
        median_v: mv,
        median_s: ms,
        log10_error_v: ev,
        log10_error_s: es,
        value_v_pass: ev.is_some_and(|e| e.abs() <= 0.5),
        value_s_pass: es.is_some_and(|e| e.abs() <= 0.5),
        valid_hops: valid,
        evaluation_hops: hops,
        valid_rate: rate,
        valid_rate_pass: rate >= 0.9,
    })
}
fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut out = vec![0.0; xs.len()];
    let mut a = 0;
    while a < order.len() {
        let mut b = a + 1;
        while b < order.len() && xs[order[b]] == xs[order[a]] {
            b += 1;
        }
        for i in a..b {
            out[order[i]] = (a + b - 1) as f64 / 2.0;
        }
        a = b;
    }
    out
}
fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    let a = ranks(a);
    let b = ranks(b);
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let cov = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>();
    let d = (a.iter().map(|x| (x - ma).powi(2)).sum::<f64>()
        * b.iter().map(|x| (x - mb).powi(2)).sum::<f64>())
    .sqrt();
    (d > 0.0).then_some(cov / d)
}
#[derive(Serialize)]
struct PathCriteria {
    variant: &'static str,
    path: &'static str,
    spearman_v: Option<f64>,
    spearman_s: Option<f64>,
    order_v_pass: bool,
    order_s_pass: bool,
    value_v_count: usize,
    value_s_count: usize,
    value_v_pass: bool,
    value_s_pass: bool,
    s_crosstalk_by_fixed_s: [Option<f64>; 3],
    v_crosstalk_by_fixed_v: [Option<f64>; 3],
    crosstalk_s_pass: bool,
    crosstalk_v_pass: bool,
    valid_hop_rates: Vec<f64>,
    valid_hops_pass: bool,
    pass: bool,
}
fn criteria(rows: &[Recovery]) -> PathCriteria {
    let estimates_v: Option<Vec<_>> = rows.iter().map(|r| r.median_v.map(f64::log10)).collect();
    let estimates_s: Option<Vec<_>> = rows.iter().map(|r| r.median_s.map(f64::log10)).collect();
    let v = estimates_v.as_ref().and_then(|v| {
        spearman(
            v,
            &rows.iter().map(|r| r.truth_v.log10()).collect::<Vec<_>>(),
        )
    });
    let s = estimates_s.as_ref().and_then(|s| {
        spearman(
            s,
            &rows.iter().map(|r| r.truth_s.log10()).collect::<Vec<_>>(),
        )
    });
    let range = |xs: Option<Vec<f64>>| {
        xs.map(|xs| {
            xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - xs.iter().copied().fold(f64::INFINITY, f64::min)
        })
    };
    let cs = std::array::from_fn(|s| {
        range(
            (0..3)
                .map(|v| rows[3 * v + s].median_s.map(f64::log10))
                .collect(),
        )
    });
    let cv = std::array::from_fn(|v| {
        range(
            (0..3)
                .map(|s| rows[3 * v + s].median_v.map(f64::log10))
                .collect(),
        )
    });
    let vc = rows.iter().filter(|r| r.value_v_pass).count();
    let sc = rows.iter().filter(|r| r.value_s_pass).count();
    let mut r = PathCriteria {
        variant: rows[0].variant,
        path: rows[0].path,
        spearman_v: v,
        spearman_s: s,
        order_v_pass: v.is_some_and(|r| r >= 0.8),
        order_s_pass: s.is_some_and(|r| r >= 0.8),
        value_v_count: vc,
        value_s_count: sc,
        value_v_pass: vc >= 7,
        value_s_pass: sc >= 7,
        s_crosstalk_by_fixed_s: cs,
        v_crosstalk_by_fixed_v: cv,
        crosstalk_s_pass: cs.iter().all(|r| r.is_some_and(|r| r <= 0.6)),
        crosstalk_v_pass: cv.iter().all(|r| r.is_some_and(|r| r <= 0.6)),
        valid_hop_rates: rows.iter().map(|r| r.valid_rate).collect(),
        valid_hops_pass: rows.iter().all(|r| r.valid_rate_pass),
        pass: false,
    };
    r.pass = [
        r.order_v_pass,
        r.order_s_pass,
        r.value_v_pass,
        r.value_s_pass,
        r.crosstalk_s_pass,
        r.crosstalk_v_pass,
        r.valid_hops_pass,
    ]
    .iter()
    .all(|p| *p);
    r
}
fn stage1(config: &AppConfig) -> Result<(Vec<PathCriteria>, Vec<String>)> {
    let mut cache = BTreeMap::new();
    let mut conditions = Vec::new();
    let mut outcomes = Vec::new();
    let mut failures = Vec::new();
    for (variant, offset, len, initial) in [
        ("base", 0, 64, (V0, S0)),
        ("seed+1000", 1000, 64, (V0, S0)),
        ("seed+2000", 2000, 64, (V0, S0)),
        ("particles128", 0, 128, (V0, S0)),
        ("initial", 0, 64, (0.001, 0.5)),
    ] {
        let mut by_path = [Vec::new(), Vec::new()];
        for (vi, v) in [1e-4, 1e-3, 1e-2].into_iter().enumerate() {
            for (si, s) in [1e-3, 1e-2, 1e-1].into_iter().enumerate() {
                let seed = 101 + (3 * vi + si) as u64 + offset;
                if let std::collections::btree_map::Entry::Vacant(entry) = cache.entry(seed) {
                    let direct = series(seed, v, s);
                    let analyzed = analyzed_observations(seed, &direct, config)?;
                    entry.insert((direct.into_iter().map(Some).collect::<Vec<_>>(), analyzed));
                }
                let (direct, analyzed) = &cache[&seed];
                for (i, (path, input)) in [("direct", direct), ("analysis", analyzed)]
                    .into_iter()
                    .enumerate()
                {
                    by_path[i].push(recover(variant, path, seed, (v, s), len, initial, input)?);
                }
            }
        }
        for rows in by_path {
            let c = criteria(&rows);
            println!(
                "stage1 {variant}/{}: rho v={:?} s={:?}; values {}/9 {}/9; pass={}",
                c.path, c.spearman_v, c.spearman_s, c.value_v_count, c.value_s_count, c.pass
            );
            for (name, pass) in [
                ("order_v", c.order_v_pass),
                ("order_s", c.order_s_pass),
                ("value_v", c.value_v_pass),
                ("value_s", c.value_s_pass),
                ("crosstalk_s", c.crosstalk_s_pass),
                ("crosstalk_v", c.crosstalk_v_pass),
                ("valid_hops", c.valid_hops_pass),
            ] {
                if !pass {
                    failures.push(format!("{variant}/{}:{name}", c.path));
                }
            }
            conditions.extend(rows);
            outcomes.push(c);
        }
        // Persist each completed variant without making stage-2 decisions early.
        write_json(
            &directory().join("stage1.json"),
            &json!({"evaluation_sec":[10,20],"conditions":conditions,"criteria":outcomes,"failures":failures}),
        )?;
    }
    let mut md = String::from(
        "# A-3 stage 1\n\nEvaluation: [10,20) s. Cross-talk limit: 0.6 log10 units. Values: at least 7/9 within 0.5 log10 units.\n\n| Variant | Path | rho v | rho s | value v | value s | max s cross-talk | max v cross-talk | min valid-hop rate | Pass |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for c in &outcomes {
        let max = |xs: &[Option<f64>]| xs.iter().flatten().copied().max_by(f64::total_cmp);
        md.push_str(&format!(
            "| {} | {} | {:?} | {:?} | {}/9 | {}/9 | {:?} | {:?} | {:.6} | {} |\n",
            c.variant,
            c.path,
            c.spearman_v,
            c.spearman_s,
            c.value_v_count,
            c.value_s_count,
            max(&c.s_crosstalk_by_fixed_s),
            max(&c.v_crosstalk_by_fixed_v),
            c.valid_hop_rates.iter().copied().fold(1.0, f64::min),
            c.pass
        ));
    }
    md.push_str("\n| Variant | Path | seed | true v | true s | median v | median s | v value pass | s value pass | valid hops / total | rate pass |\n|---|---|---:|---:|---:|---:|---:|---|---|---|---|\n");
    for c in &conditions {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:?} | {:?} | {} | {} | {}/{} | {} |\n",
            c.variant,
            c.path,
            c.seed,
            c.truth_v,
            c.truth_s,
            c.median_v,
            c.median_s,
            c.value_v_pass,
            c.value_s_pass,
            c.valid_hops,
            c.evaluation_hops,
            c.valid_rate_pass
        ));
    }
    fs::write(directory().join("stage1.md"), md)?;
    Ok((outcomes, failures))
}

#[test]
#[ignore = "A-3 registered recovery/separation, then stage 2 only after every required stage-1 pass"]
fn efe_registered_two_stage_suite() -> Result<()> {
    fs::create_dir_all(directory())?;
    let config = effective_config()?;
    provenance(&config)?;
    let (outcomes, failures) = stage1(&config)?;
    if outcomes.iter().any(|c| !c.pass) {
        let status = format!("stage 2 not run (stage 1 failed: {})", failures.join(", "));
        write_json(
            &directory().join("stage2.json"),
            &json!({"status":status,"ran":false,"stage1_failures":failures}),
        )?;
        fs::write(
            directory().join("stage2.md"),
            format!("# A-3 stage 2\n\n{status}\n"),
        )?;
        println!("{status}");
        return Ok(());
    }
    stage2(&config)
}

fn stage2_stimuli() -> Vec<(&'static str, Vec<f32>)> {
    let n = 20 * FS as usize;
    let mut brown = controls::gaussian(3, n + 2 * FS as usize);
    let mut y = 0.0;
    for x in &mut brown {
        y = 0.995 * y + *x;
        *x = y;
    }
    let mut brown = brown[2 * FS as usize..].to_vec();
    controls::normalize(&mut brown);
    let mut band = controls::gaussian(4, 21 * FS as usize);
    let f0 = (400_f64 * 800.0).sqrt();
    let q = f0 / 400.0;
    let omega = std::f64::consts::TAU * f0 / FS as f64;
    let alpha = omega.sin() / (2.0 * q);
    let b = [alpha / (1.0 + alpha), 0.0, -alpha / (1.0 + alpha)];
    let a = [
        -2.0 * omega.cos() / (1.0 + alpha),
        (1.0 - alpha) / (1.0 + alpha),
    ];
    let filter = |xs: &mut [f64]| {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for x in xs {
            let y = b[0] * *x + b[1] * x1 + b[2] * x2 - a[0] * y1 - a[1] * y2;
            x2 = x1;
            x1 = *x;
            y2 = y1;
            y1 = y;
            *x = y;
        }
    };
    filter(&mut band);
    band.reverse();
    filter(&mut band);
    band.reverse();
    let mut band = band[FS as usize / 2..21 * FS as usize - FS as usize / 2].to_vec();
    controls::normalize(&mut band);
    let mut chord: Vec<_> = (0..n)
        .map(|i| {
            [261.63, 329.63, 392.0]
                .iter()
                .map(|f| (std::f64::consts::TAU * f * i as f64 / FS as f64).sin())
                .sum::<f64>()
        })
        .collect();
    controls::normalize(&mut chord);
    // Use the inherited stimulus RNG convention for log-uniform tone frequencies.
    let mut rng = crate::core::info_terrain::info_terrain_eval::SplitMix64(5);
    let frequencies: Vec<[f64; 3]> = (0..80)
        .map(|_| std::array::from_fn(|_| (200_f64.ln() + rng.uniform() * 10_f64.ln()).exp()))
        .collect();
    let half = (0.005 * FS as f64) as isize;
    let length = FS as isize / 4;
    let tone = |sample: isize, k: isize| {
        frequencies[k as usize]
            .iter()
            .map(|f| (std::f64::consts::TAU * f * (sample - k * length) as f64 / FS as f64).sin())
            .sum::<f64>()
    };
    let mut jumps: Vec<_> = (0..n)
        .map(|i| {
            let sample = i as isize;
            let k = sample / length;
            let offset = sample % length;
            let next = if offset < half && k > 0 {
                Some(k)
            } else if offset >= length - half && k < 79 {
                Some(k + 1)
            } else {
                None
            };
            if let Some(next) = next {
                let t = (sample - next * length + half) as f64 / (2 * half) as f64;
                let f = 0.5 - 0.5 * (std::f64::consts::PI * t).cos();
                (1.0 - f) * tone(sample, next - 1) + f * tone(sample, next)
            } else {
                let fade = if k == 0 && offset < half {
                    0.5 - 0.5 * (std::f64::consts::PI * offset as f64 / half as f64).cos()
                } else {
                    1.0
                };
                fade * tone(sample, k)
            }
        })
        .collect();
    controls::normalize(&mut jumps);
    let mut white = controls::gaussian(6, n);
    controls::normalize(&mut white);
    let mut changing: Vec<_> = brown
        .iter()
        .zip(&white)
        .enumerate()
        .map(|(i, (b, w))| {
            let f = i as f64 / (n - 1) as f64;
            (1.0 - f) * b + f * w
        })
        .collect();
    controls::normalize(&mut changing);
    [
        ("V1_brown", brown),
        ("V2_band", band),
        ("V3_chord", chord),
        ("V4_jumps", jumps),
        ("V5_changing", changing),
    ]
    .into_iter()
    .map(|(n, x)| (n, x.into_iter().map(|x| x as f32).collect()))
    .collect()
}
fn measure(audio: &[f32], config: &AppConfig) -> Result<Vec<Hop>> {
    let mut stream = crate::runtime::info_terrain_efe_eval_stream(config, FS);
    let space = stream.last().space.clone();
    let mut history = SpectralHistory::new(space.clone(), FS as f64, stream.window_samples());
    let (mut observer, output) =
        Observer::new(space, FS, HOP, "habitat", config.info_terrain.trace, true);
    observer.enable_efe(config, stream.window_samples());
    let mut rows = Vec::new();
    for (id, hop) in audio.chunks_exact(HOP).take(HOPS).enumerate() {
        let frame = stream.process(hop);
        history.advance_to((id * HOP) as u64);
        observer.observe_analysis(
            id as u64,
            &frame,
            &history,
            false,
            stream.landscape_params(),
        );
        history.observe(
            (id * HOP) as u64,
            ((id + 1) * HOP) as u64,
            &frame.nsgt_power,
        );
        rows.push(
            observer
                .efe
                .as_ref()
                .unwrap()
                .trace
                .as_ref()
                .unwrap()
                .last()
                .unwrap()
                .clone(),
        );
        for _ in output.rx.try_iter() {}
    }
    drop(observer);
    ensure!(
        output.dropped.load(std::sync::atomic::Ordering::Acquire) == 0,
        "harness dropped blocks"
    );
    Ok(rows)
}
#[derive(Serialize)]
struct Window {
    seconds: [f64; 2],
    means: [Option<f64>; 5],
    defined_hops: [u64; 5],
    missing_rates: [f64; 5],
    total_hops: usize,
    valid_cell_hops: usize,
    sounding_cell_hops: usize,
    numerical_missing_cell_hops: usize,
    numerical_missing_rate: Option<f64>,
    missing_cells_by_reason: BTreeMap<String, usize>,
    missing_rule_pass: bool,
}
fn window(rows: &[Hop], start: f64, end: f64) -> Window {
    let selected: Vec<_> = rows
        .iter()
        .filter(|r| {
            let t = r.frame_id as f64 * HOP as f64 / FS as f64;
            (start..end).contains(&t)
        })
        .collect();
    let mut moments = [Moments::default(); 5];
    let mut valid = 0;
    let mut sounding = 0;
    let mut numerical = 0;
    let mut reasons = BTreeMap::new();
    for r in &selected {
        for (m, v) in moments.iter_mut().zip(r.values) {
            if let Some(x) = v {
                m.add(x);
            }
        }
        valid += r.valid_cells;
        sounding += r.sounding_cells;
        for c in &r.cells {
            if let Some(reason) = &c.reason {
                *reasons.entry(reason.clone()).or_insert(0) += 1;
                if reason == "numerical" {
                    numerical += 1;
                }
            }
        }
    }
    let total = selected.len();
    let numeric_rate = (sounding > 0).then(|| numerical as f64 / sounding as f64);
    Window {
        seconds: [start, end],
        means: moments.map(Moments::mean),
        defined_hops: moments.map(|m| m.count),
        missing_rates: moments.map(|m| {
            if total > 0 {
                1.0 - m.count as f64 / total as f64
            } else {
                1.0
            }
        }),
        total_hops: total,
        valid_cell_hops: valid,
        sounding_cell_hops: sounding,
        numerical_missing_cell_hops: numerical,
        numerical_missing_rate: numeric_rate,
        missing_cells_by_reason: reasons,
        missing_rule_pass: numeric_rate.is_some_and(|r| r <= 0.05)
            && total > 0
            && moments[0].count as f64 / total as f64 >= 0.9,
    }
}
#[derive(Serialize)]
struct Stimulus {
    name: String,
    evaluation: Window,
    early: Window,
    late: Window,
    diagnostics: Value,
}
fn record_stimulus(name: &str, rows: &[Hop]) -> Result<Stimulus> {
    ensure!(
        rows.len() == HOPS,
        "{name}: expected {HOPS} hops, got {}",
        rows.len()
    );
    let dir = directory().join("stage2-traces");
    fs::create_dir_all(&dir)?;
    let mut file = BufWriter::new(fs::File::create(dir.join(format!("{name}.jsonl")))?);
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        writeln!(file)?;
    }
    file.flush()?;
    let mut ns = Vec::new();
    let mut ess_values = Vec::new();
    let mut vanished = 0;
    let mut zero = 0;
    let mut floor = Vec::new();
    let mut max_weight = Vec::new();
    let mut pred_min = Vec::new();
    for cell in rows.iter().flat_map(|r| &r.cells) {
        if let Some(m) = &cell.metrics {
            ns.push(m.n);
        }
        if let Some(x) = cell.ess {
            ess_values.push(x);
        }
        if let Some(x) = cell.floor_fraction {
            floor.push(x);
        }
        if let Some(x) = cell.max_weight {
            max_weight.push(x);
        }
        if let Some(x) = cell.min_pred_weight {
            pred_min.push(x);
        }
        vanished += cell.vanished;
        zero += cell.zero_particles;
    }
    let dist = |mut xs: Vec<f64>| {
        xs.sort_by(f64::total_cmp);
        let n = xs.len();
        json!({"count":n,"min":xs.first(),"median":median(xs.clone()),"p90":(n>0).then(||xs[((n-1) as f64*0.9).round() as usize]),"max":xs.last()})
    };
    Ok(Stimulus {
        name: name.into(),
        evaluation: window(rows, 5.0, 20.0),
        early: window(rows, 0.0, 5.0),
        late: window(rows, 15.0, 20.0),
        diagnostics: json!({"N_distribution":dist(ns),"ESS_distribution":dist(ess_values),"max_weight_distribution":dist(max_weight),
            "minimum_pred_weight_distribution":dist(pred_min),"floor_fraction_distribution":dist(floor),"newly_vanished_particles":vanished,"zero_particle_hops":zero,
            "per_cell_per_hop":"stage2-traces/<stimulus>.jsonl","diagnostic_interval_sec":[0,20]}),
    })
}
fn render_cases(config: &AppConfig) -> Result<Vec<(String, Vec<Hop>)>> {
    let dir = directory().join("stage2-renders");
    fs::create_dir_all(&dir)?;
    let status = Command::new("cargo")
        .args([
            "build",
            "--offline",
            "--release",
            "--bin",
            "conchordal-render",
        ])
        .status()?;
    ensure!(status.success(), "renderer build failed");
    let binary = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/conchordal-render");
    let config_path = dir.join("config.toml");
    fs::write(&config_path, toml::to_string_pretty(config)?)?;
    let stream = crate::runtime::info_terrain_efe_eval_stream(config, FS);
    let settings = effective_settings(
        config,
        &stream.last().space,
        FS,
        stream.hop_samples(),
        stream.window_samples(),
    );
    let mut cases = Vec::new();
    for (name, tone, residual) in [
        ("flora_sine_residual", "sine", true),
        ("flora_sine_only", "sine", false),
        ("flora_resonant_residual", "resonant", true),
    ] {
        let script = fs::read_to_string("samples/13_flora.rhai")?;
        ensure!(
            script.matches(".fade(2)").count() == 1,
            "Flora substitution ambiguous"
        );
        cases.push((
            format!("continuity_{name}"),
            script.replace(
                ".fade(2)",
                &format!(".fade(2).tone(\"{tone}\").residual({residual})"),
            ),
            5,
        ));
    }
    for name in [
        "06_bells",
        "08_murmuration",
        "11_autumn_cycle",
        "03_gravity",
        "09_rain",
        "12_emergence_and_resolution",
    ] {
        let mut script = fs::read_to_string(format!("samples/{name}.rhai"))?;
        if name == "03_gravity" {
            script.push_str("\nwait(6.0);\n");
        }
        cases.push((
            format!(
                "{}_{name}",
                if ["06_bells", "08_murmuration", "11_autumn_cycle"].contains(&name) {
                    "V6"
                } else {
                    "continuity"
                }
            ),
            script,
            11,
        ));
    }
    let mut rows = Vec::new();
    let mut equality = Vec::new();
    for (name, script, seed) in cases {
        let case = dir.join(&name);
        fs::create_dir_all(&case)?;
        fs::write(case.join("scenario.rhai"), script)?;
        let status = Command::new(&binary)
            .arg(case.join("scenario.rhai"))
            .arg("--config")
            .arg(&config_path)
            .arg("--seed")
            .arg(seed.to_string())
            .arg("--output")
            .arg(case.join("audio.wav"))
            .arg("--report")
            .arg(case.join("report.jsonl"))
            .env("RUST_LOG", "warn")
            .stdout(fs::File::create(case.join("stdout.log"))?)
            .stderr(fs::File::create(case.join("stderr.log"))?)
            .status()?;
        ensure!(status.success(), "render failed: {name}");
        let mut hops = Vec::new();
        let mut summary = false;
        let mut matches = 0;
        for line in BufReader::new(fs::File::open(case.join("report.jsonl"))?).lines() {
            let v: Value = serde_json::from_str(&line?)?;
            if v["type"] == "info_terrain_summary" {
                ensure!(v["dropped_blocks"] == 0, "dropped render blocks");
                summary = true;
            }
            if v["type"] != "info_terrain_block" || v["stream"] != "habitat" {
                continue;
            }
            ensure!(v["input_gap"] == false, "habitat gap: {name}");
            ensure!(
                v["efe"]["effective_analysis_settings"] == settings,
                "effective analysis settings differ: {name}"
            );
            matches += 1;
            for row in v["efe"]["trace"].as_array().context("A-3 trace absent")? {
                let hop: Hop = serde_json::from_value(row.clone())?;
                ensure!(hop.frame_id == hops.len() as u64, "habitat trace gap");
                hops.push(hop);
            }
        }
        ensure!(summary && hops.len() >= HOPS, "incomplete render: {name}");
        hops.truncate(HOPS);
        equality.push(json!({"stimulus":name,"harness":settings,"render":settings,"matching_report_blocks":matches,"equal":true}));
        rows.push((name, hops));
    }
    write_json(
        &directory().join("effective-analysis-equality.json"),
        &equality,
    )?;
    Ok(rows)
}
fn hypotheses(stimuli: &[Stimulus]) -> Value {
    let get = |name: &str| stimuli.iter().find(|s| s.name == name).unwrap();
    let compare = |a: &str, b: &str, index: usize, late_a: bool, greater: bool| {
        let a = get(a);
        let b = get(b);
        let wa = if late_a { &a.late } else { &a.evaluation };
        let wb = &b.evaluation;
        let pass = wa.missing_rule_pass
            && wb.missing_rule_pass
            && wa.means[index].zip(wb.means[index]).is_some_and(|(a, b)| {
                if greater {
                    a - b >= 1e-9
                } else {
                    b - a >= 1e-9
                }
            });
        json!({"a":a.name,"a_interval":wa.seconds,"a_value":wa.means[index],"b":b.name,"b_interval":wb.seconds,"b_value":wb.means[index],"greater":greater,"a_missing_rule_pass":wa.missing_rule_pass,"b_missing_rule_pass":wb.missing_rule_pass,"pass":pass})
    };
    let group =
        |pairs: Vec<Value>| json!({"pass":pairs.iter().all(|p|p["pass"]==true),"pairs":pairs});
    let mut ha = Vec::new();
    for a in ["V1_brown", "V2_band"] {
        for b in ["V3_chord", "V4_jumps"] {
            ha.push(compare(a, b, 0, false, true));
        }
    }
    let v6 = ["V6_06_bells", "V6_08_murmuration", "V6_11_autumn_cycle"];
    let mut hn1 = Vec::new();
    for a in v6 {
        for b in ["V1_brown", "V2_band", "V3_chord"] {
            hn1.push(compare(a, b, 1, false, true));
        }
    }
    let mut hv = Vec::new();
    for a in ["V1_brown", "V2_band", "V4_jumps"] {
        for b in v6 {
            hv.push(compare(a, b, 4, a == "V4_jumps", false));
        }
    }
    let jumps = get("V4_jumps");
    let early = &jumps.early;
    let late = &jumps.late;
    let dn = early.means[1].zip(late.means[1]).map(|(a, b)| a - b);
    let ds = early.means[2].zip(late.means[2]).map(|(a, b)| a - b);
    let eligible = early.missing_rule_pass && late.missing_rule_pass;
    let npositive = eligible && dn.is_some_and(|n| n >= 1e-9);
    let s_less_n = eligible && dn.zip(ds).is_some_and(|(n, s)| n - s >= 1e-9);
    let chord = get("V3_chord");
    json!({"H-A":group(ha),"H-N1":group(hn1),"H-V":group(hv),"H-N2":{"pass":npositive && s_less_n,"delta_N":dn,"delta_S":ds,
        "pairs":[{"relation":"delta_N > 0","pass":npositive},{"relation":"delta_S < delta_N","pass":s_less_n}],"early":early,"late":late,
        "V3_reference_delta_N":chord.early.means[1].zip(chord.late.means[1]).map(|(a,b)|a-b),"V3_early":chord.early,"V3_late":chord.late}})
}
fn stage2(config: &AppConfig) -> Result<()> {
    let mut stimuli = Vec::new();
    let dir = directory().join("stage2-audio");
    fs::create_dir_all(&dir)?;
    for (name, audio) in stage2_stimuli().into_iter().chain(controls::stimuli()) {
        let name = if name.starts_with('V') {
            name.to_string()
        } else {
            format!("continuity_{name}")
        };
        let path = dir.join(format!("{name}.wav"));
        float_wav(&path, &audio)?;
        let (fs, replay) = read_wav(&path)?;
        ensure!(fs == FS && replay == audio, "WAV replay mismatch");
        stimuli.push(record_stimulus(&name, &measure(&replay, config)?)?);
    }
    for (name, rows) in render_cases(config)? {
        stimuli.push(record_stimulus(&name, &rows)?);
    }
    let hypotheses = hypotheses(&stimuli);
    write_json(
        &directory().join("stage2.json"),
        &json!({"ran":true,"status":"stage 2 run: all stage-1 variants passed","metric_order":["A","N","S_x","R","V"],"stimuli":stimuli,"hypotheses":hypotheses}),
    )?;
    let mut md = String::from(
        "# A-3 stage 2\n\n| Stimulus | A | N | S_x | R | V | valid cell-hops | valid hops | numerical missing rate |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for s in &stimuli {
        md.push_str(&format!(
            "| {} | {:?} | {:?} | {:?} | {:?} | {:?} | {} | {}/{} | {:?} |\n",
            s.name,
            s.evaluation.means[0],
            s.evaluation.means[1],
            s.evaluation.means[2],
            s.evaluation.means[3],
            s.evaluation.means[4],
            s.evaluation.valid_cell_hops,
            s.evaluation.defined_hops[0],
            s.evaluation.total_hops,
            s.evaluation.numerical_missing_rate
        ));
    }
    md.push_str("\nHypotheses and all registered pairs:\n\n```json\n");
    md.push_str(&serde_json::to_string_pretty(&hypotheses)?);
    md.push_str("\n```\n");
    fs::write(directory().join("stage2.md"), md)?;
    Ok(())
}

#[test]
fn recovery_criteria_average_ties_missing_and_cross_talk() {
    assert_eq!(spearman(&[1.0, 1.0, 2.0], &[3.0, 3.0, 4.0]), Some(1.0));
    assert_eq!(spearman(&[1.0; 3], &[1.0, 2.0, 3.0]), None);
    let mut rows = Vec::new();
    for v in [1e-4_f64, 1e-3, 1e-2] {
        for s in [1e-3_f64, 1e-2, 1e-1] {
            rows.push(Recovery {
                variant: "test",
                path: "direct",
                seed: 1,
                truth_v: v,
                truth_s: s,
                median_v: Some(v),
                median_s: Some(s),
                log10_error_v: Some(0.0),
                log10_error_s: Some(0.0),
                value_v_pass: true,
                value_s_pass: true,
                valid_hops: 100,
                evaluation_hops: 100,
                valid_rate: 1.0,
                valid_rate_pass: true,
            });
        }
    }
    assert!(criteria(&rows).pass);
    rows[0].median_v = None;
    rows[0].valid_rate_pass = false;
    assert!(!criteria(&rows).pass);
}

#[test]
fn stage2_pair_counts_ties_and_missing_data_rules() {
    let names = [
        "V1_brown",
        "V2_band",
        "V3_chord",
        "V4_jumps",
        "V6_06_bells",
        "V6_08_murmuration",
        "V6_11_autumn_cycle",
    ];
    let mut stimuli: Vec<_> = names
        .into_iter()
        .map(|name| {
            let rows: Vec<_> = (0..HOPS)
                .map(|i| Hop {
                    frame_id: i as u64,
                    values: [Some(1.0); 5],
                    valid_cells: 1,
                    sounding_cells: 1,
                    cells: vec![CellHop {
                        energy: 1.0,
                        ..CellHop::missing("first")
                    }],
                })
                .collect();
            Stimulus {
                name: name.into(),
                evaluation: window(&rows, 5.0, 20.0),
                early: window(&rows, 0.0, 5.0),
                late: window(&rows, 15.0, 20.0),
                diagnostics: json!({}),
            }
        })
        .collect();
    let result = hypotheses(&stimuli);
    for (name, count) in [("H-A", 4), ("H-N1", 9), ("H-V", 9), ("H-N2", 2)] {
        assert_eq!(result[name]["pairs"].as_array().unwrap().len(), count);
        assert_eq!(result[name]["pass"], false);
    }
    stimuli[0].evaluation.means[0] = Some(2.0);
    stimuli[1].evaluation.means[0] = Some(2.0);
    assert_eq!(hypotheses(&stimuli)["H-A"]["pass"], true);
    stimuli[0].evaluation.missing_rule_pass = false;
    assert_eq!(hypotheses(&stimuli)["H-A"]["pass"], false);
    let rows: Vec<_> = (0..100)
        .map(|i| Hop {
            frame_id: i,
            values: if i < 89 { [Some(1.0); 5] } else { [None; 5] },
            valid_cells: usize::from(i < 89),
            sounding_cells: 1,
            cells: vec![CellHop::missing(if i < 6 { "numerical" } else { "first" })],
        })
        .collect();
    let w = window(&rows, 0.0, 5.0);
    assert!(!w.missing_rule_pass);
    assert_eq!(w.numerical_missing_rate, Some(0.06));
    assert_eq!(w.defined_hops[0], 89);
}
