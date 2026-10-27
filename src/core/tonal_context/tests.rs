use super::*;
use std::{fs, path::Path};
fn read(path: impl AsRef<Path>) -> Vec<f64> {
    fs::read(path)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect()
}
fn compare(name: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    let mut max = 0_f64;
    let mut norm = 0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        if b.is_nan() {
            assert!(a.is_nan());
            continue;
        }
        assert!(a.is_finite());
        max = max.max((a - b).abs());
        norm = norm.max(b.abs());
    }
    println!(
        "{name}: max_abs={max:.16e} ref_peak={norm:.16e} max_rel_peak={:.16e}",
        max / norm
    );
    assert!(max <= tolerance, "{name}: {max} > {tolerance}");
}
#[test]
fn undefined_and_half_life() {
    assert_eq!(pearson(&[1.; W], &[1.; W]), None);
    assert_eq!(pearson(&[f64::NAN; W], &[0.; W]), None);
    let mut p = Pitch::new();
    let mut impulse = [0.; W];
    impulse[0] = 1.;
    p.integrate(impulse);
    for _ in 0..100 {
        p.integrate([0.; W]);
    }
    assert!((p.local[0] - 2_f64.powf(-100. / (FA / STEP as f64 * 0.1))).abs() < 1e-14);
    assert!((p.glob[0] - 2_f64.powf(-100. / (FA / STEP as f64 * 1.5))).abs() < 1e-14);
}
#[test]
#[ignore = "External pinned C/NumPy acquisition; TC_FIXTURES selects retained evidence"]
fn reference_acquisition() {
    let root = std::env::var("TC_FIXTURES").unwrap();
    for entry in fs::read_dir(root).unwrap() {
        let dir = entry.unwrap().path();
        if !dir.join("metadata.json").exists() {
            continue;
        }
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap()).unwrap();
        println!("fixture {}", dir.display());
        let wav = hound::WavReader::open(dir.join("input.wav")).unwrap();
        let padded: Vec<f64> = wav
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f64 / 32768.)
            .collect();
        let raw_ref = read(dir.join("raw.f64"));
        let mut ani = Ani::new();
        let mut actual = Vec::new();
        let mut i = 0;
        while actual.len() < raw_ref.len() {
            if let Some(a) = ani.tick(padded.get(i).copied().unwrap_or(0.)) {
                actual.extend(a);
            }
            i += 1;
        }
        compare(
            "ANI raw",
            &actual,
            &raw_ref,
            meta["tolerance"]["raw"].as_f64().unwrap(),
        );
        let ani_ref = read(dir.join("ani.f64"));
        let mut p = Pitch::new();
        let mut pp = Vec::new();
        let mut l = Vec::new();
        let mut g = Vec::new();
        let mut tc = Vec::new();
        for a in ani_ref.as_chunks::<40>().0 {
            if let Some(f) = p.push(*a) {
                pp.extend(f.pp);
                l.extend(f.local);
                g.extend(f.glob);
                tc.push(f.tc.unwrap_or(f64::NAN));
            }
        }
        compare(
            "PP",
            &pp,
            &read(dir.join("pp.f64")),
            meta["tolerance"]["pp"].as_f64().unwrap(),
        );
        compare(
            "L",
            &l,
            &read(dir.join("local.f64")),
            meta["tolerance"]["local"].as_f64().unwrap(),
        );
        compare(
            "G",
            &g,
            &read(dir.join("glob.f64")),
            meta["tolerance"]["glob"].as_f64().unwrap(),
        );
        compare(
            "TC",
            &tc,
            &read(dir.join("tc.f64")),
            meta["tolerance"]["tc"].as_f64().unwrap(),
        );
        let pcm = read(dir.join("pcm.f64"));
        let fs = meta["fs"].as_u64().unwrap() as usize;
        let mut model = Model::new(fs);
        let mut observed = Vec::new();
        for &x in &pcm {
            if let Some(f) = model.push(x) {
                observed.push(f);
            }
        }
        // Complete the input FIR, then reproduce C file padding and drain.
        let converted = pcm.len() * model.input.p;
        let target = converted.div_ceil(model.input.q);
        while model.input.next < target {
            if let Some(f) = model.push(0.) {
                observed.push(f);
            }
        }
        while model.raw_count < raw_ref.len() / 40 {
            if let Some(f) = model.raw(0.).1 {
                observed.push(f);
            }
        }
        let count = tc.len().min(observed.len());
        compare(
            "end-to-end TC",
            &observed[..count]
                .iter()
                .map(|f| f.tc.unwrap_or(f64::NAN))
                .collect::<Vec<_>>(),
            &tc[..count],
            meta["tolerance"]["tc"].as_f64().unwrap(),
        );
        assert_eq!(count, tc.len(), "missing completed PP frames");
        for (label, actual, key) in [
            (
                "end-to-end PP",
                observed[..count]
                    .iter()
                    .flat_map(|f| f.pp)
                    .collect::<Vec<_>>(),
                "pp",
            ),
            (
                "end-to-end L",
                observed[..count]
                    .iter()
                    .flat_map(|f| f.local)
                    .collect::<Vec<_>>(),
                "local",
            ),
            (
                "end-to-end G",
                observed[..count]
                    .iter()
                    .flat_map(|f| f.glob)
                    .collect::<Vec<_>>(),
                "glob",
            ),
        ] {
            compare(
                label,
                &actual,
                &read(dir.join(format!("{key}.f64"))),
                meta["tolerance"][key].as_f64().unwrap(),
            );
        }
        let mut converter = Resampler::<1>::new(model.input.p, model.input.q);
        let mut converted = Vec::new();
        for &x in &pcm {
            if let Some([y]) = converter.push([x]) {
                converted.push(y);
            }
        }
        while converted.len() < target {
            if let Some([y]) = converter.push([0.]) {
                converted.push(y);
            }
        }
        compare(
            "resampled PCM",
            &converted,
            &read(dir.join("converted.f64")),
            0.5 / 32768.,
        );
        let mut down = Resampler::<40>::new(1, 4);
        let trim = read(dir.join("trim.f64"));
        let mut downsampled = Vec::new();
        for a in trim.as_chunks::<40>().0 {
            if let Some(a) = down.push(*a) {
                downsampled.extend(a);
            }
        }
        while downsampled.len() < ani_ref.len() {
            if let Some(a) = down.push([0.; 40]) {
                downsampled.extend(a);
            }
        }
        compare(
            "resampled ANI",
            &downsampled,
            &ani_ref,
            meta["tolerance"]["ani"].as_f64().unwrap(),
        );

        // Transport block boundaries never enter the state equations.
        let mut blocked = Model::new(fs);
        let mut blocked_tc = Vec::new();
        let mut cursor = 0;
        let sizes = [1, 511, 512, 37, 1024];
        let mut k = 0;
        while cursor < pcm.len() {
            let end = (cursor + sizes[k % 5]).min(pcm.len());
            for &x in &pcm[cursor..end] {
                if let Some(f) = blocked.push(x) {
                    blocked_tc.push(f.tc);
                }
            }
            cursor = end;
            k += 1;
        }
        assert_eq!(
            blocked_tc,
            observed[..blocked_tc.len()]
                .iter()
                .map(|f| f.tc)
                .collect::<Vec<_>>()
        );
        if let Some(n0) = meta["n0"].as_u64() {
            let n0 = n0 as usize;
            let hz = meta["hz"].as_f64().unwrap();
            let mut prefix = Model::new(48000);
            for &x in &pcm[..n0] {
                prefix.push(x);
            }
            let (q, j, tail) = prefix.query(hz).unwrap();
            let qr = meta["q"].as_f64().unwrap();
            compare("q", &[q], &[qr], meta["tolerance"]["q"].as_f64().unwrap());
            assert_eq!(j, meta["q_frames"].as_array().unwrap().len());
            assert_eq!(prefix.n0, n0);
            assert_eq!(prefix.query(hz).unwrap(), (q, j, tail));
            println!("query hz={hz} q={q:.16e} J={j} tail_samples={tail}");
        }
    }
}

#[test]
#[ignore = "Run the compiled binary only while holding TC1a-measure timing lock"]
fn cost_and_approximation() {
    use std::{hint::black_box, time::Instant};
    let root = std::env::var("TC_FIXTURES").unwrap();
    let pcm = read(Path::new(&root).join("major-220/pcm.f64"));
    let mut context = Model::new(48000);
    for &x in &pcm[..38400] {
        context.push(x);
    }
    let own_pcm: Vec<f64> = (0..38400)
        .map(|n| 0.03 * (2. * std::f64::consts::PI * 196. * n as f64 / 48000.).sin())
        .collect();
    let mut mixed = context.clone();
    let mut own = context.clone();
    let mut mixed_us = Vec::new();
    let mut loo_us = Vec::new();
    for hop in 0..256 {
        let start = Instant::now();
        for i in 0..512 {
            black_box(mixed.push(black_box(pcm[(hop * 512 + i) % 38400])));
        }
        if hop >= 16 {
            mixed_us.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let start = Instant::now();
        for i in 0..512 {
            let m = pcm[(hop * 512 + i) % 38400];
            let x = own_pcm[(hop * 512 + i) % 38400];
            black_box(own.push(black_box(m - x)));
        }
        if hop >= 16 {
            loo_us.push(start.elapsed().as_secs_f64() * 1e6);
        }
    }
    for (label, mut values) in [("mixed_hop_us", mixed_us), ("loo_hop_us", loo_us)] {
        values.sort_by(f64::total_cmp);
        println!(
            "COST {label} mean={} p50={} p95={} p99={} max={}",
            values.iter().sum::<f64>() / values.len() as f64,
            values[values.len() / 2],
            values[values.len() * 95 / 100],
            values[values.len() * 99 / 100],
            values[values.len() - 1]
        );
    }
    let frequencies = [220., 233.081880759, 275., 330.];
    let first = ((context.n0 + 9600) * 147).div_ceil(STEP * 2560);
    let end = ((context.n0 + 28800) * 147).div_ceil(STEP * 2560);
    let mut tables = Vec::new();
    for &hz in &frequencies {
        let mut blank = Model::new(48000);
        for _ in 0..38400 {
            blank.push(0.);
        }
        let mut table = Vec::new();
        let mut j = 0;
        loop {
            let u = j as f64 / 48000.;
            let e = if j >= 28800 {
                0.
            } else if u < 0.01 {
                (1. - (std::f64::consts::PI * u / 0.01).cos()) / 2.
            } else if u >= 0.59 {
                (1. - (std::f64::consts::PI * (0.6 - u) / 0.01).cos()) / 2.
            } else {
                1.
            };
            if let Some(f) = blank.push(0.1 * e * (2. * std::f64::consts::PI * hz * u).sin()) {
                let done = f.index + 1 >= end;
                table.push(f);
                if done {
                    break;
                }
            }
            j += 1;
        }
        tables.push(table);
    }
    for name in ["major-220", "minor-220"] {
        let pcm = read(Path::new(&root).join(name).join("pcm.f64"));
        let mut ctx = Model::new(48000);
        for &x in &pcm[..38400] {
            ctx.push(x);
        }
        for (i, &hz) in frequencies.iter().enumerate() {
            let mut times = Vec::new();
            let mut q = 0.;
            for _ in 0..7 {
                let now = Instant::now();
                q = black_box(ctx.query(black_box(hz)).unwrap().0);
                times.push(now.elapsed().as_secs_f64() * 1e6);
            }
            times.sort_by(f64::total_cmp);
            let approx = || {
                let ctx = black_box(&ctx);
                let mut local = ctx.pitch.local;
                let mut glob = ctx.pitch.glob;
                let mut sum = 0.;
                let mut count = 0;
                let al = 2_f64.powf(-1. / (FA / STEP as f64 * 0.1));
                let ag = 2_f64.powf(-1. / (FA / STEP as f64 * 1.5));
                for f in &tables[i] {
                    for n in 0..W {
                        local[n] = al * local[n] + f.pp[n];
                        glob[n] = ag * glob[n] + f.pp[n];
                    }
                    if f.index >= first {
                        sum += pearson(&local, &glob).unwrap();
                        count += 1;
                    }
                }
                sum / count as f64
            };
            let now = Instant::now();
            let mut qa = 0.;
            for _ in 0..100 {
                qa = black_box(approx());
            }
            let approx_us = now.elapsed().as_secs_f64() * 1e6 / 100.;
            println!(
                "QUERY context={name} hz={hz} q={q:.16e} precomputed_q={qa:.16e} error={:.16e} direct_us_p50={} direct_us_min={} direct_us_max={} precomputed_us={approx_us}",
                qa - q,
                times[3],
                times[0],
                times[6]
            );
        }
    }
    println!(
        "STATE static_bytes={} heap_bytes={} coefficients_shared_bytes={}",
        std::mem::size_of::<Model>(),
        context.input.ring.len() * 8
            + context.down.ring.len() * 40 * 8
            + context.pitch.fani.len() * 40 * 8,
        (context.input.h.len() + context.down.h.len()) * 8
    );
}

#[test]
fn retained_c_numpy_smoke_and_snapshot_isolation() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!("smoke.json")).unwrap();
    let indices: Vec<usize> = fixture["raw_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut model = Model::new(44100);
    let mut frames = Vec::new();
    let mut pcm = Vec::new();
    for n in 0..13230 {
        pcm.push(
            (0.1 * (2. * std::f64::consts::PI * 200. * n as f64 / 44100.).sin() * 32768.).round()
                / 32768.,
        );
    }
    for chunk in pcm.chunks(511) {
        for &x in chunk {
            if let Some(f) = model.push(x) {
                frames.push(f);
            }
        }
    }
    while frames.len() < 18 {
        if let Some(f) = model.push(0.) {
            frames.push(f);
        }
    }
    let expected: Vec<f64> = fixture["pp_first"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    compare(
        "smoke PP",
        &frames[0].pp,
        &expected,
        fixture["tolerance"]["pp"].as_f64().unwrap(),
    );
    let expected: Vec<f64> = fixture["tc"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    compare(
        "smoke TC",
        &frames.iter().map(|f| f.tc.unwrap()).collect::<Vec<_>>(),
        &expected,
        fixture["tolerance"]["tc"].as_f64().unwrap(),
    );
    let mut converter = Resampler::<1>::new(1, 2);
    let mut converted = Vec::new();
    for &x in &pcm {
        if let Some([v]) = converter.push([x]) {
            converted.push((v * 32768.).round() / 32768.);
        }
    }
    let mut ani = Ani::new();
    let mut i = 0;
    let mut captured = Vec::new();
    for x in std::iter::repeat_n(0., 441).chain(converted) {
        if let Some(raw) = ani.tick(x) {
            if indices.contains(&i) {
                captured.extend(raw);
            }
            i += 1;
        }
    }
    let expected: Vec<f64> = fixture["raw"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect();
    compare("smoke raw", &captured, &expected, 1e-10);
    assert!(model.query(220.).is_none());
    let mut ctx = Model::new(48000);
    for n in 0..12000 {
        ctx.push(0.1 * (2. * std::f64::consts::PI * 220. * n as f64 / 48000.).sin());
    }
    let q = ctx.query(220.).unwrap();
    ctx.query(330.).unwrap();
    assert_eq!(q, ctx.query(220.).unwrap());
    assert_eq!(ctx.n0, 12000);
    ctx.push(f64::NAN);
    assert!(ctx.query(220.).is_none());
}
