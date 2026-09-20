use super::*;

fn frames(base: u64) -> Vec<Frame> {
    (0..20)
        .map(|i| {
            let projected = i >= 10;
            let value = |x| Feature::from(Some(x), projected);
            let rms: f64 = if projected { 0.1 } else { 0.005 };
            let mut raw = [Feature::Unsupported; 10];
            raw[2] = value(rms.log2());
            for k in 0..4 {
                raw[3 + k] = value((k + 1) as f64 * if projected { 0.2 } else { 0.1 });
            }
            Frame {
                start: base + i * 100,
                end: base + (i + 1) * 100,
                source_end: base + (i + 1) * 100,
                available: base + (i + 1) * 100,
                raw,
                energy: value(rms * rms),
            }
        })
        .collect()
}

#[test]
fn physical_windows_keep_observed_and_projected_support_separate() {
    for base in [0, (1_u64 << 53) + 99] {
        let frames = frames(base);
        let summary = summarize(
            base,
            base + 2000,
            base + 1000,
            1000,
            1.,
            frames.iter().copied(),
        )
        .unwrap();
        assert_eq!(summary.observed_fraction, [0.5; 7]);
        assert_eq!(summary.projected_fraction, [0.5; 7]);
        for (actual, expected) in
            summary
                .values
                .iter()
                .zip([0.15, 0.3, 0.45, 0.6, 0.5, 0.0525, 20_f64.log2() / 1.9])
        {
            assert!(matches!(actual, Feature::Projected(_)));
            assert!((actual.value().unwrap() - expected).abs() < 1e-14);
        }
        let clipped = summarize(
            base + 50,
            base + 1950,
            base + 1000,
            1000,
            1.,
            frames.iter().copied(),
        )
        .unwrap();
        assert!((clipped.values[6].value().unwrap() - 20_f64.log2() / 1.85).abs() < 1e-14);
        let old = summarize(
            base,
            base + 1000,
            base + 1000,
            1000,
            1.,
            frames.iter().copied(),
        )
        .unwrap();
        assert_eq!(old.projected_fraction, [0.; 7]);
        assert!(old.values.iter().all(|v| matches!(v, Feature::Observed(_))));
        let future = summarize(
            base + 1000,
            base + 2000,
            base + 1000,
            1000,
            1.,
            frames.iter().copied(),
        )
        .unwrap();
        assert_eq!(future.observed_fraction, [0.; 7]);
        assert_eq!(future.projected_fraction, [1.; 7]);
    }
}

#[test]
fn missing_support_never_becomes_silence_or_a_replacement_endpoint() {
    let mut input = frames(0);
    input[0].raw[2] = Feature::Unsupported;
    let out = summarize(0, 2000, 1000, 1000, 1., input.iter().copied()).unwrap();
    assert_eq!(out.values[6], Feature::Unsupported);
    assert_eq!(out.observed_fraction[6], 0.);
    for row in &mut input[..2] {
        row.raw[3] = Feature::Unsupported;
    }
    let out = summarize(0, 2000, 1000, 1000, 1., input.iter().copied()).unwrap();
    assert!(out.values[0].value().is_some());
    assert_eq!(out.observed_fraction[0], 0.4);
    input[2].raw[3] = Feature::Unsupported;
    let out = summarize(0, 2000, 1000, 1000, 1., input.iter().copied()).unwrap();
    assert_eq!(out.values[0], Feature::Unsupported);
    assert!(out.values[1..6].iter().all(|v| v.value().is_some()));
    for row in &mut input {
        let projected = row.start >= 1000;
        row.raw[2] = Feature::from(Some(1e-6_f64.log2()), projected);
        row.energy = Feature::from(Some(0.), projected);
    }
    let out = summarize(0, 2000, 1000, 1000, 1., input.iter().copied()).unwrap();
    assert_eq!(
        out.values[4..],
        [
            Feature::Projected(1.),
            Feature::Projected(0.),
            Feature::Projected(0.)
        ]
    );
    let empty = summarize(1000, 1000, 1000, 1000, 1., input.iter().copied()).unwrap();
    assert_eq!(empty.values, [Feature::Unsupported; 7]);
}

#[test]
fn invalid_clocks_provenance_and_overlapping_support_are_rejected() {
    for fault in 0..9 {
        let mut input = frames(0);
        match fault {
            0 => input[10].raw[3] = Feature::Observed(0.),
            1 => input[9].energy = Feature::Projected(0.),
            2 => input[0].available = 1001,
            3 => input[0].source_end = 99,
            4 => input[0].source_end = 101,
            5 => input[1].start = 99,
            6 => input[0].raw[3] = Feature::Observed(f64::NAN),
            7 => input[0].energy = Feature::Observed(-1.),
            8 => input[0].end = 0,
            _ => unreachable!(),
        }
        assert!(
            summarize(0, 2000, 1000, 1000, 1., input.into_iter()).is_err(),
            "fault {fault}"
        );
    }
    let mut input = frames(0);
    input[9].end = 1001;
    input[9].source_end = 1001;
    input[9].available = 1001;
    // Clipping an unavailable observed hop does not make it available at the issue.
    assert!(summarize(0, 1000, 1000, 1000, 1., input.into_iter()).is_err());
}

#[test]
#[ignore = "explicit arithmetic replay over frozen counterfactual targets; not prediction accuracy"]
fn replay_actual_targets_as_hypothetical_window_inputs() {
    use serde_json::{Value, json};
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};
    use std::path::Path;
    let root = std::env::var("CONCHORDAL_I10_FEATURE_DIR").unwrap();
    let output = std::env::var("CONCHORDAL_I10_WINDOW_OUTPUT").unwrap();
    let root = Path::new(&root);
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "i10-actual-action-feature-targets-v1");
    let issue = manifest["source_manifest"]["issue_sample"]
        .as_u64()
        .unwrap();
    let mut output = BufWriter::new(File::create_new(output).unwrap());
    let mut count = 0;
    for name in manifest["source_manifest"]["cases"].as_array().unwrap() {
        let name = name.as_str().unwrap();
        let path = root.join(name);
        let mut files: Vec<_> = fs::read_dir(&path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|p| {
                p.extension().is_some_and(|e| e == "jsonl")
                    && !p
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .starts_with("prefix-")
            })
            .collect();
        files.sort();
        for file in files {
            let stem = file.file_stem().unwrap().to_str().unwrap();
            let bus = stem.chars().last().unwrap();
            let prefix = fs::read_to_string(path.join(format!("prefix-bus{bus}.jsonl"))).unwrap();
            let future = fs::read_to_string(&file).unwrap();
            let frames: Vec<_> = prefix
                .lines()
                .chain(future.lines())
                .map(|line| {
                    let row: Value = serde_json::from_str(line).unwrap();
                    let raw = &row["raw"];
                    let start = raw["start"].as_u64().unwrap();
                    let feature = |value: &Value| Feature::from(value.as_f64(), start >= issue);
                    Frame {
                        start,
                        end: raw["end"].as_u64().unwrap(),
                        source_end: raw["source_end"].as_u64().unwrap(),
                        available: raw["available_end"].as_u64().unwrap(),
                        raw: std::array::from_fn(|i| feature(&raw["values"][i])),
                        energy: feature(&row["energy"]),
                    }
                })
                .collect();
            assert_eq!(frames.len(), 404);
            for cell in 0..32_u64 {
                let end = issue + (192000 * cell + 15) / 31;
                for width in [12000, 96000, 384000] {
                    let start = end.saturating_sub(width);
                    let result =
                        summarize(start, end, issue, 48000, 1., frames.iter().copied()).unwrap();
                    serde_json::to_writer(
                        &mut output,
                        &json!({
                            "case": name, "stream": stem, "cell": cell, "width_samples": width,
                            "start": start, "end": end, "issue": issue, "summary": result,
                        }),
                    )
                    .unwrap();
                    writeln!(output).unwrap();
                    count += 1;
                }
            }
        }
        println!("window arithmetic replay {name}: {count} rows");
    }
    output.flush().unwrap();
    assert_eq!(count, 76032);
}
