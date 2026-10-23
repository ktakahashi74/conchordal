//! AUD-M1 addendum 4: fixed inputs, cumulative Flora, and interval extrema.
use super::*;
use crate::life::flora::Flora;
use std::{collections::BTreeMap, io::Write};

pub(super) fn span_extrema(space: &Log2Space, scan: &[f32], low: f32, high: f32) -> (f32, f32) {
    space.assert_scan_len_named(scan, "interval_consumer_scan");
    let a = crate::core::log2space::sample_scan_linear_log2(space, scan, low);
    let b = crate::core::log2space::sample_scan_linear_log2(space, scan, high);
    let mut minimum = a.min(b);
    let mut maximum = a.max(b);
    for (i, &f) in space.centers_hz.iter().enumerate() {
        if (low..=high).contains(&f) {
            minimum = minimum.min(scan[i]);
            maximum = maximum.max(scan[i]);
        }
    }
    (minimum, maximum)
}

#[test]
#[ignore = "AUD-M1 addendum 4 registered window acquisition"]
fn aud_m1_a4_windows() {
    use sha2::Digest;
    let input = PathBuf::from(std::env::var("AUD_M1_INPUT").unwrap());
    let dir = output_dir();
    let manifest_bytes = std::fs::read(input.join("inputs-manifest.json")).unwrap();
    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let mut legacy = BTreeMap::new();
    for line in std::fs::read_to_string(input.join("v2-rows.jsonl"))
        .unwrap()
        .lines()
    {
        let row: Value = serde_json::from_str(line).unwrap();
        if row["variant"] == 4 {
            legacy.insert(
                (
                    row["fixture"].as_str().unwrap().to_string(),
                    row["frame"].as_u64().unwrap() as usize,
                ),
                row,
            );
        }
    }
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let (_, du) = erb_grid(&space);
    let reference_peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let template = ShadowFrontEnd::new(space.clone(), params.fs, 1.0 / 3.0, false, kernel.nfft());
    let mut writer =
        std::io::BufWriter::new(std::fs::File::create(dir.join("window-rows.jsonl")).unwrap());
    let mut summaries = Vec::new();
    let info = json!({"source_manifest_sha256":sha2::Sha256::digest(&manifest_bytes).iter().map(|b|format!("{b:02x}")).collect::<String>(),
        "input_directory":input,"beta":1.0/3.0,"s1":false,"spread":false,
        "flora":"six persistent states, sign of final biomass minus initial biomass over the registered 8-hop window",
        "close_pair_gate":"every observed frame: nonnegative C-maximum excess and R-minimum deficit no greater than current, no tolerance"});
    std::fs::write(
        dir.join("protocol.json"),
        serde_json::to_vec_pretty(&info).unwrap(),
    )
    .unwrap();
    for fixture in manifest["fixtures"].as_array().unwrap() {
        let name = fixture["name"].as_str().unwrap();
        let category = fixture["category"].as_str().unwrap();
        let bytes = std::fs::read(
            input
                .join("reference")
                .join(fixture["pcm"].as_str().unwrap()),
        )
        .unwrap();
        let hash = sha2::Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(hash, fixture["pcm_sha256"].as_str().unwrap());
        let pcm: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let frequencies: Vec<_> = fixture["frequencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let amplitudes: Vec<_> = fixture["amplitudes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let observe: Vec<_> = fixture["observation_hops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let onset = fixture["onset_hop"].as_u64().unwrap() as usize;
        let transient = category == "transient";
        let pair = matches!(
            category,
            "loo_close" | "close_pair_15hz" | "close_mode_body"
        );
        let separation = if pair {
            erbn(f64::from(frequencies[1])) - erbn(f64::from(frequencies[0]))
        } else {
            0.0
        };
        let mut stationary = vec![0.0; space.n_bins()];
        for (&f, &a) in frequencies.iter().zip(&amplitudes) {
            deposit(
                &space,
                &du,
                &mut stationary,
                f,
                (a_weighting_gain_pow(f) * a * a * 0.5 / params.ref_power)
                    .powf(params.loudness_exp),
            );
        }
        let mut truth_density = if transient {
            vec![0.0; space.n_bins()]
        } else {
            stationary.clone()
        };
        let leak = (-(hop as f32) / params.fs / (params.tau_ms * 1e-3)).exp();
        let mut front = template.clone();
        let mut current = SynchronousAnalysis::new(kernel.clone(), &params);
        let mut populations: Option<[Vec<Flora>; 3]> = None;
        let mut cases = Vec::new();
        for (frame, chunk) in pcm.chunks_exact(hop).enumerate() {
            if transient {
                for (old, &target) in truth_density.iter_mut().zip(&stationary) {
                    *old = leak * *old
                        + if frame >= onset {
                            (1.0 - leak) * target
                        } else {
                            0.0
                        };
                }
            }
            front.process(chunk, &params);
            current.process(chunk, &params);
            if !observe.contains(&frame) {
                continue;
            }
            let truth = evaluate(
                &truth_density,
                &space,
                &params,
                &mut scratch,
                reference_peak,
            );
            let actual = evaluate(
                &front.density_scan,
                &space,
                &params,
                &mut scratch,
                reference_peak,
            );
            let old = &current.landscape;
            let previous = &legacy[&(name.to_string(), frame)];
            assert_eq!(
                field_errors(&actual, &truth),
                previous["new"]["fields"],
                "unchanged candidate producer"
            );
            assert_eq!(
                field_errors(old, &truth),
                previous["old"]["fields"],
                "unchanged current producer"
            );
            let states = populations.get_or_insert_with(|| {
                let initial = Flora::aud_m1_start_window(&truth);
                [initial.clone(), initial.clone(), initial]
            });
            for (population, landscape) in states.iter_mut().zip([&truth, &actual, old]) {
                for flora in population {
                    flora.advance(landscape);
                }
            }
            let interval = if pair {
                let low = frequencies[0].min(frequencies[1]);
                let high = frequencies[0].max(frequencies[1]);
                let c = [&truth, &actual, old]
                    .map(|l| span_extrema(&space, &l.consonance_field_score_eff, low, high).1);
                let r = [&truth, &actual, old]
                    .map(|l| span_extrema(&space, &l.roughness01, low, high).0);
                let dc = [
                    (f64::from(c[1]) - f64::from(c[0])).max(0.0),
                    (f64::from(c[2]) - f64::from(c[0])).max(0.0),
                ];
                let dr = [
                    (f64::from(r[0]) - f64::from(r[1])).max(0.0),
                    (f64::from(r[0]) - f64::from(r[2])).max(0.0),
                ];
                json!({"low_hz":low,"high_hz":high,"erb_separation":separation,
                    "c_max_ref_new_current":c,"r_min_ref_new_current":r,"c_excess_new_current":dc,"r_deficit_new_current":dr,
                    "c_pass":dc[0]<=dc[1],"r_pass":dr[0]<=dr[1]})
            } else {
                Value::Null
            };
            cases.push(json!({"frame":frame,"interval":interval,"legacy":previous}));
            writeln!(writer,"{}",json!({"fixture":name,"category":category,"frame":frame,"interval":interval,
                "truth_h01":truth.harmonicity01,"truth_r01":truth.roughness01,"truth_c":truth.consonance_field_score,
                "new_h01":actual.harmonicity01,"new_r01":actual.roughness01,"new_c":actual.consonance_field_score,
                "current_h01":old.harmonicity01,"current_r01":old.roughness01,"current_c":old.consonance_field_score})).unwrap();
        }
        let states = populations.unwrap();
        let signs = states
            .iter()
            .map(|s| Flora::aud_m1_window_signs(s))
            .collect::<Vec<_>>();
        let mismatches = [1, 2].map(|j| {
            signs[j]
                .iter()
                .zip(&signs[0])
                .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
                .sum::<usize>()
        });
        let fake_pass = !pair
            || cases
                .iter()
                .all(|r| r["interval"]["c_pass"] == true && r["interval"]["r_pass"] == true);
        let summary = json!({"fixture":fixture,"category":category,"erb_separation":separation,"cumulative_flora_new_current":mismatches,
            "flora_pass":mismatches[0]<=mismatches[1],"interval_pass":fake_pass,"observations":cases});
        println!(
            "AUD_M1_A4_CASE {}",
            json!({"fixture":name,"flora_new_current":mismatches,"interval_pass":fake_pass,"erb":separation})
        );
        summaries.push(summary);
    }
    writer.flush().unwrap();
    std::fs::write(
        dir.join("window-summary.json"),
        serde_json::to_vec_pretty(&summaries).unwrap(),
    )
    .unwrap();
}
