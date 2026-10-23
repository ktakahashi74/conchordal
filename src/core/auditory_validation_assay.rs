//! AUD-M1 addendum 5: 47-hop Flora and unchanged eight-hop consumers.
use super::*;
use crate::life::{community::Community, flora::Flora};
use std::{collections::BTreeMap, io::Write};

fn assess(actual: &Landscape, truth: &Landscape, anchors: &[f32]) -> Value {
    let space = &truth.space;
    let hill: Vec<_> = anchors.iter().map(|&a| {
        let (correct, margin) = crate::life::voice::pitch_core::aud_m1_hill_choice(truth, a, anchors);
        let (chosen, _) = crate::life::voice::pitch_core::aud_m1_hill_choice(actual, a, anchors);
        let distance = 1200.0 * (chosen - correct).abs();
        let moved = chosen.to_bits() != a.to_bits();
        let truth_moved = correct.to_bits() != a.to_bits();
        json!({"anchor_hz":a.exp2(),"chosen_hz":chosen.exp2(),"truth_hz":correct.exp2(),"distance_cent":distance,
            "moved":moved,"truth_moved":truth_moved,"truth_margin":margin,"mismatch":distance>6.25 || moved!=truth_moved})
    }).collect();
    let pitch = hill.iter().filter(|v| v["mismatch"] == true).count();
    let ranges = [(space.fmin, space.fmax), (110.0, 880.0), (440.0, 3520.0)];
    let birth = ranges.map(|range| {
        let a = Community::aud_m1_range_mass(actual, range);
        let b = Community::aud_m1_range_mass(truth, range);
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>()
    });
    let peaks = Community::aud_m1_respawn(actual);
    let targets = Community::aud_m1_respawn(truth);
    let respawn = targets
        .iter()
        .filter(|&&i| {
            !peaks
                .iter()
                .any(|&j| 1200.0 * (space.centers_log2[i] - space.centers_log2[j]).abs() <= 6.25)
        })
        .count();
    let a = Flora::aud_m1_cell_signs(actual);
    let b = Flora::aud_m1_cell_signs(truth);
    let flora = a
        .iter()
        .zip(&b)
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum::<usize>();
    json!({"pitch_errors":pitch,"hill":hill,"birth_l1":birth,"flora_errors":flora,"respawn_errors":respawn,
        "respawn_peaks":peaks,"respawn_targets":targets.len(),"fields":field_errors(actual,truth)})
}

pub(super) fn sign_errors(populations: &[Vec<Flora>; 3]) -> ([usize; 2], Value) {
    let signs = populations
        .each_ref()
        .map(|s| Flora::aud_m1_window_signs(s));
    let errors = [1, 2].map(|i| {
        signs[i]
            .iter()
            .zip(&signs[0])
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
            .sum()
    });
    (errors, json!(signs))
}

#[test]
#[ignore = "AUD-M1 addendum 5 fixed 47-hop acquisition"]
fn aud_m1_a5_windows() {
    acquire_windows(false);
}

#[test]
#[ignore = "AUD-M1 addendum 6 boxcar and context acquisition"]
fn aud_m1_a6_windows() {
    acquire_windows(true);
}

fn acquire_windows(addendum6: bool) {
    use sha2::Digest;
    let input = PathBuf::from(std::env::var("AUD_M1_INPUT").unwrap());
    let dir = output_dir();
    let spread = std::env::var("AUD_M1_SPREAD").is_ok_and(|v| v == "1");
    let selected_s1 = addendum6 && std::env::var("AUD_M1_S1").is_ok_and(|v| v == "1");
    let reference_context =
        addendum6 && std::env::var("AUD_M1_EXACT_CONTEXT").is_ok_and(|v| v == "1");
    let selected_variant = std::env::var("AUD_M1_VARIANT")
        .ok()
        .map(|s| s.parse::<usize>().unwrap());
    let bytes = std::fs::read(input.join("manifest.json")).unwrap();
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    let source = input
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("addendum3-final/v2-rows.jsonl");
    let mut legacy = BTreeMap::new();
    for line in std::fs::read_to_string(source).unwrap().lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        legacy.insert(
            (
                row["fixture"].as_str().unwrap().to_owned(),
                row["frame"].as_u64().unwrap() as usize,
                row["variant"].as_u64().unwrap() as usize,
            ),
            row,
        );
    }
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let (_, du) = erb_grid(&space);
    let peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let variants: Vec<_> = (0..8)
        .filter(|&i| {
            (!(spread || addendum6) || i >= 4)
                && selected_variant.is_none_or(|v| v == i)
                && (!reference_context || i % 2 == 1)
        })
        .collect();
    let templates: Vec<_> = variants
        .iter()
        .map(|&i| {
            let beta = [1.0, 0.5, 1.0 / 3.0, 0.25][i / 2];
            let front = ShadowFrontEnd::new(
                space.clone(),
                params.fs,
                beta,
                if addendum6 { selected_s1 } else { i % 2 == 1 },
                kernel.nfft(),
            );
            if addendum6 {
                let mut front = front.with_statistics(hop, i % 2 == 1);
                front.reference_context = reference_context;
                front
            } else if spread {
                front.with_gaussian_spread()
            } else {
                front
            }
        })
        .collect();
    let mut rows =
        std::io::BufWriter::new(std::fs::File::create(dir.join("consumer-rows.jsonl")).unwrap());
    let mut diagnostics = std::io::BufWriter::new(
        std::fs::File::create(dir.join("frame-diagnostics.jsonl")).unwrap(),
    );
    let mut summary = Vec::new();
    let mut onset_rows =
        std::io::BufWriter::new(std::fs::File::create(dir.join("onset-rows.jsonl")).unwrap());
    std::fs::write(dir.join("protocol.json"),serde_json::to_vec_pretty(&json!({
        "manifest_sha256":sha2::Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect::<String>(),"spread":spread,"variants":variants,
        "addendum6":addendum6,"boxcar":addendum6,"s1":selected_s1,
        "reference_context":reference_context,
        "a6_variants":"4=beta1/3 narrow; 5=beta1/3 ERB; 6=beta1/4 narrow; 7=beta1/4 ERB",
        "unresolved_interval_gate":!addendum6,
        "flora_hops":47,"flora_diagnostic_hops":8,"other_consumer_hops":8,"no_tie_exclusion":true,
        "boundary":"literal numerical ERB difference >=0.75; mixed complexes retain original full-consumer diagnostic and every unresolved pair is inspected",
        "gaussian":"normal in Hz integrated against Log2Space piecewise linear interpolation hats; 8-sigma tails omitted (<1.3e-15 probability)",
        "coherence":"current hop only; lag output history advances on every sample; denominator zero means kappa=1"
    })).unwrap()).unwrap();
    for fixture in manifest["fixtures"].as_array().unwrap() {
        let name = fixture["name"].as_str().unwrap();
        let pcm_bytes = std::fs::read(input.join(fixture["pcm"].as_str().unwrap())).unwrap();
        assert_eq!(
            sha2::Sha256::digest(&pcm_bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            fixture["pcm_sha256"].as_str().unwrap()
        );
        let pcm: Vec<_> = pcm_bytes
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
        let observe: Vec<_> = fixture["judgment_hops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let flora_hops: Vec<_> = fixture["flora_hops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(flora_hops.len(), 47);
        let onset = fixture["onset_hop"].as_u64().unwrap() as usize;
        let transient = fixture["category"] == "transient";
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
        let mut current = SynchronousAnalysis::legacy_new(kernel.clone(), &params);
        let mut reference_frames = Vec::new();
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
            current.process(chunk, &params);
            reference_frames.push(if flora_hops.contains(&frame) || observe.contains(&frame) {
                Some((
                    evaluate(&truth_density, &space, &params, &mut scratch, peak),
                    current.landscape.clone(),
                    truth_density.clone(),
                ))
            } else {
                None
            });
        }
        let mut anchors: Vec<_> = frequencies
            .iter()
            .flat_map(|f| [0.5, 2.0 / 3.0, 1.0, 1.5].map(|r| f * r))
            .map(|f| f.clamp(space.fmin, space.fmax).log2())
            .collect();
        anchors.sort_unstable_by(f32::total_cmp);
        anchors.dedup();
        for (&variant, template) in variants.iter().zip(&templates) {
            let mut front = template.clone();
            let mut population47: Option<[Vec<Flora>; 3]> = None;
            let mut population8: Option<[Vec<Flora>; 3]> = None;
            let mut totals = DecisionTotals::default();
            let mut interval_pass = true;
            let mut interval_checks = 0;
            let mut worst_c = 0.0_f64;
            let mut worst_r = 0.0_f64;
            let mut max_sigma = 0.0_f64;
            for (frame, chunk) in pcm.chunks_exact(hop).enumerate() {
                front.process(chunk, &params);
                let Some((truth, old, truth_scan)) = &reference_frames[frame] else {
                    continue;
                };
                let actual = evaluate(&front.density_scan, &space, &params, &mut scratch, peak);
                if let Some(history) = &front.bank.spread {
                    max_sigma =
                        max_sigma.max(history.sigma_hz.iter().copied().fold(0.0_f64, f64::max));
                }
                if flora_hops.contains(&frame) {
                    let states = population47.get_or_insert_with(|| {
                        let s = Flora::aud_m1_start_window(truth);
                        [s.clone(), s.clone(), s]
                    });
                    for (p, l) in states.iter_mut().zip([truth, &actual, old]) {
                        for flora in p {
                            flora.advance(l);
                        }
                    }
                }
                let errors = field_errors(&actual, truth);
                let context_check = if addendum6
                    && observe.contains(&frame)
                    && front.erb_context.is_some()
                {
                    let total = front
                        .deposit_power
                        .iter()
                        .map(|&p| f64::from(p))
                        .sum::<f64>();
                    let mut error = 0.0_f64;
                    for (c, (&f, &computed)) in front
                        .bank
                        .frequency
                        .iter()
                        .zip(&front.context_power)
                        .enumerate()
                    {
                        let x = f64::from(f.clamp(1.0, params.fs / 2.0 - 1.0));
                        let b = 1.0186 * (24.7 + x / 9.265);
                        let exact = front
                            .bank
                            .frequency
                            .iter()
                            .zip(&front.deposit_power)
                            .map(|(&y, &p)| {
                                f64::from(p)
                                    * (1.0
                                        + ((f64::from(y.clamp(1.0, params.fs / 2.0 - 1.0)) - x)
                                            / b)
                                            .powi(2))
                                    .powi(-4)
                            })
                            .sum::<f64>();
                        // max(P_cb,d) is applied after the context sum in both paths.
                        let own = front.deposit_power[c];
                        error = error.max((f64::from(computed) - exact.max(f64::from(own))).abs());
                    }
                    json!({"error_over_total_power":if total>0.0{error/total}else{0.0}})
                } else {
                    Value::Null
                };
                writeln!(
                    diagnostics,
                    "{}",
                    json!({"fixture":name,"variant":variant,"frame":frame,"new_fields":errors,"context_direct_check":context_check,
                    "current_fields":field_errors(old,truth)})
                )
                .unwrap();
                if addendum6 && transient {
                    let near: Vec<_> = space
                        .centers_hz
                        .iter()
                        .map(|&f| {
                            frequencies
                                .iter()
                                .any(|&g| (erbn(f64::from(f)) - erbn(f64::from(g))).abs() <= 1.0)
                        })
                        .collect();
                    let mut error = [0.0_f64; 2];
                    let mut mass = [0.0_f64; 2];
                    let mut reference_mass = [0.0_f64; 2];
                    for i in 0..space.n_bins() {
                        let side = usize::from(!near[i]);
                        error[side] +=
                            f64::from((front.density_scan[i] - truth_scan[i]).abs() * du[i]);
                        mass[side] += f64::from(front.density_scan[i] * du[i]);
                        reference_mass[side] += f64::from(truth_scan[i] * du[i]);
                    }
                    writeln!(onset_rows,"{}",json!({"fixture":name,"variant":variant,"frame":frame,
                        "after_onset_ms":(frame-onset+1) as f64*hop as f64/f64::from(params.fs)*1000.0,
                        "density":front.density_scan,"truth_density":truth_scan,"density_mass_near_far":mass,
                        "truth_mass_near_far":reference_mass,"density_l1_near_far":error,"near_mask":near,
                        "new_fields":errors,"current_fields":field_errors(old,truth),"h01":actual.harmonicity01,
                        "r01":actual.roughness01,"c":actual.consonance_field_score,"truth_h01":truth.harmonicity01,
                        "truth_r01":truth.roughness01,"truth_c":truth.consonance_field_score,
                        "current_h01":old.harmonicity01,"current_r01":old.roughness01,"current_c":old.consonance_field_score})).unwrap();
                }
                if !observe.contains(&frame) {
                    continue;
                }
                let saved = &legacy[&(name.to_owned(), frame, variant)];
                assert_eq!(
                    field_errors(old, truth),
                    saved["old"]["fields"],
                    "unchanged current producer"
                );
                let new = if spread || addendum6 {
                    assess(&actual, truth, &anchors)
                } else {
                    assert_eq!(
                        errors, saved["new"]["fields"],
                        "unchanged point-deposit producer"
                    );
                    saved["new"].clone()
                };
                let states = population8.get_or_insert_with(|| {
                    let s = Flora::aud_m1_start_window(truth);
                    [s.clone(), s.clone(), s]
                });
                for (p, l) in states.iter_mut().zip([truth, &actual, old]) {
                    for flora in p {
                        flora.advance(l);
                    }
                }
                totals.frames += 1;
                totals.pitch_targets += anchors.len();
                totals.respawn_targets += Community::aud_m1_respawn(truth).len();
                totals.near_margin_1e3 += new["hill"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|h| h["truth_margin"].as_f64().unwrap() <= 1e-3)
                    .count();
                totals.production_ties_1e6 += new["hill"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|h| h["truth_margin"].as_f64().unwrap() <= 1e-6)
                    .count();
                for (i, data) in [&new, &saved["old"]].into_iter().enumerate() {
                    totals.pitch[i] += data["pitch_errors"].as_u64().unwrap() as usize;
                    totals.respawn[i] += data["respawn_errors"].as_u64().unwrap() as usize;
                    totals.flora[i] += data["flora_errors"].as_u64().unwrap() as usize;
                    for j in 0..3 {
                        totals.birth[i][j] += data["birth_l1"][j].as_f64().unwrap();
                    }
                }
                let mut intervals = Vec::new();
                for pair in fixture["pairs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|p| p["resolution"] == "unresolved")
                {
                    let i = pair["components"][0].as_u64().unwrap() as usize;
                    let j = pair["components"][1].as_u64().unwrap() as usize;
                    let low = frequencies[i].min(frequencies[j]);
                    let high = frequencies[i].max(frequencies[j]);
                    let c = [truth, &actual, old].map(|l| {
                        super::a4::span_extrema(&space, &l.consonance_field_score_eff, low, high).1
                    });
                    let r = [truth, &actual, old]
                        .map(|l| super::a4::span_extrema(&space, &l.roughness01, low, high).0);
                    let dc = [
                        (f64::from(c[1]) - f64::from(c[0])).max(0.0),
                        (f64::from(c[2]) - f64::from(c[0])).max(0.0),
                    ];
                    let dr = [
                        (f64::from(r[0]) - f64::from(r[1])).max(0.0),
                        (f64::from(r[0]) - f64::from(r[2])).max(0.0),
                    ];
                    interval_pass &= dc[0] <= dc[1] && dr[0] <= dr[1];
                    interval_checks += 1;
                    worst_c = worst_c.max(dc[0] - dc[1]);
                    worst_r = worst_r.max(dr[0] - dr[1]);
                    intervals.push(json!({"pair":pair,"low_hz":low,"high_hz":high,"c_max_ref_new_current":c,
                        "r_min_ref_new_current":r,"c_excess_new_current":dc,"r_deficit_new_current":dr,"pass":dc[0]<=dc[1] && dr[0]<=dr[1]}));
                }
                writeln!(rows,"{}",json!({"fixture":name,"category":fixture["category"],"resolution":fixture["pair_input_resolution"],
                    "variant":variant,"frame":frame,"new":new,"old":saved["old"],"intervals":intervals})).unwrap();
                if frame == observe[0]
                    && (name.starts_with("loo_110.0_40.0") || name == "pair15_1000.0_ratio1.0")
                {
                    std::fs::write(dir.join(format!("witness-{name}-{variant}.json")),serde_json::to_vec(&json!({"frame":frame,
                        "density":front.density_scan,"h01":actual.harmonicity01,"r01":actual.roughness01,"c":actual.consonance_field_score,
                        "truth_h01":truth.harmonicity01,"truth_r01":truth.roughness01,"truth_c":truth.consonance_field_score})).unwrap()).unwrap();
                }
            }
            let (flora47, signs) = sign_errors(&population47.unwrap());
            let (flora8, _) = sign_errors(&population8.unwrap());
            let near: Vec<_> = space
                .centers_hz
                .iter()
                .map(|&f| {
                    frequencies
                        .iter()
                        .any(|&g| (erbn(f64::from(f)) - erbn(f64::from(g))).abs() <= 1.0)
                })
                .collect();
            let partitions = [1, 2].map(|which| {
                let mut counts = [0_usize; 2];
                let truth = signs[0].as_array().unwrap();
                let actual = signs[which].as_array().unwrap();
                for (a, b) in actual.iter().zip(truth) {
                    for (i, (x, y)) in a
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(b.as_array().unwrap())
                        .enumerate()
                    {
                        counts[usize::from(!near[i])] += usize::from(x != y);
                    }
                }
                counts
            });
            let row = json!({"fixture":name,"category":fixture["category"],"resolution":fixture["pair_input_resolution"],
                "variant":variant,"beta":front.beta,"s1":if addendum6{selected_s1}else{variant%2==1},"spread":spread,
                "boxcar":addendum6,"erb_context":addendum6 && variant%2==1,"totals":totals,"flora47_new_current":flora47,
                "flora47_near_far_new_current":partitions,
                "flora8_new_current":flora8,"flora47_signs_ref_new_current":signs,"flora_targets":signs[0].as_array().unwrap().iter().map(|s|s.as_array().unwrap().len()).sum::<usize>(),
                "interval_pass":interval_pass,"interval_checks":interval_checks,"worst_c_excess_over_current":worst_c,
                "worst_r_deficit_over_current":worst_r,"max_sigma_hz":max_sigma});
            println!(
                "AUD_M1_A5_CASE {}",
                json!({"fixture":name,"variant":variant,"flora47_new_current":flora47,"interval_pass":interval_pass})
            );
            summary.push(row);
        }
    }
    rows.flush().unwrap();
    diagnostics.flush().unwrap();
    onset_rows.flush().unwrap();
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_vec(&summary).unwrap(),
    )
    .unwrap();
}
