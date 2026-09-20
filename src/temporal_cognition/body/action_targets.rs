//! Explicit offline acquisition. Counterfactual targets never enter the live observer.

use super::*;
use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{KernelAlign, NsgtKernelLog2, NsgtLog2Config, PowerMode};
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

fn pcm(path: &Path, samples: usize) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len(), samples * 4, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|bytes| {
            let value = f32::from_le_bytes(bytes.try_into().unwrap());
            assert!(value.is_finite());
            value
        })
        .collect()
}

#[test]
#[ignore = "explicit frozen-corpus acquisition; requires input and fresh output directories"]
fn acquire_action_feature_targets() {
    let input = std::env::var("CONCHORDAL_I10_PROFILE_DIR").unwrap();
    let output = std::env::var("CONCHORDAL_I10_FEATURE_DIR").unwrap();
    let input = Path::new(&input);
    let output = Path::new(&output);
    let manifest: Value =
        serde_json::from_slice(&fs::read(input.join("manifest.json")).unwrap()).unwrap();
    let medoids = manifest["schema"] == "i10-medoid-action-targets-v1";
    let onset_branches = manifest["schema"] == "i10-onset-branches-v1";
    if onset_branches {
        assert_eq!(manifest["cases"].as_array().unwrap().len(), 6);
    } else if medoids {
        assert_eq!(manifest["cases"].as_array().unwrap().len(), 8);
    } else {
        assert_eq!(manifest["schema"], "i10-actual-action-profiles-v2");
        assert_eq!(manifest["sample_rate"], 48000);
        assert_eq!(manifest["hop_samples"], 512);
        assert_eq!(manifest["issue_sample"], 14848);
        assert_eq!(manifest["horizon_samples"], 192000);
        assert_eq!(manifest["cases"].as_array().unwrap().len(), 36);
    }
    let config = TemporalBodyConfig {
        means: [0.; 6],
        deviations: [1.; 6],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
    };
    let analyzer = RtNsgtKernelLog2::new(NsgtKernelLog2::new(
        NsgtLog2Config {
            fs: 48000.,
            overlap: 0.75,
            nfft_override: Some(2048),
            kernel_align: KernelAlign::Right,
        },
        Log2Space::new(55., 8000., 96),
        None,
        PowerMode::Coherent,
    ));
    assert_eq!(analyzer.hop(), 512);
    fs::create_dir(output).expect("refuse an existing acquisition directory");
    let owner = Owner {
        id: if medoids || onset_branches {
            1
        } else {
            manifest["source_id"].as_u64().unwrap()
        },
        generation: 0,
        body_generation: 1,
        born: 0,
        changed: 0,
    };
    let mut branches = 0;
    let mut rows = 0;
    let mut prefix_comparisons = 0;
    let mut prefix_count = 0;
    let mut case_issues = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        if onset_branches && case["status"] != "acquired" {
            continue;
        }
        let name = if onset_branches {
            case["id"].as_str().unwrap()
        } else {
            case.as_str().unwrap()
        };
        let source = input.join(name);
        let destination = output.join(name);
        fs::create_dir(&destination).unwrap();
        let profiles: Value = serde_json::from_slice(
            &fs::read(source.join(if onset_branches {
                "branches.json"
            } else {
                "profiles.json"
            }))
            .unwrap(),
        )
        .unwrap();
        let issue = if onset_branches {
            profiles["render_start"].as_u64().unwrap()
        } else if medoids {
            profiles["issue_sample"].as_u64().unwrap()
        } else {
            14848
        };
        assert_eq!(issue % 512, 0);
        let decision = if onset_branches {
            profiles["decision_sample"].as_u64().unwrap()
        } else {
            issue
        };
        let future_samples = if onset_branches {
            profiles["render_end"].as_u64().unwrap() - issue
        } else {
            192000
        };
        assert_eq!(future_samples % 512, 0);
        if medoids {
            assert_eq!(profiles["sample_rate"], 48000);
            assert_eq!(profiles["hop_samples"], 512);
            assert_eq!(profiles["horizon_samples"], 192000);
            assert_eq!(profiles["model_version"], manifest["model_version"]);
        }
        let prefix: [Vec<f32>; 2] = std::array::from_fn(|bus| {
            pcm(
                &source.join(if medoids || onset_branches {
                    format!("prefix-bus{bus}.f32le")
                } else {
                    "prefix.f32le".into()
                }),
                issue as usize,
            )
        });
        case_issues.push(if onset_branches {
            json!({"case": name, "issue_sample": issue, "decision_sample": decision,
                "future_samples": future_samples, "first_fully_future_hop": decision.div_ceil(512) * 512})
        } else { json!({"case": name, "issue_sample": issue}) });
        prefix_count += issue / 512 * 2;
        let mut prefix_reference: [Option<Vec<Value>>; 2] = [None, None];
        for branch in profiles["branches"].as_array().unwrap() {
            if !medoids && !onset_branches && branch["eligible"] != true {
                continue;
            }
            let class = if onset_branches {
                &branch["input"]["class"]
            } else if medoids {
                &branch["action"]["class"]
            } else {
                &branch["class"]
            }
            .as_str()
            .unwrap();
            let offset = if onset_branches {
                branch["input"]["at"].as_u64().unwrap() - decision
            } else if medoids {
                branch["offset"].as_u64().unwrap()
            } else {
                branch["candidate_sample"].as_u64().unwrap() - issue
            };
            let stem = format!("{class}-{offset}");
            for bus in 0..2 {
                let routed = if onset_branches {
                    profiles["routing"][if bus == 0 {
                        "to_habitat"
                    } else {
                        "to_presentation"
                    }]
                    .as_bool()
                    .unwrap()
                } else if medoids {
                    profiles["routing"][bus].as_bool().unwrap()
                } else {
                    assert_eq!(
                        !branch["pcm"][bus].is_null(),
                        !profiles["prefix_pcm"][bus].is_null()
                    );
                    !branch["pcm"][bus].is_null()
                };
                let future = pcm(
                    &source.join(if onset_branches {
                        format!("{stem}-bus{bus}.f32le")
                    } else if medoids {
                        branch["pcm"][bus].as_str().unwrap().to_string()
                    } else {
                        format!("{stem}.f32le")
                    }),
                    future_samples as usize,
                );
                if (medoids || onset_branches) && !routed {
                    assert!(prefix[bus].iter().chain(&future).all(|v| *v == 0.));
                }
                let mut lane = Lane::new(analyzer.clone(), bus as u8, config);
                let mut prefix_rows = Vec::new();
                let first_prefix = prefix_reference[bus].is_none();
                let mut prefix_scan = first_prefix.then(|| {
                    BufWriter::new(
                        File::create(destination.join(format!("prefix-bus{bus}.f64le"))).unwrap(),
                    )
                });
                let mut future_rows = BufWriter::new(
                    File::create(destination.join(format!("{stem}-bus{bus}.jsonl"))).unwrap(),
                );
                let mut future_scan = BufWriter::new(
                    File::create(destination.join(format!("{stem}-bus{bus}.f64le"))).unwrap(),
                );
                let silence = [0.; 512];
                for (hop_index, chunk) in prefix[bus]
                    .chunks_exact(512)
                    .chain(future.chunks_exact(512))
                    .enumerate()
                {
                    let start = (hop_index * 512) as u64;
                    let descriptor = lane
                        .push(owner, start, if routed { chunk } else { &silence }, true)
                        .unwrap();
                    let sample = lane.history.back().unwrap();
                    assert_eq!(sample.raw.start, start);
                    assert_eq!(sample.shape, hop_index >= 3);
                    let mut row = json!({
                        "role": if start < issue { "issue_prefix" } else { "counterfactual_actual_audio_target" },
                        "raw": sample.raw,
                        "energy": sample.energy,
                        "shape_supported": sample.shape,
                        "body_descriptor": descriptor,
                    });
                    if onset_branches {
                        row["decision_position"] = json!(if start + 512 <= decision {
                            "before_decision"
                        } else if start < decision {
                            "straddles_decision"
                        } else {
                            "after_decision"
                        });
                    }
                    let scan_writer = if start < issue {
                        prefix_rows.push(row);
                        prefix_scan.as_mut()
                    } else {
                        serde_json::to_writer(&mut future_rows, &row).unwrap();
                        writeln!(future_rows).unwrap();
                        rows += 1;
                        Some(&mut future_scan)
                    };
                    if let Some(writer) = scan_writer {
                        for value in &lane.scan {
                            writer.write_all(&value.to_le_bytes()).unwrap();
                        }
                    }
                }
                future_rows.flush().unwrap();
                future_scan.flush().unwrap();
                if let Some(reference) = &prefix_reference[bus] {
                    assert_eq!(
                        &prefix_rows, reference,
                        "prefix changed: {name}/{stem}/{bus}"
                    );
                    prefix_comparisons += 1;
                } else {
                    if medoids && profiles["source_record"]["descriptor"]["bus"] == bus {
                        let actual = &prefix_rows.last().unwrap()["body_descriptor"];
                        let registered = &profiles["source_record"]["descriptor"];
                        for key in [
                            "raw_values",
                            "mask",
                            "start",
                            "end",
                            "source_start",
                            "available",
                        ] {
                            assert_eq!(actual[key], registered[key], "{name}/{key}");
                        }
                        // Value promotes f32 to f64; direct JSON serialization uses f32's short spelling.
                        for i in 0..6 {
                            assert_eq!(
                                (actual["coverage"][i].as_f64().unwrap() as f32).to_bits(),
                                (registered["coverage"][i].as_f64().unwrap() as f32).to_bits(),
                                "{name}/coverage/{i}"
                            );
                        }
                    }
                    let mut writer = BufWriter::new(
                        File::create(destination.join(format!("prefix-bus{bus}.jsonl"))).unwrap(),
                    );
                    for row in &prefix_rows {
                        serde_json::to_writer(&mut writer, row).unwrap();
                        writeln!(writer).unwrap();
                    }
                    writer.flush().unwrap();
                    prefix_scan.as_mut().unwrap().flush().unwrap();
                    prefix_reference[bus] = Some(prefix_rows);
                }
            }
            branches += 1;
        }
        println!("acquired {name}; total branches={branches}, future rows={rows}");
    }
    assert_eq!(
        branches,
        if onset_branches {
            71
        } else if medoids {
            1032
        } else {
            396
        }
    );
    assert_eq!(rows, branches * if onset_branches { 752 } else { 750 });
    assert_eq!(
        prefix_comparisons,
        if onset_branches {
            134
        } else if medoids {
            2048
        } else {
            720
        }
    );
    let registration = json!({
        "schema": if onset_branches { "i10-onset-branch-feature-targets-v1" } else if medoids { "i10-medoid-action-feature-targets-v1" } else { "i10-actual-action-feature-targets-v1" },
        "sample_rate": 48000, "nfft": 2048, "hop_samples": 512,
        "kernel_align": "right", "power_mode": "coherent",
        "rt_config": {"tau_min": 0.005, "tau_max": 0.020, "f_ref": 200.0},
        "space": {"fmin": 55.0, "fmax": 8000.0, "bins_per_octave": 96},
        "centers_log2": analyzer.space().centers_log2.iter().map(|x| f64::from(*x)).collect::<Vec<_>>(),
        "body_config": config, "source_manifest": manifest,
        "future_rows": rows, "prefix_rows": prefix_count, "branches": branches,
        "prefix_comparisons": prefix_comparisons,
        "scan_format": "f64 little-endian, one Log2Space-aligned scan per JSONL row; consult shape_supported",
        "claim": "counterfactual actual-audio targets only; no live observations, candidate accuracy, template admission or head calibration",
    });
    let mut registration = registration;
    if medoids || onset_branches {
        registration["case_issues"] = json!(case_issues);
    }
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&registration).unwrap(),
    )
    .unwrap();
}
