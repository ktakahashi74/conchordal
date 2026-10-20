use super::*;

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}

fn number(value: &Value) -> String {
    value.as_f64().map_or_else(
        || value.to_string(),
        |x| {
            if x != 0.0 && x.abs() < 1e-6 {
                format!("{x:.6e}")
            } else {
                format!("{x:.9}")
            }
        },
    )
}

fn difference(old: &Value, new: &Value) -> Value {
    json!(old.as_f64().zip(new.as_f64()).map(|(a, b)| b - a))
}

fn metrics(blocks: &[Value], version: &str) -> Value {
    let mut surprise = Moments::default();
    let mut precision = Moments::default();
    let mut counts = [0_u64; 5];
    for block in blocks {
        let v = &block[version];
        for (moments, mean, count) in [
            (
                &mut surprise,
                &v["soundgen"]["mean"],
                &v["soundgen"]["defined_hops"],
            ),
            (
                &mut precision,
                &v["precision_mean"],
                &v["precision_defined_hops"],
            ),
        ] {
            if let Some(mean) = mean.as_f64() {
                let count = count.as_u64().unwrap();
                moments.sum += mean * count as f64;
                moments.count += count;
            }
        }
        for (sum, value) in counts.iter_mut().zip([
            &block["hops"],
            &v["soundgen"]["missing_hops"],
            &v["precision_missing_hops"],
            &v["missing_surprise_bin_hops"],
            &v["missing_precision_bin_hops"],
        ]) {
            *sum += value.as_u64().unwrap();
        }
    }
    json!({"mean_surprise":surprise.mean(),"mean_precision":precision.mean(),
        "hops":counts[0],"surprise_defined_hops":surprise.count,"precision_defined_hops":precision.count,
        "surprise_missing_hops":counts[1],"precision_missing_hops":counts[2],
        "missing_surprise_bin_hops":counts[3],"missing_precision_bin_hops":counts[4]})
}

pub(super) fn write_report() -> Result<()> {
    let old_dir = root().join("results-a1");
    let new_dir = results()?;
    let expected_provenance = provenance()?;
    let mut rows = Vec::new();
    for group in ["controls", "renders"] {
        let old = read_json(&old_dir.join(format!("{group}.json")))?;
        let new = read_json(&new_dir.join(format!("{group}.json")))?;
        ensure!(
            new["provenance"]["source_sha256"] == expected_provenance["source_sha256"],
            "stale {group} measurement"
        );
        ensure!(
            old["cases"].as_array().unwrap().len() == new["cases"].as_array().unwrap().len(),
            "case set changed"
        );
        for case in new["cases"].as_array().unwrap() {
            let name = case["stimulus"].as_str().unwrap();
            ensure!(
                old["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|c| c["stimulus"] == name),
                "A-1 case absent"
            );
            let mut versions = BTreeMap::new();
            let mut blocks = Vec::new();
            for directory in [&old_dir, &new_dir] {
                let selected = if group == "controls" {
                    vec![read_json(&directory.join(format!("controls/{name}.json")))?["window_summary"].clone()]
                } else {
                    let blocks: Vec<Value> = serde_json::from_value(read_json(
                        &directory.join(format!("renders/{name}/habitat.json")),
                    )?)?;
                    blocks
                        .into_iter()
                        .filter(|b| {
                            let start = b["start_sample"].as_u64().unwrap() as f64 / f64::from(FS);
                            (2.0..20.0).contains(&start)
                        })
                        .collect()
                };
                let wav = if group == "controls" {
                    format!("controls/{name}.wav")
                } else {
                    format!("renders/{name}/audio.wav")
                };
                ensure!(
                    fs::read(old_dir.join(&wav))? == fs::read(new_dir.join(&wav))?,
                    "A-1/A-2 audio differs: {name}"
                );
                blocks.push(selected);
            }
            for version in ["original", "extended"] {
                let a1 = metrics(&blocks[0], version);
                let a2 = metrics(&blocks[1], version);
                let delta: serde_json::Map<_, _> = a1
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(key, value)| (key.clone(), difference(value, &a2[key])))
                    .collect();
                versions.insert(version, json!({"a1":a1,"a2":a2,"a2_minus_a1":delta}));
            }
            rows.push(json!({"group":group,"stimulus":name,"versions":versions}));
        }
    }
    let old_controls = read_json(&old_dir.join("controls.json"))?;
    let new_controls = read_json(&new_dir.join("controls.json"))?;
    let old_external = read_json(&old_dir.join("external.json"))?;
    let new_external = read_json(&new_dir.join("external.json"))?;
    ensure!(
        new_external["provenance"]["source_sha256"] == expected_provenance["source_sha256"],
        "stale external measurement"
    );
    let paired = external::revision_statistics(&old_external, &new_external)?;
    let mut hypotheses = Vec::new();
    for version in ["original", "extended"] {
        let find = |controls: &Value| -> Value {
            controls["hypotheses_recorded_not_asserted"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["version"] == version)
                .unwrap()
                .clone()
        };
        hypotheses
            .push(json!({"version":version,"a1":find(&old_controls),"a2":find(&new_controls)}));
    }
    let mut markdown = String::from(
        "# A-1 と A-2 の測定値\n\n基準登録第5稿、A-2 改訂第3稿。habitat 相当、評価区間 [2,20) 秒。驚きは soundgen 3.0.0 の統合規則、精度は振幅重みつき平均。差は A-2 − A-1。欠測は null。\n\n| 刺激 | 版 | A-1 驚き / 精度 | A-2 驚き / 精度 | 差 驚き / 精度 |\n|---|---|---:|---:|---:|\n",
    );
    for row in &rows {
        for version in ["original", "extended"] {
            let v = &row["versions"][version];
            markdown.push_str(&format!(
                "| {} | {version} | {} / {} | {} / {} | {} / {} |\n",
                row["stimulus"].as_str().unwrap(),
                number(&v["a1"]["mean_surprise"]),
                number(&v["a1"]["mean_precision"]),
                number(&v["a2"]["mean_surprise"]),
                number(&v["a2"]["mean_precision"]),
                number(&v["a2_minus_a1"]["mean_surprise"]),
                number(&v["a2_minus_a1"]["mean_precision"])
            ));
        }
    }
    markdown.push_str("\n欠測数。各欄は A-1 → A-2。bin-hop は (bin, hop) の組の数。\n\n| 刺激 | 版 | 全 hop | 驚き欠測 hop | 精度欠測 hop | 驚き欠測 bin-hop | 精度欠測 bin-hop |\n|---|---|---:|---:|---:|---:|---:|\n");
    for row in &rows {
        for version in ["original", "extended"] {
            let v = &row["versions"][version];
            markdown.push_str(&format!(
                "| {} | {version}",
                row["stimulus"].as_str().unwrap()
            ));
            for key in [
                "hops",
                "surprise_missing_hops",
                "precision_missing_hops",
                "missing_surprise_bin_hops",
                "missing_precision_bin_hops",
            ] {
                markdown.push_str(&format!(" | {} → {}", v["a1"][key], v["a2"][key]));
            }
            markdown.push_str(" |\n");
        }
    }
    markdown.push_str("\n## 登録仮説 1・3\n\n比較不能は null。\n\n| 仮説 | 版 | 比較 | A-1 | A-2 |\n|---|---|---|---|---|\n");
    for h in &hypotheses {
        for (hypothesis, keys) in [
            (
                "hypothesis_1",
                &[
                    "surprise_comparable",
                    "surprise_less_than_white",
                    "precision_greater_than_white",
                ][..],
            ),
            (
                "hypothesis_3",
                &[
                    "surprise_between",
                    "precision_between",
                    "direction_registered",
                ][..],
            ),
        ] {
            for key in keys {
                markdown.push_str(&format!(
                    "| {hypothesis} | {} | {key} | {} | {} |\n",
                    h["version"].as_str().unwrap(),
                    h["a1"][hypothesis][key],
                    h["a2"][hypothesis][key]
                ));
            }
        }
    }
    markdown.push_str("\n## 登録仮説 4\n\n周期刺激の 0.5 秒区間ごとの驚き。\n\n| 版 | 開始秒 | A-1 平均 / 最大 | A-2 平均 / 最大 | 差 平均 / 最大 |\n|---|---:|---:|---:|---:|\n");
    for h in &hypotheses {
        let a1 = h["a1"]["hypothesis_4"]["periodic_half_second_intervals"]
            .as_array()
            .unwrap();
        let a2 = h["a2"]["hypothesis_4"]["periodic_half_second_intervals"]
            .as_array()
            .unwrap();
        for (old, new) in a1.iter().zip(a2) {
            ensure!(
                old["start_sec"] == new["start_sec"],
                "hypothesis 4 interval changed"
            );
            markdown.push_str(&format!(
                "| {} | {} | {} / {} | {} / {} | {} / {} |\n",
                h["version"].as_str().unwrap(),
                old["start_sec"],
                number(&old["mean"]),
                number(&old["maximum"]),
                number(&new["mean"]),
                number(&new["maximum"]),
                number(&difference(&old["mean"], &new["mean"])),
                number(&difference(&old["maximum"], &new["maximum"]))
            ));
        }
    }
    markdown.push_str("\n## 登録仮説 5\n\n各区間は開始から 0.5 秒。\n\n| 版 | 開始秒 | A-1 周期 / 逸脱 最大 | A-2 周期 / 逸脱 最大 | 差 周期 / 逸脱 最大 | 逸脱 > 周期 A-1 / A-2 |\n|---|---:|---:|---:|---:|---|\n");
    for h in &hypotheses {
        for (old, new) in h["a1"]["hypothesis_5"]
            .as_array()
            .unwrap()
            .iter()
            .zip(h["a2"]["hypothesis_5"].as_array().unwrap())
        {
            ensure!(
                old["start_sec"] == new["start_sec"],
                "hypothesis 5 interval changed"
            );
            markdown.push_str(&format!(
                "| {} | {} | {} / {} | {} / {} | {} / {} | {} / {} |\n",
                h["version"].as_str().unwrap(),
                old["start_sec"],
                number(&old["periodic_max"]),
                number(&old["deviant_max"]),
                number(&new["periodic_max"]),
                number(&new["deviant_max"]),
                number(&difference(&old["periodic_max"], &new["periodic_max"])),
                number(&difference(&old["deviant_max"], &new["deviant_max"])),
                old["deviant_greater"],
                new["deviant_greater"]
            ));
        }
    }
    markdown.push_str(&format!("\n## 外部照合\n\n共通刺激 {}、原型 {}。共通 seed {}、クラスタ・ブートストラップ {} 回。再標本インデックス SHA-256: `{}`。\n\n",
        paired["common_count"],paired["prototype_count"],BOOTSTRAP_SEED,RESAMPLES,paired["resample_indices_sha256_le_u64"].as_str().unwrap()));
    for a in 0..2 {
        markdown.push_str(&format!(
            "{}\n\n| 版 | A-1 r [95% CI] | A-2 r [95% CI] | 差 r |\n|---|---:|---:|---:|\n",
            new_external["aggregation_order"][a].as_str().unwrap()
        ));
        for v in 0..6 {
            let old = &old_external["statistics"]["aggregations"][a]["versions"][v];
            let new = &new_external["statistics"]["aggregations"][a]["versions"][v];
            markdown.push_str(&format!(
                "| {} | {} [{}, {}] | {} [{}, {}] | {} |\n",
                VERSION_NAMES[v],
                number(&old["r"]),
                number(&old["ci95"][0]),
                number(&old["ci95"][1]),
                number(&new["r"]),
                number(&new["ci95"][0]),
                number(&new["ci95"][1]),
                number(&difference(&old["r"], &new["r"]))
            ));
        }
        markdown.push_str(
            "\n| 改訂差 | 差 r | 95% CI（共通再標本） | 未定義再標本数 |\n|---|---:|---|---:|\n",
        );
        for c in paired["aggregations"][a]["contrasts"].as_array().unwrap() {
            markdown.push_str(&format!(
                "| {} | {} | [{}, {}] | {} |\n",
                c["contrast"].as_str().unwrap(),
                number(&c["difference"]),
                number(&c["ci95"][0]),
                number(&c["ci95"][1]),
                c["undefined_bootstrap_resamples"]
            ));
        }
        markdown.push('\n');
    }
    markdown.push_str("N-O の改訂差を主対比、R-O と N-P-O の改訂差を補助対比とする。S は A-1 の値を保持。各版の除外理由は external.json。fit の推定の不確かさは CI に含まない。R は登録どおりの STFT/HTK-mel 近似。MP3 の先頭の無音を保持。\n\n登録からの追加逸脱なし。Gravity の末尾 6 秒待機など、A-1 の実行条件を継承。WAV は全統制刺激・Flora 構成・既存サンプルで A-1 と一致。\n");
    let payload = json!({"provenance":expected_provenance,"evaluation_window_sec":[2,20],
        "difference":"A-2 minus A-1","cases":rows,"hypotheses":hypotheses,
        "external":{"a1":old_external["statistics"],"a2":new_external["statistics"],"paired_revision_statistics":paired,
            "primary_contrast":"N-O A-2 minus N-O A-1","auxiliary_contrasts":["R-O A-2 minus R-O A-1","N-P-O A-2 minus N-P-O A-1"]},
        "additional_registration_deviations":[],"all_control_and_render_wavs_identical_to_a1":true});
    write_json(&new_dir.join("a1_vs_a2.json"), &payload)?;
    fs::write(new_dir.join("a1_vs_a2.md"), markdown)?;
    Ok(())
}
