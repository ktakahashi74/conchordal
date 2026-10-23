//! Addendum-6 delay, power calibration, and saved-onset diagnosis.
use super::*;
use crate::life::flora::Flora;
use std::collections::BTreeMap;

#[test]
#[ignore = "AUD-M1 addendum 6 delay and isolated-power acquisition"]
fn aud_m1_a6_metadata() {
    let input = PathBuf::from(std::env::var("AUD_M1_INPUT").unwrap());
    let dir = output_dir();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(input.join("manifest.json")).unwrap()).unwrap();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let mut metadata = Vec::new();
    let mut calibration = Vec::new();
    for beta in [1.0 / 3.0, 0.25] {
        let template = ShadowFrontEnd::new(
            kernel.space().clone(),
            params.fs,
            beta,
            false,
            kernel.nfft(),
        )
        .with_statistics(hop, false);
        let counts = &template.bank.statistics.as_ref().unwrap().counts;
        let a = (-(hop as f64) / f64::from(params.fs) / (f64::from(params.tau_ms) * 1e-3)).exp();
        let leaky_ms = hop as f64 / f64::from(params.fs) * a / (1.0 - a) * 1000.0;
        let rows: Vec<_> = template
            .bank
            .centers
            .iter()
            .zip(counts)
            .map(|(&f, &n)| {
                let b = beta * 1.0186 * (24.7 + f64::from(f) / 9.265);
                let radius = (-std::f64::consts::TAU * b / f64::from(params.fs)).exp();
                let group_ms = 4.0 * radius / (1.0 - radius) / f64::from(params.fs) * 1000.0;
                let boxcar_ms = n as f64 * hop as f64 / f64::from(params.fs) * 500.0;
                json!({"center_hz":f,"hops":n,"tc_ms":1000.0/(0.75*(24.7+f64::from(f)/9.265)),
                "group_ms":group_ms,"boxcar_center_ms":boxcar_ms,"leaky_center_ms":leaky_ms,
                "total_ms":group_ms+boxcar_ms+leaky_ms})
            })
            .collect();
        let selected:Vec<_>=[55.0,110.0,150.0,220.0,440.0,1000.0,4000.0,8000.0].into_iter().map(|f|
            json!({"requested_hz":f,"channel":rows.iter().min_by(|a,b|
                (a["center_hz"].as_f64().unwrap()-f).abs().total_cmp(&(b["center_hz"].as_f64().unwrap()-f).abs())).unwrap()})).collect();
        metadata.push(json!({"beta":beta,"fs":params.fs,"hop":hop,"tau_ms":params.tau_ms,
            "channels":rows,"selected_channels":selected,"erb_context_bucket_step":1.0/32.0,
            "context_moment_degree":6,"context_omission_bound_at_1khz":statistics::ErbContext::omitted_weight_bound(1000.0)}));
        for fixture in manifest["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["category"] == "single")
        {
            let pcm = read_pcm(&input.join(fixture["pcm"].as_str().unwrap()));
            let f = fixture["frequencies"][0].as_f64().unwrap() as f32;
            let amplitude = fixture["amplitudes"][0].as_f64().unwrap() as f32;
            let power = amplitude * amplitude * 0.5;
            let mut front = template.clone();
            for chunk in pcm.chunks_exact(hop) {
                front.process(chunk, &params);
            }
            let mut context = statistics::ErbContext::new(params.fs);
            context.prepare(&front.bank.frequency, &front.deposit_power);
            let unit = Complex32::from_polar(1.0, TAU * f / params.fs);
            let position = ((erbn(f64::from(f)) - front.erb_first) / front.step_erb)
                .clamp(0.0, (front.bank.centers.len() - 1) as f64);
            let lo = position.floor() as usize;
            let hi = (lo + 1).min(front.bank.centers.len() - 1);
            let fraction = (position - lo as f64) as f32;
            let narrow = (1.0 - fraction) * front.bank.energy[lo] / front.bank.gain_power(lo, unit)
                + fraction * front.bank.energy[hi] / front.bank.gain_power(hi, unit);
            calibration.push(json!({"fixture":fixture["name"],"pcm_sha256":fixture["pcm_sha256"],"beta":beta,"frequency_hz":f,
                "true_real_mean_square":power,"narrow_power":narrow,"erb_power":context.power_at(f),
                "narrow_ratio":narrow/power,"erb_ratio":context.power_at(f)/power,
                "deposit_total_ratio":front.deposit_power.iter().sum::<f32>()/power}));
        }
    }
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_vec_pretty(&json!({"delays":metadata,"single_power":calibration})).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "AUD-M1 addendum 6 saved onset attribution"]
fn aud_m1_a6_onset_diagnosis() {
    let input = PathBuf::from(std::env::var("AUD_M1_ONSET_INPUT").unwrap());
    let dir = output_dir();
    let summary: Value =
        serde_json::from_slice(&std::fs::read(input.join("summary.json")).unwrap()).unwrap();
    let (params, kernel, _) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let (_, du) = erb_grid(&space);
    let peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let mut series: BTreeMap<(String, usize), Vec<Value>> = BTreeMap::new();
    for line in std::fs::read_to_string(input.join("onset-rows.jsonl"))
        .unwrap()
        .lines()
    {
        let row: Value = serde_json::from_str(line).unwrap();
        series
            .entry((
                row["fixture"].as_str().unwrap().to_owned(),
                row["variant"].as_u64().unwrap() as usize,
            ))
            .or_default()
            .push(row);
    }
    let mut output = Vec::new();
    for ((name, variant), rows) in series {
        assert_eq!(rows.len(), 47);
        let mut populations: Option<[Vec<Flora>; 3]> = None;
        let mut mass_trajectory = Vec::new();
        for row in &rows {
            let density: Vec<_> = row["density"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let truth: Vec<_> = row["truth_density"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let near: Vec<_> = row["near_mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect();
            let masked: Vec<_> = density
                .iter()
                .zip(&near)
                .map(|(&d, &keep)| if keep { d } else { 0.0 })
                .collect();
            let mut mass = [0.0_f64; 3];
            for i in 0..space.n_bins() {
                let part = if !near[i] {
                    2
                } else {
                    usize::from(truth[i] == 0.0)
                };
                mass[part] += f64::from(density[i] * du[i]);
            }
            mass_trajectory.push(
                json!({"frame":row["frame"],"after_onset_ms":row["after_onset_ms"],
                "near_oracle_bins_near_other_bins_far_mass":mass,
                "oracle_mass":truth.iter().zip(&du).map(|(&d,&w)|f64::from(d*w)).sum::<f64>()}),
            );
            let truth = evaluate(&truth, &space, &params, &mut scratch, peak);
            let actual = evaluate(&density, &space, &params, &mut scratch, peak);
            let masked = evaluate(&masked, &space, &params, &mut scratch, peak);
            let states = populations.get_or_insert_with(|| {
                let s = Flora::aud_m1_start_window(&truth);
                [s.clone(), s.clone(), s]
            });
            for (p, l) in states.iter_mut().zip([&truth, &actual, &masked]) {
                for flora in p {
                    flora.advance(l);
                }
            }
        }
        let (errors, signs) = super::a5_validation::sign_errors(&populations.unwrap());
        let original = summary
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["fixture"] == name && s["variant"] == variant)
            .unwrap();
        assert_eq!(
            signs[0], original["flora47_signs_ref_new_current"][0],
            "same saved reference producer"
        );
        assert_eq!(
            signs[1], original["flora47_signs_ref_new_current"][1],
            "same saved candidate producer"
        );
        let near = rows[0]["near_mask"].as_array().unwrap();
        let mut partitions = [[0_usize; 2]; 2];
        let mut changed = 0;
        let mut cells = Vec::new();
        for state in 0..6 {
            for (i, position) in near.iter().enumerate() {
                let side = usize::from(!position.as_bool().unwrap());
                let before = &signs[0][state][i];
                let main = &signs[1][state][i];
                let masked = &signs[2][state][i];
                partitions[0][side] += usize::from(main != before);
                partitions[1][side] += usize::from(masked != before);
                changed += usize::from(main != masked);
                if main != before || masked != before {
                    cells.push(json!({"state":state,"bin":i,"hz":space.centers_hz[i],
                "near":position,"truth_sign":before,"main_sign":main,"without_far_sign":masked}));
                }
            }
        }
        output.push(json!({"fixture":name,"variant":variant,"main_without_far_mismatches":errors,
            "main_without_far_near_far":partitions,"current_near_far":original["flora47_near_far_new_current"][1],
            "changed_signs_by_removing_far_mass":changed,"cells":cells,"mass_trajectory":mass_trajectory}));
    }
    std::fs::write(
        dir.join("onset-attribution.json"),
        serde_json::to_vec_pretty(&output).unwrap(),
    )
    .unwrap();
}
