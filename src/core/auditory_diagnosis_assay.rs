//! AUD-M1 addendum 5: low-pair stage taps and immutable direct-PCM controls.
use super::*;
use std::io::Write;

#[test]
#[ignore = "AUD-M1 addendum 5 stage diagnosis"]
fn aud_m1_a5_diagnosis() {
    let spread = std::env::var("AUD_M1_SPREAD").is_ok_and(|v| v == "1");
    let input = PathBuf::from(std::env::var("AUD_M1_INPUT").unwrap());
    let dir = output_dir();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(input.join("manifest.json")).unwrap()).unwrap();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let (_, du) = erb_grid(&space);
    let peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let mut writer =
        std::io::BufWriter::new(std::fs::File::create(dir.join("stage-rows.jsonl")).unwrap());
    let fixtures: Vec<_> = manifest["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .chain(manifest["direct_controls"].as_array().unwrap())
        .filter(|f| {
            f["name"].as_str().unwrap().starts_with("loo_110.0_40.0_")
                || f["name"] == "pair_220.0_1.0erb"
                || (spread && f["name"] == "15hz_1000.0_ratio1.0")
        })
        .collect();
    for fixture in fixtures {
        let pcm = read_pcm(&input.join(fixture["pcm"].as_str().unwrap()));
        let frequencies: Vec<_> = fixture["frequencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let amplitudes: Vec<_> = fixture["amplitudes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let truth_power: Vec<_> = amplitudes.iter().map(|a| a * a * 0.5).collect();
        let truth_mass: Vec<_> = frequencies
            .iter()
            .zip(&truth_power)
            .map(|(&f, &p)| {
                (a_weighting_gain_pow(f) * p / params.ref_power).powf(params.loudness_exp)
            })
            .collect();
        let mut truth_density = vec![0.0; space.n_bins()];
        for (&f, &m) in frequencies.iter().zip(&truth_mass) {
            deposit(&space, &du, &mut truth_density, f, m);
        }
        let truth = evaluate(&truth_density, &space, &params, &mut scratch, peak);
        for beta in [1.0 / 3.0, 0.25] {
            for analytic in [false, true] {
                let mut front =
                    ShadowFrontEnd::new(space.clone(), params.fs, beta, analytic, kernel.nfft());
                if spread {
                    front = front.with_gaussian_spread();
                }
                for (frame, chunk) in pcm.chunks_exact(hop).enumerate() {
                    front.process(chunk, &params);
                    if frame < 56 {
                        continue;
                    }
                    let mut channels = Vec::new();
                    let mut sums = vec![[0.0_f64; 4]; frequencies.len()];
                    for c in 0..front.bank.centers.len() {
                        let raw = front.bank.frequency[c];
                        let f = raw.clamp(1.0, params.fs / 2.0 - 1.0);
                        let unit = Complex32::from_polar(1.0, TAU * f / params.fs);
                        let d =
                            2.0 * front.bank.correlation[c].norm() / hop as f32 / front.coverage(f);
                        let cp = ((erbn(f64::from(f)) - front.erb_first) / front.step_erb)
                            .clamp(0.0, (front.bank.centers.len() - 1) as f64);
                        let lo = cp.floor() as usize;
                        let hi = (lo + 1).min(front.bank.centers.len() - 1);
                        let w = (cp - lo as f64) as f32;
                        let p = ((1.0 - w) * front.bank.energy[lo]
                            / front.bank.gain_power(lo, unit)
                            + w * front.bank.energy[hi] / front.bank.gain_power(hi, unit))
                        .max(d);
                        let position = if raw > 0.0 {
                            (raw.log2() - space.centers_log2[0]) / space.step()
                        } else {
                            f32::NEG_INFINITY
                        };
                        let bp = position.clamp(0.0, (du.len() - 1) as f32);
                        let left = bp.floor() as usize;
                        let right = (left + 1).min(du.len() - 1);
                        let fraction = bp - left as f32;
                        let width = (1.0 - fraction) * du[left] + fraction * du[right];
                        let gamma = if p > 0.0 {
                            (1.0 - 1e-6 * width / p).max(0.0)
                        } else {
                            0.0
                        };
                        let aw = a_weighting_gain_pow(f);
                        let edge = if position < 0.0 {
                            (1.0 + position).max(0.0)
                        } else if position > (du.len() - 1) as f32 {
                            (du.len() as f32 - position).max(0.0)
                        } else {
                            1.0
                        };
                        let owner = frequencies
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                (erbn(f64::from(f)) - erbn(f64::from(**a)))
                                    .abs()
                                    .total_cmp(&(erbn(f64::from(f)) - erbn(f64::from(**b))).abs())
                            })
                            .unwrap()
                            .0;
                        let cent = if raw > 0.0 {
                            Some(1200.0 * (raw / frequencies[owner]).log2())
                        } else {
                            None
                        };
                        let power = f64::from(d * edge);
                        let mass = f64::from(front.channel_mass[c] * edge);
                        sums[owner][0] += power;
                        sums[owner][1] += mass;
                        sums[owner][2] += power * f64::from(cent.unwrap_or(0.0));
                        sums[owner][3] += mass * f64::from(cent.unwrap_or(0.0));
                        channels.push(json!({"channel":c,"center_hz":front.bank.centers[c],"frequency_hz":raw,
                        "energy_real_ms":front.bank.energy[c],"coherence":front.bank.coherence[c],"power_before_gate":d,
                        "power_after_gate":front.channel_power[c],"pcb":p,"gamma":gamma,"aw":aw,"mass":front.channel_mass[c],
                        "edge":edge,"component":owner,"cent":cent,
                        "lag":front.bank.spread.as_ref().map(|s|s.lag[c]),
                        "lag_coherence":front.bank.spread.as_ref().map(|s|s.coherence[c]),
                        "sigma_hz":front.bank.spread.as_ref().map(|s|s.sigma_hz[c])}));
                    }
                    let components:Vec<_>=sums.iter().enumerate().map(|(i,s)|json!({"frequency_hz":frequencies[i],
                    "true_power":truth_power[i],"true_mass":truth_mass[i],"deposited_power":s[0],"mass":s[1],
                    "power_ratio":s[0]/f64::from(truth_power[i]),"mass_ratio":s[1]/f64::from(truth_mass[i]),
                    "power_weighted_cent":if s[0]>0.0{Some(s[2]/s[0])}else{None},
                    "mass_weighted_cent":if s[1]>0.0{Some(s[3]/s[1])}else{None}})).collect();
                    let actual = evaluate(&front.density_scan, &space, &params, &mut scratch, peak);
                    let mass: f32 = truth_density.iter().zip(&du).map(|(x, u)| x * u).sum();
                    let l1: f32 = front
                        .density_scan
                        .iter()
                        .zip(&truth_density)
                        .zip(&du)
                        .map(|((a, b), u)| (a - b).abs() * u)
                        .sum();
                    let row = json!({"fixture":fixture,"frame":frame,"beta":beta,"analytic":analytic,"spread":spread,"components":components,"channels":channels,
                    "relative_density_l1":l1/mass,"fields":field_errors(&actual,&truth),"density":front.density_scan,
                    "truth_density":truth_density,"h01":actual.harmonicity01,"r01":actual.roughness01,"c":actual.consonance_field_score,
                    "truth_h01":truth.harmonicity01,"truth_r01":truth.roughness01,"truth_c":truth.consonance_field_score});
                    writeln!(writer, "{row}").unwrap();
                    if frame == 63 {
                        println!(
                            "AUD_M1_A5_STAGE {}",
                            json!({"name":fixture["name"],"beta":beta,"s1":analytic,"components":components,"fields":row["fields"],"scan_l1":l1/mass})
                        );
                    }
                }
            }
        }
    }
    writer.flush().unwrap();
}
