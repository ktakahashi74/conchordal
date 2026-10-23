use super::*;
use crate::core::landscape::Landscape;
use crate::core::psycho_state::{compute_roughness_reference, r_pot_scan_to_r_state01_scan};
use crate::core::stream::potentials::PotentialScratch;
use crate::core::stream::synchronous::SynchronousAnalysis;
use serde_json::{Value, json};
use std::path::PathBuf;

fn output_dir() -> PathBuf {
    PathBuf::from(std::env::var("AUD_M1_OUTPUT").expect("explicit acquisition output directory"))
}

#[test]
fn filter_response_matches_closed_form() {
    let space = Log2Space::new(55.0, 8000.0, 96);
    for beta in [1.0, 0.5, 1.0 / 3.0, 0.25] {
        let bank = Bank::new(&space, 48000.0, 0.25_f64.min(0.75 * beta), beta);
        let mut worst = 0.0_f64;
        for c in 0..bank.centers.len() {
            let center = f64::from(bank.centers[c]);
            let width = beta * 1.0186 * (24.7 + center / 9.265);
            for frequency in [center, center - 0.5 * width, center + width] {
                let mut probe = bank.clone();
                probe.centers = vec![bank.centers[c]];
                probe.poles = vec![bank.poles[c]];
                probe.gains = vec![bank.gains[c]];
                probe.state = std::array::from_fn(|_| vec![Complex32::default()]);
                let radius = (-std::f64::consts::TAU * width / 48000.0).exp();
                let angle = std::f64::consts::TAU * (frequency - center) / 48000.0;
                let expected = ((1.0 - radius)
                    / ((1.0 - radius).powi(2) + 2.0 * radius * (1.0 - angle.cos())).sqrt())
                .powi(4);
                let count = (24.0 * 48000.0 / (std::f64::consts::TAU * width)).ceil() as usize;
                let mut mean = 0.0_f64;
                for n in 0..count + 512 {
                    let phase = std::f64::consts::TAU * frequency * n as f64 / 48000.0;
                    probe.one(Complex32::new(phase.cos() as f32, phase.sin() as f32));
                    if n >= count {
                        mean += f64::from(probe.state[3][0].norm()) / 512.0;
                    }
                }
                let error = (mean / expected - 1.0).abs();
                worst = worst.max(error);
                // Four resonant recurrences amplify f32 rounding by O(1/(1-r)).
                let bound = 32.0 * f64::from(f32::EPSILON) / (1.0 - radius);
                assert!(
                    error <= bound,
                    "channel {c}, f={frequency}, rel={error}, roundoff bound={bound}"
                );
            }
        }
        println!(
            "AUD_M1_CLOSED_FORM {}",
            json!({"beta":beta,"channels":bank.centers.len(),"max_relative_amplitude_error":worst})
        );
    }
}

#[test]
fn reset_clone_and_initial_silence() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let space = nsgt.space().clone();
    for beta in [1.0, 0.5, 1.0 / 3.0, 0.25] {
        for analytic in [false, true] {
            let mut front =
                ShadowFrontEnd::new(space.clone(), params.fs, beta, analytic, nsgt.nfft());
            let silence = vec![0.0; nsgt.hop()];
            assert!(
                front.process(&silence, &params),
                "initial silence remains observed"
            );
            assert!(front.density_scan.iter().all(|&v| v == 0.0));
            let pcm: Vec<_> = (0..nsgt.hop()).map(|i| (i % 17) as f32 / 512.0).collect();
            front.process(&pcm, &params);
            let mut birth = front.clone();
            let mut standalone = front.clone();
            for _ in 0..32 {
                let own: Vec<_> = pcm.iter().map(|x| x * 0.25).collect();
                let mix: Vec<_> = pcm.iter().zip(&own).map(|(x, y)| x + y).collect();
                let residual: Vec<_> = mix.iter().zip(own).map(|(x, y)| x - y).collect();
                birth.process(&residual, &params);
                standalone.process(&pcm, &params);
                assert_eq!(birth.density_scan, standalone.density_scan);
                assert_eq!(birth.power_scan, standalone.power_scan);
            }
            front.reset();
            let hops = front.warmup_samples.div_ceil(nsgt.hop());
            for hop in 1..=hops {
                assert_eq!(front.process(&silence, &params), hop == hops);
                assert!(front.density_scan.iter().all(|&v| v == 0.0));
            }
        }
    }
}

#[test]
#[ignore = "AUD-M1 explicit numerical acquisition"]
fn aud_m1_v1_and_python() {
    let dir = output_dir();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(dir.join("reference/reference.json")).unwrap())
            .unwrap();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let templates: Vec<_> = [1.0, 0.5, 1.0 / 3.0, 0.25]
        .into_iter()
        .flat_map(|beta| {
            [false, true].map(|analytic| {
                ShadowFrontEnd::new(
                    kernel.space().clone(),
                    params.fs,
                    beta,
                    analytic,
                    kernel.nfft(),
                )
            })
        })
        .collect();
    let mut comparisons = Vec::new();
    let mut calibration = Vec::new();
    let mut max_energy = 0.0_f64;
    let mut max_frequency = 0.0_f64;
    let mut max_mass = 0.0_f64;
    for fixture in reference["fixtures"].as_array().unwrap() {
        let pcm = read_pcm(&dir.join("reference").join(fixture["pcm"].as_str().unwrap()));
        let mut old = SynchronousAnalysis::new(kernel.clone(), &params);
        for chunk in pcm.chunks_exact(hop) {
            old.process(chunk, &params);
        }
        let old_mass = old
            .landscape
            .subjective_intensity
            .iter()
            .zip(old.du.iter())
            .map(|(x, u)| x * u)
            .sum::<f32>();
        for (i, template) in templates.iter().enumerate() {
            let mut front = template.clone();
            for chunk in pcm.chunks_exact(hop) {
                front.process(chunk, &params);
            }
            let expected = &fixture["variants"][i];
            if fixture["name"].as_str().unwrap().starts_with("single_") {
                let mass = front
                    .density_scan
                    .iter()
                    .zip(&front.du)
                    .map(|(x, u)| x * u)
                    .sum::<f32>();
                calibration.push(
                    json!({"fixture":fixture["name"],"variant":i,"beta":front.beta,
                    "new_mass":mass,"old_mass":old_mass,"ratio":mass/old_mass}),
                );
            }
            let mut energy_rel = 0.0_f64;
            let mut frequency_abs = 0.0_f64;
            let mut coherence_abs = 0.0_f64;
            let mut power_l1 = 0.0_f64;
            let mut mass_l1 = 0.0_f64;
            let mut power_total = 0.0_f64;
            let mut mass_total = 0.0_f64;
            for c in 0..front.bank.centers.len() {
                let e = expected["energy"][c].as_f64().unwrap();
                energy_rel = energy_rel.max((f64::from(front.bank.energy[c]) / e - 1.0).abs());
                frequency_abs = frequency_abs.max(
                    (f64::from(front.bank.frequency[c])
                        - expected["frequency"][c].as_f64().unwrap())
                    .abs(),
                );
                coherence_abs = coherence_abs.max(
                    (f64::from(front.bank.coherence[c])
                        - expected["coherence"][c].as_f64().unwrap())
                    .abs(),
                );
                let p = expected["power"][c].as_f64().unwrap();
                let m = expected["mass"][c].as_f64().unwrap()
                    * (f64::from(a_weighting_gain_pow(
                        front.bank.frequency[c].clamp(1.0, params.fs / 2.0 - 1.0),
                    )) / f64::from(params.ref_power))
                    .powf(f64::from(params.loudness_exp));
                power_l1 += (f64::from(front.channel_power[c]) - p).abs();
                mass_l1 += (f64::from(front.channel_mass[c]) - m).abs();
                power_total += p;
                mass_total += m;
            }
            max_energy = max_energy.max(energy_rel);
            max_frequency = max_frequency.max(frequency_abs);
            max_mass = max_mass.max(mass_l1 / mass_total);
            comparisons.push(json!({"fixture":fixture["name"],"step":expected["step"],"analytic":expected["analytic"],
                "max_energy_relative":energy_rel,"max_frequency_hz":frequency_abs,"max_coherence_absolute":coherence_abs,
                "power_relative_l1":power_l1/power_total,"mass_relative_l1":mass_l1/mass_total}));
        }
    }
    let pcm: Vec<_> = (0..16384)
        .map(|n| {
            let t = n as f64 / 48000.0;
            (0.1 * (std::f64::consts::TAU * 220.0 * t).cos()
                + 0.07 * (std::f64::consts::TAU * 1003.1 * t + 0.8).cos()) as f32
        })
        .collect();
    let t2 = crate::temporal_cognition::t2::aud_m1_gammatone_trace(&pcm);
    let mut bank = templates[0].bank.clone();
    let mut square = vec![0.0_f64; t2.len()];
    let mut error_square = square.clone();
    let mut peak = square.clone();
    let mut max_abs = square.clone();
    for (n, &x) in pcm.iter().enumerate() {
        bank.one(Complex32::new(x, 0.0));
        for (j, (center, trace)) in t2.iter().enumerate() {
            let c = bank
                .centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (f64::from(**a) - center)
                        .abs()
                        .total_cmp(&(f64::from(**b) - center).abs())
                })
                .unwrap()
                .0;
            assert!((f64::from(bank.centers[c]) - center).abs() < 0.001);
            let [re, im] = trace[n];
            let z = bank.state[3][c];
            let error = ((f64::from(z.re) - re).powi(2) + (f64::from(z.im) - im).powi(2)).sqrt();
            let amplitude = (re * re + im * im).sqrt();
            square[j] += amplitude * amplitude;
            error_square[j] += error * error;
            peak[j] = peak[j].max(amplitude);
            max_abs[j] = max_abs[j].max(error);
        }
    }
    let t2_metrics: Vec<_> = t2
        .iter()
        .enumerate()
        .map(|(j, (center, _))| {
            json!({"center":center,
        "relative_rms":(error_square[j]/square[j]).sqrt(),"max_error_over_peak":max_abs[j]/peak[j]})
        })
        .collect();
    let warm: Vec<_> = templates.iter().enumerate().map(|(i,t)|json!({"variant":i,"channels":t.bank.centers.len(),
        "first_hz":t.bank.centers[0],"last_hz":t.bank.centers.last(),"warmup_samples":t.warmup_samples,
        "warmup_ms":t.warmup_samples as f64/48.0,"beta":t.beta})).collect();
    let beta_metadata:Vec<_>=templates.iter().enumerate().filter(|(i,_)|i%2==0).map(|(_,t)|{
        let mut maximum_error=0.0_f32;
        for j in 0..4096 {
            let f=inv_erbn(t.cover_first+j as f64/4095.0*(erbn(23999.0)-t.cover_first)) as f32;
            let unit=Complex32::from_polar(1.0,TAU*f/48000.0);
            let exact:f32=(0..t.bank.centers.len()).map(|c|t.bank.gain_power(c,unit)).sum();
            maximum_error=maximum_error.max((t.coverage(f)/exact-1.0).abs());
        }
        let delays:Vec<_>=[110.0,220.0,440.0,1000.0,4000.0].into_iter().map(|f|{
            let b=t.beta*1.0186*(24.7+f/9.265);
            let r=(-std::f64::consts::TAU*b/48000.0).exp();
            let delay=4.0*r/(1.0-r)/48.0;
            let a=(-512.0_f64/48000.0/0.010).exp();
            json!({"hz":f,"center_group_delay_ms":delay,"with_hop_and_leaky_ms":delay+511.0/96.0+512.0/48.0*a/(1.0-a)})
        }).collect();
        let b=t.beta*1.0186*(24.7+f64::from(t.bank.centers[0])/9.265);
        let r=(-std::f64::consts::TAU*b/48000.0).exp();
        let mut state=[1.0_f64;4];
        for _ in 0..16384 { let mut input=0.0; for s in &mut state { *s=r* *s+(1.0-r)*input; input = *s; } }
        json!({"beta":t.beta,"step_erb":t.step_erb,"channels":t.bank.centers.len(),"first_hz":t.bank.centers[0],
            "last_hz":t.bank.centers.last(),"coverage_max_relative_error":maximum_error,"delays":delays,
            "settle_amplitude_after_16384":state[3],"settle_compressed_mass_ratio":state[3].powf(2.0*f64::from(params.loudness_exp)),
            "estimated_s2_s3_flops_per_hop":t.bank.centers.len()*512*52})
    }).collect();
    let result = json!({"comparisons":comparisons,"calibration":calibration,"beta_metadata":beta_metadata,"t2":t2_metrics,"warmup":warm,
        "max_energy_relative":max_energy,"max_frequency_hz":max_frequency,"max_mass_relative_l1":max_mass});
    std::fs::write(
        dir.join("v1-python.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!(
        "AUD_M1_V1 {}",
        json!({"max_energy_relative":max_energy,"max_frequency_hz":max_frequency,
        "max_mass_relative_l1":max_mass,"warmup":warm,"t2":t2_metrics})
    );
    // This protects the S3-S5 formula against implementation errors, not model acceptance.
    assert!(max_mass < 0.005, "Python S3-S5 formula mismatch");
}

fn read_pcm(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn deposit(space: &Log2Space, du: &[f32], out: &mut [f32], frequency: f32, mass: f32) {
    space.assert_scan_len_named(out, "truth_density_scan");
    if frequency < space.fmin || frequency > space.fmax {
        return;
    }
    let position = (frequency.log2() - space.centers_log2[0]) / space.step();
    let lo = (position.floor().max(0.0) as usize).min(out.len() - 1);
    let hi = (lo + 1).min(out.len() - 1);
    let fraction = (position - lo as f32).clamp(0.0, 1.0);
    out[lo] += (1.0 - fraction) * mass / du[lo];
    out[hi] += fraction * mass / du[hi];
}

fn evaluate(
    density: &[f32],
    space: &Log2Space,
    params: &LandscapeParams,
    scratch: &mut PotentialScratch,
    ref_peak: f32,
) -> Landscape {
    space.assert_scan_len_named(density, "subjective_intensity_scan");
    let mut result = Landscape::new(space.clone());
    result.subjective_intensity.copy_from_slice(density);
    scratch.compute_shape(
        density,
        space,
        (&params.harmonicity_kernel, &params.roughness_kernel),
        params.roughness_ref_eps.max(1e-12),
        &mut result.harmonicity,
        &mut result.roughness_shape_raw,
    );
    r_pot_scan_to_r_state01_scan(
        &result.roughness_shape_raw,
        ref_peak,
        params.roughness_k,
        &mut result.roughness01,
    );
    result.set_analysis_metadata(
        params.roughness_kernel.params,
        params.harmonicity_kernel.params,
        params.consonance_field_kernel(),
        params.roughness_k,
        ref_peak,
        params.roughness_ref_eps,
    );
    result.recompute_consonance(params);
    result
}

fn field_errors(actual: &Landscape, truth: &Landscape) -> Value {
    let pairs = [
        (&actual.harmonicity01, &truth.harmonicity01),
        (&actual.roughness01, &truth.roughness01),
        (
            &actual.consonance_field_score,
            &truth.consonance_field_score,
        ),
        (
            &actual.consonance_field_level,
            &truth.consonance_field_level,
        ),
    ];
    let metrics: Vec<_> = pairs
        .into_iter()
        .map(|(a, b)| {
            let l1 = a
                .iter()
                .zip(b)
                .map(|(x, y)| f64::from((x - y).abs()))
                .sum::<f64>();
            let maximum = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max);
            json!({"l1":l1,"mean_l1":l1/a.len() as f64,"max":maximum})
        })
        .collect();
    json!({"h":metrics[0],"r":metrics[1],"c":metrics[2],"level":metrics[3]})
}

#[test]
fn skipped_body_hops_preserve_filter_history() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    for beta in [1.0, 0.5, 1.0 / 3.0, 0.25] {
        for analytic in [false, true] {
            let mut every =
                ShadowFrontEnd::new(nsgt.space().clone(), params.fs, beta, analytic, nsgt.nfft());
            let mut sparse = every.clone();
            for frame in 0..32 {
                let pcm: Vec<_> = (0..nsgt.hop())
                    .map(|i| {
                        let t = (frame * nsgt.hop() + i) as f64 / 48000.0;
                        (0.1 * (std::f64::consts::TAU * (110.0 + frame as f64) * t).sin()) as f32
                    })
                    .collect();
                every.process(&pcm, &params);
                if frame % 4 == 3 {
                    let before = sparse.density_scan.clone();
                    sparse.process_body_gap(&pcm, 4, &params);
                    let mut raw = every.clone();
                    raw.density_scan.fill(0.0);
                    raw.reassign(&params, pcm.len(), 1000.0);
                    let a = (-(pcm.len() as f32) / params.fs / (params.tau_ms * 1e-3))
                        .exp()
                        .powi(4);
                    for ((&actual, &old), &q) in sparse
                        .density_scan
                        .iter()
                        .zip(&before)
                        .zip(&raw.density_scan)
                    {
                        let expected = a * old + (1.0 - a) * q;
                        let roundoff = 4.0
                            * sparse.bank.centers.len() as f32
                            * f32::EPSILON
                            * (old.abs() + q.abs()).max(f32::MIN_POSITIVE);
                        assert!(
                            (actual - expected).abs() <= roundoff,
                            "body S6 intermittent law"
                        );
                    }
                    assert_eq!(sparse.bank.state, every.bank.state);
                    assert_eq!(sparse.bank.previous, every.bank.previous);
                    assert_eq!(sparse.bank.energy, every.bank.energy);
                    assert_eq!(sparse.bank.correlation, every.bank.correlation);
                    assert_eq!(sparse.channel_power, every.channel_power);
                    assert_eq!(sparse.channel_mass, every.channel_mass);
                } else {
                    sparse.skip(&pcm);
                }
            }
        }
    }
}

#[test]
fn onset_and_roundoff_floor_use_existing_pcm_gate() {
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let ref_peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    for beta in [1.0, 0.5, 1.0 / 3.0, 0.25] {
        for analytic in [false, true] {
            let template =
                ShadowFrontEnd::new(space.clone(), params.fs, beta, analytic, kernel.nfft());
            let c = template
                .bank
                .centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (**a - 7896.57).abs().total_cmp(&(**b - 7896.57).abs()))
                .unwrap()
                .0;
            let pole = template.bank.poles[c];
            let mut outputs = Vec::new();
            for target in [1000.0_f32, 2000.0] {
                let theta = TAU * target / 48000.0;
                let b = (-theta).sin() / (4.0 * pole.im);
                let a = (-theta).cos() - 4.0 * pole.re * b;
                let mut front = template.clone();
                let mut pcm = vec![0.0; hop];
                pcm[hop - 3] = 1e-9 * b;
                pcm[hop - 2] = 1e-9 * a;
                pcm[hop - 1] = 1.0;
                let mut trace = Vec::new();
                for frame in 0..40 {
                    front.process(&pcm, &params);
                    let landscape =
                        evaluate(&front.density_scan, &space, &params, &mut scratch, ref_peak);
                    trace.push((front.density_scan.clone(), landscape));
                    pcm.fill(0.0);
                    if frame == 39 {
                        break;
                    }
                }
                outputs.push(trace);
            }
            let mut max_density_l1 = 0.0_f32;
            let mut max_hrc = 0.0_f64;
            for (frame, (a, b)) in outputs[0].iter().zip(&outputs[1]).enumerate() {
                let l1 =
                    a.0.iter()
                        .zip(&b.0)
                        .zip(&template.du)
                        .map(|((x, y), u)| (x - y).abs() * u)
                        .sum::<f32>();
                let mass =
                    a.0.iter()
                        .zip(&template.du)
                        .map(|(x, u)| x * u)
                        .sum::<f32>();
                let errors = field_errors(&a.1, &b.1);
                max_density_l1 = max_density_l1.max(l1 / mass.max(1e-12));
                for kind in ["h", "r", "c"] {
                    max_hrc = max_hrc.max(errors[kind]["max"].as_f64().unwrap());
                }
                assert!(
                    l1 / mass.max(1e-12) <= 1e-3,
                    "H1 density frame {frame}: {l1}/{mass}"
                );
                for kind in ["h", "r", "c"] {
                    assert!(
                        errors[kind]["max"].as_f64().unwrap() <= 1e-3,
                        "H1 {kind} frame {frame}: {errors}"
                    );
                }
            }
            println!(
                "AUD_M1_REAL_ONSET {}",
                json!({"beta":beta,"analytic":analytic,"relative_density_l1":max_density_l1,"max_hrc":max_hrc})
            );
            let mut residue = template;
            for _ in 0..64 {
                let pcm: Vec<_> = (0..hop)
                    .map(|i| 1e-8 * (TAU * 110.0 * i as f32 / 48000.0).sin())
                    .collect();
                residue.process(&pcm, &params);
                assert!(residue.density_scan.iter().all(|&v| v == 0.0));
            }
        }
    }
}

fn make_modal_fixture(root: f32, hop: usize) -> (Value, Vec<f32>) {
    use crate::synth::{
        modes::{ModeParams, compile_mode},
        resonator::ResonatorBank,
    };
    let modes = [root, root + 15.0].map(|f| ModeParams {
        freq_hz: f,
        t60_s: 4.0,
        gain: 0.1,
        in_gain: 1.0,
    });
    let mut body = ResonatorBank::new(48000.0, 2).unwrap();
    body.set_modes(&modes).unwrap();
    let mut frequencies = Vec::new();
    let mut amplitudes = Vec::new();
    let mut phases = Vec::new();
    for m in &modes {
        let c = compile_mode(m, 48000.0);
        let r = f64::from(c.r);
        let e = f64::from(c.e);
        // Eigen-angle and impulse amplitude of the actual damped MCF matrix.
        let theta = (1.0 - r * e * e / 2.0).acos();
        frequencies.push(theta * 48000.0 / std::f64::consts::TAU);
        amplitudes.push(f64::from(m.gain) * r * e / theta.sin());
        phases.push(theta - std::f64::consts::FRAC_PI_2);
    }
    let pcm = (0..64 * hop)
        .map(|n| body.process_sample(if n == 8 * hop { 1.0 } else { 0.0 }))
        .collect();
    (
        json!({"name":format!("modal_{root}_15hz"),"category":"close_mode_body","frequencies":frequencies,
        "amplitudes":amplitudes,"phases":phases,"phase_origin_sample":8*hop,"onset_hop":8,"onset_sample":8*hop,
        "length_samples":64*hop,"observation_hops":(56..64).collect::<Vec<_>>(),
        "nominal_mode_hz":[root,root+15.0],"t60_sec":4.0,"gain":0.1,"in_gain":1.0,
        "source":"one actual ResonatorBank, two modes; equal damping is a common amplitude factor"}),
        pcm,
    )
}

#[derive(Clone, Copy, Default, serde::Serialize)]
struct DecisionTotals {
    pitch: [usize; 2],
    respawn: [usize; 2],
    flora: [usize; 2],
    birth: [[f64; 3]; 2],
    frames: usize,
    pitch_targets: usize,
    respawn_targets: usize,
    flora_targets: usize,
    near_margin_1e3: usize,
    production_ties_1e6: usize,
}

#[test]
#[ignore = "AUD-M1 addendum 3 fixed consumer acquisition"]
fn aud_m1_v2_consumers() {
    use sha2::Digest;
    use std::{collections::BTreeMap, io::Write};
    let dir = output_dir();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(dir.join("reference/reference.json")).unwrap())
            .unwrap();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let (_, du) = erb_grid(&space);
    let ref_peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let templates: Vec<_> = [1.0, 0.5, 1.0 / 3.0, 0.25]
        .into_iter()
        .flat_map(|beta| {
            [false, true]
                .map(|s1| ShadowFrontEnd::new(space.clone(), params.fs, beta, s1, kernel.nfft()))
        })
        .collect();
    let mut fixtures: Vec<_> = reference["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f.clone(),
                read_pcm(&dir.join("reference").join(f["pcm"].as_str().unwrap())),
            )
        })
        .collect();
    for root in [110.0, 1000.0] {
        fixtures.push(make_modal_fixture(root, hop));
    }
    for (f, pcm) in &mut fixtures {
        let filename = format!("{}.f32", f["name"].as_str().unwrap());
        let bytes: Vec<_> = pcm.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.join("reference").join(&filename), &bytes).unwrap();
        f["pcm"] = json!(filename);
        f["pcm_sha256"] = json!(
            sha2::Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        *pcm = read_pcm(&dir.join("reference").join(&filename));
        f["length_samples"] = json!(pcm.len());
        f["observation_hops"] = if f["category"] == "transient" {
            let onset = f["onset_hop"].as_u64().unwrap() as usize;
            json!((onset..onset + 8).collect::<Vec<_>>())
        } else {
            json!((56..64).collect::<Vec<_>>())
        };
    }
    let manifest = json!({"fs":48000,"hop":hop,"fixtures":fixtures.iter().map(|(f,_)|f).collect::<Vec<_>>(),
        "variants":templates.iter().enumerate().map(|(i,t)|json!({"variant":i,"beta":t.beta,"s1":i%2==1,"channels":t.bank.centers.len()})).collect::<Vec<_>>(),
        "reference":"stationary component masses A_w*a^2/2, only transient category receives causal S6",
        "consumer_state":{"hill_global_count":8,"hill_global_separation_cent":20,"hill_ratio_count":20,
            "hill_neighbor_step_cent":12.5,"hill_score":"Landscape C score, no move/tessitura/crowding penalty",
            "hill_rng_seed_per_call":719,"origins_outside_space":"clamp to bounds, then deduplicate","hill_temperature":0,"hill_rank_rule":1e-6,"hill_move_improvement":0.1,
            "judgment_distance_cent":6.25,"near_ties":"recorded, never excluded","respawn_count":16,
            "spawn_ranges_hz":[[55,8000],[110,880],[440,3520]],"spawn_occupancy":"empty","tension":0,
            "flora_biomass_density":[0.02,0.144,1.0],"flora_resource":[0.5,1.0],"habituation":"identity"},
        "acceptance":"each category and consumer separately; count new<=old, spawn mean L1 new<=old in every range; no-target is not pass"});
    std::fs::write(
        dir.join("inputs-manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let mut counts: BTreeMap<String, [DecisionTotals; 8]> = BTreeMap::new();
    let mut rows =
        std::io::BufWriter::new(std::fs::File::create(dir.join("v2-rows.jsonl")).unwrap());
    let mut row_count = 0;
    for (fixture, pcm) in fixtures {
        let frequencies: Vec<_> = fixture["frequencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap() as f32)
            .collect();
        let amplitudes: Vec<_> = fixture["amplitudes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap() as f32)
            .collect();
        let transient = fixture["category"] == "transient";
        let onset = fixture["onset_hop"].as_u64().unwrap() as usize;
        let observe: Vec<_> = fixture["observation_hops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mut anchors: Vec<_> = frequencies
            .iter()
            .flat_map(|f| [0.5, 2.0 / 3.0, 1.0, 1.5].map(|r| f * r))
            .map(|f| f.clamp(space.fmin, space.fmax))
            .map(f32::log2)
            .collect();
        anchors.sort_unstable_by(f32::total_cmp);
        anchors.dedup();
        let mut fronts = templates.clone();
        let mut baseline = SynchronousAnalysis::new(kernel.clone(), &params);
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
            baseline.process(chunk, &params);
            for front in &mut fronts {
                front.process(chunk, &params);
            }
            if !observe.contains(&frame) {
                continue;
            }
            let truth = evaluate(&truth_density, &space, &params, &mut scratch, ref_peak);
            let truth_signs = crate::life::flora::Flora::aud_m1_cell_signs(&truth);
            let truth_hill: Vec<_> = anchors
                .iter()
                .map(|&a| crate::life::voice::pitch_core::aud_m1_hill_choice(&truth, a, &anchors))
                .collect();
            let ranges = [(space.fmin, space.fmax), (110.0, 880.0), (440.0, 3520.0)];
            let truth_spawn =
                ranges.map(|r| crate::life::community::Community::aud_m1_range_mass(&truth, r));
            let truth_peaks = crate::life::community::Community::aud_m1_respawn(&truth);
            let assess = |actual: &Landscape| {
                let mut pitch_errors = 0;
                let mut pitch = Vec::new();
                for (&anchor, &(correct, margin)) in anchors.iter().zip(&truth_hill) {
                    let (chosen, _) = crate::life::voice::pitch_core::aud_m1_hill_choice(
                        actual, anchor, &anchors,
                    );
                    let distance = 1200.0 * (chosen - correct).abs();
                    let moved = chosen.to_bits() != anchor.to_bits();
                    let truth_moved = correct.to_bits() != anchor.to_bits();
                    let mismatch = distance > 6.25 || moved != truth_moved;
                    pitch_errors += usize::from(mismatch);
                    pitch.push(json!({"anchor_hz":anchor.exp2(),"chosen_hz":chosen.exp2(),"truth_hz":correct.exp2(),
                        "distance_cent":distance,"moved":moved,"truth_moved":truth_moved,"truth_margin":margin,"mismatch":mismatch}));
                }
                let spawn: Vec<_> = ranges
                    .iter()
                    .zip(&truth_spawn)
                    .map(|(&r, b)| {
                        let a = crate::life::community::Community::aud_m1_range_mass(actual, r);
                        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>()
                    })
                    .collect();
                let signs = crate::life::flora::Flora::aud_m1_cell_signs(actual);
                let flora_errors = signs
                    .iter()
                    .zip(&truth_signs)
                    .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
                    .sum::<usize>();
                let peaks = crate::life::community::Community::aud_m1_respawn(actual);
                let respawn_errors = truth_peaks
                    .iter()
                    .filter(|&&i| {
                        !peaks.iter().any(|&j| {
                            1200.0 * (space.centers_log2[i] - space.centers_log2[j]).abs() <= 6.25
                        })
                    })
                    .count();
                json!({"pitch_errors":pitch_errors,"hill":pitch,"birth_l1":spawn,"flora_errors":flora_errors,
                    "respawn_errors":respawn_errors,"respawn_peaks":peaks,"fields":field_errors(actual,&truth)})
            };
            let old = assess(&baseline.landscape);
            for (variant, front) in fronts.iter().enumerate() {
                let actual = evaluate(&front.density_scan, &space, &params, &mut scratch, ref_peak);
                let new = assess(&actual);
                let aggregate = &mut counts
                    .entry(fixture["category"].as_str().unwrap().to_string())
                    .or_default()[variant];
                aggregate.frames += 1;
                aggregate.pitch_targets += anchors.len();
                aggregate.respawn_targets += truth_peaks.len();
                aggregate.flora_targets += truth_signs.iter().map(Vec::len).sum::<usize>();
                aggregate.near_margin_1e3 += truth_hill.iter().filter(|(_, m)| *m <= 1e-3).count();
                aggregate.production_ties_1e6 +=
                    truth_hill.iter().filter(|(_, m)| *m <= 1e-6).count();
                for (which, data) in [&new, &old].into_iter().enumerate() {
                    aggregate.pitch[which] += data["pitch_errors"].as_u64().unwrap() as usize;
                    aggregate.flora[which] += data["flora_errors"].as_u64().unwrap() as usize;
                    aggregate.respawn[which] += data["respawn_errors"].as_u64().unwrap() as usize;
                    for j in 0..3 {
                        aggregate.birth[which][j] += data["birth_l1"][j].as_f64().unwrap();
                    }
                }
                writeln!(rows,"{}",json!({"fixture":fixture["name"],"category":fixture["category"],"frame":frame,
                    "variant":variant,"beta":front.beta,"analytic":variant%2==1,"new":new,"old":old})).unwrap();
                row_count += 1;
                if frame == *observe.last().unwrap() {
                    std::fs::write(dir.join(format!("witness-{}-{variant}.json",fixture["name"].as_str().unwrap())),serde_json::to_vec(&json!({
                        "fixture":fixture,"frame":frame,"variant":variant,"beta":front.beta,"truth_density":truth_density,
                        "new_density":front.density_scan,"old_density":baseline.landscape.subjective_intensity,
                        "new_h01":actual.harmonicity01,"new_r01":actual.roughness01,"new_c":actual.consonance_field_score,
                        "truth_h01":truth.harmonicity01,"truth_r01":truth.roughness01,"truth_c":truth.consonance_field_score,
                        "old_h01":baseline.landscape.harmonicity01,"old_r01":baseline.landscape.roughness01,"old_c":baseline.landscape.consonance_field_score,
                        "new":new,"old":old})).unwrap()).unwrap();
                }
            }
        }
        println!(
            "AUD_M1_A3_CASE {}",
            json!({"fixture":fixture["name"],"category":fixture["category"],"observed":observe})
        );
    }
    rows.flush().unwrap();
    let eligible: Vec<_> = (0..8)
        .filter(|&v| {
            counts.values().all(|c| {
                let t = &c[v];
                t.frames > 0
                    && t.pitch_targets > 0
                    && t.pitch[0] <= t.pitch[1]
                    && t.respawn_targets > 0
                    && t.respawn[0] <= t.respawn[1]
                    && t.flora_targets > 0
                    && t.flora[0] <= t.flora[1]
                    && (0..3).all(|j| t.birth[0][j] <= t.birth[1][j])
            })
        })
        .collect();
    let selected = eligible.first().copied();
    let summary = json!({"counts_by_category":counts,"eligible_variants":eligible,"selected_variant":selected,"rows":row_count,
        "acceptance":"every category and consumer separately; birth means compared in each range; no near-tie exclusions",
        "rng_seed_per_call":719,"pitch_half_bin_cent":6.25,"unconnected_consumers":"level and H/R/C diagnostics only"});
    std::fs::write(
        dir.join("v2-summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("AUD_M1_A3_SUMMARY {summary}");
    assert!(
        selected.is_some(),
        "V2 stop: no beta satisfies every category and consumer"
    );
}

#[test]
fn complex_onset_uses_existing_pcm_gate() {
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let ref_peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    for beta in [1.0, 0.5, 1.0 / 3.0, 0.25] {
        let template = ShadowFrontEnd::new(space.clone(), params.fs, beta, false, kernel.nfft());
        let mut traces = Vec::new();
        for perturbed in [false, true] {
            let mut front = template.clone();
            let mut pcm = vec![Complex32::default(); hop];
            pcm[hop - 1] = Complex32::new(1.0, 0.0);
            if perturbed {
                pcm[hop - 3] = Complex32::new(1e-9, 0.0);
                pcm[hop - 2] = Complex32::new(0.0, 1e-9);
            }
            let mut trace = Vec::new();
            for _ in 0..40 {
                front.bank.read_complex_hop(pcm.iter().copied());
                front.reassign(&params, hop, hop as f32 / params.fs);
                let l = evaluate(&front.density_scan, &space, &params, &mut scratch, ref_peak);
                trace.push((front.density_scan.clone(), l));
                pcm.fill(Complex32::default());
            }
            traces.push(trace);
        }
        let mut maximum = 0.0_f64;
        let mut maximum_density = 0.0_f32;
        for (a, b) in traces[0].iter().zip(&traces[1]) {
            let mass =
                a.0.iter()
                    .zip(&template.du)
                    .map(|(x, u)| x * u)
                    .sum::<f32>();
            let l1 =
                a.0.iter()
                    .zip(&b.0)
                    .zip(&template.du)
                    .map(|((x, y), u)| (x - y).abs() * u)
                    .sum::<f32>()
                    / mass.max(1e-12);
            maximum_density = maximum_density.max(l1);
            let e = field_errors(&a.1, &b.1);
            for k in ["h", "r", "c"] {
                maximum = maximum.max(e[k]["max"].as_f64().unwrap());
            }
        }
        println!(
            "AUD_M1_COMPLEX_ONSET {}",
            json!({"beta":beta,"relative_density_l1":maximum_density,"max_hrc":maximum})
        );
        assert!(
            maximum_density <= 1e-3 && maximum <= 1e-3,
            "complex onset PCM gate"
        );
    }
}

#[test]
fn endpoint_fade_preserves_deposited_mass() {
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let mut front =
        ShadowFrontEnd::new(kernel.space().clone(), params.fs, 1.0, false, kernel.nfft());
    let last = (front.space.n_bins() - 1) as f32;
    for (position, expected, endpoint) in [
        (-2.0, 0.0, 0),
        (-0.5, 0.5, 0),
        (0.0, 1.0, 0),
        (last, 1.0, last as usize),
        (last + 0.5, 0.5, last as usize),
        (last + 2.0, 0.0, last as usize),
    ] {
        let f = (front.space.centers_log2[0] + position * front.space.step()).exp2();
        let unit = Complex32::from_polar(1.0, TAU * f / params.fs);
        front.bank.frequency.fill(f);
        for c in 0..front.bank.centers.len() {
            front.bank.energy[c] = 0.01 * front.bank.gain_power(c, unit);
            front.bank.correlation[c] = 0.5 * hop as f32 * front.bank.energy[c] * unit;
        }
        front.density_scan.fill(0.0);
        front.reassign(&params, hop, 1000.0);
        let original: f32 = front.channel_mass.iter().sum();
        let deposited: f32 = front
            .density_scan
            .iter()
            .zip(&front.du)
            .map(|(x, u)| x * u)
            .sum();
        let bound = 16.0 * front.bank.centers.len() as f32 * f32::EPSILON;
        assert!(
            (deposited / original - expected).abs() <= bound,
            "endpoint mass fraction at {position}"
        );
        if expected == 0.5 {
            assert!(
                front
                    .density_scan
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| i == endpoint || v == 0.0)
            );
        }
    }
    for f in [-110.0, 0.0] {
        front.bank.frequency.fill(f);
        front.density_scan.fill(0.0);
        front.reassign(&params, hop, 1000.0);
        assert!(front.density_scan.iter().all(|&x| x == 0.0));
    }
}
