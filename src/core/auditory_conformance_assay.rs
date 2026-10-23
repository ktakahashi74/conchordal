//! The two registered onset perturbations with the selected front end.
use super::*;
use std::io::Write;

#[test]
#[ignore = "AUD-M1 selected real and complex onset gates"]
fn aud_m1_v3_onsets() {
    use sha2::Digest;
    let dir = output_dir();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let template = ShadowFrontEnd::new(space.clone(), params.fs, 0.25, false, kernel.nfft())
        .with_statistics(hop, true);
    let mut rows =
        std::io::BufWriter::new(std::fs::File::create(dir.join("onset-rows.jsonl")).unwrap());
    let mut summaries = Vec::new();
    for complex in [false, true] {
        let kind = if complex { "complex" } else { "real" };
        let mut inputs = [
            vec![Complex32::default(); hop],
            vec![Complex32::default(); hop],
        ];
        for input in &mut inputs {
            input[hop - 1] = Complex32::new(1.0, 0.0);
        }
        if complex {
            inputs[1][hop - 3] = Complex32::new(1e-9, 0.0);
            inputs[1][hop - 2] = Complex32::new(0.0, 1e-9);
        } else {
            let c = template
                .bank
                .centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (**a - 7896.57).abs().total_cmp(&(**b - 7896.57).abs()))
                .unwrap()
                .0;
            let pole = template.bank.poles[c];
            for (input, target) in inputs.iter_mut().zip([1000.0_f32, 2000.0]) {
                let theta = TAU * target / params.fs;
                let b = (-theta).sin() / (4.0 * pole.im);
                let a = (-theta).cos() - 4.0 * pole.re * b;
                input[hop - 3] = Complex32::new(1e-9 * b, 0.0);
                input[hop - 2] = Complex32::new(1e-9 * a, 0.0);
            }
        }
        let mut manifest = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            let bytes: Vec<_> = input
                .iter()
                .flat_map(|z| [z.re, z.im])
                .flat_map(|x| x.to_le_bytes())
                .collect();
            let name = format!("{kind}-{i}-first-hop-complex.f32");
            std::fs::write(dir.join(&name), &bytes).unwrap();
            manifest.push(json!({"file":name,"sha256":sha2::Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect::<String>(),
                "format":"interleaved real/imag little-endian f32","subsequent_silent_hops":39}));
        }
        let mut fronts = [template.clone(), template.clone()];
        let mut maximum_density = 0.0_f32;
        let mut maximum_hrc = 0.0_f64;
        for frame in 0..40 {
            let mut landscapes = Vec::new();
            for (front, input) in fronts.iter_mut().zip(&inputs) {
                if complex {
                    front.bank.read_complex_hop(input.iter().copied());
                    front.reassign(&params, hop, hop as f32 / params.fs);
                } else {
                    front.process(&input.iter().map(|z| z.re).collect::<Vec<_>>(), &params);
                }
                landscapes.push(evaluate(
                    &front.density_scan,
                    &space,
                    &params,
                    &mut scratch,
                    peak,
                ));
            }
            let mass = fronts[0]
                .density_scan
                .iter()
                .zip(&template.du)
                .map(|(d, w)| d * w)
                .sum::<f32>();
            let l1 = fronts[0]
                .density_scan
                .iter()
                .zip(&fronts[1].density_scan)
                .zip(&template.du)
                .map(|((a, b), w)| (a - b).abs() * w)
                .sum::<f32>()
                / mass.max(1e-12);
            let errors = field_errors(&landscapes[0], &landscapes[1]);
            let hrc = ["h", "r", "c"]
                .map(|k| errors[k]["max"].as_f64().unwrap())
                .into_iter()
                .fold(0.0_f64, f64::max);
            maximum_density = maximum_density.max(l1);
            maximum_hrc = maximum_hrc.max(hrc);
            writeln!(
                rows,
                "{}",
                json!({"kind":kind,"frame":frame,"relative_density_l1":l1,"fields":errors})
            )
            .unwrap();
            if l1 > 1e-3 || hrc > 1e-3 {
                rows.flush().unwrap();
                std::fs::write(
                    dir.join("onset-witness.json"),
                    serde_json::to_vec(&json!({"kind":kind,"frame":frame,"fs":params.fs,"hop":hop,
                    "inputs":manifest,"relative_density_l1":l1,"fields":errors,"tolerance":1e-3,
                    "first":fronts[0].diagnostic(),"second":fronts[1].diagnostic()}))
                    .unwrap(),
                )
                .unwrap();
            }
            assert!(l1 <= 1e-3 && hrc <= 1e-3, "registered onset PCM gate");
            for input in &mut inputs {
                input.fill(Complex32::default());
            }
        }
        summaries.push(
            json!({"kind":kind,"frames":40,"fs":params.fs,"hop":hop,"inputs":manifest,
            "max_relative_density_l1":maximum_density,"max_hrc":maximum_hrc,"pass":true}),
        );
    }
    rows.flush().unwrap();
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&summaries).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "AUD-M1 selected body history, warmup, clone, and IIR residual"]
fn aud_m1_v4_state() {
    let dir = output_dir();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let peak = compute_roughness_reference(&params, &space).peak;
    let mut scratch =
        PotentialScratch::new(&space, &params.harmonicity_kernel, &params.roughness_kernel);
    let template = ShadowFrontEnd::new(space.clone(), params.fs, 0.25, false, kernel.nfft())
        .with_statistics(hop, true);
    let mut every = template.clone();
    let mut sparse = template.clone();
    let mut body =
        with_selected_test_frontend(|| SynchronousAnalysis::new(kernel.clone(), &params));
    let mut maximum_law_error = 0.0_f32;
    for frame in 0..32 {
        let pcm: Vec<_> = (0..hop)
            .map(|i| {
                let t = (frame * hop + i) as f64 / f64::from(params.fs);
                (0.1 * (std::f64::consts::TAU * 110.0 * t).cos()
                    + 0.05 * (std::f64::consts::TAU * 150.0 * t + 0.7).cos()) as f32
            })
            .collect();
        every.process(&pcm, &params);
        if frame % 4 == 3 {
            let old = sparse.density_scan.clone();
            sparse.process_body_gap(&pcm, 4, &params);
            body.process_body_gap(&pcm, 4, &params);
            body.recompute_body_score(&params, None);
            let expected = evaluate(&sparse.density_scan, &space, &params, &mut scratch, peak);
            assert_eq!(body.landscape.harmonicity, expected.harmonicity);
            assert_eq!(body.landscape.roughness01, expected.roughness01);
            assert_eq!(sparse.bank.energy, every.bank.energy);
            assert_eq!(sparse.bank.correlation, every.bank.correlation);
            assert_eq!(sparse.channel_mass, every.channel_mass);
            let mut q = every.clone();
            q.density_scan.fill(0.0);
            q.reassign(&params, hop, 1000.0);
            let a = (-(hop as f32) / params.fs / (params.tau_ms * 1e-3))
                .exp()
                .powi(4);
            for ((&actual, &previous), &target) in
                sparse.density_scan.iter().zip(&old).zip(&q.density_scan)
            {
                let expected = a * previous + (1.0 - a) * target;
                let bound = 4.0
                    * sparse.bank.centers.len() as f32
                    * f32::EPSILON
                    * (previous.abs() + target.abs()).max(f32::MIN_POSITIVE);
                assert!(
                    (actual - expected).abs() <= bound,
                    "registered intermittent S6 law"
                );
                maximum_law_error = maximum_law_error.max((actual - expected).abs());
            }
        } else {
            sparse.skip(&pcm);
            body.skip(&pcm);
        }
    }
    let mut reset = every.clone();
    reset.reset();
    let silence = vec![0.0; hop];
    for frame in 0..kernel.nfft() / hop {
        assert_eq!(
            reset.process(&silence, &params),
            frame + 1 == kernel.nfft() / hop,
            "nfft publication hold"
        );
    }
    let mut mixed = with_selected_test_frontend(|| {
        crate::core::stream::analysis::AnalysisStream::new(params.clone(), kernel.clone())
    });
    for frame in 0..40 {
        let pcm: Vec<_> = (0..hop)
            .map(|i| 0.07 * (TAU * 220.0 * (frame * hop + i) as f32 / params.fs).cos())
            .collect();
        mixed.process(&pcm);
    }
    let mut observer = crate::core::stream::loo::Observer::new(2, 0, &mixed);
    let mut oracle = mixed.clone();
    let mut maximum_clone_error = 0.0_f32;
    let mut maximum_clone_density_l1 = 0.0_f32;
    for frame in 40..104 {
        let other: Vec<_> = (0..hop)
            .map(|i| 0.07 * (TAU * 220.0 * (frame * hop + i) as f32 / params.fs).cos())
            .collect();
        let own: Vec<_> = (0..hop)
            .map(|i| 0.05 * (TAU * 401.7 * (frame * hop + i) as f32 / params.fs + 0.3).sin())
            .collect();
        let sum: Vec<_> = other.iter().zip(&own).map(|(a, b)| a + b).collect();
        observer.process(&sum, &own);
        let reference = oracle.process(&other);
        let actual = observer.last();
        let kernel = params.consonance_field_kernel();
        for i in 0..space.n_bins() {
            let h = crate::core::float::sanitize01(actual.harmonicity[i]);
            let href = crate::core::float::sanitize01(reference.harmonicity[i]);
            let r = actual.roughness01[i];
            let rref = reference.roughness01[i];
            maximum_clone_error = maximum_clone_error
                .max((h - href).abs())
                .max((r - rref).abs())
                .max((kernel.score(h, r) - kernel.score(href, rref)).abs());
        }
        let mass = reference
            .subjective_intensity
            .iter()
            .zip(&template.du)
            .map(|(d, w)| d * w)
            .sum::<f32>();
        let l1 = actual
            .subjective_intensity
            .iter()
            .zip(&reference.subjective_intensity)
            .zip(&template.du)
            .map(|((a, b), w)| (a - b).abs() * w)
            .sum::<f32>()
            / mass.max(1e-12);
        maximum_clone_density_l1 = maximum_clone_density_l1.max(l1);
        assert!(
            maximum_clone_error <= 1e-3 && maximum_clone_density_l1 <= 1e-3,
            "cloned pre-birth history PCM gate"
        );
    }
    let mut tails = Vec::new();
    let first = template.bank.centers[0];
    let trimmed = *template
        .bank
        .centers
        .iter()
        .find(|&&f| f >= space.fmin)
        .unwrap();
    for (label, frequency) in [
        ("selected_margin", first),
        ("first_retained_at_or_above_55", trimmed),
        ("exact_55_diagnostic", 55.0),
    ] {
        let b = 0.25 * 1.0186 * (24.7 + f64::from(frequency) / 9.265);
        let radius = (-std::f64::consts::TAU * b / f64::from(params.fs)).exp();
        let count = (f64::from(params.fs)
            / (hop as f64 * 0.75 * (24.7 + f64::from(frequency) / 9.265)))
            .ceil() as usize;
        let mut state = [1.0_f64; 4];
        let mut previous = 1.0;
        let mut sum = 0.0;
        for sample in 0..kernel.nfft() {
            let mut input = 0.0;
            for stage in &mut state {
                *stage = radius * *stage + (1.0 - radius) * input;
                input = *stage;
            }
            if sample >= kernel.nfft() - count * hop {
                sum += state[3] * previous;
            }
            previous = state[3];
        }
        let power_ratio = sum / (count * hop) as f64;
        tails.push(json!({"case":label,"center_hz":frequency,"hops":count,"samples":kernel.nfft(),
            "amplitude_ratio":state[3],"compressed_instantaneous_ratio":state[3].powf(2.0*f64::from(params.loudness_exp)),
            "boxcar_power_ratio":power_ratio,"compressed_boxcar_ratio":power_ratio.powf(f64::from(params.loudness_exp))}));
    }
    std::fs::write(dir.join("summary.json"),serde_json::to_vec_pretty(&json!({"body_selected_hops":8,
        "max_absolute_s6_law_error":maximum_law_error,"max_cloned_history_hrc_error":maximum_clone_error,
        "max_cloned_history_density_l1":maximum_clone_density_l1,
        "warmup_samples":kernel.nfft(),"warmup_ms":kernel.nfft() as f64/f64::from(params.fs)*1000.0,
        "tails":tails,"tail_scope":"unit steady rotating-frame channel followed by zero input; no floor or S6 in compressed residual ratios",
        "listener":"observation ids and consumption hops deferred to M2; no shadow listener connection"})).unwrap()).unwrap();
}
