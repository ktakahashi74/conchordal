use super::*;
use crate::core::roughness_kernel::erb_grid;

fn production_analysis() -> (AnalysisStream, LandscapeParams) {
    let config = AppConfig::default();
    let core = build_analysis_runtime_core(&config, 48_000);
    let params = core.lparams;
    let stream = AnalysisStream::new(params.clone(), core.nsgt);
    assert_eq!(stream.hop_samples(), 512);
    assert_eq!(stream.window_samples(), 16384);
    (stream, params)
}

#[test]
fn lowrange_input_evidence_moves_below_55_hz() {
    let mut centroids = Vec::new();
    for frequency in [31.0, 40.0, 50.0] {
        let (mut stream, params) = production_analysis();
        let (_, du) = erb_grid(&stream.last().space);
        let mut centroid_sum = 0.0;
        let mut observed = 0;
        let mut min_mass = f32::INFINITY;
        // Fixed peak amplitude 0.1, phase 0; observe complete hops in 0.5..1 s.
        // Ordering and band membership test routing, without a one-bin accuracy claim.
        for frame in 0..(48_000 / 512) {
            let pcm: Vec<_> = (0..512)
                .map(|i| {
                    let t = (frame * 512 + i) as f64 / 48_000.0;
                    (0.1 * (std::f64::consts::TAU * frequency * t).sin()) as f32
                })
                .collect();
            let landscape = stream.process(&pcm);
            if frame * 512 < 24_000 {
                continue;
            }
            let density = &landscape.subjective_intensity;
            assert!(density.iter().all(|d| d.is_finite() && *d >= 0.0));
            let mass: f32 = density.iter().zip(&du).map(|(d, w)| d * w).sum();
            assert!(
                mass > params.roughness_ref_eps,
                "{frequency} Hz mass={mass}"
            );
            min_mass = min_mass.min(mass);
            let centroid = density
                .iter()
                .zip(&du)
                .zip(&landscape.space.centers_hz)
                .map(|((d, w), f)| d * w * f)
                .sum::<f32>()
                / mass;
            assert!(
                (30.0..55.0).contains(&centroid),
                "{frequency} Hz centroid={centroid}"
            );
            // A strict interior mode rejects density pinned to the first grid bin.
            let peak = density
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert!(peak > 0 && landscape.space.centers_hz[peak] < 55.0);
            assert!(density[peak] > density[0]);
            centroid_sum += centroid;
            observed += 1;
        }
        assert!(observed > 0);
        let centroid = centroid_sum / observed as f32;
        println!(
            "lowrange input={frequency} centroid={centroid} min_mass={min_mass} hops={observed}"
        );
        centroids.push(centroid);
    }
    assert!(centroids.windows(2).all(|p| p[0] < p[1]));
}

#[test]
fn lowrange_candidates_match_unmodified_harmonicity_law() {
    let (mut stream, params) = production_analysis();
    // Specify the reference grid independently of production_space to catch truncation.
    let reference_space = Log2Space::new(55.0 * 2.0_f32.powf(-7.0 / 8.0), 8000.0, 96);
    let reference = HarmonicityKernel::new(&reference_space, HarmonicityParams::default());
    let (_, du) = erb_grid(&reference_space);
    let mut observed = 0;
    let mut minimum_support = f32::INFINITY;
    for frame in 0..(48_000 / 512) {
        // F0=40 Hz, harmonics 2..8 only; each peak amplitude 0.1/7, phase 0.
        let pcm: Vec<_> = (0..512)
            .map(|i| {
                let t = (frame * 512 + i) as f64 / 48_000.0;
                ((2..=8)
                    .map(|k| (std::f64::consts::TAU * 40.0 * f64::from(k) * t).sin())
                    .sum::<f64>()
                    * (0.1 / 7.0)) as f32
            })
            .collect();
        let landscape = stream.process(&pcm);
        if frame * 512 < 24_000 {
            continue;
        }
        assert_eq!(landscape.space.centers_log2, reference_space.centers_log2);
        let density = &landscape.subjective_intensity;
        assert!(density.iter().all(|d| d.is_finite() && *d >= 0.0));
        let mass: f32 = density.iter().zip(&du).map(|(d, w)| d * w).sum();
        assert!(mass > params.roughness_ref_eps);
        let (expected, _) = reference.potential_h_from_log2_spectrum(density, &reference_space);
        let mut support = 0.0;
        for (i, &hz) in reference_space.centers_hz.iter().enumerate() {
            if (30.0..55.0).contains(&hz) {
                let actual = landscape.harmonicity[i];
                assert!(actual.is_finite() && actual >= 0.0);
                // Both paths apply the identical unchanged kernel to identical density.
                assert_eq!(actual, expected[i], "candidate {hz} Hz");
                support += actual;
            }
        }
        assert!(support > 0.0);
        minimum_support = minimum_support.min(support);
        observed += 1;
    }
    assert!(observed > 0);
    println!("lowrange candidate min_support={minimum_support} hops={observed}");
}

#[test]
fn lowrange_production_bank_has_extended_channels() {
    let (stream, _) = production_analysis();
    assert_eq!(stream.last().space.n_bins(), 774);
    let diagnostic = stream.aud_m1_shadow_diagnostic();
    let centers = diagnostic["centers_hz"].as_array().unwrap();
    assert_eq!(centers.len(), 174);
    // Half the final displayed millihertz permits rounding of the specified centers.
    assert!((centers[0].as_f64().unwrap() - 24.629).abs() < 0.0005);
    assert!((centers[173].as_f64().unwrap() - 8175.326).abs() < 0.0005);
}
