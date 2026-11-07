use super::*;
use crate::core::roughness_kernel::erb_grid;

#[test]
fn hr2_production_bank_and_new_input_bands() {
    let mut centroids = Vec::new();
    for frequency in [22.0_f64, 25.0, 10000.0, 12000.0] {
        let core = build_analysis_runtime_core(&AppConfig::default(), 48_000);
        let params = core.lparams;
        let mut stream = AnalysisStream::new(params.clone(), core.nsgt);
        let diagnostic = stream.aud_m1_shadow_diagnostic();
        let centers = diagnostic["centers_hz"].as_array().unwrap();
        println!(
            "HR2_BANK channels={} lowest={:.9} highest={:.9}",
            centers.len(),
            centers[0].as_f64().unwrap(),
            centers.last().unwrap().as_f64().unwrap()
        );
        assert_eq!(centers.len(), 198);
        assert_eq!(centers[0].as_f64().unwrap() as f32, 14.574448_f32);
        assert_eq!(centers[197].as_f64().unwrap() as f32, 12889.027_f32);
        assert_eq!(stream.last().space.n_bins(), 892);
        let (_, du) = erb_grid(&stream.last().space);
        let mut observed = 0;
        let mut centroid_sum = 0.0;
        let mut min_mass = f32::INFINITY;
        let mut min_gate_density = f32::INFINITY;
        // Peak amplitude 0.1 and phase zero; complete hops within 1..1.25 s.
        // This follows the window, slowest filter settling and boxcars.
        for frame in 0..118 {
            let pcm: Vec<_> = (0..512)
                .map(|i| {
                    let t = (frame * 512 + i) as f64 / 48000.0;
                    (0.1 * (std::f64::consts::TAU * frequency * t).sin()) as f32
                })
                .collect();
            stream.process(&pcm);
            if frame * 512 < 48000 || (frame + 1) * 512 > 60000 {
                continue;
            }
            let diagnostic = stream.aud_m1_shadow_diagnostic();
            let landscape = stream.last();
            let density = &landscape.subjective_intensity;
            assert!(density.iter().all(|x| x.is_finite() && *x >= 0.0));
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
                .map(|((d, w), hz)| d * w * hz)
                .sum::<f32>()
                / mass;
            let band = if frequency < 30.0 {
                20.0..30.0
            } else {
                8000.0..12500.0
            };
            assert!(
                band.contains(&centroid),
                "{frequency} Hz centroid={centroid}"
            );
            let peak = density
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert!(peak > 0 && peak + 1 < density.len());
            assert!(density[peak] > density[0] && density[peak] > density[density.len() - 1]);
            // Independently check the pre-compression local/width floor gate.
            let gate_density = diagnostic["frequency_hz"]
                .as_array()
                .unwrap()
                .iter()
                .zip(diagnostic["pcb"].as_array().unwrap())
                .zip(diagnostic["deposit_power"].as_array().unwrap())
                .filter_map(|((hz, local), power)| {
                    let hz = hz.as_f64().unwrap() as f32;
                    if !hz.is_finite() || hz <= 0.0 || power.as_f64().unwrap() <= 0.0 {
                        return None;
                    }
                    let position = ((hz.log2() - landscape.space.centers_log2[0])
                        / landscape.space.step())
                    .clamp(0.0, (du.len() - 1) as f32);
                    let lo = position.floor() as usize;
                    let hi = (lo + 1).min(du.len() - 1);
                    let fraction = position - lo as f32;
                    let width = (1.0 - fraction) * du[lo] + fraction * du[hi];
                    Some(local.as_f64().unwrap() as f32 / width)
                })
                .fold(0.0_f32, f32::max);
            assert!(
                gate_density > 1e-6,
                "{frequency} Hz local/width={gate_density}"
            );
            min_gate_density = min_gate_density.min(gate_density);
            centroid_sum += centroid;
            observed += 1;
        }
        assert!(observed > 0);
        let centroid = centroid_sum / observed as f32;
        println!(
            "HR2_INPUT hz={frequency} centroid={centroid} min_mass={min_mass} min_local_per_width={min_gate_density} hops={observed}"
        );
        centroids.push(centroid);
    }
    assert!(centroids.windows(2).all(|p| p[0] < p[1]));
}

#[test]
fn hr2_flora_uses_full_low_and_high_fields() {
    use crate::life::flora::Flora;
    use crate::scenario::{FloraSeeding, FloraSpec};
    for (extent, frequency) in [((20.0, 30.0), 25.0_f64), ((9000.0, 12000.0), 10000.0)] {
        let core = build_analysis_runtime_core(&AppConfig::default(), 48000);
        let mut stream = AnalysisStream::new(core.lparams.clone(), core.nsgt);
        let mut flora = Flora::new(
            7,
            FloraSpec::default(),
            extent,
            FloraSeeding::Uniform,
            21,
            stream.last(),
            48000.0,
            512,
        );
        let before = serde_json::to_value(flora.report(0.0, stream.last())).unwrap();
        for frame in 0..118 {
            let pcm: Vec<_> = (0..512)
                .map(|i| {
                    let t = (frame * 512 + i) as f64 / 48000.0;
                    (0.1 * (std::f64::consts::TAU * frequency * t).sin()) as f32
                })
                .collect();
            stream.process(&pcm);
        }
        let mut field = stream.last().clone();
        field.recompute_consonance(&core.lparams);
        flora.advance(&field);
        let landscape = &field;
        let after = serde_json::to_value(flora.report(1.25, landscape)).unwrap();
        let (lo, hi) = landscape
            .space
            .bin_range_of_freqs(extent.0, extent.1)
            .unwrap();
        assert!(lo < hi);
        assert!(landscape.harmonicity01[lo..=hi].iter().any(|x| *x > 0.0));
        assert!(landscape.roughness01[lo..=hi].iter().any(|x| *x > 0.0));
        assert_ne!(before["growth_level_scan"], after["growth_level_scan"]);
        assert_ne!(
            before["harmonicity_state_scan"],
            after["harmonicity_state_scan"]
        );
        assert_eq!(
            after["growth_level_scan"],
            serde_json::to_value(&landscape.consonance_field_level).unwrap()
        );
        assert_eq!(
            after["harmonicity_state_scan"],
            serde_json::to_value(&landscape.harmonicity01).unwrap()
        );
    }
}
