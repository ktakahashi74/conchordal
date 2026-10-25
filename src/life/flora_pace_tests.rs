use super::*;
use crate::core::meter::{F_BEAT_MIN, MeterShaping};
use crate::scenario::FLORA_PACE_MIN;

fn fixed_flora(hop: usize) -> (Landscape, Flora) {
    let mut terrain = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
    for i in 0..terrain.space.n_bins() {
        terrain.consonance_field_level[i] = 0.6 + 0.3 * (i as f32 * 0.03).sin();
        terrain.harmonicity01[i] = 0.4 + 0.3 * (i as f32 * 0.02).cos();
    }
    let mut f = Flora::new(
        1,
        FloraSpec::default(),
        (55.0, 8000.0),
        FloraSeeding::Uniform,
        21,
        &terrain,
        48_000.0,
        hop,
    );
    f.external_observed_frame = Some(0);
    for i in f.range.clone() {
        f.biomass_mass_scan[i] = (0.4 + 0.3 * (i as f32 * 0.08).sin()) / 96.0;
        f.resource_state_scan[i] = 0.7;
        f.external_consumption_scan[i] = 0.05 + 0.03 * (i as f32 * 0.04).cos();
    }
    (terrain, f)
}

#[test]
fn pace_scales_ecology_within_measured_substep_error() {
    // 672 samples produces 3 neutral substeps and 5 at double pace, so this
    // checks unequal discretizations rather than identical update sequences.
    let (terrain, original) = fixed_flora(672);
    let mut slow = original.clone();
    let mut fast = original.clone();
    let mut refined = original.clone();
    fast.apply_patch(&FloraPatch {
        pace: Some(2.0),
        ..Default::default()
    });
    for _ in 0..100 {
        fast.advance(&terrain);
        for _ in 0..2 {
            slow.advance(&terrain);
            let n = slow.substeps * 2;
            let dt = slow.hop as f32 / slow.fs / n as f32;
            refined.advance_ecology(dt, n);
        }
    }
    assert_eq!((slow.substeps, fast.substeps), (3, 5));
    for (name, a, b, fine) in [
        (
            "biomass",
            &fast.biomass_mass_scan,
            &slow.biomass_mass_scan,
            &refined.biomass_mass_scan,
        ),
        (
            "resources",
            &fast.resource_state_scan,
            &slow.resource_state_scan,
            &refined.resource_state_scan,
        ),
    ] {
        let error = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let discretization = b
            .iter()
            .zip(fine)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("pace scaling {name}: error={error:e} refinement={discretization:e}");
        assert!(discretization > 0.0);
        assert!(
            error <= discretization,
            "{name}: {error} > {discretization}"
        );
    }
}

#[test]
fn pace_live_patch_preserves_clock_and_birth_slew() {
    let (terrain, mut f) = fixed_flora(512);
    let shaping = MeterShaping {
        basin_hz: Some((1.0, 3.0)),
        ..Default::default()
    };
    f.observe_rhythm(TemporalObservation::default(), 0, shaping);
    let mut reference = f.clone();
    f.apply_patch(&FloraPatch {
        pace: Some(4.0),
        ..Default::default()
    });
    for frame in 1..10 {
        let now = frame * 512;
        f.observe_rhythm(TemporalObservation::default(), now, shaping);
        reference.observe_rhythm(TemporalObservation::default(), now, shaping);
        assert_eq!(
            serde_json::to_value(f.temporal.as_ref().unwrap()).unwrap(),
            serde_json::to_value(reference.temporal.as_ref().unwrap()).unwrap()
        );
        assert_eq!(
            f.temporal.as_ref().unwrap().pace,
            reference.temporal.as_ref().unwrap().pace
        );
        assert_eq!(f.pace, 4.0 * reference.pace);
    }
    f.advance(&terrain);
    f.apply_patch(&FloraPatch {
        pace: Some(1.0),
        ..Default::default()
    });
    assert_eq!(f.pace, reference.pace);
    let mut lane = Lane::new(false, 8.0);
    lane.mode = Mode::Current;
    lane.target = 0.5;
    let samples = (SLEW_SEC * f.fs * lane.target).ceil() as usize;
    for _ in 0..samples {
        lane.slew(f.fs);
    }
    assert_eq!(lane.r, 0.5);
}

#[test]
fn pace_minimum_has_strict_decay_and_finite_retirement() {
    let (terrain, mut f) = fixed_flora(512);
    f.apply_patch(&FloraPatch {
        pace: Some(FLORA_PACE_MIN),
        ..Default::default()
    });
    f.observe_rhythm(
        TemporalObservation::default(),
        0,
        MeterShaping {
            basin_hz: Some((F_BEAT_MIN, F_BEAT_MIN)),
            ..Default::default()
        },
    );
    // sane_basin widens a zero-width prior; use the actual lower beat bound.
    f.pace = FLORA_PACE_MIN * F_BEAT_MIN / 2.0;
    f.advance(&terrain);
    let dt = f.hop as f32 / f.fs * f.pace / f.substeps as f32;
    // Renewal alone is the smallest resource exponent; consumption adds to it.
    assert!((-(1.0 / RENEWAL_SEC) * dt).exp() < 1.0);
    assert!(1.0 - GROWTH_RATE * METABOLISM * dt < 1.0);
    let mut presentation = vec![0.0; f.hop];
    let mut habitat = presentation.clone();
    f.render_add(&mut presentation, &mut habitat);
    let decay = f.lanes[0].retirement_decay;
    assert!(decay > 0.0 && decay < 1.0);
    let previous =
        (-0.5 * GROWTH_RATE * (FLORA_PACE_MIN / 2.0 * F_BEAT_MIN / 2.0) * METABOLISM / f.fs).exp();
    let mut stuck = Lane::new(false, 8.0);
    stuck.mode = Mode::Retiring;
    stuck.r = 0.09337722;
    stuck.retirement_decay = previous;
    let fixed_point = stuck.r;
    stuck.slew(f.fs);
    assert_eq!(
        stuck.r, fixed_point,
        "the next smaller power of two must fail"
    );
    let mut lane = Lane::new(false, 8.0);
    lane.mode = Mode::Retiring;
    lane.r = 1.0;
    lane.retirement_decay = decay;
    // Each step must decrease a positive f32, so the number of representable
    // values from 0 to 1 is a finite upper bound, independent of elapsed time.
    let mut samples = 0_u32;
    while lane.mode != Mode::Free {
        let before = lane.r;
        lane.slew(f.fs);
        assert!(lane.r < before, "retirement stalled at {before}");
        samples += 1;
        assert!(samples <= 1.0_f32.to_bits());
    }
    eprintln!(
        "pace minimum k={FLORA_PACE_MIN} f={F_BEAT_MIN} decay={decay} retirement_samples={samples} seconds={}",
        samples as f64 / f.fs as f64
    );
}

#[test]
fn pace_one_explicit_matches_default_state() {
    let (terrain, mut omitted) = fixed_flora(512);
    let mut explicit = omitted.clone();
    explicit.apply_patch(&FloraPatch {
        pace: Some(1.0),
        ..Default::default()
    });
    for _ in 0..32 {
        omitted.advance(&terrain);
        explicit.advance(&terrain);
        assert_eq!(omitted.biomass_mass_scan, explicit.biomass_mass_scan);
        assert_eq!(omitted.resource_state_scan, explicit.resource_state_scan);
    }
}

#[test]
#[ignore = "exclusive timing acquisition; full default extent, identical states"]
fn flora_pace_advance_cost() {
    use std::hint::black_box;
    use std::time::Instant;
    let (terrain, original) = fixed_flora(512);
    for k in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0] {
        let mut f = original.clone();
        // Use the fastest supported beat to cover the ordinary 0.5..4 Hz range.
        f.pace = k * 2.0;
        for _ in 0..32 {
            f.advance(black_box(&terrain));
        }
        let mut elapsed = std::time::Duration::ZERO;
        for _ in 0..1024 {
            f.biomass_mass_scan
                .copy_from_slice(&original.biomass_mass_scan);
            f.resource_state_scan
                .copy_from_slice(&original.resource_state_scan);
            let start = Instant::now();
            f.advance(black_box(&terrain));
            elapsed += start.elapsed();
            black_box(&f);
        }
        let micros = elapsed.as_secs_f64() * 1e6 / 1024.0;
        eprintln!(
            "pace_cost k={k} beat_hz=4 bins={} substeps={} mean_us={micros:.6}",
            terrain.space.n_bins(),
            f.substeps
        );
        if k >= 16.0 && micros > 530.0 {
            break;
        }
    }
}

#[test]
fn pace_retirement_duration_record() {
    let (terrain, original) = fixed_flora(512);
    for k in [1.0, 2.0, 4.0, 8.0, 16.0] {
        let mut f = original.clone();
        f.apply_patch(&FloraPatch {
            pace: Some(k),
            ..Default::default()
        });
        f.advance(&terrain);
        let mut presentation = vec![0.0; f.hop];
        let mut habitat = presentation.clone();
        f.render_add(&mut presentation, &mut habitat);
        let mut lane = f.lanes[0].clone();
        lane.mode = Mode::Retiring;
        lane.r = 0.5;
        let mut samples = 0;
        while lane.mode != Mode::Free {
            let before = lane.r;
            lane.slew(f.fs);
            assert!(lane.r < before);
            samples += 1;
        }
        eprintln!(
            "pace_tail k={k} beat_hz=2 initial_r=0.5 samples={samples} seconds={}",
            samples as f64 / f.fs as f64
        );
    }
}

#[test]
fn pace_saturation_preserves_ecology_render_and_retirement() {
    let (terrain, original) = fixed_flora(512);
    let threshold = 2.0 * RENEWAL_SEC;
    let mut reference = original.clone();
    reference.apply_patch(&FloraPatch {
        pace: Some(threshold),
        ..Default::default()
    });
    for k in [threshold * 2.0, f32::MAX] {
        let mut saturated = original.clone();
        let mut expected = reference.clone();
        saturated.apply_patch(&FloraPatch {
            pace: Some(k),
            ..Default::default()
        });
        for hop in 0..4 {
            for f in [&mut expected, &mut saturated] {
                f.observe_rhythm(
                    TemporalObservation::default(),
                    hop * 512,
                    MeterShaping::default(),
                );
                f.advance(&terrain);
                assert_eq!(f.substeps, 35);
                assert!(
                    f.biomass_mass_scan
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0)
                );
                assert!(
                    f.resource_state_scan
                        .iter()
                        .all(|u| u.is_finite() && (0.0..=1.0).contains(u))
                );
            }
            assert_eq!(expected.biomass_mass_scan, saturated.biomass_mass_scan);
            assert_eq!(expected.resource_state_scan, saturated.resource_state_scan);
            let mut a = vec![0.0; 512];
            let mut b = vec![0.0; 512];
            let mut ah = vec![0.0; 512];
            let mut bh = vec![0.0; 512];
            expected.render_add(&mut a, &mut ah);
            saturated.render_add(&mut b, &mut bh);
            assert_eq!(a, b);
            assert_eq!(ah, bh);
            assert!(a.iter().all(|v| v.is_finite()));
            for (a, b) in expected.lanes.iter().zip(&saturated.lanes) {
                assert_eq!(a.retirement_decay, b.retirement_decay);
            }
        }
    }
}

#[test]
fn pace_sixty_four_keeps_ecology_and_pcm_finite() {
    let (terrain, mut f) = fixed_flora(512);
    f.apply_patch(&FloraPatch {
        pace: Some(64.0),
        ..Default::default()
    });
    let mut presentation = vec![0.0; 512];
    let mut habitat = vec![0.0; 512];
    for _ in 0..64 {
        f.advance(&terrain);
        presentation.fill(0.0);
        habitat.fill(0.0);
        f.render_add(&mut presentation, &mut habitat);
        assert!(presentation.iter().chain(&habitat).all(|v| v.is_finite()));
        assert!(
            f.biomass_mass_scan
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
        assert!(
            f.resource_state_scan
                .iter()
                .all(|u| u.is_finite() && (0.0..=1.0).contains(u))
        );
    }
}
