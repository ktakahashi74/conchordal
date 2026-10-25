use super::*;
use crate::core::meter::{MeterBand, MeterState};
use crate::life::phonation_engine::CandidatePoint;

#[test]
fn rearticulate_holds_tonal_frequency_through_tracking_replacement_and_tail() {
    for tone in [FloraTone::Sine, FloraTone::Resonant] {
        for active in [false, true] {
            let landscape = Landscape::new(Log2Space::new(220., 880., 96));
            let mut f = Flora::new(
                1,
                FloraSpec {
                    tone,
                    pitch_apply_mode: PitchApplyMode::Rearticulate,
                    fade_sec: 0.,
                    ..Default::default()
                },
                (220., 880.),
                FloraSeeding::Uniform,
                21,
                &landscape,
                48000.,
                512,
            );
            let mut frequencies = std::collections::BTreeMap::new();
            let mut original = None;
            let mut saw_tail = false;
            let mut saw_replacement = false;
            let mut pcm = vec![0.; 512];
            let mut habitat = pcm.clone();
            for frame in 0..180 {
                let now = frame * 512;
                // Small moves remain claimable; the final move exceeds 50 cents.
                let bin = 50
                    + if frame < 40 {
                        (frame / 10) as usize
                    } else {
                        10
                    };
                for (i, x) in f.biomass_mass_scan.iter_mut().enumerate() {
                    *x = 0.001 + 0.1 * (-0.5 * ((i as f32 - bin as f32) / 1.5).powi(2)).exp();
                }
                f.observe_rhythm(
                    TemporalObservation {
                        observed_end: Some(now),
                        known: active,
                        flux: 0.5,
                        state: MeterState {
                            beat: MeterBand {
                                freq_hz: 2.,
                                confidence: 1.,
                                amplitude: 1.,
                                phase: 0.,
                            },
                            ..Default::default()
                        },
                    },
                    now,
                    Default::default(),
                );
                let m = f.temporal.as_mut().unwrap();
                m.crossings.clear();
                if active && frame % 5 == 0 {
                    m.crossings.push(CandidatePoint {
                        tick: now + 173,
                        gate: frame,
                    });
                }
                pcm.fill(0.);
                habitat.fill(0.);
                f.render_add(&mut pcm, &mut habitat);
                for lane in f.lanes.iter().filter(|l| l.tonal && l.mode != Mode::Free) {
                    let expected = *frequencies
                        .entry(lane.generation)
                        .or_insert(lane.position.to_bits());
                    assert_eq!(lane.position.to_bits(), expected);
                    assert_eq!(lane.target_position.to_bits(), expected);
                    let (sin, cos) = (TAU * lane.position.exp2() / f.fs).sin_cos();
                    assert_eq!(lane.w, [cos, sin]);
                    let first = *original.get_or_insert(lane.generation);
                    saw_tail |= lane.generation == first && lane.mode == Mode::Retiring;
                    saw_replacement |= lane.generation != first;
                }
                let report = f.report(now as f64 / f.fs as f64, &landscape);
                for lane in report
                    .lanes
                    .iter()
                    .filter(|l| l.state != "free" && l.timing.as_ref().unwrap().generation > 0)
                {
                    let generation = lane.timing.as_ref().unwrap().generation;
                    assert_eq!(
                        lane.frequency_hz.to_bits(),
                        f32::from_bits(frequencies[&generation]).exp2().to_bits()
                    );
                }
                let events = serde_json::to_value(&f.temporal.as_ref().unwrap().events).unwrap();
                for event in events.as_array().unwrap() {
                    if active && event["action"] == "born" {
                        assert_eq!(event["scheduled_tick"], event["tick"]);
                        assert_eq!(event["tick"], now + 173);
                    }
                }
            }
            assert!(saw_tail && saw_replacement);
            assert!(frequencies.len() >= 2);
        }
    }
}

#[test]
fn live_pitch_mode_preserves_lanes_and_freezes_at_next_allocation() {
    let landscape = Landscape::new(Log2Space::new(220., 880., 96));
    let mut f = Flora::new(
        1,
        FloraSpec::default(),
        (220., 880.),
        FloraSeeding::Uniform,
        21,
        &landscape,
        48000.,
        512,
    );
    for (i, x) in f.biomass_mass_scan.iter_mut().enumerate() {
        *x = 0.001 + 0.1 * (-0.5 * ((i as f32 - 50.) / 1.5).powi(2)).exp();
    }
    f.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
    let before: Vec<_> = f
        .lanes
        .iter()
        .map(|l| {
            (
                l.generation,
                l.mode,
                if l.tonal { l.position.to_bits() } else { 0 },
            )
        })
        .collect();
    f.apply_patch(&FloraPatch {
        pitch_apply_mode: Some(PitchApplyMode::Rearticulate),
        ..Default::default()
    });
    assert_eq!(
        before,
        f.lanes
            .iter()
            .map(|l| (
                l.generation,
                l.mode,
                if l.tonal { l.position.to_bits() } else { 0 }
            ))
            .collect::<Vec<_>>()
    );
    for (i, x) in f.biomass_mass_scan.iter_mut().enumerate() {
        *x = 0.001 + 0.1 * (-0.5 * ((i as f32 - 52.) / 1.5).powi(2)).exp();
    }
    f.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
    assert_eq!(
        before,
        f.lanes
            .iter()
            .map(|l| (
                l.generation,
                l.mode,
                if l.tonal { l.position.to_bits() } else { 0 }
            ))
            .collect::<Vec<_>>()
    );
    f.apply_patch(&FloraPatch {
        pitch_apply_mode: Some(PitchApplyMode::Glide),
        ..Default::default()
    });
    f.render_add(&mut vec![0.; 512], &mut vec![0.; 512]);
    assert!(
        f.lanes.iter().enumerate().any(|(i, l)| l.tonal
            && l.mode == Mode::Current
            && l.position.to_bits() != before[i].2)
    );
}
