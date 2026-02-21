use std::sync::Arc;

use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::{NeuralRhythms, RhythmBand};
use conchordal::core::timebase::Timebase;
use conchordal::life::world_model::WorldModel;

#[test]
fn tau_tick_depends_on_theta_delta() {
    let timebase = Timebase {
        fs: 48_000.0,
        hop: 512,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let world = WorldModel::new(timebase, space);
    let rhythm_low = NeuralRhythms {
        theta: RhythmBand {
            freq_hz: 2.0,
            ..Default::default()
        },
        delta: RhythmBand {
            freq_hz: 4.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let rhythm_high = NeuralRhythms {
        theta: RhythmBand {
            freq_hz: 10.0,
            ..Default::default()
        },
        delta: RhythmBand {
            freq_hz: 0.2,
            ..Default::default()
        },
        ..Default::default()
    };
    let (tau_low, _) = world.predictor_tau_horizon_ticks(&rhythm_low);
    let (tau_high, _) = world.predictor_tau_horizon_ticks(&rhythm_high);
    let theta_period_low = timebase.sec_to_tick(1.0 / 2.0);
    let theta_period_high = timebase.sec_to_tick(1.0 / 10.0);
    assert_eq!(tau_low, theta_period_low.saturating_mul(2));
    assert_eq!(tau_high, theta_period_high.saturating_mul(8));
    assert_ne!(tau_low, tau_high);
}

#[test]
fn tau_tick_clamps_theta_delta_ratio() {
    let timebase = Timebase {
        fs: 48_000.0,
        hop: 512,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let world = WorldModel::new(timebase, space);
    let rhythm_small = NeuralRhythms {
        theta: RhythmBand {
            freq_hz: 1.0,
            ..Default::default()
        },
        delta: RhythmBand {
            freq_hz: 100.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let rhythm_large = NeuralRhythms {
        theta: RhythmBand {
            freq_hz: 50.0,
            ..Default::default()
        },
        delta: RhythmBand {
            freq_hz: 0.1,
            ..Default::default()
        },
        ..Default::default()
    };
    let (tau_small, _) = world.predictor_tau_horizon_ticks(&rhythm_small);
    let (tau_large, _) = world.predictor_tau_horizon_ticks(&rhythm_large);
    let theta_period_small = timebase.sec_to_tick(1.0 / 1.0);
    let theta_period_large = timebase.sec_to_tick(1.0 / 50.0);
    assert_eq!(tau_small, theta_period_small.saturating_mul(2));
    assert_eq!(tau_large, theta_period_large.saturating_mul(8));
}

#[test]
fn prediction_decay_reduces_far_extrapolation() {
    let timebase = Timebase {
        fs: 48_000.0,
        hop: 512,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(timebase, space);
    let rhythm = NeuralRhythms {
        theta: RhythmBand {
            freq_hz: 4.0,
            ..Default::default()
        },
        delta: RhythmBand {
            freq_hz: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    world.update_gate_from_rhythm(0, &rhythm);

    let (tau_tick, _) = world.predictor_tau_horizon_ticks(&rhythm);
    let n_bins = world.space.n_bins();
    let prev_scan = vec![0.2; n_bins];
    let last_scan = vec![0.4; n_bins];
    let prev_tick = 0;
    let last_tick = tau_tick;
    world.observe_consonance_level01(prev_tick, Arc::from(prev_scan));
    world.observe_consonance_level01(last_tick, Arc::from(last_scan));

    let near_tick = last_tick.saturating_add(tau_tick);
    let far_tick = last_tick.saturating_add(tau_tick.saturating_mul(2));
    let pred_near = world
        .predict_consonance_level01_at(near_tick)
        .expect("pred near");
    let pred_far = world
        .predict_consonance_level01_at(far_tick)
        .expect("pred far");
    let delta_near = (pred_near[0] - 0.4).abs();
    let delta_far = (pred_far[0] - 0.4).abs();
    assert!(delta_far < delta_near);
}
