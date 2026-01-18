use std::sync::Arc;

use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::Timebase;
use conchordal::life::world_model::WorldModel;

#[test]
fn world_model_pred_cache_resets_on_updates() {
    let time = Timebase {
        fs: 48_000.0,
        hop: 512,
    };
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut world = WorldModel::new(time, space.clone());
    let mut rhythm = NeuralRhythms::default();
    rhythm.theta.freq_hz = 6.0;
    rhythm.theta.phase = 0.0;
    rhythm.delta.freq_hz = 2.0;

    world.update_gate_from_rhythm(0, &rhythm);
    let scan = Arc::from(vec![0.5f32; space.n_bins()]);
    world.observe_consonance01(time.frame_end_tick(0), scan);
    assert!(world.predict_consonance01_next_gate().is_some());
    assert!(world.last_pred_next_gate().is_some());

    world.update_gate_from_rhythm(0, &rhythm);
    assert!(world.last_pred_next_gate().is_none());

    let scan2 = Arc::from(vec![0.4f32; space.n_bins()]);
    world.observe_consonance01(time.frame_end_tick(1), scan2);
    let _ = world.predict_consonance01_next_gate();
    assert!(world.last_pred_next_gate().is_some());

    let scan3 = Arc::from(vec![0.3f32; space.n_bins()]);
    world.observe_consonance01(time.frame_end_tick(2), scan3);
    assert!(world.last_pred_next_gate().is_none());
}
