use std::collections::HashMap;

use conchordal::core::log2space::Log2Space;
use conchordal::life::intent_planner::{GestureChoiceTf, choose_best_gesture_tf_by_pred_c};

#[test]
fn chooses_onset_with_higher_pred_c_peak() {
    let space = Log2Space::new(110.0, 880.0, 12);
    let onset_candidates = vec![100u64, 200u64];

    let mut scans: HashMap<u64, Vec<f32>> = HashMap::new();
    let mut scan_a = vec![0.0f32; space.n_bins()];
    let idx_a = space.index_of_freq(440.0).expect("bin");
    scan_a[idx_a] = 0.3;
    if idx_a + 1 < scan_a.len() {
        scan_a[idx_a + 1] = 0.3;
    }
    scans.insert(100, scan_a);

    let mut scan_b = vec![0.0f32; space.n_bins()];
    let idx_b = space.index_of_freq(660.0).expect("bin");
    scan_b[idx_b] = 0.9;
    if idx_b + 1 < scan_b.len() {
        scan_b[idx_b + 1] = 0.9;
    }
    scans.insert(200, scan_b);

    let f_a = space.freq_of_index(idx_a);
    let f_b = space.freq_of_index(idx_b);
    let mut make_freq_candidates = |_: u64| vec![f_a, f_b];
    let mut pred_c_scan_at = |t: u64| {
        scans
            .get(&t)
            .map(|v| std::sync::Arc::<[f32]>::from(v.clone()))
    };

    let choice = choose_best_gesture_tf_by_pred_c(
        &space,
        &onset_candidates,
        440.0,
        &mut make_freq_candidates,
        &mut pred_c_scan_at,
    )
    .expect("choice");

    assert_eq!(choice.onset, 200);
    assert_eq!(choice.freq_hz, f_b);
}

#[test]
fn ties_choose_earliest_onset() {
    let space = Log2Space::new(110.0, 880.0, 12);
    let onset_candidates = vec![50u64, 100u64];
    let mut scans: HashMap<u64, Vec<f32>> = HashMap::new();
    let scan = vec![0.0f32; space.n_bins()];
    scans.insert(50, scan.clone());
    scans.insert(100, scan);

    let f_center = space.freq_of_index(space.n_bins() / 2);
    let mut make_freq_candidates = |_: u64| vec![f_center];
    let mut pred_c_scan_at = |t: u64| {
        scans
            .get(&t)
            .map(|v| std::sync::Arc::<[f32]>::from(v.clone()))
    };

    let choice: GestureChoiceTf = choose_best_gesture_tf_by_pred_c(
        &space,
        &onset_candidates,
        f_center,
        &mut make_freq_candidates,
        &mut pred_c_scan_at,
    )
    .expect("choice");

    assert_eq!(choice.onset, 50);
}
