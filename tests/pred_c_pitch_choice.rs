use conchordal::core::log2space::{Log2Space, sample_scan_linear_log2};
use conchordal::life::intent_planner::choose_best_freq_by_pred_c;

#[test]
fn pred_c_argmax_picks_peak_bin() {
    let space = Log2Space::new(110.0, 880.0, 12);
    let mut scan = vec![0.0f32; space.n_bins()];
    let idx = space.index_of_freq(440.0).expect("bin");
    scan[idx] = 1.0;
    if idx + 1 < scan.len() {
        scan[idx + 1] = 1.0;
    }

    let f_center = space.freq_of_index(idx);
    let candidates = vec![330.0, f_center, 660.0];
    let (chosen, score) =
        choose_best_freq_by_pred_c(&space, &scan, &candidates, 400.0).expect("choice");

    assert_eq!(chosen, f_center);
    let sampled = sample_scan_linear_log2(&space, &scan, chosen);
    assert!((score - sampled).abs() < 1e-6);
}

#[test]
fn pred_c_tie_break_prefers_base_distance_then_low_freq() {
    let space = Log2Space::new(110.0, 880.0, 12);
    let scan = vec![0.0f32; space.n_bins()];
    let candidates = vec![500.0, 600.0];
    let (chosen, _score) =
        choose_best_freq_by_pred_c(&space, &scan, &candidates, 500.0).expect("choice");
    assert_eq!(chosen, 500.0);
}
