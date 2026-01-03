use conchordal::life::intent_planner::choose_freq_by_consonance;

#[test]
fn picks_fifth_over_off_ratio() {
    let neighbors = vec![440.0];
    let candidates = vec![466.16, 660.0];
    let chosen = choose_freq_by_consonance(&candidates, &neighbors, 440.0).unwrap();
    assert!((chosen - 660.0).abs() < 1e-2);
}

#[test]
fn picks_unison_when_available() {
    let neighbors = vec![440.0];
    let candidates = vec![440.0, 660.0];
    let chosen = choose_freq_by_consonance(&candidates, &neighbors, 440.0).unwrap();
    assert!((chosen - 440.0).abs() < 1e-6);
}

#[test]
fn falls_back_to_base_distance_when_no_neighbors() {
    let neighbors = Vec::new();
    let candidates = vec![400.0, 440.0, 480.0];
    let chosen = choose_freq_by_consonance(&candidates, &neighbors, 450.0).unwrap();
    assert!((chosen - 440.0).abs() < 1e-6);
}
