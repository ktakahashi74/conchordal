use conchordal::life::individual::{PitchCore, PitchHillClimbPitchCore};

#[test]
fn neighbors_produce_ratio_candidates() {
    let mut core = PitchHillClimbPitchCore::new(200.0, 440.0, 0.1, 0.1, 0.0, 0.5);
    let candidates = core.propose_freqs_hz_with_neighbors(440.0, &[440.0], 16, 8, 12.0);
    let has_fifth = candidates.iter().any(|&f| (f - 660.0_f32).abs() < 1.0_f32);
    let has_third = candidates.iter().any(|&f| (f - 550.0_f32).abs() < 1.0_f32);
    assert!(has_fifth || has_third, "expected ratio-derived candidate");
}

#[test]
fn fallback_generates_min_candidates_without_tet_steps() {
    let mut core = PitchHillClimbPitchCore::new(200.0, 440.0, 0.1, 0.1, 0.0, 0.5);
    let candidates = core.propose_freqs_hz_with_neighbors(440.0, &[], 12, 8, 12.0);
    assert!(candidates.len() >= 8);
}

#[test]
fn dedupe_is_deterministic() {
    let mut core = PitchHillClimbPitchCore::new(200.0, 440.0, 0.1, 0.1, 0.0, 0.5);
    let c1 = core.propose_freqs_hz_with_neighbors(440.0, &[440.0, 441.0], 16, 8, 1.0);
    let c2 = core.propose_freqs_hz_with_neighbors(440.0, &[440.0, 441.0], 16, 8, 1.0);
    assert_eq!(c1, c2);
}

#[test]
fn folded_candidates_stay_near_base_octave() {
    let mut core = PitchHillClimbPitchCore::new(200.0, 440.0, 0.1, 0.1, 0.0, 0.5);
    let base = 440.0_f32;
    let candidates = core.propose_freqs_hz_with_neighbors(base, &[220.0, 880.0], 24, 8, 12.0);
    for f in candidates {
        assert!(f >= base * 0.5 && f <= base * 2.0, "freq {f} out of range");
    }
}
