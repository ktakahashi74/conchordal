use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::psycho_state::{
    compose_c_statepm1_scan, h_pot_scan_to_h_state01_scan, r_pot_scan_to_r_state01_scan,
};

#[test]
fn landscape_scans_match_space_bins() {
    let space = Log2Space::new(55.0, 4000.0, 48);
    let landscape = Landscape::new(space.clone());

    space.assert_scan_len(&landscape.roughness);
    space.assert_scan_len(&landscape.roughness_shape_raw);
    space.assert_scan_len(&landscape.roughness01);
    space.assert_scan_len(&landscape.harmonicity);
    space.assert_scan_len(&landscape.harmonicity01);
    space.assert_scan_len(&landscape.consonance);
    space.assert_scan_len(&landscape.consonance01);
    space.assert_scan_len(&landscape.subjective_intensity);
    space.assert_scan_len(&landscape.nsgt_power);
}

#[test]
fn pot_state_scan_len_invariants_hold() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    let n = space.n_bins();

    let r_pot = vec![0.1f32; n];
    let mut r_state = vec![0.0f32; n];
    r_pot_scan_to_r_state01_scan(&r_pot, 1.0, 0.5, &mut r_state);

    let h_pot = vec![0.2f32; n];
    let mut h_state = vec![0.0f32; n];
    h_pot_scan_to_h_state01_scan(&h_pot, 1.0, &mut h_state);

    let mut c_state = vec![0.0f32; n];
    compose_c_statepm1_scan(&h_state, &r_state, 0.5, &mut c_state);
}

#[test]
// compose_c_statepm1_scan uses debug_assert for length checks; skip in release where it is elided.
#[cfg(debug_assertions)]
#[should_panic]
fn compose_panics_on_len_mismatch() {
    let h_state = vec![0.1f32; 4];
    let r_state = vec![0.2f32; 3];
    let mut c_state = vec![0.0f32; 4];
    compose_c_statepm1_scan(&h_state, &r_state, 0.5, &mut c_state);
}
