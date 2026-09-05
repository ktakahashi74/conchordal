//! Black-box checks for the Log2Space invariants F1/F2 (see AGENTS.md).
//!
//! `Log2Space::assert_scan_len` and its `_named` sibling are hard `assert_eq!`s,
//! so every boundary covered here panics in release as well as in debug.

use conchordal::core::consonance_kernel::{
    ConsonanceKernel, ConsonanceRepresentationParams, compose_consonance_field_level_scan,
};
use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::{Log2Space, sample_scan_linear_log2};
use conchordal::core::psycho_state::{h_pot_scan_to_h_state01_scan, r_pot_scan_to_r_state01_scan};

/// F1: every `_scan` on a Landscape is aligned to the space's bins.
#[test]
fn landscape_scans_match_space_bins() {
    let space = Log2Space::new(55.0, 4000.0, 48);
    let n = space.n_bins();
    let landscape = Landscape::new(space.clone());

    let scans: [(&str, &[f32]); 16] = [
        (
            "consonance_field_score_eff",
            &landscape.consonance_field_score_eff,
        ),
        (
            "consonance_field_level_eff",
            &landscape.consonance_field_level_eff,
        ),
        (
            "consonance_density_mass_eff",
            &landscape.consonance_density_mass_eff,
        ),
        (
            "perc_habituation_state_scan",
            &landscape.perc_habituation_state_scan,
        ),
        ("roughness", &landscape.roughness),
        ("roughness_shape_raw", &landscape.roughness_shape_raw),
        ("roughness01", &landscape.roughness01),
        ("harmonicity", &landscape.harmonicity),
        ("harmonicity01", &landscape.harmonicity01),
        ("consonance_field_score", &landscape.consonance_field_score),
        ("consonance_field_level", &landscape.consonance_field_level),
        (
            "consonance_density_mass",
            &landscape.consonance_density_mass,
        ),
        ("consonance_density", &landscape.consonance_density),
        (
            "consonance_field_energy",
            &landscape.consonance_field_energy,
        ),
        ("subjective_intensity", &landscape.subjective_intensity),
        ("nsgt_power", &landscape.nsgt_power),
    ];

    assert!(n > 0, "space must have bins for this test to mean anything");
    for (name, scan) in scans {
        assert_eq!(scan.len(), n, "{name} is not aligned to Log2Space bins");
    }
}

/// F1: scan-producing helpers write exactly `n_bins` outputs.
#[test]
fn pot_to_state_scans_preserve_length() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    let n = space.n_bins();
    assert!(n > 0);

    let r_pot = vec![0.1f32; n];
    let mut r_state = vec![0.0f32; n];
    r_pot_scan_to_r_state01_scan(&r_pot, 1.0, 0.5, &mut r_state);
    assert_eq!(r_state.len(), n);
    assert!(
        r_state.iter().all(|v| (0.0..=1.0).contains(v)),
        "r_state01 must be bounded in [0,1]"
    );

    let h_pot = vec![0.2f32; n];
    let mut h_state = vec![0.0f32; n];
    h_pot_scan_to_h_state01_scan(&h_pot, 1.0, &mut h_state);
    assert_eq!(h_state.len(), n);
    assert!(
        h_state.iter().all(|v| (0.0..=1.0).contains(v)),
        "h_state01 must be bounded in [0,1]"
    );

    let mut c_level = vec![0.0f32; n];
    let kernel = ConsonanceKernel::default();
    let repr = ConsonanceRepresentationParams::default();
    compose_consonance_field_level_scan(&h_state, &r_state, &kernel, &repr, &mut c_level);
    assert_eq!(c_level.len(), n);
    assert!(
        c_level.iter().all(|v| (0.0..=1.0).contains(v)),
        "consonance_field_level must be bounded in [0,1]"
    );
}

/// F2: the generic Log2Space boundary check itself panics, in any profile.
#[test]
#[should_panic(expected = "scan length mismatch")]
fn assert_scan_len_panics_on_short_scan() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    space.assert_scan_len(&vec![0.0f32; space.n_bins() - 1]);
}

/// F2: the named variant names the offending scan.
#[test]
#[should_panic(expected = "scan length mismatch: probe_scan")]
fn assert_scan_len_named_panics_on_long_scan() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    space.assert_scan_len_named(&vec![0.0f32; space.n_bins() + 1], "probe_scan");
}

/// F2: sampling a misaligned scan panics instead of returning a sentinel, and
/// the panic names the boundary it came from.
#[test]
#[should_panic(expected = "scan length mismatch: sample_scan_linear_log2")]
fn sample_scan_linear_log2_panics_on_misaligned_scan() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    let scan = vec![0.5f32; space.n_bins() / 2];
    let _ = sample_scan_linear_log2(&space, &scan, 220.0);
}

#[test]
#[should_panic(expected = "scan length mismatch: habituation_state_scan")]
fn apply_habituation_panics_on_short_state_scan() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    let h = vec![0.5; space.n_bins() - 1];
    let mut landscape = Landscape::new(space);
    landscape.apply_habituation(&h, 0.0, &ConsonanceRepresentationParams::default());
}

#[test]
#[should_panic(expected = "scan length mismatch: habituation_state_scan")]
fn apply_habituation_panics_on_long_state_scan() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    let h = vec![0.5; space.n_bins() + 1];
    let mut landscape = Landscape::new(space);
    landscape.apply_habituation(&h, 0.0, &ConsonanceRepresentationParams::default());
}

/// Sampling an aligned scan still uses `NEG_INFINITY` as the out-of-space
/// sentinel; only length mismatch was promoted to a panic.
#[test]
fn sample_scan_linear_log2_keeps_out_of_range_sentinel() {
    let space = Log2Space::new(55.0, 4000.0, 24);
    let scan = vec![0.5f32; space.n_bins()];
    assert_eq!(
        sample_scan_linear_log2(&space, &scan, 20.0),
        f32::NEG_INFINITY
    );
    assert_eq!(sample_scan_linear_log2(&space, &scan, 220.0), 0.5);
}

/// F2: boundaries reject mismatched scans. `compose_consonance_field_level_scan`
/// uses a hard `assert_eq!`, so this holds in release too.
#[test]
#[should_panic(expected = "h/r scan length mismatch")]
fn compose_panics_on_input_len_mismatch() {
    let h_state = vec![0.1f32; 4];
    let r_state = vec![0.2f32; 3];
    let mut c_level = vec![0.0f32; 4];
    let kernel = ConsonanceKernel::default();
    let repr = ConsonanceRepresentationParams::default();
    compose_consonance_field_level_scan(&h_state, &r_state, &kernel, &repr, &mut c_level);
}

/// F2: the output buffer is checked as well, not just the two inputs.
#[test]
#[should_panic(expected = "output scan length mismatch")]
fn compose_panics_on_output_len_mismatch() {
    let h_state = vec![0.1f32; 4];
    let r_state = vec![0.2f32; 4];
    let mut c_level = vec![0.0f32; 3];
    let kernel = ConsonanceKernel::default();
    let repr = ConsonanceRepresentationParams::default();
    compose_consonance_field_level_scan(&h_state, &r_state, &kernel, &repr, &mut c_level);
}

/// F2: the r/h state helpers also reject mismatched output buffers.
#[test]
#[should_panic(expected = "r scan length mismatch")]
fn r_state_scan_panics_on_len_mismatch() {
    let r_pot = vec![0.1f32; 4];
    let mut r_state = vec![0.0f32; 3];
    r_pot_scan_to_r_state01_scan(&r_pot, 1.0, 0.5, &mut r_state);
}

#[test]
#[should_panic(expected = "h scan length mismatch")]
fn h_state_scan_panics_on_len_mismatch() {
    let h_pot = vec![0.2f32; 4];
    let mut h_state = vec![0.0f32; 3];
    h_pot_scan_to_h_state01_scan(&h_pot, 1.0, &mut h_state);
}
