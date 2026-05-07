// v0.4 harmony control demo.
// Shows audible field states, then a guide voice snapping to each state.

seed(20260507);

let root_hz = 130.81;
let field_a_hz = root_hz;
let field_b_hz = root_hz * 1.25;
let field_c_hz = root_hz * 1.5;

let anchor = derive(harmonic)
    .brain("drone")
    .field_only()
    .amp(0.060)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(7))
    .brightness(0.22)
    .sustain_drive(0.002)
    .adsr(0.35, 0.9, 0.98, 1.6);

let reference = derive(harmonic)
    .brain("drone")
    .presentation_only()
    .amp(0.024)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(4))
    .brightness(0.16)
    .adsr(0.20, 0.55, 0.84, 1.2);

let marker = derive(harmonic)
    .brain("drone")
    .presentation_only()
    .amp(0.044)
    .sustain()
    .modes(harmonic_modes().count(5))
    .brightness(0.34)
    .adsr(0.16, 0.45, 0.82, 0.9);

let mover = derive(harmonic)
    .brain("drone")
    .presentation_only()
    .amp(0.058)
    .sustain()
    .consonance_movement()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(5))
    .brightness(0.40)
    .crowding(0.18)
    .leave_self_out_harmonics(2)
    .global_peaks(3, 140.0)
    .ratio_candidates(2)
    .adsr(0.24, 0.70, 0.82, 1.3);

scene("Consonance Field Control", || {
    set_control_update_mode("snapshot_phased");
    set_harmonic_mirror(0.0);

    let field = create(anchor, 1).freq(field_a_hz);
    let root = create(reference, 1).freq(field_a_hz);
    wait(0.8);

    let marker_a = create(marker, 2)
        .place(consonance(field_a_hz).range(1.0, 2.0).min_dist(0.95));
    wait(4.0);

    let guide = create(mover, 1)
        .place(consonance_density(140.0, 720.0).min_dist(1.4));
    wait(5.5);

    release(marker_a);
    wait(0.4);
    set_harmonic_mirror(0.35);
    field.freq(field_b_hz);
    root.freq(field_b_hz);
    let marker_b = create(marker, 2)
        .place(consonance(field_b_hz).range(0.9, 2.1).min_dist(0.95));
    wait(6.5);

    release(marker_b);
    wait(0.4);
    set_harmonic_mirror(0.85);
    field.freq(field_c_hz);
    root.freq(field_c_hz);
    let marker_c = create(derive(marker).brightness(0.42), 2)
        .place(consonance(field_c_hz).range(0.8, 1.9).min_dist(0.95));
    wait(6.5);

    release(guide);
    wait(0.6);
    release(marker_c);
    wait(0.6);
    release(root);
    release(field);
    wait(2.0);
});
