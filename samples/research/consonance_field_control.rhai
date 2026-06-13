// Harmony control assay.
// Shows audible field states, then a guide voice snapping to each state.

seed(20260507);

let root_hz = 130.81;
let field_a_hz = root_hz;
let field_b_hz = root_hz * 1.25;
let field_c_hz = root_hz * 1.5;

let anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.060)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(7))
    .brightness(0.22)
    .sustain_drive(0.002)
    .adsr(0.35, 0.9, 0.98, 1.6);

let reference = harmonic()
    .brain("drone")
    .send(presentation_bus)
    .amp(0.024)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(4))
    .brightness(0.16)
    .adsr(0.20, 0.55, 0.84, 1.2);

let marker = harmonic()
    .brain("drone")
    .send(presentation_bus)
    .amp(0.044)
    .sustain()
    .modes(harmonic_modes().count(5))
    .brightness(0.34)
    .adsr(0.16, 0.45, 0.82, 0.9);

let mover = harmonic()
    .brain("drone")
    .send(presentation_bus)
    .amp(0.058)
    .sustain()
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(5))
    .brightness(0.40)
    .avoid_neighbors(0.18)
    .leave_self_out_harmonics(2)
    .global_peaks(3, 140.0)
    .ratio_candidates(2)
    .adsr(0.24, 0.70, 0.82, 1.3);

section("Consonance Field Control", || {
    harmonic_mirror(0.0);

    let field = place(anchor, at(field_a_hz).count(1));
    let root = place(reference, at(field_a_hz).count(1));
    wait(0.8);

    let marker_a = place(marker, peaks(field_a_hz).range(1.0, 2.0).spacing(0.95).count(2));
    wait(4.0);

    let guide = place(mover, density(140.0, 720.0).spacing(1.4).count(1));
    wait(5.5);

    release(marker_a);
    wait(0.4);
    harmonic_mirror(0.35);
    field.freq(field_b_hz);
    root.freq(field_b_hz);
    let marker_b = place(marker, peaks(field_b_hz).range(0.9, 2.1).spacing(0.95).count(2));
    wait(6.5);

    release(marker_b);
    wait(0.4);
    harmonic_mirror(0.85);
    field.freq(field_c_hz);
    root.freq(field_c_hz);
    let marker_c = place(variant(marker).brightness(0.42), peaks(field_c_hz).range(0.8, 1.9).spacing(0.95).count(2));
    wait(6.5);

    release(guide);
    wait(0.6);
    release(marker_c);
    wait(0.6);
    release(root);
    release(field);
    wait(2.0);
});
