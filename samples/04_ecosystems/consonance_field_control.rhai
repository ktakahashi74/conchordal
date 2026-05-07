// v0.4 harmony control demo.
// Shows field placement, density placement, mirror modulation, and movement.

seed(20260507);

let root_hz = 130.81;
let low_band = 90.0;
let high_band = 1200.0;

let anchor = derive(harmonic)
    .brain("drone")
    .amp(0.045)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(8))
    .brightness(0.28)
    .sustain_drive(0.002)
    .adsr(0.5, 1.1, 0.98, 2.4);

let fixed_voice = derive(harmonic)
    .amp(0.042)
    .sustain()
    .modes(harmonic_modes().count(6))
    .brightness(0.34)
    .adsr(0.35, 0.9, 0.86, 1.8);

let mover = derive(harmonic)
    .amp(0.038)
    .sustain()
    .consonance_movement()
    .movement_glide(0.38)
    .modes(harmonic_modes().count(7))
    .brightness(0.44)
    .crowding(0.45)
    .leave_self_out_harmonics(3)
    .global_peaks(8, 60.0)
    .ratio_candidates(4)
    .adsr(0.45, 1.0, 0.82, 1.9);

scene("Consonance Field Control", || {
    set_control_update_mode("snapshot_phased");
    set_harmonic_mirror(0.0);

    let root = create(anchor, 1).freq(root_hz);
    wait(5.0);

    let field_group = create(fixed_voice, 4)
        .place(consonance(root_hz).range(1.0, 3.0).min_dist(0.9));
    wait(10.0);

    let density_group = create(mover, 5)
        .place(consonance_density(low_band, high_band).min_dist(0.85));
    wait(9.0);

    set_harmonic_mirror(0.55);
    root.freq(root_hz * 1.25);
    density_group.movement_glide(0.28);
    wait(10.0);

    set_harmonic_mirror(1.0);
    let high_group = create(derive(mover).amp(0.032).brightness(0.62), 4)
        .place(consonance_density(150.0, 1500.0).min_dist(0.75));
    wait(9.0);

    release(field_group);
    wait(1.0);
    release(density_group);
    wait(1.0);
    release(high_group);
    wait(1.0);
    release(root);
    wait(4.0);
});
