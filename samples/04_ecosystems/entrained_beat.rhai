// v0.4 entrained beat candidate.
// Uses agent phase, social timing, vitality coupling, and attack reward.

seed(20260509);

let root_hz = 196.0;
let beat_hz = 1.8;

let field_anchor = derive(harmonic)
    .brain("drone")
    .field_only()
    .amp(0.060)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(8))
    .brightness(0.28)
    .sustain_drive(0.002)
    .adsr(0.45, 1.0, 0.97, 2.0);

let colony = derive(harmonic)
    .amp(0.040)
    .entrained_beat(beat_hz)
    .gates(3)
    .consonance_movement()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(7))
    .brightness(0.46)
    .crowding(0.55)
    .initial_energy(0.62)
    .energy_cap(0.92)
    .metabolism(0.055)
    .action_cost(0.012)
    .viability_rate(0.20)
    .consonance_viability(0.30, 0.82)
    .dissonance_cost(0.35)
    .adsr(0.055, 0.380, 0.72, 0.85);

let late_colony = derive(colony)
    .amp(0.032)
    .entrained_beat(beat_hz * 1.25)
    .gates(2)
    .brightness(0.62)
    .crowding(0.70);

scene("Entrained Beat", || {
    set_control_update_mode("snapshot_phased");
    set_harmonic_mirror(0.05);

    let root = create(field_anchor, 1).freq(root_hz);
    wait(2.0);

    let a = create(colony, 7)
        .place(consonance_density(90.0, 780.0).min_dist(0.85));
    wait(12.0);

    let b = create(late_colony, 5)
        .place(consonance_density(160.0, 1100.0).min_dist(0.75));
    wait(10.0);

    set_harmonic_mirror(0.55);
    root.freq(root_hz * 1.5);
    a.amp(0.034);
    b.amp(0.036);
    wait(12.0);

    release(b);
    wait(1.2);
    release(a);
    wait(1.5);
    release(root);
    wait(4.0);
});
