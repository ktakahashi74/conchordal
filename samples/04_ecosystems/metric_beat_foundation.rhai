// v0.4 metric beat candidate.
// Demonstrates a legible beat surface with gate-snap articulation.

seed(20260508);

let root_hz = 220.0;
let beat_hz = 2.0;

let field_anchor = derive(harmonic)
    .brain("drone")
    .field_only()
    .amp(0.052)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(7))
    .brightness(0.24)
    .sustain_drive(0.0015)
    .adsr(0.35, 0.90, 0.96, 1.6);

let downbeat = derive(harmonic)
    .amp(0.060)
    .metric_beat(beat_hz)
    .accent(0.80)
    .gates(2)
    .pitch_mode("lock")
    .modes(odd_modes().count(5))
    .brightness(0.36)
    .adsr(0.006, 0.070, 0.42, 0.16);

let consonant_pulse = derive(harmonic)
    .amp(0.045)
    .metric_beat(beat_hz)
    .accent(0.55)
    .gates(3)
    .consonance_movement()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(6))
    .brightness(0.48)
    .crowding(0.45)
    .adsr(0.018, 0.120, 0.72, 0.34);

let offbeat = derive(modal)
    .amp(0.030)
    .metric_beat(beat_hz * 1.5)
    .accent(0.35)
    .gates(1)
    .pitch_mode("lock")
    .modes(landscape_density_modes().count(5).range(1.0, 4.0).gamma(1.2))
    .brightness(0.62)
    .adsr(0.004, 0.060, 0.36, 0.12);

scene("Metric Beat Foundation", || {
    set_control_update_mode("snapshot_phased");
    set_harmonic_mirror(0.15);

    let root = create(field_anchor, 1).freq(root_hz);
    wait(1.5);

    let beat = create(downbeat, 1).freq(root_hz * 0.5);
    wait(3.0);

    let voices = create(consonant_pulse, 8)
        .place(consonance_density(120.0, 900.0).min_dist(0.85));
    wait(12.0);

    let cross = create(offbeat, 4)
        .place(consonance_density(260.0, 1320.0).min_dist(0.75));
    wait(9.0);

    set_harmonic_mirror(0.65);
    root.freq(root_hz * 1.25);
    voices.amp(0.037);
    wait(10.0);

    release(cross);
    wait(1.0);
    release(voices);
    wait(1.0);
    release(beat);
    wait(0.8);
    release(root);
    wait(3.0);
});
