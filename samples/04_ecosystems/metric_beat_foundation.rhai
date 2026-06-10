// v0.4 metric beat candidate.
// A legible pulse that is a DEEP EMERGENT ATTRACTOR: each voice locks its onset
// phase to the shared production meter the population itself drives -- there is
// no externally imposed clock. The tempo region and attractor depth are
// director-level terrain (temporal_basin / metric_stability); a voice only
// chooses how it relates to the emergent beat (lock strength, role, phase).

seed(20260508);

let root_hz = 220.0;

let field_anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.052)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(7))
    .brightness(0.24)
    .sustain_drive(0.0015)
    .adsr(0.35, 0.90, 0.96, 1.6);

let downbeat = harmonic()
    .amp(0.060)
    .metric()
    .rhythm_role("accent")
    .cycles(2)
    .pitch_mode("lock")
    .modes(odd_modes().count(5))
    .brightness(0.36)
    .adsr(0.006, 0.070, 0.42, 0.16);

let consonant_pulse = harmonic()
    .amp(0.045)
    .metric()
    .cycles(3)
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(6))
    .brightness(0.48)
    .avoid_neighbors(0.45)
    .adsr(0.018, 0.120, 0.72, 0.34);

let offbeat = modal()
    .amp(0.030)
    .metric()
    .rhythm_role("subdivision")
    // Half a beat of phase offset reads as a syncopated note between the beats.
    .microtiming(0.5)
    .cycles(2)
    .pitch_mode("lock")
    .modes(landscape_density_modes().count(5).range(1.0, 4.0).gamma(1.2))
    .brightness(0.62)
    .adsr(0.004, 0.060, 0.36, 0.12);

section("Metric Beat Foundation", || {
    set_control_update_mode("snapshot_phased");
    harmonic_mirror(0.15);
    // Deep attractor in a ~2 Hz tempo region: the pulse forms readily and reads
    // as a stable metric heartbeat without any imposed grid.
    metric_stability(0.85);
    temporal_basin(1.8, 2.2);

    let root = place(field_anchor, at(root_hz).count(1));
    wait(1.5);

    let beat = place(downbeat, at(root_hz * 0.5).count(1));
    wait(3.0);

    let voices = place(consonant_pulse, density(120.0, 900.0).spacing(0.85).count(8));
    wait(12.0);

    let cross = place(offbeat, density(260.0, 1320.0).spacing(0.75).count(4));
    wait(9.0);

    harmonic_mirror(0.65);
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
