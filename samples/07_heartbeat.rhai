// Étude 7 — Heartbeat
// No clock is imposed. A pulse forms because the population drives one.

seed(20260508);

let root_hz = 220.0;

let field_anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.052)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(7))
    .brightness(0.24)
    .sustain_drive(0.0015)
    .adsr(0.35, 0.90, 0.96, 1.6);

let downbeat = harmonic()
    .amp(0.060)
    .metric()
    .rhythm_role("accent")
    .cycles(2)
    .anchor()
    .modes(odd_modes().count(5))
    .brightness(0.30)
    .adsr(0.010, 0.160, 0.50, 0.30);

let consonant_pulse = harmonic()
    .amp(0.045)
    .metric()
    .cycles(3)
    .seek_consonance()
    .modes(harmonic_modes().count(6))
    .brightness(0.48)
    .avoid_neighbors(0.45)
    .adsr(0.018, 0.120, 0.72, 0.34);

// The syncopation is alive: a colony voice leaning half a beat across the grid.
let offbeat = variant(consonant_pulse)
    .amp(0.030)
    .rhythm_role("subdivision")
    .microtiming(0.5)
    .cycles(2)
    .brightness(0.58)
    .adsr(0.010, 0.100, 0.55, 0.26);

section("heartbeat", || {
    harmonic_mirror(0.15);
    // A deep attractor in a ~2 Hz tempo region: the pulse forms readily.
    meter_stability(0.85);
    temporal_basin(1.8, 2.2);

    let root = place(field_anchor, at(root_hz).count(1));
    wait(1.0);

    let beat = place(downbeat, at(root_hz * 0.5).count(1));
    wait(2.0);

    let voices = place(consonant_pulse, density(120.0, 900.0).spacing(0.85).count(8));
    wait(8.0);

    let cross = place(offbeat, density(260.0, 1320.0).spacing(0.75).count(4));
    wait(6.0);

    harmonic_mirror(0.65);
    root.freq(root_hz * 1.25);
    voices.amp(0.037);
    wait(6.7);

    release(cross);
    wait(0.7);
    release(voices);
    wait(0.7);
    release(beat);
    wait(0.5);
    release(root);
    wait(2.0);
});
