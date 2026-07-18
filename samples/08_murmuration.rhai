// Sample 8 — Murmuration
// A flock drifts into step. Nothing commands it; coupling and survival do.

let root_hz = 196.0;

let field_anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.060)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(8))
    .brightness(0.28)
    .sustain_drive(0.002)
    .adsr(0.45, 1.0, 0.97, 2.0);

let colony = harmonic()
    .amp(0.040)
    .entrained()
    .cycles(3)
    .seek_consonance()
    .modes(harmonic_modes().count(7))
    .brightness(0.46)
    .avoid_neighbors(0.55)
    .endurance(8.3502)
    .recovery(3.10)
    .attack_cost_fraction(0.0194)
    .attack_recharge_fraction(0.8065)
    .consonance_viability(0.30, 0.82)
    .dissonance_penalty(0.35)
    .adsr(0.055, 0.380, 0.72, 0.85);

let late_colony = variant(colony)
    .amp(0.032)
    .entrained()
    // Locks a touch tighter, so the later cohort tips the flock toward coherence.
    .entrainment(0.65)
    .cycles(2)
    .brightness(0.62)
    .avoid_neighbors(0.70);

section("murmuration", || {
    // A medium attractor: synchronization emerges from coupling and survival,
    // never forced by a deep prior.
    meter_stability(0.45);
    temporal_basin(1.6, 2.0);

    let root = place(field_anchor, at(root_hz).count(1));
    wait(1.3);

    let a = place(colony, consonance(90.0, 780.0).spacing(0.85).count(7));
    wait(8.0);

    let b = place(late_colony, consonance(160.0, 1100.0).spacing(0.75).count(5));
    wait(6.7);

    root.freq(root_hz * 1.5);
    a.amp(0.034);
    b.amp(0.036);
    wait(8.0);

    release(b);
    wait(0.8);
    release(a);
    wait(1.0);
    release(root);
    wait(2.7);
});
