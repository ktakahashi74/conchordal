// v0.4 integrated rhythm/harmony ecology candidate.
// Combines metric beat, entrained beat, flow timing, consonance movement, and life.

seed(20260511);

let root_hz = 110.0;

let field_anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.065)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(8))
    .brightness(0.26)
    .sustain_drive(0.002)
    .adsr(0.6, 1.2, 0.97, 2.6);

let metric_body = harmonic()
    .amp(0.040)
    .metric()
    .rhythm_role("accent")
    .cycles(2)
    .pitch_mode("lock")
    .modes(odd_modes().count(5))
    .brightness(0.42)
    .adsr(0.006, 0.080, 0.45, 0.18);

let entrained_colony = harmonic()
    .amp(0.038)
    .entrained()
    .cycles(3)
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(7))
    .brightness(0.48)
    .avoid_neighbors(0.62)
    .initial_energy(0.55)
    .energy_cap(0.92)
    .metabolism(0.065)
    .action_cost(0.014)
    .viability_rate(0.26)
    .consonance_viability(0.32, 0.82)
    .dissonance_cost(0.45)
    .respawn_consonance()
    .respawn_capacity(10)
    .respawn_min_c_level(0.24)
    .respawn_background_death_rate(0.010)
    .adsr(0.055, 0.390, 0.72, 0.85);

let flow_particles = modal()
    .amp(0.023)
    .flow()
    .cycles(3)
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .modes(landscape_density_modes().count(6).range(1.0, 5.5).gamma(1.4))
    .brightness(0.74)
    .avoid_neighbors(0.45)
    .adsr(0.026, 0.330, 0.40, 0.78);

section("Conchordal Ecology", || {
    set_control_update_mode("snapshot_phased");
    harmonic_mirror(0.10);
    // Metric beat, entrained colony, and flow all share one ~2 Hz emergent
    // attractor; coverage demo, not the musical showcase.
    meter_stability(0.7);
    temporal_basin(1.8, 2.2);

    let root = place(field_anchor, at(root_hz).count(1));
    wait(1.5);

    let beat = place(metric_body, at(root_hz).count(1));
    wait(3.0);

    let colony = place(entrained_colony, density(80.0, 950.0).spacing(0.82).count(8));
    wait(10.0);

    let flow = place(flow_particles, density(180.0, 1500.0).spacing(0.68).count(9));
    wait(10.0);

    harmonic_mirror(0.55);
    root.freq(root_hz * 1.25);
    colony.amp(0.033);
    flow.amp(0.026);
    wait(10.0);

    harmonic_mirror(0.95);
    root.freq(root_hz * 1.5);
    beat.freq(root_hz * 1.5);
    colony.glide(0.20);
    wait(12.0);

    release(flow);
    wait(1.2);
    release(colony);
    wait(1.5);
    release(beat);
    wait(1.0);
    release(root);
    wait(5.0);
});
