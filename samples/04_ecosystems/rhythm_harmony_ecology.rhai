// v0.4 rhythm/harmony integration candidate.
// Stable groove, entrained agents, and non-metric flow shaped by one field.

seed(20260516);

let root_hz = 130.81;
let beat_hz = 2.0;

let field_anchor = harmonic()
    .brain("drone")
    .send(field)
    .amp(0.034)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(8))
    .brightness(0.22)
    .sustain_drive(0.0011)
    .adsr(0.7, 1.4, 0.96, 2.4);

let pulse_ground = harmonic()
    .brain("drone")
    .amp(0.052)
    .metric_beat(beat_hz)
    .accent(0.78)
    .gates(2)
    .pitch_mode("lock")
    .modes(odd_modes().count(5))
    .brightness(0.36)
    .adsr(0.006, 0.075, 0.44, 0.16);

let motif = harmonic()
    .brain("drone")
    .amp(0.034)
    .metric_beat(beat_hz)
    .accent(0.45)
    .gates(1)
    .pitch_mode("lock")
    .modes(harmonic_modes().count(5))
    .brightness(0.58)
    .adsr(0.010, 0.110, 0.34, 0.20);

let colony = harmonic()
    .amp(0.030)
    .entrained_beat(beat_hz)
    .gates(3)
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .glide(0.26)
    .modes(harmonic_modes().count(7))
    .brightness(0.48)
    .avoid_neighbors(0.55)
    .initial_energy(0.60)
    .energy_cap(0.92)
    .metabolism(0.052)
    .action_cost(0.011)
    .viability_rate(0.20)
    .consonance_viability(0.32, 0.84)
    .dissonance_cost(0.34)
    .respawn_consonance()
    .respawn_capacity(8)
    .respawn_min_c_level(0.27)
    .respawn_background_death_rate(0.006)
    .adsr(0.055, 0.360, 0.68, 0.78);

let flow_texture = modal()
    .amp(0.014)
    .flow_timing(2.05, 0.82)
    .gates(2)
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .glide(0.34)
    .modes(landscape_density_modes().count(5).range(1.2, 5.2).gamma(1.35))
    .brightness(0.70)
    .avoid_neighbors(0.38)
    .adsr(0.020, 0.260, 0.34, 0.60);

let cadence_mark = harmonic()
    .brain("drone")
    .amp(0.020)
    .metric_beat(beat_hz * 0.5)
    .accent(0.38)
    .gates(3)
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.40)
    .adsr(0.024, 0.220, 0.58, 0.50);

section("Rhythm Harmony Ecology", || {
    set_control_update_mode("snapshot_phased");
    harmonic_mirror(0.18);

    let field = place(field_anchor, at(root_hz));
    wait(1.2);

    let pulse = place(pulse_ground, at(root_hz * 0.5));
    wait(2.0);

    let theme = place(motif, at(root_hz * 1.5));
    wait(4.0);

    let agents = place(colony, density(95.0, 780.0).count(6).spacing(0.88));
    wait(8.0);

    harmonic_mirror(0.42);
    field.freq(root_hz * 1.25);
    theme.freq(root_hz * 1.875);
    agents.amp(0.032);
    wait(5.0);

    let flow = place(flow_texture, density(190.0, 1450.0).count(8).spacing(0.72));
    wait(8.0);

    release(theme);
    wait(1.0);

    harmonic_mirror(0.68);
    field.freq(root_hz * 1.5);
    agents.glide(0.20);
    agents.amp(0.034);
    flow.amp(0.017);
    let upper_theme = place(
        variant(motif).amp(0.028).brightness(0.66),
        at(root_hz * 2.25)
    );
    wait(7.0);

    let cadence = place(cadence_mark, at(root_hz * 1.5).count(2));
    wait(5.0);

    release(flow);
    wait(1.5);

    harmonic_mirror(0.28);
    field.freq(root_hz * 1.125);
    pulse.freq(root_hz * 0.5);
    upper_theme.freq(root_hz * 1.6875);
    agents.amp(0.026);
    wait(6.0);

    release(cadence);
    wait(0.8);
    release(upper_theme);
    wait(0.8);
    release(agents);
    wait(1.2);
    release(pulse);
    wait(0.8);
    release(field);
    wait(3.0);
});
