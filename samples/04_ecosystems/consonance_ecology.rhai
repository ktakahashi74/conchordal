// v0.4 harmony-first ecology demo.
// Designed to make the mechanisms audible: field setup, density seeding,
// active consonance movement, viability pressure, respawn, and mirror shifts.

seed(20260506);

let root_hz = 110.0;
let low_band = 70.0;
let high_band = 1250.0;

let settle_strategy = density(low_band, high_band).spacing(0.75);
let open_density = density(90.0, 900.0).spacing(0.8);
let mirror_density = density(80.0, 1050.0).spacing(0.85);
let high_density = density(140.0, 1500.0).spacing(0.75);
let shimmer_modes = landscape_density_modes()
    .count(9)
    .range(1.0, 5.0)
    .gamma(1.4)
    .spacing(0.65);

let field_anchor = harmonic()
    .brain("drone")
    .amp(0.075)
    .send(habitat_bus)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(8))
    .brightness(0.30)
    .sustain_drive(0.002)
    .adsr(0.4, 1.0, 0.98, 2.8);

let colony_body = harmonic()
    .amp(0.052)
    .sustain()
    .seek_consonance()
    .glide(0.32)
    .modes(harmonic_modes().count(7))
    .brightness(0.42)
    .avoid_neighbors(0.70)
    .leave_self_out_harmonics(3)
    .global_peaks(10, 55.0)
    .ratio_candidates(6);

let colony_life = variant(colony_body)
    .metabolism(0.15)
    .initial_energy(0.48)
    .energy_cap(0.85)
    .action_cost(0.020)
    .viability_rate(0.34)
    .consonance_viability(0.34, 0.84)
    .dissonance_cost(0.80);

let colony_voice = variant(colony_life)
    .respawn_consonance()
    .respawn_capacity(12)
    .respawn_settle(settle_strategy)
    .respawn_min_c_level(0.22)
    .respawn_background_death_rate(0.015)
    .adsr(0.45, 1.40, 0.86, 1.80);

let bright_colony = variant(colony_voice)
    .amp(0.045)
    .glide(0.28)
    .brightness(0.62)
    .avoid_neighbors(0.85)
    .respawn_background_death_rate(0.025);

let quiet_bridge = variant(bright_colony)
    .amp(0.001);

let quiet_final = variant(bright_colony)
    .amp(0.001)
    .glide(0.32);

let shimmer = modal()
    .brain("drone")
    .amp(0.035)
    .sustain()
    .seek_consonance()
    .glide(0.55)
    .modes(shimmer_modes)
    .brightness(0.78)
    .avoid_neighbors(0.45)
    .adsr(0.8, 1.4, 0.78, 2.2);

section("Consonance Ecology", || {
    harmonic_mirror(0.0);

    let root_field = place(field_anchor, at(root_hz));
    wait(1.5);

    let colony_a = place(colony_voice, open_density.count(4));
    wait(1.7);
    let colony_b = place(variant(colony_voice).amp(0.046), open_density.count(3));
    wait(1.9);
    let colony_c = place(
        variant(colony_voice).amp(0.040).glide(0.40),
        open_density.count(3)
    );
    wait(2.4);

    let shimmer_group = place(shimmer, high_density.count(4));
    let bridge = place(quiet_bridge, mirror_density.count(7));
    let final_colony = place(quiet_final, density(95.0, 1000.0).count(6).spacing(0.9));
    shimmer_group.amp(0.004);
    bridge.amp(0.001);
    final_colony.amp(0.001);
    wait(8.0);

    shimmer_group.amp(0.035);
    wait(10.0);

    harmonic_mirror(0.55);
    root_field.freq(root_hz * 1.25);
    bridge.amp(0.045);
    wait(12.0);

    harmonic_mirror(1.0);
    final_colony.amp(0.050);
    colony_a.amp(0.032);
    colony_b.amp(0.030);
    colony_c.amp(0.028);
    wait(5.0);

    final_colony.glide(0.26);
    shimmer_group.amp(0.020);
    wait(14.0);

    release(colony_a);
    wait(0.9);
    release(colony_b);
    wait(1.1);
    release(colony_c);
    wait(1.3);
    release(final_colony);
    wait(1.6);
    release(bridge);
    wait(1.2);
    release(shimmer_group);
    wait(3.0);
    release(root_field);
    wait(8.0);
});
