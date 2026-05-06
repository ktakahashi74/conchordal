// v0.4 harmony-first ecology demo.
// The script foregrounds the composer-facing consonance API: anchors shape the
// field, populations move through consonance, viability selects environment-fit
// voices, and mirror changes bend the harmonic terrain.

seed(20260506);

let root_hz = 110.0;
let scaffold_hz = 1.5;
let intro_sec = 8.0;
let phase_sec = 18.0;
let outro_sec = 5.0;
let settle_strategy = consonance_density(70.0, 1100.0).min_dist(0.8);
let shimmer_modes = landscape_density_modes()
    .count(10)
    .range(1.0, 5.5)
    .gamma(1.6)
    .min_dist(0.7);

let anchor = derive(harmonic)
    .brain("drone")
    .amp(0.055)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(8))
    .brightness(0.28)
    .sustain_drive(0.001)
    .adsr(0.8, 1.6, 0.98, 4.0);

let hidden_reference = derive(anchor)
    .amp(0.035)
    .mute();

let seeker = derive(harmonic)
    .amp(0.038)
    .repeat()
    .consonance_movement()
    .movement_glide(0.45)
    .modes(harmonic_modes().count(7))
    .brightness(0.36)
    .crowding(0.55)
    .leave_self_out_harmonics(2)
    .global_peaks(8, 70.0)
    .ratio_candidates(5)
    .metabolism(0.09)
    .initial_energy(0.70)
    .energy_cap(1.00)
    .action_cost(0.012)
    .viability_rate(0.18)
    .consonance_viability(0.32, 0.82)
    .dissonance_cost(0.35)
    .rhythm_freq(scaffold_hz)
    .rhythm_coupling_vitality(0.7, 0.35)
    .rhythm_reward(0.4, "attack_phase_match")
    .pulse(scaffold_hz)
    .gates(3)
    .respawn_consonance()
    .respawn_capacity(14)
    .respawn_settle(settle_strategy)
    .adsr(0.06, 0.22, 0.72, 1.2);

let modal_shimmer = derive(modal)
    .amp(0.026)
    .sustain()
    .consonance_movement()
    .movement_glide(0.90)
    .modes(shimmer_modes)
    .brightness(0.68)
    .crowding(0.35)
    .metabolism(0.04)
    .viability_rate(0.08)
    .consonance_viability(0.25, 0.75)
    .adsr(1.2, 2.5, 0.82, 4.5);

scene("Consonance Ecology", || {
    set_control_update_mode("snapshot_phased");
    set_scaffold_shared(scaffold_hz);
    set_harmonic_mirror(0.0);

    let root = create(anchor, 1).freq(root_hz);
    let reference = create(hidden_reference, 1).freq(root_hz * 1.5);
    wait(intro_sec);

    let colony = create(seeker, 10)
        .place(consonance(root_hz).range(1.0, 4.0).min_dist(0.9));
    wait(phase_sec);

    let shimmer = create(modal_shimmer, 4)
        .place(consonance_density(root_hz * 0.75, root_hz * 8.0).min_dist(0.8));
    wait(phase_sec);

    set_harmonic_mirror(0.45);
    wait(phase_sec);

    set_harmonic_mirror(1.0);
    let second_colony = create(derive(seeker).movement_glide(0.70).brightness(0.48), 6)
        .place(consonance_density(root_hz * 0.6, root_hz * 7.0).min_dist(0.9));
    wait(phase_sec);

    release(second_colony);
    release(shimmer);
    release(colony);
    release(reference);
    release(root);
    wait(outro_sec);
});
