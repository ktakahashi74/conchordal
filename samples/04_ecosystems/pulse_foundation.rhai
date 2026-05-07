// v0.4 rhythm foundation sketch.
// Uses pulse articulation, gated duration, vitality coupling, and phase reward.

seed(20260507);

let root_hz = 220.0;
let pulse_hz = 2.0;

let field_anchor = derive(harmonic)
    .brain("drone")
    .field_only()
    .amp(0.055)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.20)
    .sustain_drive(0.0015)
    .adsr(0.4, 0.9, 0.97, 1.8);

let clock_voice = derive(sine)
    .amp(0.030)
    .repeat()
    .pulse(pulse_hz)
    .gates(1)
    .pitch_mode("lock")
    .adsr(0.01, 0.08, 0.55, 0.18);

let follower = derive(harmonic)
    .amp(0.050)
    .repeat()
    .pulse(pulse_hz)
    .gates(3)
    .rhythm_freq(pulse_hz)
    .rhythm_coupling_vitality(0.85, 0.35)
    .rhythm_reward(0.70, "attack_phase_match")
    .consonance_movement()
    .movement_glide(0.32)
    .modes(harmonic_modes().count(6))
    .brightness(0.36)
    .crowding(0.40)
    .initial_energy(0.70)
    .energy_cap(0.90)
    .metabolism(0.035)
    .action_cost(0.010)
    .adsr(0.04, 0.14, 0.90, 0.62);

scene("Pulse Foundation", || {
    set_control_update_mode("snapshot_phased");
    set_harmonic_mirror(0.15);

    let drone = create(field_anchor, 1).freq(root_hz);
    wait(2.0);

    let clock = create(clock_voice, 1).freq(root_hz * 2.0);
    wait(4.0);

    let followers = create(follower, 9)
        .place(consonance_density(110.0, 880.0).min_dist(0.8));
    wait(16.0);

    set_harmonic_mirror(0.65);
    drone.freq(root_hz * 1.5);
    followers.movement_glide(0.24);
    wait(14.0);

    let cross_pulse = create(
        derive(follower)
            .amp(0.034)
            .pulse(pulse_hz * 1.5)
            .rhythm_freq(pulse_hz * 1.5)
            .gates(2),
        5,
    )
        .place(consonance_density(220.0, 1320.0).min_dist(0.7));
    wait(10.0);

    release(cross_pulse);
    wait(1.2);
    release(followers);
    wait(1.4);
    release(clock);
    wait(0.8);
    release(drone);
    wait(3.0);
});
