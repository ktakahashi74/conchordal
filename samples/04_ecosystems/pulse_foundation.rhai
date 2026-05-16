// v0.4 rhythm foundation sketch.
// Uses pulse articulation, gated duration, vitality coupling, and phase reward.

seed(20260507);

let root_hz = 220.0;
let pulse_hz = 2.0;

let field_anchor = harmonic()
    .brain("drone")
    .send(field)
    .amp(0.055)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.20)
    .sustain_drive(0.0015)
    .adsr(0.4, 0.9, 0.97, 1.8);

let clock_voice = sine()
    .amp(0.030)
    .repeat()
    .pulse(pulse_hz)
    .gates(1)
    .pitch_mode("lock")
    .adsr(0.01, 0.08, 0.55, 0.18);

let follower = harmonic()
    .amp(0.050)
    .repeat()
    .pulse(pulse_hz)
    .gates(3)
    .rhythm_freq(pulse_hz)
    .rhythm_coupling_vitality(0.85, 0.35)
    .rhythm_reward(0.70, "attack_phase_match")
    .seek_consonance()
    .glide(0.32)
    .modes(harmonic_modes().count(6))
    .brightness(0.36)
    .avoid_neighbors(0.40)
    .initial_energy(0.70)
    .energy_cap(0.90)
    .metabolism(0.035)
    .action_cost(0.010)
    .adsr(0.04, 0.14, 0.90, 0.62);

section("Pulse Foundation", || {
    set_control_update_mode("snapshot_phased");
    harmonic_mirror(0.15);

    let drone = place(field_anchor, at(root_hz).count(1));
    wait(2.0);

    let clock = place(clock_voice, at(root_hz * 2.0).count(1));
    wait(4.0);

    let followers = place(follower, density(110.0, 880.0).spacing(0.8).count(9));
    wait(16.0);

    harmonic_mirror(0.65);
    drone.freq(root_hz * 1.5);
    followers.glide(0.24);
    wait(14.0);

    let cross_pulse = place(variant(follower)
            .amp(0.034)
            .pulse(pulse_hz * 1.5)
            .rhythm_freq(pulse_hz * 1.5)
            .gates(2), density(220.0, 1320.0).spacing(0.7).count(5,));
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
