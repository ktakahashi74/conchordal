// E6-style hereditary adaptation: heredity without selection.
// Live assay with E6b rate ratios and compressed listening-time wall-clock.

seed(20260330);

let e6b_steps_cap = 12000.0;
let e6b_step_sec = 512.0 / 48000.0;
let e6b_full_run_sec = e6b_steps_cap * e6b_step_sec;
let audition_speedup = 1.0;
let scene_runtime_sec = e6b_full_run_sec / audition_speedup;
let intro_sec = 0.5;
let outro_sec = 0.5;
let anchor_attack_sec = 0.040;
let anchor_decay_sec = 0.120;
let anchor_sustain_level = 0.97;
let anchor_release_sec = 0.320;
let voice_attack_sec = 0.018;
let voice_decay_sec = 0.070;
let voice_sustain_level = 0.94;
let voice_release_sec = 0.220;
let voice_initial_energy = 0.10;
let voice_energy_cap = 0.20;

let anchor = derive(harmonic)
    .brain("drone")
    .amp(0.05)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.20)
    .sustain_drive(0.001)
    .pitch_smooth(0.12)
    .adsr(
        anchor_attack_sec,
        anchor_decay_sec,
        anchor_sustain_level,
        anchor_release_sec,
    );

let assay_voice = derive(harmonic)
    .amp(0.070)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.20)
    .sustain_drive(0.001)
    .pitch_smooth(0.12)
    .metabolism(0.12 * audition_speedup)
    .initial_energy(voice_initial_energy)
    .energy_cap(voice_energy_cap)
    .action_cost(0.0)
    .recharge_rate(0.0)
    .dissonance_cost(0.0)
    .adsr(
        voice_attack_sec,
        voice_decay_sec,
        voice_sustain_level,
        voice_release_sec,
    );

let heredity_turnover = derive(assay_voice)
    .continuous_recharge_rate(0.0)
    .respawn_background_death_rate(0.03 * audition_speedup)
    .respawn_peak_bias()
    .respawn_capacity(16);

scene("E6: Heredity Only", || {
    let drone = create(anchor, 1).freq(220.0);
    wait(intro_sec);

    let population = create(heredity_turnover, 16)
        .place(random_log(55.0, 880.0));
    wait(scene_runtime_sec);

    release(population);
    release(drone);
    wait(outro_sec);
});
