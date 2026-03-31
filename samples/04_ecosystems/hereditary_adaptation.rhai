// E6-style hereditary adaptation assay.
// This is a live simulation, not a fixed snapshot replay.
//
// Four scenes expose the same contrast as the paper assay:
// 1. Heredity + selection
// 2. Random baseline + selection
// 3. Heredity without selection
// 4. Random baseline without selection
//
// Voices keep a fixed pitch within each life. Adaptation happens across
// generations through respawn. `respawn_peak_bias()` applies parent-biased
// scene-peak proposals with a short local search, while `respawn_random()`
// keeps the same scene and survival regime but removes lineage bias.
//
// Timing note:
// The paper E6b run spans up to 12_000 steps at 512 / 48_000 s per step,
// or about 128 s of simulation time. For listening, this live assay keeps
// the same E6b rate ratios but compresses wall-clock by a fixed factor.

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

// Keep the live assay's respawn loop, but align the life constants and
// harmonic envelope to the paper E6b defaults. Wall-clock is intentionally
// faster than paper time by `audition_speedup`.
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

let heredity_selection = derive(assay_voice)
    .continuous_recharge_rate(0.10 * audition_speedup)
    .survival_signal(0.30, 0.80)
    .respawn_peak_bias()
    .respawn_capacity(16);

let random_selection = derive(assay_voice)
    .continuous_recharge_rate(0.10 * audition_speedup)
    .survival_signal(0.30, 0.80)
    .respawn_random()
    .respawn_capacity(16);

let heredity_turnover = derive(assay_voice)
    .continuous_recharge_rate(0.0)
    .respawn_background_death_rate(0.03 * audition_speedup)
    .respawn_peak_bias()
    .respawn_capacity(16);

let random_turnover = derive(assay_voice)
    .continuous_recharge_rate(0.0)
    .respawn_background_death_rate(0.03 * audition_speedup)
    .respawn_random()
    .respawn_capacity(16);

scene("E6: Heredity + Selection", || {
    let drone = create(anchor, 1).freq(220.0);
    wait(intro_sec);

    let population = create(heredity_selection, 16)
        .place(random_log(55.0, 880.0));
    wait(scene_runtime_sec);

    release(population);
    release(drone);
    wait(outro_sec);
});

scene("E6: Random + Selection", || {
    let drone = create(anchor, 1).freq(220.0);
    wait(intro_sec);

    let population = create(random_selection, 16)
        .place(random_log(55.0, 880.0));
    wait(scene_runtime_sec);

    release(population);
    release(drone);
    wait(outro_sec);
});

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

scene("E6: Random Only", || {
    let drone = create(anchor, 1).freq(220.0);
    wait(intro_sec);

    let population = create(random_turnover, 16)
        .place(random_log(55.0, 880.0));
    wait(scene_runtime_sec);

    release(population);
    release(drone);
    wait(outro_sec);
});
