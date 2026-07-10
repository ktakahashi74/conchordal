// E6-style hereditary adaptation: heredity + selection with approximate LOO selection score.
// Live assay with E6b rate ratios at 1x listening speed.

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
let voice_endurance_sec = 0.10 / (0.12 * audition_speedup);

let anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.05)
    .sustain()
    .anchor()
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

let assay_voice = harmonic()
    .amp(0.070)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(6))
    .brightness(0.20)
    .sustain_drive(0.001)
    .pitch_smooth(0.12)
    .endurance(voice_endurance_sec)
    .attack_cost_fraction(0.0)
    .attack_recharge_fraction(0.0)
    .dissonance_penalty(0.0)
    .adsr(
        voice_attack_sec,
        voice_decay_sec,
        voice_sustain_level,
        voice_release_sec,
    );

let heredity_selection = variant(assay_voice)
    .recovery(voice_endurance_sec)
    .consonance_viability(0.30, 0.80)
    .dissonance_penalty(1.0)
    .selection_approx_loo(true)
    .respawn_consonance()
    .respawn_capacity(16);

section("E6: Heredity + Selection (Approx LOO)", || {
    let drone = place(anchor, at(220.0).count(1));
    wait(intro_sec);

    let population = place(heredity_selection, random(55.0, 880.0).count(16));
    wait(scene_runtime_sec);

    release(population);
    release(drone);
    wait(outro_sec);
});
