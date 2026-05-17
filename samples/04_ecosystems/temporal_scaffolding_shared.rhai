// Temporal scaffolding: SHARED (single global pulse). Musical demo.
// All voices receive the same scaffold phase, so attack_phase_match reward
// aligns the population onto a common temporal grid — expect a steady pulse.

seed(20260418);

let run_sec = 45.0;
let intro_sec = 0.5;
let outro_sec = 0.5;
let scaffold_freq_hz = 2.0;
let voice_cycles = 3;

let anchor_attack_sec = 0.040;
let anchor_decay_sec = 0.120;
let anchor_sustain_level = 0.97;
let anchor_release_sec = 0.320;

let voice_attack_sec = 0.060;
let voice_decay_sec = 0.150;
let voice_sustain_level = 0.96;
let voice_release_sec = 0.900;

let anchor = harmonic()
    .brain("drone")
    .send(field_bus)
    .amp(0.05)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.20)
    .sustain_drive(0.001)
    .adsr(
        anchor_attack_sec,
        anchor_decay_sec,
        anchor_sustain_level,
        anchor_release_sec,
    );

let assay_voice = harmonic()
    .amp(0.060)
    .repeat()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.20)
    .metabolism(0.08)
    .initial_energy(0.60)
    .energy_cap(0.80)
    .action_cost(0.015)
    .viability_rate(0.25)
    .dissonance_cost(0.0)
    .rhythm_freq(scaffold_freq_hz)
    .rhythm_coupling_vitality(0.8, 0.4)
    .rhythm_reward(2.0, "attack_phase_match")
    .pulse(scaffold_freq_hz)
    .cycles(voice_cycles)
    .respawn_consonance()
    .respawn_capacity(12)
    .adsr(
        voice_attack_sec,
        voice_decay_sec,
        voice_sustain_level,
        voice_release_sec,
    );

section("E: Scaffold Shared", || {
    set_scaffold_shared(scaffold_freq_hz);
    let drone = place(anchor, at(220.0).count(1));
    wait(intro_sec);

    let population = place(assay_voice, random(110.0, 880.0).count(12));
    wait(run_sec);

    release(population);
    release(drone);
    wait(outro_sec);
});
