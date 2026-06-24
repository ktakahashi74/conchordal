// Étude 10 — Generations
// Voices live, starve, and are reborn where harmony can hold them.

seed(10);

let root_hz = 110.0;

let settle_strategy = consonance(70.0, 1250.0).spacing(0.75);
let open_density = consonance(90.0, 900.0).spacing(0.8);

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
    .avoid_neighbors(0.70);

let colony_life = variant(colony_body)
    .metabolism(0.15)
    .initial_energy(0.48)
    .energy_cap(0.85)
    .action_cost(0.020)
    .viability_rate(0.34)
    .consonance_viability(0.34, 0.84)
    .dissonance_cost(0.80);

// The colony regenerates itself: starved voices respawn onto consonant peaks.
let colony_voice = variant(colony_life)
    .respawn_consonance()
    .respawn_capacity(12)
    .respawn_settle(settle_strategy)
    .respawn_min_c_level(0.22)
    .respawn_background_death_rate(0.015)
    .adsr(0.45, 1.40, 0.86, 1.80);

section("generations", || {
    // A consonant ground the ecology senses but never hears.
    let root_field = place(field_anchor, at(root_hz));
    wait(1.0);

    // A population settles onto the landscape.
    let colony_a = place(colony_voice, open_density.count(5));
    wait(1.3);
    let colony_b = place(variant(colony_voice).amp(0.046), open_density.count(4));
    wait(6.0);

    // The colony grows restless: the search heats up, voices stray off
    // consonance onto dissonant ground and starve, and respawn carries the
    // survivors back toward consonant peaks.
    colony_a.temperature(0.8);
    colony_b.temperature(0.8);
    wait(9.0);

    // Calm returns: the search cools and the reborn colony settles where harmony
    // can hold it.
    colony_a.temperature(0.0);
    colony_b.temperature(0.0);
    colony_a.glide(0.26);
    wait(9.0);

    release(colony_a);
    wait(0.8);
    release(colony_b);
    wait(1.1);
    release(root_field);
    wait(3.0);
});
