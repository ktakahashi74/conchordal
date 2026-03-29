// Hereditary Adaptation: frequency lineages evolve toward consonance.
//
// A root drone establishes a harmonic landscape. A population of voices
// spawns into that landscape with metabolism and hereditary respawn.
// Voices in consonant regions live longer (higher energy from recharge),
// while dissonant voices die faster (dissonance_cost amplifies basal cost).
// On death, a new voice inherits the parent's frequency with Gaussian
// mutation. Parents are selected proportional to energy (fitness proxy).
// Over generations, the population converges toward harmonically favorable
// positions.

let root = derive(sine)
    .amp(0.12)
    .sustain()
    .metabolism(0.0)
    .pitch_mode("lock");

let organism = derive(sine)
    .amp(0.04)
    .sustain()
    .pitch_mode("free")
    .pitch_core("hill_climb")
    .metabolism(0.4)
    .initial_energy(1.0)
    .continuous_recharge_rate(0.6)
    .dissonance_cost(1.5)
    .respawn_hereditary(0.05)
    .respawn_capacity(3)
    .respawn_min_c_level(0.15);

scene("Hereditary Adaptation", || {
    let drone = create(root, 1).freq(110.0);
    wait(1.0);

    let population = create(organism, 16)
        .place(random_log(90.0, 440.0))
        .pitch_mode("free");
    wait(30.0);

    release(population);
    release(drone);
    wait(1.0);
});
