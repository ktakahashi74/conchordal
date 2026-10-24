// Tempo prior for ecological pace; confirmed habitat beats can change the pace.
temporal_basin(1, 1);
// Succession assay: population.
seed(43003);
let sun = harmonic().brain("drone").sustain().anchor().amp(0.05).send(habitat_bus)
    .modes(harmonic_modes().count(8)).adsr(0.6, 0.5, 0.95, 2.0);
let seeker = harmonic().amp(0.04).sustain().seek_consonance()
    .glide(0.4).avoid_neighbors(0.6);
let population = place(sun, at(110));
wait(2);
let seekers = place(seeker, random(120.0, 1000.0).count(5));
let growth = place(flora().amp(0.035).fade(2).residual(true), consonance(110, 1760));
wait(86);
release(population);
release(seekers);
release(growth);
wait(4);
