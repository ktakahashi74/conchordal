// Tempo prior for ecological pace; confirmed habitat beats can change the pace.
temporal_basin(2, 2);
// Succession assay: rate8.
seed(43006);
let sun = harmonic().brain("drone").sustain().anchor().amp(0.05)
    .modes(harmonic_modes().count(8)).adsr(0.6, 0.5, 0.95, 2.0);
let population = place(sun, at(110));
wait(2);
let growth = place(flora().amp(0.035).fade(2).residual(true), consonance(110, 1760));
wait(86);
release(population);
release(growth);
wait(4);
