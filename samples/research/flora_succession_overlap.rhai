// Succession assay: overlap.
seed(43008);
let sun = harmonic().brain("drone").sustain().anchor().amp(0.05)
    .modes(harmonic_modes().count(8)).adsr(0.6, 0.5, 0.95, 2.0);
let population = place(sun, at(110));
wait(2);
let first = place(flora().rate(4.0).amp(0.035).fade(2).residual(true), consonance(110, 880));
let second = place(flora().rate(4.0).amp(0.035).fade(2).residual(true), consonance(220, 1760));
wait(86);
release(population);
release(first);
release(second);
wait(4);
