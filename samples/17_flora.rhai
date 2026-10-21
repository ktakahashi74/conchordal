// Sample 17 — Flora
// A sounding Population shapes the terrain; Flora grows around it.
// The halo can settle into a held spectrum. No beat carrier is used.

let sun = harmonic().brain("drone").sustain().anchor().amp(0.05)
    .modes(harmonic_modes().count(8)).adsr(0.6, 0.5, 0.95, 2.0);
let moss = flora().amp(0.035).fade(2);

let population = place(sun, at(110));
wait(2);
let growth = place(moss, consonance(110, 1760));
wait(12);

// Only the Population is retuned. Flora receives the change through sound.
population.freq(146.83);
wait(12);
release(population);
wait(6);
release(growth);
wait(2);
