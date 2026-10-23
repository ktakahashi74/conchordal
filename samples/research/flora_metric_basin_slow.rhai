// Metric Flora tempo-prior assay: slow.
seed(21);
// A sounding Population shapes the terrain; Flora grows around it.
// The halo can settle into a held spectrum. No beat carrier is used.

output_gain(7.5);
temporal_basin(1, 2);

let sun = harmonic().brain("drone").sustain().anchor().amp(0.05)
    .modes(harmonic_modes().count(8)).adsr(0.6, 0.5, 0.95, 2.0);
let moss = flora().metric().amp(0.035).fade(2);

let population = place(sun, at(110));
wait(2);
let growth = place(moss, consonance(110, 1760));
wait(90);
release(population);
release(growth);
wait(2);
