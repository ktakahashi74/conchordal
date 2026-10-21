// Tuning only. Evaluation uses another drone, extent, and disjoint seeds.
seed(41101);
// Habitat-only drone keeps the measurement WAV free of its direct signal.
let sun = place(harmonic().brain("drone").sustain().anchor().amp(0.06)
    .modes(harmonic_modes().count(16)).send(habitat_bus), at(110));
wait(2);
let growth = place(flora().amp(0.03).fade(2),
    consonance(110, 1760));
wait(32);
release(growth);
release(sun);
wait(2);
