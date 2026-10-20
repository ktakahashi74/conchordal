// Held-out mixing evaluation: never use this seed to choose margins.
seed(42102);
let sun = place(harmonic().brain("drone").sustain().anchor().amp(0.06)
    .modes(harmonic_modes().count(16)).send(habitat_bus), at(146.83));
wait(2);
let growth = place(flora().amp(0.03).rate(0.5).fatigue(6).fade(2),
    consonance(146.83, 2349.3));
wait(32);
release(growth);
release(sun);
wait(2);
