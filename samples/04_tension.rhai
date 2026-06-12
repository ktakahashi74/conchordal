// Étude 4 — Tension
// The director leans on the terrain until it strains,
// then lets the gravity come home.

seed(4);

let ground = harmonic()
    .brain("drone")
    .amp(0.05)
    .sustain()
    .anchor()
    .brightness(0.3);

let chord = harmonic()
    .amp(0.04)
    .sustain()
    .adsr(0.4, 0.6, 0.85, 1.8);

section("tension and release", || {
    harmonic_mirror(0.1);
    place(ground, at(110.0));
    wait(2.0);
    place(chord, peaks(110.0).range(1.0, 4.0).count(5).spacing(0.9));
    wait(6.0);

    harmonic_mirror(0.8);   // the terrain loses its coherence
    wait(8.0);

    harmonic_mirror(0.1);   // release
    wait(8.0);
});
