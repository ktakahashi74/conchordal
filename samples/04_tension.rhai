// Étude 4 — Tension
// The director leans on the terrain until it strains,
// then lets the gravity come home.

seed(4);

let ground = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.05)
    .sustain()
    .anchor()
    .brightness(0.3);

let chord = harmonic()
    .amp(0.04)
    .sustain()
    .seek_consonance()
    .glide(0.5)
    .avoid_neighbors(0.6)
    .adsr(0.4, 0.6, 0.85, 1.8);

section("tension and release", || {
    place(ground, at(110.0));
    wait(1.3);
    place(chord, consonance(110.0).peak().range(1.0, 4.0).count(5).spacing(0.9));
    wait(14.6);
});
