// Étude 5 — Settling
// Scattered by chance, the voices glide to where the field can hold them.

seed(5);

let ground = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.05)
    .sustain()
    .anchor()
    .brightness(0.28);

let seeker = harmonic()
    .amp(0.04)
    .sustain()
    .seek_consonance()
    .glide(0.4)
    .avoid_neighbors(0.6)
    .adsr(0.3, 0.4, 0.85, 1.6);

section("settling", || {
    place(ground, at(110.0));
    wait(1.0);
    place(seeker, random(120.0, 1000.0).count(7));
    wait(10.7);
});
