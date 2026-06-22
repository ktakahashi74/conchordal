// Étude 2 — Constellation
// Six ways to enter a field: a line, the peaks, the density,
// the strain, the gaps, and chance.

seed(2);

let star = sine()
    .amp(0.05)
    .sustain()
    .anchor()
    .adsr(0.3, 0.3, 0.8, 1.8);

section("a line", || {
    place(star, line(220.0, 880.0).count(5));
    wait(3.3);
});

section("the peaks", || {
    place(star, at(110.0));
    wait(0.7);
    place(star, consonance(110.0).peak().range(1.0, 4.0).count(5).spacing(0.9));
    wait(4.0);
});

section("the density", || {
    place(star, at(110.0));
    wait(0.7);
    place(star, consonance(110.0, 880.0).count(7).spacing(0.8));
    wait(4.0);
});

section("the strain", || {
    place(star, at(110.0));
    wait(0.7);
    place(star, dissonance(140.0, 880.0).count(5).spacing(0.8));
    wait(4.0);
});

section("the gaps", || {
    place(star, consonance(110.0).peak().range(1.0, 4.0).count(4).spacing(0.9));
    wait(0.7);
    place(star, gap(140.0, 880.0).count(4).spacing(0.8));
    wait(4.0);
});

section("chance", || {
    place(star, random(110.0, 880.0).count(5));
    wait(3.3);
});
