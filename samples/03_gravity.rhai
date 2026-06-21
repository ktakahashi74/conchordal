// Étude 3 — Gravity
// The same root under two suns. The peaks listen to what is sounding,
// not to a chart.

seed(3);

let body = harmonic()
    .amp(0.04)
    .sustain()
    .adsr(0.2, 0.3, 0.8, 1.5);

let sun = harmonic()
    .brain("drone")
    .amp(0.06)
    .sustain()
    .anchor()
    .adsr(0.6, 0.5, 0.95, 2.0);

section("a dark sun", || {
    let s = place(variant(sun).brightness(0.12), at(110.0));
    wait(1.3);
    place(body, consonance(110.0).peak().range(1.0, 4.0).count(6).spacing(0.9));
    wait(4.7);
    release(s);
    wait(1.0);
});

section("a brilliant sun", || {
    let s = place(variant(sun).brightness(0.85), at(110.0));
    wait(1.3);
    place(body, consonance(110.0).peak().range(1.0, 4.0).count(6).spacing(0.9));
    wait(4.7);
    release(s);
    wait(1.0);
});
