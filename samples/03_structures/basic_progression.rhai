seed(9);

let guide = sine()
    .amp(0.18)
    .sustain()
    .send(field_bus);
let pad = harmonic()
    .amp(0.11)
    .sustain()
    .adsr(0.2, 0.3, 0.8, 1.0)
    .pitch_mode("lock");

let chord = |root, dur| {
    let settle = 0.20;
    place(guide, at(root).count(1));
    wait(settle);
    let strat = peaks(root).range(1.0, 3.0).spacing(1.0);
    place(pad, strat.count(4));
    wait(dur - settle);
};

section("Basic Progression", || {
    play(chord, 110.0, 3.0);
    play(chord, 146.83, 3.0);
    play(chord, 196.0, 3.0);
    play(chord, 130.81, 4.0);
});
