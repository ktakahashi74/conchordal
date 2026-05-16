let pad = harmonic()
    .amp(0.13)
    .sustain()
    .adsr(0.2, 0.3, 0.8, 1.0);

let chord = |root, dur| {
    let strat = peaks(root).range(1.0, 3.0).spacing(1.0);
    place(pad, strat.count(4));
    wait(dur);
};

section("Basic Progression", || {
    play(chord, 110.0, 3.0);
    play(chord, 146.83, 3.0);
    play(chord, 196.0, 3.0);
    play(chord, 130.81, 4.0);
});
