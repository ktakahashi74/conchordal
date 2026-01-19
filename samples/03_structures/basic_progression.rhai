let pad = derive(harmonic)
    .amp(0.25)
    .phonation("hold")
    .adsr(0.2, 0.3, 0.8, 1.0);

let chord = |root, dur| {
    let strat = consonance(root).range(1.0, 3.0).min_dist(1.0);
    create(pad, 4).place(strat);
    wait(dur);
};

scene("Basic Progression", || {
    play(chord, 110.0, 3.0);
    play(chord, 146.83, 3.0);
    play(chord, 196.0, 3.0);
    play(chord, 130.81, 4.0);
});
