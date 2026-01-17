seed(14);
let pad = derive(harmonic)
    .amp(0.2)
    .phonation("hold")
    .adsr(0.2, 0.2, 0.8, 1.0);
let bass = derive(sine).amp(0.36).phonation("hold");

let chord = |root| {
    create(bass, 1).freq(root);
    let strat = harmonicity(root).range(2.0, 5.0).min_dist(1.0);
    create(pad, 3).place(strat);
    wait(1.5);
};

scene("Cadence II V I", || {
    play(chord, 146.83);
    play(chord, 196.00);
    play(chord, 130.81);
});
