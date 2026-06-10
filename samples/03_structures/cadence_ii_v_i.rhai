seed(14);

let pad = harmonic()
    .amp(0.1)
    .sustain()
    .adsr(0.2, 0.2, 0.8, 1.0)
    .anchor();
let bass = sine().amp(0.18).sustain();

let chord = |root| {
    let settle = 0.20;
    place(bass, at(root).count(1));
    wait(settle);
    let strat = peaks(root).range(2.0, 5.0).spacing(1.0);
    place(pad, strat.count(3));
    wait(1.5 - settle);
};

section("Cadence II V I", || {
    play(chord, 146.83);
    play(chord, 196.00);
    play(chord, 130.81);
});
