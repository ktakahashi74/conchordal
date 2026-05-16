let bass = sine().amp(0.1).sustain();
let pad = harmonic().amp(0.05).sustain().brightness(0.7);

let step = |root, dur| {
    place(bass, at(root).count(1));
    let strat = peaks(root).range(2.0, 5.0).spacing(1.0);
    place(pad, strat.count(3));
    wait(dur);
};

section("Autumn Cycle", || {
    play(step, 110.0, 1.6);
    play(step, 146.83, 1.6);
    play(step, 196.0, 1.6);
    play(step, 130.81, 1.6);
    play(step, 174.61, 1.6);
    play(step, 246.94, 1.6);
    play(step, 164.81, 1.6);
    play(step, 110.0, 1.8);
});
