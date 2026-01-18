let bass = derive(sine).amp(0.35).phonation("hold");
let pad = derive(harmonic).amp(0.14).phonation("hold").timbre(0.7, 0.2);

let step = |root, dur| {
    create(bass, 1).freq(root);
    let strat = harmonicity(root).range(2.0, 5.0).min_dist(1.0);
    create(pad, 3).place(strat);
    wait(dur);
};

scene("Autumn Cycle", || {
    play(step, 110.0, 1.6);
    play(step, 146.83, 1.6);
    play(step, 196.0, 1.6);
    play(step, 130.81, 1.6);
    play(step, 174.61, 1.6);
    play(step, 246.94, 1.6);
    play(step, 164.81, 1.6);
    play(step, 110.0, 1.8);
});
