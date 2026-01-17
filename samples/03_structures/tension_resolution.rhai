seed(15);
let sustain = derive(harmonic)
    .amp(0.2)
    .phonation("hold")
    .adsr(0.2, 0.3, 0.8, 1.0);
let tension = derive(harmonic).amp(0.25).phonation("hold").timbre(0.9, 0.3);

let block = |root, dur| {
    let strat = harmonicity(root).range(1.5, 4.0).min_dist(0.8);
    create(sustain, 3).place(strat);
    wait(dur);
};

scene("Tension Resolution", || {
    play(block, 110.0, 1.5);
    create(tension, 4).place(random_log(165.0, 600.0));
    wait(1.0);
    play(block, 110.0, 1.8);
});
