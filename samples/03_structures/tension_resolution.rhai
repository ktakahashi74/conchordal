let sustain = harmonic("harmonic")
    .amp(0.1)
    .sustain()
    .adsr(0.2, 0.3, 0.8, 1.0);
let tension = harmonic("harmonic").amp(0.13).sustain().brightness(0.9);

let block = |root, dur| {
    let strat = peaks(root).range(1.5, 4.0).spacing(0.8);
    place(sustain, strat.count(3));
    wait(dur);
};

section("Tension Resolution", || {
    play(block, 110.0, 1.5);
    place(tension, random(165.0, 600.0).count(4));
    wait(1.0);
    play(block, 110.0, 1.8);
});
