let anchor = derive(sine).amp(0.4).phonation("hold");
let voice = derive(sine).amp(0.2).phonation("hold");

scene("Mirror Dualism", || {
    create(anchor, 1).freq(261.63);
    flush();
    wait(0.8);

    set_harmonicity_mirror_weight(0.0);
    for i in 0..4 {
        let strat = harmonicity(261.63).range(1.0, 3.0).min_dist(0.9);
        create(voice, 1).place(strat);
    }
    wait(1.5);

    set_harmonicity_mirror_weight(1.0);
    for i in 0..4 {
        let strat = harmonicity(261.63).range(0.8, 2.5).min_dist(0.9);
        create(voice, 1).place(strat);
    }
    wait(1.5);
});
