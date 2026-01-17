seed(24);
let anchor = derive(sine).amp(0.6).phonation("hold");
let slider = derive(sine).amp(0.4).phonation("hold");
let swarm = derive(sine).amp(0.15).phonation("hold");

scene("Drift Flow", || {
    let a = create(anchor, 1).freq(65.41);
    let s = create(slider, 1).freq(138.59);
    flush();
    wait(1.0);

    s.freq(220.0);
    flush();
    wait(1.5);

    release(s);
    wait(0.5);

    for i in 0..5 {
        let strat = harmonicity(130.0).range(1.0, 4.0).min_dist(1.0);
        create(swarm, 1).place(strat);
        wait(0.6);
    }

    a.freq(87.31);
    flush();
    wait(1.0);
});
