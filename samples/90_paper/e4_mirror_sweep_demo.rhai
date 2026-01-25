// E4 demo: mirror weight sweep with probe placement.
let anchor = derive(sine).amp(0.1).phonation("hold");
let probe = derive(sine).amp(0.04).phonation("hold");

scene("E4 Mirror Sweep Demo", || {
    let base = create(anchor, 1).freq(196.0);
    flush();
    wait(0.6);

    for w in [0.0, 0.25, 0.5, 0.75, 1.0] {
        set_harmonicity_mirror_weight(w);
        let strat = consonance(196.0).range(0.8, 2.5).min_dist(0.9);
        create(probe, 2).place(strat).mode("free");
        wait(0.8);
    }

    release(base);
    wait(0.3);
});
