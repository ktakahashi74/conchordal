let kick = derive(sine).amp(0.6).phonation("decay").adsr(0.01, 0.1, 0.0, 0.2);
let drone = derive(harmonic).amp(0.2).phonation("hold").timbre(0.5, 0.05);
let bugs = derive(sine).amp(0.2).phonation("decay");

parallel([
    || {
        for i in 0..8 {
            create(kick, 1).freq(60.0);
            wait(0.5);
        }
    },
    || {
        let pad = create(drone, 1).freq(110.0);
        flush();
        wait(4.0);
        release(pad);
        wait(0.5);
    },
    || {
        for i in 0..6 {
            let strat = harmonicity(400.0).range(1.0, 2.0).min_dist(1.0);
            create(bugs, 2).place(strat);
            wait(0.75);
        }
    }
]);
