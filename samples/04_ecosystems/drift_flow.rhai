let anchor = sine().amp(0.3).sustain();
let slider = sine().amp(0.2).sustain();
let swarm = sine().amp(0.08).sustain();

section("Drift Flow", || {
    let a = place(anchor, at(65.41).count(1));
    let s = place(slider, at(138.59).count(1));
    wait(1.0);

    s.freq(220.0);
    wait(1.5);

    release(s);
    wait(0.5);

    for i in 0..5 {
        let strat = peaks(130.0).range(1.0, 4.0).spacing(1.0);
        place(swarm, strat.count(1));
        wait(0.6);
    }

    a.freq(87.31);
    wait(1.0);
});
