let anchor = derive(sine).amp(0.2).sustain();
let voice = derive(sine).amp(0.1).sustain();

scene("Mirror Dualism", || {
    let mirror_warmup_sec = 0.2;

    create(anchor, 1).freq(261.63);
    wait(0.8);

    set_harmonicity_mirror_weight(0.0);
    wait(mirror_warmup_sec);
    for i in 0..4 {
        let strat = consonance(261.63).range(1.0, 3.0).min_dist(0.9);
        create(voice, 1).place(strat);
    }
    wait(1.5);

    set_harmonicity_mirror_weight(1.0);
    wait(mirror_warmup_sec);
    for i in 0..4 {
        let strat = consonance(261.63).range(0.8, 2.5).min_dist(0.9);
        create(voice, 1).place(strat);
    }
    wait(1.5);
});
