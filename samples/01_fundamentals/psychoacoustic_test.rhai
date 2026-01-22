scene("Psychoacoustic", || {
    let seed_voice = derive(sine).amp(0.12).phonation("hold");
    create(seed_voice, 1).freq(330.0);
    flush();
    for i in 0..6 {
        let strat = consonance(330.0).range(0.8, 3.0).min_dist(1.0);
        create(seed_voice, 1).amp(0.08 + i * 0.005).place(strat);
        wait(0.5);
    }
});
