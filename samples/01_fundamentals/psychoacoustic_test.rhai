section("Psychoacoustic", || {
    let seed_voice = sine().amp(0.12).sustain();
    place(seed_voice, at(330.0).count(1));
    for i in 0..6 {
        let strat = peaks(330.0).range(0.8, 3.0).spacing(1.0);
        place(seed_voice.amp(0.08 + i * 0.005), strat.count(1));
        wait(0.5);
    }
});
