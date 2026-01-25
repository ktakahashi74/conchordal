// E5 demo (conceptual sketch): rhythmic entrainment flavor with two pulse trains.
let click = derive(sine)
    .amp(0.18)
    .phonation("decay")
    .adsr(0.01, 0.08, 0.0, 0.15);

scene("E5 Rhythmic Entrainment Demo", || {
    parallel([
        || {
            for i in 0..8 {
                create(click, 1).freq(80.0);
                wait(0.5);
            }
        },
        || {
            for i in 0..6 {
                create(click, 1).freq(160.0);
                wait(0.666);
            }
        }
    ]);
});
