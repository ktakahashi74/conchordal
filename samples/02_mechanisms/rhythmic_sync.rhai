let click = derive(sine)
    .amp(0.2)
    .phonation("decay")
    .adsr(0.01, 0.1, 0.0, 0.2);

parallel([
    || {
        for i in 0..8 {
            create(click, 1).freq(60.0);
            wait(0.5);
        }
    },
    || {
        for i in 0..6 {
            create(click, 1).freq(120.0);
            wait(0.666);
        }
    }
]);
