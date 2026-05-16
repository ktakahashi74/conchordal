let click = sine("sine")
    .amp(0.2)
    .repeat()
    .adsr(0.01, 0.1, 0.0, 0.2);

parallel([
    || {
        for i in 0..8 {
            place(click, at(60.0).count(1));
            wait(0.5);
        }
    },
    || {
        for i in 0..6 {
            place(click, at(120.0).count(1));
            wait(0.666);
        }
    }
]);
