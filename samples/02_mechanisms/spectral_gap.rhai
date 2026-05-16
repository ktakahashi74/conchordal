let grain = sine("sine")
    .amp(0.09)
    .repeat()
    .adsr(0.01, 0.1, 0.0, 0.3);

section("Spectral Gap", || {
    for i in 0..8 {
        place(grain, random(120.0, 2200.0).count(1));
        wait(0.4);
    }
});
