let grain = derive(sine)
    .amp(0.18)
    .phonation("decay")
    .adsr(0.01, 0.1, 0.0, 0.3);

scene("Spectral Gap", || {
    for i in 0..8 {
        create(grain, 1).place(random_log(120.0, 2200.0));
        wait(0.4);
    }
});
