seed(11);
let anchor = derive(sine).amp(0.6).phonation("hold");
let seeker = derive(sine).amp(0.4).phonation("hold");

scene("Interaction", || {
    let anchor_group = create(anchor, 1).freq(65.41);
    flush();
    wait(1.0);

    let seekers = create(seeker, 2).place(harmonic_density(90.0, 180.0));
    wait(2.0);
    release(seekers);
    wait(0.5);
});
