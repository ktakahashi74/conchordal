seed(22);
let anchor = derive(sine).amp(0.12).phonation("hold");
let seeker = derive(sine).amp(0.08).phonation("hold");

scene("Emergent Harmony", || {
    let base = create(anchor, 1).freq(65.406);
    flush();
    wait(1.0);

    create(seeker, 1).freq(92.499);
    create(seeker, 1).freq(69.296);
    flush();
    wait(2.0);

    base.freq(87.307);
    flush();
    wait(1.5);
});
