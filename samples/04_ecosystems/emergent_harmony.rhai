let anchor = derive(sine).amp(0.06).sustain();
let seeker = derive(sine).amp(0.04).sustain();

scene("Emergent Harmony", || {
    let base = create(anchor, 1).freq(65.406);
    wait(1.0);

    create(seeker, 1).freq(92.499);
    create(seeker, 1).freq(69.296);
    wait(2.0);

    base.freq(87.307);
    wait(1.5);
});
