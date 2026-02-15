// E2 demo (conceptual sketch): emergent harmony around an anchor.
let anchor = derive(sine).amp(0.08).phonation("hold");
let seeker = derive(sine)
    .amp(0.04)
    .phonation("hold")
    .pitch_core("peak_sampler");

scene("E2 Emergent Harmony Demo", || {
    let base = create(anchor, 1).freq(110.0);
    flush();
    wait(0.8);

    create(seeker, 1).freq(165.0);
    create(seeker, 1).freq(137.5);
    flush();
    wait(1.5);

    base.freq(220.0);
    flush();
    wait(1.0);
});
