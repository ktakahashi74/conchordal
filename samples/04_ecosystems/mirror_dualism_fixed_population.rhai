// Mirror Dualism variant: fixed population, mirror switch only.
let anchor = derive(sine).amp(0.2).phonation("hold");
let voice = derive(sine).amp(0.1).phonation("hold");

scene("Mirror Dualism (Fixed Population)", || {
    let base = create(anchor, 1).freq(261.63);
    flush();
    wait(0.8);

    let probes = create(voice, 8)
        .place(consonance(261.63).range(0.8, 3.0).min_dist(0.9))
        .mode("free");
    flush();
    wait(0.6);

    for w in [0.0, 1.0] {
        set_harmonicity_mirror_weight(w);
        flush();
        wait(1.5);
    }

    release(probes);
    release(base);
    wait(0.3);
});
