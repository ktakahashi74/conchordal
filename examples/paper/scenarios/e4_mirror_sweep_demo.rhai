// E4 demo: harmonic-regime phase diagram with fixed population.
let anchor = derive(harmonic)
    .amp(0.09)
    .phonation("hold")
    .timbre(0.82, 0.14)
    .pitch_mode("lock");
let probe = derive(harmonic)
    .amp(0.035)
    .phonation("hold")
    .timbre(0.72, 0.22)
    .pitch_mode("free")
    .pitch_core("peak_sampler");

scene("E4 Mirror Sweep Demo", || {
    let mirror_warmup_sec = 0.4;
    let sample_sec = 6.0;

    let base = create(anchor, 1).freq(196.0);
    flush();
    wait(1.0);

    // Keep population fixed across the sweep to avoid density confounds.
    let probes = create(probe, 12)
        .place(consonance(196.0).range(0.8, 2.5).min_dist(0.9))
        .mode("free");
    flush();
    wait(0.8);

    for w in [0.0, 0.25, 0.5, 0.75, 1.0] {
        print(`E4 mirror_weight=${w}`);
        set_harmonicity_mirror_weight(w);
        flush();
        wait(mirror_warmup_sec);
        wait(sample_sec);
    }

    release(probes);
    release(base);
    wait(0.3);
});
