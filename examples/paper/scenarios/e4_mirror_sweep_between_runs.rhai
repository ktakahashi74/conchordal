// E4 demo: between-run sweep with population reset per mirror weight.
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

scene("E4 Mirror Sweep Between Runs", || {
    let n_probes = 8;
    let mirror_warmup_sec = 0.2;
    let settle_sec = 1.8;
    let gap_sec = 0.6;
    let weights = [0.0, 0.25, 0.5, 0.75, 1.0];

    for w in weights {
        set_harmonicity_mirror_weight(w);
        flush();
        wait(mirror_warmup_sec);

        let base = create(anchor, 1).freq(196.0);
        let probes = create(probe, n_probes)
            .place(consonance(196.0).range(0.8, 2.5).min_dist(0.9))
            .mode("free");
        flush();
        wait(settle_sec);

        release(probes);
        release(base);
        flush();
        wait(gap_sec);
    }
});
