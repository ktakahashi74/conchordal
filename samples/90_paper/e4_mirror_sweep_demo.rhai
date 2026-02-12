// E4 demo: fixed-population step response for mirror-weight sweep.
let anchor = derive(sine).amp(0.1).phonation("hold");
let probe = derive(sine).amp(0.04).phonation("hold");

scene("E4 Mirror Sweep Demo", || {
    let base = create(anchor, 1).freq(196.0);
    flush();
    wait(0.6);

    // Keep population fixed across the sweep to avoid density confounds.
    let probes = create(probe, 8)
        .place(consonance(196.0).range(0.8, 2.5).min_dist(0.9))
        .mode("free");
    flush();
    wait(0.4);

    for w in [0.0, 0.25, 0.5, 0.75, 1.0] {
        set_harmonicity_mirror_weight(w);
        flush();
        wait(1.5);
    }

    release(probes);
    release(base);
    wait(0.3);
});
