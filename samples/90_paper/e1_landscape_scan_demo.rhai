// E1 demo: anchor plus ratio sweep for landscape intuition.
let anchor = derive(sine).amp(0.1).phonation("hold");
let sweep = derive(sine).amp(0.05).phonation("hold");

scene("E1 Landscape Scan Demo", || {
    let base = create(anchor, 1).freq(220.0);
    flush();
    wait(0.5);

    let probe = create(sweep, 1).freq(220.0);
    flush();

    for r in [1.0, 1.25, 1.333, 1.5, 2.0, 0.5] {
        probe.freq(220.0 * r);
        flush();
        wait(0.6);
    }

    release(probe);
    wait(0.2);
    release(base);
    wait(0.2);
});
