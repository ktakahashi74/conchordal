// v0.4 flow timing candidate.
// Non-metric onset flow over consonance-biased pitch regions.

seed(20260510);

let root_hz = 146.83;

let field_anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.052)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(8))
    .brightness(0.22)
    .sustain_drive(0.0018)
    .adsr(0.6, 1.4, 0.96, 2.4);

let droplet = modal()
    .amp(0.027)
    .flow()
    .cycles(3)
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .modes(landscape_density_modes().count(6).range(1.0, 5.0).gamma(1.5))
    .brightness(0.74)
    .avoid_neighbors(0.42)
    .adsr(0.024, 0.300, 0.38, 0.72);

let stream = harmonic()
    .amp(0.026)
    .flow()
    // A longer hold makes this layer read as a slow stream under the droplets.
    .cycles(4)
    .seek_consonance()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(5))
    .brightness(0.50)
    .avoid_neighbors(0.55)
    .adsr(0.045, 0.540, 0.62, 1.10);

section("Flow Timing Field", || {
    set_control_update_mode("snapshot_phased");
    harmonic_mirror(0.25);
    // Flow voices barely couple, so no beat forms: the terrain priors are gentle
    // and the texture stays non-metric over consonance-biased placement.
    metric_stability(0.1);
    temporal_basin(1.0, 2.6);

    let root = place(field_anchor, at(root_hz).count(1));
    wait(2.0);

    let low_stream = place(stream, density(80.0, 720.0).spacing(0.80).count(7));
    wait(8.0);

    let high_drops = place(droplet, density(220.0, 1600.0).spacing(0.65).count(10));
    wait(12.0);

    harmonic_mirror(0.80);
    root.freq(root_hz * 1.25);
    low_stream.amp(0.023);
    high_drops.amp(0.030);
    wait(12.0);

    release(high_drops);
    wait(1.0);
    release(low_stream);
    wait(1.2);
    release(root);
    wait(4.0);
});
