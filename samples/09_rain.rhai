// Étude 9 — Rain
// Time without a beat, falling along the field.

seed(9);

let root_hz = 146.83;

let field_anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.052)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(8))
    .brightness(0.22)
    .sustain_drive(0.0018)
    .adsr(0.6, 1.4, 0.96, 2.4);

let droplet = modal()
    .amp(0.027)
    .flow()
    .cycles(3)
    .seek_consonance()
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
    .modes(harmonic_modes().count(5))
    .brightness(0.50)
    .avoid_neighbors(0.55)
    .adsr(0.045, 0.540, 0.62, 1.10);

section("rain", || {
    // Flow voices barely couple, so no beat forms: the priors stay gentle.
    meter_stability(0.1);
    temporal_basin(1.0, 2.6);

    let root = place(field_anchor, at(root_hz).count(1));
    wait(1.3);

    let low_stream = place(stream, consonance(80.0, 720.0).spacing(0.80).count(7));
    wait(5.3);

    let high_drops = place(droplet, consonance(220.0, 1600.0).spacing(0.65).count(10));
    wait(8.0);

    root.freq(root_hz * 1.25);
    low_stream.amp(0.023);
    high_drops.amp(0.030);
    wait(8.0);

    release(high_drops);
    wait(0.7);
    release(low_stream);
    wait(0.8);
    release(root);
    wait(2.7);
});
