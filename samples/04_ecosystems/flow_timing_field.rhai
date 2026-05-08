// v0.4 flow timing candidate.
// Non-metric onset flow over consonance-biased pitch regions.

seed(20260510);

let root_hz = 146.83;

let field_anchor = derive(harmonic)
    .brain("drone")
    .field_only()
    .amp(0.052)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(8))
    .brightness(0.22)
    .sustain_drive(0.0018)
    .adsr(0.6, 1.4, 0.96, 2.4);

let droplet = derive(modal)
    .amp(0.027)
    .flow_timing(2.6, 0.95)
    .gates(3)
    .consonance_movement()
    .pitch_apply_mode("gate_snap")
    .modes(landscape_density_modes().count(6).range(1.0, 5.0).gamma(1.5))
    .brightness(0.74)
    .crowding(0.42)
    .adsr(0.024, 0.300, 0.38, 0.72);

let stream = derive(harmonic)
    .amp(0.026)
    .flow_timing(1.25, 0.88)
    .gates(4)
    .consonance_movement()
    .pitch_apply_mode("gate_snap")
    .modes(harmonic_modes().count(5))
    .brightness(0.50)
    .crowding(0.55)
    .adsr(0.045, 0.540, 0.62, 1.10);

scene("Flow Timing Field", || {
    set_control_update_mode("snapshot_phased");
    set_harmonic_mirror(0.25);

    let root = create(field_anchor, 1).freq(root_hz);
    wait(2.0);

    let low_stream = create(stream, 7)
        .place(consonance_density(80.0, 720.0).min_dist(0.80));
    wait(8.0);

    let high_drops = create(droplet, 10)
        .place(consonance_density(220.0, 1600.0).min_dist(0.65));
    wait(12.0);

    set_harmonic_mirror(0.80);
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
