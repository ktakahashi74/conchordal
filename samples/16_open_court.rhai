// Sample 16 — Open Court
// The court fills, two paths cross it, and the crowd leaves in layers.

temporal_basin(1.4, 1.8);
meter_stability(0.65);

let crowd = harmonic()
    .brain("drone")
    .metric()
    .cycles(1)
    .seek_consonance()
    .amp(0.035)
    .brightness(0.35)
    .temperature(0.12)
    .measure_accent(0.8)
    .leave_self_out(true)
    .adsr(0.015, 0.08, 0.4, 0.12);
let path = harmonic()
    .brain("drone")
    .sustain()
    .seek_consonance()
    .amp(0.045)
    .brightness(0.35)
    .temperature(0.12)
    .pitch_apply_mode("rearticulate")
    .leave_self_out(true)
    .adsr(0.015, 0.08, 0.4, 0.12);
let air = harmonic()
    .brain("drone")
    .sustain()
    .entrained()
    .seek_consonance()
    .amp(0.035)
    .brightness(0.35)
    .temperature(0.12)
    .proposal_interval(0.9)
    .pitch_apply_mode("glide")
    .glide(0.08)
    .leave_self_out(true)
    .adsr(0.04, 0.12, 0.4, 0.2);

section("open court", || {
    let near = place(
        crowd.rhythm_role("accent").proposal_interval(1.5),
        line(110.0, 330.0).count(3)
    );
    wait(8.0);
    let far = place(
        crowd.rhythm_role("subdivision").proposal_interval(0.75),
        line(220.0, 660.0).count(3)
    );
    wait(8.0);
    let quick = place(path.proposal_interval(0.5), at(660.0));
    wait(8.0);
    let long = place(path.proposal_interval(2.0), at(495.0));
    wait(8.0);
    let drifting = place(air, at(880.0));
    wait(24.0);
    release(drifting);
    wait(8.0);
    release(long);
    wait(8.0);
    release(quick);
    release(far);
    wait(8.0);
    release(near);
    wait(2.0);
});
