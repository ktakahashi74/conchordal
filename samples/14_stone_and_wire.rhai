// Sample 14 — Stone and Wire
// Two turning stones mark the ground; a wire crosses and falls silent.

output_gain(7.5);

temporal_basin(1.8, 2.2);
meter_stability(0.65);

let stone = harmonic()
    .brain("drone")
    .metric()
    .cycles(1)
    .seek_consonance()
    .amp(0.045)
    .brightness(0.35)
    .temperature(0.12)
    .measure_accent(0.8)
    .leave_self_out(true)
    .adsr(0.015, 0.08, 0.4, 0.12);
let wire = harmonic()
    .brain("drone")
    .sustain()
    .seek_consonance()
    .amp(0.045)
    .brightness(0.35)
    .temperature(0.12)
    .proposal_interval(0.75)
    .pitch_apply_mode("rearticulate")
    .leave_self_out(true)
    .adsr(0.015, 0.08, 0.4, 0.12);

section("stone and wire", || {
    let low = place(stone.rhythm_role("accent").proposal_interval(1.5), at(165.0));
    wait(8.0);
    let high = place(stone.rhythm_role("subdivision").proposal_interval(0.75), at(330.0));
    wait(8.0);
    let crossing = place(wire, at(660.0));
    wait(48.0);
    release(crossing);
    wait(4.0);
    release(high);
    wait(8.0);
    release(low);
    wait(2.0);
});
