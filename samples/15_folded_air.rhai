// Sample 15 — Folded Air
// A long thread stays; its companion breathes, joins, and breaks away.

let strand = harmonic()
    .brain("drone")
    .sustain()
    .seek_consonance()
    .amp(0.045)
    .brightness(0.35)
    .temperature(0.12)
    .proposal_interval(2.0)
    .pitch_apply_mode("rearticulate")
    .leave_self_out(true)
    .adsr(0.015, 0.08, 0.4, 0.12);
let breath = harmonic()
    .brain("drone")
    .sustain()
    .entrained()
    .entrainment(0.05)
    .cycles(1)
    .rest(0.5)
    .seek_consonance()
    .amp(0.045)
    .brightness(0.35)
    .temperature(0.12)
    .proposal_interval(0.5)
    .pitch_apply_mode("rearticulate")
    .rearticulate_above(75)
    .articulation_persistence(0.75)
    .glide(0.03)
    .leave_self_out(true)
    .adsr(0.015, 0.08, 0.4, 0.12);

section("folded air", || {
    let long = place(strand, at(220.0));
    wait(8.0);
    let companion = place(breath, at(330.0));
    wait(16.0);
    companion.pitch_apply_mode("glide");
    wait(16.0);
    companion.pitch_apply_mode("gate_snap");
    wait(16.0);
    companion.pitch_apply_mode("rearticulate");
    companion.rearticulate_above(150.0);
    wait(8.0);
    release(companion);
    wait(8.0);
    release(long);
    wait(2.0);
});
