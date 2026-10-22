// Sample 13 — Three Windows
// Three lights cross an unseen room; each leaves by its own door.

output_gain(5.0);

let ground = harmonic()
    .brain("drone")
    .anchor()
    .sustain()
    .amp(0.045)
    .brightness(0.35)
    .send(habitat_bus);
let strand = harmonic()
    .brain("drone")
    .sustain()
    .seek_consonance()
    .amp(0.045)
    .brightness(0.35)
    .temperature(0.12)
    .pitch_apply_mode("rearticulate")
    .glide(0.03)
    .leave_self_out(true)
    .adsr(0.015, 0.08, 0.4, 0.12);

section("three windows", || {
    let floor = place(ground, at(220.0));
    let long = place(strand.proposal_interval(2.0), at(330.0));
    wait(8.0);
    let quick = place(strand.proposal_interval(0.5), at(495.0));
    wait(16.0);
    let middle = place(strand.proposal_interval(1.0), at(440.0));
    wait(32.0);
    release(middle);
    wait(8.0);
    release(quick);
    wait(8.0);
    release(long);
    release(floor);
    wait(2.0);
});
