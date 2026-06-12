// Étude 1 — A Single Voice
// One voice appears, holds its breath, and leaves.

let voice = sine()
    .amp(0.08)
    .sustain()
    .adsr(0.8, 0.4, 0.85, 2.5);

section("a single voice", || {
    let v = place(voice, at(220.0));
    wait(4.0);
    release(v);
    wait(2.0);
});
