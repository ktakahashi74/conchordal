let anchor = sine("sine").amp(0.06).sustain();
let seeker = sine("sine").amp(0.04).sustain();

section("Emergent Harmony", || {
    let base = place(anchor, at(65.406).count(1));
    wait(1.0);

    place(seeker, at(92.499).count(1));
    place(seeker, at(69.296).count(1));
    wait(2.0);

    base.freq(87.307);
    wait(1.5);
});
