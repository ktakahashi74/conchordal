seed(15);

let root = sine()
    .amp(0.05)
    .sustain()
    .pitch_mode("lock");
let pad = sine()
    .amp(0.04)
    .sustain()
    .pitch_mode("lock");
let tension = sine()
    .amp(0.045)
    .sustain()
    .pitch_mode("lock");
let resolve = sine()
    .amp(0.025)
    .sustain()
    .pitch_mode("lock");

let tension_pair = |dur| {
    place(tension, at(233.08).count(1));
    wait(0.20);
    place(tension, at(311.13).count(1));
    wait(dur - 0.20);
};

let release_mark = |dur| {
    place(resolve, at(220.0).count(1));
    place(resolve, at(329.63).count(1));
    wait(dur);
};


section("Tension Resolution", || {
    place(root, at(110.0).count(1));
    place(pad, at(220.0).count(1));
    place(pad, at(329.63).count(1));
    wait(1.00);
    play(tension_pair, 1.15);
    wait(0.15);
    play(release_mark, 0.75);
    wait(0.85);
});
