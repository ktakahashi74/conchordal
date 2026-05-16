let anchor = sine("sine").amp(0.2).sustain();
let voice = sine("sine").amp(0.1).sustain();

section("Mirror Dualism", || {
    let mirror_warmup_sec = 0.2;

    place(anchor, at(261.63).count(1));
    wait(0.8);

    harmonic_mirror(0.0);
    wait(mirror_warmup_sec);
    for i in 0..4 {
        let strat = peaks(261.63).range(1.0, 3.0).spacing(0.9);
        place(voice, strat.count(1));
    }
    wait(1.5);

    harmonic_mirror(1.0);
    wait(mirror_warmup_sec);
    for i in 0..4 {
        let strat = peaks(261.63).range(0.8, 2.5).spacing(0.9);
        place(voice, strat.count(1));
    }
    wait(1.5);
});
