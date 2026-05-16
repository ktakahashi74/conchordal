// Mirror Dualism variant: fixed population, mirror switch only.
let anchor = sine("sine").amp(0.2).sustain();
let voice = sine("sine").amp(0.1).sustain();

section("Mirror Dualism (Fixed Population)", || {
    let mirror_warmup_sec = 0.2;
    let sample_sec = 1.3;

    let base = place(anchor, at(261.63).count(1));
    wait(0.8);

    let probes = place(voice, peaks(261.63).range(0.8, 3.0).spacing(0.9).count(8))
        .pitch_mode("free");
    wait(0.6);

    for w in [0.0, 1.0] {
        harmonic_mirror(w);
        wait(mirror_warmup_sec);
        wait(sample_sec);
    }

    release(probes);
    release(base);
    wait(0.3);
});
