// Fixed-frequency exposure, withdrawal and re-exposure; not autonomous return.
seed(20260905);

let stimulus = harmonic().brain("drone").amp(0.06).sustain().anchor()
    .modes(harmonic_modes().count(8)).sustain_drive(0.002)
    .adsr(0.01, 0.01, 1.0, 0.02);

section("exposure", || {
    let first = place(stimulus, at(220.0));
    wait(10.0);
    release(first);
});
section("withdrawal", || { wait(16.0); });
section("re-exposure", || {
    let second = place(stimulus, at(220.0));
    wait(10.0);
    release(second);
    wait(0.1);
});
