// I9 development material: 30 minutes with intervening content and repeated phrases.
// Construction labels and timings are test metadata, never observer inputs.
seed(2026091309);

let tone = harmonic()
    .brain("drone")
    .anchor()
    .amp(0.035)
    .sustain()
    .adsr(0.08, 0.15, 0.82, 0.45);

fn motif_a(body) {
    for hz in [220.0, 275.0, 247.5] {
        let v = place(body, at(hz));
        wait(3.5);
        release(v);
        wait(0.5);
    }
}

section("continuous development and returns", || {
    for cell in 0..60 {
        let root = 130.0 + ((cell * 43) % 230).to_float();
        let color = 0.15 + 0.1 * (cell % 6).to_float();
        if cell == 0 || cell == 5 || cell == 27 || cell == 58 {
            motif_a(variant(tone).brightness(0.35));
        } else {
            let v = place(variant(tone).brightness(color), at(root));
            wait(5.0);
            v.freq(root * 1.125);
            wait(5.0);
            release(v);
            wait(2.0);
        }
        let lower = place(variant(tone).brightness(0.2), at(root * 0.75));
        let upper = place(variant(tone).brightness(color), at(root * 1.6));
        wait(6.0);
        upper.freq(root * 1.25);
        wait(6.0);
        release(lower);
        wait(3.0);
        release(upper);
        wait(3.0);
    }
});
