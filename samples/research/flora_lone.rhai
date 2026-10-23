// Closed-loop assay: a lone Flora in the middle register.
seed(41001);
let growth = place(flora().rate(4.0).amp(0.03).fade(2).residual(true),
    random(110, 1760));
wait(32);
release(growth);
wait(2);
