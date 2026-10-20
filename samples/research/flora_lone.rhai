// Closed-loop assay: a lone Flora in the middle register.
seed(41001);
let growth = place(flora().amp(0.03).rate(0.5).fatigue(6).fade(2),
    random(110, 1760));
wait(32);
release(growth);
wait(2);
