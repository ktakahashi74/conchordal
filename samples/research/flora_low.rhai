// Closed-loop assay: the low register has the longest analysis delay.
seed(41002);
let growth = place(flora().rate(4.0).amp(0.03).fade(2).residual(true),
    random(55, 220));
wait(32);
release(growth);
wait(2);
