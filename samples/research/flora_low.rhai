// Closed-loop assay: the low register has the longest analysis delay.
seed(41002);
let growth = place(flora().amp(0.03).fade(2),
    random(55, 220));
wait(32);
release(growth);
wait(2);
