// Tempo prior for ecological pace; confirmed habitat beats can change the pace.
temporal_basin(1, 1);
// Closed-loop assay: the low register has the longest analysis delay.
seed(41002);
let growth = place(flora().amp(0.03).fade(2).residual(true),
    random(55, 220));
wait(32);
release(growth);
wait(2);
