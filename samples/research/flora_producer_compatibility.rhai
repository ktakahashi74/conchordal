// Bounded producer compatibility check before the full offline campaigns.
seed(44001);
let growth = place(flora().amp(0.03).fade(2).residual(true), random(110, 1760));
wait(4);
release(growth);
wait(2);
