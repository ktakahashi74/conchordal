// Ecological pace from the scene tempo prior; no acoustic beat following.
temporal_basin(1, 1);
// Closed-loop assay: a lone Flora in the middle register.
seed(41001);
let growth = place(flora().amp(0.03).fade(2).residual(true),
    random(110, 1760));
wait(32);
release(growth);
wait(2);
