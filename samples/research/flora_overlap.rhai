// Tempo prior for ecological pace; confirmed habitat beats can change the pace.
temporal_basin(1, 1);
// Closed-loop assay: two Flora share an overlapping extent through sound.
seed(41003);
let first = place(flora().amp(0.03).fade(2).residual(true),
    random(110, 880));
let second = place(flora().amp(0.03).fade(2).residual(true),
    random(220, 1760));
wait(32);
release(first);
release(second);
wait(2);
