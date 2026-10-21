// Closed-loop assay: two Flora share an overlapping extent through sound.
seed(41003);
let first = place(flora().amp(0.03).fade(2),
    random(110, 880));
let second = place(flora().amp(0.03).fade(2),
    random(220, 1760));
wait(32);
release(first);
release(second);
wait(2);
