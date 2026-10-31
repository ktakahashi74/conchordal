// V7: continuous shared/external history with explicit population turnover.
endless();
seed(21);
let drone = place(
    sine().brain("drone").sustain().anchor().amp(0.06)
        .adsr(0.02, 0.1, 1.0, 0.2),
    at(196.0)
);
let renewing = place(
    sine().entrained().amp(0.04).cycles(2)
        .respawn_random().respawn_capacity(1).respawn_min_consonance(0.0)
        .respawn_background_death_rate(0.1)
        .adsr(0.01, 0.1, 0.8, 0.2),
    random(220.0, 880.0)
);
