let anchor = sine().amp(0.3).sustain();
let seeker = sine().amp(0.2).sustain();

section("Interaction", || {
    let anchor_group = place(anchor, at(65.41).count(1));
    wait(1.0);

    let seekers = place(seeker, density(90.0, 180.0).count(2));
    wait(2.0);
    release(seekers);
    wait(0.5);
});
