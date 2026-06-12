// Étude 6 — Bells
// Struck bodies. The last bell lets the field choose its partials.

seed(6);

let bell = modal()
    .amp(0.10)
    .sustain()
    .anchor()
    .adsr(0.005, 0.9, 0.2, 2.4);

section("a bar", || {
    place(
        variant(bell).brightness(0.58).modes(modal_table("uniform_aluminum_bar").count(6)),
        at(196.0)
    );
    wait(2.1);
});

section("a stiff string", || {
    place(
        variant(bell).brightness(0.6).modes(stiff_string_modes(0.004).count(9)),
        at(130.81)
    );
    wait(2.1);
});

section("a glass", || {
    place(
        variant(bell).brightness(0.85).modes(modal_table("wine_glass").count(6)),
        at(330.0)
    );
    wait(2.1);
});

section("a bell of the field", || {
    let ground = place(
        harmonic().brain("drone").send(habitat_bus).amp(0.05).sustain().anchor().brightness(0.26),
        at(110.0)
    );
    wait(1.0);
    place(
        variant(bell)
            .amp(0.07)
            .brightness(0.7)
            .modes(landscape_density_modes().count(10).range(1.0, 5.5).gamma(1.5).spacing(0.7)),
        at(220.0)
    );
    wait(3.3);
    release(ground);
    wait(1.3);
});
