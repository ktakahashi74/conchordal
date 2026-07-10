// Seq-brain audition: short authored markers inside a living texture.

seed(20260711);

let terrain = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.045)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(6))
    .brightness(0.24);

let living = harmonic()
    .amp(0.035)
    .entrained()
    .cycles(3)
    .seek_consonance()
    .endurance(8.0)
    .recovery(3.0)
    .attack_cost_fraction(0.02)
    .attack_recharge_fraction(0.50)
    .consonance_viability(0.30, 0.82)
    .dissonance_penalty(0.50)
    .adsr(0.04, 0.20, 0.72, 0.45);

let authored = modal()
    .brain("seq")
    .amp(0.11)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(5))
    .brightness(0.72)
    .adsr(0.01, 0.12, 0.65, 0.18);

section("Seq Authored Events", || {
    place(terrain, at(110.0));
    place(living, consonance(120.0, 880.0).count(5));
    wait(2.0);

    place(authored, at(660.0));
    wait(2.0);
    place(variant(authored).brightness(0.90), at(495.0));
    wait(2.0);
    place(variant(authored).amp(0.14), at(742.5));
    wait(2.0);
});
