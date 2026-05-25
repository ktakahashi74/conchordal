print("TEST_OK: listener_twin_tension_resolution");

seed(21);

let root_hz = 130.81;

let hidden_reference = harmonic()
    .amp(0.075)
    .send(habitat_bus)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(6))
    .brightness(0.24)
    .adsr(0.25, 0.40, 0.96, 0.80);

let audible_reference = harmonic()
    .amp(0.060)
    .send(presentation_bus)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(5))
    .brightness(0.30)
    .adsr(0.18, 0.35, 0.92, 0.75);

let audible_tension = harmonic()
    .amp(0.050)
    .send(presentation_bus)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(5))
    .brightness(0.34)
    .adsr(0.08, 0.18, 0.84, 0.55);

section("Listener Twin Hidden Habitat", || {
    set_control_update_mode("snapshot_phased");
    place(hidden_reference, at(root_hz).count(1));
    place(hidden_reference, at(root_hz * 1.5).count(1));
    place(hidden_reference, at(root_hz * 2.0).count(1));
    wait(1.40);
});

section("Listener Twin Stable Presentation", || {
    place(audible_reference, at(root_hz).count(1));
    place(audible_reference, at(root_hz * 1.25).count(1));
    place(audible_reference, at(root_hz * 1.5).count(1));
    place(audible_reference, at(root_hz * 2.0).count(1));
    wait(2.20);
});

section("Listener Twin Resolvable Tension", || {
    place(hidden_reference, at(root_hz).count(1));
    place(hidden_reference, at(root_hz * 1.5).count(1));
    place(hidden_reference, at(root_hz * 2.0).count(1));
    wait(0.30);

    let leading = place(audible_tension, at(root_hz * 0.9375).count(1));
    let fourth = place(audible_tension, at(root_hz * 1.3333).count(1));
    let fifth = place(audible_tension, at(root_hz * 1.5000).count(1));
    let ninth = place(audible_tension, at(root_hz * 2.2500).count(1));
    wait(1.15);

    release(leading);
    release(fourth);
    release(fifth);
    release(ninth);
    wait(0.20);

    place(audible_reference, at(root_hz).count(1));
    place(audible_reference, at(root_hz * 1.25).count(1));
    place(audible_reference, at(root_hz * 1.5).count(1));
    place(audible_reference, at(root_hz * 2.0).count(1));
    wait(1.10);
});

section("Listener Twin Settled Presentation", || {
    place(audible_reference, at(root_hz).count(1));
    place(audible_reference, at(root_hz * 1.25).count(1));
    place(audible_reference, at(root_hz * 1.5).count(1));
    place(audible_reference, at(root_hz * 2.0).count(1));
    wait(2.20);
});
