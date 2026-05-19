seed(15);

let field_tone = harmonic()
    .brain("drone")
    .amp(0.060)
    .send(habitat_bus)
    .sustain()
    .pitch_mode("lock")
    .modes(harmonic_modes().count(5))
    .brightness(0.24)
    .adsr(0.20, 0.60, 0.98, 0.90);

let seeker = sine()
    .amp(0.046)
    .sustain()
    .seek_consonance()
    .glide(0.20)
    .neighbor_step_cents(50.0)
    .move_cost(0.08)
    .improvement_threshold(0.006)
    .exploration(0.08)
    .persistence(0.35)
    .avoid_neighbors(0.08)
    .metabolism(0.08)
    .initial_energy(0.65)
    .energy_cap(0.85)
    .viability_rate(0.22)
    .consonance_viability(0.24, 0.72)
    .dissonance_cost(0.35)
    .adsr(0.12, 0.25, 0.82, 0.80);

let support = variant(seeker)
    .amp(0.024)
    .glide(0.24)
    .global_peaks(6, 45.0)
    .ratio_candidates(3)
    .exploration(0.14)
    .persistence(0.42)
    .avoid_neighbors(0.20);

section("Tension Resolution", || {
    set_control_update_mode("snapshot_phased");

    let v_root = place(field_tone, at(98.00).count(1));
    let v_third = place(field_tone, at(123.47).count(1));
    let v_fifth = place(field_tone, at(146.83).count(1));
    let v_seventh = place(field_tone, at(174.61).count(1));
    wait(0.80);

    let leading_line = place(seeker, density(120.0, 128.0).count(1).spacing(0.20));
    let seventh_line = place(seeker, density(170.0, 180.0).count(1).spacing(0.20));
    let support_colony = place(support, density(90.0, 260.0).count(4).spacing(0.55));
    wait(3.20);

    release(v_root);
    release(v_third);
    release(v_fifth);
    release(v_seventh);

    let i_root = place(field_tone, at(130.81).count(1));
    let i_third = place(field_tone, at(164.81).count(1));
    let i_octave = place(field_tone, at(261.63).count(1));
    let i_high_third = place(field_tone, at(329.63).count(1));
    wait(5.60);

    release(leading_line);
    release(seventh_line);
    release(support_colony);
    wait(1.20);
    release(i_root);
    release(i_third);
    release(i_octave);
    release(i_high_third);
    wait(2.00);
});
