let modal_density = derive(modal)
    .pitch_mode("free")
    .timbre(0.72, 0.20)
    .modes(
        landscape_density_modes()
            .count(20)
            .range(1.0, 3.5)
            .min_dist(0.9)
            .gamma(1.15)
    );

let modal_bar = derive(modal)
    .pitch_mode("lock")
    .freq(220.0)
    .timbre(0.62, 0.12)
    .modes(modal_table("uniform_aluminum_bar").count(6));

let harmonic_odd = derive(harmonic)
    .pitch_mode("lock")
    .freq(110.0)
    .timbre(0.74, 0.0)
    .modes(odd_modes().count(12).range(1.0, 6.0));

scene("Modes Generation Showcase", || {
    let field = create(modal_density, 8)
        .place(consonance_density_pmf(110.0, 880.0).min_dist(0.8))
        .mode("free");
    let bar = create(modal_bar, 1);
    let odd = create(harmonic_odd, 1);

    flush();
    wait(6.0);

    release(field);
    release(bar);
    release(odd);
    flush();
});
