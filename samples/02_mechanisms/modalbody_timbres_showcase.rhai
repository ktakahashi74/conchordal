let table_uniform_aluminum_bar = derive(modal)
    .pitch_mode("lock")
    .freq(196.0)
    .amp(0.12)
    .timbre(0.58, 0.08)
    .modes(modal_table("uniform_aluminum_bar").count(6));

let table_uniform_wooden_bar = derive(modal)
    .pitch_mode("lock")
    .freq(196.0)
    .amp(0.12)
    .timbre(0.42, 0.04)
    .modes(modal_table("uniform_wooden_bar").count(6));

let table_xylophone = derive(modal)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .timbre(0.70, 0.10)
    .modes(modal_table("xylophone").count(6));

let table_vibraphone_1 = derive(modal)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .timbre(0.62, 0.14)
    .modes(modal_table("vibraphone_1").count(5));

let table_vibraphone_2 = derive(modal)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .timbre(0.66, 0.16)
    .modes(modal_table("vibraphone_2").count(5));

let table_wine_glass = derive(modal)
    .pitch_mode("lock")
    .freq(330.0)
    .amp(0.10)
    .timbre(0.86, 0.05)
    .modes(modal_table("wine_glass").count(6));

let gen_harmonic = derive(modal)
    .pitch_mode("lock")
    .freq(110.0)
    .amp(0.10)
    .timbre(0.72, 0.06)
    .modes(harmonic_modes().count(14).range(1.0, 10.0));

let gen_odd = derive(modal)
    .pitch_mode("lock")
    .freq(110.0)
    .amp(0.10)
    .timbre(0.64, 0.09)
    .modes(odd_modes().count(10).range(1.0, 12.0));

let gen_power = derive(modal)
    .pitch_mode("lock")
    .freq(130.81)
    .amp(0.10)
    .timbre(0.68, 0.15)
    .modes(power_modes(1.35).count(14).range(1.0, 12.0));

let gen_stiff_string = derive(modal)
    .pitch_mode("lock")
    .freq(130.81)
    .amp(0.10)
    .timbre(0.60, 0.18)
    .modes(stiff_string_modes(0.003).count(14).range(1.0, 11.0));

let gen_custom = derive(modal)
    .pitch_mode("lock")
    .freq(146.83)
    .amp(0.10)
    .timbre(0.74, 0.11)
    .modes(
        custom_modes([1.0, 2.02, 3.11, 4.22, 6.03, 8.91, 12.7])
            .jitter(4.0)
            .seed(42)
    );

let landscape_seed = derive(modal)
    .pitch_mode("free")
    .amp(0.045)
    .timbre(0.64, 0.08)
    .modes(harmonic_modes().count(16).range(1.0, 4.0));

let landscape_density = derive(modal)
    .pitch_mode("free")
    .amp(0.05)
    .timbre(0.70, 0.16)
    .modes(
        landscape_density_modes()
            .count(16)
            .range(1.0, 3.5)
            .min_dist(0.9)
            .gamma(1.15)
            .jitter(4.0)
            .seed(7)
    );

let landscape_peaks = derive(modal)
    .pitch_mode("free")
    .amp(0.05)
    .timbre(0.74, 0.14)
    .modes(
        landscape_peaks_modes()
            .count(14)
            .range(1.0, 3.5)
            .min_dist(0.9)
            .jitter(4.0)
            .seed(13)
    );

scene("ModalBody Timbres Showcase", || {
    let g1 = create(table_uniform_aluminum_bar, 1);
    wait(1.4);
    release(g1);

    let g2 = create(table_uniform_wooden_bar, 1);
    wait(1.4);
    release(g2);

    let g3 = create(table_xylophone, 1);
    wait(1.4);
    release(g3);

    let g4 = create(table_vibraphone_1, 1);
    wait(1.4);
    release(g4);

    let g5 = create(table_vibraphone_2, 1);
    wait(1.4);
    release(g5);

    let g6 = create(table_wine_glass, 1);
    wait(1.4);
    release(g6);

    let g7 = create(gen_harmonic, 1);
    wait(1.4);
    release(g7);

    let g8 = create(gen_odd, 1);
    wait(1.4);
    release(g8);

    let g9 = create(gen_power, 1);
    wait(1.4);
    release(g9);

    let g10 = create(gen_stiff_string, 1);
    wait(1.4);
    release(g10);

    let g11 = create(gen_custom, 1);
    wait(1.4);
    release(g11);

    let bed = create(landscape_seed, 8)
        .place(consonance_density_pmf(110.0, 880.0))
        .mode("free");
    wait(2.0);

    let g12 = create(landscape_density, 3)
        .place(consonance_density_pmf(110.0, 880.0))
        .mode("free");
    wait(1.8);
    release(g12);

    let g13 = create(landscape_peaks, 3)
        .place(consonance_density_pmf(110.0, 880.0))
        .mode("free");
    wait(1.8);

    release(g13);
    release(bed);
});
