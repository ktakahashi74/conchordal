let modal_sustain = derive(modal).sustain();

let table_uniform_aluminum_bar = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(196.0)
    .amp(0.12)
    .brightness(0.58)
    .modes(modal_table("uniform_aluminum_bar").count(6));

let table_uniform_wooden_bar = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(196.0)
    .amp(0.12)
    .brightness(0.42)
    .modes(modal_table("uniform_wooden_bar").count(6));

let table_xylophone = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .brightness(0.70)
    .modes(modal_table("xylophone").count(6));

let table_vibraphone_1 = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .brightness(0.62)
    .modes(modal_table("vibraphone_1").count(5));

let table_vibraphone_2 = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .brightness(0.66)
    .modes(modal_table("vibraphone_2").count(5));

let table_wine_glass = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(330.0)
    .amp(0.10)
    .brightness(0.86)
    .modes(modal_table("wine_glass").count(6));

let gen_harmonic = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(110.0)
    .amp(0.10)
    .brightness(0.72)
    .modes(harmonic_modes().count(10));

let gen_odd = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(110.0)
    .amp(0.10)
    .brightness(0.64)
    .modes(odd_modes().count(6));

let gen_power = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(130.81)
    .amp(0.10)
    .brightness(0.68)
    .modes(power_modes(1.35).count(6));

let gen_stiff_string = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(130.81)
    .amp(0.10)
    .brightness(0.60)
    .modes(stiff_string_modes(0.003).count(9));

let gen_custom = derive(modal_sustain)
    .pitch_mode("lock")
    .freq(146.83)
    .amp(0.10)
    .brightness(0.74)
    .modes(
        custom_modes([1.0, 2.02, 3.11, 4.22, 6.03, 8.91, 12.7])
            .jitter(4.0)
            .seed(42)
    );

let landscape_seed = derive(modal_sustain)
    .pitch_mode("free")
    .amp(0.045)
    .brightness(0.64)
    .modes(harmonic_modes().count(4));

let landscape_density = derive(modal_sustain)
    .pitch_mode("free")
    .amp(0.05)
    .brightness(0.70)
    .modes(
        landscape_density_modes()
            .count(16)
            .range(1.0, 3.5)
            .min_dist(0.9)
            .gamma(1.15)
            .jitter(4.0)
            .seed(7)
    );

let landscape_peaks = derive(modal_sustain)
    .pitch_mode("free")
    .amp(0.05)
    .brightness(0.74)
    .modes(
        landscape_peaks_modes()
            .count(14)
            .range(1.0, 3.5)
            .min_dist(0.9)
            .jitter(4.0)
            .seed(13)
    );

scene("Modal Table: Uniform Aluminum Bar", || {
    let g = create(table_uniform_aluminum_bar, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Table: Uniform Wooden Bar", || {
    let g = create(table_uniform_wooden_bar, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Table: Xylophone", || {
    let g = create(table_xylophone, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Table: Vibraphone 1", || {
    let g = create(table_vibraphone_1, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Table: Vibraphone 2", || {
    let g = create(table_vibraphone_2, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Table: Wine Glass", || {
    let g = create(table_wine_glass, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Gen: Harmonic Modes", || {
    let g = create(gen_harmonic, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Gen: Odd Modes", || {
    let g = create(gen_odd, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Gen: Power Modes", || {
    let g = create(gen_power, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Gen: Stiff String Modes", || {
    let g = create(gen_stiff_string, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Gen: Custom Modes", || {
    let g = create(gen_custom, 1);
    wait(2.2);
    release(g);
    wait(0.3);
});

scene("Modal Landscape: Density Modes", || {
    let bed = create(landscape_seed, 8)
        .place(consonance_density_pmf(110.0, 880.0))
        .mode("free");
    wait(3.0);

    let g = create(landscape_density, 3)
        .place(consonance_density_pmf(110.0, 880.0))
        .mode("free");
    wait(2.4);

    release(g);
    release(bed);
    wait(0.4);
});

scene("Modal Landscape: Peaks Modes", || {
    let bed = create(landscape_seed, 8)
        .place(consonance_density_pmf(110.0, 880.0))
        .mode("free");
    wait(3.0);

    let g = create(landscape_peaks, 3)
        .place(consonance_density_pmf(110.0, 880.0))
        .mode("free");
    wait(2.4);

    release(g);
    release(bed);
});
