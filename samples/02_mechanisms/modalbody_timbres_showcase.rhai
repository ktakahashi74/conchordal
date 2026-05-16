let modal_sustain = modal("modal").sustain();

let table_uniform_aluminum_bar = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(196.0)
    .amp(0.12)
    .brightness(0.58)
    .modes(modal_table("uniform_aluminum_bar").count(6));

let table_uniform_wooden_bar = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(196.0)
    .amp(0.12)
    .brightness(0.42)
    .modes(modal_table("uniform_wooden_bar").count(6));

let table_xylophone = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .brightness(0.70)
    .modes(modal_table("xylophone").count(6));

let table_vibraphone_1 = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .brightness(0.62)
    .modes(modal_table("vibraphone_1").count(5));

let table_vibraphone_2 = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(220.0)
    .amp(0.11)
    .brightness(0.66)
    .modes(modal_table("vibraphone_2").count(5));

let table_wine_glass = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(330.0)
    .amp(0.10)
    .brightness(0.86)
    .modes(modal_table("wine_glass").count(6));

let gen_harmonic = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(110.0)
    .amp(0.10)
    .brightness(0.72)
    .modes(harmonic_modes().count(10));

let gen_odd = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(110.0)
    .amp(0.10)
    .brightness(0.64)
    .modes(odd_modes().count(6));

let gen_power = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(130.81)
    .amp(0.10)
    .brightness(0.68)
    .modes(power_modes(1.35).count(6));

let gen_stiff_string = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(130.81)
    .amp(0.10)
    .brightness(0.60)
    .modes(stiff_string_modes(0.003).count(9));

let gen_custom = variant("modal_sustain", modal_sustain)
    .pitch_mode("lock")
    .freq(146.83)
    .amp(0.10)
    .brightness(0.74)
    .modes(
        custom_modes([1.0, 2.02, 3.11, 4.22, 6.03, 8.91, 12.7])
            .jitter(4.0)
            .seed(42)
    );

let landscape_seed = variant("modal_sustain", modal_sustain)
    .pitch_mode("free")
    .amp(0.045)
    .brightness(0.64)
    .modes(harmonic_modes().count(4));

let landscape_density = variant("modal_sustain", modal_sustain)
    .pitch_mode("free")
    .amp(0.05)
    .brightness(0.70)
    .modes(
        landscape_density_modes()
            .count(16)
            .range(1.0, 3.5)
            .spacing(0.9)
            .gamma(1.15)
            .jitter(4.0)
            .seed(7)
    );

let landscape_peaks = variant("modal_sustain", modal_sustain)
    .pitch_mode("free")
    .amp(0.05)
    .brightness(0.74)
    .modes(
        landscape_peaks_modes()
            .count(14)
            .range(1.0, 3.5)
            .spacing(0.9)
            .jitter(4.0)
            .seed(13)
    );

section("Modal Table: Uniform Aluminum Bar", || {
    let g = place(table_uniform_aluminum_bar, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Table: Uniform Wooden Bar", || {
    let g = place(table_uniform_wooden_bar, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Table: Xylophone", || {
    let g = place(table_xylophone, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Table: Vibraphone 1", || {
    let g = place(table_vibraphone_1, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Table: Vibraphone 2", || {
    let g = place(table_vibraphone_2, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Table: Wine Glass", || {
    let g = place(table_wine_glass, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Gen: Harmonic Modes", || {
    let g = place(gen_harmonic, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Gen: Odd Modes", || {
    let g = place(gen_odd, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Gen: Power Modes", || {
    let g = place(gen_power, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Gen: Stiff String Modes", || {
    let g = place(gen_stiff_string, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Gen: Custom Modes", || {
    let g = place(gen_custom, at(440.0).count(1));
    wait(2.2);
    release(g);
    wait(0.3);
});

section("Modal Landscape: Density Modes", || {
    let bed = place(landscape_seed, density(110.0, 880.0).count(8))
        .pitch_mode("free");
    wait(3.0);

    let g = place(landscape_density, density(110.0, 880.0).count(3))
        .pitch_mode("free");
    wait(2.4);

    release(g);
    release(bed);
    wait(0.4);
});

section("Modal Landscape: Peaks Modes", || {
    let bed = place(landscape_seed, density(110.0, 880.0).count(8))
        .pitch_mode("free");
    wait(3.0);

    let g = place(landscape_peaks, density(110.0, 880.0).count(3))
        .pitch_mode("free");
    wait(2.4);

    release(g);
    release(bed);
});
