// Rhai version of harmonic_density.json5
add_agent("seed", 440.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 });
wait(1.0);
for _ in 0..4 {
    spawn(
        "hdens",
        #{ mode: "harmonic_density", min_freq: 200.0, max_freq: 1200.0, temperature: 0.7 },
        #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.12, envelope: #{ attack_sec: 0.01, decay_sec: 0.25, sustain_level: 0.25 } },
        3,
        0.2,
    );
    wait(1.0);
}
finish();
