// Rhai version of spectral_gap.json5
add_agent("seed", 440.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 });
wait(1.0);
for _ in 0..4 {
    spawn(
        "gap",
        #{ mode: "spectral_gap", min_freq: 260.0, max_freq: 1400.0 },
        #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.6 },
        2,
        0.2,
    );
    wait(1.0);
}
finish();
