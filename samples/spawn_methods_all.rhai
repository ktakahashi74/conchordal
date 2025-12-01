// Rhai version of spawn_methods_all.json5

// seed tone
add_agent("seed", 440.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 });

spawn("rand_all", #{ mode: "random_log_uniform", min_freq: 220.0, max_freq: 880.0 }, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5 }, 2, 0.2);
wait(0.8);
spawn("harm_all", #{ mode: "harmonicity", min_freq: 300.0, max_freq: 900.0 }, #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.1, envelope: #{ attack_sec: 0.02, decay_sec: 0.25, sustain_level: 0.3 } }, 2, 0.22);
wait(0.8);
spawn("lowharm_all", #{ mode: "low_harmonicity", min_freq: 250.0, max_freq: 1000.0 }, #{ type: "decay", initial_energy: 0.9, half_life_sec: 0.4 }, 2, 0.2);
wait(0.8);
spawn("hdens_all", #{ mode: "harmonic_density", min_freq: 180.0, max_freq: 1200.0, temperature: 0.6 }, #{ type: "sustain", initial_energy: 1.1, metabolism_rate: 0.12, envelope: #{ attack_sec: 0.01, decay_sec: 0.3, sustain_level: 0.25 } }, 3, 0.18);
wait(0.8);
spawn("zero_all", #{ mode: "zero_crossing", min_freq: 200.0, max_freq: 1500.0 }, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.45 }, 2, 0.2);
wait(0.8);
spawn("gap_all", #{ mode: "spectral_gap", min_freq: 260.0, max_freq: 1400.0 }, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.6 }, 2, 0.2);
wait(2.0);
finish();
