// Rhai version of spawn_strategies_sample.json5

spawn_agents("rnd", #{ mode: "random_log_uniform", min_freq: 220.0, max_freq: 880.0 }, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.4 }, 3, 0.2);
wait(1.5);
spawn_agents("harm", #{ mode: "harmonicity", min_freq: 330.0, max_freq: 990.0 }, #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.08, envelope: #{ attack_sec: 0.05, decay_sec: 0.2, sustain_level: 0.4 } }, 2, 0.25);
wait(0.8);
spawn_agents("gap", #{ mode: "spectral_gap", min_freq: 260.0, max_freq: 1040.0 }, #{ type: "decay", initial_energy: 0.8, half_life_sec: 0.6 }, 1, 0.18);
wait(1.7);
spawn_agents("hdens", #{ mode: "harmonic_density", min_freq: 150.0, max_freq: 1200.0, temperature: 0.5 }, #{ type: "sustain", initial_energy: 1.2, metabolism_rate: 0.12, envelope: #{ attack_sec: 0.01, decay_sec: 0.25, sustain_level: 0.2 } }, 4, 0.22);
wait(0.5);
spawn_agents("zero", #{ mode: "zero_crossing", min_freq: 200.0, max_freq: 1600.0 }, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.45 }, 2, 0.2);
wait(0.5);
spawn_agents("lowharm", #{ mode: "low_harmonicity", min_freq: 180.0, max_freq: 1440.0 }, #{ type: "decay", initial_energy: 1.1, half_life_sec: 0.35 }, 2, 0.2);
wait(15.0 - (1.5 + 0.8 + 1.7 + 0.5 + 0.5));
finish();
