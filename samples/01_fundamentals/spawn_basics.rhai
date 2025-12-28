// Mixed strategies in one sweep (quick listening tour)
let decay_short = #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.4 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, habituation_sensitivity: 1.0, exploration: 0.0 }
};
let sustain_soft = #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.04 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, habituation_sensitivity: 1.0, exploration: 0.0 }
};

spawn_agents("rnd", #{ mode: "random_log_uniform", min_freq: 200.0, max_freq: 1500.0 }, decay_short, 3, 0.18); wait(0.9);
spawn_agents("harm", #{ mode: "harmonicity", min_freq: 260.0, max_freq: 1200.0 }, sustain_soft, 2, 0.22); wait(0.8);
spawn_agents("gap", #{ mode: "spectral_gap", min_freq: 220.0, max_freq: 1400.0 }, decay_short, 2, 0.18); wait(0.9);
spawn_agents("hdens", #{ mode: "harmonic_density", min_freq: 160.0, max_freq: 1400.0, temperature: 0.7 }, sustain_soft, 3, 0.2); wait(0.8);
spawn_agents("zero", #{ mode: "zero_crossing", min_freq: 180.0, max_freq: 1600.0 }, decay_short, 2, 0.18); wait(0.8);
spawn_agents("lowharm", #{ mode: "low_harmonicity", min_freq: 200.0, max_freq: 1500.0 }, decay_short, 2, 0.18);
wait(1.5);
finish();
