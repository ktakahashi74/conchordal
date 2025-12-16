add_agent("seed", 330.0, 0.2, #{ type: "decay", initial_energy: 1.0, half_life_sec: 4.0 });

let fast = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.3, attack_sec: 0.01 };
let med = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.6, attack_sec: 0.02 };

spawn_agents("rand", #{ mode: "random_log_uniform", min_freq: 200.0, max_freq: 1500.0 }, fast, 3, 0.18); wait(0.6);
spawn_agents("harm", #{ mode: "harmonicity", min_freq: 260.0, max_freq: 1200.0 }, med, 2, 0.2); wait(0.6);
spawn_agents("low", #{ mode: "low_harmonicity", min_freq: 220.0, max_freq: 1400.0 }, fast, 2, 0.18); wait(0.6);
spawn_agents("dens", #{ mode: "harmonic_density", min_freq: 180.0, max_freq: 1600.0 }, med, 3, 0.18); wait(0.6);
spawn_agents("zero", #{ mode: "zero_crossing", min_freq: 200.0, max_freq: 1500.0 }, fast, 2, 0.18); wait(0.6);
spawn_agents("gap", #{ mode: "spectral_gap", min_freq: 240.0, max_freq: 1600.0 }, fast, 2, 0.18);
wait(1.5);
finish();
