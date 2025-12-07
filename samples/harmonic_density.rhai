// Harmonic density (probabilistic) spawn demo
add_agent("seed", 440.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 });
wait(1.0);
let method = #{ mode: "harmonic_density", min_freq: 200.0, max_freq: 1200.0, temperature: 0.7 };
let life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.12, envelope: #{ attack_sec: 0.01, decay_sec: 0.25, sustain_level: 0.25 } };
for i in 0..4 {
    spawn_agents("hdens", method, life, 3, 0.2);
    wait(1.0);
}
wait(2.0);
finish();
