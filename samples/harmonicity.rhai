// Harmonicity spawn demo
add_agent("seed", 440.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 });
wait(1.0);
let method = #{ mode: "harmonicity", min_freq: 220.0, max_freq: 880.0 };
let life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.1, envelope: #{ attack_sec: 0.02, decay_sec: 0.3, sustain_level: 0.3 } };
for i in 0..4 {
    spawn_agents("harm", method, life, 2, 0.22);
    wait(1.0);
}
wait(2.0);
finish();
