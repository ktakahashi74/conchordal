// Low harmonicity (dissonant) spawn demo
add_agent("seed", 440.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 });
wait(1.0);
let method = #{ mode: "low_harmonicity", min_freq: 220.0, max_freq: 880.0 };
let life = #{ type: "decay", initial_energy: 1.1, half_life_sec: 0.6 };
for i in 0..4 {
    spawn_agents("lowharm", method, life, 2, 0.2);
    wait(1.0);
}
wait(2.0);
finish();
