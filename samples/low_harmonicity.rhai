// Rhai version of low_harmonicity.json5
add_agent("seed", 440.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.8 });
wait(1.0);
for _ in 0..4 {
    spawn_agents("lowharm", #{ mode: "low_harmonicity", min_freq: 220.0, max_freq: 880.0 }, #{ type: "decay", initial_energy: 1.1, half_life_sec: 0.6 }, 2, 0.2);
    wait(1.0);
}
finish();
