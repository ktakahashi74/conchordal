// Rhai version of sample_scenario.json5

// t = 0.0
add_agent("lead", 330.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5 });
add_agent("pad", 440.0, 0.4, #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.15, envelope: #{ attack_sec: 0.05, decay_sec: 0.1, sustain_level: 0.2 } });

wait(0.6);
set_freq("lead", 550.0);
wait(0.4);
set_amp("lead", 0.1);
wait(0.2); // reach ~1.2s
wait(0.8); // reach ~2.0s
set_freq("pad", 660.0);
wait(0.6); // reach ~2.6s
remove("lead");

wait(2.4); // reach ~5.0s
finish();

// Repeat burst at 8s (will not execute if finish stops earlier)
wait(3.0); // to 8.0s
for i in 0..10 {
    spawn_agents("burst", #{ mode: "harmonicity", min_freq: 400.0, max_freq: 1200.0 }, #{ type: "decay", initial_energy: 0.8, half_life_sec: 0.3 }, 2, 0.15);
    wait(0.15);
}
