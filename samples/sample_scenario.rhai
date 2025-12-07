// Rhai version of sample_scenario
// Timeline mirrors json5: lead enters at 0, pad at ~1.2, edits, burst loop later.

// lead start
add_agent("lead", 330.0, 0.25, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5 });
wait(0.6);
set_freq("lead", 550.0);
wait(0.4);
set_amp("lead", 0.1); // t = 1.0

// pad entry at t ≈ 1.2
wait(0.2);
add_agent("pad", 440.0, 0.4, #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.15, envelope: #{ attack_sec: 0.05, decay_sec: 0.1, sustain_level: 0.2 } });
wait(0.8);
set_freq("pad", 660.0); // t ≈ 2.0
wait(0.6);
remove("lead"); // t ≈ 2.6

// pause until 5.0s
wait(2.4);
// burst loop starting at 5.0s (instead of mid-run finish)
wait(3.0); // to 8.0s
let burst_method = #{ mode: "harmonicity", min_freq: 400.0, max_freq: 1200.0 };
let burst_life = #{ type: "decay", initial_energy: 0.8, half_life_sec: 0.3 };
for i in 0..10 {
    spawn_agents("burst", burst_method, burst_life, 2, 0.15);
    wait(0.15);
}
wait(2.0);
finish();
