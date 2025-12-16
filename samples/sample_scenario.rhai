// Minimal sample scenario for tests
add_agent("seed", 440.0, 0.1, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.01 });
wait(0.5);
finish();
