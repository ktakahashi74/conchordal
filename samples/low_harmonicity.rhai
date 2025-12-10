print("Low Harmonicity: deliberate rub");

add_agent("anchor", 330.0, 0.2, #{ type: "decay", initial_energy: 1.0, half_life_sec: 3.0, attack_sec: 0.1 });
wait(0.4);

let method = #{ mode: "low_harmonicity", min_freq: 180.0, max_freq: 1200.0 };
let life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.02 };

for i in 0..6 {
    spawn_agents("rub_" + i, method, life, 2, 0.18);
    wait(0.55);
}

wait(1.5);
finish();
