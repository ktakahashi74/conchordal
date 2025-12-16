print("Zero-Crossing: hovering around the neutral plane");

add_agent("seed", 220.0, 0.2, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.9 });
wait(0.5);

let method = #{ mode: "zero_crossing", min_freq: 160.0, max_freq: 1600.0 };
let life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.55 };

for i in 0..6 {
    spawn_agents("zc_" + i, method, life, 2, 0.18);
    wait(0.65);
}

wait(1.5);
finish();
