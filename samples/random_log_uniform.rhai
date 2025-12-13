print("Random Log-Uniform: spray then thin out");

let method = #{ mode: "random_log_uniform", min_freq: 160.0, max_freq: 1800.0 };
let life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.45 };

scene("spray");
for i in 0..6 {
    spawn_agents("spray_" + i, method, life, 3, 0.18);
    wait(0.4);
}

scene("fade");
for i in 0..3 {
    spawn_agents("tail_" + i, method, life, 1, 0.14);
    wait(0.6);
}

wait(1.5);
finish();
