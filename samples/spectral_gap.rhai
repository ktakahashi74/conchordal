print("Spectral Gap: weave through silence");

let method = #{ mode: "spectral_gap", min_freq: 120.0, max_freq: 2200.0 };
let life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.32 };

section("stitch");
for i in 0..24 {
    spawn_agents("gap_" + i, method, life, 1, 0.18);
    wait(0.16);
}

wait(1.2);
finish();
