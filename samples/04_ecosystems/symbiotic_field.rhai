print(">>> Hybrid Ecosystem: Machine, Plant, and Life <<<");

// 1. Machine (Seq): Strict rhythm using config maps
// No play_tone() helper; use add_agent with brain: "seq"
scene("Phase 1: Mechanism");
let seq_conf = #{ brain: "seq", duration: 0.1 };
parallel(|| {
    for i in 0..8 {
        add_agent("kick", 60.0, 0.6, seq_conf);
        wait(0.5);
    }
});
wait(4.0);

// 2. Plant (Drone): Ambient texture
scene("Phase 2: Vegetation");
let drone_conf = #{ brain: "drone", sway: 0.2 };
// Using "harmonic" body with "drone" brain -> Flexible combination!
let timbre = #{
    mode: "harmonic",
    stiffness: 0.0,
    brightness: 0.5,
    comb: 0.0,
    damping: 0.5,
    vibrato_rate: 0.3,
    vibrato_depth: 0.01,
    jitter: 0.05,
    unison: 0.05
};
add_agent("pad_low", "harmonic", 110.0, 0.3, timbre, drone_conf);
wait(2.0);

// 3. Life (Entrain): Standard autonomous agents
scene("Phase 3: Inhabitants");
// Legacy style config (brain defaults to "entrain")
let bio_life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.6 };
let bio_method = #{ mode: "harmonicity", min_freq: 400.0, max_freq: 800.0 };
spawn_agents("bugs", bio_method, bio_life, 5, 0.2);

wait(5.0);

// 4. Global Control
scene("Phase 4: High Pressure");
set_global_coupling(2.0);
set_roughness_tolerance(0.1);
wait(5.0);

finish();
