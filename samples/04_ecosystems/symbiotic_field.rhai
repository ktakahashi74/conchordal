print(">>> Hybrid Ecosystem: Machine, Plant, and Life <<<");

// 1. Machine (Seq): Strict rhythm using config maps
scene("Phase 1: Mechanism");
parallel(|| {
    let kick_method = #{ mode: "random_log_uniform", min_freq: 60.0, max_freq: 60.0 };
    let kick_patch = #{
        body: #{ amp: 0.6, method: "sine" },
        phonation: #{ type: "clock", density: 1.0, legato: 0.0 }
    };
    for i in 0..8 {
        spawn("kick", 1, #{ method: kick_method }, kick_patch);
        wait(0.5);
    }
});
wait(4.0);

// 2. Plant (Drone): Ambient texture
scene("Phase 2: Vegetation");
let drone_patch = #{
    body: #{
        amp: 0.3,
        method: "harmonic",
        timbre: #{ brightness: 0.5, inharmonic: 0.0, width: 0.05, motion: 0.05 }
    },
    phonation: #{ type: "hold" }
};
// Using a harmonic body for a drone-like pad.
let pad_method = #{ mode: "random_log_uniform", min_freq: 110.0, max_freq: 110.0 };
spawn("pad_low", 1, #{ method: pad_method }, drone_patch);
wait(2.0);

// 3. Life (Entrain): Standard autonomous agents
scene("Phase 3: Inhabitants");
let bio_patch = #{ body: #{ amp: 0.2, method: "sine" } };
let bio_method = #{ mode: "harmonicity", min_freq: 400.0, max_freq: 800.0 };
spawn("bugs", 5, #{ method: bio_method }, bio_patch);

wait(5.0);

// 4. Global Control
scene("Phase 4: High Pressure");
set_global_coupling(2.0);
set_roughness_tolerance(0.1);
wait(5.0);

end();
