print(">>> Hybrid Ecosystem: Machine, Plant, and Life <<<");

// 1. Machine (Seq): Strict rhythm using config maps
// No play_tone() helper; use add_agent with temporal core "seq"
scene("Phase 1: Mechanism");
let seq_conf = #{
    body: #{ core: "sine" },
    temporal: #{ core: "seq", duration: 0.1 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};
parallel(|| {
    for i in 0..8 {
        add_agent("kick", 60.0, 0.6, seq_conf);
        wait(0.5);
    }
});
wait(4.0);

// 2. Plant (Drone): Ambient texture
scene("Phase 2: Vegetation");
let drone_conf = #{
    body: #{
        core: "harmonic",
        mode: "harmonic",
        stiffness: 0.0,
        brightness: 0.5,
        comb: 0.0,
        damping: 0.5,
        vibrato_rate: 0.3,
        vibrato_depth: 0.01,
        jitter: 0.05,
        unison: 0.05,
        partials: 16
    },
    temporal: #{ core: "drone", sway: 0.2 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};
// Using harmonic body with drone temporal core.
add_agent("pad_low", 110.0, 0.3, drone_conf);
wait(2.0);

// 3. Life (Entrain): Standard autonomous agents
scene("Phase 3: Inhabitants");
let bio_life = #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.6 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};
let bio_method = #{ mode: "harmonicity", min_freq: 400.0, max_freq: 800.0 };
spawn_agents("bugs", bio_method, bio_life, 5, 0.2);

wait(5.0);

// 4. Global Control
scene("Phase 4: High Pressure");
set_global_coupling(2.0);
set_roughness_tolerance(0.1);
wait(5.0);

finish();
