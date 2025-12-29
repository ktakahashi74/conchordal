print("Spectral Gap: weave through silence");

let method = #{ mode: "spectral_gap", min_freq: 120.0, max_freq: 2200.0 };
let life = #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.32 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

scene("stitch");
for i in 0..24 {
    spawn_agents("gap_" + i, method, life, 1, 0.18);
    wait(0.16);
}

wait(1.2);
finish();
