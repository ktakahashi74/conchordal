print("Harmonicity demo: climb into peaks, then settle");

// Seed: One-shot long decay
add_agent("seed", 330.0, 0.24, #{
    body: #{ core: "sine" },
    articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 2.0, attack_sec: 0.05 },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
});
wait(0.5);

let method = #{ mode: "harmonicity", min_freq: 200.0, max_freq: 1200.0 };
let life = #{
    body: #{ core: "sine" },
    articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.4, attack_sec: 0.02 },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

for i in 0..6 {
    spawn_agents("harm_" + i, method, life, 2, 0.16 + i * 0.01);
    wait(0.7);
}

wait(1.5);
finish();
