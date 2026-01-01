// Mixed strategies in one sweep (quick listening tour)
let decay_short = #{
    body: #{ core: "sine" },
    articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.4 },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};
let sustain_soft = #{
    body: #{ core: "sine" },
    articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.04 },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

spawn("rnd", 3, #{ amp: 0.18, method: #{ mode: "random_log_uniform", min_freq: 200.0, max_freq: 1500.0 }, life: decay_short }); wait(0.9);
spawn("harm", 2, #{ amp: 0.22, method: #{ mode: "harmonicity", min_freq: 260.0, max_freq: 1200.0 }, life: sustain_soft }); wait(0.8);
spawn("gap", 2, #{ amp: 0.18, method: #{ mode: "spectral_gap", min_freq: 220.0, max_freq: 1400.0 }, life: decay_short }); wait(0.9);
spawn("hdens", 3, #{ amp: 0.2, method: #{ mode: "harmonic_density", min_freq: 160.0, max_freq: 1400.0, temperature: 0.7 }, life: sustain_soft }); wait(0.8);
spawn("zero", 2, #{ amp: 0.18, method: #{ mode: "zero_crossing", min_freq: 180.0, max_freq: 1600.0 }, life: decay_short }); wait(0.8);
spawn("lowharm", 2, #{ amp: 0.18, method: #{ mode: "low_harmonicity", min_freq: 200.0, max_freq: 1500.0 }, life: decay_short });
wait(1.5);
finish();
