// Autonomous Harmonic Progression demo
scene("anchor_intro");

// Phase 1: anchor on C2
add_agent(
    "anchor",
    65.406, // C2
    0.12,
    #{
        body: #{ core: "sine" },
        temporal: #{
            core: "entrain",
            type: "sustain",
            initial_energy: 1.0,
            metabolism_rate: 0.0,
            envelope: #{ attack_sec: 0.02, decay_sec: 0.2, sustain_level: 1.0 }
        },
        field: #{ core: "pitch_hill_climb" },
        modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    }
);
set_commitment("anchor", 0.9);

wait(4.0);

scene("seekers_arrive");
// Phase 2: seekers at dissonant intervals (F#, C#)
add_agent("seeker_fsharp", 92.499, 0.08, #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.2, half_life_sec: 6.0 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
});
add_agent("seeker_csharp", 69.296, 0.08, #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.2, half_life_sec: 6.0 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
});
set_commitment("seeker_fsharp", 0.25);
set_commitment("seeker_csharp", 0.25);

wait(8.0);

scene("anchor_moves");
// Phase 3: anchor hops to F2; swarm should re-lock on F-major colors
set_freq("anchor", 87.307); // F2
set_commitment("anchor", 0.8);

wait(10.0);
finish();