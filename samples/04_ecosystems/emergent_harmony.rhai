// Autonomous Harmonic Progression demo
scene("anchor_intro");

// Phase 1: anchor on C2
let anchor_method = #{ mode: "random_log_uniform", min_freq: 65.406, max_freq: 65.406 };
let anchor = spawn(
    "anchor",
    1,
    #{
        amp: 0.12,
        method: anchor_method,
        life: #{
        body: #{ core: "sine" },
        articulation: #{
            core: "entrain",
            type: "sustain",
            initial_energy: 1.0,
            metabolism_rate: 0.0,
            envelope: #{ attack_sec: 0.02, decay_sec: 0.2, sustain_level: 1.0 }
        },
        pitch: #{ core: "pitch_hill_climb" },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
        }
    }
);
set(anchor, #{ persistence: 0.9 });

wait(4.0);

scene("seekers_arrive");
// Phase 2: seekers at dissonant intervals (F#, C#)
let seeker_life = #{
    body: #{ core: "sine" },
    articulation: #{ core: "entrain", type: "decay", initial_energy: 1.2, half_life_sec: 6.0 },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};
let seeker_fsharp_method = #{ mode: "random_log_uniform", min_freq: 92.499, max_freq: 92.499 };
let seeker_csharp_method = #{ mode: "random_log_uniform", min_freq: 69.296, max_freq: 69.296 };
let seeker_fsharp = spawn("seeker_fsharp", 1, #{ amp: 0.08, method: seeker_fsharp_method, life: seeker_life });
let seeker_csharp = spawn("seeker_csharp", 1, #{ amp: 0.08, method: seeker_csharp_method, life: seeker_life });
set(seeker_fsharp, #{ persistence: 0.25 });
set(seeker_csharp, #{ persistence: 0.25 });

wait(8.0);

scene("anchor_moves");
// Phase 3: anchor hops to F2; swarm should re-lock on F-major colors
set(anchor, #{ freq: 87.307 }); // F2
set(anchor, #{ persistence: 0.8 });

wait(10.0);
end();
