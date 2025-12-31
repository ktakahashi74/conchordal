print(">>> Comparison: Major Gravity vs Mirror Dualism <<<");

// ==========================================
// Shared Setup: The Anchor
// ==========================================
// Center on C4 (261.63Hz).
// "C" persists in both phases, but the attraction landscape around it changes.
let anchor_life = #{ 
    body: #{ core: "sine" },
    temporal: #{
        core: "entrain",
        type: "sustain", 
        initial_energy: 10.0, 
        metabolism_rate: 0.0, 
        envelope: #{ attack_sec: 2.0, decay_sec: 0.5, sustain_level: 1.0 }
    },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

add_agent("anchor", 261.63, 0.4, anchor_life);
wait(2.0);

// ==========================================
// Phase 1: Common Root (Standard / Major)
// ==========================================
// mirror_weight = 0.0
// Standard gravity based on the Overtone Series.
// With C4 active, its harmonics (E, G, Bb...) and C-rooted chords become stable.
// Result: Emergence of "C Major" tonality.
scene("Phase 1: Major Gravity");
print("--- [1] Weight 0.0: Common Root (Major) ---");

set_harmonicity(#{ mirror: 0.0 });
wait(0.5);

// Use harmonic_density with moderate temp to allow 3rds/5ths to emerge
let method_major = #{
    mode: "harmonic_density",
    min_freq: 130.0,
    max_freq: 1050.0,
    temperature: 0.5,
    min_dist_erb: 1.0
};
let life_pulse = #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.8 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

let voices_major = [];
for i in 0..6 {
    // Probabilistically selects C, E, G (C Major Triad)
    let voice = spawn_agents("voice_maj_" + i, method_major, life_pulse, 1, 0.2);
    voices_major.push(voice);
    wait(0.8);
}

wait(2.0);
// Clean up Phase 1 voices
for v in voices_major {
    v.remove();
}

// ==========================================
// Phase 2: Common Overtone (Mirror / Minor)
// ==========================================
// mirror_weight = 0.8
// Inverted gravity based on the Undertone Series.
// Searches for "hidden roots" that contain C4 as an overtone.
// - C is the 3rd harmonic of F (F-A-C)
// - C is the 5th harmonic of Ab (Ab-C-Eb)
// Result: Gravity flips to the minor side (F Minor / Ab Major) sharing C.
scene("Phase 2: Mirror Gravity");
print("--- [2] Weight 0.8: Common Overtone (Minor) ---");

// Invert gravity using map-based API
set_harmonicity(#{ mirror: 1.0 }); 
wait(1.0); // Allow landscape to update

let method_mirror = #{
    mode: "harmonic_density",
    min_freq: 130.0,
    max_freq: 1050.0,
    temperature: 0.5,
    min_dist_erb: 1.0
};

for i in 0..6 {
    // Probabilistically selects F, Ab (F Minor / Ab Major context)
    spawn_agents("voice_min_" + i, method_mirror, life_pulse, 1, 0.2);
    wait(0.8);
}

wait(3.0);
finish();
