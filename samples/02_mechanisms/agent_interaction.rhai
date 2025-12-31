print(">>> Scan & Pursue Demo: Intelligent Harmony <<<");

// ==========================================
// Phase 1: Stable drone (C major context)
// ==========================================
scene("Phase 1: Anchor C2");
print("--- [1] Anchoring C2 (65.41Hz) ---");

// Anchor (immobile)
let anchor = add_agent("anchor", 65.41, 0.6, #{
    body: #{ core: "sine" },
    temporal: #{
        core: "entrain",
        type: "sustain",
        initial_energy: 10.0,
        metabolism_rate: 0.0,
        envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 1.0 }
    },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
});
anchor.set_commitment(1.0);

wait(1.0);

// ==========================================
// Phase 2: Inject seekers (dissonance -> resolution)
// ==========================================
scene("Phase 2: The Seekers");
print("--- [2] Spawning Seekers at Dissonant Intervals ---");

// Relative to C2 (65.41), drop dissonant agents:
// F# (tritone): 92.5 Hz -> likely resolves to G (98.0) or F (87.3)
// C# (minor 2nd): 69.3 Hz -> likely resolves to C (65.4) or D (73.4)

let life_seeker = #{
    body: #{ core: "sine" },
    temporal: #{
        core: "entrain",
        type: "sustain",
        initial_energy: 2.0,
        metabolism_rate: 0.0,
        envelope: #{ attack_sec: 1.0, decay_sec: 1.0, sustain_level: 0.5 }
    },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

// Place them slightly off the comfy spots
let seeker_tritone = add_agent("seeker_tritone", 92.5, 0.4, life_seeker);
let seeker_m2 = add_agent("seeker_m2", 140.0, 0.4, life_seeker); // near C3 (~130 Hz)

// Set commitment (higher = more stubborn)
seeker_tritone.set_commitment(0.5);
seeker_m2.set_commitment(0.67); // moves more slowly

print(">>> Observe: They should hop to the nearest consonant peak (G or C) <<<");
wait(5.0);

// ==========================================
// Phase 3: Tectonic shift (modulation)
// ==========================================
scene("Phase 3: Modulation to F");
print("--- [3] Moving Anchor to F2 (87.31Hz) ---");

// Force-move the anchor; landscape flips.
// Formerly comfy spots (C, G) may become dissonant relative to F.
anchor.set_freq(87.31);

// Agents scan periodically; they should notice F overtones feel better and retarget.

wait(8.0);

// ==========================================
// Phase 4: Swarm behavior (harmonic density)
// ==========================================
scene("Phase 4: Swarm Intelligence");
print("--- [4] Spawning a swarm ---");

// Add many bodies; peaks they create become targets, so watch for clustering vs dispersing.
let method = #{ mode: "harmonic_density", min_freq: 200.0, max_freq: 800.0 };
for i in 0..4 {
    let swarm = spawn_agents("swarm_" + i, method, life_seeker, 1, 0.15);
    swarm.set_commitment(0.56);
    wait(0.5);
}

wait(5.0);
finish();
