print(">>> Scan & Pursue Demo: Intelligent Harmony <<<");

// ==========================================
// Phase 1: Stable drone (C major context)
// ==========================================
scene("Phase 1: Anchor C2");
print("--- [1] Anchoring C2 (65.41Hz) ---");

// Anchor (immobile)
add_agent("anchor", 65.41, 0.6, #{
    type: "sustain",
    initial_energy: 10.0,
    metabolism_rate: 0.0,
    envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 1.0 }
});
set_commitment("anchor", 1.0);

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
    type: "sustain",
    initial_energy: 2.0,
    metabolism_rate: 0.0,
    envelope: #{ attack_sec: 1.0, decay_sec: 1.0, sustain_level: 0.5 }
};

// Place them slightly off the comfy spots
add_agent("seeker_tritone", 92.5, 0.4, life_seeker);
add_agent("seeker_m2", 140.0, 0.4, life_seeker); // near C3 (~130 Hz)

// Set commitment (higher = more stubborn)
set_commitment("seeker_tritone", 0.5);
set_commitment("seeker_m2", 0.67); // moves more slowly

print(">>> Observe: They should slide to the nearest consonant peak (G or C) <<<");
wait(5.0);

// ==========================================
// Phase 3: Tectonic shift (modulation)
// ==========================================
scene("Phase 3: Modulation to F");
print("--- [3] Moving Anchor to F2 (87.31Hz) ---");

// Force-move the anchor; landscape flips.
// Formerly comfy spots (C, G) may become dissonant relative to F.
set_freq("anchor", 87.31);

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
    spawn_agents("swarm_" + i, method, life_seeker, 1, 0.15);
    set_commitment("swarm_" + i, 0.56);
    wait(0.5);
}

wait(5.0);
finish();
