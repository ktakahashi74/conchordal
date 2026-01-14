print(">>> Scan & Pursue Demo: Intelligent Harmony <<<");

// ==========================================
// Phase 1: Stable drone (C major context)
// ==========================================
scene("Phase 1: Anchor C2");
print("--- [1] Anchoring C2 (65.41Hz) ---");

// Anchor (immobile)
let anchor_method = #{ mode: "random_log_uniform", min_freq: 65.41, max_freq: 65.41 };
let anchor_tag = "anchor";
spawn(
    anchor_tag,
    1,
    #{ method: anchor_method },
    #{ body: #{ amp: 0.6, method: "sine" }, phonation: #{ type: "hold" } }
);
set(anchor_tag, #{ pitch: #{ persistence: 1.0 } });

wait(1.0);

// ==========================================
// Phase 2: Inject seekers (dissonance -> resolution)
// ==========================================
scene("Phase 2: The Seekers");
print("--- [2] Spawning Seekers at Dissonant Intervals ---");

// Relative to C2 (65.41), drop dissonant agents:
// F# (tritone): 92.5 Hz -> likely resolves to G (98.0) or F (87.3)
// C# (minor 2nd): 69.3 Hz -> likely resolves to C (65.4) or D (73.4)

// Place them slightly off the comfy spots
let seeker_tritone_method = #{ mode: "random_log_uniform", min_freq: 92.5, max_freq: 92.5 };
let seeker_m2_method = #{ mode: "random_log_uniform", min_freq: 140.0, max_freq: 140.0 };
let seeker_tritone_tag = "seeker_tritone";
spawn(
    seeker_tritone_tag,
    1,
    #{ method: seeker_tritone_method },
    #{ body: #{ amp: 0.4, method: "sine" }, phonation: #{ type: "hold" } }
);
let seeker_m2_tag = "seeker_m2";
spawn(
    seeker_m2_tag,
    1,
    #{ method: seeker_m2_method },
    #{ body: #{ amp: 0.4, method: "sine" }, phonation: #{ type: "hold" } }
); // near C3 (~130 Hz)

// Set persistence (higher = more stubborn)
set(seeker_tritone_tag, #{ pitch: #{ persistence: 0.5 } });
set(seeker_m2_tag, #{ pitch: #{ persistence: 0.67 } }); // moves more slowly

print(">>> Observe: They should hop to the nearest consonant peak (G or C) <<<");
wait(5.0);

// ==========================================
// Phase 3: Tectonic shift (exploration)
// ==========================================
scene("Phase 3: Modulation to F");
print("--- [3] Moving Anchor to F2 (87.31Hz) ---");

// Force-move the anchor; landscape flips.
// Formerly comfy spots (C, G) may become dissonant relative to F.
set(anchor_tag, #{ pitch: #{ freq: 87.31 } });

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
    let swarm_tag = "swarm_" + i;
    spawn(
        swarm_tag,
        1,
        #{ method: method },
        #{ body: #{ amp: 0.15, method: "sine" }, phonation: #{ type: "hold" } }
    );
    set(swarm_tag, #{ pitch: #{ persistence: 0.56 } });
    wait(0.5);
}

wait(5.0);
end();
