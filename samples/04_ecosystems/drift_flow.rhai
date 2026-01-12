print(">>> Drift & Flow: Physics Verification Demo <<<");

// ==========================================
// Phase 1: Gravity Test (adhesion experiment)
// ==========================================
scene("Phase 1: Gravity Pull");
print("--- [1] Anchoring C2 (65.41Hz) ---");

// Base tone (static anchor)
let anchor_method = #{ mode: "random_log_uniform", min_freq: 65.41, max_freq: 65.41 };
let anchor_tag = "anchor";
spawn(anchor_tag, 1, #{ amp: 0.6, method: anchor_method, life: #{ 
    body: #{ core: "sine" },
    articulation: #{
        core: "entrain",
        type: "sustain", 
        initial_energy: 10.0, 
        metabolism_rate: 0.0,
        envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 1.0 }
    },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
} });
// Anchor stays put (high persistence = resists movement)
set(anchor_tag, #{ pitch: #{ persistence: 1.0 } });

wait(1.0);

print("--- Injecting Dissonant Tone (C#3) -> Watch it hop to C3 or E3 ---");
// Drop a semitone-above tone (C#3 ≈ 138.59) above C3 (130.81)
// Check whether the pull guides it toward C3 or E3
let slider_method = #{ mode: "random_log_uniform", min_freq: 138.59, max_freq: 138.59 };
let slider_tag = "slider";
spawn(slider_tag, 1, #{ amp: 0.4, method: slider_method, life: #{
    body: #{ core: "sine" },
    articulation: #{
        core: "entrain",
        type: "sustain",
        initial_energy: 2.0,
        metabolism_rate: 0.0,
        envelope: #{ attack_sec: 0.5, decay_sec: 0.5, sustain_level: 1.0 }
    },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
} });

// Low persistence = moves easily
set(slider_tag, #{ pitch: #{ persistence: 0.33 } }); // eager to move

wait(4.0);

// ==========================================
// Phase 2: Perceptual Adaptation
// ==========================================
scene("Phase 2: Perceptual Push");
print("--- [2] Perceptual adaptation (self + environment) ---");

// Expect perceptual traces to grow under the slider

print("--- Slider should be pushed away by red arrow ---");
// As boredom accumulates, the peak flattens into a valley, nudging the agent to a neighboring harmonic.
wait(6.0);

// ==========================================
// Phase 3: Mass Migration (exploration and pursuit)
// ==========================================
scene("Phase 3: Modulation");
print("--- [3] Spawning Swarm & Moving Anchor ---");

remove(slider_tag);
// Lightly reset environment

// Drop a swarm (spread across C overtones)
let method = #{ mode: "harmonicity", min_freq: 130.0, max_freq: 800.0 };
let life = #{
    body: #{ core: "sine" },
    articulation: #{
        core: "entrain",
        type: "sustain",
        initial_energy: 5.0,
        metabolism_rate: 0.05,
        envelope: #{ attack_sec: 1.0, decay_sec: 1.0, sustain_level: 0.5 }
    },
    pitch: #{ core: "pitch_hill_climb" },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

for i in 0..5 {
    let swarm_tag = "swarm_" + i;
    spawn(swarm_tag, 1, #{ amp: 0.15, method: method, life: life });
    set(swarm_tag, #{ pitch: #{ persistence: 0.56 } }); // hop loosely
    wait(0.5);
}

wait(2.0);

print(">>> MOVING ANCHOR to F2 (87.31Hz) <<<");
print(">>> Watch the blue arrows flip direction! <<<");

// Move anchor to F2.
// Landscape flips: C-major valleys disappear, F-major basins emerge.
// Swarm should move toward F overtones (A, C, Eb...).
set(anchor_tag, #{ pitch: #{ center_hz: 87.31 } }); 

wait(8.0);

print(">>> Releasing Anchor (Dissolution) <<<");
remove(anchor_tag);
// Removing the anchor flattens the landscape; swarm should disperse/decay.
wait(5.0);

end();
