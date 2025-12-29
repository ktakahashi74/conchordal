print(">>> Experiment: Self-Organizing Rhythm <<<");

// ============================================================
// Setup: percussive so synchronization is easy to hear
// ============================================================

// Long-lived agents, but the sound is a short pluck.
// Sharp attack makes timing offsets and sync obvious.
let life_plucking = #{
    body: #{ core: "sine" },
    temporal: #{
        core: "entrain",
        type: "sustain",
        initial_energy: 1.5,
        metabolism_rate: 0.05, // slow to die
        envelope: #{ 
            attack_sec: 0.005, // very sharp attack
            decay_sec: 0.1,    // quick decay
            sustain_level: 0.0 // percussive blip (discrete rhythm)
        }
    },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};

// ============================================================
// Phase 1: The Seed (metronome-driven entrainment)
// ============================================================
scene("Phase 1: Metronome");
print("--- [1] Injecting strong beat (Delta band driver) ---");

// 1. Place a strong kick (60 Hz) to shake the low/delta band and set a reference.
add_agent("kick", 60.0, 0.5, #{ 
    body: #{ core: "sine" },
    temporal: #{
        core: "entrain",
        type: "sustain", 
        initial_energy: 2.0, 
        metabolism_rate: 0.0, // does not die
        envelope: #{ attack_sec: 0.01, decay_sec: 0.2, sustain_level: 0.0 }
    },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
});

wait(2.0); // allow rhythm to settle

// ============================================================
// Phase 2: The Swarm (group injection)
// ============================================================
scene("Phase 2: Recruitment");
print("--- [2] Spawning agents with random phases ---");

// 2. Inject many agents (200–800 Hz).
// Phases start messy, then get pulled toward the kick (delta band).
let method_flock = #{ 
    mode: "harmonic_density", 
    min_freq: 200.0, 
    max_freq: 800.0, 
    temperature: 1.0 
};

// Add in four waves to thicken layers
for i in 0..4 {
    spawn_agents("follower", method_flock, life_plucking, 4, 0.08);
    wait(1.0); // pause before next batch
}

print("--- Listening for synchronization... ---");
wait(4.0); // give time to synchronize

// ============================================================
// Phase 3: Emergence (self-sustain)
// ============================================================
scene("Phase 3: Emergence");
print("--- [3] REMOVING THE METRONOME (Watch self-sustain) ---");

// 3. Remove the parent kick; no external reference remains.
// Agents try to keep rhythm by mutual coupling alone.
remove("kick");

// Success if the groove persists (with some wobble) after the leader vanishes.
wait(6.0);

// ============================================================
// Phase 4: Perturbation (disrupt and resync)
// ============================================================
scene("Phase 4: Perturbation");
print("--- [4] Injecting counter-rhythm (High freq) ---");

// 4. Inject high clicks with a different tempo to perturb the groove.
// Watch whether high/alpha stimulation micro-adjusts timing.
let life_click = #{
    body: #{ core: "sine" },
    temporal: #{ core: "entrain", type: "decay", initial_energy: 0.8, half_life_sec: 0.2 },
    field: #{ core: "pitch_hill_climb" },
    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
    perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
};
let method_click = #{ mode: "random_log_uniform", min_freq: 2000.0, max_freq: 4000.0 };

for i in 0..8 {
    spawn_agents("noise", method_click, life_click, 2, 0.1);
    wait(0.25); // fast pulse
}

wait(3.0); // observe post-perturbation settling

// Done
finish();
