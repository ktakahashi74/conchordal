print(">>> Experiment: Self-Organizing Rhythm <<<");

// ============================================================
// Setup: percussive so synchronization is easy to hear
// ============================================================

let click_phonation = #{ type: "clock", density: 1.0, legato: 0.0 };

// Startup stimulus: a single short click to bootstrap rhythm extraction.
let starter_method = #{ mode: "random_log_uniform", min_freq: 100.0, max_freq: 100.0 };
spawn(
    "starter",
    1,
    #{ method: starter_method },
    #{ body: #{ amp: 0.4, method: "sine" }, phonation: click_phonation }
);

// Long-lived agents, but the sound is a short pluck.
// Sharp attack makes timing offsets and sync obvious.
let follower_patch = #{
    body: #{ amp: 0.08, method: "sine" },
    phonation: #{ type: "interval", density: 0.7, legato: 0.1 }
};

// ============================================================
// Phase 1: The Seed (metronome-driven entrainment)
// ============================================================
scene("Phase 1: Metronome");
print("--- [1] Injecting strong beat (Delta band driver) ---");

// 1. Place a strong kick (60 Hz) to shake the low/delta band and set a reference.
let kick_method = #{ mode: "random_log_uniform", min_freq: 60.0, max_freq: 60.0 };
let kick_tag = "kick";
spawn(
    kick_tag,
    1,
    #{ method: kick_method },
    #{ body: #{ amp: 0.5, method: "sine" }, phonation: click_phonation }
);

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
    spawn("follower", 4, #{ method: method_flock }, follower_patch);
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
remove(kick_tag);

// Success if the groove persists (with some wobble) after the leader vanishes.
wait(6.0);

// ============================================================
// Phase 4: Perturbation (disrupt and resync)
// ============================================================
scene("Phase 4: Perturbation");
print("--- [4] Injecting counter-rhythm (High freq) ---");

// 4. Inject high clicks with a different tempo to perturb the groove.
// Watch whether high/alpha stimulation micro-adjusts timing.
let method_click = #{ mode: "random_log_uniform", min_freq: 2000.0, max_freq: 4000.0 };

for i in 0..8 {
    spawn(
        "noise",
        2,
        #{ method: method_click },
        #{ body: #{ amp: 0.1, method: "sine" }, phonation: click_phonation }
    );
    wait(0.25); // fast pulse
}

wait(3.0); // observe post-perturbation settling

// Done
end();
