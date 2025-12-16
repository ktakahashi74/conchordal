print(">>> Kuramoto Verification: Entrainment & Self-Organization <<<");

// ------------------------------------------
// Click voice: short, sharp pluck for timing
// ------------------------------------------
let life_click = #{
    type: "sustain",
    initial_energy: 1.5,
    metabolism_rate: 0.0,
    envelope: #{ 
        attack_sec: 0.005, // instant attack
        decay_sec: 0.1,    // quick decay
        sustain_level: 0.0 // no tail
    }
};

// ==========================================
// Phase 1: The Metronome (external driver)
// ==========================================
scene("Phase 1: External Driver");
print("--- [1] Metronome starts (Listen for the beat) ---");

// Inject a strong external pulse to shake Delta waves and set a reference beat.
// This is a system-driven event, not the agents’ own rhythm.
let kick_life = #{ type: "decay", initial_energy: 1.5, half_life_sec: 0.1 };

// Kick loop: every 0.6s (BPM 100), 20 hits
parallel(|| {
    for i in 0..20 {
        add_agent("kick", 60.0, 0.8, kick_life);
        wait(0.6);
    }
});

wait(2.0); // let the metronome settle

// ==========================================
// Phase 2: The Swarm (entrainment)
// ==========================================
scene("Phase 2: Entrainment");
print("--- [2] Spawning autonomous agents (Watch them sync) ---");

// Agents start with random internal clocks (0.5–3 Hz) but entrain to the kick beat.

let method_random = #{ 
    mode: "random_log_uniform", 
    min_freq: 400.0, 
    max_freq: 1200.0 
};

// Spawn 5 agents
spawn_agents("follower", method_random, life_click, 5, 0.2);

print(">>> Observing synchronization... (Wait 10s)");
wait(10.0);

// ==========================================
// Phase 3: Autonomy (metronome stops)
// ==========================================
scene("Phase 3: Self-Organization");
print("--- [3] METRONOME STOPS. (Agents maintain the groove) ---");

// Metronome ends. Synchronized agents keep the beat via mutual coupling—listen for emergent self-maintenance.

// Observe until agents die out
wait(10.0);

finish();
