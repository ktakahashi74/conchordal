print(">>> Harmonic Circle: Spread Voicing <<<");

// ==========================================
// Timbre definitions
// ==========================================
let life_drone = #{
    type: "decay",
    initial_energy: 10.0,
    half_life_sec: 8.0,
    attack_sec: 2.0
};

let life_pulse = #{
    type: "decay",
    initial_energy: 2.0,
    half_life_sec: 0.6,
    attack_sec: 0.01
};

// ==========================================
// Phase 1: Gravity (establish tonality)
// ==========================================
scene("Phase 1: Gravity");
print("--- [1] Establishing Harmonic Root (C2 + G2) ---");

add_agent("root", 65.41, 0.5, life_drone); // C2
wait(1.0);
add_agent("fifth", 98.00, 0.4, life_drone); // G2
wait(1.0);

// ==========================================
// Phase 2: Rhythm Driver
// ==========================================
scene("Phase 2: The Clock");
print("--- [2] Injecting Rhythm Driver ---");

add_agent("kick", 65.41, 0.6, #{ 
    type: "decay",
    initial_energy: 5.0,
    half_life_sec: 0.8,
    attack_sec: 0.005
});
wait(3.0);

// ==========================================
// Phase 3: The Chord (probabilistic spread)
// ==========================================
scene("Phase 3: Voicing");
print("--- [3] Spawning agents with Harmonic Density ---");

// Note: harmonicity -> harmonic_density
// temperature:
//   0.1 = snaps to the strongest peak (unison-like)
//   1.0 = spreads naturally by consonance
//   2.0 = occasionally samples low-consonance spots (jazzy/noisy)
let method_spread = #{ 
    mode: "harmonicity", 
    min_freq: 130.0, // C3
    max_freq: 1050.0, // up to about C6
    temperature: 0.5 // a bit sharp to favor clean voicings
};

for i in 0..6 {
    // Drop 6 voices; they should probabilistically land on C/E/G/B partials.
    spawn_agents("voice_" + i, method_spread, life_pulse, 1, 0.22);
    print(" -> Add Voice " + i);
    wait(1.0); 
}

print("--- Listening to the chord... ---");
wait(5.0);

// ==========================================
// Phase 4: Autonomy
// ==========================================
scene("Phase 4: Autonomy");
print("--- [4] Removing Anchor & Kick ---");

remove("kick");
// Also fade the drones; test whether the 6-voice interaction alone holds the chord.
// (harmonic_density choices should still resonate without the anchor)
set_amp("root", 0.0);
set_amp("fifth", 0.0);

wait(10.0);
finish();
