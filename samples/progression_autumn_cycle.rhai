print(">>> Cycle of Fifths (Fixed Crowding) <<<");

fn play_step(name, root, dur) {
    // 1. Lifecycle configuration
    let pad_life = #{ 
        type: "decay", 
        initial_energy: 1.0, 
        half_life_sec: 1.0, 
        attack_sec: 0.6,
        // Note: Check if your Rust config maps this. Usually 'habituation_sensitivity' is in the agent config, not lifecycle.
        // Configuration to avoid crowding during autonomous movement
        crowding_sensitivity: 2.0 
    };
    
    let bass_life = #{ 
        type: "decay", 
        initial_energy: 1.0, 
        half_life_sec: 0.6, 
        attack_sec: 0.2,
        // Bass goes its own way (ignores crowding)
        crowding_sensitivity: 0.0 
    };

    // 2. Spawn method configuration (Critical section)
    // Explicitly set 'crowding' to avoid overlap during initial placement
    let pad_method = #{ 
        mode: "harmonicity", 
        min_freq: root * 2.0, 
        max_freq: root * 5.5,
        // Set high to encourage dispersion
        // Note: The Rust struct field is likely named 'crowding', not 'crowding_sensitivity'.
        crowding: 5.0 
    };

    print(" -> " + name + " @ " + root + "Hz");
    
    // Spawn bass
    add_agent("bass_" + name, root, 0.38, bass_life);
    
    // Wait for the bass to ripple through the landscape
    wait(0.25);

    for i in 0..3 {
        // Spawn pads sequentially
        // By waiting each time, the next pad can sense the impact (crowding)
        // caused by the previous pad on the landscape.
        spawn_agents("pad_" + name + "_" + i, pad_method, pad_life, 1, 0.14);
        wait(0.2); 
    }

    // Wait for the remaining duration
    // (Simplified logic: assumes total wait fits within duration)
    wait(dur);
    
    remove("bass_" + name);
    
    // Wildcard removal depends on implementation (verified: supported in population.rs)
    // Note: If wildcard fails, remove individually.
    remove("pad_" + name + "_*");
}

let chords = [
    #{ name: "Am",   freq: 110.00, dur: 1.6 },
    #{ name: "Dm",   freq: 146.83, dur: 1.6 },
    #{ name: "G7",   freq: 196.00, dur: 1.6 },
    #{ name: "Cmaj", freq: 130.81, dur: 1.6 },
    #{ name: "Fmaj", freq: 174.61, dur: 1.6 },
    #{ name: "Bdim", freq: 246.94, dur: 1.6 },
    #{ name: "E7",   freq: 164.81, dur: 1.6 },
    #{ name: "Am",   freq: 110.00, dur: 1.8 },
];

scene("cycle");

// Set global habituation/crowding parameters (if not already set)
// Prevents energy accumulation in one spot by triggering 'boredom' quickly
set_habituation_params(0.5, 2.0, 1.0); 

for ch in chords {
    play_step(ch.name, ch.freq, ch.dur);
}

wait(1.0);
finish();
