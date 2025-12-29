print(">>> Cycle of Fifths (Fixed Crowding) <<<");

fn play_step(name, root, dur) {
    // 1. Lifecycle configuration
    let pad_life = #{ 
        body: #{ core: "sine" },
        temporal: #{ 
            core: "entrain",
            type: "decay", 
            initial_energy: 1.0, 
            half_life_sec: 1.0, 
            attack_sec: 0.6
        },
        field: #{ core: "pitch_hill_climb" },
        modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };
    
    let bass_life = #{ 
        body: #{ core: "sine" },
        temporal: #{ 
            core: "entrain",
            type: "decay", 
            initial_energy: 1.0, 
            half_life_sec: 0.6, 
            attack_sec: 0.2
        },
        field: #{ core: "pitch_hill_climb" },
        modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };

    // 2. Spawn method configuration (Critical section)
    // Use `min_dist_erb` to avoid overlap during initial placement
    let pad_method = #{ 
        mode: "harmonicity", 
        min_freq: root * 2.0, 
        max_freq: root * 5.5,
        // Minimum ERB distance between spawned fundamentals
        min_dist_erb: 1.0
    };

    print(" -> " + name + " @ " + root + "Hz");
    
    // Spawn bass
    add_agent("bass_" + name, root, 0.38, bass_life);
    
    // Wait for the bass to ripple through the landscape
    wait(0.25);

    for i in 0..3 {
        // Spawn pads sequentially
        // By waiting each time, the next pad can sense the impact caused by the previous pad on the landscape.
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

for ch in chords {
    play_step(ch.name, ch.freq, ch.dur);
}

wait(1.0);
finish();
