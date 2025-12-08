print("Long Progression: Cycle of Fifths (Autumn Leaves style)");

// --- Setup Functions ---

fn play_chord(root_freq, id_str) {
    // Define lifecycles locally
    let pad_life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.12, envelope: #{ attack_sec: 0.8, decay_sec: 0.5, sustain_level: 0.7 } };
    let bass_life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.08, envelope: #{ attack_sec: 0.2, decay_sec: 0.1, sustain_level: 0.95 } };

    let bass_tag = "bass_" + id_str;
    let pad_tag = "pad_" + id_str;

    print("  -> Chord [" + id_str + "] Root: " + root_freq + " Hz");

    // 1. Set the Anchor (Bass)
    // Strong amplitude to warp the landscape
    add_agent(bass_tag, root_freq, 0.45, bass_life);

    // 2. Scatter Upper Structure
    // Allow a wider range (up to 6th harmonic) to find colorful extensions (9th, 13th)
    let method = #{ 
        mode: "harmonicity", 
        min_freq: root_freq * 2.1, 
        max_freq: root_freq * 6.0 
    };
    
    // Spawn 4 agents for a richer texture (7th chord density)
    spawn_agents(pad_tag, method, pad_life, 4, 0.18);
}

fn clear_chord(id_str) {
    remove("bass_" + id_str);
    remove("pad_" + id_str);
}

// --- Progression Data (Cycle of Fifths) ---
// Key: C Major / A Minor
// A2=110, D3=146.8, G2=98.0, C3=130.8, F2=87.3, B2=123.5, E3=164.8

let chords = [
    #{ name: "VI_Am", freq: 110.00, dur: 2.0 }, // Am7
    #{ name: "II_Dm", freq: 146.83, dur: 2.0 }, // Dm7
    #{ name: "V_G",   freq: 98.00,  dur: 2.0 }, // G7
    #{ name: "I_C",   freq: 130.81, dur: 3.0 }, // Cmaj7 (Resolution 1)
    
    #{ name: "IV_F",  freq: 87.31,  dur: 2.0 }, // Fmaj7
    #{ name: "vii_B", freq: 123.47, dur: 2.0 }, // Bm7b5 (High Tension!)
    #{ name: "III_E", freq: 164.81, dur: 2.0 }, // E7 (Dominant to Am)
    #{ name: "VI_Am_End", freq: 110.00, dur: 4.0 } // Am (Final Resolution)
];

// --- Execution Loop ---

section("start");
wait(0.5);

for i in 0..chords.len() {
    let c = chords[i];
    let id = i.to_string();
    
    // Play new chord
    play_chord(c.freq, id);
    
    // Wait for duration
    wait(c.dur);
    
    // Clean up
    clear_chord(id);
    
    // Short overlap/gap adjustment
    wait(0.1); 
}

finish();
