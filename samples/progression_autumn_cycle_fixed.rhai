print("Long Progression: Cycle of Fifths (Latency Fixed)");

fn play_chord(root_freq, id_str) {
    let pad_life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.12, envelope: #{ attack_sec: 0.8, decay_sec: 0.5, sustain_level: 0.7 } };
    let bass_life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.08, envelope: #{ attack_sec: 0.2, decay_sec: 0.1, sustain_level: 0.95 } };

    let bass_tag = "bass_" + id_str;
    let pad_tag = "pad_" + id_str;

    print("  -> Chord [" + id_str + "] Root: " + root_freq + " Hz");

    // 1. Anchor: Bass
    add_agent(bass_tag, root_freq, 0.35, bass_life);

    // --- CRITICAL FIX: Wait for the Landscape to form ---
    // Give the audio engine ~200ms to render the bass and update the potential field.
    wait(0.2); 

    // 2. Scatter Upper Structure
    // Now the landscape has peaks at 3rd, 5th, etc.
    let method = #{ 
        mode: "harmonicity", 
        min_freq: root_freq * 2.1, 
        max_freq: root_freq * 6.0 
    };
    
    // Slight boost to Pad amplitude for visibility
    spawn_agents(pad_tag, method, pad_life, 4, 0.22);
}

fn clear_chord(id_str) {
    remove("bass_" + id_str);
    remove("pad_" + id_str);
}

// --- Progression Data ---
let chords = [
    #{ name: "VI_Am", freq: 110.00, dur: 2.0 },
    #{ name: "II_Dm", freq: 146.83, dur: 2.0 },
    #{ name: "V_G",   freq: 98.00,  dur: 2.0 },
    #{ name: "I_C",   freq: 130.81, dur: 3.0 },
    #{ name: "IV_F",  freq: 87.31,  dur: 2.0 },
    #{ name: "vii_B", freq: 123.47, dur: 2.0 },
    #{ name: "III_E", freq: 164.81, dur: 2.0 },
    #{ name: "VI_Am_End", freq: 110.00, dur: 4.0 }
];

section("start");
wait(0.5);

for i in 0..chords.len() {
    let c = chords[i];
    let id = i.to_string();
    
    play_chord(c.freq, id);
    
    // Subtract the wait time used inside play_chord to keep total rhythm tight
    wait(c.dur - 0.2);
    
    clear_chord(id);
    wait(0.1); 
}

finish();
