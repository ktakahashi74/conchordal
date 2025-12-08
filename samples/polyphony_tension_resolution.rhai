print("Polyphony: Anchor (L1) + Tension (L2)");

// Function to manage a polyphonic chord step
fn play_poly_step(root_freq, chord_name, duration) {
    // --- Lifecycle Definitions ---
    // Bass: Stable, sustained
    let bass_life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.1, envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 1.0 } };
    // Pad: Slow swell, blending
    let pad_life = #{ type: "sustain", initial_energy: 0.9, metabolism_rate: 0.15, envelope: #{ attack_sec: 0.8, decay_sec: 0.5, sustain_level: 0.7 } };
    // Spice: Short, decaying "stings" or wandering lines
    let spice_life = #{ type: "decay", initial_energy: 0.8, half_life_sec: 0.6 };

    let id = chord_name;
    print(">>> Chord: " + chord_name + " (Root: " + root_freq + "Hz)");

    // 1. [Level 1] The Anchor (Bass)
    // Establishes the harmonic field.
    add_agent("bass_" + id, root_freq, 0.4, bass_life);
    
    // WAIT: Allow physics to form the landscape peaks
    wait(0.2); 

    // 2. [Follower] The Pads
    // They seek the "peaks" created by the bass (Perfect 5th, Maj 3rd, etc.)
    // Search range: 2 octaves above root
    let pad_method = #{ 
        mode: "harmonicity", 
        min_freq: root_freq * 2.0, 
        max_freq: root_freq * 4.0 
    };
    spawn_agents("pad_" + id, pad_method, pad_life, 3, 0.15);

    // 3. [Level 2] The Tension (Spice) - DELAYED ENTRY
    // We inject this halfway through to create movement/rub.
    wait(duration * 0.3);

    // Strategy: varying by chord function
    if chord_name == "V_G7" {
        // On Dominant (V), we want HIGH tension.
        // Use "LowHarmonicity" to find notes that clash/rub against the stable notes.
        // This often finds tritone substitutes or altered extensions (b9, #11).
        let tension_method = #{ mode: "low_harmonicity", min_freq: root_freq * 2.5, max_freq: root_freq * 5.0 };
        print("    ! Injecting Tension (LowHarmonicity)");
        spawn_agents("spice_" + id, tension_method, spice_life, 2, 0.18);
    } else {
        // On Tonic/Pre-dominant (I, II), we fill "Spectral Gaps".
        // This creates melodic interest without extreme dissonance.
        let fill_method = #{ mode: "spectral_gap", min_freq: root_freq * 3.0, max_freq: root_freq * 6.0 };
        print("    * Injecting Color (SpectralGap)");
        spawn_agents("spice_" + id, fill_method, spice_life, 2, 0.12);
    }

    // Wait out the rest of the duration
    wait(duration * 0.7 - 0.2);

    // Clean up
    remove("bass_" + id);
    remove("pad_" + id);
    // Note: We let "spice" decay naturally (it's decay type), or remove if sustain.
}

// --- Progression: II - V - I - VI (Loop) ---
// Dm7 -> G7 -> Cmaj7 -> A7alt
let seq = [
    #{ freq: 146.83, name: "II_Dm7", dur: 2.0 },
    #{ freq: 98.00,  name: "V_G7",   dur: 2.0 },
    #{ freq: 130.81, name: "I_Cmaj", dur: 3.0 },
    #{ freq: 110.00, name: "VI_A7",  dur: 2.0 } 
];

section("polyphony_start");
wait(0.5);

for i in 0..seq.len() {
    let step = seq[i];
    play_poly_step(step.freq, step.name, step.dur);
    wait(0.1); // Breathing room
}

// Final resolution to C
play_poly_step(130.81, "Final_C", 4.0);

finish();
