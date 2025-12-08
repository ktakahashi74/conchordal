print("Harmonic Progression: II - V - I via Anchor Bass (Fixed Scope)");

// Function to play a "chord" by setting a bass anchor and scattering upper structures
fn play_chord(root_freq, tag_suffix) {
    // Define lifecycles INSIDE the function to avoid Rhai scope errors
    let pad_life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.15, envelope: #{ attack_sec: 0.5, decay_sec: 0.5, sustain_level: 0.8 } };
    let bass_life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.1, envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 0.9 } };

    let bass_tag = "bass_" + tag_suffix;
    let pad_tag = "pad_" + tag_suffix;

    // 1. Anchor: The Bass Note
    print("  -> Anchor: " + bass_tag + " @ " + root_freq + "Hz");
    add_agent(bass_tag, root_freq, 0.4, bass_life);

    // 2. Structure: The Pads
    // Search harmonic spots relative to the CURRENT landscape (dominated by the bass).
    let method = #{ 
        mode: "harmonicity", 
        min_freq: root_freq * 2.0, 
        max_freq: root_freq * 5.0 
    };
    
    spawn_agents(pad_tag, method, pad_life, 3, 0.2);
}

fn clear_chord(tag_suffix) {
    remove("bass_" + tag_suffix);
    remove("pad_" + tag_suffix);
}

// Sequence: Dm7 (II) -> G7 (V) -> C (I)
// Reference C4 = 261.63 Hz.

section("II_Dm7");
// D3 (146.83Hz). 
// Natural harmonicity often pulls to Major 3rd (F#), making it D Major-ish.
// But context (memory) might affect this. Let's listen.
play_chord(146.83, "ii");
wait(2.0);

section("V_G7");
clear_chord("ii");
// G2 (98.00Hz).
play_chord(98.00, "v");
wait(2.0);

section("I_C");
clear_chord("v");
// C3 (130.81Hz).
play_chord(130.81, "i");
wait(3.0);

finish();
