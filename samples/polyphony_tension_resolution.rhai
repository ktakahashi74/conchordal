print("Polyphony: anchor + tension + resolution");

fn chord_block(root, name, sustain_amp, tension_amp, dur) {
    let bass_life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.1 };
    let pad_life = #{ type: "decay", initial_energy: 0.9, half_life_sec: 0.9, attack_sec: 0.4 };
    let sting_life = #{ type: "decay", initial_energy: 0.8, half_life_sec: 0.5 };

    // Anchor bass
    add_agent("bass_" + name, root, sustain_amp, bass_life);
    wait(0.18);

    // Pads seek harmonic peaks above the anchor
    let pad_method = #{ mode: "harmonicity", min_freq: root * 2.0, max_freq: root * 5.0 };
    spawn_agents("pad_" + name, pad_method, pad_life, 3, sustain_amp * 0.6);

    // Midway, inject tension using low_harmonicity
    wait(dur * 0.4);
    let tension_method = #{ mode: "low_harmonicity", min_freq: root * 1.5, max_freq: root * 4.0 };
    spawn_agents("tension_" + name, tension_method, sting_life, 2, tension_amp);

    // Resolve: refresh harmonicity and thin tension
    wait(dur * 0.35);
    spawn_agents("resolve_" + name, pad_method, pad_life, 2, sustain_amp * 0.5);
    wait(dur * 0.25);
}

section("Imaj7");
chord_block(130.81, "Cmaj7", 0.32, 0.18, 2.0);

section("IIm7");
chord_block(146.83, "Dm7", 0.3, 0.18, 1.8);

section("V7");
chord_block(196.00, "G7", 0.34, 0.2, 2.2);

section("Imaj7_resolve");
chord_block(130.81, "Cmaj7_final", 0.36, 0.14, 2.4);

wait(1.0);
finish();
