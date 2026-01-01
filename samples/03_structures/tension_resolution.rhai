print("Polyphony: anchor + tension + resolution");

fn chord_block(root, name, sustain_amp, tension_amp, dur) {
    let bass_life = #{
        body: #{ core: "sine" },
        articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.1 },
        pitch: #{ core: "pitch_hill_climb" },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };
    let pad_life = #{
        body: #{ core: "sine" },
        articulation: #{ core: "entrain", type: "decay", initial_energy: 0.9, half_life_sec: 0.9, attack_sec: 0.4 },
        pitch: #{ core: "pitch_hill_climb" },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };
    let sting_life = #{
        body: #{ core: "sine" },
        articulation: #{ core: "entrain", type: "decay", initial_energy: 0.8, half_life_sec: 0.5 },
        pitch: #{ core: "pitch_hill_climb" },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };

    // Anchor bass
    let bass_method = #{ mode: "random_log_uniform", min_freq: root, max_freq: root };
    spawn("bass_" + name, 1, #{ amp: sustain_amp, method: bass_method, life: bass_life });
    wait(0.18);

    // Pads seek harmonic peaks above the anchor
    let pad_method = #{ mode: "harmonicity", min_freq: root * 2.0, max_freq: root * 5.0 };
    spawn("pad_" + name, 3, #{ amp: sustain_amp * 0.6, method: pad_method, life: pad_life });

    // Midway, inject tension using low_harmonicity
    wait(dur * 0.4);
    let tension_method = #{ mode: "low_harmonicity", min_freq: root * 1.5, max_freq: root * 4.0 };
    spawn("tension_" + name, 2, #{ amp: tension_amp, method: tension_method, life: sting_life });

    // Resolve: refresh harmonicity and thin tension
    wait(dur * 0.35);
    spawn("resolve_" + name, 2, #{ amp: sustain_amp * 0.5, method: pad_method, life: pad_life });
    wait(dur * 0.25);
}

scene("Imaj7");
chord_block(130.81, "Cmaj7", 0.32, 0.18, 2.0);

scene("IIm7");
chord_block(146.83, "Dm7", 0.3, 0.18, 1.8);

scene("V7");
chord_block(196.00, "G7", 0.34, 0.2, 2.2);

scene("Imaj7_resolve");
chord_block(130.81, "Cmaj7_final", 0.36, 0.14, 2.4);

wait(1.0);
end();
