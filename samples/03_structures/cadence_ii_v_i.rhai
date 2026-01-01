print("II - V - I anchor walk");

fn play_chord(root, tag) {
    let bass_life = #{
        body: #{ core: "sine" },
        articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.1 },
        pitch: #{ core: "pitch_hill_climb" },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };
    let pad_life = #{
        body: #{ core: "sine" },
        articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.9, attack_sec: 0.4 },
        pitch: #{ core: "pitch_hill_climb" },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };
    let pad_method = #{ mode: "harmonicity", min_freq: root * 2.0, max_freq: root * 5.0 };

    let bass = add_agent("bass_" + tag, root, 0.36, bass_life);
    wait(0.18);
    let pads = spawn_agents("pad_" + tag, pad_method, pad_life, 3, 0.2);
    wait(1.4);
    bass.remove();
    pads.remove();
}

scene("ii");
play_chord(146.83, "ii"); // Dm7

scene("v");
play_chord(196.00, "v"); // G7

scene("i");
play_chord(130.81, "i"); // Cmaj

wait(1.0);
finish();
