print("Polyphony: anchor + tension + resolution");

fn chord_block(root, name, sustain_amp, tension_amp, dur) {
    // Anchor bass
    let bass_method = #{ mode: "random_log_uniform", min_freq: root, max_freq: root };
    spawn(
        "bass_" + name,
        1,
        #{ method: bass_method },
        #{ body: #{ amp: sustain_amp, method: "sine" }, phonation: #{ type: "hold" } }
    );
    wait(0.18);

    // Pads seek harmonic peaks above the anchor
    let pad_method = #{ mode: "harmonicity", min_freq: root * 2.0, max_freq: root * 5.0 };
    spawn(
        "pad_" + name,
        3,
        #{ method: pad_method },
        #{ body: #{ amp: sustain_amp * 0.6, method: "sine" }, phonation: #{ type: "hold" } }
    );

    // Midway, inject tension using low_harmonicity
    wait(dur * 0.4);
    let tension_method = #{ mode: "low_harmonicity", min_freq: root * 1.5, max_freq: root * 4.0 };
    spawn(
        "tension_" + name,
        2,
        #{ method: tension_method },
        #{ body: #{ amp: tension_amp, method: "sine" }, phonation: #{ type: "hold" } }
    );

    // Resolve: refresh harmonicity and thin tension
    wait(dur * 0.35);
    spawn(
        "resolve_" + name,
        2,
        #{ method: pad_method },
        #{ body: #{ amp: sustain_amp * 0.5, method: "sine" }, phonation: #{ type: "hold" } }
    );
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
