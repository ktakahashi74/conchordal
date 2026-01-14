print("II - V - I anchor walk");

fn play_chord(root, tag) {
    let pad_method = #{ mode: "harmonicity", min_freq: root * 2.0, max_freq: root * 5.0 };

    let bass_method = #{ mode: "random_log_uniform", min_freq: root, max_freq: root };
    let bass_tag = "bass_" + tag;
    spawn(
        bass_tag,
        1,
        #{ method: bass_method },
        #{ body: #{ amp: 0.36, method: "sine" }, phonation: #{ type: "hold" } }
    );
    wait(0.18);
    let pads_tag = "pad_" + tag;
    spawn(
        pads_tag,
        3,
        #{ method: pad_method },
        #{ body: #{ amp: 0.2, method: "sine" }, phonation: #{ type: "hold" } }
    );
    wait(1.4);
    remove(bass_tag);
    remove(pads_tag);
}

scene("ii");
play_chord(146.83, "ii"); // Dm7

scene("v");
play_chord(196.00, "v"); // G7

scene("i");
play_chord(130.81, "i"); // Cmaj

wait(1.0);
end();
