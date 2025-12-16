print("II - V - I anchor walk");

fn play_chord(root, tag) {
    let bass_life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.1 };
    let pad_life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.9, attack_sec: 0.4 };
    let pad_method = #{ mode: "harmonicity", min_freq: root * 2.0, max_freq: root * 5.0 };

    add_agent("bass_" + tag, root, 0.36, bass_life);
    wait(0.18);
    spawn_agents("pad_" + tag, pad_method, pad_life, 3, 0.2);
    wait(1.4);
    remove("bass_" + tag);
    remove("pad_" + tag);
}

scene("ii");
play_chord(146.83, "ii"); // Dm7

scene("v");
play_chord(196.00, "v"); // G7

scene("i");
play_chord(130.81, "i"); // Cmaj

wait(1.0);
finish();
