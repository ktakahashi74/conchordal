print("Cycle of Fifths (settle time baked in)");

fn play_step(name, root, dur) {
    let bass_life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.5, attack_sec: 0.25 };
    let pad_life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 1.2, attack_sec: 0.7 };
    let pad_method = #{ mode: "harmonicity", min_freq: root * 2.2, max_freq: root * 5.5 };

    add_agent("bass_" + name, root, 0.34, bass_life);
    wait(0.25); // allow landscape to warp
    spawn_agents("pad_" + name, pad_method, pad_life, 3, 0.2);
    wait(dur);
    remove("bass_" + name);
    remove("pad_" + name);
}

let chords = [
    #{ name: "Am",  freq: 110.00, dur: 1.6 },
    #{ name: "Dm",  freq: 146.83, dur: 1.6 },
    #{ name: "G7",  freq: 196.00, dur: 1.6 },
    #{ name: "Cmaj",freq: 130.81, dur: 1.6 },
    #{ name: "Fmaj",freq: 174.61, dur: 1.6 },
    #{ name: "Bdim",freq: 246.94, dur: 1.6 },
    #{ name: "E7",  freq: 164.81, dur: 1.6 },
    #{ name: "Am",  freq: 110.00, dur: 2.0 },
];

section("cycle_fixed");
for ch in chords {
    play_step(ch.name, ch.freq, ch.dur);
}

wait(1.0);
finish();
