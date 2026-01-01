print(">>> Cycle of Fifths (Fixed Crowding) <<<");

fn play_step(name, root, dur) {
    let pad_life = #{
        articulation: #{ core: "entrain", type: "decay", half_life_sec: 1.0, attack_sec: 0.6 }
    };

    let bass_life = #{
        articulation: #{ core: "entrain", type: "decay", half_life_sec: 0.6, attack_sec: 0.2 }
    };

    let pad_method = #{
        mode: "harmonicity",
        min_freq: root * 2.0,
        max_freq: root * 5.5,
        min_dist_erb: 1.0
    };

    print(" -> " + name + " @ " + root + "Hz");

    let bass_method = #{ mode: "random_log_uniform", min_freq: root, max_freq: root };
    let bass = spawn("bass_" + name, 1, #{ amp: 0.38, method: bass_method, life: bass_life });
    wait(0.25);

    let pads = [];
    for i in 0..3 {
        let pad = spawn("pad_" + name + "_" + i, 1, #{ amp: 0.14, method: pad_method, life: pad_life });
        pads.push(pad);
        wait(0.2);
    }

    wait(dur);

    remove(bass);
    for pad in pads {
        remove(pad);
    }
}

let chords = [
    #{ name: "Am",   freq: 110.00, dur: 1.6 },
    #{ name: "Dm",   freq: 146.83, dur: 1.6 },
    #{ name: "G7",   freq: 196.00, dur: 1.6 },
    #{ name: "Cmaj", freq: 130.81, dur: 1.6 },
    #{ name: "Fmaj", freq: 174.61, dur: 1.6 },
    #{ name: "Bdim", freq: 246.94, dur: 1.6 },
    #{ name: "E7",   freq: 164.81, dur: 1.6 },
    #{ name: "Am",   freq: 110.00, dur: 1.8 },
];

scene("cycle");

for ch in chords {
    play_step(ch.name, ch.freq, ch.dur);
}

wait(1.0);
end();
