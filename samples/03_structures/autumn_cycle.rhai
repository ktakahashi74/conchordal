print(">>> Cycle of Fifths <<<");

fn play_step(name, root, dur) {
    print(" -> " + name + " @ " + root + "Hz");

    // Use hold so the bass sustains across the step.
    let bass_patch = #{
        body: #{ amp: 0.38, method: "sine" },
        phonation: #{ type: "hold" },
        pitch: #{ constraint: #{ mode: "lock", freq_hz: root } }
    };
    let bass_tag = "bass_" + name;
    spawn(bass_tag, 1, bass_patch);

    // Let the bass register in the landscape before sampling pads.
    wait(0.20);

    // Pad timbre is fixed; pitch is sampled via harmonicity.
    let pad_patch = #{
        body: #{ amp: 0.14, method: "sine", timbre: #{ brightness: 0.7, width: 0.2 } },
        phonation: #{ type: "hold" }
        // Leave pitch.center_hz unset (sampling decides it).
    };

    // Harmonicity sampling (band-limited around the root).
    let pad_method = #{
        mode: "harmonicity",
        min_freq: root * 2.0,
        max_freq: root * 5.0,
        min_dist_erb: 1.0
    };
    let pad_opts = #{ method: pad_method };

    let pads = [];
    for i in 0..3 {
        let pad_tag = "pad_" + name + "_" + i;

        // Sample pad pitch at spawn time using harmonicity.
        spawn(pad_tag, 1, pad_opts, pad_patch);

        pads.push(pad_tag);
        wait(0.15);
    }

    wait(dur);

    remove(bass_tag);
    for pad in pads { remove(pad); }
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
