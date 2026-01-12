print("Autumn cycle");

let chords = [
    #{ name: "Am",   freq: 110.00, dur: 1.6 },
    #{ name: "Dm",   freq: 146.83, dur: 1.6 },
    #{ name: "G",    freq: 196.00, dur: 1.6 },
    #{ name: "C",    freq: 130.81, dur: 1.6 },
    #{ name: "Fmaj", freq: 174.61, dur: 1.6 },
    #{ name: "Bdim", freq: 246.94, dur: 1.6 },
    #{ name: "E7",   freq: 164.81, dur: 1.6 },
    #{ name: "Am",   freq: 110.00, dur: 1.8 }
];

for chord in chords {
    let bass_tag = "bass_" + chord.name;
    let pad_tag = "pad_" + chord.name;

    spawn(bass_tag, 1, #{
        body: #{ amp: 0.38, method: "harmonic" },
        phonation: #{ type: "drone", density: 1.0, legato: 1.0 },
        pitch: #{ constraint: #{ mode: "lock", freq_hz: chord.freq } }
    });

    spawn(pad_tag, 2, #{
        body: #{ amp: 0.14, method: "harmonic", timbre: #{ brightness: 0.7, width: 0.2 } },
        phonation: #{ type: "drone", density: 1.0, legato: 1.0 },
        pitch: #{ constraint: #{ mode: "lock", freq_hz: chord.freq * 2.0 } }
    });

    wait(chord.dur);

    remove(bass_tag);
    remove(pad_tag);
}

wait(0.2);
end();
