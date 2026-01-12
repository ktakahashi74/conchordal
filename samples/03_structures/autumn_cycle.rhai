print("Autumn cycle");

let chords = [
    #{ name: "Am",   freq: 110.00, dur: 1.6 },
    #{ name: "Dm",   freq: 146.83, dur: 1.6 },
    #{ name: "G7",   freq: 196.00, dur: 1.6 },
    #{ name: "Cmaj", freq: 130.81, dur: 1.6 },
    #{ name: "Fmaj", freq: 174.61, dur: 1.6 },
    #{ name: "Bdim", freq: 246.94, dur: 1.6 },
    #{ name: "E7",   freq: 164.81, dur: 1.6 },
    #{ name: "Am",   freq: 110.00, dur: 1.8 }
];

for chord in chords {
    spawn("bass_" + chord.name, 1, #{
        body: #{ amp: 0.38 },
        pitch: #{ center_hz: chord.freq }
    });
    spawn("pad_" + chord.name, 2, #{
        body: #{ amp: 0.14 },
        pitch: #{ center_hz: chord.freq * 2.0 }
    });
    wait(chord.dur);
}

end();
