print("Basic progression");

let chords = [
    #{ name: "Am",   freq: 110.00, dur: 3.0 },
    #{ name: "Dm",   freq: 146.83, dur: 3.0 },
    #{ name: "G7",   freq: 196.00, dur: 3.0 },
    #{ name: "Cmaj", freq: 130.81, dur: 3.0 },
    #{ name: "Fmaj", freq: 174.61, dur: 3.0 },
    #{ name: "Bdim", freq: 246.94, dur: 3.0 },
    #{ name: "E7",   freq: 164.81, dur: 3.0 },
    #{ name: "Am",   freq: 110.00, dur: 4.0 }
];

for chord in chords {
    spawn("bass_" + chord.name, 1, #{
        body: #{ amp: 0.6 },
        pitch: #{ center_hz: chord.freq }
    });
    spawn("pad_" + chord.name, 2, #{
        body: #{ amp: 0.3 },
        pitch: #{ center_hz: chord.freq * 2.0 }
    });
    wait(chord.dur);
}

end();
