print("Harmonicity demo: climb into peaks, then settle");

spawn("seed", 1, #{ body: #{ amp: 0.24 }, pitch: #{ center_hz: 330.0 } });
wait(0.5);

for i in 0..6 {
    spawn("harm_" + i, 2, #{
        body: #{ amp: 0.16 + i * 0.01 },
        pitch: #{ center_hz: 440.0 }
    });
    wait(0.7);
}

wait(1.5);
end();
