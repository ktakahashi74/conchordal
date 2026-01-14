print("Spectral Gap: weave through silence");

let method = #{ mode: "spectral_gap", min_freq: 120.0, max_freq: 2200.0 };

scene("stitch");
for i in 0..24 {
    spawn(
        "gap_" + i,
        1,
        #{ method: method },
        #{
            body: #{ amp: 0.18, method: "sine" },
            phonation: #{ type: "interval", density: 0.6, legato: 0.1 }
        }
    );
    wait(0.16);
}

wait(1.2);
end();
