print("Harmonicity demo: climb into peaks, then settle");

// Seed: One-shot long decay
let seed_method = #{ mode: "random_log_uniform", min_freq: 330.0, max_freq: 330.0 };
spawn("seed", 1, #{ method: seed_method }, #{ body: #{ amp: 0.24 } });
wait(0.5);

let method = #{ mode: "harmonicity", min_freq: 200.0, max_freq: 1200.0 };

for i in 0..6 {
    spawn(
        "harm_" + i,
        2,
        #{ method: method },
        #{ body: #{ amp: 0.16 + i * 0.01 } }
    );
    wait(0.7);
}

wait(1.5);
end();
