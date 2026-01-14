// Mixed strategies in one sweep (quick listening tour)
spawn(
    "rnd",
    3,
    #{ method: #{ mode: "random_log_uniform", min_freq: 200.0, max_freq: 1500.0 } },
    #{ body: #{ amp: 0.18 } }
);
wait(0.9);
spawn(
    "harm",
    2,
    #{ method: #{ mode: "harmonicity", min_freq: 260.0, max_freq: 1200.0 } },
    #{ body: #{ amp: 0.22 } }
);
wait(0.8);
spawn(
    "gap",
    2,
    #{ method: #{ mode: "spectral_gap", min_freq: 220.0, max_freq: 1400.0 } },
    #{ body: #{ amp: 0.18 } }
);
wait(0.9);
spawn(
    "hdens",
    3,
    #{ method: #{ mode: "harmonic_density", min_freq: 160.0, max_freq: 1400.0, temperature: 0.7 } },
    #{ body: #{ amp: 0.2 } }
);
wait(0.8);
spawn(
    "zero",
    2,
    #{ method: #{ mode: "zero_crossing", min_freq: 180.0, max_freq: 1600.0 } },
    #{ body: #{ amp: 0.18 } }
);
wait(0.8);
spawn(
    "lowharm",
    2,
    #{ method: #{ mode: "low_harmonicity", min_freq: 200.0, max_freq: 1500.0 } },
    #{ body: #{ amp: 0.18 } }
);
wait(1.5);
end();
