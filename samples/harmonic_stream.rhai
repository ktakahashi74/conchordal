print("Harmonic Density Stream: 50 notes");

// Strategy: harmonic density sampling (uses current consonance as a PDF)
let method = #{
    mode: "harmonic_density",
    min_freq: 150.0,
    max_freq: 1500.0,
    temperature: 1.5, // wider spread for variety
};

// Lifecycle: sustain, designed to exhaust energy around 2 seconds
let life = #{
    type: "sustain",
    initial_energy: 1.0,
    metabolism_rate: 0.5, // consumes 0.5 per second
    envelope: #{
        attack_sec: 0.1,
        decay_sec: 0.5,
        sustain_level: 0.8
    }
};

section("stream");

// Seed tone so density has something to latch onto
spawn_agents("base", method, life, 1, 0.14);
wait(0.25);

// 50-note stream
for i in 0..200 {
    spawn_agents("voice", method, life, 1, 0.08);

    // Reinforce base every 10 notes to keep the field consonant
    if i % 10 == 0 {
        spawn_agents("base_layer", method, life, 1, 0.12);
    }

    wait(0.15);
}

// Let the tail ring out
wait(3.0);
finish();
