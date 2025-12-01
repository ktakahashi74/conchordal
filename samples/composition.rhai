print("Composition Start (No BPM)");

// Functions
fn pattern_kick() {
    let life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.1 };
    for i in 0..4 {
        add_agent("kick", 60.0, 0.5, life);
        wait(0.5);
    }
}

fn pattern_bass() {
    let life = #{ type: "sustain", initial_energy: 1.0, metabolism_rate: 0.1, envelope: #{ attack_sec: 0.05, decay_sec: 0.2, sustain_level: 0.5 } };
    add_agent("bass", 110.0, 0.3, life);
    wait(1.0);
    add_agent("bass", 165.0, 0.3, life); // 5th
    wait(1.0);
}

// --- Section 1 ---
section("intro");

parallel(|| {
    pattern_kick();
});

parallel(|| {
    pattern_bass();
});

// Time advanced inside parallel blocks; advance here to set section length
wait(2.0);

// --- Section 2 ---
section("break");
add_agent("cymbal", 2000.0, 0.1, #{ type: "decay", initial_energy: 1.0, half_life_sec: 1.0 });
wait(2.0);

finish();
