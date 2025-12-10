print("Mini composition: pulse -> bloom -> dissolve");

// Simple Euclidean-ish pattern helper (quarter-ish grid)
fn pulse(tag, freq, amp, count, spacing) {
    for i in 0..count {
        add_agent(tag + "_" + i, freq, amp, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.18 });
        wait(spacing);
    }
}

section("pulse");
pulse("kick", 60.0, 0.45, 6, 0.25); // drives the field

section("bloom");
parallel(|| { // bass anchor + slow pads
    add_agent("bass", 96.0, 0.35, #{ type: "decay", initial_energy: 1.1, half_life_sec: 0.5, attack_sec: 0.05 });
    wait(0.4);
    add_agent("pad", 192.0, 0.28, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.7, attack_sec: 0.4 });
});
parallel(|| { // shimmer hits
    pulse("shimmer", 1200.0, 0.12, 8, 0.125);
});
wait(1.0);

section("dissolve");
let melt = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.35 };
let drift = #{ mode: "harmonic_density", min_freq: 200.0, max_freq: 1400.0, temperature: 1.0 };
for i in 0..6 {
    spawn_agents("drift", drift, melt, 2, 0.16);
    wait(0.2);
}
wait(1.0);
finish();
