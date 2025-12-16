print("Harmonic Density Stream: flowing 1/f voices");

// PDF driven by the landscape; wide window to let consonance peaks pull.
let method = #{ mode: "harmonic_density", min_freq: 70.0, max_freq: 2600.0, temperature: 0.5, crowding: 2.0 };
let life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.6, attack_sec: 0.08 };

scene("seed");
spawn_agents("anchor", method, life, 1, 0.12);
wait(0.35);

scene("flow");

// set_harmonicity(#{ mirror: 1.0});  // major/minor

for i in 0..48 {
    // occasional louder anchors to re-center the PDF
    if i % 12 == 0 { spawn_agents("anchor", method, life, 1, 0.16); }
    spawn_agents("voice_" + i, method, life, 1, 0.09);
    wait(0.12);
}

wait(1.0);
finish();
