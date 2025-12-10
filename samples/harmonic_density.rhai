print("Harmonic Density drift (temperature-biased)");

// Seed a light anchor so the PDF stabilizes
add_agent("anchor", 330.0, 0.18, #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.9 });
wait(0.5);

let method = #{ mode: "harmonic_density", min_freq: 160.0, max_freq: 1500.0, temperature: 0.9 };
let life = #{ type: "decay", initial_energy: 1.0, half_life_sec: 0.7, attack_sec: 0.02 };

// Four waves of density, slowly cooling the PDF
for i in 0..4 {
    let amp = 0.14 + i * 0.02;
    spawn_agents("hdens_" + i, method, life, 3, amp);
    wait(0.6);
}

wait(1.2);
finish();
