print(">>> Harmonic Circle: Spread Voicing <<<");

// ==========================================
// 音色定義
// ==========================================
let life_drone = #{
    type: "decay",
    initial_energy: 10.0,
    half_life_sec: 8.0,
    attack_sec: 2.0
};

let life_pulse = #{
    type: "decay",
    initial_energy: 2.0,
    half_life_sec: 0.6,
    attack_sec: 0.01
};

// ==========================================
// Phase 1: Gravity (調性の確立)
// ==========================================
section("Phase 1: Gravity");
print("--- [1] Establishing Harmonic Root (C2 + G2) ---");

add_agent("root", 65.41, 0.5, life_drone); // C2
wait(1.0);
add_agent("fifth", 98.00, 0.4, life_drone); // G2
wait(1.0);

// ==========================================
// Phase 2: Rhythm Driver
// ==========================================
section("Phase 2: The Clock");
print("--- [2] Injecting Rhythm Driver ---");

add_agent("kick", 65.41, 0.6, #{ 
    type: "decay",
    initial_energy: 5.0,
    half_life_sec: 0.8,
    attack_sec: 0.005
});
wait(3.0);

// ==========================================
// Phase 3: The Chord (確率的分散)
// ==========================================
section("Phase 3: Voicing");
print("--- [3] Spawning agents with Harmonic Density ---");

// 【変更点】 harmonicity -> harmonic_density
// temperature: 
//   0.1 = ほぼ最大値しか選ばない（Unison）
//   1.0 = 協和度に従って自然にバラける（Natural）
//   2.0 = 協和度が低いところもたまに選ぶ（Jazzy/Noise）
let method_spread = #{ 
    mode: "harmonicity", 
    min_freq: 130.0, // C3
    max_freq: 1050.0, // C6付近まで広げる
    temperature: 0.5 // 少し鋭めに（きれいな和音を狙う）
};

for i in 0..6 {
    // 6体を投入。確率的に C, E, G, B などの倍音に散らばるはず
    spawn_agents("voice_" + i, method_spread, life_pulse, 1, 0.22);
    print(" -> Add Voice " + i);
    wait(1.0); 
}

print("--- Listening to the chord... ---");
wait(5.0);

// ==========================================
// Phase 4: Autonomy
// ==========================================
section("Phase 4: Autonomy");
print("--- [4] Removing Anchor & Kick ---");

remove("kick");
// ドローンも消して、残った6体の相互作用だけで和音が保たれるか実験
// （harmonic_densityで選ばれた音は物理的に協和しているので、ドローンが消えても響き合うはず）
set_amp("root", 0.0);
set_amp("fifth", 0.0);

wait(10.0);
finish();
