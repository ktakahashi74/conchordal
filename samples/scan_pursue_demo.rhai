print(">>> Scan & Pursue Demo: Intelligent Harmony <<<");

// ==========================================
// Phase 1: 安定したドローン (C Major Context)
// ==========================================
scene("Phase 1: Anchor C2");
print("--- [1] Anchoring C2 (65.41Hz) ---");

// アンカー（動かない）
add_agent("anchor", 65.41, 0.6, #{
    type: "sustain",
    initial_energy: 10.0,
    metabolism_rate: 0.0,
    envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 1.0 }
});
set_drift("anchor", 0.0);

wait(1.0);

// ==========================================
// Phase 2: 探索者の投入 (不協和音 -> 解決)
// ==========================================
scene("Phase 2: The Seekers");
print("--- [2] Spawning Seekers at Dissonant Intervals ---");

// C2 (65.41) に対して、以下の不協和音程でエージェントを投入します。
// F# (Tritione): 92.5Hz -> おそらく G (98.0Hz) か F (87.3Hz) に解決するはず
// C# (Minor 2nd): 69.3Hz -> おそらく C (65.4Hz) か D (73.4Hz) に解決するはず

let life_seeker = #{
    type: "sustain",
    initial_energy: 2.0,
    metabolism_rate: 0.0,
    envelope: #{ attack_sec: 1.0, decay_sec: 1.0, sustain_level: 0.5 }
};

// 意図的に「少しズレた場所」に置く
add_agent("seeker_tritone", 92.5, 0.4, life_seeker);
add_agent("seeker_m2", 140.0, 0.4, life_seeker); // C3(130)の近く

// 【重要】ここで「追従速度」を設定
// 値が大きいほどキビキビ動く。小さいとポルタメントがかかる。
set_drift("seeker_tritone", 1.0);
set_drift("seeker_m2", 0.5); // こちらはゆっくり動く

print(">>> Observe: They should slide to the nearest consonant peak (G or C) <<<");
wait(5.0);

// ==========================================
// Phase 3: 地殻変動 (転調)
// ==========================================
scene("Phase 3: Modulation to F");
print("--- [3] Moving Anchor to F2 (87.31Hz) ---");

// アンカーを強制移動。ランドスケープの地形が一変する。
// 以前の「正解」だった場所（C, G）は、今は不協和な場所（Fに対しての2度など）になる可能性がある。
set_freq("anchor", 87.31);

// エージェントは定期的に周囲をスキャンしているため、
// 「あ、こっち（Fの倍音）の方が居心地が良い」と気づき、
// ターゲットを更新して移動し始めるはず。

wait(8.0);

// ==========================================
// Phase 4: 群れの挙動 (Harmonic Density)
// ==========================================
scene("Phase 4: Swarm Intelligence");
print("--- [4] Spawning a swarm ---");

// 多数の個体を投入。お互いが作るピークも探索対象になるため、
// 互いに集まってコードを形成する（Cluster）か、分散するかが観察できる。
let method = #{ mode: "harmonic_density", min_freq: 200.0, max_freq: 800.0 };
for i in 0..4 {
    spawn_agents("swarm_" + i, method, life_seeker, 1, 0.15);
    set_drift("swarm_" + i, 0.8);
    wait(0.5);
}

wait(5.0);
finish();
