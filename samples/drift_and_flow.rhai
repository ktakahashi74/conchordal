print(">>> Drift & Flow: Physics Verification Demo <<<");

// ==========================================
// Phase 1: Gravity Test (吸着実験)
// ==========================================
scene("Phase 1: Gravity Pull");
print("--- [1] Anchoring C2 (65.41Hz) ---");

// ベース音 (不動のアンカー)
add_agent("anchor", 65.41, 0.6, #{ 
    type: "sustain", 
    initial_energy: 10.0, 
    metabolism_rate: 0.0,
    envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 1.0 }
});
// アンカーは動かないように設定
set_drift("anchor", 0.0);
set_habituation_sensitivity("anchor", 0.0);

wait(1.0);

print("--- Injecting Dissonant Tone (C#3) -> Watch it slide to C3 or E3 ---");
// C3 (130.81) の半音上、C#3 (138.59) を投入
// 強力な Drift Rate を与え、青矢印（引力）によって C3 または E3 に吸い込まれるか確認
add_agent("slider", 138.59, 0.4, #{ type: "sustain", initial_energy: 2.0, metabolism_rate: 0.0, envelope: #{ attack_sec: 0.5, decay_sec: 0.5, sustain_level: 1.0 } });

set_drift("slider", 2.0); // 高めの値で明確に動かす
set_habituation_sensitivity("slider", 0.0); // まずは飽きさせない

wait(4.0);

// ==========================================
// Phase 2: Habituation Test (飽きの実験)
// ==========================================
scene("Phase 2: Habituation Push");
print("--- [2] Enabling Habituation (Red Arrows appear) ---");

// 環境の「飽きやすさ」を強化
// 赤矢印が slider の下から伸びてくるはず
set_habituation(3.0, 2.0); // weight=3.0 (強烈), tau=2.0s (早い)

// エージェントの感受性をON
set_habituation_sensitivity("slider", 1.0);

print("--- Slider should be pushed away by red arrow ---");
// 飽き（赤矢印）が蓄積すると、ポテンシャルの山が削られ、谷になる。
// 結果、エージェントは「居心地が悪く」なり、隣の倍音へ移動し始めるはず。
wait(6.0);

// ==========================================
// Phase 3: Mass Migration (転調と追従)
// ==========================================
scene("Phase 3: Modulation");
print("--- [3] Spawning Swarm & Moving Anchor ---");

remove("slider");
// 環境をリセット気味に
set_habituation(0.5, 5.0); // マイルドに戻す

// 群れを投入 (Cの倍音上に分布)
let method = #{ mode: "harmonicity", min_freq: 130.0, max_freq: 800.0 };
let life = #{ type: "sustain", initial_energy: 5.0, metabolism_rate: 0.05, envelope: #{ attack_sec: 1.0, decay_sec: 1.0, sustain_level: 0.5 }};

for i in 0..5 {
    spawn_agents("swarm_" + i, method, life, 1, 0.15);
    set_drift("swarm_" + i, 0.8); // ぬるっと動く設定
    set_habituation_sensitivity("swarm_" + i, 1.5); // 敏感
    wait(0.5);
}

wait(2.0);

print(">>> MOVING ANCHOR to F2 (87.31Hz) <<<");
print(">>> Watch the blue arrows flip direction! <<<");

// アンカーを F2 へ移動。
// これによりランドスケープの地形が激変し、Cメジャーの安定点が消滅、Fメジャーの安定点が出現。
// 群れが一斉に F の倍音 (A, C, Eb...) へ向かってドリフトするはず。
set_freq("anchor", 87.31); 

wait(8.0);

print(">>> Releasing Anchor (Dissolution) <<<");
remove("anchor");
// アンカーが消えると地形が平坦になり、群れは拡散・消滅に向かう
wait(5.0);

finish();
