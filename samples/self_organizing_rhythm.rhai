print(">>> Experiment: Self-Organizing Rhythm <<<");

// ============================================================
// 設定: 同期を聞き取りやすくするための「パーカッシブ」な設定
// ============================================================

// 長く生きるが、音自体は短い「プラック（撥弦）音」
// アタックが鋭いため、タイミングのズレや同期が明確に聞こえます。
let life_plucking = #{
    type: "sustain",
    initial_energy: 1.5,
    metabolism_rate: 0.05, // なかなか死なない
    envelope: #{ 
        attack_sec: 0.005, // 非常に鋭いアタック
        decay_sec: 0.1,    // すぐ減衰
        sustain_level: 0.0 // 音は切れる（リズムの「点」を作る）
    }
};

// ============================================================
// Phase 1: The Seed (メトロノームによる強制引き込み)
// ============================================================
scene("Phase 1: Metronome");
print("--- [1] Injecting strong beat (Delta band driver) ---");

// 1. 強力なキック音 (60Hz) を配置
// これがランドスケープの Low/Delta 帯域を激しく揺らし、リズムの基準を作ります
add_agent("kick", 60.0, 0.5, #{ 
    type: "sustain", 
    initial_energy: 2.0, 
    metabolism_rate: 0.0, // 死なない
    envelope: #{ attack_sec: 0.01, decay_sec: 0.2, sustain_level: 0.0 }
});

wait(2.0); // リズムが安定するのを待つ

// ============================================================
// Phase 2: The Swarm (群れの投入)
// ============================================================
scene("Phase 2: Recruitment");
print("--- [2] Spawning agents with random phases ---");

// 2. 多数の個体を投入 (200Hz - 800Hz)
// 最初は位相がバラバラなので「ザワザワ」した音になりますが、
// 徐々にキックのリズム（Delta帯域）に引き込まれていきます。
let method_flock = #{ 
    mode: "harmonic_density", 
    min_freq: 200.0, 
    max_freq: 800.0, 
    temperature: 1.0 
};

// 4回に分けて投入し、層を厚くする
for i in 0..4 {
    spawn_agents("follower", method_flock, life_plucking, 4, 0.08);
    wait(1.0); // 次の投入まで少し待つ
}

print("--- Listening for synchronization... ---");
wait(4.0); // 同期が進むのをじっくり待つ

// ============================================================
// Phase 3: Emergence (自律維持)
// ============================================================
scene("Phase 3: Emergence");
print("--- [3] REMOVING THE METRONOME (Watch self-sustain) ---");

// 3. 親（キック）を削除
// これにより「正解」がなくなります。
// 個体たちは「仲間の音」だけを頼りに、リズムを維持しようとします。
remove("kick");

// ここでリズムが崩壊せず、少し揺らぎながらも「グルーヴ」が残れば成功です。
// まさに「盆踊り」のように、中心がなくなっても輪が回り続ける現象です。
wait(6.0);

// ============================================================
// Phase 4: Perturbation (撹乱と再同期)
// ============================================================
scene("Phase 4: Perturbation");
print("--- [4] Injecting counter-rhythm (High freq) ---");

// 4. 全く違うテンポ感の音（高域のクリック音）を入れて撹乱してみる
// High/Alpha帯域が刺激され、既存の個体のタイミングが微調整（Micro-timing）されるか観察
let life_click = #{ type: "decay", initial_energy: 0.8, half_life_sec: 0.2 };
let method_click = #{ mode: "random_log_uniform", min_freq: 2000.0, max_freq: 4000.0 };

for i in 0..8 {
    spawn_agents("noise", method_click, life_click, 2, 0.1);
    wait(0.25); // 早いパルス
}

wait(3.0); // 撹乱後の落ち着きを確認

// 終了
finish();
