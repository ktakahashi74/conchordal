print(">>> Kuramoto Verification: Entrainment & Self-Organization <<<");

// ==========================================
// 設定: 聞き取りやすい「クリック音」
// ==========================================
// 減衰が速い音（Pluck）にすることで、発音タイミングのズレが明確に分かります。
let life_click = #{
    type: "sustain",
    initial_energy: 1.5,
    metabolism_rate: 0.0,
    envelope: #{ 
        attack_sec: 0.005, // 瞬時のアタック
        decay_sec: 0.1,    // 短い減衰
        sustain_level: 0.0 // 音を残さない
    }
};

// ==========================================
// Phase 1: The Metronome (外部強制力)
// ==========================================
scene("Phase 1: External Driver");
print("--- [1] Metronome starts (Listen for the beat) ---");

// 外部からの強力なパルスとして、定期的に「重いキック音」を投入します。
// これが環境の Delta波（0.5-4Hz）を強制的に揺らし、同期の「正解」を作ります。
// ※ Individualの自律リズムではなく、システム側で正確に刻むイベントです。
let kick_life = #{ type: "decay", initial_energy: 1.5, half_life_sec: 0.1 };

// 0.6秒間隔（BPM 100）で10回鳴らすループを並行実行
parallel(|| {
    for i in 0..20 {
        add_agent("kick", 60.0, 0.8, kick_life);
        wait(0.6);
    }
});

wait(2.0); // メトロノームが定着するのを少し待つ

// ==========================================
// Phase 2: The Swarm (追従者の投入)
// ==========================================
scene("Phase 2: Entrainment");
print("--- [2] Spawning autonomous agents (Watch them sync) ---");

// 自律エージェントを投入します。
// 彼らの内部時計（rhythm_freq）は 0.5Hz〜3.0Hz の間でランダムですが、
// 環境で鳴っている「0.6秒間隔のキック」のDelta波に引き込まれ、
// 次第にそのタイミングに合わせて鳴くようになります。

let method_random = #{ 
    mode: "random_log_uniform", 
    min_freq: 400.0, 
    max_freq: 1200.0 
};

// 5体を投入
spawn_agents("follower", method_random, life_click, 5, 0.2);

print(">>> Observing synchronization... (Wait 10s)");
// ここで、バラバラだった高音のクリック音が、低音のキックと揃い始めるのを確認してください。
wait(10.0);

// ==========================================
// Phase 3: Autonomy (メトロノーム消失)
// ==========================================
scene("Phase 3: Self-Organization");
print("--- [3] METRONOME STOPS. (Agents maintain the groove) ---");

// メトロノーム（ループ）が終了し、静寂が訪れます。
// しかし、同期したエージェントたちが互いの音（Delta波の残響）を感じ取り合い、
// リズムを崩さずに維持（または集団で緩やかに変速）し続ける「創発的同期」を確認します。

// エージェントの寿命が尽きるまで観察
wait(10.0);

finish();
