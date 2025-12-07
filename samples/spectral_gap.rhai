print("Spectral Gap Filling Demo");

// "spectral_gap" モード:
// 現在の音響エネルギーが最も低い（静かな）周波数帯域を探して生成します。
// 結果として、音が重ならず、互いに隙間を埋め合うような配置になります。
let method = #{
    mode: "spectral_gap",
    min_freq: 100.0,
    max_freq: 2000.0
};

// 減衰音（短い音）
let life = #{
    type: "decay",
    initial_energy: 1.0,
    half_life_sec: 0.3
};

section("main");

// 0.15秒ごとに、その瞬間の「隙間」に音を置く
for i in 0..30 {
    // タグ "gap", 1個生成, 振幅 0.2
    spawn_agents("gap", method, life, 1, 0.2);
    wait(0.15);
}

wait(2.0);
finish();
