print(">>> Cycle of Fifths (Organic Update) <<<");

// ヘルパー: 指定範囲のランダムな周波数を返す（簡易版）
fn random_offset(base, amount) {
    // Rhaiにrand()がない場合を想定し、システム時間等は使えないため
    // 決定論的な擬似ランダム、または単純なオフセットを使用します。
    return base + amount; 
}

fn play_step(name, root, dur) {
    // 1. Bass (Anchor): 動かない、重い、飽きない
    let bass_life = #{ 
        type: "sustain", 
        initial_energy: 10.0, 
        metabolism_rate: 0.0, 
        envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 0.5 },
        // 新しい有機的パラメータ
        commitment: 1.0,              // 絶対に動かない
        habituation_sensitivity: 0.0, // 飽きない（場所に留まる）
        crowding_sensitivity: 0.0     // 混雑を気にしない
    };

    // 2. Pad (Seeker): 動きやすい、分散する
    let pad_life = #{ 
        type: "sustain", 
        initial_energy: 5.0, 
        metabolism_rate: 0.0, 
        envelope: #{ attack_sec: 1.0, decay_sec: 0.5, sustain_level: 0.6 },
        // 新しい有機的パラメータ
        commitment: 0.3,              // 協和点を探して動き回る
        habituation_sensitivity: 1.0, // 同じ場所にいると飽きる
        crowding_sensitivity: 2.0     // 他と重なるのを嫌う（和音を作る）
    };

    print("Playing: " + name);

    // ベース（アンカー）を配置
    add_agent("bass_" + name, root, 0.6, bass_life);
    
    // ランドスケープが形成されるのを少し待つ
    wait(0.2); 

    // パッドを配置（自律的に和音構成音を探させる）
    // ルートの少し上から配置し、GravityとConsonanceで落ち着かせる
    add_agent("pad_" + name + "_1", root * 1.5, 0.3, pad_life); // 5th付近
    add_agent("pad_" + name + "_2", root * 2.0, 0.3, pad_life); // Octave付近
    add_agent("pad_" + name + "_3", root * 1.25, 0.3, pad_life); // 3rd付近 (Major/Minorは地形で決まる)

    wait(dur);

    // クリーンアップ
    remove("bass_" + name);
    remove("pad_" + name + "_1");
    remove("pad_" + name + "_2");
    remove("pad_" + name + "_3");
}

let chords = [
    #{ name: "Am",   freq: 110.00, dur: 3.0 },
    #{ name: "Dm",   freq: 146.83, dur: 3.0 },
    #{ name: "G7",   freq: 196.00, dur: 3.0 },
    #{ name: "Cmaj", freq: 130.81, dur: 3.0 },
    #{ name: "Fmaj", freq: 174.61, dur: 3.0 },
    #{ name: "Bdim", freq: 246.94, dur: 3.0 },
    #{ name: "E7",   freq: 164.81, dur: 3.0 },
    #{ name: "Am",   freq: 110.00, dur: 4.0 },
];

scene("cycle_organic");

// ランドスケープ設定（飽きの早さなどを調整）
// set_habituation_params(weight, tau, max_depth)
set_habituation_params(0.5, 4.0, 1.0); 

for ch in chords {
    play_step(ch.name, ch.freq, ch.dur);
}

wait(2.0);
finish();
