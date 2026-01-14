print(">>> Cycle of Fifths (Organic Update) <<<");

// ヘルパー: 指定範囲のランダムな周波数を返す（簡易版）
fn random_offset(base, amount) {
    // Rhaiにrand()がない場合を想定し、システム時間等は使えないため
    // 決定論的な擬似ランダム、または単純なオフセットを使用します。
    return base + amount; 
}

fn play_step(name, root, dur) {
    // 1. Bass (Anchor): 動かない、重い、飽きない
    let bass_patch = #{ body: #{ amp: 0.6, method: "sine" }, phonation: #{ type: "hold" } };

    // 2. Pad (Seeker): 動きやすい、分散する
    let pad_patch = #{ body: #{ amp: 0.3, method: "sine" }, phonation: #{ type: "hold" } };

    print("Playing: " + name);

    // ベース（アンカー）を配置
    let bass_method = #{ mode: "random_log_uniform", min_freq: root, max_freq: root };
    let bass_tag = "bass_" + name;
    spawn(bass_tag, 1, #{ method: bass_method }, bass_patch);
    
    // ランドスケープが形成されるのを少し待つ
    wait(0.2); 

    // パッドを配置（自律的に和音構成音を探させる）
    // ルートの少し上から配置し、GravityとConsonanceで落ち着かせる
    let pad_1_method = #{ mode: "random_log_uniform", min_freq: root * 1.5, max_freq: root * 1.5 };
    let pad_2_method = #{ mode: "random_log_uniform", min_freq: root * 2.0, max_freq: root * 2.0 };
    let pad_3_method = #{ mode: "random_log_uniform", min_freq: root * 1.25, max_freq: root * 1.25 };
    let pad_1_tag = "pad_" + name + "_1";
    let pad_2_tag = "pad_" + name + "_2";
    let pad_3_tag = "pad_" + name + "_3";
    spawn(pad_1_tag, 1, #{ method: pad_1_method }, pad_patch); // 5th付近
    spawn(pad_2_tag, 1, #{ method: pad_2_method }, pad_patch); // Octave付近
    spawn(pad_3_tag, 1, #{ method: pad_3_method }, pad_patch); // 3rd付近 (Major/Minorは地形で決まる)

    wait(dur);

    // クリーンアップ
    remove(bass_tag);
    remove(pad_1_tag);
    remove(pad_2_tag);
    remove(pad_3_tag);
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

for ch in chords {
    play_step(ch.name, ch.freq, ch.dur);
}

wait(2.0);
end();
