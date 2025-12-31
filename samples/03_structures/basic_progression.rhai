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
        body: #{ core: "sine" },
        temporal: #{
            core: "entrain",
            type: "sustain", 
            initial_energy: 10.0, 
            metabolism_rate: 0.0, 
            envelope: #{ attack_sec: 0.1, decay_sec: 0.1, sustain_level: 0.5 }
        },
        field: #{ core: "pitch_hill_climb" },
        modulation: #{ core: "static", persistence: 1.0, exploration: 0.0 },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };

    // 2. Pad (Seeker): 動きやすい、分散する
    let pad_life = #{ 
        body: #{ core: "sine" },
        temporal: #{
            core: "entrain",
            type: "sustain", 
            initial_energy: 5.0, 
            metabolism_rate: 0.0, 
            envelope: #{ attack_sec: 1.0, decay_sec: 0.5, sustain_level: 0.6 }
        },
        field: #{ core: "pitch_hill_climb" },
        modulation: #{ core: "static", persistence: 0.3, exploration: 0.0 },
        perceptual: #{ tau_fast: 0.5, tau_slow: 20.0, w_boredom: 1.0, w_familiarity: 0.2, rho_self: 0.15, boredom_gamma: 0.5, self_smoothing_radius: 1, silence_mass_epsilon: 0.000001 }
    };

    print("Playing: " + name);

    // ベース（アンカー）を配置
    let bass = add_agent("bass_" + name, root, 0.6, bass_life);
    
    // ランドスケープが形成されるのを少し待つ
    wait(0.2); 

    // パッドを配置（自律的に和音構成音を探させる）
    // ルートの少し上から配置し、GravityとConsonanceで落ち着かせる
    let pad_1 = add_agent("pad_" + name + "_1", root * 1.5, 0.3, pad_life); // 5th付近
    let pad_2 = add_agent("pad_" + name + "_2", root * 2.0, 0.3, pad_life); // Octave付近
    let pad_3 = add_agent("pad_" + name + "_3", root * 1.25, 0.3, pad_life); // 3rd付近 (Major/Minorは地形で決まる)

    wait(dur);

    // クリーンアップ
    bass.remove();
    pad_1.remove();
    pad_2.remove();
    pad_3.remove();
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
finish();
