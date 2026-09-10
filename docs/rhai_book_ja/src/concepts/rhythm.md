# リズム：時間の知覚と個体の参加

拍は、音から推定する反復的な時間の基準です。同じ音を参照するVoiceが、同じ拍位置で
発音する必要はありません。三つのプリセットは、異なる参加の仕方を選びます。

- **metric** — 共有された拍への同期を意図します。結合を強めるほど発音位置が拍へ近づきます。
- **entrained** — 自身のペースを持ち、参加時に記憶した周囲の音の特徴と、音が重なる予測から発音時刻を調整します。
- **flow** — 不規則な発音間隔の基準を保ち、弱い結合で周囲の音に応答します。基準の速さを反復周期へ寄せません。

entrainedとflowは、過去のhabitat音声から得た短い未来の予期を使います。Voice自身が
合成した音も記録し、自分の反復を競合相手として扱わないようにします。発音予定の近傍で
鳴る、参加の1周期まで待つ、今回は見送る、を比較します。重なりは予定する保持時間と
ADSRのreleaseを含む範囲で評価します。見送りを続けるほどコストが増すため、重なりだけを
理由に永続的な沈黙へ落ち着くことはありません。実際に発音を指令したときだけ周囲との
関係を記憶します。証拠が乏しいときは
身体のペースを保ち、共通位相へ置き換えません。重なりの評価は三つの帯域のエネルギーを
使う工学的な近似であり、音楽的な受入は検証中です。

## Director水準の地形

演出者は次の操作で、metricが参照する生成側の拍の地形を調整します。
entrainedとflowの身体の速さや、反復予測の周期候補を直接指定する操作ではありません。

- `meter_stability(value)` — `[0,1]`のattractor depth。実在するperiodicityに対する
  basinを深くするだけで、非metric inputからbeatを捏造しない。
- `temporal_basin(min_hz, max_hz)` — 創発beatが引き寄せられるtempo領域。
  `consonance(min, max)`の時間軸版で、beatを配置したりmeasureを強制したりはしない。

## Voiceごとのプリセットと調整

`metric()`、`entrained()`、`flow()`は速さの引数を取りません。`place()`より前に
PopulationSpecへ設定します。metricは地形が誘導する拍へ近づきます。entrainedとflowは
2 Hzの固有ペースから始まります。entrainedは周囲の反復から参加の周期を更新します。
flowは不規則な間隔を作る基準を保ち、周囲の音に応じて待つ・見送る判断を続けます。
実際の発音間隔は変わり得ますが、全Voiceが従う共通テンポや発音位置は指定しません。

`.entrained()`と`brain("entrain")`を混同しないでください。前者はonset timing、後者は
Voiceの発音上の生と代謝を選びます。独立しており併用できます。

Voiceごとの設定で、参加の仕方を調整できます。

- `entrainment(strength)` — `[0,1]`の影響の強さ。entrainedとflowでは音から得た文脈、metricでは拍位相への引力に作用します。`0`なら固有のペースを保ちます。
- `rhythm_role("beat"|"subdivision"|"accent"|"texture")` — metrical role。
  `accent`は共有meterを強く駆動するonsetを出し、反復downbeatからmeasureを創発できる。
- `microtiming(amount)` — metricの目標位置を拍位相でずらします。範囲は`[-0.5, 0.5]`、`0.5`は半拍です。entrainedとflowへ固定された発音位置を割り当てる設定ではありません。
- `measure_accent(amount)` — 検出したmeasureに沿って、発音ごとの強弱を弱く変える設定です。
  範囲は`[0,1]`、既定値は`0`。強度の倍率は`1 + 0.35 * amount * confidence * cos(phase)`です。
  位相の原点は観測した強拍に合わせ、measureが未検出なら強度を変えません。
  発音時刻は指定せず、`rhythm_role`とは独立して働きます。

```rhai
meter_stability(0.85);     // attractor depth: how readily a pulse forms
temporal_basin(1.8, 2.2);   // tempo region the emergent beat gravitates toward

let beat = harmonic()
    .metric()
    .rhythm_role("accent")  // a strong onset that drives the shared beat
    .cycles(2);

let entrained = harmonic()
    .entrained()
    .cycles(2);

let drift = harmonic()
    .flow()
    .cycles(1);

let offbeat = harmonic()
    .metric()
    .microtiming(0.5)       // a half-beat offset reads as syncopation
    .cycles(2);

place(beat, at(110.0));
place(entrained, consonance(110.0).peak().count(3));
place(drift, consonance(300.0, 1200.0).count(4));
place(offbeat, at(220.0));
wait(12.0);
```

同じ軸では最後の指定が優先されます。調整値は記憶され、対応するプリセットを選んだ時に適用されるため、
`entrainment(0.8).metric()`と`metric().entrainment(0.8)`は同等です。
`duration_range(...).adaptive_duration()`と逆順も同様です。

## 明示的なwhen/duration（Tier 2）

presetの下には`once()`、`pulse(rate_hz)`、`while_alive()`、`cycles(n)`、
`adaptive_duration()`があり、`duration_range`、`duration_curve`、`shorten_on_drop`で
調整できます。

## 生存としてのtiming（Tier 3）

既存のアーティキュレーション制御には、`rhythm_coupling_vitality(lambda_v, v_floor)`、
`rhythm_reward(rho_t, "attack_phase_match")`、`rhythm_freq(freq_hz)`があります。
Gated発音では位相一致報酬を使いません。下の例でも、この報酬による同期は起きません。

```rhai
let pulse_voice = harmonic()
    .repeat()
    .pulse(2.0)
    .cycles(2)
    .rhythm_freq(2.0)
    .rhythm_coupling_vitality(0.8, 0.4)
    .rhythm_reward(0.4, "attack_phase_match");

place(pulse_voice, at(165.0));
wait(8.0);
```

## Scaffold（research control）

scaffold functionは比較assay用の外部pulseを強制します。

```rhai
set_scaffold_off();
set_scaffold_shared(2.0);
set_scaffold_scrambled(2.0, 17);
```

同期が比較や実演に不可欠な場合だけ使います。entrainedとflowの参加を成立させる補助として
専用の拍打ちを追加しません。

## 身体の周期と既存の代謝制御

`cycles(n)`は、entrainedとflowでは設定された固有周期、metricでは実効的な拍周期を使います。
entrainedの参加周期は音響の反復へ追従しますが、保持時間とその重なり予測の音長は固有周期から求めます。
したがって、周囲のペースが変わっても一音の保持を自動的に引き延ばしません。
周期への追従では自分の周期内の位置を保ち、共有の発音位相を指定しません。
flowの音長には既存の3倍の係数も適用します。`rhythm_freq()`はアーティキュレーションの
内部振動子を設定し、参加方策の固有ペースを直接設定しません。

Gated発音では自律attackが無効になります。発行したonsetは協和度に基づく通常の回復を
受け、`attack_phase_match`による位相一致報酬は使いません。共有リズムを読む音高探索や
振幅変調は別の経路として残ります。専用の拍打ちは、同期の実演や比較に必須の場合だけ使います。
