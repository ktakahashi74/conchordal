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

- `meter_stability(value)` — 拍を引き込む深さ（attractor depth）を`[0,1]`で指定します。
  既定値は`0`で、中立の状態です。拍が形成されやすい度合いに加え、好みのテンポに
  寄せて拍を読む強さも、同じ深さで決まります。好みの中心は`temporal_basin`の中心、
  帯域を置かなければ2 Hzです。これは作曲者が置く、モデル化した聴き手の文化層の
  事前分布です。実際の周期性への引き込みを強める操作であり、非周期の入力から
  拍を作り出しません。
- `temporal_basin(min_hz, max_hz)` — 拍が適応できるテンポの帯域を指定します。
  拍はその中心から始まり、適応も帯域内に限られます。stabilityが`0`なら、中心を
  好む重みも、中心へ戻す力も足しません。深くするほど、中心に寄せた拍の読みと、
  確信度に依存する中心への復元が強まります。`consonance(min, max)`の時間軸版であり、
  拍を配置したり、小節を強制したりはしません。同じ整形済みの中心を、`metric()`の
  Voiceが自走を始める速さにも使います。この初期値はstabilityに依存しません。
  basinを置かなければ、metricの初期値は従来どおり2 Hzです。

basinはscene全体の初期条件です。scriptで最後に指定した値を、`place()`と`create()`で
作る全てのmetricのPopulationに使います。Populationの配置後や`wait()`の後で指定しても、
同じsceneの初期値を上書きします。演奏途中で切り替える指定にはなりません。

身体が最初に刻む周期と、音から確認された拍は区別します。metricの各Voiceは別々の
初期位相を持ちます。音から拍を確認するまでは確信度と共有拍への結合は0で、確認後に
聴こえた拍へ近づきます。再生成したVoiceはPopulationに保存した初期設定から始まり、
親が演奏中に得た周期を継ぎません。entrainedとflowの固有ペース、`pulse(rate_hz)`で
明示した速さは、この初期化の対象外です。

既定の帯域で4 Hzの等間隔の入力を最初から観測する対照では、stabilityが`0`なら
4 Hzの拍、`1`なら2 Hzの拍の二分割として読みます。深さは駆動の強さと証拠を読む
速さにも従来どおり作用し、テンポの好みだけを独立に動かす操作ではありません。
間隔の仮説を初めて確認するまでは確信度を0に保ちます。ただし、確認後の不規則な
入力が偶然まとまり、一時的に確信度が上がる場合は残ります。

## Voiceごとのプリセットと調整

`metric()`、`entrained()`、`flow()`は速さの引数を取りません。`place()`や`create()`より前に
PopulationSpecへ設定します。metricはbasinの中心、指定がなければ2 Hzから自走を始め、
音から確認された拍へ近づきます。entrainedとflowは2 Hzの固有ペースから始まります。entrainedは周囲の反復から参加の周期を更新します。
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

下で説明する明示の持続と参加の併用を除き、同じ軸では最後の指定が優先されます。調整値は記憶され、対応するプリセットを選んだ時に適用されるため、
`entrainment(0.8).metric()`と`metric().entrainment(0.8)`は同等です。
`duration_range(...).adaptive_duration()`と逆順も同様です。

## 持続音を保ちながら参加する

`sustain()`と`entrained()`または`flow()`を両方明示すると、Voiceは音を保ちながら、
参加の機会ごとに鳴らす・待つ・見送るを判断します。書く順には依存しません。
`sustain().entrained()`と`entrained().sustain()`は同じ形です。
鳴らすと選んだ時は、旧いToneを離し、そのsampleの身体の音高で新しいToneを立ち上げます。
待つ・見送ると選んだ時は、鳴っているToneを保ちます。見送りを無音の間隙には変えません。
音高が同じでも、参加の機会で打ち直す場合があります。

```rhai
let p = place(
    harmonic().sustain().entrained().seek_consonance().glide(0.03),
    consonance(220.0, 660.0).count(1)
);
wait(3.0);
release(p);
```

既定の`"glide"`では、新しい音高目標を採用した時に、そのsampleでの発音も、まだ決めていない合法な参加の機会へ
候補として渡します。既存の時刻、音響の文脈、重なりの費用で比較し、同じ費用なら既存の
候補を残します。既に決めた発音の予定は動かしません。採用時の発音が選ばれなければ、
音高は滑らかに移ります。採用と時計の発音が同じsampleに重なった時は、一回だけ発音します。
移動中に時計が発音を選んだ時も、その時点の身体の音高を使い、移動先へ先に跳びません。
release、死亡、removeは保留中の要求を取り消します。発行した発音は、通常のenergyと
音響の経路へ一回だけ渡ります。

参加の予測には、既存の有限長の代表的な音のfootprintを使います。音を続ける案と、
旧音を離して打ち直す案の波形を、その場で厳密に描いて比べる仕組みではありません。
フレーズの境界の検出や、音程による打ち直しの閾値も追加しません。

この併用で音高の適用方法を省略すると`"glide"`になります。音高が自由なVoiceへ
`"gate_snap"`を明示すると、採用した行き先を一つ保留し、参加の機会で鳴らすと選ぶまで
現在の音高を保ちます。鳴らすと選んだ時に旧いToneを離し、保留した音高で新しいToneを
立ち上げます。保留中に別の行き先を採用すれば置き換えます。待つ・見送る間は音も保留も
保ち、滑りません。保留がなければ、現在の音高を打ち直します。

GateSnapは採用時刻の発音候補を追加せず、参加の費用と見送りの規則をそのまま使います。
同じsampleでは採用を先に処理し、その後の機会で使います。未来の採用を先の発音に
使いません。release、死亡、removeは保留を取り消します。

```rhai
let p = place(
    harmonic().sustain().flow().seek_consonance().pitch_apply_mode("gate_snap"),
    consonance(220.0, 660.0).count(1)
);
wait(3.0);
p.pitch_apply_mode("glide");
wait(3.0);
release(p);
```

演奏中のGlide、GateSnap、Rearticulateの切り替えでは、鳴っているToneを保ち、余分な発音を起こしません。
未出力の要求と、保留していた発音用の行き先は取り消します。Glideへ戻すと、音高の判断側に
残っている目標へ、次の制御更新から滑り始めます。GateSnapやRearticulateへ替えた後は、その後の採用を
保留します。同じmodeをもう一度指定した場合は保留を保ちます。再生成したVoiceはmodeを
継ぎますが、親の演奏中の保留は継ぎません。参加のプリセット自体は配置前の指定です。

この形に`pitch_apply_mode("rearticulate")`を添えると、`rearticulate_above(cents)`で
滑りと保留を組み合わせられます。最後の実発音音高から測り、正の境界以下の採用は
旧い保留を取り消して滑り、超える採用は滑りを止め、次に鳴らす機会まで行き先を保留します。
保留のない機会は、その時点の身体音高で打ち直します。待つ・見送る時も既に始まった滑りは
続きます。どの打ち直しでも最後の実発音音高を更新し、境界は時計の発音を間引きません。
境界の省略と0はGateSnapと同じです。参加の時計がないHoldのRearticulateは採用時に
発音し、この形では鳴らすと選んだ機会で発音します。

演奏中の境界変更は、それまで選んだ挙動を次の採用まで保ちます。mode変更では未出力の
要求、保留、選んだ滑りを取り消し、Toneと判断側の目標を保ちます。配置前の書く順に
依存しません。再生成したVoiceは設定を継ぎ、親の保留、滑りの選択、最後の発音音高は
継ぎません。

既定のWhileAliveと、`while_alive()`だけでは、この形を選びません。`sustain()`だけ、
`entrained()`だけ、`flow()`だけの挙動は変えません。`metric()`、`pulse()`、`repeat()`と
`sustain()`の併用は、従来どおり後の指定が優先されます。`rest()`がない場合は、後に
`cycles()`や`adaptive_duration()`で有限の音長を指定すると、持続する形を置き換えます。
`rest()`があれば音長と持続＋参加を独立に保ち、音を離し、実際の余韻の終了から間を
数えます。[音を切って間を置く](consonance.md#音を切って間を置く)を参照してください。

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
