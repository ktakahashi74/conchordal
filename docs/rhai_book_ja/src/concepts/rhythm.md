# リズム：時間の知覚と個体の参加

拍は、音から推定する反復的な時間の基準です。同じ音を参照するVoiceが、同じ拍位置で
発音する必要はありません。

## プリセットが保つ時間関係

プリセットは、時刻を決める方策が保つ関係を選びます。発音の機会と実際の打ち直しは
別です。生存過程、発音ゲート、有限音長、休止、発音の方針によって、発音を控えたり
生きたToneを伸ばしたりする場合があります。

| プリセット | 保つもの | 変わってよいもの | 保証しないもの | `entrainment(x)`の意味 |
| --- | --- | --- | --- | --- |
| `entrained()` | Voiceの生存中に固定された身体の固有の速さ | 身体の尺度への近さで重み付けし、支持された反復へ参加周期を寄せます。費用によって予定近傍の発音、参加の一周期までの待ち、見送りを選びます | 等間隔の発音列、Voice間で違う発音の速さを保ち続けること、共有拍との位相一致 | 周期の適応と発音時刻の費用に対する音響文脈の影響。省略値`0.5` |
| `flow()` | 不規則な間隔を抽選する身体の固有の尺度。反復周期への適応はしません | 上下限のある間隔を更新し、音響費用に応じて時刻の変更、待ち、見送りを選びます | 規則的な発音列、拍、固有周期と実発音の平均間隔の厳密な一致 | 発音時刻の音響費用への影響。抽選の尺度を反復へ適応させる量ではありません。省略値`0.05` |
| `steady()` | 固有周期と、その後の自分の発音予定の格子。時刻はsampleへ丸めます | 文脈は予定時刻で鳴らすか見送るかを選び、実行上の制約でも発音が抜けます | 発音数、音高の結果、相手の時刻への追従 | 鳴らすか見送るかの音響費用への影響。周期と予定時刻は動かしません。省略値`0.5` |
| `metric()` | 音から確認された生成側の拍への引力と、指定した拍位相のずれ。確認前は独立した初期位相から始めます | 拍とその確信度に応じて実効的な速さと位相が変わります。確認前は自走し、実行上の制約や発音方針でも打ち直しを控えます | 拍そのものの確認、厳密な位相一致、独立した非整数比のテンポ関係 | 拍の確信度に応じた速さと位相への引力。省略値`0.95` |

entrained・steady・flowは音を参照する局所的な参加、metricは拍への同期です。
`entrainment(0)`なら、表に示した音響文脈による時刻への作用をなくします。
flowの不規則な間隔の抽選は残ります。生成側のmeterが持つのは一つの拍と整数比の
分割・小節であり、独立した非整数比のテンポを結ぶネットワークはありません。

entrainedの参加周期を更新するときは、まだ確定していない将来の予定までの残り時間を、
新旧周期の比で写します。確定した予定を新しい音響文脈で動かしません。

entrainedとflowは、過去のhabitat音声から得た短い未来の予期を使います。Voice自身が
合成した音も記録し、自分の反復を競合相手として扱わないようにします。発音予定の近傍で
鳴る、参加の1周期まで待つ、今回は見送る、を比較します。重なりは予定する保持時間と
ADSRのreleaseを含む範囲で評価します。自分で見送った回数が増すほど次の見送りの
費用は増え、実際の発音指令で回数をリセットします。発音したときだけ周囲との関係の
記憶更新を予約し、対応する実音の観測が届いてから更新します。待ち、見送り、未実行の
予定を発音した経験にはしません。保持時間と重なり予測の音長は固有周期から求めるので、
参加周期の適応によって各音を引き延ばしません。三つの帯域のエネルギーを使う評価は
工学的な近似であり、聴覚上のマスキングや協和・融合のモデルではありません。

## 連続する音から反復を読む

TOMLの最上位に`temporal_meter_t2 = true`を置くと、production meterはhabitatの連続音から
反復を読みます。どのセクションよりも前に指定します。既定値の`false`では発行された
発音イベントを使います。群のアクセントや到着予測を設定する`[temporal_t2]`とは独立です。

音から反復間隔の候補を得て、その後の音で支持を確かめます。周波数成分の変化と実際の
発音が位相と強度の手掛かりです。確信度は0から始まり、反復の証拠を表します。
拍の位相が正しい確率ではありません。欠けた観測は証拠を加えません。
`metric()`は、この確信度を伴う共有拍に追従します。ListenerTwinは提示側の発音から
独立した拍を読みます。オフライン描画は決定的な観測の締切を使い、演奏時は更新時点で
届いている観測を使います。

## 自分の発音予定を保つ：`steady()`

steadyで過ぎた機会や合法でない機会を進めるときも、固有周期の整数倍で進め、格子を
ずらしません。持続音の初回のHold発音は、後続の格子とは別に従来どおり即時です。
費用の項や重みは変えません。いまの到来費用は鳴らすか見送るかを変えません。候補が
予定時刻一つで、見送りの比較から到来費用を除くためです。

```rhai
let voices = place(harmonic().steady().sustain().intrinsic_rate(2.0)
    .seek_consonance().pitch_apply_mode("gate_snap"), line(220.0, 330.0).count(2));
wait(6.0);
release(voices);
```

timingのプリセットは後の指定が優先されます。`flow().steady()`はsteady、
`steady().flow()`はflowを選びます。音長は別軸の既存規則に従います。配置前の指定であり、
演奏中に呼ぶ語ではありません。明示の`sustain()`との順は問わず、見送った機会では
現在のToneを保ちます。Glide、GateSnap、Rearticulateとその境界、`tie`（省略時0）、
`articulation_persistence`、有限音長と`rest`の併用と優先順はままです。
Glideでも音高を採用した時刻の発音候補を足しません。音高の適用方法を演奏中に替えても
Toneを保ち、その変更だけでは発音しません。再生成では保存した方策と速さを継ぎ、
初期位相、due、予定、演奏状態は新しく始めます。

## Director水準の地形

演出者は次の操作で、metricが参照する生成側の拍の地形を調整します。
entrained・steady・flowの身体の速さや、反復予測の周期候補を直接指定する操作ではありません。

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

basinはscene全体の初期条件です。scriptで最後に指定した値を、`place()`で
作る全てのmetricのPopulationに使います。Populationの配置後や`wait()`の後で指定しても、
同じsceneの初期値を上書きします。演奏途中で切り替える指定にはなりません。

身体が最初に刻む周期と、音から確認された拍は区別します。metricの各Voiceは別々の
初期位相を持ちます。音から拍を確認するまでは確信度と共有拍への結合は0で、確認後に
聴こえた拍へ近づきます。再生成したVoiceはPopulationに保存した初期設定から始まり、
親が演奏中に得た周期を継ぎません。entrained・steady・flowの固有ペース、`pulse(rate_hz)`で
明示した速さは、この初期化の対象外です。

既定の帯域で4 Hzの等間隔の入力を最初から観測する対照では、stabilityが`0`なら
4 Hzの拍、`1`なら2 Hzの拍の二分割として読みます。深さは駆動の強さと証拠を読む
速さにも従来どおり作用し、テンポの好みだけを独立に動かす操作ではありません。
間隔の仮説を初めて確認するまでは確信度を0に保ちます。ただし、確認後の不規則な
入力が偶然まとまり、一時的に確信度が上がる場合は残ります。

## Voiceごとのプリセットと調整

`metric()`、`entrained()`、`steady()`、`flow()`は速さの引数を取りません。`place()`より前に
PopulationSpecへ設定します。時間関係と結合の省略値は上の表のとおりです。
参加の身体の尺度は`intrinsic_rate(hz)`で指定できます。

`.entrained()`と`brain("entrain")`を混同しないでください。前者はonset timing、後者は
Voiceの発音上の生と代謝を選びます。独立しており併用できます。

Voiceごとの設定で、参加の仕方を調整できます。

- `entrainment(strength)` — `[0,1]`の影響の強さです。方式ごとの意味と省略値は上の表を参照してください。選んだプリセットの中で作用の強さを変えます。
- `rhythm_role("beat"|"subdivision"|"accent"|"texture")` — metrical role。
  `accent`は共有meterを強く駆動するonsetを出し、反復downbeatからmeasureを創発できる。
- `microtiming(amount)` — metricの目標位置を拍位相でずらします。範囲は`[-0.5, 0.5]`、`0.5`は半拍です。entrained・steady・flowへ固定された発音位置を割り当てる設定ではありません。
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

## 身体の固有の速さを置く

`intrinsic_rate(hz)`は、`entrained()`・`steady()`・`flow()`で初代として生成されるVoiceの
身体の尺度をHzで置きます。固有周期は`1/hz`です。発音の間隔を命じたり、共通の拍を
置いたりする語ではありません。entrainedは聴こえた反復へ参加の周期を寄せますが、
固有の値は生存中ずっと保ちます。steadyは発音予定の周期も動かしません。flowはこの尺度から不規則な更新間隔を作ります。
どの形も機会を見送れ、entrainedとflowは待つ判断もするため、身体の速さを違えても、実際の発音の
二つの速さが永久に保たれる保証はありません。

`place()`より前にPopulationSpecへ指定します。参加のプリセットとの
前後は問いません。演奏中の変更には使えません。再生成ではPopulationのtemplateの値を
読み、新しい初期dueと演奏状態から始めます。親が適応した参加周期、発音予定、発音の
方針は継ぎません。

同じ固有周期を、有限音長の`cycles`と`duration_range`、`rest(periods)`、重なり予測の
代表的な保持長にも使います。速さを変えると、音の長さと間の秒数も一緒に変わります。
flowの音長には3倍係数が残り、ADSRのreleaseは秒指定のままです。有限音長を
指定しない持続音を、固有周期ごとに自動で切る働きはありません。

```rhai
let slow = harmonic().sustain().entrained().intrinsic_rate(0.5)
    .seek_consonance().pitch_apply_mode("gate_snap");
let quick = harmonic().intrinsic_rate(2.0).sustain().flow()
    .seek_consonance().pitch_apply_mode("gate_snap");
let a = place(slow, at(220.0));
let b = place(quick, at(330.0));
wait(6.0);
release(a);
release(b);
```

ここでの速さは使い方の例で、推奨値ではありません。最終の指定がentrained・steady・flowなら、
有限の`0.01〜200 Hz`だけを受け入れます。範囲の外と有限でない値は、配置前のscript errorに
なります。この範囲は実行を保護する値で、音楽的な推奨範囲ではありません。flowの更新間隔には
別の制限`0.035〜4秒`もあるため、極端な速さでは実際の間隔はrateに比例しません。
最終の形がmetric、pulse、repeat、onceなど対象外なら、値が不正でも警告して無視します。
省略時は2 Hzの経路を使い、metricのbasinの規約も変えません。

`rhythm_freq(hz)`はEntrainのアーティキュレーション内部の振動子を設定し、この参加の
尺度は変えません。`proposal_interval(seconds)`は音高を提案する機会の間隔であり、
発音を命じたり、身体の固有の速さを変えたりする語ではありません。

## 持続音を保ちながら参加する

`sustain()`と`entrained()`・`steady()`・`flow()`のいずれかを明示すると、Voiceは音を保ちながら、
参加の機会ごとに鳴らすか見送るかを判断します。entrainedとflowは時刻を待つ判断もします。書く順には依存しません。
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

entrainedとflowの既定の`"glide"`では、新しい音高目標を採用した時に、そのsampleでの発音も、まだ決めていない合法な参加の機会へ
候補として渡します。時刻、音響の文脈、重なりの費用で比較し、同じ費用なら
候補を残します。既に決めた発音の予定は動かしません。採用時の発音が選ばれなければ、
音高は滑らかに移ります。採用と時計の発音が同じsampleに重なった時は、一回だけ発音します。
移動中に時計が発音を選んだ時も、その時点の身体の音高を使い、移動先へ先に跳びません。
release、死亡、removeは保留中の要求を取り消します。発行した発音は、通常のenergyと
音響の経路へ一回だけ渡ります。

参加の予測には、有限長の代表的な音のfootprintを使います。音を続ける案と、
旧音を離して打ち直す案の波形を、その場で厳密に描いて比べる仕組みではありません。
フレーズの境界の検出や、音程による打ち直しの閾値も追加しません。

この併用で音高の適用方法を省略すると`"glide"`になります。音高が自由なVoiceへ
`"gate_snap"`を明示すると、採用した行き先を一つ保留し、参加の機会で鳴らすと選ぶまで
現在の音高を保ちます。鳴らすと選んだ時に旧いToneを離し、保留した音高で新しいToneを
立ち上げます。保留中に別の行き先を採用すれば置き換えます。待つ・見送る間は音も保留も
保ち、滑りません。tieを省略するか0なら、保留がない機会でも現在の音高を打ち直します。

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
`entrained()`だけ、`flow()`だけの挙動は変えません。`pulse()`、`repeat()`と
`sustain()`の併用は、従来どおり後の指定が優先されます。明示の`sustain()`と`metric()`は、
下の持続＋同期を選びます。`rest()`がない場合は、後に
`cycles()`や`adaptive_duration()`で有限の音長を指定すると、持続する形を置き換えます。
`rest()`があれば音長と持続＋参加を独立に保ち、音を離し、実際の余韻の終了から間を
数えます。[音を切って間を置く](consonance.md#音を切って間を置く)を参照してください。

## 発音の方針を保つ

自由音高の持続＋参加に`articulation_persistence(p)`を添えると、Voiceが音高の採用や
参加の機会をまたいで、発音の方針を保ちます。対象は、
`sustain()`と`metric()`／`entrained()`／`steady()`／`flow()`を明示し、`pitch_apply_mode("rearticulate")`と
正の`rearticulate_above(cents)`を指定した形です。配置前の指定で、演奏中の
Populationには呼べません。

```rhai
let p = place(
    harmonic().sustain().entrained().seek_consonance()
        .pitch_apply_mode("rearticulate").rearticulate_above(50)
        .articulation_persistence(0.75),
    consonance(220.0, 660.0).count(1)
);
wait(3.0);
release(p);
```

この例の境界と保持確率は比較用の値で、推奨値ではありません。Voiceは、境界と
参加の規則に従う**Base**から始まります。もう一つの**Connect**では、Toneが生きている間、
大小どちらの採用でも滑り、参加が鳴らすと選んだ機会の発音を控えて、現在の音を伸ばします。
これは費用による見送りとは別の意図による抑止です。新しいTone、発音による回復、
発音由来の参加文脈の学習は起こしません。

小さい採用と費用による見送りはConnect、大きい採用と新しく鳴らすと選んだ機会はBaseを
提案します。`0 < p < 1`では、現在と反対の提案だけで専用の乱数を一回引き、確率`p`で
方針を保ち、`1-p`で替えます。同じ提案や待ちのhopでは引き直しません。既に決めた
将来の機会は、実行まで同じ判断を使います。Connectへ替わる時は旧保留を取り消し、
最後の実発音音高を保ちます。取り消した採用を後で再演しません。

pは**つなぐ割合ではなく、方針の続きやすさ**です。保持はBaseにも働くため、pを
上げてもConnectの時間や発音数は単調には動きません。長さは出来事の回数で決まり、
秒や拍の数を指定しません。この保持は惰性であり、聴こえた終止、輪郭、間とは結び付かず、
フレーズの境界を推定する働きもありません。

既定の0と端の1は、私有状態や追加の乱数を通らず経路を使います。1では出生時の
Baseのまま替わりません。有限値は0〜1へ丸め、正の無限大は1、負の無限大とNaNは0に
します。対象外の最終指定は警告して無視します。書く順には依存しません。再生成では
templateのpだけを継ぎ、方針、乱数の途中、保留、最後の発音音高を継がずBaseから
始まります。演奏中のmode変更はBaseへ戻します。同じmodeの再指定と境界だけの変更は
保持を続け、境界0の間は私有経路を休止します。

有限音のOff、`rest()`、余韻終了、間、release、死亡、removeは保持より優先します。
Toneが無ければ、Connectも合法な初回発音や休止後の復帰を塞ぎません。方針は個々の
発音や休止をまたぎますが、一音ごとに伸ばすか切るかを選ぶ働きではありません。

## 行き先がない機会では音を伸ばす

持続音を参加の機会で打ち直す形に、配置前の`tie(p)`を添えられます。pは、行き先の保留が
ない時、いまのToneを打ち直さずに伸ばす確率です。自由音高の`sustain()`と`metric()`／`entrained()`／`steady()`／
`flow()`に、明示のgate_snapまたはrearticulateを組み合わせる形が対象です。境界0も含みます。
省略時は、持続＋同期では1、持続＋entrained／steady／flowの参加では0になります。参加のVoiceは
費用によって機会を見送れます。明示の`tie(0)`は保留の無い機会でも打ち直し、`tie(1)`は
対象のToneを必ず伸ばします。中間では機会ごとに両方が混ざります。固定音高とGlideは
tieの対象外で、metricの交差ごとに打ち直します。音楽的な最適値を決めたものではありません。

保留があれば、その行き先へ跳んで発音します。正の境界以下の採用は既に滑っており、
保留を作らないので、そのToneも伸ばす対象です。厳密に同じ音高だけに限りません。
gate_snapでは小さい採用や距離0の採用も保留するため、その機会は打ち直します。
発音は減り、機会から発音が抜けた列になります。機会の時刻、音高の提案、費用、
見送りの式は変えません。すべての音をつなぐレガートの量や、フレーズ単位の切り替えではなく、
聴こえた区切りとも結び付きません。

```rhai
let voices = place(harmonic().sustain().entrained()
    .pitch_apply_mode("rearticulate").rearticulate_above(50).tie(0.5),
    line(220.0, 330.0).count(2));
wait(3.0);
release(voices);
```

この境界と中間の確率は使い方の例で、推奨値ではありません。伸ばした機会では新しいTone、
発音の代謝、発音由来の文脈学習を生じず、費用の見送りとしても学習しません。機会は消費し、
後で再演しません。`articulation_persistence`を併せると、Connectの抑止を先に使い、tieの
乱数は引きません。Baseの保留ありは発音し、保留なしだけにtieを使います。tieで伸ばしても
Base／Connectの提案や保持は変えません。大きい移動も滑らせ、発音を控えるConnectの
「つなぐ」方針とは別の量です。

Toneが無い時の合法な初回や、有限Off・余韻終了・`rest()`の後の発音は塞ぎません。
release、死亡、remove、休止の合法性が優先します。演奏中にpを変える指定はありません。
modeがGlideになっている間は休止し、対象に戻れば再び働きます。`place`と`create`で
書く順に依存せず、再生成はtemplateの値だけを継ぎ、途中の乱数や保留、最後の発音音高は
継ぎません。有限値は0〜1へ制限し、正の無限大は1、負の無限大とNaNは0にします。
対象外の最終指定は警告して無視します。0ではtieの状態も乱数も使わず、1では乱数を引きません。
中間だけが個体別の専用系列を、機会の実行時に一度使います。


## 明示的なwhen/duration（Tier 2）

presetの下には`once()`、`pulse(rate_hz)`、`while_alive()`、`cycles(n)`、
`adaptive_duration()`があり、`duration_range`、`duration_curve`、`shorten_on_drop`で
調整できます。

## リズムを生存へ結びつける

`rhythm_freq(hz)`はEntrainのアーティキュレーション振動子を設定します。
`rhythm_coupling_vitality(lambda_v, v_floor)`は、その影響を活力に応じて変えます。
`rhythm_reward(rho_t, "attack_phase_match")`が効くのは、`brain("entrain")`の連続持続音で
自律的な発音が起きる場合だけです。有限のGated発音や、持続音と結合した発音機会では
協和度による通常の回復を使います。そこで報酬を明示すると、警告を出して無視します。

```rhai
let living = harmonic().brain("entrain").sustain()
    .endurance(8.0).recovery(4.0)
    .rhythm_freq(2.0)
    .rhythm_coupling_vitality(0.8, 0.4)
    .rhythm_reward(0.4, "attack_phase_match");
place(living, at(165.0));
wait(8.0);
```

身体の固有周期は有限の音長と休止時間の基準です。参加の間隔が周囲の反復へ追従しても、
固有周期に基づく保持時間は引き延ばしません。flowの有限保持時間には3倍の係数を使います。
`rhythm_freq()`が設定する振動子と、`intrinsic_rate()`が設定する参加の身体的な尺度は別です。

## Scaffold（research control）

scaffold functionは比較assay用の外部pulseを強制します。

```rhai
scaffold_off();
scaffold_shared(2.0);
scaffold_scrambled(2.0, 17);
```

同期が比較や実演に不可欠な場合だけ使います。entrained・steady・flowの参加を成立させる補助として
専用の拍打ちを追加しません。

## 持続音を拍へ同期させる

`sustain()`と`metric()`を両方明示すると、生きたToneを保ちながら、同期時計の
整数交差を発音の機会に使います。書く順は問いません。共有の拍は音から確認された拍です。
確認される前は、tempo basinの中心と独立した初期位相から自走します。外から拍を置く語ではありません。

初回はHoldと同じ即時の発音です。その後は拍の交差ごとに機会があります。自由音高の
GateSnap／Rearticulateでは、既定は行き先の採用を保留した拍で発音し、ほかの拍では
生きたToneを伸ばします。`tie(1)`と同じです。毎拍打つには`tie(0)`と書きます。
固定音高とGlideは、交差ごとにその時の身体音高で打ち直します。Glideは発音の間も滑ります。
GateSnapは最新の採用先を次の合法な交差まで保留します。境界0のRearticulateも同じです。
正の境界では小さい採用が旧保留を取り消して滑り、大きい採用は保留されます。
同じsampleの採用は、交差の発音より先に扱います。

```rhai
let wire = harmonic().brain("drone").sustain().metric().seek_consonance()
    .pitch_apply_mode("rearticulate");
let voices = place(wire, at(660.0));
wait(4.0);
release(voices);
wait(2.0);
```

自由音高のGateSnap／Rearticulateに`tie(p)`を明示すれば、その値を使います。省略値は
配置前に解決し、演奏中のmode切替で対象に入っている間も同じ規則が働きます。
持続＋entrained／steady／flowの参加は省略時0のままで、費用による見送りがあります。
tieの演奏中のsetterは追加しません。`articulation_persistence`はBase／Connectを
保ちますが、同期には費用による見送りも、決めた予定を待つ状態もありません。方針の提案は
採用と発音の交差から来ます。`rest()`は明示の有限音長、実際の余韻の終了、間を組み合わせます。
その間も時計は進み、控えた交差を後で再演しません。

役割、確認された小節の強調、microtimingの意味は同じです。固定音高も持続＋参加と同様に
扱えますが、tieとpersistenceは自由音高だけの量です。entrainmentや役割だけを指定した
Coupledには、持続＋同期の意味を広げません。
