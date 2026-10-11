# Consonance Field — 音高を評価する地形

Conchordalの知覚coreである**Landscape**はhabitat busを聴き、log-frequency spaceへ
変換して二つのpotentialを計算します。**roughness**はcritical band内の振幅変動による
感覚的不協和、**harmonicity**はperiodicityとtemplate matchingです。この二つを
組み合わせた**Consonance Field**は、placement、movement、prediction、survivalが
読む、周波数上の評価地形です。

Consonance Fieldはhabitat busから計算されるため、そのbusへ送られたVoiceが
ほかのVoiceの読むConsonance Fieldを変形します。presentation busだけへ送られたVoiceは
変形しません。和声はchord chartではなく、このfeedback loopから生まれます。
potential、score、level、mass、densityの関係は
[VoiceとLandscape — 音と環境のフィードバック](ecological_loop.md)を参照してください。

## Consonance Fieldへの配置：`consonance`、`dissonance`、`edge`、`gap`

Consonance Fieldに応じたPlacementでは、Field内の**対象領域**を指定します。既定では確率分布から位置を選び、
`.peak()`を付けると決定的な極値を選びます。

- `consonance` — consonanceが高い位置（harmonic center、fusion）。
- `dissonance` — consonanceが低い位置（tension、cluster、color）。
- `edge` — consonance/dissonanceの境界（metastableな中間）。
- `gap` — 空いたregister（空間を埋め、maskingを避ける）。

`consonance(root)`はroot周辺のharmonic windowを取り、`range()`で倍音範囲を指定します。
どの配置先にも、絶対周波数の`(min_hz, max_hz)`範囲を渡せます。

範囲の両端には、音源が扱える1〜20000 Hz内の有限値を指定します。大小が逆でも使えます。
初期配置は微小な揺らぎ（jitter）を加える場合も、重みがすべてゼロ、または全域が占有済みで
代替の抽選を行う場合も、指定範囲を守ります。Consonance Fieldに基づくPlacementは
解析帯域（現在55〜8000 Hz）との共通部分を使います。共通部分がなければ評価の根拠を
得られないため、指定範囲内の対数一様分布へ
切り替わります。`random()`は指定範囲を直接使います。この範囲は初期配置の指定であり、
その後の`seek_consonance()`による移動を同じ範囲へ制限するものではありません。

```rhai
let anchor = harmonic()
    .brain("drone")
    .amp(0.06)
    .sustain()
    .anchor();

let voice = harmonic()
    .amp(0.04)
    .sustain();

section("field placement", || {
    place(anchor, at(110.0));
    wait(1.0);

    place(voice, consonance(110.0).peak().range(1.0, 4.0).count(6).spacing(0.9));
    wait(6.0);
});
```

追加指定のない配置先は**密度分布**として扱われます。「無作為だが調和的」なのではなく、
協和のモデルから導かれ、指定範囲内で正規化された分布です。配置は配置後の振る舞いから
独立しています。`dissonance`へ入り`anchor()`でclusterを保持することも、
`seek_consonance()`で解決へ向かわせることもできます。

```rhai
let cloud = harmonic().amp(0.035).sustain();

place(cloud, consonance(90.0, 1200.0).count(10).spacing(0.8));
wait(8.0);
```

## 配置時の緊張度：`tension(τ)`

`tension`を指定しないConsonance Placementは、通常の方法で実現されます。`.peak()`なら
最も強いpeakを選び、既定のdensityなら通常のConsonance密度分布を作ります。
`tension(τ)`は、最大値より一段**下**のfield-scoreへ配置を偏らせます。
`τ ∈ [0, 1]`がtension degreeで、`0`は通常の配置を変えず、大きい値ほど弱く
metastableなstepを対象にします。指定範囲内の評価値で
`target = L_max − τ·(L_max − L_min)`です。movementにおけるsearch `temperature`と
対になる、spawnをどれだけ解決した位置に置くかのdialです。`.peak()`なら最も近いstepへ
最寄りの段へ固定し、確率分布なら対象値の周辺へ重みを集中させます。

```rhai
let tense = harmonic().amp(0.035).sustain();

// A metastable step below the strongest peak — placed, not resolved.
place(tense, consonance(110.0, 1200.0).peak().tension(0.4).count(6).spacing(0.8));
wait(6.0);
```

Consonance Fieldに依存しないPlacementには、対数周波数上で一様な`random(min_hz, max_hz)`と、幾何的な
`at(hz)`、`line(start_hz, end_hz)`があります。

## 周波数をroot比で名づける

`at(hz)`と`line(start_hz, end_hz)`は絶対周波数を取ります。標準的な書き方は、
鳴っている一つのrootをHzで名づけ、他のpitchをその比として導くことです。完全5度なら
`root_hz * 1.5`、完全4度なら`root_hz * 4.0/3.0`、registerを上げるなら
`root_hz * 2.0`です。比はField自身が読む物理量なので、周波数関係として考えられます。

平均律のdecimal、たとえばD3を表す`146.83`も動作しますが、12音のsymbol gridを裏口から
持ち込みます。sceneで実際に鳴っているrootの比を優先してください。

```rhai
let root_hz = 110.0;
let voice = harmonic().amp(0.04).sustain();

place(voice, at(root_hz * 1.5));
wait(2.0);
```

## Consonance movement：`seek_consonance`

VoiceがConsonance Field上のより良い位置を能動的に探すときは`seek_consonance()`を使います。
自由な山登り探索と音高推移の初期値を設定します。同じ移動を遅く、または速く
動かしたいときは`glide(tau_sec)`を使います。

```rhai
let mover = harmonic()
    .amp(0.045)
    .sustain()
    .seek_consonance()
    .glide(0.35)
    .avoid_neighbors(0.6)
    .global_peaks(8, 70.0)
    .ratio_candidates(5);

place(mover, consonance(80.0, 900.0).count(8));
wait(12.0);
```

`avoid_neighbors(strength)`は混雑を避ける反発を加え、移動するVoiceがすべて同じ山へ重なるのを防ぎます。
移動の反対は`anchor()`です。固定されたVoiceは音高を保ち、ほかのVoiceが読むConsonance Fieldだけを
変形します。`at()`または`freq()`で指定したVoiceは暗黙に固定されます。

移動の反映方法は発音様式から決まります。持続するVoiceは滑らかに移り、再発音するVoice
（`pulse()`、`metric()`、`entrained()`、`flow()`）は発音開始ごとに新しい音高へ
移ります。`pitch_apply_mode("gate_snap")`と`pitch_apply_mode("glide")`で、
この反映方法を明示的に選べます。

持続するVoiceで、新しい音高の行き先を採用するたびに発音し直したい場合は、
参加の時計を付けず、`pitch_apply_mode("rearticulate")`を指定します。移動の始まりで旧いToneをreleaseし、
新しいToneは頭から新しい音高でattackします。棄却した提案、同じ行き先、
glideの途中の更新では発音し直しません。提案の間隔は行き先を選ぶ機会の間隔であって、
一定の音符を刻む時計ではありません。発音を拍に合わせて遅らせる処理もありません。

```rhai
let moving = harmonic().sustain().seek_consonance()
    .pitch_apply_mode("rearticulate");
let voices = place(moving, line(220.0, 440.0).count(3));
wait(12.0);
voices.pitch_apply_mode("glide");
wait(12.0);
```

演奏中にglideとrearticulateを切り替えても、鳴っているToneを止めたり、
切り替えだけで新しい発音を起こしたりしません。まだ出力していない再発音の要求を取り消し、
次の採用から新しいmodeを使います。再出生するVoiceにもPopulationのtemplateから
同じmodeが渡ります。`glide(tau)`はGlideと、打ち直しの境界以下で滑る時の移動の時定数です。
`anchor()`などで固定されたVoiceにrearticulateは作用しません。

対象は音高が自由なHoldのVoiceです。自由音高のGated発音とrearticulateの組み合わせは、
builderの呼び出し順にかかわらず、配置時の最終specでscript errorになります。
演奏中の対象外の指定も、状態を書き換える前に拒否します。`sustain()`でHoldにするか、
Gatedではgate_snapまたはglideを選んでください。参加の時計がないHoldには、同音高の
発音だけを求める指定はありません。持続＋参加では機会による現在音高の打ち直しも
できます。Voice自身が一音ごとに方策を選ぶ機能は含まれません。

### 音高の移動と発音の結びつきを選ぶ

行き先を採用する時と、参加の機会で鳴らすと選ぶ時は別です。次の表で、移動を音の区切りに
どう結びつけるかを選べます。境界は創始のVoiceに置く発音の気質で、作曲者が選ぶ量です。
50や100 centは使い方の例であり、聴覚の原理から決まる値や推奨値ではありません。

| 書きたい形 | 明示する指定 | 発音の区切り |
|---|---|---|
| レガート | `sustain().pitch_apply_mode("glide")` | 初回に発音し、その後の移動は滑る |
| 採用ごとに打ち直す | `sustain().pitch_apply_mode("rearticulate")` | 境界を省略するか0にすると、採用ごとに打ち直す |
| 小さく動く時は滑る | `sustain().pitch_apply_mode("rearticulate").rearticulate_above(50)` | 最後の発音音高から行き先までが50 centを超えた採用で打ち直す |
| 持続しながら参加し、音高は滑る | `sustain().entrained().pitch_apply_mode("glide")` | 鳴らすと選んだ機会で、その時点の身体音高を打ち直す |
| 持続しながら参加し、発音時に跳ぶ | `sustain().flow().pitch_apply_mode("gate_snap")` | 鳴らすと選んだ機会で、最後に保留した行き先へ跳んで打ち直す |
| 持続しながら参加し、移動の大きさで分ける | `sustain().entrained().pitch_apply_mode("rearticulate").rearticulate_above(50)` | 小さい採用は滑り、大きい採用は鳴らす機会まで保留する |

参加の各行では、entrainedとflowをそれぞれ選べます。時計の違いは
[リズム](rhythm.md)の章で説明します。

参加の時計がない場合、正の境界を指定した`rearticulate_above(cents)`は、最後に実際に発音した音高からの正味の変位を測ります。
経路の距離を足し合わせる量ではなく、元の音高へ戻れば相殺されます。境界以下なら同じToneで
`glide(tau)`に従って滑り、境界を超えれば新しいToneを立ち上げます。選択は次の採用まで
保持します。両端がLog2Spaceの正確なbin中心ならbin差をcentへ換算し、それ以外は連続した
log2座標の差を使います。行き先の丸め直しやepsilonは加えません。同じ未出力hopに先行する
合法な発音があれば、その音高を基準にします。取り消した要求を実発音の記憶へ残しません。

参加の時計がない場合、省略値0は採用ごとに打ち直す従来の形です。負値と有限でない値は0にし、上限は置きません。
演奏中に境界を変えても、それだけでは発音せず、次の採用から使います。Rearticulate以外への
指定は警告して無視します。配置前の書く順に依存しません。再生成したVoiceは境界を継ぎますが、
親が最後に発音した音高は継ぎません。

```rhai
let moving = harmonic().sustain().seek_consonance()
    .pitch_apply_mode("rearticulate").rearticulate_above(50).glide(0.03);
let voices = place(moving, consonance(220.0, 660.0).count(2));
wait(3.0);
voices.rearticulate_above(100);
wait(3.0);
release(voices);
```

参加の時計を持つRearticulateでは、同じ距離の測り方を使い、打ち直しの時刻は時計に
委ねます。正の境界以下の採用は、その時刻から滑り始め、発音せず、旧い保留を取り消します。
境界を超える採用は進行中の滑りを止め、身体の音高を保ち、最新の行き先を一つ保留します。
鳴らすと選んだ機会で保留を消費し、そこへ跳んで打ち直します。保留がなければ、その時点の
身体音高で打ち直します。滑っている途中なら、前の発音と音高が違う場合もあります。
どちらの打ち直しでも最後の実発音音高を更新します。待つ・見送るなら音と保留を保ち、
既に始まった滑りは続きます。同じsampleでは採用を先に処理し、採用時刻の発音候補を
追加しません。境界で時計の発音を間引きません。境界の省略と0はGateSnapと同じ形で、
距離0の採用も保留します。演奏中の切り替えは[リズム](rhythm.md)の章で説明します。

正の境界を持つこの形では、配置前の`articulation_persistence(p)`でBase／Connectを
既存の出来事にまたがって保てます。移動の境界やつなぐ割合を変える指定ではありません。
[発音の方針を保つ](rhythm.md#発音の方針を保つ)を参照してください。

### 音を切って間を置く

持続音と参加の機会を併せ持つVoiceに`rest()`を指定すると、発音ごとに有限の音長で
音を離し、鳴り終わってから能動的な間を置きます。音長は`cycles(n)`、または
`adaptive_duration()`とその調整語で指定します。音長、明示の`sustain()`、
`entrained()`または`flow()`、`rest()`は、書く順によらず独立に保持します。
有限の音長を暗黙に補う処理はありません。明示の`while_alive()`は有限の指定を消し、
その後に有限の音長を指定し直さなければ配置時にerrorになります。

予定したOffで通常のreleaseを始め、余韻は切りません。rendererが同じsource IDと
generationに属する全Toneの終了を確認してから、間を数えます。`rest()`は確認時の
固有周期一つ分、`rest(periods)`は作曲者が置いた倍数のあいだ発音を禁止します。
周期が既知なら`rest(0)`は余韻の終了だけを待ちます。周期が未知の場合も、音が
鳴り終わっていない場合も、代わりの値や期限を置かず待ちます。負値と有限でない値は
0にし、有限の非負値に上限は置きません。

間が終わると、既存の参加の判断で鳴らす・待つ・見送るを選べます。間の終わりだけで
発音を強制しません。GateSnapは発音中、余韻待ち、間のあいだも最新の行き先を保留し、
次に鳴らすと選んだ機会で消費します。Rearticulateも余韻待ちや間のあいだに採用を扱い、
小さい採用は旧保留を取り消し、大きい採用は保留します。次の実発音まで、最後の発音音高を
基準にします。Off後も身体の滑りは続きますが、離した余韻は動かしません。Glideでも
Off後の余韻に音高更新を渡しません。
release、死亡、removeが優先します。再生成したVoiceは設定を継ぎ、演奏の状態は
新たに始めます。

```rhai
let notes = harmonic().sustain().entrained()
    .adaptive_duration().duration_range(0.2, 0.2).rest(2);
let voices = place(notes, at(330.0));
wait(3.0);
release(voices);
```

ここでの0.2周期と間の2周期は使い方の例で、推奨値ではありません。`rest`は音長の
語と同じく配置前の指定です。明示の持続＋entrained／flowと有限の音長が必要で、
Glide、GateSnap、Rearticulateを選べます。対象外の配置は状態を書き換える前に拒否します。
`rest`がなければ既存の規則を保ちます。この形では全ての音が有限長で終わり、Voiceが
一音ごとに伸ばすか切るかを選ぶ機能は含まれません。

研究scriptには`pitch_core()`など機構水準のcontrolもありますが、作品では
`seek_consonance()`を入口にし、発音し直すか滑らかに移るかを選んでください。

## Consonance viabilityとrespawn

`consonance_viability(low, high)`はconsonance window、`recovery(seconds)`は最大回復に
要する時間を定義します。適合度の高いVoiceは支えられ、低いVoiceは基準
`endurance(seconds)`へ近づきます。既定では自分の音響的な痕跡を近似的に除く
**環境相対**評価です。Consonance Field全体への適合度を問う場合だけ`viability_scope("total")`を使います。

再生成は循環を生態系として閉じます。Voiceが死亡すると、再生成の方針に従って代わりのVoiceが
現れます。`respawn_consonance()`は、energyで重み付けして選んだliving parentの周辺へ
偏らせながら、field scoreの高いpeakから選びます。`respawn_capacity(count)`はliving
membership上限です。`respawn_settle(placement)`はpolicy本来のcandidateを置き換えず、
replacementのcandidate poolへそのPlacementを追加します。

```rhai
let settle = consonance(70.0, 1100.0).spacing(0.8);

let ecology = harmonic()
    .amp(0.04)
    .repeat()
    .pulse(1.5)
    .cycles(3)
    .seek_consonance()
    .glide(0.45)
    .endurance(8.0)
    .recovery(4.0)
    .attack_cost_fraction(0.017)
    .attack_recharge_fraction(0.70)
    .consonance_viability(0.32, 0.82)
    .respawn_consonance()
    .respawn_capacity(14)
    .respawn_settle(settle);

place(ecology, consonance(70.0, 1100.0).count(14));
wait(30.0);
```

完全なlifecycle/respawn surfaceは[APIリファレンス](../reference/api.md)、articulation、
発音、音高の振る舞い、生存の区別は[Population — Voiceをまとめる持続単位](voice_life.md)、
エネルギーと再生成のcycleは
[VoiceとLandscape — 音と環境のフィードバック](ecological_loop.md)を参照してください。

## Landscape-awareな音色

Fieldはpitchだけでなく音色も形づくれます。`modal()` bodyはmode patternを取ります。
`landscape_density_modes()`はdensity massの強い位置を間隔を空けて決定的に選び、
`landscape_peaks_modes()`はField levelの強いlocal peakを間隔を空けて選びます。
そのため、bellのpartialをConsonance Fieldがすでにsupportする位置へ置けます。

```rhai
let shimmer_modes = landscape_density_modes()
    .count(10)
    .range(1.0, 5.5)
    .gamma(1.6)
    .spacing(0.7);

let shimmer = modal()
    .amp(0.025)
    .sustain()
    .seek_consonance()
    .modes(shimmer_modes)
    .brightness(0.7);

place(shimmer, consonance(200.0, 1600.0).count(4));
wait(8.0);
```

mode patternの生成関数には`harmonic_modes()`、`odd_modes()`、`power_modes(beta)`、
`stiff_string_modes(stiffness)`、`custom_modes([ratios])`、`modal_table(name)`、
`landscape_density_modes()`、`landscape_peaks_modes()`があります。
