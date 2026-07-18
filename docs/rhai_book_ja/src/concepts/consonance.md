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
snapします。例外が必要なら`pitch_apply_mode()`を使います。研究scriptには
`pitch_core()`など機構水準のcontrolもありますが、作品では`seek_consonance()`と
`glide()`を優先してください。

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
