# VoiceとLandscape — 音と環境のフィードバック

habitat busへ送られたVoiceはLandscapeを変え、変化したLandscapeはplacement、movement、
viabilityへ影響します。この章では、その実行時feedbackと、そこから生じるenergy、death、
respawnの過程を追います。

```text
Voice bodies
    |-- presentation bus --> listener / ListenerTwin
    |
    `-- habitat bus --> Landscape --> placement, movement, viability
                              ^                         |
                              `------ new sound <------'
```

初期状態ではすべてのVoiceが両方のバスへ送られ、聴衆と生態系は同じ物理的な出来事を共有します。
意図的に分離する方法は[ルーティングとListenerTwin](routing.md)を参照してください。

## 音からLandscapeへ

Landscapeはhabitat busをlog-frequency spaceで解析し、次を計算します。

- **roughness potential** — critical band内の干渉とbeating。
- **harmonicity potential** — periodicityとvirtual rootへのsupport。
- **Consonance Field** — candidate frequencyの評価に使う合成地形。

Voiceがhabitat bus上の周波数成分を変えると、これらの走査結果も変わります。そのため
`consonance(...).peak()`は固定scale degreeを返しません。現在鳴っているものによって
答えが変わります。

## Score、level、mass、density

同じLandscapeから複数の表現が導かれます。関連はありますが交換可能ではありません。

| 表現 | 範囲 | 用途 |
|---|---:|---|
| potential | kernel依存 | raw roughness/harmonicity output。 |
| field score | 非有界の実数 | 位置比較、hill-climb、placement tension。 |
| field level | `0..1` | 振る舞いと生存可能性に使う有界な信号。 |
| density mass | 非負 | stochastic placementの正規化前weight。 |
| density / PMF | 選択範囲内で総和1 | 密度分布からの標本抽出。 |

`.peak()`は極値を選びます。既定の密度配置はPMFから標本を取るため、複数のVoiceは
一つの周波数区間へ重ならず、支えられた領域の周囲に分布します。
`tension(degree)`は最大peakより一定のfield-score stepだけ下を狙います。
生存可能性の範囲は、有界または環境相対の適合度を読みます。レポートの接尾辞も同じ区別を
保ち、`mean_c_field_score`と`mean_c_field_level`は別の量です。

## 配置と移動は別の判断

配置はVoiceがどこへ入るか、音高の振る舞いは入った後に何が起きるかに答えます。

```rhai
let fixed_strain = harmonic()
    .amp(0.035)
    .sustain()
    .anchor();

let resolving_strain = harmonic()
    .amp(0.035)
    .sustain()
    .seek_consonance()
    .glide(0.4);

section("two responses to dissonance", || {
    place(fixed_strain, dissonance(140.0, 900.0).count(3));
    place(resolving_strain, dissonance(140.0, 900.0).count(3));
    wait(6.0);
});
```

両方のPopulationが不協和な領域へ入ります。一方はその状態を保ち、もう一方は解決へ向かう
出発点として扱います。

## 自分ではなく環境を評価する

Voiceは、後で自ら評価するConsonance Fieldへ自身のエネルギーも加えます。そのままでは、強いVoiceは自分の
footprintを聴くことでviableに見えるかもしれません。そこで`consonance_viability()`は
既定で環境相対評価を有効にし、自分の寄与を近似的に除いて適合度を判断します。

これは生存規則であり、音の送り先を決める規則ではありません。Voiceはhabitat busへ寄与し、ほかの
すべてのVoiceが読むConsonance Fieldを変え続けます。自分も含むConsonance Field全体への適合度を問いたい場合だけ
`viability_scope("total")`を使います。

## Energy、death、replacement

生態学的lifecycleは`brain("entrain")`に属し、energyは`0..1`へ正規化されます。

1. `endurance(seconds)`がzero-fit時の基準寿命を定める。
2. attackごとに`attack_cost_fraction`を消費する。
3. consonantなattackは最大`attack_recharge_fraction`まで回復できる。
4. `recovery(seconds)`が連続回復を有効にし、viability windowが現在位置での回復量を決める。
5. energyがゼロになるとVoiceは死亡し、音の減衰へ入る。
6. 再生成の方針があれば、Populationは代わりのVoiceを生み出せる。

```rhai
let settlement = consonance(70.0, 1100.0).spacing(0.8);

let ecology = harmonic()
    .brain("entrain")
    .entrained()
    .cycles(2)
    .seek_consonance()
    .endurance(8.0)
    .recovery(4.0)
    .attack_cost_fraction(0.017)
    .attack_recharge_fraction(0.70)
    .consonance_viability(0.32, 0.82)
    .respawn_consonance()
    .respawn_capacity(8)
    .respawn_settle(settlement);

place(ecology, consonance(70.0, 1100.0).count(8));
wait(20.0);
```

respawn policyは異なる作曲上の問いに答えます。

- `respawn_random()` — parentの系譜を作りません。Populationを最初に配置したPlacementから
  candidateを作り、現在のscene scoreで重み付けして選びます。一様random配置ではありません。
- `respawn_hereditary(sigma_oct)` — living parentをenergyで重み付けして選び、その近くへ
  offspring候補を作り、現在のField levelが最も高いcandidateを採用します。
- `respawn_consonance()` — living parentをenergyで重み付けして選び、そのparentの周辺へ
  偏らせながらfield scoreの高いpeakから選びます。
- `respawn_capacity(n)` — Populationが維持する生存個体数の上限。未指定時は初代の個体数で、
  明示値をfounder数より小さくはできない。
- `respawn_settle(placement)` — そのPlacementからcandidateを追加します。respawn policy本来の
  baseline candidateも一つ残ります。

respawn後も`Population`と`population_id`は維持され、`voice_id`と`generation`が変わります。
レポートの記録により、この世代交代を観測できます。

## 実行時のfeedbackと人間による修正

- **実行時のfeedback**は自動です。音がLandscapeを変え、LandscapeがVoiceの振る舞いと
  生存を変えます。
- **人間による修正**はrunの間に行います。実行し、聴き、reportを調べ、scenarioを修正して
  再実行します。

実行して結果を確かめ、修正する具体的な手順は[演奏](../tutorial/observing.md)で説明します。
