# リズム：一つの結合連続体

Conchordalのリズムは、別々の時計を寄せ集めたものではありません。**Communityから生まれる
共有拍の上に、一つの結合連続体があります。** Communityは、結合振動子からなる一つの拍を
駆動します。個々のVoiceは位相振動子として働き、結合の強さに応じて発音位置をその拍へ
近づけます。外から押し付ける格子はありません。VoiceがCommunity自身の拍へどれほど強く
同期するかによって、まとまりも、その欠如も生まれます。

三つのrhythm「family」は連続体上の三領域にすぎません。

- **metric** — 強い結合。深いattractorとして安定したpulseに聞こえる。
- **entrained** — 中程度の結合。時間とともに同期するがdriftも残る。
- **flow** — ほぼゼロの結合。自由なrenewal processによる非metric texture。

リズムはconsonance、viability、movement、respawnと同じ生態系に属します。timingが生存へ
影響し、生存がtimingを再編成できます。

## Director水準の地形

directorはConsonance Fieldの操作と対称的にrhythmic terrainを形づくります。これらは
scheduleではなくsoft priorであり、創発が働き続けます。

- `meter_stability(value)` — `[0,1]`のattractor depth。実在するperiodicityに対する
  basinを深くするだけで、非metric inputからbeatを捏造しない。
- `temporal_basin(min_hz, max_hz)` — 創発beatが引き寄せられるtempo領域。
  `consonance(min, max)`の時間軸版で、beatを配置したりmeasureを強制したりはしない。

## Voiceごとのプリセットと調整

Tier 1 presetの`metric()`、`entrained()`、`flow()`はrate引数を取りません。tempo領域は
Voiceごとではなく、演出者が一度だけ設定する地形の性質です。`place()`より前に
`PopulationSpec`へ設定します。

`.entrained()`と`brain("entrain")`を混同しないでください。前者はonset timing、後者は
Voiceの発音上の生と代謝を選びます。独立しており併用できます。

Voiceごとの設定で、結合連続体上の位置を細かく調整できます。

- `entrainment(strength)` — `[0,1]`の結合。free（`0`）からlocked（`1`）。
- `rhythm_role("beat"|"subdivision"|"accent"|"texture")` — metrical role。
  `accent`は共有meterを強く駆動するonsetを出し、反復downbeatからmeasureを創発できる。
- `microtiming(amount)` — `[-0.5, 0.5]`のsigned beat-phase offset。`0.5`は半beatずれ、
  syncopationとして聞こえる。

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

timingを生存と再編成へ影響させるには`rhythm_coupling_vitality(lambda_v, v_floor)`と
`rhythm_reward(rho_t, "attack_phase_match")`を使い、internal oscillatorの直接設定には
`rhythm_freq(freq_hz)`を使います。

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

demoやassayには有用ですが、リズム作曲の抽象ではありません。作曲されたリズムは結合連続体と
地形から生じるべきです。
