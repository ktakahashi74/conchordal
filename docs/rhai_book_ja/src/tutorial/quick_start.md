# クイックスタート

リアルタイムinstrumentでシナリオを実行します。DSP性能のためrelease modeを推奨します。

```bash
cargo run --release -- samples/01_a_single_voice.rhai
```

## 最小の音

```rhai
place(sine().amp(0.08).sustain(), at(440.0));
wait(2.0);
```

`place(population_spec, placement)`は、現在のスクリプト時刻にPopulationをただちに配置します。
返された`Population`への参照を使うと、実行中のPopulationを更新したり終了したりできます。

## 基本オブジェクト

- **PopulationSpec**は、初代として生成されるVoiceとPopulation全体の方針を記述します。
  `sine()`、`harmonic()`、`modal()`、`saw()`、`square()`、`noise()`から始めます。
- **Variant**は`variant(population_spec)`でspecificationを複製します。
- **Placement**はPopulationをどこへ配置するかを決めます。協和、不協和、その境界、空いた音域を
  `consonance()`、`dissonance()`、`edge()`、`gap()`で選べます。既定では候補を確率分布として
  扱い、`.peak()`を付けると最も強い一点を選びます。ほかに`random()`、`at()`、`line()`があります。
- **Population**は`place()`が返す安定した参照です。この時点で初代のVoiceが
  存在し始め、実行中に使えるメソッドは現在のVoiceを更新します。
- **Section**はPopulationの有効範囲を定め、終了時に自動で解放します。

Populationは、配置後も同一性を保つ一つの安定した参照です。`.count(6)`を指定したPlacementでは、
そのPopulationに6つの初代のVoiceが生まれます。後でVoiceが死亡して再生成されても、構成員と世代が
変わるだけでPopulationの同一性は保たれます。詳しいオブジェクト構造と生存過程は
[Population — Voiceをまとめる持続単位](../concepts/voice_life.md)を参照してください。

```rhai
let population_spec = harmonic()
    .amp(0.08)
    .sustain()
    .brightness(0.35);

section("plain entry", || {
    place(population_spec, line(220.0, 440.0).count(3));
    wait(4.0);
});
```

## Consonance Fieldへ配置する

`consonance(root_hz).peak()`は、基準周波数の周辺で協和の評価が高い位置へVoiceを
配置します。Fieldはsystemが知覚したものによって形づくられるため、anchorが入ると
peakの位置も変わります。

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

Consonance Fieldの働きと、その中でVoiceが移動し、生き残り、再生成される仕組みは
[Consonance Field — 音高を評価する地形](../concepts/consonance.md)で説明します。

## 実行中のPopulationを更新する

発音体、振る舞い、生存過程、再生成の方針は、`place()`より前に`PopulationSpec`へ設定します。
返された`Population`には、配置後に意味を持つ操作、つまり実行中の更新と終了だけが
公開されます。[APIリファレンス](../reference/api.md)には、両者のメソッドが区別して
表示されます。

```rhai
let spec = harmonic().amp(0.04).sustain();
let population = place(
    spec,
    consonance(220.0).peak().count(3)
);
population.amp(0.02); // live patch on running voices
population.glide(0.8);
wait(3.0);
release(population);
```

## 完成した小品

```rhai
seed(7);

let anchor = harmonic()
    .brain("drone")
    .amp(0.05)
    .sustain();

let colony = harmonic()
    .amp(0.035)
    .sustain()
    .seek_consonance()
    .glide(0.4)
    .avoid_neighbors(0.6);

section("emergence", || {
    place(anchor, at(110.0));
    wait(2.0);

    place(colony, consonance(90.0, 900.0).count(8).spacing(0.8));
    wait(8.0);
});
```

## 次に読む章

- [エディタ設定](editor_setup.md) — completion、hover documentation、diagnostics。
- [演奏](observing.md) — report、filter、`--seed`による再現。
- [Population — Voiceをまとめる持続単位](../concepts/voice_life.md) — PopulationSpec、Voice、brain、phonation、生存、release。
- [VoiceとLandscape — 音と環境のフィードバック](../concepts/ecological_loop.md) — 音と知覚環境がfeedbackを作る仕組み。
- [Consonance Field — 音高を評価する地形](../concepts/consonance.md) — Field、density、movement、viability、respawn。
- [リズム](../concepts/rhythm.md) — 結合連続体とdirectorが形づくるリズム地形。
- [ルーティングとListenerTwin](../concepts/routing.md) — 生態系が知覚する音、聴衆が聴く音、観測される状態。
- [タイムラインと構造](../concepts/timeline.md) — 配置境界、有効範囲、再利用可能な身振り、並行する流れ。
- [サンプル](../reference/samples.md) — 機能を順番に試すための道筋。
