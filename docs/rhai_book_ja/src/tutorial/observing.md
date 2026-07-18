# 演奏

[クイックスタート](quick_start.md)では、最小のscenarioを書いて実行するところまでを
説明しました。本章では、リアルタイムinstrumentを起動して演奏を始める、終了する、
GUIなしで演奏する、その結果を記録・再現する、という一連の手順を説明します。

## GUIで演奏する

リアルタイムinstrumentでscenarioを起動します。

```bash
cargo run --release -- samples/10_generations.rhai
```

scenarioをcompileした後にGUIが開き、初期設定では自動的に演奏が始まります。scriptが
大まかな時間構造を与えます。`place()`がPopulationを導入し、`wait()`がscript上の時刻を進め、
`section()`と`play()`は内部で作ったPopulationを有効範囲の終了時にreleaseします。その間も、
Voiceの振る舞い、Landscape、それらのfeedbackはリアルタイムに変化し続けます。

進行バーはscenario上の時刻を表示します。演奏中は、その右側にあるボタンを押すと
演奏を終了します。ウィンドウを閉じるかCtrl-Cを押しても終了します。このボタンは一時停止では
なく、GUI上でscriptを実行中に書き換えることもできません。scenarioを変更するときは、演奏を終了し、
Rhai fileを編集して、もう一度起動します。

GUIを先に用意し、合図に合わせて開始するには`--wait-user-start`を付けます。準備ができたら
Spaceを押すか、開始ボタンをクリックします。

```bash
cargo run --release -- samples/10_generations.rhai --wait-user-start
```

scenarioが終わっても、初期設定では最終状態を確認できるようにGUIが残ります。進行バー
右端のボタンまたはウィンドウの閉じるボタンで終了します。scenarioの終了と同時に閉じるには
`--wait-user-exit=false`を付けます。

## GUIなしで演奏する

`--nogui`を付けると、すぐに演奏を始め、scenarioの終了時にプログラムも終了します。GUIは
表示しませんが、初期設定では既定の音声出力先から音を出します。

```bash
cargo run --release -- samples/10_generations.rhai --nogui
```

音を出さず、simulationやreport作成だけを行う場合に限り`--play=false`を使います。

```bash
cargo run --release -- samples/10_generations.rhai --nogui --play=false --report run.jsonl
```

`--nogui`では、開始待ちと終了待ちの両方が無効になります。offline WAV出力は別の操作で、
`conchordal-render`が担当します。リアルタイムinstrumentである`conchordal`はaudioを
記録しません。

## 聴く・確かめる・修正する

scenarioは、どのPopulationをいつ配置し、どのような方針で振る舞わせるかを指定します。
ただし、演奏中の正確な音高、リズム同期、生存、respawnの結果までは事前に決まりません。
それらは、変化し続けるLandscapeとの相互作用から生まれます。scenarioを実行して聴き、
実際に起きたことを確かめ、scriptを修正して、もう一度実行します。

instrumentである`conchordal`は、どのbuild profileでもaudioをdiskへ書きません。
演奏は、音が鳴り終わるとinstrument内には残らない一過的な出来事（ephemeral）として
設計されています。

## 演奏を再現する

各runはseedをlogへ出します。

```text
scenario seed: 3821650944810716341 (replay with --seed 3821650944810716341)
```

最上位のサンプルは毎回新しいseedから始まり、一つの固定結果ではなくsystemのvariationを
示します。残したいrunが得られたら、表示された値で正確に再現します。

```bash
cargo run --release -- samples/10_generations.rhai --seed 3821650944810716341
```

`conchordal-render`でも同じ`--seed`を使い、保存した演奏をWAVにできます。
script内の`seed(...)`は実行中に初期seedを上書きするため、常にcommand-line flagより優先します。
呼び出し方に関係なくscript自体を再現可能にする必要がある場合だけ使ってください。

## レポート

`--report`へ書き込み先を指定して実行します。

```bash
cargo run --release -- samples/10_generations.rhai --report run.jsonl
```

`--report`は`conchordal` instrumentのflagで、`--nogui`でも動作します。
audioを描画する`conchordal-render`にはありません。

fileはJSON Linesで、1行が1 record、`type` fieldで種類を示します。

- `meta` — 最初に記録される有効なscenario seed。
- `scene_marker` — 各`section("name", || { ... })`の開始位置。以後のrecordを直前の
  markerごとにまとめられます。
- `spawn` / `respawn` / `death` — Voiceごとの世代交代。出現周波数、親となるVoice、
  設定寿命、energy枯渇時刻、envelope tailを含む観測寿命、初期consonance、死亡時PLV。
- `onset` — Voiceごとの発音開始時刻、強度、周波数、位相同期、足場の状態。
- `population_step` — 再生成待ちの`alive_count: 0`も含むPopulation size、
  平均周波数、Consonance Field score/level、周波数entropy。
- `listener_state` — `ListenerTwin`の4つの知覚level、beat/subdivision/measure追跡、解析遅延。
- `rhythm_observation` — Community全体の瞬間的なKuramoto orderと環境rhythm state。
  `rhythm_summary` — Community全体およびPopulationごとのonset density、IOI規則性、
  burstiness。Kuramoto summaryはCommunity全体にだけ付きます。
- `listener_confidence_summary` — beat confidenceのpeakと終盤window。
- `dcc_pressure` — listener由来のtension pressureとDCCが加えたpitch temperature bonus。
- `phonation_gate_open` — phonation gateが開いた時刻と、その時点のconsonance値。

未加工のJSONLは「Population 3の構成員が実際に入れ替わったのはいつか」「anchorが入った直後、
`tension_level`はどう変化したか」といった狭い問いに直接答えます。

## レポートを読む

streamはplain JSONLなので、任意のJSON toolでfilterできます。次は全deathについて寿命と
初期consonanceを取り出します。

```bash
jq -c 'select(.type == "death")
       | {time_sec, population_id, configured_endurance_sec,
          energy_depletion_sec, lifetime_sec, first_k_mean}' run.jsonl
```

`scene_marker`の時刻でbucketに分ければ、「第IV sectionでcolonyは飢えたか、その時の
consonanceはいくつか」を直接読めます。専用digest toolはありません。重要なsummaryは
作品ごとに異なるため、問いに合わせた一度限りのfilterやscriptの方が固定formatより有効です。

## GUI

GUIは実行中のLandscapeとListenerTwinの状態をリアルタイムに表示します。レポートは、聴いて気づいたが
すべての数値を同時には覚えられなかった瞬間を、演奏後に読むためのものです。
