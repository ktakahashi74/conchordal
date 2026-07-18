# タイムラインと構造

シナリオは音響標本を一つずつ予定表へ並べません。スクリプト上の時刻を進め、その時点で
Populationを配置し、更新または終了する時刻を記述します。

## `place`、`wait`、`flush`

`place()`は現在時刻に初代のVoiceをただちに配置し、安定した`Population`への参照を
返します。`wait(seconds)`は保留中の更新を発行して時刻を進めます。`flush()`は
時刻を進めずに保留中の更新だけを発行します。

```rhai
let spec = harmonic()
    .amp(0.04)
    .brightness(0.35)
    .sustain();

let population = place(spec, consonance(220.0).count(3));
population.amp(0.025); // Live patch at the same script time.
flush();               // Emit it; the cursor does not move.
wait(2.0);             // Advance two seconds.
release(population);
```

意味上の境界は`wait()`や`flush()`ではなく`place()`です。発音体、振る舞い、生存過程、
再生成の初期設定は`PopulationSpec`に属し、実行中の更新と終了だけが`Population`に属します。

## `release`、`section`、`play`

`release(population)`は明示的なreleaseです。`section(name, callback)`と
`play(callback, ...)`は有効範囲を作り、関数が戻ると内部で作られたPopulationを
自動的にreleaseします。有効範囲の終了時にはreleaseより前に保留中の更新を発行します。

名前の付いたformには`section`を使います。名前はreportの`scene_marker`になります。
引数を取る再利用可能なgestureには`play`を使います。

```rhai
let gesture = |root_hz, duration_sec| {
    place(
        harmonic().amp(0.035).sustain(),
        consonance(root_hz).peak().count(3)
    );
    wait(duration_sec);
};

section("two gestures", || {
    play(gesture, 110.0, 2.0);
    play(gesture, 165.0, 2.0);
});
```

各`play`が作るPopulationは呼び出しの終了時にreleaseされます。外側の`section`はレポートの印を
提供し、その直下で作ったものを所有します。

## Parallel timeline

`parallel([callbacks])`は現在時刻から複数のcursorをforkします。全branchは同時に始まり、
記述後の主時刻は最長の分岐の終端へ進みます。各分岐も一つの有効範囲であり、戻ると内部の
Populationをreleaseします。

```rhai
section("overlap", || {
    parallel([
        || {
            place(sine().amp(0.05).sustain(), at(220.0));
            wait(3.0);
        },
        || {
            wait(1.0);
            place(harmonic().amp(0.03).sustain(), at(330.0));
            wait(1.0);
        }
    ]);
});
```

第一branchはforkから3秒後、第二branchは2秒後に終わるため、main cursorは3秒進みます。
両分岐の有効範囲が重なる区間だけ、両方のVoiceが同時に鳴ります。

## スクリプト上の時刻と実行時の振る舞い

スクリプト上の時刻は制御事象の時刻を決めます。その時刻のLandscapeを予測するものでは
ありません。`wait()`でscenarioが進む間も、movement、coupled rhythm、death、respawnは
実行中も続きます。

- scriptはmacro structureを書く。
- Communityは瞬間ごとの振る舞いを生成する。
- レポートは、記述された区間で実際に起きたことを示す。

[演奏](../tutorial/observing.md)で説明した手順を使って結果を確かめ、scenarioを修正します。
特定の実現を再現するときだけ`seed(...)`または`--seed`を使います。
