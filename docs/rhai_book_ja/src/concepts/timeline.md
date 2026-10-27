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

## シーン全体の設定と途中の変更

大域的な設定の名前は、いつ適用されるかを表します。`set_`のない
`meter_stability`、`temporal_basin`、`control_update_mode`、`scaffold_off`、
`scaffold_shared`、`scaffold_scrambled`、`seed`、`output_gain`はシーン全体の初期条件です。
`wait()`の後に書いても、最後の指定が全体へ適用されます。`temporal_mode`も
初期条件ですが、イベントを置く前に最上位で一度だけ宣言します。

`set_roughness_aversion`、`set_global_coupling`、`set_pitch_objective`は、
その時点のスクリプト時刻に変更を置きます。`wait()`で変更の時刻を選べます。
配置、解放、有効範囲を作る語は、引き続き操作そのものを表します。

## 曲全体の再生音量

`output_gain(db)`で曲全体の再生音量をdB単位で指定します。整数と小数のどちらも使え、
省略時は`0` dBです。`seed`と同じく、`wait()`の後の宣言も含めて最後の指定が曲全体に
適用されます。途中の音量変化やフェードを作る操作ではありません。

```rhai
output_gain(3.0);
play(|| {
    place(sine().amp(0.04).sustain(), at(220.0));
    wait(2.0);
});
```

指定値は設定ファイルの`[audio].output_gain_db`（既定値`+4.5` dB）に加算され、
振幅を`10^((設定のdB + 曲のdB)/20)`倍にします。楽器のスピーカー出力と、オフライン描画の
WAV出力の両方で、出力リミッターの直前に適用されます。リミッターを無効にしても音量指定は
有効です。曲の指定を`0`にすると、設定ファイルで定めた再生音量を保ちます。音量を上げると、
ピークを抑えるリミッターが働く場合があります。

変わるのは再生音量だけです。音響解析、ListenerTwin、生態、レポート、UIの音量表示には
元の水準の音が渡ります。Populationの発音そのものや、生態に与える影響を変えるときは
`amp()`を使います。

値は有限で、振幅倍率が有限かつ正の`f32`として表現できる必要があります。設定値と加算した
結果も同じ条件で検査し、不正な指定には宣言位置を付けてエラーを返します。この制限は
`f32`の表現範囲によるものです。作品の音量目標や、それ以外のdB範囲は設けていません。

## 既知の終わりへ向けた表情

`expressive_section(name, arch_weight, ritard_weight, callback)`は、作者が明示した
最後の区間に強弱の弧と減速を付けます。Populationの所有と自動解放は`section`と同じです。
二つの重みは初期の指定だけに使い、有限の非負値を受け取ります。0では中立、1以上では
完全適用に飽和します。語を使わなければ従来の音を保ち、両方0なら`section`と同じ処理を通ります。

```rhai
temporal_basin(1.6, 2.4);
meter_stability(0.55);
let pluck = modal().metric()
    .modes(stiff_string_modes(0.004).count(12))
    .brightness(0.42).adsr(0.004, 0.18, 0.0, 0.35).amp(0.04);
let pulse = place(pluck, line(196.0, 392.0).count(3));
section("opening", || { wait(4.0); });
expressive_section("ending", 1.0, 1.0, || {
    wait(8.0);
    release(pulse);
});
```

曲線にはTodd（1992）の単層の強度とFriberg–Sundberg（1999）のq=3の減速を使います。
これらは候補モデルです。作者の区間に割り当てること、振幅の床を加えること、生成時計を
一時的に固定することは作者規則です。句や終止を知覚したという意味ではありません。
既定の聴き手の型へ自動で加える機能でもありません。

初版の対象は、自由音高の再発音を伴わないOnceのVoiceと、持続しない同期型の`metric()`の
Voiceです。Flora、参加型の時計、持続する`metric()`、`follow_own_sequence`との併用、
`rest`は対象外です。対象外を含むシーンでは警告を出し、表情の効果全体を無視します。
既存の語の指定は消しません。非零の区間は一つだけとし、全Populationを終端までに解放します。
例のように外側で作ったPopulationには明示的な`release`が必要です。関数を抜ける際の自動解放は
内部で作ったPopulationだけに働きます。区間後の配置や更新はできません。不正な重みや区間も
警告して効果を無視します。装置の設定時には、Onceの発音と解放が同じ配送hopに入る場合と、
変換後600秒を越える場合を、時間写像の適用前に診断します。

強弱は既に鳴っている余韻を含め、両busへ分ける前の発音に作用します。発音行為の直接の
エネルギー更新には掛けません。完全適用では区間中央の振幅が√2倍になります。
減速の完全適用では区間長が約1.403倍になり、末端の速度は入口の0.3倍になります。
同期時計の入口の位相速度を保存し、区間内では速度の再抽選も停止します。感覚の観測、生態、
身体の包絡は実時間のままです。gate単位の音長は同じ写像に従い、解放の開始は遅くなりますが、
その後のフェードは元の秒数を保ちます。入口より前に予約済みのOffは元の実時刻を保ちます。
減速の重みは正ならどれだけ小さくても時計を固定し、0なら従来の時計を維持します。
区間名や予定された終端をListenerTwinの終止の知覚の根拠へ渡すことはありません。

Once は出生時に発音する形に限ります。Once の `phonate_when_viable()` は初版の対象外です。
