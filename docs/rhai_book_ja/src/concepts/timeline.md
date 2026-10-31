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
終わりを定めた曲では、`wait()`の後に書いても、最後の指定が全体へ適用されます。`temporal_mode`も
初期条件ですが、イベントを置く前に最上位で一度だけ宣言します。

`set_roughness_aversion`、`set_harmonicity_preference`、`set_global_coupling`、
`set_pitch_objective`は、
その時点のスクリプト時刻に変更を置きます。`wait()`で変更の時刻を選べます。
配置、解放、有効範囲を作る語は、引き続き操作そのものを表します。

## 曲全体の再生音量

`output_gain(db)`で曲全体の再生音量をdB単位で指定します。整数と小数のどちらも使え、
省略時は`0` dBです。終わりを定めた曲では、`seed`と同じく、`wait()`の後の宣言も含めて最後の指定が曲全体に
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

## 終わりを定めずに演奏する

`endless()`を宣言すると、脚本が最後まで進んでも演奏は続きます。イベントが
尽きた後もVoiceは活動し、世代交代も続きます。すべてを解放した場合も自動終了せず、
無音のまま演奏者の終了操作を待ちます。`endless()`は最上位に一度だけ、再生イベントを
置く前、時刻を進める前に書きます。

```rhai
endless();
let drone = place(sine().amp(0.04).anchor().sustain(), at(220.0));
wait(2.0);
drone.amp(0.025);
flush();
```

GUIの **End** または一度目のCtrl-Cで、残っているPopulation、Flora、Rainをそれぞれに
指定した長さでフェードさせて終了します。すでに始まっているフェードも最後まで待ちます。
**Exit**、ウィンドウを閉じる操作、二度目のCtrl-Cは即時終了です。

`--duration <sec>`を指定すると、その演奏時刻に終了を要求します。同じ処理区間に
予定されたイベントを配ってから解放するので、出力には解放後のフェードや余韻も含まれます。
終わりを定めた曲が指定時刻より先に終わる場合は、そのまま終了します。

終わりを定めない曲を`conchordal-render`でWAVに書く場合、`--duration`が必須です。
楽器でも、この種の曲に`--report`や`--profile`を付ける場合は`--duration`が必要です。
計測記録には処理区間数の上限があるので、フェードの余地を残してください。
WAVは起動時の長さの検査に加え、書き込み中にもRIFF形式の上限を検査します。
楽器自体には録音機能はありません。

終わりを定めない曲には、次の三つの規則があります。

- `seed`、`output_gain`、`temporal_mode`、`scaffold_*`、
  `control_update_mode`、`meter_stability`、`temporal_basin`は、最初に時刻を
  進める`wait()`より前に宣言します。いずれも演奏全体の初期条件です。
- 自分自身の音を除外するPopulationには、配置前のPopulationSpecで
  `leave_self_out(true)`を指定します。その能力を後から新たに追加する更新はできません。
  配置時に指定してあれば、無効にしてから再び有効にする更新はできます。
- `parallel`は使えません。

終わりを定めない脚本は専用のスレッドで、演奏時刻の少し先まで逐次評価します。
終わらないループには、譜面時刻を進める`wait()`を入れます。

```rhai
endless();
loop {
    section("breath", || {
        let p = place(sine().sustain(), at(220));
        wait(4);
    });
    wait(2);
}
```

最初に時刻を進める`wait()`で演奏全体の初期条件を確定します。時刻を進めずに脚本が
返る場合は、その時点で確定します。時刻0のイベントは確定するまで実行しません。
`metric()`で拍に同期するVoiceの初期値には、最後の`temporal_basin`宣言を反映します。
最初のイベントより前に`wait()`を書いてもかまいません。初期条件の確定後に脚本で
エラーが起きると、以後の評価は止まりますが、確定済みのイベントと生態系の演奏は
続きます。System欄で評価済み時刻、完了、エラー、同じ呼び出し箇所ごとにまとめた
警告を確認できます。

繰り返し配置するPopulationやFloraは、明示的に解放するか、処理を終えて解放する
`section`や`play`の内側で作ります。Voiceが自然に死んでも、未解放のPopulationの
記録は残ります。配置やRhai配列を解放せず増やし続ける場合、その保持量は作者側の管理です。

終わりを定めない曲のコンパイルのみの実行とWAV出力には、`--duration`が必要です。
その範囲を越える`wait()`では、範囲の時刻で評価を止め、`section`や`play`の内側で
作ったPopulationとFloraを解放します。この解放は同じ時刻の演奏終了要求より先に
実行します。`--compile-only`では
`end: open (evaluated through t=…)`を表示します。有限曲は従来どおり一括評価し、
`parallel`も利用できます。1処理区間内の操作数や、操作に付随するデータの保持量が
受け渡し用のキューの容量を超えると、WAV出力とコンパイルのみの実行は、同じ譜面時刻で
容量超過のエラーになります。

## 脚本から使える機能

終わりの有無にかかわらず、すべての脚本で`eval`とファイルを読む`import`を禁止します。
`sleep`と`timestamp`もエラーになります。演奏上の時刻を進めるには`wait()`を使います。
実時間を`timestamp`や`elapsed`で読み取る経路と、`sleep`で評価を止める経路を閉じることで、
脚本とseedから演奏予定が定まります。診断用の`print`と`debug`は引き続き使えます。

### 調的文脈の観測を予約する

終わりを定めない曲で正の重みの`follow_tonal_context`を配置する場合、時刻を進める前、
その配置より前に`observe_tonal_context()`を宣言します。これは作者規則です。演奏開始時から
調的文脈の観測器と表を用意し、後から生まれるVoiceにもそれまでに聴いた履歴を渡します。
時刻が進んだ後の宣言と、宣言なしの正の重みの配置は、位置付きの脚本エラーになります。
有限の曲では宣言は不要です。書いた場合は警告して無効にし、従来どおり配置の先読みにより
観測器の要否を決めます。

終わりを定めない曲では、共有のonset habituationを演奏開始時から観測します。
後の`rest_on_habituation`の配置を予見できないためです。有限の曲は正の重みの利用者が
ある場合だけ観測器を用意します。`harmonic_rhythm`も、終わりを定めない曲では最初に
時刻を進める`wait()`より前に宣言します。
