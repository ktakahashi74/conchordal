# 第2段階の実装・比較結果

2026-09-05。`a5447e1`からのローカル変更を対象とする。小節アクセント比較への作者の選好回答を含む。
前提と順序は[Manifesto整合・beta計画](manifesto-alignment-and-beta.md)、判断境界は
[作者への引継ぎ](author-decision-handoff-2026-09-05.md)に記録する。

## 小節アクセント

`.measure_accent(amount)`を追加した。範囲は0–1、既定値は0。
生成側が検出したmeasureの位相と確信度から、発音強度に
`1 + 0.35 * amount * confidence * cos(phase)`を掛ける。
位相原点は強いオンセットのphasorへ揃える。検出がなければ中立となる。
発音時刻を直接変更する処理ではないが、音の変化がmeterや生存へ戻るため、その後の軌跡は変わりうる。
固定拍子、専用の拍打ちVoice、temporal scaffoldは加えていない。

Sample 08の二つの`.entrained()`へ値0または1だけを加えた対照を、seed 1・21・42で生成した。
音源ごとの正規化は行っていない。

| seed | 最初のアクセントがあるhop（秒） | 有効時の発音強度の範囲 | 全体RMS差（有効−無効） |
|---|---:|---:|---:|
| 1 | 1.8133 | 0.8891–1.2209 | −0.129 dB |
| 21 | 1.5680 | 0.8540–1.2082 | −0.436 dB |
| 42 | 1.6853 | 0.8762–1.1769 | −0.141 dB |

無効時の発音強度は全て1。最初のアクセントより前はPCMと状態が一致し、その後はPCMに差がある。
全6音源にクリッピングと全無音はない。強拍位置を反転した検出対照、無効・未検出・確信度ゼロの中立性、
一様なオンセットを実際の発音エンジンへ戻す60秒の対照を検査した。
一様入力の対照では、終盤に架空のmeasureを作らない。

実音側のmeasure確信度は一律には改善しない。平均値はseed 1で0.124→0.097、
seed 21で0.133→0.199、seed 42で0.094→0.133だった。
したがって「小節が音楽として聞こえる」は機械判定で埋めない。
作者には、**数拍ごとに反復する強弱を聞き取れるか、どちらを採用するか**を別々に尋ねた。

作者の回答は「Bがこのましい」。直前の案内が「まずseed 1」だったため、seed 1への回答として記録する。
回答自体にはseedの明示がないため、seed 21・42へは一般化しない。
seed 1のBは小節アクセント追加なし（`measure_accent=0`）、Aは追加あり（`1`）である。
対応表と、試聴用・元renderの両WAVのSHA-256を照合した。
BのSHA-256は`aac028f097e85af9121812070f0fe7adc8dc2a526b65d5ccc1b6cde56df1c142`。
この選好を受け、公開Sample 08は追加なしの現状を維持し、機構の既定値も0を保つ。
反復する強弱を聞き取れたかは未回答であり、選好から検出成功・失敗を推定しない。
回答と対応条件は成果物内の`listening_feedback.json`にも保存した。

成果物: `target/stage2-evaluation/2026-09-05-measure-final/`。
再実行: `python3 scripts/evaluate_stage2.py measure --output <新しい出力先>`。
`audition.html`はA/Bを伏せた共通プレイヤー、対応表は`audition_mapping.json`に保存した。

## Habituationの回復と再訪

`--report`に生成時刻1秒ごとの`habituation_scan`を追加した。Log2Spaceの座標と、状態・raw score・
effective scoreの全scanを記録する。借用したsliceから出力し、reportなしのhop経路には追加しない。
移動する`tracked_bin`は従来の意味を維持する。

220 Hzを10秒提示し、16秒取り去り、10秒再提示した。固定した同じビンの値は次のとおり。

| 条件 | 9秒（刺激中） | 25秒（退出後） | 35秒（再提示） |
|---|---:|---:|---:|
| off | 0 | 0 | 0 |
| 回復8秒 | 0.76631 | 0.01233 | 0.76646 |
| 回復80秒 | 0.76631 | 0.51088 | 0.77420 |

表はseed 1。seed 21・42も同じ関係を満たす。これは実際の音声解析からの回復と再応答の検査であり、
スクリプトによる再提示を自律的な再訪とは呼ばない。

既存の60秒の再訪assayでは、「持続する基準音」が約10秒で死亡していた。
研究sampleの基準音を`brain("drone")`に修正し、全測定区間の生存と各基準音のonsetが1回だけであることを検査した。
専用の拍打ちは加えていない。

固定した50-cent領域を使い、同じVoiceが100 cent以上離れた発音を示した後、2秒以上経て同じ領域へ戻る候補を数えた。
観測がないだけでは退出と扱わず、別Voiceやrespawnを同じVoiceの帰還として数えない。

| 条件 | seed 1 | seed 21 | seed 42 |
|---|---:|---:|---:|
| off：再訪候補 | 3 | 2 | 2 |
| 回復8秒：再訪候補 | 62 | 66 | 64 |
| 回復80秒：再訪候補 | 55 | 49 | 53 |
| 回復8秒：退出後に状態が20%以上低下した候補 | 36 | 19 | 17 |
| 回復80秒：同じ条件の候補 | 6 | 2 | 2 |

隣接領域の候補は相関するため、この件数を独立した周期数や統計的有意性と解釈しない。
また、これは退出中の状態低下の検査であり、再訪直前の盆地の順位回復までを保証しない。

さらに、研究試験のscenario IRでVoice固有の適応を切り、temperatureとcrowdingもゼロにした9条件を実行した。
同じruntime wiringと音声解析を通した結果、habituation有効時には3 seedとも再訪候補が0となった。
offは0・1・0。この条件ではhabituation単独で反復的な帰還を生む証拠が得られていない。
この対照は三つの制御を同時に変えるため、Voice固有の適応だけの因果効果を同定する試験ではない。
保存reportを再検査すると、全9条件で40〜59.9秒の移動Populationの生存数は5を維持し、
全Voiceが59秒以降にも発音していた。再訪候補0を集団の消滅や発音停止で説明する結果ではないが、
移動が止まる原因までは特定していない。固定110・165・220・330・440 Hzの状態も保存した。
追検査は`target/stage2-evaluation/verification-2026-09-05/decision-audit.json`に記録する。
**固定領域の回復は検証済み。自律的な閉ループの総合判定はPARTIAL、既定値はoffを維持する。**
新しい帰還機構や既定値の変更を、この結果から自動的に導入しない。

成果物: `target/stage2-evaluation/2026-09-05-habituation-persistent/`、
`target/habituation-isolation/1788611323/`。
再実行: `python3 scripts/evaluate_stage2.py habituation --output <新しい出力先>`、
`cargo test --lib habituation_isolation_campaign -- --ignored --nocapture`。

### 2026-09-06：Voice固有の適応だけを外す対照

前の対照に残った温度との交絡を切り分けるため、現在の時間参加実装で27条件を実行した。
現行設定、移動Populationの適応だけを無効化、適応を無効化してtemperature・crowdingも0、
の3群と、habituation off／回復8秒／回復80秒、seed 1・21・42を組み合わせた。
今回は移動するPopulation 3の5 Voiceだけを変更し、2つの持続する基準音は全群で同じ設定にした。
crowdingはこのfixtureでは元から0であり、後の2群間の実際の差はtemperatureの指定だけである。

コンパイル後のscenario IRを保存し、適応だけを外した群では`AdaptationControl.enabled`の
true→falseだけが変わったことを9組すべてで照合した。configと初期spawnの記録も対応する
3群で完全一致した。この無効化はVoice側のboredom・familiarity・self補正を外すもので、
共有環境のHabituationFieldや時間的な参加方策を無効化する操作ではない。

| 群・条件 | seed 1 | seed 21 | seed 42 |
|---|---:|---:|---:|
| 現行設定・habituation off：再訪候補 | 0 | 10 | 0 |
| 現行設定・回復8秒：再訪候補 | 49 | 72 | 36 |
| 現行設定・回復80秒：再訪候補 | 58 | 44 | 50 |
| 適応だけ無効・habituation off：再訪候補 | 0 | 0 | 0 |
| 適応だけ無効・回復8秒：再訪候補 | 0 | 0 | 0 |
| 適応だけ無効・回復80秒：再訪候補 | 0 | 0 | 0 |
| 適応・temperature・crowding無効：各habituation条件 | 0 | 0 | 0 |
| 現行設定・回復8秒：退出後に状態が20%以上低下した候補 | 18 | 20 | 16 |
| 現行設定・回復80秒：同じ条件の候補 | 3 | 4 | 7 |

全27条件で、40〜60秒の移動Populationは5 Voiceを保ち、全Voiceが58秒以降にも発音した。
基準音は60秒の測定区間を生存し、各onsetは1回だけだった。再訪候補0を集団の消滅や
発音停止で説明する結果ではない。habituation offでは全scanの状態0とraw/effective scoreの
同一性も確認した。再訪候補の定義は前節と同じで、同じVoiceの発音時点での周波数に基づく。
隣接領域の件数は相関し、独立した周期数や音楽的な価値を表さない。

今回のfixtureでは、温度を保ってもVoice固有の適応を外すと再訪候補が消えた。
これで適応だけの条件付き効果を切り分けたが、Habituation単独による閉ループの成立は
示していない。回復8秒の総候補数が80秒より常に多いわけでもなく、回復時間と再訪件数の
単調な関係は主張しない。状態の低下と、元の盆地の順位回復・帰還の因果関係は別の検証である。
総合判定はPARTIAL、既定値offを維持する。

成果物: `target/stage2-evaluation/2026-09-06-habituation-adaptation-isolation/`。
`manifest.json`が実行先`target/habituation-isolation/1788691705/`を指し、
`adaptation-assessment.json`に設定差、初期状態、再訪候補、生存・発音、固定領域を保存した。
ソース191ファイルと実行したtestバイナリも保存した。通常704テスト、研究campaign 1件、
Clippy全targetが通過し、全出力・終了値を記録した。変更は`src/runtime/mod.rs`の
`#[cfg(test)]`内の比較処理だけで、通常の演奏コードは前版と同一。音声ファイルは生成していない。
再実行コマンドは前節と同じで、今回から27条件を生成する。

## DCCの実用範囲

`max_temperature_bonus=0.1`を固定し、結合0・0.1・0.25・0.5・1を3 seedで比較した。
結合ゼロと既定値の一致、`pressure = tension * strength`、bonus上限を確認した。
解析欠落、空の受信待ち、古い解析結果では圧力ゼロを保ち、有効な解析が戻った後だけ再開する回帰も通過した。

厳密な固定入力には、研究試験内のscenario IRで寿命20秒のSeqGate基準音を使う。
Droneは共有リズムで振幅も変わるため、busだけを分けても入力固定を保証しない。
固定入力試験の全15条件では、seedごとにpresentation WAVとlistener観測が完全一致した。
そのうえで、**移動するVoiceの音高軌跡も全強度で同一**だった。bonus最大値は結合1で0.00698。
この入力は移動するVoiceから独立しているため、解決時間の測定対象にはしない。

別の閉ループ試験では、移動するVoiceをhabitatとpresentationの両方へ送る。
持続基準音を保ち、音高を強制的に変えず20秒観測した。

| seed | 結合0の音高移動総量 | 結合1と0の最大音高差 | 低tensionへ移るまで |
|---|---:|---:|---:|
| 1 | 612.8 cent | 0 cent | 1.079秒 |
| 21 | 1171.0 cent | 0.0274 cent | 2.199秒 |
| 42 | 247.8 cent | 0 cent | 0.620秒 |

低tensionは「配置後に0.05を超え、その後0.05以下が1秒以上続く」という診断条件であり、音楽的な終止ではない。
時刻は全強度で同じ。最後の5秒の音高移動は全条件0 centだった。
seed 1・42は結合0と1のPCMも一致する。明確な探索・解決の改善を示していないため、
強度を選ぶ試聴を必須にはせず、既定値0を維持する。

成果物: `target/stage2-evaluation/2026-09-05-dcc-isolated/`。
再実行: `python3 scripts/evaluate_stage2.py dcc --output <新しい出力先>`。
runnerは閉ループのrelease renderと、固定入力の研究用test binaryを別々に保存する。
後者は公開Rhai APIへ研究用の設定を足さず、既存IRとruntime wiringを使う。

## 検証と成果物の同一性

- 通常のRust全671テスト、Clippyを通過。研究用の長い2 campaignは通常実行ではignoreし、別途どちらも実行・成功した。
  必須の全出力と終了値は`test_report.txt`、`test_status.txt`に保存した。
- Python比較ツール52テストを通過。研究sample本体の固定seedを維持し、保存した比較用コピーだけCLIのseedへ委ねる。
- `cargo check --all-targets`、日英mdBook、Zolaのローカルビルド、`git diff --check`も通過した。
- 各renderのsource snapshot、設定、seed、バイナリ、WAV、reportを保存した。小節アクセントのrelease renderは
  `5c0d927b7802462632a3e960a2fb5f1b01ea3c28b67db8639bdfeb12b45cc210`。
  他のcampaignは各`manifest.json`と`stage2_manifest.json`を正本とする。
- 実機は64 harmonic Voice、warmup 5秒＋測定30秒。アクセント1、habituation on、DCC 0.25を同時に有効化した。
  固定scaffoldなし。測定中3904 onsets、強度0.8659–1.1217、全38回のhabituation scanを確認した。
  M2経路で出力不足・callback errorともゼロ。hop p99 3.538 ms、最大3.945 ms、予算10.667 ms。
  これは新しい経路の短時間試験であり、同期8 Hzや10分の性能を新バイナリで再認定するものではない。
  記録は`target/stage2-rt-20260905/`。profile buildは
  `499f57a8f2d739be10c880673cd66b71b1532c20ee00ae16c7259adf9cd8c31f`。

seedの上書き、基準音の寿命、Droneの入力非独立性で失敗した先行campaignも残した。
それらを最終結果へ合算しない。日英API参照はregistryから生成し、今回変更した技術ノートの節を同期した。
技術ノート全体の再監査やbeta全体の完了を宣言するものではない。
