# flow対照の修正と実行負荷の測定 — 2026-09-05

この文書の測定結果と初回試聴は、commit `935df53`までのpulseあり版を対象とする。
作者の方針によりsample 12の専用pulseを除去した。現行版の構成と検証は
[専用の拍打ちを除いた記録](sample12-without-beat-carrier-2026-09-05.md)を参照する。
以下の旧測定値は保存時のsource・操作時刻・Population番号に対する記録として保持する。

flowを除いた脚本でも将来のVoice ID予約を揃える修正を実装した。
sample 12の24条件を再実行し、flow有無の全12組で、登場前18.7秒までのPCMと状態記録の完全一致を確認した。
続いてMOTU M2へ出力する36条件の負荷試験がすべて成功し、測定窓の出力不足はゼロ、
最大hop p99は2.651 msだった。条件と制約を以下に記録する。

## flow比較の契約

`conchordal-render --reserve-runtime-ids-through N`を追加した。
runtimeで再生成するVoiceのIDを、指定値とScenario中の最大IDのうち大きい方より上に配置する。
既定値0は従来の予約を維持し、小さい指定でもScenarioの予約範囲を縮めない。
`u64::MAX`はCLIと実行APIの両方で拒否する。Rhaiの新しい操作やダミーPopulationは追加していない。

```bash
python3 scripts/evaluate_resolution.py
```

比較runnerは全variantへ19を指定する。root 1、pulse 1、colony 8、flow 9の合計に対応する。
配置数、操作文、wait列、各操作までのwait数を検査するため、配置数や操作時刻が変わると失敗する。
flow配置を18.7秒から24秒へ移しても古い計測窓のまま実行できる穴も塞いだ。

受入条件は同じ温度・音高移動・seedのflow有無について、半開区間[0,18.7)秒の次の一致とする。

- PCMの形式、frame数、全sampleのSHA-256。
- spawn、respawn、death、onset、Population、rhythm、ListenerTwin、DCC、Habituation、発音ゲートの記録。
  wall-clockのhop時間と終了時の集計は比較から除く。
- 記録の欠落や不一致はcampaign全体の失敗とする。確認結果と再生成IDは`flow_pre_intervention.json`へ保存する。

## 再実行結果

- [24条件の試聴・結果一覧](../../target/resolution-evaluation/2026-09-05-matched-ids/index.html)
- [介入前12組の一致](../../target/resolution-evaluation/2026-09-05-matched-ids/flow_pre_intervention.json)
- [検証記録](../../target/resolution-evaluation/2026-09-05-matched-ids/validation.json)・[manifest](../../target/resolution-evaluation/2026-09-05-matched-ids/manifest.json)
- [旧campaignを拒否する陰性対照](../../target/resolution-evaluation/2026-09-05-matched-ids/previous_campaign_rejection.json)

全24条件が成功した。全12組の介入前PCM・状態が一致し、seed 42の最初の再生成IDは
flow有無とも20となった。旧campaignへ同じ検査を適用すると、問題があったseed 42の4組だけが
失敗し、残る8組は一致した。flowありの全12WAVは、前回の音量・glide修正版と完全一致した。
保存したsource・設定・バイナリ・WAV・JSONLのハッシュも一致し、全編無音やPCM飽和標本はなかった。

以下は早期窓24.0〜27.3秒と直前窓18.7〜24.0秒の平均tension差について、
flow ONからOFFを引いた対応差である。温度・音高はON=1、OFF=0とする。

| 温度・音高 | seed 1 | seed 21 | seed 42 |
|---|---:|---:|---:|
| 00 | +0.009793 | +0.000608 | +0.003953 |
| 01 | +0.000273 | −0.012801 | −0.010475 |
| 10 | +0.006776 | +0.011187 | −0.002577 |
| 11 | +0.002133 | +0.004728 | −0.008056 |

seed 42では、温度・音高の組によりflowの対応差の符号が変わる。
4組を平均した値も前回の+0.003918から今回は−0.004289へ変わった。
将来のID予約による交絡は、効果の方向の解釈にも影響していた。
旧比較の値をflow集団の寄与としてそのまま使わない。

今回も3seedすべてで早期tensionが低下するvariantはなく、111の音は前回と変わっていない。
これは対照条件の修正であって、新しい解放効果を音へ加えた変更ではない。
flow登場後は配置抽選、集団構成、局所相互作用が変わるため、その後の乱数系列の一致は保証しない。
比較するのはflow集団を加える操作の効果であり、flowの音だけを加算する効果ではない。

## 負荷計測の実装

楽器へ`--nogui --profile PATH`を追加し、通常reportの有無とは独立に負荷を記録できるようにした。
workerの全hop時間、測定中の生存Voice数、機器情報、出力不足、callbackエラーを保存する。
`profile-alloc`を有効にしたビルドでは、worker threadのRust割当要求の回数とbytesも記録する。
profile保存用の容量は開始前に確保し、JSON書き込みは終了時だけ行う。
音声ファイルを保存する経路は楽器へ追加していない。

同時に次の不備を修正した。

- headlessで要求した音声機器の初期化に失敗しても、計算だけの実行へ切り替わって正常終了し得た。
  初期化失敗を非0終了にした。
- 終了hopの後もringに空きがあると、同じframe IDで処理を繰り返し得た。
  終了状態で内側の生成ループを抜けるようにした。
- profileの作成・書き込み失敗と容量超過を終了値へ伝え、欠落した計測を成功扱いしないようにした。

runnerは既知の仮想sink、callback未観測、再生進行が示されない条件を性能合格にしない。
大きいringに音を貯めただけで成功する境界も回帰テストで検査する。
詳細な測定範囲と制約は[負荷比較手順](runtime-load-evaluation.md)を参照する。

## 実機36条件の結果

- [全条件CSV](../../target/rt-evaluation/2026-09-05-device-host/summary.csv)・[JSON](../../target/rt-evaluation/2026-09-05-device-host/summary.json)
- [manifest](../../target/rt-evaluation/2026-09-05-device-host/manifest.json)・[検証記録](../../target/rt-evaluation/2026-09-05-device-host/validation.json)
- [計測中の物理出力接続](../../target/rt-evaluation/2026-09-05-device-host/device-route.json)
- [sandbox内での音声初期化失敗](../../target/rt-evaluation/2026-09-05-device-sandbox/device_blocker.json)

最初はsandbox内で`Audio init failed: No default config`となり、残る35条件を省略した。
sandbox外で同じ手順を実行すると音声接続に成功し、全36条件を完走した。
PipeWireの接続情報から、計測中のconchordalストリームが`hw:M2`の物理sinkへ
2本のactive linkで接続していることも確認した。これは一条件の実行中の観測で、全期間の経路監視ではない。

```bash
python3 scripts/evaluate_rt.py --mode device
```

測定機はAMD Ryzen 9 9950X、32 logical CPU、affinity 0〜31、Linux 7.0.0-30-generic。
Rust 1.95.0、release、既定のsimd-wideとprofile-allocを有効にした。
基準commitは`390ae1e5c382c87dc1abae8bfd0673b3e778529d`と未コミット差分で、
source 174ファイルと実行バイナリを保存した。バイナリSHA-256は`94607a1a8124bf9f`で始まる。

CPAL backendはALSA、device名は`Default Audio Device`、実sample rateは48000 Hz、
2 channels、hop 512 samples、ring容量4800 mono framesだった。
seed 1、warmup 5秒、要求測定窓5〜15秒の中の完全な937 hopを各条件で集計した。
実際の集計窓は5.002667〜14.997333秒で、release後の待機は2秒である。
自分のビルド・Rust/Pythonテスト・別campaignのrenderを完了してから、36条件を逐次測定した。

下表のp99はreport無効/有効 × DCC 0/0.25の4条件中の最大値。
割当は各条件の1 hop平均の範囲である。

| Voice数 | 音色 | 最大hop p99 ms | 割当回数/hop reportなし | 割当回数/hop reportあり |
|---:|---|---:|---:|---:|
| 4 | sine | 0.164 | 16.14〜16.15 | 23.14〜23.15 |
| 4 | harmonic | 0.140 | 16.11〜16.12 | 23.12〜23.13 |
| 4 | modal | 0.145 | 16.14〜16.18 | 23.15〜23.18 |
| 16 | sine | 0.564 | 18.58〜18.74 | 27.59〜27.75 |
| 16 | harmonic | 0.505 | 18.59〜18.69 | 27.60〜27.69 |
| 16 | modal | 0.448 | 18.59〜18.71 | 27.62〜27.74 |
| 64 | sine | 2.651 | 22.41〜22.58 | 38.39〜38.58 |
| 64 | harmonic | 1.943 | 22.42〜22.77 | 38.42〜38.77 |
| 64 | modal | 1.395 | 22.38〜23.27 | 38.39〜39.28 |

全36条件で、測定窓の生存Voice数が指定数と一致し、出力不足とcallbackエラーはゼロだった。
hop時間10.667 msに対し、個別hopの最大時間も6.162 msに収まり、測定窓内の超過hopはなかった。
workerの割当要求量は条件平均で約27.1〜36.7 kB/hopだった。保持メモリ量ではなく、
analysis thread・callback・native mallocの割当は含まない。

今回の固定sustain集団、3音色、単一seed、10秒の測定窓では64 Voiceまで性能条件を満たした。
短い音を密に重ねる作品、より多いmode数、長時間演奏、別の機器やCPUにも同じ上限を保証する結果ではない。
出力不足はworkerの生成時刻に沿って採ったring不足の累積差であり、hardware xrunの測定ではない。
offline版の待ち時間込みp99と直接比較して処理が高速化したとは判断しない。
今回は音声機器を実際に使用できたため、同じ36条件のoffline-checkは追加実行していない。

全source・設定・バイナリ・各条件の成果物のSHA-256を確認した。負荷campaign内にWAVは作成されていない。
通常Rustテスト659件、profile-alloc付き660件、Pythonテスト35件が成功した。
両ビルドでClippyと全ターゲット検査、format検査も通過した。
通常テストの全出力・終了値は`test_report.txt`・`test_status.txt`、
feature付きの記録は`target/rt-validation/feature-test-report.txt`・`feature-test-status.txt`に保存した。

## 作者の試聴報告（2026-09-05）

[修正前後の試聴](../../target/resolution-evaluation/2026-09-05-controls-fixed/comparison.html)を案内した後、
作者から「glideするようにはなっている。 音楽的にはわからない」と報告があった。
続いて「glideがonになって、協和音におちつきやすいことはわかる。 音量についてはわからない」と補足があった。
現在の評価は次のとおりとする。

- glideの可聴性に加え、glide ONの試聴条件で協和音へ落ち着きやすいことが聞き取れた。
- 音量操作の効果は判定保留である。
- 曲の区切り・終止感については、まだ肯定・否定の判断が示されていない。

局所的な協和への落ち着きについて作者の確認を得た。これは今回の試聴条件の評価であり、
glideだけの因果効果や曲全体の受入条件を検証した結果とは区別する。
聴いたseed、版ごとの優劣、変化を感じた時刻は指定されていないため、
個別条件の試聴票には結果を転記しない。

## 次の判断

専用pulseを除いた現行sample 12は、試聴ページの案内後に作者確認を得た。
続く音量比較ではglide ONと基準音の帰還を共通にし、flowの減音の有無だけを比較する2条件に絞った。
現行版では探索冷却、glide切替、基準音の帰還、flowの減音が20.3秒に移っている。
以下の条件でseed 1の[2音源とA/B試聴ページを生成した](flow-amplitude-audition-2026-09-05.md)。
介入前のPCM・状態一致を確認し、作者は「差は聞き取れない／落ち着きは同程度」と判定した。
この条件で減音の音楽的な寄与は確認できなかった。終止感の判定は残っている。

| 条件 | 20.3秒以降のflowのamp |
|---|---|
| 減音なし | 0.020を維持 |
| 減音あり | 0.014へ変更（現行sampleと同じ） |

探索冷却・glide切替・基準音の帰還・Voice配置・seedは共通にする。
介入前[0,20.3)秒のPCMと状態記録の一致を検査する。
まず1seedの2条件で、同じ前半を聴いてから20.3〜23.6秒の変化を比べる。再生音量は固定し、音源別の音量正規化は行わない。
23.6秒のflow解放以降は別の操作を含むため、この比較窓には含めない。
条件名を隠したA/B表示を用い、まず差が聞き取れるか、次に協和への落ち着きの印象が変わるかを
「判断できない」も含めて記録する。提示順と条件の対応も保存する。
この比較は20.3秒のflowの減音を対象とし、glide単独の効果や24.9秒のcolonyの減音には一般化しない。
flowの減音は共有Landscapeにも影響するため、完成音源全体の再生音量を下げる比較とは区別する。

tensionの値だけを目標にした係数調整は先行させない。
先行文脈による帰還感を扱う必要が確認できた場合に、presentationだけを読む受動的な短期期待モデルを具体化する。
今回の音量差の非検出や旧版での判断保留だけを根拠に、新しい期待・終止モデルが必要とは判断しない。
今回の性能条件ではworkerに余裕があったため、割当をゼロにする目的だけの改修を先行させない。
より密な発音や長時間演奏で未達を検出した場合に、残る割当箇所やanalysis/callback側の負荷を絞って調べる。
