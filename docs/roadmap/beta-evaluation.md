# beta 比較の実行手順と読み方

更新日: 2026-09-05

この文書は、[Manifesto整合計画](manifesto-alignment-and-beta.md)の第1段階として、
現行版の動作と音を同じ条件で比較する手順を定める。個別の実測値と試聴結果は実行後に記録する。
音楽的な合格、実機のリアルタイム性能、メモリ割当数は、この手順を用意した時点では未判定・未測定である。

## 標準の12条件

リポジトリのルートから実行する。

```bash
python3 scripts/evaluate_beta.py
```

既定では次の4 sampleをseed `1`・`21`・`42`で比較する。sample本体にはseedを追加しない。

| Sample | 調べる挙動 | 現行sampleのPopulation対応 |
|---|---|---|
| [07 Heartbeat](../../samples/07_heartbeat.rhai) | 拍の可聴性、声部追加とoffbeatの影響 | 1: habitatのroot、2: beat、3: voices、4: cross |
| [08 Murmuration](../../samples/08_murmuration.rhai) | 同期の立ち上がり、集団の生存と交代 | 1: habitatのroot、2: colony a、3: colony b |
| [09 Rain](../../samples/09_rain.rhai) | 非拍節的なまとまり、間隔のばらつき | 1: habitatのroot、2: low stream、3: high drops |
| [12 Emergence and Resolution](../../samples/12_emergence_and_resolution.rhai) | 拍、緊張と解放、構成の区切り | 1: habitatのroot、2: pulse、3: colony、4: flow |

Population番号は保存したsampleの配置順に対応する。sampleを変更した場合は対応も確認する。
複数seedの結果を個別に残し、一つのseedの成功をsample全体の成功へ置き換えない。

## 実行と成果物

runnerはrelease版の`conchordal-render`を使い、各条件のWAVと`--report`のJSONLを
同じ実行から得る。楽器`conchordal`には音声保存機能を追加しない。
設定、sampleのソース、実行条件のmanifestと生データを保存し、`index.html`を比較と試聴の入口にする。
試聴票は`listening.csv`であり、初期状態は未判定となる。判定はCSVへ手動で記入する。
`index.html`での再生操作だけでは試聴票は更新されない。

条件を絞る例:

```bash
python3 scripts/evaluate_beta.py \
  --samples 07_heartbeat 12_emergence_and_resolution \
  --seeds 1 21 \
  --config config.toml \
  --output target/beta-evaluation/selected-run \
  --timeout 300 \
  --warmup-sec 2
```

| 引数 | 意味と既定値 |
|---|---|
| `--samples` | sampleの拡張子を除いた名前、または`.rhai`ファイルのパス。複数指定は空白区切り。既定は表の4 sample |
| `--seeds` | seedを空白区切りで指定。既定は`1 21 42` |
| `--config` | 保存して使用する既存TOML。既定はリポジトリルートの`config.toml` |
| `--output` | 新規の出力ディレクトリ。省略時は`target/beta-evaluation/<UTC時刻>/`。既存ディレクトリは拒否する |
| `--skip-build` | releaseビルドを省略して既存binaryを保存・使用。sourceとbinaryの対応は未検証としてmanifestへ記録する |
| `--binary` | 使用する既存release版rendererのパス。`--skip-build`との併用が必要 |
| `--timeout` | 各条件のrenderの制限時間、単位は秒。既定300。ビルドには別の制限時間を使う |
| `--warmup-sec` | hop時間統計から除く先頭のシミュレーション秒数。既定2。ListenerTwinとrhythm集計の窓は変更しない |

標準実行では`cargo build --release --locked --bin conchordal-render`でビルドする。
新しいsourceの比較には標準実行を使う。

| 出力 | 内容 |
|---|---|
| `manifest.json` | HEAD、対象source・設定・binaryのSHA-256、版情報、実行コマンドと成否 |
| `config.toml`、`source/`、`bin/` | 実行に使った設定、対象source、binaryのコピー |
| `tracked-head.patch`、`staged.patch`、`unstaged.patch` | 保存対象sourceのHEAD差分、stage済み差分、未stage差分 |
| `summary.csv`、`summary.json` | 条件別の主要指標。全体のrhythm値が含まれるため、Population別の結果も併読する |
| `index.html`、`listening.csv` | WAV再生と記録への入口、手動記入する試聴票 |
| `<連番_sample>/seed-N/audio.wav` | その条件のpresentation音声 |
| `<連番_sample>/seed-N/report.jsonl`、`run.log` | 同じ実行のJSONLとログ |
| `<連番_sample>/seed-N/metrics.json` | Population別rhythm、ListenerTwin時系列・統計、hop時間統計などの詳細 |

sourceコピーと差分の対象は、Cargo設定、Rust source、sample、実行script、test、AGENTS等の
実行再現に使う許可対象である。リポジトリ全ファイルの複製ではない。
各条件の`ok`とcampaignの`complete`は実行・成果物検証の成否を表し、音楽的な合格を表さない。
失敗・timeoutはその状態とログを残し、成功条件の数値で補完しない。

比較結果を扱うときは、commitだけでなく未コミット差分、実行に使ったsource、設定、
seed、sample rate、hop、ビルド条件を一組として保持する。
CLIで要求したseedとJSONLの`meta.seed`が一致していることも確認する。
旧版の[測定表](v0.4.0-rhythm-report-runs.md)は再設計前の記録であり、新しい結果へ混ぜない。

## 指標を読むときの区別

### 集団全体、Population、Voice

現行JSONLのキーは`population_id`と`population_step`である。
[過去のrhythm schema](v0.4.0-rhythm-report-schema.md)にある`group_id`・`group_step`を
現行ファイルのキーとして使わない。`population_id: null`の`rhythm_summary`が全体集計に当たる。
onsetのなかったPopulationにはrhythm summaryがないため、生存状態は`population_step`も確認する。

| 指標 | 対象と解釈 |
|---|---|
| `onset_count`、`onset_density_hz` | 生成側の発音イベント数。複数Voiceの同時発音も個別に数える |
| `ioi_cv`、`beat_stability`、`burstiness` | 対象イベント列の間隔統計。全体、Population、各Voiceで意味が変わる |
| `mean_plv` | attack PLVを観測した発音イベントの平均。Coupled発音経路ではこの観測がなく、`null`となる |
| `kuramoto_order_mean/max`、`sync_emergence_sec` | 全体の内部同期。Population別summaryでは未算出 |
| `one_over_f_slope` | 時間binごとのonset数系列の傾きの推定 |
| `ioi_one_over_f_slope` | `(population_id, voice_id, generation)`別IOI系列から得た傾きの平均 |
| `population_step` | 生存数、平均周波数、habitat地形上の平均評価、周波数占有のentropy |
| `listener_state` | presentation音声を入力にしたstability、resolvability、tension、attention、meter |
| `dcc_pressure` | ListenerTwinから生成側へ加える探索圧力とtemperature bonus |

`beat_stability`は`clamp(1 - ioi_cv, 0, 1)`である。同時発音による0秒のIOIが増えると、
聞き取れる拍が存在しても全体の値は低下する。値を比較するときは同じ集計単位を使う。
Kuramoto orderは、共有リズムに結合するVoiceごとのarticulation oscillatorの位相から、
各Voiceのphase offsetを除いて計算される。実際の発音を決めるCouplingClockのonset位相とは
別の状態であるため、Kuramoto orderの高さだけで実発音の同期を判定しない。
`sync_emergence_sec`は内部orderが0.70以上で2秒続いた区間の開始時刻であり、
聴き手が拍を感じ始めた時刻は試聴で別に記録する。

未算出の`null`は保持する。0への置き換えは、未観測と低い値を混同させる。
`metric()`・`entrained()`・`flow()`のCoupled発音では、自律attack側のPLV観測を更新しない。
この経路の`mean_plv: null`は同期不足を意味しない。未観測を0として出力した過去の記録も、
同期不足の根拠として使わない。
短いIOI系列では傾きが得られない場合がある。傾きの数値だけで1/f構造の成立や音楽的な自然さを合格にしない。

### habitat、presentation、試聴

onsetと`population_step`はpresentationだけに絞った記録ではない。
`mean_c_field_score/level`は生存VoiceをhabitatのLandscape上で評価した平均であり、
音量による重みづけも行わない。`alive_count=0`に伴う平均0を、低い協和性の観測として数えない。

ListenerTwinの`tension_level`は`(1 - stability_level) * resolvability_level`に基づくモデル値である。
Scenarioの`temperature`、habitat上の評価、ListenerTwinの値、試聴した緊張感を別の項目にする。
音声入力の欠落や分析遅延も記録し、観測のない区間を滑らかな有効値で埋めない。

WAVのpeakとRMSはPCM16値を32768で正規化した値である。`clipping_fraction`は量子化後に
両端値となった標本の割合、`silence_fraction`は絶対値1以下の標本の割合を表す。
これらは保存音声の振幅診断として使う。無音区間には意図した間や減衰尾部も含まれ、
clipping率にはリミッター前の歪みは直接現れない。実際に聞こえた音切れや歪みも試聴票へ記録する。

### 時刻、窓、末尾

現行onsetの`time_sec`はhop単位で記録される。sample rate 48000、hop 512なら
時間解像度は約10.67 msとなる。個々の発音のsample単位の時刻はJSONLから復元できないため、
それより細かなmicrotimingの精度判定には使わない。

既存`rhythm_summary`の集計窓は最初のonsetから最後のonsetまでであり、
`onset_density_hz`もその長さを分母にする。別途固定した窓で集計する場合は、その開始・終了と分母を明示する。
`listener_confidence_summary`の`beat_confidence_late_mean`は記録全体の最後25%を使うため、
releaseと減衰尾部を含みうる。立ち上がり、声部が揃った区間、解放操作、尾部を分けて確認する。

## Sample 12の操作窓

次の時刻は現行scriptの`wait()`から求めた名目時刻である。実行時のhop境界と分析遅延も併記する。
I〜Vはscriptのコメント上の段階であり、独立した`scene_marker`としては記録されない。
全体を一つの`section("emergence and resolution", ...)`が囲む。

| 開始秒 | 終了秒 | 操作と意図 |
|---:|---:|---|
| 0.0 | 2.3 | Genesis。habitat-onlyのrootを配置。presentationはまだ無音 |
| 2.3 | 6.0 | Pulse。pulseを配置 |
| 6.0 | 15.4 | Colony。8 Voiceのcolonyを追加 |
| 15.4 | 18.7 | Tension操作。colony temperatureを0.85へ上げ、rootとpulseを1.5倍の周波数へ移す |
| 18.7 | 24.0 | flowを追加し、Tension段階を継続 |
| 24.0 | 27.3 | Resolution操作。temperatureを0へ戻し、flowを減音、rootとpulseを元の周波数へ戻す |
| 27.3 | 28.6 | flowをrelease |
| 28.6 | 33.9 | colonyをさらに減音 |
| 33.9 | 35.9 | colonyをrelease |
| 35.9 | 37.2 | pulseをrelease |
| 37.2 | 41.2 | rootをreleaseし、section終端まで待機 |

Tension/Resolutionは操作意図のラベルである。対応する窓でListenerTwinのtensionがどう変化したか、
試聴でどの時刻に緊張と解放を感じたかをそれぞれ記録する。
操作直後の値を読むときは`analysis_lag_frames * hop / sample_rate`による遅延を確認する。
無音とrelease尾部の増加だけを、安定した生きたcolonyへの解決として扱わない。

## 試聴と性能の判定

`index.html`から同じ条件のWAVと記録をたどり、まず音を聴いて、変化を感じた時刻を残す。
07〜09は拍の可聴性、同期の立ち上がり、非拍節的なまとまりを評価する。
12は拍、緊張から解放への変化、曲全体の区切りを別々に評価する。
その後に操作時刻と指標を照合し、一致と不一致の両方を残す。
未試聴の条件は未判定のまま保持する。

hop時間統計のwarmup除外は先頭2秒を既定とする。対象は`time_sec >= warmup_sec`の標本であり、
p95/p99は順位`(n - 1) * p`の線形補間で計算する。除外後の標本がない場合は`null`となる。
この除外はListenerTwinとrhythmの集計には適用しない。
頭2秒を除くだけでは音楽的な定常状態を保証しない。
たとえば12でcolonyが入るのは6秒、flowが入るのは18.7秒である。
比較するPopulationと操作窓を合わせて指定する。

オフライン生成では実機の出力callbackが動かない。処理時間や待ち時間を測れても、
実際の機器でのunderrun、出力latency、安定した同時Voice数は別の再生測定で確認する。
今回の12条件比較では実機RT合格を未判定とし、割当数も未測定として扱う。
音楽的な合格は作者の試聴判定を残すまで未判定とする。
