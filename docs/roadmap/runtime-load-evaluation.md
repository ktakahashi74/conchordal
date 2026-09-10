# 実行負荷の比較手順

固定したVoice数と音色で、workerの処理時間、音声出力の不足、Rustのメモリ割当を測定する。
持続音に加え、短い音の反復発音、追加集団による一時的な負荷増大とrelease後の回復も扱う。
この手順は楽器`conchordal`を使い、音声ファイルを保存しない。
機器なしの確認と音声機器を使った確認は、別のcampaignとして保存する。
[反復発音・長時間・回復の結果と時計修正](dense-long-recovery-2026-09-05.md)に、2026-09-05の実機結果を記録した。

## 実行

```bash
# 音声機器を使用する。release + profile-allocをビルドして実行する。
python3 scripts/evaluate_rt.py

# 機器が使えない環境での明示的な計算負荷・割当確認。
python3 scripts/evaluate_rt.py --mode offline-check

# 対象を絞る例。reportとDCCの4組は維持する。
python3 scripts/evaluate_rt.py --voices 4 --bodies sine
```

既定は4・16・64 Voice × sine・harmonic・modal × report無効/有効 × DCC結合0/0.25の36条件。
seedは1、warmupは5秒、測定は10秒、release後の待機は2秒とする。
`--seed`、`--warmup-sec`、`--duration-sec`、`--config`、`--timeout`、`--output`で変更できる。
既存の出力先は上書きしない。

### 参加方策の負荷

`dense/recovery`のpulse時計と、entrained/flowの参加方策は異なる経路である。
参加方策の性能を測る場合は、対応するworkloadを明示する。

```bash
python3 scripts/evaluate_rt.py --workload entrained --voices 4 16 64 \
  --bodies sine harmonic modal --reports 1 --dcc 0
python3 scripts/evaluate_rt.py --workload flow --voices 4 16 64 \
  --bodies sine harmonic modal --reports 1 --dcc 0
```

実際の参加プリセットと`cycles(1)`を使う。entrainedの周期追従、flowの固有周期による
不規則な更新、両者の音響文脈・重なり予測を通す。flowの保持には既存の3倍係数が掛かる。
attackによる消費を無効にし、長いenduranceで測定中のVoice数を保つ。
report有効時は窓内で各Voiceが2回以上発音したことを要求し、4回/秒というpulse用の
密度条件は使わない。report無効時は発音の継続を独立には検証しない。
共有scaffoldによって負荷を作り替えないよう、`--timing synchronized`との併用は拒否する。
この負荷の合格を、音楽的な関係の持続や任意の音長・同時Tone数への保証に読み替えない。

2026-09-06、周期追従をentrainedだけに適用した版で、両workloadの各36条件をMOTU M2へ出力した。
seed 21、48 kHz・hop 512、warmup 5秒・測定10秒。72条件すべてで測定窓の出力不足ゼロ、
callbackエラーゼロ、hop p99は10.667 msの処理予算以内だった。

| workload | 4 Voice | 16 Voice | 64 Voice |
|---|---:|---:|---:|
| entrained：各12条件の最大hop p99 | 0.556 ms | 0.855 ms | 1.815 ms |
| flow：各12条件の最大hop p99 | 0.731 ms | 1.625 ms | 5.065 ms |

report有効の36条件では各Voiceの反復発音も確認した。約0.5秒間隔でPIDに対応する
PipeWireの経路を観測し、72条件すべてでactiveな経路の出力先がM2だった。
連続した経路追跡や機器側xrunの検査ではない。観測処理の負荷を含む。
標準08・09もseed 21で実機基準を満たした。
[比較記録](rhythm-perception-evidence-2026-09-06.md#entrainedflowの実機負荷)に条件と成果物を示す。


### 反復発音・長時間・回復

```bash
# 64 Voiceの反復発音を3音色で20秒ずつ測る。
python3 scripts/evaluate_rt.py --workload dense --voices 64 \
  --reports 1 --dcc 0.25 --duration-sec 20

# harmonicの反復発音を10分測る。起動と終了の余裕を含むtimeoutを指定する。
python3 scripts/evaluate_rt.py --workload dense --voices 64 --bodies harmonic \
  --reports 1 --dcc 0.25 --duration-sec 600 --timeout 750

# 16 Voiceから一時的に合計512 Voiceへ増やし、16 Voiceへ戻す。
python3 scripts/evaluate_rt.py --workload recovery --voices 16 --bodies harmonic \
  --reports 1 --dcc 0.25 --overload-voices 512 --stress-sec 5 \
  --duration-sec 10 --timeout 180
```

`--workload`は`sustain`（既定）・`dense`・`recovery`・`entrained`・`flow`。
`--timing`は`adaptive`（既定）・`synchronized`。後者は同期した同時発音の負荷を測る、
明示的な同期性能アッセイであり、`dense/recovery --reports 1`でだけ使用できる。
16 Hzの共有θ scaffold、蓄積率32、既存の2ゲート間隔制限で8回/秒を駆動する。
蓄積率はgateの重みが掛かっても発音を維持する値で、実発音率ではない。専用の音声Voiceは加えない。
各Voiceの回数を窓長×8（境界の1発分まで許容）、IOIを125 ms（reportの1 hop分と時刻丸め誤差を許容）と照合し、
reportが`scaffold_mode=shared`を示すことも要求する。音楽的な創発の合否は判定しない。

```bash
python3 scripts/evaluate_rt.py --workload dense --timing synchronized --voices 64 \
  --bodies sine harmonic modal --reports 1 --dcc 0.25 --duration-sec 30
python3 scripts/evaluate_rt.py --workload dense --timing synchronized --voices 64 \
  --bodies harmonic --reports 1 --dcc 0.25 --duration-sec 600 --timeout 750
python3 scripts/evaluate_rt.py --workload recovery --timing synchronized --voices 16 \
  --bodies harmonic --reports 1 --dcc 0.25 --overload-voices 1024 --duration-sec 10
```

`--reports 0 1`と`--dcc 0 0.25`は既定の比較集合であり、上記のように対象を絞れる。
`dense`と`recovery`は蓄積率8（`adaptive`）または32（`synchronized`）、1周期の保持、attack 3 ms・decay 25 ms・
sustain level 0.30・release 50 msを指定する。attackによるエネルギー消費とvitalityによる
rate低下を無効にする。`adaptive`では`pulse`は適応的なθゲート時計に従うため、固定8 Hzの時計ではない。
指定Hzだけで高密度に発音したとみなさない。report有効時は、各測定窓で全Voiceの発音と
Voiceあたり平均4回/秒以上をJSONLから確認し、同じreport hopで同じVoiceが複数回発音する
異常な集中も拒否する。report無効時の密度は未検証となる。
専用の拍打ちVoiceを加える試験ではない。


初回試験ではθ時計の位相更新漏れによる大量の偽ゲートが平均発音数を押し上げていた。
修正後はこの密度条件に届かない場合があり、その試行は正常な高密度負荷の合格とは扱わない。
詳細は[θ時計と終了音追跡の修正](gate-clock-stability-2026-09-05.md)を参照する。
[同期負荷の実測](synchronized-load-2026-09-05.md)では64 Voice・3音色が8回/秒を満たした。
回復試験は512 Voiceでは過負荷未成立、1024 Voiceでは負荷超過と16 Voiceへの回復を確認した。
[同じharmonic条件の600秒試験](synchronized-long-harmonic-2026-09-05.md)も8回/秒を満たし、
測定窓とrelease中の出力不足ゼロ、hop p99 1.592 msとなった。終了後の最終集計中に増えた
callback累積値は、生成中のhop記録と分けて保存している。

`recovery`ではwarmup後、`--duration-sec`のbaseline、`--stress-sec`の負荷区間、
追加集団のrelease、2秒の待機、baselineと同じ長さの回復窓の順に実行する。
`--overload-voices`は追加数ではなく負荷区間の合計Voice数。振幅はこの最大Voice数で正規化し、
baselineと回復窓でも同じ値を維持する。各窓の全hopで指定した生存Voice数を検査する。

`phases`に3窓の計測値を保存する。baselineが性能条件を満たし、負荷区間で出力不足または
hop p99の予算超過を観測し、回復窓が再び性能条件を満たした場合だけ`recovery_pass=true`とする。
負荷超過が起きなければ回復は未判定。負荷を戻しても不足が続けば不合格。
意図的な負荷超過を含む実行全体の`device_rt_pass`は`null`とし、全区間が性能条件を満たしたとは扱わない。
実行成功と性能判定は別なので、runnerの終了値だけで回復成功を判断しない。

長時間試験でもprofileの100000 hop上限は共通であり、warmup・回復・終了待機を含めて収める。
10分の成功を、時間上限のない演奏やメモリリークがない証拠へ一般化しない。

既定の脚本は長いenduranceを持つsustain音を配置し、`seek_consonance()`を使う。
再生成は設定しない。測定窓の全hopで`Voice::is_alive()`の数が指定数と一致することを検査する。
総振幅の増大を抑えるため、Voiceごとのampを`0.06 / sqrt(Voice数)`とする。
これはこの負荷試験の条件であり、対応Voice数をあらゆる音色・作品へ一般化する根拠ではない。
DCC 0.25も比較用の値であって推奨設定ではない。

既存バイナリを使う場合は`--skip-build --binary PATH`を指定する。
この場合、sourceとバイナリの対応は未検証としてmanifestに記録する。
割当計測には`profile-alloc`を有効にしたビルドが必要で、無効なビルドの割当値は`null`となる。

## 記録する範囲

楽器の`--nogui --profile PATH`は、通常の`--report`から独立した負荷記録である。
`--compile-only`とは併用しない。profileとreportは別ファイルへ保存する。

| 項目 | 契約 |
|---|---|
| worker時間 | `process_hop`の処理時間。report有効時の書き込みを含む。workerの待機sleep、profile行の保存、終了時の集計・書き込みは含まない |
| 区間別時間 | schema 2ではanalysis待ち、Landscape更新、listener待ち、Voice更新、report、合成・転送、合成後の処理を記録する。合成後にはmeter・解析への送信・UI・hop reportを含む。計測点間の小さな処理は全体時間にだけ含まれる |
| 合成内部 | `synthesis_us`は`render_route_us`の内数。両者を合計しない。`rendered_tone_count`はそのhopでScheduleRendererが処理するTone数であり、生存Voice数とは別に記録する |
| p95/p99 | 指定した測定窓に完全に含まれるhopだけを使い、`(n−1)×p`で線形補間する |
| worker割当 | 成功したRust `alloc`・`alloc_zeroed`・`realloc`の要求回数と要求bytes。保持メモリ量ではない |
| 割当の対象外 | analysis thread、音声callback、native malloc、開始時の準備、終了時の集計 |
| 出力不足 | callbackがringから読めずゼロ補完したmono frame数。hardware xrunではない。生成時刻の測定窓境界で取得した累積差であり、ring以降の遅延を補正した物理再生区間の集計ではない |
| 機器情報 | CPAL backend、device名、実sample rate、channel数、ring容量、callbackとエラーの数 |

`profile-alloc`が無効な通常ビルドには割当計測用のglobal allocatorを組み込まない。
有効なビルドではthread-localのスカラーカウンターで計測し、計測用処理からの割当を避ける。
ただし、計測による実行時間の負担はあるため、結果のビルド条件に必ず残す。

profileは開始前に100000 hop分を確保し、終了時にまとめてJSONへ書く。
48 kHz・hop 512では約17.8分が上限となる。上限を超えても途中で容量を増やさず、
`truncated`と欠落数を記録し、終了値を非0にする。途中の欠落を含むprofileは受け入れない。
区間別時間の追加時計読み取りは`--profile`指定時だけ行い、hop内の計測に新しいheap割当を加えない。
評価ツールは必須フィールド、区間時間の合計、合成時間が合成・転送時間を超えないことを検査する。
以前のschema 1はその実行に保存された評価ツールで読む。

reportを有効にすると、DCCがゼロでもListenerTwinの解析が起動する。
したがってreport無効/有効の差には解析経路の差も含まれ、JSONL書き込みだけの負荷差ではない。

## 機器を使った判定

実行の成功、性能条件の合格、物理的な出力先の確認、作者の試聴合格は区別する。

- `--play=true`で音声初期化に失敗したら非0終了し、計算だけの実行へ切り替えない。
  runnerは失敗理由を`device_blocker.json`へ記録し、残りの同じ機器での試行を省略する。
- callback未観測、既知のnull/dummy/loopback出力、割当計測なし、測定区間を消費した証拠が不足する場合は性能判定を保留する。
- 全生成frame数からring容量を引いた値を、callbackへ渡ったframe数の下限とする。
  これが測定窓の末尾に達しない場合は、大きいringに貯めただけで合格にしない。
  callback frame数とwall timeもこの下限と整合する必要がある。
- 証拠が揃った条件で、測定窓の出力不足ゼロ、callbackエラーゼロ、hop p99がhop時間以下を性能条件とする。
  個別hopの超過数と最大時間も併記する。
- CPALのdevice名だけでは物理的な接続先まで確認できない。特にALSAの`default`は経路名であり、
  `physical_output_verification`の未確認表示を残す。結果を公表する場合は実際の出力先も確認する。

`offline-check`では音声機器を開かず、出力不足は`null`、`device_rt_pass`も`null`とする。
そのp99と割当数はボトルネックを探す材料になるが、実機での合格や対応Voice数の宣言には使わない。

## 成果物と再現条件

出力先は`target/rt-evaluation/`内の新しいディレクトリ。Git管理外のローカル生成物である。
manifestにsource・設定・バイナリのSHA-256、ビルドコマンド、CPU、affinity、seed、測定窓を保存する。
各条件に脚本・設定・profile・ログを保存し、report有効時はJSONLも保存する。
`summary.csv/json`に実行状態、性能判定、p95/p99、割当、出力不足、機器情報をまとめる。

音楽的な判断は[Manifesto整合・beta計画](manifesto-alignment-and-beta.md)と試聴票で扱う。
この負荷試験の成功だけで、拍・解放・終止が聴き取れるとは判断しない。
