# 実行負荷の比較手順

固定したVoice数と音色で、workerの処理時間、音声出力の不足、Rustのメモリ割当を測定する。
この手順は楽器`conchordal`を使い、音声ファイルを保存しない。
機器なしの確認と音声機器を使った確認は、別のcampaignとして保存する。

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

脚本は長いenduranceを持つsustain音を配置し、`seek_consonance()`を使う。
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
