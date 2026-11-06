# Browser engine, audio and GUI

Web Worker 内の正式な `conchordal::StepEngine` が一 hop ずつ演奏を生成する。
Host は raw presentation PCM に設定と台本の output gain および native と同じ
limiter を適用し、専用 MessagePort の transferable ArrayBuffer で AudioWorklet
へ渡す。GUI は独立した wasm の egui/WebGL で描画する。音声の保存経路はない。
`www/` は計測ページであり、repository 直下の Astro `web/` とは別である。

中核は `default-features = false, features = ["web"]`。GUI crate は `gui/` に置き、
表示データ型だけを中核と共有する。GUI wasm だけが `browser` feature を使う。
endless と追加の解析 thread を必要とする台本・設定は入口で拒否する。

## Build

Rust、`wasm32-unknown-unknown` target、wasm-bindgen-cli **0.2.130** が必要。
`build.sh` は両 wasm と JS glue を生成する。link stack は検証済みの 8 MiB、
`+simd128` と wasm target 専用の rustfft `wasm_simd` を使う。native の FFT
feature 選択は変えない。固定乱数 backend の build flag は使わない。

blanche では割り当て済み target と cargo wrapper を指定し、MemAvailable が
12 GiB 以上あることを確認して、子ジョブだけに 8 GiB の上限を掛ける。

```sh
export CARGO=/home/shafi/lwrk/conchordal/.orchestration/bin/cargo-slot
export CARGO_TARGET_DIR=/home/shafi/.cache/conchordal-test-tmp/intw1-target
export TMPDIR=/home/shafi/.cache/conchordal-test-tmp/intw1-tmp
export CARGO_BUILD_JOBS=4
mkdir -p "$CARGO_TARGET_DIR" "$TMPDIR"
prlimit --as=8589934592 -- bash crates/conchordal-web/build.sh
```

`www/pkg/` と `www/gui-pkg/` は生成物で git 管理しない。
HTTPS または localhost が必要。COOP/COEP header は不要。

## WP-6：GUI を含めた手元の測定

統合済みの `www/` 全体（両 pkg directory と台本を含む）を測定機へ置く。
同じ機械で次の配信器を起動する。引数の port は空いているものを選ぶ。
既存の port 8000 の測定配信には触れない。

```sh
python3 /home/shafi/lwrk/conchordal/.orchestration/units/WP-6/serve.py \
  /home/shafi/lwrk/conchordal/.worktrees/intw1-20261106/crates/conchordal-web/www 8011
```

MacBook Air では、この配信器をコピーした場所と page directory に置き換える。
数値は配信器と同じ directory の `metrics.jsonl` に追記される。

1. 測定機の Firefox または Chrome で `http://127.0.0.1:8011/?seed=21&buffer=250`
   を開く。GUI canvas は viewport 全体を使い、設定と数値はその下にある。
2. 出力音量を確認し、GUI の Start または下の「開始」を押す。通常のページは
   端末の音声出力を使う。自動検証の専用無音 sink はページ自身では作らない。
3. GUI を表示し foreground のまま、DEMO-B1 全長約 230 秒の `ended` を待つ。
   `underruns = 0`、`missingFrames = 0` と生成・消費 frame 数の一致を確認する。
   GUI の End は graceful end、Exit は即時停止。下のボタンでも操作できる。
4. 最小先行残量、hop 平均・p99・最大、snapshot 平均・p99・最大、GUI decode・UI・
   CPU frame の各時間、browser・端末・背景負荷を記録する。250 ms で途切れた時だけ
   `buffer=500` で再測定する。別の性能比較をする場合は、その比較の取得条件に従う。

URL の seed と入力欄が空なら、開始のたびに host の `crypto.getRandomValues()`
から seed を作る。明示した seed は再使用する。台本内の seed 指定があれば優先する。
初期 PCM の到着を待ってから再生し、自然終了では残りを排出する。

## 計測と送信の境界

Worker は native の `UI_MIN_INTERVAL` と同じ **33 ms の wall time** を最短間隔とし、
`ui_frame().web_snapshot().encode()` を呼ぶ。最後の hop でも最終 snapshot を送る。
bytes は main thread へ所有権を移し、そのまま `push_snapshot()` に渡す。
JS の JSON 変換は挟まない。hop 時間には計算、limiter、PCM コピー、snapshot の
構築・encode・転送呼出しを含む。snapshot 時間は構築・encode・wasm からのコピーを
独立集計する。GUI decode は Rust 内の復号、UI は egui の UI 関数、CPU frame は
eframe の CPU 描画時間であり、GPU 完了時間ではない。

5 秒ごとと状態遷移時に `/metrics` へ性能項目だけを明示列挙して POST する。
音声、snapshot、波形、peak、非零 sample 数、台本、エラー本文は送らない。
送る項目は状態、sample rate、buffer 設定と残量、生成・消費・欠損 frame 数、underrun、
hop/snapshot/GUI の時間分布、browser 情報、論理 CPU 数、device memory、tab の非表示時間。
ページ内の runId は既存計測方式を維持する。公開版の同意と収集先は WP-5 の決定事項。
GUI の測定は演奏開始ごとにリセットし、その演奏全体を集計する。

GUI が依存する eframe には clipboard、storage、file-drop 等の標準機能があるが、
アプリはそこへ音声や音から作った値を渡さない。表示用 waveform は memory 内だけで
GUI に渡し、音声は専用 AudioWorklet port だけに渡す。
