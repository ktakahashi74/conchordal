# F3e 第四段階: 連続 gate の延期政策と実数値消費

日付: 2026-09-27。状態: 隔離 worktree の試験専用結果。元登録は `body-aware-fitness-f3e-phase4-registration-v1-20260927.md`（SHA-256 `8e49b3464984e788f149002a1d7627d0d1721bf89bd817854386ff435dca76d0`）、改訂登録は `body-aware-fitness-f3e-phase4-registration-20260927.md`（SHA-256 `8b256a3a38184c7011cc3408594dd1e3917bae1793eb12851886203d6906cb91`）。最初の対象検査後、レビューで延期時間の二重計上を発見したため、元登録を保全し、残量 0 の規則を改訂登録に追記してから修正版を再取得した。既存の phase2-v2／phase3 capsule は変更していない。

初回検査は身体表消費 1、score 利用 145、直接表一致、失効負例一致を示したが、延期解除後の蓄積時計を検査していない。元実装は通常の 0.01 秒だけを減算し、約 8 hop 分を残す。後続 proposal の adaptation へ過去の延期時間を重複計上し得るため、初回結果を政策の成功としない。改訂版は全経過時間を一度だけ使って残量 0 とし、次 hop の通常周期まで検査した。

従来の fallback を 1 gate 実行すると、固定 Sine、固定 target・current pitch でも RNG が進み、後続候補 132 個のうち 3 個は旧表に exact pitch bit がない。従来配送の P=1／L=8 では Ready 機会 0 だった。完成表だけの保持や有界 queue は、この全候補欠落を解決しない。

任意の延期政策では、1 source の実 Voice に全 9 hop で `dt=512/48000` の decide／commit を実行した。hop 0–7 は proposal のみ延期し、target、salience、adaptation、RNG と基音 440 Hz の bit が不変だった。PitchController の時計、Voice の articulation／body／lifecycle は進めた。8 hop の蓄積時間を次の実 proposal に一度だけ渡し、残量を 0 にした。さらに次 hop の通常周期への復帰を検査した。実 thread で Tone 72 hop 代表密度を計算し、明示 barrier で完了を待ってから hop 8 の新 gate に渡した。そこで F3b fresh batch（支持終端 sample 4,608）を受理し、同じ 9 hop の混合 PCM で進めた共有 habituation 版 9 の環境で採点した。Ready 機会 1、実身体表消費 1、score 利用 145。直接表対照と target、salience、adaptation、終了 RNG、commit 後基音が一致した。旧 habituation 版と異なる body generation は各々拒否され、旧 fallback 対照と一致した。

密度 thread は gate で `join` する明示 barrier。仮想 8 hop の間に実 thread が完了した証拠ではなく、壁時計内の有効評価率ではない。固定 Sine・固定 pitch・固定 routing・固定 epoch の一判断の機能検査。theta 入力は毎 hop 固定で、位相交差の効果は未検証。body generation 負例は表の期待値だけを変更したもの。実 Capture の世代変化、変動する身体、glide中の current pitch、route変更、通常 runtime の非同期配線、連続実時間完了は未取得。実時間測定は cargo/render の専有枠で別に取得する。

最終 `cargo test -- --nocapture` は lib 1129 件成功・41 件 ignore、全 suite 失敗 0。`test_status.txt` は `cargo test exit=0 @ 2026-09-27T11:54:54+09:00`、全 stdout/stderr は `test_report.txt`。`cargo fmt --all --check` と標準 `cargo clippy -- -D warnings` は成功。live-paced 測定の取得前登録は `body-aware-fitness-f3e-phase4-live-registration-20260927.md`（SHA-256 `6192332d6b52447eb076332e77d639545eb74816687aa9f58600e4e3ce32297c`）。release test binary は `/home/shafi/lwrk/conchordal/.worktrees/target-body-fitness/release/deps/conchordal-ca9750681d3d7936`、SHA-256 `353bbcc40b485e6776a4ecd9ef4a58cbd5040a4c6ed763dbe8aa42776ceb962a`。live測定は未取得。

統合用 source 系譜: phase3封印 `source.tar.gz` 内 `src/life/pitch_controller.rs` SHA-256 `65ac7266a4513ce59fbaa34c75b9283e87ff57b9e936c82b3b4f723b9ff01bf5`、`src/runtime/body_fitness_f3e_delivery_tests.rs` SHA-256 `3a9dbf82b0e4fc69e531d1201d4aaacff3f473efc96f1db57db51875d433e9b8`、`src/core/source_removed_worker.rs` SHA-256 `81c1db5048886045bd8398ef078126bf37c3763c8d3d95c53a90fe3cdcb056ce`。第四段階は順に `90e31cd19b3dd3da0c10d8c7f7cf6cc0e9202557d6408f3ae721e20c04e8fddb`、`50b36ef696677b360d26de947ed5c65e9340c88eae469d8fa9c7d78a6a346527`、`561630a52576ec57226c51e8242619a79a2492f12b22b788983384bd3b6d7427`。
