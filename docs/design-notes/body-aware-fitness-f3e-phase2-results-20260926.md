# F3e 第二段階: offline 明示 barrier 配送の結果

日付: 2026-09-26。状態: `cfg(test)` の固定 fixture における初回取得結果。独立レビューで密度準備と環境採点の責務混合、Ready 後の待機要求遷移不足が判明した。修正・再取得結果は [第二段階 v2](body-aware-fitness-f3e-phase2-v2-results-20260927.md)。初回 source／結果／ログは独立 capsule に保持する。取得前登録の本文コピーは `target-body-fitness/f3e-phase2-registration-20260926-235100/registration.md`、SHA-256 は `b8266f646904aaae6a416ef94eac55dda83645487317b5ab867f5e352b74d913`。第一段階 capsule は変更していない。

実 Tone と独立 source PCM を 48 hop レンダリングし、F3b worker の支持終端 24,576 sample の `ReceivedBatch` を取得した。共有 habituation 版 48 と source 除去環境を一度固定し、候補ごとの実代表身体を 72 hop 解析して F2 採点した。明示 barrier 配送では、最初の NoGate で要求追加 0・RNG 不変、続く sample 24,576 と 25,088 の二つの実 Voice gate で `Pending` と旧 scorer fallback を確認した。各 gate の target、salience、adaptation、終了 RNG は job 無し対照と一致した。各 commit 後の要求 B/C は更新済み target、current pitch、RNG から作られ、C が B を置換した。古い A は完了後に破棄し、C だけを完成結果として保持した。完成後の NoGate でも結果を消費せず、sample 25,600 の実 gate で `ReceivedBatch::accept` 後に身体 score 146 件を利用した。この判断と commit は準備表を直接渡す F3d 対照と一致した。

4 source 容量 fixture は source ごと running 1＋latest pending 1、完成済み未消費結果は running 枠を置き換える構造を検査した。保持最大 8 件、9 件目の同一 source 要求は pending を置換、第 5 source は拒否。古い実行中 job の完了後、最新待機 request だけが running に進んだ。この容量 fixture の環境は構造検査用であり、4 source の実 PCM・各 Voice の正常採点は第一段階に封印した 10 件が別の証拠である。

独立 worker で期限切れと失効を調べた。支持終端 24,576 に対し判断 sample 29,696 は age 5,120 で `Stale`。別 worker へ不正 epoch を提出した後、保留していた batch は `Invalidated(Processor(EpochMismatch))`。元の `Rejection` を `PreparedRejection::EnvironmentRejected` 内に保持し、双方とも身体 score 利用 0、実 Voice gate の旧 fallback と job 無し対照の target、salience、adaptation、終了 RNG が一致した。正常系列への失効混入はない。

`cargo test --lib f3e_delivery -- --nocapture` は 3 件成功。最終 `cargo test -- --nocapture` は lib 1124 件成功、全 suite 1208 件成功・40 件 ignore・失敗 0、`test_status.txt` は `cargo test exit=0 @ 2026-09-26T23:51:59+09:00`。`cargo fmt --all --check`、標準 `cargo clippy -- -D warnings` は成功。`cargo clippy --all-targets` も成功し、lib test の既存警告 17 件だけを記録した。

配送器と gate の利用は `cfg(test)` 限定。barrier による完了順・保持上限の試験であり、代表身体計算の並列性、実 thread の遅延、通常 runtime の有効評価率、任意 phonation 更新を測っていない。現在の queue は固定 source 1・4 と固定 recipe の offline 境界に留まる。通常 runtime の worker 配線や F4 の代謝・出生効果を採用した結果ではない。
