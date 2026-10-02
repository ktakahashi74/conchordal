# F3e 第三段階: 仮想完了遅延と連続 gate の取得結果

日付: 2026-09-27。状態: `cfg(test)` の sample 時計による offline 明示配送診断。取得前登録は `target-body-fitness/f3e-phase3-registration-20260927-000500/registration.md`（SHA-256 `ed29154b38d50b2814eb45d62ed9c676ed12176b2420de6f77e8829d716c156d`）。第二段階 v2 capsule は変更していない。今回の 128 hop は壁時計性能、実 thread の完了遅延、通常 runtime の有効評価率の証拠ではない。

主行列は source 1／4、gate 周期 P=1／4 hop、仮想完了遅延 L=1／2／4／8 hop の 16 条件。hop 0 に初回要求を置き、最初の gate は hop P、同時刻では完了を gate より先に処理した。下表の数値は source 1 件あたりの Ready 機会で、source 4 件の条件ではそれぞれ 4 倍だった。

| gate 周期 P | L=1 | L=2 | L=4 | L=8 |
| --- | ---: | ---: | ---: | ---: |
| 1 hop | 127 | 0 | 0 | 0 |
| 4 hop | 31 | 31 | 31 | 0 |

Ready 機会 0 の条件では、同じ配送則で完了 q を消費する機会が生じない。P=1, L=2／4／8 と P=4, L=8 が構造的飢餓反例。一方、L=P でも完了先行の主行列では Ready に届く。初回要求直後の hop 0 に gate を出す別位相では P=L=1／4 とも Ready 機会 0（source 1／4 の全 4 条件）。初期位相も結果を左右した。

この 20 条件は queue 状態試験。確定破棄 job と仮想 Ready job の数値 q／F2 採点を省略し、`numeric_skipped` を別記した。したがって `actual_body_consume=0` は試験設計上の値であり、「実測有効評価率 0」ではない。全 gate で実 Voice の旧 proposal／commit を通し、job 無し twin と target、salience、adaptation、終了 RNG、commit 後基音が一致した。非 gate hop の新要求と RNG 消費は 0。保持最大は source ごと 2、4 source 全体で 8。

別の回復条件は source 1、P=4、L=8。hop 0–60 の 16 gate は Pending fallback、hop 64–79 を休止し、hop 72 に最後の最新要求から実 Tone 代表密度 q を準備した。破棄確定の先行完了 8 件では q を計算しなかった。hop 80（sample 40,960）に実 source PCM から F3b worker が作った fresh `ReceivedBatch` を受理し、同じ 80 hop の混合 PCM から共有 habituation 版 80 を適用した環境で F2 score 表を作成した。旧版 79 の表は実 Voice gate が `HabituationChanged` で拒否し、job 無し twin と RNG を含む fallback が一致。版 80 の表は実 Voice で 1 件受理され、候補 score 利用 143 件、直接表を渡した対照と target、salience、adaptation、終了 RNG、commit 後基音が一致した。密度準備時の source／body generation／Recipe／route／epoch／target／current pitch／RNG／space は判断時値で上書きしていない。

回復条件は source PCM と実 Voice の身体を固定するため、`landscape_weight=0`、`move_cost_coeff=10`、基音 440 Hz bit 不変という登録制約を使った。この条件の正例は、任意の制御・発声変化で非同期準備を使える証拠ではない。通常 runtime への非同期配線、実 thread の性能、有効評価率、公開既定の採用も未検証。

対象配送試験 6 件成功。最終 `cargo test -- --nocapture` は全 suite 1211 件成功、40 件 ignore、失敗 0。`test_status.txt` は `cargo test exit=0 @ 2026-09-27T00:20:30+09:00`。`cargo fmt --all --check`、標準 `cargo clippy -- -D warnings` は成功。`cargo clippy --all-targets` は成功し、既存の test target 警告 17 件のみ。対応する source、登録、検査ログと JSON 行は phase3 capsule に保存する。

封印先は `.worktrees/target-body-fitness/f3e-phase3-sealed-20260927-002141/`。
SHA256SUMSのSHAは `a1fe5fba7e99165cd2bc099e382477b2ab4c93190fc7760c945abe07662416ea`。
source確認による補足: 主行列ではgateを指定したhopだけVoiceへ`dt=0.01`を渡し、
非gateには`dt=0`を渡している。配送の仮想hop時計とVoiceへ渡す積分時間を分けた試験であり、
通常runtimeの全hop積分や実時間gate生成を再現しているわけではない。
