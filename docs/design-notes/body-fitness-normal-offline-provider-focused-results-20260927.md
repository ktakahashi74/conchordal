# 通常 offline metabolism provider の focused 検査

日付: 2026-09-27。対象は `src/runtime/body_fitness_f4a_tests.rs` の `normal_offline_metabolism_*` 3件。これは通常 Voice が保持する `OfflineMetabolism` の局所検査であり、10秒の paced 実時間取得ではない。

初回実行は3件中1件成功、2件失敗。cache 試験は `commit_decided_control` の前後で articulation recipe が変わるのに同一 recipe と仮定していた。WhenViable 試験は factory の許容範囲を省略し、低い point level でも門を通した。両 fixture を修正し、同一 recipe の hit、実基音変更の miss、環境 score 更新だけで density を再利用して score を再評価する境界を追加した。source id・generation・birth sample、epoch、鮮度、時刻加算 overflow、space/scan の拒否も検査した。初回失敗出力は実行ツール内の記録のみで、別ファイルには保存していない。

最終実行: `RUST_BACKTRACE=1 CARGO_TARGET_DIR=/home/shafi/lwrk/conchordal/target/body-fitness-metabolism-build-20260927 cargo test --lib normal_offline_metabolism -- --nocapture`。3件成功、0件失敗。全出力はmain repository側の `target/body-fitness-metabolism-build-20260927/provider-focused-final.log`。正常値は実 Tone を72 hop描画した直接参照と比較し、point poison と独立した score/level、現在基音と target の差、lifecycle energy、Gated onset と WhenViable point gate を確認した。

この検査は非同期通常 runtime、実時間費用、音声デバイスでの実行を証明しない。全体suite、Clippy、release検査の最終結果は[第十三版b統合記録](body-fitness-metabolism-validation-20260927.md)を参照する。
