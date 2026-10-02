# F3e 第二段階 v2: 密度準備と環境採点を分離した配送結果

日付: 2026-09-27。状態: `cfg(test)` の offline 明示 barrier 取得。取得前登録は v1 と同じ `target-body-fitness/f3e-phase2-registration-20260926-235100/registration.md`（SHA-256 `b8266f646904aaae6a416ef94eac55dda83645487317b5ab867f5e352b74d913`）。v1 source／結果／ログは `f3e-phase2-sealed-20260926-235159` に保存した。独立レビューで見つかった責務混合と Ready 遷移の不足を修正した版であり、v1 capsule を上書きしない。

修正版の `Request` は live source／body／control／RNG と候補 bit、`Log2Space` 幾何だけを保持する。barrier で完了する `Ready` は各候補の実 Tone 72 hop 代表密度と `Identity` を保持し、環境 C と F2 score を含まない。判断 sample 25,600 で元の `ReceivedBatch` を `accept` し、共有 habituation 版 48 を適用した当該 source 除去環境から全候補の F2 score 表を作り、実 Voice gate へ渡した。同じ Ready 密度に対して判断時の C を全 bin で +0.25 した反事実では、全候補の score が +0.25（絶対誤差 2e-5 未満）になった。これにより身体密度の先行準備と、受理環境での採点の分離を確認した。候補 `Identity` は準備時と表生成時で一致した。

登録した NoGate→Pending×2→post-commit C による B 置換→A 完了破棄→C 完了→正の実 gate 系列は 4 件の対象試験の一つとして通過。身体 score 利用 146 件、旧 fallback 二件と job 無し対照の target／salience／adaptation／RNG 一致、正の判断・commit と F3d 直接表対照の一致を維持した。4 source は保持最大 8、第五 source 拒否、同一 source 待機要求の置換を確認した。追加の反例では A が Ready のとき新しい B を受けると A を破棄して B を running に置き、さらに待機 C があっても B 完了後に C だけが running へ進む。未消費 Ready の後ろに pending が取り残されない。期限切れ `Stale` と `Invalidated(Processor(EpochMismatch))` は独立 worker から元理由を保持したまま、実 Voice gate の旧 fallback と一致した。

`cargo test --lib f3e_delivery -- --nocapture` は 4 件成功。最終 `cargo test -- --nocapture` は lib 1125 件成功、全 suite 1209 件成功・40 件 ignore・失敗 0、`test_status.txt` は `cargo test exit=0 @ 2026-09-27T00:02:17+09:00`。`cargo fmt --all --check`、標準 `cargo clippy -- -D warnings` は成功。`cargo clippy --all-targets` は成功し、既存 lib test 警告 17 件のみ。

この試験の成功条件は手動 barrier の固定完了順。全 proposal gate 後に新要求を出し続けると、現在の配送規則では計算中 job が完了するたび最新待機要求に置換され、完成 q が消費されず有効評価率 0 になり得る。単候補密度の費用からこの可能性は現実的だが、この試験は壁時計性能や実 thread の有効評価率を測っていない。通常 runtime の非同期配線、任意 phonation 更新、F4 の効果は未実装・未検証。
