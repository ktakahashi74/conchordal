# Body fitness preparation: 取消の局所検証結果（2026-09-27）

対象は v14 recovery 作業木の `BodyFitnessAction` と、各 source slot が保持する実際の preparation worker。ここでの結果は局所 unit test の範囲に限る。全体試験、通常 renderer の取得、性能計時は実施していない。

## 実装境界

- Request と Ready/Failure は `serial`、完全な `SourceIdentity`（ID、generation、birth sample）、body generation、analysis epoch を持つ。action は受信した identity と現在の pending identity の完全一致を確認し、不一致なら現在の pending を消さない。
- 実時間 action は body recipe の変更または source retirement を検出すると、pending serial に対して協調的取消を要求する。worker は開始時、各候補の境界、候補計算直後、全候補終了後の Ready 返却前に取消を確認する。候補一つの内部では停止しない。72-hop の一候補分が残る可能性がある。
- 取消要求と worker 完了が競合し、Ready が先に生成済みでも、action は取消要求済みの応答を `obsolete` として扱い、Voice に導入しない。取消要求後の応答は `last_cancelled_preparation` に1件保持する。新しい Ready が `last_preparation` を上書きしても、この記録は残る。source slot を再割当すると記録を消す。
- 同じ Voice ID と generation が再利用されても、batch の birth sample が変われば旧 source を retired と判断する。
- offline deterministic action は wall-clock に依存する取消を要求せず、受信を明示的に待つ。旧 source の結果は identity と retirement により破棄する。

## 取得結果

コマンド: `CARGO_TARGET_DIR=/home/shafi/lwrk/conchordal/target/body-fitness-recovery-build-20260927 RUST_BACKTRACE=1 cargo test --lib runtime::body_fitness_action::tests -- --nocapture`

最終取得は **5 passed、0 failed**。内訳は次の通り。

1. 既存の routing 変更試験: 旧提案の消費を拒否し、次の実 worker job で回復。
2. body 変更試験: 旧 serial を取消要求し、旧 RNG と target が不変のまま旧応答を破棄。新 body generation の job を消費し、直近取消記録を report 上に保持。
3. retirement 試験: Voice 消失時に pending を取消要求。旧 Ready は導入されない。
4. birth sample 再利用試験: 同一 ID/generation の新 source に旧応答を導入せず、新 source identity の job で回復。offline 取消要求は0件。
5. 想定外 identity 試験: 古い応答が新 pending identity を消さず、後続の一致する応答を受理。

最初の取得では、body 変更試験の `yield_now` 50,000 回による短い poll が worker 完了前に尽き、当該1件のみ失敗した（3 passed、1 failed）。これは論理上の取消失敗を示すものではない。poll を30秒の生存確認上限と1ms sleepに変更し、待機中の fixture support sample を進めて再取得した。30秒は試験ハング検出用で、処理速度の合否閾値ではない。

両取得の生ログはファイル保存していない。記録は当該作業対話の tool 出力のみ。再実行による生ログの補充は行っていない。
