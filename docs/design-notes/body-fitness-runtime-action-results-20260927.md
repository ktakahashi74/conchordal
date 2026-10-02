# 通常runtimeでの身体評価消費（2026-09-27）

状態: 隔離した第十一版の機能検証を通過。作業場所は `.worktrees/body-fitness-action`。取得前の条件は [登録](body-fitness-runtime-action-registration-20260927.md) に固定した。mainへの採用、commit、pushは行っていない。

## 実装

`body_fitness_action = true` により、通常runtimeで自己音を除いた環境に対する身体評価をpitch提案へ渡す。観測も同時に有効になる。既定値はfalse。現在の対象は48 kHz、512 sample/hop、最大4 source、local/non-ratio候補である。

配線時に4本の持続workerを作り、各workerの入出力を各1件に制限した。代表密度cacheは256件かつentry領域1 MiB以下。環境scoreはcacheしない。Recipeは実Voiceの身体・ADSR・代表modulatorから再構成し、振幅1・hold 48,000 sample・観測72 hopを固定する。実際に鳴った音の将来予測ではなく、候補比較用の代表発音である。

観測結果のsource集合、出生識別、時刻、最大4,800 sampleの経過時間を照合する。候補密度の完成後、採点に当hopのhabituationを反映する。現pitchだけの変化は許容し、身体、route、control、target、RNG、epoch、space、habituationは消費時にも照合する。判断待ちと拒否ではpitch提案だけを保留する。発音・身体のcommit・寿命は継続し、一度消費した後の同hop内の別substepでも旧評価へ戻らない。

候補列挙は環境値を読む局所peak探索ではなく、一定値scorerで局所範囲の全binを準備する。消費経路ではPreparedBodyScoresが環境scoreを置き換えるため、共有Landscapeから読むのはfrequency spaceとobjective modeだけである。解析設定の更新時は観測を失効させる。この確認により、source別Landscapeを毎hop複製する一時案は削除した。

## 検証経過

通常renderの先行試験では、seed 7・Sine 1声/4声・0.6秒について、準備済み評価の消費、missing flagとfalseのWAV/通常記録一致、ON二回のWAV/行動記録一致を確認した。その後、各sourceの消費、cache上限、採点時の観測経過時間も検査項目に加えた。

最初の全suiteは1171成功・5失敗・45 ignoreで停止した。失敗は以前の実験fixtureのRecipeと、今回消費時に再構成する実VoiceのRecipeとの不一致である。最初のログとsource hashは `target/runtime-action-validation/attempt1/` に保持した。実身体照合は弱めず、fixtureを実Voiceから作る形へ修正した。失敗5件の個別再検査を通過し、最終全suiteも通過した。拡張した通常render試験では保存先の親ディレクトリ未作成による失敗も修正し、最終suiteで証拠保存を含めて通過した。

## 最終結果

最終全suiteは1263成功・0失敗・45 ignore。`test_status.txt` は `cargo test exit=0 @ 2026-09-27T12:47:27+09:00`。fmt、標準Clippy、全target check、通常release buildも成功した。

通常renderの最後の定期記録（frame 48、sample 24,576、0.512秒）では、1声条件の消費は10回、4声条件は各10回（合計40回）だった。これは0.6秒のシーン全体の最終累計ではない。観測supportと採点時刻はともに24,576で、許容経過時間内。cacheは1声で163件/470,360 byte、4声では各162–165件/467,600–475,880 byteだった。ON二回のWAV・行動記録は一致し、missing flagとfalseも一致した。このSine fixtureではON/OFFのWAVも同一であり、音響差が出たとは主張しない。

既定OFFの通常releaseは、取得前に固定した4条件（I4 bounded Sine、both-on Sine/Harmonic/Modal、各4声）すべてで第六版のWAV・登録対象記録と一致した。比較器、入力、source、binary hashは取得前に `target/integration-runtime-v11-regression-20260927/plan.json` に固定し、結果は同ディレクトリの `comparison.json` / `manifest.json` に保存した。

- source: `target/integration-source-20260927-v11/manifest.json`（src/tests/Cargoの400ファイル）。SHA-256 `3a8bd1cfaa38bfefa34fe1428b1c34492eea1232613438bb0c7293369569cd7c`。
- 通常render証拠: `target/runtime-action-validation/normal-render-evidence/`。WAV、JSONL、config、scenarioとその生成test、summaryを保持。summary SHA-256 `f1b400831188ccf76e10a00545f729df8cbb665601c8f9a18a6bc62aac544768`。
- release binary SHA-256: `2386fb249d8e29d469b94793103932c04ea649cd9bdbe8f12a1e34e709d55210`。
- release回帰plan SHA-256: `137dcdfee00024d3e71a1a97aa9046db19ad9691a2c66aeb76092ae92e4dc701`。
- 最終ログと機械可読対応表: `target/runtime-action-validation/validation.json`。最初の失敗ログは同ディレクトリの `attempt1/` に保持。

これらの相対pathは `.worktrees/body-fitness-action` を基準とする。旧 `.worktrees/integration-fitness` の第十版sourceは保存したままで、第十一版へ上書きしていない。

## 到達範囲

Offline renderは候補worker完了を明示的に待つ決定的な機能試験である。通常instrumentの配線はpollingだが、今回のrender成功を実deviceの締切・処理能力の合格とはしない。sourceが変わる場合、古い結果は保留または拒否となる。通常runtimeのIDはu64 wrapまで単調増加する。低層APIで同じhop内に同一id/generationを明示再利用する操作は今回の試験対象外。

キャッシュ上限は解析template、Ready表、プロセス全体を含まない。明示ONではRecipe照合、要求候補、採点表に追加の確保がある。固定pitchにも候補準備を行うため、その費用削減は残る。初回spawn、parentありPeakBiased、global/ratio候補、変動身体の非同期追随、長時間の実時間動作、音色遺伝、作者採用は未完了。I11の14/16とModal記憶参照の既存不合格は、この試験によって変更しない。
