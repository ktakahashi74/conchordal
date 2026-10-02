# 第十四版: 取消後の非Sine実時間追随

[取得前登録](body-fitness-recovery-registration-20260927.md)に従い、同じseed 17、Harmonic 220 Hz／Modal 330 Hz、10秒・938 hop、frame 300のbrightness変更、frame 600のcontrol変更をOFF／ON各一回取得した。結果はOFF合格、ON不合格。身体変更後の新世代消費はframe 480で、期限frame 432を超えた。第十三版bのframe 522より早い観測だが、期限内回復や全体の実時間合格とはしない。

## 固定した取得と判定

取得は2026-09-27 14:57:37–14:57:57 JST。前後のホスト確認で他のcargo・rustc・render・試験プロセスは0。実音声デバイスを使わず、既存の模擬出力先へ描画した。取得器 `src/runtime/body_fitness_runtime_live_tests.rs` は第十三版bとbyte一致し、実際のScenarioとIRのSHAも各modeで一致。新しい出生flagと代謝flagはOFF。

| 項目 | OFF | ON |
|---|---:|---:|
| 判定 | 合格 | 不合格 |
| 有効hop | 938 | 938 |
| 全窓のVoice数 | 2 | 2 |
| 出力underflow | 0 | 0 |
| hop予算超過 | 0 | 0 |
| hop p50 | 0.276642 ms | 0.261697 ms |
| hop p95 | 0.548357 ms | 0.596945 ms |
| hop p99 | 0.636333 ms | 0.818689 ms |
| hop最大 | 1.211808 ms | 1.372639 ms |

ONはsupport age上限4800 samples、各sourceのcache上限、Modal身体世代の不変、control変更後の回復と継続消費を満たした。観測workerの受理912件、停止なし。frame 288／432／576／720／912のHarmonic消費は6／8／12／30／55、Modalは4／6／15／22／28。frame 432のHarmonic累積8回は新世代の消費を含まず、期限内回復ではない。frame 912では各cacheが256 entry・727,040 byte、完了84件・消費83件・拒否1件。拒否は後のcontrol変更に対する `Consume(ControlChanged)`。

## 取消と冷計算の分離

旧身体のserial 14はsample 151040（frame 295）に投入。身体変更のsample 153600（frame 300）で取消を要求し、sample 154112（frame 301）に取消応答を受信した。135候補のうち85候補が完了し、worker計算区間の壁時間は70,115 µs。取消要求1件と実取消応答1件が一致し、旧結果からの新しい決定は行わなかった。

新身体serial 15はsample 154624（frame 301の描画後、frame 302の開始境界）に投入され、sample 245760（frame 480）に受信・消費された。全135候補を完了し、worker計算区間は1,905,687 µs。取消応答の後に約1.91秒の冷計算が残る。sample座標はhop境界で、requestは直前hopの描画後、受信と決定は当該hopの先頭を表す。

最初の冷計算もHarmonic 133候補で1,883,951 µs、Modal 135候補で1,943,734 µsだった。これらはCPU時間ではなく、スケジューリング停止も含むworker内の壁時間。候補生成規則と72 hopは変更していないが、非同期の決定時刻が変わると軌跡と候補集合が変わる。第十三版bの新世代132候補と今回135候補の時間を、同一負荷の速度比較とは扱わない。

代表解析のscratch再利用は[独立数値比較](body-fitness-spectral-reuse-results-20260927.md)を通過した。今回の取得だけでは、その変更単独の速度効果を分離できない。取消の1 hop応答は確認できたが、冷計算を期限内に終える対策が引き続き必要。

OFFのhabitat／presentation PCM SHAは第十三版bと一致し、`b88c69cc650663a16bac036f2a2b09a7accf431239c42a6f8b956276477cbfce`。ONの音声は非同期判断の時刻が異なるため別であり、今回の一回取得を反復再現性や一般的な音楽的効果へ拡張しない。

## 保存先と次の単位

隔離worktree `.worktrees/body-fitness-recovery/target/runtime-non-sine-v14-20260927/` に実行ファイル、取得command、登録コピー、stdout/stderr、exit 101、raw、summaryを保存。全suiteとsourceの固定は[統合検証](body-fitness-recovery-validation-20260927.md)を参照する。第十三版bのsource/binary/rawと未達判定も変更していない。

次は冷計算の処理内訳と、有界な候補並行準備を含む短縮案を別登録で比較する。候補順、RNG、cache scope、取消応答、数値結果を保持し、追加threadと解析状態のメモリを明示する。今回の期限や候補数、観測窓を結果後に緩めない。通常出生と代謝の同時接続、非同期代謝、長期生態、音色遺伝、作者採用は引き続き未完了。
