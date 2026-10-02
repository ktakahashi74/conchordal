# 通常runtimeの模擬出力先による実時間取得結果（2026-09-27）

[登録](body-fitness-runtime-live-registration-20260927.md)のDrone改訂版（第十二版b）は、1声・4声のON/OFF各一回、計4条件を通過した。48 kHz、512 samples/hop、seed 7、固定Sine身体、音高Freeの `line(f, f)` 配置、10秒の条件である。実device/callback、非Sineの実時間追随、長時間運転の合格ではない。

## 観測結果

全条件で主観測窓frame 0–937の938 hopに指定声数を維持し、callback相当の不足sample数とhop予算超過はともに0だった。予算は10.666667 ms。総profileは939 hopで、Finishはframe 938。ONのobserver停止はなく、frame 912までの受理batchは各912。cache hitがあり、支持時刻のageは0–4800 samples、各cacheは256 entry / 1 MiB以内だった。frame 912の各cacheは256 entry、727040 bytes。

| 条件 | hop p50 ms | p95 ms | p99 ms | 最大 ms | frame 912までの各声の消費回数 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1-source-off | 0.083 | 0.263 | 0.298 | 0.638 | — |
| 1-source-on | 0.189 | 0.543 | 0.699 | 0.853 | 90 |
| 4-source-off | 0.272 | 0.473 | 0.565 | 2.235 | — |
| 4-source-on | 0.460 | 1.231 | 1.702 | 2.648 | 121/134/135/139 |

消費回数は最終定期reportのframe 912の累積値であり、窓末尾frame 937や全sceneの総数ではない。ONの主窓全体では実target・基音の変更も記録した。1声のtarget変更は91回、4声は17/18/14/20回。ただしtarget変更数と評価消費数は別の量で、改善がなければ評価を消費してもtargetは変わらない。

配線は約159–167 ms、配線後prefill待ちは約1.9–3.5 ms。source観測workerの生成はframe 0のhop費用に含む。consumerの最大起床jitterは条件ごとに約67–79 μs。rawには各予定/実際時刻、sample数、全profile、report、両busのPCM digestを保存した。音声はメモリでdigestを計算して破棄し、device再生もWAV保存も行っていない。ON/OFFのdigest差は記録値であり、非同期取得一回ずつの比較から音響上の優劣や作者の採用を主張しない。background workerのCPU時間・全体メモリ上限はこのhop profileだけでは証明しない。

## 初回不合格と改訂

初回のSeq条件は4条件ともframe 94でVoiceが0となり、938 hopの生存条件を満たさなかった。Seqの既定durationが1秒で、endurance指定はその寿命を変更しなかった。欠損0・hop超過0でも、この初回を10秒間の稼働合格とは扱わない。旧rawは `target/runtime-live-v12-20260927/`、旧sourceは `target/integration-source-20260927-v12/` に保持した。

その後、再取得前にDroneへfixtureを改訂し、SpawnのDrone/Free条件を読込時に検査した。閾値とseed、周波数、窓長は変更していない。修正版のsource差分は試験harness一ファイルのみ。初回と修正版をまとめたbest-run選択ではなく、異なる事前登録条件の取得として区別する。

## 保存先

- 修正版raw、取得log、exit code、test binary: `.worktrees/body-fitness-ecology/target/runtime-live-v12b-20260927/`
- source manifest: `target/integration-source-20260927-v12b/manifest.json`、SHA-256 `539c12283edd1e3963e84a79e0dd05573683459d42584a6b4e9b04ba4c9f9cb0`
- release test binary SHA-256: `62bfab3a4e95753ad81de856a3478314a8507660ee115cdc3f45740d669f6d4c`
- source・全体試験・取得物の対応: `target/ecology-validation/validation.json`

実時間取得の直前にホストのcargo/test/render停止を確認し、結果を `exclusive-precheck-drone.json` に保存した。取得は登録どおり各条件一回。通常の全体試験は1271成功・0失敗・46 ignoreで、上記paced試験はその後にignoreを明示解除して別実行し、exit 0だった。機能試験、実時間条件、試聴、作者採用は別の判定を維持する。
