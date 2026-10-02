# 身体評価 第十二版の統合検証（2026-09-27）

隔離worktree `.worktrees/body-fitness-ecology` の第十二版bを固定した。通常renderのHarmonic/Modal移動・身体変更・制御失効、初回Field出生の試験入力、通常runtimeのSine/Drone実時間条件を検証した。mainの通常実装への採用は行っていない。

## 実装と結果

- 周波数代表発音へI11の位相保持Recipeを流用したため、Droneで身体世代が毎hop変わり、準備判断を一度も消費できなかった。[改訂](body-fitness-frequency-phase-revision-20260927.md)では周波数比較用のDrone初期位相だけを0へ固定した。実発音、自己除去PCM、I11 onsetの位相契約は維持した。独立の位相・sway_rate・brightness試験も通過した。
- [通常render v2](body-fitness-runtime-changes-v2-registration-20260927.md)ではseed 17のHarmonic/Modalが実targetと基音を変更し、frame 144までに各30回の準備判断を消費した。別sceneのbrightness変更後はHarmonicの身体世代1→2を照合し、control変更で旧判断を拒否した後、frame 192までに40回へ回復した。Modalは世代1のまま41回。各scene二回のWAV・action・population記録は一致した。数値は定期report時点であり、scene総数ではない。
- 同じVoiceのroute変更は実density workerを通す内部境界試験で拒否と回復を確認した。通常Rhaiに同一Voiceのlive route変更APIは追加していない。
- [F4f初回出生](body-aware-fitness-f4f-initial-results-20260927.md)はConsonance Peak/Densityの全bin身体準備、既存の連続jitter、0 mass/全占有fallback、独立WeightedIndex、破損拒否の5試験を通過した。出生配線は `cfg(test)` 内で、Dissonance/Edge等の未取得範囲は結果文書に残した。
- [通常runtimeの実時間取得](body-fitness-runtime-live-results-20260927.md)はseed 7、Sine/Drone、1声・4声、10秒、模擬出力先でON/OFFの4条件を通過した。全窓声数維持、underflow 0、10.666667 ms超過0。ONの最大hop費用は1声0.853 ms、4声2.648 ms。frame 912の各声消費は90回と121/134/135/139回だった。

## 失敗の保持

初回の位相失効（消費0）、`at(freq)` が音高Lockを設定する旧移動fixture、Seqの既定寿命1秒による初回実時間取得失敗を別記録として保持した。`line(f,f)` とDroneへのfixture改訂は各再取得前に明記し、合否閾値は緩めていない。旧取得を修正版の事前登録成功へ読み替えない。

## 固定物と検査

最終通常suiteは1271成功・0失敗・46 ignore。`test_status.txt` は `cargo test exit=0 @ 2026-09-27T13:29:36+09:00`。全出力は同worktreeの `test_report.txt`。format、標準Clippy、全target check、release test buildが通過した。paced試験はignoreを明示解除して別実行し、13:30:47にexit 0。追加の `clippy --tests` は従来・並行コードの警告で失敗した記録を残し、標準Clippyの合格と区別する。

403ファイルのsource capsuleは `target/integration-source-20260927-v12b/`。manifest SHA-256は `539c12283edd1e3963e84a79e0dd05573683459d42584a6b4e9b04ba4c9f9cb0`。`target/ecology-validation/validation.json` はsource、全体試験、通常render最終取得、初回/修正版実時間取得の全rawを対応づける。validation SHA-256は `397e74b452f9a71406c96aa0f016f73298ced39d16d887d1deca153a36f05c18`。修正版release test binaryもrawと別にコピーして保持した。

## 次の単位

非Sine・身体変更の非同期実時間追随と長時間のcache/延期率、F4のparentありPeakBiased、未取得出生条件、通常runtimeの代謝・出生配線を残す。固定身体の長期生態・選択効果、F5の資源/感度/試聴/作者採用、F6の音色遺伝も未完了。今回の実時間合格はSine/Drone・10秒・模擬出力先に限定する。I11正式費用14/16とModal記憶参照0の未達は今回の結果で変更しない。mainのsource変更、commit、pushは行っていない。
