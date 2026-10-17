# 設計ノートの索引

実装を読む入口と、設計・取得・判断の履歴を分けて案内する。
現在の作業範囲は[現行計画](../roadmap/plan-current.md)、文書全体の入口は[docs/README](../README.md)にある。
過去の記録が指す取得物の制約は[証拠の所在](../operations/evidence-locations.md)を参照する。
ファイルは移動・改名していない。

## 中核のノート

- [timbre](timbre.md)。身体の合成とcoreの境界、音色を評価するlistener model。
- [listener-twin](listener-twin.md)。提示された音を聴くListenerTwinの設計。
- [DCCの認知神経階層](dcc-neurocognitive-hierarchy.md)。時間階層ごとの役割と設計の根拠。
- [時間構造DCCの設計レビュー](dcc-hierarchy-review.md)。階層設計へのレビュー記録。
- [neural-rhythm-meter](neural-rhythm-meter.md)。神経リズムと拍のモデル。
- [rhythm-temporal-niche](rhythm-temporal-niche.md)。拍、位相、時間的ニッチの再設計。
- [lifecycle-time-domain](lifecycle-time-domain.md)。生存・回復の時間領域への再パラメータ化。
- [voice-movement-redesign](voice-movement-redesign.md)。占有を用いるVoiceの移動。
- [crowding-unison-avoidance](crowding-unison-avoidance.md)。混雑とユニゾン回避。
- [placement-vocabulary](placement-vocabulary.md)。配置のfield targetを直交させる語彙。
- [tension](tension.md)。移動の温度と配置の相対level。
- [並列継続の作業境界](parallel-continuation-20260927.md)。2026-09-27の隔離作業と取得の履歴。
- [技術ノート設計台帳・英語](technote-ledger.md)。節番号とanchorを保った設計判断の履歴。
- [技術ノート設計台帳・日本語](technote-ledger.ja.md)。同じ台帳の日本語版。

## body-fitnessの登録・結果・判断

`body-aware-fitness-*`と`body-fitness-*`は同じ系統の記録である。
以下は2026-09-26〜29を中心とする履歴の対応表であり、箇条書きの各行を「登録 → 結果 → 判断・限界」の順に並べた。
「判断・限界」はその記録での解釈と適用範囲を指す。作者の採択、本体への統合、科学的受入を一括して表すものではない。
取得前草案は登録と区別し、未収録の結果を成功と読まない。
日付がファイル名に無い記録は本文の日付を読む。二つのprefixも日付の有無も、採用順序を意味しない。

全体の計画とその改訂は[旧身体適応度計画](body-aware-fitness-plan.md)に残る。
2026-10-07以後の採択と受入の見直しは[台帳§9.3.57](technote-ledger.ja.md#two-tier-rule)、
同期の身体評価などの実装結果は[台帳§9.3.58](technote-ledger.ja.md#realtime-acceptance-scope)を読む。
下記の試験専用・未達という当時の記録を、現在の採択で上書きしない。

### 最小参照とF3への境界（2026-09-26）

- **F1／F2の最小参照**。[登録](body-aware-fitness-reference.md) → [結果](body-aware-fitness-results-20260926.md) → [判断・限界](body-aware-fitness-results-20260926.md#残る工程)。関連 [f0-audit](body-aware-fitness-f0-audit.md)・[synthesis-review-20260926](body-fitness-synthesis-review-20260926.md)。
- **F3a・解析状態**。[登録](body-aware-fitness-f3a-registration.md) → [結果](body-aware-fitness-results-20260926.md) → [判断・限界](body-aware-fitness-results-20260926.md#f3a-最大4-sourceの解析状態)。関連 [runtime-handoff](body-aware-fitness-runtime-handoff.md)。
- **F3の費用・密度前処理・明示drive**。[登録1](body-aware-fitness-cost-probe-registration.md)・[登録2](body-aware-fitness-density-only-registration.md)・[登録3](body-aware-fitness-driven-cost-registration.md) → [結果](body-aware-fitness-results-20260926.md) → [判断・限界](body-aware-fitness-results-20260926.md#f3追加費用-密度専用経路と明示drive付きprocessor)。
- **F3b〜d・配送、実Voice観測、一判断**。[登録1](body-aware-fitness-f3b-registration.md)・[登録2](body-aware-fitness-f3c-shadow-registration.md)・[登録3](body-aware-fitness-f3d-decision-registration.md) → [結果](body-aware-fitness-results-20260926.md) → [判断・限界](body-aware-fitness-results-20260926.md#f3bd-配送実voice観測一判断への接続)。関連 [candidate-preparation](body-aware-fitness-candidate-preparation.md)。

### F3eの候補準備と配送（2026-09-26〜27）

- **第一段階**。[取得前草案](body-aware-fitness-f3e-preparation-draft.md) → [結果](body-aware-fitness-f3e-phase1-results-20260926.md) → [判断・限界](body-aware-fitness-f3e-phase1-results-20260926.md)。
- **第二段階・初回とv2**。[登録](body-aware-fitness-f3e-phase2-registration-20260926.md) → [結果1](body-aware-fitness-f3e-phase2-results-20260926.md)・[結果2](body-aware-fitness-f3e-phase2-v2-results-20260927.md) → [判断・限界](body-aware-fitness-f3e-phase2-v2-results-20260927.md)。
- **第三段階・仮想完了遅延**。[登録](body-aware-fitness-f3e-phase3-registration-20260927.md) → [結果](body-aware-fitness-f3e-phase3-results-20260927.md) → [判断・限界](body-aware-fitness-f3e-phase3-results-20260927.md)。
- **第四段階・延期政策**。[登録1](body-aware-fitness-f3e-phase4-registration-v1-20260927.md)・[登録2](body-aware-fitness-f3e-phase4-registration-20260927.md) → [結果](body-aware-fitness-f3e-phase4-results-20260927.md) → [判断・限界](body-aware-fitness-f3e-phase4-results-20260927.md)。
- **第四段階・live-paced補助取得**。[登録](body-aware-fitness-f3e-phase4-live-registration-20260927.md) → [結果](body-aware-fitness-f3e-phase4-live-results-20260927.md) → [判断・限界](body-aware-fitness-f3e-phase4-live-results-20260927.md)。
- **F3e5・候補密度cache**。[登録](body-aware-fitness-f3e5-cache-registration-20260927.md) → [結果](body-aware-fitness-f3e5-cache-results-20260927.md) → [判断・限界](body-aware-fitness-f3e5-cache-results-20260927.md)。

### F4の代謝と出生の一機会（2026-09-26〜27）

消費境界は[接続前監査](body-aware-fitness-f4-consumer-audit.md)を読む。

- **F4a・成人代謝**。[登録](body-aware-fitness-f4a-metabolism-registration.md) → [結果](body-aware-fitness-f4a-results-20260926.md) → [判断・限界](body-aware-fitness-f4a-results-20260926.md)。
- **F4b・Random出生**。[登録](body-aware-fitness-f4b-birth-registration.md) → [結果](body-aware-fitness-f4b-results-20260927.md) → [判断・限界](body-aware-fitness-f4b-source-review-20260927.md)。関連 [f4b-birth-draft](body-aware-fitness-f4b-birth-draft.md)。
- **F4c・Hereditary初回とv2**。[登録1](body-aware-fitness-f4c-hereditary-registration.md)・[登録2](body-aware-fitness-f4c-v2-registration-20260927.md) → [結果1](body-aware-fitness-f4c-results-20260927.md)・[結果2](body-aware-fitness-f4c-v2-results-20260927.md) → [判断・限界](body-aware-fitness-f4c-v2-results-20260927.md#検証と保存)。関連 [f4c-hereditary-draft](body-aware-fitness-f4c-hereditary-draft.md)。
- **F4d・settle補助候補**。[登録](body-aware-fitness-f4d-registration-20260927.md) → [結果](body-aware-fitness-f4d-results-20260927.md) → [判断・限界](body-aware-fitness-f4d-results-20260927.md)。
- **F4e・親なしPeakBiased**。[登録](body-aware-fitness-f4e-registration-20260927.md) → [結果](body-aware-fitness-f4e-results-20260927.md) → [判断・限界](body-aware-fitness-f4e-results-20260927.md)。
- **F4e・親ありPeakBiased**。[登録](body-aware-fitness-f4e-parent-registration-20260927.md) → [結果](body-aware-fitness-f4e-parent-results-20260927.md) → [判断・限界](body-aware-fitness-f4e-parent-results-20260927.md)。
- **F4f・初回Field spawn**。[登録](body-aware-fitness-f4f-initial-registration-20260927.md) → [結果](body-aware-fitness-f4f-initial-results-20260927.md) → [判断・限界](body-aware-fitness-f4f-initial-results-20260927.md#未取得と解釈限界)。
- **F4f・第十三版の残余**。[登録](body-aware-fitness-f4f-v13-residual-registration-20260927.md) → [結果](body-aware-fitness-f4f-v13-residual-results-20260927.md) → [判断・限界](body-aware-fitness-f4f-v13-residual-results-20260927.md#残る範囲)。

### 隔離統合と通常runtimeへの接続（2026-09-26〜27）

第1〜18版の集約は[統合検証の記録](body-fitness-integration-validation-20260926.md)にある。各版の結果を一つの合格へまとめない。

- **統合第九版・第十版**。[登録1](body-fitness-integration-v9-registration-20260927.md)・[登録2](body-fitness-integration-v10-registration-20260927.md) → [結果](body-fitness-integration-validation-20260926.md) → [判断・限界](body-fitness-integration-validation-20260926.md#第十版-候補密度cacheとglide再利用peakbiased全bin評価)。
- **通常runtimeの観測**。[登録](body-fitness-runtime-observation-registration-20260927.md) → [結果](body-fitness-runtime-observation-results-20260927.md) → [判断・限界](body-fitness-runtime-observation-results-20260927.md)。
- **途中出生と退役**。[登録](body-fitness-runtime-lifecycle-registration-20260927.md) → [結果](body-fitness-runtime-lifecycle-results-20260927.md) → [判断・限界](body-fitness-runtime-lifecycle-results-20260927.md)。
- **延期中の実状態変化**。[登録](body-fitness-deferred-state-registration-20260927.md) → [結果](body-fitness-deferred-state-results-20260927.md) → [判断・限界](body-fitness-deferred-state-results-20260927.md)。
- **glide中の密度再利用**。[登録](body-fitness-glide-reuse-registration-20260927.md) → [結果](body-fitness-glide-reuse-results-20260927.md) → [判断・限界](body-fitness-glide-reuse-results-20260927.md)。
- **通常runtimeの行動消費**。[登録](body-fitness-runtime-action-registration-20260927.md) → [結果](body-fitness-runtime-action-results-20260927.md) → [判断・限界](body-fitness-runtime-action-results-20260927.md#到達範囲)。
- **非Sineとcontrol変更**。[登録1](body-fitness-runtime-changes-registration-20260927.md)・[登録2](body-fitness-runtime-changes-v2-registration-20260927.md) → [結果1](body-fitness-runtime-changes-v2-registration-20260927.md)・[結果2](body-fitness-ecology-validation-20260927.md) → [判断・限界](body-fitness-ecology-validation-20260927.md#失敗の保持)。関連 [frequency-phase-revision](body-fitness-frequency-phase-revision-20260927.md)。
- **Sine／Droneのpaced取得**。[登録](body-fitness-runtime-live-registration-20260927.md) → [結果](body-fitness-runtime-live-results-20260927.md) → [判断・限界](body-fitness-runtime-live-results-20260927.md#初回不合格と改訂)。
- **通常offline代謝**。[登録](body-fitness-offline-metabolism-registration-20260927.md) → [結果1](body-fitness-offline-metabolism-results-20260927.md)・[結果2](body-fitness-normal-offline-provider-focused-results-20260927.md) → [判断・限界](body-fitness-metabolism-validation-20260927.md#失敗の保持と残余)。
- **非Sine・途中変更のpaced取得**。[登録](body-fitness-non-sine-changes-runtime-live-registration-20260927.md) → [結果](body-fitness-non-sine-changes-runtime-live-results-20260927.md) → [判断・限界](body-fitness-metabolism-validation-20260927.md#非sine実時間取得)。

### 取消・再利用・offline出生の統合（2026-09-27）

- **代表スペクトルの再利用**。[登録](body-fitness-spectral-reuse-registration-20260927.md) → [結果](body-fitness-spectral-reuse-results-20260927.md) → [判断・限界](body-fitness-spectral-reuse-results-20260927.md)。
- **旧世代取消と回復**。[登録](body-fitness-recovery-registration-20260927.md) → [結果1](body-fitness-cancellation-results-20260927.md)・[結果2](body-fitness-recovery-results-20260927.md) → [判断・限界](body-fitness-recovery-validation-20260927.md#検証範囲と失敗履歴)。
- **初回Field出生・初回とv2**。[登録1](body-fitness-offline-birth-registration-20260927.md)・[登録2](body-fitness-offline-birth-v2-registration-20260927.md) → [結果](body-fitness-offline-birth-results-20260927.md) → [判断・限界](body-fitness-offline-birth-results-20260927.md#初回の失敗を保持)。
- **冷候補の並行準備**。[登録](body-fitness-cold-parallel-registration-20260927.md) → [結果1](body-fitness-cold-parallel-focused-results-20260927.md)・[結果2](body-fitness-cold-parallel-results-20260927.md) → [判断・限界](body-fitness-cold-parallel-validation-20260927.md#局所検査と失敗履歴)。
- **初回出生と代謝の結合**。[登録](body-fitness-birth-metabolism-registration-20260927.md) → [結果](body-fitness-birth-metabolism-results-20260927.md) → [判断・限界](body-fitness-birth-metabolism-validation-20260927.md#検査と修正履歴)。関連 [birth-metabolism-coupling-draft](body-fitness-birth-metabolism-coupling-draft-20260927.md)。
- **親ありrespawn一機会**。[登録](body-fitness-offline-respawn-registration-20260927.md) → [結果](body-fitness-offline-respawn-results-20260927.md) → [判断・限界](body-fitness-offline-respawn-validation-20260927.md#判定の範囲)。関連 [offline-respawn-draft](body-fitness-offline-respawn-draft-20260927.md)・[offline-respawn-focused-review](body-fitness-offline-respawn-focused-review-20260927.md)・[respawn-peak-reference](body-fitness-respawn-peak-reference-20260927.md)。
- **反復respawn**。[登録](body-fitness-repeated-respawn-registration-20260927.md) → [結果](body-fitness-repeated-respawn-results-20260927.md) → [判断・限界](body-fitness-repeated-respawn-validation-20260927.md#独立対照と残余)。関連 [repeated-respawn-draft](body-fitness-repeated-respawn-draft-20260927.md)・[repeated-respawn-focused-review](body-fitness-repeated-respawn-focused-review-20260927.md)。

### 草案・事後診断・方式の再設計（2026-09-27〜29）

- **短い記憶区間**。独立した登録の対応は[統合記録](body-fitness-integration-validation-20260926.md)を参照 → [試験専用の探索結果](body-fitness-partial-memory-results-20260927.md) → [不合格と保証の境界](body-fitness-partial-memory-results-20260927.md#不合格と保証の境界)。
- **全消費者の比較**。[設計草案](body-fitness-all-consumers-draft-20260927.md)・[実装準備](body-fitness-three-consumer-integration-plan-20260927.md) → このディレクトリには対応する全比較の取得結果を収録していない → [実装レビュー](body-fitness-all-consumers-implementation-review-20260927.md)。[移動停止理由の診断登録](body-fitness-movement-diagnostic-registration-20260927.md)が参照する別worktreeの結果は、この索引からの合格判定には使わない。
- **非同期出生**。[取得前の入力草案](body-fitness-async-birth-acquisition-draft-20260928.md)・[境界観測草案](body-fitness-boundary-observation-draft-20260928.md) → 別worktreeの結果への参照は各草案に保存 → [旧結果の現在地](body-fitness-runtime-restart-20260929.md#6-旧結果の現在地)。
- **計画レビューと再出発**。[旧計画](body-aware-fitness-plan.md) → [レビュー](body-fitness-plan-opus55-review-20260929.md) → [所見への対応](body-fitness-plan-review-response-20260929.md)・[方式と時刻契約の判断](body-fitness-runtime-restart-20260929.md)。レビューの完了と方式の本番採用を区別する。
- **直接部分音モデルv1**。[登録](body-fitness-direct-model-contract-20260929.md) → [結果](body-fitness-direct-model-results-20260929.md) → [判断・限界](body-fitness-direct-model-results-20260929.md#現在の結論)。関連 [direct-damping-diagnostic](body-fitness-direct-damping-diagnostic-20260929.md)。
- **部分音群モデルv2**。[登録](body-fitness-partial-groups-contract-20260929.md) → [結果](body-fitness-partial-groups-results-20260929.md) → [判断・限界](body-fitness-partial-groups-results-20260929.md#近似の境界と次の判断)。関連 [partial-groups-draft](body-fitness-partial-groups-draft-20260929.md)。
- **部分音群の採点位置**。[登録](body-fitness-anchor-readout-diagnostic-20260929.md) → [結果](body-fitness-anchor-readout-results-20260929.md) → [判断・限界](body-fitness-anchor-readout-results-20260929.md#結論)。
