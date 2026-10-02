# F3d: 一回の Voice 音高判断への身体評価接続

日付: 2026-09-26。状態: 取得前登録。対象は隔離 worktree の `#[cfg(test)]` 経路だけであり、通常 runtime の採用ではない。

## 固定条件と入力

- `PitchHillClimbPitchCore` の局所候補のみを使う。`global_peak_count = 0`、ratio 候補無効。温度 0、`landscape_weight = 10`、移動費用 0、tessitura gravity 0、proposal interval 0.01 秒、初期 target 440 Hz。乱数候補三個の生成は通常の `PitchController` が持つ乱数器で行う。
- 48 kHz、512 sample/hop、既存 runtime 解析設定。source 1 は 440 Hz の Harmonic 身体を持つ実 `Voice`、source 2 は 466 Hz の Sine Tone。両方とも sample 0 に出生。各 source の実 Tone PCM を独立 renderer で採り、混合 renderer と共に 64 hop 連続投入する。代表身体密度は候補ごとに 72 hop 観測する。
- 候補準備時に実 `PitchController` の乱数器を複製し、`evaluation_pitches_for_probe` の全候補を得る。実 Tone 合成と `AnalysisStream` 前処理で各候補の代表密度を作る。準備中から判断まで Voice の target、pitch control、身体 recipe、乱数状態を変えない。候補は `f32` bit 値で照合する。
- F3b worker の同一 `ReceivedBatch` を、source id・世代・出生、epoch、支持終端、受信時刻、判断時刻で一度だけ受理する。対象 source を除いた一つの `Landscape` だけを全候補へ使う。初版は共有 habituation 無効、状態 0 とし、raw C と effective C の一致を検査する。
- 判断前に全候補を F2 `evaluate` へ通し、無支持・非有限・候補欠落なら判断を中止する。scorer 内では Tone 合成、NSGT、F2 評価をしない。全 scorer 呼出しで固定表の身体平均 C score を引き、既存の目的符号、landscape weight、移動費用、tessitura、crowding、adaptation の補正を同じ順序で適用する。
- 実乱数器の消費前に準備時の `SmallRng` 複製と現在状態を直接比較する。その時点の再複製から候補集合を再生成し、全 bit key が表に存在し、表 score が有限であることを確認する。target と Log2Space の全 bin 座標も準備時と一致させる。

## 一判断の合格条件

既存 Voice の `decide_pitch_target_with_listener_pressure` が通常 gate を通り、`PitchController` の実乱数器を使って `propose_with_scorer` を一回実行する。局所 peak 抽出も候補採点も同じ身体 score 表を読む。判断後の target、salience、adaptation、範囲 clamp と `commit_decided_control` は既存経路を通る。通常 `None` 経路の挙動は変えない。

試験は、受理 batch と候補密度の source/候補同一性、全 scorer 呼出位置の表内支持、身体 score の正の利用件数、旧 C field を変えても身体 score 表で選ぶ一判断の不変性、固定乱数入力に対する target・salience・終了乱数状態の一致、gate 不成立時に提案・乱数消費しない点を検査する。判断後の target は開始値から変わり、commit 後の身体周波数も target に一致する必要がある。固定 466 Hz 条件で動かなければ失敗として記録し、同じ取得中に周波数や閾値を調整しない。`force_set_target_pitch_log2`、旧 C による候補抽出、過去乱数状態の書き戻しは使わない。

候補キー欠落、準備後のRNG状態不一致、target不一致、周波数座標不一致の四つの拒否例も検査する。
各失敗の前後で呼出時の実RNG状態が一致することを要求する。これらは固定表の消費境界の検査であり、
一般の非同期待機中にbodyやcontrolが変わる場合の失効管理を実装したとは扱わない。
成功した一判断では候補数・採点表の利用件数・環境支持終端・初期／目標／commit周波数・salienceをJSONへ保存する。

## 範囲と失敗停止

候補準備と worker 配送を明示 barrier で同期する offline 試験。実時間 worker への配線、非同期予約・失効処理、PeakSampler、global/ratio 候補、代謝・出生、作者既定は対象外。準備後の target/control/body/RNG 変化、epoch/source 不一致、期限切れ、候補表不足は成功した身体判断として数えない。
