# F3e 第三段階: 仮想完了遅延と連続 gate の取得前登録

日付: 2026-09-27。状態: 取得前登録。第二段階 v2 の source／結果 capsule `target-body-fitness/f3e-phase2-v2-sealed-20260927-000217` は変更しない。ここでは sample 時計上の配送因果と飢餓条件を検査する。壁時計性能、通常 runtime の非同期配線、公開既定の合否は扱わない。

## 仮想遅延の主行列

48 kHz、512 sample/hop、128 hop（hop 0–127）、source 数 1／4、proposal gate 周期 P=1／4 hop、仮想完了遅延 L=1／2／4／8 hop の 16 条件を固定する。source は id 1–4、身体／初期基音は Sine 440 Hz、Harmonic 440 Hz、Modal 466 Hz、Sine 660 Hz。各 source の最初の要求を hop 0 に予約し、主行列の**最初の gate は hop P** とする。同じ hop に完了と gate がある場合、完了を先に処理する。gate 後に実 Voice の commit を終えてから次要求を一件だけ予約する。非 gate hop では追加要求 0。全 gate で実 `Voice` → `PitchController` の旧 proposal／commit を通し、job 無しの同条件 twin と target、salience、adaptation、終了 RNG、commit 後基音を照合する。未完なら `Pending` を理由として実 gate に渡す。

主行列は**queue 状態だけ**の診断とする。破棄が確定した完了 job では Tone 代表密度を数値計算しない。Ready に到達した job もこの行列では q を計算・採点せず、`ready_opportunity` と `numeric_skipped` を数えた上で旧 proposal を実行する。`actual_body_consume` は設計上 0 であり、これを「実測有効評価率 0」と呼ばない。`ready_opportunity=0` の条件だけが、この固定配送則の構造上、身体表を消費できない飢餓反例になる。期待境界は P=1／L=1 と P=4／L=1,2,4 で ready 機会が正、P=1／L=2,4,8 と P=4／L=8 で 0。数値は取得後に変更しない。保持上限は source ごと running／Ready 1＋latest pending 1、全体 8。

初期位相の別反例として、hop 0 の予約直後に最初の gate を出す P=L=1 と P=L=4 を source 1／4 で同じ 128 hop 走らせる。この場合、最初の pending が同時刻完了より先に発生し、その後も completion が pending により置換され得る。主行列の L=P と混同せず、`ready_opportunity=0`、旧 fallback 一致を別行で検査する。

各条件の `BODY_FITNESS_F3E_PHASE3` JSON 行に source 数、P/L、初期位相、gate 数、要求／置換／破棄／Ready 完了／Ready 機会／`numeric_skipped`／実身体表消費／Pending fallback、source ごとと全体の最大保持数を出す。gate と完了の sample 時計も条件に結び付ける。Ready 機会と実身体表消費を合算しない。NoGate では要求追加 0、RNG 消費 0、配送完了の受信は可能だが Voice 適用は次 gate まで保留する。

## 休止後の実数値回復

別の 1 source 回復条件は P=4、L=8、hop 0 に予約と即時 gate、hop 0／4／…／60 を連続 fallback、hop 64–79 は gate なし、hop 80 に再開 gate と固定する。連続期に pending が既にある完了は queue 遷移と破棄だけを数え、数値計算しない。最後の最新要求は休止中の hop 72 に実 Tone の 72 hop 代表密度を計算して Ready にし、hop 80 まで温存する。この密度は環境 C、F2 score、habituation 版を保持しない。

環境は別途、出生 sample 0 から同じ固定 Sine Tone の source PCM と混合 PCM を 80 hop レンダリングして F3b worker へ渡す。支持終端・受信・判断をいずれも hop 80 の sample 40,960 とし、その `ReceivedBatch::accept` を一度行う。共有 habituation は同じ 80 hop の混合 PCM で毎 hop 進め、版 80 の state を source 除去環境へ一回適用する。Ready の q と候補 `Identity` から、この**判断時の**環境・版で F2 score 表を作る。準備時の source／body generation／Recipe／route／epoch／target／current pitch／RNG／space の情報は上書きしない。旧 habituation 版 79 を表に束縛した負例では実 Voice gate が `HabituationChanged` で拒否し、job 無し twin の fallback と target、salience、adaptation、終了 RNG が一致する。同じ Ready を版 80 で採点した正例では表利用を 1 件以上確認し、直接表を渡す対照の target、salience、adaptation、終了 RNG、commit 後基音と一致する。負例と正例は同じ履歴から分岐した独立 Voice で行う。固定 source PCM と live Voice の基音を一致させるため、回復条件の pitch control は `landscape_weight=0`、`move_cost_coeff=10` に固定し、連続期を通じて基音 440 Hz の bit 不変を要求する。成立しなければ失敗として記録し、値を調整しない。

数値回復条件以外では仮想 Ready から fake q や過去 score を live proposal へ渡さない。判断時の batch 期限 4,800 sample、既存 F3e の理由・失効検査を緩めない。境界 `<`、`=`、`>`、即時 gate 位相、休止後回復を別結果として報告する。
