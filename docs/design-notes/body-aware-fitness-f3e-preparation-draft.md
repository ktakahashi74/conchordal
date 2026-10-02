# F3e: 非同期候補準備の予約と失効（取得前ドラフト）

日付: 2026-09-26。状態: 静的調査に基づく取得前ドラフト。実装、数値取得、通常 runtime への採用を宣言しない。
出発点: [runtime 引継ぎ](body-aware-fitness-runtime-handoff.md)、[F3d 一判断登録](body-aware-fitness-f3d-decision-registration.md)、[候補準備](body-aware-fitness-candidate-preparation.md)。

## 現在の責務と不足

F3d は固定 Voice の一判断を通した。`PitchHillClimbPitchCore::evaluation_pitches_for_probe` が局所格子、target 近傍、乱数候補の全採点位置を列挙し、`PreparedBodyScores` は target の bit、`Log2Space` の全 bin、`SmallRng` の完全一致、候補表の有限性・完全性を実 RNG 消費前に検査する。`PitchController::update_pitch_target` の通常 gate、補正、target・salience・適応更新、Voice の commit は残る。ただし表の設置は試験側の明示 barrier であり、非同期準備中に変わる状態を総合照合していない。

| 既存の識別子・入口 | 現在保証する範囲 | F3e で残る照合 |
| --- | --- | --- |
| `SourceIdentity { id, generation, birth_sample }` と `ReceivedBatch::accept` | 同一 epoch、source 集合、提出・受信・判断 sample 順、支持終端から 4,800 sample 以内。worker 失効も受信時と採用時に確認 | 身体世代、候補 recipe、routing、PitchController 状態は含まない |
| `BodyCapture::token(id, generation)` | slot と body generation。現実装の世代更新は身体 kind・unison・ratios 等の限定項目 | brightness などを含む完全 recipe の同一性は別途必要。slot index だけを永続 ID にしない |
| `footprint::Identity { source_id, body_generation, recipe_hash }` | `Recipe` の身体、周波数、hold、ADSR、modulator、平滑化、sample rate 等を hash 化 | 周波数を含むので候補ごとに identity が異なる。現在基音の identity 一個で候補表全体を証明しない。routing は hash 外 |
| `Voice::effective_control.pitch`、`PitchController` | 実 pitch 制御と private RNG、target を所有 | `PitchControl` は `PartialEq` を持たず、F3d の表は current pitch・control・body・route・epoch を検査しない |
| `WorkerState::hab_ecology` と `current_landscape` | `apply_landscape_updates` が共有 habituation を進め、次に `advance_population` が Voice を判断 | F3d は habituation 無効。source 除去環境へ適用する共有状態の版と時点を結び付ける必要 |

`Voice::footprint_recipe(fs)` の `freq_hz` は現在の `SoundBody` 基音。候補密度を作る際は各候補の `freq_hz` を使う。`PhonationBatch.routing` は `effective_control.body.routing` から出るが、既存 Tone の `ScheduleRenderer` は Tone ごとに routing を保持する。したがって「現在の control の routing が同じ」だけで過去 PCM の経路を証明しない。source PCM と混合 habitat PCM は出生から同じ routing 事象を観測し、route 変更時は結果を失効させる。通常 runtime の `body_capture` は observing かつ `temporal_body` 設定時だけ存在する。未起動のとき body generation を便宜的な 0 としない。初版の独立試験では `temporal_cognition::body::Capture::spawn` を明示的に起動し、`prepare` に出生時から全 Voice の `(id, generation, body_snapshot)` を渡して `token` を取得する。標準化設定は `TemporalBodyConfig { means: [0.0; 6], deviations: [1.0; 6], accent_means: [0.0; 2], accent_deviations: [1.0; 2] }` とし、記述子の判定値は使わない。後の通常 runtime 配線には既存 `body_capture` の有効化条件か同等の Voice 所有世代が必要であり、既定の temporal 設定を暗黙に変えない。

## 最小予約境界

初版は固定身体の `PitchHillClimb`、大域候補 0、ratio 候補なし、最大 4 source に限る。一つの Voice について、準備開始時に live RNG を**複製**し、その private copy で候補を列挙する。live RNG はこの時点で進めない。準備 job は既存型を保持する小さい平坦な記録で足りる。最低限、`SourceIdentity`、body generation、候補ごとの `Identity` と密度、準備時 `SmallRng`、target/current pitch の `f32::to_bits()`、候補幾何に関係する `Log2Space` の全 bin、実効 pitch control の照合値、実効 body recipe と habitat route、解析 epoch、準備 sample を持つ。環境受理後に別途作る一判断の score 表は、候補 bit key と有限 score を持つ。4,800 sample の鮮度期限は `ReceivedBatch` の環境に適用し、状態一致の代表身体密度へ機械的に移さない。新しい一般 pool や識別子 newtype は作らない。

control の照合を `PitchControl` の一部の手書き推測だけに任せない。最初は job に `PitchControl` を clone し、判断入口で各数値を bit 比較、列挙・scorer・gate・commit に影響する enum と option も一致させる、または `Voice::apply_effective_control` の全更新に増分 revision を付けて一致させる。後者を採る場合も変更漏れの経路を試験で潰す。`Debug` 文字列比較は使わない。body は `BodyCapture` 世代に加えて、現在の recipe から再計算した `Identity` と全候補 recipe identity を照合する。route は両 bus の bit 対を独立照合する。control・body・route を一つの曖昧な generation に折り畳まない。

第一段階の Seq 固定 scene では `Voice::footprint_recipe` は `None` なので、代表 `Recipe` の hold／ADSR／modulator／平滑化は実 Tone の生成条件から取得して fixture 中は固定する。判断時には実 Voice の `body_snapshot` と現在基音を再反映して照合する。任意の phonation 制御更新でこれら固定項目が変化する場合の検出は、この段階では未対応として拒否または対象外にする。PitchControl 以外の実効制御も、固定条件を崩す変更まで網羅したとは主張しない。listener の temperature bonus、現在の neighbor と adaptation は準備候補のキーや密度を変えないため、失効理由にせず判断時の値を既存補正順で使う。これらを準備時へ巻き戻さない。

受理する判断では、まず `ReceivedBatch::accept` を一度だけ呼ぶ。同一 `OutputBatch` 内の対象 `SourceIdentity` の環境だけを使い、epoch・支持終端・source 集合を再確認する。判断時点の共有 habituation 状態を一回だけ固定し、版を現在 hop または単調更新番号で記録する。その一つの状態を対象 source 除去 `Landscape` に適用し、全候補の F2 score を計算する。候補途中で別 batch や後続 hop の habituation を読まない。`HabituationField` の `state()` と `theta()` はあるが、現状は版番号を持たないため、版の発行点と `apply_landscape_updates` の順序を明示する必要がある。解析設定 epoch と habituation 版は別物として記録する。

実 RNG 消費前の入口で、source id／Voice generation／birth、body generation、全 recipe、route、epoch、target bit、current pitch bit、pitch control、`SmallRng` 完全一致、space 幾何、score 表の全キー・有限性、worker の期限と失効を検査する。現在基音は `PitchController::update_pitch_target` に渡る `Voice::body.base_freq_hz()` と同じ値を使う。Glide 中は target が不変でも current pitch が変わるので、初版では厳格に拒否する。この規則では滑走し続ける Voice の job が連続失効し、有効評価率が 0 になり得る。照合から `propose_with_scorer` までの同一 runtime worker の判断呼出し内で Voice／control／habituation を更新しない。これは cpal callback 上の処理ではない。scorer 内の assert は最終防壁に残すが、通常の失効判定に使わない。

どれか一つでも不一致、worker 無効、score 無支持なら job を破棄し、その**現在 gate** の既存 `propose_target_with_crowding_salience` を現在の live RNG で呼ぶ。過去 RNG の書き戻し、過去 gate の追実行、`force_set_target_pitch_log2`、旧 C での身体候補抽出はしない。fallback でも `PitchController` の経過時間処理、neighbor occupancy、target・salience・適応、range clamp、Voice commit の順序を保存する。身体評価を使えなかった gate は理由付き欠測に数え、有効評価に含めない。

## 実呼出しへの最小接続と二段階

第一段階は `#[cfg(test)]` の消費境界だけを加える。fixture が準備 job と独立の live source 台帳（birth sample、解析 epoch、capture token）を Voice に渡す。`Voice::decide_pitch_target_with_listener_pressure` が毎回、実 `id`／`metadata.generation`、`effective_control`、`body_snapshot`／現在基音、route とその台帳を読んで、既存 `update_pitch_target_with_listener_pressure` → `PitchController::update_pitch_target` の呼出しへ渡す。照合と身体表／旧 scorer の分岐は後者の**実際の `should_propose` 分岐内**、RNG を触る前に置く。試験専用の別 proposal 関数で gate を模倣しない。`Community` の decide→commit 順と Voice の公開された通常 `None` 経路は維持する。現在の F3d `PreparedBodyScores` にはこの判断用の source・body・control 文脈がないため、test-only の小さい平坦な記録をその外側に一つ持たせ、F3d の固定試験は残す。入口の失敗を panic ではなく理由付きで旧分岐へ返し、既存 scorer 内 assert はプログラミング誤りの防壁とする。

第二段階で source ごとの「計算中 1＋最新待機 1」の配送を足す。候補密度の worker と受理済み source 除去環境の結合は runtime worker 側で行い、`PitchController` に Tone、NSGT、queue、epoch 所有を入れない。worker 結果は必要時に取り出し、通常の判断呼出しへ一回だけ渡す。gate の受理／失効／未完という小さい test-only 結果だけを Voice 側へ返し、再予約は同じ control substep の `commit_decided_control` 後の状態から発行する。`Community` の二相経路では全 Voice の decide 後に commit するため、decide 直後の古い身体基音から予約しない。前段の実 Voice 失効試験を先に通し、配送試験で結果到着順を変えても同じ失効規則を使う。通常 runtime の常時有効化、作者既定はこの二段階に含めない。

拒否理由は成功／失敗の bool に畳まず、少なくとも `Pending`、`WorkerInvalid`、`SourceMismatch`、`BodyGenerationMismatch`、`RecipeMismatch`、`RouteMismatch`、`EpochMismatch`、`TargetChanged`、`CurrentPitchChanged`、`ControlChanged`、`RngChanged`、`SpaceMismatch`、`MissingOrUnsupportedScore`、`Stale` を区別する。`ReceivedBatch::accept` の `Rejection` は元の理由も保持する。gate 不成立時は消費も fallback proposal も起きないので、別の `NoGate` として数える。重複理由は worker の失効、batch の epoch／source／時計／鮮度拒否、job 未完、body generation、target、current pitch、recipe、route、control、RNG、space、共有 habituation 版、score の順で最初の一件を記録する。受理件数・失効件数・fallback 件数・身体 score 利用件数を sample 時計付きで出し、理由付き欠測を有効評価に含めない。

## 計算中の変化と上限

候補密度の準備は環境と独立だが、現在の代表 Tone 解析は一候補約 42 ms の参照費用を要し、準備中の target 変更は十分起こり得る。job 全体は古い target へ適用しない。初回 prefetch は出生・初期 control と body の確定後、最初の proposal gate より前に始める。gate 到達時に未完なら、その gate は現在の RNG と target で旧 proposal を実行する。gate 結果を保存し、**同じ control substep の commit 後**に更新済み RNG・target・current pitch・control・body・route から次の要求を作り、まだ計算中の旧 job があれば最新待機要求へ置く。旧 job 完了時は失効として捨て、最新待機要求だけを開始する。gate が生じない hop でも既存要求の完了を受け取れるが、live 状態への適用は次の実 gate までしない。これにより「未完→旧 proposal が状態変更→完了時失効」の循環は有界化するが、評価率の正値は保証しない。10 ms gate が代表密度準備より速ければ連続失効による評価率 0 を失敗として報告し、gate 間隔や許容を取得後に変更しない。

初版の配送は source ごとに計算中 1 件、置換可能な最新待機要求 1 件まで、全体で対象 source 最大 4 とする。新しい target が何度変わっても待機要求を上書きし、変更回数に比例して queue を伸ばさない。実行中 job の中断を必須にせず、失効結果は計算資源だけを消費したものとして測る。期限切れや長い Glide ではその gate を fallback とし、状態が安定してから次の準備を試みる。即時の同期再解析や、計算完了まで runtime worker を待たせる経路は作らない。

再要求は同一 source の proposal gate 一回につき最大一件、gate 不成立の hop ではゼロ件と固定する。source ごとの queue 保持上限は計算中 1＋最新待機 1、4 source 全体で最大 8 件（計算中 4＋待機 4）。旧 job 完了時の待機要求開始は新しい要求数に数えず、最新状態との再照合で失効した待機要求は捨てる。候補密度の再利用は、候補 bit、`footprint::Identity`、代表解析設定が完全一致する場合だけ許す。target が変わっても密度そのものは再利用可能だが、古い job の候補集合や採点表を現在の判断へ流用しない。

## 段階的な取得登録案

最初の取得は通常 runtime の配線を使わない offline 明示配送とする。48 kHz／512 sample、1 および 4 固定身体、出生 sample 0 からの source PCM、48 hop（支持終端 24,576 sample）の環境、72 hop の各代表密度、局所・no-ratio とする。以前の草案の環境 64 hop は F3c 固定 scene の終了 0.55 秒を越えるため、正の scene 取得前に 48 hop へ訂正した。F3c と同じ seed 1／4、1 source は Sine 440 Hz、4 source は Sine 440 Hz・Harmonic 440 Hz・Modal 466 Hz・Sine 660 Hz の順、両 bus は初期 ON／ON と固定する。route 失効例だけ habitat／presentation の片側を変更する。F3d の Harmonic 440 Hz＋Sine 466 Hz の正の移動試験 64 hop は別に保持する。fixture は上記設定の `Capture::spawn` を sample 0 より前に起動し、各 hop の `prepare` 後に `(slot, body_generation)` を取り、job と独立の live 台帳へ渡す。worker の sample 時計と 4,800 sample 期限をそのまま使う。1 source の正常一判断では、同一受理 batch・単一 habituation 版、全候補の F2 採点、正の表利用件数、target／salience／commit／終了 RNG の一致を要求する。4 source では全 source の分離と、各 Voice が自分だけを除いた環境を得ることを確認する。PCM と受理環境の比較は F3c で登録した bit 一致／F1 数値許容をそのまま使う。habituation 有効ケースは共有履歴を独立に一回進め、全候補が同じ版を使ったことを検査する。

失効試験は、準備後の live RNG、target、current pitch、pitch control、body recipe の値変更、body generation、route、epoch、source 世代／出生、期限切れ、worker 失効、候補 key 欠落・無支持を一件ずつ与える。全件で身体表の利用 0、理由分類、fallback の proposal・salience・適応・終了 RNG が「初めから job なし」の同じ gate と一致することを要求する。対照 Voice にも当該変更を同じ sample で適用する。特に body generation が変わらない brightness 更新と、target 不変の Glide を含める。変更を判断直前にも入れ、入口検査前後で live RNG を余分に消費しないことを確認する。

長時間計算の試験は barrier で worker 完了を遅らせ、途中で target を複数回更新する。実時計 sleep の偶然ではなく sample 時計と明示 barrier で順序を固定する。失効 job は一度も採用されず、source あたり実行中 1・最新待機 1、4 source 全体で同時保持 8 以下、新規要求は source／gate あたり 1 以下、期限後採用 0 を確認する。fallback 後の要求は commit 後の current pitch と RNG から作られ、古い job が終了すると最新待機要求だけが走ることを確認する。ここまでの結果で失効率・有効評価率・queue 滞留・計算時間を記録し、別の取得前登録を確定してから実 thread／通常 runtime の一部へ延ばす。1／4 source の検査を 64 Voice 常時利用や実時間合格と読み替えない。

F4 の代謝・出生効果、途中参加 source の履歴再構成、大域／ratio 候補、公開既定はこのドラフトの範囲外。
