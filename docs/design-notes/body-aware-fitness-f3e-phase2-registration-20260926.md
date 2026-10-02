# F3e 第二段階: offline 明示 barrier 配送の取得前登録

日付: 2026-09-26。状態: 取得前登録。第一段階の封印版は `target-body-fitness/f3e-phase1-sealed-20260926-233646` に保存し、変更しない。今回の対象は `cfg(test)` の配送と実 Voice gate 接続だけであり、通常 runtime の thread 配線、任意 phonation 更新、実時間性能の合格を含めない。

## 固定 fixture と時計

48 kHz、512 sample/hop、epoch 37、出生 sample 0、seed 1 と 4 を使う。source は 1 件または 4 件。4 件は id 順に Sine 440 Hz、Harmonic 440 Hz、Modal 466 Hz、Sine 660 Hz、両 bus は ON/ON。Tone は第一段階と同じ hold 48,000 sample、ADSR 5 ms/0/1/200 ms、SeqGate 1 秒、平滑化 0、振幅 0.06。source PCM と混合 PCM を独立 renderer で 48 hop 作り、F3b worker の最終 `ReceivedBatch` は支持終端 24,576 sample、受信 sample 24,576 とする。共有 habituation はその同じ 48 hop から一度だけ進め、版 48 を全 source の同じ判断に用いる。代表身体は実 Tone の 72 hop を解析する。候補は有限局所 `PitchHillClimb`、global peak 0、ratio 無し。Body fitness の score 表は各候補の実密度と、受理された当該 source 除去環境から作る。環境の期限は `ReceivedBatch::accept` の 4,800 sample をそのまま使う。

1 source の配送系列を明示 barrier で固定する。最初の gate 前に要求 A を予約するが完了させない。`dt=0.005` の NoGate では要求数 0、proposal 0、RNG 不変。続く `dt=0.006`、判断 sample 24,576 の gate は `Pending` で旧 scorer に fallback する。対照の job 無し Voice と target、salience、adaptation、終了 RNG を bit 一致させ、実 Voice の commit 後に要求 B を一件予約する。このとき A が計算中、B が最新待機。判断 sample 25,088 の次 gate も A 未完のまま `Pending` fallback とし、commit 後に要求 C を一件予約して B を置換する。A の barrier を解放したとき、その完成物を消費せず破棄し、C だけを実行へ移す。C を完了させ、判断 sample 25,600 の gate に配送する。ここでは同じ受理 batch・habituation 版で正の身体 score 利用を要求し、準備表を直接渡す F3d 対照と target、salience、adaptation、終了 RNG、commit 後基音を一致させる。A/B/C の予約は各時点の live RNG、target、current pitch、control、body recipe、route、source identity を複製する。とくに B/C は直前の commit 後に作り、古い RNG を書き戻さない。現実の計算時間は barrier で決め、sleep に依存させない。

4 source の容量 fixture では各 source に計算中 1、最新待機 1 を保持し、全体上限 8 を毎操作で検査する。同じ source の追加要求は最新待機 1 件を置換し、source を増やす第 5 要求は容量拒否する。完成済み未消費結果はその source の「計算中」slot を置き換え、第三の保持 slot を作らない。したがって完成済み結果も含めた保持上限は source ごと 2、全体 8。計算中 A が終わった後に最新待機だけが開始し、古い完成物は Voice へ設置しない。NoGate で新規要求 0、各 proposal gate 後の新規要求は source ごと最大 1。NoGate 中の完了受信は許すが、次の実 gate まで結果を Voice へ適用しない。4 source の実 PCM／source 除去環境と各 Voice の正常採点は第一段階の封印済み 10 件を独立の前提とし、容量 fixture は配送上限と順序を検査する。

## 理由、失敗条件、ログ

実 `Voice::decide_pitch_target_with_listener_pressure` → `PitchController::update_pitch_target` の `should_propose` 内で、`Pending`、受理済み job、`ReceivedBatch` 拒否を区別する。NoGate は gate 結果と別に数える。`ReceivedBatch::accept` の `Rejection` は元の値を保持し、特に支持終端 24,576 に対する判断 sample 29,696（age 5,120）の `Stale`、worker を不正 epoch 提出で失効させた後の `Invalidated(Processor(EpochMismatch))` を区別する。この二つは正常系列とは独立の worker／Voice で取得し、正常結果へ混入させない。失効結果は表を設置せず同じ gate で旧 scorer へ進み、job 無し対照の target、salience、adaptation、終了 RNG と一致する。`Pending` も同じ対照に一致する。環境拒否や古 job を成功表利用に数えない。終了後に `BODY_FITNESS_F3E_PHASE2` の JSON 行で sample、source id、要求数、置換数、破棄数、最大同時保持数、理由、score 利用件数、終了 target/RNG 対照を保存する。

取得後に gate 間隔、期限、body 条件、環境長、候補集合、許容を変更しない。失敗した fixture とログを残し、別条件が必要なら新登録に分ける。第二段階の結果を 64 Voice 常時動作、通常 runtime の非同期処理、F4 の効果に拡張して解釈しない。
