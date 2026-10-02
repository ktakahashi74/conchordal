# F3e 第四段階補助取得: live-paced worker 有効率

日付: 2026-09-27。状態: 取得前登録。機能検査の明示 join barrier とは分け、専有枠で取得する。通常 runtime 全体への配線の検査ではない。

`cargo test --release --lib --no-run` でビルドした release test binary を使う。48 kHz、512 sample/hop、938 hop（10.005333 秒）を source 1 と 4 で各一回。全 source は ID 1–4、固定 Sine 440 Hz、amp 0.06、同じ身体 Recipe、独立 Voice、独立 source PCM。混合 PCM は同じ Tone を重ね、各 source の `ScheduleRenderer` と F3b `Worker` に渡す。Tone の hold は 480,000 sample＝10 秒であり、最後の1 hop は release 境界後の PCM を含む。全 hop で `Voice` decide／commit に実 `dt=512/48000` を渡す。source身体と実 Voice の基音・Recipe の一致を毎 hop 検査し、変化すれば試行不合格。pitch は `landscape_weight=0`、`move_cost_coeff=10`、基音 440 Hz bit 固定。これ以外の身体変動へ一般化しない。

密度要求は source ごと最大 Running 1、Ready 1。実 thread で72 hop代表密度を計算し、channel で完了を送る。計時中 gate は `try_recv` のみ。計時中 `join` なし。F3b も `submit` と `try_receive_latest` のみで、計時中 blocking receive なし。F3b queue失効は元理由を保存して試行不合格。Running／Ready 中は proposal を延期し、target/RNGを保持、経過時間は最初の有効 gate に一度だけ適用し残量0。Ready到着後の**次の** hop gateで、同時点までの F3b 最新受理 batch と共有 habituation 現版で score 表を作り、実 Voice の失効検査で採用する。F3b batchが届かない gate は Ready を保持し、数値採点しない。消費後の新要求は commit 後の live Voice とその時点の Capture 世代から予約する。過去のRNGを書き戻さない。

`Instant` で予定hop開始時刻を `start + hop*512/48000` に固定し、予定前なら sleep。`start_jitter` は実開始と予定開始の差。`deadline_miss` は各 hop の処理完了が**その hop の予定開始 + 10.667 ms**を越えた件数とする。両者を別集計する。source別の要求・実thread完了時刻と所要時間・測定窓内完了件数／join後回収件数・Ready機会・実数値消費・score利用・失効理由・F3b受信/失効・最大保持、全体のdeadline miss、最大/平均start jitterをJSONへ出す。測定中のログ表示なし。残 thread の join と最終回収は938 hop終了後のみ。

成功条件は各sourceの `actual_body_consume > 0`、全sourceの失効理由別coverage（0件も明記）、最大保持がsourceごと2以下・全体8以下、全完了時刻の記録、deadline miss 0。source別正の消費があってもdeadline missがあれば「実時間成功」と呼ばない。固定身体の有効評価率と通常runtime全体の有効評価率を分ける。測定結果を見て設定変更やbusy時の再測定からbest選択をしない。
