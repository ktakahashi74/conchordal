# 身体評価 ON の通常 runtime を対象とする paced 取得前登録

日付: 2026-09-27。状態: Drone改訂版の取得完了。結果は[取得結果](body-fitness-runtime-live-results-20260927.md)を参照。対象は隔離 worktree `.worktrees/body-fitness-ecology` の通常 `wire_runtime`、`worker_loop`、source 除去観測、持続候補準備 worker、PitchController 消費である。既存 F3e live-paced 試験は構成部品の試験であり、この登録の全 runtime 取得とは区別する。

## 入力と経路

release の ignored crate 内試験を専有時間枠で一度だけ実行する。固定 seed 7、48,000 Hz、512 sample/hop、1 声または 4 声、すべて Sine。配置周波数は順に 220、330、440、550 Hz。各 Voice は `line(f, f)` により指定周波数へ配置し、`drone`、habitat と presentation の両 bus、amp 0.06、sustain、設定値としての endurance 30 秒、sustain_drive 0.2、ADSR (0.005, 0, 1, 0.2)、`seek_consonance()`、local/non-ratio hill-climb、global peaks 0、ratio candidates 0、temperature 0、proposal interval 0.05 秒とする。Drone の生存には endurance 設定を使用せず、scenario の Finish まで生きる。開始時に全声を配置し、`wait(10.0)` で終了する。同じ scenario と設定で `body_fitness_action=false` / `true` を各声数につき一回ずつ実行する。`true` は source 除去観測も有効にする。比較時は listener/report/profile の設定も揃える。

既存の private `wire_runtime(WiringOptions)` を直接呼ぶ。`deterministic_analysis=false`、`deterministic_footprints=false`、`start_playing=true`、`wav_tx=None`、device handle なし。presentation の `audio_prod` は test 内の容量 512 sample の `ringbuf` producer に接続する。別 test consumer は最初の一 hop が満たされるまで待ち、その時刻を pacing 原点とする。その後、512/48,000 秒ごとに最大 512 sample を取り出して破棄する。production `worker_loop` の `audio_prod=Some` 分岐をそのまま通し、音声 callback、スピーカー、WAV writer は使用しない。これは production runtime の simulated sink 付き pacing であり、実 device/callback の保証ではない。

時刻の主観測窓は frame 0..937 の 938 hop とする。frame 0 の開始 sample は 0、frame 937 の開始 sample は 479,744。`wait(10.0)` の Finish は最初の該当 hop（frame 938、開始 sample 480,256）で配送される。音の release tail と worker join は合わせて主観測窓の後として集計し、主観測窓へ混ぜない。実際の終了 frame と総hop数も保存し、想定差があれば失敗として記録する。4本の候補 worker の生成は `wire_runtime` の費用に含める。source 除去 worker の生成は初回 `observe` の中で行われ、frame 0 の hop profile に含まれる。prefill は音声を ringbuf へ push した時点で成立するため、frame 0 の `process_hop` 完了とは同義ではない。配線時間、配線後の prefill 待ち、frame 0 の hop 費用を別々に保存する。

## 取得項目

consumer 側は pacing 原点からの各予定時刻、実際の起床時刻、取得 sample 数、512 未満の callback 相当 underflow、最大/平均 start jitter を記録する。初回 prefill が30秒以内に成立しない場合は停止要求と失敗を保存する。通常 runtime profile は各 `process_hop` の経過時間、analysis/listener wait、population、render、post-render、割当数、主観測窓の p50/p95/p99/最大、10.6667 ms を超えた hop 数を記録する。profile は hop 毎の行をメモリへ追加し、終了後にだけ JSON を書く。別の既存 JSONL report には `body_fitness_observation` の受理・停止理由と、`body_fitness_action` の source 別実消費、延期、拒否、完了、cache hit/miss/byte、support と判断 sample を保存する。report I/O は hop 費用に含まれるので、この取得の条件として明記する。actionの最終定期記録は主窓末尾 frame937 ではなく frame912 の値として明示する。test 内の既存 render probe は hop ごとの source 数、実 Voice の基音と target の変化、および両 bus の bit 列 digest だけをメモリに記録する。probe と report の費用は ON/OFF 両条件で揃える。

記録には scenario/config/source/test binary の SHA-256、実行 command、exit code、全 raw profile/report、集計 JSON を含める。初回 prefill、測定窓、測定窓後の release tail・join・報告書出しを合わせた壁時間を分けて記す。ON と OFF の音声 digest、声数、target 変化は比較値であり、ON/OFF が一致することを要件にしない。配線時の候補 worker 起動と各 background worker の CPU 時間は hop profile に含まれない。probe の digest 計算は hop profile に含まれ、ON/OFF で同一条件とする。

## 判定

前提は frame 0..937 で指定声数が存在し、声数上限と 48 kHz/512 境界を満たし、observer が停止せず、全 source に受理された解析があること。ON は report の `mode=realtime`、各 source の `consumed_decisions > 0`、少なくとも一件の cache hit、support age 0..4,800 sample、cache 上限 256 entry / 1 MiB を要求する。callback 相当 underflow 0、主観測窓の worker hop 超過 0、start jitter と deadline の全標本保存を実時間適合の条件とする。OFF は既定経路の基準値として同じ検査を行う。いずれか失敗しても raw を残し、条件変更や best run 選択をしない。取得後も device callback、音響装置、長時間・変動身体・出生退役一般への合格とは扱わない。

並行する cargo/render が終了するまで壁時間取得を始めない。先に harness の format/compile と機能境界だけを確認し、専有枠で release 取得する。実行時は `CONCHORDAL_BODY_FITNESS_LIVE_DIR` に新しい保存先、`CONCHORDAL_BODY_FITNESS_SOURCE_SHA256` に凍結source manifestのSHA-256を指定し、`cargo test --lib --release normal_runtime_action_paced_without_audio_device -- --ignored --nocapture` の全出力と同一shellのexit codeを保存する。

## 取得前改訂

壁時間取得前に、配置を `at(f)` から `line(f, f)` へ変更した。`at(f)` は配置時に `PitchMode::Lock` を設定する。`seek_consonance()` の提案を計算しても、PitchController の同じ hop の Lock 分岐が target を固定周波数へ戻す。旧登録と harness は `target/ecology-validation/live-prereg-at/` に保存した。改訂版は scenario 読込直後、全 Spawn の `PitchMode::Free` と固定幅ゼロの Linear 配置を検査し、意図しない Lock を取得前に失敗させる。その他の取得条件と判定閾値は変更しない。旧版で壁時間は一度も取得していない。

初回専有取得の raw は `target/runtime-live-v12-20260927/raw/` に保存した。4 case すべて 938 hop、consumer underflow 0、予算超過 0 だが、全 Voice が frame 94 で消え、frame 912 の action report は source 空であった。各 population の `population_step` も 0.992 秒で alive_count 0。原因は `seq` の既定 duration 1.0 秒であり、`.sustain().endurance(30.0)` は Seq の寿命を延ばさない。取得前の fixture 改訂として `brain("seq")` を `brain("drone")` に変更する。Drone の articulation は生存継続し、scenario Finish が退役を決める。旧版の harness と登録は `target/ecology-validation/live-prereg-seq-v12/` に保存した。新 harness は scenario 読込時に全 Spawn の Drone articulation を検査する。全窓生存、source 消費、underflow、hop 費用などの合否条件は緩めない。この初回取得を成功例とは扱わない。
