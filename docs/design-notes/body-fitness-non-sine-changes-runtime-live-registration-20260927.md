# 非 Sine 身体・control 変更の通常 runtime paced 取得前登録

日付: 2026-09-27。状態: 取得前。対象は隔離 worktree `.worktrees/body-fitness-metabolism` の通常 `wire_runtime` / `worker_loop` である。第十二版の Sine/Drone 取得とは別の ignored release 試験であり、その再取得をこの試験の結果として数えない。この試験は非 Sine 身体・control の変更時にも既存身体評価の消費が続くかを測る。新しい代謝 provider の試験ではない。

## 固定入力と経路

48,000 Hz、512 sample/hop、seed 17、10 秒。Harmonic Voice を `line(220, 220)`、Modal Voice を `line(330, 330)` で frame 0 に配置する。両 Voice は `brain("drone")`、`PitchMode::Free`、`seek_consonance()`、habitat/presentation 両 bus、amp 0.06、sustain、設定値として endurance 30 秒、sustain_drive 0.2、ADSR (0.005, 0, 1, 0.2)、proposal interval 0.05 秒、move_cost 0、temperature 1、glide 0.04 秒、global peaks 0、ratio candidates 0 とする。開始 brightness は Harmonic 0.30、Modal 0.60。各 Spawn の身体種別、Drone、Free、周波数を scenario 読込直後に検査する。

`wait(3.2)` の後、frame 300 で Harmonic の brightness を 0.85 に変更する。さらに `wait(3.2)` の後、frame 600 で Harmonic の landscape_weight を 0.0 に変更する。最後に `wait(3.6)` で 10 秒に達し、Finish は frame 938 で配送する。主観測窓は frame 0..937 の 938 hop。frame 288 を身体変更前、432 を身体変更後、576 を control 変更直前、720 を control 変更後、912 を終点近傍の定期報告として扱う。Scenario IR に2つの UpdatePopulation と Finish の時刻・対象を事前検査し、イベント時刻と実際の配送 frame は raw report と scenario に残す。

`body_fitness_action=false` と `true` の二条件を一回ずつ取得する。この取得の対象外である通常 metabolism flag `body_fitness_metabolism_offline` は両条件で OFF とし、設定ファイルに値を保存する。既存 `wire_runtime(WiringOptions)` をそのまま使い、`deterministic_analysis=false`、`deterministic_footprints=false`、`audio_prod=Some` に容量512の ringbuffer producer を接続する。別 test consumer が 512/48,000 秒周期で取り出して音を破棄する。`wav_tx=None`、audio device/callback なし。report と hop profile は両条件で有効。候補 worker 起動は配線時間、source 除去 worker 起動は初回 hop profile に含む。report と probe の費用は hop profile に含み、background worker の CPU 時間は含まない。

## 保存と判定

各条件で scenario、事前検査した Scenario IR の debug 表示 `scenario-ir.txt`、config、全 raw JSONL report/profile、集計、音声 bit digest、source/test binary SHA-256、実行 command と exit code を保存する。配線、prefill、主観測窓、後処理の壁時間を分ける。consumer の各予定時刻・実起床時刻・取得 sample 数・jitter、各 hop 費用の p50/p95/p99/最大を記録する。実際の source 別 `consumed_decisions`、body generation、`last_consumed` の support/decision sample と score_uses、cache entry/byte/hit、拒否理由も保存する。終点近傍の action report は frame 912 の定期標本であり、frame 937 の総数ではない。

両条件で全 938 hop に2声が存在し、consumer underflow 0、主窓 worker hop の 10.6667 ms 超過 0 を要求する。ON では realtime mode、observer の受理と非停止、両 source の実消費 >0、cache hit >0、各定期報告に現れる `last_consumed` の support age 0..4,800 sample、score_uses >0、cache 256 entry / 1 MiB 以下を要求する。frame 288 と 432 の比較で Harmonic の body generation 増加、更新後世代の消費、Modal の世代不変を要求する。frame 576 と 720 の比較で Harmonic の control 更新後の再消費、frame 912 までの継続消費を要求する。主窓全体で両 Voice の target と実基音が各1回以上変化することを要求する。拒否理由 `Consume(ControlChanged)` は準備完了と更新 gate の相対時刻に依存するため raw に記録し、合否の必須条件にはしない。この負条件は既存の内部境界試験が検証する。ON/OFF の bit digest や軌跡の一致も要求しない。

初回 prefill が30秒以内に成立しない、worker が終了する、または例外が出る場合、明示的な失敗を保存する。失敗後に条件や閾値を変えて同取得の成功例としない。第十二版の Sine 取得と同様、これは production runtime の simulated sink による専有 paced 取得であり、実 audio device、可聴品質、他の身体・生態・長時間への保証ではない。壁時間取得は他の cargo/render が停止した専有枠でのみ行う。

実行時は `CARGO_TARGET_DIR=/home/shafi/lwrk/conchordal/target/body-fitness-metabolism-build-20260927`、`CONCHORDAL_BODY_FITNESS_NON_SINE_CHANGES_DIR` に新規保存先、`CONCHORDAL_BODY_FITNESS_SOURCE_SHA256` に凍結 source manifest のSHA-256を指定する。対象は `cargo test --lib --release normal_runtime_non_sine_changes_paced_without_audio_device -- --ignored --nocapture`。全出力と同一 shell で得た exit code を保存する。取得前の no-run compile は壁時間結果として扱わない。
