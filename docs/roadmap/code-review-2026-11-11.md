# コード全体レビューと改善案（2026-11-11）

`main` の `cb7c29d6` を対象に、crate全体（`src/` 約16.7万行、`crates/conchordal-web/`、`gui/src/`）を読み取りのみでレビューした結果と改善案を記録する。
実装には着手していない。本文の行番号はすべて `cb7c29d6` 時点のものである。

<a id="scope"></a>

## 範囲と方法

- 6分野に分けてレビュー担当のエージェント（Fable 5.2）を並列に走らせ、統括側で結果を統合した。
  分野は `src/core/`、`src/life/` のVoice・Community・pitch部、`src/life/` のFlora・phonation・音響部（`src/synth/` を含む）、`src/temporal_cognition/` と `src/listener_twin/`、`src/scripting/`・`src/scenario*`・`src/config.rs`・`src/cli.rs`・`src/bin/`、`src/runtime/`・`src/audio/`・`src/ui/`・web側である。
- 各担当には `CLAUDE.md` の契約（Log2Space の `_scan` 不変条件、SIMD のbit同一性、hop経路の割当予算、`core/float.rs` の共有helper、ゼロ重みでの乱数消費不変、Air-Gap、`pub(crate)` 方針）を照合対象として渡した。
- 機械的な検査は2点だけ実行した。`cargo clippy --all-targets -- -D warnings` は警告0で通過した。`cargo check --lib --features loo-diagnostic` も通過した。
- `cargo test` は実行していない。どの指摘も実行による再現はしていない。
- 精読の範囲外: `tests/`、`scripts/`、`web/`、`src/ui/` の描画本体、`gui/replay`、`src/life/body_metabolism.rs`、`src/life/report.rs`、`src/life/support_action.rs`（後の3つは流し読み）。

<a id="legend"></a>

### 確認状況の表記

- **確認済**: 統括側が該当行を読み、欠陥の機構を確認した。
- **報告**: レビュー担当の読解だけに基づく。統括側では再確認していない。
- **数値未実行**: 利得・誤差などの数値は担当の机上計算か、Pythonでのf32模擬による。crate自体は実行していない。

<a id="order"></a>

## 推奨する着手順

1. [A. 実害のある不具合](#a)の11件を、それぞれ回帰テスト付きで直す。各件とも数行から数十行の局所修正で済む。
2. [B. 作者判断が要る項目](#b)は、該当箇所に手を入れるときに決める。A群の着手には不要である。
3. [C. 読者のない計算と実時間負荷](#c)のうち、読者がないことをgrepで確認できたもの（C1–C3、C6）を削除する。
   読者がないことはテストで担保できるので、削除の前提として計測は置かない。効果を記録したい場合は既存のruntime profileを使う。
4. [E1](#e1)（CIでのwasm32ビルド）は小さい変更で済み、A10と `step.rs` 経路の無検査状態を解消するので、早めに入れる。
5. [D. 構造の改善](#d)は `pitch_core.rs` と `scripting/engine.rs` から始める。

---

<a id="a"></a>

## A. 実害のある不具合

<a id="a1"></a>

### A1. HV-9 が既定構成のVoiceで無言のまま働かない（確認済）

- 場所: `src/life/community.rs:1006-1012`
- 欠陥: `update_loo_ownership` は内部にtemporal重み用の分岐（`:917` 以降）を持つが、呼出条件が `leave_self_out || selection_environment` だけである。
  Voiceの `loo_input.relations` を更新する経路は、この呼出と `harmonic_rhythm` 時の `:548` しかない。
  ```rust
  if self.voices.iter().any(|v| v.effective_control.pitch.leave_self_out || v.selection_environment) {
      self.update_loo_ownership(landscape);
  }
  ```
- 失敗シナリオ: 全Voiceが既定の `leave_self_out(false)` で、`harmonic_rhythm` を使わず、`follow_harmonic_relations(0.8)` を指定する。
  - `relations` が None のまま残り、`pitch_core.rs:731, 892` のHV-9項が警告なしに働かない。
  - `pitch_controller.rs:429` は既定値の `relation_source`（id 0）で `record_opportunity` を呼び、全Voiceの機会がid 0に帰属する。worker側への影響は未確認。
  - 無関係なPopulationの1体を `leave_self_out(true)` にすると、全Voiceで有効になる。
  - `:975-984` の「support movement requires leave_self_out」警告も同じ条件でしか出ない。
- 経緯: 条件行の最終変更は10/22で、HV-9の追加（`e4dc467c`、11/02）のときに更新されなかった。
- テストの穴: 既存テストは `update_loo_ownership` を直接呼ぶ。HV-9のfixtureはすべてLOO Voiceか `harmonic_rhythm` を併用している。
- 修正案: 無条件で呼ぶ（担当によれば `loo_observation_delivery_steady_state_allocates_nothing` で割当0を確認済）。
  または条件に「temporal重み ≠ 0 または `relations.is_some()`」を加える（後者は重みが0に戻ったときの反映に必要）。
  全既定Voice＋temporal重みのCommunityテストを追加する。

<a id="a2"></a>

### A2. Floraでゼロ重みのbypassが漏れる（確認済）

- 場所: `src/life/flora.rs:1027`（分岐条件は `:981`）
- 欠陥: 両重みが0に戻ると、Floraは `Use.weight` だけを消す。`continuation` と `decision` が残り、`Use::active()` が true のままになる。
  ```rust
  } else if let Some(active) = &mut self.harmonic_use {
      active.weight = 0.;
  }
  ```
  対照として、`temporal_cognition/harmonic_relations/consumer.rs:83-91` の `Use::refresh` は両重み0で `weight`・`continuation`・`decision` をすべて消す。Voiceは `community.rs:951` で必ずrefreshを通り、spawnは呼出側で遮断される。漏れているのはFloraだけである。
- 失敗シナリオ: `follow_harmonic_continuation(0.5)` の後で両方を0にする。その後も古いDecisionで次の処理が動く。
  - `flora_metric.rs:299-303` の `allow_now` が `harmonic_act_now=false` を返し、`flora.rs:1439` でhold、`flora.rs:1622` でbirthを抑止する。
  - `flora.rs:1307` の `action_difference` と、`flora.rs:1447/1557/1629` の `choose_scored`（preview計算を伴う）。
  - 継続期間は `expires_at`（最古anchor＋600秒）までである。
- `CLAUDE.md` の「Both zero bypass prediction/preview/new draws」に反する。テスト（`runtime.rs:1193`）は `weight` だけの経路しか見ていない。
- 修正案: else分岐を削除し、`Use::refresh` を無条件で通す（冒頭が両重み0を処理する）。または `Use::disable()` を追加する。

<a id="a3"></a>

### A3. `parallel` の中で外側handleに出したpatch・releaseが誤った時刻で発行される（確認済）

- 場所: `src/scripting/engine.rs:1713-1724`、`src/scripting/context.rs:592, 632, 646, 1285`
- 欠陥: branch終了時の `pop_scope` → `collect_actions(Some(depth))` は、depthが一致しないPopulationを飛ばす。外側handleのpendingが残ったまま、次のbranchで `cursor = start_time` に巻き戻る。最初の `wait` の `emit_pending_actions` が、巻き戻った時刻で全pendingを発行する。
- 失敗シナリオ: `let p = place(...); parallel([|| { wait(4); p.amp(0.4); }, || { wait(1); }]);` で、`UpdatePopulation` がt=4ではなくt=0に出る。`release(p)` も同様に4秒早く解放される。最後のbranchが最長でない場合は、逆に `max_end` まで遅れる。
- 文書との不一致: `docs.rs:373, 550, 627` は「Live patches emit at wait, flush or scope end」とするが、外側handleはscope終了時に出ない。
- 修正案: branch終了時、`pop_scope` の前に `ctx.emit_pending_actions(pos)` を呼ぶ。回帰テストは上の台本でevent時刻が4になることを確かめる。

<a id="a4"></a>

### A4. Modalの Nyquist超modeが0.49·fsに折り畳まれて大きな利得で鳴る（機構は確認済、利得は数値未実行）

- 場所: `src/life/sound/modal_engine.rs:400-415`（`ModeShape::build_modes`）、`src/synth/modes.rs:69`（`compile_mode`）
- 欠陥: `pitch_hz * ratio` がNyquistを超えてもmuteせず、`compile_mode` が周波数だけを `0.49·fs` にclampし、gain・in_gainはそのまま使う。
- 担当の試算: MCFの出力ピークは b1/cos(πf/fs) で、1 kHzで×1.0、12.5 kHzで×1.46、20 kHzで×3.86、0.49·fsで×29.6。
  既定の `modal()`（harmonic 16 modes、brightness 0.6）、fs 48 kHzで、23.52 kHz成分は帯域内RMS比でpitch 2 kHzのとき+4.9 dB、4 kHzで+14.6 dB、8 kHzで+18 dB（44.1 kHzでも+3.5/+12/+17 dB）。
- 影響: 不可聴成分がlimiter・meter・WAVのheadroomを占める。Voice音域の上限8 kHzまで動く `seek_consonance()` のmodal Voiceで到達する。
- 同根の問題: `render_prototype=true` のとき、`modal_engine.rs:103-106` の `assert!(self.phase3_residual_bound().is_finite())` が、clampされたmodeとt60 ≥ 約1.3秒で発火し、workerがpanicする（担当の再計算で1.1秒は有限、1.3秒はINF）。
- 対照: OscillatorBankはcullする（`oscillator_bank.rs:465-476`）。Rainは0.45·fs以上をmuteする（`rain.rs:591-603`、`fill_object_modes`）。未対応はModalだけで、テストもない。
- 修正案: `build_modes` と `bank_forecast`（`modal_engine.rs:185-198`）で、`freq_hz >= 0.45*fs` のmodeを `gain=0, in_gain=0` にする（Rainと同じ規則）。
  帯域内の 1/cos 利得の補正は音色の変更になるため、作者判断として切り離す。

<a id="a5"></a>

### A5. NaNが `f32::clamp` を素通りしてengineに届く（amp経路と登録側は確認済、他は報告）

- 場所: `src/scenario/control.rs:110`（`set_amp_clamped`）、`src/scripting/engine.rs:2461, 2473`（`meter_stability`）ほか
- 欠陥: `f32::clamp` はNaNをそのまま返す。Rhaiの浮動小数除算は無検査で、`0.0/0.0` で台本からNaNを作れる。数値builderの登録（`engine.rs:2844` の `register_population_spec_numeric_overloads`）にも有限性の検査がない。
- 確認した経路:
  - `amp(NaN)` → `self.body.amp = amp.clamp(0.0, 1.0)` → `SineBody.amp`（`life/sound_body.rs:659`）→ 標本に加算（同 `:115`）→ NaNのPCM。初期specとlive patchの両方。render経路に `is_finite` のguardはない。
  - `meter_stability(NaN)` → `core/meter.rs:521` の `force_gain = 1.0 + NaN` → `beat_r` と全confidenceがNaNになり、rhythmとFlora meterへ伝わる（`core/meter.rs:515, 849` も同じclamp）。
- 同型の箇所（下流は未確認）: `control.rs:115`（`set_timbre_brightness_clamped`）、`spec.rs:751`（`set_entrainment`）、`:882`（`set_pulse_lock`）、`:890`（`set_social`）、`:900`（`set_duration_range`）、`:982`（`set_adsr` のsustain）、`:1019`（`set_respawn_min_c_level`）、`scenario.rs:458, 460, 471`（`CoupledTimingSpec::sanitized`）。`set_duration_curve`（`spec.rs:906`）は検証自体がない。
- 修正案: `core::float::sanitize01` に置き換える（`CLAUDE.md` の「helperを書き直さない」規則とも合う）。`meter_stability` は `MeterNetwork::set_shaping` で一度だけsanitizeし、台本側でも警告を出す。
- 補足: `freq(NaN)` は `pitch_controller.rs:548` でguardされていて無害である。

<a id="a6"></a>

### A6. `ModePattern.count(-1)` がworkerのpanicになる（確認済）

- 場所: `src/scripting/engine.rs:1769-1771`
- 欠陥: `pattern.with_count((n as usize).max(1))` で負値が `usize::MAX` に化ける。spawn時に `Vec::with_capacity(count)`（`core/mode_pattern.rs:465, 489, 514`）でcapacity overflowのpanicになる。台本エラーではなくworker側のpanicである。Rainの `impact_modes` / `object_modes` も同じ経路を通る（`life/rain.rs:166, 190`）。上限もないので巨大な値はOOMになる。
- 修正案: `n < 1` と上限超過を位置付きの台本エラーにする（Rainの `objects` と同じ方式、`engine.rs:420`）。

<a id="a7"></a>

### A7. `remember_pitch_set` の48 kHz検査が実際の標本化率を見ない（確認済）

- 場所: `src/runtime/scenario_load.rs:274`（検査）、`src/core/stream/analysis.rs:89`（`assert_eq!(self.params.fs, 48000.)`）、`src/runtime/wiring.rs:464-466, 866`
- 欠陥: 検査は `load_scenario` だけにあり、対象は設定ファイルの `audio.sample_rate` である。assertは実行時の標本化率に対してかかる。
- 失敗シナリオ:
  - `--play` で48 kHz非対応の機器（44.1 kHz固定など）→ `init_runtime` 内でassertがpanicし、位置付きのエラーにならない。
  - 設定が44.1 kHzで機器が48 kHz → 動くのに拒否される。
  - `StepEngine::new(.., 44100)` と `remember_pitch_set(w>0)` → wasmがtrapする。同梱のページは `AudioContext({sampleRate: 48000})` なので現状は発火しない。
- 不整合: TCは同じ前提を警告に落としてskipする（`wiring.rs:467`）。
- 修正案: 検査を `prepare_runtime` に移して位置付きの `Err` を返し、`load_scenario` からは削除する。`StepEngine` も同じ経路でErrを返す。TCと同じ警告＋skipにする選択肢もある。

<a id="a8"></a>

### A8. headlessの自然終了でringに残った音が再生されない（確認済）

- 場所: `src/runtime/wiring.rs:1016-1040`、`src/runtime/worker.rs:394-402, 946-959`、`src/audio/output.rs:396-402`
- 欠陥: workerは最終hopをpushした直後に `exiting` を立ててreturnする。`run_headless` はringが空になるのを待たずに `rt` をdropし、cpal streamが止まる。ringの排出を待つコードはない。
- 失敗シナリオ: `--nogui` では `wait_user_exit` が強制的にfalseになる。ring容量は `2 × latency_ms`（既定で約100 ms）で、workerは常にほぼ満杯まで詰める。終了の検出は50 msごとのpoll。末尾の約40–100 msのrelease tailやfadeが毎回切れる。GUIは `watch_for_close` の200 msのsleepで偶然排出される。終端が無音なら実害はない。
- 修正案: 自然終了（`state.finished` かつ failuresが空）のとき、`exiting` を立てる前に、worker_loopで `prod.occupied_len() == 0` まで待つ。待ち時間の上限はring時間の2倍とし、`recovering` と強制停止で中断する。`exiting` はstop flagと共用なので、待機はstoreの前に置く。

<a id="a9"></a>

### A9. listener更新のblocking送信でdeadlockし得る（送信と容量は確認済、到達timingは報告）

- 場所: `src/runtime/analysis_sync.rs:452-459`、`src/core/analysis_worker.rs:279-314`、`src/runtime/wiring.rs:534-536`、`src/runtime/worker.rs:746-752`
- 欠陥: 実時間経路でworkerは `listener_analysis_update_tx.send(update)`（容量8）をblockingで呼ぶ。listener threadはOrderedで `result_tx.send`（容量4）をblockingで呼び、更新の受け取りはbatchの先頭だけで行う。workerがresultを読むのは更新を送った後である（`apply_landscape_updates` → `wait_for_listener`）。
- 失敗シナリオ: listener解析がhop予算を超え（backlogが64に張り付き、1 batchが0.7秒以上）、その間に台本が9 hop以上続けてlandscape更新を出す（例: `set_roughness_aversion` を `wait(0.05)` 刻みでramp）。workerが9個目でblockし、listenerが5個目のresultでblockして相互に待つ。workerはこの間 `exiting` を見ないので、ExitもCtrl-Cも効かず、`App::drop` のjoinが永久に待つ。
- 修正案: 実時間経路では `try_send` にする。Fullのときはworker側にmerge済みの `pending_listener_update` を保持して次のhopで再送する（mergeは `life/community/actions.rs:421` に既存）。field解析側は `Latest` publishがnon-blockingなので現状安全である。

<a id="a10"></a>

### A10. wasm32でIPEMの標本位置が約609秒で桁あふれする（算術は確認済、wasm実行は未実施）

- 場所: `src/core/tonal_context/mod.rs:70-79`、`src/core/tonal_context/pitch_set.rs:444`、`src/core/tonal_context/decoder.rs:127-128`
- 欠陥: 標本位置の積（`self.next * self.q`、p=147、q=320 など）を `usize` で計算している。wasm32では32 bitである。
- 失敗シナリオ: 増加率は 48000×147 = 7,056,000/秒で、2^32に608.7秒で達する。wasmのreleaseにはoverflow checkがない（`Cargo.toml` はdevだけ）ので無言でwrapする。`right >= self.count` が成り立たなくなり、入力1標本ごとに位置のずれた出力が出る。`Tracker::process` は `samples = 0` になりPS学習が止まる。native（64 bit）には影響しない。`remember_pitch_set` はopt-inのExperimentalである。
- 修正案: `count` / `next` / `n0` / `frame.index` と派生する積を `u64` にする。またはresamplerの位相を有界な剰余で持つ。

<a id="a11"></a>

### A11. `analysis.nfft` が1〜7だと検証を通った後に起動時panicする（確認済）

- 場所: `src/config.rs:642-645`（`(1..=1 << 18).contains(&self.analysis.nfft)`）、`src/core/nsgt_kernel.rs:147`（`assert!(len >= 8, ...)`）
- 修正案: 設定検証の下限を8にする。

<a id="a-low"></a>

### その他の不具合（低優先度）

Voice・Community・pitch:

- `src/life/articulation_core.rs:596-607`: `while self.rhythm_phase >= 2.0 * PI` に回数の上限がない。`set_global_coupling` は台本側で無検証（`engine.rs:2529-2540`）、`actions.rs:136` も `value.max(0.0)` だけである。coupling 約1e10以上で `rhythm_phase` が2^27を超え、f32の減算が吸収されて無限ループになる（報告）。修正案: 1 substepのwrap回数に上限を設け、非有限は0に戻し、couplingを飽和上限でclampする。
- `src/life/voice.rs:2161-2171`: 新しいtoneは `continuous_drive` 0で始まり、次のhop頭の `ToneCmd::Update` まで `sustain_drive` が届かない（報告、可聴性は未確認）。修正案: `ToneSpec` に `continuous_drive` を持たせる。
- `src/life/community.rs:570-591`: `population_sway` が `cleanup_dead`（`:1422-1431`）で剪定されない。endlessの場面でsway付きPopulationが入れ替わると単調に増え、死んだPopulationの1/f過程も毎hop進む（報告）。修正案: `population_sway.retain(|id, _| self.populations.contains_key(id))`。
- `src/life/pitch_core.rs:1339-1344, 2754-2780`: PeakSamplerは `DEFAULT_MOVE_COST_*` に固定され、`move_cost*`・`global_peak*`・`ratio_candidate*` を警告なしに無視する（報告）。修正案: 反映するか、非対応のcontrolに警告を出す。

Flora・音響:

- `src/life/modal.rs:153-166`、`src/life/sound/modal_engine.rs:309-311, 334`: ModalBodyのspectral self-reportは64回に1回しかpitchに同期しない。移動後のlive patchでHV-3のtemplateが出生時のpitchで作られる（報告、support重み≠0のときだけ）。修正案: `project_spectral` に現在のpitchを渡し、ダミーのrenderを削除する。
- `src/life/rain.rs:422-428`: slotの奪取にfadeがない。同時発音の需要が64の2倍を超えると、−40 dB以上の残響が急に切れる（報告、可聴性は未確認）。

core・解析:

- `src/core/harmonicity_kernel.rs:358-361`: `accumulate_shifted` の境界がoff-by-oneで、`h_pot_scan` の端のbinが小さくなる。production gridでは約20 Hzと12.5 kHzの端だけでVoice音域の外である（報告）。参照実装（`:408-411`）とlevel別テストも合わせて直す。
- `src/temporal_cognition/context.rs:244`: `arrival_probability` はhorizonがちょうど `rate` 標本（1秒）のforecastでしか埋まらない。設定例は0.1秒なので常にNoneになる。読み手は `viewdata_web.rs:137` だけである（報告）。
- `src/listener_twin/prediction.rs:157, 209-217`: `shape_var` がf32の大きな数同士の差で、ほぼ静的なtextureでは0に落ち、coverageが0になる。影響はUIとreportだけ（報告、数値未実行）。修正案: f64化と周期的な再計算。

runtime・audio・web:

- `src/runtime/worker.rs:515-521, 986-988`: `finish_run` の `wait_for_analysis` が最終hopを待たない。最終hopのsupportがsummaryに入るかがthreadのtiming次第になる（報告）。
- `src/runtime/worker.rs:184-189, 384-386`: `exit_during_recovery` が、終了後の待機中（`wait_user_exit`）にも発火する。機器の抜き差しでGUIが即終了する（報告）。修正案: 条件に `&& !self.finished`。
- `src/audio/output.rs:208-214, 297`、`src/audio/recovery.rs:51-62`: defer_start時のrecovery基準時刻がstream構築時のままで、tonalの場面ではDecoder構築を待つ間に開始直後のstreamを破棄・再構築し得る（報告、timing依存）。修正案: `start_prepared` で `Recovery::new(Instant::now())`。
- `src/runtime/worker.rs:833, 923-944`: L/R meterはUI frameを送ったhopのpeakだけを使う。約4 hopに1回しか反映されず、疎な衝撃音で小さく表示される（報告）。修正案: 前回の送信成功以降の最大値を保持する。`monitor.rs:101-113` のheadroom警告がbus値基準である点も同時に見直す。
- `src/app.rs:141, 227-235, 299-318`: watchdogの強制終了はjoin結果がOkでも常に `exit(1)`。`runtime_error` の赤いlabelは表示される前にCloseされるので実質的に死んでいる（報告）。
- `src/audio/output.rs:36-48`、`src/runtime/worker.rs:435-446`: ringに常時 `latency_ms` の約2倍を溜める。GUIのstrobeやplayheadが音より約90 ms先行する。`latency_ms` ≤ 約5.3では容量がhopと等しくなり毎hop underrunするが、検証は通る（報告）。修正案: 充填目標を `target_frames` にし、検証で容量 ≥ 2 hopを要求する。
- `src/audio/limiter.rs:181-186`: `LimiterMode::None` は非有限値を素通しする（報告）。修正案: Noneでも `finite_or` だけは適用する。
- `crates/conchordal-web/www/player.js:54-57`、`gui/src/frame.rs:285-300`: serde_jsonは非有限floatを `null` で出し、非Optionの欄でdecodeが失敗すると演奏ごと停止する。nativeはlimiterで続行する（報告、NaNの発生源は未特定）。修正案: snapshotの例外でframeを捨てる、または `web_snapshot()` で非有限を0に置き換える。
- `crates/conchordal-web/www/audio-worklet.js:59-107`: `process()` 内で毎quantum `subarray()`・`splice()`・report objectを作り、render threadのGC停止要因になる（報告）。修正案: 固定長のFloat32Array ringにする。
- 未確認: `StepEngine` は `follow_tonal_context` を拒否も警告もしない。tableの構築コスト（`CLAUDE.md` では約1 GB/80秒）がwasmでどうなるかは確かめていない。

scripting・CLI:

- `src/cli.rs:7-8, 23-24`: `--play` と `--wait-user-start` が `num_args = 0..=1` で `require_equals` がない。`conchordal --wait-user-start scene.rhai` で台本パスをbool値として消費してparse errorになる（clapの仕様からの推論、未実行）。`--wait-user-exit` は対処済（`cli.rs:19, 96`）。修正案: 同じく `require_equals = true` とテスト。
- `src/scripting/engine.rs:2675-2731`: `follow_harmonic_relations/continuation` の範囲外・非有限値と、`set_support_*_preference` の非有限・負のnoveltyに台本側の診断がない。runtime（`life/community/actions.rs:141-173`）が位置なしの `tracing::warn!` で捨てるだけで、`--compile-only` はWarnings 0になる（報告）。`dwell` などと同じく位置付き警告にする。
- `src/scripting/placement.rs:101, 140`: `at()` と `line()` は無検証で、`at(0.0/0.0)` や `line(-100, 0)` が無警告で1 HzのVoiceになる。field配置の境界はhard error（`context.rs:855-876`）なので扱いが揃っていない（報告）。
- `src/scripting/score.rs:235`: endlessで `harmonic_rhythm` が後からpatchを拒否するとき、警告が `Position::NONE` になる（報告）。
- 無言の値変換: `cycles(n)` は0以下を1にし2^32以上を切り捨てる（`engine.rs:971`）。`wait` は負値とNaNを無言で0にし、`wait(1.0/0.0)` は `duration_sec = inf` のまま `validate_scenario` を通る（`context.rs:1287`）。負のseedの扱いが3通りある（`seed()` はerror、ModePatternの `seed()` は警告、`scaffold_scrambled` は無言で0、`engine.rs:2618, 2636`）（報告）。

---

<a id="b"></a>

## B. 作者判断が要る項目

A群の着手には不要で、それぞれ該当箇所に手を入れるときに決めれば足りる。

<a id="b1"></a>

### B1. attack途中に届いた通常のOffでgainが跳ぶ（確認済）

- 場所: `src/life/sound/envelope.rs:34-52`
- 欠陥: `with_release` が `hold_end` を書き換え、`attack_len()` が既に鳴った区間までさかのぼって短くなる（`self.attack_ticks.min(self.hold_end.saturating_sub(self.onset).max(1))`）。
- 失敗シナリオ: `adsr(0.6, ...)` のsustain Voiceがonsetの0.18秒後にOff（死亡、rest/gap、`shutdown_at`）になると、gainが0.30から次の標本で1.0（+10 dB）に跳び、そこからreleaseする。非Holdのtoneでも、holdが60秒の仮置き（`schedule_renderer.rs:1085-1089`）でOffがoff_tickのhopに届く（`phonation_engine.rs:786-808`）ため、音長がattackより短く、OnとOffが別のhopに入ると起きる。既定のattack 10 msでは高速なflowだけが該当する。
- 現状: 置換のOffは `with_replacement_release` で対策済である。通常のOffは既存テスト（`envelope.rs:122-124`、`original.with_release(60).gain_at(60) == 1.`）が旧法則を明示的に固定している。
- 問い: 発音開始後の通常のOffにも、置換と同じ凍結則（attackを凍結し、到達したlevelを保持してrelease）を適用するか。
- 推奨: 適用する。clickの発生源であり、置換のOffと規則が分かれている理由が見当たらない。
- 効くところ: 短い音と早く解放される音の音が変わる。固定テストの更新と、`tone_energy` のrelease予測の同則化が要る。

<a id="b2"></a>

### B2. HV-9のtransformed mappingが実データでほとんど働かない（報告、数値未実行）

- 場所: `src/temporal_cognition/harmonic_relations/matching.rs:754-765, 1008`、`src/temporal_cognition/harmonic_relations/adapter.rs:100`、`src/life/schedule_renderer.rs:117`
- 欠陥: transformed mapping（移調・時間伸縮）は、誤差0と宣言された値にbit一致を求める（`if error == 0. { self.conflict |= self.exact.is_some_and(|old| old != value); ... }`）。productionのproducerは誤差0を宣言するが（`timing_error: 0.`、`Value::exact(f64::from(state.1.log2()))`）、値には丸めが乗る。
- 連鎖: `time_fit.push((dx/dy).ln(), 0, ..)` が厳密な経路に入る。`dx = fl(s1/sr) - fl(s0/sr)` は同じ標本差でもbitが一致しないので、3個目のanchorで `conflict` になり、`time_scale` が無効化され、`matching.rs:1008` で `compatibility_weight = 0` になる。pitchもf32のlog2の丸めで、静止pitch以外は2組目でconflictする。
- 結果: 実データでtransformedの仮説が生き残るのは、長さ2でpitchが静止しているときだけになる。source seedの半分（`matching.rs:434`）が空振りする。テスト（`tests.rs:379, 501`）は長さ2と整数値しか見ていない。同じ根を持つものとして、`derived.rs:168` のcategorical arrivalの時刻envelopeがある。
- 問い: 修正に進むか。
- 推奨: まず実際のproducerの出力を使った再現テストで不作動を確かめる。確かめられたら、時間は整数の標本差から `dx/dy` を作り、pitchはf32のlog2の表現誤差を `Value.error` に載せる（adapterはsupportについて既に丸めを載せている、`adapter.rs:73, 113`）。
- 効くところ: 直すと、生き残る仮説の集合が変わり、HV-9の振る舞いが変わる（Author ruleの試聴に影響し得る）。

<a id="b3"></a>

### B3. log域のpitch平滑がtargetの0.4〜0.8 cent手前で止まる（報告、数値はf32模擬）

- 場所: `src/life/sound/tone.rs:1004-1015`（`:781` のコメントは「one rounding step」とする）
- 欠陥: `α·(ln t − ln c)` が `ulp(ln c)/2` 未満になると進まない。
- 担当の模擬: 440→466.16 Hzは466.053 Hzで止まる（0.40 cent）。4000→4237.85 Hzは4235.91 Hzで止まる（0.79 cent、1.9 Hz）。
- 失敗シナリオ: 同じpeakに逆方向からglideした2つのVoiceが0.8〜1.6 centずれ、高域のunisonで1〜4 Hzのうなりが出る。
- 問い: 停止時にtargetへsnapするか。
- 推奨: snapする（`next == current && current != target` のときtargetへ）。意図しないうなりを除き、`settled_pitch_hz` とforecastの条件（`:782-783`）も簡素になる。
- 効くところ: 波形が変わる。互換性の契約はない。

<a id="b4"></a>

### B4. temporal observerとVoiceごとの `OwnSoundHistory` がconsumerの有無と無関係に常時動く（報告）

- 場所: `src/runtime/worker.rs:310-311`、`src/runtime/phases.rs:212-218`、`src/core/history_prediction.rs:289-314`
- 現状: 全runで `AcousticTemporalExpectation` を構築し、全Voiceに誕生時 `OwnSoundHistory::new` を作る。consumerはparticipation clockとreportだけである。Voice誕生ごとにworker thread上で約0.87 MBのcopyと2.97 MBの予約が起き、10 msのwindowごとに57特徴のGramを更新する。`CLAUDE.md` の割当予算に記載がない。
- コメント「Track from birth, including before a later switch into participation」から、意図的な設計である。
- 問い: participationのない場面でも誕生時から追跡を続けるか。
- 推奨: 有限の場面ではFlora meterやonset habituationと同じ台本の事前走査でgateする。endlessの場面は現状どおり開始時から確保する。最低限、`comparisons` を最初の `issue_comparison` で確保する。
- 効くところ: participationのない場面のworker負荷とメモリが減る。後からparticipationへ切り替わるVoiceの扱いを事前走査で拾えるかが実装上の確認点になる。

<a id="b5"></a>

### B5. spawnとturnoverのseedを `DefaultHasher` で導出している（報告）

- 場所: `src/life/community.rs:373-381, 407-413`
- 欠陥: stdの `DefaultHasher` はrelease間での安定を保証しない。toolchainの更新で、seedを固定したassayの配置が変わり得る。
- 問い: 安定したmixerに替えるか。
- 推奨: 既存のSplitMix形のmixer（`pairwise_split_sign`、`own_sequence::next`）に替える。ただし既存の出力が変わり、登録済みfixtureの再取得が要るので、次にfixtureを取り直す機会にまとめて行う。
- 効くところ: 単独で行うとfixtureの再取得だけが追加の作業になる。

---

<a id="c"></a>

## C. 読者のない計算と実時間負荷

コストの数値は演算数と構造体定義からの算出で、実測していない。

<a id="c1"></a>

### C1. `landscape.roughness` 用の2回目の畳み込みに読者がない（確認済）

- 場所: `src/core/stream/synchronous.rs:175-181`
- `compute_shape`（正規化した密度 → `roughness01`）とは別に、生の密度で2回目の畳み込みを毎hop行っている。production上の読み手は `runtime/analysis_sync.rs:81` のmoveだけで、UIは `roughness01`（`ui/windows.rs:888`）を読む。他はテストとscan長のassertである。
- 規模: production grid（892 bin、±4 ERB）で1回約19万tap。`AnalysisStream::process` は常に `full=true` で、LOO observerも同じ経路を通る（`stream/loo.rs:188`）。observerごとにRのコストがほぼ倍になり、`CLAUDE.md` が警告する解析threadの直列コストに直結する。
- 修正案: 呼出と `Landscape::roughness` の欄を削除するか `#[cfg(test)]` にする。`tests/log2space_scan_invariants.rs:37` と、bit比較のテスト（`stream/analysis.rs:436`、`stream/synchronous.rs:332`、`life/pitch_loo_conformance_tests.rs:425, 455`、`life/flora_heard_tests.rs:391`、`core/harmonic_support/tests.rs:430`）を更新する。

<a id="c2"></a>

### C2. habituationが毎hopでHを再投影し、`landscape.harmonicity` と重複する（コードは確認済、差分は報告）

- 場所: `src/runtime/analysis_sync.rs:153-156`、`src/core/harmonicity_kernel.rs:157-167`
- 同じframeの同じ密度から、解析側が既に `landscape.harmonicity` を算出している（`stream/synchronous.rs:160-167, 184`）。再投影は `vec!` 3本（約13.8 KB）と32回のshift-addを、新しいframeがなくても毎hop行い、listener frameごとにも行う。
- 修正案: `hab.advance_from_parts(&landscape.consonance_field_level, &landscape.harmonicity, dt)` にし、`potential_h_from_log2_spectrum` をテスト専用にする。
- 差分: 質量がeps以下の無音時、保存済みのscanは0で埋まるが、再投影は残渣をpeak 1に正規化する。数hopの減衰尾だけが異なる。bit一致は未確認。
- `CLAUDE.md` の「owned harmonicity projection」の記述も更新する。

<a id="c3"></a>

### C3. `sequence_control` を持つVoiceがいると毎hop `Landscape` を丸ごと複製する（確認済）

- 場所: `src/life/community.rs:1013-1022`
- `.any(|v| v.sequence_control.is_some()).then(|| Arc::new(landscape.clone()))`。`Landscape` はderive Cloneで、`Vec<f32>` のscanが十数本と `Log2Space`、`loo` の各Observationの `h_scan`/`r_scan` を含む。毎hop 20以上の割当（690 binで44 KB超）が起き、次のhopで解放される。tonal profile（SQとMEが既定で有効）では常に該当する。`CLAUDE.md` の割当予算に記載がなく、`community_steady_hops_reuse_...` のテストはsequence Voiceを含まない。
- 修正案: Communityがsnapshotを1つ保持し、欄ごとの `clone_from` で容量を再利用する。collect時に参照で渡せばArcは要らない。`sequence_execution.rs:726-728` のfallbackのcloneも削除する。

<a id="c4"></a>

### C4. `SpectralHistory` がreporterなしでも毎hop動く（報告）

- 場所: `src/core/analysis_worker.rs:94-100, 207-213, 232-234`、`src/core/spectral_history.rs:44-102`
- 構築条件は `stream.nsgt_power_enabled()`（GUI・tap・Floraのresourceで真）だが、snapshotの読み手はJSONLのreportだけ（`runtime/phases.rs:503-517`）である。
- 規模: 298 lane × 8 age × 4 ch × 104項 ≈ 99万積和/hop/stream。publicationごとに `Vec<[f32; 8]>` 892要素（28.5 KB）とArc。既定のGUIではecologyとlistenerの両threadで、webの `StepEngine` では単一のhop threadで2 stream分を負担する。
- 修正案: reporterの有無を `AnalysisState::new` に渡して構築条件にする。`CLAUDE.md` の「NSGT power and its spectral history are enabled ... for GUI」の文も更新する。

<a id="c5"></a>

### C5. `forecast()` の `contrast` 配列の値に読者がない（報告）

- 場所: `src/core/temporal_expectation.rs:964-975`
- production上の唯一の利用は `life/temporal_participation.rs:441` の `contrast_at(now).is_some()`（範囲の判定）である。temporal contextを要するVoiceがいるhopごとに `predict_at` を201回（各回で全lagを走査）呼び、`TemporalForecast` のcopyも2.4 KB増える。
- 修正案: 欄とloopを削除し、範囲の述語に置き換える。

<a id="c6"></a>

### C6. HV-9の `Relations`・`Contexts`・`Accumulator` にproductionの読者がない（読者なしは確認済、最悪計算量は報告）

- 場所: `src/temporal_cognition/harmonic_relations.rs:3`（`#![allow(dead_code)]`）、`:378-381, 405, 415`
- `Model::relations()` / `contexts()` / `support_context()` の呼出は、src・tests・gui・crates・examplesのどこにもない。boundedなworkerがobservationごとに計算している。`allow(dead_code)` がこれを隠している。
- 最悪の計算:
  - `derived.rs:412` の `substitute`: 最古のanchorから走査し、最悪でO(38,400)×hellingerをeventごとに最大2回。
  - `context.rs:46-83` の `Contexts::expire`: 先頭anchorの失効ごとに全memberを再集計（O(KMAX × members)）。`CLAUDE.md` の「Do not restore whole-history ... rescoring」に近い形である。
  - `context.rs:284-314` の `same_summary`: O(members²)（到達性は未確認）。
  - `context.rs:35` の `VecDeque::with_capacity(38_400)` × 272 B ≈ 10.4 MB。
- 修正案: `CLAUDE.md` はarrivalとcadenceを「records」として扱うので、削除ではなく遮断が妥当である。`cfg(test)` にするか、report有効時だけ `relations.observe`・`contexts.complete`・accumulatorを回す。`allow(dead_code)` を外し、残りをcompilerに示させる（[D9](#d9)）。

<a id="c7"></a>

### C7. ListenerTwinの `prediction_enabled` がproductionで常にtrue（報告）

- 場所: `src/runtime/worker.rs:256`、`src/listener_twin/mod.rs:63, 280`
- 切替は `#[cfg(test)] set_prediction_enabled` だけにある。消費者は `ui_frame.rs:222` とreport（`phases.rs:618`）だけで、headlessかつreportなしでもE0 predictor（256×binの履歴、frameごとのpeak探索、発行ごとのretrieval）が動く。`CLAUDE.md` の「Dominant-peak search runs only inside `prediction_enabled`」は閉じることのないgateになっている。
- 修正案: `prediction_enabled = UIあり || reporterあり`。

<a id="c8"></a>

### C8. 合成の標本ごとの無駄（報告、コストは未計測）

- glide中の全lane再構築: `src/life/sound/oscillator_bank.rs:572-577, 764-779`。pitchの変化が1e-6を超えると毎標本 `refresh_pitch_state`（laneごとにcosとsin）と `apply_spectral_damping`（laneごとにpowf）を行う。Voiceのtoneは常にpitch平滑tau 0.02秒（`voice.rs:311`）なので、pitch Updateごとに約0.1秒、連続glide中は常時になる。`:775` のコメント「never per sample」と矛盾する。16 partial × unison 9 = 144 laneで、標本・Voiceごとにtrig 288回とpowf 144回。修正案: damped gainは64標本のcadenceでだけ計算し、回転係数はcadenceとcentの閾値で更新する。Nyquistのcull判定だけ毎標本に残す。
- 定数引数のlibmを毎標本再計算: `src/life/sound/tone.rs:608, 676, 1004-1015`、`src/life/articulation_envelope.rs:32`。sineのtone 1本で標本ごとにln×2とexp×2〜3を、pitch収束後も続ける。修正案（bit同一）: decay係数を構築時にcacheし、pitch stepは (current, target) → next を1件memoする。
- 減衰状態がsubnormalの固定点に張り付く: `src/synth/resonator.rs:259-288`、`oscillator_bank.rs:604-607, 616`、`tone.rs:608`。入力0のMCFはsubnormal域で格子回転になり0に落ちない。FTZ/DAZの設定はない。`modal().sustain()` で約12.7×t60（2〜20秒）後から全modeが毎標本subnormal演算になる。Intel x86ではsubnormalの乗算にassistがかかり、ARMとZenでは軽微と見込まれる。前例として `nsgt_rt.rs:479-481` が同じ現象を明示的にflushしている。修正案: 同じ方式の閾値flushを、forecast側（`SineForecast.boost_decay`、`BankForecast`）にも同じ規則で入れる。SIMDの全levelで同じ判定にする。

<a id="c9"></a>

### C9. その他の割当と無駄な計算（報告）

- note-onごとに約12個のVec: `src/life/schedule_renderer.rs:1090-1099` → `oscillator_bank.rs:104-220`、`tone.rs:243`。sine 1 laneでも12個前後を確保し、終了時に解放する。`CLAUDE.md` の予算に記載がない。修正案: SoAを1つのblockにまとめ、padding前の中間Vecを廃止する。必要ならfree-list。
- PeakSamplerの提案ごとの確保: `src/life/pitch_core.rs:1376-1397, 2366, 2436`、`src/life/adaptation.rs:28-44`、`FeaturesNow::from_occupancy_scan`（`pitch_controller.rs:467`）。Voiceあたり約2秒に1回。HillClimbの `ProposalScratch` を共用する。
- HV-9のDecisionをpublicationごとに丸ごとclone: `src/temporal_cognition/harmonic_relations/runtime.rs:499-503, 865`。HMAX=16個の全atomと、consumerが読まない欄（`Atom.mapping` など）を運び、古い `Arc<Decision>` がaudio worker上でdropされる（Vec約50本のfree）。修正案: consumer用の薄いstructをpublishし、古いArcはLOO packetと同じ形のrecycle channelで返す。
- `consumer.rs:352-370` の `action_difference`: 乗数が0でも `consequence`（preview最大2回）を評価する。乗数0なら先に0を返す。
- `pitch_core.rs:617-624`: sequenceが有効ならtrace無効でもcounterfactualの選択を毎提案行う。`pitch_core.rs:2689-2693`: `environment_score` が使わない `rearticulation_score` を毎substep再構築する。`community.rs:534-542` と `generator_model.rs:76-88`: 予測を計算してからgateを判定し、Vecと `Arc::from` を二重に確保する。
- `src/life/flora.rs:1868-1934, 2051-2116`: Floraのrenderが Free laneも毎標本処理する。rest有効時は `flora_span_gap.rs:17-51` が毎標本全laneを走査し、f64のexpを呼ぶ。
- `src/viewdata_web.rs:224-235`: web snapshotがLOO supportを持つ全Voiceのsupport scanを33 msごとに計算する。nativeは選択した1 Voiceだけを描画時に計算する。
- `src/core/stream/synchronous.rs:17`、`src/core/stream/analysis.rs:172, 199-200`: LOO observerが使わないNSGTのstreaming状態をcloneする（nfft 16384で約328 KB/observer、記録済みの初回LOO packet 586,496 Bの半分超）。
- production上の読者がない毎hopの計算: `Landscape::consonance_density` と `consonance_field_energy`（`core/landscape.rs:409-412, 461`）、`Bank.coherence`（`core/auditory_frontend.rs:249`）。
- `src/temporal_cognition/body.rs:609, 548`（opt-inのtemporal_body経路）: hop経路でblockingの `Mutex::lock` と約23 KBのcopy、毎hop 256 KBのzero fill。同種の `worker.rs:764` は `try_lock` を使う。

---

<a id="d"></a>

## D. 構造の改善

<a id="d1"></a>

### D1. `src/life/pitch_core.rs`（5.2k行）

- `PitchCore` traitはgeneric・dynの利用がなく、enum dispatchだけで使われる（`:50-92, 933-958`）。`propose_target` はテスト専用で、`_current_freq_hz` と `_features` は全実装で未使用である。
- `adjusted_pitch_score_impl`（`:1609-1731`）は位置引数が19個で、うち3つ（`reference_loo_mode`、`reference_harmonics`、`body_ratios`）はproductionで `let _ =` として捨てている（`:1720-1725`）。
- setter 8組がHillClimbとPeakSamplerで逐語的に重複している（`:193-246` と `:1198-1270`）。
- `melodic_decision`・`sequence_decision`・`ps.decision`・`tonal.selection_known` がRefCell/Cellの副経路になっている（`:121-132, 1096-1123`、`pitch_controller.rs:506-538, 581-588`）。trace用のcounterfactualが上書きするので手動で退避・復元しており、副経路を足すたびに復元漏れの危険がある。
- 改善案: traitを撤去し、`ScoreCtx` structと共通のparams structを入れ、`ProposalOutcome { proposal, melodic, sequence, pitch_set, tonal_known }` を戻り値にする。`too_many_arguments` のallowは11から4前後に減り、約200行が減り、f32引数の順序の取り違えを防げる。

<a id="d2"></a>

### D2. `src/life/voice.rs`

- `tick_control_with_listener_pressure` と `update_articulation_with_listener_pressure` は本体が同一である（`:1407-1448, 1533-1574, 1720-1769`）。
- `Voice::metadata()` と `Voice::apply_update` は呼出がない。`tick_control`・`decide_pitch_target`・`tick_phonation`・`tick_phonation_into`・`release_gain`・`force_set_pitch_log2` はcrate内テスト専用の `pub` である。
- `Community::add_voice`（`community.rs:415-464`）はテスト専用で、`spawn_one`（`:806-847`）の設定を約45行複製しており、`harmonic_member` と観測の配線が既に欠けている。
- 改善案: まずテスト専用のwrapperを削除する（`too_many_arguments` 11件のうち7件は同じ経路の薄いwrapper）。残る `decide_pitch_from_frequency` と `DeferredPitch::push` は同じ引数の組を運ぶので、借用の `PitchStepInput` 1つで共有する。

<a id="d3"></a>

### D3. `src/scripting/engine.rs`（3.1k行）

- `register_fn` が150回、`let diagnostics = ctx.lock()...clone()` が71回ある。
- 引数なしのPopulationSpec builder 15個（`seek_consonance`、`anchor`、`sustain`、`repeat`、`metric`、`entrained`、`steady`、`flow`、`once`、`while_alive`、`phonate_when_viable`、`adaptive_duration`、`rest`、`respawn_random`、`respawn_consonance`）が同じ形で、既存の `register_population_spec_numeric_methods` と同じtable方式にできる。
- enum文字列のsetter 6個（`move_cost_time_scale`、`pitch_apply_mode`、`pitch_core`、`brain`、`rhythm_role`、`viability_scope`）も同じ形である。
- `numeric_overloads!` があるのに、`avoid_neighbors`、`global_peaks`、`intrinsic_rate`、`respawn_hereditary`、`scaffold_*`、`meter_stability`、`wait`、`temporal_basin`、`output_gain` はFLOAT/INTを手書きしている。
- handleのabsent/inactiveの定型が8箇所ある（`:1890-2260`）。警告のkindが `<name>_absent` / `population_patch` / `population_update_inactive` の3流儀に分かれている。`with_live_population(ctx, handle, label, pos, |p| ...)` 1つで解消できる。
- 汎用helper内の `if name == "rest_on_habituation"` の特例（`:2847`）。検証が3箇所（helper、`spec.rs:862` の無検証setter、`context.rs:1070-1072`）に分散し、最後のものは到達不能である。f64重みのsetter 4個（`spec.rs:383-418`）と登録10個（`engine.rs:1009-1100`）も同文である。
- Flora・Rainの数値parameterが文字列でdispatchされ（`spec.rs:1073-1118, 1206-1222`）、名前の列が4箇所に分散し、それぞれ `_ => unreachable!()` を持つ。追加時の不一致は台本実行時のpanicになる。(name, min, max, spec setter, Option<patch setter>) のtable 1本にする。
- 効果: 数百行の削減と警告kindの統一。

<a id="d4"></a>

### D4. scriptingの不要な状態と公開面

- `context.rs:13-28`: `PopulationState::spawned` は常にtrueで（`:1143`）、`&& p.spawned`（`:595, 649`）は意味がない。`respawn_policy`・`crowding_target_same/other` は `spec` の複製で、`engine.rs:1914-1917` で二重に書いている。`ScopeFrame {}`（`:57`）は空で、depthのcounterで足りる。`if !ids.is_empty()`（`:1150`）は常にtrue。
- `placement.rs:180, 257` の `other =>` の腕は直前の `matches!` で到達不能である。`ControlUpdate::timbre_inharmonic` / `timbre_motion`（`scenario/control.rs:404-405`）にはproducerがない。
- `scripting/mod.rs:14-20` の `pub use`（`ScriptContext`、`Bus`、`BusSet`、`Placement`、`PopulationHandle`、`PopulationSpecHandle`）と `ScriptHost::create_engine`（`engine.rs:132`）・`load_script`（`:2741`）にcrate外のconsumerがない。外部の利用は `load_script_with_warnings` と `defs_gen` だけである。`pub(crate)` にする。
- `evidence.rs:300` の `CONFIG_DOCS` は `AppConfig` の末端keyの手書きの写しで、`defs_gen.rs:236-262` は重複と空のevidenceしか検査しない。全Optionを埋めた `AppConfig` をtomlにしてleaf keyの集合と比べるテストを追加する。現時点では73 keyが一致している。

<a id="d5"></a>

### D5. `src/runtime/`

- `prepare_runtime`（`wiring.rs:266-768`）が約500行で、台本の走査が8回あり、同じ述語が重複している。`resource_power`（`:403-408`）＝ `has_flora`（`:659-664`）、`tonal_enabled`（`:456-458`）＝ `prepare_tonal`（`:838-840`）、harmonicの `valid`（`:418`）＝ `step.rs:83` の `active`、LOO Spawnの述語（`:627-645`）＝ `phases.rs:49-66`。output gainの計算も `scenario_load.rs:284-295` と `step.rs:100-106` で二重になっている。
  `prepare_tonal` と `tonal_enabled` がずれるとtableの構築前に機器が始まる。[A7](#a7)も検査の分散が原因である。
  改善案: `ScenarioNeeds::scan(&Scenario)` の1 passを `init_runtime`・`prepare_runtime`・`StepEngine::new` で共有する。
- webのinline解析（`step.rs:225-275, 333`）は `make_step`（`core/analysis_worker.rs:257-322`）の手書きの複製で、出荷経路は `cargo check` しか通っていない。PCM一致のテスト `compare_offline` は `cfg(all(test, feature = "native"))` で、native-inlineを検証している。現時点で挙動の差は見つかっていない。
  改善案: `handoff::Slot` を `Rc` で共有して `make_step` を両featureで共通にし、出荷経路を検証経路と一致させる。

<a id="d6"></a>

### D6. Flora・participation・renderer

- `Flora::update_tones`（`flora.rs:1252-1780`、約530行）で同じ式が繰り返される。50 centのclaim述語が6回（`flora.rs:1273, 1315, 1333, 1360, 1463`、`flora_metric.rs:336`）、entry scoreの式が4回（`1547-1554, 1612-1620, 1645-1649, 1655-1659`）、retirementのdecay式が3回（`flora.rs:1701, 1803`、`flora_metric.rs:244`）、noisy laneの帯域係数blockが2回、nearest peakの選択が2回。`Peak::claimable_from(&Lane)`・`entry_score(&Peak)`・`retirement_decay()` をhelperにし（3回以上の使用で `CLAUDE.md` の規則に適合する）、HV-3の分岐（`1282-1437`）を別関数に分ける。
- participationのemitter 3つ（`sustained_participation.rs:162-187`、`bounded_participation.rs:79-104`、`shape_participation.rs:176-200`）で、glideの `ToneCmd::Update` 発行blockが3回、adoption適用blockが2回重複する。`Voice::emit_glide_update(..)` を抽出し、3経路のpitch適用規則が修正漏れで分かれる危険を除く。
- rendererのtone単位の処理が2経路に複製されている（`schedule_renderer.rs:664-765` と `819-897`）。`render_sample_major_oracle` は名前に反してPX-1のproduction経路である（`:767`）。`RoutedTone` に1標本の処理を集約し、名前を `render_sample_major` にする。

<a id="d7"></a>

### D7. 音響部の死んだコード

- `ModeShape::Harmonic`（`life/sound/modal_engine.rs:14-19, 340-351, 373-399`）はsrc・tests・examplesのどこでも構築されない。
- `SoundBody::articulate_wave`（`sound_body.rs:58, 104, 242, 477`、`modal.rs:136-151`）は呼出元がない。ModalBodyは `ModalEngine` をself-reportのためだけに保持している。削除するときはbuild時のRNG drawの消費順を保つ必要がある。
- `oscillator_bank.rs:153-167` の `BodyKind::Modal` の腕は `AnyBackend::from_snapshot` 経由で到達しない。

<a id="d8"></a>

### D8. `src/core/`

- `core.rs:24` の `pub mod nsgt_kernel`: crate外の利用は `tests/config_restore.rs` の `KernelAlign` だけで、offline API（`analyze`、`analyze_psd`、`BandCoeffs`、`WinLenCap`）はテスト専用、`win_len_cap` はproductionで常にNoneである。
- `mode_pattern.rs:204-363` と `365-672` の評価器2本を手動で同期している。`eval` の `space` 引数は `_space` に流れるだけである（`:534, 602`）。
- `meter/continuous_recurrence.rs:776-802` で20欄を手動でresetしており、`with_input` と `set_shaping` と同じ内容である。
- flux→driveの式 `(flux.max(0.0) * 500.0).tanh().clamp(0.0, 1.0)` が3箇所（`temporal_expectation.rs:882`、`runtime/phases.rs:999`、`listener_twin/mod.rs:516`）にあり、+infの扱いが3番目だけ異なる。
- 解析側の `harmonic_relations_weight` / `harmonic_continuation_weight`（`stream/analysis.rs:309, 316`）は書くだけで読まれない。
- PSの `coefficients.clone().into()` が毎hop Vec→Arcで二重にcopyする（`tonal_context/pitch_set.rs:303-314`）。`harmonic_support::Snapshot` が公開のArcにFFTのscratchを同梱し、`make_mut` のcopyでscratchも複製される（`harmonic_support.rs:144-161`）。

<a id="d9"></a>

### D9. `src/temporal_cognition/` のテスト専用・未読のitem

- `#![allow(dead_code)]` を外すと露出するもの: `Model::predict`（`harmonic_relations.rs:285`）、`freeze_learning` / `remove_correspondences` とその分岐、`Matcher::endpoint`（`matching.rs:452`）、`Prediction::for_query`（`matching.rs:223`）、`Graph::edges` / `edge_attributes` / `EdgeAttributes` / `EdgeKind` / `EdgeId`（`graph.rs:4-15, 631-739`）、`edge_capacity`、書くだけの `Adapter.baseline`（`adapter.rs:14, 91`）と `Anchor.slot`（`graph.rs:477, 521`）、Errを返す経路のない `Graph::insert` のResult（`graph.rs:420`）。
- 同じ重複trajectory・event flagの検査が `adapter.rs:59-67` と `graph.rs:169-183` の2箇所にある（各O(S²)）。`Graph::validate` に一本化する。`matching.rs:590-595` の `min_step` はloop不変なので `Graph::new` で1回計算する。
- `proposals/producer.rs:106-108, 315-341`: 毎hop57要素の大きな配列をstackに複数本置き、うち `continuations`・`packed`・`saved` は不要である。`recurrence.rs:202` のコメントと `#[inline(never)]` は、stackの余裕が薄いことを示している（frame sizeは手計算のみ）。
- その他の残骸: `feature_projection.rs:7-38` の `Feature::Projected`、`arrival.rs:83-100` の `probability()` の第2要素（常に0）と `Forecast.evaluations`（常に0）、`resources.rs:99` の `Meter::frame` の `epoch` / `table_ns` と `rejected()`、`accents.rs:8-12` の `Delivery` の欄、`grouping.rs:38-39` の `Correlations.paired_*`。段間の再検証（`feature_projection.rs:69-83` と `features.rs:194-207`、weightsの正規化検査3箇所）。

<a id="d10"></a>

### D10. 共有helperの書き直しと小さな重複

- `sanitize_nonnegative_finite` 相当の手書き: `pitch_core.rs:194, 202, 224, 249, 1176, 1199, 1228, 1236, 1248`、`pitch_controller.rs:334`、`metabolism_policy.rs:40`、`tone.rs:729-733, 740-744`、`phonation_engine.rs:818-822`、`core/consonance_kernel.rs:72-76`。
- `finite_or` 相当: `phonation_rules.rs:185-191`、`synth/modes.rs:72-81`、`core/meter.rs:950-959`（`sane_basin`）、`core/nsgt_rt.rs:488-510`。
- `core/float.rs:2` は「single 0..1 clamp」と宣言するが、テスト外のcoreに生の `clamp(0.0, 1.0)` が約57箇所ある（`meter.rs` 13、`landscape.rs` 10）。全置換ではなく、NaNが通り得る箇所（[A5](#a5)）を優先する。
- splitmix64のstepが4箇所（`flora.rs:107-114`、`tone.rs:1048-1056`、`oscillator_bank.rs:956-962`、`synth/resonator.rs:10-16`）。`Ramp` が2つあり丸めの規則が異なる（`flora.rs:124-164` はceilと閉形式、`rain.rs:64-104` はroundと累積）。
- 合法音域（`freq ± range_oct.clamp(0,6)/2`）の計算が7箇所で微妙に異なる（`pitch_controller.rs:59-78, 251-258, 281-293`、`harmonic_voice.rs:41-44`、`voice.rs:663, 677, 1049`）。centerのfallbackとfield境界との交差の有無が揃っていない。`target_bounds` に集約する。
- `community/frequency.rs` の累積重みの抽選が4箇所（`:631-641, 656-665, 848-856, 863-870`）、`gap_ref` が3箇所（`:469-475, 566-572, 728-731`）。

<a id="d11"></a>

### D11. `CLAUDE.md` のF2規則に反する無名のassert

- `src/life/span_gap.rs:140-142` と `src/life/report.rs:1046` が、`CLAUDE.md` が「production callerがない、使うな」とする無名の `assert_scan_len` を呼んでいる。`assert_scan_len_named(…, "onset_habituation_state_scan")` に替える。

---

<a id="e"></a>

## E. CIと横断的な方針

<a id="e1"></a>

### E1. CIがwasm32を一度もビルドしない（確認済）

- `.github/workflows/ci.yml` は `cargo check --lib --no-default-features --features web` をhost targetで実行するだけで、`crates/conchordal-web` と `gui --features browser` をwasm32でビルドしない。`target_arch = "wasm32"` のcfgを持つコードは8箇所ある。
- Cargo.lockが3つ（root、`gui/`、`crates/conchordal-web/`）あり、`build.sh` は `--locked` でそれぞれを使うので、依存のversionが分かれ得る。
- 改善案: Linuxのjobに `rustup target add wasm32-unknown-unknown` と、engine（`crates/conchordal-web`）とgui（`--features browser --lib`）の `cargo check --target wasm32-unknown-unknown` を追加する。[A10](#a10)のような32 bit固有の問題はcheckでは見つからないので、必要なら別途wasmでの長時間テストを検討する。

<a id="e2"></a>

### E2. 非有限のPCMの扱いが箇所ごとに揃っていない（機構は報告、発生源は未特定）

- panic: `core/temporal_history.rs:41`（`assert!(band_rms.is_none_or(|v| v.iter().all(|x| x.is_finite() && *x >= 0.0)))`、`temporal_expectation.rs:200-201` と `spectral_history.rs:60-69` から無検査で到達）、`listener_twin/contour.rs:227`（report有効時）。
- 無効化: TC（`tonal_context/mod.rs:262-265`）。skip・unknown: `runtime/phases.rs:995, 1102`。素通し: `LimiterMode::None`。web: decode失敗で演奏停止。
- IIRの状態はNaNに固着し、gapでresetされるまで戻らない（`auditory_frontend.rs` の `Bank.state`、`nsgt_rt.rs:258, 478`、`stream/dorsal.rs:61-65`）。
- 確認できた発生源は[A5](#a5)の `amp(NaN)` 経路である。
- 改善案: 境界の1箇所でhopを検査し、非有限なら「未観測」として扱う（既存のgap分岐を流用し、filter状態をresetする）。値のassertは `debug_assert` にする。

<a id="e3"></a>

### E3. `CLAUDE.md` の割当予算と記述が実態とずれている

- 記載のない割当: `Landscape` の毎hop複製（[C3](#c3)）、note-onごとのVec（[C9](#c9)）、`OwnSoundHistory`（[B4](#b4)）。
- 実態と異なる記述: spectral historyのGUIでの有効化（[C4](#c4)）、habituationの「owned harmonicity projection」（[C2](#c2)）、`prediction_enabled` のgate（[C7](#c7)）。
- 対応するコードを直すときに、同じ変更で記述を更新する。

<a id="e4"></a>

### E4. `loo-diagnostic` featureが「Temporary」のまま残っている

- `Cargo.toml` の説明は「Temporary proposal replay for the isolated T-31 prototype」である。`cargo check --lib --features loo-diagnostic` は通るが、CIでは検査していない。
- 残すならCIの検査に加え、不要なら削除する。

<a id="e5"></a>

### E5. 手書きのAVX2 dispatchと `mul_add`（報告）

- `src/core/history_prediction.rs:490-504` の `is_x86_feature_detected!("avx2")` と `unsafe` は、crate内で唯一の生のfeature検出である。`CLAUDE.md` は `fearless_simd` 経由とlevel別テストを求めるが、該当するテストがない。
- `history_prediction.rs:163` と `core/compensated.rs:72` の `mul_add` は、std の `mul_add` が正確に丸めるのでbit同一性には問題がない。ただしFMA命令のないtarget（wasm32、portableなx86-64）ではlibmの `fma` 呼出になり、`forecast()` ごと最大38,988回呼ばれる（速度低下は未計測）。
- 改善案: `Level::as_avx2()` と `vectorize` に置き換え、two-productはDekker/Veltkampにする。

<a id="e6"></a>

### E6. `clippy::too_many_arguments` のallowが58箇所

- `voice.rs` と `pitch_core.rs` に各11箇所集中している。[D1](#d1)と[D2](#d2)で大半が解消する。

---

<a id="sound"></a>

## 健全と確認した範囲

各担当が読んで問題を見つけなかった範囲を記す。いずれも読解による確認である。

- **config**: 全21のstructに `deny_unknown_fields`。`validate` の数値範囲と、`load_or_default` のerrorの意味（明示パスの不在・不正key・範囲外は失敗し、暗黙パスのときだけ既定値を書き出す）。
- **Air-Gap**: `main.rs` は `run_render` を参照せず、`cli.rs` に出力パスのflagはない。`WavOutput` は `wiring::run_render` とテストだけで使われる。`page.js` のallowlistは `outputPeak`・`nonzeroFrames`・seed・error・snapshotを送らない。eframeのwebにはpersistence featureがない。
- **生成docs**: `defs_gen::check` がengineの登録と `FN_DOCS` を照合する（存在、arity、tier、Live/Initialとreceiver）。initial-onlyとliveの区分は `CLAUDE.md` の記載と一致する。endlessの予約（`observe_tonal_context()`、`observe_harmonic_relations(max_sources)`）は位置付きerrorで、有限の場面では警告して無視する。`listener_profile` の既定値とtonalの重みも一致する。
- **thread間の同期**: cpal callbackに割当・lock・logはなく、atomicだけを使う。field解析の `Latest` publishはnon-blockingで、`wait_for_analysis` はexitingとdisconnectで抜ける。analysis threadのpanicはDisconnectedとして失敗に変わる。LOOのrecycleは1つのVecが循環し、結果のpublishより先に返却される。
- **wiring**: `init_runtime`・`run_render`・`StepEngine` はすべて `prepare_runtime` を通る。`src/runtime` から `crate::ui` への参照はない。headlessとofflineでは `ui_tx = None` でUI frameを作らない。
- **数値核**: `auditory_frontend` の `read_lanes` はscalarと同じ順のComplex32演算でreductionがない。`roughness_kernel` は固定8 laneと固定のreduction treeで `mul_add` がなく、level別のbitテストがある。`harmonicity_kernel` は入力をsanitizeし、`assert_scan_len_named` とlevel別テストを持つ。PSの `f64x4` 評価器はscalarのchannel順を保ち、wasm simd128を含むlevel別bitテストがある。scan長のassertは座標の検査より先にある。`nsgt_kernel` / `nsgt_rt` にhopごとの割当はない。
- **Voice行動**: Voiceの行動が読む値はすべて `_eff` である。ゼロ重みでの乱数消費の不変性はME・SQ・PS・TC・dwell・tail roughness・sway・rest・phraseで守られ、counterfactualはRNGをcloneし、HV-9の `dedicated_draw` は状態を持たない。`advance_scratch`・`gate_open_scratch`・`dead_id_scratch`・HillClimbの `ProposalScratch` は再利用されている。ME・SQ・PPMの資源枠は `CLAUDE.md` と一致する。
- **Flora・phonation**: pace則（`beat_pace·min(k, 20·r)`）とratioの選択条件は `CLAUDE.md` と一致する。Floraの `synthesize`・`OscillatorBank::process_sample`・`ResonatorBank::process_sample` は `Level::baseline()` を使い、`mul_add` がない。Floraのadvance・render・metric allocate・support scratchは既存bufferのclearと再充填で、定常的な割当はない。span-gapの状態機械、CouplingClockの交差tick、PhonationEngineのscratch再利用、TemporalParticipationのtick一致に矛盾はない。ecology側に音響体の種類による分岐はない。
- **HV-9のprotocol**: hopの右端のrecordは次のbatchまで残り、audio側の `expected_query` とworker側の `query` は同じrecordの集合から導かれる。容量（anchors ≤ records ≤ 38,400）、indexの走査上限、batch 1回の失効、入力欠落時の中立化も記述どおりである。
- **ListenerTwin**: F2のassert、grid変化時だけの確保、eligibleとmarginalで同じ `reference_audible` filter、Copy snapshotへのqの凍結、DCCが `static_pull_level` だけを読むこと。

---

<a id="next"></a>

## この記録の扱い

- 着手した項目は、この文書の該当節に完了したcommitを追記するか、[現行計画](plan-current.md)に移す。
- 行番号は `cb7c29d6` 時点のものなので、着手時には該当行を読み直してから直す。
- B群の判断は、該当箇所に手を入れるときに作者に求める。いま決める判断はない。
