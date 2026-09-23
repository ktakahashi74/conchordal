# I11-1: onset-now／delayed onset比較の実装前登録

2026-09-21。I10の狭い範囲の完了（[I10記録](i10-body-outcome.md)）に続く、I11の最初の一機構の登録である。
初稿〜第3稿へのAstraレビュー（いずれも「不可」。`target/i10-bank-forecast-inputs-20260921/astra-review-i11-1-draft1.md`、
同 `-draft2.md`、`-draft3.md`、`-draft4.md`）を反映した第5稿。第4稿の判定は「条件付き可」で、
条件2件（§5.9の比較対象、第1段の必須範囲）を本稿で反映した。本文書は実装前の登録であり、機能の成立を主張しない。登録するのは、比較する量、集計の尺度、
有界窓、四つの時刻と締切、入力が不明のときの規則、両候補を同じ観測時点から評価する契約、供給元の出力形、
off基準、検査計画である。数値は全て宣言した工学規則で、人の評定への適合は行わない
（[milestones §1.3](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)）。

I11は「支持された関係一つ→実消費者→音響差」を対象T行・消費者ごとに閉じる。ここで閉じる一機構は、
T1（自声の身体予測）とT2（到来期待）を、実消費者 `src/life/temporal_participation.rs` の
発音時刻決定へ入力として返す経路である。第1段が返すのは「代表onset条件付きの身体予測」であり、
実kick・実rhythms・実seedに条件付けたI10の動作条件付き予測そのものではない（§4.2、§6）。T1を第1段、T2を第2段とし、登録・実装・完了記録を段ごとに分ける。
第1段は本文書で登録済み。第2段は配送契約と量の形まで登録し、Hazard設定の凍結とCDF格子の最終値は
第2段の実装前に追記する（§4.5）。

## 1. 対象と非対象

対象は、発音機会を持つVoiceが「基準の到来時刻で鳴る」か「前後にずらして鳴る」かを決める既存の候補選択に、
I10で得た自声の予測energyと、I6の到来期待を入力として与えることである。決定の主体、費用の符号・重み、
skipの方策は `TemporalParticipation`（ALife側）に属する。DCC側は確率と予測energyを供給するだけである。
音高の決定は既存のtheta gateとhill-climbに委ね、時刻の決定と独立に扱う。

非対象は次のとおり。共同posteriorや全帰結投影（研究拡張）。候補評価の結果を学習へ返すこと。
release／gap／wait／skipの候補評価（既存の診断のまま）。作者採用（A1／A2）、資源受入（R2）、実機受入（A3）。

## 2. 現状の消費者（確認済みの事実）

`TemporalParticipation::candidate(now, end, onset_allowed, forecast)`（`src/life/temporal_participation.rs`）
は、予約 `planned` が無いとき、次の到来時刻 `due_frame` を中心に候補格子を作り、費用最小の候補を予約する。

- `period_frames` は固有周期 `intrinsic_period_frames` を初期値とし、`update_reference` で外部予測の
  反復周期へ追従した値である。固有周期そのものではない。`flow_depth > 0` では追従しない。
- 格子は `offset = -2..=20`。負側は `due + offset * width / 2`、正側は `due + offset * period_frames / 20`。
  `offset = 0` は `max(due_frame, earliest)`。`earliest` は直前onsetから最大onset率で決まる下限で、
  `now` より後になり得る。`width = min(0.2 * period_frames, 0.06 * fs)`。`earliest` 未満の候補は捨てる。
- 費用は次のとおり。条件を満たさない項は加えない。

```
cost(at) = displacement²
         + coupling * H(at)                      [onset_allowed かつ memory と context が有る場合]
         + coupling * 6 * s * Ov(at) / norm      [onset_allowed かつ s != 0 かつ forecast が有る場合]
displacement = (at - due_frame) / period_frames
H   = 記憶した外部帯域energy形状と予測形状の二乗Hellinger距離（片側だけ無音なら1、両方無音なら0）
Ov  = sum_k sum_b own_b(k) * ext_b(at + delay_k) / (own_b(k) + ext_b(at + delay_k) + 1e-12)
norm = max(1e-12, sum_b own_profile_b * sum_k power_k)
s   = overlap_sensitivity（呼び出し元が0.8で固定）
```

- 自声footprintは現状ADSRの代理である。`duration = hold_sec + release_sec` を64等分し、ADSR包絡の二乗
  `power_k` を重み、帯域配分は静的 `own_profile`（3帯域）とする。身体の応答、modulator、残存toneは含まない。
- 外部予測は `TemporalForecast::sustained_energy_at(at + delay_k)`。予測地平を越えると観測平均へ落ちる。
- `forecast` は、`energy_window_after(max(due - width, now))` と `energy_window_after(due + period + 80 ms)`
  の両方が得られる場合だけ費用へ入る。未支持の候補を安く見せない規則である。
- 同点は格子順で先の候補（負側が先）が残る（厳密な `<` 比較）。
- skipは `onset_allowed` かつ `forecast` が有り、自声の総energyが正、`s > 0`、かつ
  `best_cost > 1 + skipped_cycles` のとき、`due_frame += next_interval()` で一周期見送る
  （flow時の間隔は `next_interval()` が決める）。skipは行動の決定であり、報酬でも実音でもない。
- 予約後は次の `candidate()` で再評価しない。予約時刻が `now` を過ぎると予約を捨て、`due` を進める。
  `resolve(tick, sounded)` が予約を消費し、鳴った場合だけ `last_onset` と文脈を記録する。
- 選択後、`ParticipationContextPrediction` にonset時刻、予測の観測時点、対象窓、予測帯域energy、外部footprintを
  記録し、reportへ出す。

呼び出し元は `src/life/phonation_engine.rs`（640〜680行付近）で、hop経路上で同期的に呼ぶ。
`TemporalForecast` は `src/runtime/mod.rs` がhopごとに `observer.forecast()` で同期生成し、必要なVoiceへ渡す。

## 3. 供給元の出力形（確認済み）

### 3.1 外部音の短期予測: `src/core/temporal_expectation.rs`

- 観測窓は `round(fs / 100)` sample（48 kHzで480 sample、10 ms）。予測の刻みは `FORECAST_STRIDE = 2` 窓、
  `FORECAST_LEN = 201` 点。地平は起点から200刻み（20 ms × 200 = 4.0秒）。
- 各点に3帯域（低・中・高）の `band_energy` と、事象率基準に対する符号付き `contrast`。
- `observed_frame()` は完成した観測窓の終端（予測の起点）。`available_through_frame` は受領済み入力の終端で、
  これより前の時刻の `band_energy_at` は `None`。`energy_window_after(frame)` は `frame` を窓境界へ切り上げた
  窓を検査するため、`frame` 自体が締切前でも成功し得る。締切の検査は窓の `start` に対して行う。
- `sustained_energy_at(frame)` は地平内では `band_energy_at`、地平外では観測平均 `background_energy`。
- 反復lagの探索範囲は `ceil(0.25 / step) ..= floor(2.0 / step)`（10 ms刻みで約0.25〜2.0秒）。
  `MAX_DELAY = 400` は配列容量であり探索上限ではない。
- 自声の除外（own-excluded）は供給側で行い、habitat busを対象とする。

### 3.2 自声の予測energy: I10の候補energy

- `ToneEnergy`（`src/life/self_prediction/projection.rs`）は、renderer包絡、制御予測（絶対時刻 `issued_at`、
  `valid_until`、振幅更新、平滑化、modulator状態）、予約release、sine／bank搬送波を持つ。
- `project_window(retained, added, action_at, intervention, bus, interval, use_coherent)`
  （`src/life/action_candidates/energy.rs`）は区間を16 binへ分け、`Window { energies, incoherent_energies,
  coherent_energies, mean, .. }` を返す。`use_coherent = true` でも、coherentが得られないbinは中央点の
  incoherent値へ戻る。したがって `Some(Window)` は全binの支持を意味しない。
- 予測は候補worker（別thread、容量64の有界queue、逐次処理）で計算する。hop経路では計算しない。
- 支持条件: pitch定常、continuous drive 0、motion 0、17 lane未満、発音中の再impulseなし。
- 登録済み71分岐での順位一致は固定development素材への限定付き。1 recordの費用はsine約7 ms、
  harmonic約80〜210 ms、modal約60〜130 ms。flow条件では要求の55〜98%が容量dropになる。

### 3.3 到来期待: `src/temporal_cognition/arrival.rs`

- `Forecast { model, version, group, source_start, source_end, available, issued_at, horizon_end,
  last_accent, elapsed_seconds: [f64; 2], reset_unknown, probability: Option<[f64; 2]>,
  observed_survival, evaluations }`。`version` はモデル版であり更新番号ではない。
- 予測対象は「発行時刻の後、`horizon_end` までに次のaccentがあるか」。`probability` の二値は、経過時間と
  resetの取り方の全ての可能性（途中の位相極値を含む）を区間演算で囲んだ確率の下界・上界である。
  不確実性が無ければ二値は等しい。`reset_unknown` は未解消のreset不確実性を示す。`None` は計算不能。
- Hazardモデルの18係数・標準化・horizonは設定で明示し、内部推定しない（[I6記録](i6-period-arrival.md)）。
  `Periodic` モデルは周期peakから段階関数で与える。
- 音群枠は全体で最大7。観測worker内で作られ、共有snapshotとしてreportとUIへ出る。発音選択への
  直接配送は無い。

## 4. 登録する比較

### 4.1 候補集合と同一観測時点契約

候補集合は§2の既存格子とする。新しい格子は作らない。「基準候補」は `offset = 0`（到来時刻、
`earliest` による繰り下げを含む）、「遅延候補」は正側、「前倒し候補」は負側であり、いずれも勝ち得る。
本文書の「onset-nowとdelayed onsetの比較」は、基準候補の費用と、それ以外の候補の最小費用との差を指す。

一回の `candidate()` 呼び出しの中で、全候補を同じ `now`、同じ `forecast`（同じ `observed_frame`）、
同じ自声footprint record、同じ到来CDF表から評価する。候補ごとに供給元を取り替えない。いずれかの入力が
候補集合の一部でしか得られない場合は、その入力を全候補から外す（§4.9）。

### 4.2 自声の予測footprint（T1、第1段）

定義。Voiceの現在のrecipeから、次の手順で代表toneを作り、その予測energyを footprint とする。
代表toneは選択前に作るため、選択後に決まる量（実際のkick、予約release、onset時刻、tone_id）は
代表値で置く。これは宣言した近似であり、置いた値を全て記録する。

- 代表値。相対onset `T0 = 0`（footprintはonsetからの経過時間の関数なので絶対時刻を持たない）。
  要求を発行した絶対sample時刻は別に `requested_at` として記録し、識別には含めない。
  hold は `set_sound_duration` に渡される `hold_sec`（`hold_theta / base_rate_hz`）を `round(hold_sec * fs)`
  sampleにした値。Hold modeでは `ToneSpec.hold_ticks = Tick::MAX` なので4秒で打ち切る。kick強度は `1.0`。
  seedは `modal_phase_seed(source_id, 0, 0)`。制御予測の `NeuralRhythms` は `NeuralRhythms::default()`。
  `tone_id` は0。ADSRが `None` なら `Tone::from_parts` の既定包絡を使う。
- 代表tone。`Tone::from_parts(time, T0, hold, freq_hz, amp, body, modulator, adsr)` に
  `set_smoothing_tau_sec(recipe.smoothing_tau_sec)`、`seed_modal_phases(seed)`、
  `schedule_planned_kick(OnsetKick { strength: 1.0 })`、`arm_onset_trigger(1.0)` を適用する。
  振幅更新、予約release、残存toneは含めない。
- `ToneEnergy` は `energy_projection.rs` の `issued` と同じ手順で凍結する。`prediction_sine(T0)`、
  `prediction_bank(T0)`、`prediction_control(T0, &NeuralRhythms::default())`、`prediction_parameters(None)`、
  `scheduled_release = None`（holdが包絡の終端を与える）。`Tone::from_parts` が `None` を返す場合
  （amp 0、非有限値）は `unsupported`。
- 評価。`project_window(retained = [], added = Some(([true, false], tone)), action_at = T0,
  intervention = (None, None), bus = 0 (habitat), interval = [T0, T0 + D), use_coherent = true)`。
  `D = min(release_end - T0, 4 * fs)`。`project_window` が `None` を返す場合（`D < 16`）は `unsupported`。
  4秒で打ち切った場合は `truncated = true` を記録する。
- 採用する値。`Window.coherent_energies[k]`（k = 0..15）が全て `Some` のときだけ、その値を `E_k` とする。
  一つでも `None` なら footprint は `unsupported` とし、`E_k` を採用しない。`incoherent_energies` は使わない。
- 重み。`power_k = E_k / max_k E_k`。全16 binが `Some(0)` なら既知の無音であり、`power_k = 0`、
  出所は `body` のままで、重なり寄与は0とする。これは `unsupported` と区別する。
- 帯域配分は現状どおり静的 `own_profile`。I10の予測は総energyで帯域を持たない。
- 時刻移動。footprintは経過時間 `delay_k = (k + 0.5) * D / 16` の関数として保持し、候補時刻 `at` へ
  平行移動する。onset時刻ごとのphase seedの違い、実際のkick強度、残存toneとの干渉、DroneSwayの
  rhythms依存は含めない。宣言した近似である。

識別と再利用（2026-09-22に[改訂](#識別規則の改訂2026-09-22)）。footprint recordは `(source_id, body_generation, recipe_hash)` で識別する。`recipe_hash` は
代表toneの生成入力の全て、すなわち身体snapshotの全field、`amp` と `freq_hz` のf32 bit、代表hold（sample）、
ADSRの4値のbit（`None` は別符号）、modulator spec、`smoothing_tau_sec` のbit、代表kick `1.0`、
代表seed、`fs` のSHA-256とする。代表値を固定したことで生成入力は識別に全て含まれ、recordは識別が
一致する限り周期をまたいで再利用できる。識別が変わった時点で旧recordは失効する。recordには識別、
`E_k`、`D`、`truncated`、代表値、`requested_at`、workerが計算を終えた `computed_at`、Voiceが受け取った
`received_at` を保存する。

計算場所と要求規則。候補worker上で計算し、Voiceへ返す。hop経路に残るのは、識別の計算、要求の送出、
返却の受け取り、§4.4の費用計算だけである。

- worker側の要求は、既存の候補packet queueとは別の、Voice数（64）を容量とする専用queueへ入れる。
  workerはfootprint要求を候補packetより先に取り出す。worker本体は逐次なので、処理中の候補packet一つ分の
  遅延は受ける。
- Voiceあたり未完了の要求は一つ。未完了中に識別が変わった場合、その要求を「置換済み」と印し、
  返却が届いても使わずに捨て、返却後に最新の識別で再送する。
- 返却queueの容量もVoice数。返却がdropされた場合は未完了を解除し、次のhopで再送する。
- 失効したrecordは新しい要求を抑止しない。同じ識別の要求は一つにまとめる。
- 返却の遅延 `received_at - requested_at` と `proxy(absent)` の比率を報告し、§5.7で実測する。

### 4.3 外部音の短期予測

現状どおり `sustained_energy_at(at + delay_k)` を使う。`delay_k` は§4.2の16 binの中心。地平外は観測平均へ
落ちる既存規則を保つ。不明条件は§2の `forecast` filter と同じ。

`Some` の中でのproxy。footprintが `proxy(*)` のとき、または設定が `footprint = "proxy"` のとき、
ADSR代理を16 binの中心で標本化し、`power_k` とする。`D` は、識別が一致するbody recordがあればその `D`、
無ければ `hold_sec + release_sec` を4秒で打ち切った値とする。`Some` の中では body と proxy の差が `power_k` の値だけになる。`None` の64分割・
打ち切りなしはそのまま残る（§4.10）。

### 4.4 重なり量

§2の式のうち `power_k` の出所（ADSR代理→I10予測）と分割数（64→16）だけを変える。定数6、`s = 0.8`、
`norm` の形は既存値を維持する。

### 4.5 到来項（T2、第2段）

供給。観測workerは、支持のある各音群について「発行時刻から `t` までに次のaccentが来る確率」の表
`F_g(t)` を作る。`t` は `issued_at` を起点とする20 ms刻みの201点（起点を含み、4.0秒まで。§3.1の外部予測と
同じ格子）。Hazardモデルでは経過時間が既知のときだけ、`integral(elapsed, elapsed + Δt, context)` で
`1 - exp(-∫)` を各格子点まで求める。経過時間が不確実（`elapsed_seconds` の二値が異なる、または
`reset_unknown`）のときは表を作らず不明とする。上下界の表は配送しない。Periodicモデルでは予測到来時刻での
段階関数とする。表と共に `issued_at`、`horizon_end`、音群handle、bus、`periodic: bool`（周期peakの有無）を
配送する。表は既存の共有snapshot（UI／reportが読むもの）に載せ、runtimeがhopごとに読み、`TemporalForecast`
と一緒にVoiceへ渡す。Engineの係数・履歴・contextは配送しない。CDFの計算はEngine内に閉じる。

音群集合。一回の `candidate()` で使う集合は、`now` 時点で表を持ち、`issued_at <= now < horizon_end` かつ
`periodic = true` で、habitat busに属する音群とし、全候補で共通に固定する。表を持たない音群、
`periodic = false` の音群、期限外の音群は集合から除く（到来項全体を不明にするのは集合が空のときだけ）。
集合内の音群について `at ± w` が `horizon_end` を超える候補が一つでもあれば、その音群を集合から除いて
やり直し、除いた音群を報告する。自声の混入は、I10の
`consumer::Binding`（Voice→prototype対応）と共有snapshotの `groups[prototype]` 対応表から、同じbus・
身体世代・action profileのモデル版が一致し、対応表の期限内にある音群を除いて避ける（現行
`consumer.rs` のモデル一致検査と同じ条件）。対応は記述子による関連付けであり、自声混入の完全な検出では
ない。bindingが無い場合は除外せず、その旨を報告する。集合が空（`n = 0`）なら到来は不明。

量。候補 `at` に対し、`P_g(at) = F_g(min(at + w, horizon_end)) - F_g(max(at - w, issued_at))`。`w = width`
（§2）。格子点の間は線形補間する。`P(at) = (1 / n) * sum_g P_g(at)`（一様重み）。

費用。ALife側の方策として `coupling * w_arr * (1 - P(at))` を加える。`w_arr` は設定値（既定1.0、範囲0〜4）。
到来に乗る候補を安くし、外す候補を高くする。同期を促す方向だけを持つ。この項は skip の閾値比較には
含めない（`best_cost` から到来項を除いた値で skip を判定する）。回避（裏拍）の選好はScenarioの領域とし、
本機構は持たない。

不明。集合が空（`n = 0`）のときだけ到来項を全候補から外す。音群単位の除外理由（表が無い、
`issued_at > now`、`now >= horizon_end`、`horizon_end < at + w`、`periodic = false`、経過時間が不確実で
表が作られなかった、積分失敗）は音群ごとに報告する。

第2段の実装前に追記する事項: 試験用Hazard設定（18係数、標準化、horizon）の凍結値とhash、共有snapshotへの
載せ方と `groups[prototype]` 対応表の参照形。これらは結果を見て変えない。

### 4.6 集計と選択

```
cost(at) = displacement²
         + coupling * H(at)                         [§2の条件]
         + coupling * 6 * s * Ov(at) / norm         [§2の条件。power_k の出所は §4.2]
         + coupling * w_arr * (1 - P(at))           [第2段。設定で有効化し、既知の場合]
```

選択は費用最小、同点は格子順で先（既存）。skipの判定は到来項を除いた費用で行う（§4.5）。
報告には、基準候補の費用、選択候補の費用と `offset`、各項の値、入力ごとの状態を加える。
状態の語彙は、footprintが `body | proxy(unsupported) | proxy(absent) | proxy(stale)`、外部予測が
`known | unknown(reason)`、到来が `known | unknown(reason)` とする。

### 4.7 有界窓

- 候補格子は既存のまま打ち切らない。周期が長いVoiceでは正側候補が4秒を超え得るが、その場合も
  §2の `forecast` filter が `due + period + 80 ms` の支持を要求するため、外部項は全候補から外れる。
  これは既存の挙動であり、変更しない。
- footprintの `D` は4秒で打ち切る（§4.2）。評価の終端は `at + D` で、外部予測の地平を越える部分は
  観測平均へ落ちる（既存）。
- 到来項は `horizon_end` を超える候補で不明（§4.5）。
- 候補範囲（格子）、音長（`D`）、予測支持地平（外部4.0秒、到来 `horizon_end`）は別々の量である。

### 4.8 四つの時刻と締切

各入力について、観測終端（データの終わり）、発行時刻（計算した時点）、受領時刻（Voiceが受け取った時点）、
失効時刻を区別する。

- 外部予測: 観測終端は `observed_frame`、締切は窓の `start >= available_through_frame`。同じhopで生成し
  受領する。遅延として `now - observed_frame` を報告する。
- footprint: 発行時刻 `requested_at`、受領時刻 `received_at <= now`。失効は識別の不一致のみ（§4.2）。
  受領前は `proxy(absent)`、識別不一致は `proxy(stale)`。遅延として `received_at - requested_at` を報告する。
- 到来: 発行時刻 `issued_at <= now`、失効は `now >= horizon_end`。同じ音群の複数版は `issued_at` が最大の
  ものだけを使う。
- footprintの近似の限界。代表値（kick 1.0、seed、rhythms既定、hold）で作るため、実際に鳴るtoneの
  energyとは異なる。差は§5.2の状態介入ではなく、A1の聴取と、I10のonset予測（実kick・実seed）との
  比較で別に報告する。
- 予約との関係。予約は選択時の入力で行い、予約後に新しい入力が届いても再選択しない（既存の契約を維持）。
  予約後にrecipeが変わった場合も予約は保持し、次の周期から新しいfootprintを使う。報告には選択時刻と
  発音時刻の両方を記す。
- 全ての入力は `now` 以前の観測だけから作る。候補時刻より後の観測は使わない。

### 4.9 入力が不明のときの規則

- footprintが `proxy(*)`: ADSR代理へ戻し、状態を報告する。比較は続ける。
- 外部予測が不明: 既存規則どおり `H` と `Ov` を全候補から外す。
- 到来が不明: 到来項を全候補から外す。
- 一部の候補だけで不明: その入力を全候補から外す。候補ごとに入力の有無を変えない。
- 全入力が不明: 費用は `displacement²` のみ。ただし `period_frames` は過去の追従値を保つので、
  `coupling = 0` の状態と同じ選択になるとは限らない。同じ状態上で費用が同じ形になる、という意味である。
- 不明は既知の0（無音、到来確率0）と区別し、報告に理由を残す。不明を理由に学習へ信用を与えない。

### 4.10 off基準

- 設定 `[temporal_onset_comparison]` は `Option`、既定 `None`。`None` では既存の経路をそのまま通す。
  64分割のADSR代理、打ち切りなし、演算順序、状態更新の全てを変えない。新しい経路は `Some` のときだけ
  分岐する。bit一致の検査は§5.4。
- `Some` の欄: `footprint = "body" | "proxy"`（省略時 `body`）、`arrival = false`（省略時）、
  `arrival_weight = 1.0`（範囲0〜4、範囲外は設定エラー）。`arrival = true` で到来表の配送が無い場合は
  不明として扱い、エラーにしない。
- `coupling = 0` は外部入力に由来する費用項（`H`、重なり、到来）を全て消し、周期追従も止める。
  `displacement²` は残る。`overlap_sensitivity = 0` は重なり項だけを消す。両者は既存の意味のまま。
- 環境変数は使わない。I10の `CONCHORDAL_DISABLE_CANDIDATE_ENERGY` は診断用で、ここでは参照しない。

## 5. 検査計画

I11の完了条件（[milestones §2](milestones.md)）に沿って、第1段は1〜7の合格と9の分布報告の完了を、
第2段は全項目を要件とする。

1. 入力一致。`candidate()` が使ったfootprint record（識別、`T0`、`received_at`）、外部予測（`observed_frame`）、
   到来表（`issued_at`）と、reportの記録が一致する。独立参照（Python）で各項を再計算し照合する。
2. 状態介入。同じ観測で `footprint = proxy` と `body` を切り替えたとき、費用差が§4.4の `power_k` の差だけで
   説明できる。到来項の有無も同じ。
3. 期限。`received_at > now` のrecord、識別不一致のrecord、`start < available_through_frame` の窓、
   `issued_at > now` または `horizon_end` 超の到来表が使われない。予約後に届いた入力で再選択しない。
4. off基準。二つに分ける。(a) 設定 `None` の実行で、音声（WAV）と学習系record（I10の`candidates_off.py`が
   比べる10種）が基準commitのbinaryとbit一致する。基準commitのSHAと対象artifactは第1段のtest開始時に
   登録fileへ固定する。(b) 候補計算は、同一の凍結packetを基準版と新版で再評価し、record内容をbit比較する。比較対象は
   `Record` の決定的なfield（候補の入力・窓・energy・状態・識別）とし、`processing_us`、`max_processing_us`、
   worker資源計測欄、`received_at` などの実時間欄を除く。除いたfieldは検査記録に列挙する。
   liveの候補recordは飽和で採否が変わるため（同一条件の再実行で件数が揺れる）、件数・drop・欠落は
   別集計とし、内容一致の対象にしない。
5. APIの作用（第1段でも必須）。二つの比較を行う。(a) 設定 `Some(footprint = body)` と
   `Some(footprint = proxy)` を、同じrecord・同じ `D`・同じ16 bin・同じ外部入力で比べ、選択候補の差と
   その音声差（WAV）が§4.4の `power_k` の差だけから生じることを示す。これがT1固有の作用の検査であり、
   これに差が出ない素材だけでは第1段を通過しない。(b) `Some` と `None` の差は分割数・打ち切りの差を含む
   ため、参考として記録する。素材はsine／harmonic／modal × 外部pulse。
6. 部分欠測と世代交換。footprintが `absent → body → stale → body` と遷移する実行で、状態の報告と
   使用値が一致する。身体世代の交換で旧recordが使われない。未完了中の識別変更で旧返却が捨てられ、
   最新識別で再送される。返却dropの後に再送される。
7. hop経路と飽和。候補選択は `advance_population` 内で走り、profileの `population_us` に入る。設定 `Some` と
   `None` で、`population_us`、`synthesis_us`、hop全体 `elapsed_us` それぞれの中央値・p99・最大と、予算超過
   hop数を4／16 Voiceの交互3反復で比べる。許容は各欄の中央値とp99が±5%以内、最大の差が1 ms以内、
   超過hop数が同数。flow条件でのfootprint返却遅延の分布（中央値・p99・最大）と `proxy(absent)` の比率、
   置換済み要求と返却dropの件数を報告する。
8. 到来（第2段）。CDF表がEngineの内部積分と格子点で一致する。`elapsed_seconds` の二値の不一致、
   `reset_unknown`、`periodic = false`、期限外が§4.5どおりに音群を集合から外し、空集合で不明になる。
9. 代表条件と実条件の差（第1段、分布報告）。12条件の実行で発行された各onset機会について、
   実際に発行された `ToneEnergy`（実kick、実seed、実rhythms、予約release。`energy_projection.rs` の
   `issued` と同じ凍結）を `added` とし、`retained = []`、bus 0、区間 `[onset, onset + D_rep)`
   （`D_rep` は同じ機会で使われた代表footprintの `D`）を16 binで offline に再投影する。新tone単独の
   energyであり、残存tone込みの候補評価のSummaryは使わない。両者を `power_k` へ正規化し、比較量は
   `max_k |power_k^actual - power_k^rep|` と、同じ外部予測で計算した `Ov(at)` の差の絶対値。
   実側で `coherent_energies` に `None` があるbinを含む機会は比較から除き、件数を報告する。
   両側が既知の無音なら差0とする。合否ではなく分布（中央値・p95・最大）を報告し、A1の材料とする。

## 6. 完了と採用の判定

- I11の元の依存「対応するR4・R5の合格」は、§1.3によりR4・R5が研究拡張へ移ったため、本体では各Iの
  契約検査（O03／O09）と作者の聴取（A1）に置き換わる。本登録は適合係数を持たず、R4の対象がない。
- 第1段の技術完了は§5の1〜7の合格と9の分布報告の完了で、主張は「代表onset条件付きの身体予測を実消費者へ返し、
  proxyに対する選択・音響差を示した」に限る。実kick・実seed・実rhythmsに条件付けた予測の消費は主張しない。
  第2段は§4.5の追記と§5の全項目。
- 生成作用の採用は、`None` と `Some` の可聴差を作者が聴いて判断すること（A1／A2）。それまで既定は
  `None` のままとし、O15（因果的な関係効果）・O18（liveの期待・行動判定）は主張しない。

## 7. 資源の前提と順序

2026-09-21の判断（推奨採用）により、候補workerの資源対策はR2に残し、I11は登録と数値核を先に進める。
footprint要求は専用queueと優先取り出し（§4.2）で候補packetの飽和から分離するが、worker本体は逐次であり、
処理中の候補packet一つ分の遅延は受ける。その影響は§5.7で実測してから述べる。第2段は到来表の配送が
要るため、第1段の後に置く。

## 8. 未決と限界

- footprintはonset時刻に対して不変な近似で、残存toneとの干渉とseed差を含まない。I10の順位一致とは別の
  主張であり、参加費用の入力としての妥当性はA1の聴取でしか判定しない。
- 帯域配分は静的 `own_profile` のまま。bank laneの周波数から帯域energyを作ることは第1段に含めない。
- 到来項の一様重みと、経過時間が不確実なときに不明へ落とす規則は宣言した規則で、根拠は工学的である。
- 到来項は同期方向だけを持つ。回避の選好はScenario側に置く。
- 本登録の数値（`w_arr = 1`、`w = width`、16 bin、4秒、代表kick 1.0、代表seed）は結果を見て調整しない。変更する場合は
  新しい登録として日付と理由を残す。


## 第1段の実装状態（2026-09-22）

登録どおりに第1段（T1 footprint）を実装した。実装はFable統括のもとOpusへ4 unitに分けて委譲し、統合後に
全体を検査した。commit前の実走確認は最小限で、§5の全項目は次の単位で取得する。

- 実装の所在。worker側の核は `src/life/action_candidates/footprint.rs`（`Recipe`／`Identity`／`Request`／
  `Record`／`compute`）と `energy.rs` の専用queue（容量64、候補packetより先に取り出す）。消費者側は
  `src/config.rs` の `[temporal_onset_comparison]`、`src/life/temporal_participation.rs` の `Some` 分岐
  （16 bin、`BodyFootprint`、状態語彙、report 8欄）、`phonation_engine.rs` の `FootprintTracker`（Voiceごとの
  未完了1件・置換済み印・再送）、`voice.rs` の `footprint_recipe`、`runtime/mod.rs` の要求送出と返却routing。
- 登録からの実装上の差。`Request` の入力は `Recipe` に束ねた。`computed_at` はworkerにhop時計が無いため
  `requested_at + 実測経過` で埋め、実時間欄として扱う。`hold` のhashは4秒clamp前の値。`BodySnapshot` と
  `RenderModulatorSpec` の `Serialize` はtest限定なので、識別hashは手書きの正準byte符号化をSHA-256に通す。
  `Unsupported` recordは `d_samples = 0` で保持し、proxyの `D` は `hold + release` の打ち切り値へ落とす。
  返却dropの解除経路は作っていない（未完了1件・容量64・毎hop drainのため生じない。runtimeにコメント）。
- 検査。全Rust 1,154成功・失敗0・35 ignored（`cargo test exit=0 @ 2026-09-22T12:47:17+09:00`）、
  `cargo clippy -- -D warnings` 通過。§5.6の遷移（absent→body→stale→body）、置換、再送、`Unsupported`／
  `BodySilent`、予約後の非再選択、`None`／`proxy` では要求ゼロ、を単体testで確認。
- §5.4(a)。`None` のrender 12条件を基準commit `32a6389` のbinaryと比べ、WAVは全条件でSHA一致、学習系10種の
  recordは新8欄を除いて全件一致（`target/i11-stage1-20260922/bit-identity.json`。基準側は新欄を持たないため、
  比較scriptは新欄を実時間欄と同じく除外する）。
- 実走。4 Voiceのsine／harmonic／modal flowで `body` と `proxy` のrenderを1回ずつ取得した。`body` では
  `footprint_source` が sine 53／56、harmonic 55／56、modal 53／56 で `body`、残りは `proxy(absent)`（最初の
  機会）または `proxy(stale)`（recipe変更直後）。全recordが16 bin。選択候補の分布は `body` と `proxy` で
  異なる（例: modal-flow-4 で `body` は前倒し候補7件を含み、`proxy` は含まない）。これは経路が動く確認であり、
  §5.5(a)の判定ではない。
- 測定script。`target/i11-stage1-20260922/`（`plan.json`、`register.py`、`acquire.py`、`bit_identity.py`、
  `api_effect.py`、`hop_path.py`、`representative_gap.py` の骨組み、`README.md`）。入力15件のSHAと基準commitを
  登録済み。

未実施（次の単位）。§5.1の独立参照、§5.2、§5.3の実走、§5.5(a)(b)の12条件、§5.7の交互3反復、§5.9の
offline再投影（Rust側の入口が未実装）。第1段の技術完了はこれらの通過後に判定する。第2段（到来項）は未着手。


## 第1段の測定（2026-09-22、§5.5と§5.7）

Opus agentに委譲して取得した。scriptの修正はなく、成果物は `target/i11-stage1-20260922/`
（`api-effect.json`、`hop-path.json`、`summary-20260922.md`、各render／profile）。判定はここで行う。

- §5.5(a) `body` 対 `proxy`（12条件のrender、同一入力）。選択候補が変わりWAVも変わった条件は6件:
  harmonic-flow-4／16／64（3／56、10／205、9／857）、modal-flow-4／16／64（16／55、44／204、67／853）。
  sine-flowの3条件は選択差0でWAV一致。sine-holdの3条件は発音機会が無く比較対象なし。
  T1固有の作用（代表footprintがproxyと異なる選択を生む）はharmonic／modalで観測され、sineでは出なかった。
  sineの16 bin footprintはADSR代理と形が近く、この素材では差が閾値を越えないためである。
- §5.7 `body` 対 `none`（4／16 Voice、report有無、交互3反復。登録範囲16組）。全欄合格は3組
  （sine-hold-4 report、sine-hold-16 report、sine-flow-16 no-report）、不合格26件。内訳は次のとおり。
  - `population_us` の中央値・p99: 4 Voice flowの3条件で+8.5〜+17.5%。中央値の絶対値は5〜9 µsで、識別hashと
    要求送出がそのまま乗った量。±5%の相対許容は、この小さい絶対値に対して厳しすぎる可能性があるが、
    許容は登録値のまま不合格として記録する。
  - `elapsed_us` の最大: sine-hold-4 no-reportとmodal-flow-4で1 ms超（+1.5〜+1.9 ms）。単発の最大値で、
    反復間のばらつきの範囲か本変更の影響かは、この3反復では分離できない。
  - `over_budget_hops`: harmonic-flow-4／16で不一致だが、超過1 hopはnone／proxy側の1反復に出ており、
    body側は全反復0。bodyが超過を増やした証拠ではない。
  - `synthesis_us` p99と `elapsed_us` p99の±5%超が数組（−14〜+19%）。synthesisは本変更が触らない区間であり、
    計測のばらつきの大きさを示す。
- footprint計数。要求＝完了で全件、drop 0。置換済みが要求の56〜88%（recipeの変化頻度が高く、要求の多くが
  返却前に識別を失う）。返却遅延の中央値10.7〜85.3 ms、最大277 ms（harmonic-flow-16）。`proxy(absent)` は
  harmonicの2条件で1〜2%、他は0。`none`／`proxy` では要求ゼロ。

判定。§5.5(a)はharmonic／modalで成立、sineでは不成立（差が出る素材が無い）。§5.7は登録許容で不合格が残る。
第1段の技術完了はまだ主張しない。次の単位で扱うのは、(1) `population_us` の増分を絶対値（µs）でも登録し、
相対±5%が意味を持つ下限を宣言すること、(2) 反復を増やして最大値のばらつきを分離すること、(3) 置換済み比率が
高い原因（recipe hashに入る量のうちhopごとに揺れるもの、例えば `amp` や `freq_hz` の平滑化中の値）を特定し、
識別を意図どおり「recipe変更時だけ」にすること。(3)は登録の識別規則の見直しになるため、変更する場合は
新しい登録として残す。§5.1〜5.3の独立参照と§5.9の再投影は未実施。



## 識別規則の改訂（2026-09-22）

§5.7で置換済み要求が56〜88%と高かった原因を特定し、§4.2の識別規則を改める。§8の「変更する場合は
新しい登録として日付と理由を残す」に従う改訂登録である。§4.2の本文は改訂前の登録として残す。

- 原因。`Voice::footprint_recipe` が組むrecipeに、発音中の状態としてhopごとに動く値が二つ入っていた。
  一時計測（4 Voice flow、sine／harmonic／modal、各4,760の連続hop対）で前hopから変わった欄は、modulatorの
  `initial_env_level` が4,642（97.5%）、`amp` が350〜506、`initial_state` が214だった。`freq_hz`（3〜7）、
  `hold`・`adsr`（各4）は実際のrecipe変更である。`initial_env_level`／`initial_state` はentrained包絡の現在値、
  `amp` は `compute_target_amp()`（`body.amp × gate × release × vitality`）である。返却遅延（数hop）より短い
  周期で識別が変わるため、要求の大半が返却前に置換済みになっていた。
- 包絡の現在値はfootprintに届かない。代表toneは `T0` にkick 1.0を予約し、`RenderModulator::kick_planned` が
  包絡を `(Attack, 0)` から始め直す。予測側の `ControlForecast::gain_at` もkick以降は `(Attack, 0)` から数え、
  初期値を読まない。autonomous pulseの位相は `AmplitudeModel` に入らない。`initial_state`／
  `initial_env_level`／`phase_0_1` だけを変えたrecipeのrecordは、3身体とも `energies`・`d_samples`・`state` が
  bit一致した（test `the_live_envelope_state_moves_neither_the_footprint_nor_the_pinned_identity`）。
- `amp` は倍率としてだけ働く。`amp` を0.61倍・1.7倍にすると `energies` は `amp²` 倍になり、正規化後の `power` の
  差は最大1.5e-14（丸め）、`d_samples` と `state` は一致した。footprintは `power_k = E_k / max_k E_k` だけを
  使うので、`amp` はfootprintの値を変えずに識別だけを変えていた。

改訂内容。

1. modulatorの代表化。`EntrainPulse` の `initial_state = Idle`、`initial_env_level = 0`、autonomous pulseの
   `phase_0_1 = 0` に固定してから識別と計算に使う（`footprint::representative_modulator`）。recordの値は
   変わらない（上記のbit一致）。`SeqGate` はspecに現在値を持たず、kickでtimerが戻るので変更しない。
   `DroneSway.phase` はkickで戻らず予測に効くため、生成入力として残す（下記の限界）。
2. 代表振幅。recipeから `amp` を外し、代表値 `1.0` を置く。代表kickと同じく固定した代表値として
   `representative.amp` に記録する。§4.2の「`amp` と `freq_hz` のf32 bit」は「`freq_hz` のf32 bitと代表振幅
   1.0」と読み替える。
3. 「`Tone::from_parts` が `None` を返す場合（amp 0、非有限値）は `unsupported`」のうちamp 0は、身体の
   `amp_scale = 0` の場合だけになる。gate・release・vitalityが0の間も、footprintは身体と包絡から作る。
   onsetの発行は従来どおり `target_amp > AMP_EPS` で止まるので、振幅0のVoiceが発音することはない。
4. 変更は `footprint_recipe` の中、`footprint_hold()?` より後だけにある。`None`／`proxy` の経路はここを
   通らないので変わらない。`body` の経路は、recordの値が丸め差を除いて同じまま受け取れるrecordが増えるため、
   `body`／`proxy(stale)` の選択比率が変わり、WAVも変わる。§5.5・§5.7の既存の測定値は改訂前の実装のものである。

測定（再現条件）。登録の12条件の入力（`target/i11-stage1-20260922/`）は今回の作業環境に無いため、同じ構成の
再現条件を別に作った。`temporal_mode("observe")`、`[temporal_body]` は恒等尺度、
`[temporal_onset_comparison] footprint = "body"`。`sine()`／`harmonic()`／`modal()` に
`.flow().cycles(3).adsr(0.03, 0.3, 0.5, 0.7)` を付け、`consonance(220, 880)` へ4／16 Voiceを12秒置く。
`conchordal --nogui --play=false --profile` の `candidate_energy` 計数を各3反復した。置換済み／要求の比率は次のとおり。

- 改訂前。4 Voiceで98〜99%（要求637〜1,331）、16 Voiceで86〜99%（要求1,890〜4,143）。
- 改訂1だけ。4 Voiceで69〜80%（要求48〜69）、16 Voiceで73〜84%（要求516〜680）。
- 改訂1と2。4 Voiceで0〜18%（要求11〜17）、16 Voiceで0〜14%（要求62〜95）。全条件でdrop 0、要求と完了は同数。

残る置換済みはharmonic／modalに出る。発音中に `freq_hz` が動く、実際のrecipe変更である。sineでは0。
これは§5.7の登録測定ではない。§5.5・§5.7は登録の12条件で再取得する。

限界。

- `DroneSway.phase` は生成入力として識別に残る。drone articulationのVoiceがparticipation clockを持つと、
  識別は毎hop変わる。今回の再現条件（entrained）には現れない。代表位相を置くかはfootprintの値を変える判断
  なので、別の登録で扱う。
- pitchが連続して動く間（glide中）は `freq_hz` が毎hop変わりうる。実際の生成入力なので識別に残す。


## 第1段の再取得（2026-09-23、識別規則改訂後の§5.5と§5.7）

識別規則の改訂commit（`77e07b6`）を反映したbinaryで、§5.5(a)と§5.7を再取得した。旧計測（`target/i11-stage1-20260922/`）は上書きしていない。新しい取得先は `target/i11-stage1-rerun-20260923/`。

- HEADは `77e07b6`（直前は識別規則改訂本体の `db1eae4`）。release binaryを同日に再buildした
  （`conchordal` SHA-256 `313ef557a8175730f247713d6ad909c0b90e03bac0eba65bcb17fe452af37e5e`、
  `conchordal-render` SHA-256 `199fec1385aa75811616205299e35551fbd01e6ae8ff639c65c142d56c718d39`）。
- 入力15件のSHA-256は `plan.json` の登録値と全件一致（不一致0件）。`register.py` は再実行していない
  （入力は2026-09-22の凍結のまま）。`plan.json`・`README.md`・全script（`acquire.py`／`api_effect.py`／
  `hop_path.py`／`bit_identity.py`／`representative_gap.py`／`register.py`）は複写のみで変更していない。
- 取得範囲は今回の依頼どおり§5.5(a)と§5.7に限る。§5.5(a)はrender modeで`body`対`proxy`を12条件全て
  取得した（旧計測と同じ範囲）。§5.7はinstrument modeの`body`対`none`を、登録範囲である4／16 Voiceの
  8条件（sine-hold-4/16、sine-flow-4/16、harmonic-flow-4/16、modal-flow-4/16）だけ、report有無×3反復で
  取得した。`proxy`と64 Voiceの8条件はこの再取得に含めていない（§5.7の許容判定は`body`対`none`のみを
  要求し、64 Voiceは`registered_by_5_7 = false`で判定対象外のため）。variantの反復順回転は
  `body`／`none`の2値で行い、登録文書にある3値回転（none/body/proxy）とは異なる。全120件
  （render 24件、report/no-report 96件）の実行はすべて exit 0、drop・失敗0件。

### §5.5(a): `body` 対 `proxy`（12条件、render）

| 条件 | 旧（改訂前） differing/comparable | 新（改訂後） differing/comparable | WAV一致性 |
|---|---|---|---|
| sine-hold-4/16/64 | 0/0 | 0/0 | ok（発音機会なし） |
| sine-flow-4/16/64 | 0/56, 0/205, 0/857 | 0/56, 0/205, 0/857 | ok |
| harmonic-flow-4/16/64 | 3/56, 10/205, 9/857 | 3/56, 10/205, 8/857 | ok |
| modal-flow-4/16/64 | 16/55, 44/204, 67/853 | 25/55, 44/204, 66/851 | ok |

`wav_consistency`（選択差の有無とWAV差の有無が一致するか）は12条件全てで`ok`。判定は旧計測と同じく、
harmonic／modalでT1固有の作用が観測され（`t1_effect_observed`は6/12条件で真、旧計測と同じ内訳）、
sineでは観測されない。

数値そのものは複数条件で変化した。原因は改訂の意図どおり、`body`のfootprint_sourceが
`proxy(stale)`／`proxy(absent)`へ落ちる頻度が下がり、より多くの機会で実際に代表footprintが使われる
ようになったことである。modal-flow-4では旧計測でfootprint_sourceが`body`53／`proxy(stale)`3だったのに対し、
新計測は`body`56／`stale`0（全件解決）。この3件の解決だけで、下流の状態を通じてdiffering数が16→25へ
9件動いた（単一の入力差が後続の候補選択へ連鎖する影響で、footprint_source変化数と1対1ではない）。
harmonic-flow-4/16はfootprint_source内訳が変わっても（例: harmonic-flow-16は旧`stale`1／`absent`1→新
`absent`4、旧`body`203→新`body`201）、differing数（3、10）は不変だった。harmonic-flow-64／modal-flow-64
はdiffering数・comparable数がわずかに動いた（9→8、853→851など）。64 Voiceは実行が長くhopが多いため、
上流での小さな選択差が後続の発音機会数へ連鎖した結果と考えられる。64 Voiceも§5.5(a)の登録範囲
（12条件のrender）に含まれ、どちらの条件でも選択差は残るので、T1の作用の判定は変わらない
（2026-09-23訂正。当初「登録範囲外」と書いたのは§5.7の範囲との取り違え）。

### §5.7: `body` 対 `none`（4／16 Voice、report有無、交互3反復、登録8条件）

全16組（8条件×report有無）中、全欄合格は7組（sine-hold-4 report／no-report、sine-hold-16
report／no-report、sine-flow-16 no-report、harmonic-flow-16 report／no-report）。旧計測の全欄合格3組
（sine-hold-4 report、sine-hold-16 report、sine-flow-16 no-report）と比べ、sine-hold-4／16のもう一方の
report/no-reportとharmonic-flow-16の両方が新たに合格した。不合格9組の内訳:

| 条件・mode | 不合格欄 |
|---|---|
| sine-flow-4 report | population_us中央値+15.3%、p99 -15.4% |
| sine-flow-4 no-report | population_us中央値+12.4%、p99 -14.3% |
| sine-flow-16 report | synthesis_us p99 +5.7% |
| harmonic-flow-4 report | population_us p99 -12.1% |
| harmonic-flow-4 no-report | population_us中央値+11.3%、p99 -6.2%、over_budget_hops（body=[0,0,0]、none=[1,0,0]） |
| modal-flow-4 report | population_us中央値+23.4%、p99 -6.0%、elapsed_us最大+1802.8 µs |
| modal-flow-4 no-report | population_us中央値+9.0%、elapsed_us最大+1638.6 µs |
| modal-flow-16 report | population_us p99 +7.1%、elapsed_us p99 +8.9%、over_budget_hops（body=[1,0,0]、none=[0,0,0]） |
| modal-flow-16 no-report | elapsed_us p99 +9.7% |

不合格の性質は旧計測と同じ二種に分かれる。(1) `population_us`の中央値・p99の相対差は、絶対値が
5〜9 µs台のまま±5%相対許容に対して依然厳しい（旧計測で指摘した論点は今回も未解決、許容は変更していない）。
(2) `elapsed_us`最大の1 ms超過（modal-flow-4）とover_budget_hopsの1 hop差（harmonic-flow-4 no-report、
modal-flow-16 report）は単発の値であり、3反復では反復間ばらつきと本変更の影響を分離できない
（旧計測と同じ限界）。over_budget_hopsの差はいずれも`none`側または`body`側の1反復にのみ出ており、
一貫して`body`が超過を増やした証拠にはならない。

footprint計数（`body` variant、report mode、3反復合計）は次のとおり。要求＝完了が全条件で成立し、drop 0。

| 条件 | 要求 | 置換済み(superseded) | 置換済み比率 |
|---|---|---|---|
| sine-hold-4/16 | 0 | 0 | ― |
| sine-flow-4 | 24（8×3） | 0 | 0% |
| sine-flow-16 | 96（32×3） | 0 | 0% |
| harmonic-flow-4 | 24 | 6（2,2,2） | 25% |
| harmonic-flow-16 | 96 | 22（7,7,8） | 23% |
| modal-flow-4 | 24 | 0 | 0% |
| modal-flow-16 | 96 | 0 | 0% |

旧計測の置換済み56〜88%から、登録8条件では0〜25%へ下がった（改訂の目的どおり）。返却遅延
（`footprint_received_at - footprint_requested_at`、report jsonlから算出）は次のとおり。

| 条件 | 件数 | 中央値 | p99 | 最大 | footprint_source内訳 |
|---|---|---|---|---|---|
| sine-flow-4 | 168 | 0.0 ms | 0.0 ms | 0.0 ms | body 100% |
| sine-flow-16 | 615 | 0.0 ms | 0.0 ms | 0.0 ms | body 99.5%、proxy(stale) 0.5% |
| harmonic-flow-4 | 168 | 32.0 ms | 53.3 ms | 53.3 ms | body 100% |
| harmonic-flow-16 | 615 | 85.3 ms | 106.7 ms | 106.7 ms | body 98.9%、proxy(stale) 0.3%、proxy(absent) 0.8% |
| modal-flow-4 | 168 | 21.3 ms | 32.0 ms | 32.0 ms | body 100% |
| modal-flow-16 | 612 | 32.0 ms | 64.0 ms | 64.0 ms | body 99.5%、proxy(stale) 0.5% |

返却遅延の分布自体は旧計測（中央値10.7〜85.3 ms、最大277 ms）と同程度の桁で、harmonic-flow-16が
最も遅い点も変わらない。`proxy(absent)`はharmonic-flow-16の0.8%のみで、他条件は0（旧計測の
「harmonicの2条件で1〜2%」よりわずかに下がった）。

### 判定

§5.5(a)は改訂後binaryでも成立を維持する（harmonic／modalでT1固有の作用、sineでは不成立）。数値は
置換済み比率の低下により旧計測から動いたが、判定を左右する`t1_effect_observed`の内訳は変わっていない。
§5.7は登録許容で不合格が残る（7/16合格、旧3/16から改善）。第1段の技術完了はまだ主張しない
（§5.1〜5.3の独立参照、§5.9のoffline再投影は未実施のまま）。

次の単位で扱う項目を更新する。(1) `population_us`の増分を絶対値（µs）でも登録し、相対±5%が意味を持つ
下限を宣言すること（未解決、今回も同じ論点が残った）。(2) 反復を増やしてelapsed_us最大値とover_budget_hops
のばらつきを分離すること（未解決）。(3) 置換済み比率が高い原因の特定と識別規則の改訂は今回**完了**
（`77e07b6`）。次に残るのは(1)(2)と、§5.1〜5.3の独立参照、§5.9のoffline再投影（Rust側入口が未実装）である。


## §5.7の許容にA/A雑音幅の下限を加える（2026-09-23、取得前の登録）

§5.7の登録許容（中央値・p99が±5%、最大の差が1 ms、超過hop数が同数）は、2026-09-22と09-23の測定で
繰り返し`population_us`を落とした。その欄は絶対値が5〜9 µs台で、相対±5%が0.3 µs程度を意味する。
許容が機械の雑音より狭いのか、`body`が本当に遅いのかを、結果を見てから許容を緩めて判定すると事後調整に
なる。そこで**同じ設定どうしの比較（A/A）の差を雑音幅とし、その最大値を許容の下限（floor）に置く**規則を、
`body`対`none`の取り直しの**前**に登録する。§8の「変更する場合は新しい登録として日付と理由を残す」に従う
改訂登録である。元の許容は削除せず、判定を併記して残す。

### A/Aの取得

比較の両側とも`None`設定である。取得先`target/i11-stage1-aa-20260923/`の`inputs/config-body.toml`は
同ディレクトリの`inputs/config-none.toml`のバイト複写で（`plan.json`の`aa_substitution`に両者の
SHA-256と一致の検証を記録）、`hop_path.py`が出す`tolerance_body_vs_none`は設定が同一の2本の実行の差に
なる。`acquire.py`と`hop_path.py`は`target/i11-stage1-20260922/`からの無改変の複写である（SHA-256一致を
検証済み）。範囲は§5.7の登録範囲と同じ16組（8条件＝4身体条件×4／16 Voice、report有無）、交互3反復、
96実行すべてexit 0・失敗0。生データは`runs-aa.json`、`hop-path-aa.json`、floorの算出は`aa-floor.json`。
`target/`はgit管理外なので、これらはこの機械の上にだけある。

A/Aは元の許容で16組中6組が不合格になった。**設定が同一なのだから、落ちた欄は許容がその欄の雑音幅より
狭いことを意味する。**

| 組 | 元の許容で落ちた欄 |
|---|---|
| sine-hold-16 report | over_budget_hops（[0,0,0]対[0,0,1]） |
| sine-flow-16 no-report | over_budget_hops（[0,0,0]対[0,0,1]） |
| harmonic-flow-4 report | population_us中央値+8.5%、over_budget_hops（[1,0,0]対[0,0,0]） |
| harmonic-flow-4 no-report | population_us中央値−5.4% |
| modal-flow-4 report | population_us中央値−9.4%、p99 −13.3% |
| modal-flow-4 no-report | population_us p99 −11.3% |

### floorの定義と値

floorは欄ごとに、16組のA/A差の絶対値の**最大値**とする。時間欄は`|A − A|`をµsで、超過hop数は
3反復の合計の差`|Σbody − Σnone|`を件数で取る（元の許容はsort済みlistの同値比較なので、差を数える
規則をここで定義しておく）。

| 欄 | floor | 最大を与えた組 |
|---|---|---|
| population_us 中央値 | 1.270 µs | sine-flow-16 report |
| population_us p99 | 70.181 µs | modal-flow-4 report |
| population_us 最大 | 468.202 µs | sine-flow-16 report |
| synthesis_us 中央値 | 62.571 µs | sine-flow-16 report |
| synthesis_us p99 | 73.500 µs | harmonic-flow-16 no-report |
| synthesis_us 最大 | 584.653 µs | modal-flow-16 no-report |
| elapsed_us 中央値 | 71.580 µs | modal-flow-16 report |
| elapsed_us p99 | 127.040 µs | harmonic-flow-16 no-report |
| elapsed_us 最大 | 648.174 µs | sine-hold-16 report |
| over_budget_hops | 1 hop | sine-hold-16 report |

### 登録する規則

`body`対`none`の判定を、欄ごとに次で行う。

- 中央値・p99: `|body − none| <= max(0.05 × none, floor)`
- 最大: `|body − none| <= max(1 ms, floor)`
- 超過hop数: `|Σbody − Σnone| <= floor`

floorが効くのは`0.05 × none`がfloorを下回る欄、つまり絶対値の小さい欄だけである。`population_us`の
中央値（`none`が5 µs台なら5%は0.3 µs）ではfloorが効き、`elapsed_us`の中央値（`none`が2 ms台なら5%は
100 µs）では従来どおり5%が効く。最大の3欄はfloorがいずれも1 ms未満なので、1 msが引き続き効く。

### この規則の限界

- floorは**この機械・この時点・この16組**での雑音幅である。別の機械、別の負荷、別の条件集合では
  取り直す。この文書の値を普遍的な下限として他所へ持ち出さない。
- 3反復1回分のA/Aから取った最大値であり、雑音幅の上界を証明したものではない。A/Aを増やせばfloorは
  上がりうる。
- 組ごとではなく16組全体の最大を取っているので、静かな組にはfloorが緩すぎる。組ごとのfloorは
  データが足りないため置かない。
- floorはあくまで「この差は雑音と区別できない」の線であって、`body`が`none`より速い保証ではない。
  floor以内の差は**合否の判定に使わない**という意味しか持たない。


## §5.7の取り直しと二つの許容による判定（2026-09-23）

上の規則を登録した後に、`body`対`none`を§5.7と同じ手順で取り直した。取得先は
`target/i11-stage1-rerun2-20260923/`（8条件×report有無×交互3反復、96実行すべてexit 0・失敗0）。
入力は登録凍結のバイト複写、`acquire.py`・`hop_path.py`は無改変の複写（いずれもSHA-256一致を検証済み）。
元の許容の判定は`hop-path.json`、A/A規則の判定は同ディレクトリの`aa_floor_check.py`が出す
`aa-floor-verdict.json`。floorは`target/i11-stage1-aa-20260923/aa-floor.json`から読むだけで、
この取得の結果からは再計算していない。

合格は**元の許容で6/16組、A/A規則で8/16組**。判定が変わったのは2組だけである
（sine-flow-4 report、harmonic-flow-16 no-report、いずれも不合格→合格）。

### `population_us`中央値は決着した

規則を作る動機になった欄は、A/A規則では**16組すべて合格**する。相対差は最大+21.4%
（harmonic-flow-4 report）まで出るが、絶対差は1.23 µsでfloorの1.27 µsに収まる。

| 組 | body | none | 差 | 相対 | 許容 |
|---|---|---|---|---|---|
| harmonic-flow-4 report | 6.97 µs | 5.74 µs | 1.23 µs | +21.4% | 1.27 µs（floor） |
| modal-flow-4 no-report | 6.09 µs | 5.17 µs | 0.92 µs | +17.8% | 1.27 µs（floor） |
| sine-flow-4 report | 5.88 µs | 5.13 µs | 0.75 µs | +14.6% | 1.27 µs（floor） |
| sine-flow-16 no-report | 112.65 µs | 115.73 µs | 3.08 µs | −2.7% | 5.79 µs（5%） |

`none`が5 µs台の4／16 Voiceのhold・flow条件ではfloorが効き（16組中10組）、`none`が115〜120 µs台の
16 Voice flow条件では従来どおり5%が効く。**識別hashと要求送出が`population_us`の中央値へ乗せる量は、
この機械のA/A雑音と区別できない**、というのが今回の結論である。floorが5%より広く効いたのは
`population_us`中央値10組・同p99 8組・`synthesis_us`中央値8組・同p99 8組・`elapsed_us` p99 2組で、
最大の3欄はfloorがいずれも1 ms未満のためどこでも効いていない。

### A/A規則でも残る8組

| 組 | 落ちた欄（差 > 許容） |
|---|---|
| sine-hold-16 no-report | synthesis_us p99 184.8 > 98.5 µs |
| sine-flow-4 no-report | elapsed_us中央値 136.6 > 99.2 µs、同p99 152.4 > 130.1 µs |
| harmonic-flow-4 report | population_us p99 78.5 > 70.2 µs |
| harmonic-flow-4 no-report | population_us p99 95.2 > 70.2 µs |
| modal-flow-4 report | elapsed_us最大 1781.4 > 1000.0 µs |
| modal-flow-4 no-report | elapsed_us最大 2036.9 > 1000.0 µs |
| modal-flow-16 report | population_us p99 168.3 > 90.4 µs、elapsed_us p99 546.0 > 404.2 µs |
| modal-flow-16 no-report | population_us p99 99.9 > 92.9 µs、elapsed_us p99 597.7 > 400.2 µs |

性質は三つに分かれる。

1. **bodyの経路では説明できない不合格。** sine-holdの2条件は`body`設定でもfootprint要求が0件である
   （本取得の計数で確認。下表）。要求を一件も出していない条件で`synthesis_us`のp99が落ちるのは、
   本変更の作用ではありえない。sine-flow-4も要求24件・置換0件・返却遅延0.0 msで、`elapsed_us`の
   中央値が136 µs動く理由が本変更側に無い。これらは、**1回分のA/Aから取ったfloorがその欄の雑音幅を
   まだ下回っている**ことを示す。登録時に限界として書いたとおりで、floorは上界の証明ではない。
2. **再現した唯一の候補。** modal-flow-4の`elapsed_us`最大は、本取得で1781.4／2036.9 µs、同日午前の
   取り直しでも1802.8／1638.6 µsと、独立した2回の取得の4組すべてで1 msを超えた。同じ組のA/A雑音は
   262.6／301.1 µsで、超過はその6倍前後にあたる。現在の証拠では雑音に帰せられない。§5.7で残る
   実質的な問題はこれである。
3. **判定を保留する残り。** harmonic-flow-4とmodal-flow-16のp99の超過はfloorの1.1〜1.9倍で、
   取得2回では反復間ばらつきと本変更の作用を分離できない。

### footprint計数（`body`、report、3反復合計）

| 条件 | 要求 | 置換済み | drop | 返却遅延 中央値／p99／最大 | footprint_source |
|---|---|---|---|---|---|
| sine-hold-4／16 | 0 | 0 | 0 | ― | ― |
| sine-flow-4 | 24 | 0 | 0 | 0.0／0.0／0.0 ms | body 100% |
| sine-flow-16 | 96 | 0 | 0 | 0.0／0.0／0.0 ms | body 99.5%、proxy(stale) 0.5% |
| harmonic-flow-4 | 24 | 6 | 0 | 32.0／53.3／53.3 ms | body 100% |
| harmonic-flow-16 | 96 | 22 | 0 | 85.3／106.7／106.7 ms | body 99.2%、proxy(absent) 0.5%、proxy(stale) 0.3% |
| modal-flow-4 | 24 | 0 | 0 | 21.3／32.0／32.0 ms | body 100% |
| modal-flow-16 | 96 | 0 | 0 | 32.0／64.0／64.0 ms | body 99.5%、proxy(stale) 0.5% |

要求＝完了、drop 0、置換済み0〜25%。同日午前の取り直しの値と件数・遅延とも一致しており、識別規則の
改訂の効果は取得をまたいで安定している。

### 判定

§5.7は元の許容でも新しいA/A規則でも全欄合格に至っていない（6/16、8/16）。ただし不合格の中身は
`population_us`の中央値から、p99・最大へ移った。中央値の増分は改訂した受入規則（A/A floor）に適合し（雑音と区別できないことの証明ではない。2026-09-23訂正）、
残る実質的な問題はmodal-flow-4の`elapsed_us`最大（1 msの約1.8〜2.0倍、2回の取得で再現）に絞られた。
第1段の技術完了は判定しない（§5.1〜5.3と§5.9は別マシンで進行中）。

次の一単位。(1) modal-flow-4の`elapsed_us`最大の超過が、どのhopで、`advance_population`の
どの区間で起きているかをprofileから特定する。(2) A/Aを複数回に増やし、floorを最大値でなく分布で
置き直す（現在のfloorは`synthesis_us`と`elapsed_us`で雑音幅を下回っている）。(3) §5.1〜5.3の独立参照と
§5.9のoffline再投影（別マシン）。


## 独立参照による入力一致・状態介入・期限の検査（2026-09-23）

§4.6は報告に「各項の値」と入力ごとの状態を加えると登録しているが、第1段の実装は基準候補と選択候補の
費用しか出していなかった。独立参照が費用を再計算するには候補ごとの項と入力が要るため、登録どおりの
報告を追加し、それを読む独立参照を作った。

- 報告。`participation_decision` は `[temporal_onset_comparison]` が `Some` で、かつreportが付くときだけ、
  決定ごとに一件出る。`now`、`due_frame`、`period_frames`、`width`、`earliest`、係数、`memory`、
  `own_band_energy`、代理の保持と包絡、footprintの出所・識別・現在の識別・`requested_at`／`received_at`・`D`・
  16 binの遅延と `power_k`、外部予測の `observed_frame` と `available_through_frame`、格子23点それぞれの
  `at`・`displacement²`・文脈項の距離と入力・重なり和と16 binの外部energy・費用、選択、`skipped_cycles` と
  skipの有無を持つ。`body_footprint` は配送された各recordを、置換済みで捨てたかどうかと共に出す。
  `None` の経路と、reportを付けない実行では何も出さない。
- 独立参照。`scripts/verify_i11_stage1.py` はRustを読まず、登録の式を書き直して照合する。§5.1は、使った
  footprintが配送されたrecordと一致すること、代理のpowerとspanが§4.2・§4.3に従うこと、全候補の格子・
  各項・費用・選択・skip判定の再計算、後に出る `participation_context` が同じ決定を述べること。§5.2は、
  同じ決定のまま `power_k` だけを代理へ置き換え、選択が変わる決定を数える。§5.3は、未来のrecordを
  使わないこと、現在の識別と異なるrecordをbodyとして使わず `proxy(stale)` へ落とすこと、置換済みrecordを
  使わないこと、`available_through_frame` より前の外部energyと窓を使わないこと、予約が残る間に再選択
  しないこと。整合したreportが全検査を通ることと、7種の違反をそれぞれの検査が捉えることを
  `tests/test_verify_i11_stage1.py` で確認した。
- 結果（再現条件）。識別規則の改訂節と同じ4 Voice flow（sine／harmonic／modal、habitat bus）を
  `body` と `proxy` でrenderし、6本のreportで全検査が通った。各reportの決定は112〜113件、候補2,576〜2,599件、
  そのうち文脈項が効いた候補2,369件、重なり項が効いた候補2,461件、外部energy 39,376点。§5.2で選択が
  変わった決定は、sine 0／107、harmonic 19／102、modal 37／107。skip、`proxy(absent)`、`proxy(unsupported)` は
  この実行に現れず、Rust側の単体testだけが扱う。登録の12条件での実行は未実施。
- 変更前との一致。本変更の前の実装（`77e07b6`）と比べ、`None` では旧版どうし・新版どうし・旧版と新版の
  いずれもWAVと `onset` が一致した。`body` でも、habitatとpresentationの両busへ送るharmonic条件で同様に一致した。
  報告の追加は決定を変えない。
- 見つかった性質。`body` のrenderは再実行で同じにならない。同じbinaryの再実行で `footprint_received_at` が
  107件中101件異なり、`proxy(stale)` と `body` が入れ替わる決定もあった。footprintの配送がworkerの実時間に
  依存するためで、第1段の実装からある性質である。この条件では `onset` は一致したが、habitat専用のharmonic
  条件では旧版と新版で `onset` が食い違った（受領時刻の揺れで説明できる）。§5.5(a)のように別々の実行を
  比べる検査はこの揺れを含み、§5.2の反実仮想（同じ決定で `power_k` だけを入れ替える）は含まない。
- 資源。報告の追加はreport付きの実行だけでhop経路の仕事を増やす。§5.7のreport有りの8組は、本変更の後に
  取り直す必要がある。A/Aのfloorは `none` どうしなので影響を受けない。


## offline renderでのfootprint配送の決定化（2026-09-23、登録）

`body` のrenderは、footprintの受領時刻がworkerの実時間に依存するため、再実行で同じにならない（前節）。
§5.5(a)のように別々の実行を比べる検査と、研究assayの再現にはこれが障害になる。§4.8の時刻の扱いを、
offline renderに限って次のように改める。§8に従う改訂登録である。

- 規則。`conchordal-render` だけで、hopの冒頭で出した要求を、同じhopの受け取り処理（集団更新の後）で
  全件受け取る。workerの計算が終わるまで待つ。受領時刻は要求と同じhopの `now` で、決定で使うのは次の
  hopから。workerはfootprint要求を先着順に処理し、Voiceごとの未完了は1件なので、受け付けた要求は全件
  返り、返却queue（容量64）は溢れない。
- 対象外。instrumentは `--play=false` でも実時間の配送のままとする。§5.7の負荷にworkerを待つ時間を
  混ぜないためである。
- §4.8の読み替え。offline renderでは受領の遅延が0 hop、決定への反映が1 hop後になる。liveで測った
  受領遅延の分布（中央値10.7〜85.3 ms、最大277 ms）はofflineには現れず、offlineの `proxy(absent)`／
  `proxy(stale)` の比率はliveより低くなる。offlineの結果は最短の配送を仮定したものである。
- 確認。同じ設定の `body` のrenderを2回行い、WAVと、`onset`・`participation_decision`・
  `participation_context`・`body_footprint`（実時間の欄 `computed_at` を除く）が一致する（test
  `body_footprint_renders_repeat_exactly`）。再現条件の3身体で独立参照の全検査が通り、全決定が `body` を
  使った。§5.2で選択が変わる決定は、sine 0／107、harmonic 16／107、modal 37／107。
- 影響。`body` のoffline renderの結果が変わるため、§5.5(a)は取り直す。`None` の経路は要求を出さないので、
  待ちも生じない。


## A/A floorを複数passの分布から置き直す（2026-09-23、取得前の登録）

09-23に登録したfloorは1回分のA/A（16組）の最大値だった。同日の`body`対`none`で、**本変更が触れない欄が
そのfloorを超えて落ちた**。sine-holdの2条件は`body`設定でもfootprint要求が0件なのに`synthesis_us`のp99が
落ち、sine-flow-4は要求24件・置換0件・返却遅延0.0 msなのに`elapsed_us`の中央値が136 µs動いた。1回分の
最大値がその欄の雑音幅を下回っていたということである。標本を増やしてfloorを置き直す。取得の前に、
pass数とfloorの取り方をここで固定する。§8の「変更する場合は新しい登録として日付と理由を残す」に従う
改訂登録である。

### 固定する取得の形

- **A/A 1本（pass）** = §5.7の登録範囲と同じ16組（8条件＝4身体条件×4／16 Voice、report有無）を交互3反復。
  両側とも`None`設定（取得先の`inputs/config-body.toml`は`inputs/config-none.toml`のバイト複写。
  SHA-256の一致を検証して`plan.json`に記録する）。`acquire.py`・`hop_path.py`は無改変で使う。
- **pass数 P = 3**。連続して取得し、計測中は他の重い処理を走らせない。
- 1 passにつき、`hop_path.py`が3反復を「各反復の統計値の中央値」へ畳んだ値を使う（§5.7と同じ畳み方）。
  1組1 passから欄ごとに差が1つ出る。時間欄は`|body側 − none側|`をµs、超過hop数は3反復の合計の差
  `|Σ − Σ|`を件数で取る。
- 標本数は欄ごとに **P × 16 = 48**。

### floorの取り方

**floor[欄] = 48標本の最大値**とする。分布のp50・p95・pass毎の最大も記録するが、判定には使わない。

最大値を採る理由。floorは「この差は雑音と区別できない」の線であり、超えた差だけを議論の対象にしたい。
上側の分位点（例えばp95）を採るとA/A自身が欄ごとに5%落ちる。欄は10個あるので、組単位では4割前後が
落ちる勘定になり、雑音の線として使えない。

**収束の確認（判定には使わない）**。pass 1〜2の32標本から取ったfloorを pass 3の16標本に当て、超えた
件数を報告する。0件に近ければfloorは足りており、多ければP=3でもまだ足りない。この数字でfloorを
調整することはしない（それは事後調整になる）。足りないと分かった場合は、次の登録でPを増やす。

### 判定の規則（形は09-23の登録から変えない）

- 中央値・p99: `|body − none| <= max(0.05 × none, floor)`
- 最大: `|body − none| <= max(1 ms, floor)`
- 超過hop数: `|Σbody − Σnone| <= floor`

§5.7の取り直しは、**元の登録許容・09-23の1 pass floor・本節の3 pass floorの三つすべてで判定して併記する**。
どれか一つで置き換えることはしない。

### report有りの組の読み方（取得前に宣言する）

`aebae2b`の`participation_decision`は`[temporal_onset_comparison]`が`Some`かつreportが付くときだけ出る。
`none`は`None`設定なので、同じreport付きでも決定の報告を出さない。したがってreport有りの8組の
`body`対`none`は、**footprintの経路の費用と§4.6の決定報告の費用を合わせて**測っている。footprintの経路
だけを見るのはreport無しの8組である。A/Aは両側とも`None`で決定報告を含まないので、floorは雑音の推定
として有効なまま使える（report有りの`body`側に乗る決定報告の費用は、雑音ではなく実際の仕事である）。
report有りで差が出ても、それをfootprintの経路の費用と読み替えない。


## §5.5(a)の取り直し（2026-09-23、`a6637db`の決定的配送）

offline renderでのfootprint配送を決定化した`a6637db`で、登録12条件の§5.5(a)を取り直した。取得先は
`target/i11-stage1-effect-a6637db-20260923/`（12条件×`body`／`proxy`、report付きrender、24実行すべて
exit 0・失敗0）。入力は登録凍結のバイト複写、`acquire.py`・`api_effect.py`は無改変の複写（SHA-256一致を
検証済み）。

### 先に再現性を確かめた（modal-flow-4、1条件）

同じ設定の`body` renderを2回行って比べた（`target/i11-stage1-repeat-check-20260923/`）。

- WAVはSHA-256が一致（`6efaf95b…`）。
- record型28種のうち**24種が完全一致**。`onset` 57件、`participation_decision` 60件、
  `participation_context` 56件、`participation_outcome` 56件、`population_step` 2,364件、
  学習系recordを含む。§5.5(a)が読む欄（`selected_offset`・`footprint_source`・`selected_cost`・
  `reference_cost`・`onset_frame`）は56件すべて一致した。
- 一致しなかったのは6種で、内訳は`body_footprint`の`computed_at`のみ（8件中6件、`a6637db`が
  実時間の欄として明記したもの）、`hop_timing`（611件、`elapsed_us`など実時間）、`body_observation`
  （62件、`max_processing_us`と`worker_resources.*_ns`）、`self_sound_observation`（1件、同種）、
  `temporal_observation`（126件、`worker_resources`のhistogram）、`body_candidate_energy`（86件）。
- **`body_candidate_energy`は実時間の欄ではない。** I10のcandidate energy workerが非同期に評価した
  内容そのもの（区間・energy・binの数）が run 間で変わる。decisionにもWAVにもこの条件では届いて
  いないが、report全体のbyte比較は今後も一致しない。再現するのは「WAVと決定に関わるrecord」であって、
  report全体ではない。

### 結果（`77e07b6`の実時間配送との比較）

| 条件 | `77e07b6` 選択差 | `a6637db` 選択差 | `body`のfootprint源の変化 |
|---|---|---|---|
| sine-hold-4／16／64 | 0/0 | 0/0 | 発音機会なし |
| sine-flow-4 | 0/56 | 0/56 | body 56（変化なし） |
| sine-flow-16 | 0/205 | 0/205 | body 204, stale 1（変化なし） |
| sine-flow-64 | 0/857 | 0/857 | body 855, stale 2（変化なし） |
| harmonic-flow-4 | 3/56 | 3/56 | body 56（変化なし） |
| harmonic-flow-16 | 10/205 | 10/205 | body 201, absent 4 → body 204, stale 1 |
| harmonic-flow-64 | 8/857 | 8/857 | body 839, absent 15, stale 4 → body 856, stale 2 |
| modal-flow-4 | 25/55 | 25/55 | body 56（変化なし） |
| modal-flow-16 | 44/204 | 44/204 | body 203, stale 1（変化なし） |
| modal-flow-64 | 66/851 | 66/851 | body 843, stale 11 → body 852, stale 2 |

**選択差の件数は12条件すべてで一致した。** `t1_effect_observed`も6/12で変わらない（harmonic／modalで
成立、sineでは不成立）。配送の決定化が変えたのはfootprintの出所の内訳で、`proxy(absent)`は
harmonicの2条件から消え、`proxy(stale)`は64 Voiceで11→2・4→2へ下がり、その分`body`が増えた。

この一致は、実時間配送のときの揺れが**この取得ではたまたま選択を動かさなかった**ことを意味する。
配送が決定的になった今後は、同じ条件の再実行で同じ数値が出る。§5.5(a)を別々の実行の比較として
成立させるための前提が揃ったということであり、数値そのものが変わったわけではない。

### 独立参照（`a6637db`、12条件）

同じ24本のreportに`scripts/verify_i11_stage1.py`をかけ、**22種の検査3,558,101件すべて通過、失敗0**。
§5.2で選択が変わった決定は121／3,372（sineは全条件0、harmonic 1／56・6／205・7／863、
modal 22／56・38／204・47／864）。`aebae2b`での同じ集計（121／3,346）と変化数は同じで、分母が増えたのは
`proxy(stale)`へ落ちる決定が減って`body`を持つ決定が増えたためである。footprint源の合計は
`aebae2b`の`body` 3,524・`absent` 55・`stale` 19から、`body` 3,550・`absent` 39・`stale` 9になった。

`proxy(absent)` 39件は揺れではなく構造的なものである。要求を出したhopで受け取っても決定で使うのは
次のhopからなので、各Voiceの最初の決定にはfootprintが無い。

### 限界

- `a6637db`が登録したとおり、offlineは**最短の配送を仮定**している。liveで測った受領遅延（中央値
  10.7〜85.3 ms、最大277 ms）はここに現れず、`proxy(absent)`／`proxy(stale)`の比率はliveより低い。
  §5.5(a)の数値をliveの挙動として読まない。
- 再現性を確かめたのは1条件（modal-flow-4）である。12条件すべてで2回renderして確かめてはいない。
- 本節は§5.5(a)のみで、§5.7の数値は含まない。§5.7の取得（A/A 3 passと`body`対`none`）は
  `aebae2b`のbinaryで行っており、判定はまだ付けていない。


## §5.7を三つの許容で判定する（2026-09-23、A/A 3 pass floor）

取得前に登録した規則どおりに、A/A 3 passからfloorを置き直し、`body`対`none`の16組を三つの許容で
判定した。取得はいずれも`aebae2b`のbinary（A/Aは`target/i11-stage1-aa3-20260923/pass1〜3/`、
`body`対`none`は`target/i11-stage1-rerun3-20260923/`、計384実行すべてexit 0・失敗0）。§5.5(a)を
取り直した`a6637db`はoffline renderの配送だけを変える改訂で、instrumentは実時間の配送のままである。

### floor（48標本の最大）

| 欄 | 3 pass floor | p50 | p95 | pass毎の最大 | 09-23の1 pass floor |
|---|---|---|---|---|---|
| population_us 中央値 | 2.090 µs | 0.27 | 1.31 | 2.1 / 1.3 / 2.0 | 1.270 |
| population_us p99 | 109.621 µs | 14.47 | 54.13 | 79.9 / 50.2 / 109.6 | 70.181 |
| population_us 最大 | 447.032 µs | 75.84 | 303.20 | 312.7 / 447.0 / 372.5 | 468.202 |
| synthesis_us 中央値 | 101.580 µs | 9.97 | 41.58 | 51.4 / 39.2 / 101.6 | 62.571 |
| synthesis_us p99 | 223.631 µs | 21.84 | 73.75 | 53.6 / 223.6 / 71.3 | 73.500 |
| synthesis_us 最大 | 386.653 µs | 45.77 | 243.98 | 273.0 / 386.7 / 344.2 | 584.653 |
| elapsed_us 中央値 | 171.471 µs | 28.05 | 98.07 | 98.1 / 111.6 / 171.5 | 71.580 |
| elapsed_us p99 | 244.752 µs | 68.43 | 178.28 | 178.3 / 244.8 / 160.8 | 127.040 |
| elapsed_us 最大 | 552.173 µs | 149.25 | 334.07 | 552.2 / 382.1 / 338.8 | 648.174 |
| 超過hop数 | 1 | 0 | 0 | 0 / 1 / 0 | 1 |

中央値・p99の6欄はすべて1 pass floorより大きくなった（`synthesis_us` p99は73.5→223.6 µsと3倍）。
1回分の標本が雑音幅を下回っていたという診断どおりである。最大の3欄は逆に小さくなったが、規則が
`max(1 ms, floor)`なのでどのみち1 msが効き、判定は変わらない。

**収束の確認**（登録どおり、判定には使わない）。pass 1〜2の32標本から取ったfloorをpass 3の16標本に
当てると、160件（16組×10欄）中4件が超過した——`population_us` p99が1件（harmonic-flow-16 report
109.6 > 79.9）、`synthesis_us`中央値が2件（sine-flow-16 report 101.6 > 51.4、同 no-report 57.2）、
`elapsed_us`中央値が1件（sine-flow-16 report 171.5 > 111.6）。2 passでは足りず、3 passでもまだ
完全には収束していない。floorはこの数字で調整していない。

### 判定（16組）

| 組 | 元の許容 | 1 pass floor | 3 pass floor | 3 passで落ちた欄 |
|---|---|---|---|---|
| sine-hold-4 report | ○ | ○ | ○ | — |
| sine-hold-4 no-report | × | × | ○ | — |
| sine-hold-16 report／no-report | ○ | ○ | ○ | — |
| sine-flow-4 report | × | × | ○ | — |
| sine-flow-4 no-report | × | × | ○ | — |
| sine-flow-16 report／no-report | ○ | ○ | ○ | — |
| harmonic-flow-4 report | × | ○ | ○ | — |
| harmonic-flow-4 no-report | × | ○ | ○ | — |
| harmonic-flow-16 report | × | × | × | population_us中央値 8.8 > 6.0 µs（5%が効く） |
| harmonic-flow-16 no-report | × | ○ | ○ | — |
| modal-flow-4 report | × | × | × | elapsed_us最大 1610.5 > 1000.0 µs（1 msが効く） |
| modal-flow-4 no-report | × | × | × | elapsed_us最大 1919.8 > 1000.0 µs（1 msが効く） |
| modal-flow-16 report | × | × | × | elapsed_us p99 552.8 > 410.2 µs（5%が効く） |
| modal-flow-16 no-report | × | × | × | elapsed_us p99 667.2 > 397.3 µs（5%が効く） |

**合格は、元の許容で5/16、1 pass floorで8/16、3 pass floorで11/16。**

残る5組はいずれも**floorが効いていない**。落ちた欄の許容を決めているのは5%か1 msであり、floorを
広げても変わらない。`population_us`の中央値は16組中15組が合格で、唯一落ちたharmonic-flow-16 reportは
`none`が119.65 µsの大きい基準に対して8.81 µs（+7.4%）増えたもので、09-23に問題にした「絶対値5 µsに
相対5%」の話とは別である。同じ条件のno-reportは合格しているので、**report有りの`body`側にだけ乗る
§4.6の決定報告の費用**（取得前に宣言した非対称）が最も素直な説明になる。report有りは5/8、
report無しは6/8の合格で、全体としては決定報告が判定を大きく動かしてはいない。

footprint計数（`body`、report、3反復合計）は要求＝完了・drop 0で、置換済みはharmonic-flow-4が6/24、
harmonic-flow-16が24/96、他は0。返却遅延の中央値はsine 0.0 ms、modal-flow-4 21.3 ms、
harmonic-flow-16 85.3 msで、09-23の取得と同程度である。

### modal-flow-4の`elapsed_us`最大の超過hop（手順どおり特定した）

この欄は3回の独立した取得（09-23午前 1802.8／1638.6 µs、同午後 1781.4／2036.9 µs、今回
1610.5／1919.8 µs）で一貫して1 msを超えており、§5.7で唯一再現していた問題である。profileから
hopを特定した。

- **超過hopは6実行すべてframe 401**（body report／no-report × 3反復）。`none`側の最大hopは
  11・191・501・561とばらけている。
- frame 401で増えている相は`analysis_wait_us`だけである。body 7770.8 µs に対し`none`の同じframeは
  1512.9 µs。他の相は一致する（`synthesis_us` 656.8対648.6、`rendering.capture_delivery_us`
  304.6対299.1、`rendered_tone_count` 16対16）。**`population_us`は6.5 µs**で、候補選択の仕事は
  この hop でほとんど動いていない。
- **footprintの時刻とは重ならない。** modal-flow-4のfootprint要求は1実行あたり8件で、
  `requested_at`は0と512サンプル（frame 0と1）、`received_at`は0〜2048サンプル、`computed_at`は
  22〜564である。frame 401（サンプル205,312）の前後に要求・計算・受領はいずれも無い。置換済み0件、
  drop 0件。
- `analysis_wait_us`が3 msを超えるhopは、**両変種ともframe ≡ 1 (mod 10)** に限って現れる
  （body 23件、none 30件）。分析thread側の配送が10 hop周期で、hopがそれを待っている。中央値は
  body 1262.7 µs・none 1262.4 µs、p99は5533.1対5539.5でほぼ同じ。**spikeの数は`none`の方が多い。**
- 最大のspikeは、その周期hopの直前にonsetが入ったときに出る。bodyは3反復ともhop 400にonsetがあり
  （次のhop 401が周期hop）、`none`は395〜401にonsetが無く、frame 401の待ちは1.3〜1.5 msに留まる。
  上位spikeでも frame 301（onset 298）、frame 481（onset 478）と同じ並びが見える。
- 走行全体では差がない。`elapsed_us`の合計は report が body 1.43／1.46／1.42 s に対し
  `none` 1.42／1.42／1.41 s、no-report は body 1.36／1.36／1.38 s に対し`none` 1.38／1.37／1.36 s。
  予算超過hopは611 hop中、body no-report rep1の1件を除いて全実行0である。

**原因の候補と根拠。**

1. **最も支持される候補**: 落ちているのは「分析threadを待った時間の、走行中で最大の1点」であり、
   その待ちは両変種に共通する10 hop周期の構造である。`max`という統計は各実行から外れ値を1つずつ
   拾って比べるので、分布がほぼ同じでも、どのhopが最大を引くかで1 ms程度は動く。根拠は、spikeの
   周期性、`population_us`が6.5 µsであること、footprintの活動がframe 0〜1で終わっていること、
   中央値とp99が一致すること、spikeの件数は`none`の方が多いこと。
2. **除外できない候補**: bodyがhop 400にonsetを置いたために、次の周期hopでの分析が実際に重くなった。
   onsetの位置が違うのはT1が選択を変えた結果である（§5.5(a)でこの条件は25/55の決定が異なる）。
   この場合の費用は「機構が計算に使う時間」ではなく「機構が変えた音楽の分析にかかる時間」であり、
   §5.7が測ろうとしているhop経路の負荷とは別物になる。今回の証拠では両者を分離できない。
3. **除外した候補**: 候補選択そのもの（当該hopの`population_us` 6.5 µs）と、footprintの要求・計算・
   受領（frame 401の前後に一件も無い）。飽和・drop・置換済みもこの条件では0件である。

修正は行っていない。この観察は、`elapsed_us`の最大を別々の実行の間で比べる判定が、機構の負荷では
なく分析待ちの外れ値を拾いうることを示す。判定の形を変える場合は新しい登録として残す。

### 判定のまとめ

§5.7は三つの許容のいずれでも全欄合格に至っていない（5/16、8/16、11/16）。floorを3 passの分布から
置き直したことで、雑音と区別できない差による不合格は解消した。残る5組は、(1) harmonic-flow-16
report の`population_us`中央値＝report有りにだけ乗る決定報告の費用が最も素直な説明、
(2) modal-flow-4の`elapsed_us`最大＝分析待ちの外れ値で、footprintの経路とは時間的に重ならない、
(3) modal-flow-16の`elapsed_us` p99＝同じ分析待ちの分布がp99へ効いたもの、に整理できる。
第1段の技術完了は判定しない（§5.9のoffline再投影が未実施）。


## 決定報告の書き出しを報告の相へ移す（2026-09-23）

§5.7でharmonic-flow-16 reportだけ `population_us` 中央値が+8.81 µs（+7.4%）増えた件を調べ、実装の
誤りを見つけた。`participation_decision` のJSON書き出しを集団更新（`advance_population`）の中で行って
いたため、report付きの実行では書き出しの費用が `population_us` に計上されていた。既存の
`participation_outcome` は集団更新では集めるだけで、書き出しは後の相で行う。同じ形に直し、決定recordは
集団更新で集め、`emit_hop_reports` で書き出す。

移す前後で、同じ条件のoffline renderのWAVと、`participation_decision`・`participation_context`・`onset`・
`body_footprint`（`computed_at` を除く）が一致した。report有りの8組の§5.7は、この修正の後に取り直す。


## §5.7の`elapsed_us`をhop自身の仕事で判定する（2026-09-23、取得前の登録）

§5.7は機構がhop経路に加える負荷を測る検査である。`5c877d4` の分析で、`elapsed_us` の不合格はその負荷では
ないことが分かった。modal-flow-4の最大はframe 401に集中し、増えた相は `analysis_wait_us` だけ（`body`
7770.8 µs、`none` 1512.9 µs）で、`population_us` は6.5 µsだった。`analysis_wait_us` が3 msを超えるhopは
両変種ともframe ≡ 1 (mod 10) に限られ、解析threadの10 hop周期の受け渡しをhopが待っている。最大の外れ値は、
周期hopの直前にonsetが入ったときに出る。modal-flow-16のp99も同じ分布から来ている。待ちの増加は機構の
計算ではなく、機構が変えた音楽の解析時間である。

- 規則。`elapsed_us` の3欄（中央値・p99・最大）と予算超過hop数を、hopごとの
  `own_us = elapsed_us - analysis_wait_us - listener_wait_us` から算出して判定する。許容は現行のまま
  （中央値とp99は `max(0.05 * none, floor)`、最大は `max(1 ms, floor)`、超過hop数は `floor` 以内）。
  超過hopは `own_us` がhop予算を超えたhopとする。floorは既存のA/A 3 pass（`none` 対 `none`）から、同じ規則
  （48標本の最大）で `own_us` について算出する。
- 併記。生の `elapsed_us` とその判定、`analysis_wait_us`・`listener_wait_us` の中央値・p99・最大を、
  `body` と `none` に分けて報告する。音楽が変わったことによる解析負荷の差は、資源受入（R2）で扱う。
- 変えないもの。`population_us` と `synthesis_us` の検査はそのまま。
- 事後性。この規則は `5c877d4` の結果を見た後に置く。判定には、決定報告の書き出しを報告の相へ移した
  `c41fdb3` の後の新しい取得を使い、その取得の前にこの登録を固定する。A/Aのfloorを登録したときと同じ扱いである。


## §5.9の入口と再現条件での分布（2026-09-23）

§5.9のoffline再投影の入口を、I10のoffline評価と同じ形で作った。製品側のコードは変えていない。

- 取得。ignored test `runtime::body_profiles::representative_gap::acquire_representative_gap`
  （入力 `CONCHORDAL_I11_GAP_INPUTS`、出力 `CONCHORDAL_I11_GAP_OUTPUT`）。入力の `registration.json`
  （schema `i11-representative-gap-v1`）はcaseごとにscriptとconfigとそのSHA-256を持つ。各caseを、test専用の
  probeとreportを付けてofflineで実行する（footprintは決定的な配送）。probeは発行された各onsetを、
  `ToneSpec` とkickから `Tone::from_parts` で組み直し、実kick、seed `modal_phase_seed(source, onset, tone_id)`、
  そのhopのrhythms、予約releaseで、I10の `issued` と同じ手順で凍結する。実行後、reportの
  `participation_decision`（skipでなく、`round(selected_at)` がonsetに一致するもの）から `D_rep` を取り、
  `project_window` の `retained = []`、`added = ([true, false], issued)`、bus 0、区間
  `[onset, onset + ceil(D_rep))`、coherentで16 binを求めて `gap.jsonl` に書く。
- 集計。`scripts/summarize_i11_representative_gap.py` は、実側のenergyを最大binで正規化し、
  `power_gap = max_k |power_k^actual - power_k^rep|`、選択候補の外部energyでの重なり和の差
  `overlap_gap`、それを `coupling * 6 * s / norm` に通した費用項の差 `term_gap` を、footprintの出所ごとに
  中央値・p95・最大（nearest rank）で出す。支持の無いbinを含むonset、決定の見つからないonset、重なり項が
  効いていない決定は数えて除く。両側が既知の無音なら差0。重なり式は `verify_i11_stage1.py` と共有する。
  testは `tests/test_summarize_i11_representative_gap.py`。
- 再現条件の結果（識別規則の改訂節と同じ4 Voice flow、`body`）。onsetはsine 107、harmonic 107、modal 106で、
  すべて決定と対応が付き、16 binすべてに支持があった。決定はすべて `body`。

  | 条件 | `power_gap` 中央値／p95／最大 | `term_gap` 中央値／p95／最大 |
  |---|---|---|
  | sine | 5.89e-3／6.03e-3／6.12e-3 | 4.46e-4／8.63e-4／1.07e-3 |
  | harmonic | 2.12e-2／2.19e-2／2.20e-2 | 5.75e-3／6.83e-3／7.03e-3 |
  | modal | 3.69e-5／1.01e-4／1.35e-4 | 6.32e-6／1.78e-5／2.40e-5 |

  代表条件（kick 1.0、代表seed、既定rhythms、release予約なし）と実条件の差は、この素材ではpowerで
  最大2.2%、費用項で最大0.007だった。登録の12条件での取得は、入力がリポジトリに入った後に行う。
- 限界。実側も発行時点の凍結であり、発行後の振幅更新と残存toneとの干渉は含まない（§5.9の定義どおり、
  新しいtone単独のenergy）。


## §5.7が全欄合格する（2026-09-23、`own_us`判定と決定報告の修正の後）

`c41fdb3`（決定報告の書き出しを報告の相へ移す）と`6f85f42`（`elapsed_us`を`own_us`で判定する登録）の後に、
`body`対`none`を16組で取り直した。取得は`029795c`のbinary、出力は`target/i11-stage1-rerun4-20260923/`
（8条件×report有無×交互3反復、96実行すべてexit 0・失敗0）。**リポジトリ化した入力
（`docs/roadmap/temporal-dcc/i11-inputs/`、`fe42afc`）から取得した初回**でもある。

### `own_us`のfloor（取得前に固定した）

既存のA/A 3 pass（`none`対`none`、`aebae2b`）から、登録どおり48標本の最大で算出した。

| 欄 | `own_us` floor | 生の`elapsed_us` floor |
|---|---|---|
| 中央値 | 107.020 µs | 171.471 µs |
| p99 | 156.901 µs | 244.752 µs |
| 最大 | 552.433 µs | 552.173 µs |
| 超過hop数 | 1 | ― |

待ちを引くと中央値とp99のfloorは下がる（171→107、245→157）。A/Aの雑音幅のうち、その分は解析待ちの
ばらつきだったということである。最大はほぼ同じで、規則上は1 msが効く。生の`elapsed_us`の3値は
`aa-floor3.json`と一致し、`own_us_check.py`が`hop_path.py`の畳み方を再現していることを裏づけた。
収束確認（判定には使わない）はpass 1〜2のfloorをpass 3へ当てて3件超過。

### 判定

**登録の組み合わせ（`population_us`・`synthesis_us`は3 pass floor、`elapsed_us`の3欄と超過hopは`own_us`）で、
16組すべてが全欄合格した。落ちた欄は0である。** §5.7が全欄で通ったのはこれが最初である。

| 判定 | 合格 |
|---|---|
| 登録の組み合わせ（`own_us`） | **16/16** |
| 参考: 3 pass floor（生の`elapsed_us`のまま） | 13/16 |
| 参考: 元の登録許容 | 8/16 |

余裕も確認した。最も許容に近いのは`population_us`でharmonic-flow-4 reportのp99が73.1/109.6 µs（67%）、
`own_us`ではsine-flow-16 reportのp99が171.0/242.7 µs（70%）で、いずれも許容の7割以内である。
超過hop（`own_us`基準）は全16組で`body`・`none`とも0だった。

**`c41fdb3`の効果。** 09-23に唯一残っていた`population_us`中央値の不合格（harmonic-flow-16 report）は
解消した。同じ欄が、決定報告の書き出しを集団更新の外へ出す前は body 128.46 µs／none 119.65 µs（差
8.81 µs、+7.4%、許容5.98）で不合格、後は body 121.45 µs／none 118.05 µs（差3.40 µs、+2.9%、許容5.90）で
合格である。**report有りの`body`側にだけ乗っていた費用が、決定traceのJSON符号化だったことが実測で
確かめられた**（09-23にそう推定して記録していたものの裏づけになる）。

### 生の`elapsed_us`と待ちの併記（登録が求める報告）

生の`elapsed_us`で判定すると13/16で、落ちるのは3組である。

| 組 | 生の`elapsed_us`で落ちた欄 | 同じ組の`analysis_wait_us`最大（body／none） |
|---|---|---|
| modal-flow-4 report | 最大 1585.6 > 1000.0 µs | 8050／6041 µs |
| modal-flow-4 no-report | 最大 1713.5 > 1000.0 µs | 8036／6100 µs |
| modal-flow-16 report | p99 513.2 > 404.6 µs | 6301／6015 µs |

modal-flow-4では、生の`elapsed_us`最大の差1.59〜1.71 msに対し、同じ組の`analysis_wait_us`最大の差が
2.0〜1.9 msある。**待ちを引けば差は消える**（`own_us`中央値の差は8.8 µsと2.6 µs）。`5c877d4`で
「解析待ちの外れ値を`max`が拾っている」と診断したとおりの構造が、独立した4回目の取得でも同じ形で出た。

待ちの分布（中央値／p99／最大、µs）。`listener_wait_us`は全組で`body`・`none`とも中央値4〜6 µs、
p99 256〜269 µsで差がない。`analysis_wait_us`の中央値は多くの組で両変種とも1260 µs前後で、
差が出る組は向きが揃わない（harmonic-flow-16はreportで`body`が高く1494対1265、no-reportでは`none`が
高く1505対1287）。系統的に`body`が待つわけではない。

音楽が変わったことによる解析負荷の差は、登録どおり資源受入（R2）で扱う。

### footprint計数（`body`、report、3反復合計）

要求＝完了・drop 0。置換済みはharmonic-flow-4が6/24、harmonic-flow-16が26/96、modal-flow-4が1/24、
他は0。返却遅延の中央値はsine 0.0 ms、modal-flow-4 21.3 ms、modal-flow-16 32.0 ms、
harmonic-flow-4 42.7 ms、harmonic-flow-16 85.3 ms。これまでの取得と同程度である。

### 残るもの

§5.7は全欄合格したが、第1段の技術完了は§5の1〜7の合格と9の分布報告の完了を要件とする（§6）。
§5.9はRust側の入口が入り（`12bb02b`）、再現条件では通ったが、登録12条件での取得が未実施である。
§5.4bの凍結packetの再評価も未実施のまま。`own_us`による判定は、機構がhop経路に加える負荷を見る
という§5.7の趣旨に沿わせたものであり、解析待ちが増える事実そのものを無くしたわけではない。


## §5.9を登録12条件で取得する（2026-09-23）

`12bb02b`の入口を、リポジトリ化した`registration.json`（`fe42afc`）の12条件へ当てた。取得は
`CONCHORDAL_I11_GAP_INPUTS`／`_OUTPUT`を渡した ignored test `acquire_representative_gap`（`029795c`の
ソース、`[profile.dev]` の `opt-level = 1` で192.6秒、test ok）。出力は`target/i11-gap-20260923/`
（条件ごとに`report.jsonl`と`gap.jsonl`、計約780 MB）。集計は
`scripts/summarize_i11_representative_gap.py`で、`summary.json`に置いた。§5.9は合否ではなく分布の
報告であり、A1の材料である。

### 件数

投影できたonsetは**3,390件**で、`coherent_energies`に`None`を含む機会（`excluded_unsupported_bin`）は
**0件**だった。`no_decision` 84件はsine-holdの3条件（4＋16＋64）の、各Voiceが開始時に1回だけ鳴らす持続音である。
Hold modeの発音は参加の発音時刻決定（`candidate()`）を経ないので、対応する決定と代表footprintが無く、
比較の対象外になる（2026-09-23補足）。
`overlap_not_applied` 39件は、選択候補に重なり項が適用されていなかった決定で、`proxy(absent)`の
39件と一致する（各Voiceの最初の決定にはfootprintが無い）。この39件は`power_gap`だけが出て、
`overlap_gap`と`term_gap`は算出されない。

footprint源別の投影件数は`body` 3,342、`proxy(absent)` 39、`proxy(stale)` 9。

### `body`（3,342件）の分布

| 素材 | 条件 | `power_gap` 中央値／p95／最大 | `overlap_gap` 最大 | `term_gap` 最大 |
|---|---|---|---|---|
| sine | flow-4／16／64 | 2.64e-2／3.67e-2／3.96e-2<br>2.69e-2／3.87e-2／4.30e-2<br>2.73e-2／4.23e-2／5.26e-2 | 8.88e-4<br>5.80e-4<br>3.24e-4 | 1.51e-3<br>1.52e-3<br>1.66e-3 |
| harmonic | flow-4／16／64 | 8.62e-2／9.77e-2／1.00e-1<br>8.70e-2／1.05e-1／1.31e-1<br>8.70e-2／1.07e-1／1.34e-1 | 3.26e-3<br>1.35e-3<br>5.81e-4 | 1.08e-2<br>1.29e-2<br>1.16e-2 |
| modal | flow-4／16／64 | 2.63e-3／1.03e-2／1.09e-2<br>2.28e-3／9.35e-3／1.77e-2<br>2.31e-3／1.12e-2／2.22e-2 | 2.13e-6<br>1.96e-6<br>1.15e-6 | 2.36e-4<br>2.99e-4<br>1.01e-3 |

素材による差が大きい。**代表footprintが実際に鳴ったtoneを最もよく追うのはmodal**（`power_gap`中央値
2.3e-3）で、**最も外れるのはharmonic**（同8.7e-2、最大1.34e-1）。sineは中間（2.7e-2）。Voice数を増やしても
中央値はほとんど動かず、最大だけが伸びる。

`12bb02b`の再現条件（4 Voice flow、habitat bus）では`power_gap`最大2.2e-2・`term_gap`最大7.0e-3だったので、
**登録12条件はそれより厳しい**（`body`で`power_gap`最大1.34e-1、`term_gap`最大1.29e-2）。再現条件の値を
登録条件の代表値として読まない。

### 代理に落ちた決定（48件）

| 源 | 件数 | `power_gap`（sine／harmonic／modal） |
|---|---|---|
| `proxy(absent)` | 39 | 1.09e-1〜1.12e-1／7.78e-1〜7.80e-1／1.00 |
| `proxy(stale)` | 9 | 1.90e-1〜1.95e-1／6.51e-1〜6.76e-1／9.97e-1 |

代理（ADSRの包絡）と実際のtoneの差は、`body`より1〜3桁大きい。modalではほぼ1.0、つまり16 binの
形がまったく違う。`proxy`のときに費用へ入る`power_k`が実条件をほとんど表していないということであり、
`proxy`を基準候補として使う§4.3の設計そのものの限界を示す。件数は48/3,390（1.4%）に留まる。

`proxy(stale)` 9件の`overlap_gap`と`term_gap`はいずれも厳密に0だった。選択候補の外部energyが全ゼロの
窓で起きた決定のためで（48要素すべて0、`overlap`も0.0と確認）、実側・代表側の`Ov`がともに0になる。
重なり項が効かない窓での差なので、0であること自体に意味は無い。

### 限界

- offlineの最短配送を仮定した取得である（§4.8のoffline改訂）。liveの受領遅延は含まず、
  `proxy(absent)`／`proxy(stale)`の比率もliveより低い。代理の`power_gap`が大きいことの影響は、
  liveではこの48件より広い範囲に及びうる。
- 実側も発行時点の凍結で、発行後の振幅更新と残存toneとの干渉は含まない（§5.9の定義どおり）。
- 分布の報告であって合否ではない。この差を許容するかはA1の聴取で判断する。


## §5.4bを対の再実行で確かめる（2026-09-23、取得前の登録）

§5.4bは「同一の凍結packetを基準版と新版で再評価し、record内容をbit比較する」と登録した。packetには
共有table（`consumer::Publication`）、外部予測、traceなど約10種の型が入り、直列化と読み戻しの仕組みが
無い。基準commit側にも評価の入口を移植する必要がある。同じ確かさを、次の方法で得る。§8に従う改訂登録である。

- 同値の根拠。候補energyの評価関数（`evaluate`、`evaluate_scheduled`、`project_window`、`ratios`、
  `release_trace`）は、I11直前の `103a32f` から本文が変わっていない。差分は、workerのループから
  `process_packet` への切り出し（同値。packet返却の成否を戻り値で返す）と、footprint用queueの配線に限られる。
  `None` ではhopごとの状態が基準とbit一致する（§5.4a）ので、両版は同じpacketを作る。
- 方法。基準（`32a6389`）と新版（取得時のHEAD）で、登録12条件を `config-none.toml`、offline render、
  report付きで実行する。`body_candidate_energy` を
  `(source_id, source_generation, tone_id, issued_at, decision_at, scope)` で突き合わせ、実時間の欄
  `processing_us` 以外の全欄をbit比較する（`scripts/compare_i11_candidate_records.py`）。片側にしか無い
  recordは、候補workerの飽和による採否の揺れとして件数だけを数え、比較しない。
- 合格。12条件すべてで、突き合わせたrecordが全件一致し、突き合わせが0件の条件が無いこと。
- 予備確認。`77e07b6` の版と `aebae2b` 時点の版を、再現条件3身体の `None` で比べた。突き合わせた345件は
  全件一致し、片側にしか無いrecordは162件だった。scriptのtestは
  `tests/test_compare_i11_candidate_records.py`。
- 基準の公開。基準commit `32a6389` はblancheのローカルにしか無いので、タグ `i11-stage1-baseline` として
  公開し、別のマシンでも基準binaryを作れるようにする。


## §5.4bの取得と判定（2026-09-23）

上の登録どおり、基準版と新版の対で`None`を再実行し、候補recordを突き合わせた。

### 基準タグは`103a32f`に付けた（`32a6389`と同値である根拠つき）

タグを付ける前に`32a6389`を調べたところ、**HEADの祖先ではなかった**。同じ登録commitが
`103a32f`として本流にあり、履歴が並行していた（`32a6389`→`94cf6fe`→`69e728f`、
`103a32f`→`f0b0421`→`b34ce96`と、同じ件名で別のhashが並ぶ）。`32a6389`固有のcommitは18件あり、
これをpushすると本流とは別系統の履歴がremoteに残る。

両者が基準として同値であることを確かめた。

- `src/` treeが完全一致（どちらも `83b98b08bacf3c01e768e3708687845f87c6d0e7`）。`Cargo.toml`と
  `Cargo.lock`も一致。
- 全体の差分は`web/static/pdf/conchordal_alife2026_slides.pdf` 1件だけで、ビルドは読まない。
- **`103a32f`からビルドした`conchordal-render`のSHA-256は
  `3614d998e07c04c7af1913f97f515cb0edf0be4e2aec6b18331703e67fa58b5e`** で、`plan.json`が
  2026-09-22に記録した基準binaryの値と**バイト単位で一致した**。§5.4aが使ったbinaryそのものが
  再現できている。

そこでタグ `i11-stage1-baseline` は`103a32f`に付け、注釈にこの同値性を記した。remoteへはタグだけを
pushした。以後、基準binaryは `git worktree add <dir> i11-stage1-baseline && cargo build --release` で
どのマシンでも作れる。

### 取得

登録12条件を`config-none.toml`・offline render・report付きで、基準版と新版の両方で実行した。
出力は`target/i11-stage1-54b-20260923/{baseline,new}/`（各12実行、失敗0）。
基準版のrenderは`3614d998…`（`plan.json`の記録と一致）、新版は`df3c360eb08935c1f1218e10cd5830a118cc6967ce1c63a3ab5e1125e0beccc9`（`65a0314`。この改訂はsrc/を変えていないので`029795c`のビルドと同じ）。

### 判定

`body_candidate_energy`を`(source_id, source_generation, tone_id, issued_at, decision_at, scope)`で
突き合わせ、`processing_us`以外の全欄をbit比較した。

| 条件 | 突き合わせ | 一致 | 不一致 | 基準のみ | 新のみ |
|---|---|---|---|---|---|
| sine-hold-4／16／64 | 448／1,792／7,168 | 同 | 0／0／0 | 0／0／0 | 0／0／0 |
| sine-flow-4／16／64 | 180／333／917 | 同 | 0／0／0 | 13／32／34 | 11／39／22 |
| harmonic-flow-4／16／64 | 75／84／117 | 同 | 0／0／0 | 3／8／18 | 3／8／17 |
| modal-flow-4／16／64 | 80／93／134 | 同 | 0／0／0 | 5／26／17 | 5／23／16 |
| **合計** | **11,421** | **11,421** | **0** | 156 | 144 |

**登録の合格条件をどちらも満たす。** 突き合わせた11,421件すべてが一致（不一致0件）で、突き合わせが
0件の条件は無い（最少はharmonic-flow-4の75件）。片側にしか無いrecord（基準のみ156件、新のみ144件）は、
登録どおり候補workerの飽和による採否の揺れとして件数だけ数えた。sine-holdの3条件では片側のみが
0件で、揺れがまったく出ていない。

これで§5.4は(a)(b)ともに合格した。§5.4(a)は`None`のWAVと学習record 10種が基準とbit一致すること
（2026-09-22、12条件）、(b)は候補計算のrecord内容が一致することである。

### §6の技術完了の要件（現状の一覧。完了の宣言はしない）

§6は第1段の技術完了を「§5の1〜7の合格と9の分布報告の完了」と定めている。現状は次のとおり。

| 項 | 内容 | 状態 | 記録 |
|---|---|---|---|
| §5.1 | 入力一致（独立参照での再計算） | 合格 | 12条件24本のreportで22種・3,558,101件が失敗0 |
| §5.2 | 状態介入（`power_k`だけを代理へ置換） | 合格 | 121／3,372の決定で選択が変わる。sine 0、harmonic 1〜7、modal 22〜47 |
| §5.3 | 期限（未来・置換済み・stale・availability・再選択） | 合格 | 同じ24本で失敗0 |
| §5.4 | off基準 (a) bit一致 / (b) 候補recordの一致 | 合格 | (a) 12条件でWAVと学習record一致／(b) 本節、11,421件一致 |
| §5.5 | APIの作用 (a) `body`対`proxy` | 合格 | harmonic／modalの6条件で選択差とWAV相違（`t1_effect_observed` 6/12）。sineでは差が出ないが、登録は「差が出ない素材だけでは通過しない」と定めており、harmonic／modalで出ている |
| §5.6 | 部分欠測と世代交換 | 合格（単体testのみ） | `absent→body→stale→body`、置換、再送、`Unsupported`／`BodySilent`、予約後の非再選択をRustの単体testで確認（2026-09-22）。登録12条件の実走では skip と `proxy(unsupported)` は現れない |
| §5.7 | hop経路と飽和 | 合格 | `own_us`判定で16組すべて全欄合格 |
| §5.9 | 代表条件と実条件の差（分布報告） | 完了 | 登録12条件で3,390件を投影。`body`の`power_gap`中央値はmodal 2.3e-3〜harmonic 8.7e-2 |

§5.8（到来）は第2段の項目で、第1段の要件ではない。

**要件は形式的にはそろった。ただし技術完了はここでは宣言しない**（別に判断する）。判断の際に見ておく
点を挙げておく。

- §5.6は登録12条件の実走ではなく単体testで満たしている。skipと`proxy(unsupported)`は12条件の実行に
  現れていない。
- §5.7の合格は、`elapsed_us`を`own_us`（解析待ちを引いた値）で判定する2026-09-23の改訂登録に依る。
  生の`elapsed_us`では13/16である。待ちが増える事実そのものは残り、資源受入（R2）へ回している。
- §5.9は分布の報告であって合否ではない。代理に落ちた48件の`power_gap`はmodalでほぼ1.0で、
  `proxy`を基準候補に使う§4.3の限界を示している。
- §5.5(a)はsineでは成立しない。素材によって作用が出ないこと自体は登録の想定内だが、主張の範囲は
  harmonic／modalで観測した選択差と音響差に限る。
- 生成作用の採用（A1／A2）は別で、既定は`None`のままである。O15・O18は主張しない。


## 第1段の完了判定へのレビューと対応（2026-09-23）

§6の要件がそろった時点で、技術完了の宣言文案と本文書の全文をAstraに渡し、宣言してよいかを問うた。
判定は「不可」。経路の成立と、harmonic／modalでの選択と音響の差には根拠があるとしたうえで、
「§5の1〜7合格、9完了」はまだ立証されていないとした。識別規則とoffline配送の改訂は、機構上の根拠が
あるとして合格誘導とは見なされなかった。指摘は6件で、次のとおり対応する。宣言はしない。

### §6の改訂（登録）：§5.7は待ちを除く処理負荷に限る

- 指摘。`own_us` は機構の直接の処理負荷を測れるが、選択が変わったことで増えた解析待ちも機構の影響に
  含まれる。元の総hop負荷の要件を満たしたことにはならない。
- 改訂。§6の技術完了の要件のうち§5.7は、「機構がhop経路に加える処理負荷（`population_us`、
  `synthesis_us`、`own_us`）」の合格に限る。総hop負荷（解析待ちを含む `elapsed_us`）は第1段の主張に
  含めない。総負荷の差（生の `elapsed_us` で13/16、元の登録許容で8/16。modal-flow-4の最大は解析待ちの
  差1.9〜2.0 msで説明がつく）は資源受入（R2）で扱う。
- 事後性。この改訂は結果を見た後であり、技術完了の範囲を狭める方向の改訂である。

### §5.4bの改訂（登録）：共通recordに限った内容一致

- 指摘。共通キーでの突き合わせは、同一packetが入力だったことを保証しない。片側にしか無い300件の除外も
  含め、凍結packetの比較と同等であることは立証されていない。
- 改訂。§5.4bの要件を「基準版と新版の対の再実行で、共通キーを持つ候補recordの内容（`processing_us`を
  除く全欄）が一致すること」に改める。主張はこの範囲に限る。登録当初の「同一の凍結packetの再評価」は
  **未実施**として残す。片側にしか無いrecordは飽和による採否の揺れとして件数だけを報告する。
- §5.4a。bit一致は `77bd4c2` の時点で測ったもので、その後のコード（`77e07b6`、`aebae2b`、`a6637db`、
  `c41fdb3`、本節の返却drop）を含む最終版での再実行が要る（残作業）。

### §5.6：返却dropの解除と再送を実装した

- 指摘。「返却dropの解除経路は作っていない」と「返却drop後の再送を満たした」が整合しない。§4.2は
  「返却がdropされた場合は未完了を解除し、次のhopで再送する」と登録していた。
- 実装。workerは返却queueが満杯でrecordを捨てたとき、その識別を別の通知queue（容量64）へ送る。
  runtimeは毎hopその通知を取り出し、該当Voiceの未完了の要求を解除する（`FootprintTracker::release`）。
  解除されたVoiceは次のhopで同じ識別を再送し、再送として数えられる。offlineの決定的配送は、返却と
  通知のどちらかを受け取るまで待つので、dropがあっても止まらない。
- 確認。trackerのtest（`a_dropped_reply_releases_the_request_and_the_next_hop_resends_it`）と、返却を
  取り出さずに要求を積んでdropを起こすworkerのtest（`every_accepted_footprint_returns_or_is_released`。
  実時間の配送と決定的配送の両方で、受け付けた要求は返却か解除のどちらかになり、解除数はdrop数に一致する）。
  dropの無い条件では、変更の前後でWAVと決定に関わるrecordが一致した。
- 範囲。§5.6は単体testで確かめたものであり、登録12条件の実行にはdrop、skip、`proxy(unsupported)` が
  現れない。

### 記述の訂正

- A/A floorで合格した欄について「雑音と区別できないことが確定」と書いた箇所を、「改訂した受入規則
  （A/A floor）に適合」に改めた。floorは条件横断の最大値であり、雑音と区別できないことの証明ではない。
- §5.5の再取得の節で、64 Voiceを「登録範囲外」と書いた誤りを訂正した（§5.5(a)の範囲は12条件のrender）。
- §5.9の `no_decision` 84件は、sine-holdの各Voiceが開始時に1回だけ鳴らす持続音で、参加の発音時刻決定を
  経ないため対応する決定が無いことを補った。
- §5.9の代理に落ちた48件は、「実際のtone対ADSR代理」の差であり、「実際のtone対代表身体予測」とは別の
  母集団である。これらの機会では代表身体予測が使われておらず、代表身体予測との差は算出していない。
  主な分布（`body` 3,342件）はこの48件を含まない。

### 残作業（blancheで行う。データが要る）

1. §5.4aを最終版で再実行する（`None` の12条件で、基準binaryとのWAVと学習record 10種のbit一致）。
2. §5.7の最終取得について、登録した1 passの規則での判定、待ちの中央値・p99・最大、返却遅延のp99・最大を
   補う。
3. §5.5(a)について、`body` と `proxy` で最初に選択が分かれた決定を条件ごとに取り出し、その決定の入力
   （`D`、外部energy、`memory`、係数）が一致し、違いが `power_k` だけであることを示す。§5.5(b)
   （`Some` 対 `None` の参考比較）を記録する。
4. §5.9の `overlap_gap` と `term_gap` の中央値・p95を表に加え、`body` と代理を分けた母集団の表にする。
5. 再現性の確認の節（modal-flow-4）の「28種のうち24種が一致」と「6種が不一致」の数の矛盾（24＋6＝30）を、
   元の出力から数え直して訂正する。

これらがそろったら、改訂した§6の範囲で宣言文を書き直し、Astraに再レビューを頼む。
