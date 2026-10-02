# 通常offline出生と代謝の結合案（実装前draft）

状態: 第十四版の封印済み実装を読んだ設計案。コード変更、結合取得、性能取得はまだ行っていない。対象は決定的な `conchordal-render` における初回Field出生と身体代謝の併用だけとし、既定OFF、実時間instrument拒否、既存の単独flagの挙動を維持する。`body_fitness_action` と `body_fitness_observation` との併用、親付きrespawn、動的解析設定は対象外。

## 現在の呼出し順と結合時の失敗点

`process_hop` は `wait_for_analysis` で主解析を受け、`apply_landscape_updates` でhabituationを適用し、`advance_population`、`render_and_route_audio`、`BodyFitnessObserver::observe` の順に進む。`advance_population` 内では `OfflineBirth::install` が主解析の実frame番号から支持終端を得た後、`Conductor::dispatch_until_with_birth` → `Community::apply_action_with_birth` → Field全binの代表身体評価 → 既存配置選択・jitter → `spawn_one` が実子を追加する。続いて代謝の環境を各Voiceにinstallし、`Community::advance_with_listener_pressure` のlifecycleとGated onsetが実Voiceの代表Recipeを評価する。報告と死者整理はその後で、音声レンダリングはさらに後となる。

現状の代謝は時刻0の全Voiceだけを認める。時刻0以外では `BodyFitnessObserver::decision_batch(&state.pop.voices, now_tick)` が、直前の描画hopから受理したsource集合と、現在のVoice集合との**完全一致**を要求する。出生hopではdispatch前の環境Voiceが1声、dispatch後は子を含む2声なので、旧1声batchに対するこの呼出しは `None` となり、続く `expect` で停止する。現在の結合禁止を取り去るだけでは動かない。`OfflineMetabolism::install` の `initial_shared` も `now == 0 && support_end == 0` に限定され、子の出生時共有環境を表現できない。

## 境界を保つ最小接続

1. 結合専用のシナリオ検証は、時刻0の固定環境Voice 1声と後続のField子1声、各SpawnのVoice ID・population ID一意、同時生存2声、解析設定不変、Release・respawn・途中Updateなしに限定する。出生だけONの既存受理範囲と代謝だけONの既存受理範囲は変更しない。併用許可はこの交差条件のみに置き、action/observation併用とinstrumentは引き続き拒否する。FieldのConsonance/Peakを最初の取得対象とし、他のField variantを実装済みであることと今回の取得対象であることを分ける。
2. dispatch前に現在のVoice集合と、observerが最後に受理した `SourceIdentity { id, generation, birth_sample }` の集合を固定する。旧batchは**この出生前集合との完全一致**、epoch一致、支持時刻と決定時刻の順序、hop終端まで4800 samples以内、Log2Space一致を検証してから既存Voiceにだけ配る。`decision_batch` の一般的なsource数・identity照合を緩めたり、旧batchの任意部分集合を受理したりしない。dispatch後の実Voice集合は出生前集合に子ID 1個だけ加わり、既存Voiceのidentityが不変で、実子の世代0・population・member・実基音が出生診断と一致する、と確認する。ID衝突・子の追加失敗・別Voiceの消失があれば失敗とする。
3. 子の出生hopだけ、出生評価が見た `state.current_landscape` を代謝にも渡す。これは同じhopの `apply_landscape_updates` 済み共有環境であり、**子の発音前**の解析支持である。主解析の `last_analysis_frame` から実支持終端を算出し、出生診断と代謝receiptの `support_end_sample`・`decision_at_sample`・epoch・spaceを照合する。時刻をシナリオの秒数から作らない。既存Voiceは同hopでも旧batchの自己除去環境を使う。全声を共有環境へ戻さない。出生時の環境originは `InitialShared`（時刻0）と別の `BirthShared` として記録し、後者は `source.birth_sample == now_tick > 0` と実出生の照合を必須にする。
4. `OfflineMetabolism` のsource所有者は出生時に確定した `(id, generation, birth_sample)` とする。`VoiceMetadata` 自体にはbirth sampleがないため、今回の一意IDと出生前後集合差、出生hop開始sampleを突き合わせて固定する。observerは最初にそのVoiceを見たhop開始時刻を `birth_sample` に入れ、以後同じidentityを保持する。実子の `OfflineMetabolism` を生成した後に `Voice::tick_articulation_lifecycle` が実基音・実Bodyから72hopの代表密度を評価する。初回共有環境でも通常と同じscore/level→Entrain energy更新を走らせる。既存声の代謝更新回数・順序を出生有無で減らさない。
5. 出生hopのレンダリング前に `ScheduleRenderer::prepare_self_sound` が新Voiceのself slotと512サンプルのbufferを作る。`render_with_prediction_matches` はhop冒頭でself habitatをゼロクリアし、実際にhabitatへ送ったToneだけを加算する。無音の子でも長さ512の**ゼロ自己PCM**がある。続くobserverは混合habitat PCMと全Voiceのself habitat PCMを提出する。`Processor::step_validated` は新 `SourceSlot` を**出生hopの混合PCMを処理する前**の共有 `AnalysisStream` からcloneするため、子slotは出生前までの実混合履歴を継ぐ。その後、共有streamは混合PCM、各slotは `mixed - own` を出生hop分だけ処理する。子がそのhopで無音なら `own == 0` で、出生hopの共有混合音を子の自己除去履歴へ正しく追加する。決定的rendererは同hop終端のbatchを `receive_one_offline` で受理する。次hopの開始時に2声の完全一致batchを取り、子も `BirthShared` から `SourceRemoved` へ切り替える。自己PCMが非零になるまで切替を遅らせない。
6. 主解析とobserverは別の `AnalysisStream`。出生時共有評価と翌hop自己除去評価の時刻・epoch・space・habituation版を個別に照合する。主解析は固定遅延上限1 frame、出生評価は最大4800 samples、代謝はhop終端まで最大4800 samplesという現在の境界を守る。時刻0だけ既存の空初期履歴共有を使う。静的シナリオのepochは0で、更新を認めるまで一般的なepoch移行を装わない。受理batch欠落、worker停止、PCM不連続、誤ったsource identity、将来/古い支持、space不一致では点評価・ゼロ値・古い表への暗黙fallbackをしない。

## 次の最小取得案

新しい取得条件として事前登録してから実行する。seed 7、48 kHz/512、既定kernel・habituation、時刻0にSine 440 Hz Drone Sustainを両busへamp 0.06で配置する。`wait(0.6826667)` の後、Harmonic Entrain Sustain（brightness 0.7、endurance 2秒、recovery 10秒、attack cost/recharge 0、両bus amp 0.06）を `consonance(380.0, 520.0).peak().spacing(0.0)` で1声出生させる。出生後は `wait(0.4)` とし、frame 96まで子の実energyを観測する。比較条件は「出生だけON」と「出生＋代謝ON」の各2回。出生選択条件を両modeで同一にして、代謝入力の効果だけ比較する。各mode内でWAV、spawn、出生診断、population、代謝診断（後者は結合ONのみ）の再現性を要求する。mode間ではWAVの一致を要求しない。

出生診断の全候補と実最終Hz・Recipe identityを従来どおり照合する。新診断は出生hopと次hopの各source receiptを少なくとも一回保存し、`id/generation/birth_sample`、環境origin、epoch、space識別、support end、decision sample、身体generation、score/level、実Entrain energyを示す。出生hopで子は `BirthShared`、環境Voiceは既存batchの `SourceRemoved`。両者の決定sampleは出生hop開始、支持は出生前実解析の右端、ageは許容内。次hopは両者 `SourceRemoved`、子のidentityは出生hop開始sampleを保持し、observer受理batchは2source。子が出生hopで無音でもゼロ自己PCMでslotができ、次hopに有効なsource-removed評価があることを局所試験とreportで確認する。frame 96の子の `population_step.mean_entrain_energy` は結合ONの代謝receiptに対応し、出生だけONとの差が事前固定閾値 `1e-6` を超えることを要求する。差が出なければその条件は不達と記録し、取得後に係数や閾値を変えない。Sine populationはEntrain数0・平均energy nullを確認する。

必須負例は、初回環境なし・時刻0 Field子、同hop複数Spawn、重複IDまたはpopulation、第三Voice、Release/respawn/解析設定変更/途中Update、instrumentとaction/observation同時ONの配線前拒否。内部境界では、dispatch前batchのsource欠落・余分・birth sample違い、実出生失敗、子の誤世代、worker欠測・不連続、self PCM欠落または長さ違い、supportの未来/期限切れ、epoch/space不一致、出生翌hopに子がbatchにいない場合を失敗として確認する。無音の**長さ512のゼロ自己PCM**は負例ではなく正例とする。通常の一点評価や旧環境へのfallbackを禁止する。

この取得は1環境Voiceと1子の初回出生、2声の身体代謝更新までを示す。親選択、respawn、source入替え、動的解析構成、非同期実時間の公平政策、長期生態、計算費用は別の取得に残す。
