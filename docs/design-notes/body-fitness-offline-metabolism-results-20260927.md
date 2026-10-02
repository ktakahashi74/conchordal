# 第十三版: 通常renderの身体代謝入力（取得結果）

登録: [body-fitness-offline-metabolism-registration-20260927.md](body-fitness-offline-metabolism-registration-20260927.md)。固定したseed 7、48 kHz/512、0.64秒、Harmonic 440 Hz Entrain SustainとSine 466 Hz Drone、両bus・amp 0.06、movement OFFで取得した。metabolism OFF/ONを各二回。検査実装は `tests/body_fitness_metabolism.rs`。取得後にseed、身体、時刻、閾値は変更していない。

## 生記録と判定

この節のfocusedログと通常render生記録の `target/` は `/home/shafi/lwrk/conchordal/target/` を指す。

最終focused検査は2件成功。ログは `target/runtime-metabolism-validation/focused-final.log`、生のシナリオ、設定、WAV、JSONL、stderr、数値要約は `target/body-fitness-metabolism-build-20260927/runtime-metabolism-evidence/run-557-1790485072372899919/`。ヘッドレスinstrumentの設定拒否の生記録は同じevidence親ディレクトリの `headless-557-1790485072372899189/`。

OFF/ON各二回で、同条件のWAV、`population_step`、代謝recordが一致した。OFFの代謝recordは0件、ONはframe 0と48の2件。四回のWAV SHA-256はすべて `908e260dbc33af7a155473ebf53a47eb270fe4a4c060ef7cd4b18001635f7926`。この固定条件では音声差は出ず、代謝状態の差を検査している。

frame 48のHarmonic実Entrain energyはOFF `0.78194094`、ON `0.7596431`、差 `0.02229784`。登録閾値 `1e-6` を超えた。ONの代謝receiptも同じ実energyを保持し、JSON表示精度をf32へ戻してbit一致を確認した。`population_step` のHarmonic Entrain数は1、Droneはframe 0と48でEntrain数0・平均energy null。

ONのframe 48 receiptはHarmonic現在基音440.0 Hz、body generation 336、score -0.224905014、level 0.38940597、support endとdecision sampleはともに24576。Drone現在基音466.000061 Hz、body generation 1、score -0.35328817、level 0.3303558、同じsample時刻。両sourceで正のband massとsupport age上限を検査した。Harmonicのdensity build 336・hit 56、Droneのbuild 1・hit 391で、各合計はlifecycle/onset評価数と一致した。frame 0だけ共有初期履歴origin、frame 48はsource別の受理済み環境。

### generation 336の原因と意味

これはVoice出生世代でもHarmonic音色の336回変更でもない。生記録では両sourceの出生 `generation` はframe 0/48とも0。frame 0のHarmonicはlifecycle評価8回、body generation 1、density build 1・hit 7。frame 48は累積評価392回、body generation 336、build 336・hit 56。Droneは同じ392回の評価でもbody generation 1、build 1・hit 391。`OfflineMetabolism::evaluate` は基音だけ0にした代表 `Recipe` のhashが前回と異なるたびに独立の `body_generation` を増やし、完全Recipe identityの不一致なら密度を再生成する。したがって、この数値は代表Recipeの変化回数とcache再生成回数を示す。

固定シナリオではHarmonicの音色制御、基音、ADSR、代表holdは変わらない。Entrain Holdの代表modulatorには `KuramotoCore::rhythm_freq` が `autonomous_pulse.rate_hz` として入る。`update_phase` が `omega_rad / TAU` でこの値を更新し、代表化はlive位相・初期envelopeを固定する一方、rateを保持する。`Identity` はrateのf32 bit列をhashする。この経路がHarmonicの反復invalidationsを説明する。既存JSONLにはrate/hashそのものの列がないので、336個の個別rate値を生記録から直接再現したという主張ではない。

rateは代表 `Tone` のmodulatorへ渡り、render中のpulse位相加算には使われる。しかし代表密度生成は `NeuralRhythms::default()` を72 hop進める。ここではthetaのmagとalphaが0のまま、Entrain既定閾値0.04/0.2を満たせず、自律pulseが周回しても再attackしない。さらにこのSustainはkick後の正のsustain level 0.9へ留まる。したがって、今回の代表PCMと密度はrate変化で変わらない。hashにrateが残るため、336回のbuildはこの条件では過剰なcache無効化。代謝差は環境に対するscore/level評価と実energy更新の証拠であり、336種類の異なる可聴Body密度の証拠ではない。この所見はsourceと保存済みrawの監査で、追加renderは行っていない。

`conchordal --play=false --nogui` にoffline専用flagを渡す負例は起動拒否を確認した。ログとstderrを保持した。

初回取得は四renderを完走し、最後のreceipt対population energyのJSON `f64` 数値直接比較だけ失敗した。前者 `0.7596430778503418`、後者 `0.7596431` は同じf32値の異なるJSON表記。生記録は `run-536-1790484991822029339/`、失敗ログは `target/runtime-metabolism-validation/normal-render-first.log`。比較を双方f32へ戻したbit一致に修正し、登録済み実energy差と他の条件は維持した。修正後の中間成功ログは `normal-render-f32-repair.log`。

この検査が示す範囲は通常render経路の実Voice代謝入力と実energy差。非同期実時間代謝、出生経路、長期生態、代謝による音声差は未取得。全suite、format、Clippy、全target checkと別の実時間取得の判定は[第十三版b統合記録](body-fitness-metabolism-validation-20260927.md)を参照する。
