# 身体評価 第十三版bの統合検証（2026-09-27）

隔離worktree `.worktrees/body-fitness-metabolism` で、第十二版bから、通常offline renderの代謝入力、初回Field出生の残余境界、parentありPeakBiased出生を進めた。非Sineの身体・control変更は別の実時間取得で検査する。mainの通常実装への採用とは区別する。

## 通常renderの代謝

[登録](body-fitness-offline-metabolism-registration-20260927.md)に従い、既定falseの `body_fitness_metabolism_offline` を決定的offline配線だけに接続した。lifecycleはautonomous/glide後、Gated onsetは実onsetへのrecharge直前の現在基音と代表身体を評価する。target基音や移動候補表は使わず、72 hopの実Toneから求めた密度を自己除去済み環境のF2 score/levelへ渡す。frame 0の共有初期履歴は別originとして報告する。

[固定シーンの取得](body-fitness-offline-metabolism-results-20260927.md)では、seed 7、0.64秒、Harmonic 440 Hz EntrainとSine 466 Hz Drone、移動OFFで、代謝OFF/ONを各二回実行した。frame 48の実Entrain energyはOFF 0.78194094、ON 0.7596431、差0.02229784。各modeの音声・population・代謝記録は再現した。この条件では四回のWAVも同一であり、音声差や生存選択を示した結果ではない。

[通常providerの内部検査](body-fitness-normal-offline-provider-focused-results-20260927.md)は、自己除去済み環境と独立した72 hopのTone参照、現在基音とtargetの分離、point毒入れからの独立、scoreとlevel、実Gated onsetとpoint gateの分離を確認した。同じRecipeなら密度を再利用して現在環境で再採点し、基音変更で再生成する。source ID・出生世代・出生時刻、epoch、支持鮮度、時刻overflow、space/scanの不一致を拒否した。

静的監査では、途中出生直後のVoice集合が前hopの受理環境と一致しない経路を確認した。今回のflagは初期4声・ID一意までとし、途中Spawn、ReleasePopulation、SetRespawnPolicy、解析設定変更を配線前に拒否する。instrumentの拒否は音声デバイス初期化より前に行う。source解析workerのshutdown時の再panicを防ぎ、renderは全threadをjoinしてから最初のエラーを返す。これらの負例も検査した。

固定Harmonicの代表Recipe世代336は音色変更数ではない。Entrainのlive pulse rateがhashに残って変化するため、今回の代表条件ではPCMへ影響しない値の変化でも密度を再生成する。過剰なcache無効化を未解決の費用課題として残した。非同期実時間代謝の成立や費用受入へは外挿しない。

## 出生の試験専用接続

[F4f残余](body-aware-fitness-f4f-v13-residual-results-20260927.md)は11件成功。Dissonance/EdgeのPeak/Density、狭い範囲とclamp、最終spacing、実出生event、全占有fallback、既定OFFとGap/Uniformを検査した。二音環境の候補抽選確率で、Sine対Harmonicの全変動距離は0.056282911、Harmonicのrho 0対3は0.101807393。反復出生による頻度推定ではない。

[parentありPeakBiased](body-aware-fitness-f4e-parent-results-20260927.md)では、二親の実energy更新から重み付き親選択、予約した子の44 binと32局所点の身体評価、出生までを照合した。選択親1、子ID 4・世代1、最終440 Hz。局所探索は動いたが、この条件では中心からの周波数移動は無い。高い最低levelで出生を拒否し、親energyなど6種類の古い入力も拒否した。両出生経路とも `cfg(test)` であり、通常runtime出生配線の完成ではない。

## 非Sine実時間取得

[専有取得](body-fitness-non-sine-changes-runtime-live-results-20260927.md)はOFF合格・ON不合格。両条件とも2声・938 hopを維持し、underflow 0、hop予算超過0。ONの最大hop費用は1.183347 msだった。一方、frame 300の身体変更後、登録期限frame 432までの新世代消費が無く、初消費はframe 522だった。旧世代pendingの完了待ちと132候補の冷計算を確認した。frame 912までにはHarmonic 46回・Modal 40回の消費があるが、期限未達は解消扱いにしない。旧計算の取消と冷計算費用の分解・改善を、次の優先単位とする。

## 固定物と最終検査

最終全suiteは1285成功・0失敗・47 ignore。`test_status.txt` は `cargo test exit=0 @ 2026-09-27T14:14:54+09:00`。全出力は同worktreeの `test_report.txt`。format・標準Clippy・全target check・release test buildは通過した。ignoreを明示解除した非Sine実時間取得は別実行でexit 101、OFF合格・ON不合格。通常suiteの合格にこの失敗を混ぜていない。

405ファイルのsource capsuleは `target/integration-source-20260927-v13b/`。manifest SHA-256は `2bc58ef9bffdb12eb50c042a1bcc678d9197d2d081f9d27d60b67bd263823b61`。初回第十三版からのsource変更は `src/life/voice.rs` のOption照合だけ。初回sourceは `target/integration-source-20260927-v13/`、最初のsuite/Clippy記録は `target/metabolism-validation/initial-v13/` に保持した。

`target/metabolism-validation/validation.json` は179個の検証artifactと9個の登録・結果文書をhashで対応づける。SHA-256は `8a13443dfed808e4efce3b53d1d8362f8828748be961e02d66b36998bad99f25`。通常renderの最終全suite取得、初回JSON比較失敗、親あり出生、provider focused、初回/最終suite、専有実時間取得と保存binaryを含む。mainへ同期するのは計画・結果文書で、sourceは隔離worktreeに保持する。

## 失敗の保持と残余

通常renderの初回energy照合は、同一f32を異なる精度でJSON表記した値のf64直接比較で失敗した。f32に戻したbit比較へ直し、登録したenergy差の閾値は維持した。providerの初回2失敗は、更新前後のRecipe同一性とWhenViable初期境界に関するfixtureの誤りだった。後者の初回出力は会話内のみで、独立ログファイルは無い。最初の全suiteは1285成功だったが、標準Clippyが追加経路のunnecessary_unwrapを2件検出した。第十三版bではOption照合へ置換し、再検査した。初回sourceと全suite/Clippyログも保持する。

残る実装は通常runtimeの出生接続と公平な非同期代謝政策。固定身体の長期生態、移動のみ・生存のみ・全接続の除去比較、長時間の資源と延期率、F5の感度・試聴・作者採用、F6の音色遺伝も未完了。I11正式費用14/16とModal記憶参照0の未達は変更しない。mainのsource変更、commit、pushは行っていない。


## 次の取得単位

読み取り監査では、offline専用出生評価器をWorkerStateが所有し、出生機会だけConductorからCommunityへ可変参照で渡す構造を次候補とした。試験用thread-localをproductionへ昇格しない。最初は出生flag単独で、frame 0の環境音源に約64 hop後のField子1声を加え、予約子の全候補F2、順位または抽選、最終Recipe・出生eventを通常renderで照合する。

出生前の共有PCM環境は、まだ発音していない子に自己除去を施さず使う。代謝と同時接続する段階では、既存Voiceはイベント適用前の集合で受理した自己除去環境、新生児の初回は共有環境へ分ける。cleanup後に生まれる子の無音PCM slotをrender前に準備し、出生時刻つきsourceとして次hopへ引き継ぐ。これらの遷移を検査するまでは、第十三版bの途中出生・respawn拒否を維持する。親ありrespawnはその後の独立取得とする。この段落は未実装の次工程案である。
