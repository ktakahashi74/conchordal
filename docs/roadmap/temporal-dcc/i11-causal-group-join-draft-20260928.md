# I11：自己音除去と共有音響群を結ぶ通常取得案

2026-09-28。これは新しい隔離版の実装・取得前草案であり、入力、統合source、検査器、binaryの凍結済み登録ではない。実装も数値取得もまだ行っていない。既存の[三座標投影結果](../../../target/i11-group-projection-20260928/results-v2.md)は、単独Aの完備92組で群との算術的一致、重畳Bの87組で不一致、reset Cの67組中66組で一致を示した。Aの一致は、同じ入力と同じ群配分を使った構成上の整合であり、物理音源から群への帰属や自己群除外の正例ではない。既存rawを再集計してこの問いの正例へ数え直さない。

## 一件の目的と完了範囲

通常`conchordal-render`の各判断に対し、実際に消費した到来snapshotと、その判断以前に受理済みの自己音除去観測を、producerのhop・bus・identity・支持時刻で結ぶ。共有群を記述した各hopについて、同じ群配分を混合音と自己音除去後の音の**それぞれの解析power**へ適用し、群ごとの変化と残存を記録する。単独音源と別音源重畳の両方を、未見の固定入力で全件取得する。自己音除去後にも群成分が残る例、受信が間に合わない例、CDFが無い例を除外しない。

この一件はreport-onlyの因果対照と時計・由来の整合を確立する診断である。群を「自分だけの群」と判定する新しい閾値、`body→medoid→group`の置換、`ArrivalSet::select`での新しいhard除外、到来係数・CDF・乱数・既定動作の変更は含めない。自己音を除くとgroup powerが下がっても、NSGTの履歴・平滑化、音源間の位相干渉、群分割・mergeがあるため、その差を物理的所有率や独立到来事象とは呼ばない。後続のhard除外には、他者由来の残存と欠測をどう扱うかを別途取得前に決め、通常の実除外と反例を検証する必要がある。

## 統合対象と取得位置

新しい隔離worktreeで、封印済みI11系sourceとSourceRemoved系sourceの由来・差分を先に固定する。主な接続点は次の通り。

| 継承候補 | この草案で現物確認した由来 |
| --- | --- |
| I11側：[`.worktrees/i11-window-inputs`](../../../.worktrees/i11-window-inputs/) | 封印済みの518-file [source manifest](../../../.worktrees/i11-window-inputs/target/i11-window-inputs-fixed-source/source-manifest.json) SHA-256 `f064313c63a9d6768757fb35a2338e0034be58198f010bd643f5abcbe2e3f76d`、[source archive](../../../.worktrees/i11-window-inputs/target/i11-window-inputs-fixed-source/source.tar.gz) `f0d2286d40971fdd008043c80264dae94e0fc5a5a5e61327fbafa29060b31f85`、195項の[最終索引](../../../.worktrees/i11-window-inputs/target/i11-window-inputs-final-index.json) `40b6e4e182173d9e7209665c9157e3f4ac8f5e015c05afbc34a3b10701568e7c`。今回の実装元とする版は、取得後訂正されたPython検査器を含むこの最終索引に明示して選ぶ。 |
| SourceRemoved側：[`.worktrees/body-fitness-public-guard`](../../../.worktrees/body-fitness-public-guard/) | [継承manifest](../../../.worktrees/body-fitness-public-guard/target/inherited-source-manifest.json) SHA-256 `5f10c3ef55d10766fddc940c4f130ac6e78644f3c5884de1159274d13890928e`は、その前の封印済みboundary-observation sourceからの系譜を示す。封印済み506-file [fixed-source-v1 manifest](../../../.worktrees/body-fitness-public-guard/target/public-guard-validation/fixed-source-v1/source-capsule/source-files.json) SHA-256 `559e7c39622e9129c989781fca5b0255dd744b4f5a67a3633f3836fbba772f2f`。[最終索引](../../../.worktrees/body-fitness-public-guard/target/public-guard-validation/final-index-v1.json)は1191項・237,321,718 byte、SHA-256 `23035ef81d75892561c45269cdd7cb7d74200dbef09bc2a210efc0e4ef473da7`。必須検査と取得を終えて封印された版である。 |

両worktreeをそのまま同一版とみなさない。統合前に双方のmanifestのpath集合とbyte差分を取り、I11の到来判断・時間観測とSourceRemovedのworker・observer・runtime配線について採用元をファイル単位で記録する。予定する実装差分は、下表の入口におけるreport-only producer証拠、判断前の非blocking受理参照、二段joinの識別・時計、必要なconfig/report配線と局所反例に限定する。既存の音響群形成、SourceRemovedのPCM演算、I11の費用式・CDF・乱数・自己群hard除外を差分へ紛れ込ませない。`runtime/mod.rs`は両系とも変更が大きいため片方を丸ごと上書きせず、`process_hop`の描画前後順序とsnapshot/observer配線を行ごとに照合する。両封印元の上記manifest/index/hashを新しい取得planへ固定する。

| 既存入口 | 必要な接続 |
| --- | --- |
| I11の`src/life/community.rs`、`src/life/temporal_participation.rs`、`src/life/arrival_cost.rs` | 判断が保持した`ArrivalContext`、`ArrivalProvenance`、候補窓、選択・除外群と理由をそのまま記録する。`self_group`は現行のroute・身体Record・Binding・共有assignment・CDF検査を変えない。 |
| I11の`src/temporal_cognition/observation.rs`、`context.rs`、`group.rs`、`trajectory.rs` | 共有producerが受理した各habitat hopの`power_scan`、peak、実assignment rows/weights、group handles、source epoch、frame start/end/support/available、受理・欠測・退役をreport-onlyで保存する。後のsnapshotから過去の配分を推測しない。 |
| SourceRemoved系の`src/core/source_removed_analysis.rs`、`source_removed_worker.rs`、`src/runtime/body_fitness_observer.rs` | 実際のmixed habitat PCMと同一hopの各Voice自己PCMから既存workerが作る`mixed − own`の解析結果を利用する。`OutputBatch`のframe、support end、source id/generation/birth sample、提出・受理・判断時刻、欠落・拒否理由を保存する。音声から別の擬似群を作らない。 |
| 統合版`src/runtime/mod.rs`、必要なら`src/config.rs`・報告writer | `body_fitness_observation`相当のopt-inでobserverを起動し、I11のarrival報告と同時にproducer側証拠を配送する。action・metabolism・birthの新しい作用を有効化しない。通常音声、候補、費用、RNGへ診断値を戻さない。reporter無しとarrival OFFでは、この診断の配列複製・追加heap割当を発生させない。 |

現SourceRemoved observerは48 kHz・512 sample/hop、最大4 source、受理結果の年齢4800 sample以下を要求する。現実装のworker epochは0で固定される一方、I11のtemporal source epochは入力更新で変わる。数値が同じだけで両epochを同一視しない。統合版はruntimeの分析再構成・source退役時に両系のepoch対応を明示して記録する。対応を記録できないepochはjoin不明とする。現observerの`decision_batch`はactionから呼ばれる入口であり、I11だけをONにしても自動で判断直前に呼ばれない。統合版には、直前に届いたbatchを判断前に非blockingで受理・照合し、report用に参照できる最小入口が必要である。この入口はI11の費用計算へ結果を返さない。既存の`body_fitness_observation`と`temporal_onset_comparison.arrival`を同時指定できるか、統合sourceのconfig検証と局所反例で確かめる。未対応なら取得前に明示的なopt-in配線を完成させ、暗黙に別機能を有効にしない。

## 二段階joinと採否

I11の参加判断は`advance_population`中、当該hopの描画より前に起こる。SourceRemovedの`observe`はそのhopの`render_and_route_audio`後である。したがってhop `f`の音を除いた結果を、hop `f`の判断が既に消費した入力として結ばない。以下の二段を独立に検査する。

1. **producer同hop比較**：bus 0、48 kHz/512 sample、明示的に対応付いたepoch、同一frame start/endとsupport end、同一mixed PCM取得、同一source id/generation/birth sampleの自己PCMを要求する。共有側のframeが受理され、source側も受理され、時刻とscan gridが一致した場合だけ、共有側で実使用した`w_g(k)`をmixed解析powerとsource-removed解析powerの双方に適用する。群slot 7の残余配分も保持する。片側の解析履歴は独立なので、`power(mixed) − power(own) = power(mixed − own)`とは仮定しない。全binの有限性、非負性、配分和、group handle/generation、履歴resetを検査する。正常な観測ゼロと未支持・未受信は別状態にする。
2. **後続判断へのas-of join**：`participation_decision.now`以前に実受理されたproducer記録だけを候補とし、判断が保持したsnapshot frame/source epoch、`Shared.end_sample`、群handle/generation、CDFのissued/horizon、実採用・除外集合へ対応させる。群配分はproducer hop当時の値を用い、判断時の最新配分で過去hopを書き換えない。`SourceRemoved.support_end_sample <= received_at_sample <= decision.now`を要求し、observerの4800 sample年齢条件とI11側の各窓・期限条件を独立に適用する。実判断が消費しなかった後着結果を、後から過去判断の証拠へ挿入しない。複数の適格hopは群の実支持窓と取得順序で全件対応させ、好都合な一hopだけを選ばない。

報告schemaは判断ID/Voice identity、source birth sample、body generation、route、producer-local sequence、frame/support/available/received/decision時計、epoch対応、群handle、group slotと配分行、CDFのissued/horizon/periodic/採否、mixed・removedの全bin scanまたは独立再演に足る固定入力、各失敗理由を含む。大きなscanはreport有りの場合だけ有界queueで運ぶ。送信drop、未完了・重複sequence、末尾欠落、source数不一致は明示し、完全joinへ数えない。worker間のJSON到着順は因果順序の根拠にしない。producer同hop比較と判断as-of対応は別々の状態として記録する。それぞれ`complete`、`no_eligible_cdf`、`source_missing`、`shared_missing`、`clock_mismatch`、`epoch_unknown`、`identity_mismatch`、`stale`、`report_dropped`等の主理由と副理由を保持し、判断総数とproducer hop総数を別分母として各数を示す。CDFが無い群や自己群候補0も分母から消さない。

独立検査器はproducerのpeak/assignmentから`w_g`を再構成し、各binで配分保存とmixed・removed投影を再計算する。`ArrivalProvenance`からCDF全表、候補窓の確率、採用集合と費用も既存の独立検算規則で再計算する。PCM→NSGTそのものはこの一件の独立実装対象外であり、producerのpowerを入力として信頼した境界を明記する。source-removedが群の物理的所有を証明したとは扱わない。`SourceRemoved`のepoch誤対応、birth違い、bus違い、未来結果、古い結果、handle再利用、旧group配分の流用、報告末尾欠落、支持されたゼロと欠測の取り違えを合成反例にする。

## 新規scene案と取得前固定

新しい二つのresearch sceneを作る。単独Sはhabitatにも送るharmonic flow参加Voice一つを持続させる。重畳Mは同じ参加Voiceに、別idのhabitat音源一つを固定時刻から重ねる。候補は参加Voice 170 Hz、追加音源340 Hz、固定seed `20261217`、48 kHz/512 sample、約14秒で、前のA/B/Cとは別scene byteを用いる。周波数・時間・音量・route・script上のrelease、全config値は**取得前**にRhai入力をcompile-only検査して確定し、初回render後にCDF正例を探して変更しない。専用beat carrierを後付けして群を作らない。Mの追加源によって参加Voiceの実PCMが変わり得るため、同一prefixを想定せず、sourceごとの支持とPCM hashを確認した範囲だけで「混合のみの介入」と述べる。CDF群0、decision0、SourceRemoved欠測、epoch不一致、完全join0でも両sceneの結果を保持する。

両sceneで統合診断ONのreport有りを各2回、report無しを各1回、SourceRemoved診断OFF・到来設定同一のreport有りを各1回取得する案とする。ONの二反復は判断・join分類の再現性を見る。report有無ではWAV byteを比較する。報告されない判断を推測で比較せず、同入力の決定的局所fixtureで候補・費用・RNGの不変を別に検査する。observer ON/OFFの両reportでは、WAV byteと、既存判断・候補・CDF記録から新診断fieldだけを除いた内容を比較する。SourceRemovedの非同期受理時計には揺れがあり得るため、診断分類までbit一致を無条件要求しない。音声・既存決定の変化は非干渉不合格として保存する。到来OFFの取得を加えるなら、ONとの選択差を許し、observer/report有無の音声比較と混同しない。

取得開始前に、新統合sourceの全ファイルcapsuleと両継承元との差分、configと二sceneの原byte、seed、モデル・尺度・閾値0.25・hazard係数、build profile、binary、checker/反例、全実行argv、出力先・判定条件をplanのSHA-256で固定する。旧I11/SourceRemovedの封印source・raw・索引は変更しない。Rust source変更時はrepositoryの`cargo fmt --all`、標準Clippy、全target check、`RUST_BACKTRACE=1 cargo test -- --nocapture`のstdout/stderrと同shell exit記録を終えてから取得する。失敗runもWAV・JSONL・stdout/stderr・status・hashを残す。実device性能や通常稼働資源の受入は本取得で主張しない。

## 判定と残る問い

技術判定は二段joinの完全性、既存CDF/費用の独立再計算、非干渉、欠測・未来結果・世代交代の反例を対象にする。科学的な結果は、各群で自己音除去後の投影量と残存の符号・大きさ、単独/重畳での分布、完全joinかunknownかを全件報告する。完全joinが一件以上あっても、`ArrivalSet::select`による**実自己群除外の正例は依然0**であり、この診断だけをI11受入としない。残存を許すhard除外の意味、群split/mergeと複数Voice寄与、音源別の非線形power、群・CDFが同じ支持窓を代表するかは未解決のまま記録する。

I9の研究拡張は本単位の前提にも代替証拠にも使わない。[I10記録](i10-body-outcome.md)の実身体候補energy予測は、固定development素材の4,936実音対で一致した限定成果であり、未知素材への転用・認知的妥当性・R2/A1/A3受入を証明しない。I11の通常自己群除外0とは別の判定として保持する。


## 2026-09-28 統合取得後の状態

隔離版 `i11-causal-group-join` でreport-only接続を実装し、全Rustテスト1197成功・失敗0・ignored36、固定S/M計8renderを完了した。各sceneの4 WAVとreport付き3本の判断28件は同一。保存powerとassignment rowから各1314hop中1291hopを投影でき、判断前履歴はS11件・M12件に対応した。実report契約を誤読したchecker v1〜v3と失敗/偽不一致を保存し、同rawをv4で修復監査した。

新固定Mでは既存規則の自己群除外1件がOFFを含む同一判断に現れた。従来の通常自己群除外0は過去素材に限定した記録となる。新observerによる判断変更ではなく、body/Binding/CDF完全joinと物理的帰属は依然未確認である。結果は隔離版の `docs/roadmap/temporal-dcc/i11-causal-group-join-results-20260928.md`、最終索引SHA `8c79033ee1d73d7e3dc2c106ca6b1b2b4087918e0735b25f6d53c7696cd7cf3f` に封印した。main source採用とI11全面受入れは行っていない。


### 2026-09-28 保存済み全判断の追加照合

[事後解析v4](../../../target/i11-complete-join-20260928/results-v4.md)でON4取得の全112判断・5,256 producer hopを照合した。提出・受理・shared input・asofの分母とkeyを全件確認し、整合エラー0。完全対応は0、unknownは112である。GroupPrototypesの同hop独立報告と消費値の直接一致は14判断で成立した。残り98判断は間引きによる欠測であり、Mの自己群除外各1判断もここに含まれる。身体Record・Binding・assignment・CDF・群履歴の個別一致を、全段階の成立へ繰り上げない。

次の記録追加は、消費snapshotのproducer原点、route/birthの独立原点、群予報の発行原点を分けて設計する。物理的source帰属はこれらのevent追加だけでは証明できない。v1のschema誤読、v2の同hop報告順誤判定、v3の末尾受理分母検査不足を別版として保存した。v4は12反例を通過し、独立レビューと主担当の25入力hash照合を終えた。新しい音声取得やmain実装の変更は行っていない。


### 2026-09-28 原点記録版の全判断照合

隔離版 `i11-origin-records` の全8renderは正常終了し、旧版の全8 WAV・report付き6本各28判断と完全一致した。終端未消費観測を誤って欠測とした新checker v1の4失敗を保持し、sourceの終了後drain経路に対応する別版v2を固定した。v2は実shared producer入力と対応する終端1観測だけを分母外へ分離し、途中欠測・未来参照・同frame差替えを拒否する。17局所検査と30固定入力hashの照合後、同じ4 ON rawを再監査した。

全112判断で消費snapshot、現行Voice registry、実出生原点の照合が成立した。全5,256 producer hopとの対応に整合エラーはなく、body/Binding/CDF/群履歴までの `complete_observable_join` はMの二反復で各1判断、残り110判断はunknownだった。成立した判断は両方ともvoice1・now387584・群(bus0,epoch0,generation63)で、既存の自己群除外に対応する。同一scene/seedの同一判断の再現であり、独立した二条件の正例ではない。Sの各28判断はBinding欠測、Mの各27判断はBinding欠測25・assignment未対応2を残した。CDF群欠測は重なる副理由として保持した。物理的source所有、保持中Toneのrouting、I11の全面受入れは未証明のままである。
