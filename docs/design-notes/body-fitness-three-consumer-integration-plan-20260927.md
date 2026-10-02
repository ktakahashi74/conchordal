# F4 三消費者統合の実装準備

2026-09-27。これは[全消費者草案](body-fitness-all-consumers-draft-20260927.md)、[実装レビュー](body-fitness-all-consumers-implementation-review-20260927.md)、[F4計画](body-aware-fitness-plan.md)に続く設計文書であり、数値fixtureの取得前登録、結合済み実装、取得結果ではない。対象は移動、代謝、出生候補の三消費者を同じ通常offline実行で独立に点／身体へ切り替え、同一Populationの異種founderと非遺伝の固定子を比較する最初の単位である。

## 二つの固定源版と結合境界

同hop重なり修正版の固定415ファイル源版と、出生なし四条件比較の固定419ファイル源版は同じGit HEAD `06a4772c43d06b41b44753bb0be891f23e16b93e`から分岐した。両作業木の現`src/`を読み取り比較すると、差は9ファイルに限られる。ただし移動診断が進む作業木を結合の入力として直接使わず、各固定manifestに載る実ファイルとhashを先に再照合する。以後の記述は固定版の差分に対する結合手順である。

共通部分は共有baseを使う。重なり版からは`runtime/mod.rs`の前児翌hop receiptのcleanup前照合、`next`にcleanup前集合を記録してから`birth`にcleanup後集合を記録する順序、`community/respawn.rs`と`offline_respawn.rs`の退役tailを数えた投影source容量preflight、対応する故障注入・通常renderer試験を保持する。出生なし比較版からは`ScoreBasis`、`OfflineMetabolism::new_with_basis`の同一準備・点score置換、`Ready::score_with_basis`と`BodyFitnessAction::new_with_basis`の同一prepared gate、`request`が実Recipe identityとgenerationを更新してから捕捉する修正、四条件の専用設定・検証・reportと試験を移す。両源版で完全一致するソースは再実装しない。

`runtime/mod.rs`、`community/respawn.rs`、`offline_respawn.rs`は意味上の競合箇所であり、どちらかのファイルを丸ごとコピーしない。比較版は旧`voices.len() <= 3`容量検査とcleanup後の前児報告を残しているため、そこを統合版の基準にしない。重なり版の`offline_respawn_tests.rs`にだけある「保持中の死者が第四sourceを占める場合のRNG・ID・counter不変」試験と、`runtime/mod.rs`にだけある同hop重なり試験も落とさない。結合後に既存OFF経路、重なり版の通常renderer事例、比較版の四条件を別々に回帰させてから三消費者へ進む。結合先、source archive、試験ログは新版として別に固定し、415／419版の証拠を上書きしない。

## 同一Populationの異種founderと固定子

現`Action::Spawn`は全memberへ一つの`VoiceSpec`を渡し、`on_spawn_action`はそのaction内でmember indexを0から振る。`ensure_population_state`は同一Populationへの次のSpawnで単一templateを更新する。したがって別Populationを親poolとして跨がせたり、同一PopulationへのSpawn二回や`cfg(test)`専用注入で異種founderを作ったりしない。`UpdatePopulation`も全memberに同じcontrol patchを適用するため、個体別身体宣言の代用にしない。

通常`conchordal-render`が読めるRhai入口で、単一Populationの一回のtime-zero spawnに、順序付きfounder `VoiceSpec`列と固定child `VoiceSpec`を明示する。最小案は専用の`place_members([founder_spec_0, founder_spec_1, founder_spec_2], child_spec, placement)`を一つ追加し、単一のScenario actionにmember別spec・一意のID・member indexを保持する形とする。通常の`place`と既存`Action::Spawn`は維持し、専用actionを通常のdispatchと`spawn_one`へ通す。入力検証はfounder数3、全specの共通の非身体条件、同じPopulation policy、固定子spec、respawn capacity、初期配置と時刻を確認する。身体method、代表Recipeに効くbody control、初期f0は明示値をそのまま記録する。Harmonic／Modalの同一amp設定を同一放射RMSと扱わず、音色と初期f0の交絡は別対照へ残す。

`RuntimePopulationState`のrespawn用templateは、最後のfounderではなく明示した`child_spec`を保存する。親抽選前にそのspecを固定し、planned childのID、population ID、member index、系譜generation、parent IDと合わせて候補身体を構成する。親は更新後energyによって選ぶが、親の身体やcontrolを子へコピーしない。`evaluate_child_body`の仮Voiceと実`spawn_one`を同じ固定spec・確定metadata・候補Hzで作り、Recipe hash、body snapshot、route、最終Hzを照合する。初期member indexは0,1,2、最初の子は3とし、同一Population内で重複させない。音色遺伝はこの単位の機能に含めない。

移動比較の入力には`at()`を使わない。現行の`place(..., at(freq))`は`set_freq_lock_clamped`を通り、先に指定した`seek_consonance()`のFreeをLockへ上書きする。単一PopulationのLinear配置など、初期周波数を割り当ててもpitch modeをLockへ変えない経路を使う。assay validatorはIR中のfounderと固定子のFreeを検査し、通常spawn後とrespawn後の実VoiceでもFreeを検査・記録する。既存の出生なし四条件はLock下の採点消費だったため、その音高不変を閾値や移動費用だけの結果とは扱わない。

## 時計、source、出生環境

三消費者は同じ解析系のepoch/space/habituation版、decision clockとsupport終端を照合する。既存Voiceの移動・代謝は各sourceのSourceRemoved batchを使い、未出生の子だけは後述のshared環境を使う。代謝contextはhop前のbatchから各生存Voiceへinstallし、onsetとlifecycleをそのhopで更新する。actionの`before`は同じbatchからprepared判断をinstallし、通常のVoice進行後に`after`で結果を確定する。前機会の子の翌hop SourceRemoved receiptが期限を迎えたら、死亡の有無に関係なく**cleanup前**に完全identity `{id,generation,birth_sample}`、support、decision clockで検証する。その同hopに新しい単独死亡があっても、検証済みの前機会は新出生を妨げない。親poolはこのhopの代謝・発音・lifecycle更新後、cleanup前の生存親2声と実energyから作る。stale receipt、親pool不足、投影source容量超過は新機会の親RNG・spawn counter・子ID/member index更新前に拒否する。ただしhop全体の通常advanceを巻き戻す意味ではない。

自己除去解析の上限は、現在生きるVoice数ではなく、保持中の死者tailと新生児を含む実source集合で4。三founderの死亡後も旧sourceがtailを保持するなら、その集合を残したまま子を追加して4以内か投影する。旧死者をobserverから外すのは`should_retain`が偽になったcleanup後であり、残るtailは新子が聞く環境に残す。容量を満たさない機会を黙って消費したり、容量のためにtailを早期除去したりしない。

出生候補の評価時点では子の自己PCMは存在しないため、点・身体の両basisとも**同一の出生前shared effective Landscape**を使う。親のSourceRemoved環境へ置き換えず、旧死者のtailも環境に含める。候補生成、Peak/Density・占有、全bin／局所／最終Hz、閾値、RNGを共通にし、出生候補のscore/levelだけを点補間か予定子の代表密度加重平均へ切り替える。子の出生hopは代謝receiptなし、自己PCMのhop長は全ゼロでslotだけ用意し、翌hopにその子自身のSourceRemoved環境と初回receiptを作る。記録の`next`はcleanup前source集合、`birth`はcleanup後source集合と明記し、機会番号・spawn sequence・子の完全identityで結ぶ。

## 三消費者の独立切替と比較

既存boolean flagのOFFを点対照として読まず、通常offline専用の三つのscore basis（action、metabolism、birth）を一つの明示設定にする。点条件も同じobserver、prepared gate、代謝receipt、出生候補処理を通り、当該消費者のscore/levelだけを同時点・同環境の点値へ置換する。`in_band_mass`は両条件で実代表身体から得た診断値を保持する。旧flag同時ON拒否をこの新設定に限って意味を整理し、通常OFF経路と既定は維持する。

三軸の全8条件を取得対象とする（順序は移動／代謝／出生）。`PPP`、`BPP`、`PBP`、`PPB`、`BBP`、`BPB`、`PBB`、`BBB`。基準、移動のみ、生存のみ、出生のみ、各二者結合、全接続を欠かさない。親重みは代謝後energyの派生なので、代謝basisを変えた条件では親poolの重み・抽選結果も変わり得る。出生basisの直接効果は、移動・代謝basisを固定した対応ペアの**最初の共通出生機会**で、親pool、親RNG snapshot、予定子spec、候補範囲と機会時計が同じ場合だけ読める。後続のsource集合やRNG列の一致は要求しない。四条件の相互作用を含む全条件比較は、同じ結果になったことだけを「効果なし」の証明にしない。

数値fixtureは別の取得前登録で、固定founderの身体・初期f0・放射量、固定子body、seed、代謝係数、期間、死亡／出生機会、解析設定、許容差を先に決める。各条件を最低2回走らせ、条件内WAV・決定的recordの一致を検査する。初回分岐前には完全source identity、候補集合とRecipe、support/decision時計、epoch/space/habituation、RNG snapshotまたは既存の完全消費検証を一致させる。各判断のReady／拒否／保留、有効score使用数、採択proposal、targetと実f0、receipt、更新後親energyと抽選確率、子候補score/levelと実Hz、出生hopと翌hopのsource集合を保存する。診断用RNG probe一値だけを全状態一致の証明にはしない。

F4の移動介入を成立とするには、同じ前分岐入力を持つ対応ペアで有効prepared判断を実消費し、少なくとも一つの採択targetまたは実f0が移動basisによって変わることを取得前に要求する。`score_uses > 0`だけでは足りない。[移動停止診断の登録](body-fitness-movement-diagnostic-registration-20260927.md)は現fixtureの採点内訳と不採択理由を調べる別単位であり、その結果に合わせて本比較のseedや係数を事後選別しない。異身体のenergy・生存時間差と親選択・出生地点の差も分けて報告し、同一身体のf0交換、放射RMSの対照、親重みを固定した直接出生効果は必要なら別登録にする。

既存の`tests/body_fitness_action_changes.rs`には、Drone、`move_cost(0)`、`temperature(1)`、`glide(0.04)`、`proposal_interval(0.05)`、seed 17、220/330 Hzで非Sineの通常action移動が起きた先例がある。これは新しい移動fixtureの候補を設計する際の既知の入力に限る。当時のaction単独positiveは、三消費者同時経路や点／身体の差の因果証拠ではない。現行四条件の停止診断へこの設定を混入させない。

最初の統合取得は四source以内の三founderと最初の子を扱う。後続の同hop重なりは結合回帰として保持するが、8条件での任意回数の反復出生、同ID再利用、長期生態、実時間費用、作者採用、音色遺伝までをこの単位の合格へ含めない。source上限や異種親poolの条件を満たさない取得は欠測または拒否として明示し、要求を二founderや別Populationへ縮めない。
