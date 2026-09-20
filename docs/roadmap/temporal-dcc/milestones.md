# 時間構造DCCの実行マイルストーン

更新日: 2026-09-20。状態: 2026-09-20に機構選択規則を採用し、本体をT1・T2・T4の短期反復期待とI12bへ縮小した（§1.3）。T3評定head・I7・I8・I12a／c／dは研究拡張へ移管。以下の2026-09-18までの記録は保存する。
旧状態（2026-09-18）: ユーザーの「一旦停止して、マイルストーンの再編案を実装して」に基づき採用。
**I8までの初期診断版の技術完了を維持。I9の範囲変更は `6a1a8c0`、I10の機構選定は `329f638` にcommit済み。I10のonset／release／shutdown診断、身体世代付き六記述値、動作条件付き自声予測を通常renderer／report／UIへ接続し、検査を完了した。I10全体は未完了。私有traceの有界Rust数値filterを検証し、その入力となる共有memoryの保持・検索確率上下限と、2秒の観測支持・anchorによるreference inventoryを通常接続した。実Voiceの発行時予測・reference凍結と、原観測順のtrace更新も通常接続した。候補帰結の数値核も独立参照で検証した。実音prototypeの適合と私有body／共有groupの通常descriptor照合も接続した。実rendererと共有するenvelopeを発行時予測へ固定し、音声保存を検査した。7 classのoffline実身体比較素材396分岐を取得・検証した。私有traceの候補／default時刻fitと、局所own-excluded energyの中心補間・16点診断も通常reportへ接続した。区間支持bitsetと実音6,336条件の積分監査を追加し、16点近似の誤差・順位反転を記録した。両busの私有source-energy予測・16窓の実音教師・両対照・学習前誤差も通常renderer／report／UIへ接続した。指令発行時の自声energyとhabitat自声除外予測の比率診断も通常report／UIへ接続した。10個の生特徴の数値式を観測／候補で共用し、出所タグとraw投影方針を検査した。7 classの比較素材は512／480 hopの発行前PCM付き取得へ拡張し、各396分岐を独立検証した。前史付き素材を通常の私有特徴抽出へ通し、396分岐・両bus・297,000未来hopの十生特徴と原支持も独立照合した。観測／候補共通の物理窓集計を通常phraseへ接続し、76,032窓の数値・出所・支持も独立照合した。7 classの身体入力則を取得器と共有し、時刻生成を通常traceへ接続した。396分岐の実音・metadataは不変。実Voiceのpolicy事実と既存指令／活動／予約onsetのdefault診断を通常renderer／report／UIへ接続し、268 recordを独立照合した。未予約の未来機会は未知のまま保持し、両素材の音声・既存予測は不変。既存8medoidの元前史と同一の7 class・1,032分岐を取得し、774,000未来hopの特徴を独立照合した。JSONのf64往復精度も修正した。条件付き身体行動profileをhash固定で通常workerへ読み込み、7 classのprimitive特徴窓・report・UIへ接続した。全1,032 bindingのf64とmaskを照合し、通常両busの未来72窓と音声・観測の不変性を検査した。候補articulationの有界scratch予測とphraseの250 ms入力・観測支持付きgroupingも通常接続した。元の未知質量と観測を保持し、閉形式・枝刈り・通常両busの数値と音声不変を検査した。固定expected segmentとの候補残差比較と、raw closure・経路別continuationも通常接続した。実音二素材の両busで独立数値照合と観測・音声不変を確認した。発行時coverage・音群重み・priorを固定したraw head混合も通常接続し、三素材の両busで支持とbackoffを独立照合した。共同正規化から候補帰結への数値経路も独立全列挙と照合した。局所成分の遷移と有界tuple合成を追加し、12ケースのDecimal全列挙・既存articulation遷移・未知枠・安定tie・scratch再利用を検査した。候補ごとのhead内音群支持を保持する契約へ修正し、通常raw混合と集計を共用した。10ケースの独立数値参照と三素材・両busの分布、観測・音声不変を検査した。候補accent密度を通常continuation head・report・UIへ接続し、4-hop支持と未来888窓、入力からheadへの変化、観測・音声不変を検査した。候補arrivalをcontinuation・report・UIへ接続し、合成profileで入力から分布まで、通常五素材で観測・音声不変を検査した。実音から凍結した112件の音群は固定medoid照合外であり、実音の候補点予測正例は未証明。articulation proposal核を通常観測と共同tuple側で共用化し、実音二素材の観測・候補表の数値差ゼロと音声不変を確認した。phrase親参照の不正をsection更新前に拒否する境界も接続し、拒否後の復帰と実音二素材の状態・音声不変を検査した。発行時の実Voice時計点と更新前周期を通常default診断の候補時刻列へ接続し、463記録の独立再計算、実時計との対応、三素材・両busの観測と音声不変を検査した。未来発音の許可・確約とは区別する。実身体396分岐・両busの転用監査で、descriptor距離だけでは無音busへの誤対応と候補energy順位反転が生じると確認した。既存profileを実Voice適格性へ昇格せず、身体条件と実policy defaultを含む転用検証が必要。新しい6素材から実policy defaultの未来指令と身体条件を取得し、保存JSON再生・発行時状態の複製再生を両busの実音と一致させた。無追加指令の対照との差も記録し、未来入力は教師に限定した。発行前に確定したrelease予約と身体envelopeからcontinue／release／gapの局所実音分岐を取得し、両busの56分岐・自然終端後の候補除外・予約脱落対照・元のpolicy音声不変を検査した。実時計の送出前機会と通常一回の発音recipeを結び、元のdueと予約releaseをreport／UIへ接続した。39 recipe、過去のdueとの30照合、通常9記録、元の172音声ファイルと共通531観測の不変性を検査した。同じ実機会の7 class・71分岐を固定recipeと既知releaseから取得し、両bus・568物理窓のlocal default差分を検査した。通常私有特徴抽出へ通した53,392未来hopの十生特徴と判断をまたぐhopの支持も独立照合した。既知の予約releaseを通常onset予測へ接続し、Off到着hop前後のattackを保持した。36条件、実機会52件の入力、既存172音声と503共通観測の不変性、短いreleaseの398予測座標変化を検査した。同一medoidのrouted busを使う比較版を検証したが、無音身体へprototypeの既存音を持ち込み順位反転が増えた。routing単独補正の採用を退け、通常loaderもこの比較版を拒否する。通常source-energy予測で実際の残存toneと指令toneを分離し、各routing・予約Off・世代と有界走査を接続した。70予測・2,240窓の成分、172音声と438共通観測の不変性を検査した。単純energy加算の固定／条件付き誤差は増加し、backend・modulator・位相を含む予測精度は未達。実modulator状態・予定kick・発音開始を指令／残存toneのenergy予測へ接続した。70予測・2,240窓の独立再計算、172音声と429共通観測の不変性を確認した。2,220窓の固定／条件付き誤差は合計で減少したが、固定誤差が増える3素材もあり、身体backend・位相と候補転用の精度は未達。通常の指令／残存toneと候補7 classで身体energy核を共用した。実機会71分岐・両busの9,088点と568窓のdefault差を独立照合し、通常97予測・172音声・483共通観測の不変性を検査した。実音とのenergy順位反転54／4,936対と、遅延onsetで位相相殺によるenergy減少を加算priorが逆符号に予測する反例を確認した。候補比較はoffline段階であり、位相を含む身体応答と通常消費は未達。実sine位相・seed・impulse boostとcarrier交差項を通常／候補energy予測へ接続した。sine-flowの1,520候補対の順位反転30→0、見落とした差8→0となり、相殺の逆符号も解消した。非sine等は明示的に旧近似へ戻す。通常70予測の成分と選択、172音声と478共通観測の不変性も検査した。全backendの精度と位相積の計算費用は未達。通常の候補許可と共有候補モデルの実身体条件付けは未接続。通常経路の共有proposal・path寿命・条件付き特徴・score更新は未接続。評価時刻の確定、全head特徴・帰結表、未予約時刻を含むactual-Voice/default対応・全資源検査が残る。**
旧目標「M0を完了させて」を今回の範囲拡張とは解釈しない。
共有候補表の失効を即時反映し、同一時計での再構築を100 ms以上に制限した。同一全軌道のCellと、articulationの完全hop途中状態を再利用する。60条件と合成448セル位置を直接計算と完全照合し、実音3素材の共通283表・音声・観測不変を確認した。直接呼出し65,184→53,635回に加え、1,535,132遷移hopの再計算を省いた。release最大構築は114.655→82.217→63.389 ms。表単独でも全worker 50 ms枠を超え、全Voice負荷・worker／hop全体の資源受入は未達。

共有worker全体の費用・公開遅延を全frame／100 ms窓の有界histogramへ接続した。実音6条件・両busの6,960 frameを測定し、sustain／stairで50 ms枠超過を確認した。4／16／64 Voice条件も取得したが、対応prototypeは最大1–2個であり全負荷受入ではない。音声・予測値は一致した一方、16／64 Voiceの到着時刻は変動し、厳密な全記録一致は不合格として保持する。観測queueは参照仕様の32件へ修正し、両busで飽和・復帰・EOFを検査した。6条件の再取得でも50 ms超過と到着時刻差が残る。その後、全構築の対応prototype数0〜8を累積計数へ接続した。16 Voiceの公開最大0に対し、実構築最大2を確認した。rate計算の1,496 byte設定コピーを参照渡しへ変え、既存6条件と非ゼロ係数対照の音声・数値を照合した。共通780表と789観測は一致し、16／64 Voiceの到着時刻差は残る。既存6条件の最大worker窓は65.456→57.542 ms、非ゼロ係数対照は75.461→68.096 ms。50 ms予算と全8 prototype負荷は未達。さらに候補の2秒／250 ms特徴窓へ渡すprofile frameを窓に重なる区間だけに制限し、13終端×2窓を全frame参照と照合した。7条件の共通784表・799観測・音声は一致、64 Voiceの到着時刻差は残る。最大worker窓は既存6条件56.459 ms、非ゼロ係数65.328 msとなり、50 ms超過は引き続き6窓。articulationの4個のframe共通入力を一度標準化して参照渡しする版も検証した。値渡し版は交互反復で退行したため不採用。参照渡し版は両素材・両busの3往復すべてで総処理費用が減り、通常7条件の共通783表・音声も一致した。最大worker窓は既存6条件54.112 ms、非ゼロ係数65.637 msで、50 msは未達。続いて異なる軌道間の完全一致raw prefixを再利用し、部分hop・巻戻し・group交代と224セル位置の直接一致を検査した。7条件のarticulation実計算は4,915,279→496,073 hop、共通781表・音声は一致。884窓で50 ms超過0、最大18.647 msとなった。交互3往復でも総表費用73.44〜76.68%減少、最大窓19.555 ms。現在7条件の超過は解消したが、全8 prototype・全I10処理・liveの受入ではない。全8 profileを1／7音群へ強制対応させる検査専用の16条件も取得した。336構築・602,112試行位置、最終28,672位置の直接一致を確認した。一方、gesture・phrase・表だけで最大52.882 ms、336完全窓中17窓が50 ms超過。closure headは未支持で、通常の距離門を通した照合でもない。次は7音群でのgesture／phraseと表の費用を絞り、全head・全worker・通常共同推論・候補消費を引き続き接続・検証する。

区間端を再利用する位相積分を通常／候補予測へ接続した。直接波形和84追加条件と71分岐の順位、通常172音声の不変性を確認し、計算費用を削減した。複合条件を含む24条件の核単独費用も記録した。共有budgetと全Voice負荷の受入は未完であり、I10全体の完了条件は維持する。


実身体の受理済み単一onsetから、7 classの条件付きenergyを別workerで計算し、通常report／UIへ接続した。6素材・9記録・両bus2,792窓、音声・既存予測学習・共通566観測の不変性を確認した。対象外2件は未対応として保持する。offline71分岐の独立照合も通過した。共有候補表の通常consumerは後述の追加接続を参照。実身体適合、非energy帰結、全資源受入は未完。I10完了範囲の回答待ちも維持する。[接続記録](i10-body-outcome.md)、[登録](i10-live-candidate-energy.json)を参照。

実身体の即時／遅延onset候補へ、同じ発行時の私有trace分布を凍結して接続した。offline共有観測の公開完了同期も修正した。同じ同期処理を含む対照との6素材比較で、既存音声・energy・私有学習は不変。100組の差分を独立照合し、支持あり22組・非ゼロ20組を確認した。全Rust 1,184成功。共有候補表の通常consumerは後述の追加接続を参照。別head、実身体適格性、全資源受入は未完。[接続・同期記録](i10-body-outcome.md)、[登録](i10-body-trace-candidates.json)を参照。

## 1. 変更の範囲

[完成計画](../temporal-dcc-completion.md)の実行単位を、技術実装I、研究判定R、最終受入Aに分ける。
旧M0–M9は、元の要求と保存済み証拠を参照する番号として残す。旧M0を初期実装の進捗箱として使わない。
I0の完了は旧M0全体の完了ではない。2026-09-13の変更では、本体の必須範囲をT1–T6と
I12a–I12dの四接続とし、I9・I12e・T7・O16を研究拡張へ移した。MR1–MR3、実時間・
作者採用・air-gapと、残る対象の数値・認知・可聴性の基準は維持する。研究拡張の未達を合格へ変更しない。

前の「17%」は撤回する。異なる規模の終了条件の件数であり、工数や残り時間の割合ではない。
I/R/Aの完了件数も一つの進捗率へ合算しない。

### 1.1 採用スコープと認知機構の選定（2026-09-13）

（2026-09-13時点の記録。本体範囲の現行規則は§1.3。）

原理と採否の正本は[technote-ledger §9.3.55](../../design-notes/technote-ledger.ja.md#i9-scope-decision)。
Scenarioは作者の構成・環境条件、生成側は身体と認知環境に応じた局所判断、ListenerTwinは
実際のpresentationに対する選択した聴取応答を担う。全曲構成の自律推定や人間の包括的な複製は本体の前提にしない。
Scenarioの帰還指示を、聞こえた帰還の証拠としてTwinへ流用しない。

次の機構追加では、対象応答、対応する認知神経機構と一次根拠、抽象化する状態・更新・相互作用、
生成側／ListenerTwinの利用先、音への具体的な作用、競合説明を区別する比較を先に一組で定める。
部位名の対応表、評定への適合、既存の数値実装だけでは機構対応の採用としない。
全時間階層を埋めるために新しい機構を追加せず、選んだ作用に必要な記憶だけを導入する。
既存I0–I8の技術完了は維持し、その認知的採用はR4/R5等の未達判定と分ける。

終端をあらかじめ決めない継続運転（indefinitely running）も設計対象とする。状態更新は観測済み入力に因果的に従い、
予定曲長・EOF・全曲完了感を要求しない。有界な記憶でも長い過去の影響は保持でき、全履歴の保存を要求しない。
継続運転の実装・実時間受入は未検証。A3では宣言した運転条件の資源・鮮度・復旧を検査する。
30分は研究素材または耐久検査の一条件であり、全用途への全曲記憶要件ではない。

I9・I12eは研究拡張として保留し、自動継続の対象にしない。再開時は対象となる聴取応答と機構、
有限作品か継続運転か、必要な保持範囲、元の結果との比較条件を先に定める。
O16はその長期能力の検証・採用を止めるが、本体のI10・I11・I12a–I12d・A4を止めない。
I4/I8で使う記憶の正しさ・資源検査は引き続き必須。R2/A3は実際に走る処理をすべて数え、
研究拡張への移管を理由に既存の計算費用や不具合をゼロへ置き換えない。

保存したJSON数値参照・fit表は元の研究条件の記録であり、新スコープの実験登録を代用しない。
実験を組み替える場合は対象・モデル・依存・負荷・fit条件を再登録する。今回の変更は文書上の採用範囲と工程に限る。

### 1.2 次工程の機構選定：動作に条件づけた自声予測（2026-09-13）

「commitして、次の工程に進んで」に基づき、対象機構と音への作用を具体化した。
最初の実装候補は、**実行する動作と身体状態から、その後に生じる自声の特徴と時刻を予測し、実音で確かめる機構**とする。
これは開発上の選定であり、著者による音の採用や神経機構の実証ではない。
一次根拠、抽象化、現行コードとの差、比較課題の正本は[technote-ledger §9.3.55](../../design-notes/technote-ledger.ja.md#i10-sensorimotor-contract)。

| 今回の一単位 | 終了点と次の実装境界 |
|---|---|
| 対象 | I10の身体・帰結入力を使う一機構の選定。I9は保留を維持する |
| 具体化した動作 | 動作指令と実際の発音を別記録にし、発行済み予測を観測後に照合する。将来のI11では、自声の予測音長と外部音の文脈からonset-now／delayed onsetを比較する |
| 今回の終了点 | 一次研究とその限界、因果的な状態更新、所有者、実消費者、反証可能な比較、既存処理との差を日英台帳へ記録した。モデル実装・実験実施は含まない |
| 次の技術単位 | 固定した身体・方策の一Voiceについて、実行指令→routing後の私有実音→時刻・支持つき帰結の診断を通常経路へ一往復させる。最初はonsetを対象とし、I10全体の完了とは区別する |
| 必要な検査 | 指令は発音の証拠ではないこと、失敗／既知無音／欠測の区別、候補の非学習、予測発行前の入力限定、bus・世代・順序、off音声一致。既存の数値参照は出所を維持して再利用する |
| 残る判断 | 自声予測の数値モデル・尺度・時間窓を実装前に登録し、固定モデルと動作非依存モデルを比較する。I11の生成作用には該当R4/R5、音の採用にはA1/A2が必要 |

想定する記憶は有界な動作―音対応の状態と、未照合の短い観測窓であり、曲長に比例する出来事一覧ではない。
既存の `PrivateTimingTrace` は文脈に対する参加時刻の記憶であり、この自声予測器と同じ機構として扱わない。
記憶を二重に追加せず、共有できる観測と別に必要な状態を次の技術単位で確定する。
この機構選定時点では文書・コードの照合と一次資料の確認までを実施した。続く初期onset接続は[I10記録](i10-body-outcome.md)を参照する。
Rust実装の残工数は未計測であり、見積りを合格条件にしない。

### 1.3 機構選択規則による本体範囲の縮小（2026-09-20、同日のレビューを反映）

原理と判定の正本は[technote-ledger §9.3.55](../../design-notes/technote-ledger.ja.md#mechanism-selection-rule)。
和声側と同じ分担を時間構造に適用する。一次根拠と同定済み認知神経機構があり、既存モデルまたは閾値・尺度を明示した工学則で書け、
人の新規収集なしに作者自身の試聴で採否を決められる機能だけをDCC側に置く。禁止するのは本体の生成へ接続する表現の新設・採用であり、
保存済みの研究実装と既定offの受動診断は保持できる。それ以外の生成作用はALifeの局所判断とScenarioに委ね、
ListenerTwinは受動的な観測指標に留める。I9除外はこの規則の最初の適用だったと位置づける。

本体と研究拡張の担当を次のように改める。§2〜§4の表は元の定義を保存し、適用範囲はこの節に従う。

| 区分 | 本体（2026-09-20以降） | 研究拡張（保留） |
|---|---|---|
| T行 | T1 articulation、T2 拍・小節、T4のうち短期の反復期待 | T3のgroove／参加意欲評定head、T4の変形をまたぐ対応・想起、T5 phrase・終止、T6 section・帰還、T7 全曲 |
| 技術実装I | I0–I3、I4のうち `temporal_cognition/reference_inventory.rs` の有界因果参照、I5、I6、I10、I11（本体T行の一機構ずつ）、I12b（下記二経路）、I13 | I4の `matcher.rs`（coarse／DTW）・`memory.rs` の順序付き／orderless検索・`query.rs` のscheduler・`recall.rs`／`recall/graph.rs` の対応graph、I7、I8、I9、I12a、I12c、I12d、I12e |
| 研究判定R | R2、R3（実際に動く全処理の数値・回収・安定性検査） | R1、R4、R5 |
| 最終受入A | A1（作者自身の試聴だけで完了可能。他者試聴は任意の参考で判定に必須としない）、A2、A3、A4（この縮小範囲の監査） | 研究拡張採用時の該当T行の四判定 |
| O義務 | O01–O04、O09–O15、O17–O22（本体T行と接続に限る。本文は§4と完成計画§4.1） | O05–O08、O16（長期記憶容量の採用条件、2026-09-13に移管済み） |

- T3のうち固有周期・参加周期・鳴る／待つ／見送る（`src/life/temporal_participation.rs`）はVoiceの身体と作者の参加傾向であり、ALife側でそのまま使う。groove／参加意欲の評定head（`groove.rs`・`ratings.rs`・`[temporal_groove]`）は2026-09-20にmainから削除し、研究拡張ブランチへ保存した。
- I4の分割: 本体に残すのは `src/core/temporal_expectation.rs`・`history_prediction.rs` の短期反復・帯域エネルギー予測（I11の外部短期予測の供給元）、`recall.rs` の受領証拠の保持・減衰状態、その保持状態と音響支持から作る `reference_inventory.rs` の2秒支持・anchorによる有界因果参照（I10私有traceの入力。`Stream` は `recall::Snapshot` を入力にとる）。研究拡張へ移すのは `matcher.rs` のcoarse／DTW、`memory.rs` の順序付き／orderless episode検索、`query.rs` のscheduler、`recall.rs` のprefix query発行、`recall/graph.rs` の対応graph。現在の `recall.rs` は保持状態の生成とprefix query発行を同じ型で担うため、この分離はコード上は未実施であり、`[temporal_memory]` 明示時はrecall全体が既定off診断として動く。分離はI4の残作業として§5に登録し、I10の完了条件には含めない。
- I12bの本体範囲は、I6の到来期待がgesture候補時刻へ作用する経路と、観測accentが周期推定へ入る既存経路の二つ。除去比較はこの二経路それぞれ。前者の更新則・閾値・尺度はI11の実装前登録で確定し、登録前は本体採用を保留する。独立したgrouping推論は本体に追加しない。
- I7・I8のRust実装（`phrase.rs`・`section.rs`・`whole.rs`・`hazard.rs`と各fixture）は2026-09-20にmainから削除した。実装・fixture・検査記録はbranch `research/temporal-dcc-extension`（tag `research/temporal-dcc-extension-20260920`）に保存し、再開時はそこから復元する。既定off診断としてmainに残す方針は、本体が読まない約2万行をtest buildへ抱えることになるため撤回した。
- 本体のA4は、I9／I12eに加えてI7・I8・I12a／c／d・T3評定head・T5・T6を監査対象から外す。四接続の統合判定はI12b二経路の除去比較へ置き換える。
- I11の対象は本体T行に限る。本体の閾値・尺度は各Iの契約検査（O03／O09）で登録し、人の評定へ適合させない。
- 既存の数値参照（39／82成分descriptor、109係数head、episode graph、fit表）は研究拡張ブランチと`docs/roadmap/temporal-dcc/*.json`に保存し、本体の依存へ戻さない。
- 継続運転、air-gap、既定off、Scenario指示≠聴いた帰還の証拠、因果的状態更新、R2／A3の全処理計数は変更しない。
- I10は§2行の狭い範囲で閉じる。2026-09-18〜20の共同posterior・全帰結投影・phrase再重み付けの作業は研究拡張の保存記録とし、`src/temporal_cognition/joint/`・`consequence/`・`long_form.rs` はmainから削除した。残作業の仕分けとI11入力契約は[I10記録](i10-body-outcome.md#i10の範囲確定と残作業の再定義2026-09-20)。
- 候補表は観測文脈（`src/temporal_cognition/context.rs`）から作る。列はfeature窓、articulation投影、短窓の四座標、grouping支持、arrival、accent密度、文脈三座標に限る。closure／continuation／groove由来の列は削除した。
- 記憶の保持・検索は固定スパンのepisodeで動く。phrase由来のcue封緘、対応graph、prefix query、coarse commitment cacheは削除した。`matcher.rs`・`memory.rs`・`query.rs`の探索はI4の研究拡張として残り、分離は§5の残作業。
- 本体の範囲判定は文書上の変更だが、上記のコード削除と候補表の作り替えは2026-09-20に実施済み。通常演奏の既定動作（全`temporal_*`がNone）と生成音は変更していない。

## 2. 技術実装の完了点

（以下の表は元の定義を保存する。本体／研究拡張の適用範囲は§1.3に従う。）

Iの完了は、指定した入力から実際の消費者・診断まで通り、変更に必要な契約検査と必須テストが通ること。
部品が`cfg(test)`で動くだけでは、通常runtime接続を求めるI項目を完了にしない。
未校正の研究用出力は識別し、通常の生成作用は有効化しない。各行は後続全モデルの完成を要求しない。

| ID | 動作として示す成果物・終了条件 | 必要な入力・依存 | 旧要求の対応 |
|---|---|---|---|
| I0 基準と交換契約 | source・設定・比較入力を識別し、最初の数値仕様と小型MR1を再現できる。残る義務を担当へ割り当てる | 既存基準・仕様 | M0の実装準備、W0–W2 |
| I1 順序付き実観測 | 実音→NSGT→二busの順序付き観測→診断表示。欠落・epoch・EOF・report独立性・off音声一致を検査する | I0 | M1の初期経路、W3 |
| I2 軌跡の継続 | 実音のピーク→ridge handle・継続・退役→診断。明示尺度、支持時刻、欠落後の再開を検査する | I1 | M2のridge |
| I3 音群と生特徴 | 同じ実音→音群配分・成立・退役→groupごとのraw descriptor／accent→診断。更新前handle、energy保存、欠測を検査する | I2 | M2の音群・特徴 |
| I4 記憶照合の往復 | 実観測の区間を保存し、後続のqueryで照合結果を診断へ返す。初出／再出、順序変更、元支持、遅延・退役拒否を一つの経路で示す | I3、既存matcher・prefix・scheduler | M3の最初の検索経路 |
| I5 gestureと受け渡し | attack／continuation／release／gapとgesture unionを実観測から更新し、音群をまたぐ継続と未知を診断へ渡す | I3、必要なI4出力 | M2のarticulation、T1 |
| I6 周期と到来 | 実accent→周期・まとまり候補→到来予測→時刻付き診断。省略・非拍節・複数解釈とMR2の交換を示す | I3 | M2の拍・小節、T2 |
| I7 フレーズと終止 | 同じ実音の先行文脈→継続／境界候補・別の終止推定→診断。境界と終止を区別する | I4・I5・I6の利用可能な観測 | M4、T5 |
| I8 sectionと帰還 | 実区間記述→継続・対比・変形した帰還→想起焦点・診断。登録39／82成分と記憶の実消費を通す | I4・I7 | M3残部、M5のsection、T4・T6 |
| I9 長期の聴取応答（研究拡張・保留） | 対象機構・応答・必要な記憶範囲を再設定してから検証する。旧whole head・graph・30分容量比較の成果と失敗は[I9記録](i9-whole-context.md)に保持。本体の必須工程から除外 | 機構選定、対象に必要なI8出力 | 旧M5の長尺、O16、T7 |
| I10 身体と帰結入力 | 実Voiceの身体・実行結果→私有trace・候補帰結の数値経路→診断。固定方策、未実行候補の除外、両busの対応を検査する | I3・I4、既存の身体 | M3・M6のtrace、O13・O14、R2への負荷入力 |
| I11 一機構の局所生成 | 支持された関係一つ→実消費者→音響差を示す。入力一致、状態介入、期限、off基準、APIの作用を検査する | 対象I、対応するR4・R5の合格 | M6。対象を明記して一機構ずつ閉じる |
| I12a articulationとphrase | 双方向の更新を通し、この接続だけの除去比較を実行できる | I5・I7・該当I11 | M7の第1接続 |
| I12b beat/groupingとgesture | 同上。異なる参加位置とgestureの関係を保持する | I5・I6・該当I11 | M7の第2接続 |
| I12c motifとphrase | 同上。対応・想起とphraseの相互更新を通す | I4・I7・該当I11 | M7の第3接続 |
| I12d phraseとsection | 同上。局所境界と長い文脈の相互更新を通す | I7・I8・該当I11 | M7の第4接続 |
| I12e sectionと長期聴取文脈（研究拡張・保留） | 再設定したI9を生成へ返す用途を選ぶ場合だけ、対応する相互更新と除去比較を検証する。本体の必須接続から除外 | 採用したI9・I8・該当I11 | 旧M7の第5接続 |
| I13 作者の操作経路 | scenario指定→演奏→状態・無効理由の確認→編集・再実行を、成立した機能で通す。Rhai・UI・文書・実行sampleを同期する | I1、公開するI11・I12の実消費者 | M8のAPI／UI、O21 |

I11は対象T行・消費者ごとに完了を記録する。一つの成功で全局所生成を完了にしない。
I12は親集計名であり独立の実装作業ではない。本体はI12a–I12dをそれぞれ閉じ、T3のgroove／参加意欲と
役割の相互作用を四接続の統合版のR4・R5・A1で判定する。I12eを採用する研究拡張では影響範囲を再判定する。
MR3はI1以降の最初の利用可能な境界で実施し、fit・保存形式の利用先が増えた行で追加する。

## 3. 研究判定と最終受入

| ID | 判定する内容 | 通過まで止める行為 |
|---|---|---|
| R1 研究登録・実施準備 | 全対象の特徴・課題・対照・素材・提示条件・分割・機関手続き・同意・回答管理・pilot手順を、元の規定時点までに固定する | 対応する人の収集・pilot、未登録の検証。pilot結果に依存する募集数等は検証前に固定する |
| R2 全処理の数値preflight | O04の両bus・64 Voice・非ゼロ／最大候補負荷、全worker・コピー・容量・headroomを、同じRust処理で検査する | stage-1収集。未実装処理のゼロ算入、単体timingによる全体合格 |
| R3 回収・安定性・感度 | O10–O13の登録条件で、実音尺度を使うepisode／trace回収、event-head予測安定性、感度比較を行う | 対応する収集・freeze・パラメータ採用。合成回収の成功を人的実現可能性へ転用しない |
| R4 適合・校正 | 有限fit表・予算、stage 1–4、OOF、利用先別校正、交換後の再適合を完了する | 対応するモデル・行動列の採用。部分版の合格を統合版へ転用しない |
| R5 認知課題 | 本体T1–T6のGate 2を独立課題・対照で検証する。T3のgroove／参加意欲を含む。T7は研究拡張の採用時だけ別判定 | 対応する機構の生成接続・認知的採用 |
| A1 関係の可聴性 | 本体T1–T6、研究拡張では採用したT7のGate 3。三記述・foil・不明回答・共同成功条件、因果介入と接続除去の比較 | 可聴な関係効果の採用・主張 |
| A2 作者採用 | 本体T1–T6、研究拡張では採用したT7のGate 4。長い試聴、宣言した音楽条件、既存sample、旧template置換・既定変更の個別判断 | 作者採用、O19置換、別途承認を要する既定変更 |
| A3 実時間・通常運用 | 第6節の実機・4／16／64 Voice・長時間・鮮度・復旧・音声・API／表示一致を検査する | 実時間運用と通常運用の最終採用。R2はこれを代用しない |
| A4 本体完了 | I9/I12eを除く本体I、R1–R5、A1–A3、O16を除く適用O、T1–T6×四判定、四接続・回帰・文書を監査する。研究拡張と旧全階層仕様の完了は別集計 | 宣言した本体範囲の完了。T7・全曲記憶の能力は含めない |

I0–I10の技術着手に、R1–R5の一括完了を要求しない。各Iの入力契約・数値fixtureはその接続前に検査する。
R2は必要な全処理とI10等の負荷入力が揃ってから実行する。規定された保守的な費用算入以外で未実装分を埋めない。
R1の手続き条件は全ての人の収集に適用する。R2・R3等の追加条件は元の規定対象に適用し、実施順序を循環させない。
R/Aは対象T行・component・版ごとに判定できる。一つの部分合格で全対象を合格にせず、必要な統合再判定を残す。
適用スコープは本書§1.1と[完成計画](../temporal-dcc-completion.md)に従う。旧M定義の全階層要件を本体へ戻さない。

## 4. O義務の担当と停止条件

下表は旧担当から新担当への対応。義務本文・成果物・数値基準は完成計画§4.1と設計メモ§10.4に対応し、適用範囲は§1.1に従う。
担当の移管は義務の完了ではない。「各I」はその変更が影響する行だけを意味する。

| ID | 新担当 | 維持する停止条件 |
|---|---|---|
| O01 | I0; 各Iの変更時 | 出所が不明な比較・旧成果の転用 |
| O02 | I0; A4 | 実装の裏付けがない現在形の能力主張 |
| O03 | I0–I13; R1; R4 | 必要な座標・尺度・mask・対照等が未登録の模擬検査、収集、fit |
| O04 | R2 | stage-1収集。単体性能からの全体合格 |
| O05 | R1 | 素材・分割・機関手続き・同意等が未確定の人の収集（pilotを含む） |
| O06 | R1; R3; R5 | 対応する検査が不足した収集・freeze。制作上の試聴を独立した認知検証へ転用しない |
| O07 | R1; R4; R5 | 未登録の検証、対照・検出力が不足したcomponent freeze |
| O08 | R1; A1 | Gate 3の聴取検証 |
| O09 | I0–I13; R4 | 該当する数値特徴の適合・解釈、未検査経路の採用 |
| O10 | R3; R4 | 影響するT行のfreeze |
| O11 | I4; R3; R4 | stage-1収集、episodeパラメータ採用・長期文脈の主張 |
| O12 | R3; R4; I5–I12 | 境界／type収集、未再適合モデルの採用、統合版・行動列のfreeze |
| O13 | I10; R3; R4 | stage-1収集／適合、私有trace採用・旧template置換 |
| O14 | I10; I11; A3 | ordinal生成列の有効化 |
| O15 | I11; I12a–I12d; A1（研究拡張ではI12e） | 因果的な関係効果の主張 |
| O16 | 研究拡張I9; R3 | 対象とする長期聴取能力の検証・採用。本体工程とA4は止めない |
| O17 | A3 | 実時間採用 |
| O18 | I6; I11; A3 | liveの期待・行動判定 |
| O19 | R5; A1; A2; A4 | 旧templateの置換。off／受動・既定基準版の変更は認めない |
| O20 | A4 | 本体T1–T6・四接続の完了。研究拡張は別に監査 |
| O21 | I1; I11; I13; A2; A3; A4 | 通常運用・API／UI／文書完成・既定変更。操作・表示・内部状態が一致しない場合も止める |
| O22 | 再編計画; A4 | 設計登録・実装・実証を混同した完了報告 |

## 5. 現在地と再開時の一単位

範囲再改訂（2026-09-20）: §1.3の機構選択規則を採用し、本体をT1・T2・T4の短期反復期待、I0–I3、I4の有界因果参照、I5、I6、I10、I11、
I12b、I13へ縮小した。T3評定head、I7、I8、I12a／c／dは研究拡張へ移管し、R1・R4・R5は本体の完了条件から外した。
同日、研究拡張のRust実装をmainから削除した。checkpointは `d579688`、保存先はbranch `research/temporal-dcc-extension`
（tag `research/temporal-dcc-extension-20260920`）。削除したのはjoint／consequence／long_form、phrase／section／whole／
groove／ratings／hazard、対応fixtureと生成スクリプト。候補表は新しい観測文脈モジュールから作り直し、記憶の照合は
明示した距離則へ置き換えた。`src/temporal_cognition/` は35,342行（削除前は約51,000行）。
以下の2026-09-19までの判断待ち（工程表I10と後続拡張範囲の不一致）は、この改訂で範囲側を確定したものとして扱う。
通常演奏の既定動作（全`temporal_*`がNone）と生成音は変更していない。
I10の残作業は三つ。実身体転用の妥当性（非sine backendを含む順位反転の解消）、release・任意身体・両busへの候補評価の展開
（§1.2の契約「onsetのみの成立はI10全体の完了ではない」に由来し、新たな不足ではない）、縮小した負荷の資源引渡し。
2026-09-19監査の残依存6件のうち、共同posterior・全帰結投影・phrase再重み付け・参加意欲headの4件は研究拡張へ、
実身体転用と資源引渡しの2件が本体に残る。2026-09-18〜20の共同posterior・全帰結投影の作業は研究拡張の保存記録とし、
`joint/`・`consequence/`・`long_form.rs` はmainから削除した。仕分けの詳細、新しい完了監査表、I11入力契約は
[I10記録](i10-body-outcome.md#i10の範囲確定と残作業の再定義2026-09-20)、監査は `target/i10-scope-confirmation-20260920/audit.json`。
次の一単位は実身体転用の妥当性。
I4の残作業として、`recall.rs` から `matcher.rs`／`memory.rs`／`query.rs` のepisode探索を分離することを登録する
（I10の完了条件外、A4前に実施）。phrase由来のcue封緘・対応graph・prefix query・coarse commitment cacheは2026-09-20に削除済みで、
記憶は固定スパンのepisodeで動く。`reference_inventory` はその検索結果を入力にとるため、探索の分離には代替の照合規則が要る。
I11の比較量・有界窓・情報締切・欠測時の選択規則と、I12b到来経路の更新則・尺度は、I11の実装前登録で確定する。登録前はI11・I12bの機能成立を主張しない。

解決済み（2026-09-20に発見・同日解決）: 研究拡張の削除直後、私有参加traceが信用を割り当てなくなった。
原因は設定の意味の変化だった。phrase cueがある間は、cueのcommitがepisodeを封緘し、`span_hops` はquery span
だけを決めていた。cueを外すと `span_hops` が封緘するepisodeの長さそのものになる。検査設定の8 hop（85 ms）では
毎秒約11本のepisodeを封緘し、16枠のbankを1.4秒で総入れ替えしてしまう。削除前後の実測は、検索行314→66、
参照付き在庫34→5、参照378→44、信用付きtrace 16→0。
検査設定を1秒のepisode（`span_hops = 96`）と32枠へ改めたところ、封緘570→24、退避554→8となり、
信用・候補/default時刻差・学習済み予測の再利用がすべて回復した。`src/config.rs` の `span_hops`・`episodes` に
この意味を明記した。コード側の変更は不要で、`paired_lookup` の窓規則も変えていない。

残依存のsource監査（2026-09-19）: 共同正規化・全帰結の核・参加意欲の109座標標準化組立は監査時点でtest build限定。
通常版はraw closure／continuation二列であり、全帰結の残作業は単なる呼出し追加ではない。
工程表I10と後続の拡張範囲の不一致について判断を求めた。未実装列の削除・完了扱いは行わない。
判断待ちと独立の実身体/default要件は継続する。確定未来control更新より前のenergy支持の回復は、以下の版で完了した。
監査結果は`target/i10-completion-dependencies-20260919/audit.json`、詳細は[I10記録](i10-body-outcome.md)末尾。
この監査時点ではsourceを変更していない。後続のcontrol支持修正の検査は以下に記録する。

最新のI10検査（2026-09-20）: 全5成分の局所候補生成とID割当を一つのproducerへまとめ、
元観測・cache・modelを不変借用したbatchからownerへ渡した。欠測中のhazard退出はunknownへ移す。
8更新・21,943 tupleの確率照合、12実matcher支持窓の24条件、全group退役、ID／score失敗を確認した。
全5成分既知540 tuple・360保持経路、grouping欠測下の帰還126 tupleを検査。候補競合による帰還切捨ても記録した。
全Rust 1,302成功、fmt／通常Clippy／全target check／release通過。変更はcfg(test) joint内9件。
記録は`target/i10-joint-local-producer-20260920/summary.json`。通常source不変のため新renderなし。
残るstage1 phrase入力、joint cue確定投影、全scoreと通常接続は未完。
全帰結、身体転用、default、統合・資源引渡しも未完。I9保留、I11無効、at-action維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: 元query全件のretrieval特徴を局所sectionへ供給し、
元音響履歴からpruning後のphrase endingを投影した。元支持・退役・欠測・差替えをowner入口で照合する。
80候補／16保持の独立score検査、30更新の音響moment検査、accent容量損失、622 owner経路を確認した。
全Rust 1,300成功、fmt／通常Clippy／全target check／release通過。
3実renderはWAV・393候補record・434時間観測が一致。記録は`target/i10-joint-evidence-projection-20260920/summary.json`。
全成分producer、条件付きphrase残余、joint cue確定投影、全scoreと通常接続は未完。
全帰結、身体転用、default、統合・資源引渡しも未完。I9保留、I11無効、at-action維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: 実観測・前回section・確定Returnから局所section候補を生成し、
sectionとcorrespondenceの整合をtuple生成段階へ接続した。16候補／86優先度評価の上限を維持。
24条件の独立全直積参照と照合し、実matcher12支持窓から36復帰／Development経路をownerへ採用した。
全Rust 1,297成功、fmt／通常Clippy／全target check／release通過。変更はcfg(test) joint内だけ。
記録は`target/i10-joint-local-sections-20260920/summary.json`。
全成分producer・ID/source結合、phrase／retrieval／cue投影、全条件付きscoreと通常接続は未完。
全帰結、実身体転用、default、統合・資源引渡しも未完。I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: 実音響支持・確定context・既存section Headから共有proposalを生成し、ownerへ根拠を結合した。
静音／欠測・同率・重複role・単調key、5更新22proposal、実matcherの12共有復帰・12退役拒否を検査。
候補生成scoreは明示した工学則であり、観測log-potentialや校正済み確率とは区別する。
全Rust 1,296成功、fmt／通常Clippy／全target check／release通過。変更はcfg(test) joint内だけで、通常renderは反復しない。
記録は`target/i10-joint-shared-producer-20260920/summary.json`。
次は局所tupleの全producer、joint cue確定投影、全条件付きscoreと通常接続。
全帰結、実身体転用、default、統合・資源引渡しも未完。I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: 共有経路ごとに型付きの前回summaryと元の観測時刻を保持した。
同じkeyの別経路を統合せず、q(local|shared)、既知ゼロ、unknown／枝刈りを分ける。group順序変更・欠測・世代交代も検査。
全Rust 1,294成功、fmt／通常Clippy／全target check／release通過。変更はcfg(test) joint内だけで、通常renderは反復しない。
記録は`target/i10-joint-shared-state-20260920/summary.json`。
keyの実生成と最強supported共有proposal、全条件付きscore、joint cue確定投影・通常接続はまだ未完。
全帰結、実身体転用、default、統合・資源引渡しも維持する。I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: sectionの音響入力・phrase支持蓄積を通常版と共同ownerで共用した。
section unknown中もphrase支持を保持し、選択groupingとgroupごとの欠測を反映する。12更新622経路の支持を独立時間計算と照合。
22拒否（うち10回は正規化後の更新失敗）と公開状態保全を検査。通常3素材393候補・434観測・音声は前版と一致。
全Rust 1,293成功、fmt／通常Clippy／全target check／release通過。joint／consequenceはcfg(test)。
記録は`target/i10-joint-section-inputs-20260920/summary.json`、残件は同directoryの`remaining-audit.json`。
残りは共有状態／全score、通常共同推論、全帰結、実身体転用、default、統合・資源引渡しの6実装群。
既に通常接続したgroove／desire headは再実装しない。次は共有状態とproducerの組立を進める。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: section復帰を選択済みfresh correspondenceと世代付きepisodeの確定context支持へ結合した。
実cue／matcherの100更新を3条件で検査。共同ownerのunknown復帰24件・development12件・60拒否後の状態保全を確認した。
同率context支持はIDで決定し、支持ゼロでは復帰しない。通常3素材393候補・434観測・音声は前版と一致した。
全Rust 1,292成功、fmt／通常Clippy／全target check／release通過。joint／consequenceはcfg(test)。
記録は`target/i10-joint-section-returns-20260920/summary.json`。
次はsectionの全条件付き入力組立、共有状態・全score・通常接続。実身体転用・default・統合検査も未完。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: section履歴を共同ownerへ保持し、条件付きproposalへ再利用した。
8更新・248保持・27 return・5 development、正規化後の8回の更新失敗と再試行を検査した。
section arena両側は30,347,520 bytesを予約。これは全資源受入ではない。
全Rust 1,291成功、fmt／通常Clippy／全target check／release通過。共同ownerはcfg(test)。
記録は`target/i10-joint-section-owner-20260920/summary.json`。
次はsection入力と過去context復帰、共有状態・全条件付きscore・通常接続。I10の実身体転用・default・統合検査も未完。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: sectionの解釈状態を重みから分離し、通常版と共同proposalで共有した。
5遷移・欠測・遅延accentとbuffer再利用を検査。共同遷移質量12条件の最大誤差2.3e-16。
通常3素材の393候補・434観測・音声は旧版と一致。全Rust 1,290成功、fmt／通常Clippy／全target check／release通過。
記録は`target/i10-joint-section-state-20260920/summary.json`。通常接続は共通section更新までで、jointはまだcfg(test)。
次はsectionの共同保持、共有状態と全条件付きscore、通常接続。I10の共同推論・実身体転用・default・統合検査は未完。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: groupingの元payloadを共同ownerへ保持した。
90更新・8,930保持・420継承・89回のgroup順序変更を検査。86回の拒否後再試行も通過した。
単一accentから群化なしをadmitする不備を修正し、period pair／peakの支持を必要とした。
全Rust 1,287成功、fmt／通常Clippy／全target check／release通過。変更はcfg(test) joint内。
記録は`target/i10-joint-grouping-owner-20260920/summary.json`。
次はsection／共有状態と全条件付きscore、通常接続。I10の共同推論・実身体転用・default・統合検査は未完。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: 実grouping inventoryから共同候補を作る入口を追加した。
元accent key・支持区間・残差を保持し、16 fresh＋stay＋群化なし＋unknownを区別する。
90更新・1,231候補payloadを検査。遷移質量の最大誤差5.6e-17。探索不足からの新規群化なしを抑止した。
全Rust 1,285成功、fmt／通常Clippy／全target check／release通過。通常sourceの実行処理は不変、render反復なし。
記録は`target/i10-joint-grouping-20260920/summary.json`。jointはまだcfg(test)。
次はgroupingの共同保持、section／共有状態と全条件付きscore、通常接続。I10の共同推論・実身体転用・default・統合検査は未完。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: 実matcherの対応payloadを共同ownerへ保持する経路を追加した。
96更新で元query・変換・残差の継承、欠測時のmask、退役対象の除外、95回の拒否後再試行を検査した。
通常3素材の393候補・434観測・音声は旧版と一致。全Rust 1,282成功、fmt／通常Clippy／全target check／release通過。
記録は`target/i10-joint-correspondence-owner-20260920/summary.json`。jointはまだcfg(test)。
次は残るgrouping／section／共有状態と全条件付きscore、通常接続。I10の共同推論・実身体転用・default・統合検査は未完。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: 実matcherからcorrespondenceの16 fresh＋stay＋known no-memory＋unknownを組み立てる入口を追加した。
元queryの探索範囲を通常reportとsectionへ伝達し、容量・変換範囲の不足から新規no-memoryを出さない。
208候補・55 query、空の記憶／容量不足／範囲外を検査。通常3素材の1,982 queryでflag公開を確認した。
追加fieldを除く393候補・434観測・音声は旧版と一致。全Rust 1,280成功、fmt／通常Clippy／全target check／release通過。
記録は`target/i10-joint-correspondence-20260920/summary.json`。correspondenceのListとjointはまだcfg(test)。
次は対応payloadの共同保持、残るcomponent・共有選択・全条件付きscoreの接続。I10の共同推論・実身体転用・default・統合検査は未完。
I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-20）: phraseのforeground・終了descriptor・継続link更新を通常版とjointで共通化した。
共同ownerが親slotを検査し、枝刈り後だけ実履歴を保持する。6更新・162履歴と拒否後の再試行を検査した。
通常3素材の393候補・434観測・音声は旧版と一致。全Rust 1,274成功、fmt／通常Clippy／全target check／release通過。
記録は`target/i10-joint-phrase-state-20260920/summary.json`。joint全体はまだcfg(test)。
実観測からの全特徴・admission・残るcomponent・共有選択を次に接続する。二つのphrase arenaは計2,150,400 bytes、全資源受入は未完。
I10未完了。共同推論・実身体転用・残るdefault・統合検査を維持。I9保留、I11無効、R2／A3は別工程。

前版のI10検査（2026-09-20）: 実観測のarticulation入力・状態遷移を通常Gestureと共同proposalで共通化した。
共同ownerは元観測のattack／release／gap支持と開始時刻・censoringを保持し、次入力へ渡す。
独立60条件と6更新を検査。通常3素材の393候補・434観測・音声は旧版と一致した。
全Rust 1,271成功、fmt／通常Clippy／全target check／release通過。joint全体はまだcfg(test)。
記録は`target/i10-joint-articulation-20260920/summary.json`。次は残るcomponent状態・共有選択・条件付きscoreの接続。
I10未完了。共同推論・実身体転用・残るdefault・統合検査を維持。I9保留、I11無効、R2／A3は別工程。

前版のI10検査（2026-09-20）: 共同経路の前回重みと親identityを内部保持する層を追加した。
独立した10更新との最大重み差1.67e-16。bus／epoch・音群世代・親経路・観測順・拒否後の再試行と、
32共有／57,856局所候補の有界処理を検査した。全Rust 1,268成功、fmt／通常Clippy／全target check／release通過。
追加の全target Clippyは既存6ファイルの18指摘で失敗し、今回の変更ファイルへの指摘は0として別記録した。
jointはまだcfg(test)、release binaryは前単位と同一。通常接続やcomponent履歴の実所有を完成したとはしない。
記録は`target/i10-joint-owner-20260920/summary.json`。次は実component producerとproposal組立・条件付きscoreの接続。
I10未完了。共同推論・実身体転用・残るdefault・統合検査を維持。I9保留、I11無効、R2／A3は別工程。

前版のI10検査（2026-09-20）: 既知の予約振幅更新を実身体／default予測へ接続した。
各Tone先頭4件を凍結し、同時刻順・平滑化・release時の予約消去を反映する。音高・駆動と超過分は支持境界を保持。
実Toneの120条件照合を通過。通常の振幅変更素材で16 record・224窓を追加支持し、既存支持の喪失・値変更は0。
音声・実観測・非energy学習・共有profile対応は旧版と一致。全Rust 1,264成功、fmt／Clippy／全target check／release通過。
記録は`target/i10-scheduled-amplitude-20260920/summary.json`。通常共同推論・実身体転用・残るdefault・統合の4大工程は維持。
I10未完了、I9保留、I11無効。次は通常共同推論の状態所有と実条件特徴の接続を確認する。R2／A3は別工程。

前版のI10検査（2026-09-20）: 実身体の振幅平滑化を、発行済み指令と候補／defaultのenergy予測へ接続した。
現在値・目標値・一sample係数を凍結し、次の予約更新までを評価する。平滑化中でも共有prototypeの適格性は拡張しない。
実Tone波形の最大絶対差0.000553未満。通常の振幅変更素材で300 record・3,976窓の予測支持を回復した。
音声・実観測・私有trace・共有profile対応は旧版と一致。一定振幅37 recordの既存予測も識別子等を除いて一致。
全Rust 1,262成功、fmt／Clippy／全target check／release通過。記録は`target/i10-amplitude-smoothing-20260920/summary.json`。
次は既知の予約control更新を時刻以後の実身体／default予測へ反映する。共同推論・実身体転用・default・統合の4大工程は維持。
I10未完了、I9保留、I11無効。R2／A3は別工程。

前版のI10検査（2026-09-19）: 中断されていたsection共同遷移adapterを完成し、54条件の独立数値照合と
既存sectionの帰還／展開、組合せ選択からNormalizerまでを検査した。同一文脈内の展開は境界を維持するが、完全stayとは区別する。
全Rust 1,260成功、fmt／Clippy／全target check／release通過。変更はtest build内に限定され、release binaryは変更前と同一。
通常共同推論は未接続。記録は`target/i10-section-joint-transitions-20260919/summary.json`。
追加の既存110 Hz実音三実行で、通常consumerに128組のcandidate／default評定が到達し、各head116組に支持あり。
距離閾値・モデルを維持した実bindingで検査し、独立計算の分布最大差1.39e-16未満。三実行の音声、head追加前後の実学習は一致。
候補間の評定差は全てゼロ。取得recipeでの到達正例であり、実身体転用精度の検証は未完了。
記録は`target/i10-actual-consumer-heads-20260919/summary.json`。詳細は[I10記録](i10-body-outcome.md)末尾。
残り4大工程と工程帰属の未回答を保持。次は実身体転用誤差の実recipe／control／既存Tone別の切分けと接続条件。
I10未完了、I9保留、I11無効、at-action評価維持。R2／A3は別工程。

前版のI10検査（2026-09-19）: 候補groove／参加意欲の基礎評定を共有候補表・身体consumer診断・UIへ接続した。
対応する1音群だけを54入力から再評価し、他音群、発行時観測率・重み・温度・priorを固定する。
混合→温度→prior退避の順序と未支持質量を維持し、原観測・学習を更新しない。共同事後分布による再重み付けは別工程。
通常実音一回で公開434候補、支持あり856評定を独立計算と照合し、最終分布の最大差は2.23e-16未満。
音声・実学習・既存特徴値は一致。音群構成変更時の空候補表撤回68観測と、それに伴う未成立理由の変化を別検査した。
実身体とのpaired正例はこの素材では0件。固定対応の通常数値経路では8評定の変化とconsumer到達7候補を確認した。
releaseで発生したstack overflowは保持snapshotのheap移動で解消した。stack上限は変えず、固定領域を57,568 byte削減。
全Rust 1,258成功、fmt/clippy/all-targets/release通過。候補表最大11,258,096、publication 1,434,272 byte／bus。
記録は`target/i10-candidate-groove-heads-20260919/summary.json`。残り4大工程は共同推論、実身体転用、default対応、統合検査。
転用精度・通常consumer正例・共同推論の工程帰属の未回答は未完のまま保持する。I10未完了、I9保留、I11無効。R2／A3は別工程。

前版のI10検査（2026-09-19）: 候補のword／groupingの4特徴を通常共有候補表へ接続した。
候補評価時刻で終わる8秒窓に残る実観測だけを集計し、未来word・groupingや学習更新は作らない。
発行時2,868座標が通常値と一致。90%支持境界・epoch・容量損失・既知0・原状態不変を独立した数値fixtureで検査した。
通常Frontendから候補表への検査用固定対応で、支持ありgroupingが3候補へ届いた。実身体転用の精度は別要件。
通常実音一回の公開490候補では4特徴は未支持のまま。音声・実学習・318候補の既存値・332観測は不変。
全Rust 1,256成功、fmt/clippy/all-targets/release通過。候補表とscratchは最大10,305,968 byte／bus。
記録は`target/i10-candidate-word-features-20260919/summary.json`。54個の候補raw入力の接続・検査が完了し、提示した残り工程は5大工程。
次は候補groove／参加意欲評定。その後は共同推論、実身体転用、default対応、統合検査。
共同推論の工程帰属の未回答を維持する。I10未完了、I9保留、I11無効。R2／A3は別工程。

前版のI10検査（2026-09-19）: 候補の42時刻特徴を通常共有候補表へ接続した。音群間12本と音群内1本の有界scratchを再利用し、
発行時の実参照を固定して候補accentを加える。実観測・周期選択・学習は変更しない。発行時の30,114座標が通常値と一致し、
周期・非周期・音群内の候補分布を独立したtimestamp差分布と照合した。通常音響処理から候補表への数値fixtureでは、
明示した検査用対応で4候補・4仮想間隔・16座標変化を確認した。descriptor転用の精度はこの対応固定で証明しない。
通常実音一回の公開490候補は時刻支持0件のまま保持。音声・実学習・318候補の既存値・332観測は不変。
全Rust 1,254成功、fmt/clippy/all-targets/release通過。候補表とscratchは最大9,945,968 byte／bus。全負荷受入ではない。
記録は`target/i10-candidate-timing-features-20260919/summary.json`。提示した7大工程のうち候補42時刻入力の接続・検査を閉じ、
残り6大工程。次はword／groupingの4候補入力。その後は候補groove／参加意欲評定、共同推論、実身体転用、default対応、
統合検査。共同推論の工程帰属の未回答を維持する。I10未完了、I9保留、I11無効。R2／A3は別工程。

前版のI10検査（2026-09-19）: 候補評価の8密度窓を通常共有候補表へ接続した。実観測のprefixと仮の未来支持を別記録にし、
既存の候補accent走査を共用する。発行時点の5,736窓が通常密度と一致。通常実音一回で公開候補490件を検査し、
未来部分1,400窓をprofileから独立再計算して一致した。この素材の該当未来accent数は0、正の未来accentは数値fixtureで検査した。
音声・実学習・318候補の既存値・332観測は不変。全Rust 1,251成功、fmt/clippy/all-targets/release通過。
候補表の保持量は1 bus当たり849,600 byte増加した。全負荷性能の受入ではない。
記録は`target/i10-groove-density-projection-20260919/summary.json`。次は残る42時刻入力と4 word／grouping入力の候補射影、
発行時の混合・支持を固定したgroove／参加意欲の候補評定。共同推論・実身体転用・残るdefaultと工程帰属の未回答、
R2／A3を維持する。I10未完了、I9保留、I11無効。

前版のI10検査（2026-09-19）: 明示設定した独立groove／参加意欲headを通常recurrence／report／UIへ接続した。
共有54入力を109座標へ写し、headごとの係数・閾値・温度・priorで評定分布を計算する。未支持の音群質量を保持し、
物理観測率とhead支持でpriorへ戻す。設定なしではheadを追加しない。両bus・二headで計656件の支持あり評定を得た。
独立Python再計算との最大差はpredictor 1.78e-15、最終分布3.34e-16未満。音声・実学習・318候補・332観測は不変。
全Rust 1,248成功、fmt/clippy/all-targets/release通過。検証係数は未fitであり、聴取者の予測精度や候補の効果は未証明。
記録は`target/i10-groove-heads-20260919/summary.json`。次は発行時支持を固定した候補射影。
共同推論・実身体転用・残るdefaultと工程帰属の未回答、R2／A3の未達を維持する。I10未完了、I9保留、I11無効。

前版のI10検査（2026-09-19）: 残る12生入力を通常recurrence／reportへ接続し、既存42入力と合わせた54入力を公開した。
8密度窓は既存accent台帳と実観測秒数を使い、128件容量超過を欠測とする。word entropy／二乗・遷移surpriseと
8秒grouping平均は音響groupingから計算し、元の観測対の二重学習、欠測、世代交代、90%支持を検査した。
通常音響fixtureでは密度707 frame・grouping平均8 frame・約26組の学習を確認した。実音比較は密度1,990件・
grouping平均24件、16秒密度64件の独立再計算が一致。wordの最大支持は17.57%であり、entropy／surpriseは欠測を維持した。
初版release実行のstack overflowをGDBで切り分け、音響解析後の診断集計を別関数に分離して解消した。
修正版は音声・実学習・318候補・既存332観測が一致。全Rust 1,243成功、fmt/clippy/all-targets/release通過。
記録は`target/i10-groove-inputs-20260919/summary.json`。次は独立groove／desire headの明示設定と通常消費、続いて候補射影。
実身体転用・共同推論／全headの工程帰属・R2／A3の未達は維持する。I10未完了、I9保留、I11無効。

前版のI10検査（2026-09-19）: 時刻履歴からmode・全modeへの残差分散・3×14の生特徴を通常reportへ接続した。
exact f64 plateau、wrap／端点／一様分布／16 mode、欠測と適用外ゼロ、選択履歴の重みを検査した。
残差分散はtimestamp差の連続分布を積分し、閉形式と独立601×601点積分で照合。最大差5.27e-07未満。
実音素材の新規取得は一回。6支持履歴・60既知座標のmode／特徴写像を独立再計算し、音声・学習・
318候補・共通332観測と元の履歴を保持した。全Rust 1,237成功、fmt/clippy/all-targets/release通過。
記録は`target/i10-timing-features-20260919/summary.json`。次は残る12生特徴とgroove／desire headの接続。
実身体転用・共同推論／全headの工程帰属・R2／A3は未解決。I10未完了、I9保留、I11無効を維持する。

前版のI10検査（2026-09-19）: 残る帰結入力のうち、実音群の相対時刻履歴を通常recurrence／reportへ接続した。
順序付き42組×周期／非周期と7個の音群内履歴、計91履歴を8秒・128件へ制限する。
周期の更新前固定、同時accent、中央値参照、1秒の差替待ち、欠測・世代・容量退避を検査した。
通常観測fixtureでは支持638 frame。実音の振幅変化素材では両bus各3件の音群内周期履歴を確認した。
flow／pulseは支持0件のまま保持。3素材の音声・学習・754候補・共通806観測は一致した。
全Rust 1,231成功、fmt/clippy/all-targets/release通過。初回の既存テストの一時WAV消失失敗も保存し、
単体と全体の再実行は成功した。1記録56 byte、48 kHz／512 hopでは1 bus当たり745,152 byteの所有量。
記録は`target/i10-auditory-timing-20260919/summary.json`。mode／dispersionと42入力の特徴写像、
残る12入力、head／候補への射影は未接続。次は履歴からの特徴写像。I10未完了、I9保留、I11無効。
実身体への転用、共同推論・全headの工程帰属、R2／A3の未達を今回の入力接続で解消したとは扱わない。

前版のI10検査（2026-09-19）: 通常の遅延onset候補がdefaultの初期位相を使い回す欠落を修正。
実renderer・検証用予測と位相生成を共用した。通常workerと実分岐の3,072 sample比較で最大誤差1.93e-07。
通常110 Hz flowの旧新比較では実音・学習を保持し、遅延66候補の物理予測が変化。他候補と共有headは一致した。
全Rust 1,220成功、fmt/clippy/all-targets/release通過。記録は`target/i10-candidate-phase-20260919/summary.json`。
実身体へのprototype転用、非energy帰結、既に提示した全headの範囲判断は残る。I10未完了。

前版のI10検査（2026-09-19）: 予約control更新前の予測を全て未知にしていた経路を修正。
最初の更新時刻まで支持を保持し、更新後・未収束ampを未知にする。候補onsetの移動で期限は動かさない。
実rendererの250 ms窓の正例、両bus音声・実学習の不変、共有prototypeの制限維持を検査した。
全Rust 1,219成功、fmt/clippy/all-targets/release通過。通常一素材の音声・37候補値・共通52観測が一致。
記録は`target/i10-control-support-20260919/summary.json`。全性能行列の再実行なし。I10未完了。
残作業は、実身体への転用失敗と帰結経路の未接続部分。共同推論・全headの今回の範囲は回答待ち。

前版のI10検査（2026-09-19）: 元prototypeの取得bus／routingを保持し、実Voiceへの誤転用を通常consumerで拒否する門を追加した。
loaderの元bus一致・routing検査、公開表とprofileの整合、両busでの実candidate workerを検査。全Rust 1,217成功、必須検査通過。
通常110 Hz素材の3実行で音声・学習・37候補の身体／私有値と共有表128組は不変。
保存転用72 ownerのうちrouting不一致24件を拒否。残るrouting一致24件にも候補energy順位反転があり、転用の採用は未成立。
拒否24件に距離0.25以内の代替prototypeはなく、単なる選び直しでは解決しない。
実身体への転用、関係支持・invitation・全代替uncertainty、残るdefault接続を引き続き進める。
詳細は[I10記録](i10-body-outcome.md)末尾、証拠は`target/i10-prototype-routing-20260919/summary.json`。
I10未完了、I9保留、I11作用無効、at-action評価を維持。R2／A3の受入は別工程。

前回のI10検査（2026-09-18）: 実Voice判断時計から共有表の音響classへの対応を修正した。
同一特徴軌道の証明を持つcontinue／wait／skipと、同じ選択cellのonset classを対応させ、実動作は変更しない。
全Rust 1,215成功、必須検査通過。通常110 Hz素材の4実行で音声・学習・37候補の身体／私有値は不変。
paired 64→128、unknown_cell 64→0。回復値の出所・支持・差分を照合した。
非ゼロordinal差は0件。身体転用の妥当性・relation／invitation／全代替uncertainty・残るdefault接続は未完。
詳細は[I10記録](i10-body-outcome.md)末尾、証拠は`target/i10-context-clock-20260918/summary.json`。
I10未完了、I9保留、I11作用無効、at-action評価を維持する。

前回のI10検査（2026-09-18）: 受理済み予約だけを持つidle身体を通常候補診断へ接続した。
予約defaultは一度だけ保持し、release／gap、取消、未知制御、世代、容量、新recipeへの置換を扱う。
7条件×3 routingの実rendererで音声・実帰結・学習不変を確認。全Rust 1,213成功、必須検査通過。
通常2素材・6実行の音声・既存1,306候補記録・共通364観測は不変。
通常2素材にはidle予約がなく、新機能の正例はrenderer検査。全負荷受入は実施していない。
未予約機会を含むdefault・実身体に対応した非energy帰結・身体転用と残る通常接続は未完。
次は残る帰結の入力から通常診断までを確認する。I12／R2／A3は別工程として保持する。
詳細は[I10記録](i10-body-outcome.md)末尾、証拠は`target/i10-idle-queued-20260918/summary.json`。
I10未完了、I9保留、I11作用無効を維持する。

前回のI10検査（2026-09-18）: 学習済みrelease headを、実身体候補とdefaultのTone別終端へ接続した。
複数Toneの終端を別々に保持し、gapによる予約除外、即時release、未知・範囲外、元のintrinsic periodを扱う。
全Rust 1,209成功、Clippy・全target check・fmt・release build通過。
4 Voiceの4素材・新旧14実行で音声・学習・既存2,330候補記録・共通652観測は不変。
6,836の条件付きfitと1,466の非ゼロ差、発行済みOffから実終端への698対応を確認した。
flowと持続音metricでは支持0。立ち上がりと減衰を持つmetricで周期参照と正例を得た。失敗結果も保持する。
自然終端への統計的転用・Voice全体の集約・生成作用・全負荷合格は未確定。次は予約onsetだけでidleの身体の候補診断。
詳細は[I10記録](i10-body-outcome.md)末尾、証拠は`target/i10-release-trace-20260918/summary.json`。
I10未完了、I9保留、I11作用無効を維持する。

前回のI10検査（2026-09-18）: 候補比率を4秒窓の補間から実envelope・controlの直接評価へ置き換えた。
候補別の身体支持、予約release・gap、16点間の未知、標本ゼロと既知無音を区別する。
全Rust 1,205成功、Clippy・全target check・fmt・release build通過。
hold／flow各4 Voiceの6実行で音声・実観測学習・比率以外の878記録・95,856物理窓・共通252観測は一致した。
11,982候補の比率式と支持座標を独立照合。短いreleaseの8／16／32点・全sample参照も検査したが、
実音全体の精度／順位、切捨てた物理energy質量、性能合格は未証明。全個体数の負荷再試験は実施していない。
詳細は[I10記録](i10-body-outcome.md)末尾、証拠は`target/i10-direct-body-footprint-20260918/summary.json`。
次は私有release／gapと実身体終端の対応。I10未完了、I9保留、I11作用無効を維持する。

前回のI10検査（2026-09-18）: 環境energy予測の帯域走査を共用し、全周期候補が未成立の場合の同一予測を再利用した。
旧計算とのbit一致と学習状態を使う独立数値検査、全Rust 1,199成功、必須検査を通過。
6条件18実行の音声・実観測学習・全18,462候補記録・250,365比率候補・共通756観測は前版と一致した。
短期wall計測では64 Voiceの個体処理平均がhold約18.8%、flow約11.9%減少したが、
全hop p99はhold約15.8〜16.7 ms、flow約20.1〜24.2 msであり、性能条件は未達である。
追加の核単独ベンチと反復負荷実行は行わず、CPUの速度向上や正式受入は主張しない。
証拠は`target/i10-energy-preview-cost-20260918/summary.json`、詳細は[I10記録](i10-body-outcome.md)末尾。
工数に関する問いを受け、I10の機能経路とI12/R2/A3の接続・受入を混在させた進め方を見直す。
今回の性能修正で区切り、次は機能上の残項目へ戻る。各機能追加に後段の性能合格を要求して延長し続けない。
全要求と範囲回答待ちの記録は保持し、I10未完了、I9保留、I11作用無効を維持する。

前回のI10検査（2026-09-18）: 候補とdefaultのhabitat energy比率診断を通常reportへ接続した。
既存4秒・16窓の身体energyを窓中心の線形補間モデルとして使い、実指令の比率数値核を共用した。
指令のない活動身体にも既存20 Hz標本時の自声除外予測を凍結し、同一支持区間で重なり・可聴性proxyの差を取る。
実envelopeを直接評価する最終footprintとは区別し、範囲外の質量はこの補間モデルの質量だけを記録する。
全Rust 1,198成功、Clippy・全target check・fmt・release build通過。
6条件18実行で音声・実観測学習・既存候補数値・共通756観測は一致。
18,462記録・250,365候補を独立照合し、支持あり250,065候補、非ゼロ差分229,498件を確認した。
一方、64 Voiceの全hop p99はhold約16.2〜16.8 ms、flow約20.7〜24.7 msへ悪化した。
reportなしの個体処理区間の平均は約0.89〜0.94 ms増加し、追加した同期環境予測の費用と整合する。
短期wall時間でありCPU原因の確定や正式受入ではない。40 ms／8.5333 ms条件は引き続き未達。
次はこの環境予測取得の費用を削減し、直接envelope・私有release/gap・残る帰結／default接続を進める。
[I10記録](i10-body-outcome.md)末尾と[候補比率登録](i10-candidate-energy-ratios.json)を参照。
I10未完了、I9保留、I11作用無効を維持する。

前回のI10検査（2026-09-18）: 候補送出をhop内の全batch・更新予約の適用後へ移し、同sourceの後続指令を取り込んだ。
先頭の適格recipeがduplicate／invalidでも、次に実際に受理された適格recipeを選ぶ。
既存active packetを同じpool内で置き換え、一source一件・20 Hz・64件上限を維持した。
Off／Updateを含む活動身体にも拡張し、指令後の継続基準と先頭の実指令を区別する。
全Rust 1,196成功、Clippy・全target check・fmt・release build通過。
6条件18実行の音声・実観測学習・共通756観測・既存候補数値は前版と一致し、新たに167記録を取得した。
64 Voiceの全hop p99はhold約10.3〜10.6 ms、flow約15.1〜18.8 msで8.5333 ms条件を超過し、
私有100 ms完全窓p99も40 ms超である。短期wall時間の失敗を保持する。
[hop全体のdefault接続登録](i10-hop-default.json)と[I10記録](i10-body-outcome.md)末尾を参照。
予約だけでidleの身体・未予約機会・制御更新の解析予測・非energy帰結・身体転用・残る通常接続・全資源受入は未完。
I10未完了、I9保留、I11作用無効を維持する。

前回のI10検査（2026-09-18）: 受理済みの一機会の候補予測を、そのbatchの全指令・更新予約の適用後に固定し、
他の受理済みTone・予約onset・既知releaseを基準と候補の双方へ保持した。先頭の受理指令と焦点機会を別記録にし、
複合defaultを単一classの共有prototypeへ対応させない。gapの半開区間内の予約抑止、release tailの保持、
未対応の制御更新を未知とする境界も検査した。全Rust 1,194成功、Clippy・全target check・fmt・release build通過。
6条件18実行の音声・実観測学習・共通756観測は前版と一致し、既存候補の数値も不変。
候補18,289件から18,295件へ増え、旧未対応の複合指令6件を通常診断へ接続した。
[既知default接続登録](i10-known-default.json)と[I10記録](i10-body-outcome.md)末尾を参照。
対象はsampled batchの既知状態であり、未発行の将来方策、Off単独、予約だけでidleの身体、同一hop後続batchまでを
網羅するfull defaultではない。非energy帰結・身体転用・全資源受入も未完。I10未完了、I9保留、I11作用無効。

前回のI10検査（2026-09-18）: NSGTの不変kernelを共有し、128 cloneの確保要求を124.8 MBから6.3 MBへ削減した。
解析状態は独立に保持し、入力分岐・片側reset・元の破棄後も独立参照とbit一致。全Rust 1,190成功。
6条件18実行の音声・診断内容も前版と一致した。ただし追加12実行の交互比較では、
私有frame合計wall費用が1.5〜5.0%増え、50 ms超の窓も増えた。メモリ削減と計時退行を分けて保持する。
[不変kernel共有登録](i10-nsgt-shared-kernel.json)を参照。負荷条件は未達。次は複数指令・予約を含む
full defaultの通常接続へ戻る。I10未完了、I9保留、I11作用無効。

前回のI10検査（2026-09-18）: NSGTの毎hopのFFT作業用メモリ確保を、構築時の確保と再利用へ変更した。
確保回帰検査は修正前14回・28,672 bytesで失敗、修正後は両power mode・FFT長・reset／clone後も0で通過。
6条件18実行の音声・診断内容は前版と一致し、全Rust 1,189成功。追加12実行の交互比較で、
64 Voiceの私有frame合計wall費用は5.7〜8.9%減少した。ただし全hop p99が増えた対と50 ms超の私有窓もあり、
負荷条件は未達。[FFT作業領域登録](i10-nsgt-scratch.json)を参照。
次は不変kernelの複製費用と、残る候補帰結／defaultの通常接続。I10未完了、I9保留、I11作用無効。

前回のI10検査（2026-09-18）: 私有bodyの全frame／100 ms窓計器と、report非依存の背景終端profileを接続した。
hold／flow各4・16・64 Voiceの18実行で、6組の音声、候補18,289件等の診断内容、共通756観測は前版と一致。
私有611 frame・66窓の費用保存と、report有無での終端回収を確認した。Rust 1,189成功、Python 29成功。
64 Voiceの私有窓p99は全4条件で50 ms超〜75 ms以下となり、40 msのwall時間screenは不合格。
全hop p99もholdでreportなし11.0164 ms、あり10.9278 ms、flowで17.2421／18.0312 msと8.5333 ms超過。
offline完了待ちを含む短期wall時間であり、CPU時間・正式preflight・実機受入ではない。
[背景資源登録](i10-background-resources.json)を参照。次は私有解析費用と残る候補帰結／defaultの通常接続。
I10未完了、I9保留、I11作用無効を維持する。

前回のI10検査（2026-09-18）: 私有bodyのoffline提出をsnapshot公開完了まで同期した。liveは非待機を維持。
hold／flow各4・16・64 Voiceの24実行で、変更前後・反復の音声が一致。修正後の反復とinstrumentでは、
候補18,289件・身体descriptor19,824件・予測／学習・私有trace・共通756観測の診断内容が一致した。
計時と非同期出力順は比較から分離した。旧非同期参照から変わったbinding・支持・学習値は保存。
Rust 1,189成功。同期後64 Voice flowの全hop p99はreportなし15.1642 ms、あり16.8990 msで、
offline完了待ちを含む。負荷合格とはしない。[公開順序登録](i10-body-publication-order.json)を参照。
次は私有workerの全窓費用・reportなし背景計器と、残る候補帰結／defaultの通常接続・検証。

前回のI10検査（2026-09-18）: hold／flow各4・16・64 Voiceの6秒screen、計24実行を完了した。
6組のobserve／off音声・onset・Population記録は完全一致。64 Voice flowの全hop p99はreportなし10.2485 ms、
あり11.3673 msで8.5333 msを超過した。他5条件は短期p99以内だが、正式な60秒warmup＋600秒受入ではない。
候補18,289件のcapacity／output dropは0、共有workerの対応prototypeは最大3個。
私有workerの全窓p99、reportなし背景計器、候補queue／鮮度、全8 prototypeと実機の検査は残る。
[短期負荷登録](i10-population-preflight-screen.json)を参照。
続いて通常profileへrenderer内訳6区間を追加し、18実行と旧binary再実行を取得した。
64 Voice flowはreportなし全hop p99 10.2029 ms、sample処理は合成区間合計の約70%。
音声は全6条件一致したが、一部の身体診断は時刻・予測値まで変動し、全記録一致は不合格。
旧binaryの再実行でも到着時刻差を確認した。上記の公開順序修正はこの発見に基づく。
[内訳計時登録](i10-render-phase-profile.json)を参照。Rust 1,188成功、Python 28成功。

前回のI10検査（2026-09-18）: 同じrequestの物理窓を厳密再利用し、同じTone／時刻のenergy二重評価も除去した。
各段階の8素材・665候補記録・音声・既存学習・共通910観測は一致。窓計算は92,408→49,672回。
交互3往復で窓再利用は合計費用34.4〜48.5%減、続く時刻energy再利用はさらに5.7〜34.5%減。
ただし最大値の退行した比較があり、最終版の個別背景request最大1.676 ms。
正本の1.6 ms条件は全個体の新規判断を合計した一hopのp99であり、この個別最大とは比較できない。
全Voice／worker／live受入は未成立。
[窓再利用登録](i10-energy-reuse.json)・[時刻energy再利用登録](i10-energy-point-reuse.json)を参照。
身体適合・非energy帰結・full default・共同推論等は未完。I10完了とはしない。


前回のI10検査（2026-09-18）: 継続中の既存身体からcontinue／release／gapを通常診断へ接続した。
8素材で651件・87,656窓、非ゼロenergy差48,792窓。音声・既存学習・既存14候補recordと共通910観測は一致。
静的110 Hzの既存prototype取得条件では共有表の64組が通常consumerへ到達した。
ただしclosureは支持なし、continuation差はゼロ。250 ms後評価の比較でも差ゼロであり、非ゼロordinal帰結は未証明。
身体転用の適格性・非energy帰結・未予約機会／複数指令・全資源受入は未完。I9保留、I11作用無効を維持する。
[継続身体候補の登録](i10-active-body-candidates.json)と[I10記録](i10-body-outcome.md)末尾を参照。


前回のI10検査（2026-09-18）: 共有候補表の不変publicationを実Voiceの候補packetへ接続し、
両bus・世代・時刻・期限・未知セル・default支持の検査を実装した。7素材のWAVと既存学習・energy、
共通858観測は一致。14候補recordへ28表が到達したが、実音のpaired正例は0件。
発音対象bus24件は固定descriptor閾値外、残った4対応は無音busとして拒否した。
次は実身体・実defaultに条件づけた帰結の通常正例を示す。閾値緩和で適格性を代用しない。
I10全体は未完了。詳細は[I10記録](i10-body-outcome.md)末尾と[共有consumer登録](i10-shared-consumer.json)。


| 対象 | 現在の状態・証拠 |
|---|---|
| I0 | 保存版の実装準備・MR1は検証済み。[W0–W2記録](m0-rust-preparation.md)。旧M0全体の合格ではない |
| I1 | 初期観測経路は実装・検証済み。[観測記録](m1-observation.md)。モデル境界の追加検査は該当する後続Iへ割り当てる |
| I2 | 直前版で実レンダーのridge継続、音声一致、二bus隔離、欠落・epochを確認。全テスト905成功・0失敗・28ignore、通常Clippy通過（2026-09-12 22:21 JST）。[テスト状態](../../../test_status.txt)は可変ファイルなので時刻と対象版を照合する |
| I3 | 完了。音群frontend・raw descriptor／accentをobserverと共通snapshotへ接続し、UI表示を追加。実レンダーのenergy保存・旧handle帰属・二bus隔離・音声一致、欠落・epoch検査を通過。最終検査状況は[接続記録](i3-i4-observation-memory.md) |
| I4 | 完了。実音の音群特徴→明示した診断区間→保存→後続prefix query→順序付き照合→診断を通常runtimeへ接続。初出／再出・順序変更・元支持・期限・音群とepisode退役を検査。最終検査状況は[接続記録](i3-i4-observation-memory.md)。未校正の診断であり生成作用なし |
| I5 | 完了。実音の音群特徴→4状態とunknown→単一文脈のgesture union→UI／reportを接続。契約・検査状況は[I5記録](i5-gesture.md) |
| I6 | 実accent→周期・まとまり候補→交換可能な到来head→UI／reportを接続。省略・欠測・非周期・複数候補と小型MR2を検査。[I6記録](i6-period-arrival.md)。完了 |
| I7 | 初期のphrase foreground／4出口候補・独立した終止／継続headを、I4の発行済み予測と実観測からUI／reportへ接続。[I7記録](i7-phrase-closure.md)。初期接続の完了 |
| I8 | 初期診断版の技術実装・検証完了。登録39/82成分、0.5秒lagの確定phrase記憶・whole-prefix/fallback、群ごと16局所section候補、内部展開・対比・変形した帰還・想起焦点を通常observer/UI/reportへ接続。実音の一致対照、元query/episode/context、30秒phrase保存、音声不変性を検査。Rust 979成功、Clippy・全target check通過。後続全モデル・R/Aの範囲は[I8記録](i8-section-return.md)に保持。 |
| I9 | 研究拡張へ移管・保留。以下は移管前の技術記録。whole診断head・対応graph・3比較controlを接続。新developmentの旧設定は両busとも30分完走、11,888 episode・3,696退避。全探索で前2帰還の検索損失と第3帰還の退避を確認。16,384件・最大1,024候補をtransport／section／graphまで接続し、1,024候補の165秒実音検査で最初の参照支持59件へ到達。16,384件／512候補で両bus全replayを実行中。取得時計とgraphの対応保持を修正し、165秒の同一source比較を実施。保持モデルのRust数値参照と通常経路の確定順序も検査。Rust 1,006成功、Clippy・全target check通過。第3入力・保持モデルの接続・全曲graph参照・O16採用は未完。[I9記録](i9-whole-context.md)の再設定条件に従う。旧長尺runの実行中表示は当時の状態であり、現在の稼働を保証しない。 |
| I10 | onset／release／shutdown、64 Voice・二busの私有身体記述値に続き、活動／終端の5座標と将来の六記述値の自声予測を接続。発行時予測→実観測照合→更新、固定／動作非依存対照、対応反転、欠測・世代・容量・支持時刻を検査。通常binaryの実音照合とoff／observe×report有無の音声一致も確認。私有traceの有界Rust filterもPythonの60積分条件・112更新・22 lookupと照合した。共有memoryの保持・retrieval score上下限を通常worker／report／UIへ接続し、低支持退避・16件切捨て・実音の非干渉を検査。共有reference inventoryも実音から通常worker／report／UIへ接続し、複数anchor・欠測・世代交代・質量保存を検査。実Voiceの発行時reference／full・elapsed-only予測の凍結と、原観測順のonset／release信用更新を接続。固定素材の実音で同じVoiceの次回予測まで確認し、presentation-onlyの非学習・音声不変性を検査した。候補帰結の数値核を追加し、60条件のDecimal参照と照合。距離・時刻lookup・default差分を含む6検査を追加。実音288 recordの8-medoid適合と通常身体worker／report／UIでの照合を接続。36素材の音声一致、1,512公開recordの独立照合、世代／退役時の消去を検査。共有groupの2秒descriptor cacheとprototype対応も通常worker／report／UIへ接続。物理窓・即時退役・bus／epoch隔離・音声不変性を検査。全Rust 1,066成功、通常Clippy・全target check通過。prototype適合用Python 4検査は前段で通過。既存episodeへの明示的再帰割当、候補帰結の通常接続、I10全体の資源検査は未完。[I10記録](i10-body-outcome.md) |
| I11以降、R/A | 既存の数値参照・部分実装を再利用する。動作接続・研究・受入の合格は未確認。詳細な旧成果と残条件は[M0台帳](m0.md)から参照する |

今回の対象**I3・I4**を閉じた。全テスト912成功・0失敗・28ignore（2026-09-12 22:57:45 JST）、
通常Clippy・全target check通過。続くI5は上表の記録を参照する。
I5を閉じた。全テスト921成功・0失敗・28ignore（2026-09-13 09:07:52 JST）、
通常Clippy・全target check通過。
I6を閉じた。全テスト930成功・0失敗・28ignore、通常Clippy・全target check通過。
[I6記録](i6-period-arrival.md)へ接続・検査結果を集約する。I7初期診断版を閉じた。全テスト941成功・0失敗・28ignore、通常Clippy・全target check通過。
I7の接続・検査結果は[I7記録](i7-phrase-closure.md)へ集約する。続く週次使用率50%超までの継続指示により、I8「sectionと帰還」に着手した。
I8初期診断版は技術完了。I9の自動継続は終了した。I10の最初の候補と生成への用途は§1.2に具体化した。
onsetに続きrelease／shutdown診断も[I10記録](i10-body-outcome.md)の範囲で接続・検査を完了した。六記述値もVoice別・bus別の私有実音から身体世代・原支持付きで診断へ接続した。動作条件付き自声予測も実装前登録したモデルと両対照で照合・学習まで接続した。私有traceの有界Rust filterはPython参照と照合済み。共有memoryの保持／retrieval scoreも通常接続・検査済み。reference inventoryも通常経路へ接続した。実Voiceへの発行時保持と実帰結更新も通常接続・検査済み。候補帰結の数値核も独立参照と照合した。実音288 recordのPAM prototypeを通常身体worker／report／UIへ接続し、36素材の音声不変性と1,512 recordの独立距離照合を検査した。共有group descriptor cacheとprototype対応も通常経路へ接続した。次は候補feature・帰結表の通常接続と全I10資源測定。
I9を待つ依存は設けない。R/Aの収集・適合・可聴性・作者採用・全負荷検査は未完了。
I9の範囲変更を含む15文書は `6a1a8c0`、続く機構選定の5文書は `329f638` にcommitした。既存の実装群と、I10のonset／release接続は未commitで保持する。

直近のI10修正（2026-09-18）: 共同正規化が0音群を拒否する境界を修正。共有文脈だけの更新と初期unknown、全音群退役後の局所slot消去を数値核で検査した。
独立Decimal全列挙は旧25例を保ち4例追加。Rust検査2件で0→8→0→1→0音群、scratch再利用、不正入力拒否と復帰を確認した。
共同推論の通常経路・状態所有・実特徴scoreは引き続き未接続。詳細は[I10記録](i10-body-outcome.md)末尾と[i10-joint-normalization.json](i10-joint-normalization.json)。

直前の負荷検査: articulation内部を分けて計時し、最大割当のビルド条件とrecurrence時間の二重加算を訂正した。
以前の負荷試験はdev/test opt-level 1。release opt-level 3・LTOで同じ入力を取得し、数値の一致を確認した。
releaseの非計時最大値は動作時点34.649 ms、250 ms後46.158 ms、1秒後71.703 ms。前二条件の50 ms超過は0、1秒後は336回。
合成入力・強制割当のtest executableによる結果であり、通常worker全体やVoice/audio deviceの受入ではない。productionアルゴリズムと既定評価時刻は変更していない。
releaseのarticulation内訳は遷移・分岐展開67.38%、sort/merge 21.29%、順位付け・枝刈り7.25%。費用削減候補は分岐展開。
評価時刻の採用、共同推論・全head・実身体消費・全資源受入は残る。
負荷の詳細は[i10-articulation-kernel.json](i10-articulation-kernel.json)。次は共有・局所の状態所有と実特徴scoreの通常接続。I9は保留を維持する。

## 6. 過剰な準備を防ぐ実行規則

- 着手前に対象I、前後で変わる動作、必要な検査、今回の終了点を一行ずつ定める。I/R/Aの終了条件を作業中に追加しない。新しい要求は担当行へ記録する。
- 部品の網羅性を高める前に、同じ入力から実消費者まで一度通す。後続全階層の完成を現在Iの前提にしない。
- 動作を示せないまま実装作業が60分に達したら、阻害要因・残作業・見積りを見直す。検証プロセスの待ち時間は区別する。新しい機構を追加して現在の単位を延長しない。
- 見積りは次の一単位について記録する。実装・検証・人の判断待ちを分け、実測不足は未計測とする。時間枠は残工数の予測ではない。
- Rust変更にはリポジトリ必須のテスト・ログ保存を行う。必要な検査が通れば、新しい変更・失敗・未解決の懸念なしに再検査を増やさない。最大全負荷・全感度・全回収を各Iで繰り返さない。
- 通常の結果記録は本書の現在地と一つの検査記録に集約する。新しい段階別Markdown／JSON／source archiveは、独立した再現実験や比較版の凍結に必要な場合だけ作る。既存成果は削除しない。
- 進捗報告は「完成した動作／検証対象の版／未接続部分／次の一単位」。旧M0の件数比やI/R/Aの単純平均を進捗率にしない。
- 人の判断が必要なR/Aだけを待ち状態にする。ユーザーの明示した停止・再開・終了範囲を優先し、I7終了時の停止は、続く週次使用率50%超までの継続指示で更新された。使用率を定期確認し、超過時に進捗・検証・再開点を保存して停止する。
