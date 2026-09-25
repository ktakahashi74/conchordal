# R2：本体範囲の全処理資源検査・実装前登録草案

状態：草案（未レビュー・未凍結）。実装・取得なし

**2026-09-24の対象版改訂。** 最初の正式主判定は、I12bの取得・終結と基準版固定、I4の第一・第二段階、I10再検証、T1のERB帯域同時・前方マスキング地形、T2の包絡変調スペクトル地形、I11-2／I12bの新版再登録・再取得を経た統合版を対象とする。地形追加前のsource／binaryや既存I11・I12b取得を正式R2の反復へ算入しない。現時点のI12b取得は0件であり、対象統合版のsource、binary、設定、モデル、素材SHAは未確定である。T1・T2の担当をI11-3／I11-4とするのは提案であり、作者未承認。以下の行列・予算は新版で再確認する草案で、取得の凍結を意味しない。

2026-09-24追記：作者の「推奨」により、R2-1は通常採用・既定値変更の資源前提とする方針、R2-2はI4第二段階とI10再検証後の最終構成を最初の主判定にする方針を選択した。草案全体のレビュー・数値の凍結・実行開始とは区別する。

2026-09-24の作業ツリーを静的に調査した案である。I12bと並行して作成したため、以下のsource行番号は調査位置であり封印版の識別ではない。入力、source、binary、実行順、指標の分母、閾値、検査器は、独立レビュー後の別作業で固定する。I12bの結果は本草案へ先取りしない。

同日の独立した静的レビューで、現計器の計測境界、要求数の分母、測定末と終了drainの区別、対照の対応範囲を修正した。これは登録全体のレビュー完了や閾値の承認を意味しない。

## 1. 目的と、R2が止める対象

正本は[milestones §1.3](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)、同§3のR2／A3、および[完成計画 O04](../temporal-dcc-completion.md#41-メモから引き受ける必須事項)である。旧R2は両bus・64 Voice・想定／最大負荷の全処理を同時に検査し、人のstage-1収集を止める条件だった。R1が研究拡張へ移った現在、その停止対象は本体にはない。R2の義務自体を削除したという記述はなく、実際に走る処理をすべて数える規則は維持されている（`milestones.md:52–55,151–161`）。

**判断R2-1（方針選択済み）：I11／I12bの生成作用の通常採用と、その作用を有効にする既定値変更について、R2を必要な資源前提とする。** 作者採用A1／A2や実機受入A3をR2で代用しない。A3の最終受入にもR2の結果を渡すが、故障の切り分けや機器条件の確認のためのA3診断取得まで禁止しない。A3の取得開始全体をR2通過待ちにする代案は選ばなかった。実機の出力不足やringの実消費を確認しないと判断できないR2項目があり、循環する停止条件を避けるためである。研究拡張で人の収集を再開する場合の旧停止条件は別途維持する。

本案の「R2通過」は、登録した構成と負荷範囲について全処理の費用・容量・鮮度を確認したという意味に限る。I12bの新旧full非退行120欄、cutとの差240欄、作者による音の採用、無期限運転の成立とは別である。

## 2. 実際に走る処理の目録

有効化は設定とScenarioに依存する。各runに、有効／無効／未計測の三状態と理由を保存する。以下は実装済み経路の目録であり、すべてが既定で実行されるという意味ではない。

| 経路 | sourceの確認位置 | 数える処理・状態・配送 |
|---|---|---|
| 起動・配線 | `src/runtime/mod.rs:939,998–1085` | habitat／presentation解析、両共有observer、私有body、candidate worker、ListenerTwinの有効条件。解析入力64枠、結果4枠、更新8枠を各busで記録する。GUI・report・profileによる起動条件も保存する |
| hop処理区間 | `src/runtime/mod.rs:2020–2315,2321,2363,2412,2462,2728,2901,3085,3128,3145` | 解析待ち→地形更新→listener待ち→snapshot取得→Voice更新→report→合成・配送→meter・UI等。現profileの`elapsed_us`はrow保存直前までであり、worker pacing、row保存、終了集計を含まない。各相の分位値を足して代用しない |
| 音響解析 | `src/core/analysis_worker.rs:20–97`; `src/core/stream/analysis.rs:76–107` | 両busのNSGT、Landscape、spectral history、snapshot clone。habitatはLatest、listenerはOrdered。batch用Vec、結果sendの待ち、欠落時reset、古い結果の破棄も計数対象 |
| 共有認知observer | `src/temporal_cognition/observation.rs:19,153–188,252–268,453–629` | 各bus32枠の入力queue。ridge、音群、raw descriptor、accent、gesture、period／grouping、memory、reference inventory、prototype、action table。受領・欠測・配送drop・拒否・終端を別計数する |
| I4の現存検索 | `src/temporal_cognition/recall.rs:331–445,625–637,750–807`; `src/temporal_cognition/memory.rs:104–319` | 明示memory設定ではSpanの確保・prefixコピー、scheduler、coarse／DTW、path／residual、retention、検索上下限、reference inventoryが実行される。研究拡張に分類された費用も現binaryで走る限り含める |
| 私有body worker | `src/temporal_cognition/body.rs:17–18,306–351,570–668` | 最大64 Voice×2bus、PCM bufferは4枠。NSGT、descriptor、prototype、lock／copy、返却と完了通知。deterministic配送と非同期配送を区別する |
| prototype・候補表 | `src/temporal_cognition/body_model.rs:71–116,138`; `src/temporal_cognition/observation.rs:706–719` | 8枠のdescriptorと実際のmedoid数、全prototype比較、実際の対応、表の生成・コピー・公開・消費。8個を読み込むことと8個に実支持があることを区別する |
| 候補energy／footprint | `src/life/action_candidates/energy.rs:16,733–768,845–875,967–1037` | candidate packet pool・入力・出力は各64枠。footprint要求・結果・返却drop通知も各64枠。submitted、capacity drop、未支持、完了、出力drop、処理最大、footprint要求／完了／置換／再送／解放を区別する |
| 到来CDF配送・消費 | `src/temporal_cognition/observation.rs:721–741`; `src/runtime/mod.rs:2070–2092`; `src/life/temporal_participation_arrival.rs:92–94`; `src/life/temporal_participation.rs` | 201点CDF生成、Arc公開、版ごとの初回受領時計、23 onset候補の適格性・自己除外・P評価。現在の生成作用はhabitat由来。両busのperiod計算と、両busからCDFを消費することを混同しない |
| 共有参照→私有trace | `src/temporal_cognition/observation.rs:595–629,693–703`; `src/runtime/mod.rs:2114`; `src/life/participation_trace.rs:229–264,333–375` | inventoryとretained IDの共有、Voiceごとの参照凍結、期限・世代拒否、信用更新。参照を無効にして得た軽量化を同じ機能の資源改善としない |
| 音声・UI・記録 | `src/runtime/mod.rs:2935–2936,3145`; `src/audio/output.rs:18–43,46`; `src/runtime_profile.rs:18–119` | 2本のaudio Arc、ring容量・実消費・不足、callback、UI snapshot、report直列化、profile。instrumentはWAVを書かない |

AGENTS.mdのDSP Efficiency節は、steady-stateの5種類の割当箇所と、条件付き／起動時／life内部／別threadの割当を区別している。これは完全な計数表でも「5回以下」という閾値でもない。**判断R2-2の一部として、通常releaseの時間測定に加え、別の`profile-alloc` buildによる割当診断を登録する案を推奨する。** 計測buildの時刻を通常releaseと混ぜない。現計器は主workerのRust割当のみであり、保持量・analysis thread・callback・native mallocを数えない（`src/runtime_profile.rs:155,233–291`）。他threadの費用を0とする根拠には使わない。

### 現計器では足りないもの

- `HopProfile`は相別wall timeを持つが、thread CPU timeではない。CPU headroomを主張するにはprocess／thread CPU時間と採取区間の対応が必要である。
- `resources::Snapshot`はframe／table／delivery等の全run histogramで、100 ms窓も全run集計である（`src/temporal_cognition/resources.rs:9–124`）。warmupを除いた600秒だけの精密p99は現在の終端histogramから復元できない。§2.2では測定窓ごとの有界集計と、登録閾値に対する超過計数を最小案とする。精密p99値を別途必要とする場合に区間rawを登録する。
- queueの容量はsourceから確認できるが、全queueの最大占有率、滞留時間、enqueue待ち、lock待ち、コピーbytes、全threadの保持メモリは現profileだけでは揃わない。不足欄は「未計測」とし、取得前に追加する計器とその費用を登録する。
- candidateの`Stats`は要求と処理の一部を数える。prototype別の支持／採用、消費者までの有効到達、単位時間あたりの最大要求率は別集計が要る。candidate出力dropとfootprint出力dropは別の失敗である。

### 2.1 計器を追加する場合の登録案

不足計器はI4後の最終構成へ別工程で追加する。次は全処理計数を満たすための設計案であり、この文書から実装を開始しない。

- 全主runで同じ通常release・profileを使い、report有無を変える。`profile-alloc`は別診断とする。profileの`elapsed_us`にはreportのhop timing書込みも入る一方、そのreport自身の`elapsed_us`は自分の書込みを含まないため、二列を同じ母集団として混ぜない（`src/runtime/mod.rs:2262–2315`）。snapshot／参照受領等、相別欄に独立して現れない処理は全体時間に含まれる。phase合計との残差を隠さず、候補判断1.6 msの範囲を既存phase名だけから推定しない。
- hop／callbackへ計測のための新しい確保、文字列整形、ファイル出力、待つlockを加えない。所有threadの固定カウンター、事前確保の有界な記録域を使う案とし、受け渡しが必要な計器も非待機とする。計器自身の欠落・容量超過を数え、欠測を0や合格へ置き換えない。OSのCPU／RSS採取はhop外で行い、採取周期・対象thread・wall区間を登録する。
- queueごとに容量・投入試行・受理・取出し・破棄理由・最大滞留を対応づける。厳密な最大占有を取れないsnapshot計器は「観測した最大」とし、queue全履歴の最大と呼ばない。dropだけでなくfootprintを優先処理するcandidate workerの候補待ちも残す（`src/life/action_candidates/energy.rs:884–931`）。
- warmup／測定／終了drainの境界はsample tickとwall時刻の対応で固定する。workerの100 ms窓はsample時計上の窓であり、100 msのwall CPU区間ではない。境界をまたぐ窓、epoch reset、未完窓を別欄にし、全run histogramを600秒区間の値へ読み替えない。headroomには、既存profileが除くrow保存等の計器費用と、worker pacing／機器消費の遅れを別途照合する。意図的なpacing sleep自体は処理時間へ足さず、周期・締切の遅れと区別する。

追加計器の受入は、計測区間と分母の検査、既定None等の登録回帰、計器有無の費用差を対象とする。費用差の測定は将来の別診断であり、現時点で計器が無負荷・RT-safeであるとは主張しない。

### 2.2 主判定に必要な最小データ契約案

次の欄を既存profileと終端counterへ接続する案である。新しい汎用計測frameworkや全eventの無制限logは要求しない。表の「追加」は未実装であり、未観測値を埋めるものではない。共通識別は登録版、run、所有thread／経路、必要なbus・owner・epoch・body世代とする。

| 所有者・既存接続 | 時計と母集団 | 既存欄に加える最小の観測と集計 |
|---|---|---|
| 主worker：`HopProfile`、`process_hop` | 整数frame・sample tick、単調wall時計。§5の完全hop集合 | 既存の全hop・相別wall・Voice／Tone数・出力不足を保持する。計時外のrow保存等は別の処理費用として集計する。欠番・重複・非有限値・計器dropを完全性欄へ出す。既存100,000 hop上限を事前照合し、途中拡張・間引きをしない |
| 両busの解析／共有observer、私有body、候補worker：`resources::Meter`、candidate `Stats` | 処理対象のsource sample tickと単調wall開始／終了。共有observerの100 ms窓はbus／epoch別、私有body・候補の共通処理はworker全体で一度だけ | 計器のある経路は全run集計を残して測定窓別の`count`・合計・最大・histogram・登録閾値超過数を追加する。各windowの完全／欠測／境界横断件数も保存する。現計器のない解析費用とcandidate分布は同じ範囲を明記して追加し、既存最大値だけからp99を作らない。table等の内数を二重加算しない |
| 主workerの要求側、candidate worker、結果消費側：`Stats`とfootprint配送 | 要求発行sample tickで測定対象を分類し、投入／取出しの単調wall差でqueue滞留を測る | 候補はpool取得前の要求機会、投入、受理、支持不成立、容量drop、出力drop、処理完了、消費／破棄、pendingを別にする。footprintは投入試行と固有identity、受理、再送、完了、結果／drop通知受領、消費／破棄を区別する。既存の有界pool・queueに付く集計であり、capacity・観測最大占有・最古滞留・未説明未回収、主要payloadのcopy bytes・保持枠数を経路ごとに残す |
| 共有／私有のprototype所有者、主workerの表・CDF・参照消費者：`Shared.assignments`、`body::Snapshot`、到来判定 | producer側はsource／available sample tick、消費側はdecision tick。公開版と世代を照合する | bus×prototype別の有効比較・実支持／対応・対応期間を固定サイズの集計へ出す。表を読み込んだ数と実対応数を分け、私有assignmentが現JSONでは省略される点を補う。必要な経路の公開数・有効受領数・適格decision数・実使用数、stale／未来／世代不一致等の拒否理由を集計する。既知0・既知非ゼロ・未知を分け、分母0を成功率へ変換しない |
| process／各threadのCPU、RSS、音声callback：OS採取と既存音声counter | 主workerが測定窓へ入る／出るwall境界と採取時刻。sample時計の対象件数とは別 | CPU累積差と実wall長、RSS観測時刻・値・観測最大、callback実消費・不足・errorの境界差を保存する。hop外の採取周期・遅延を記録し、採取点間のRSS最大や物理再生区間の正確な境界を観測したとは呼ばない。機器なし・thread別値なしは未観測のままとする |

測定窓を`W`とすると、frame費用はsource hopがWに完全に入るもの、要求系列はissue tickがWに入るもの、判断系列はdecision tickがWに入るものをそれぞれ母集団とする。境界をまたぐworker窓と測定前に発行された要求は別欄へ置く。W内に発行されW後に返った要求もdrainまで追跡し、W内完了件数だけで受理数と照合しない。CPUはWに対応するwall区間の実行費用であり、この要求母集団のCPU費用と同一ではない。

copy bytesは実際に複製したpayloadを対象とし、Arcの参照共有を指す先のPCM全量の複製として数えない。固定サイズ×実行回数から求める欄と実測する欄を区別し、確保要求bytes、保持上限、RSSと同一視しない。

有界集計は所有threadで更新し、凍結した容量・overflow counterを持つ。counter overflowや収集欠落があれば完全性失敗である。queueの受理数は結果の配送や消費を保証しないため、入力停止後に「受理済み＝消費済み＋説明付き終端破棄＋未説明未回収」の終端照合を行う案とする。右辺は各要求を一度だけ分類し、完了・中間配送や、同じ結果を複数decisionで使った回数を重ねて足さない。再送は別の受理試行として照合し、固有identity数とも分ける。全要求の履歴保存は追加せず、pool・queue・処理中枠に基づく未完要求の有界状態と累積終端計数で照合する。生成に必要な結果の破棄は説明できても資源失敗になり得るため、回収合格とdrop許容の判定を分ける。

## 3. 引き継ぐ失敗と適用範囲

1. **I10：候補飽和。** [縮小負荷の資源測定](i10-body-outcome.md#資源測定残作業の三件目r2への引渡し)（`i10-body-outcome.md:6710–6749`）では、6.5秒／611 hopのflow条件で要求の55〜98%が容量dropだった。record処理中央値はsine約7〜8 ms、harmonic約80〜210 ms、modal約60〜130 ms、最大約1.4秒。候補要求頻度・窓の重複・搬送波の二乗費用を引き継ぐ。これは全8 prototype・長時間・期限内消費の合格ではない。
2. **I10：64 Voiceのhop超過。** 同節では10.667 msの予算に対し、sine-holdが73 hop、sine／harmonic／modal flowが470／486／485 hop超過した。reportなしでもほぼ同数。変更前にも見られた全体負荷であり、追加機構だけの差を小さくしても解消したことにはならない。
3. **I11-1：待ちを含む総負荷は未受入。** [第1段レビュー](i11-onset-comparison.md#第1段の完了判定へのレビューと対応2026-09-23)と[限定技術完了](i11-onset-comparison.md#stage1-technical-completion)を引き継ぐ。`own_us`への変更は結果取得後の主張範囲の縮小である。生の`elapsed_us`では13/16、元許容では8/16で、解析待ちを含む負荷はR2の対象に残った。R2では待ちを除外した値だけで合格にしない。
4. **I11-2：原試行11欄不合格。** [§5.7](i11-2-results.md#57固定floorによる性能主判定は11欄不合格)の主判定560欄は549合格・11不合格。 [§6](i11-2-results.md#6原試行の未達判定と事後限定受入)の事後限定受入は、AC給電・蓋開放・覚醒統制下の追加252本を別証拠として採用したもので、原失敗の取消しではない。追加試行にも137.379 msのhopが1件あり、floorなしの元規則では239欄不合格だった。
5. **I12b：受け取り欄のみ。** source／binary／検査器のSHA、192 runの完了数、全120欄の非退行判定、240欄の符号付き資源差、各runの到来消費／周期切断件数、原失敗と留保を受領する。すべて未記入である。I12bが成功してもR2の全8 prototype・64 Voice・全worker・長時間検査を代替しない。

## 4. 条件行列案と取得規模

**判断R2-2：運用対象と資源予算をレビュー前に選ぶ。** 対象機器、OS／toolchain、sample rate／hop、実機出力の要否、GUIの扱い、onset比較・memory・habituation等の有効設定、期待負荷と最大負荷を一つのmanifestへ列挙する。推奨は48 kHz／512 sampleを最初の一環境とし、I4分離後に通常採用する構成を主判定にする案である。別機器・別sample rateへの一般化はしない。I4分離前の構成を測る場合は検索費用も含め、後で同じ版の結果として流用しない。

行列の案は次のとおりである。本数は作業量の見積りであり、取得開始の承認でも必須本数の確定でもない。

- 中心行列：sine／harmonic／modalの3身体 × 4／16／64 Voice × 期待／最大の2負荷 × report有無 × 3反復 = **108 instrument run**。すべて両busを同時に動かす。busを別runへ分割して同時負荷の代用にはしない。
- 期待負荷は通常のflowと有効な非ゼロ候補要求を持つ。最大負荷は、登録した音群数、同時Tone数、onset／release機会、候補窓・候補数・搬送波数・表更新率・queue投入率を上限まで与える。holdだけや飽和で提出が減った状態を「最大候補負荷」と呼ばない。要求総数を分母として残す。
- `None`診断基準：3身体 × 3 Voice数 × report有無 × 3反復 = **54 run**。本数に負荷の2水準は入っていないため、第一案では中心行列の期待負荷54 runだけに対応させる。最大負荷と混合身体には対応対照がない。この範囲での差は診断用で、絶対headroomの合否を代用しない。最大負荷にも対応差を求める場合は追加54 run、長期合計252 runへ取得前に改訂する。
- 全8 prototypeの補完：登録8 medoidすべてを用いる混合身体入力で、3 Voice数 × 2負荷 × report有無 × 3反復 = **36 run**。各prototypeの比較実行と実支持／対応を両busで個別に報告する。4 Voiceで8 prototypeを同時に占有できるとは仮定せず、時間交替を使う条件と64 Voiceの同時最大条件を区別する。必要な素材・到達数は未登録であり、届かないprototypeを架空の割当で埋めない。
- 配送の切り分け：habitat-only／presentation-only × 3身体 × 3 Voice数 × report有無 = **36短期run**。これは両bus同時主判定に加える診断である。habitat専用pulse等は同期が実験変数の条件に限り、一般の補助拍として追加しない。

長期198 runを各60秒warmup＋600秒測定とすると、音声時計上で**36.3時間**、短期36 runを各6.5秒なら約3.9分が加わる。終了drain、冷却、build、機器確認は含まない。非実時間処理なら壁時計時間は変わる。規模が過大なら、主張する構成・対照・反復の役割をレビューで明示して縮約し、**取得前**に本数を確定する。結果を見て重い条件だけ除く縮約は行わない。

この36.3時間は新版統合構成を一度測る場合の現行行列案の音声時計であり、地形追加前の暫定測定を加算した費用ではない。T1／T2を別々に本取得する案は採らず、各地形の局所検証とI11-2／I12bの再取得を別の費用として見積もる。T1・T2がCPU、メモリ、queue、結果の鮮度、候補負荷を変える範囲は§2の全経路目録で再点検し、T1のマスキング計算とT2の変調解析・周期推定を実行する所有thread、両bus、Voice数、report条件をmanifestへ追記する。現行計器の行番号・上限・費用は新版で再調査し、未計測を0にしない。

I11-2／I12bのシナリオ、seed、body／Hazard設定、checksum検査器は候補素材として参照できる。ただし短い入力を延長すれば別入力である。全8 prototypeの占有、64 Voice、長い保持、最大要求率が満たされるかを再登録し、旧試行を新行列の反復へ算入しない。

各反復のseedは事前に固定し、report有無と対応する`None`には同じseed・外部入力・身体・routeを割り当てる。`None`が切る設定を一覧化し、I4の有界参照、私有trace、到来、habituation等の有効条件を省略しない。主条件は有効にする処理の実要求・実到達が非ゼロであることを適格性として検査する。`None`は複数経路を切る診断であり、一機構だけを切った因果対照ではない。さらに生成作用のon／offで発音数や音響が変われば処理量も変わるため、同じseedでも差を機構の純粋な追加CPU費用とは呼ばない。固定入力・固定作用列の対照を必要とする場合は別に登録し、R2の絶対判定とA1／I12bの接続除去を代用しない。

### runの識別・反復・適格性

登録版と一つの機器・計器構成に対して、予定runの識別因子を「行列区分、身体入力、Voice数、負荷、report有無、反復番号、route」とする案である。中心は3身体×3 Voice数×2負荷×2 report×3反復、Noneは同じ3身体×3 Voice数×期待負荷のみ×2 report×3反復、混合身体は一つの登録入力集合×3 Voice数×2負荷×2 report×3反復であり、routeはいずれも両busである。区分を省略しないため、異なる役割のrunが同じIDにならない。短期36本は別区分として3身体×3 Voice数×2 report×2片bus route、反復は各1回とし、長期198本の反復へ混ぜない。

3反復の第一案は、同じ登録入力・seedでprocessを毎回起動し直す3回の取得である。機器・スケジューリング変動を観察するもので、3人の標本、3つの音楽素材、seed感度試験とは呼ばない。seed値と提示順はまだ作らない。中断した試行はattemptを別識別して保存し、同じ予定runの再取得を新しい反復として算入しない。再取得の可否・扱いは取得前に定める。

適格性と資源判定を次の順に分ける。いずれも結果を見て重いrunを除外する規則ではない。

1. **完全性。** 必須計器・設定・版・測定区間が揃い、欠番、計器drop、非有限値、識別重複がないことを先に検査する。不足時は主判定不能とし、得られたrawと資源失敗は保持する。
2. **負荷を与えた証拠。** 取得前に、Voice／Tone数、要求機会・表更新の率と継続区間、窓・候補・搬送波数、prototype対応範囲の必要値を条件ごとに列挙する。最大負荷の要求率はpool取得やbackpressureで省略する前の機会を分母にし、投入率・受理率・処理率と並記する。例えば要求機会1,000件、受理100件なら、100件だけを最大負荷の要求数にしない。要求機会自体が登録値に届かない試行は最大負荷として不適格である。機会は足りるが飽和で受理・実使用が不足した試行は、その不足とdrop等の資源失敗をともに残す。軽くなった処理時間だけで最大負荷合格としない。
3. **必要経路の実行。** 混合身体行列ではbus×8 prototypeそれぞれの実支持／対応を要求する。比較演算だけの実施、空の対応、同一groupへの複数prototype対応を、8つの独立した音源が働いた証拠へ読み替えない。4 Voiceでの時間交替と64 Voiceでの同時条件を事前の区間指定で区別する。必要な表・CDF・参照の消費は対応する適格decision数と実使用数で検査し、0件や登録下限未満ならその経路の検査未達である。Noneで意図して切る経路にはこの非ゼロ要件を課さず、無効理由を記録する。
4. **資源予算。** 適格な各runへ§5の絶対予算を適用する。未達runの既知の超過・drop・誤消費は、適格性不足を理由に取り消さない。適格性不足と資源失敗は併存し得る。反復中央値や他prototypeの成功で不足を補わない。

率・継続区間・有効消費の下限値は最終構成と負荷素材に依存するため未凍結である。未記入のまま取得を開始せず、実装後の計器・素材確認と主取得を分ける。

### 4.1 測定対象版に固定するfootprintの出所

作者の確定判断(b)は[設計台帳「身体の時定数と拍の時定数の最終的な配置」](../../design-notes/technote-ledger.ja.md#body-beat-time-constants)に従う。現在の本体はI11-1の `footprint = "body"`、すなわち**合成器由来の代表予測**のままA4まで閉じる。学習したfootprintへの置き換えは現在のA4にも今回のR2の合否にも含めず、A4の後の次の単位として登録する。この決定と、後に選択したR2／I4の工程方針は別の判断である。

後続取得のmanifestには、source／binary／設定のSHAに加え、`footprint_origin = synthesizer_representative`、I11-1実装版、代表recordの生成規則・身体パラメータ・D・bin・打切り規則、profile／prototype表の識別、I4の段階と参照規則の版を固定する案とする。既存の自声学習が動くことと、footprint自体が学習予測であることを区別する。現草案は値を凍結していない。

学習版へ置き換えた場合は、旧R2へ結果を継ぎ足さず、変更の影響範囲を明示した**新しい登録**で次を測り直す。

- footprint要求の頻度・payload・確保／コピー、専用の要求／結果／返却drop通知queue、占有・滞留・飽和・未回収・再送・置換・解放。
- 識別hashの生成・照合費用と意味。身体・代表条件・モデル版や更新世代のどれで無効化するか、hash不一致時の破棄・再要求・stale退避・誤消費を測る。識別の省略を軽量化と扱わない。
- 候補workerの予測・投影・energy計算、通常候補とfootprintの競合、要求から消費までの鮮度。学習・モデル公開・私有body／共有observerの費用が変わるなら、その所有thread・コピー・保持量も含める。
- hop pathのうち、snapshot／到来・参照受領、Voice更新と候補要求・footprint消費、合成配送内のrecord回収とfootprint配送、reportの該当相。相別費用に加え、待ちを含むhop全体、音声配送・callback不足、CPU・メモリへの波及も再測定する。影響が広ければR2の再測定範囲を広げる。
- [I11-1のA1](a1-i11-audition.md)を新版について再実施する。旧版のA1／R2を学習版の成立へ自動継承しない。

<a id="i4-r2-order"></a>

### 4.2 I4との条件付き工程順

前回の判断4（本書R2-2のうち最初の運用範囲）は、I4第二段階とI10再検証後を選択した。[I4の引渡し順](i4-recall-separation.md#r2-handoff-order)も同じ選択を反映する。以下の第一分岐を採用方針とし、後二分岐は選ばなかった代案として残す。第一段階は旧producerを残す責務分離であり、I4残作業の完了ではない。

- **第二段階後を主判定にする方針（選択済み）**：I12bの取得・作業終結と基準版の固定 → I4第一段階のbit回帰 → 第二段階の新規則登録・実装 → 研究検索の配置判断とI10／影響する生成経路の再検証 → T1地形とそのA1 → T2地形 → I11-2／I12bの新版再登録・再取得 → T2のA1とT1証拠の回帰照合 → 最終処理目録・不足計器の検証とR2登録 → R2 → R3／A2／A3の最終受入・A4監査。T1とT2のA1は対象版を固定したうえで並行可能である。T2変更後のT1回帰で証拠を引き継げない範囲はT1のA1を再実施する。R2の正式取得はI10再検証と両地形の統合まで待つ。R2合格をI4実装の着手条件にはしない。
- **第一段階後に先行検査する案**：I12b終結・基準固定 → I4第一段階のbit回帰 → 旧producerを含む版のR2登録・取得 → 第二段階・研究検索の配置・I10再検証 → 最終構成のR2を別登録する。第一段階までで足りるのは、その暫定版だけを対象とする検査である。bit一致はCPU時間・割当・queue費用の一致を意味せず、第二段階後の資源受入へ流用しない。先行検査の目的・本数も未承認である。
- **I4分離前を最初の対象にする代案**：I12b終結後の現構成を固定し、現存検索を含めてR2を行う。後続のI4分離・I10再検証後には新版のR2登録が要る。前回の判断4を採らない選択肢である。

旧検索を本体依存として採用し直す代案をI4-1で選ぶなら、本体範囲とI4完了条件の改訂が先に必要であり、第一段階だけを現在の定義の「I4分離完了」とは呼ばない。研究検索をmainに残す選択でも、通常経路が呼ばない費用を仮算入せず、選んだ運用構成で実行する費用を数える。

全分岐ともR2対象は§4.1の合成器由来版である。学習版の置き換えをI4や今回のR2の待ち条件へ追加しない。作者の確定判断(b)による順序は、現在のA4 → 学習footprintの次単位 → I11-1のA1と影響するR2の再実施である。I4が生成音や参照を変える場合のA1証拠の引継ぎは別途照合し、既存WAVの試聴だけで新版まで受け入れない。

上の矢印は資源検査の依存順である。A4には[milestones §1.3・§3](milestones.md#3-研究判定と最終受入)に従い、縮小後の本体I（I13を含む）、R3、A1／A2、適用Oの未完項目も残る。R2からA3へ進んだことだけで、これらを通過したとは扱わない。

## 5. 指標・集計・閾値の案

以下はすべて未確定の提案である。旧契約を根拠とする値も、新スコープに適用するには取得前レビューが必要である。

- **hop headroom：** 予算 `B = 10^6 × hop / sample_rate` µs。48 kHz／512では10,666.667 µs。主案は待ちを含むhop処理区間のwall p99 ≤ 0.8 B（8,533.333 µs）とする。根拠は旧`reference_operating_envelope`の20%余裕であり、現在の実現可能性を示す測定値ではない。現`elapsed_us`の除外範囲は§2.1のとおりで、この一列から全thread・計器込みのheadroom成立は導かない。全hopの中央値・p99・最大・`elapsed_us > B`件数と率も必須とする。`own_us`は補助指標に限る。
- **予算超過：** 最も厳しい案は測定区間の超過0件である。p99だけを必須にして少数超過を許す案を採るなら、許容件数・連続長・最大遅延を事前に別登録し、0件案と混同しない。反復中央値で単発最大や悪いrunを隠さない。
- **worker：** 旧値の共有／私有100 ms窓wall p99 ≤ 40 ms、合成された時間判断のp99 ≤ 1.6 msを検討出発点とする。現計器の`over_limit`の50 ms／25 msはそれぞれ処理／配送の計器設定であり、この40 msや消費期限と同義ではない。壁時計の並列区間を合計してCPU予算や全判断1.6 msの達成を主張しない。判断経路の開始・終了点を新たに特定する。
- **鮮度・配送：** source→available→issue→receipt→decisionのsample時計と、wall公開遅延を別に測る。publication版、Key、owner／epoch／世代、期限切れ理由を保持する。未来・別世代・期限切れpacketの誤消費は0件を必須案とする。既知到来の消費率、body／proxy(absent)／proxy(stale)、publication置換率と遅延の許容上限は未決である。既知0・未知・重み0を合算しない。
- **drop：** 期待負荷で要求の容量drop・出力drop・source欠測・解析配送dropを0とする案を出発点とする。最大負荷でdropを許すなら、診断候補の欠落と生成に必要なfootprint／CDFの欠落を分け、許容率・復旧時間・有効消費の下限を取得前に定める。要求率を抑えて合格させる変更は別の実装・登録にする。現candidateの`capacity_dropped`には支持判定前のpool取得失敗も含まれるため、`capacity_dropped / (submitted + capacity_dropped)`は既存計器での診断比率であり「支持付き要求の失敗率」ではない。`unsupported`、未提出のpending、pool取得前に省略した機会は別欄とし、必要な試行母数を新計器で確認する。footprintの`footprint_requested`は受理数なので、投入試行数は`footprint_requested + footprint_dropped`である。同一identityの再送は試行数に含め、固有identity数・再送数を別集計する（`src/life/action_candidates/energy.rs:947–1002`）。
- **容量・コピー・メモリ：** 各queueの上限・最大占有・飽和時間、確保／解放数、コピーbytes、各所有者の保持上限、process RSSの開始／測定末／最大を報告する。有界状態の上限遵守を必須案とする。測定区間末には正当なin-flightが残り得るため、その件数・最古時刻を保存し、新規入力停止後の登録済みdrain期限内に、受理済み要求の未説明の未回収を0件にする案とする。処理完了、結果配送、消費、明示破棄、退役は別状態であり、完了件数だけで回収を示さない。RSS上限MB・定常傾き・drain期限の値は機器・計器の確定後に登録する。RSS増加だけをリークと断定せず、allocatorの保持と所有状態を照合する。
- **実音声：** 機器を含む構成を選んだ場合、callback不足・エラー0件、測定末までの実消費、実出力先を確認する。`--play=false`では値は未観測であり0ではない。A3の長時間復旧・API／UI一致・音の採用は別に残る。

分位値の主案は、runごと・事前固定測定区間のraw列に対する昇順`floor(q×(N−1))`番目とする。I11-2／I12bとの計算規約を合わせる意図である。既存`runtime-load-evaluation.md`の線形補間とは異なるため、旧集計値を直接継ぎ足さず必要ならrawから双方を併記する。run別値を全件示し、絶対予算は各runで判定する。反復中央値は代表値、最大は全run最大、超過数は分母付き合計とする。

主案の測定窓はsample時計の`[60 × sample_rate, 660 × sample_rate)`に完全に入るhopとし、判定には整数の`frame_idx × hop`を使う。48 kHz／512ではframe 5,625〜61,874の56,250 hopが完全に入り、他の設定で端をまたぐhopは別欄に残す。開始時に容量を決め、欠番・重複・`truncated`・計器dropは完全性失敗として扱う。全run／warmup／drainも保持し、測定窓外の失敗は別記する。profile内蔵summaryは全run・線形補間であるため、この窓の主判定を直接代用しない。

histogramしかないworker分布は区間として報告し、上端を精密p99と呼ばない。`synthesis_us`は`render_route_us`の内数、rendering各相はsynthesisの内数、CDF／table生成はworker frame費用の内数であり、重ねて加算しない。

workerの最小計器でも、測定窓の全N件と登録閾値Tを厳密に超えた件数`n_gt`があれば、p99の**閾値通過**は判定できる。N>0、0始まりの順位`k = floor(99 × (N − 1) / 100)`に対して、`p99 ≤ T`は`n_gt ≤ N − 1 − k`と同値である。Tと等しい値は超過に数えず、整数ns等の登録した単位で比較する。これは精密p99値の復元ではない。N=0は判定不能であり、欠測や非有限値をNから除いて通過させない。現`over_limit`の50 ms／25 msは別の閾値なので40 ms等の判定へ読み替えず、登録したTの計数を追加するか、そのTを境界に持つ完全なhistogramから算出する。

I11-2の固定A/A floorは、旧条件の非退行比較の参考併記に限る案を推奨する。`elapsed_us_max`の58.280 msや超過318 hopという幅を、新R2の絶対deadlineに加えない。新しい雑音調査を行う場合も、回数・終了条件・用途を本取得前に登録する。floorを収束するまで増やしたり、失敗後に広げたりしない。

## 6. 既存記録の位置づけ

- [hardware-baseline.json](hardware-baseline.json)：`inventory_only_not_a_feasibility_pass`。機器・旧運用条件の目録であり、物理出力や現binaryの合格ではない。
- [feasibility-preflight.json](feasibility-preflight.json)：`unmet_partial_measurement; full_simultaneous_preflight_not_run`、`full_preflight_result = null`。研究拡張を含む旧全処理の要求・部分測定である。未実装処理を0で埋めない。
- [i10-background-resources.json](i10-background-resources.json)：`report_independent_background_meter_verified_resource_screen_failed`。短期screenで、64 Voice私有窓p99が(50,75] msとなり40 msを超えた記録を含む。旧研究条件の記録として保持する。
- [i10-render-phase-profile.json](i10-render-phase-profile.json)：`renderer_phase_measurement_verified_audio_preserved_diagnostic_reproducibility_failed`。計時内訳の検証と診断再現性の失敗を別々に引き継ぐ。後続の限定受入と置換して消去しない。
- [runtime-load-evaluation.md](../runtime-load-evaluation.md#記録する範囲)：計器と機器判定の意味を借りる。profile容量100,000 hop（48 kHz／512なら約17.8分）、warmupと全run背景meterの違い、非実機値の限界を守る。30分以上のA3を現在のprofile容量のまま完了したと扱わない。

JSONは今回、識別・状態・要求／限界の小さな欄を確認した。全raw・全checksum・旧機器の現状は再監査していない。

## 7. 後続取得の手順案

1. R2-1の停止対象、R2-2の運用構成・予算、§4.2のI4段階、§4.1のfootprint出所と実装版を確定する。全実行処理と足りない計器を一覧化し、実装が必要なら別工程で追加・検証する。本草案から実行しない。
2. 新しい実験ディレクトリへ入力・seed・route・prototype表・設定・source／binary・toolchain・検査器のSHAを固定する。既存I10／I11／I12bの成果物を上書きしない。最大条件の件数、全8 prototypeの実支持、非ゼロ消費の適格性を先に定義する。
3. AC給電・蓋開放・sleep抑止を開始時・各run前後で検査し、OS電源ログを保存する。build・test・他の取得・重いagent処理を同時実行しない。温度・電源・冷却間隔・画面状態も登録する。バッテリー運用の支持は別行列である。
4. 条件の順序はseed付きの事前固定順とし、report有無・負荷・身体をblock内で回転する。反復を逐次実行し、各runは60秒warmup＋600秒測定を第一案とする。全実行・測定区間・終了drainを別記し、warmupを任意に延ばさない。
5. instrumentはprofile／reportのみを書く。音声回帰が必要な場合は別の`conchordal-render`試行を事前登録する。instrumentへWAV出力を追加しない。deterministic renderの待ちとlive非同期の費用を混ぜない。
6. failed／interrupted run、全hop、drop、環境逸脱、未完queueを保持する。live PIDやlockを確認せず再開・再実行しない。逸脱は環境統制失敗として残し、都合のよい再取得で置換しない。
7. 独立検査器で完全性→適格性→絶対資源予算→参考の相対差の順に判定する。失敗があれば対象の構成と原因を記録する。R2／A3／作者採用を分けて報告し、失敗を合格へ書き換えない。

## 8. 判断点と草案の終了点

- 方針選択済み：R2-1は通常採用・既定値変更の資源前提（前回1）、R2-2はI4第二段階・I10再検証後（前回4）。I4-1／I4-2（前回2／3）は二段階分離と、代替規則検証・保存確認後の研究branch移管を選択した。割当診断・機器出力・最大負荷・本数・予算の詳細は未凍結であり、この方針選択だけで実装・取得開始とはしない。
- 別の判断で決まる：I4を分離してmainから研究検索を除くなら、その後のR2主判定には除去済み検索を要求しない。分離前またはmainに検索を残す構成なら、実行された検索費用をすべて数える。いずれも過去の失敗は保持する。
- 後段の登録事項：I4後の最終構成と計器の確定後、機器出力・GUI・対象負荷、主張に必要な本数、drop率・鮮度・メモリ上限・drain期限・prototype素材を取得前に固定する。数値予算と運用範囲の採用に必要な作者判断はこの時点でまとめる。198 run・36.3時間は現行案の規模であり、今その実行承認を求めるものではない。別機器・バッテリー条件、A3の長時間復旧行列はさらに別の登録で扱う。

本草案の終了点は、失敗の引渡し先、現処理の目録、計器不足、行列・判定の提案、必要な判断がレビュー可能になった状態である。R2の登録完了・計器実装・性能成立を意味しない。
