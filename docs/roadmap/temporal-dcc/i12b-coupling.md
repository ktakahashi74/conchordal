# I12b：到来期待・観測accentと発音gesture時刻の二経路比較（実装前登録）

状態：**実装前登録。入力・判定器のfreeze前、未実装・未取得。I12b完了判定ではない。** 2026-09-24時点のコードを読んで介入境界を定めた。[Astraの本文仕様レビュー](../../../target/i12b-20260924/prereg-review.md)は可。入力bytes、seed、設定、実行順、判定器、source版のSHAと最終照合は別のfreeze記録を正本とする。結果に合わせて本登録を書き換えて合格扱いにしない。

## 1. 本体範囲と現在地

[milestones §1.3](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)のI12bは二経路に限る。(F) I6の到来期待がgesture候補時刻へ作用する経路。(B) 観測accentが周期推定へ入る経路。各々を独立に除去して比較する。ここでいう「gesture候補時刻」は、Voiceの`TemporalParticipation`が選ぶonset-now／delayed-onsetの発音機会時刻を指す。I5の共有gesture分類・unionや`action_profiles::Table`の固定32時刻を新たな時刻選択器にしない。`Table`は既定offsetの`evaluation_at`へarticulationとarrivalを別の特徴として投影しているだけである（`src/temporal_cognition/action_profiles.rs:208–273`）。独立したgrouping推論、T3評定head、I12a／c／dを本体へ戻さない。

Fの実装は[I11-2結果](i11-2-results.md)の限定受入範囲にある。`Recurrence::arrival_issue`が実音群の201点CDFを発行し、同一snapshotをruntimeがVoiceへ渡し、`TemporalParticipation`が23候補の到来確率をbase費用へ加える（`src/temporal_cognition/proposals/frontend/recurrence.rs:410–447`、`src/temporal_cognition/observation.rs:708–729`、`src/runtime/mod.rs:2495–2532`、`src/life/temporal_participation.rs:590–615, 745–785`）。I11-2のT2は非到来入力prefix、同じCDFによる独立介入、選択・WAV差を検査済み。この結果はFの局所作用の根拠として再利用できる。ただしI11-2の`arrival=false`ではCDF生成も止まるため、今回の「同じ表を届けたまま消費だけ切る」F除去腕そのものを取得済みと呼ばない。

Bの入力経路も存在する。`Recurrence::consume_acoustic`は、旧assignmentのgroup handleへ帰属した実accentを`Estimator::deliver`へ渡し、周期peakとgroupingを更新する。同じ実accentを別途`arrival::Engine::advance`へ渡して実時計・interval・resetを更新する（`src/temporal_cognition/proposals/frontend/recurrence.rs:231–286`）。I6は実accent→周期・まとまり候補→到来予測の診断を通常runtimeへ接続したが、[I6記録](i6-period-arrival.md)もI12bの双方向結合を未完とする。現実装には周期入力だけを切るB腕と、Bを通ったperiod／phase→CDF→Voice候補→音声の独立した除去比較がない。

## 2. 三腕と切断点

三腕は同じシナリオ、seed、binary、I11-2で凍結したHazard係数と身体設定を使う。介入は演奏開始前に固定し、epoch途中で切り替えない。新しい第四腕や成功するまでの素材追加は行わない。

| 腕 | 観測accent→台帳・実時計 | 観測accent→周期pair／peak | CDFの発行・Voice受領 | 到来費用 |
|---|---|---|---|---|
| `full` | 有効 | 有効 | 有効 | I11-2の`arrival_weight = 1.0` |
| `forward-cut` | 有効 | 有効 | 有効 | `arrival = true`、`arrival_weight = 0.0`。既知Pに対して項は`Some(0.0)` |
| `backward-cut` | 有効 | **無効** | 有効 | `arrival = true`、`arrival_weight = 1.0` |

Fの切断は既存設定だけで実施する。`arrival=true`を維持するので、`forward-cut`でも観測worker・CDF表・publication identity・受領時刻・known件数を検査できる。既知Pの費用項が0であること、baseと選択が同じ入力の独立再計算に一致することをbitで確認する。I11-2の`arrival=false`対照は旧回帰として残すが、今回のF切断と混同しない。

Bには小さな診断制御を追加する。`[temporal_period]`の`period_accent_input`（省略時`true`、`false`はB腕のみ）を`Recurrence`生成時に固定する。`true`は既存report形を守るためserialize時に省略し、Bの`false`だけ`period_parameters.period_accent_input`へ出す。**`Recurrence`から`Estimator::deliver`の呼出し全体をskipしない。** `Estimator`内部で`Ledger::deliver`による原accentの所有・重複拒否・時刻・容量処理と戻り値を保ち、周期pairへの挿入・sums・peak更新だけ抑止する。抑止時もviewの受領時刻・失効・空peakを進め、古いpeakを復活させない。台帳は実accentの観測支持を持つが、pair／peak由来の周期支持は無いことをreportの腕識別とともに明示する。grouping候補とpeakに依存するtiming予測が変わるのはBの下流変化として扱う。`Ledger`が新規受領した実accentの周期寄与を切った回数だけ`period_accent_cut_deliveries`を増やす。観測snapshotとprofileの`background.shared[bus]`へB時のみ累積値を出し、epoch間でもrun全体を数える。full／F切断／`None`ではこの新fieldを省略する。

`delivered_accent`は`auditory_timing::Input.accent`にも渡る（`recurrence.rs:239–241, 328–332`）。B腕でもその実accent受領を保つ。`arrival::Engine::advance(raw, accent, context, ...)`へのrawとaccentも保ち、last accent、elapsed／interval更新、reset／uncertain処理、原観測のsource clock、bus／epoch／handle世代はfullと同じ原観測で進める。I5 `Gesture::advance`は同じacoustic snapshotとridgeを読む別経路なので抑止しない（`src/temporal_cognition/observation.rs:472–507`）。抑止するのは周期pairから生じるpeak、そこからのphase／word、下流のCDF採否・形と候補費用であり、音響検出や実accentそのものではない。Hazard生成器はpeak不在でも数値CDFを作り得るが、I11-2のVoice consumerは`periodic=false`の群を採用しない（`src/temporal_cognition/arrival/cdf.rs:98`、`src/life/temporal_participation_arrival.rs:166`）。B腕のCDF数値の有無、周期適格性、VoiceにとってのP unknownを別々に記録し、未知を既知0へ変換しない。初回比較前のraw accent、観測支持、last accent、reset状態の一致を検査する。周期由来の支持は切断後に変わってよい。

## 3. 同一状態と因果順序の検査

全腕で、原音声入力、seed、Voice構成・route、分析hop、旧assignment、group handle／世代、body binding、CDF発行・受領時計をreportへ残す。観測更新内では原音響→accent検出→旧ownerへのcredit→周期推定→grouping→到来更新の処理順を維持し、その後はCDF発行→非同期の受領→Voice候補比較→発音という因果順序を検査する。発行・受領・決定は複数hopにまたがり得る。未来accent・新生groupへのcredit、旧epochの再利用、発行後に届いた表の過去候補への適用を拒否する。欠測・遅配・EOFはI6／I11-2の未知・失効規則のままとし、`None`、既知P=0、重み0を別に報告する。

Fでは最初の選択・skip分岐までperiod、grouping、forecast、CDF、自己除外集合、非到来入力と候補`at`／offsetをbit一致で確認する。唯一の介入差は到来係数であり、各候補のP、base、到来項、総費用と選択を独立再計算する。skipは選ばれた候補の加算前baseで判定する。既知CDF使用が0件ならFの正例に数えない。

Bでは最初の周期差より前の原PCM／NSGT入力、音響accentの時刻・観測支持・owner、台帳、I5 Gesture入力を一致させ、周期pair／peak差が最初の**意味的な**介入差であることを確認する。腕設定と切断件数という自明な差は別に報告し、最初の意味的な差へ数えない。実accent時計・resetと`auditory_timing`への実accent受領も一致させる。次にfull側の周期支持と使用可能な既知CDF、B側のperiodic・CDF・群採否を比較する。B側のpeak消失→`periodic=false`→`nonperiodic`除外→P unknown／到来項なしは正しい下流連鎖であり、両腕の既知CDFを要求しない。CDF数値がB側に残る場合も、periodic適格性とVoice採否を別に検査する。単なるunknown化だけでは成功に数えず、この連鎖が候補費用、選択またはskip、WAVまで伝わることを要求する。別groupへの再帰属だけ、先行した音響差だけでも成功に数えない。

周期切断は`action_profiles`のperiod由来列、grouping、timing予測にも正当に波及し、後段のVoice非到来base費用まで変わり得る。従ってBの通常runtime比較と、到来packetだけを差し替える**直接媒介介入**を別gateにする。通常runtimeでは切断前の原観測・身体・候補格子の同一prefixと、period→CDF採否／値→Voice到来項→選択／skip→WAVという順の総効果を必須とする。最初の周期差以後の`action_profiles`・baseは同一と仮定せず全差を報告する。直接媒介介入では**実走full側**の`TemporalParticipation`状態・23候補・全非到来入力を凍結し、同時刻の実走full／Bから取得した対応publicationだけを交換して、CDF採否／P／到来項からの選択またはskip差を独立再計算する。packetは同じ観測cut・epoch・group系譜に属し、決定`now`で両腕とも受領済み・有効でなければならない。profile Key、Binding、owner、受領・発行時刻と期限は実値のまま照合し、交換に都合のよいidentity／時刻の書換えを禁止する。対応不能なcaseと理由は全件報告し、陽性へ数えない。これにより到来経路の決定への制御直接効果を確認するが、この介入だけで通常runtimeの音響差を説明し尽くしたとは言わない。切断前prefix不一致は媒介介入で救済しない。初回の音響分岐以後は閉ループの入力が変わり得るため、その後の全record一致を要求しない。

## 4. 取得行列と終了条件の案

I11-2の外部pulse付き固定6条件（sine／harmonic／modal × 4／16 Voice）を三腕ともreport有／無でrenderし、各WAV・reportと実行provenanceを保存する。pulseはhabitat、対象Voiceはpresentationという元routeを維持する。6条件×3腕×report有無＝**36本**、さらに旧12条件の新binary `None`を**12本**取得し、offline計**48本**とする。report有無のWAV一致を各腕で確認する。旧入力bytes・seed・設定はSHAで固定し、今回の三腕設定・実行順・判定器・source／binary版を**結果取得前**にfreezeする。旧baseline binaryのoffline再取得もI11-2の全852本の再取得も行わない。新素材が必要なら結果を見て追加するのでなく別登録として扱う。

各身体について、Fは4または16 Voiceの少なくとも一条件で候補選択またはskipと対応WAVが変わることを必須とする。Bは少なくとも一つの**同一条件**で§3の総効果と制御直接効果の**両gate**が成立し、候補選択またはskipと対応WAVが変わることを必須とする。FとBの陽性条件は同じ身体内で異なってよい。六条件の全結果を報告し、差なし・決定なし・full側の適格な既知CDF消費なし・B側の実accent切断なしを正例へ混ぜない。Bの総効果は通常runtimeでperiod pair／peak差→CDF値またはperiodic採否差→到来項差→選択／skip差→WAV差まで追う。Bの制御直接効果は§3の凍結入力・実packet差替えで別に必須とし、実走base差があればCDFを実走差の唯一の原因とは呼ばない。Fではfullとforward-cutが同じCDFを受け取ったことを必須とする。異なる参加位置とgesture状態がどのgroup・Voice・時刻で成立したかを、`temporal_observation`と`participation_decision`の対応で記録する。

既存I11-2のT2、数値核、body自己除外、旧12 `None`／旧24 offの証拠は同じ封印source版に限り再利用する。新binaryの`None`旧12は封印済みI11-2 `None`のWAV SHAと既存10種の学習recordへ照合し、新診断metadata以外の除外を増やさない。新binaryの**period-on `full`**新6はI11-2保存済み同設定のWAV・既存report recordとbit照合する。既存の実時間欄など旧登録で認めた除外だけを引き継ぐ。現設計では新`full`に増えるdeterministic fieldはなく、新metadata除外は空集合とする。実装で必要と判明した場合はrecord種とキーを取得前の入力planに列挙して固定する。`None`だけの非干渉確認では足りない。非同期candidate recordは旧§5.4bと同じく共通keyの決定的内容を比較し、片側だけの件数とdropを別記する。飽和による片側recordを黙って`zip`したり、それだけで失敗にしない。音響・accent・period・CDF・決定の因果核や最初の作用前prefixの欠落は失敗とする。

境界検査は、最初／重複accent、容量・失効、欠測→復帰、遅配、EOF、group出生・退役・新世代、bus／epoch不一致、周期peakなし、CDF既知0／unknown、arrival weight0、予約後新CDFを含む。`Estimator`のB腕で台帳・`auditory_timing`実accent・arrival resetがfullと一致し、周期pair／peakだけが空になる単体検査を置く。三腕のpublication、受領、候補trace、初回分岐の独立照合とWAV差は通常runtimeの統合検査で行う。既定`None`は新診断設定を暗黙に有効化せず、候補版の従来WAV・学習recordの形を維持する。

追加した制御は有界状態と既存scratchを使い、通常fullのhop pathに新規allocationを入れない。性能は上記6条件×report有無の12組で、新binary三腕を各3反復＝**108 instrument実行**、封印済みI11-2 binaryの同じ`full`を各3反復＝**36実行**、計**144実行**とする。旧binaryも新binaryと同じAC給電・蓋開放・覚醒統制下で交互に取得する。新A/A 216本を追加必須にしない。`conchordal` instrumentはair-gapを守り音声を保存しない。

実装非退行の主判定は新`full`対封印旧`full`だけの**一側上限**とする。全hopから各runの`population_us`・`synthesis_us`・待ち込み`elapsed_us`の中央値／p99／最大と予算超過hop数を算出し、時間欄は3反復中央値、超過数は3反復合計で比較する。中央値／p99は`new−base <= max(0.05×base, 旧floor)`、最大は`new−base <= max(1,000 µs, 旧floor)`、超過数は`new−base <= 旧floor`。12組×10欄＝**全120欄の合格をI12b技術完了に必須**とし、失敗は未達としてrawとともに保存する。軽量化は失敗とせず、同じ機能・有効消費はWAV・決定・観測・件数の別gateで検査する。旧I11-2の固定A/A floorは事前に存在する工学幅として再利用し、新環境の雑音推定や統計的同等性の証明とは呼ばない。`elapsed_us_max`など約58 msの大きな幅も隠さない。旧floor付き両側絶対差規則と、floorなしの元±5%／1 ms／超過同数規則も全120欄で併記し、原I11-2の合否へ遡及適用しない。

新`full`対`forward-cut`・`backward-cut`は各12組×10欄、計**240欄**の符号付き資源差と分布を診断として全件報告し、差の絶対値で切断腕を不合格にしない。三腕のworker処理・配送、period pair／peak更新、CDF生成／評価・失敗理由、候補使用件数、drop／置換、総hop timingを同じ入力で記録する。各instrument反復・report有無の**個々のrun**で、`full`と`forward-cut`には既知CDF消費1件以上、`backward-cut`には実accentの周期切断1件以上をprofile counterで要求し、他runの件数で補わない。この適格性を取得前planへ固定する。測定値と既存I11-2の負荷表を別母集団として示し、R2の全8 prototype・両bus・長時間受入やA3のlive deadlineへ昇格させない。source変更時は`AGENTS.md`所定の全`cargo test`ログ・同一shell終了status、fmt、clippyと対象統合検査を保存する。原I11-2の11欄不合格、事後限定受入、未fitの診断Hazard、既定`None`、A1／A2・R2・A3の別判定は維持する。これらの別判定全体をI12bの追加必須条件にはしない。
