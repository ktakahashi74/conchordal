# I4：有界因果参照と研究用episode検索の分離・設計登録草案

状態：草案（未レビュー・未凍結）。実装・取得なし

2026-09-24追記：作者の「推奨」により、I4-1は二段階分離、I4-2は代替規則の検証と保存確認後に研究検索を研究branchへ移してmainから除く方針を選択した。R2は第二段階とI10再検証後を待つ。実装・移管・凍結はまだ行わない。

2026-09-24の作業ツリーに対する静的調査である。I12bの未commit変更を含むsourceを読み、変更していない。行番号は調査位置であり封印版ではない。実装はI12bのcommit後、別worktreeで行う前提である。

同日の並列静的レビューで、旧残差の尺度、圧縮後の支持位置、型依存、第一段階の比較境界を再確認し、第二段階の候補を具体化した。本書全体の登録レビュー・凍結・成立判定は未完了である。以下のcommit・worktree・build・取得は後続工程の記述であり、今回の文書作業では行わない。

## 1. 残す責務と現状のずれ

正本は[milestones §1.3](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)と同§5のI4残作業（`milestones.md:215–217`）である。本体に残すものは短期反復・帯域エネルギー予測、受領証拠の保持・減衰、それらと音響支持から作る2秒窓・anchorによる有界因果参照である。coarse／DTW、ordered／orderless episode検索、query schedulerは研究拡張とする。I4の分離をI10の過去の限定完了へ遡って追加条件にしない。

**単なるファイル移動では依存は切れない。** 現在の`reference_inventory`は`recall::Snapshot.retrieval`からepisode別weightを読む。そのweightは、coarse／DTWの残差を得点化し、保持状態と合成した結果である。検索を止めてretentionだけ残しても、従来の参照入力は生成できない。

もう一つの注意は「prefix query削除」の意味である。09-20の記録はphrase由来のcue封緘・prefix query等の削除を述べるが、現在も`Recall::advance`は固定スパンの途中で`Span::prefix`を作りschedulerへsubmitする（`src/temporal_cognition/recall.rs:402–418`）。phrase cue由来の機構が消えたことと、prefix生成がすべて消えたことを同一視しない。本分離は残る固定スパンqueryも対象にする。

## 2. sourceで確認した依存

### 2.1 inventoryが直接読む欄

`src/temporal_cognition/reference_inventory.rs:139–146,329–443`の`Stream::advance`は`&recall::Snapshot`を受け取るが、直接読む最上位欄は`retrieval`だけである。

- 群選択・因果条件：`retrieval[].group`、`evaluated_at_sample == cut`、`available_at_sample <= cut`。
- biasと出所：`no_memory_bias`、`query_id`、`support_start_sample`、`support_end_sample`、`available_at_sample`。bias不一致はエラーになる。
- 質量：`discarded_weight[2]`と`entries[].weight[2]`。群の音響assignment比率×2秒窓coverageを掛ける。
- 同一性：`entries[].episode`、`generation`。inventory側でepochとperiodic／nonperiodic familyを加えてKeyを作る。

`retention`、`acquisition`、`latest`、`group_queries`、検索件数、`acoustic_score`、`availability`、`score`はinventoryから直接は読まない。ただしweightの生成に間接依存するため、不要な計算と即断できない。

音響側からは旧／保持group handle、assignment rows、rawの支持・availability・known量、実accentを読む。period側からは同じcutの有効群、元支持、支持最大のpeakを読む（`reference_inventory.rs:157–328`）。直近2秒の支持ring、欠測後anchor破棄、最大7群、最大16参照、lower bound信用、未割当／検索切捨て／inventory切捨て／anchorなしの分離を保つ。

ここで`search_discarded`へ入るのは、得点・weightを計算した後の16件超過分である。coarse検索で評価されなかった候補の質量を推定した値ではない（`recall/retrieval.rs:225–229`、`reference_inventory.rs:345–348`）。また`unassigned = 1 - assigned`は切捨てやanchorなしを既に含む残量であり、理由別診断をもう一度加えて全質量としない。`weight`の上下限も二つの独立したPMFとは扱わない。維持する質量条件は、採用した下限信用の和が1以下で、その補数を未割当に置くことである。

### 2.2 private traceはSnapshotを直接読まない

`src/temporal_cognition/private_trace.rs`には`recall::Snapshot`の入力がない。読むのは`Reference`、Key、anchor interval／period／weight、保持Key、観測支持である（`private_trace.rs:3,37,83,448–527`）。間の配線は次のとおりである。

1. `observation.rs:595–629`でmemory snapshotからinventoryを作り、同:693–703で`reference_inventory::Context { inventory, retained }`を公開する。
2. `retained`は`Recall::retained_ids`が現episode bankの`(id,generation)`から作るArcである（`recall.rs:130–140`）。inventoryの16参照だけから再構成すると、現時点で低weightの保持episodeまで退役扱いにしてしまう。
3. `runtime/mod.rs:2114`付近から`life/action_observation.rs:190–199`を経て`life/participation_trace.rs:229–264`へ配送する。epoch resetと保持Keyによるtrace退役を行う。
4. `participation_trace.rs:333–375,516`以降で、発行時点の有効inventoryとretained IDを照合し、参照・anchor・Voice固有周期を凍結する。inventoryの発行時鮮度は100 ms以内であり、retrievalのquery終端に対する0.5秒条件とは別である。後から得た新しい参照を過去の動作へ適用しない。

### 2.3 Recallが検索に依存する箇所

- 型の所有：`recall.rs:3,58–70,106–126`。`memory::Episode`、`memory::Residuals`、`query::Scheduler`、transport controller、pending dispatch／report／matchesを同じRecallが持つ。
- 初期化・配線：`observation.rs:61,252–268`。memoryの明示設定でRecallを作り、retention設定があるとinventoryも作る。既定Noneではこの診断を起動しない。
- query：`recall.rs:331–418`。群ごとのSpan、二hop cadence、最大128 knot、明示query cadence、固定spanのprefixを用いる。
- 検索：`recall.rs:422–445`から`memory::ordered_index(..., true, candidate_limit)`へ入る。`memory.rs:104–176`のcoarse候補化と同:260–319のDTW／変換試行が残差を作る。orderless関数もmainにあるが、この通常Recall呼出しはorderedである。
- 受領と保持：`recall.rs:625–637,686–748,750–807`。queryを先に行ってからepisodeを封緘し、自己照合を防ぐ。満杯の退役、transportの期限・世代照合、scheduler完了後に群のmatchesを更新する。
- retrieval：`recall/retrieval.rs:135–249`。受領済み・期限内・音響支持あり・query終端が0.5秒未満という条件で、episodeごとの最良残差得点を取り、保持availability上下限と合成して16件まで公開する。
- retentionも完全に独立ではない。`recall/retention.rs:333`は`super::memory::MAX_EPISODES`を参照する。保持容量の定数と残差表現の所有を切り分ける必要がある。Coarse入力の型が残っていても、現episode封緘は`coarse = None`である（`recall.rs:710`付近）。

## 3. 「明示した距離則」の実体

実装は`MatchSnapshot::acoustic_score`（`src/temporal_cognition/recall/retrieval.rs:96–131`）、設定は`TemporalRetentionConfig`（`src/config.rs:441–453`）である。曖昧なmatch、retentionなし、観測残差なしは得点を作らない。

観測数をN、支持のある座標集合をJとすると、距離は「Jの各座標の `sqrt(coordinate_squared_error[j] / N) / scales[j]` の平均」に、存在する場合のmotion RMS／`motion_scale`とinterval RMS／`interval_scale`を加えたものである。座標の分母は`coordinate_count[j]`ではなくNであり、bit保存の段階ではこの実装上の選択も維持する。得点は `−distance / match_temperature − edit_penalty × (inserted + deleted) / N`。支持のない座標を平均値で埋めない。

残差の単位にも注意が要る。現在の`matcher.rs:533–541`はpitch変換後の座標差を既に`scales[j]`で割って二乗し、`memory.rs:300–302`がその値をそのまま渡す。その後に上の`acoustic_score`が再び`scales[j]`で割るため、座標項には二段階で尺度が適用される。第一段階ではこれも保存し、同時に「修正」しない。第二段階で生座標から尺度を一度だけ適用する案は、この点でも別モデルである。

その後、retentionのlog availability上下限に`logadd(v, ln(1e−300))`を適用して得点を加え、`no_memory_bias`を含むprefix／suffixのlogaddでweight上下限を作る（`retrieval.rs:177–237`）。順序とtie break、最大16件の切捨て質量も出力へ影響する。

**判定：この得点式だけでは検索を置き換えられない。** 入力残差はDTW path、coarse候補選択、pitch／tempo変換、挿入・削除、band edgeによる曖昧性から来る（`memory.rs:260–319`）。得点式にfit係数がないことと、入力が検索から独立していることは別である。

## 4. 分離案と数値影響

### 4.1 先に行う、動作を保存した責務分離

推奨する第一段階は、検索結果を受け取る処理と、受領後の保持・参照重み計算を分けることである。新しい汎用frameworkは作らず、現在のretrieval Group／Entryに相当するcrate内のplain structを境界とする。inventoryには必要な7群の参照証拠だけを渡し、私有traceには従来のContextを渡す。

この段階では旧検索producerをadapterとして残し、episodeの封緘順・ID・退役、query cadence・受領cut、残差、f64演算順、上下限、tie、16枠、retained ID列を一切変えない。型やファイルの所有だけを変える。この設計なら決定的な本体出力のbit一致を**目標にできる**。実測していないため一致成立とは記さない。また検索依存がadapterの内側に残るので、これだけでI4分離完了とはしない。

比較は二層に分ける。同一の保存観測列を境界へ与える検査では、bus・epoch・group世代・cutを鍵に、retrievalの支持／受領／評価時刻・ID順・全weightの浮動小数点bit、retained ID列、inventoryと私有traceの入力を全件比較する。通常実行ではWAVと登録学習recordの一致、実際に参照・信用が生じた件数を検査し、非同期候補recordは同一入力の共通件の内容と片側件数を分ける。実時間計器・非同期の出力行順だけを事前に列挙して除外し、意味を持つ受領cut・支持・世代・学習順序は除外しない。共通件だけの一致を全出力一致と呼ばず、0件照合や全参照消失も成功としない。

### 4.2 検索を外すための有界照合候補

第二段階の候補は、固定スパン・先頭からの経過sampleを対応づける有界参照照合である。名称を変えただけのDTWや、研究拡張への隠れた呼出しを残さない。以下は静的レビュー後の第一案であり、実装済み規則でも凍結済みの登録でもない。

- 群はbusごとに最大7、保持bankは各busの全群で共有する32枠、区間は96 hopとする案である。根拠は09-20の短すぎるepisodeで信用が消えた修正記録であり、一般的な最適値ではない（`milestones.md:232–239`）。96 hopは48 kHz／512 sampleなら1.024秒であり、秒数ではない。欠測で時間を詰めず、96 hopの物理区間を超えて伸長しない。全欠測の区間は保持しない。
- 比較入力はrawの十座標と支持から二hop幅の固定binを48個まで直接作る。96 hop ÷ 2 hop = 48であり、旧descriptorの128 knot上限は新規則へ持ち込まない。区間先頭を時刻0とし、完成したbinのみ使う。各座標はbinの全sampleが取得済みで、被覆するraw座標が全て既知の場合だけ、重なりsample数で加重した平均値を持つ。rawの元支持端・availabilityはcut以前を要求する。末尾の未完成binと欠測binを後の観測で補って過去queryへ戻さない。容量を超える設定は登録外として拒否し、適応圧縮・間引きで黙って収めない。
- この入力は既存の圧縮済み`matcher::Knot`から復元しない。`descriptor::Block`は取得時間の総量とmomentを保持するが、merge後の区間内の取得位置は復元できず、`Block::matching`は90%被覆で座標maskを立てる（`descriptor.rs:5–18,85–150`）。その平均を細かいbinへ配って「実際の共通支持」とする案は撤回する。第一段階では既存圧縮を保存し、第二段階ではこの有界入力へ置き換える。
- queryと候補は同じ相対bin番号だけを比較する。pitch shift、tempo倍率、DTW、挿入・削除による最適経路、全順列探索は使わない。比較窓長を`L = min(queryの経過sample数, 候補区間長)`とし、窓内の完成binだけ比較する。候補は同bus・同epochで、区間終端がquery開始以前、availabilityがcut以前の保持区間に限り、32枠をID昇順に全比較する。同じgroup世代を要求せず、現実装と同じく過去の別群も参照候補にできる。queryのgroup世代と候補episodeの世代は別々に検査する。
- 座標jについて共通既知bin集合を`B_j`、件数を`n_j`、共通支持sample数を`S_j`とする。第一案では`n_j >= 2`かつ`S_j / L >= 0.9`の座標だけを集合Jへ入れ、Jが空なら得点を作らない。未完成末尾もLの分母には残る。距離は`mean_j∈J(sqrt(sum_b∈B_j((q[b,j]-e[b,j])^2) / n_j) / scales[j])`、得点は`-distance / match_temperature`とする案である。生座標へ尺度を一度だけ適用し、未知を0差へ埋めない。binは昇順、座標は0〜9順にf64で集計する。
- 初回案の得点にはmotion・interval・編集項を含めない。既存motion／intervalはDTWで対応した隣接knotから作られており、尺度だけでは新規則を定義できない（`matcher.rs:568–608`）。固定bin同士のinterval比は常に1であるため、別の時間対応を導入せずに従来項を移せない。項を未知や観測済み0として報告するのではなく、新規則の式とschemaから区別する。必要性が示された場合は、方向差分等を別の技術案として登録してから追加する。
- coverage 0.9、最低2 bin・1座標は候補値であり、妥当性は未検証である。感度比較0.8／0.9／1.0と差分の判定項目を取得前に固定し、I10の望ましい結果に合う値を選ばない。得点温度・`scales`・`no_memory_bias`は初回比較では保存済みI10設定値を使う案とし、候補入力・支持・距離・棄却理由を残す。同じ数値を用いても旧モデルとの同等性は主張しない。
- 発行頻度は保存済みquery cadenceのsample換算値を初回案へ引き継ぎ、群ごとの最新の確定prefixを最大一件だけ凍結する。受領済み旧結果の失効確認、前cutで発行した結果の受領、新照合、区間封緘、weight更新の順と、終端イベントでの受領を別々に登録する。旧版の受領は`recall.rs:313–328,750–807`に従い、group退役判定が先である。新しいpending状態も最大7群分に固定する。ID・支持・発行／完了／受領cutを分け、次の受領機会で期限・世代・保持中IDを再確認する。同じsample cutの終端イベントを排除するためだけの架空の遅延は加えない。
- 0.5秒未満のquery鮮度、各群の公開16参照、全inventoryの16参照、下限信用、no-memoryへの未割当、anchor・period規則、保持IDによる退役を維持する。検索schedulerの優先順位がなくなり複数群を同じcutで照合できるため、公開時刻が旧版と同じとは仮定しない。各busの最大7群×32候補×48 bin×10座標＝107,520座標比較に加え、入力形成・retention・公開の費用と保持量をR2へ渡す。この上限は実時間予算を満たす証拠ではない。

この候補は対応pathと候補集合を変えるため、`retrieval.weight`、inventory、私有trace、I10の学習recordのbit一致を一般には保証できない。上記第二段階を「完全にbitを保った最適化」として登録してはならない。既定Noneでの不変と、memory明示時の意図したモデル変更を分ける。

**判断I4-1（方針選択済み）：第一段階でbit保存を検査し、その後に第二段階を新しい規則として登録する。** 旧coarse／DTWを本体の依存として採用し直す代案は選ばなかった。旧検索のままファイルだけ分けて、探索分離済みとは呼ばない。第二段階の具体的な照合則・数値は別途レビューと凍結を要する。

### 4.3 第二段階の差分許容と不成立条件の案

以下を次の登録レビューへ渡す技術契約とする案である。新旧で同じになる欄と、モデル変更により変わる欄を分け、旧版への近さだけを新モデルの受入条件としない。数値許容も未凍結である。

- **必ず保存する契約。** bus・epoch・queryの群世代・episode世代を混同しない。queryについて`支持終端 <= availability <= 発行 <= 完了 <= 受領 <= 期限`を要求し、候補の元支持・availability、保持中ID、query前の候補終端も検査して、未来情報や退役結果を使用しない。群退役は受領より先に処理し、後から得た参照で発行済みtraceを書き換えない。既知0と未知、下限信用と上限、未割当と理由別損失を区別する。上記の上限を超える状態を作らず、同じ入力・版では同じID順・tie解決・結果を得る。`None`と固定方策の観測非干渉については§5のWAV／学習record条件を満たす。
- **意図する変化。** 比較入力、支持座標集合、候補の距離・順位・weight、群ごとの結果発行数／公開cut、inventory、私有traceとその後の学習、参照を消費する候補費用・生成選択は変わり得る。長い欠測で区間が96 hopを超えた旧処理から固定区間へ変える場合、封緘境界とepisode IDの対応も変わり得るため、元支持区間を併記して比較する。該当欄の新旧一致は要求しないが、入力から初めて分岐するcutと理由を、入力支持・距離・正規化・受領・消費の順に追えることを要求する。query終端と受領cutを削除して一致させない。旧schemaの0埋めや、全記憶を無効にした対照を新規則の成功にしない。
- **比較前に棄却する入力。** 不一致のbus／epoch／世代、元支持またはavailabilityの未来超過、候補区間終端のquery開始超過、非有限値、非正の尺度・温度、上限外設定、重複または逆転したID／時刻は契約違反として拒否する。`L <= 0`、完成binが2未満、全座標の共通支持不足は通常の未支持であり、得点を作らない。この二種を別理由で記録し、未支持を距離0としない。coverage 0.9の境界は`10*S_j >= 9*L`という整数sample数で判定する案とし、丸め許容で不足を通さない。
- **数値比較の案。** 整数、ID、mask、採否、tie順、支持区間は完全一致とする。同じ実装の同じ入力の再実行と第一段階は浮動小数点bitも一致とする。別実装の独立数値参照に対してのみ、有限な距離・得点・weightに`abs(actual-reference) <= 1e-12 * max(1, abs(reference))`を用いる案とする。採否の分岐や世代・時刻にこの許容を流用しない。各weightは0〜1で下限が上限以下、inventoryの採用下限和は既存実装と同じ`1 + 1e-12`以下とし、未割当はその補数から得る。生じた違反を後処理のclipで隠さない。
- **不成立として戻す条件。** 上の契約または独立数値参照が落ちる、旧有効条件を全部未支持にして有効経路を検査できない、期待した参照付きinventory→信用更新→同じVoiceの後続利用が消える、固定方策の観測非干渉／`None`回帰が崩れる場合は技術完了としない。I10の12条件には参加決定のないholdも含まれるため、全12条件への正の信用を一律には要求しない。取得前に保存済みI10の入力・期待する有効経路・空でよい対照を条件ごとに対応づけ、その既存要件を維持する。対応できない条件は確認未完であり、合格母数から黙って除外しない。R2未測定を、この段階の性能合格にも置き換えない。

### 4.4 文書上の最小計算例と固定する自由度

次の例は式と境界を検算するための人工入力である。実音の取得、コードの検査結果、正式fixtureではない。

- **距離と尺度。** 一座標、二つの完全既知binでquery値`[0, 2]`、候補値`[0, 0]`、尺度2、温度1とする。`n=2`、coverage=1、距離は`sqrt((0^2+2^2)/2)/2 = 0.7071067811865476`、得点はその負値となる。旧DTW残差が同じ対応で作られたときは先に尺度が適用され、既存得点式の座標距離は0.3535533905932738となる。新旧の差は許容誤差ではなく、登録する尺度則の変更である。全座標が未知なら、この0や負値のいずれも出さない。
- **共通支持。** 二hop binが10個あり、queryはbin 0、候補はbin 1だけ未知とする。各側は90%既知だが共通は8 binで、`S/L = 8/10 = 0.8`となり0.9案では未支持である。同じbin 0が未知なら共通9 binで0.9となり境界を通る。4 hop時点の完全既知2 binはcoverage=1、5 hop時点では未完成末尾も分母に残りcoverage=4/5となる。後者を失敗と記録せず、登録した未支持として扱う。
- **受領と期限。** 48 kHz、hop 512の例でquery支持終端・発行・完了をsample 4096、期限を例示用に8896とする。pending結果を次cut 4608で受け取る際、所有群とepisodeがまだ同じ世代で保持されていれば受理できる。4608で群を退役させた場合は先に無効化する。期限8896ちょうどは期限条件を満たすが、9216なら拒否する。EOFイベントが4096で来た場合は、既に完了したpendingを同じcutで受け取れる。いずれも、受領前のVoice発行へ結果を遡及しない。別に、query支持終端から24,000 sample経過した時点は、期限内であっても0.5秒未満という鮮度条件を満たさない。

初回案では32保持枠、96 hop区間、2 hop/bin、48 bin容量を一組のモデル定数として登録し、独立した容量調整キーを追加しない。7群・各群16公開件・inventory16件は既存の境界として維持する。尺度10個、温度、no-memory bias、保持係数、発行cadenceと期限は比較入力manifestに明示し、保存済みI10値をそのまま初回値に用いる。coverageの0.8／0.9／1.0だけを事前登録する感度比較とし、motion・interval・編集項の係数を新設定に残さない。正式な実装名は後で決めるが、旧設定キーを同名のまま別の意味へ読み替えない。初回案の定数変更は同じ取得内の調整ではなく、規則版と比較登録の変更として扱う。

## 5. I10の既存証拠をどう扱うか

I10の限定完了はその版・固定development素材に対する過去の判断として保存する。新参照規則の成功へ自動で継承しない。第一段階で同じ入力にbit一致すれば、その一致範囲を新版へ引き継ぐ。第二段階に差が出た場合は次を再取得・再照合する計画とする。

- `temporal_observation.memory.retrieval`と`reference_inventory`：元支持・出所・上下限・16件の切捨て・未割当を比較し、差が初めて出るcutを記録する。
- `private_participation_trace`、発行時の参照凍結、onset／releaseの信用更新、同じVoiceの後続予測、低支持退避、presentation-only非学習、epoch／世代／退役・欠測境界を再確認する。
- [I10のrelease・任意身体・両bus検査](i10-body-outcome.md#release任意身体両busの検査と縮小した負荷の資源引渡し2026-09-21)の12条件と学習record比較を、旧入力・seed・設定・除外規則を保存したまま再取得する。候補recordは非同期の片側件数と共通内容を分ける。query cadence変更の影響を時刻欄の除外で隠さない。
- 候補energyの数値核や自声descriptorの数値参照は、入力も演算も変わらない範囲のみ継承可能である。shared referenceを入力とするtrace／候補費用は別に照合する。新規則がI11／I12bのbase・選択まで変えた場合は、その生成経路の回帰・因果比較も再登録する。
- `None`のWAVと登録学習recordは従来とbit一致を必須案とする。明示memory時も固定方策の観測非干渉WAVは必須だが、生成へ参照を消費する設定のWAVは不変と仮定しない。参照を空にして音だけ一致させた結果は有効機能の不変ではない。

## 6. 分離後の配置・設定・report

本体には`core/temporal_expectation.rs`、`history_prediction.rs`、descriptorのうち有界観測に必要な部分、受領時計・retention、参照証拠のweight計算、`reference_inventory.rs`、`private_trace.rs`を残す案とする。既存`recall/{clock,retention,retrieval}.rs`の移設は責務境界に従う。新ファイル名は実装時に決め、不要なtraitやwrapperは追加しない。`memory::Residuals`のように研究検索の型に置かれた共通値は本体側のplain structへ移す。

移設監査には実行時の呼出しだけでなく、型とtestの依存を含める。`descriptor.rs`は`matcher::Knot`と`memory::Descriptor`を返し、`transport.rs:74,154`も`memory`の容量定数を読む。`descriptor/tests.rs`にはDTW／ordered検索のtestがあり、inventory fixtureも`recall::retrieval`に依存する。第一段階では値を変えず境界を移し、最終段階では本体の観測・保持・受領に必要な型／上限だけを本体へ残し、検索exportと検索専用testを研究側へ移す。`Residuals`全体を将来の本体へ恒久的に残すことまでは要求しない。通常build・test buildの双方で、残る本体から研究moduleへのimportがないことを確認する。

研究側は`matcher.rs`、ordered／orderless探索、scheduler、研究検索への固定span prefix query発行と検索診断adapterである。§4.2の有界入力で現在までのprefixを使うこと自体は禁止しない。研究側を本体型のproducerにできても、本体からそのproducerを必須にしない。

**判断I4-2（方針選択済み）：第一段階の比較を済ませ、I4-1の本体代替規則を検証した後に研究branchへ保存してmainから除く。** mainに既定off診断として残す代案は選ばなかった。09-20のI7／I8削除と同じく、本体が読まない大きなtest buildを抱えない利点がある。ただし既存の研究branchに後日の修正がすべて保存済みとは未確認であり、削除前にsource・fixture・設定・数値参照・再現手順の保存を検証する。実際の移管・削除は本草案の依頼範囲外である。

### 6.1 研究保存の読み取り確認と残る保存対象

2026-09-24、`git log --all`と`git show`で確認した範囲では、milestonesに記された`research/temporal-dcc-extension` branchと`research/temporal-dcc-extension-20260920` tagは、このcheckoutのローカルrefとして解決できなかった。remoteを取得していないため、別の保存先に存在しないとは判断しない。

一方、縮小前checkpoint `82573d1c927fa8d24cf2dca4682bd43cdb168de8`はローカルから読める。ただしその`acoustic_score`はfit係数を読む旧版である。以後の`ff1c685b8f0a51e845844610c31478cbae1ecfa5`で明示距離則、`975b108f097f88eb3353c446707ac5241609df4f`でphrase等の削除、`db1eae4cf01fc3ed9511f4a3cd9f6653a6cb35f4`で取得時計の修正が入っている。checkpointに到達できることだけで現在の研究producerが保存済みとは呼ばない。

削除前には、I12b後に固定する基準commitを起点に、(1) matcher／memory・index／query・cache／recallとそのtest、(2) descriptor／transport／clock／retention／retrievalと本体側契約の対応、(3)設定・schemaと生成物、(4)検索用Python／Rust数値参照・fixture・検査手順、(5)比較入力manifest・binary識別・結果を、復元可能な一組として確認する。共用のclock／retention等を本体から削除する指示ではない。研究側に必要な版を保存し、残る本体との所有を区別する。

本調査のHEAD `b357e61f0136bc3856c96dc17e330c2377cd0c34`に対し、上の主要検索source・その子module・検索参照script・`tests/fixtures/temporal_cognition/`には作業ツリー差分がなかった。一方、`src/config.rs`、`tests/test_evaluate_temporal_matcher_kernel.py`、I3／I4記録には未commit差分がある。matcher検査にはDarwinのlink指定とlibm依存の前提修正が含まれる。未commitの観測配線やruntime等も含め、HEADだけで現在の実行環境を再現できるとは扱わない。保存先refの確認、現在版の保存、別環境での復元検査は未実施であり、移管開始前の条件として残す。

I10記録の取得plan `target/i10-reduced-resources-20260921/plan.json`は、指定された親ディレクトリをこのcheckoutで確認できなかった。ただし追跡済み[I11登録入力](i11-inputs/README.md)を追加調査し、I10からバイト複写した12 scenarioと`config-none.toml`を回復確認した。13件全てが`i11-inputs/plan.json`の登録SHAと一致し、None設定のSHA `b9b93fb3e1cadfbd85bde87ea9223f23d4f795393b61ed874c5b1384e4f920f0`は同planのI10元設定SHAとも一致した。`register.py:70–90`に複写元とvariant生成規則が残る。

設定が要求する身体profileも、`target/i11-same-record-mac-20260923/action-profiles.bin`が登録SHA `033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5`と一致した。元設定にはLinuxの絶対パスが残るため、後続取得では原本を維持して実際のprofile所在との対応をmanifestへ明記する。これによりI4比較の入力作成を、元入力が全て失われたという理由で止める必要はない。

一方、旧I10の取得plan・実行script・report／WAV・条件別の参照信用結果・正確な除外scriptは回復していない。I10正本に残る12条件、学習record10種、候補on／off対照の判定等は新しい比較登録の要件として使えるが、後続I11の除外規則を旧I10のものと断定しない。入力の同一性、既存記録の引用、新しいI4比較、旧取得そのものの再現は区別する。今回確認した範囲と実装入口は[入力回復記録](../../../target/r2-i4-parallel-20260924/round2/frontier-i4.md)に記す。

### 6.2 設定とreportの扱い

設定は現`[temporal_memory]`の`retention`、`candidates`、`scales`、`span_hops`、`episodes`、`query_cadence_ms`、`deadline_ms`を監査対象とする（`src/config.rs:422–453`）。第一段階ではキーの意味とserialize形を保つ。最終分離では、本体設定と研究設定を区別する案を推奨する。`candidates`やcadenceを本体規則へ転用して同名のまま意味を変えない。新名称・値は未凍結である。

全configの`deny_unknown_fields`に従い、削除した研究キーは明示的エラーとし、黙って無視したり旧検索へfallbackしたりしない。alphaでは互換aliasを既定の義務にしない。保存済み実験設定は旧binary用の証拠として残し、新設定は派生manifestで対応づける。

reportは第一段階の値・キー・型を保存する。最終分離で旧`query_id`／検索件数等が意味を失うならschema／モデル版を更新し、全件0で「検索が成功した」形を残さない。inventoryの参照元ID・元支持・発行／受領cutに何が対応するかを新契約に列挙する。UIとreportは同じsnapshotから読み、生成の既定Noneは維持する。

## 7. 実装順・衝突・検査計画

1. 選択済みのI4-1／I4-2方針に基づいて詳細を詰め、独立レビューを受ける。I12bのcommit完了と測定プロセス停止を確認する。別worktreeでもCPUは共有するため、I12b取得中にbuildしない。
2. commitされた共通基準を別worktreeへ固定し、source／binary／設定／入力／report版を保存する。未commitのI12b変更をcopyして「基準commit」と呼ばない。
3. inventoryの狭い入力型とretained ID所有を分け、旧producer adapterで第一段階のbit回帰を行う。境界を変えた時点で演算順・時刻・ID順・tie・容量・退役を確認する。
4. 第二段階の本体照合規則・閾値・差分許容を別登録する。新旧を同じ観測列で比較し、差の理由と最初の分岐を記録する。旧producerは検証用対照であり、通常本体のfallbackにしない。
5. I4-2に従って研究コードとfixtureを配置し直す。mainに残るRustを参照するPython corpusは維持し、削除したRust専用の参照は研究側へ閉じる。import graphを確認してから整理する。
6. 上記I10再検証、`None`回帰、設定拒否・TOML往復、schema・UI／report対応を行い、R2へ最終構成の処理目録を渡す。技術完了とR2／A3受入を分ける。

調査時の`git status --porcelain`で、将来の分離と重なる未commit変更は少なくとも`src/temporal_cognition/observation.rs`、`src/runtime/mod.rs`、`src/temporal_cognition/tests.rs`にある。さらに`src/config.rs`、`src/temporal_cognition.rs`、`src/life/action_observation.rs`、`src/life/phonation_engine.rs`、`src/life/temporal_participation.rs`、`src/runtime_profile.rs`、`tests/render_binary.rs`、`Cargo.toml`にも変更がある。これらは今回編集していない。`recall.rs`／`reference_inventory.rs`が現在cleanでも、同じ型を参照する変更の意味的競合は残る。作業開始直前にstatusと対象diffを再確認する。

検査の配置は次の案とする。今回は一つも実行しない。

- module内部：retention／参照照合のinlineまたは既存内部testで、未知・既知0・空候補、同距離tie、尺度、不一致世代、future／stale、0.5秒境界、capacity、欠測復帰を検査する。
- 第二段階の数値参照では、生座標への尺度一回適用、欠測の位置が異なる対、二hop境界・未完成末尾・長い欠測、部分支持が別座標へ分散した場合、退役直前のpending結果、同一cutの複数群を確認する。検索の全件0化や無音化だけで有効経路を通過したことにしない。
- 既存inventory test（`src/temporal_cognition/reference_inventory/tests.rs:103,153,185,210,252`）で、2秒物理支持、二anchor、退役の非移譲、検索損失と16件切捨て、periodの未来支持拒否を維持する。private traceのPython数値参照と既存Rust testは、入力／式が不変なら継承する。
- 公開／複数module：`tests/render_binary.rs`の両bus観測非干渉、inventory→私有traceの実到達、発行時凍結と後続予測を検査する。旧memory有効／新本体参照有効の差を未使用経路の0件で通さない。
- `tests/log2space_scan_invariants.rs`は音響scan境界を壊していないことを確認する既存検査として全体検証に含める。今回の分離がscan境界を変えなければ、同じ実装を写した新規testは増やさない。
- 実装時にはAGENTS.md所定の全cargo test・fmt・clippy等と同一shellの終了statusを保存する。草案段階ではcargoを実行しない。

<a id="r2-handoff-order"></a>

### 7.1 R2が待つ段階と、footprintの版

[R2 §4.2](r2-preflight.md#i4-r2-order)の前回判断4も、I4-1（前回2）・I4-2（前回3）と合わせて推奨方針を選択した。次の第一分岐を採用方針とし、後二分岐は選ばなかった代案として残す。実行開始や登録値の凍結を意味しない。

- **R2の主判定を第二段階後に置く方針（選択済み）**：I12b終結・基準版固定 → 第一段階のbit回帰 → 第二段階の新規則登録・実装 → 研究検索の配置と§5のI10／影響する生成経路の再検証 → 最終処理目録・計器検証・R2登録 → R2 → A3の最終受入・A4監査。R2はI10再検証まで待つ。I4実装にR2合格を先に要求する循環は作らない。
- **第一段階後のR2先行検査を選ぶ案**：I12b終結・基準固定 → 第一段階のbit回帰 → 旧producerを含む暫定版のR2登録・取得 → 第二段階・研究検索の配置・I10再検証 → 最終構成のR2を別登録する。第一段階だけで足りるのは暫定版の検査である。旧検索費用を計上し、I4分離完了や第二段階の資源受入とは呼ばない。出力のbit一致から処理時間の一致は導けない。
- **分離前のR2を選ぶ案**：I12b終結後の現構成を固定して現存検索も数える。I4の二段階とI10再検証後、新版のR2を別登録する。これは「最初の運用範囲をI4分離後にする」案を採らない場合である。

上の矢印はI4からR2への資源検査経路であり、A4までの全要件を列挙したものではない。A1／A2、R3、I13および本体に適用するO項目等は、[milestones](milestones.md)の現行範囲に従って別途監査する。

I4-1の代案で旧検索を本体依存として採用し直すなら、既存の本体範囲と§8の完了条件を先に改訂する必要がある。単なる責務分離を最終分離と読み替えない。I4-2で研究検索をmainに残す場合も、通常本体から依存を切ることと、研究診断を同時に起動して費用を数えるかを分ける。設定・schema・fixtureの扱いとI10証拠の継承範囲は、この二判断に従って決まる。

作者の確定判断(b)は[設計台帳](../../design-notes/technote-ledger.ja.md#body-beat-time-constants)に従う。どの分岐でも今回のR2へ渡すfootprintは、I11-1の `footprint = "body"`、合成器由来の代表予測である。学習版への置き換えは現在のA4に含めず、A4の後の次の単位とする。その時点で[I11-1のA1](a1-i11-audition.md)と[R2のfootprint関連負荷](r2-preflight.md#41-測定対象版に固定するfootprintの出所)を新しく登録して再実施する。I4の完了に学習版を追加要求しない。一方、I4第二段階が参照・候補費用・生成音を変えた場合は、footprintの出所が同じでもI10・I11／I12b・A1の影響範囲を再照合する。

## 8. I4残作業の完了条件

本体の保持・参照生成・私有traceへの入力が、研究用coarse／DTW・ordered／orderless探索・query scheduler・研究検索へのprefix query producerを呼ばずに、登録した有界かつ因果的な規則で成立することを完了条件とする。保持IDと世代・支持・受領時計・上下限・anchor・質量保存・失効を維持し、第一段階のbit回帰と、第二段階の登録済み差分および影響するI10／生成経路の再検証を通す。既定Noneの音声・学習record不変、設定／report版の意味、研究実装と証拠の保存、最終処理目録のR2への引渡しを確認して初めてI4分離を技術完了とする。作者採用・R2・A3・A4全体の完了は含めない。

## 9. 判断点

- 方針選択済み：I4-1はbit保存の責務分離の後に本体照合を新規則として登録する二段階案（前回2）。I4-2は代替producer検証・保存確認後の研究branch移管（前回3）。R2は第二段階・I10再検証後（前回4）。いずれも技術的成立の判定や実装開始ではない。
- 別の判断で決まる：新照合規則を採るなら、明示memory時のI10証拠の再検証とreport／設定の意味の更新が必要になる。研究コードをmainから除くなら、それに専用の設定・test・参照の移管も必要になる。
- 技術レビューで詰める：§4.2の固定bin入力・距離式・容量・受領順・coverage感度・差分許容は未凍結の第一案である。現在の依頼では実装／取得へ進まない。名前、数値参照、境界検査、保存対象の列挙は技術作業として進められ、項目ごとに作者へ選択を求めない。現在の有界因果参照という本体範囲を変える、支持要件を緩めて不成立を受け入れる、登録後の許容を結果に合わせて変える必要が生じた場合は、根拠と代案をまとめて作者判断へ戻す。
