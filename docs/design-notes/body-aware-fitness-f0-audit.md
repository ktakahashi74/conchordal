# 身体スペクトル適応度 F0 現在地監査

日付: 2026-09-26。対象: blanche の `/home/shafi/lwrk/conchordal`、commit `06a4772c43d06b41b44753bb0be891f23e16b93e`。取得前監査から I11-1 の登録36本と判定まで追記した記録であり、I11 全体・I12b の完了判定ではない。

## 確認できた状態

- I11-1 は、同一代表 record を `body` と `proxy` へ配送する修正 `7f76e0b` を含む。詳細記録末尾は、修正後の登録12条件、`None` の §5.4a／b、§6 の再判定を未実施とする。後続文書の「限定技術完了」は、この未取得分の証拠に代わらない。
- I11-1 の再取得入力と判定手順は `docs/roadmap/temporal-dcc/i11-inputs/README.md` にある。登録 profile `target/i10-action-profiles-20260918.bin` の SHA-256 は `033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5`。基準 `target/i11-stage1-54b-20260923/baseline/render` の SHA-256 は `3614d998e07c04c7af1913f97f515cb0edf0be4e2aec6b18331703e67fa58b5e`。基準12条件の WAV と report は全件存在する。取得時には登録入力全件の hash を再検査する。
- I11-2 は `i11-onset-comparison.md` §4.5 で、試験用 Hazard 18係数と hash、共有 snapshot と対応表の参照形を実装前の追記事項として残す。同記録は第2段を未着手とする。現 source の `arrival_weight` は設定・検証に存在するが、これだけを到来費用の実消費や I11-2 技術完了の証拠としない。
- I12b は `milestones.md` の2026-09-24引継ぎで取得・技術完了が未判定。現 checkout、既知の別 worktree、`target/`、`/tmp` で専用結果 manifest と保存記録を確認できなかった。192 run の取得手順と封印された実行版は現 checkout にない。別機器の未収録成果が存在しないとは断定しない。
- I4 の二段階分離は草案で、実装・凍結前。第一段階は既存出力の bit 保存を目標とし、第二段階は研究検索の代替規則を要する。共有 runtime を変更する F3 はこの配線と直列化する。

## 取得条件と未解決点

I11-1 は新しい `target/i11-same-record-<日時>/` に、登録12条件 × `body`／`proxy`／`none` の offline render 36本を取得する。変更後 release binary の source と binary SHA、36本の実行台帳を保存する。§5.5(a) の同一 record・D・16 bin・外部入力と最初の選択差、§5.4a の WAV・学習記録一致、§5.4b の共通候補 record 一致、24 report の独立参照を README の検査器で判定する。片側だけの record と無分岐条件は件数を残し、除外して合格としない。既存の §5.7 と §5.9 の限定的な結果は、新取得で自動的に更新されたとは扱わない。

取得時点の関連ジョブ、固定 source、登録入力の hash、基準物の存在を確認する。`acquire.py` は各 run 後に `runs.json` を更新するが、同じ出力先への安全な再開機能はない。中断・失敗時は台帳と process を確認し、終了を推定して自動再起動しない。旧取得先を上書きしない。

F1 の自己除去・身体スペクトル参照仕様と、F2 の静的評価契約は I12b と独立に整理できる。現在の blanche では I12b の稼働 source・取得器・登録と関連ジョブを確認できず、「取得中の終了待ち」は現状に適用できない。2026-09-26 の作業判断として、`06a4772` を基準に隔離 worktree で F1/F2 の数値コード編集を進め、I11-1 の36本取得・判定が終わるまで同じ機器での追加 build・実験を行わない。I12b の実装・登録・取得は独立した未完事項として残し、完了扱いにしない。I4 の変更、F3 の通常 runtime 接続、R2／A1／A4 はそれぞれ別の証拠と判定を要する。

## I11-1 再取得と判定（2026-09-26）

取得前に関連 job が無いこと、`src/` に差分がないこと、登録15入力の hash、profile と基準 binary の hash、基準12条件の WAV・report 全件を確認した。`06a4772` の release binary を構築し、`target/i11-same-record-20260926-193414/` に36本を取得した。`runs.json` は36本すべて exit 0。source commit、各 Rust source の hash、binary の hash は同じ出力先に保存した。

登録手順の結果、36本の WAV・report が存在し、§5.5(a) は harmonic／modal の6条件で選択差と WAV 差、sine の6条件で無分岐となった。最初の分岐以前の非 power 入力は12条件すべて一致。§5.4a は基準12条件の WAV と10種の学習 record が一致。§5.4b は共通候補 record 11,382件で不一致0件。片側だけの record は別集計のまま残した。

独立参照の初回実行は `verification-status.txt` に exit 1 として保存した。`verify-stage1.json` の失敗9件は、すべて `proxy` 設定で古い record の識別が残ったときに、検査器が `proxy(stale)` を要求したためだった。修正後の仕様では `proxy` 設定は同じ record の時間幅を使うが power は代理値とし、出所を `proxy(setting)` と記録する。`scripts/verify_i11_stage1.py` のこの条件だけを修正し、元の JSON は残したまま `verify-stage1-corrected.json` へ再検査した。24 report の全検査に失敗0件。`sine-hold` の6 report は登録どおり決定0件で、検査器の終了コード1を許容した。検査器修正後の SHA-256 は `3e930b2f44b74e70a8a0ae17f8691147c2f74ec86bcc1f83de9bd3c3c4ae04ac`。

以上は I11-1 の登録36本と明示した判定の証拠。I11-2、I12b、総 hop 資源、作者試聴、R2／A1／A4 の合格へ転用しない。I11-1 第1段の技術完了を宣言する場合は、既存 §5.7・§5.9 と今回の結果、検査器の事後修正を合わせて再レビューする。

### 検査器修正の回帰

`tests/test_verify_i11_stage1.py` に、旧識別の代表 record から時間幅だけを受け取る `proxy(setting)` の例と、旧識別の body power を使おうとする違反例を追加した。現検査器の3 test は通過した。前者は全検査が通り、後者は `5.3 body identity is current` と `5.3 stale identity falls back` の双方で失敗する。`git show HEAD:scripts/verify_i11_stage1.py` の修正前本文を一時 module として前者へ適用すると、`5.3 stale identity falls back` だけが1件失敗する。この再現は既存の取得結果を変更しない。

### §5.7・§5.9 の版と終了条件

| 条件 | 既存の成立証拠 | 現版との境界 |
|---|---|---|
| §5.7 hop経路と飽和 | `target/i11-stage1-rerun4-20260923/runs-hop4.json` は96実行すべて exit 0。`own-us.json` は登録後の A/A 3 pass floor と待ちを除く `own_us` により16組中16組合格。生の `elapsed_us` は13組合格、3組不合格。返却遅延・置換済み・drop と待ちの分布も詳細記録にある | 測定 binary は `029795c`。その後 `1a26d95` の返却 drop 解除・再送、`7f76e0b` の proxy 側 record 要求が入った。36本のoffline renderでは更新できなかったため、下記の現版専有測定を別途行った。現版の登録組合せは14/16合格であり、旧版の16/16を継承しない。総 hop 負荷と全 worker 資源は元から R2 の別課題 |
| §5.9 代表条件と実条件の差 | `target/i11-gap-20260923/summary.json` は登録12条件で3,390 onset を投影。支持なし bin 0、hold の決定なし84。`body` 3,342件と代理48件を別母集団で、`power_gap`、`overlap_gap`、`term_gap` の中央値・p95・最大を報告済み。ここは分布報告であり合否条件はない | 取得は `029795c` の ignored test。後続変更は body の代表スペクトル計算を直接変えていないが、`06a4772` で同じ投影は再取得していない。今回の36本の WAV・report は発音時に凍結した実 `ToneEnergy` の独立再投影を含まない。新版の分布として転記しない |

この表は36本の取得直後に整理した旧版継承の限界を示す。その後、§5.9 は現版で再投影し、§5.7 は下記の専有測定で再取得した。I11-1 全体の現版技術完了は、§5.7 の直接処理欄が2組不合格のため宣言しない。36本の成功や旧版の合格で代替しない。

### I11-2・I12b の実装前に不足する登録

I11-2 では `i11-onset-comparison.md` §4.5 に従い、試験用 Hazard の18係数・標準化・horizon の値と hash、20 ms／201点 CDF の最終格子と積分・補間規則、共有 snapshot と `groups[prototype]` 対応の正確な欄、bus・群・身体世代・モデル版・時刻の照合を固定する必要がある。`arrival = true` の有効入力、既知0と不明、期限切れ、自己群除外、到来項だけを切る対照、skip に項を入れない規則、独立数値参照、作用差の分母・判定も実装・取得前に具体化する。現行の設定欄 `arrival_weight` の存在だけでは、この経路を成立と判定できない。

I12b では本体に残す二経路、すなわち到来期待から gesture 候補時刻への作用と、観測 accent から周期推定への作用を別々に切る位置・状態保持・期待する変化を登録する。前者の更新則・閾値・尺度は I11 の実装前登録で確定する条件。取得には source／binary／検査器、シナリオ・seed・body／Hazard設定・route・入力の hash、実行順と測定区間、各経路の非ゼロ消費の適格条件、対の比較指標と事前の合否規則、元失敗の扱いを固定した manifest が要る。`r2-preflight.md` の192 run、非退行120欄、符号付き資源差240欄は受け取り欄で、現 checkout に実行済み結果や封印された取得仕様を示すものではない。T2 地形を入れた版では I11-2／I12b の比較を再登録・再取得する工程も別に残る。

### I11-2 実装前登録の草案（履歴）

以下は `i11-onset-comparison.md` §4.5、現行の `arrival::Engine`、`observation::Snapshot` と `action_profiles::consumer` を突き合わせたもの。試験用 Hazard 入力と独立参照は [`i11-stage2-test-registration.md`](../roadmap/temporal-dcc/i11-stage2-test-registration.md) に別登録した。通常既定の作者採用、Rust 実装、結果判定は行っていない。

| 項目 | 現行 source・記録から固定できる内容 | 実装前に残る判断 |
|---|---|---|
| Hazard 入力 | `TemporalPeriodConfig` は `model`、18係数、8個ずつの `means`／`deviations`、`horizon_sec` を明示要求。`arrival::Engine` は `softplus(c・x)` を毎秒 hazard とする。`x[0]=1`、`x[10]` は周期位相の cos（標準化後）、`x[11]` は sin。I6 の定数2/秒、`c[0]=log(expm1(2))`・残り0、平均0・偏差1・地平0.1秒は数値対照で、I11-2 の作用試験値ではない | I13 記録の未適合診断値 `c[0]=-2, c[10]=4`、残り0を試験用に固定した。`model="hazard"`、平均8個0、偏差8個1、`horizon_sec=4.0`、48 kHz。入力 TOML 全バイトの SHA-256 は `a7993ce065a52e156a19ef82c97559842184674c3562bdd4b65d6c7e06e12485`。係数の適合済み尺度、通常既定値、作者採用とは呼ばない |
| CDF 数値 | §4.5 は `issued_at` から20 ms刻み、0〜4.0秒の201点、線形補間、候補窓 `at±width`、群の一様平均を規定する。48 kHz なら格子間隔960 sample、地平192,000 sample。現在の Hazard 積分は2点 Gauss–Legendre、1／2／4／8／16分割で最大62評価、収束許容 `1e-8+1e-6×|積分|`。位相係数が非ゼロなら各分割幅を周期の4分の1以下に制限する | 技術案: 同じ `Context` と既知の経過時間を用い、各20 ms 区間の積分を足して累積量 `H_k`、`F_k=-expm1(-H_k)` とする。4秒を一度に積分すると最短周期0.125秒で16分割幅0.25秒となり既存上限に反する。20 ms は0.125/4秒より短い。1区間でも未解決なら群の表全体を不明とし、補間・差分前に `0≤F_k≤1` と単調性を検査する。200区間×最大62評価×最大7群／bus の費用は R2 事前測定を要する。格子の sample 丸めと48 kHz以外の扱いも封印する |
| 供給と自己群除外 | `observation::Snapshot` は bus、source epoch、sample rate、`period`、`group_prototypes`、`action_profile_features` を保持。`period.groups[0..7]` に群 handle と forecast、`group_prototypes.assignments[0..8]` に prototype→群の descriptor 対応がある。`action_profile_features.groups[prototype]` にも群 handle があるが、profile と gesture が無いと snapshot 自体が作られない。`consumer::Binding` は声の source id／generation、body generation、bus、end／available、body model version、prototype を持つ | CDF は共有観測 snapshot に bus・epoch・group handle・`issued_at`・`horizon_end`・`periodic`・201値を群別に加える案。自己群判定は Binding の所有者／身体世代／bus／model版を照合し、`group_prototypes.assignments[b.prototype].key=(群epoch,群generation)` と同じ bus から handle を復元する案。action profile snapshot がある場合、その `groups[b.prototype]` とも一致検査する。欠けていても除外判定を依存させない。対応表の `end_sample` と Binding の `end` の既存24,000 sample鮮度規則を援用するか、I11-2 用の期限を別定義するか固定する。descriptor 関連付けに基づく除外であり、完全な自己音検出とは呼ばない |
| 消費と不明 | §4.5 は `issued_at≤now<horizon_end`、`periodic=true`、habitat bus の群集合を全候補共通に固定。`elapsed_seconds[0]≠[1]`、`reset_unknown`、積分失敗は表なし。群集合が空なら到来項なし。非空なら `P_g(at)=F_g(min(at+w,end))-F_g(max(at-w,issued_at))`、`P` は群平均、費用へ `coupling×arrival_weight×(1-P)` を加え、skip は到来項抜きで判定。設定は arrival既定false、weight既定1、許容0〜4 | `Forecast::valid_for` は現 source では `cut≤horizon_end` だが §4.5 の群採用は厳密な `now<horizon_end` とする。`at+w` が地平外の候補が一つでもある群は集合から除外し、理由を報告。既知の `P=0` と表なしを別記録。`arrival=true`／到来項だけを切る対照、独立数値参照、分母・作用判定、候補集合の固定位置を登録してから実装する |

特に `c[0]=-2,c[10]=4` は I13 に書かれた未適合診断値の試験用転記であり、現行 I11-1 入力（全係数0、地平1秒、arrival無効）からの推定ではない。I13 原文は「公開候補はAPI決定でも既定値変更でもない」「`arrival_weight` は実消費者の作者規則候補」「T2変調による `P` の尺度・解釈を確認してから恒久操作を決める」と区別する。したがって作者判断が必要なのは恒久操作・既定採用・可聴差の受入であり、隔離した試験用数値の固定と独立数値検査には要求しない。共有対応表の期限、48 kHz以外の時計、実装対参照の誤差許容、作用差の母集団は技術登録として取得前に解決する。R2 の資源判定も別に残る。

### I11-2 の隔離実装と局所取得（2026-09-26、上の草案を更新）

`.worktrees/i11-stage2` にだけ数値CDF、共有snapshot、Voiceの到来費用を実装した。Hazard試験係数、48 kHz／20 ms／201点、独立参照の誤差許容、24,000 sample鮮度、全候補共通の群集合、自己群対応の因果条件、skip判定に使う最小基底費用を [`i11-stage2-test-registration.md`](../roadmap/temporal-dcc/i11-stage2-test-registration.md) に取得前固定した。`arrival=false` の通常既定とmainの `src/` は変更していない。上の「実装前に残る判断」は当時の草案として読み、試験用技術値は登録文書の後続決定を優先する。

3素材9本の取得先は `target/i11-stage2-comparison-20260926-204923/`。source patch、追加source、全Rust source hash、binary hash、入力hashを保存した。[結果記録](../roadmap/temporal-dcc/i11-stage2-results-20260926.md)によると、Sineは到来既知0で無分岐、HarmonicとModalは初回分岐前の非到来入力が一致し、到来既知の選択差とWAV差を得た。旧基準 `06a4772` に対するNoneのWAV・学習record10種は3素材すべてbit一致。候補record共通keyの内容差0、片側だけの件数は分けた。これは試験用の局所作用と3素材のOFF回帰であり、I11-2全項目、作者採用、R2、I12bの完了ではない。CDF欠測の内部原因別報告と実走の自己群除外も残る。

### I11-1 §5.9 の現版再取得

mainの `src/` がcommit `06a4772` と一致することを確認し、登録12条件の `acquire_representative_gap` を新規 `target/i11-gap-06a4772-20260926-205241/` に実行した。ignored test exit 0、12条件の集計は投影3,390件・決定なし84件、支持なしbinによる除外0件。`body` 3,342件、`proxy(absent)` 39件、`proxy(stale)` 9件。`summary.json` のSHA-256 `e3c574b7c11e1f05fb8c9f00c0eb03489afdc7062bfc2d7339ac3930b4bf71b0` は旧 `target/i11-gap-20260923/summary.json` とbit一致。今回のsource全hash、登録hash、試験binary（SHA-256 `d4ac7f0c9c96f900a442f63f5bf36a663413481b2947b8029500dfa623117fba`）、test出力・status、端数内訳は同名の `-capsule/` に保存した。

旧記録で未確認だった代理39件の `D_rep` はすべて `86399.99771118164` sample。登録の整数再投影は `ceil(D_rep)=86400` sampleで、端数増分は0.002288818359375 sample。16 binの代理座標との丸め境界を残すが、分布の最大を含め現版と旧版の集計は完全一致した。§5.9は合否ではなく分布報告であり、この一致を実際の音色妥当性や作者採用とは呼ばない。

### I11-1 §5.7 の現版専有測定準備

取得前の条件整理。登録文書の §5.7「A/A雑音幅」と「`own_us`」は、同じ機械・時点のA/A差の最大をfloorとし、A/A 3 passの48標本から `own_us` floorを算出する規則を技術判定へ組み込む。新しい専有窓ではA/A 3 passも取り直した。`analysis_wait_us` 等、機構が変えた音楽に伴う解析負荷の受入は、同記録が明示する別のR2資源判定であり、A/A floorと混同しない。旧floorの流用は歴史的感度比較まで。

取得器、登録15入力のA/A用バイト複写（`body`ラベルも `config-none.toml` と完全一致）、06a版instrument／renderの保存binary、検査器と3 pass floor生成器を `target/i11-resource-06a4772-prep-20260926-210125/` に封印した。`prep-manifest.json` は各hash、`run-resource.sh` は実行前に封印24ファイル、元planと元入力15件、共有profileのhash、現mainの06a版source treeと出力先4個の不存在を確認した。順序はA/A 96本×3 pass、現版body／none 96本、`hop_path`・A/A floor・`own_us` の判定。floor生成器は旧3 passから旧floor数値を厳密再現した。各実行後 `runs.json` を保存し、中断時の同一出力先への自動再開は設けなかった。

旧実走の所要合計はbody／none 96本950.753秒、A/A 3 pass各956.315／967.223／966.635秒、計3,840.926秒（約64分）。準備時には75〜90分の専有枠を見込んだ。現版の実測所要は下記へ分けて記録する。

### I11-1 §5.7 の現版専有測定結果（2026-09-26）

21:25:55 JSTに封印runbookを開始し、22:30:53 JSTに判定まで終了した。A/A 3 passとbody／noneの各96本、計384本はすべてexit 0。report有無・4／16 Voiceの登録8条件を交互3反復で取得し、各実行のprofile・log、report modeのjsonlを保存した。各passの実行時間合計は966.162／961.506／966.502／958.850秒、計3,853.020秒。取得中に他のbuild・test・renderを開始せず、session終了後にCPU専有を解除した。sourceは `06a4772c43d06b41b44753bb0be891f23e16b93e` の209 Rustファイルで4 pass間のhash集合が一致し、instrument／renderのSHA-256は `6e4f1c77ef21ad4d7bc8fb03a6da313daaebcab15a27ff043cfc73a65acf376d`／`eb852e34e78a9e0ec40465065ae12ca09fe818ecace08d8002375133ee3623fa`。入力・検査器・結果のhashと実行台帳は同取得先の `prep-manifest.json`、`resource-capsule-manifest.json` に保存した。

登録後の組合せ規則では、`population_us` と `synthesis_us` をA/A 3 passのfloorで、`elapsed_us` の3欄と超過hopを待ちを除いた `own_us` で判定する。16組中14組合格、2組不合格。`harmonic-flow-16 report` の `population_us` 中央値差は6.421 µs、許容5.966 µs（A/A floor 2.571 µs）。`harmonic-flow-16 no-report` の `synthesis_us` p99差は346.873 µs、許容261.851 µs（floorと同じ）。`own_us` は16組すべて合格し、`own_us` 基準の超過hopはbody／noneとも全組0。旧raw `elapsed_us` は11/16合格で、改訂後の直接処理判定には使わない。`aa-floor-verdict.json` の総合10/16は旧raw `elapsed_us` 欄も含む値であり、改訂登録の14/16と区別する。合成した16行と不合格欄は `registered-technical-verdict.json` に保存した。

reportの `population_us` 中央値差は各反復でbody側が大きく、8.191／5.831／6.980 µs。no-reportの `synthesis_us` p99差は605.434／129.391／7.771 µsで反復間の変動が大きい。A/A floorはこの専有窓の48比較の最大差であり、統計的な雑音上界ではない。取得中の既知のbuild・test・render競合は避けたが、ホストの全負荷を継続記録しておらず、外部CPU競合と機構由来の費用をこの結果だけで分離できない。登録欄で観測した不合格は保持し、再測定・許容緩和で取り消さない。一方、待ちを含む総hop負荷と全worker資源の受入は別のR2条件であり、この結果から実害の有無やR2合格を宣言しない。
