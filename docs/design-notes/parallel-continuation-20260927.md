# 並列継続の作業境界

2026-09-27。依頼は「計画を並列で進める。astraである必要がないところはsol-6に投げて」。この記録は再開先を特定するための索引であり、タスク全体の完了判定ではない。主計画は[身体評価F0–F6](body-aware-fitness-plan.md)と[時間DCCの工程](../roadmap/temporal-dcc/milestones.md)。既定採用、作者受入、実機受入、commit、pushは実施していない。

## 今回の分担

| 系統 | 作業場所 | 進める内容 | 証拠と後続境界 |
| --- | --- | --- | --- |
| F4同hop遷移 | `.worktrees/body-fitness-repeated-respawn` | 前児の翌hop receipt確認と新しいrespawnを同hopで順次処理する。退役source、残存tail、RNG/counter/ID変更前の拒否も検査する | v18bの保存source/binary/rawを保持し、新規登録・通常renderer・局所fault試験・全suiteを別取得する |
| F4移動／生存比較 | `.worktrees/body-fitness-factorial` | 出生なし固定異種二身体について、点／点、身体／点、点／身体、身体／身体を同じ環境とprepared時計で比較する | 親pool・出生候補は含まない。後続で同一Populationの異種founder、非遺伝の固定子身体、三消費者の比較へ進む |
| I11-2検算 | `.worktrees/i11-stage2` | 既存CDF理由配送を再実装せず、通常recordの費用・到来表・選択・skipと登録§5.1–5.3の独立再計算を補う | 保存recordの間引きで再現できない入力は未検証と明示し、必要な追加取得を別登録する |

各系統の実装・検算は`gpt-6-sol`へ委譲し、主担当は設計上の交絡、実装の順序、証拠の範囲、共有記録を確認する。worktreeごとの変更所有者を分け、性能計測は並列実行に混ぜない。各src変更は所定の全`cargo test`と同shell終了コード、fmt、標準Clippy、all-targets checkを残す。

## F4比較版の出所

新しい`body-fitness-factorial`はmain基底`06a4772c43d06b41b44753bb0be891f23e16b93e`に、保存済みv18bの414ファイルを重ね、全ファイルhashを照合した。作業中の同hop修正を途中でコピーしていない。

- 保存元: `.worktrees/body-fitness-repeated-respawn/target/integration-source-20260927-v18b/`。
- `manifest.json` SHA-256: `c235a2f40b25799bff59e099b8ff8f6395b3dde68a11ef43904fa548091c89b6`。
- `source.tar.gz` SHA-256: `aedb390e5a5d6600fc1a2d5b920c3b59c59b3a43d1c22800d1a5b56ab16cacbb`。
- 継承manifest: `.worktrees/body-fitness-factorial/target/factorial-validation/inherited-v18b-manifest.json`。

比較の[実装レビュー](body-fitness-all-consumers-implementation-review-20260927.md)は、旧flag OFFでは環境と時計が同時に変わる問題を確認した。点条件でも同じ代表密度を準備し、質量は実測診断として保持し、消費するscore/levelだけ同じ地形の点値へ置換する。初期取得は同gain条件であり、身体間の放射RMSが等しいとは扱わない。

## 比較版の初回失敗と回復修正

出生なし比較の初回PP取得ではframe4に二声が一度ずつprepared判断を消費した後、frame10のRecipe不一致を経て保留が続いた。固定音色でもEntrainの自主発振rateがsubstepで変わるため、古いRecipeを拒否すること自体は正しい。別の不備として、render後の`BodyFitnessAction::request`が新Recipeをcaptureする際にslotのRecipe/generationを更新せず、次hopの`before`が新しい応答までobsoleteとして捨てていた。

失敗版はmain checkoutから見て`.worktrees/target-body-fitness-factorial/failure-v1/`へsource/binary/rawを固定した（比較worktree内部のtargetではない）。修正はrequest時のslot identity更新に限定し、live Recipeの凍結や一致検査の迂回は行わない。scene・seed・係数・時計を維持したPP再取得はframe59まで有効消費22・拒否2となり、継続消費が回復した。同hop修正版は別のsource版であり、この修正を自動的に継承したとは扱わない。

## 四条件の差と移動が起きなかった範囲

比較版は全検査と固定を完了した。1330成功・0失敗・48 ignore、`cargo test exit=0 @ 2026-09-27T18:39:05+09:00`。fmt、標準Clippy、all-targets checkも成功。[証拠manifest](../../.worktrees/target-body-fitness-factorial/factorial-sealed-20260927-183905/capsule.sha256)の11項目にsource、binary、初回失敗、raw、全test logをまとめた。source archive SHA-256は`15b98351a4c4506cf87b6b307e90803ba16f0f5a474351adce9191b077f2011e`。主担当はmanifest全項目と419ファイルの現sourceを照合した。通常rendererの最終8取得も独立に読み、各modeの60 frame比較recordとWAVの二回一致、各声11判断で採点使用が正であること、全frameのtarget・現在f0不変、energy表を確認した。

[比較結果](../../.worktrees/body-fitness-factorial/docs/design-notes/body-fitness-factorial-results-20260927.md)では、全条件で有効消費22・拒否2。初回移動判断は同じ候補と時計で点／身体scoreが異なり、生存方式の切替ではHarmonicのframe1 score差がenergyへ伝わった。一方、全条件のtarget・現在f0は不変で、WAVも全条件同一。死亡はなく、生存時間差も評価できない。Modalのframe1 scoreは両方式0であり、登録にはない「各声とも初回差必須」という過強なテスト断言の失敗と取得後の修正を結果に残した。

最初の静的レビューは`seek_consonance()`がFree/HillClimbへ解決する部分だけを読み、「Lockではない」と誤って判断した。後続の診断で訂正した。実fixtureの`at(440)`／`at(466)`は`scripting/mod.rs`の配置処理で`spec.set_freq`を呼び、`PitchControl::set_freq_lock_clamped`が**その後にLockへ上書きする**。診断preflightのHarmonic frame4ではbest adjusted 0.757799、baseline 0.004199、改善幅0.753600で閾値0.1を超え、proposalはlog2 8.8646927へ変わったが、controllerのLock分岐がtargetを元のlog2 8.7813597へ戻した。これはrange clampやGlideの遅さではない。主担当も配置からLockまでのコードを直接照合した。費用や閾値が不移動の原因という先の推定は撤回する。

`score_uses`は評価器の消費回数であり、最終targetの変更回数ではない。[診断登録](body-fitness-movement-diagnostic-registration-20260927.md)の条件を保ち、実使用mode、proposal、Lockでの上書きと最終targetを全判断で確認する実装・取得を進めている。登録の分類にはLock overrideがなく、追加発見として結果へ記録し、range clampと同一の分類へ書き換えない。旧四条件の音高・音声不変とenergy差という観測結果は変わらないが、このfixtureからFree移動での効果なしを結論しない。身体と初期f0の割当交換や、実際にFreeとなる配置の比較は別登録へ分ける。今回の結果を見てseedや係数を探し、既登録の成功例へ置換しない。

## 同hop重なり修正版の確定

[結果](../../.worktrees/body-fitness-repeated-respawn/docs/design-notes/body-fitness-overlap-results-20260927.md)と[取得索引](../../.worktrees/body-fitness-repeated-respawn/target/overlap-validation/validation.json)を保存した。1331成功・0失敗・48 ignore、`cargo test exit=0 @ 2026-09-27T18:21:31+09:00`。fmt、標準Clippy、all-targets checkも成功。主担当は415ファイルの現sourceとmanifest、archive、binary、全test logのhashを独立照合し、test集計も一致した。前機会の`next`はcleanup前、次機会の`birth`はcleanup後の集合であり、旧v18bのrecordと意味を混同しない。容量拒否は新しいspawnの抽選・counter・ID変更前であり、そのhopの通常advance全体の巻戻しではない。

## 再開時の確認

I11-2の今回の独立検算単位は完了した。[結果](../roadmap/temporal-dcc/i11-stage2-results-20260926.md)に既存362判断と追加60判断、CDF440表の照合、Python14検査、正確な消費snapshotの照合が残ることを記録した。主担当もreport hash、検査器とplan hashを照合し、新規反例6件を再実行して成功を確認した。I11-2全体の完了ではない。

続いて[正確な消費入力の記録](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-consumed-arrival-registration-20260927.md)を別登録し、[結果](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-consumed-arrival-results-20260927.md)を固定した。同じ到着費用計算が使うsnapshot・CDF・群選択・自己群照合の入力をreport-onlyで保存し、ON241判断・OFF181判断の独立監査はエラー0。既知23判断529候補の確率は消費CDFから再計算して差0。旧10本の音声とprovenance以外の判断は一致した。主担当もhash照合、368ファイルのsource capsuleと現物の比較、監査とPython8件を再実行した。Rust1170成功・0失敗・36 ignore、fmt・標準Clippy・all-targets check成功。自己群除外と複数採用群の通常正例は各0で、全体の受入は未完。既存の間引き観測から入力を推定したとは扱わない。

今回の三系統は実装・検査・証拠固定と共有計画への反映まで完了し、次の移動診断は登録のみである。再開時は各worktreeの最新の登録・結果・`test_status.txt`と、実際の稼働ハンドルを確認する。保存した版と以後の作業中sourceを混同しない。別版の同hop修正・四条件比較・I11を一つの検証済み実装として扱わない。出生なし四条件はF4全消費者比較の前段であり、F5/F6、I11全体、R2/A1–A4の完了を代用しない。

## 次の継続: 移動診断・複数群・三消費者の統合

上記の三系統を確定した後、同じSol担当へ次工程を委譲した。上記test集計とhashは固定旧版の証拠であり、これからの変更へ転用しない。

- `body-fitness-factorial`では登録済み移動診断を実装する。旧capsuleの11項目を再照合し、`.worktrees/target-body-fitness-factorial/movement-old-fixed-20260927/`へ複製して書込不可にした後で着手する。scene・係数・期間を維持し、実消費時の採点内訳の記録と独立再計算、旧版との判断・音声不変性を検査する。
- `i11-stage2`では既存の凍結binary/modelで[複数群取得](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-multigroup-results-20260927.md)を完了した。通常2回とも28判断中19判断に複数群、算術平均の監査エラー0、判断配列・WAV一致。主担当も再検算した。2周期刺激から最大4群が生じており、独立音源との一対一対応は別である。自己群除外0の一方、habitatへ送らない参加者にも静音Record由来のbus0 Bindingが存在した。誤除外は起きていないが潜在機構を確認したため、[自己群証拠の必要条件](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-self-group-evidence-registration-20260927.md)を別登録して実装した。Record.active・prototype一般の意味や凍結係数を変えず、I11専用でcurrent routeと同一消費Recordのスペクトル証拠を要求する。ON269判断・OFF181判断の監査エラー0、旧11本のWAV・新証拠と自己群理由以外の判断内容は一致した。全1171テスト成功、fmt・標準Clippy・all-targets check成功。主担当も541ファイルのsource capsule、15入力hash、全監査と比較を再確認した。自己群正例は未達のまま。
- 三消費者の統合担当は、同hop版と四条件版の差分、同一Population内の異種founder、非遺伝の固定child spec、三消費者の独立切替と順序を[統合計画](body-fitness-three-consumer-integration-plan-20260927.md)へ具体化した。両固定版は同じHEADだが、四条件版には同hopの順序・投影容量検査がなく、同hop版には四条件のscore切替・request時のRecipe更新がない。共有ファイルの丸ごと置換で結合しない。設計確認後、`.worktrees/body-fitness-three-consumer`を同じHEADで作成し、固定419ファイルarchiveを展開・全hash照合した。`target/three-consumer-validation/inherited-factorial.json`に継承元を記録した。ここで意味上の結合と既存回帰、異種founder・三軸の実装へ進む。移動診断中の作業木を入力としてコピーしない。三軸の数値fixtureは実行前に別登録する。

I11自己群証拠の[結果と通常正例欠測の診断](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-stage2-self-group-evidence-results-20260927.md)は固定済み。habitat接続かつスペクトル証拠ありの221判断は全件Bindingなしで、現在の凍結モデルとの対応段階が未達である。モデルを新規訓練してholdoutへ適用する案は設計のみ。診断担当は次に、固定した三消費者v1の数値・比較の独立レビューを担当する。

### 三消費者v1の独立レビュー

固定v1の16本はすべてexit 0、各条件の2回でWAVと主要な決定記録が一致した。ppp/bppはframe15までの共通履歴の後、Voice3のtargetが9.58344269/9.44802666、実Hzが896.065735/876.593689へ分かれた。消費時刻7680、候補132件のkey/recipe、事前target/RNG probeは同じである。ただし補正scoreと選択時の乱数draw列は欠測しており、report単独の独立選択再計算は未達。ppp/pbpではframe1にlevelとenergyが変わり、frame19の親重みに届くが、選択親はともにID3のままだった。全substepのlevelも欠測しており、energy更新総和の独立再計算とは扱わない。

ppp/ppbの初出生は親pool・親RNG probes・選択親・peak・局所候補Hzが同一であり、記録された局所score最大を独立に求めると、出生児と一致する463.478485/462.141846 Hzになった。これらは22hopの限定比較であり、長期生存や生態的選択の証拠ではない。環境全scanの欠測を同一環境の完全な観測証明へ読み替えない。

v1 binaryの取得時hashはplanにあり、主担当もv2変更前に現物を照合した。しかし外部build pathをv2が上書きしたため、v1 binary現物は保存されていない。v1のsource capsuleとrawは保持し、この保存上の限界を明示する。通常Spawnの不要Vecを除いたv2は別取得し、同一性と全検査を確認してから最終版として記録する。

### 固定配置の移動診断を確定

[最終結果](../../.worktrees/body-fitness-factorial/docs/design-notes/body-fitness-movement-diagnostic-results-20260927.md)はrun-2006の8本、各22判断の全176件がLock上書き。全件で改善閾値を越えたproposalが初期targetへ戻った。点補間・身体積分のbest/baseline raw誤差0、adjusted score最大差5.5879e-8。旧scene/config/WAV、60frame共通report、同条件2回の追加診断が一致し、主担当も独立検査器を再実行した。reportなしで診断を生成しない局所検査も通過した。

全検査は1331成功・0失敗・48 ignore、19:32:22 JST exit0。途中の旧子プロセスによるログ競合を除き、一意logへ単独取得した最終記録であり、NULなし、通常test_report/statusへのcopy一致を主担当も確認した。途中suiteは中断として別保存する。固定capsuleは`.worktrees/target-body-fitness-factorial/movement-diagnostic-sealed-20260927-193222/`、全16 hashを主担当でも検証した。三消費者統合版とは別sourceの診断証拠である。

### 今回の三系統の保存状態

三消費者は[最終v2結果](../../.worktrees/body-fitness-three-consumer/docs/design-notes/body-fitness-three-consumer-results-20260927.md)を固定した。全16本exit0、各条件の反復一致とv1/v2のWAV・決定記録一致。三軸を単独で変えた4組ずつに移動Hz・energy・出生Hzの直接差がある。親重みは変わるが選択親は同じであり、長期生存は未評価。全1336テスト成功・0失敗・48 ignore、19:40:49 JST exit0。fmt・標準Clippy・all-targets check・英日mdbook build成功。主担当も422source・固定binary・16入力・32 raw hash、全16本の旧新比較、test集計・NULなし・通常log/statusへのcopy一致を確認した。

移動診断、I11自己群証拠ゲート、三消費者統合の今回分は検証と記録反映まで完了した。次の具体的な単位は三消費者の確率移動・代謝substepのreport-only独立再計算。I11の新規訓練／holdout案は未実施であり、旧rawへモデルを適合していない。三つのworktreeは別実装で、I11と三消費者や追加移動診断を一つの統合済み版とは扱わない。F4全体、F5/F6、I11全体の完了判定、main採用・commit・pushは行っていない。

三消費者v2の最終[検証索引](../../.worktrees/body-fitness-three-consumer/target/three-consumer-validation-v2/validation-index.json)はSHA-256 `4dbd5e399ffa191213817944c068aed0e51d104b25cd641f1d12e0a19a63f46a`。主担当は索引内96ファイルのサイズとhashを再照合し、全件一致を確認した。

## 次の継続: 実消費の独立再計算とI11訓練設計

前回は三系統の実装・取得・検証と結果固定を完了したため、進展ありと分類する。最新v2と計画を再読し、三消費者のRust計測実装／正式取得を既存Sol担当、同worktreeの独立Python検算器を別のSol担当へ分担した。固定scene・8条件・seed・22hopを保持し、確率移動の実drawと補正score、代謝の全substepと逐次energy適用をreport-onlyで取得する。別担当はI11の独立訓練／model凍結／holdoutを具体的な事前登録へ進める。取得済み11本へ再適合せず、係数やseedを結果に合わせない。

主担当は[生存比較の残条件](body-fitness-all-consumers-draft-20260927.md)を現v2から整理した。背景死亡による出生とenergy枯渇による生存差を区別し、背景死亡0を拒否する現validator、生存親2声／一機会一死亡、tailを含む上限4、固定Sine子による異種性の消失が長期化の障壁と確認した。今回の短い取得をそのまま長期生存の完了へ読み替えない。

三消費者の[計測登録](../../.worktrees/body-fitness-three-consumer/docs/design-notes/body-fitness-three-consumer-trace-registration-20260927.md)を確認した。energyは各f32演算順で再計算し、一般のscore許容差1e-5で微小なenergy効果を消さない。既存親poolと実threshold drawも記録して親抽選を再計算する。演算・RNGを計測目的で増やさず、reportなしのheapを増やさない。

I11の[新系統の事前登録](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-self-positive-calibration-registration-20260927.md)をレビューし、取得へ進めた。旧モデルの訓練288記録は第6座標が全て0だった。新規6訓練sceneから各4声・3cutの72枠を固定規則で採り、既存fitterの8 medoidと0.25を維持して別3sceneをON/OFF検証する。主担当は10入力fileと旧source capsule・binary hashを照合した。最新窓の不適格を別窓へ差し替えず、train/holdoutのseed探索や旧rawへの適合はしない。新bodyモデルと版が合わない旧action profilesはこの新系統から除くため、旧11本との同一条件比較ではない。

### 新訓練の取得後訂正と独立照合

原登録はすべての`body_observation`をfinishedと要求し、原collectorはその行を一つと仮定していた。通常reportには途中経過と最終summaryの62行があり、6本とも原検査は不合格になった。[取得後訂正1](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-self-positive-calibration-correction-1-20260927.md)で最終summaryのfinishedと全行の異常counter 0を要求する形式契約へ直した。原登録・原collector・6件不合格のselectionは現物とhashを保存した。取得前からこの規則だったとは扱わない。raw再取得や選定枠の変更はない。

主担当は6本のrawから72枠の最新`(end, available)`を独立に再選定し、選定結果・時計・スペクトル条件・hashを照合した。訂正後v2/v3 corpusはbyte同一、SHA-256 `6838dae1e3bb394a3b40fe4ec62f0eddb7f570721a12037569f95da9c8e92c43`。第6座標の母標準偏差は`0.4234971790863734`で、旧訓練のゼロ分散はこの新系統では解消した。

固定モデル`2cc5cd87cb4f19937b60515c50212082823d0213ecf3f5d419f13016d20cf02c`の自己hash、正規化corpus hash、6座標の平均・標準偏差、8 medoidの原記録、72件の最短距離・割当・適合判定、目的値も主担当が別計算で一致を確認した。訓練内の0.25適合は18/72であり、訓練結果をholdoutの成立と扱わない。`preholdout-plan.json`の18ファイルと6 scene参照のhashは一致し、ON/OFF設定差はarrivalのbool一つだけだった。

三消費者の独立移動検算器では、Pythonのf64で同点境界を評価するとRustのf32と判定が逆転する具体例を主担当が確認した。例えばbest=0.5、次score=0.5000010132789612では、f64の`best + 1e-6`比較とf32加算後の比較が異なる。実f32演算順と独立rawから補正scoreまでの誤差伝播へ修正し、音源の入力やRNGを変更せずに検証する。

### I11新モデルのholdout陰性を固定

[新系統の結果](../../.worktrees/i11-stage2/docs/roadmap/temporal-dcc/i11-self-positive-calibration-results-20260927.md)は、6訓練・6holdoutの全render exit0、ON/OFF各178判断の数値監査エラー0。ONでは178 habitat route → 173消費Record → 164 spectral Record → 14 Binding → 14共有model一致 → 0群対応 → 0自己群除外だった。主担当もrawから自己群状態と除外0を再集計した。旧訓練の第6座標ゼロ分散を解消しても、通常自己群正例は成立しなかった。

14件の共有割当tupleと完全一致するarchived観測を使った補助検算では、13件の最近傍距離がすべて0.25超、1件は群候補なしだった。これは群対応の不成立を再現するが、decisionが消費したdescriptor snapshotとのpointer同一性を証明しない。身体の私的履歴と混合群の履歴は同じ6座標式でも観測対象と時間窓が異なる。閾値の緩和やseed探索へ進めず、このモデル・seedの陰性結果を保持する。

ModalのみON/OFFの最初の選択分岐がnow=47104で生じ、Sine/Harmonicでは判断・WAVの差がなかった。これはarrival項の効果であり、自己群除外の効果ではない。collector反例4件、raw改竄反例2件、Python構文検査が通過した。今回のI11単位は`src/`変更なし、旧固定binary/sourceと1171成功の旧test logを維持した。主担当が最終[100件索引](../../.worktrees/i11-stage2/target/i11-self-positive-calibration-20260927/artifact-index.json)の全サイズ・hashを再確認した。索引SHA-256は`f8582537f01cfbd044435d8db32534d08ad0e06867ade7151ce47fc296ddd7b7`。この取得・検証単位は完了だが、I11全体は未完了である。

### 三消費者traceの正式取得とテスト期待の訂正

423ファイルの初回trace sourceを固定し、同じsourceから作ったbinary現物を保存して全16本を取得した。主担当は全sourceのarchive/current/hash、16入力、32 raw hash、旧v2との16 WAV byte一致を確認した。固定binary SHA-256は`941c09160a0d26fe69c6ceed37bfd0f318bbc14c421f3495fcee0c144c4dbd17`。

Python独立監査は16本の旧通常record projectionと条件内反復、176成功移動判断、8272 lifecycle substep、親poolの64員を照合した。検算対象の最大数値誤差は0で、energy各stageのULP差も0。身体積分の独立raw検算は各判断のbest/baselineに限り、その他の身体候補はprepared rawを入力とする補正・分岐検算である。adaptation状態の生成、親RNG状態全体は再構成していない。実onsetイベントは0件なので、その経路の数値実証は追加していない。採点前入力が共通となる移動frame4と代謝frame0を、実結果の初分岐とは区別する。移動のraw scoreはbasisによってframe4から異なり得る。ppp/bppの実target/f0はframe15、ppp/pbpのscore/level/energyはframe1で初めて分かれる。

初回full suiteは、新規のreportなしテストが既存comparison用`candidate_scores`にもNoneを期待したため1件失敗した。主担当は旧v2 capsuleにも`self.comparison.then(...)`があると確認した。訂正は`cfg(test)`の期待を`.is_none()`から`.is_some()`へ変えた1箇所のみで、追加traceがNoneとなる検査は保持した。失敗版source・log・status・rawを残し、訂正版423ファイルを別capsuleに保存した。主担当も全archive hashとsource差分1箇所を照合した。訂正版manifest SHA-256は`ca329b6e6518b91dde869b8c2015ef2fa777fd0b39811552628c12dc70804c68`。訂正版から`cargo build --bin conchordal-render`で再buildした直後のbinaryは取得版とbyte一致した（dev profile、opt-level=1、debug情報あり）。共用debug pathは後続のtest buildで上書きされた。取得用の固定binaryと16本のrawは保持し、下記のproduction回帰発見後は別のv2 source/binaryへ移行した。このtest期待だけを訂正した版のfull suiteは、下記のquota失敗として保存した。

### 次単位の読み取りレビュー

最終full suite中にSol担当が次の設計境界を読み取り確認した。追加実装・取得はまだ行っていない。

有限コホートの最小入口は既存2軸factorialの拡張とする案が有力である。異種2 founderを別Populationの共有音場で追い、出生設定なしの4条件を保てる。三消費者版からは実消費traceを再利用し、同一Populationの3 founderと出生を伴う長期比較は別単位にする。basal適用直後のenergy 0、retrigger停止、Idle、Voice cleanup、renderer側の源別tail終了、observer source退役を別のendpointとする。self音声slotの解放だけではrendererのtail終了を証明できない。Finishで未観測のendpointは右打切りであり、取得失敗とは区別する。source最後の消失・同hop複数枯渇・pending actionの退役・源別tail帰属を数値登録前に確認する。policy None時の既存spawn counter更新は、それだけを理由に変更しない。

I11の次診断ではprivate sourceの窓/reset、shared群の混合履歴、eligibilityと100ms cacheを分離する。実消費したRecordとshared snapshotの窓・6座標・coverage・選別理由を保存し、後から同じtupleの観測を探す補助照合から進める。単独sourceと第2source重畳の比較ではtarget自身のPCM同一性も必要で、未確認なら混合のみの効果と判定しない。同一交差窓の診断再集計は候補だが、実判定を変更しない経路と取得条件を先に登録する。モデル・尺度・0.25は固定し、今回holdoutへの再適合はしない。

### trace初回版の追加停止条件

テスト期待を訂正した第2回full suiteもexit101となった。`habituation_field_assay`の4件が`Disk quota exceeded`でreport書き込みに失敗しており、main filesystemの容量不足ではなく、`usrquota`付き`/tmp` tmpfsのユーザー割当量が原因だった。主担当は今回の失敗一時ファイル`conchordal_hab_5137_*`の8件、345,837,937 bytesを`target/three-consumer-trace-validation/failed-suite-temp-5137/`へcopyし、全SHA照合後に元一時ファイルだけを除去した。退避記録は同ディレクトリの`custody.json`に残した。他案件の`/tmp`やbuild cacheは削除しない。以降の全検査は作業ディスク上の専用`TMPDIR`を使う。

さらに静的確認でMetropolis計測のproduction回帰が見つかった。旧分岐`delta >= 0 || draw < exp(...)`はdeltaがNaNでもdrawを消費するが、初回計測版の`delta < 0`を条件とした記録はdrawを消費しなかった。固定16本のprepared scoreは有限で、その限定結果は保持する。ただし任意scorerのRNG不変契約を満たさないため、初回計測版を最終成功版としない。元の比較分岐とdraw→exp順序を保持する修正、NaN反例、別source/binary固定、16本の再取得とfull suiteを新版として進める。先のテスト期待だけの訂正とは異なり、production source変更を伴う版である。

主担当は旧同点抽選の`rng.random_range(0.0..1.0) < 0.5`がf64へ推論される一方、計測fieldのf32型が新式の推論を変えていた点も確認した。v2では同点drawを元のf64で保存し、旧式f64 drawの全bit・pick・最終RNGとの局所照合を追加した。Metropolis drawは元からf32なので維持する。初回16本の同点draw実件数は0であり、この型差による同取得の実値変更はない。NaNと同点の両反例、fmt・標準Clippy・all-targets checkは通過した。NaN修正だけの仮capsule/binaryは正式plan・取得前に保全し、[v2差分登録](../../.worktrees/body-fitness-three-consumer/docs/design-notes/body-fitness-three-consumer-trace-v2-registration-20260927.md)に両修正を記載してから正式版を固定する。

v2正式版は424ファイルで、manifest SHA-256 `4a38980497dcc0ada3bb9a87b1f750733b83028949d367f67cc600b7c4944f83`、capsule `86a7e8f6692e9a0745deab100e195bec6e08f47bc051402c496985777e8bb28a`、固定binary `cf79d88e920d76e24060499eb096c3dc69402d85909229cfe7a60c0ddc5e2bf8`、取得plan `bb6054071e0a4e224a3fc032f29d28ba2504411e9ca1004e34de0e342f34e1b4`。主担当が全424ファイルのarchive/current/hashとこれらの宣言値を照合した。取得と全検査の保存先は`target/three-consumer-trace-validation-v2/`であり、初回版と混ぜない。

### trace v2の最終確定

[v2結果](../../.worktrees/body-fitness-three-consumer/docs/design-notes/body-fitness-three-consumer-trace-v2-results-20260927.md)の16本は全exit0。移動176判断・代謝8272 substep・親pool64員の独立検算は全件通過し、監査JSONは初回版とbyte同一だった。主担当も16入力・32 raw hash・全WAVの旧2版との一致を照合した。さらに時間2項のみを除いた2976 recordをcanonical JSON文字列で比較し、診断値や負のゼロの表記を含め完全一致を確認した。Python検算器は変更せず、[別索引](../../.worktrees/body-fitness-three-consumer/target/three-consumer-trace-validation-v2/python-audit-index.json)の63項を主担当もサイズ・hash照合した。

最終full suiteは専用TMPDIRで2026-09-27 20:53:43 JSTにexit0。40群、1341成功・0失敗・48 ignore。前回quotaで失敗した習慣化4試験も通過した。主担当が集計、NUL 0、通常test_report/statusへのbyte同一copyを確認した。最終log SHA-256は`dd94803455839fda74d8dd6acc09e4ee256357b6227601a4535cd63fa6712daf`。fmt・標準Clippy・all-targets checkもv2源版で通過している。

最終[検証索引](../../.worktrees/body-fitness-three-consumer/target/three-consumer-trace-validation-v2/validation-index.json)は122項、SHA-256 `03881612220d6f92c8e586c7ec8756e1cdfdb9bfd9cc6d70bba06af6e093723a`。主担当も122項の全サイズ・hash、現在source424ファイルとmanifestの一致を確認した。初回版のテスト期待誤り・quota失敗・RNG回帰とその成果物は別保存し、最終版の合格へ混ぜない。

この継続では三消費者traceの実装・取得・独立検算・全検査と、I11新モデルの訓練／holdout陰性結果の固定を完了した。F4全体の長期生存・選択作用、F5/F6、I11全体の終了条件は未完了。次は上記の出生なし有限コホートと、private body／shared群の窓・混合・選別を切り分ける診断を具体的に登録する。各worktreeの実装は別系統であり、main sourceへの採用やcommit・pushは行っていない。

## 次の継続: 有限コホートと実消費descriptor診断

旧証拠の変更を避けるため、HEAD `06a4772c43d06b41b44753bb0be891f23e16b93e` から二つの隔離worktreeを作成した。`.worktrees/body-fitness-cohort` へ固定trace v2 capsule（SHA-256 `86a7e8f6692e9a0745deab100e195bec6e08f47bc051402c496985777e8bb28a`）を展開し、424 manifest entryを照合した。`.worktrees/i11-descriptor-diagnostic` へ固定self-evidence capsule（SHA-256 `40dff6201169739dc1b86d93d2220432eaf63aeb24d80b95944a6ab74e952481`）を展開し、541ファイルをarchive内容と照合した。継承元の検証結果はこれからの変更の合格証拠には転用しない。

Sol担当をF4 Rust実装、独立Python検算、I11診断実装へ分担した。F4は局所退役検査と数値登録を先行し、孤立音RMS校正は通常amp 0.06を基準に最小RMSへ減衰する。混合音場やFree移動後のRMS一致とは呼ばない。I11は実decisionが消費したprivate Recordとshared descriptor、filter/cacheの由来をreport-onlyで保存し、凍結model・尺度・閾値0.25を保つ。交差窓再集計は今回の範囲から外し、窓差だけが原因だとは確定しない。targetのPCMが条件間で一致しない取得は、混合だけを変えた因果対照と扱わない。

登録・実装は進行中で、正式数値取得は未実施。以後のbuild/testでは各worktreeの `target/tmp` をTMPDIRに指定し、先の `/tmp` user quota失敗を避ける。既存取得と失敗ログを削除しない。

### 取得前レビューと局所検査

F4の[有限コホート登録](../../.worktrees/body-fitness-cohort/docs/design-notes/body-fitness-cohort-registration-20260927.md)は16セル×2反復、seed 7、300 hopに固定した。正式profileはrelease、最大4並列の4本×8 batchとする。source 0、pending action退役、Voice退役後の源別tail、同hop二声枯渇、report有無のWAV一致を局所検査した。RMS校正はWAVの量子化値ではなくrendererの源別f32音声を使用し、窓のみの保存PCMと全長WAVのsample offsetを分離する。実spawnの身体snapshot、Harmonic source 1／Modal source 2、generation 0、初発tone 0を局所fixtureで確認した。ただし、校正本体へtone 0を反映したという初期確認は誤りであり、後述の全suiteで実ToneSpecの不一致が判明した。校正の固定励振と通常Entrainの時間変動発音は同一とは扱わない。

I11の[descriptor診断登録](../../.worktrees/i11-descriptor-diagnostic/docs/roadmap/temporal-dcc/i11-descriptor-diagnostic-registration-20260927.md)は凍結モデルのまま単独source／第2source重畳の二条件とする。PCM比較binは取得前にhop境界へ整列し、`[0,96256)`、`[96256,192512)`、`[192512,312320)`に固定した。script上の変更時刻とは別の区間である。spread 0のunison更新は実発音数を増やさず、観測側のRecipe世代をresetする。cache再利用時には除去されたgroupの代替を再選択しないため、消費descriptorからの新しいnearest診断と、実cache assignmentの正しさを分けて検査する。

I11の最初の全suiteでは、SnapshotのCopyからderive Cloneへの変更後に既存habituation試験4件がworker stack overflowで失敗した。失敗ログと修正前source capsuleを保存した。既存Copy fieldの直接コピーと更新先への `clone_from` に変更し、stackサイズ増量やunsafeを使わず該当4件を通過した。この局所回復を全suite合格とは扱わず、最終全検査を別実行する。両系統ともこの時点では正式取得・最終受入は未完了である。

### I11実消費descriptor診断の確定

[診断結果](../../.worktrees/i11-descriptor-diagnostic/docs/roadmap/temporal-dcc/i11-descriptor-diagnostic-results-20260927.md)は二条件ともexit 0、18／17判断すべてBindingなしである。descriptorと到来CDFの独立監査は誤り0。前のPCM比較binは全hopのSHA一致、中央binは158 hopの明示的unsupported、後のbinは不一致だった。支持窓の完全一致は0組であり、窓だけの説明、混合だけの説明はいずれも確定できない。世代reset後の成熟区間で距離を計算できた5件も固定閾値0.25を超え、最大寄与は全件スペクトル広がり座標だった。実自己群除外の通常正例は依然0である。

最終Rust全suiteは1174成功・0失敗・36 ignore、21:32:27 JST exit 0。fmt、標準Clippy、all-targets check、Python反例4件を通過した。実binaryはdev profile（opt-level 1、debug有効）でありreleaseではない。取得前planのprofile記載漏れは別provenanceで補い、planを改変しなかった。unsupported PCMをmalformedにも数えた取得前Python判定器は現物を保持し、取得後v2で区分のみ修正した。区間の判定と取得rawは不変。主担当は510 sourceの取得前一致、12入力hash、全test集計、最終34-entry索引（SHA-256 `34510d5dce82947405304d1496c77f2556829a746c9eb0768aa64f493e16ebb1`）を照合した。

### F4有限コホート初版の中断と訂正

校正v1の保存PCMと記録値の算術は独立監査に合格したが、実ToneSpecのidentityを保証していなかった。校正tone 1→0の修正が既存の `outcome_batch` 試験helperへ誤適用され、校正本体はtone 1のまま、JSONだけtone 0と記録していた。Modal位相はtone identityに依存するため、登録不一致として校正・主取得v1を不合格とした。全suiteも既存renderer試験6件で失敗し、後続integrationには進んでいない。主担当と担当agentの両方が凍結前版とのdiffで誤適用を確認した。

取得をexit 130で中断し、最終確認で10本完了・2本partialを保存した。中断直後の暫定8本完了／4本partialは最終値ではない。`target/body-fitness-cohort-validation/interruption.json` に、その時点で存在した122 rawファイルのhashを記録した（SHA-256 `3e2b65337202dc9615e59067d252fb2da0eeee977d60bb212f11e1189cb09cb3`）。後の再照合で、親shellのexit130後も取得子プロセスが継続していたと判明した。この索引は中間snapshotであり、取得停止後の固定索引ではない。失敗source、固定binary、校正、旧plan、検査logは保持する。修正版はvalidation-v2へ分離し、metadataを実batchから生成する。独立sourceレビュー、局所6件の回復、全Rust suiteの完走を先行してから、同じ32条件を校正から再取得する。seed・期間・係数・合否閾値を結果に合わせて変えない。


### 有限コホートv2の校正とsource独立確認

v2は旧test helperのTone IDを1へ復元し、校正用OnとToneSpecを登録どおり0へ変更した。校正metadataは実batchのsource/generation/toneから生成する。独立担当と主担当が該当sourceを読み、SHA-256 `9332b48f58f106421c2bdee68e307462f51fa7fec5e5c5f57a2f6de01c386cbe` の一致を確認した。登録補遺を含む426-file capsuleのmanifestは `82e0a520184cefaa0305bb80003746bb8e6879abf931457b5b7bf75835167b29`。主担当が全426ファイルのサイズ・hashを照合した。

校正8本を別root `target/body-fitness-cohort-validation-v2/` に取得した。固定release校正binaryは `559e91df407c9bcb81c489b18b87d1e81bd720665162f4c6e449d4bf362e0118`、校正JSONは `5771715dcca09be83ce3676deedad33deafcc05299173d53efe2cf1cc68a0c6c`。実Tone IDは全行0。主担当も8組のsource f32 PCM・WAVのhash、9600 sample、source identity、独立RMSとpeakを照合した。目標RMSは0.017562035876685644であり、旧v1の校正値を流用していない。全Rust suiteは進行中で、その完走を主32本の開始条件とする。


旧v1取得子は最終的に8 batch・32 runをすべてexit0で完走した。親shell exit130と取得子の完走は別の状態である。主担当は最終224 rawファイルのサイズ・hashを再照合した。最終失敗版索引 `v1-final-failed-index.json` のSHA-256は `06e6da0c5b6bdc07b75fb586115a516616dfd3f34b986c0b33f66ef1ee04c0b7`。実Tone ID不一致とfull suiteの6失敗があるため、32本完走を登録適合や成功へ読み替えない。旧interruption索引は中間snapshotとして保持する。v2 runnerには停止markerから子をterminate/joinしstatusを保存する経路を追加し、sleep子による局所確認を行った。


### 有限コホートv2の全検査通過と正式取得開始

全Rust suiteは41群、1347成功・0失敗・48 ignore、2026-09-27 22:07:38 JSTにexit0で完走した。主担当もlogの全結果・NULなし・通常test_reportへのbyte同一複写を確認した。log SHA-256は `43b4399359ec693ae38149da1932bee3ee1b5dd6c7fa7314e6b609c6b6554032`。fmt、標準Clippy、all-targets checkもexit0だった。

正式取得前の監査で、既存offline rendererはFinishを処理した後にも音声hopを描画すると確認した。観測reportはframe 0..299、Finishはsample 153600のまま固定する。WAVは301 hop以上・512 sample整列を要求し、Finish後の付随音声を生存endpointへ算入しない。release残存時は付随区間が1 hopを超えるため、終端1 hopや無音を一般条件にしない。WAV全体のa/b一致は維持する。登録補遺・Python検査器だけを修正し、Rust sourceと検査済みbinary入力は変えていない。

最終426-file source manifestは `6e48318a6cf00d3ead1ff1c750e6ca88da6f5e0d00926f82cd9647fc56a530c9`、登録は `d8ee38dca4bb65500d0fabb06d57dfa5bb6fed063e01a2c69bc7e15847222a7d`。主担当が全sourceのサイズ・hashと前capsuleとの差分3ファイルを照合した。最終planのSHA-256は `41693aeff127596eef6415aaa6b0d2cd2299800efdafabe81bd2ac636da747d8`、固定release rendererは `e6faf2d63b3f4ada78666c1058628467d4a16cb2231bec1b78d4b88e391a43cd`、formal checkerは `47e7063908efe1d5a71764a303e429b33b4fc35b657abb93d39f5adfd10a490f`。plan版名の期待文字列不一致を取得前に一行訂正し、旧plan/checkerを保持した。主担当が32 scene/configの計64 hash、16セル内のa/b入力一致、校正gainのf32値、各固定artifactを照合し、独立静的監査の通過後に4本並列×8 batchを開始した。

登録した旧factorial fixture回帰も、継承元trace v2と今回full suiteの保存済みartifactで検査した。主担当の独立再演で8 WAVのbyte一致、既存4064 recordの値・順序一致を確認した。除外は新規cohort record、旧baselineには存在しない6 source field、実測wall-clock fieldのみ。追加取得は行っていない。回帰記録 `inherited-factorial-regression.json` のSHA-256は `419f59341e8925b2a1b39c8e660b728faa4ab389e4f9f8535b65b011923112d7`。


### 有限コホートv2の最終確定

[結果文書](../../.worktrees/body-fitness-cohort/docs/design-notes/body-fitness-cohort-results-20260927.md)を固定した。32本すべてexit0、2026-09-27 22:26:43 JSTに取得を終了した。16セルのa/bは全WAVと決定的reportが一致し、16組の単一軸比較も独立監査を通過した。代謝106,740 lifecycle eventの各energy stageは独立f32再演と0 ULP差、levelの最大絶対差は5.960464477539063e-8。登録窓の24,420 eventでは実scan・density・duからscore/massを検算し、残る82,320 eventは保存scoreを条件入力とした。onsetは0件であり、通常取得によるonset経路の実証には使わない。

16セルの各2 founderについて、初回energy 0、retrigger停止、Idle、Voice cleanup、accepted observer退役、renderer tail完了をFinish前に観測した。a/bを含む64 source観測の右打切りは0。初回枯渇はframe205..211、cleanupは206..212、tail完了は234..286だった。生存変化の向きは身体・初期f0・振幅条件に依存する。例としてHの移動PP→BPは正順通常で8 substep遅延し、正順孤立RMSでは20 substep早まった。身体評価の一方向の利益や音色固有の生存優位を結論しない。孤立固定音の校正は、主実験中の混合・Free移動音声のRMS等量性を証明しない。

[独立監査](../../.worktrees/body-fitness-cohort/target/body-fitness-cohort-validation-v2/python-cohort-audit-v2.json)のSHA-256は `c483ca8e7e4e939bf8dceed612b465da7cac6c52973b17b02ef3f09494f41977`。主担当も32 runのexit0と出力128 hashを確認し、最終[1583項索引](../../.worktrees/body-fitness-cohort/target/body-fitness-cohort-validation-v2/validation-index.json)の全サイズ・hashを照合した。索引SHA-256は `17d6c9aa4cf3bf7c567c4bbce588ae25c844af25b74f9c224f3b445b980ca875`、結果文書は `f2f54a226cec30bf9c8c316fbae67f13e3849d7f7a31f47af7f066ccec0c2e74`。失敗v1、各preflight版、校正・取得binary、source、検査器・反例、rawとlogは別々に保存した。

この取得・検証単位は完了した。親pool、同一Populationの異種親選択、出生地点評価、子の継承、長期選択、実時間性能、作者受入、F5/F6は残る。I11も今回の記述子診断は確定したが、private gate失敗とBinding不成立の解消は未完了である。mainへのsource採用、commit、pushは行っていない。


### 次単位: 実枯渇から親選択・出生への接続とI11同窓入力診断

前回は有限コホートとI11記述子診断の取得・独立検算・証拠固定を完了した進捗回だった。全体目標は未完了のまま継続する。Sol担当3名を、F4実装、F4比較登録・独立検算、I11入力診断へ再分担した。封印済みworktreeは変更しない。新 `.worktrees/body-fitness-selection` はcohort v2のpost-wav capsule426ファイルを全hash照合して継承した。新 `.worktrees/i11-window-inputs` はI11診断の固定source archive510 regular fileを照合して継承し、その後に取得後verifier v2と反例testの2ファイルだけを現確定版から適用した。各worktreeの `target/inherited-source-manifest.json` に由来とhashを残す。専用target/TMPDIRを用い、旧ログ・索引を上書きしない。

F4は同一PopulationのHarmonic／Modal親候補と短enduranceのSine founderを使い、背景死亡0で実代謝枯渇から親pool、energy重みと実draw、予定固定子の出生先採点、実出生、翌hop自己除去receiptまでをつなぐ案を採った。短寿命founderは出生機会を作る明示介入であり、一般生態の自然な世代交代とは扱わない。8 consumer mode、初期f0交換、通常振幅／孤立RMS対照、a/bを維持する。読み取りで逆順Linearの出生候補域が一点に潰れることを発見したため、初期配置順を保って `freq_range_hz` のLinear候補域を正規化する修正と回帰を先行する。f0対照を削って容易な条件へ縮めない。校正は新しい実source identityとTone IDに合わせて別取得する。親確率分布への作用と実親choiceの変化を区別し、seedや期間を結果から選び直さない。数値条件はこれから別登録で固定する。

I11は安定単独、同一target＋第二源、単独recipe resetの3条件案とする。private Laneとshared Contextの全hop実入力、accent、reset/eviction、群へのenergy割当をreport-only経路で記録し、実消費された元窓の再現と同一交差窓のshadow集計を独立に行う。旧rawには全hop入力がないため、その再集計を推測で補わない。群handleを音源IDと見なさず、unsupportedとsupported zeroを区別する。固定model・尺度・0.25 gateは変えず、PCMからDSPを独立再構築した検査とも区別する。登録、実装、検証はこれからであり、I11不成立を解消したとは記録しない。


次F4の[取得前登録](../../.worktrees/body-fitness-selection/docs/design-notes/body-fitness-selection-registration-20260927.md)は、H1／短寿命Sine2／Modal3、親380↔520 Hz・Sine450 Hz、8 mode×f0割当×通常／孤立RMS×a/bの64本、seed7、120hop=1.28秒へ具体化した。親・子endurance2秒、trigger0.5秒、recovery10秒、attack energy増減0。期間はtriggerの基礎消費−最大回復とSine減衰の上界、親のenergy下界から設定し、取得後に延長しない。新5固定音の校正は15360 sample・窓[4800,14400)、実source/gen/toneに一致させ、相対RMS誤差0.005以内・peak<1を先に固定した。逆Linearのrange修正、実Communityでの逆順520/450/380、三消費者のみzero-hazard受理の局所3試験は通過した。初回fixtureのModal factory未登録によるfallback失敗は試験側を訂正し、通常runtimeの登録変更ではないと区別した。全suiteと正式64本は未実施。

[I11新登録](../../.worktrees/i11-window-inputs/docs/roadmap/temporal-dcc/i11-window-inputs-registration-20260927.md)はseed20261101、安定単独・安定重畳・単独resetの各7秒をa/b、計6取得へ固定した。入力tupleからのnative窓再現を先行し、失敗したpairを同窓比較のpositiveに数えない。モデル・尺度・gateを固定し、数値許容・境界の不確実判定・dev build profileも取得前に登録した。読み取りで、群energy割当がOk(None)を返すとgroup scanの前hop値が残る経路を確認したため、traceは成立状態と当該scanの有効性を区別する。新src実装は進行中で、取得結果はまだない。


### 実枯渇から出生への局所確認とI11入力診断の最終Rust検査

F4の通常offline rendererによるforward／BBB／通常振幅の120hop局所試験は、短寿命source 2の実代謝枯渇から二親pool、実一出生、生後hopの子PCMゼロ、翌hopのSourceRemoved receipt、出生hop全sourceのfull-density記録、四源のtail追跡まで通過した。逆順配置と校正五音のactual body snapshot・初発Tone ID 0も局所検査を通過した。これは正式64本の比較結果ではない。旧positive-hazard三消費者のintegration二件も通過した。独立Python検算はenergy・親CDF・全調製候補score・endpointを扱い、保存されたpeak集合と選択indexを条件とする局所探索の再演を追加中である。peak抽出とWeightedIndex内部乱数の独立再現とは区別する。

I11新入力診断の最終Rust suiteは31群、1176成功・0失敗・36 ignore、2026-09-27 23:10:07 JST exit0だった。主担当も全結果を集計し、logにNULがないことを確認した。`test_report.txt` のSHA-256は `dd1de1798fd5a0b4fb89f96fa4ad062951ef6491f7d34d8e292b6a86ef303c76`。その時点の518-file source archiveは `f0d2286d40971fdd008043c80264dae94e0fc5a5a5e61327fbafa29060b31f85`、manifestは `f064313c63a9d6768757fb35a2338e0034be58198f010bd643f5abcbe2e3f76d`。主担当がarchive内全regular fileとlive全ファイルのサイズ・hashを照合した。途中版の23:02:25 JSTの検査は別保存し、最終版の根拠へ流用していない。

I11の独立検算でnative集計式との対応、f32入力の丸め、キャッシュ生成時刻、terminal emitted/dropと全sequence、両busのowner/hop集合、unsupportedと観測された無音の区別を確認した。両側とも全座標欠測の一致はpositiveに含めない。共有群の特徴量とContext入力の対応検査、および反例を最終化してからcheckerを別固定する。Rustの最終検査後にPythonや登録文書だけが変わる場合は、その差分を取得planに明記する。正式六本の後に、単独AのON reporterなし、OFF reporterあり／なしの三本を実施する順序も取得前に固定した。ON/OFF間の行動一致は要求せず、各設定内のreport有無でWAV一致を検査する。


### I11入力診断の取得・独立検算と部分hop補遺

I11の固定plan（SHA-256 `140ac78fed55b16c127a84921c032a0a39ff3aedfdb42bda941959ff65241f53`）について、主担当が18入力hash、9コマンドの順序・seed・report指定、ON/OFFの差がarrival値だけであることを確認してから取得を開始した。9本は全件exit0。主担当も43 rawファイルのサイズ・hash、9終了状態、a/b三組とON/OFF各設定内のreport有無二組のWAV byte一致を独立照合した。raw manifestのSHA-256は `a301378cb89de8a35d35826fc047601904c7405f538e9be0b8c3ef7a5d59af36`。

[主診断結果](../../.worktrees/i11-window-inputs/docs/roadmap/temporal-dcc/i11-window-inputs-results-20260927.md)では、各条件20判断のうちbody Recordがある19件を全件再現し、消費した群記述子も単独A95／重畳B91／reset C95件を再現した。元登録の同窓完全対は各a/bで72／69／55件、全対で身体と共有群の6座標が不一致だった。安定単独Aではprivate Bindingが2判断成立したが、両者とも共有群への割当は`assignment_unmapped`。B/CのBindingは0、実自己群除外は全条件0。前単位のBinding0を今回へ引き継がない。

独立Python checkerは取得前12反例を通っていたが、実rawで局所変数による関数名の隠蔽、群の帰属が無効なhopの特徴量成立条件、空native記述子とRecord公表時刻の混同を発見した。v1〜v4とその失敗を保存し、productionの契約に沿うv5で6 ON報告の誤り0、旧到来CDF検算も全件合格を確認した。v5は `480ecc6cddacb4e725509344779906bda49c3b26611545e924d112c26c8aa32e`。source、raw、モデル、尺度、0.25 gate、元planは変更していない。Record公表時刻の照合は、この決定的取得でworker clockがendに一致する範囲に限る。

主担当は数値rawを読む前に、96000 sampleの2秒窓が512 sample hopに整列しないことを算術から指摘し、左端の部分hopをproductionと同じ重みで扱う[別解析補遺](../../.worktrees/i11-window-inputs/docs/roadmap/temporal-dcc/i11-partial-window-analysis-registration-20260927.md)を固定した。[補遺結果](../../.worktrees/i11-window-inputs/docs/roadmap/temporal-dcc/i11-partial-window-analysis-results-20260927.md)はA13／B0／C5の追加完全対、Cの支持hop欠落4。完全18対は全件不一致で、最大寄与は全件log RMSだった。元解析の成功件数には合算しない。補遺v1/v2の元checker由来の失敗も保持し、最終補遺v3は全6 exit0。別agentが支持条件、重み、native再現先行、保存結果の全件集計とa/b同値を読み取り確認した。

A/Bのtarget PCMは139 hop一致・501不一致で、連続一致prefixは138 hopまで。第二源による純粋な混合入力差という解釈を後半へ延ばせない。A/Cは全WAVと共有側の全producer入力が一致した一方、reset後のprivate支持158 hopが欠測となった。支持された482 hopのtarget PCMに不一致はない。これらは入力境界とresetの診断であり、物理的な音源と群の一般的対応、通常の自己群除外正例、I11全体の受入は未完了である。

I11主診断・補遺・各失敗版を含む[最終195項索引](../../.worktrees/i11-window-inputs/target/i11-window-inputs-final-index.json)を固定した。SHA-256は `40b6e4e182173d9e7209665c9157e3f4ac8f5e015c05afbc34a3b10701568e7c`。主担当も全195現物のサイズ・hashを独立に照合した。この診断単位は完了し、対象worktreeの封印済み証拠は以後変更しない。


### F4実枯渇・親選択単位の全検査、校正、64条件取得開始

新selection版は42群、1354成功・0失敗・48 ignore、2026-09-27 23:37:44 JSTにfull cargo testがexit0で完走した。主担当も集計・NULなし・通常test_report/statusへのbyte一致を確認した。log SHA-256は `3f167a29677eefb84e39adf51f3d5dc15e9d639fe439d9f55f11e07a7b7ca856`。fmt、標準Clippy、all-targets checkも通過した。旧positive-hazard BBBのWAVと既存209 record投影も一致した。逆順試験で初期Hzのlog2往復誤差を厳密等号とした失敗は現物を残し、nominal Hzの局所許容のみ訂正した。source frozen 431項はmanifest SHA-256 `2eddb7dfc2100c0fde4a743ca0008781eb0f7b725393a7f7136f3d10cdf53456`で、主担当がcapsuleとlive全現物を照合した。

release校正libtestを固定し、校正plan `f075197b6b0e174bf8b3638e7682f2a590d638729add73c0d193380dac9c9b37`に従って五組×基準／減衰後の10音を取得した。source H1／Sine2／Modal3、generation0、初発Tone0、各9600 sampleのf32窓とWAVを保持した。校正JSONは `d53cc9c7860fc784e8ecaab9204f4e5392ceef38b70194b6f42718762718b692`、Rust記録の目標RMSは0.017670218785683478。独立fsumとの差は登録内で、減衰後全五組が相対0.005以内だった。主担当も全10窓のRMS、identity、f32/WAV hashを照合した。初稿planの0.06 f32 bits手記誤りは数値取得前に実pack値へ訂正し、初稿を別保存した。

正式[64条件plan](../../.worktrees/body-fitness-selection/target/body-fitness-selection-validation/acquisition-plan.json)はSHA-256 `ed97260fb3d7b6918a3e28698039b32ed0e0fcf4ac5bd9765b94e124303fafe3`、固定release rendererは `82de9905f5569d8eb10a75c749939069d6632f5767b81e3554ea95804e804f4e`。独立担当の静的監査と主担当の全128 scene/config hash、32組a/b入力一致、64順序／4並列×16 batch／argv／source／校正／checkerの照合後に取得を開始した。runnerの停止markerから子をterminate/joinしstatusを保存する経路もdummy子で確認した。SIGINT一般への保証はしない。正式結果はこれからであり、局所BBBの因果経路成功を64条件の成功へ読み替えない。


### I11既存rawの事後探索: 全音源と音響群の量の違い

封印後の既存6 ON rawだけを対象とした[追加解析](../../target/i11-union-boundary-20260927/results.md)を別rootへ保存した。これはAの例とBinding二判断を既に見た後の探索に基づく再解析であり、元登録の前向き確認ではない。主担当が最終38 artifactとplanの23入力のサイズ・hashを照合し、a/b結果のbyte一致を確認した。[最終索引](../../target/i11-union-boundary-20260927/final-index.json)は `e6d740fa3119c5151991f5e932e23211b7529612fad0168f9941dcd34c75f22d`、planは `8ba37827ef347e32703d52730b96271d422571a0bc28e5ae370fe05d74e05b15`。Rust変更・追加renderはない。

単独Aでは、target privateの640 hopの音声energyが同時計の混合habitat energyと一致し、非ゼロ626 hopでは8 slot（残余を含む）のunion scanとprivate scanも一致した。全sourceの量を個別groupの部分量と比較していることが、log RMS差の一因となる。例の28-hop同窓では群のenergy分率0.0004914085819400978が差5.49539466323003を正確に説明し、主担当も独立計算で照合した。共通窓の全完全対72／69／55件についても、source/mixとmix/groupの対数比へ分解して消費tupleと照合した。Bのsource/mix比は波形の交差項を含むため、物理的寄与率とは見なさない。

初版の「Cでも支持されたscanは常に一致する」という期待は、実raw試行で反証された。初版登録・checker・A/B試行・C失敗を保存した上で、同じ数値境界のままCの一致426／不一致39／scan支持なし3を負結果として分けた。さらにprivate支持欠測158 hopを分母に残した。plan固定前の実raw試行があるため、この改版を盲検の事前確認とは呼ばない。native RMS再計算の同値は固定6報告のhop整列した右端とcausalなaccepted入力に限定し、一般の部分右hop・未来availabilityへ拡張しない。

Aの実Binding二判断では、同じ消費tupleに対する仮想的なbody→group直接距離と、実際のmedoid→group距離も分けた。最初の判断は0.2421670904対0.3434593285、次は0.3027109720対0.4196997703だった。第一例では直接距離だけなら0.25以下でも、実二段階照合では0.25を超える。これは量子化経路の数値差であり、直接比較へ変更すべきという採用判断や、その群が物理的自己音源である証明ではない。実自己群除外の正例は0のまま維持する。


### 2026-09-28: I11次工程の設計境界

読み取り監査で、現経路はprivate全source記述子とshared部分群記述子を固定prototypeで橋渡しする代理関連付けであると確認した。[元登録](../roadmap/temporal-dcc/i11-stage2-test-registration.md)も完全な自声検出を保証していない。今回の量の不一致や二段階距離差だけで、group handleを物理source identityへ読み替えたり、直接距離へ置換したりしない。

次の候補は二つに限定する。第一はprivate per-hop scanを同時刻のshared群配分へ投影するreport-only診断で、単独音源の全群和・群別energyの算術整合を検査する。同じ配分作用素による一致は独立したsource同定の正例ではない。第二は混合音と対象sourceを除いた音を同一支持窓で解析する反実仮想比較である。対象群の変化と他源群の保持を検査できるが、干渉や群の分裂・融合により、差だけから固有の帰属を推定できるとは限らない。両案ともroute-off、zeroとunsupported、reset、時刻不一致、群対応不明を反例またはunknownとして扱う。

この段階は設計所見であり、実装・取得登録ではない。固定medoid、尺度、0.25 gateを維持し、結果に合わせた再fitやseed探索を先行させない。I11の実CDF群除外の通常正例、配送鮮度、資源の受入は別に残る。


### 有限F4比較後に優先する技術上の未達

F4正式64本の完了前に、次工程を読み取りで整理した。同種の小規模出生assayを無条件に増やすのではなく、現統合版の通常action ONにおける持続稼働と、代謝・出生・respawnの非同期配送を優先する。前者では身体・route・controlの反復変更について、requestからReady、実消費までの全列、失効・延期率、プロセス全体のCPU・メモリを追う。[第十五版b](body-fitness-cold-parallel-results-20260927.md)の10秒6条件は限定合格だが、変更後最初の冷job詳細は定期reportによる上書きで不明であり、cacheの1 MiB/sourceは全プロセスの資源上限ではない。旧第十三・十四版の失敗も、現selection版の失敗として転用しない。

現selection版の代謝・出生・respawn設定はblockingなexact offline専用である。非同期化では、欠測中の代謝更新と出生保留、結果失効、source増減、actionとの同時利用、有効評価率を取得前に契約化する。[主計画の実時間方針](body-aware-fitness-plan.md#f3-の実時間方針)どおり、音色ごとの計算負荷による評価欠落を音響的な選択圧と取り違えない。今回の有限因果鎖の結果は、その契約や現版の長時間・実device検証の代用ではない。F5の作者採用とF6の遺伝は別工程として維持する。


非同期代謝を先行する最小案も読み取りで整理した。既存source-removed observerから環境を得て、代表身体の一つの密度だけを有界queueで準備し、消費時の環境scanで既存の純粋評価式を使う。現F3のRequestはhill-climb候補集合とRNGに依存するため、そのまま転用できない。source・Recipe identity、取消、非blocking受信の規則を再利用する案であり、全消費者用の汎用workerを先行して作らない。source別workerを増やす場合は、action側の最大8計算threadへの上乗せ、lane共有、queue競合、全CPU・RSS予算が未決として残る。

重要な境界は、現代謝が各64 sample制御substepの自律pitch更新後に、実Hzを含むRecipe identityで密度を照合する点である。Glide中に前hopの密度をそのまま使ったり、実Hzをtargetへ置き換えたりしない。初段は固定pitch・固定Recipeのexact消費に限定し、各substep/onsetでidentity・route・source世代・epoch・空間・鮮度が不適格なら既存評価へfallbackして理由を残す。observerの `now-support<=4800` に加え、現代謝の `now+512-support<=4800` を満たす必要がある。初hop、退役後Ready、ID再利用、queue満杯、正当なscoreゼロ、lifecycleとonsetの別評価を局所反例に含める。 queue満杯だけを理由に既に有効なReadyを破棄せず、必要な密度が未着か不適格な場合にfallbackする。1・4声のpaced取得では身体別の有効率とfallback、queue滞留、全資源、hop超過・underflowを登録する。可変pitchの高有効率には別の近似・連続密度契約が必要であり、固定pitchでの配線成功をその代用にしない。この案はまだ実装・取得していない。


### 2026-09-28: 実枯渇・親選択・固定子出生の64本を確定

[結果文書](../../.worktrees/body-fitness-selection/docs/design-notes/body-fitness-selection-results-20260927.md)を固定した。正式取得は2026-09-27 23:47:21から09-28 00:22:45 JSTまで、64本・16 batchすべてexit0。32セル内a/bのWAVと決定的report投影が一致した。主担当も全64終了状態、出力256 hash、入力128 hash、32組のWAV一致、plan・binary不変を独立照合した。a/bは決定性検証であり、独立した科学的反復ではない。

全16 batchの検算を再利用し、固定比較関数で各軸16対応対を集約した。集約SHA-256は `26ba3a1e4dd72d194f1333f9da7eb3b07a4fab1d2abfc159d2874510ac9530a4`。移動介入では全16対でtarget・実Hz・energyが分かれ、代謝介入では全16対でscore・level・energyが分かれた。代謝から更新後親energyへの作用は選択確率を変えたが、実親choiceの変化は0だった。出生basis違いによる親抽選機会の重複を除くと固有8対であり、8対すべてが確率差あり・実choice差なし。出生basisの直接16対は死亡identity、親pool・draw・実親、環境、時計、全bin候補のidentityを一致させ、実子Hz差10対、差なし6対だった。主担当も集約の件数と表の重複構造を再計算した。

lifecycle 184,072 event中44,680はfull-density、残り139,392は保存scoreを条件とするenergy再演である。全energy stageの差は0 ULP、score/mass差0、level最大差は5.960464477539063e-8。onsetは0で、通常onsetの正例とは扱わない。出生の局所候補検算は保存peak集合と選択indexを条件とし、peak抽出やWeightedIndex内部乱数の独立再実装ではない。

trigger id 2のenergy 0は全runでframe48、その後のIdle、出生、Voice集合からの除去、observer退役、renderer tail終了を区別して記録した。出生時に残る死者Voiceは翌hopで集合から除去されるため、出生frameと除去frameを同じラベルへ揃えない。Voice除去はframe49–56、tail完了はframe53–60だった。全runで出生一件と翌hopの子のSourceRemoved receiptを確認した。一方、両親と子の計192 source観測では生存終端がFinishまで現れず、右打切りである。今回の成功は単一seed・有限期間・親の音色を継承しない固定子の因果鎖に限り、音色固有の生存優位や一般的な選択係数を主張しない。

全1354テスト・0失敗・48 ignoreの固定記録を保持した。fmt／標準Clippy／all-targets checkは専用log不足を補うため同じsourceで09-28 00:30:20 JSTに記録付き実行し、全exit0だった。sourceとfull suiteは変更・再取得していない。準備時のModal fixture失敗は初回raw未保存を明記し、逆順nominal Hz厳密等号の失敗と校正planのf32 bits訂正前は保存現物へリンクした。正式64本の失敗と混同しない。

[最終1000項索引](../../.worktrees/body-fitness-selection/target/body-fitness-selection-validation/validation-index.json)はSHA-256 `4ee4964e223402f217edd508336fc0d2e2b9f11692908b3976eb745f45d41237`、結果文書は `b482eadbade6e6df22ae1cbeb94675e460d037c184ffcecac8889d30979bfaf8`。主担当も全1000現物、計2,390,550,585 byteのサイズ・hashを独立照合した。この取得・検証単位は完了し、封印済みworktreeを以後変更しない。主計画先頭を今回の結果と次の非同期化・持続稼働へ更新した。F0–F6とI11の全体目標は未完了で、main source採用・既定変更・commit・pushは行っていない。


### 2026-09-28: 非同期代謝と現統合版actionの持続検証を開始

前単位は正式64本、独立検算、結果文書、1000項の証拠固定まで完了し、全体目標は未完のまま継続する。新しい `.worktrees/body-fitness-async-metabolism` と `.worktrees/body-fitness-action-sustained` を同じHEADから作り、封印selection source capsule431ファイルの全サイズ・hashを照合して継承した。取得後に確定したselection Python checkerと反例testの2ファイルだけを追加上書きし、Rust基準は封印版のまま保持した。両worktreeの `target/inherited-source-manifest.json` はSHA-256 `55d62868f1bea03e80fc6d6a1d043474ce213082403a37eb771ff2e1f002a813`で、親の1000項索引、source capsule、2上書きの由来を記録する。

Sol担当を、非同期代謝実装、同契約・取得前登録と独立検算、現統合版actionの持続検証へ再分担した。最初の実装は代表身体一密度を有界queueで準備し、各substep/onsetの実Recipeと適格なSourceRemoved環境で評価する通常runtimeの明示ON経路を目指す。固定pitchはexact消費の正例であり、可変pitchやrecipe/route変更の評価欠落を分母から除かない。旧評価fallbackと理由・有効率を記録し、出生・respawnの非同期化まで一度に一般化しない。別worktreeではrequestから実消費までの診断欠落を修正し、非Sine身体/controlの反復変更と固定route対照で持続性と全プロセス資源を測る準備を進める。同一Voiceのroute更新は現Rhai APIに存在しないため、この診断のためにAPIを増やさず、局所反例としてのroute変更と通常sceneの固定routeを区別する。

両者の正式paced資源取得は実装・全検査・source固定・事前登録の後に専有枠で直列に行う。並列cargoやrenderと競合した値を実時間受入へ使わない。I11の封印済み診断を変更せず、自己群除外の通常正例0、非同期出生、可変pitchでの高有効率、通常採用と作者受入、F6の未了は維持する。


新契約のレビューでは、代表密度Readyと消費環境を分離した。密度はRecipe・完全source identity・route・解析epoch・空間が一致すれば再利用し、環境scanやhabituation更新だけでは破棄しない。scoreは最新の適格環境から毎回計算し、以前のscoreを保持しない。主有効率は初hop、環境欠測、pending、実Hz不一致を含む全実評価機会を分母とし、適格環境だけの条件付き率は補助診断とする。代謝workerは全sourceで一つを共有する案となり、最大4sourceのround-robinと要求別取消tokenを使う。既存actionとの併用時の身体準備計算threadは最大8+1であり、他のruntime threadを含む全資源は別に測る。

固定pitchと固定Recipeも区別する。Entrain/Sustainは基音が固定でもautonomous pulseのrateがsubstepごとに変わり得る。exact正例はGatedの休符・onsetを含めた実Recipe hash安定性を局所検査してから入力固定し、Sustainの変化は動的Recipe対照に残す。また現Gated処理は非空onset batchを一回採点して複数onsetへ適用できるため、評価batch数と実onset適用数を別に計数する。action持続取得は旧身体回復132frame・control回復120frameを反復イベントへ維持し、3%の全source-hop有効率など今回追加する粗い資源上限と区別して登録する。

実装初稿のレビューでは、空onset batchが直前lifecycleの実適用score/levelを診断列で上書きする不具合を発見した。非空batchにだけonset記録を付ける修正と、energy stageの入力と評価記録のbit一致による独立反例検査を進める。取消後にslotを解放しても未完了serialを終了時inventoryから落とさないこと、queue-full試行を正式requestのserialと混同しないこと、trace容量と実drop数も確認対象にした。4source超過は通常scriptから到達しうるためassertで演奏を停止せず、非同期代謝を停止して従来評価へ戻し、理由と未完了処理を記録する。対象外sourceまで有効率を検証したとは扱わない。

action持続sceneの局所parseでは、`wait(3.2)`のf32累積により末尾の変更が意図したframe2100/2400より1hop遅れる点を正式取得前に確認した。登録と検算の時計を実dispatchへ合わせ、身体132frame・control120frameの期限長は維持する。診断有無の局所比較は候補score・判断・Hz・RNGを対象とし、end-to-end PCM同値を直接検証したとは主張しない。資源samplerは対象外processから100ms間隔で観測し、採取区間とwait4による全寿命CPU/RSSを区別する。これらは取得準備中の所見であり、通常paced有効率・資源ゲートの合格結果ではない。

主担当は非同期代謝worktreeの既存 `body_fitness_runtime_live_tests` に固定Rhai/TOMLを受け取る取得入口を追加した。`--play=false` CLIはaudio producerなしで直進し、report付きでは解析も決定的経路になるためpaced証拠に使わない。既存のringbufferと模擬callbackをそのまま利用し、入力現物、声数、prefill、938hopの消費、jitter、underflow、profileを保存する。入口の合格は配送時間と固定声数に限り、代謝有効率・数値再演は独立checkerへ分けた。通常instrumentへの音声disk書込みは追加していない。

取得案は固定GatedのSine/Harmonic/Modal単独、固定四源、動的GatedのModal単独・四源（後者はactionもON）、固定f0 HoldのHarmonic単独の計7sceneに整理し、それぞれ代謝OFF/ONを独立paced反復a/bで取得する28本とした。正式取得はまだ行っていない。Gated/Lockの局所検査では188hopでonset・休符とRecipe hash安定を確認した。一方、`.glide(0.04)`は時定数だけの指定でGatedの自動適用方式はGateSnapだったため、動的sceneには取得前に `.pitch_apply_mode("glide")` を明示する。共通の代謝係数は上限clampによる作用の隠蔽を避けるため式からendurance20秒・recovery30秒へ変更する方針とした。formalでonset energy係数はゼロを維持し、onsetへの配送と適用件数の検証を非ゼロenergy因果効果の実証と混同しない。

その後の通常配線による非paced局所probeでは、F-Sで938hop・lifecycle7,504件・onset batch18件を記録した。D-Mでは実HzとRecipeの変化を確認したが、R-Hの当初入力では938hopすべてでRecipe hashが一値、実Hz440固定であり、意図した動的Recipe対照は成立しなかった。正式取得前のv2では、R-Hを同一Voice・固定f0 Holdのまま、既存Population APIによるbrightness 0.7→0.3（3.2秒）→0.7（6.4秒）の明示変更へ改める。初版入力・manifest・rawを保存し、非Hz Recipeの二遷移と再準備を検証する役割へ変更した由来を記す。自発的なHold pulse-rate変化は未実証として残し、局所probeの有効率や時間を正式pacedの合格結果へ転用しない。

独立checkerはさらに、F-S初版rawのframe46でcumulative onsetが増えた一方、hop別onset件数とevent列には記録がない欠陥を検出した。runtimeは `collect_phonation_batches` の後に非同期代謝の `before/start_hop` を呼んでいたため、onsetは前hopの環境で採点され、その記録が後から消えていた。これは表示だけの問題ではなく消費時の環境・時計の不一致であり、正式取得前にdispatch後・phonation前へ更新順序を修正し、全機会とenergyを再検算する。初版rawと反例は保持し、先のproducer累計値をonset経路の検証成功とは扱わない。

action持続版の全suiteには記録上の別問題が生じた。最初の実失敗は終了診断行のtypeと決定的reportへの変動計時で、修正後の中間suiteを入力現物化の追加前に中断した。その後のsuiteは09-28 01:31:38 JSTにexit0を記録したが、log内の連続NUL28byteが後に可読文字へ変わり、status後の01:34:47にも同inodeへ書込みがあった。主担当も当初報告hashと現物hashの不一致を確認した。中断した旧child/teeによる後書きが疑われるが、原因の断定と単独runの証拠とは区別する。変化後現物は `body-fitness-action-sustained/target/action-sustained-validation/root-log-integrity-observation/` にサイズ・mtime・hash・statusとともに保護した。このsuiteは終了コード0が観測されたものの完全なlogの単独由来を保証できず、完了扱いを保留する。旧inodeを隔離し、次の検査は固有run pathへstdout/stderrと同shell終了値を保存してからcanonical記録へ写す。正式paced取得はこの整合確認と再検査の後とする。

host processの追加確認により、01:24:17開始のaction cargoは上記status時刻後もselection統合testと二つのrendererを実行していた。したがって01:31:38のexit0を当該走行の終了値とする扱い、42群・1,357passの単独走行結果としての扱いは撤回した。旧log inodeはrenameで隔離し、旧cargo・tee・子renderer・status出力shellの終了を確認してから固有pathのv4を一次検査記録とする。

非同期代謝の順序修正版は別の非paced局所F-S rawで独立全件監査を通過した。938hop、lifecycle7,504件、onset batch18件・実適用18件、全energy stage差0 ULPで、旧frame46はonset一件→lifecycle八件の順に回復した。R-H v2も実Hz440固定のままframe300/600でRecipe hashが二度遷移し、準備要求・完了は各三件だった。いずれも正式paced取得前の配線・入力成立確認であり、実時間の率・資源ゲートは未判定である。

旧全suiteのcargo・test・renderer・teeが実際に終了したことをhost processで確認し、非同期代謝v2とaction持続v4をそれぞれ再利用しない固有log pathで実行している。割込み応答のexit130だけでは子process終了の証拠にならなかった。新検査では同shell内の終了値と全出力を対応させ、実終了後にsource・log・statusを照合する。

修正版の非paced局所D-MとF-4も全評価とenergyの独立検算を通過した。D-Mはlifecycle7,504件・onset batch15件に対してbody消費0で、変動Hz下の高有効率は示していない。F-4は各source7,504件のlifecycleを記録し、初回Lockで466／494／523 Hzがlog2往復により2／1／1 ULPだけ変化した後に固定された。初hopを分母から除かず、nominal入力との比較だけ最大2 ULPを許容する取得前補正とし、Ready消費時のRecipe一致はbit厳密を維持する。各Ready個別のspace fingerprintはtraceにないため、formal独立検査は固定global空間とscan長の範囲に限り、Ready空間同一性の根拠は実装と局所反例へ明示的に分ける。

非同期代謝の順序修正版は固有pathの全suiteを2026-09-28 01:56:29 JSTにexit0で終了した。主担当も43群・1,368 pass・0 fail・49 ignored、NULなし、canonicalへのbyte一致を確認した。一次log SHA-256は `d89db5075e0012d2f26652bc8791b2f43218b4535215f528e7e4b32f6b40013a`、statusは `a1ea7ff1828583ba55248098cec579bf2b55fa488b29b83d3386b531501986d7`。これはRust検査の完了であり、正式paced 28本と資源ゲートはまだ未取得である。

action checkerの取得前レビューでは、control更新を跨ぐ旧要求の成功消費を別分類で許す案を撤回した。`Ready::score_with_basis` は要求時controlを保持し、`PreparedVoiceDecision::validate` はlandscape_weightのbit不一致を拒否するためである。要求・Readyが更新後まで残ることと、旧controlでの消費成功は区別する。最終候補source v3は436項、manifest SHA-256 `485bde23b9be4223ccdb6999a16f7bdb481c2967b0ff304564ec8b3685c9c8a8`、archiveは `11dcf6f9491bb90b24bcd8ce6e4ae103ba6394386333bb0c2d0ff5c640a37b89`。主担当が全現物・capsule・archiveを照合し、稼働中v4全suite開始時からの差分は登録文書とPython checker／反例testの3ファイルのみ、Rustは不変と確認した。途中解釈版v2も保存した。

action持続v4全suiteも2026-09-28 01:59:15 JSTにexit0で終了し、主担当が42群・1,357 pass・0 fail・50 ignored、NULなしとcanonical一致を確認した。一次log SHA-256 `0ec983c69a8461e0769127d3c69a088639f3930b79c9d02f34d8e58a21ed1993`、status `6823f14c873434cbd34ed476ba78eb4d8ff5a28973755ce642be4c17d97c1ee0`を採用し、先の混在v3 logは代用しない。

非同期代謝はsource440項（manifest `351da88f380343ee775ac8265a88817c800343027904cbfc247e000479a0a1e2`）とrelease test binary（59,022,704 byte、`56d81fa7a4abf78881921edcad605d2fae3eec7934324649c3d4cf00e9d55568`）を固定した。主担当が全現物・capsuleおよびRust build入力249項との一致を確認した。Python検算はworker履歴の未失効Readyを各sourceの消費serialへ接続し、未要求・未完了・取消済・失効済の見かけの数値一致を拒否する。測定終了のstat/status読取raceは終了を再確認してtail欠測に分類し、稼働中の欠測は依然として整合失敗とする。

action正式8本のplanは `b692cdafb04c4228e459e579c9958bf40d6bf2d1c6df05841296457d3c6f84b9`、wrapperは `412e4ef1515da8bf777b6aeec9d463a1be0c843fb0d6129b97f103c1558fd668`。各case前後に30固定現物と436 sourceを照合し、通常の科学・資源gate失敗は保存して後続caseを取得する。全担当の重い処理停止とhostのcargo／rustc／renderer終了を確認後、action8本を先に専有取得する段階へ進んだ。

action正式8本は全runner exit0、全case前後hash一致で終了した。固定checker v1（audit SHA-256 `0968924b86c152b45009978318dc5eadcfa5c7280d2680b142151b4377de2a97`）は6/8合格とし、both-on／split-onに各3件の `ready_unaccounted`、split-onに資源採取欠測を報告した。小eventとsourceの照合で、6件はいずれも `requested→preparation_ready→consume_rejected(ControlChanged)` であり、実装は拒否後にReadyを破棄していた。旧controlの誤消費ではなく、検算器が拒否をReadyの終端に含めない不備だった。固定v1と全rawを保持し、取得後v2解析を別現物に分離する。split-onの `missing proc status fields: {Threads: 1}` は実際の採取欠測として不合格を維持する。全8条件の測定済CPU／RSS／thread上限は通過したが、欠測を資源全体の合格へ読み替えない。

非同期代謝の正式v1はplan `41857fec332db97f9e97f270ab25f26a213dce92fc2fd2ec9feda05f095ac142` の全28条件を実行したが、全件がpaced開始前のharness assertionでexit101となった。固定TOMLが `[playback] wait_user_start=false, wait_user_exit=false` を省略し、終了待機の既定 `wait_user_exit=true` が入口の条件に反したためである。開始待機の既定はfalseであり、v2では両方を明示する。音声・配送は一度も実行されず、性能陰性や有効率0とは扱わない。失敗inventory SHA-256 `274a47fe67ecb320606d5b4f85c8602713aab23792b11c8a66bfd59197f4f2c0` と全log／plan／入力を保存する。取得v2はこの2設定だけを明示する構造補正とし、全14 TOMLのparse・他key不変・全Rhai byte不変を確認して別rootへ固定する。Rust・release binary・科学閾値は変更しない。

action取得後v2監査は7/8合格となり、split-onの資源採取欠測1件だけを不合格として保持した。全ONの実消費数は単独Sine307、4声Sine2,080、非Sine both326、split328。非Sineの身体変更後回復は最長93／92 frameで132 frame期限内、control変更後は31／43 frameで120 frame期限内。結果文書と最終索引 `301d6c1c7a9db91067433205a7651e7147b766d9dd4e7898bbc7d8c3e7562ed2` を固定し、主担当も1,445現物・626,659,830 byteの全サイズ・hashを独立照合した。取得後v2 auditは `0da10e6cc62d9319064f83e4d2f0d43b96d85e91ba196f0bbfb47493cfe92038`。この取得・監査単位は完了だが、全条件合格・実device採用には未達である。

非同期代謝v2はsource440項manifest `5240337c453a150999ab136d48c787cc1102016ac0af337f0ffdd85a1b3d48b7`、plan `08497d4a1deb9aeb6111bc5c502795f18c76f6cd7bca0a92e36b7fa0a9c0bf40` として別rootに固定した。主担当が14 TOMLをparseし、待機2項目の追加以外の全key値と全Rhai byteの不変を確認した。v1 capsuleとの差は登録・入力generator・checker反例testの3ファイルだけで、Rust・binaryは不変。v1 planが指すlive登録等はv2となったが、取得当時の全hashに一致するv1現物を旧capsuleに保持している。action証拠固定と重い処理停止の後、v2正式28本の専有取得を開始した。

v2 binaryコピーはbyteを保存した一方で実行modeを落としており、初回Popenはcase開始前にPermissionErrorとなった。4つの起動失敗現物をmanifest `058bf679598b7e0ffc46ea8e922129245b39a482f77067af9c6624801bb45b00` とともに保全し、元版と同じmode 0775へ復元、同一byte hash／同一planでv2bを起動した。正式v2bは28/28 process exit0・配送summary合格で終了し、一次custody288項・1,095,944,679 byte（`2910be7ab60cb50a93ab76415afc4af54210ee2632e4d5286a90a75936a2a92b`）を固定した。

固定Python正式監査は不合格を返した。ON14条件の全評価機会・Ready履歴・密度積分とenergy更新は通過し、energy stage差は全件0 ULP。固定単独と4声はlifecycle有効率99.36–99.79%、固定f0のbrightness二変更は99.36%。可変HzのModal単独は5/7,504（約0.067%）、action併用4声はsource別21.96–34.54%であり、動的Hzでの高有効率は未達である。正式監査JSONは `6fcd03fa26ac97ace0b50d9e305468af6f03223d602f92b37f6810a3f25c6e14`。

資源判定は別に不成立だった。8 OFF条件で採取VmHWMがwait4最大RSSをわずかに上回り、checkerの厳密大小比較が失敗した。Linux公式文書はSMPのRSS集計が非同期で精密とは限らないと説明するが、この説明だけで今回の差の原因を断定しない（https://docs.kernel.org/filesystems/proc.html）。F-4／D-4の4 pairは全寿命最大RSSのON−OFF差が256 MiBを超えた。harnessはruntime thread join後にreport全文をVec<Value>へ読む。採取ピークがruntime終了後・thread2の時点であることは終了後解析の寄与と整合するが、未登録の時間窓へ切り替えて合格にはしない。固定監査は20/28 case、2/14 pairの通過にとどまり、他の率・数値正例と資源ゲートの未達を分けて保存する。

非同期代謝の結果文書をSHA-256 `35c74790b5ab8197d0722e07b329389e505d231ea3b56199f7b7cd7ad73e4ab4`、最終索引を `2f3ba1c47c14c5206d2c2da84eb6e7928d9a485c2327c862e14eb9ac2af961a9` として固定した。主担当が1,493項・1,693,002,199 byteの全サイズとhashを独立照合し一致した。局所旧版、正式v1入口失敗、v2起動前失敗、正式v2bの28本、主監査不合格と補助診断、全Rust検査、両capsule・binaryを同索引から追跡できる。この取得・監査単位は完了し、登録資源ゲート未達、動的Hz低有効率、実device・作者受入、非同期出生、F6とI11残件は未完のまま維持する。主計画を更新した。次工程は取得器の終了後全文readbackと資源計数・欠測契約を修正して事前固定したうえで再取得することであり、今回の基準を事後緩和して合格にはしない。


### 2026-09-28: 資源取得器の修正と同条件再取得を開始

前工程は二つの隔離版について正式取得・独立監査・証拠索引まで完了し、actionの資源欠測と非同期代謝の資源上限未達が次の実装判断を定めた。全体目標は未完了のまま、新しい `body-fitness-action-resource` と `body-fitness-async-resource` を同じHEADに作成した。前者は封印action source v3の436項、後者は封印非同期代謝v2の440項を全サイズ・hash照合して継承した。継承manifestは順に `e3bd6cc526a750bb983b0c6f7f31b7b160d278b5d81e41fb4d7ca67c6c7021f4`、`2d54d1a371db4739a68681e94597ca3fa668a765e441ee189ab6ca6823e4d1ad`。旧worktree・capsule・結果は変更しない。

Sol担当は非同期harnessのRust修正、同Python資源checker／登録／独立検算、action sampler／checker／同8条件再取得へ分担した。Rust変更はテスト専用harnessの終了後readbackに限る。JSONLを一行ずつ全件parseして不正行検出を維持し、主窓のaction／observation行だけを保持する。既存summary式・順序・判定は保ち、保存済みrawで必要recordの同値を検査する。音声runtime・身体採点・代謝・入力・既存の科学閾値は変えない。action側はRustが不変なら親のbinaryと全Rust検査をhash由来付きで継承する。

資源samplerは一回の `stat→status` 採取と、その直後の同PIDに対する `wait4(WNOHANG)` を結びつける。procファイル消失・Zombie・必要field欠落という限定的な取得不能が生じ、同じiterationで実際に終了をreapした場合だけ `terminal_overlap` として時刻・段階・元の欠測・観測state・wait4結果を残す。stateだけでは終了を認定せず、未reap欠測、権限エラー、値のparse不正等は不合格にする。terminal overlapは最終iteration一件のみ、数値の補間やゼロ埋めは行わない。異なるRSSカウンタ間の厳密順序は要求せず、それぞれの実値と差を保存する。全寿命wait4 RSSの増分・絶対上限、CPU、配送・率・回復期限は従来基準を維持し、旧取得の判定を書き換えず新しい8本／28本で再検査する。

新取得には採取の網羅品質条件を追加した。100 ms周期に対し、開始から初回有効標本、隣接有効標本間、最終有効標本から全寿命終了までの各gapを200 ms以下とする（浮動小数丸め許容1 ns）。有効標本を2件以上要求し、実測最大gapも保存する。これは旧取得を救済する規則ではなく、新しい取得前条件である。欠測・品質・科学的判定の不合格は保持して全固定caseを継続し、source・binary・入力の同一性破損は停止する。

action新取得はPython反例40件を通過し、source438項（manifest `ea13ca1ba24b3c7f4313d97198e699771ec5365c4933e5a435db9879b3462884`）とplan `7f65d57108283fa52451a1ab739061bb9170708e0714b020e924df1e28f5e55b` を固定した。主担当がlive／capsule全現物と継承Rust・Cargoの不変を確認した。旧release binary `7d44ea38caf3a49988a6d45497a9f16e0fc9d6b08c7eea86b31bfa2a9ec6b0bb` と16入力をbyte同一で継承し、Rust全検査1,357 passも由来付きの継承記録として扱う。新しいRust全検査を実行したとは呼ばない。正式取得は非同期側のRust全検査・release build・旧raw読戻し同値監査が終了した後の専有枠で行う。

非同期harnessの新しい全Rust検査は2026-09-28 02:54:51 JSTに終了した。主担当が固有logとstatus、canonicalのbyte一致、43群・1,369 pass・0 fail・49 ignored、NULなし、旧実行processの終了を確認した。log SHA-256は `1a517672e06a1d817b0e6abeb2df701680a6e1797f6125f2ee6b5c6e157003d3`。fmt、標準clippy、all-targets checkも通過した。追加1件は読戻しの保持順序と無関係な不正JSONの拒否を検査する。続いて専用targetのrelease test binaryを構築する。

非同期release buildは02:57:13 JSTにexit0で終了した。固定binaryは58,968,320 byte、mode0775、SHA-256 `897757d06e79419855c4edfeac5c173b9a37fc3d7e86f55edbad25c61ae5e635`。旧28本のsummary入力投影を独立に再計算し、action80行・observation320行と全28summaryの該当項目が一致した（監査 `cfad8227e229dc94dc3163830e7aaac70aebb41486a74b1614a7fc66161bc376`）。これは集計に必要な入力と式の照合であり、production音声同値の証明へ拡張しない。Python反例25件も通過した。

新source442項のmanifestは `642b03101c0f80b2587e5a7322b8b51f394622fca317e8a255731d07303d022e`、28件planは `31edcf90fef1e26c0bce80d4238d82cf1a21e867da9009758bf5cbd4e2e98331`。主担当もlive／capsule全項とbinaryを照合した。旧入力のRhai7本・TOML14本はbyte同一で、開始・終了待機はともにfalse。Rust・build・大raw監査の終了とhost残留processなしを確認し、action8本、次に非同期28本の順で専有取得へ進んだ。

action新8本は03:04:53 JSTに全runner exit0で終わり、各case前後の固定現物・live source照合を通過した。非同期新28本も03:10:18に全runner exit0で終わった。重い監査は両取得終了後に並列実行した。action固定checkerは8/8合格（audit `173b063ef0fcae3bb0c193b146e372034929f031893ad5255fa5f6532e2c21a1`）。ON実消費数は317／2,156／325／350、body変更後の最長回復95／94 frame、control31／15 frameで従来の132／120 frame内だった。CPU最大4.087 core相当、wait4最大RSS254.6 MiB、threads最大15。8本すべて最終stat Zombieと同iteration reapを記録し、稼働中の欠測は0だった。旧status欠字段事象を新実走で再現したわけではなく、この分岐の検証は局所反例に限る。

非同期固定checkerは28/28 case・14/14 pair合格（audit `e5adaf01044e6ce0adf29aa9d4b6625689cadf4b470f95147ba6dbb5a47eedc0`）。ON14条件の195,104 lifecycle評価と385 onset batchを検算し、全energy stage差0 ULP。固定pitchの有効率99.467–99.787%、brightness二変更99.360%。全28本の稼働中欠測は0、最終終了重複は各1件、最大採取gap0.100855秒。全寿命wait4 RSS増分は0–2.898 MiB、ON絶対最大64.219 MiB、CPU増分最大0.813 coreで、旧256 MiB／1 GiB／1.5 core上限を維持して通過した。主担当も資源JSONから全14組の差と欠測・gapを再計算した。

動的Hzの問題は残る。D-Mは両取得とも469 request・468 completion・Ready 1・cancel 467・Finish pending 1、有効率5/7,504（0.06663%）。D-4はaで15.67–28.12%、bで19.19–25.35%であり、旧取得との差を性能改善と扱わない。動的条件には有効率の登録下限を置いていないため、この新監査全通過は可変Hz高有効率の成立を意味しない。結果文書と証拠索引へこの境界を保存する。

action結果本文を `f60d7e2317f6b5c0fe25cdabb224c11125a73aacee37f026a96a850b43e3e6fe`、最終索引を `e912ba75f025818ac5c4916ccb64b27f745b6fd73da0d5abd3efdcad791e50be` として固定した。主担当が553項・504,068,714 byteの全サイズとhashを独立照合し一致した。非同期結果本文は `16672d1cac0a69ab23ac7e1c7b790d81b5f4598c68d2ac932865647a37b81693`、最終索引は `1dbad513568d1c8b14c34527a83a5e0aef5c683521c85865d5169357c3edf4fa`。こちらも主担当が770項・1,147,674,083 byteを全照合した。両取得・監査・証拠固定の単位は完了し、主計画の現在地を更新した。全体目標は継続中であり、次は動的Hzの実Recipe更新による低有効率と非同期出生・respawnを独立した単位として進める。F5の実device・作者受入、F6、I11の通常自己群除外正例は未完のままである。mainのsource採用・既定変更・commit・pushは行っていない。

### 2026-09-28: 動的Hz密度familyと非同期出生の分離実装

前単位は資源条件の再取得・検算・証拠固定を完了しており、全体目標には動的Hz低有効率と非同期出生・respawnが残る。最新の封印済み非同期資源版source442項を、同じHEADの新worktree `body-fitness-dynamic-density` と `body-fitness-async-birth` へ全サイズ・hash照合して継承した。両継承manifestは `dc2593edefc84ed8dadda298ead38ed6aeb4307ba5264649546434e6ff6562b4`。旧封印版は変更しない。Sol担当を動的HzのRust実装、同数値契約・独立検算、非同期出生の設計実装に分けた。主担当はモデル境界と統合判断を扱う。

動的Modalの既存rawでは1 hop内に中央値8種類の実Hzがあり、1.333 msの生命更新ごとにほぼ別Recipeとなる。取消を含むcompletion中央値10.886 msを完全密度の費用と混同しない。serialを成功Readyに結びつけた完全72 frameの計算はD-Mで13.695 ms、固定F-Mで13.562 ms、D-4で約13.2–15.6 msだった。以前のF3 glide再利用は固定candidate Hzに限るため、その根拠で代謝のactual Hz照合を外すことはできない。新候補は非Hz身体条件を厳密に保つ周波数密度familyであり、実Hzを引数として隣接log2 nodeのraw subjective densityを明示補間し、最新の自己除去環境で従来のERB正規化とF2採点を行う。旧厳密密度とは別の近似契約として、固定した幅・身体・周波数点の全候補を実Tone oracleへ照合してからruntime接続へ進める。格子外やnode欠測は外挿せずfallbackする。有限の誤差取得を全Hz保証とは呼ばない。

非同期出生では、重い候補身体密度の準備を最新環境での採点から分離する。Readyを使う実hopで環境を再評価し、respawnの親drawもその時点の生存pool・更新後energyで一度だけ行う案を具体化する。現Voiceの身体乱数seedはIDだけでなくspawn frameに依存するため、要求時の仮Voiceを未来の実Voiceと同じ身体と仮定しない。新opt-in経路は子IDと身体生成seedを予約し、実生成時はそのseedを使い、生命時計・birth_sampleは実際の出生時刻にする契約を先に記録する。親metadataが非遺伝templateの身体乱数に入らないこと、予約Recipeと実生成前の子Recipe一致、dead tailを含む容量、取消・queue・RNG・初回自己除去を検査対象とする。通常OFFの即時出生と既存offlineの挙動は維持する。これらは着手時の設計であり、通常非同期出生の実消費完了をまだ主張しない。

動的密度の純粋補間境界2件と、実Rhai→VoiceのRecipe同一性preflightが通過した。基準5 familyのframe0/frame16および機械選定したactual Hzの全Recipe hashを厳密照合した。六幅（100／50／25／12.5／3.125／0.78125 cent）、5 family、5,444 node、10,278 query、1,216環境を固定した。主担当の入力レビューで、D-M単独自己除去地形は全ゼロとなりscore・energy差が自明に消える点を検出したため、数値取得前に既指定D-4の8地形（無音2・非平坦6）を層化queryへの交差対照として加えた。actual event比較は同一source・同一hop環境のまま保つ。旧v1は数値未実行として保全し、v2 plan `d611351422c95f4568ec6fe9cf26b24ec3fcab19cafe3c4e8fe0fa7a394618c7`、登録 `15614eb11546b97dc301b71f960de8a1987e4784293a25d52375cc48ce3e65e5` を固定した。

出生版では初回FieldとPeakBiasedの最終Hzにも追加密度が必要なため、全bin表と局所256候補の有界cacheを分離した。最新環境による選択の再演で不足Hzを追加要求し、実親drawやspawnを部分的にcommitしない。opt-in config、seed frame分離、単一worker、初回Field予約、respawn死亡機会予約、runtime pollの最初の配線はlib checkを通過した。Finish取消・出生時計・非平坦環境への実出生と全検査はまだ未完了であり、コンパイル成功を出生経路の成立と扱わない。

probe planの静的検査で、v2のactual Hz選定に登録上除外する同一f32 bitsの重複が残っていた。数値oracle未実行のままv2を保全し、重複除外後のv3を `a801e2b42e3ed76b6163c67a1a6d136cf9c579b43750d9204734d49e2fbd7695`、登録を `b3e8ab4b6d0175961d9a31614e3e1777c2f710f9a8f99a6c3a07061e3627016e` として固定した。v3は5 family・5,444 node・9,120 query・1,027環境である。誤差結果を見て入力を変更したものではない。

非同期出生の局所試験では、初回Field予約からworker Readyを経た実Voice commit、PeakBiasedの全binと追加局所密度からgeneration 1の実子commitの2正例が通過した。これは通常runtimeの出生hop／次hop receiptと全資源条件の取得より前の内部検査であり、F4全体の成立とは区別する。

数値取得前のchecker自査でenergy stageのdt・basal delta・continuous deltaの独立照合不足を検出し、旧v3を保全したままv3b `09198f8e868db43fb3e5e34052ad3a9dbe77de34fb7d21e1ba6d026d59941339` を固定した。planと登録は不変である。447 sourceのmanifestは `602b61d9ffbba32c0299651621ef2cd24636f3569d01899ce4ebe02a82be89cc`、release test binaryは `2e7555ec70142ef6e42379dc87f4c59a13a610ca353e5acb08b0d8927a02e162`。probeは03:53:16 JSTにexit0で終了し、5,444 node・9,120 queryの生成は全件成功した。

固定checker v3bの判定はexit1、全30 family×格子幅で登録誤差条件未達だった（audit `57f7daeb26fd023e5ce3c0f0b7bb449436b39be6cec4b18ce557934f74653a38`）。F2計算失敗は0、記録energy stageの独立照合は通過したが、密度補間そのものの差が大きい。最細0.78125 centでも正規化mass L1の最大値は0.857–1.529であり、通常runtimeへの接続を進めない。主担当がrawを独立に再計算したSine source1の実Hz 652.19873046875では、補間のmassはbin342へ0.76456、bin343へ0.22936だったのに対し、実Tone参照はbin343へ0.99393、bin342へ0だった。解析sourceは選別したpeakのmassを単一binへ配置するため、bin切替を跨ぐ両端密度の線形混合が実密度と大きく違う機序と整合する。全5 familyの最細幅で最大L1を示した点はいずれも既存通常実行から機械選定したactual Hzであった。全幅の失敗と層化／actualの内訳を保持し、閾値や格子を取得後に変更して合格と扱わない。

出生版はH/R変更前後の密度bit不変と、pending予約を保持し環境epochを進めて最新Cで再採点する局所試験を追加した。実行順の確認で、非同期commitは当該frameのrender前である一方、既存self PCM準備より後であると判明した。出生clockを翌frameへずらす案を撤回し、当該frameのbirth_sampleを維持してrender前に新子の自己音声slotを準備する配線を追加した。通常rendererを通る局所正例で、実子出生、512値のゼロ自己PCM、次hopのsource-removed receiptとsupport/birth_sample一致を確認した。2秒scene・380–520Hzの44候補ではFinish時点で未完了だった記録も保存し、狭域430–450Hzの正例を一般の供給期限達成へ読み替えない。登録sceneと資源条件の取得は引き続き未完了である。

出生版の追加レビューでは、pendingの件数だけでmember indexを算出すると先頭予約の取消後に後続予約と重複する条件を検出した。予約済み最大member indexを含めて次値を決める修正と、member 4／5の先頭取消後にmember 6を予約する反例が通過した。また、環境epochはAction受付時ではなく共有Cの再計算完了時に進め、旧Cへ新epochを付けない。固定reservation ticketと各job serialを分けて全配送記録を連結し、reporterなしでは自己PCM統計や診断receipt取得を実行しない。

通常rendererの局所正例は、offline再生のlogical時間内に非同期workerが完了する条件にも依存していた。これは先の成功観測を否定しないが、実行速度に依存しない回帰試験の根拠にはならない。全検査へ組み込む出生配線試験は完了配送を明示制御する対照へ移し、通常配送での成功率・期限は独立した登録取得で扱う。

単純補間の代わりに、数値モデルを変えないexact 72-frame経路の費用分解を次単位として準備した。新worktree `body-fitness-exact-cost` は旧封印済みasync-resourceの442項を全byte/hash照合して継承し、継承manifestを `57e8e07927f9a01177cbee679b1e9d58e7b57d948f7fcb34eaa28dc4d5d01c0e` とした。Tone初期化・描画、AnalysisStream clone/reset、NSGT、peak抽出、残りのfrontendを分け、計時なし経路との密度・massのf32 bits一致を要求する取得案である。時間の合否や通常runtimeの高有効率をここで宣言せず、最適化箇所を選ぶ前の測定とする。現時点では入力登録・探針準備段階で、計測は未実行。現在の補間結果のsource/binaryは変更しない。

動的密度版の全Rust検査は2026-09-28 04:14:07 JSTにexit0で終了した。主担当が43群・1,371 pass・0 fail・51 ignored、NULなし、固有logとcanonicalのbyte一致、host側の該当target子process残留0を確認した。log SHA-256は `530e6ca0bcc8b200e6b32824ba7a63bf845b26a3d1a9d64e88474d9e5863bbd1`。fmt、標準Clippy、all-targets checkも通過した。これは探針・補間実装を含むコード検査の通過であり、全30条件の数値誤差未達を変更しない。単純補間のproduction接続は行わず、全取得・監査・補助内訳の証拠固定を進める。出生版のsourceが安定した後は、同版の全検査を次の直列枠で実行する。

動的密度の陰性結果は本文 `99b9ea308b15d2687e6134ae9adb2acd4877a339962fe278f07211a4577c9434`、最終索引 `927f82579d7cd84e22c553a656726df71d307dc74cfcf3c9f19f8f8a2c3fca4d` として固定した。主担当も482現物・269,522,077 byteの全サイズとhashを独立照合し一致した。この取得・数値不合格判定・全コード検査・証拠固定の単位は完了である。全体目標は継続し、同worktreeを変更せず次のexact費用分解と出生経路の残検査へ進む。

### 2026-09-28: exact費用入力と出生取得の事前固定

exact費用分解は旧D-M／D-4の5 familyについて、440 Hzと各lifecycle ordinal 0／1876／3752／5628／7503を機械選定した。重複を統合した29入力を各16対、非計時先行と計時先行の交互順で測る。旧封印oracleとの交差は11入力である。計時境界は両方とも関数呼出し・unwrap直後までで、関数内のローカル破棄を含み、返却BodyFootprintの破棄を含まない。template cloneは通常代謝workerのper-job費用へ足さない。登録文の失敗保存契約をpartial rawとnonzero process statusへ正確化してから最終入力を固定する。数値計測はまだ開始していない。

出生版の中間source468項の全Rust検査は04:35:13 JSTにexit0で終了し、43群・1,382 pass・0 fail・49 ignoredとなった。固有logは3,736,417 byte、NULなしで、対象source全項のcapsuleが保存されている。その後に通常配送harnessと独立検算用の診断を追加しているため、この中間結果を最終sourceの検査結果へ流用しない。新しい取得案は[非同期出生の通常配送草案](body-fitness-async-birth-acquisition-draft-20260928.md)に分け、9 scene・ON/OFF・a/bの36本、30秒、狭域と55–8000 Hz全域、最新親energyを持つ非遺伝respawnを含めた。構文・IRのcompile-onlyは9本通過したが、通常配送の成功・資源判定は未取得である。

OFF不変対照用に、旧async-resourceの封印済みsource442項からrelease rendererを再構築した。全項のbuild前後hashは一致し、再構築binaryはSHA-256 `4e48af0ab9b29baeb7bed2e83b231320057f3efe38d979c0b1b805694f6feb0d`、12,888,424 byteである。出生worktreeの `target/old-off-reference/seal.json` にsource・toolchain・build log・binaryの系譜を固定した。これは旧時点に封印したbinaryではなく、旧sourceから今回再構築した対照である。旧configには存在しない出生OFFキーだけを除き、他キーを同一にして比較する。

出生の取得器を `scripts/run_body_fitness_async_birth.py`、資源独立検算を `scripts/check_body_fitness_async_birth_resources.py` として隔離版へ追加した。旧代謝取得器のproc採取とwait4の同iteration契約を共有し、旧呼出しの既定値は維持した。資源検算も既存の系列・欠測検査を共有し、出生workerと代謝workerの数を分けて検査する。関連29 Python検査が通過し、さらに固定planから科学checkerを削除してhashを再計算しても拒否する反例を追加して通過した。レビューに基づき、科学checkerと反例を取得planの必須依存に固定した。

R系では生存capacity3と、死者tailを含む物理Voice上限4を区別した。入力v2はmax_sourcesだけを3から4へ訂正し、Rhai9本・TOML18本はv1と全byte一致した。harnessも全hopの実Voice数上限を検査する。正式取得の判定条件は隔離版 `docs/design-notes/body-fitness-async-birth-acquisition-registration-20260928.md` にまとめた。旧新版のOFF音声同値対照は全9 sceneで代謝・action身体評価もOFFとし、R系の通常配送取得から分離した。ここでも新旧configは旧版に存在しない出生OFFキーだけが異なる。これらは取得前準備であり、通常配送の正例や資源条件の合格はまだ得ていない。

exact費用版の全Rust検査は04:57:13 JSTにexit0、43群・1,371 pass・0 fail・51 ignoredで終了した。主担当も固有logのNUL0、集計、SHA-256 `b646a9cf6a87c0c598591345af4505ca025606f85f293c86c0cbf25186160181` と同shell statusを確認した。final source447項・44,090,311 byteのlive/capsuleを全照合し、manifestは `7ef6d6f37692be954e9b52c87f9eacfdc0401c4fd018651ac4bb2cd0f785e9aa`。host対象子process0を確認して専有計時へ進んだ。

固定plan `47482f7730cde871d285ef316fc3de45df3f4bb86fbc824607152517b785707b` とrelease binary `ab0d751e7af663e0b04902bbb97b6e74fc3f7727de57c0795d0986f4f250a9da` による単回取得は04:58:55 JSTにexit0、14.52秒で終了した。rawは1,706,732 byte、SHA-256 `aab7999817823e1d543d8d78ff0d143fbe2d0c51b050ea88004db1a7882ca065`。固定Python監査 `86ce055349c2516f471d8d321a37c9a5b229716e908eb0b66f047d65f105c666` は29入力・464対すべてのbits同値と旧直接oracle11件一致、交互順、leaf合計とwallの関係を検証して通過した。主担当もrawの464一意対、密度・mass・frame数の完全一致、全leaf合計≤wallを別集計で確認した。

測定leaf総和の88.337%はNSGT、9.625%はTone render、1.464%はpeak抽出、0.536%は残りfrontendだった。query別timed wall中央値は13.5721–13.9384 ms。これは局所profilingであり、通常paced runtimeの期限や動的Hz有効率の合格ではない。次の調査対象はNSGT内部のring再構成・FFT・sparse積和・smoothingであり、inclusiveなNSGT値だけから内部のどれが支配的かは断定しない。結果本文と最終索引の固定を進め、出生版は次の最終Rust検査枠へ進む。

次のNSGT内部費用分解用に、同HEADの新worktree `body-fitness-nsgt-cost` を作成した。exact版final source447項を全サイズ・hash照合して継承し、継承manifestは `2e6bf64138ec30de229a4103c56f83122dddf8c0169d17c20bf36c7e465e1f6e`、親manifestは上記 `7ef6d6...` である。現在のexact版結果固定を先に完了し、新版だけでring再構成・FFT・sparse積和・smoothingを計時する。入力29件・交互16対と数値bits維持の条件を引き継ぐ。新計測はまだ行っていない。出生版の最終全検査は別targetで進行中であり、新版全検査・正式計時とは直列に調整する。

exact費用分解の結果本文は `7656f7ab63a8c02a1fa3dbdfa619b158f02cb5a91065b971a3bf583800d72fcc`、最終索引は `ccc1851a6bd74326abfbed3d6f4a17755402c87286d10fb0c6a69005d4e0374c` として固定した。主担当も944現物・258,683,377 byteの全サイズとhashを独立照合し一致した。この局所費用分解の取得・数値同値監査・全コード検査・証拠固定は完了である。全体目標は継続し、担当2名は新NSGT費用分解へ移った。出生版は選択親自身のbody使用を含む17科学反例、関連4 Python suite計33 pass、独立read-onlyレビューに追加blocking所見なし。最終Rust全検査がfactorial区間で進行中であり、release構築・36本の正式通常配送・旧OFF音声同値比較は未完了である。main source採用・既定変更・commit・pushは行っていない。

出生R系の独立検算では記録pool各親の実energyとCDFを照合する一方、代謝source記録にis_alive／population IDがないため、pool外の全適格親を独立列挙した証拠にはならない。この境界を取得前登録へ追記した。Rust sourceは変更せず最終全検査を継続する。

### 2026-09-28: NSGT内側の忠実な計時と出生最終検査

NSGT内側の探針では、sparse積和とsmoothingを二つのpassへ分割する案を採らず、元のband loop・メモリ走査順・f32演算順を保った。四つの計時区間はring書込、ringからFFT bufferへの再配置、FFT、sparse積和とsmoothingを含むband loopである。690 bandごとのtimerは追加しない。各bandを独立計時する必要があるかは、この内訳が判明した後に判断する。局所検査はSine／Harmonic／Modalの最終footprint同値と連続job resetに加え、coherent／incoherent各72 hop、途中reset前後の全band出力bits一致を通過した。新しい正式計時はまだ実行していない。

出生の取得用最終capsuleはRust検査用468項へPython取得器・科学検査器・資源検査器・反例・登録を加えた478項で、manifest SHA-256 `c78dde91ad9f4f5253c759c23ff32c15089f3d0a3eaa2521cd57ec927abd1b42`。主担当がliveとcapsuleの478項・44,312,650 byteを全照合した。Rust468項は最終全検査用sourceと同一で、Python関連4 suite・34件はexit0（05:10:24 JST）。Rust全検査はfactorial394.03秒、代謝14.93秒を通過し、再生成経路を検査中である。新release binaryの完成後、固定済み9 sceneの旧新版OFF比較と36本の通常配送取得へ進む。OFF比較器 `target/run-off-parity.py` は入力・自身・旧新binaryのhashを各実行前後に照合し、全9 sceneのWAV byte一致と全失敗を保存する。まだ比較実行は開始していない。

I11の次の診断は[群への身体投影草案](../roadmap/temporal-dcc/i11-group-projection-draft-20260928.md)へ切り出した。既存shared groupの配分をprivate scanにも適用し、全体量と部分量を混同せず三つの座標を比較する有限再解析案である。単独Aで同じ入力を同じmaskへ通す一致は構成上の整合性であり、source identityや通常自己群除外の正例とは呼ばない。既存medoid/閾値の再調整、CDF変更、SourceRemovedによる通常runtimeの除外はこの草案の実装範囲へ含めない。既知の無音とunsupportedも分ける。まだ新しい再解析の入力・許容誤差を固定しておらず、数値取得・source変更はない。

出生の最終Rust全検査は05:18:18 JSTにexit0で完了し、43群・1,382 pass・0 fail・51 ignoredとなった。主担当も3,736,685 byteの固有log、NUL0、同shell status、canonicalとのbyte一致を確認した。log SHA-256は `3280d69a44c1c5df8d45191fd8abfbc946c963d6113cd92a09dbb0bd663b0459`。release rendererを `f9c28b3d938caa398ea02d546b8c3b930930781734382d8b719cc0df58cb3123`、正式配送test binaryを `1286a89a6d05ca642dc34be95c92612b100f4f2fb28b8a0501f745d45943609c` として固定した。36-case planは `8a3d3e0522769617c4c153b2789dc81444f0a374c96ff4302b55ebe8b6613b2d` で、主担当による全固定入力・source・binaryのload検査も通過した。

旧新OFF対照は05:25:08 JSTに全9 scene成功で終了した。結果 `target/off-parity-v1/results.json` のSHA-256は `224cc18e5afa0940bc706dd7f8f2d74a90018166d8df16de6087d6de96d9a983`。主担当も18 WAVのhash・長さ、48 kHz／1,440,768 samples、旧新byte一致を独立確認した。birth・metabolism・actionの身体評価をすべてOFFとしたrenderer対照であり、R系の代謝ON通常配送や出生ONの同値を示すものではない。正式36取得後は固定helper `801ff1d7bafac49917c58f756b5bd5c467fb22e780c1bedea20b628ccce3c219` によりON18本の密度oracle、全36科学監査、全体資源監査を順に実行し、失敗と欠測も分母へ残す。

NSGT内側計時版の全Rust検査は05:35:18 JSTにexit0で完了した。主担当が43群・1,372 pass・0 fail・53 ignored、NUL0、canonicalと固有log/statusのbyte一致を独立確認した。log SHA-256は `09281192a079f3e29c07af3a0faa85ec8b0219eabb5c9b7ab2ef7541ef03c93f`。対象のhost build/test子processが0であることを確認し、固定source451項 `71690cb24d878e034ad705b9765e243e1695c1ac07c8e6d155f6c1deaeb72110`、release test binary `8773a144385ac636d71b71508f68e90fc2df619747912b4f616e762441fe0452`、plan `2d447eeda8b61290df9fccf4eef3994ea4bf80d2d0306fcc948858cae5a9f4ef` による正式専有計測を開始した。

NSGT内部費用の単回取得は05:36:53 JSTにexit0、14.51秒で完了した。raw SHA-256は `b2573ce6f118bd095ae9f2376f147224c28196b830f14e54073e6bf203e43a78`、固定Python監査は `a8ee458156554bce96009356f7a2ab453854a29b6e3fd4ebe0ce1731a22cf722` で通過した。主担当も29入力・464一意対のreference/timed完全一致、内訳総和とinclusive NSGTの一致を独立集計した。FFT長16,384、690 band、coherent、sparse entry総数347,437。NSGT総和5.620秒の92.105%がband loop、7.013%がFFT、0.642%がring再配置、0.177%がring書込、0.062%が残差だった。band loopは計測全wall6.368秒の81.289%を占める。元の単一band loopを保持した測定であり、sparse積和とsmoothing個別の内訳や改修後の速度向上は未測定である。

NSGT重監査の終了後、出生の固定36-case planによる通常配送取得を開始した。session5891、取得器log/statusは隔離版 `target/async-birth-formal-v1/acquisition-run.log`／`.status`。この間は他の重いbuild・全検査・render・raw監査を止める。oracleと科学・資源監査は全36本終了後に行い、途中の失敗も保持する。

NSGT内部診断の最終索引は `0b4ea67c15722b0d0cca96ea4cf363bce243aac8b37046a4e890a53c5d9154eb`、本文は `fdae65d1ca80aff39d2169b7cdd532582b6ff0998b00561222cd346be24e6752` として固定した。主担当も501現物・114,919,120 byteの全サイズとhashを独立照合した。次のu32添字化は隔離実験草案に留め、まだsourceを変更しない。現計時のwallからband loopを引いた値だけでも平均2.568 msであり、旧D-Mの1.333 ms更新に対する高有効率を局所軽量化から推定しない。

出生36本の通常配送取得は05:56:46 JSTにexit0で完走した。主担当も36本のreport存在、36件の取得PASS、同shell status、log SHA-256 `6ac669b26b417678e9cad1e51a2b1d070c9324478fa186226301ac26bdd621e5` を確認した。固定postrunは05:58:58 JSTにexit1。資源検算は全36条件・18 ON/OFF組で通過したが、科学判定はOFF18通過、ON18全件未達だった。全36本のintegrity_errorsは空。ON16条件で合計20子が実出生したが次hop receiptのavailableは全件0で、`next_receipt_unavailable` を保持する。広域I-W4のON a/bは各3予約・3要求・出生0で、aは2失敗とFinish時未完了1、bは3失敗だった。出生数・初期選択数・member順の条件を満たさなかった。oracleは18実行がexit0だが広域2本は密度行0であり、非空16本の計4,004密度照合と区別する。受入基準を事後変更せず原因診断へ進む。

I11三座標投影の18合成反例を通過し、6報告・30依存物のplan `824ff7b76f9115f5cdc02d11152bd36a18303243b75d94964d1b8b8056cddaa0` を固定した。最初の再解析v1は欠測PCM digestのNone処理でTypeErrorとなりexit1、結果JSONなし。固有log/statusと使用source/planを保全し、欠測を音声同一へ置き換えないv2修正を準備する。まだ投影診断の数値結果や通常自己群除外の正例は得ていない。

出生postrunの独立集計ではCPU ON−OFF差最大1.059806 core、最大RSS差4,112 KiB、ON絶対最大59,008 KiB。科学checkerのF2最大差は2.9802322387695312e-8で、全36本にintegrity errorはなかった。密度再生成は非空16本・4,004行のRecipe/hashが一致した。postrun系譜索引 `2aa08bddfe86c14ec85a204c8630f10c4d97e6c5e4176f797f410e128fb6bdc7` は主担当も191現物・14,469,031 byteを全照合した。

出生次hop未達のsource診断では、通常のdecision_batchは保存済みlatestだけを読み、非同期出力の受理は後段render後のobserveに置かれていた。出生hopでは旧source集合をIdentityMismatchで拒否し、次hop冒頭でも更新前のlatestを読む。一方、配線fixtureはdeterministic=trueで同hopのreceive_one_offlineを完了してから判断していた。これが局所配線成功と通常配送全20 receipt未達の違いである。後続の受理累計増加は永久欠測でないことと整合するが、元の次hop冒頭条件を満たしたとは呼ばない。次版では判定前の非blocking受理更新で既着かつidentity・support・age条件を満たすbatchを使えるかを先に検討する。広域では約9.65秒の全bin要求がunsupported_densityとなったが、現記録に失敗Hz/index/理由variantがないため位置は断定しない。元の30秒・帯域・成功条件を緩めず、別版の診断を準備する。

I11 v1の失敗14項は `137674fbe8fa87066a0e670314039070fd895f9097bde97055f82cd217f89c12` のmanifestへ原byteで保全した。v2はsupported時の32-byte digestとunsupported時のnullを区別し、19反例を通過した。固定plan `b92f62dcd8c0ba73a978dfb065e84b41a03f61ed57d01dcb40f0fd7db68bca48` の6報告再解析は06:02:53 JSTにexit0、audit `1ad17ee37aa290080b6748647bf87e1bfb53a39002745f0d38165126fb60d741`、1,059,340 byteとなった。各a/bは一致。固有281組のうちAは95→三座標完備92・一致92、Bは91→完備87・一致0、Cは95→完備67・一致66。三座標完備のうち旧六座標も完備なのは各85／69／60であり、分母を混ぜない。Bの完備窓は共通PCM prefix15、初差を跨ぐ2、以後70に分けた。Cの24組は支持hop欠落、158 PCM hopはunsupportedのまま残る。A一致は構成整合性で、通常自己群除外は依然0である。

I11三座標診断の結果本文は `713e830c8bd81691d51686f761eca8c3a414d9bf3ae23677bd4f8f37b2031e18`、最終索引は `07cc5c5b18ceff9aedf5bebba256621a41a31fca6d4e5dfb7072b14c0f8ac6ae` として固定した。主担当も62現物・481,580,086 byteを全サイズ・hash照合した。v1失敗とv2修正を含む有限再解析単位は完了し、通常の自己群除外の受入とは分ける。

出生通常配送の結果本文は `af3d90d2f17c4e9e1afd74c95b629b1e33f705c81d29a2c69f0a2c2159d99eb2`、最終索引は `3b6ecf5ed022e2f9e56e9a346cd17fe69d16fdaf690b3ad3e073182f8357cdd7` として固定した。主担当も2,155現物・3,725,039,786 byteの全サイズとhashを独立照合した。この単位の実装・検査・36取得・全監査・陰性理由の診断・証拠固定は完了であり、出生の科学gateはON0/18の未達として維持する。次単位は別sourceで判定前の非blocking受理更新と広域unsupportedの位置診断を検証する。NSGT u32添字化は後順位の局所軽量化案とし、高有効率の成立と混同しない。全体目標とF5/F6/I11の独立受入は継続中。main source採用・既定変更・commit・pushは行っていない。

### 2026-09-28: 出生の判定前受理と広域失敗位置の別版

前単位は取得・監査・陰性理由の診断・証拠固定まで完了した。次は同HEADの隔離worktree `body-fitness-async-birth-receive` を作り、出生封印版の478 source・44,312,650 byteをlive/capsule双方に全照合して継承した。継承manifestは `2d93a2da63cba8e74842e5d90fc9ebd554f7f9e42b9fc8dfd7a0a30338b68807`、親source manifestは `c78dde91ad9f4f5253c759c23ff32c15089f3d0a3eaa2521cd57ec927abd1b42`。元27入力とmanifestもbyte同一で `target/inherited-inputs-v1` にコピーした。旧封印版は変更しない。

担当を判定前の非blocking受理、密度失敗entryの有界診断、登録・Python検算に分けた。受理更新は出生ONで実出生後の次advance前に限定し、reporter有無に依存させず、出生OFFのaction/代謝の受理時計を広く変更しない。元workerのidentity・birth sample・epoch・support・age・clock単調性を保ち、未着結果を待たない。失敗診断は要求配列のentry index・Hz bits・Unsupported variant・成功済み件数・全件数を残し、全tableの失敗を0密度や点評価へ置換しない。旧9 scene・30秒・36本の科学/資源判定条件を維持し、正式再取得前にはE-M-on-aとI-W4-on-aの同条件2本を別の診断分母として固定する。まだ新sourceの検査・取得結果ではない。


### 2026-09-28: 次hop受理の2件診断

隔離版 `body-fitness-async-birth-receive` の482ファイルを固定し、元と同じ30秒条件からE-M-on-a、I-W4-on-aの2件を別分母で取得した。source manifestは `b924fa1f1b587c27eb8c5846d066d8db4c44fd3ad2b4b83c92924a4698ac908d`、release lib test binaryは `a16a6b353699b9a3a6e668ae4c417354e56ba1b940ca4e06a72d53b016d1eebd`。E-Mはconsume 1、出生hop自己PCMゼロ、次hop開始receipt利用可能1、固定checkerの未達gate 0となった。空環境の正例であり、非平坦C・親energy選択の正例ではない。

I-W4は3予約・consume 0。ticket 1/2とも候補index 687（7845.2822265625 Hz）で `NoInBandMass`、成功済み687/全690となり、ticket 3はFinish時submitted=true/ready=falseで右打切りだった。このvariantは非正massと非有限mass/scanを含み、実質量ゼロや端bin脱落の確定ではない。2件のtransport/samplerは通過、postrunは広域の出生数・選択数・member順序の未達を保持してexit 1。正式36件の再取得やmainへの採用は行っていない。詳細は隔離版 `docs/design-notes/body-fitness-async-birth-receive-results-20260928.md`。新版全Rust suiteは診断取得・oracle終了後に実行し、43 group・1386 passed / 0 failed / 51 ignored、2026-09-28 06:46:19 JSTに同shell exit 0で完走した。canonical log/statusも一致した。次は入力不変の局所probeでPCM、NSGT power、ピーク抽出、mass/scanを分けて原因を識別する。


次の原因診断用に `body-fitness-birth-endpoint-probe` を同HEADから分離し、受理診断版482ファイルをcapsuleと全照合して継承した。継承manifestは `56eb29108bbf4737a69385cce2272cda5a7b66f522a3af468a8d14a2c53f6220`。Rust側のcfg(test)計測と登録・最小Python checkerを分担して準備する。対象は同じ予約jobの候補686–689と低域288、72frameのPCM/raw NSGT power/最終peak/mass/scan。旧失敗時のRecipe全体は未保存であり、別取得で実際のRecipeをcaptureして系譜を固定する。全peak抽出の独立再実装は行わず、有限性・非正値・局所極大・閾値までで原因を絞る。現診断版の全suite中は追加build・重い数値取得を開始しない。


受理診断版の最終索引は `body-fitness-async-birth-receive/target/async-birth-receive-validation/final-index-v1.json`、SHA-256 `a76aa26c21807efe5676219e88b78086a994901f0b037c73b962d91b4bb91696`。1070ファイル・242,061,553 byteをrootが全size/hash照合した。結果文書SHA-256は `8721ba7f6a1481123fb203714b319f08f1397448920cecdcc0a1e8907367fd8e`。旧receive treeを凍結し、新endpoint probe treeの局所compile/testを解除した。数値probe取得は新source/schema/checker/入力/判定条件の固定まで未実施。


### 2026-09-28: 高域端点の実Jobプローブ

`body-fitness-birth-endpoint-probe` の487 source・44,418,665 byteを固定し（manifest `3df97d837cab93f0666bb61ba58526371e89a344336929d267654089b494563e`）、同I-W4-on-aの実Job 2件を取得した。終了後、各index 288/686/687/688/689を72frameずつ解析した。release lib test binaryは `a81458e826e091f50a8cfe88225c5aad485033881c9c383a2ba21793a1665e85`、planは `e60e68b1c443f1400b7045c713ac876c847cf03c08eade33d07f6f27b857236f`。取得と独立checkerはexit 0、全10 Recipe hashと原本binaryのidentity/shape/hashが一致し、非有限値はなかった。

両ticketとも440 Hzと7788.8394 Hzでは身体密度が成立した。一方7845.2822 / 7902.1343 Hzは全72 frameでPCM非ゼロ、NSGT正power、閾値通過の内側局所極大が存在したが、最終peak・主観強度massは全frameゼロだった。7959.3931 Hzは端binだけが最大で内側局所極大がなく、同じくmassゼロだった。今回のNoInBandMassは非有限値や無音ではなく、ピーク選別後のゼロmassと識別できた。監査SHAは `6fc0aad11c672e1f55e3b715d734ae03375544ce8821708c5818fa941e41c676`。通常の出生成功・資源正例には算入しない。

詳細は隔離版 `docs/design-notes/body-fitness-birth-endpoint-probe-results-20260928.md`。固定sourceの全suiteは取得後に完走し、43群・1387成功・失敗0・ignored 51（同shell exit 0、2026-09-28 07:18:57 JST）。rootが全ログの件数・hash・NUL 0とcanonical log/statusのbyte一致を検証した。mainのテスト記録は置換していない。prominence以降は初回独立監査の対象外だったため、保存rawの左右谷・10dB判定を別登録/別script/別planへ固定してから検算する。旧source・旧判定は維持し、mainへの採用は行っていない。


保存rawの追加prominence検算も完了した。別plan `ae7aefe7136ecd9bf68c5cfaeab6c5efa0cf16c2a3f387c7ad197894c9df36ae` を実計算前に固定し、全720 frame・2200候補を照合、10 dB通過1895/不通過305/境界未解決0だった。index 687は3.44002–4.42113 dB、688は0.84218–1.10675 dBで全frame不通過。両者とも右側探索は最上端へ達し、右谷がbaseを決めた。689は内側候補化されない。追加audit SHAは `b498034e1ed3d677c834102b08887fef83a93e405eab83ec92744a7b163ab112`。現在の除外段は識別できたが、帯域外の谷、別解析範囲・境界規則の妥当性、全690候補/全3出生の改善は未検証である。

費用側は別worktree `body-fitness-nsgt-u32-index` を同HEADから分離した。既封印NSGT-cost source451ファイルをlive/capsule全照合して継承し、継承manifestは `54479243fb0bbb01c186f576af1607e834399dfbf716bff493e2c467c0e58190`（親source `71690cb24d878e034ad705b9765e243e1695c1ac07c8e6d155f6c1deaeb72110`）。usize→u32だけを候補とし、flattenや積和順変更を混ぜない。型変更前witnessと型変更後の同コードでindex/weight/output bitsを照合する準備を進め、費用比較は別binaryのold→new→new→oldという2対の順序反転を候補として登録する。元29入力・各16反復を維持するが、反復を独立campaignと数えず、起動/構築と内部72frame費用を分ける。まだ新形式の同値性・速度改善・採用を示す取得結果ではない。

端点診断版は全検証後に封印した。`target/endpoint-probe-validation/final-index-v1.json` は1109ファイル・212,499,976 byte、SHA-256 `91399f7b99261e5741ffcf15dfbe4161390a07c755c29007acc4ebf2d92d1f6e`。rootが全entryの長さ/hashを再照合した。隔離結果文書のSHA-256は`b5f1e4fc037dd6fb04356a03452aa1f4ec7f536a1fc19c3e21ffc939aa015365`。以後この隔離版は変更しない。次のNSGT u32 index比較は別worktreeで旧usize+witness版の固定から開始する。通常出生の受入れは未完了のまま維持する。

封印内 `target/boundary-semantics-next-design.md` の「prominenceの実数値・谷位置は今回未計算」は追加解析前の設計時点を示す。現在の結果は同版の結果文書と追加監査JSONであり、上記の通り687/688の10 dB不通過と右端までの探索を確認済み。封印済み設計文書を遡及更新せず、次の登録ではこの時点差を明記する。

NSGT u32比較の変更前baselineは、witness追加のみの旧usize source 451ファイルを固定した。manifest SHA-256 `893f8bd30161075097647a645129d6778fc66056e416417a6791f239c473b8e6`、release binary `a7288d863f98974da1f6d1c744a62c5f2e5c491c13781feed445548c64b7877d`。rootがcapsule全451件を照合した。局所sparse witnessのCenter alignmentは両PowerMode各347370要素で通過し、正式29入力のRight alignment・347437要素とは別fixtureとして保持する。型変更後の同値・全suite・正式ABBA性能取得はまだ未完了であり、速度向上は未確認。

### 2026-09-28: NSGT u32局所同値と追加帯域観測の準備

旧usize版とu32版の両release witnessで、Center alignmentの疎係数347370件（coherent/incoherent）と72 frame・frame36 resetのRT出力hashが一致した。新版binaryは`4ef9e9ef5d8d7eb1becc28e789bf703408eae11fc780f30f1b87bb53ae5a5e60`、sealは`5110a11003823f6f57331d5ec46715e98cb1bed37c38c563926698f0b863f790`。rootが新版451 sourceのlive/capsule全件を照合し、変更2Rust fileを確認した。tupleサイズ検査はcoherent 16→12 byte、incoherent 16→8 byteで通過した。正式Right alignment・347437件・29入力の同値/速度比較はまだ未取得である。

必須全suite初回は1256成功・10失敗・ignored 53、exit 101で終了した。指定TMPDIR `target/tmp` が未作成で、9件はENOENT、1件は既定設定fileが作られないという失敗だった。初回log/statusを`target/nsgt-u32-validation/fullsuite-v1/`へ保持し、sourceを変えずディレクトリを作成して`fullsuite-v2/`へ全suiteを再実行中。mainのテスト記録は置換しない。

高域境界は[追加帯域観測草案](body-fitness-boundary-observation-draft-20260928.md)を別に作成した。新worktree `body-fitness-boundary-observation` はendpoint版487 sourceをlive/capsule双方で照合して継承し、継承manifestは`b25d2ced11252f1b2c004b89dcb4b273d1f7d8a30bd997244dc282d3a613178e`。保存済みPCMの固定追加帯域観測をcfg(test)内で準備し、取得はNSGT比較終了後とする。旧690候補の支持、30秒内の全3出生、通常運転の受入れは未達のまま維持する。

NSGT比較の正式ABBA planは`body-fitness-nsgt-u32-index/target/nsgt-u32-validation/abba-plan-v1.json`、SHA-256 `2551997f96d394fb1c58917e71186e155ced074274299adcb8b528f505edbbb4`へ固定した。Python反例9件・静的checker・runner事前検査は通過し、出力root `abba-v1/` は未作成。fullsuite-v2は実行session 41458で継続中、factorial/代謝/respawn群を通過してselection群へ進んだ。全suiteと他の重負荷終了を確認してからrootが正式4 campaignを取得する。結果文書は隔離版`docs/design-notes/body-fitness-nsgt-u32-index-results-20260928.md`、封印補助は`target/seal_nsgt_u32.py`に準備済みだが未実行。

境界観測版のcfg(test) Rust probeとgeometry-only exportも実装・静的レビュー済み。旧保存powerと旧再生、旧再生と拡張prefixのbit照合を別計数し、全raw保存後に不一致を報告する。cargo・新しい数値取得は未実行。Python checkerは旧690の対照、旧duを保つ混合列、native拡張列を分けて準備中である。

### 2026-09-28: NSGT u32比較は主指標で退行

u32版fullsuite-v2は43群・1374成功・失敗0・ignored 53、同shell exit 0（07:53:22 JST）で完走した。rootがログSHA `5e68df74fc4b0c8325211c337a2e6f5057b5961539ec3f986f903c595398f789`、件数、NUL 0、canonical log/statusのbyte一致と別inodeを確認した。旧v1のTMPDIR不備による失敗記録は保持した。

正式ABBAは全4 child exit 0、全1856内部比較行の密度690 bin・mass・Recipe・frame数が旧rawとbit一致した。しかし主指標の元関数72 frame jobは旧中央値13.829/13.842 msに対し新20.857/20.877 ms。対応全928組・全29query・全5familyで新が遅く、合計比は1.511493 / 1.509993だった。一方、計時版band loopと計時版wallは全組で短縮し、二経路の大小関係が逆転した。process全体wall/CPUはほぼ同じ、RSSは新が約5 MiB低い。計時版の短縮だけを元経路の改善へ読み替えず、u32版を速度改善として採用しない。

正式audit SHAは`d61f3c72ffdf5a6b50a8322cb060434f64cb54794591ba3c9a27347767dc0e7e`。root/独立担当の全raw集計は一致し、偶奇順序・時計ラベル・返却tupleの誤対応は見つからなかった。費用逆転の原因は未同定。固定binaryの静的逆アセンブルは追加時間測定と区別して保存した。境界追加帯域観測は別のusize系統を維持し、u32の速度改善を前提にしない。

隔離版 `body-fitness-nsgt-u32-index/target/nsgt-u32-validation/final-index-v1.json` は1455ファイル・281,690,568 byte、SHA-256 `2220b7ca7b5fdf78a6bfcf372f5b0a08e81d5766ac85ed302d7003a0e7e7fef2`。rootが全entryのsize/hashを再照合した。結果文書SHA-256は`6738f199cb79131281986528a23aa4150627e8b2e01f369b9e0c377142d317a8`。以後この隔離版は凍結する。通常runtime・動的Hz更新期限・広域全690候補/全3出生の受入れは未完了である。


### 2026-09-28: 保存PCMの追加帯域観測と右終端の分離

隔離版`body-fitness-boundary-observation`で旧55–8000 Hz・690 binと拡張55–16000 Hz・786 binを比較した。固定source 500ファイルのmanifestは`4a4c95aadec8288d23eb4b0dda85c637c8fb8efecd9bcf3c33f4926e254f35d1`、正式planは`27935eb48c28859379b9c6eecaab8c8c0d157dc59c23047ef2ef6e43fccfa21e`。保存済み実Jobの10 query×72 frameを再生し、旧保存→旧再生と旧再生→拡張prefixは各496,800値がbit一致した。numeric fault 0、replay/checker/runner exit 0。旧中心Hz/log2は一致し、ERB幅はbin689だけ異なるため、旧公開密度を保つmixed列とnative拡張列を分けた。

mixedは2200候補中10 dB通過2187・不通過13、nativeは2776候補中通過2475・不通過301、閾値付近未解決はいずれも0だった。元候補687/688は両ticket全72 frameで追加帯域の谷と後続上昇を観測し、10 dBを通過した。689はmixedで旧端点として除外、nativeで各72 frame通過。440 Hzの固定index288は各65/72候補で、旧「frame内に何らかのpeakがある」という72/72集計と区別する。初回audit SHAは`74ca8be3ce60a30bab951622991a0ba04a0a8bcd5847a310a9511dcb519319a9`。

初回checkerの右端到達は真の探索打切りと同義ではなかった。元raw/source/auditを保持し、別plan `4662df405ea01be13fcd56ce6ad54fd2fa9c46dc7d7d49688df9a9fecae3529d`と合成10反例を固定して全4976候補を分類した。mixedの右端到達2066件はすべて真の打切り、nativeの2642件中1件は最終binで候補より高い値に達した停止だった。谷class表示・10 dB分岐の変更は0。追加audit SHAは`bd27659dc69dfd31a88fabab53850846ffa6cf72824664583fd639b5f7d95a08`、独立照合も一致した。

公開密度と候補を固定した右域延長ではprominenceは非減少となるため、有限観測での通過は下界として保持できる。真の打切りを一律不通過にせず、停止理由・実際の谷と後続上昇・10 dB下界を分ける。ただしnativeのERB幅/床変化、最終peak選択、身体mass、全690候補、通常出生、全3出生の30秒期限はこの保証外である。次は共有frontendで公開690 binと右guardを分離する最小契約を検討し、guardだけの強峰、単調tail、公開帯域のmass保存、内側候補とguard無効時の同値を固定反例にする。production変更はまだ行っていない。


境界観測版の全suiteは43群・1387成功・失敗0・ignored 53、同shell exit 0（08:24:45 JST）で完走した。rootがログSHA `af5ef9ee9966bfb4542010e2bc4706eac0b4b5440fa65baaabf68d166e7272a2`、件数、NUL 0、canonicalのbyte一致と別inodeを確認した。`target/boundary-validation/final-index-v1.json`は1127ファイル・223,682,384 byte、SHA-256 `6f0023a234c2caa134d5b43294dd8f4908d8bc7599268f83d403384663fb034f`。rootが全entryを照合し、この隔離版を凍結した。結果文書は隔離版`docs/design-notes/body-fitness-boundary-observation-results-20260928.md`、SHA-256 `cf3a1c6d5509c1fcd2c650fbba94c23eb8d8516f1b985b59fcdfceca3d1582eb`。main source・mainテスト記録は変更していない。


### 2026-09-28: 公開帯域を保つguard実装の分離

次の隔離版`body-fitness-public-guard`を同HEADから作成し、境界観測版の500 sourceをlive/capsule照合して継承した（継承manifest `5f10c3ef55d10766fddc940c4f130ac6e78644f3c5884de1159274d13890928e`）。旧du[689]とguard側duの混合をそのまま形状判定へ使うと偽の境界局所最大を作り得るため、前段結果文書の最小案を修正した。新案では観測全域の連続したERB幅で形状を判定し、公開690 binだけから床・候補・候補順位を決める。質量は旧公開ERB幅による別densityを使い、形状floorのmaskに通る公開binだけを従来のf32演算順で積分する。旧候補/最終peakの完全保持は要求しない。

RT NSGTは一つの拡張kernelとFFTを使い、公開space/周波数/process_hop出力を690 binに保つ。crate内の全観測値を共有frontendへ渡し、通常知覚・代表身体の両経路を同じ規則へ結線した。公開形状だけでprominenceが通過する内側候補は保持し、guard依存の追加候補と公開端点689には探索範囲内の谷と後続上昇、10 dB通過を要求する。true right censorの一律拒否は行わない。guardの強峰を公開massへ足さない。

実装と保存PCM probeを分担し、相互read-onlyレビューを実施した。Clippy初回は旧frontend入口2件がテスト専用になったdead_codeで失敗し、cfg(test)へ限定して再検査を通過。all-targets初回は新テストの借用中accessor呼び出しがE0502となり、assert順序を修正して再検査を通過した。失敗log/statusは保持した。release buildと局所反例・必須全suite、10固定PCMの正式取得は継続作業であり、まだ身体mass改善・出生受入れを示す結果ではない。main sourceは変更していない。


公開guard版のrelease binaryを固定した（59,558,352 byte、SHA-256 `deec856806abac4ee0c6159615e9cf10b5a133308c09c5d910eef596be4c306a`）。局所releaseフィルタ13成功・失敗0・ignored 1、Python反例19成功。source 506ファイル・44,699,655 byteのmanifestは`559e7c39622e9129c989781fca5b0255dd744b4f5a67a3633f3836fbba772f2f`、保存PCM取得planは`5ab249472f43c7fe532b80efe12dadc4879e783befbd6ff943ee581464795ff8`。静的checkerとrunner事前確認は通過し、実PCM取得は未実施。

必須全suite初回はdefault並列で1288成功・1失敗・ignored 54、exit 101（08:47:23 JST）。非同期actionの回復テストが30秒待機上限へ達し、後続統合群は未実行だった。初回log SHAは`181b76224bd6148ac829ce20b97977e078ac3701892f2150e4b5bb2a72b2c024`、固有原本を保持した。同sourceの当該単独テストは2.16秒で成功。sourceや30秒閾値を変えず、`RUST_TEST_THREADS=4`の全suiteを固有`target/public-guard-fullsuite-v2/`へ再実行中である。これは機能検査の実行負荷を制御する条件であり、通常の出生・資源ゲートを緩めない。default並列での失敗原因はまだ断定しない。


次段のI-W4単case取得を、保存PCM結果を見る前に固定した。`body-fitness-public-guard/target/public-guard-birth-next/plan-v2.json`のSHA-256は`2ed5467bea189c8d1b8668a10f682b96885e9bff4bc3eb5d7f7ddb2aae11ace3`。旧scene/config、seed 7、30秒、690候補、3出生と次hop receiptを維持し、全2,813hopの予算、underflow 0、単ONのRSS上限を別判定する。binary/sourceは10 PCM planと完全一致、全506 live/capsuleと親planの14資産を照合する。runnerの合成陰性5件とcheck-onlyは成功した。初回plan-v1は継承manifestのschema誤読により静的照合で失敗し、原plan/log/statusを保存してv2へ修正した。paced取得はまだ実行しておらず、単ONから36case・18対の相対資源ゲート通過を主張しない。


### 2026-09-28: I11の次接続を二段joinへ具体化

`docs/roadmap/temporal-dcc/i11-causal-group-join-draft-20260928.md`（SHA-256 `0a443dbcfdece60eda500173aa847a2ecb2ad8df9dee2ecfde4dcc037731887d`）に、封印I11 sourceと未封印public-guard/SourceRemoved sourceの由来・統合対象を記録した。I11判断は描画前、SourceRemoved取得は描画後なので、producer同hop比較と後続判断のas-of参照を分離する。未対応epoch・未来結果・欠測・他者音残存をunknownとして残し、report-onlyの対応付けからhard自己群除外へ自動昇格しない。新規scene/seedと非干渉取得は草案であり、実装・数値取得はまだない。

I10の現文書は固定development素材4,936実音対で身体候補energy予測の一致を記録している。未知素材への転用は未確認であり、I11通常自己群除外0と区別する。以前の限定記録を根拠にI10全体を「実身体正例0」と一般化しない。


全suite v2は`RUST_TEST_THREADS=4`で43群・1397 passed / 0 failed / 54 ignored、exit 0（09:20:44 JST）で完走した。log 3,740,332 byte、SHA-256 `41186468701e72c4d20e5188c5046b836928405509ed61d2d314106ecbdaa6e0`、status SHA-256 `3efef3eb0e5293b8b941bb1da9b3743e32f6f1eb2460ed29a2924533a13a2ab2`。rootでNUL 0、canonical byte一致・別inodeを再確認した。初回default並列での30秒待機失敗は残す。

固定10 PCMの正式取得を一度実行した。Rust replayはexit 0、720 frameを保存し、旧保存データの再現、旧公開powerと新公開prefix、通常知覚と代表身体の公開power/scan/massが全frame bit一致した。一方、独立checkerは公開mass検算でexit 1となり、auditは未完成である。原因は`audit_frame`が既にshape densityへ変換した配列を、powerを期待する`classify_frame`へ渡した二重変換だった。固定source/checker/plan/rawを変更せず、別の監査版と非一様ERB幅の反例を準備する。閾値の緩和やPCM再取得は行わない。

rawの直接集計では新実装の全720 frameで正のmass/scanが得られた。元index288は両ticketとも旧/新72 frame、686は旧70→新72、687/688/689は旧0→新72である。選択bin集合の変化は計436 frame。これはrawの出力事実であり、独立形状・質量検算の完了とは分ける。出生取得は監査完了まで保留する。


保存PCMの訂正版監査は、修復plan `c286c27a3187da6ca494af9c17de469b070da59f09cf53b701121935f51146e9`で同じ32 rawを再監査しexit 0となった。不一致0・未解決0・正値720/720 frame、選択bin変化436。audit SHAは`59a5091d902e09a9dd6f2657d800a439843b42c20944b70437b39e17464111dc`。別担当の独立点検でも選択924 peakの支持違反0、公開mass相対差最大約2.06e-7。旧checker失敗と修復static初版の配置パス失敗を保存し、production source/binary/rawは変更していない。

同じ固定binaryでI-W4-on-aを一度実走した。取得と単case資源は成功、oracle exit 0、科学checker exit 1。ticket 1の690候補表はすべて準備され、oracle 690行のRecipe/densityと一致、F2差最大0だった。ただし最初のReadyのworker computeは20.752728秒。bin167から184.24945068359375 Hzを選択後、実周波数の密度をserial 4で要求し、ticket 2の全表待ちを含むqueueに残った。30秒Finishではticket 1の追加密度とticket 2が未Ready、ticket 3未submitted、出生0・次receipt 0。unsupported 0を未完了の3ticket全支持へ読み替えない。全3表・3出生は不合格である。

全2,813hop・underflow 0・hop超過0、hop最大6723.429 us、whole-process 1.39564 core equivalents、MaxRSS 80,720 KiB。単ON資源は通過したが、OFF対照・36case相対資源は未評価。runnerの`oracle_integrity_pass=false`は全3表条件との合成であり、取得済690行の検算不一致ではない。結果SHAは`04b95f6a882a05fee622122da172b56708701ead812ddfcb4da304e76e83d990`。次は全表計算費用と追加密度のqueue遅延を分けて対策を検討する。main source・既定設定は変更していない。


公開guard隔離版を封印した。`target/public-guard-validation/final-index-v1.json`は1191ファイル・237,321,718 byte、SHA-256 `23035ef81d75892561c45269cdd7cb7d74200dbef09bc2a210efc0e4ef473da7`。rootが全entryのsize/hashを再照合した。結果文書SHAは`36555c0373b93d04d65c5d08b5f6b9fb12c5f7e598594045f4fda9bcefe63c7d`。元失敗、訂正監査、出生不合格、独立690行照合を共に収録し、以後このtreeは変更しない。次段案`target/public-guard-birth-next/next-step-draft.md`は全690×72 frame準備費用と実Hz追加密度のqueue優先を分ける。次版実装・取得は未着手。main source・mainテスト記録は変更していない。


### 2026-09-28: 出生queueとNSGT費用を別隔離版で実装中

封印public-guardの506 sourceをlive/capsuleで再照合し、同HEADから`body-fitness-birth-cooperative`と`body-fitness-nsgt-runs`を作成した。継承manifest SHAはそれぞれ`032d84738af7d8d7a511787394e6b456977b14947ec6c31a7f655ce711448737`、`19066bc3a7c8a6ac3aeee580ca00e356f59ea3ca5e3d4c5c3ac3beba683afa70`。親の失敗・数値・source・最終索引は変更していない。

cooperative版はworker 1本、full input/output各bounded 1を維持し、exact local用bounded 1を追加した。候補1件単位でfull/localを交互に進め、各jobの候補順・72 frame・Recipe・Hz bitsを保持する。レビューで新localの受信ごとに優先flagを戻すとsingleton localの連続でfullが止まる問題を発見し、実行後の交互順だけを保持するよう修正した。bounded outputへdummy completionを入れてlocal1の完了送信を止め、local2/fullを待機させる実scheduler反例で順序を検証する。Clippy・all-targets・fmt、async_birth局所16件は成功。計算時間と他job実行を含む経過時間を別記録にした。全suiteは4threadsで開始済みであり、正式paced取得はまだない。

nsgt-runs版は、固定release binaryの疎積和に各entryの境界検査が残ることを逆アセンブルで確認し、安全な連続index群ごとのslice検査へ変更した。usize tupleと積和順、offline解析を維持し、unsafeは導入していない。Clippy・all-targets・fmt、両PowerMode/Center/Right/72frame/reset/clone/prefixの局所検査2件が成功。全suiteを4threadsで実行中。性能は未測定であり、旧u32実験の退行を無視した採用はしない。次は親の完成済み1表690候補を既存の通常oracle入口で旧/新/新/旧の4 process比較し、全density/Recipe bits、外側wall、wait4 CPU/RSSを照合する登録を作った。各対のCPU比≤0.95はこの固定素材での採用候補の基準であり、30秒3出生や36case受入れを意味しない。

両新treeの保存PCM checkerには親の二重density変換を訂正し、非一様ERB幅を通るframe全体の反例を追加、各20件成功。新出生runnerでは初期3全表と追加local要求を分離し、全oracleのintegrityと全3表の成立を別に検査する。まだsource/binary/取得planの最終固定前である。機能suiteは32 logical/16 physical CPU、各4threadsで並行実行する条件を記録し、正式な性能・paced取得は全heavy job終了後に行う。main sourceとmain test_statusは変更していない。


I11は新隔離版`i11-causal-group-join`を作成し、封印I11 archiveの518ファイルを全byte/hash照合して展開した。継承manifest SHA `d7cd628aa45523bf6271ae64488c0cee5f731f4ccfe685b8957fc76f70ef0b1e`。public-guardからSourceRemovedのcore2ファイルとobserverを選択的に移植し、I11判断・CDF・group・RNGは維持する。guard DSPやF4出生全体は移植しない。共有producerとSourceRemoved提出PCMのdigest、描画後accepted batch、判断前as-ofの時刻を別eventで報告する配線を実装中。epochの数値0を同一性とみなさず、未来結果・欠測の反例と独立join checkerを準備している。source採用元は新treeの`target/integration-lineage.json`に記録。cargo局所検査の準備段階で、数値取得はまだない。


2026-09-28の独立レビューで、NSGT比較runnerの`assert __debug__`自体がPython `-O`で除去される問題を発見した。module冒頭の明示的例外へ置き換え、generatorも同じ条件で拒否する。全690行の不足・並べ替え・個体違い・非有限値・byte破損と最適化モードを含む5テストが成功した（`target/nsgt-runs-abba/checker-tests-v3.*`）。出生監査は初期3×690表の完備、初期2070行のoracle一致、localを含む全取得行のintegrityを別gateとして全て必須とし、反例11件が成功した。正式取得はまだ実施していない。

I11の初読では、同じsample時刻でも描画後のacceptedを描画前as-ofへ混入できる点と、同じgroup handleを持つ別Voice履歴を拾える点を指摘した。accepted sequence・epoch・frame・receiptとreport出現順の照合、provenance ownerに対応するsourceへの限定を追加中。group履歴対応とbody binding/CDFまでの完全joinを区別し、未実装部分をunknownとして残す。


両候補のsource capsuleを固定した。NSGT版は510ファイル・44,729,526 byte、SHA-256 `26a857e1d898e1a2b5ac9307a2c09ef70d82ea1c28bcc9e56cdfbd370b3e9983`。協調出生worker版は510ファイル・44,760,039 byte、SHA-256 `10205bd7afdeeb151c0b6accc317d5fc295c06cdb19c4b506835696019144349`。親506に対する差分も保存し、前者はRust2ファイル、後者はRust1ファイルと、共通のchecker訂正・各登録/runnerだけである。全suiteとrelease buildは未完了なので、source固定を検証完了とは数えない。

I11のSourceRemoved局所12件、因果join checkerの12反例が成功した。履歴各行にもas-of前のevent位置とaccepted sequence上限を適用する。scene compile・fmt・all-target完了後に全suiteを2threadsで並行実行する条件を追加した。既存2suiteは各4threads、host32logical CPUで機能検査としての並行であり、性能取得は全heavy終了後に限定する。


I11統合版の必須全suiteは32群、1197 passed / 0 failed / 36 ignored、2026-09-28T10:20:04+09:00にexit 0で完了した。log SHA-256は`7cf73ccf51e807b4b6b431ef83dce9d0cb48f4b99bb5f1facfd620e4776e4793`。rootでNUL 0、canonical logのbyte一致・別inode、終了statusを確認した。release renderer構築と取得plan固定へ進む。これはreport-only配線の機能検査であり、正式S/M取得、body/CDF完全join、自己群除外受入れは未完了。


I11はsource 537ファイル（manifest SHA `54898039746251802854647ebf8bf5c766583861629985cb9cf665552d730284`）、release renderer SHA `3396670abc14a55ec1370df52265b00a5736efa8c7942809b8f82b86bab62651`、正式8case plan SHA `150a64762bbeeb81d0096708cd6c46384915b8815eef4676e6d1d54c2b74e7c0`を固定した。rootが18資産・537 live source・archive全537格納fileを照合し不一致0。旧plan `8ab16d...`はpost-acquisition argv具体化前の草案として別保存した。Python検査は取得器・非干渉比較を含む69件成功。

取得順の運用を見直し、I11のoffline 8caseはF4機能suite/release buildと並行して開始する。I11は決定論的なWAV/判断同値と因果参照を調べる機能取得であり、wall時間や受理時刻差をRT性能の結果へ転用しない。NSGT ABBAおよび30秒paced出生の性能・資源評価は、I11を含む全heavy job終了後に単独実行する制約を維持する。source/config/閾値・case集合は変えない。


協調出生worker版の必須全suiteは43群、1400 passed / 0 failed / 54 ignored、2026-09-28T10:39:02+09:00にexit 0で完了した。logは3,741,327 byte、SHA-256 `5b81a27a9d80464b3c77b75b18eb16f97d2dc75f529c54644ac9c963a64d7266`。rootでNUL 0、canonical byte一致・別inodeを確認した。release binaryの構築へ進む。並行機能suiteの所要時間を性能結果へ転用しない。


NSGT連続index群版の必須全suiteも43群、1399 passed / 0 failed / 54 ignored、2026-09-28T10:41:12+09:00にexit 0で完了した。logは3,740,783 byte、SHA-256 `1ac7beedbaa7156de50e4cf2c954b5238704c2f9f66d6b4292bd118a3082100e`。rootでcanonical byte一致・別inodeとNUL 0を確認した。I11の正式8renderも全exit 0、全case前後で固定資産・source一致。I11の科学判定は取得後17検査を実行中、F4の性能・paced取得はまだ行っていない。


両F4候補の保存PCMを新binaryで各一度取得し、10入力・720frameのRust replayと独立checkerが全てexit 0、不一致0・未解決0・正値720/720だった。NSGT ABBAは固定旧/新/新/旧4processを全heavy終了後に単独実行し、全690候補のRecipe・density binary/JSONが完全一致した。しかしwhole-process CPU時間は旧A20.867668秒、新A28.967105秒、新B29.161941秒、旧B20.872057秒、対の新/旧比1.388133/1.397176。事前基準≤0.95を満たさず、速度採用は不合格。境界検査を減らす設計から速度向上を推定しない。原因は未確定で、mainへ採用しない。

続けて協調worker I-W4-on-aを単独paced取得した。初期1表690候補とexact-Hz追加1候補の計691行がoracleと一致し、integrity error 0。full Readyはsample1000960、compute19.848551秒。local要求は同sample、local Ready/実出生はsample1004032（20.917333秒）、local compute28.824 ms。実Hz184.24945068359375と実Recipeを保持し、出生1・自己PCM1・次hop受領1を確認した。残り2個体の全表は30秒Finishで未Ready。3出生・全3表・36case受入れは未達であり、旧親の出生0から単ON一回の出生1への前進と区別する。

単ON資源はtransport/hop通過、whole-process1.419739 core equivalents、MaxRSS80,952KiB。OFF相対資源は未取得。今回の出生は初期配置で、親選択のenergy因果経路の証拠へ転用しない。元source/plan/rawは固定のまま独立監査と最終索引を作成中。


I11の固定8取得は全exit0。S/MそれぞれON-report2本・ON-report無し・OFF-reportの4 WAVが全byte一致し、report付き3本の判断列は各28件で同一だった。v5/windowとstage2/CDF費用検算は各6reportすべてexit0。新causal checker v1は実際にはflatなparticipation_decisionをnestedとして読んで4件失敗し、別copy v2で1行修復した。v2ではf32の異なるJSON表記を数値列の完全一致で比較したため偽space_mismatchとなった。v3のf32 bits正規化はこの表記差を除いたが、各hopへgeometry配列を要求する追加条件が初回だけ配列を配信するproducer契約に反した。v4では初回geometryを継承し、SourceRemoved全binと、後続で明示されたgeometryにはf32 bits完全一致を要求する。1bit改竄・非有限・後続正常省略の反例を含む15件が成功した。各旧checker/plan/失敗結果を保持し、Rust/binary/rawは再取得・変更していない。

v4は同じ4 ON-reportすべてで1314生成hopのうち1291投影成功、assignment_absent3、no_source20。単独Sの判断28件中11件、重畳Mでは12件に判断前の同owner・同群履歴が対応した。残りはas-of SourceRemoved取得までは確認できるが選択群履歴対応に至らない。body/Binding/CDFをまたぐ完全joinは引き続きunknown_not_implementedで、自己群帰属の受入れではない。

新固定M sceneでは、既存I11規則の自己群除外がnow387584・voice1・群(bus0,epoch0,generation63)で1件観測された。OFF-reportにも同じ判断があり、新observerによる判断変更ではない。従来の『通常自己群除外0』は過去の取得素材に限った記録として更新する。この群とownerの判断以前の投影履歴は232hop、最後のend387584でmixed0.03268212802346264、source-removed0.000525074679315833だった。これは保存powerと同じassignment rowの投影値で、所有率や自己群の物理的同定を意味しない。完全join・物理帰属の受入れは未達のまま、独立結果と最終索引へ保存する。


三つの隔離版を封印し、rootが索引全entryのsize/hashを再照合して不一致0を確認した。NSGTは1136項・298,229,673 byte、index SHA `0179d68ea15dfe8b40c5f2cb0a923dbed316387960486e278533262a721c694d`。協調workerは648項・189,315,581 byte、index SHA `a08dd0ace4c9eccff0d74650dd0f086bfc0000a9d35c04e7f1af0243714b3f6a`。I11は720項・1,184,425,422 byte、index SHA `8c79033ee1d73d7e3dc2c106ca6b1b2b4087918e0735b25f6d53c7696cd7cf3f`。索引作成中logを初回索引へ含めた不整合もattempt原本として保存し、最終索引は閉じた証拠だけを対象にした。

結果文書は各treeの `docs/design-notes/body-fitness-nsgt-runs-results-20260928.md`、`docs/design-notes/body-fitness-birth-cooperative-results-20260928.md`、`docs/roadmap/temporal-dcc/i11-causal-group-join-results-20260928.md`。rootはさらにI11のM-on-a、end387584、owner1/gen0、群63について、checkerを使わず保存power690binと実rowから2投影値を再計算して完全一致を確認した。これは1時点の保存power投影の点検で、物理帰属の独立検証ではない。root検証記録は `target/parallel-continuation-20260928/` に保存した。

次の未完了事項は、全690候補表の費用対策と3出生/30秒・36case資源受入れ、I11のbody/Binding/CDF完全joinと自己群帰属の独立証拠。main source・既定値へ隔離版を採用しておらず、goal全体の完了にはしない。


### 2026-09-28: 全候補費用の分布とI11全判断照合

協調workerの全表19.848551秒から、同費用の3全表を約29秒の準備区間内に終える目安は約2.06倍のthroughputと算出した。これは未完了ticketの同費用仮定であり、実測した3表費用ではない。旧計時から見積もるtemplate clone全廃の節約は約5.3 ms/表で、必要な約10.19秒短縮の主対策にならない。追加workerも同版ON/OFFのCPU予算を通した結果はない。[費用調査](../../target/body-density-cost-next-20260928/analysis.md)の6参照現物を主担当も全hash照合した。

封印public-guardのkernel sourceを直接取り込むstandalone constructor probeで、48 kHz、nfft 16384、hop 512、Right/Coherent、55–16000 Hzの786帯域を再構築した。全center Hz/log2 bitsが封印geometryと一致した。公開690帯域は347,437 entries、高域96帯域は444,414 entries、合計791,851 entries。全31,926 run、run長中央値13。連続4帯域のlane利用率は90.56%。主担当も786行のrun長合計、4lane分母、6現物hashを独立再計算した。高域96帯域だけで積和entryの56.1%を占める。これは構築時の分布であり速度測定ではない。[機械語・分布調査](../../target/nsgt-next-cost-20260928/analysis.md)に保存した。単純な4積先計算はcodegen上scalarのままで採らず、独立帯域間のSIMDを数値同値の次候補とした。

I11は保存済みON4取得の全112判断・5,256 producer hopを[事後照合v4](../../target/i11-complete-join-20260928/results-v4.md)した。各28判断/1,314 hop、整合エラー0、完全対応0、unknown112。消費GroupPrototypesと独立な同hop報告の直接一致はS各4、M各3の14判断。Mの自己群除外は各1判断でbody Record・Binding・assignment・CDF・latest accepted・窓内の投影履歴が個別に通過したが、その消費snapshotの直接報告は欠測。route/birthと予報発行の独立原点も別の未取得証拠として明記した。

v1の旧audit schema誤読、v2のreport直列化順誤判定を保存した。runtimeは判断に渡した同じtemporal_framesを後置報告するため、v3では同hopのasof間だけ後置snapshotを許す。旧v1/v2の入力testを元bytesへ復元し、主担当もv1/v2の各24入力とv3の25入力のhashを全照合した。取得後に条件を設計した探索解析であり、事前登録の科学的合格へ読み替えない。mainのRust本体・既定動作は変更していない。

全producer分母に対するaccepted/shared末尾の欠落を拒否する検査をv4に追加した。v3を保存し、別planの25入力を固定して4件を再解析した。12反例が通過し、結果件数は不変。別担当の独立レビューでも追加blockingなし。主担当もv4の全入力hash、各28判断/1,314 producer、整合エラー0とunknown28を確認した。


独立4帯域AVX2のstandalone積和prototypeは、封印kernelの786帯域×6入力種の4,716実band比較と、末尾padding・空/異長laneを合わせた4,740 laneでreal/imagの全bit一致を確認した。非零subnormal、signed zero、±MIN_POSITIVEを含み、不正FFT indexはgather前assertで拒否する。対象機械語はpacked積4・gather2、FMA0。最初のtiny入力がゼロへunderflowした試験は保持し、非零assertを加えた最終logを別保存した。[数値proof](../../target/nsgt-next-cost-20260928/band-simd-proof-v1/proof-summary.json)のsource/binary/依存rlib/参照sourceを含む11現物hashを主担当も照合した。これは積和の限定同値証拠であり、実Tone・72frame・全密度・通常runtime・速度の試験ではない。次は隔離版で同じ算術を本体へ接続し、実入力の全bit同値を確認した後に専有ABBAへ進む。

I11事後解析の最終索引は71成果物・7,113,520 byte、SHA-256 `758c1aff9f7cf95bcb422c7781b91114e8455d53cb68524d3af27f8327481895`。主担当が71件全サイズ/hash一致を確認した。今回の変更は調査・検査artifactと計画記録のみで、mainのsrc、既定動作、公開technoteは変更していない。


### 2026-09-28: SIMD本体接続とI11原点記録の隔離実装

前段を数値proofと欠測特定による前進と分類し、二つの新worktreeで継続した。`body-fitness-nsgt-band-simd`は協調workerの封印510ファイル（親manifest SHA `10205bd7afdeeb151c0b6accc317d5fc295c06cdb19c4b506835696019144349`）、`i11-origin-records`は因果join版537ファイル（`54898039746251802854647ebf8bf5c766583861629985cb9cf665552d730284`）から継承し、全現物hashを照合した。gitの基準commitは共に `06a4772`。初回のsource集合確認は親側のPython bytecodeを含めて不一致となったため、`__pycache__`だけを集合対象外として再確認した。原本の封印sourceは変更していない。

NSGT担当は独立4帯域のcoherent AVX2積和をRT kernelへ接続する。転置係数はconstructorで一回作りArc共有し、hop中のallocationを増やさない。index検証、非AVX2/incoherentのscalar経路、各帯域内の加算順、scale/norm/smoothing、公開690帯域とguard96帯域を維持する。別担当が保存PCM/全690表の同値検査と旧・新・新・旧の外側CPU計測を登録する。全suiteと実入力同値の後に性能計測へ進む。局所proofから速度成立を推定しない。

I11担当はpublishされたsnapshotのruntime取込原点、実出生のRuntimeEvent原点、判断前のVoice/body-generation/current route原点をreport-onlyで加える。snapshot内forecast/CDFの原点と消費値を別eventで照合する。current routeは実Toneのroutingや音響寄与と同一視しない。OFF/reportなしの判断・音声を対照する。既存S/M入力・モデル・seed・閾値は保持し、物理source帰属を記録追加だけで成立させない。機能検査は並行可、性能・paced資源取得は全heavy終了後の単独実行を維持する。


NSGTのsource capsuleは517ファイル・44,810,529 byte、SHA `81fb4a2b00b32512146790f4efa333fd324fd7e5e3cb3d7556032118449c43ec`。親とのsrc252path集合は一致し、差分はnsgt_rt.rsだけ。局所2検査、fmt、標準Clippy、all-targets checkを通過し、全suiteを4threadsで実行中。factorialの無出力区間では同session72441を維持し、host上で新しいrenderer子processのCPU稼働を確認した。停止や失敗と決めつけて再実行していない。

I11原点記録版は全suite32群、1199 passed / 0 failed / 36 ignored、2026-09-28T11:53:40+09:00にexit0で完了した。log SHA `5fdd398e68480d27032411ac37299eb9e4f87c9fc4e866b8c082cedcfd4549b8`。主担当もNUL0、canonicalとのbyte一致・別inode、終了statusを確認した。新原点のJSON表記は型付きSerializeで判断側と一致させる。Snapshot取込後はMutexGuardを解放してから報告し、出生原点は同hop判断行より前に出す。原点単独のchecker成功で止めず、前段の全body/Binding/CDF/群履歴joinへ接続して正式8取得へ進める。物理的source帰属は別の未証明条件として残す。


I11原点版の統合checkerと登録を主担当が読取レビューした。親v4の全producer対応・Binding/model/閾値・半開forecast履歴・CDF検算を残し、消費snapshotの欠測判定だけを取込原点の全判断照合へ置き換えた。最新取込、判断より前のevent順、実出生と初観測出生の区別、現行routeと保持中Toneのrouteの区別を確認した。Python関連94検査が成功。543ファイル・44,501,860 byteのsource manifest SHA `d66edcd97c53570544821fe4c474965cca601ef956a46ed094cf7cc6ec04ce49` を固定し、主担当も全543現物のサイズ/hash一致を確認した。release build後に旧S/M全8条件との音声・判断同値と新原点の全分母を検査する。限定された保存入力joinとI11全受入れは区別する。


I11原点版のrelease binaryは12,218,944 byte、SHA `1787005fd9ced847ebb622e0870d58ed2590ef0a3e74d5eba4689d0f01354379`。正式plan SHA `8494ea8ac1597da6290033e30b7358fc19d4e2dec2f1397dfe08118944adcff4` の37資産とsource543件を主担当も照合し、旧8条件のscene/config bytes・seed・case順一致を確認した。全8renderがexit0、毎case前後の入力hash一致。独立比較でも全8 WAVとreport付き6本各28判断が旧版と完全一致した。記録は `target/parallel-continuation-20260928/i11-origin-noninterference.json`。

重畳M-on-aのnow387584・owner(1,0,1)を主担当がcheckerを使わず原行から点検した。消費snapshot原点frame756・取込clock387584、現行registry、birth_sample0が判断行より前に存在し、GroupPrototypes、forecast群、パラメータ、owner/body-generationが消費値と完全一致した。自己群除外は1件。これは1判断の限定照合で、全判断の完了や物理帰属を主張しない。

新origin checker v1は終端観測に対して偽の欠測を報告した。runtimeは終了後に解析publisherをdrainし、最後のsnapshotをtemporal_observationとして報告するが、そのsnapshotは次の判断へ取り込まれない。S-on-aでは判断取込原点の最終frame1312に対し、この観測はframe1313だった。sourceのfinished分岐とraw順序を確認し、取得/source/planを変更せず、初版失敗を保存して別版checkerで終端未消費観測を区別する。旧causal v4/window/stage2と新原点checkerの結果は別に記録する。


### 2026-09-28 原点記録版の全判断照合

隔離版 `i11-origin-records` の全8renderは正常終了し、旧版の全8 WAV・report付き6本各28判断と完全一致した。終端未消費観測を誤って欠測とした新checker v1の4失敗を保持し、sourceの終了後drain経路に対応する別版v2を固定した。v2は実shared producer入力と対応する終端1観測だけを分母外へ分離し、途中欠測・未来参照・同frame差替えを拒否する。17局所検査と30固定入力hashの照合後、同じ4 ON rawを再監査した。

全112判断で消費snapshot、現行Voice registry、実出生原点の照合が成立した。全5,256 producer hopとの対応に整合エラーはなく、body/Binding/CDF/群履歴までの `complete_observable_join` はMの二反復で各1判断、残り110判断はunknownだった。成立した判断は両方ともvoice1・now387584・群(bus0,epoch0,generation63)で、既存の自己群除外に対応する。同一scene/seedの同一判断の再現であり、独立した二条件の正例ではない。Sの各28判断はBinding欠測、Mの各27判断はBinding欠測25・assignment未対応2を残した。CDF群欠測は重なる副理由として保持した。物理的source所有、保持中Toneのrouting、I11の全面受入れは未証明のままである。


I11原点版の[結果](../../.worktrees/i11-origin-records/target/i11-origin-post-acquisition-v2/results.md)と最終索引を固定した。索引は167現物・1,262,020,708 byte、SHA `568d60a734939ee2f7e44541a046bbf4c319b72dbe86e37059ad1db8815e804b`。相対pathは原点版worktree基準、旧索引参照は絶対pathとして、主担当も全167件のサイズ/hash一致を確認した。初回の主担当確認は全pathを絶対扱いして停止しただけで、成果物の不一致ではなかった。修正確認記録は `target/parallel-continuation-20260928/i11-origin-final-index-verification.json`。当該単位の原点追加・取得・保存入力照合を完了とし、計画全体や物理帰属の完了とはしない。

NSGT全suiteは同じsession72441を維持し、factorial、代謝、再出生を通過してselectionの通常renderer比較を実行中。終端成功後は担当が主担当作成のrelease helperを一度だけ実行する。保存PCM、全690表の専有ABBA、通過時だけ30秒paced出生という順序を維持する。次段paced runnerは旧18資産と追加8系譜・前提資産を検証する別target版として準備した。SIMD→協調worker→public-guardの二段由来、ABBA不合格・別binary/source・raw改竄を拒否する15局所検査を通過したが、正式planとpaced取得は未実施である。


### 2026-09-28 原登録への要件再照合とSIMD全suite完了

前ターンを、I11の新取得・原点照合・封印による前進と分類した。I11原登録 `i11-onset-comparison.md` §4.5はBinding欠測なら自己群を除外せず報告し、CDF欠測群を集合から除いて集合が空のときだけ到来をunknownとする。またdescriptorによる関連付けを完全な自声検出と区別している。従って112判断すべてのcomplete化、物理音源の完全同定を完了条件へ追加しない。原点版で残った110 unknownはこの限定照合の分類であり、それ自体を実装不合格としない。既存の明示未達・未取得は、§5.7のI11-1現版直接処理2条件とI11-2到来ON版の費用比較を中心に改めて確認する。§4.5のCDF未生成原因の群別報告は古い結果では未実装と記録されていたが、現隔離sourceではarrival/cdf.rsからrecurrence.rsを通じarrival_cost.rsへ具体理由を配送済みである。主担当もこの経路と局所反例を読取確認した。既実装の再実装と、未取得の費用比較を混同しない。

NSGT SIMD全suiteは2026-09-28T12:27:44+09:00に43群・1402 passed / 0 failed / 54 ignoredで終了した。log 3,741,520 byte、SHA `e4cd904b1e4b5074686d76bdf098a82789b1fbb4c5e24f9df795b40d201b6550`。主担当もNUL0、canonical byte一致・別inodeと終了statusを確認した。固定release lib-test binaryは59,649,768 byte、SHA `ebbe5a30b6e01eae502664b93fc75a1b51ee9b2a4552ea43d5e86000f2809ccb`、build metadata SHA `47e6fd7ea1d31583d30afb511263450fe78b1a37f2db64d989976572a693520d`。主担当がbinary、全298Rust/Cargoのbuild前後manifest、全517source manifest、test log、Cargo config、build scriptの現物hashを照合した。保存PCMとABBAはこの固定binaryで別取得する。


NSGT SIMDの保存PCM正式plan SHA `62a03df702e98e12dc698db75500d3cee54f4362fe7bf4c9cc28a960b5df2c16` は14資産を固定した。10 query×72 frameの再生・独立検算はcomplete、不一致0。主担当も全10 JSONLを親協調worker版の同queryとbyte比較して全一致を確認した。ABBA plan SHA `39779e6814b9e858f3d662f78893c2dd240d992a3e1270d865886befee774dd8` は28資産を固定し、hostのbuild/test/render終了を読取確認して旧・新・新・旧を取得した。

四processはすべてexit0、全690候補のRecipe/oracle/密度byte一致。CPU秒はold-a20.405899、new-a22.431098、new-b22.574020、old-b20.223488で、新/旧比は1.0992457622と1.1162278238だった。登録した両比≤0.95をともに満たさず、SIMD案は不採用。外側runnerのexit1は数値不一致やchild失敗ではなく速度gate不合格を表す。結果SHA `84a6f2502e446e0f2e17e851d1390ad26fa9c516a1ad11ddbc0f36fdc78d1003`。主担当も全4 statusとoracleを再照合してCPU比を再計算した。次段paced計画の発行・取得は行わず、準備稿15反例の状態で保持する。元の全表19.85秒・30秒3出生未達は解消していない。


SIMD版の[結果文書](../../.worktrees/body-fitness-nsgt-band-simd/docs/design-notes/body-fitness-nsgt-band-simd-results-20260928.md)はSHA `eeb207332e5479c04830dcf9ac76be329fdbb7094da20864d125e5e3ee2de4c1`。最終索引は657現物・271,800,961 byte、SHA `5cee704102a9e0a5b082f5655aecc21798ae74a17ac206c85e6ad036813cd1df`、状態 `functional_equal_performance_gate_failed`。主担当も全entryのサイズ/hash一致を確認した。保存先は `target/nsgt-band-simd-validation/final-index-v1.json`、主担当照合は `target/parallel-continuation-20260928/nsgt-band-simd-final-index-verification.json`。以後このsource・取得・結果を変更せず、次案は別artifactで調べる。

I11の[正本要件監査](../../target/i11-2-acceptance-gap-20260928.md)を主担当が確認した。次の資源比較は現到来モデルを共通baseとする新A/B/C（None／body到来OFF／body到来ON）、既登録8scene、report有無・3反復、A/A3passの432本を取得前に具体化する。旧§5.7の1秒horizon・旧medoid・action profileとは異なる現モデルであり、旧2未達を解消した証拠へ転用しない。現CDF契約の4秒horizon、モデル・係数の固定由来と全config差分を明示する。旧checkerの欠測profile skipを引き継がず、run/hop欠落と非有限値を判定前に拒否する。取得はまだ開始していない。


### 2026-09-28: 代表密度共有の境界と次の費用比較

3予約の密度表共有を既存source・封印reportで調べた。同じseed_frame94・body_seed7でもchild IDは2/3/4で、代表生成の乱数とphase identityが異なる。保存されたticket1/2の同Hz 5点ではRecipe hashが全点異なり、自律pulse rateは6.3711786 Hzと6.1087828 Hzだった。index288の密度hashも異なる。ticket3全表は未取得なので全2070密度の相違は主張しないが、現exact契約で1表を3予約へ配る根拠はない。[監査](../../target/body-density-sharing-audit-20260928/findings.md)の原データ2件と代表生成に関わる両版7 source件ずつ、計14件のhashを主担当も照合した。監査JSON SHA `72497060195656a06762ee18e348a918e8324b7c3eaff15a0a386226abd314d5`。

SIMD退行の次候補として、gather二命令だけをscalar loadとlane packへ置き換えたstandalone積和を検証した。4740 laneの全real/imag bitsが旧scalar・gatherと一致し、対象機械語はgather0・FMA0だった。主担当も22成果物のサイズ/hashを照合した。manifest SHA `a01ad54eb65ecd0cf9a82d965b8e3ee0bdc29e67570bea528533d9c2fe956e8d`。この時点で速度は未測定であり、先の全690表ABBA不合格は変更しない。三方式を同じ786帯域・固定FFTで比べる局所計時へ進む。

I11資源比較用instrumentは封印原点版543 sourceからbuildを終えた。binary SHA `62737eb1860787e0ea93cf232f14ec9c85bdb56b8895680e81b2cd1a576453e4`、21,321,608 byte。build前後のsourceとCargo configは一致し、8 scene×3 configのcompile-onlyは全24件成功した。取得前reviewでprofile末尾削除とsummary件数改変を組み合わせた欠測を拒否する611 hop固定検査を加えた。hop予算も固定geometry由来の値へ照合する。旧準備planは保存し、数値取得前の検査強化として新planを固定する。正式432本の性能取得は局所計時終了後、heavy処理が重ならない区間で開始する。


scalar／gather／packの局所計時は固定FFT・全786 bandで各144 pass、計432 passを単回取得した。CPU中央値は387254.5／413429.5／271945 ns、中央値比pack/scalarは0.70223845。全passの出力digestは一致した。主担当も事前planの14入力hash、CSV全432行と各144行、中央値を独立再計算した。最終索引13成果物はSHA `e7acdbb6bfe10d094cc93a7f9902d8135b891669fafccf20c7fbc02a9b9e6937`。固定synthetic FFTに対する積和だけの約30%短縮であり、全690候補×72 frameや通常runtimeの速度改善ではない。次の隔離本体候補を準備するが、I11資源取得中はbuild・test・replayを重ねず、コード編集と検証登録だけを進める。


I11の資源取得前plan最終版はSHA `0349269d2fd023629abde589f4bd1611ee9618ef3a308d2a8490c0969f21a3d3`。旧plan/preplanはv1として保持し、新版は固定hop予算・host開始記録の照合を追加した。Python反例9件とpy_compileが成功した。主担当は22固定入力のpreflight、543 live source、全432 jobとlabelの一意性、8 sceneの旧bytes、onset節以外のA/B/C一致を確認した。AA288本とA/B/C各48本であり、C対Aを主判定とする。別担当も原§5.7の分位点・3rep集約・48差のfloor・予算超過差の式に重大欠落がないと確認した。検査結果は `target/parallel-continuation-20260928/i11-2-resource-plan-verification.json`。host開始/終了snapshotと1秒CPU系列を追加して正式取得を指示した。まだ取得完了・資源合格とはしない。


正式取得v2は出力先を相対path `runs-v2` で渡したため、repoをcwdとする子processのreport/profile作成先と食い違った。全432本がexit1で終わり、profileは0件だった。主担当もjournal・先頭log・終了statusを確認した。`runs-v2/` と `formal-v2-run.*`、host before/after、失敗auditを保存し、性能値は存在しない失敗取得と分類する。固定plan/source/binaryは変えず、出力先をrepo内の絶対path `target/i11-2-resource-preparation-20260928/runs-v3` で渡して再取得する。全432 argvのprofile/report絶対path確認を先に行う。

新隔離版 `body-fitness-nsgt-pack` はSIMD親の封印517ファイルを継承し、`src/core/nsgt_rt.rs` のgather部分だけをscalar load＋packへ置き換えた。主担当も親現物とのdiffを読み、係数・mask・積和順・出力部分に変更がないと確認した。新src SHA `27a54c14d2649d8e5cd6cab2eb7485bafd257226fefc22ec018c60b3604291b9`。rustfmt確認だけ通過し、cargo/test/replayはI11専有取得の終了後へ保留した。検証登録草案は `target/nsgt-pack-validation-next-20260928/registration-draft.md`。比較基準は旧協調workerとし、失敗したSIMD版を速度分母にしない。


v3は絶対出力pathで同planの432本を開始した。host名前空間の開始snapshotで他heavy processなしを確認し、session16832を保持する。先頭取得はexit0でprofileが生成され、主担当も最初の2件で611 hop・固定geometry・全数値と終端の検査通過を確認した。全432本の終了・資源判定は未完了。


### 2026-09-28: I11取得継続とpack本体検査の固定

前単位を局所速度実測・新source実装・I11取得開始による進展と分類した。同じI11取得session16832は担当側でrunningを確認し、主担当もhostのrunner PID4038523とその子instrumentの稼働を確認した。主担当から他agent所有sessionへ直接pollした際のUnknown process idは取得停止の証拠ではなく、再起動していない。現在88/432本がexit0。scene長6.5秒と実取得wallは別であり、先頭約1秒の値もHarmonic系約20秒の値へ外挿しない。

packのコメント訂正後source-v2は517件、manifest SHA `d699296cd221bb15e1c46b81fe47e44ab31fb48e64836b27805cba2a2fdbc7cf`。主担当も全live/capsule hashと、SIMD親との差分が `src/core/nsgt_rt.rs` 一件のみであることを確認した。検査runnerは局所2検査、fmt、Clippy、all-targets、全suite4threadsを同じsourceで実行し、stdout/stderrと同shell statusを別inodeのcanonicalへ保存する。新pack用ABBA準備scriptはsource517外として別hash固定し、軽量Python反例9+3件が成功した。旧協調workerを性能基準とし、絶対pathのargvを要求する。I11全432本とpostflightが実session終端へ到達した場合だけ、担当が機能検査を開始する条件付き承認を出した。性能取得には別の専有枠を設ける。

I11原登録§5の[部分版対応監査](../../target/i11-requirement-version-audit-20260928/requirements.md)を追加した。現版と旧stage2でfootprint/participation/CDF/recurrenceの計算sourceはbyte一致し、arrival_costの差はprovenance字段だけだった。現版全suiteの該当試験とassertも読取確認した。ただしこのsource継承だけでNoneの全WAV回帰・同packet回帰・API音響作用・12条件分布を一括合格にはしない。§5.7で明示されたflow返却遅延、proxy(absent)比率、置換/drop件数は、今回の取得後に保存reportから補遺として集計する。直接処理欄の数値判定とは分ける。


### 2026-09-28: I11旧証拠の時点訂正

§5.4a/b・§5.5a・§5.9の[版継承監査](../../target/i11-requirement-version-audit-20260928/inheritance-review.md)を作った。初期監査で9月24日のD丸め・external_energy差による§5.5a未達を後続版へ誤って対応付けたが、主担当が9月26日の同一record取得現物を確認し訂正した。`first-divergence.json` SHA `2ef328a1376ea68f116c3f3856292a6d534be02af72ac42d46cb9acbf8787a6b` はharmonic/modalの6条件でpower以外の入力差なし・先行差なしと選択分岐を記録する。`bit-identity.json` SHA `5c0616068d21c5fe3fa45df9d5cd95fd26e1b0e29efade83c5955116a482f255` はNone12/12合格、候補recordは共通11,382行で差0、片側195/158を分離した。

§5.9も旧9月23日取得だけでなく、`target/i11-gap-06a4772-20260926-205241/` の新基準再取得を確認した。主担当がsummary現物と旧summaryのbyte一致を照合した（SHA `e3c574b7c11e1f05fb8c9f00c0eb03489afdc7062bfc2d7339ac3930b4bf71b0`）。投影3,390機会の分布は9月26日の基準版で再現済みであり、既済の作業を未実施へ戻さない。一方、現原点版d66…への継承は別問題で、同一record修正後の06a4772からの差分がNone音声・学習record・候補packet・実ToneEnergy投影へ到達するかを静的に追っている。変更pathの数だけで全36本再取得を要求しない。

pack release固定helperはsource manifest、Cargo config、全suite43群/1402成功/0失敗/54ignoreを固定し、RUSTFLAGS/CARGO_ENCODED_RUSTFLAGSとbinary modeも保存するよう補強した。helper SHA `7cf781cb41063fef17c50a4838e79346950c310ce95ace7c49b75fb70d8837f6`。まだ実行しておらず、I11全取得の専有枠を維持する。


I11の[到達経路監査](../../target/i11-requirement-version-audit-20260928/route-inheritance.md)では、9月26日の同一record版から現原点版への変更を旧登録設定で追った。到来/group/PCM trace枝は無効、候補時刻の式・到来なしの最小費用・既存writerを維持し、footprint/energy/Tone/代表gap投影核はbyte一致した。数値経路の限定継承は成立するが、現binaryの12条件WAV・record一致という実測とは区別する。元12条件の分母を維持してNone/body/proxyの36 renderを別取得前計画へ具体化する。None結果は§5.4a/bへ、body/proxyは§5.5へ共用する。§5.9の再投影は行わず、9月26日版の分布を時点付きで保持する。旧baseline raw約853MBのhash読取りもI11性能取得終了後へ送り、今はrunner・検査の静的準備のみ進める。

### 2026-09-28: 原点版36本回帰の準備確認

主担当が `target/i11-origin-regression-preparation-20260928/` の登録、plan生成器、runner、比較結果判定を読んだ。旧12 scene・3 config・seedに加え、外部action-profile blob SHA `033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5`、基準12 WAV/report、固定原点543 sourceとrenderer、比較器・準備scriptを取得前assetに含める。全argvは絶対path、cwdはrepo rootに固定する。36取得の厳密なcase/variant/labelと4比較器の集合を要求し、共通候補0件や比較器欠落を合格にしない。

Sine holdは旧結果でも比較判断0件であり、Sine flowの比較あり・非分岐とは分けて保存する。後者の空分母は拒否する。合成7検査が成功し、旧9月26日の4成果JSONだけを一時領域でpath正規化して判定へ渡すschema検査も成功した（hold比較数0/0/0、flow比較数60/219/920）。取得rawの再読取りやrenderはしていない。登録SHA `55e518e1355f2e7e9370e821d931fce180a6881e49313fc3e11f4b8311474ddc`、runner SHA `129abb09d563aaf2ca30677c62b25431453247b6a6a76a895ab6e6ea0df784d1`。主担当の確認記録は `target/parallel-continuation-20260928/i11-origin-regression-preparation-review.json`。正式plan発行と36取得は、進行中I11資源測定の終端・監査結果を確認してから判断する。

packの検査runnerとrelease helperも読取確認した。I11担当の実session終端・postflight・host-after確認による専有枠解放後、既に承認した機能検査から全成功時のrelease固定buildまで進める。実binaryのgather/FMA照合も記録し、PCM一致・全690候補ABBAの性能取得はその結果を受けて別途調整する。現時点でpackの本体性能・I11資源合格は未判定。

### 2026-09-28: I11配送・欠測補遺の集計準備

前単位は36本回帰準備の実装・検査による進展だった。次の継続では同じI11 session16832のlive応答に加え、host runner PID4038523と子instrumentの稼働を再確認した。取得を再起動せず、空いた担当へ§5.7補遺のsource照合と集計器準備を委譲した。

`target/i11-2-resource-supplement-preparation-20260928/` に登録とcollectorを作成し、主担当もsource契約と集計処理を読んだ。旧定義のparticipation_context件数重み遅延と、body_footprint返却event一件一回の遅延を別の母集団として保存する。proxy(absent)比率の分母はcontext数。終端profileの要求・完了・入力drop・返却drop・置換・再送・解除はrun累積値であり、返却event件数や個別未返却要求の情報へ置き換えない。

到来のknown判断数、非null候補slot数、CDF確率付きslot数、判断ごとのCDF群数分布・延べ群数と除外理由を分離する。Bの到来無効を観測0件とせず、Cのknownと群数の対応、確率の有限性と範囲を確認する。既存資源auditの同一runs内参照と到来state集計へ照合し、直接処理欄の合否は変更しない。合成4検査が成功し、主担当は固定planからB/C各24、合計48 reportが選ばれることも確認した。実取得reportはまだ読んでいない。主担当のhash付き確認記録は `target/parallel-continuation-20260928/i11-footprint-supplement-preparation-review.json`。実集計は全432本の終端と元資源auditの後に行う。

### 2026-09-28: 次候補の幅変更に伴う静的費用

I11同一processの稼働を再確認して待機を継続し、その間に既存constructor TSVだけで隣接bandの4 laneと8 laneの幾何量を比較した。主担当も入力3件のhashと全786 bandの順、実entry 791,851、連続group最大長の和を独立再計算した。4 laneは197 group・218,591 step・利用率90.5631%、8 lane仮定は99 group・115,236 step・利用率85.8945%。entry payloadは13,989,824 Bから14,750,208 Bへ760,384 B増える。同じ旧sparseを保持する場合の合計は26,659,440 Bから27,419,824 Bとなる。Box/Arc等のheaderやFFT bufferを含む全メモリ量ではない。

[静的結果](../../target/nsgt-width-geometry-20260928/findings.md)は命令・cache・CPU時間の測定ではなく、8 laneの実装や計時は行っていない。現packの全690候補比較を先に判定し、その結果が必要とする場合だけ次候補を選ぶ。

### 2026-09-28: I11資源比較の完走と不合格の固定

同じsession16832がexit0で終了した。v3の全432子はexit0、profile432件、各611 hop、postflight22項目が一致し、host-afterを保存した。専有測定の終了確認後、pack機能検査と36本回帰の正式plan固定を並行開始した。主監査は整合エラー0・証拠受理だが、登録C（到来ON）対A（None）は5/16条件のみ合格で全体不合格、補助C対B（到来OFF）は7/16条件合格。監査exit1はこの登録不合格を表す。失敗を外れ値として捨てず、閾値も変えていない。A/Aのfloorは登録された48組の差の実測最大であり、統計的な雑音上界ではない。

主担当も全144 test profileからpopulation_us・synthesis_us・own_usの中央値/p99/最大、予算超過数を独立再集計し、元auditの432統計欄と一致を確認した。`target/parallel-continuation-20260928/i11-resource-independent-reduction.json` に入力hashと再集計を保存した。per-hopでownからpopulation/synthesisを引いた残りの時間も診断用に集計した。sine-hold-4のreport中央値はA/B/Cが108.821/108.441/261.232 µs、reportなしでは81.531/82.520/83.441 µsだった。sine-hold-16のreportは145.690/143.161/660.414 µsで、report側の別経路を調べる根拠になるが、これだけで特定の処理へ因果帰属しない。

配送補遺は48/48 report・profile・exitを照合し、Cの2,508判断中known369、CDF確率付き候補8,487、判断ごとの延べCDF群432を記録した。Cの返却・要求・完了は各360、入力/返却dropは0、superseded31。proxy(absent)はharmonic-flow-16の6/600 context（1%）で、全C contextは2,325。主比較のA/B/Cいずれもown予算超過hopは0だが、処理時間の許容差不合格を取り消さない。結果は `target/i11-2-resource-preparation-20260928/result-v1.md`、最終index SHA `5d9c81d83c70bdfe397daf2f4d6e83efcfefa1db903c98346f393a2b09d629ce`。主担当も1567項・14,084,289,016 byteの全サイズ/hashを照合し、error0を `i11-resource-final-index-verification.json` に保存した。旧v2失敗を含む取得先は以後変更しない。

36本回帰は旧baseline12組861,171,277 byteと24 asset・15 inputを固定し、正式plan SHA `ca9dceb1b22fa113af092442483c3c236c2ec676d23c04c7d4400dba0e5c5745` の事前検査が成功した。主担当もレビュー済み準備scriptと正式planのhash対応を確認した。ただし新資源証拠で現版の不合格が確定したため、36 renderはまだ開始しない。report側の追加費用とreportなしpopulation費用をsourceと保存profileで調べ、意味・記録を維持する修正範囲を決めてから対象版を確定する。原点版の旧回帰が新たに合格したという主張はしない。

### 2026-09-28: I11費用削減候補の隔離実装

[追加費用診断](../../target/i11-resource-cost-diagnosis-20260928/findings.md)はsine-hold-4のreport C−Bでreports中央値+35.26 µs、render_route+125.53 µs、内包synthesis+4.89 µsを確認した。各中央値の差を加算可能な因果分解とは扱わない。同条件rep0のreportはBが18,603,987 B、Cが109,990,947 Bで、追加された全scan traceの同期JSON化と整合した。

当初候補の解析worker側preencodingは、実装前のack境界確認で見直した。今回の`--play=false`は決定的解析を使い、bodyのCapture::endがrenderer内でworker完了を待つ。ack前JSON化ではrender_routeの費用がsynthesisへ移るだけになる。shared側もanalysis_waitへ移る可能性がある。この案を性能改善として実装・採用することはしない。

新隔離tree `.worktrees/i11-resource-cost` はHEAD06a4772へ原点版archiveの543ファイルを復元し、親と全hash一致を確認してから編集を開始した。report側の候補はarrival+report時だけ使う有界FIFOと一つのwriter threadへ変更した。一般recordは既存Serializeでbytes化、重い二種類のowned traceはwriter側で既存ReportRecordのまま符号化する。全recordの呼出順・値・bytes・改行を保ち、満杯時の待ち、flush完了、IO失敗、Finish/Dropのdrainを検証する。producerと解析ackは変更しない。全体CPU改善とhop側同期費用の削減を区別し、queue payload、解析待ち、全elapsed、最終drain、process CPU/RSSも後続検証対象とする。まだ実装途中であり、速度・回帰の成立は未判定である。

独立してCDF窓確率の二重計算を削減する変更を同treeのarrival_cost.rsとtemporal_participation.rsへ入れた。固定23 slotで妥当性検査時の値を保持し、群全体が採用された後だけ、旧group順でf64加算する。追加heap allocationは無い。主担当の差分review後、最初の二窓が有効で最後の窓が無効になる群を用い、途中値の混入を検出する一つの試験へ修正した。CDF側のCargoはまだ実行せず、report側と合わせた一回の全体検査を実装担当が行う。元の封印source・取得結果とmain sourceは変更しない。

### 2026-09-28: report writerの失敗反例と小規模screen準備

arrival専用writerの初稿は、同じ入力のJSON byte・record順と満杯queueの局所試験を通過したが、書込失敗時のflush試験が停止した。静的レビューのblocker未発見という所見を、この実測で更新した。writerのReceiverが切断されてもproducer側Senderが残る間、未処理Flush内のreply Senderがqueueに残り、ackだけを待つ側が終了できない。writer終了通知とflush ackの二つをselectする修正を入れ、write開始→Flush enqueue→故意Errまたはpanicという順序を固定した反例、満杯時の待ち、明示flushなしのDrop drainを再検査中。最初の中断・timeout結果は保存し、全体検査成功とは扱わない。

次の費用比較は四scene（sine-hold-4/16、harmonic-flow-16、modal-flow-16）×report有無×旧新新旧の32processを予定する。旧正式試験のseed・scene・C設定・611 hopを維持し、外側wait4のprocess CPU/RSSと終了までのwallをhop統計に併記する。これは診断であり、旧全16条件の資源ゲートや元12scene×3条件の機能回帰を代替しない。草案は `target/i11-resource-cost-diagnosis-20260928/screen-registration-draft.md`。新source/binaryは未固定で、取得はまだ許可していない。

`target/i11-resource-cost-screen-preparation-20260928/run_screen.py` を用意した。32本の順序・入力固定、絶対pathかつ新規出力先、child終了値と起動失敗の保存、前後hashを検査する。child非zeroのoutput/CPU/RSS保持、起動不能、欠測/順序変更/asset変更、既存/相対出力先という4つの局所試験が成功した。実際のinstrument計測は未実行。

report比較契約の事前確認では、旧版の同scene/seed別processの先頭1500行でも型順序とtrace値に差があった。これは部分検査であり、全recordの差の規模は別途調べる。old/new全report一致を無条件hard gateにはせず、各runのsequence・emitted件数・drop・終端・判断内部整合を検査し、旧新の差を診断として保存する。writer単体の同一入力byte/順序一致とCDFの旧計算bit一致は維持する。

NSGT packは同じ全suite sessionを継続し、factorial内の最終caseを通過してrespawn統合試験へ進んだ。factorialの終端を全suite完了とは扱わず、release固定buildとPCM/速度取得は引き続き後段に置く。

同じ継続単位でwriter focused-v2が4/4成功した（2026-09-28 14:42:44 JST、exit0）。主担当もstatusと全結果欄、故意panicを捕捉して成功したこと、debug出力がsourceに残っていないことを確認した。旧v1のexit130と単独timeoutのexit124は保持した。CDF変更を合わせたformat/Clippy/all-targets/fullsuiteはこれから実行するため、統合検査や速度改善の成立はまだ主張しない。

### 2026-09-28: I11費用候補の統合検査とscreen監査器

新候補のClippy初回はReportMessageのvariant寸法差（InputTrace 10024 B、FeatureTrace 544 B）で失敗した。arrival+reportのhandoffだけをBox化し、producerのtrace型とDirect経路を維持した。format、標準Clippy、all-targetsが成功し、source543ファイルをSHA `1b393a84aff0f5650b8814c7562b5a8e4e85f4c777d05b166b00661b0c2172d4`で固定した。主担当も全live fileのhashと封印親との差分を確認し、変更はarrival_cost.rs、report.rs、temporal_participation.rs、runtime/mod.rsの4つだけだった。

全cargo testはsession67174で終端exit0（14:51:27 JST）、32群・1204成功・0失敗・36ignored。主担当はNUL0、log210152 B、SHA `68291615f8bda1e354e59925b534939d0a67237e63e7a376b86eedbbf4e20060`、canonicalのreport/statusが保存版と同bytesで別inodeであることを独立確認した。確認先は `target/parallel-continuation-20260928/i11-resource-cost-fullsuite-verification.json`。この検査は実装の機能証拠であり、資源基準の合格ではない。次に同sourceからinstrumentとoffline rendererを固定buildする。

旧focused-v1について前記exit130はwrapper中断だった。主担当のhost process点検で旧cargo/test子が残っていることが判明したため、cmdと親子関係を確認し、旧test PID4096786だけSIGTERMで停止した。cargo4088627とshell4088568は自然終了し、主担当も指定3PIDの消滅を独立確認した。旧cargoの最終exitは101で、wrapper130と経緯は `target/arrival-writer-focused-v1.lifecycle.json` に分離保存した。新fullsuiteの成功と混同しない。

screenの統合監査器 `audit_screen.py` を追加した。32本のjob/command/child exit、wait4 CPU合計とRSS、旧validatorによる611hop等の完全性、reports/render_routeを含む8指標の分布、report内部整合を照合し、旧新a/bを別々に示す。元の統計関数 `int(frac*(N-1))` のindex規約を継承する。欠測時に数値比較を出さず、各runの原情報とerrorを保存する。独立レビューで指摘された非object profileも明示errorへ落とした。runner/監査の10反例が成功し、保存旧版8profileのschema smokeも成功した。

比較器v3は9反例成功、SHA `012b076ccb9d45d7403b21c31b490724bdfaf907333f6bfeffe51c060a8b21de`。旧hold2本は各15516行、旧harmonic-flow-16の2本は各50926行を全行検査し、内部整合は成功した。flow2本のtype件数と順序付き鍵は一致したが、8typeのraw hashに差があった。各flowはknown77、CDF候補1771、延べselected群96で、状態と仕事量も記録する。kind別最大JSON行長も保存するが、64倍をRustのRAM上界とは呼ばない。例えば同じ旧flowでも最大行は295282 Bと315077 Bだった。

sandbox内のprocess一覧だけではホストの他作業を観測できないことを確認し、screen実行器にhost PID namespace `pid:[4026531836]` の前後検査を追加した。隠れたprocess一覧を拒否する反例も成功した。source/binary/正式plan固定前、またpack検査・buildとの重複中は取得しない。packは同じsession56399でrespawn4件、selection3件、three_consumer2件を通過し、残りの統合検査へ進んでいる。

packも同じsession56399が14:54:14 JSTにexit0で終端した。主担当は43群・1402成功・0失敗・54ignored、NUL0、log3741187 B（SHA `088c6a16ae2758b651db507a12511212c8b3e8c4078d0471ca78fef7007a08c1`）、canonical report/statusの同bytes・別inode、固定source517件の全hashを独立確認した。`target/parallel-continuation-20260928/pack-fullsuite-verification.json` に保存し、承認済みrelease helperへ進めた。I11もinstrument/rendererのpair release buildを開始したが、両版ともまだ性能取得は行っていない。

### 2026-09-28: packの固定全表CPU改善とI11 screen開始

両release buildが完了し、主担当がbinaryのbytes/mode/hash、metadataとlog/status、source543または517件の前後一致を確認した。I11 instrumentはSHA `0a61b25465029f376caa6a8a841300ca001f606e3d64f180b7eae3fb09ac4026`、rendererは`2018ad0b28c3490f199f159428df8ac1adb1a924de47edbe70baa60bb37c9125`。pack libtestは59732160 B・0775、SHA `512473ad17f8f28cadf099ffa02f4e31d4eefea817404fa2c43eb85bfebbb42a`、metadata SHA `85ed9a6c98a0475f6f874b1169dc2249c52acccf2628e4be5fab6e476bd96ecb`。実binaryのanalyze_one_and_updateとcoherent_four_band_sumにgather/FMA命令はなかった。

pack保存PCM plan SHA `c72be342cc5550e22d698a82f54b1e246a52f5705f3bb4082581e5c6e0fdc9f1` のreplay/checkerがexit0、complete。主担当も10query・720frameを旧協調workerのrawと直接byte照合して一致した。ABBA事前検査ではcheckerがmanifestの親hash keyを誤参照してKeyErrorとなった。性能取得前に実key `parent_source_manifest_sha256` へ一箇所修正し、負例を含む10検査を通した。旧plan055b9bb0…と失敗statusを残し、新plan SHA `f69976bc1cd568a52014735fe93414eae73ee0df8963ac53ace831c78093f2a2` を固定した。source/binary/PCMと速度基準は変えていない。

host namespace `pid:[4026531836]` で他のheavy processが無いことを確認し、session68032で登録順の4processを一回だけ取得した。全childとrunnerはexit0、全density/oracleは旧baselineとbyte一致。CPU秒はold-a 20.669809、new-a 14.848029、new-b 14.805454、old-b 20.293686。対応比は0.7183437931／0.7295596276で、両方0.95以下となった。結果SHA `8b68968c016d6438f900f4e792b4921a675d7569bceaf6216619c235e4c01135`。主担当も全4出力のbyteと両比を独立再計算した。後続host witnessはheavy0、source517 live/capsuleのpostflightも一致した。対象は固定690候補表を処理するwhole processであり、通常runtimeの受入れではない。

30秒・3出生・全690表のI-W4-on-a paced試験は、既存単case runnerの親manifest固定を新版へ対応させる準備に移った。既存の科学・oracle・資源判定式は維持し、source変更や出生数の緩和は行わない。その準備中に、I11の固定32process screenを先行させた。plan SHA `44648a863963e7610b3ca94f65a038b44b2c575200970df3afb1574631564811` の48assets、32unique job、元Cの4scene/config、旧新instrumentを主担当も照合し、error0だった。hostでheavy0を確認してsession95520を開始した。出力は `target/i11-resource-cost-screen-acquisition-20260928/runs-v1`、取得中は他担当を小script編集だけに限定する。全体§5.7の再合格とはまだ判定しない。

新I11の元12scene×3条件回帰planは `target/i11-resource-cost-regression-preparation-20260928/acquisition-plan-final.json`、SHA `370b667de256f63e55c659e26c060ee840d4f8dfc42d13ad6c10a567acde8534`。8軽量反例と36本/基準12組の静的検査が成功し、取得は未実行。初回静的planは登録文の時制修正前のhashを持つため未使用版として保存した。元36条件はarrival指定が無く、config.rsの既定falseが適用される。新CDFと非同期writerの到来ON経路は、既存の親原点8条件と同じWAV・判断・原点照合を新版でも再検証する準備を別に進める。自己群の追加正例や110 unknownの解消を新しい要件へ変えない。

### 2026-09-28: I11 screen完了とpack paced開始

I11の32本は全child exit0、前後照合成功、固定監査もexit0・整合error0で完了した。audit SHA `9344cc66ae2d6c8a8fe162121dfd4a5969ad7f7193c9d287f0ed84ac5f93da4e`。全profile各611hop、reportあり16本のtrace emitted一致・drop0を確認した。report付き8対のown中央値は155–718µs、p99は231–848µs短縮したが、process CPU比は0.865–1.007で混在する。modal reportのcandidate energy submitted/completedは旧125から新117へ減ったため、全仕事量を揃えた効率改善とは呼ばない。no-reportのp99は3対で増加した。主担当は全16対と仕事量欄を読み、[結果文書](../../target/i11-resource-cost-screen-acquisition-20260928/result-v1.md)も照合した。旧§5.7のC/A 5/16という不合格は変更しない。

pack paced plan SHA `9c16d34061197265ec2cd9c921405f243aae0481b6a1e54b57790b7fd63152be` を主担当も旧単case runnerとdiff照合し、30秒・3出生・全690表、既存oracle/科学/資源式の維持と事前検査成功を確認した。host PID namespace `pid:[4026531836]`・競合heavy0を確認し、session85734で単回取得を開始した。他担当のbuild/render/大hashを停止し、性能取得の専有を維持する。

到来ONの新版原点8条件planも固定した。SHA `a8539e4354900b96f5ff099fafd09069cf5d5b1b2b870a22c7ef9fa8afacd5f6`、4軽量反例と8 WAV/6 report・543 sourceの静的照合が成功した。8条件と元36条件のrenderはまだ開始していない。

pack pacedは全postまで終了した。取得子exit0、2813 active hop、hop予算超過・underflowとも0、oracle取得済み密度一致、単case資源観測を通過した。一方、科学checkerは3出生条件に対して2出生・翌hop receipt2件となりexit1だった。子2は15.445333秒・184.24945068359375 Hz、子3は29.845333秒・7062.9052734375 Hz、子4の全表はFinish時点でpendingだった。主担当も全1382密度の長さ付きSHA、Recipe hashとraw対応を独立検算した（`pack-paced-independent-verification.json`）。全表のcomputeは14.394375秒と14.378023秒。速度改善で旧1出生から2出生へ増えたが、登録3出生・全3表条件は未達であり閾値・期間は変更しない。

性能専有の解放後、I11の元36条件（session92389）と原点8条件（session97958）の機能回帰を開始した。主担当も両正式planのcheck-onlyを実行して成功を確認した。最大2 rendererで並行し、性能数値の根拠には使わない。screenの最終索引は89件・4,854,121,263 byte、SHA `60ffdcb5548b33ecb8f767fd7e095a0d35239418d2d35f43a9ca870299bcbd67`。主担当が全サイズ・hashを照合してerror0だった。

packの[結果文書](../../.worktrees/body-fitness-nsgt-pack/docs/design-notes/body-fitness-nsgt-pack-results-20260928.md)と最終索引を固定した。索引SHA `4ea374b22d368dcf88c3afdee821693ddb0be0449e18e67049cb7ebbfdaeec97`、1,744現物・401,490,786 byteを主担当も全サイズ・hash照合し、差0だった。I11 screenも32 profile各611 hopについて8指標×3統計、計768セルを主担当が独立再計算して一致した。これらの検算は、packの3出生未達やI11の正式資源不合格を変更しない。

### 2026-09-28: I11費用候補の44本機能回帰完了

元36条件は全child exit0・4後検査成功、result SHA `acfd43945442029c8d00efb2e1fff5dccef47e424abf0311144e3905dd29c45b`。None12本のWAV・10種学習recordが一致し、共通候補11,329件は差0、片側だけは基準248・新版157件だった。harmonic/modal各4/16/64 Voiceの6条件はbody powerだけが異なる最初の分岐でselected offsetとWAVが変化した。sine-holdの比較判断0と、sine-flowの60/219/920判断で非分岐を分けた。

原点8条件は全child exit0・21登録後検査・4 complete-joinが成功した。集約SHA `f851ceae7b63eafb13ad875d15f5d9a9a4d3b8385ec6e4e49eec1ff3294a252a`。旧8 WAVと6 report各28判断が一致し、到来ON4 reportの112判断・5256 producerで完全対応2件・unknown110件を再現した。主担当も旧新rawの全8 WAVと6 report判断を直接比較した。complete plan生成時の親raw plan誤写像による失敗は保存し、取得済みrawは変更せず写像訂正版で検査した。これは機能回帰であり資源予算の合格ではない。

新版432件資源planの初版をreviewし、登録文のasset件数訂正、writerの最終drainを含むwait4 CPU/RSS記録とhost PID namespaceの拒否条件を取得前に追加する判断をした。元の432順序・scene/config・統計式・閾値は維持し、追加資源欄は記述に限定する。未使用初版planも保存する。

### 2026-09-28: 8帯域局所proofと新版資源planの固定

8帯域packのstandalone疎複素積和は、封印constructorによる全786実帯域・6入力種と境界合成例でscalar/4帯域とのbit一致を確認した。8帯域比較4776 laneには末尾padding・合成laneが含まれる。nativeとAVX2最小buildの双方で成功し、最小targetの該当関数はgather/FMA0だった。主担当がproof manifest SHA `865a0ac9e2ab64887ddc7664059a00536a04ca467feb749dba4ceaa10de13a17` の20参照現物をhash照合し、別担当の読取レビューも重大問題なしだった。production関数を直接呼ぶ全経路検査ではなく、72frame/密度/runtimeは未検証。局所timing plan SHA `55b839a93429038949bfe1a01fbce96fa30fae48186cd9d62b4d51e6e95f9cea` を固定し、全786帯域・scalar/pack4/pack8各144pass、64warmup、6順序×24cycle、同じFFT/digestで単回専有取得へ進めた。

新版I11の432件資源planは SHA `8d90441972aa2f54f9eedb66ca0eb7531de0cf36171e474ff06d519ce53f1cba`。rootも27asset、全job辞書/order、13入力asset、旧audit/statistic byte同一を確認した。wait4でchild終了までのuser/system/合計CPU/maxRSSを保存し、host namespaceの不一致は出力作成前に拒否する。12 Python反例が成功した。8帯域局所計時終了後に専有取得する準備状態で、まだ432件は開始していない。

I11の44本機能回帰最終索引は995件・6,142,937,641 byte、SHA `fc7e0e39bbe10a96e6ef4be4ccf56a9bdfcd162777397d1e5da592ce0e83e453`。主担当が全現物を独立hash照合して差0だった。

8帯域局所timingは単回exit0、全432行・各144pass・出力digest `6445087a15e56d9b` が一致した。CPU中央値はscalar566145 ns、pack4 329329.5 ns、pack8 318014.5 nsで、pack8/pack4は0.9656423126だった。主担当もCSV全行・中央値・全plan assetsの取得後hashを独立照合した。約3.4%という疎積和の局所短縮を、全690候補表や3出生の成立へ拡張しない。本体への8帯域組込みは保留し、別候補として4帯域entryの独立active配列をFFT indexの未使用下位bitに格納して64から48 byteへ減らす案を静的検討する。まだ実装・計時は行わない。

全担当の重い処理終了後、新版I11 432件へ専有開始指示を出した。正式plan SHA `8d90441972aa2f54f9eedb66ca0eb7531de0cf36171e474ff06d519ce53f1cba`、host-beforeの実namespace/競合確認を起動前条件とする。進行中の旧不合格は保存し、今回の結果が出る前に合格へ変更しない。

新版432取得はhost-visible session25851で開始した。出力は `target/i11-resource-cost-432-preparation-20260928/runs-v1`、runner logは同prepの `formal-v1-run.log`。host-beforeは実namespace `pid:[4026531836]` とheavy0を確認済み。担当が同sessionを追跡し、432終端・postflight・host-after・厳格監査・B/C48report補遺まで続ける。他担当はbuild/render/大hash/計時を停止した。

NSGTのactive-bit案は二担当の静的確認で致命的反例なし。現行index上限内でactive `2k+1`、inactive0、load offset `encoded & ~1`、mask `encoded << 31` を使えば4laneの64byte entryを48byteへ減らせる設計である。218591 entryの型上の削減は3,497,456 byte。実sizeof・codegen・速度は未検証で、容量だけから速度を推定しない。検討は `target/nsgt-pack-active-index-design-20260928.md`、source変更はまだない。

### 2026-09-28: 資源取得中の次候補準備と残条件再点検

新版I11のsession25851は同じhandleでliveを確認し、83/432まで全child exit0だった。sceneごとに処理時間が異なるため、観測間隔やquiet状態を停止とは扱わず、再起動しない。登録した全順序と全失敗記録を保持する。他担当は引き続きbuild/render/大hash/計時を止めている。

active-bit metadataの次候補は `target/nsgt-pack-active-index-preparation-20260928/` にstandalone proofと局所timingのRust sourceを準備した。786帯域・6入力種、活動index0・最大有効index・空/異長/末尾・非活動sNaN・範囲外拒否をbit照合する設計で、scalar/旧pack4/encoded pack4の各144passを比較する。rustfmtの構文確認だけ成功し、compile・実行・性能値は未取得。production sourceはまだ変更していない。一般nfft上限2^18の契約を保つため、固定16384にだけ収まるu16への縮小は今回含めない。

主担当は費用改訂候補のreport FIFO・flush完了通知・Drop drain/join、CDF一群の途中欠測を累積へ反映しない処理を読取確認し、追加の修正要件は見つからなかった。記録は `target/parallel-continuation-20260928/i11-resource-cost-source-review-v1.md`。これは新たな実行試験ではない。計画全体の残条件も再点検し、元の明示gateと任意一般化・研究拡張を分ける。I10の後続4,936実音対を無視して旧正例0へ戻さず、I11のunknown解消や完全な物理source帰属を新しい完了条件にしない。

[残条件の読取監査](../../target/body-aware-fitness-remaining-gates-20260928/audit.md)は、明示された必須条件と証拠の外挿限界を別列にした。F1の任意route全支持、F2の全音色一般化、F4の実選択親の変化や無期限長期生態、I11の物理source帰属・保持Tone route・unknown全解消を、新しい合格条件に加えない。F6実装はF4/F5の採用範囲確定後へ残す。

active-bitの[取得前草案](../../target/nsgt-pack-active-index-preparation-20260928/registration-draft.md)をreviewした。局所計時でCPU中央値new/pack≤0.95、その後の固定全690表ABBAで両対new/pack≤0.95を取得前の工学screenとし、旧scalarで既に得た速度改善を新候補の効果に混ぜない。数値一致・全Rust検査・720frameのPCM同値を先行し、通過後の30秒3出生条件を維持する。新source/binary/正式planは未固定。8帯域の3.4%短縮は記述的screenであり、事前数値gateの不合格だったとは記録しない。

### 2026-09-28: I10通常consumerの現在地を訂正

[実sourceの読取監査](../../target/i10-shared-consumer-current-audit-20260928/audit.md)で、共有候補表のread-only通常consumerは接続済みと確認した。runtimeのsnapshot key/時刻一致try_lock、実Voice packetへの表Arc/binding凍結、背景workerでのcandidate/default Pair、通常reportへの経路がある。主担当も `runtime/mod.rs:2057`、`action_candidates/energy.rs:774`、`action_profiles/consumer.rs:177` の接続とidentity/時刻/bus/routing拒否を読んだ。

初回14record/28表のpaired0だけで止めず、後続の37record/128組・各head116支持という記録を区別した。評定差は全0で、実身体に適格な共有モデル転用と非ゼロ帰結の証明は別に残る。4,936実音対は別のenergy順位実験であり、この共有prototypeの適格性へ転用しない。初期I10範囲と後続拡張の境界を推定で確定せず、source接続を未実施として再実装しない。残条件監査のI10欄も訂正した。


### 2026-09-28: I10範囲確定の見落としを訂正、動的Hzの前段案を確認

直前節の「I10初期範囲と後続拡張の境界は未確定」「実身体適格転用が残る」という判定を撤回する。主担当が `i10-body-outcome.md:6490–6568` と `milestones.md:88–110` を直接確認した。2026-09-20に狭いI10へ確定し、共同posterior・全head・全資源受入は研究拡張またはR2へ分離済みだった。2026-09-21監査では、実身体転用の4,936対、release・任意身体・両busへの展開、資源測定と引渡しの3件を完了している。共有prototypeの旧評定差0と現energy予測の証拠は異なるが、その違いを理由にI10へ研究拡張の検証義務を戻してはならない。主計画のI10行は既に正しく限定完了を記しており、変更不要。今回の残条件監査を訂正し、後続変更が影響する部分の再検証、R2と作者・実機受入を別に維持する。

動的Hzの[peak前補間案](../../target/body-fitness-dynamic-prepeak-design-20260928.md)をsourceと照合した。72frameの平滑化済み観測NSGT power（公開690＋guard96）をnodeごとに保存し、補間後に既存frontendのpeak抽出・時間正規化を再実行する案である。主担当は `body_footprint.rs:50`、`stream/analysis.rs:131`、`landscape_spectral.rs:155` のreset、guard入力、f32 scan加算とf64 mass加算の順序を確認した。1 node 226,368 byteで、全6,145 nodeの常駐は約1.391 GB/familyとなる。旧652.19873046875 Hzの反例から現在のguard付きdirect oracleに対して検証する設計であり、精度・供給有効率の成立は未証明。既存の精度閾値は緩めない。数値probeはまだ実行していない。

I11の正式432取得は同じsession25851で140件まで全child exit0。rootもhost側のrunner PID4143663の稼働を確認した。取得の専有を維持し、他候補のcompile・計時は後に置く。


### 2026-09-28: active-index取得器を準備、無音suffix案は保留

active-index局所screenの `make_timing_plan.py` と `run_timing.py` を準備した。正式planは将来のproof/check/build/binary現物を要求し、今は未生成。scalar/pack4/active-index各144 pass、64 warmup、6順序×24、固定digest、CPU中央値C/B≤0.95を維持する。rootの読取reviewで入力の取得後照合不足を見つけ、子の成功・非zero・起動失敗いずれもstatus、host-after、全asset postflightを残すよう取得前に修正した。担当はPython構文・最適化実行拒否・閾値両側・起動失敗の小反例を確認した。Rust compile、数値probe、正式取得はまだない。

別のexact高速化案として、代表音の終端後にFFT入力ringが完全ゼロとなる区間を省略できるか静的に調べた。しかし固定出生oracleは `body_fitness_runtime_live_tests.rs:1100` でhold48,000 sampleを与え、観測は72×512=36,864 sampleである。`Voice::representative_body_recipe` はこのholdをそのままRecipeへ入れるため、release後の無音suffixが窓内にあるとは仮定できない。真の全ゼロring検出は別候補としては考えられるが、この固定3出生の費用改善を見込む根拠はなく、実装しない。既存72frameの短縮やhold変更も行わない。

I11は同一session25851で236/432、失敗0まで進行した。正式結果が出るまで元の判定を維持し、heavy処理の専有を継続する。


### 2026-09-28: 動的Hzの三点probe草案を準備

[三点probe準備](../../target/body-fitness-dynamic-prepeak-preparation-20260928/preparation.md)を作成した。旧v3 planからD-4-on-aのSine source1・generation0・body generation1、frame864、lifecycle ordinal6912、実Hz652.19873046875と両node652.1536254882812／652.447998046875、Recipe hashと同hopの690-bin環境を小fixtureへ抽出した。大きい旧rawは再走査していない。

将来の別隔離treeへ適用するcfg(test) sourceと1行module追加patchをtargetへ置いた。既存の代表Tone probeとproduction関数の平均scan/massを照合し、保存PCMを同じstreamへ再投入して全786帯域のpowerを捕捉する。補間後は既存SpectralFrontEndの72frameを再演し、重み0/1の端点で直接nodeのpeak・平均scan/massとのbit一致を確認する設計。主担当は旧dynamicのf64 log2からf32への重み、f32積和順と、実Rhai→Spawn→VoiceのRecipe復元を直接照合した。失敗は正式runnerのnonzero/panic log/partial出力として保持し、Unsupportedや端点不一致による取得失敗と、完走rawで判定する精度不合格を分ける。fixture静的照合・rustfmt構文検査・patch適用のdry-runまで成功し、Rust compile・probe・精度判定は未実施。封印済みsourceは変更しない。

I11正式取得は同じsession25851で385/432まで全exit0。性能専有を継続し、active-index局所screenをこの三点probeより先に実行する順序を維持する。


### 2026-09-28: 新版I11資源432件が完走、10/16で未達

session25851は17:02:00 JSTにexit0、432/432子processがexit0で終了した。rootもrunner PID4143663と子processの終了をhost側で確認した。27assetのpostflightを保持し、journalの全job identity・wait4・CPU和・終端状態を主担当が別途検査してerror0だった（`target/parallel-continuation-20260928/i11-resource-cost-432-journal-verification.json`）。

固定[監査](../../target/i11-resource-cost-432-preparation-20260928/runs-v1/audit.json)は整合error0、登録C/Aが10/16合格、補助C/Bが13/16合格。own_us全16は通過したが、population_usの6件が未達。Sine-flow4のreport/no-report p99はON−Noneが−75.281／−96.861 µs、許容74.240 µsで、速い側への差だった。Harmonic-flow16のreport/no-report p99は+102.541／+99.620 µs、許容94.105／95.531 µs。Modal-flow16のreport中央値は+7.040 µs、許容6.149 µs、no-report p99は+106.861 µs、許容93.923 µs。対称絶対差の基準を後から変更せず、6件とも不合格に保持する。

Sol担当の[独立再計算](../../target/i11-resource-cost-432-independent-review-20260928/findings.md)は全432 profile×611hop、AA48組、16条件の全統計・合否を別実装で照合し一致した。主担当もその処理と6件の数値・符号を確認した。48/48 reportのfootprint補遺と、432/432 childのwait4 CPU/RSS/終了までのwall記述補遺も整合した。補遺collectorの小testはAAも含むgroup総数を48と誤記して最初に失敗し、実際の80（AA32＋test48）へ訂正した検査2件が通過した。失敗記録も保持する。

取得とhost-afterの記録後、active-index standaloneのcompile/proof/checkだけを並行開始した。局所計時はI11の後監査と索引作業が終わるまで待つ。main採用・既定変更・commitは行っていない。


### 2026-09-28: I11最終索引を照合、active-index局所速度条件は未達

新版I11の最終索引SHAは `63f151e2957f54858250ae1630ad408bd6504f5f6105f6cf3d07112a31ef7d27`。主担当が1,148現物・14,015,515,689 byteを全hash照合し、差0だった。結果・入力・raw・失敗した補遺test初版も固定し、以後は変更しない。

active-indexはnativeとAVX2最小の双方で4,748 laneのbit一致、実Entry64→48 B、対象関数gather/FMA0を確認した。rootもbuild入力7件・plan/buildの32参照現物を照合した。正式局所plan SHA `d24d810342ffc6cc848b389b29b7d94cab9f9b65a91d504137ab978d1fd4b48b` を固定し、全担当の大処理終了後に実host namespace `pid:[4026531836]`・heavy0を確認して一回取得した。child exit0、全432行のdigest一致、23asset postflightも一致した。一方、CPU中央値はscalar466419.5 ns、pack4 319165 ns、active-index309185 ns、比0.9687309072で、登録条件0.95以下に未達だった。runner exit1を保持し、再試行・本体への組込みは行わない。主担当もCSV全行と中央値・比を独立再計算した（`active-index-timing-verification.json`）。

次は動的Hzの三点probeを封印packの別隔離コピーへ入れ、compile・必須検査から進める。I11は6不合格のうち増加4件と減少2件の仕事量差を読取診断する。NSGTには別候補として、4band全laneが活動する共通prefixだけmask/blendを省き、残りを旧masked tailへ渡す設計を静的検討する。速度値や3出生成立を先取りしない。

### 2026-09-28: population残差を診断、二候補の数値検査を準備

[I11の残差診断](../../target/i11-resource-cost-residual-v2-20260928/findings.md)では、失敗6条件それぞれの3反復×611 hopで生存Voice数がA/C一致する一方、render後の `schedule_renderer.active_tone_count()` は異なることを確認した。この値は新規onset件数でもpopulation処理中のTone数でもない。Sineの負方向2件はC/B補助比較の許容内で、到来CDF付き候補も0であり、CDF最適化による高速化とは同定できない。残る正方向4件も、population一括timerだけではCDF選別・Voice tick・forecast・ecology等の個別費用を分解できない。次の診断は外側区間の時刻・仕事量と観測追加費用を取得前に固定する準備へ進める。既存432件の判定は変更しない。

動的Hzの新しい隔離コピーは封印packの登録517pathと全hashを照合し、cfg(test)の新probeとmodule宣言だけを追加した。fmt・標準Clippy・all-targets検査は通過し、全Rust testsを実行中。三点の科学probeは未実行である。checkerには既存のL1・mass・score・level精度条件を先に固定し、未検証の全7504点energy再演と区別する。

NSGT共通prefix候補のstandalone native proofは実786帯域×6入力と8合成groupを照合した。全218,591 entryのうち174,557 entry（79.86%）が4lane共通prefixとなるが、これは速度値ではない。AVX2最小環境のcodegen・数値照合と取得器の固定を続け、性能計測は他のbuild・検査が終了した専有枠で一度行う。

続いてnativeとAVX2最小のbuild/proof/checkがすべて通過した。初回nativeと出力pathを変えた確認buildの実行file hash同一を求めた検査は失敗し、その記録を保持した。確認binary自体にproof/checkを実施し、正式計時対象を `probe-native-confirmed` に固定した。計時plan SHAは `7678c8df25bceb7d6544a6e0c7b5571935c0213dd5e8930eb99bc7da7a90fc96`。主担当も32 asset、build入力の前後一致、両対象関数のFMA/gather0を確認した（`target/parallel-continuation-20260928/common-prefix-plan-verification.json`）。性能計時はまだ行っていない。

動的Hzの端点については、既存probeのRust assertが両nodeの72frame peak・平均scan/massをbit照合することを主担当が読取確認した。元captureを重複出力する任意変更は追加せず、現sourceの全テストを続ける。Pythonによる独立端点照合とは呼ばず、4精度条件は保存された既存出力から独立判定する。I11の次診断草案は3scene×2report mode×A/B/C×3反復の54本と、同binaryの計測無54本を交互順の隣接pairにする108本。6つの連続・非重複区間を測り、計測追加費用と非同期仕事量差を保持する。source patch・binary・checker固定前の実行不能draftであり、取得済みとは扱わない。

### 2026-09-28: 三点checkerを照合、I11区間診断を隔離実装

主担当が動的Hzの新treeを親517登録pathと再照合し、差分は `src/runtime/mod.rs` の宣言追加と新probe fileだけ、登録元の欠落0だった（`target/parallel-continuation-20260928/prepeak-source-verification-v1.json`）。Solの独立read-onlyレビューでも、72frame reset、guard786のshapeと公開690のmass、log2重み・f32積和に重大な齟齬は見つからなかった。checkerのmass相対差は旧v3bと同じf32総量を使い、PMFはf64総量で正規化する。peak tupleと補助spectral mean massの有限・形状検査を追加したが、Pythonによるpeak抽出の独立再演とは呼ばない。取得前文書の「Modal」はfixtureの `kind=sine` と一致しないため、固定前にSineへ訂正する。Rust sourceは追加変更せず、同じ全テストsession11809を継続する。

I11の新隔離コピー `.worktrees/i11-population-breakdown` は、封印archiveの543 regular fileをmanifest `1b393a84…` と全照合して作成した。編集は `src/runtime/mod.rs` と `src/runtime_profile.rs` に限り、明示した診断envで6区間の時刻・件数を別sidecarへ保存する。旧HopProfile schemaは維持し、U/Dは固定Serialize入力でbyte一致を検査する。区間内の同期enqueue/lock/FIFO待ちは壁時計へ含まれる。Uも変更後binaryの無効経路なので、共通分岐やcodegen変化をU/D差から除去したとは扱わない。容量超過・欠測行・sidecar書込失敗時のprofile保持を検査してから、fmt・Clippy・all-targets・全Rustテストへ進む。

hostは32 CPUであり、性能を判定しないI11必須検査は2jobs/2test threadsで動的Hzの既存統合テストと並行する順序に改めた。NSGTの正式局所計時は双方のheavy終了まで延期する。I11の108診断取得と動的Hzの三点科学probeはいずれも未実行である。

### 2026-09-28: 二つの全suite通過、共通prefixの局所条件は未達

I11区間診断の全検査v1は、sidecar書込失敗を起こす反例で `write` と `flush` のエラー文言を過限定したassertが1件失敗した。失敗log/statusを保持し、そのassertだけを修正したv2はlib全成功だが、復元archiveにsamplesがなく統合検査が失敗した。fixtureを補ったv3でもRhai.toml不足が判明したため、親checkoutからテスト入力65ファイルをbyte同一で補完した。元543 source manifestは変更せず、追加fixture manifest SHA `83547acf3a34607f6f3dc749dee2bc8dfeb80a824c0448df41d42143219c312f` を別に固定した。主担当も親・コピー双方を全hash照合し、543＋65＝608の固有pathを確認した。archiveだけで全テスト入力が揃うとは扱わない。

v4全suiteは18:00:31 JSTにexit0、32群1,206成功・0失敗・36ignored。fmt・Clippy・all-targetsはsource不変のv2通過を維持し、canonical test_report/statusはv4からbyte同一・別inodeで保存した。主担当も全群集計・log SHA `e88a1efd3e721bf2079c8a9c50428c566f413bc7e1ea3ab1231fcc6318621c4f` とcanonicalを確認した。最終src差分はruntimeとruntime_profileの2件のみ。108本取得器は終了値・wait4 CPU/RSS・全入力前後hashに加え、build/source/binary/toolchainの系譜と終了後plan・親plan・host-beforeの不変を検査する。診断checkerは候補処理時間を仕事量件数から分離し、body_candidate_energyのprocessing_usを判断内容の差に数えない。

動的Hzの同一session11809も18:01:44 JSTにexit0、43群1,402成功・0失敗・55ignoredとなった。主担当の独立集計とlog SHAは `d0366e5edf475eebb29cf13e284457e3a582d91c7c158dd561acf46fcc51bea3`。親517＋新probeのsource manifestとは別に、samples31とroot README.md／Rhai.tomlの計33テスト入力を固定し、親・コピーとも全hash一致を確認した。fixture manifest v2 SHAは `0217916ce014df2851c475e2efdfa057bf735b6db914dde2e099e617c9b38402`。新しい科学probeはignoredのままであり、全テスト通過を三点精度の成立には読み替えない。

両系統のheavy終了後、18:03:35 JSTに実host namespaceと対象heavy0を確認し、登録plan `7678c8df…` の共通prefix局所計時を一度実行した。child exit0、全432行のdigest一致、32asset postflight一致。一方、CPU中央値はscalar366,775 ns、旧pack267,295 ns、common-prefix257,935 nsで、C/Bは0.96498250996、約3.5%短縮に留まった。登録0.95以下を満たさずrunner exit1、再計時・本体組込みなし。主担当も全行の順序・digest・中央値・比と入力を再照合した（`common-prefix-timing-verification-v1.json`）。専有枠を解放し、I11と動的Hzのrelease buildを2jobsずつ並行開始した。次のNSGT案は、代表身体の複数time-frameをSIMD laneにして係数を共有する方式のread-only設計調査であり、通常知覚hopやcooperative取消境界を変えず成立するかは未確認である。


### 2026-09-28: peak前補間の三点診断が通過

動的Hzのrelease buildは18:09:24 JSTに終了した。主担当が固定plan SHA `85bd9c4c3187eed82d29a2223750adb9c3d41dd9f41e742ac45f75c91f16fda9` の11資産、518 source、33 fixture、build metadata内11参照の現物を照合し、取得を承認した。単回probeは18:15:54–55 JSTにexit0で終了し、入力postflightと端点Rust bit assertが通過した。rawは2,281,982 byte、SHA `6afbdc7d98c19fa657f56ae6c45204698693aa2646b69afb3d387dd38394b646`。

登録Sine一点の4条件はすべて通過した。normalized ERB mass L1は2.4326023271e-6（基準1e-3）、fitness mass相対差2.0370958765e-4（1e-3）、score差1.4901161194e-8（1e-4）、level差5.9604644775e-8（5e-5）。主担当が保存mean scanから別途4値を再計算し、checkerと一致した（`target/parallel-continuation-20260928/prepeak-independent-quality-v1.json`）。peak抽出前powerの補間が旧反例一点を改善する局所証拠であり、全family／query、single interval energy、7504 step replay、通常runtimeの成立には読み替えない。

I11診断版のrelease buildも完了した。108本は同じ新binaryのU/D隣接54pair、元432のうち3scene×2mode×3variant×3反復を維持する。取得前レビューでfixture manifestのsource_rootが親treeを指し、新treeの65 fixtureをrunnerが再確認していない箇所を発見した。親と新treeの両方を前後照合する修正と、新treeだけ変更する反例を追加して10件通過した。先に固定したplan-v1/v2と途中test失敗は保持し、修正後の別planを固定してから取得する。取得済み432の合否は変更しない。


三点結果の比較範囲を補足する。旧dynamic-densityのruntime constructorはguardなし、今回のpack継承版は公開690にguard96を加えた解析である。したがって今回の通過は現行guard付きdirect oracleに対する局所精度であり、旧L1値との差を補間方式だけの改善量と扱わない。次の30点案も旧最大L1による反例選定であり、新方式の全query最大誤差を保証するものではない。

I11は最終plan-v3 SHA `c747f1f149345b01cd8f6db7b5eaa886db87c243b5df0c2ac6e04bbcd161bb9e` を固定した。主担当が729参照を照合し、host namespaceとheavy終了を確認して108回取得を承認した。host-beforeは18:19:27 JST、担当session10777で開始し、25回まで全child exit0。取得中は他担当のcompile・probe・計時を停止し、軽量な次段準備だけを進める。区間median/p99の和を外側統計と同一視せず、同hop区間＋residualの加法関係を用いる。


その後、保存rawによる旧新比較でこの一点の範囲をさらに確認した。D-4:s1/e6912/d1536のdirect mean scanは690/690 bin、両端nodeも各690/690 bin、spectral mean mass、Hz bits、Recipe hashが一致した。旧L1 1.52913113に対して新L1 2.4326e-6となったのはdirect参照値の変更によるものではなく、補間scanの4 binが変わった結果である。bin342の旧97.6307907は新0、bin343の旧29.2352676は新126.8774719となった。この一点での比較可能性と、他の周波数・音色でguard影響が未検証である点を分ける。封印済み三点証拠は変更しない。


### 2026-09-28: I11区間診断108本が完走、仕事量差を保持

18:42:22 JSTに担当session10777がexit0で終了し、108/108 child exit0、729参照のpostflight一致となった。journal SHAは `f3c3e031eac2003fcf07edaefaf32593ff01638024f5c415d249f306b46e710b`。主担当も全job identity、54組の順序、wait4 CPU和、host-before/after/CPU記録のhashと終端を検査した（`i11-population-journal-verification-v1.json`）。固定summary v2は54/54組・18/18条件・診断32,994/32,994 hop、整合エラー0だった。主担当の別実装でも108 profileと54 sidecarを集計し、全hopの区間和と残差を検算、702統計値が完全一致した。

U/Dの生存Voice数とrender後Tone数は全組で一致したが、候補energy workerの件数は45/54組で異なり、粗い観測仕事量一致は9/54に留まった。reportあり27組の時刻項目を除いたrecord列もすべて不一致であり、順序や挿入による差を参加判断そのものの変更とは読み替えない。区間別p99はenergy context・判断処理区間が大きいが、区間統計の和をpopulation統計へ足さず、旧432の失敗量への直接帰属も行わない。旧受入れ10/16合格・全体不合格を維持する。

性能取得終了後、4 frame方向SIMD standaloneのnative/AVX2最小build・proofまでと、別隔離版でのprepeak30点screenの全Rust検査を開始承認した。NSGTのscratchは各bin `[re0,re1,re2,re3,im0,im1,im2,im3]` の524,288 byteで、転置費用を計時に含める。初回buildは合成反例のarray長推論E0284で止まり、失敗logを保持して型注記だけ修正した別v2へ進む。計時は両系統のheavy終了後の別承認、30点数値取得もsource/binary/plan固定後の別レビューとする。


4 frame SIMDの型注記修正版v2はnativeと最小AVX2のbuild/proof/checkを通過した。6系列×4 frame×786帯域＝18,864 band-frameでbits一致、入力digestは両buildで同一。実逆アセンブルで疎積和loop内のFMA/gather/shuffleは0、loop後の出力整列にはvshufpdが1件ある。主担当もbuild証拠31参照と固定計時planの39参照を照合した。plan SHAは `140febc609686f836e4ebcc31ded132d017182228df291cb238430ac9aa94404`、転置込みCPU中央値C/B≤0.95を維持する。計時はprepeak30版の全Rust検査が終了するまで待機し、現時点で速度向上は主張しない。

prepeak30版 `.worktrees/body-fitness-dynamic-prepeak-30` は親518 source＋33 fixtureを全hash照合した別コピーにprobeの最小差分だけを適用した。旧3 reportのSHAも全一致、30 fixture生成・fmt通過。専用target/TMPDIRと2jobsでClippy以降を進行中であり、新版の全suite・release・科学probe完了とはまだ扱わない。


I11 report27組の事後identity照合では、`(voice_id, now, due_frame)` の判断集合2,988件がU/Dで完全一致した。record全体では1,334件に差があったが、存在を全件要求した選択5字段 `selected_offset/selected_at/selected_cost/skipped_cycles/skipped` と `onset_allowed` は差0だった。候補値差は5判断。主なrecord差は `footprint_received_at` 1,193件と `footprint_requested_at` 329件である（字段別件数は重複する）。None条件の判断0も明示した `posthoc-decision-fields-v2.json` を保存し、v1の欠測双方をget(None)で同一扱いし得る検査を補強した。参加判断の最終出力一致と背景worker仕事量の非同一を分離し、旧資源受入れは変更しない。

prepeak30版はfmt・標準Clippy・all-targetsを通過し、全Rust suite session96151で進行中。source差分はprobe1件のみ、旧3reportの現物SHA一致を確認した。releaseと30点数値probeは未実施。


I11区間診断の[結果本文](../../target/i11-population-breakdown-acquisition-20260928/result-v1.md)と最終索引を固定した。index SHA `d6be5e75aa05f17b0a6ad392a2e1a125a04a0cb3a2a02258c0d329e577f54d4a`、434現物・7,035,945,529 byteを主担当も全hash照合して差0（`i11-population-final-index-verification-v1.json`）。108取得・固定集計・事後の判断字段補足を含むこの診断単位は完了した。I11資源受入れ自体は未達のまま。prepeak30版の全suiteとNSGT time4の未実施計時を次の技術作業として継続する。

### 2026-09-28 継続: 30点版の同一性と次の局所計時

主担当がprepeak30版の親518 sourceと33 fixtureを独立照合した。変更はprobe一件だけで、capture・frontend replay・補間・端点照合・fitness計算の数値本体は封印3点版とbyte一致した（`target/parallel-continuation-20260928/prepeak30-source-numeric-review-v1.json`）。全suiteのfactorial区間は964.65秒で通過したが、全suite終端・release・30点取得はまだ未完了。性能計時との競合を避け、suite終了後にtime4を一度だけ計時してからreleaseへ進む。

I11の次候補は、成熟energy forecastの201個の保持率をconstructorで計算するstandalone検査に限定した。[登録](../../target/i11-energy-retention-preparation-20260928/registration.md)とplan SHA `79121a21bc6f49dc3702e3b984850f2d3785e6b76acaa178fe668010d49721f6` を固定した。96状態の201 horizon×3帯域、57,888値のbit一致、任意aheadの1,152値、96回の再生成を検査済み。主担当も12資産と旧参照関数の封印ソース一致を照合した。32 processのABBA計時で成熟CPU中央値比≤0.95を事前条件とするが、計時は未実施。投影したhistory構造体の局所検査であり、production layout、実forecast頻度、通常hop費用、旧I11資源判定の改善は未検証。

NSGT time4の既存bit証拠は合成FFT入力の4 frame疎積和までを拘束する。実統合時にはTone/FFT/平滑/frontend/平均の順序に加え、block内・frame3→4境界・72 frame終端の取消、部分表の非公開、古いidentityのReady棄却と応答遅延を別検査する。read-only整理は `target/nsgt-time-batch-integration-boundaries-20260928.md` に保存した。

### 2026-09-28 全suite完走と2候補の単回計時

prepeak30版は19:29:04 JSTに全suite exit0、43群1,402合格・0失敗・55無視で完走した。主担当もcanonical report SHA `abb46dd7f70fcc42e8107d9f67f40cbfa09a5a94ae63ca94d570754795c31da5` と全群の集計を照合した。新sourceは518件44,828,424 byte、manifest SHA `e36cc7a4d6cd9b9f12448d5be3e530da7d3102595c9e2ea4de121f2983784e67`。全heavy/hash終了後、実ホストnamespaceと競合build/test/render不在を記録し、次の2計時を順次一回だけ実施した。

- time4は全432行・digest・39資産のpostflightを通過し、pack4 CPU中央値1,111,329 ns、time4転置込み541,730 ns、比0.48746140881773087で事前0.95以下を通過した。主担当の独立CSV再集計も一致した（`time4-independent-reduction-v1.json`）。合成FFTの局所合格として封印し、親packから別隔離treeで代表72 frame専用統合へ進む。通常runtimeの採用、全690表、30秒3出生の合格とはまだ扱わない。
- I11保持率cacheは全32 process exit0、mature/coldの出力digest一致、12固定資産一致を確認した。成熟CPU中央値は参照0.0990495秒、候補0.0971065秒、比0.9803835456009368で事前0.95以下に届かず不合格。主担当も全raw/journalから独立再計算した（`i11-retention-independent-reduction-v1.json`）。coldの中央値は参照0.001683秒／候補0.001473秒、constructorは参照0.008694秒／候補0.0133445秒だが、状態構築とdigestを含むprocess総CPUであり、constructor側はdigest操作数も非対称である。純粋なconstructor単体性能とは呼ばない。再計時・production実装は行わず、この候補を閉じる。旧I11資源10/16合格・全体不合格は不変。

ホスト前後とrunner終了記録は `target/parallel-continuation-20260928/host-*-time4-retention-v1.json` と両driver log/statusに保存した。計時終了後、prepeak30版release buildを開始した。新binaryと正式30点planの照合が済むまで科学probeを開始しない。

局所2候補の最終索引も主担当が全現物を再hashした。time4は48項11,737,062 byte、index SHA `bbb1a179278b9775c9e2d7dd1d319cdd3c15c73b3ddd184872d386ec20e354ea`。I11保持率案は72項4,642,730 byte、index SHA `9848292b98e4e499c92387f395788110499c08ebb4a7b75cd562e35e0ab1fc10`。time4の相対pathを最初のroot照合器が絶対pathと仮定して停止したが、索引のrootから解決するよう直して全件一致を確認した。候補本体や取得結果の失敗ではない。

prepeak30のreleaseは固定binary SHA `4e965e3669f46df86be24233d1b23c22d64d58c34d3f6b1c5168739c21be274a`、build metadata SHA `288443cde108f526ca832505b4124b300d70b3f11525adba609b42344894696d` で固定した。正式plan SHA `7fd13d0ab5c9cfd4d3d5f8903c895ef9b404418c7226c0f605b71e51a412b192` の64個のunique資産・build descriptor、compiler artifact原本と固定copy、source前後を主担当が照合した。未使用のoutput先を確認して全30点の単回科学probeを開始した。

prepeak30の単回取得は19:36:00–25 JSTに完了した。全30 child exit0、postflight一致、integrity失敗0。登録した4精度条件は17/30合格・13/30不合格で、runner exit1を科学判定として保持する。主担当は全raw・log・fixtureのhashと120品質scalarを独立照合し、すべてexact matchだった（`prepeak30-independent-quality-v1.json`）。5familyの格子順12/24/48/96/384/1536の合格列は、それぞれFFFPPP、FFFPPP、FFPPPP、FFPPPP、FFPPFP。Modalのq28（d384）はL1=0.07009934946399432のみ閾値を超え、同じ照会時刻のd96とd1536は通過した。最細選択点の5/5合格を全照会や単調な格子収束へ一般化しない。追加取得・閾値変更なしでこのscreenを固定する。

I11の別候補は成熟forecastのloop順をlag外側・horizon内側へ変更する。各horizonのlag加算順、cold、任意aheadの元関数、new/resetは維持し、保持率201値はforecast内で計算する。参照式抽出、固定12資産、96状態・57,888予測値bit一致を主担当も照合した。任意aheadの1,152値は元関数の有限性検査であり、別候補式とのbit比較ではない。plan SHA `ce982a071331e667191d785ba903d27bb98fb867f83cc13d98f12ec38555e69c` を固定し、他のbuild/test/renderと大量hashを止めた実ホスト枠で24processを一度だけ取得した。

全child exit0、mature/cold内digest一致、資産hash一致。成熟CPU中央値は参照0.1016265秒・候補0.0932725秒、比0.9177970312861312で事前0.95以下を通過した。主担当の独立全raw/journal再集計も一致した（`i11-loop-interchange-independent-reduction-v1.json`）。coldは参照0.001421秒・候補0.001547秒の記述対照である。投影historyを使い起動・状態構築・digestを含む局所process CPUであり、通常hop費用の結果ではない。封印resource-cost親543 sourceから新隔離候補へ同変更だけを実装し、実historyのbit境界検査と全Rust suiteへ進む。旧constructor cache案の不合格と旧432資源不合格は維持する。

prepeak30のModal q27/q28/q29は照会Hz bits1149061864だけでなく、Recipe hash、frame112、保存環境、直接oracleの全72frame power/peak、mean scan/mass、fitnessもbit一致した。変わるのはnode Hzと補間格子である。同じ照会・同じ直接oracleに対する96→384→1536のPASS→FAIL→PASSとして結果文を訂正した。全30点では幅ごとに照会を別選定したものもあるため、この局所反例と全群集計を分ける。

prepeak30の最終索引は755件529,224,910 byte、SHA `0db0179887edd60f94138f5b483777dc8eb46576759875d2677bad8c4ed29bd8`。主担当も全現物hashを照合した。I11 loop交換の局所screenも52件4,598,364 byte、index SHA `e7fdf9cff72d46954087fbd7ae02bf1eb7127ab1a56afce5c3f8aa40cbf36073` を全照合して固定した。両索引の以後の変更は行わない。

継続先はNSGT time4の新隔離tree `.worktrees/body-fitness-nsgt-time4`（親pack517 source継承）と、I11 loop交換のresource-cost親543 sourceからの新隔離実装である。time4は初回all-targets checkと72frame×786帯域のframe別一致focused検査を通過し、身体生成・取消の検査と全suiteへ進行中。I11は通常構造体の実装・bit境界検査へ進む。いずれも局所計時を通常runtime受入れへ読み替えず、次の科学・資源取得は新source/binary/planの確認後に行う。

### 2026-09-28 time4とI11 production候補の独立レビュー

time4の親517 sourceに対する変更は `nsgt_rt.rs`、`stream/analysis.rs`、`body_footprint.rs`、`community/async_birth.rs` の4件。独立読取レビューを `target/nsgt-time4-independent-review-20260928.md`（SHA `317cc0b9777ee2ec777506361dd41cd7714ab06992102348fbda5d8f2b09150f`）へ固定した。通常process_hop不変更、4hopのTone/FFT・band内平滑・frontend・平均の順序、取消時の部分表非公開、候補ごとのfresh解析状態とActiveJob内scratch再利用にblocking所見なし。core72×786、frontend72、Sine/Harmonic/Modal身体72のbit一致とworker取消のfocused検査は通過した。初回Clippyの形式指摘を保存しv2で通過、all-targetsも通過。全suiteはsession5167で実行中、source編集を止めて終端を待つ。4 test threadsは検査条件として記録し、性能比較へ使わない。

I11新隔離treeは `.worktrees/i11-energy-loop-production`。親543 sourceとの差分は `temporal_expectation.rs` 1件で、読取レビューは `target/i11-energy-loop-independent-review-20260928.md`（SHA `541a10f7db3fa285e2bd67627457de1524d8d91ce51f86abfbc67f3836926208`）。成熟forecastの各horizon内lag加算順、旧energy_at、新規構築/reset、cold分岐の保存にblocking所見なし。実historyを用いた境界testを強化した。実48 kHz stepは0.01と同bitsなので、このproduction testの96実行は72一意状態であり、standaloneのordinalを変えた96状態と区別する。親のcanonical test report/statusを別保存し、新候補は未完了状態へ更新した。全suite後、固定親rendererと36＋8条件の機能回帰、通常資源判定への接続を別登録で確認する。

I11 production候補は19:53:22 JSTに全suite exit0。主担当も32群1,204合格・0失敗・36無視、NUL0、固有log SHA `fcfc236944f0caf481597e5f800523975be68fea67274eccb51be53df61507e6`、canonical reportのbyte一致と別inode、同shell終了statusを照合した。親543 source全件を主担当も照合し、新版の変更1件と `TemporalExpectation` 構造体・new・旧energy_atのbyte不変を確認した（`i11-loop-production-source-review-v1.json`）。65 fixtureも親・新双方で全hash一致した。source後検査も通過し、fresh releaseと36＋8条件の機能回帰plan準備へ進む。資源432と通常受入は未実施のまま。

I11 releaseはrenderer SHA `4c1579b134411de7fd822faf987e6c1f574d6f9844d6eb61b71579fc9c795f5c`、instrument SHA `d3c3a5715e66bab327122d060265c32cf395a9468e4577aa64f132c5881192f5` で固定した。source manifest SHA `f021034565d87d04008b8902bd0c7f8aa6eec9c86c950c263de1786336fc9b2a`、archive SHA `8ca4af8b3ae6fb64e5803950a6667162270b05b3e6ade9077caf9530eb6fbe74`。主担当は98資産、archive内543 source、binary原本/固定copy、build前後source/configを照合した。36本plan SHA `cb01abd3529d00cb7d29f4caec4fd6596b4ff66e3cb7ec0168dfa1067cc72076` は旧inputs/baseline/runs/criteriaと一致。8本plan SHA `388654b465cacfebbc0aab20bb4d38c14e36a48455f7a17a7f31ce2a88feaec6` もrenderer/output/self-plan参照の置換以外にargv・後検査・limits・基準出力の差はなかった。初回root比較器はself-plan参照の置換を省いて停止したが、plan不変更のまま正規化を訂正し一致を確認した。36本の単回機能回帰をsession47819で開始し、完走後8本へ進む。NSGT suiteとの同時実行であり、時刻/CPUを性能証拠には使わない。資源432は別の専有枠・正式登録を待つ。

動的Hzの次段は幅96/1536の全prior_actual 2,030行を対象に準備する。主担当も旧planから機械的に抽出し、各幅1,015行、環境適格2,010行・initial_shared環境欠測20行を確認した。さらにD-4:s1のon-a/on-b・e0・両幅の4行は実Hz=低端=440 Hzで、20欠測行に含まれる（`prepeak-actual-population-review-v1.json`）。新しい隔離cfg(test) probeはnull環境とこの端点を明示処理し、分布L1/massは取得、score/levelは欠測理由付きnullで成功に算入しない。capture/補間演算/端点replayは維持する。2010行へ分母を縮めず、全2030の完結性と新source/binary/planを確認するまで数値取得しない。限定cache/cold費用・全synthetic/multi-environment・energy再演・paced高有効率は別の未検証事項である。


### 2026-09-28 I11の36条件回帰と動的Hz全実点版の確認

I11 loop交換版の36取得・固定4後検査はすべてexit0で完了した。主担当も全36組のraw WAV/reportのSHAを照合し、旧None全12 WAVと時刻等の登録除外字段を除いた10種学習record列を独立再計算して一致を確認した。共通候補11,386件はprocessing_us以外すべて一致し、旧側だけ191件・新側だけ323件を別集計した（`target/parallel-continuation-20260928/i11-loop-functional36-independent-review-v1.json`）。body/proxyのHarmonic/Modal6条件は選択差とWAV差を持ち、Sine-flow3条件は選択差0、Sine-hold3条件は判断0だった。到来8条件を続けて取得中で、36条件の既定arrival=falseと区別する。並行suite中の時間は性能証拠へ使わない。

全prior_actual版 `.worktrees/body-fitness-dynamic-prepeak-prior-actual` は親518 sourceのうち診断probe 1件だけを変更した。主担当も親全件、継承33 fixture、2,030新fixtureと旧planのquery/family/node/environmentの一致を照合した。capture/replay helperと補間数値本文は封印30点版とbyte一致した（`prepeak-prior-actual-source-fixture-review-v1.json`）。probe SHAは `494152fc32ec9162c5648e8dd9d8040ab87a8dd84e0a0c93df32a658f6c64d2b`、selection SHA `0002db0a4810de1d182044df2b53ecea2f56d74937c1cc736cf6c66a56b78fd5`、fixture index SHA `df8841c57a5a14a2a87f70aa0fd1efc8e133a6fa9b8ef791cb44116fed36659c`。20欠測点のmassは既存ERB演算を零Cで通して得るが、score/levelを人工環境で成功とせず理由付きnullで出力する。低端4行はRustとPython双方が直接点と低端点の保存bitsを比較する。一般nodeのfrontend replayにPython独立再実装はない。fmt・標準Clippy・all-targetsとPython反例8件が通過し、全Rust suiteはsession66982で進行中。正式source/binary/plan固定前の数値取得は行わない。


I11到来8本も全child exit0とpostflightを通過した。主担当は新raw30件・旧基準14件をhash照合し、全8 WAVのbyte一致を確認した（`i11-loop-origin8-acquisition-review-v1.json`）。登録21後検査は全exit0で、主担当もargv・log/output SHAを照合した。続く4 complete joinも全exit0。補助plan SHA `f84e401cb283b0f315ef9fe08dc8f1b4a99a6047e23f12f0c1f75aefe61b20ff` と35入力を全照合し、112判断・5,256 producer、complete_observable_join 2件・unknown 110件を再現した（`i11-loop-origin-postcheck-review-v1.json`）。物理的source帰属、retained Tone route、通常自己群除外正例の追加達成とはしない。機能回帰36＋8条件は完了し、最終索引と新432条件の登録準備へ進む。旧資源不合格は不変。

time4のfocused検査範囲を登録草案と照合した。実測は72frame×786 smoothed power、72frame公開scan/mass、3身体の最終平均bits、poll20のblock内取消、ActiveJobの候補間取消と部分表棄却である。PCM/FFT複素中間bits、frame3→4や最終frameを狙った取消、非AVX2強制fallbackは直接未測定。Incoherent/非AVX2がscratchを生成せず旧経路を通る点はsource構造確認であり、実測と分ける。後続720PCMはordinary経路の不変対照、全690はtime4 workerと旧cancellable oracleの直接対照として扱う。


I11 loop交換版の[機能回帰結果](../../target/i11-energy-loop-production-results-20260928/results.md)と最終索引を固定した。index SHA `593b347c652f20cb24354570850b2e3eac9157a50fc66b10d0f1747f09a61aa3`、997現物・4,125,839,375 byte。主担当も全現物のsize/hashを照合して差0を確認した（`i11-loop-functional-final-index-review-v1.json`）。新432条件は登録準備へ進むが、取得は未実施。time4全suiteと動的Hz2,030点版全suiteは継続中であり、終端前に合格とはしない。


### 2026-09-28 time4全suite完了とI11資源plan固定

time4全suiteは20:26:23 JSTにsession5167でexit0となった。主担当も43群1,405合格・0失敗・54無視、NUL0、report SHA `87305dcf472ec8435bee505a503bee3070c9f0337d168ae45a5d75b6caf3dc88`、canonicalの別inode・同byteを確認した。suite前517 sourceと現物全hashが一致し、親との差分は登録4件だけだった（`time4-production-suite-review-v1.json`）。source capsuleとreleaseを準備し、取得順は新720PCM→全690 ABBA→30秒3出生を維持する。

I11 loop交換版の[資源432条件plan](../../target/i11-energy-loop-resource-432-preparation-20260928/plan.json)はSHA `b81ea224031dad24c07191041b27371dfecbe1c6225e8546b30024e5ef34818f` で固定した。主担当は旧親planとの13字段、全432 jobと順序、seed20260918、611hop、AA floorを含む監査・統計参照がJSON一致すると確認した。実行・監査等6コードはbyte不変、binaryを含む28資産の現物hash/modeも一致した（`i11-loop-resource432-plan-review-v1.json`）。14局所検査とpreflightも通過した。新候補の通常資源値はまだ未取得であり、旧対称判定10/16合格・全体不合格は維持する。性能取得はNSGT・動的Hz版のbuild/test/大量検算を終え、ホスト専有を確認してから一度だけ実施する。


全prior_actual 2,030点版は別Sol担当が生成器・取得器・検査器・Rust差分を独立読取した（`target/prepeak-prior-actual-independent-review-20260928.md`）。取得前の行除外・閾値緩和につながる所見はなく、2010適格/20欠測、4低端、10群の保持を確認した。正式planは未発行なので、binary/source/全assetと未使用output先の照合は後段へ残す。一般frontend/peakのPython独立再実装は対象外である。rootも同レビューを読んだ。動的Hz版の同一全suiteは実ホストでfactorialのbody_body-a rendererがCPU稼働し、case進行を確認した。出力待ちを停止と誤認せず同じsessionを継続する。


time4 release libtestを固定した。binaryは59,745,776 byte、SHA `0fe55ecace67a46bd56215b11879ec978600490e21eb419991d06507ad830e1d`。build metadata SHAは `7ecd2ba1531d92c7ce016f5695582a6cf76b5f674d419e67d7d252654decd97b`、source manifest SHAは `394b0491440695db0b37c553ecae22e5c98ff5090423cb1372e28e0f906a40b8`（517件44,827,685 byte）。主担当は全sourceのlive/capsule双方とbuild前後298 Rust/Cargo、16 build参照、compiler artifact原本とfixed copyを照合した（`time4-release-custody-review-v1.json`）。実buildはCARGO_BUILD_JOBS=4で、共通方針2との差を記録した。再buildはせず、build時間を性能証拠へ転用しない。以後のbuildは2jobsとする。metadataに直接載っていないbuild JSONL/stderr/statusとorigin artifactもroot証拠に束縛し、正式取得planへ含める。空いたSol担当に720PCM固定planと旧pack/cooperative照合準備を分担し、ownerは全690 ABBAとpaced登録を進める。数値・性能取得はまだ未開始。


### 2026-09-28 time4のPCM同値とABBA入口の不一致修正

PCM inner plan SHA `28fadd956261e602f8d7a05f944eba928b169114e5007c8d44184534c681fb0a`、outer plan SHA `a095fae638380d88312923a3124773a681eb9b2ecea47d0944129d51bcb6a23c` を主担当が照合した。47資産と元10queryのPCM/power/scan 30件が固定hashと一致し、出力先未使用を確認した。20:36:38–42 JSTに主担当が一度だけ取得し、replay/checkerともexit0・comparison completeとなった。全10 query×72 frameのJSONL計18,979,351 byteがpack/cooperative双方とbyte一致した（`time4-pcm-plan-review-v1.json`、`time4-pcm-independent-review-v1.json`）。ordinary saved-PCM経路の不変対照であり、新time4 workerを直接計測した証拠ではない。

続くABBA草案の実行入口を主担当がsourceまで追い、`async_birth_representative_density_oracle` が新binaryでも旧 `representative_subjective_intensity_cancellable` を直接呼ぶと確認した。time4の接続先は `ActiveJob::advance` だけであり、この草案を実行してもtime4の速度は測れない。取得前に停止し、正式planは未固定・実測は未実施のまま草案を保持する。別Sol独立レビュー `target/nsgt-time4-abba-independent-review-20260928.md` も同じ経路不一致を主所見とした。副所見として、4本oracle JSON相互一致だけで保存baseline JSONへの直接byte照合が欠けており、新取得器ではこれも固定基準に含める。誤った経路を測って速度不合格を作らない。

新しい比較用コピーは `.worktrees/body-fitness-nsgt-time4-job-probe` と `.worktrees/body-fitness-nsgt-pack-job-probe`。両者へbyte同一のcfg(test)固定Job入口を加え、同じscenario/reserve/全690Hzから実 `ActiveJob::new/advance` を使う。旧数値oracleは独立経路のまま残す。新入口は正式source/binary/planの確認前に試走しない。一般の生産コード・通常設定・封印parentは変更せず、必要fixture継承、source差分レビュー、全Rust検査とreleaseを両版で行う。比較用入口の追加でtime4取得枠は先送りし、動的Hz2,030点版は同一suite→release→正式planを並行継続する。


固定Job診断入口のtime4側初稿を主担当が読取り、引数を既存tupleで整理し、終了表のticket/serial/child_id照合を補った。最終差分は `async_birth.rs` のcfg(test) Community helperと `body_fitness_runtime_live_tests.rs` の新ignored testの2件だけ。helper部分を除くと親async_birth.rsへbyte完全復元し、runtimeの旧oracleを含む全prefixもbyte不変だった（`time4-job-probe-source-review-v1.json`）。helper SHA `29007f78c4d7568b1d435e540e196eaa8061bd61b33375e994895c1e67114874`、追加test SHA `6e3414582d7a191ced598c675d38b259c083bc8fbcf5f41d12d412d14461617d`。旧側へはこの追加2blockだけを同byteで移植する。ファイル全体のコピーは旧ActiveJobをtime4へ置換するので行わない。新ignored690自体は未実行のまま、通常全suiteでコンパイルと既存回帰を検査する。計測入口のenvからendpoint capture指定も除き、固定Job以外の診断処理が混入しないようにする。

旧pack-job-probe treeではsource517外の検査入力33件が欠けていた。元33fixture manifestと親packの実物がすべて一致すると確認し、親実物をcopyして新treeでも全hashを照合した。source archive単独と全suite再現に必要な検査入力を分けて記録する。


旧pack-job-probe側も主担当が独立確認した。親517 sourceと追加33 fixtureは固定hash一致。差分は追加helper/testの2fileだけで、helper除去後の旧async_birth.rs全文、追加test除去後の旧runtime全文が親にbyte一致した。追加helper/testは新time4側と同byte、旧async_birth.rsにtime_four参照の混入はなかった（`pack-job-probe-source-review-v1.json`）。この版で両側の専用target・2jobsによる検査を開始した。両ignored690 probeは取得前未実行を維持する。


全prior_actual版の全suite session66982は20:50:01 JSTにexit0で完走した。新canonicalへ直接teeし、同shellでstatusを書いた記録であり、別logからのコピーではない。主担当も43群1,402合格・0失敗・55無視、NUL0、3,748,711 byte、report SHA `67b6f68846b4574b1c864ef68b89f1608df53b70d53b85b50e4231578734174d` を確認した。source capsuleは518件44,831,077 byte、manifest SHA `335014fe5cb871716eaadc5bd438f9fd7975b18930e331867bd1a2500261b8be`。主担当は全live/capsule、追加33fixtureを照合し、親30点版からの差分がprobe1件だけと確認した（`prepeak-prior-actual-suite-freeze-review-v1.json`）。jobs2 release session61898へ進行し、2,030点の数値取得は新binaryと正式planの確認後とする。


### 2026-09-28 全prior_actual 2,030点の正式取得開始

releaseは2jobsで完了し、fixed libtestは59,791,168 byte、SHA `dd441a9c0589b7c219fba79ef57db69c3e69ba195610a02c620a832316a75ddb`、build metadata SHA `cb3180aa15d9d223bb0973d21cf79d3971f0877925f7caac69df2c638c88507d` となった。正式plan `target/prior-actual-validation/probe-plan-v1.json` はSHA `f3207fd1d59e29d9216ee9d23867d95f6ebccf858a91598630894484981cfbd8`。主担当は2,062参照資産のsize/hash、実compiler artifactとfixed binary、build/check status、旧親planから抽出した全prior_actualの2,030件の順序とfixture queryを独立照合した。score適格2,010、環境欠測20、低端一致4件を保持し、出力先未使用も確認した（`prepeak-prior-actual-formal-plan-review-v1.json`）。

担当は固定runnerをsession10296で一度だけ開始した。全childの失敗・部分出力を保持し、再試行や閾値変更はしない。新旧Job-probe版の通常suiteと並行する精度取得であり、取得時間は性能の証拠にしない。実ActiveJobのABBA、30秒paced、I11資源432条件は重い取得・build・検算が終わるまで待機する。

主担当の独立後検算器 `target/parallel-continuation-20260928/review_prior_actual_metrics.py` も用意した。登録checkerをimportせず、全保存mean scanからERB分布L1・mass相対差・score/level差を再計算し、child status・raw/log hashを照合する。担当の読取レビューでもschemaと丸め順にblocking所見はなかった。まだ取得終端前なので未実行である。独立peak/frontend再演の代用とはしない。


### 2026-09-28 新Job-probe版の30秒取得器を適合

主担当が新time4-job-probeの `target/time4-paced-v1` を分担し、旧pack向けに固定されていたPCM・ABBA参照を、新版の明示入力へ変更した。新PCMのinner/outer plan、比較ファイル、root検算とresultを相互照合し、ABBA planの新binary/source・runner・同PCM参照とも一致させる。実行前後に固定ABBA検証器の全資産照合を呼ぶ。endpoint capture環境を除去し、作業directoryを新版へ明示する。30秒I-W4-on-a、3表各690、3出生・翌hop receipt、underflow/hop超過0の基準は変えない。

Python反例12件が通過し、旧PCM証拠の取り違え6変種も拒否した。旧pack取得器との比較では `run_logged`、`table_counts`、`audit_single_resource` の本文がbyte一致した（`time4-job-paced-adaptation-review-v1.json`）。Sol担当の読取レビューでもblocking所見はなかった。後段の科学oracleは旧直接計算関数を維持し、速度比較だけが実ActiveJobを通る。これは取得器準備であり、正式plan・新PCM・ABBA・30秒取得は未実施である。

長時間factorialは出力がしばらく増えなくても同sessionがliveだった。主担当のホスト読取でも新treeはpoint_body-a rendererがCPU327%、旧treeはbody_point-bがCPU310%で稼働し、条件進行を確認した。再起動していない。旧側suiteの実argvはtest-threads=2、新側は4であり、構築・機能検査の条件として区別する。


### 2026-09-28 全prior_actual 2,030点の数値精度未達

単回runner session10296は21:34:01 JSTにexit1で終端した。全2,030子はexit0、取得後照合は成功、登録checkerの整合失敗は0だった。rawは4,652,969,757 byte。exit1は四精度条件の未達によるものであり、実行エラーとは分ける。score適格2,010点の同時合格は1,810点、不合格200点。環境欠測20点の分布L1・ERB massは20/20合格だが、score/levelの合格には数えない。

幅96は819/1,005、幅1536は991/1,005が四条件合格。最細幅でもD-4:s2に13件、D-4:s3に1件の不合格が残った。指標別の失敗数はL1 199、mass 53、score 104、level 104であり、重複を含む。主担当は全子status・raw/log hashを照合し、保存mean scanから8,080指標を独立再計算して全値一致を確認した（`prepeak-prior-actual-independent-quality-v1.json`、session95938 exit0）。閾値変更・失敗除外・再取得は行わず、通常runtimeへの補間接続は進めない。層化照会、複数環境、7,504step energy再演、冷cache費用と高有効率は別の未達工程として残す。

固定監査後、主担当は最細幅の全14不合格を機械的に抽出して保存power/peak列を比較した。各照会でpeak bin列が変わるframeはちょうど1件。13件のD-4:s2ではpeakが1個増減し、D-4:s3の1件ではbin534/533が切り替わった（`prepeak-prior-actual-finest-failures-posthoc-v1.json`）。これは失敗集合全件の事後記述であり、独立frontend再演や因果同定、改善案の検証ではない。

全prior_actualの結果文書と最終indexを封印した。index SHA `1ce564be6caac21546ecefe36ee0e04841b9507a12fba0c7e7480f3333852e0e`、11,326現物・5,170,285,242 byte。主担当も全entryのsize/hashを照合して一致した（`prepeak-prior-actual-final-index-review-v1.json`）。失敗200件を含む原本、入力、source/build、独立検算を保持し、以後変更しない。

さらに登録取得済み全2,030点に対し、各72 frameのlow/high/interpolatedでsorted peak-bin集合が同一、という事後条件を点検した（`prepeak-prior-actual-endpoint-topology-posthoc-v1.json`）。幅96では適格1,005点の一致0。幅1536では一致353点のうち352点が元精度条件を通るが、D-4:s2:D-4-on-b:e1728:d1536は四条件すべて不合格のままだった。不一致側には成功639点と不合格13点がある。この簡単な集合一致条件は誤差保証ではなく、通常有効率の証拠にもならない。新しい採用gateとして登録しておらず、再取得・source変更は行わない。残る保存反例のframe25と実frontend分岐を読取診断する。

### 2026-09-28 新旧Job-probeの全suite完了

time4側session72097は21:39:28 JST exit0、43群1,405合格・0失敗・55無視。report 3,742,224 byte、SHA `a03adb231999ef7002ac352e8b5b3d8a5d7ec81ae9dae9e6e6d1c3e759db7e1b`。主担当はcanonical report/statusの別inode・同byte、source live/capsule517件とfixture33件のhash一致を確認した（`time4-job-probe-suite-review-v1.json`）。旧pack側session45079は21:40:18 JST exit0、43群1,402合格・0失敗・55無視。report 3,741,539 byte、SHA `59677f874e91cbdb2b9b6c7e7f834ce90f4c77062551f42ceb38b835ebac3a75`。こちらもcanonical2件の別inode・同byte、suite前source517件と現物の一致、fixture33件を主担当が確認した（`pack-job-probe-suite-review-v1.json`）。両ignored690 probe自体は未実行。新旧2jobsのreleaseへ進み、同じ新版binaryで新720PCM→実Job690 ABBA→30秒pacedの順序を維持する。


### 2026-09-28 実ActiveJobのABBAと30秒3出生の通過

新旧Job-probeのreleaseを2jobsで構築した。新binary SHA `ab38994ed58e1b86e298ddd493dce93425e8adc03ec4fd519980b0170304c865`、旧binary SHA `61ad9aff8c7e3bda38948d58c61e416ab8644f8b01e279b604e8160bb4635ca7`。source manifestは新 `500b45276d123f8df650986c2df4787506b505e382215a67f34231329b5314a6`、旧 `4e95769ff0a4ad53fcafb365bb7446d93fec4b2e31875cbbe21349109d4aac89`。主担当は各517 sourceと33 fixture、build前後298 Rust/Cargo、compiler artifactと固定copy、raw build記録まで照合した（`time4-job-probe-release-review-v1.json`、`pack-job-probe-release-review-v1.json`）。新版の新720PCM取得はreplay/checker exit0、10 query×72 frame計18,979,351 byteが旧pack/cooperative双方とbyte一致した。これは通常saved-PCM経路の不変対照である。

正式ABBA plan SHA `bd99ff278f18a31f56fe8d381bea7bc3d4634e001ad9d3bc84d3b110498e648d` の54資産を主担当が照合し、別Solも実Job入口・planを読取確認した。ホストに他の重処理がないことを記録し、session80862で一度だけ旧A→新A→新B→旧Bを取得した。CPU秒は14.579045、7.472303、7.574491、14.411417。新旧比0.5125372067／0.5255896072で両対とも0.95以下を通過した。全4 oracle JSON・密度が固定baselineとbyte一致し、主担当も原status・hash・CPU比を再計算した（`time4-job-abba-independent-review-v1.json`）。実ActiveJobの全690表を含むprocess比較であり、Worker/channelを含む通常配送は次段で判定した。

paced生成器の初回呼出しはABBA resultの引数path誤りでplan作成前に失敗した。raw取得・source・gate変更なしで引数だけを訂正し、経緯を `generate-v1-observation.json`、成功log/statusを `generate-v2.*` に保持した。正式paced plan SHA `f2f1cd8f1c17eeba71e51abb7874f85cb4af0e511c1f4178839bf7c57e855d50` の26資産と旧登録の同一入力・条件を照合し、静穏ホストでsession19294により一度だけ取得した。runner、科学checker、旧直接計算経路の独立oracleは全exit0。3表各690を完備し、sample413696／787456／1170944で3出生、各+512 sampleで翌hop receiptを確認した。全2,813 hopで予算超過0、consumer underflow callback/sampleとも0だった。

主担当は保存された2,073 oracle行の全float密度、長さ・offset・hash・recipe・周波数をreport各表へ独立照合した。birthのrecipe/Hzと対応する局所表、翌hop receiptの時刻、pending/event drop 0も一致した（`review_time4_job_paced.py`、`time4-job-paced-independent-review-v1.json`）。whole-process CPU 38.949919秒、wall30.407843秒、1.280917 core、wait4 maxRSS146,436 KiB。旧版は準備完了2表、新版は3表なので、この値を同仕事量の新旧費用比較とはしない。OFF相対資源は未観測である。出生hop自己PCM3件は全zero/no_phonationであり、有音の実身体移送の正例ではない。初期出生seed7の模擬sink1条件の成功に限定し、親energy・他scene・実device・作者受入とF4全体は未完了とする。原結果を封印後、静穏ホストをI11 loop資源432条件へ引き渡す。

全prior_actualの追加読取では、残ったframe25反例でbin543の局所peak候補自体はlow/high/direct/interpolatedの全系列に存在した。5% mass fractionのpruning前余裕はdirectが+3.78266e-5、補間が−2.07871e-6で、directだけが保持された。保存されたdirectの最終mass0.0905929は再配分後の値であり、閾値直前massではない（`target/prepeak-prior-actual-frame25-diagnosis-v1/findings.md`）。保存sourceと分岐に沿う事後診断であり、全frontendの独立因果検証や新しい誤差保証ではない。封印済み200失敗の判定と通常runtime未接続を維持する。


新旧Job-probeの最終索引は `body-fitness-nsgt-time4-job-probe/target/time4-job-probe-validation/final-index-v1.json`、SHA `b3bdb8677316f9a3b9194be974dd445cebca89e7e100445f4d676ae55b4215b9`、2,375項・555,939,324 byte。主担当も全size/hash一致を確認した（`time4-job-probe-final-index-review-v1.json`）。初回の主担当検算は索引内相対pathをmain cwdから解決して停止したため、索引生成treeを基準に修正した。取得・索引には変更なし。重い証拠読取りを完了し、I11 loop資源432条件の担当へ静穏ホスト確認後の単回取得開始を指示した。後検算用の独立集計器は旧式を維持して新pathへ適合し、旧結果10/16という固定出力だけを実測件数からの集計へ替えた。取得前には実行しない。

I11 loop資源432条件はhost-visible PID namespace `pid:[4026531836]`、他のcargo/rustc/render/計測runner不在を確認し、owner session41111で単回取得を開始した。開始時刻は 2026-09-28T13:10:11.959586+00:00。途中42/432件で子の非ゼロ終了0。固定plan・順序を維持し、同sessionを追跡する。この途中記録は資源合否ではなく、終了後の固定監査と全profile独立集計を待つ。


### 2026-09-28 次の出生36条件の取得準備と動的代謝の範囲確認

次の草案と最小起動helperを `body-fitness-nsgt-time4-job-probe/target/time4-next-acceptance-v1/` に用意した。最初の草案は親time4 source/binaryを参照していたため、実測成功版Job-probeのsource `500b4527…` / binary `ab38994e…` へ訂正した。旧草案は未実行の誤参照履歴としてpointerだけを残す。新草案の9scene×ON/OFF×a/b=36、ON18 oracle、科学36、OFF相対資源18対、R系の親energy連結は旧登録を維持する。起動helperはendpoint診断環境を除去して現treeのcwdを明示し、既存strict inner plan取得器は変更しない。別Solの独立読取でも数値・順序・判定条件の変更はなかった（`independent-review-draft-v1.md`）。正式outer planの資産照合と固定、取得はまだ実施していない。I11測定中は軽量準備だけを進める。

動的Hz代謝のsource読取を別文書 `body-fitness-nsgt-time4-job-probe/target/time4-metabolism-feasibility-v1/read-review.md` に記録した。time4は出生ActiveJobだけに接続され、代謝workerは従来72-frame関数のままである。代謝は完了収集後に現VoiceのRecipeを再照合し、Hz identityが変わればReadyを失効させる。実評価も完全Recipe identityとHz bitsを確認する。前hopの計算が完了しても現hopHzの変化で使えない場合があるため、出生計算の高速化を動的代謝の高有効率へ転用しない。これはsourceから確認した条件付き制約であり、新しい取得結果や改善実装ではない。

36条件の外側plan生成器と起動・後検査helperの準備を完了した。実測Job-probeのsource/binary、build・fixture・全suite、PCM・実Job ABBA・pacedのplan/result/root照合を束ね、strict内側planは変更しない。軽量反例4件は成功。主担当も実schema、source/binary連結、出力先分離、起動前後照合、endpoint診断env除去、失敗保持を読取確認した（`time4-birth36-preparation-source-review-v1.json`）。正式planの生成・全資産固定と36取得はI11完了後に残す。

出生36条件の正式固定は `.worktrees/body-fitness-nsgt-time4-job-probe/target/time4-next-acceptance-v1/freeze_36.py --freeze` に引数を集約した。20資産、manifest＋27入力、36case宣言、未使用のinner/postrun/outer出力3pathを静的確認し、主担当も実測版への参照と生成器の引数対応を読取確認した（`time4-birth36-freeze-call-review-v1.json`）。`--freeze` はまだ実行していない。I11 session41111の測定・監査が終わってから正式planを生成し、全assetの現物照合後に36件へ進む。


### 2026-09-28 ユーザー指示の再確認：不要な性能チューニングを止める

ユーザーから「無駄な性能チューニングはしないように指示した」と再確認があった。出生の時間方向4-frame化は、登録された通常runtimeで全候補準備が間に合わず出生が成立しない具体的な障害への修正だった。一方、I11ではown処理予算超過0の既測定結果がある中で、対称の細かな比較差を満たすための性能探索を広げすぎた。新しい高速化候補、反復的な性能再取得、局所速度条件の追求を停止する。

進行中のI11 session41111は固定432件の取得と既登録の監査・集計までで保存し、不合格でも次のチューニングへ自動進行しない。旧版の判定・閾値・原結果は書き換えない。完了後は、残る条件が機能・数値・実時間動作に必要かをユーザーの指示に照らして分類し、微小な性能差を理由に全体作業を延長しない。出生36条件の準備や動的Hzの精度未達についても、機能・精度の障害と速度向上そのものを分ける。性能変更は具体的な動作不成立または実用上の予算超過を解消する範囲に限定する。


### 2026-09-28 原登録との照合による次工程の限定

[I11の残要件の読取レビュー](../../target/i11-requirement-realignment-20260928/read-review.md)を原登録と現roadmapへ照合した。I11-1の同record機能比較は現版で通過済みであり、旧末尾の未取得記述を再作業の根拠にしない。§5.7の14/16、I11-2の旧5/16と次版10/16は各版の判定として残す。Binding欠測110判断を全件成功へ変えることや、物理的sourceの完全帰属を新たな必須条件にはしない。進行中432件の終端後も、相対費用判定の不合格だけを理由に性能探索を再開しない。I10の限定技術完了を再開しない。

出生36条件は既登録の9場面について、通常配送・数値整合・親energyによる選択・実用上のCPU/RSS予算を検証する。単一の初期出生成功では未検証の範囲であり、局所的な速度差を追う追加試験ではない。実測済みtime4 Job-probe版と旧入力を固定して一度取得する準備を継続し、I11の専有測定・監査終了後に正式planを固定する。 [正式固定前の独立読取レビュー](../../.worktrees/body-fitness-nsgt-time4-job-probe/target/time4-next-acceptance-v1/review-final-v1.md)では、元登録の条件・順序、実測版への参照、inner/outer連結、子環境と失敗保持に阻害所見はなかった。正式planの現物照合は固定後に行う。

動的Hzの[選定点メタデータ集計](../../target/prepeak-next-feasibility-v1/findings.md)は、1,015点が全75,040機会の疎な選定であり、全時系列の更新頻度やcache再利用率を示さないことを確認した。四node集合と容量は算術上の仮定にとどまり、精度改善の証拠はない。四node実装や格子探索へ自動進行せず、現Hz一致・Ready失効の契約を保ったまま機能障害を解く次手をsourceから確認する。

動的Hzの[source読取](../../target/dynamic-hz-functional-next-20260928/read-review.md)では、managerがhop開始Recipeに対して全体で最大1 job/hopを送る一方、512-sample hop内の64-sample制御ではpitch更新後に各評価が行われる構造を確認した。主担当も最新資源版のmanager・Voice・評価側で同じ順序と実Recipe/Hz bits照合を確認した。8評価のHzがすべて異なれば、同じ一つのReadyで全8評価を覆えない。したがってworkerの単純な高速化だけをD-M高有効率の解とはしない。未来Recipeを先行計算する案には、評価前に実Recipeが確定する証拠が必要だが、これは全解法の必要条件ではない。現Hzから登録誤差内の密度を得る別計算法も論理上残るものの、現在の補間は精度未達。旧Ready流用、Hz固定、閾値緩和、新しい性能探索は開始しない。

[動的Hzの要件境界](../../target/dynamic-hz-functional-next-20260928/requirements.md)を初回登録と主計画へ照合した。可変pitchの初回登録には有効率下限がなく、実Hz変化、誤消費0、全分母・理由の保存が条件である。最新資源版の28/28条件・14/14対の合格と、D-M 5/7,504の低率を両立する記録として保持する。固定pitchの95%／80%を動的条件へ移したり、全7,504機会の消費を後付けしたりしない。一方、低率の取得を継続的な動的生態比較の成功とは呼ばない。必要率と欠測時処理の事前登録は、その比較を行う段階の条件として残す。現在の次実行は出生36条件であり、高有効率のための補間探索を並行で再開しない。


### 2026-09-28 物理16コア・SMT有効での並行継続

ユーザーから並列計算の影響を過度に考慮しなくてよいとの指示があり、I11終端まで出生取得を待つ方針を解除した。実機のlscpuはAMD Ryzen 9 9950X、1 socket、16 cores/socket、2 threads/core、論理CPU 0–31すべてonline。sysfsはsmt/active=1、control=onだった（[実機記録](../../target/parallel-continuation-20260928/host-cpu-topology-20260928.json)）。I11の既存取得は再開・再起動せず続け、出生runnerの実開始時刻を環境補足へ記録する。

出生36条件の最初の正式固定は取得前に停止した。元source manifestはfile_countを持つ一方、旧取得器はcountを要求し、KeyErrorとなった。freeze-v1.log/statusと空のacquisition-v1を保持する。実装・binary・元manifestは変更せず、同じfilesを持つ派生目録にcountだけを加え、内外planで元目録との一致を検証する修正を行う。新しい正式出力はv2とし、条件・数値閾値・入力順序は維持する。これは試験本体の失敗や性能結果ではない。

出生36条件の正式v2 planを固定し、主担当が28資産の現物hash/size、元目録と派生目録の全内容一致（count追加のみ）、binary、36順序とON18件、未使用出力を照合した（[固定照合](../../target/parallel-continuation-20260928/time4-birth36-fixed-plan-review-v2.json)）。outer SHA `712967d5ec0bb7b6412f17a655d147b6ca49e8038d4229cb98cbbd5f38b178ed`、inner SHA `acbe23281dfcda1cff4df17d33a7eb48db9656dba792bf287ad985247860ab05`。派生目録SHA `4b8d97c704080e37f2e7cdf9de21d56aac56b34a0e96121e6d0b1ea4091071ce`、元目録500b4527とbinary ab38994eは不変更。schema修正後、innerのdescriptorがpath/hashのみである点も合わせ、反例7件を通過した。

取得は2026-09-28T14:05:19.878174Z（23:05:19 JST）、主担当session32802で一度開始。新rootはJob-probe treeの `target/time4-next-acceptance-v1/acquisition-v2`。wrapper開始記録・log・終端statusを同親dirへ保存する。I11の並行境界はjournal index376の実行中であり、出生開始後の最初のI11 jobはindex377。最初のE-M-off-a取得はPASSで、同じrunnerを追跡する。これは個別の取得成否であり、科学・資源の全判定は後検査を待つ。

ユーザーから出生試験の意味と待ち時間について確認があった。30秒は観測窓であり、音声処理を停止する待ちではない。一方、広域単例の出生時刻は開始約8.6／16.4／24.4秒であり、即時出生の実用受入を証明しない。出生36条件は既登録の配送・数値・資源を判定し、実用上の許容遅延は別に明確化する。閾値や取得順序を事後変更しない。


### 2026-09-28 I11 loop版432件の終端と独立集計

owner session41111は23:21:00 JSTにexit0で終了し、全432子もexit0、postflightを保存した。固定auditは整合エラー0で、登録済み相対費用判定によりexit1。主担当は保存432 profileを独立再計算し、48 A/A組からのfloorと16比較の全数値・合否がauditに一致した（[集計](../../target/i11-energy-loop-resource-432-independent-review-20260928/independent-reduction.json)、recompute-v1.statusはexit0）。到来ON対Noneは13/16合格、全体不合格。機能回帰36＋8件の通過とは別の判定として保持する。

未達はharmonic-flow-16 no-reportのpopulation p99差+139.431 µs（許容122.971）、modal-flow-16 reportのpopulation p99差+124.271 µs（同122.971）、modal-flow-16 no-reportのpopulation中央値差+10.770 µs（許容6.06055）。最後の条件ではsynthesis p99がONで249.992 µs速い方向にも対称許容224.252 µsを超えた。own各指標は16/16合格、ON/Noneの同時計の予算超過は0。後半の出生取得との重なりを含む一回の記録であり、旧10/16との差を純粋なコード効果とは断定しない。ユーザー指示に従い追加候補・反復取得へ進まず、既登録の補遺と封印を完了する。

出生36取得はroot session32802が2026-09-28T14:23:43.427073Zにexit0で終了した。主担当が全36取得statusのplan/source/binary、入力一致、transport/sampler成功を照合し、全条件2813 hop・予算超過0・underflow0を確認した（[取得照合](../../target/parallel-continuation-20260928/time4-birth36-acquisition-review-v2.json)）。登録済みpostrunはSol担当session40857で一度開始し、ON18 oracle→science36→resource1の順を維持する。

別の[待ち時間集計](../../target/time4-birth-latency-scope-20260928/on18-v1.json)は全36の終端statusを確認してON18 reportを読み、予約26・出生26・未観測0・整合エラー0だった。主担当も全26件の予約frameから出生sampleまでの差とinitial/respawn別の集計算術を照合した。初期20件は2.058667–24.650667秒（中央値3.450667）、respawn 6件は2.421333–2.560000秒（中央値2.501333）。広域ON-aは8.053333／16.106667／24.650667秒、ON-bは7.850667／16.096000／24.458667秒。先の単ON封印取得とは別の測定である。すべて試験sample時計で、許容遅延の合否や実device体感値ではない。

ユーザーから「24秒もあって音楽は成立するの？」との指摘があった。既存Voiceの演奏は継続するが、要求した時刻から数秒～24秒後に新Voiceが加入するため、音楽上の加入・補充タイミングを保てない。rootは現非同期方式の通常採用を不可とし、進行中の固定後検査は診断結果の保存として完了する。出生待ちを短縮する局所速度探索へは戻らず、重い準備の前倒しと出生時刻の保証を設計上の次の論点とする。消費時の最新環境・占有・親energyによる選択を、古い状態の抽選へ変えて解決したとは扱わない。

### 2026-09-29 Opus 5.5レビューと再出発

[レビュー所見への対応](body-fitness-plan-review-response-20260929.md)を確定し、[Q1–Q6の静的確認と現行方針](body-fitness-runtime-restart-20260929.md)を記録した。一般的な子表在庫は未来の実ID・出生frame・身体入力を前確定できない条件を覆えず、通常実装へ進めない。身体入力まで固定できる予定Spawn等の前倒し準備のみ補助候補とし、次は実子の確定部分音・modeからその場で作る本番用身体分布の契約を一案として定義する。方式は未採用。予定・未予告Spawnとrespawnは、既存の出生機会hop内でのVoice生成を設計目標とし、初発音と翌hop観測を区別する。この条件の下で、対象負荷、出生処理のCPU時間とRSS上限、未準備時の状態遷移を試作前に定める。

既登録のtime4後検査は[postrun-v2](../../.worktrees/body-fitness-nsgt-time4-job-probe/target/time4-next-acceptance-v1/postrun-v2/postrun-results.json)でoracle 18、science 36、resources 1、失敗0、`pass=true`、SHA-256 `92c6ab7b27d2030bf9674ea64e6e19b7c97fcd065667383fb8dbe8bed34ed2c6`。今回の確認範囲は保存結果JSON・status・入力hashで、全DSPの独立再実行や同hop実用合格ではない。旧36条件の一括再取得、NSGT局所高速化、I11性能探索、音色遺伝には進まない。9月28日までの次手は各時点の履歴とし、[主計画の現行工程](body-aware-fitness-plan.md#現行工程2026-10-02)を優先する。
