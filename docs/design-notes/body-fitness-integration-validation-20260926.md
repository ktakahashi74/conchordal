# I4・I11-2・身体評価の隔離統合と検査順

日付: 2026-09-26。基底: `06a4772c43d06b41b44753bb0be891f23e16b93e`。
統合先: `.worktrees/integration-fitness`、branch `work/integration-fitness-20260926`。
mainのsrc、既定設定、現在のA4基準版は維持する。

## 統合するもの

| 入力 | 状態と保存した基準 | 統合で確認する境界 |
|---|---|---|
| `.worktrees/i4-reference` | 第一段階36条件回帰、第二段階の有界producer・実消費、memory=None 12条件回帰まで合格。stage2 source capsule保存済み | observationの所有、memory選択、参照inventory、clock・retentionの共通化 |
| `.worktrees/i11-stage2` | 3素材ON/OFF・旧None回帰とCDF独立参照を合格。CDF未生成理由追加後の全テストと9条件の旧音声・記録回帰も通過 | 同じobservationへ追加したCDF有無・arrival消費、自己群identity、既定OFF |
| `.worktrees/body-aware-fitness` | F1/F2〜F3dの数値・配送・実Voice一判断と全テストを通過。統合版の入力は封印したF3dまで。後続F3eは別の検査単位 | 解析設定、source PCMの所属、欠落・受信時計、既存runtimeの観測非干渉 |

最初にI4・I11-2のsourceと検査素材を、共通基底から三方向で合わせた。
`config.rs`、`temporal_cognition.rs`、`observation.rs`、`reference_inventory/tests.rs` の4共有ファイルは
機械的な衝突なく統合できた。これは動作の合格ではない。入力sourceのhashと統合箇所を
統合先 `target/integration-source-20260926/manifest.json` に保存した。
以後の未検査変更は再同期してから最終sourceを封印する。各入力の既取得capsuleは保持する。

21時台後半に三入力の現差分を再同期した。旧統合先に不明な追加変更がないことを確認し、
身体評価の17ファイルとI11の診断更新4ファイルを反映した。共有箇所は6ファイル、出力全体は47ファイル。
入力・出力のSHA、初回manifestのSHA、共有箇所の解決は
`target/integration-source-20260926-v2/manifest.json` に保存した。formatと差分検査は通過したが、
build・test・renderは未実施である。共有箇所を含め、検査結果はこの版へ改めて対応づける。
その後の第三版は`target/integration-source-20260926-v3/manifest.json`に三入力と出力49ファイルのSHAを固定した。
第三版のformatと差分検査は通過した。その後、sourceを維持したまま通常build、全cargo test
（1228成功・0失敗・40 ignore）、全targetのcheckも通過した。228 sourceのSHAとログは
統合先`target/integration-validation-20260926/validation.json`に保存した。release renderは同capsuleに封印した。
初回の標準Clippyは別worktree由来のpackage cacheを再利用してexit 0だったが、後述の新規3件を検査できていなかった。
package成果物を除去した同じ第三版で3件を再現したため、初回exit 0をlint合格の根拠から外した。

## 検査順と解釈

1. I11 §5.7の専有取得終了後、F3bの実thread・queue試験、density-onlyと旧解析の全bin bit比較、
   I11-2のCDF理由配送と数値不変を各隔離版で先に確認する。
2. 確認済み差分を統合先へ同期し、sourceの共通変更を再レビューする。
   通常buildとtest buildの双方を確認し、必須の全cargo test・format・lint結果を保存する。
3. F3cでは `wire_runtime` の実Voice・source PCM採取からworker受信までを通し、独立した他者-only
   解析と比較する。明示offline配送なので、実時間の鮮度や資源受入へは転用しない。
4. 統合版の既定OFFは登録した基準素材で旧版の音声・主要recordと照合する。I4有効とI11 arrival有効は
   両者の参照・CDFが実際に届いたことも数える。0件の無作用を機能保存として合格にしない。
5. density-onlyの費用は同じbinaryの旧／新経路を交互順で比較する。持続励振した4身体のProcessor費用と、
   通常runtimeへの接続後の全体費用は別の取得とする。元のI11 §5.7は旧基準版の評価であり、統合新版のR2ではない。

予定した検査が失敗すれば、その失敗を保存し、入力・許容・既定値を結果に合わせて緩めない。
候補身体の準備と移動への採用、代謝・出生への接続、作者による聴取・既定採用は、この統合だけで完了とはしない。

## 統合runtime回帰の取得前登録

統合第三版の通常runtimeを、保存済みの隔離版成果と比較する。取得先は
`.worktrees/integration-fitness/target/integration-runtime-regression-20260926/`。
`plan.json`に入力・設定・旧WAV／report・旧比較器・第三版manifest・封印済みrenderのSHAと全25コマンドを固定し、
取得前に保存する。取得器は開始時に全hashを再検査し、封印済みrelease
`conchordal-render`を複製してbinary SHAを追加記録する。中断後の同一出力先への自動再開はしない。

| 境界 | 件数 | 入力・設定 | 判定 |
|---|---:|---|---|
| I4 memory=None | 12 | `sine-hold`、`sine-flow`、`harmonic-flow`、`modal-flow`の各4／16／64 Voice。I4保存済み`config-memory-none.toml` | 旧06aの同条件WAV byte、10種の学習record、temporal observationを比較。memory／inventory／bounded_memoryの非出力、旧版と同じ共通候補の内容、片側件数を確認 |
| I4 bounded | 1 | `sine-flow-4`、I4保存済み`config-bounded.toml` | 旧I4 smokeのWAV・学習recordと共通候補を比較。bounded evidence、重み付きinventory、同じkeyのprivate trace、旧memory非出力を正数で確認 |
| I11 arrival ON／OFF／None | 9 | `sine-flow-4`、`harmonic-flow-4`、`modal-flow-4` × 3設定。I11診断v2の同じ入力 | 診断v2保存結果のWAV byte、10種の学習record、成功CDF、temporal observationとparticipation decisionを比較。共通候補の全値を比較し、片側件数は別報告。OFF／NoneにCDF理由が出ないこと、ONの既知消費と局所作用判定も照合 |
| 両ON | 3 | I11の3素材、ON設定の`temporal_memory`表だけをI4 bounded設定の表へ置換 | 新条件。WAV一致を要件にしない。各素材にbounded evidence・inventoryと成功CDFが正数、Harmonic／Modalで到来knownの消費が正数、旧memory非出力を確認。Sineは先行I11でeligible knownが0の対照として残す |

合計25本。両ONの設定は、I11 ONに`model = "bounded_reference"`を1行追加したもの。
TOMLとして読んだ`temporal_memory`表（retentionを含む）がI4 bounded設定と等しく、
そのほかの表はI11 ONと等しいことを取得前に機械確認する。
先行I11診断v2のONでは成功CDFがSine 120、Harmonic 134、Modal 74件。
一方、Sineの参加判断60件でeligible knownは0、HarmonicとModalでは各1件。
成功CDFの生成と参加判断への既知到来の実消費を別に数える。

学習recordは先行§5.4aの10種、候補は既存の6フィールドkeyで比較する。候補workerの非同期飽和による片側recordを
内容差として隠さず件数を保存する。clock診断の除外はI4保存比較器の`_us`／`_ns`接尾辞と登録済み明示キー、
I11診断v2比較器の既存集合に限定する。数値、ID、支持時刻、選択、CDF、理由、到来費用は除外しない。
統合版と診断v2版の間では追加診断の差を新たに許さない。判定失敗時は生データを保持し、設定・閾値・除外を事後変更しない。

この取得の通常runtimeは身体評価F3の`cfg(test)`経路を呼ばない。確認できるのは旧pitch経路と、I4／I11配線の
非干渉・実消費までである。F3の通常runtime採用、callback／queue費用、R2、音楽的評価を判定しない。

## 第三版の25条件結果とModalの不達

25本のrenderはすべてexit 0。I4 memory=Noneの12条件、bounded単独1条件、I11の9条件は、
旧WAV・主要記録・共通候補値・対象となるCDF／観測の比較を通過した。
両ONのSineとHarmonicも登録した実消費条件を満たした。
Modalは成功CDF74件と到来known 2件がある一方、bounded evidenceとinventoryが0件だった。
したがって比較は24/25条件通過、全体は不合格とする。条件を事後変更して合格へ読み替えない。

Modalのbounded observationは124件、query最大350、received最大334、退役を理由にした拒否16件で、
stored episodeは0件だった。報告されたretained groupのIDは390種類、観測された最長区間は81 hopであり、
96 hopのBuilderが完成する前に群が交代する説明と整合する。ただし報告は間引かれているため、
全hopの寿命や破棄理由を確定した結果ではない。後続は閾値を変えず、全hop診断でこの境界を確認する。

取得前本文はcapsuleの`registration-precollection.md`、初回manifestは`manifest-before-diagnosis.json`へ
当時のSHA完全一致で保存した。追加診断とmanifest更新の経緯は`diagnosis-supplement-manifest.json`に分けた。
生データ、元のplan、失敗判定は保持している。

## 第四版: 新規lint3件の修正と影響範囲の再検査

`bounded_match.rs`のu64偶奇判定3箇所だけを、同じ意味の`is_multiple_of(2)`へ変更した。
モデル、96 hop、coverage、距離、既定設定は変えていない。第三版の49ファイルをSHA照合してから
`target/integration-source-20260926-v4/predecessor-v3-overlay.tar`へ保存した。
初回lintの訂正は同じ場所の`v3-lint-correction.json`に残した。

第四版の全cargo testは23:12:06 JSTにexit 0（1228成功・0失敗・40 ignore）。
format、標準Clippy、全targetのcheck、release buildも通過した。
all-targets Clippyには既存テスト箇所17件が残るが、新規3件は解消した。
source・binary・全ログは`target/integration-validation-20260926-v4/validation.json`に対応づけた。

影響するI4 bounded単独1条件と両ON3条件だけを新しく登録して再取得し、第三版と比較した。
4条件ともWAV・既存記録・共通候補・成功CDF・実消費集計が一致した。
`target/integration-lint-v4-regression-20260926/`に登録・全コマンド・生データ・比較を保存した。
これは構文修正の非干渉を示す。Modalの参照0と第三版の全体不合格は変わらない。

## 全hop診断: 固定保存窓に届かないModalの追跡群

別worktree `.worktrees/integration-memory-diagnostic` で、各hopの群の出生・退役と
Builderの経過を計測した。既存の3素材・両ON設定・96 hop窓・距離閾値を固定し、
診断を足した版と第四版で、WAV、主要記録、共通候補、CDF、非診断のtemporal記録が一致した。

各busのModalでは212群が生まれ、205群が退役し、EOF時に7群が残った。
退役群の最大寿命と未完成Builderの最大経過はともに90 hop、EOFのBuilderは最大51 hopだった。
96 hop満了は0、保存時の無観測・鮮度拒否も0で、そもそも保存関数へ到達していなかった。
Sineは6件、Harmonicは3件を保存した。二busとも件数の保存則を確認した。
これで「同一handleの追跡が保存窓より先に終わる」という直接の仕組みは確認できた。
群が退役する上流の理由や、その音色依存性まではこの集計で断定しない。

診断版の全cargo testは1228成功・0失敗・41 ignore、formatと標準Clippyも通過した。
登録、source、binary、生データ、旧版との比較は同worktreeの
`target/memory-diagnostic-20260926/manifest.json`に保存した。
この診断は第三版の不合格を解消する変更ではない。保存窓や群の扱いを改訂する場合は、
別のモデルと取得登録として検証する。

## 退役理由の第二段診断と統合第五版

全hop診断を退役イベントの理由へ広げたところ、各busのModalの205件、Harmonicの147件は
すべて追跡容量による退役だった。観測された非活動を理由とする退役と理由不明はともに0件だった。
`superseded` は別の機構であり、退役件数へ加算しない。登録、source、raw、比較は
`.worktrees/integration-memory-diagnostic/target/memory-diagnostic-phase2-20260926/` に保存した。
この入力では、限られた追跡枠の交代が96 hopの保存窓へ到達できない直接の理由となる。
他の入力や音色一般に同じ因果を外挿しない。

第五版は第四版に、F3e第一段階の実Voice照合、F4aの成人代謝と発声費用の限定接続、
記憶診断第二段階を取り込んだ。F3e第二段階、F4bの出生、短い保存区間の実験は含めない。
共通baseと封印sourceを三方比較し、I11のarrival対応発声入口へF4aの事前照合を配置した。
照合は発声エンジンの更新と出力buffer消去より前に行う。二つの呼出入口が同じ照合を通る。

全cargo testは2026-09-26 23:55:01 JSTにexit 0、1232成功・0失敗・42 ignoreだった。
format、標準Clippy、全target check、release buildも通過した。
all-targets Clippyは第四版と同じ既存17件が残り、新規エラーはなかった。
232件のRust source／Cargoファイルと57件の統合出力を、検査前後にSHAで照合した。
`target/integration-source-20260926-v5/manifest.json` のSHAは
`679df6b352f2823e963e39b57653f740ad44a34a2d6b05674e074f14cd477e4d`。
結果とbinaryは `target/integration-validation-20260926-v5/validation.json` に対応づけた。

通常releaseの影響4条件を、第四版と同じ入力・比較規則で新しく取得した。
4条件ともWAV byte、主要記録、非診断temporal、共通候補、成功CDFと実消費集計が一致した。
取得前planのSHAは `149632c827b3f9e827e790d521235785a8d721a15b3d464961cce38434dc784c`。
全記録は `target/integration-runtime-v5-regression-20260926/` に保存した。
身体評価の接続と診断は引き続き試験専用であり、通常runtimeの有効評価率を示す結果ではない。
Modal参照0と、元の25条件の全体不合格も維持する。

## 第六版: Random出生、配送第二段階、部分区間保存の統合

第五版に、F3e第二段階v2、F4bのRandom出生一機会、退役時部分区間保存の試験専用実装を加えた。
封印source同士の差分を共通版から取り込み、三方mergeに競合はなかった。
F3e第三段階とF4cのHereditary接続は、この版には含めない。
F4bのrecipe一致は、実子から取得したbody・基音・代表modulatorと、宣言した固定励振条件の一致である。
実子のToneSpec全要素を独立に取得した一致とは扱わない。[独立レビュー](body-aware-fitness-f4b-source-review-20260927.md)に境界を記録した。

全cargo testは2026-09-27 00:20:25 JSTにexit 0、1240成功・0失敗・43 ignore。
format、標準Clippy、全target check、release buildも通過した。all-targets Clippyの既存17件は
messageと対象ファイルが前版と一致し、新規エラーはなかった。
234件のRust source／Cargoと62件の統合出力を固定したsource manifestは
`target/integration-source-20260927-v6/manifest.json`、SHAは
`a4b8b6b010932637b79e1b0f7a6b6414df56e4b62e8078e4947fd40ce55b63e6`。
全ログとrenderは `target/integration-validation-20260927-v6/validation.json` に対応づけた。

登録した通常releaseの4条件は第五版とWAV byte、主要記録、非診断temporal、共通候補、CDF、
実消費集計が一致した。取得前planのSHAは
`b761b8b86311a1abbaf2b672745d0c4d3c76ab27734e7d9a2fd6f67f34be3fea`。
生データと比較は `target/integration-runtime-v6-regression-20260927/` に保存した。
部分区間保存は通常releaseでは無効であり、この回帰は新モデルの採用試験ではない。
部分区間保存の別取得における厳格OFF比較の不合格と、元の25条件の不合格を維持する。

## 第七版: 配送の連続gate診断を統合

第六版へF3e第三段階の封印済み配送試験を取り込んだ。二つのsource tarの全memberを比較し、
差分は試験専用の `src/runtime/body_fitness_f3e_delivery_tests.rs` 一つだけだった。
F4cのHereditary試験は未統合である。

全cargo testは1242成功・0失敗・43 ignore。format、標準Clippy、全target checkとrelease buildも通過した。
all-targets Clippyの既存17件は同じで、新規エラーはない。
source manifestは `target/integration-source-20260927-v7/manifest.json`、SHAは
`c1ce8d6c54dcbe3cea660976d4606432c796b2e8297f891d636e8d219f1ddf99`。
全ログ・終了コード・source対応は `target/integration-validation-20260927-v7/validation.json` に保存した。

通常releaseのrenderは第六版とbyte一致し、SHAは
`1a112159df211bf0088823c1b3fe706c7e8dcbd61edde7b3ece56fb0f52e8868` だった。
取得前登録に従い、同じbinaryによる第六版の通常4条件比較を対応づけ、新たな4本取得は行わなかった。
全cargo testに含まれる既存render integration testsは第七版でも実行した。
通常runtimeの身体評価採用、実時間有効率、作者採用の証拠へは拡張しない。

## 第八版: 非ゼロ代謝からHereditary出生一機会への接続

第七版の234 source/Cargoファイルを保存済みhashと照合してから、[F4c v2](body-aware-fitness-f4c-v2-results-20260927.md)を取り込んだ。変更は `src/life/community/respawn.rs` の試験用接続、同階層の `respawn/f4c_offline.rs`、`src/runtime/body_fitness_f4a_tests.rs` の試験ヘルパー可視性の三箇所である。外部APIは拡張していない。

最初の全suiteはコピー元の更新時刻を保持したことで古いtest binaryを再利用し、F4cが検査対象に入っていなかった。この実行は第八版の証拠から除外した。更新時刻を修正した再ビルドで試験ヘルパー `tone_batch` の可視性不足を検出し、隔離元と同じ `pub(crate)` へ修正した。両ログを保存し、登録した数値条件は変更していない。

最終全cargo testは2026-09-27 11:18:57 JSTにexit 0、1243成功・0失敗・43 ignore。F4cの実行成功行を照合し、結果JSONも隔離元とbyte一致した。fmt、標準Clippy、全target check、release buildを通過した。all-targets Clippyは今回再実行していない。

235件のRust source／Cargoを検査前後で照合し、`.worktrees/integration-fitness/target/integration-source-20260927-v8/` に保存した。source manifestのSHA256は `26819adc96ac6b0b26115c68a1cc1cc25775364b87e2edc59bbdeb19a4fbc32d`。ログ・終了値・source対応は同worktreeの `target/integration-validation-20260927-v8/validation.json` に保存した。

通常release renderのSHA256は `316f431b9db1a646e9a949f37ae44ea83a1b99dc7a82f49c8f77d9295eb4a2e8` で、前版とはbyte不一致だった。取得前手順に従い、I4 bounded Sineと両ONのSine/Harmonic/Modalの固定4条件を再取得した。第六版に対し、全条件のWAV byte、主要記録、非診断temporal、共通候補、成功CDF、実消費集計が一致した。取得planのSHA256は `5fe649edf21742c991a1a201ad3d81d1d391679356ea7384d1f91f5706ead373`。生データと比較は同worktreeの `target/integration-runtime-v8-regression-20260927/` に保存した。

F4cの作用は試験専用の固定身体・一機会に限定する。通常runtimeへの採用、音色遺伝、実時間有効率の受入は未実施であり、元のModal参照0、I11資源2条件、部分記憶の厳格OFF比較の未達も維持する。

## 第九版: 提案延期、settle出生、明示ONの通常観測

[取得前登録](body-fitness-integration-v9-registration-20260927.md)に基づき、第八版の235 source/Cargoファイルをhash照合してから、F3e第四段階、F4d、通常runtime観測を取り込んだ。共有workerの差分はnonblocking受信のcrate内公開で一致し、観測側の診断を保持した。コピーは更新時刻を引き継がず、新sourceで再ビルドした。独立レビューでは延期時間の二重計上修正と消費前照合を確認した。

全cargo testは2026-09-27 11:54:31 JSTにexit 0、1251成功・0失敗・44 ignore。新規の延期・出生・通常観測・途中出生退役の成功行を照合した。fmt、標準Clippy、全target check、release buildも通過した。44件のignoreには別取得のlive-paced配送を含む。all-targets Clippyは再実行していない。

src・tests・Cargoの397ファイルを取得前後および独立レビューで照合し、`target/integration-source-20260927-v9/` に封印した。manifestのSHA256は `6ced43228964f9104f705e0305c33ad2c2900433e4d056ef20338a9711a89e98`。全検査と由来は `target/integration-validation-20260927-v9/validation.json` に対応づける。

通常release renderのSHA256は `9d1362885d72bed36a563971f3f3429052c291cd45a554ced137b7e9bf2e6a91`。第八版とは異なるため、登録済みのI4 bounded Sineと両ONのSine/Harmonic/Modalを再取得した。全4条件で第六版とのWAV byte、主要記録、非診断temporal、共通候補、CDF、実消費集計が一致した。取得planのSHA256は `1a65bb1c41edac566de15b51e4cfff130c277e8e2481362995994cf264e4d0c6`。生データと比較は `target/integration-runtime-v9-regression-20260927/` に保存した。

通常ビルドに追加したのは、既定OFFの自己除去観測である。通常renderで1/4 sourceのON/OFF音声・行動記録一致、5 sourceの理由付き拒否、途中出生と退役のidentityを確認した。観測値を身体評価の行動入力へは渡していない。候補提案の延期、身体表の消費、代謝・出生の身体評価は引き続き試験専用である。

live-paced配送は元の隔離worktreeで、他のcargo/renderを止めた専有枠で別取得した。1/4 source各938 hop、固定Sine 440 Hzで各source5回の実消費、deadline miss 0、F3b失効0だった。各source1件の完了は測定窓外で、窓内消費へ加算していない。通常runtime全体の有効率・実device性能とは分ける。元のModal参照0、I11資源2条件、部分記憶の厳格OFF比較の未達も維持する。

## 第十版: 候補密度cacheとglide再利用、PeakBiased全bin評価

[取得前登録](body-fitness-integration-v10-registration-20260927.md)に基づき、第九版の397 source/tests/Cargoファイルから各差分の由来を確認した。F3e5のcacheは密度だけを保持し、現在環境でのscore化は毎gate行う。glideの照合変更はlocal/non-ratio候補に限定し、候補全集合とcandidate Recipe Identityの照合を維持する。F4eでは身体評価前の点地形による候補切り捨てを行わず、range内全binから身体scoreでpeakを抽出する。変更は試験専用である。

統合準備の初回は相対cwd誤指定でsource保存が失敗したため、そのまま始まったsuiteを中断し、exit 130とログを保存した。正しいsourceを統合した全suiteは1258成功。その後、追加all-targets lintで見つかったcache試験の余分な参照2箇所だけを除去し、最終sourceでも全suiteを再実行した。最終終了は2026-09-27 12:21:53 JST、exit 0、1258成功・0失敗・45 ignore。新規7試験の実行成功行を照合した。fmt、標準Clippy、全target check、release buildも通過した。all-targets Clippyの既存警告群は修正対象としていない。

最終397ファイルのsource manifestは `target/integration-source-20260927-v10/manifest.json`、SHA256は `f371d9ae4d2dc0e97b119553f7cb0067592bf18dfd247478012c0ab8970ef0a8`。参照修正前のsourceも別保存した。結果対応は `target/integration-validation-20260927-v10/validation.json` に集約する。

通常release renderは第九版と異なり、SHA256 `f725c7d37ce6f2268c20e37f098bd4ac16c8278e6397e3a053625214414a7d3c`。固定4条件を第六版と再比較し、全条件でWAV byteと登録対象記録・集計が一致した。取得plan SHA256は `e298b6940a4512b31ca1d5ed5dbb8b0a23fe56c6cd11d4fbb5b061ff39e65820`、保存先は `target/integration-runtime-v10-regression-20260927/`。試験の参照2箇所修正後にreleaseを再ビルドし、同じbinaryであることをbyte照合したため、同じ4条件を再取得していない。

別worktreeのcache専有live取得では、各938 hopで1 sourceの実消費196回、4 sourceの実消費194/193/195/194回、deadline missとF3b失効は全条件0だった。窓内完了と実際のgate消費、最終join後回収は分けて記録する。これは固定身体の試験binaryの結果で、統合第十版の通常runtimeが身体評価で行動した証拠ではない。通常ビルドの行動入力、初回spawn、parentありPeakBiased、音色遺伝、作者採用、既存三種の未達は引き続き未完である。


## 第十一版: 通常runtimeへの明示ON行動接続（2026-09-27）

作業sourceは `.worktrees/body-fitness-action`。第十版の397ファイルに新規3ファイルを加えた400ファイルを固定した。旧 `.worktrees/integration-fitness` の第十版は保持している。[取得前登録](body-fitness-runtime-action-registration-20260927.md)と[結果](body-fitness-runtime-action-results-20260927.md)に配線・境界・失敗経過を記録した。

最初の全suiteはRecipe不一致5件で停止（1171成功・5失敗・45 ignore）。fixtureを実Voice由来へ修正し、最終全suiteは1263成功・0失敗・45 ignore、fmt/標準Clippy/全target check/release build成功。最終statusは `2026-09-27T12:47:27+09:00`。通常renderでは各sourceの準備済み判断消費と再現性を確認し、既定OFFのrelease固定4条件も第六版と一致した。短いSine fixtureでON/OFF WAVは同一であり、音響差や実時間性能は未確認。

source manifest SHA-256: `3a8bd1cfaa38bfefa34fe1428b1c34492eea1232613438bb0c7293369569cd7c`。通常release binary: `2386fb249d8e29d469b94793103932c04ea649cd9bdbe8f12a1e34e709d55210`。詳細の機械可読対応表は同worktreeの `target/runtime-action-validation/validation.json`。mainの通常sourceへの採用、commit、pushは行っていない。

## 2026-09-27 第十二版b: 非Sine移動と実時間条件

[第十二版の記録](body-fitness-ecology-validation-20260927.md)へ続く。隔離worktreeは `.worktrees/body-fitness-ecology`。1271成功・0失敗・46 ignore、format、標準Clippy、全target check、release test buildを通過した。Sine/Drone・10秒・1声/4声・ON/OFFの専有paced取得は全4条件成功。Seqの寿命による初回4条件不合格を別保存した。

最終source manifest SHA-256: `539c12283edd1e3963e84a79e0dd05573683459d42584a6b4e9b04ba4c9f9cb0`。対応表は同worktreeの `target/ecology-validation/validation.json`、SHA-256 `397e74b452f9a71406c96aa0f016f73298ced39d16d887d1deca153a36f05c18`。通常renderの非Sine移動と身体・制御変更は二回一致を確認。出生は初回Fieldの試験専用入口までで、通常runtimeへの代謝/出生接続、長期生態、作者採用は未完了。


## 第十三版b: 通常offline代謝と非Sine実時間未達（2026-09-27）

[第十三版b統合検証](body-fitness-metabolism-validation-20260927.md)へ詳細を保存した。隔離worktreeは `body-fitness-metabolism`。全suiteは1285成功・0失敗・47 ignore、format・標準Clippy・全target check・release test buildは合格。通常renderの代謝ON/OFFで実energy差と再現性、Field出生残余とparentありPeakBiasedの試験専用接続を検査した。

別の非Sine・身体/control変更の専有取得はOFF合格・ON不合格。2声・938 hopでunderflow/hop予算超過0だが、身体変更後の初消費frame 522が登録期限432を超えた。失敗rawとbinaryを保持し、旧世代pendingの取消と新世代密度の冷計算を次の改善対象とする。通常runtime出生、非同期代謝、長期生態と作者採用は未完了。mainのsourceへは採用していない。


## 第十四版: 取消と通常offline出生（2026-09-27）

固定source408ファイルの隔離版 `.worktrees/body-fitness-recovery` は1299成功・0失敗・48 ignore、format、標準Clippy、全target check、release lib test構築を通過。通常offline初回Field出生、取消、scratch再利用を統合した。[固定物と検証](body-fitness-recovery-validation-20260927.md)に全suite・source/binary・rawを集約する。

非Sine専有取得はOFF合格・ON不合格。取消1 hop、冷計算約1.91秒、新世代消費frame 480で期限432未達。underflowとhop予算超過は0。第十三版bの未達と今回の未達を保持する。通常出生の二回一致・独立数値照合を、実時間の出生/代謝接続や長期生態の完了へ拡張しない。mainの通常実装と既存の未達判定は変更していない。


## 第十五版b: 冷準備の並行化と登録期限内復帰（2026-09-27）

`.worktrees/body-fitness-cold-parallel` に第十四版の全408 sourceを検証して引き継ぎ、候補準備・action診断・新規局所試験の3ファイルだけを変更した。固定sourceは409ファイル、manifest SHA-256 `30d7d1b19289799a767bd02366d04556b5138779bcc5cedb1b37765d77cf9e43`。初回v15の通常lib import不備を保存したうえで、修正版v15bの1303成功・0失敗・48 ignore、標準Clippy、全target check、release test buildを確認した。

[検証記録](body-fitness-cold-parallel-validation-20260927.md)と[取得結果](body-fitness-cold-parallel-results-20260927.md)にsource・binary・rawを対応づけた。非Sine身体/control変更2条件とSine 1/4 sourceの4条件を固定binaryで各一回取得し、すべて通過。前二版で失敗したframe 432の新身体消費条件はこの版で成功した。定期reportに残らない冷jobの正確な時間は補わない。

通常offline出生＋代謝はまだ併用拒否。次の結合案を保存したが、今回の実装には含まない。第十五版bは保存単位であり、F0〜F6全体、実機・長時間・作者受入の完了ではない。mainのsrc、既定設定、A4基準版は維持した。


## 第十六版d: 初回Field出生と通常offline代謝の同時接続（2026-09-27）

`.worktrees/body-fitness-birth-metabolism` は第十五版bの固定409 sourceを基準とし、通常offlineの出生と代謝を限定sceneで併用する。最終sourceは410ファイル、manifest SHA-256 `01b8cc1c04aa9ffc2d79fc015f74852f7fb6d488215bff934a4394b8b4fddcaa`。全suiteは1311成功・0失敗・48 ignore、2026-09-27 16:01:11 JST終了、exit 0。fmt・標準Clippy・全target checkも通過した。

[検証記録](body-fitness-birth-metabolism-validation-20260927.md)と[取得結果](body-fitness-birth-metabolism-results-20260927.md)に、一次render4回・固定binary・全source・局所検査・失敗履歴を対応づけた。出生hopの子だけBirthShared、翌hopは2sourceともSourceRemovedとなり、子の出生sampleと世代を維持した。frame 96のenergyは出生のみ0.85603476、代謝併用0.85717773。WAVと出生位置はmode間同一だった。

初回compileと試験clock/型、正常なcrowding初期設定の誤拒否の修正を記録し、条件を後から調整して成功に変更していない。呼出し順のレビューにより既定OFFの毎hop heap追加を除去し、Finish後tailと短すぎる/複数Finishを区別した。第十五版bの専有実時間取得は旧版の証拠として保持し、新版の実時間性能へ読み替えない。

[親ありrespawnのdraft](body-fitness-offline-respawn-draft-20260927.md)を次の設計単位とする。長期生態、非同期代謝、音色遺伝、作者受入は未完了。mainのsrc・既定設定・A4基準版は維持した。


## 第十七版f: 通常offline親付きrespawn一機会（2026-09-27）

`.worktrees/body-fitness-respawn` に第十六版dの全410 sourceを照合して引き継ぎ、親の更新後energyから予定子の身体評価、実出生、翌hopの自己除去receiptへ接続した。最終source414ファイル、manifest SHA-256 `22791e93aac37af82c606877d4fb5f17f68079e8fcb223aa46192cc128a58ae9`。全suite1322成功・0失敗・48 ignore、fmt・標準Clippy・全target checkを通過した。

[検証記録](body-fitness-offline-respawn-validation-20260927.md)と[数値結果](body-fitness-offline-respawn-results-20260927.md)に、固定binary・source・通常render5回・拒否対照・局所fault・失敗履歴を対応づけた。登録seed7の最初の死亡はframe143/Voice3、親Voice1、子Voice4。ON/OFFの親energyと出生Hzが異なり、各modeのWAVと対象recordは二回一致した。子の出生hopのreceipt不在と512ゼロPCM、翌frame144のSourceRemovedとenergy更新を確認し、Finish frame150で終了した。

比較は代謝・出生の同時切替であり、効果の単独分離ではない。通常peak一覧を別実装で抽出した証拠とは扱わず、記録済候補からのRNG選択再計算と別環境fixtureのTone積分を区別する。第十五版bの計時を現版へ転用しない。次は[反復respawnの設計案](body-fitness-repeated-respawn-draft-20260927.md)。mainのsrc・既定設定・A4基準版は維持した。


## 第十八版b: 反復respawnと完全source identity（2026-09-27）

`.worktrees/body-fitness-repeated-respawn` は第十七版fの414 sourceを照合して引き継ぎ、5ファイルを変更した。最終source414ファイル、manifest SHA-256 `c235a2f40b25799bff59e099b8ff8f6395b3dde68a11ef43904fa548091c89b6`。全suiteは1327成功・0失敗・48 ignore、fmt・標準Clippy・全target checkを通過した。

[検証記録](body-fitness-repeated-respawn-validation-20260927.md)と[取得結果](body-fitness-repeated-respawn-results-20260927.md)に固定物を集約した。通常offline二機会で第一子4が第二親となり、第二子5は系譜generation2を持った。予定子・実子、各機会の全source、出生sample、出生hopの無音自己PCM、翌hopのSourceRemoved receiptとenergy更新を照合した。固定templateからの子生成であり、音色遺伝ではない。

初回v18のintegration失敗はcleanup後reportのbirth_sample欠落による。v18bでreportとruntimeの完全identity照合を追加し、登録sceneと期待値を維持した。旧一機会・高閾値拒否も最終suiteで再取得した。peak抽出は別言語実装で照合し、記録scoreの独立音響測定とは区別する。

同hopの前児確認と新機会の重なりはcleanup後に拒否する制限が残り、全登録faultの独立注入完了ではない。次は[固定異種身体と全消費者比較](body-fitness-all-consumers-draft-20260927.md)。現版の実時間性能・作者受入、F4/F5/F6全体、main採用は未完了。
