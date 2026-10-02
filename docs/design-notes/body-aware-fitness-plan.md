# 自己スペクトルを用いる適応度評価の実装計画

作成: 2026-09-26。最終更新: 2026-10-02。状態: 本番方式の再設計中（旧取得は保存済み）。

実行上の優先事項（2026-09-28ユーザー再確認）：不要な性能チューニングは行わない。性能変更は具体的な動作不成立・実用上の予算超過の解消に限定し、微小な速度差や局所screenの目標値を追う追加最適化は止める。取得済みの判定は保持し、既存比較gateが本来の機能要件を超える場合はその違いを明示する。

9月28日取得に適用した並行方針（履歴）：ユーザーの指示によりI11と出生36条件を並行して進めた。実機はRyzen 9 9950X、物理16コア・論理32 CPU、SMT有効と確認した。当時の専有待ち指示より、この追加指示を優先した。測定中の重なりは記録するが、それだけを理由に再測定しない。

出生の実用判断：予約から最大約24.65秒、死亡後補充で約2.5秒を待たせる現非同期方式は、演奏中の予定時刻にVoiceを加える用途へ採用しない。既存試験は数値・配送診断として完了・保存するが、実用合格とはしない。出生要求後の全候補準備を前提にしない。9月28日時点の前倒し準備案は、9月29日の判断により、身体入力を前確定できる予定Spawn等の補助手段に限定する。最新環境・親energyによる選択を過去へ移す修正は行わない。

出生待ち時間の境界：30秒は取得窓であり、許容出生遅延ではない。[単ONの時計照合](../../target/time4-birth-latency-scope-20260928/read-review.md)では予約約1.003秒から実出生まで7.616／15.403／23.392秒を要した。音声処理は継続するが、予約子は出生まで発音・集団参加しない。配送・数値検査の通過を即時出生の実用受入へ読み替えず、初期spawnとrespawnの許容遅延を別に明確化する。

### 現行工程（2026-10-02）

現在のF2は、旧失敗を保存したまま、描画と元の解析モデルに忠実な隔離参照を検証する工程である。[Source5の通常検証記録](../../target/f2-host-safe-resource-revision-20261001/source5-neutral-readback-preparation-v1/actual-neutral-exit-report-v1.json)は1,448成功・0失敗・67 ignore、必須4工程の終了0を記録する。これは `cfg(test)` の参照実装の検証であり、通常runtimeの身体分布producerや同hop出生を実装・受け入れた証拠ではない。

Source5の728候補×72 frameの精度取得は再起動で中断した。[中断記録](../../target/post-reboot-continuation-20261001/source5-reboot-interruption-v1.json)ではterminalと閉じたmanifestがなく、数値判定は未確認のままである。旧出力を保存し、自動再取得や未完出力のSavedReader評価へ進めない。新しい同条件取得には既存の具体案に対する作者判断が必要で、独立SavedReaderと元の精度判定はその後の別工程に残る。

最大容量側は、実2048 public bin／2333 observation binの入力、隔離consumer、通常検証1,453成功、入力だけの単回診断、凍結runtimeの結合まで[完了記録](../../target/post-reboot-continuation-20261001/maximum-runtime-root-binding-v1/evidence-completion-v1.json)を持つ。元wrapperの終了1と別読戻しによる閉包は区別して保存した。元の5回容量観察の初回取得は、専用8 GiB・swap 0のscopeと外部監視、排他時間窓で終了した。[一次終了照合](../../target/maximum-primary-final-join-preparation-20261002-v1/external-final-v1.json)はSource exit 101、scope exit 1、observer exit 0を保存している。[保存summaryの読取診断](../../target/maximum-failed-summary-diagnostic-root-20261002-v1/diagnostic-root-final-v1.json)は診断scope・observerとも終了0で完了し、元のSource失敗を保持したまま内訳を検証した。5反復・11,520候補のProducedは0、Failedは1,900、NotAttemptedは9,620で、各反復では0／380／1,924だった。最初の候補は第1frameでboundの登録上限67,108,864を超え、各bodyは累積訪問上限68,719,476,736に到達して残りを未試行にした。5つのchecksumは全て完成scoreの空総和0として一致しており、容量成功や精度の陽性証拠ではない。Source終了後・controller閉包前に報告された同cgroupのmemory.peakは310,730,752 bytes、前後のkernel OOMは0、完全な一次ログに対象killイベントは0だった。最終瞬間のpeakとkill-sourceフィールドの未確定状態は別に保持し、容量受理はfalseのままとする。排他枠は実PID・cgroupの消失と監視の終了を確認して解放した。独立診断と科学的受理を混同せず、自動再取得しない。これはSource5精度取得の再試行ではない。標準711候補・pattern711の資源判定、通常の同hop16出生・32Voice共存、描画照合と本番採用も未完である。完了したCODE接続を再び未実装へ戻さず、各未完条件を別々に閉じる。

### 2026-09-30の設計判断

9月30日の[保存仕事量と再利用の成立性判断](../../target/body-fitness-feasibility-20260930/decision-v1.md)では、現interval経路の本番化へ進まないという技術判断を記録した。kernel定数・frame内の履歴積は条件付きで再利用できるが、保存traceの約15.35億band訪問、約15.16億H/suffix/sqrt処理、候補ごとの72 frame処理は残り、prominence・mass・scoreを返す軽量producerもない。統括は入力hashと主要counterを独立に照合した。CPU時間や速度比の判定ではなく、追加cache試作・性能取得・同じ保存範囲の再取得を進める根拠がないという結論である。数学診断の成功、旧近似の失敗、閾値と同hop出生条件は保持する。

Task 2と共有する出口は、描画正本が返す処理済みf32 densityと実身体・観測identityであり、最新C_effとの旧積分を維持する。疎表現へのpackだけでは構築費用を解消しないため、新しいadapterは作らない。次に必要なのは、phase・時相・frontend応答を保持しながら候補×72 frame×band/laneのどの処理を除くか、一案の入出力と仕事量根拠を定める設計である。安価な構築法は未成立で、目的関数・描画参照・許容差の変更やrender-onlyへの工程変更は採用していない。

同日の[位相付き構築案](../../target/body-fitness-cold-design-20260930/proposal.md)と[限定Astraレビュー](../../target/body-fitness-cold-design-20260930/astra-review/review.md)では、bodyの制御状態から実疎kernelの応答を複素momentへまとめる一案を検討した。motion角はbank内共有であり、8sample更新数にlane数を掛けた値を不可避の下界とはしない。定常区間の長さdに対する応答は次数d−1の多項式として表せる一方、区間からbandへの転送とpeak・masked massの探索を減らす具体的手順が残る。peak後の履歴は活動支持だけで同じf32計算順を保つ余地があるが、この局所省略だけを出生問題の解決とはしない。本案はHOLD、追加試作・取得へ進めない。

有限の精度・費用検査を始める前に、全未知入力の包括的証明を必須としない。必要なのは、登録済み負荷領域を保つ実装可能な手順、準備を含む静的仕事量、先験的な誤差機構、有限検査と未解決・容量超過時の明示的な失敗である。今回の停止理由は包括的証明の欠如ではなく、残る二つの主要loopを消す手順がない点にある。契約一般の不可能性や作者方針変更の必要性は証明しておらず、旧gate・同hop・最新C/親energy・描画正本は維持する。

音色Phase 2の無音対照について、[作者採用済みの限定規則](../superpowers/specs/2026-09-29-body-policy-author-decisions.md)を隔離参照へ実装した。[保存入力60条件の再評価](../../target/timbre-silence-reference-20260929/evidence-report.md)は、正mass48条件の旧fitness・代謝・親選択記録を完全一致で維持し、完全zero PCMの12条件を実Voice代謝から親選択まで評価した。無音では音響回復と不協和追加費用を0とし、既存基礎費用と実行済み操作の費用を残す。固定条件の最終energyは0.34950000047683716、親確率は0.3478373399094321だった。独立検算、局所7検査、全Rust検査（1212成功・42 ignore）、標準Clippy・all-targets checkを通過した。旧unsupported取得は保存し、音響DSPを再取得していない。これは試験専用の意味規則の検証であり、本番代謝、通常休符判定、F2受入れ、Phase 2全体完了へ読み替えない。

9月29日採用の[音色・合成計画](../superpowers/plans/2026-09-29-timbre-synthesis.md) D6–D10 と [milestones §5の調整記録](../roadmap/temporal-dcc/milestones.md#2026-09-29音色合成計画との調整)を適用する。進行中のF2単位と順序は変えない。**身体footprintの正本は代表描画であり、直接式・部分音群は描画照合を通った範囲だけで使える近道である。** 別の作者規則へ置き換えてv1/v2の未達を解消したとは扱わない。照合の許容差と実測差を記録し、既存のscore 0.025、level 0.0125、旧gap 0.1以上の順位逆転0という判定と失敗結果を保持する。

新しい生態系コードに `BodyKind` 分岐を増やさず、`action_candidates/footprint.rs`・`energy.rs`、`temporal_cognition/body.rs`、`self_prediction` で身体内部値を新たに読まない。既存の分岐・自己予測の移設は音色計画Phase 3で行い、本単位で先取りしない。F2で定める近道の入力・出力はTask 2の任意の自己モデル面と共同で定義する。作曲者の設定対象は聴き手モデルの価値判断・事前分布、初期身体・配置、マクロ形式の三つとし、感覚モデルや生態系へ委ねた形質を新しい操作項目にしない。`timbre-valuation-meter`（`a3a56fd`）はmain未統合として扱い、I12b終結・基準版固定後の統合点まで現在の登録条件へ混ぜない。

[Opus 5.5レビューと採否](body-fitness-plan-review-response-20260929.md)を受け、[Q1–Q6の静的確認と再出発の契約](body-fitness-runtime-restart-20260929.md)を第一工程として完了した。以下の9月26–28日付の進捗・次手は各時点の履歴であり、この現行工程と§5–§7の改訂を優先する。全候補×72 frameの実合成・NSGTは数値参照として保存し、出生要求後の全表計算を本番方式にしない。

一般の子表在庫は、未来の実ID・出生frame・身体入力を常に前確定できず、通常実装へ進めない。予定済みで身体入力まで固定できる子の前倒し準備だけを補助手段とする。F2内で、実子の確定した部分音・modeからその場で身体分布を作る[直接モデルv1の契約](body-fitness-direct-model-contract-20260929.md)を固定し、隔離した数値試作を実装した。方式は未採用。予定Spawn・未予告Spawn・respawnとも既存mainの出生機会hop内にVoiceを生成する設計目標を保ち、音響初発音は発声規則に従う。局所高速化、I11性能探索、音色遺伝、旧36条件の一括再取得には進まない。

直接モデルv1は旧参照との意味判定に不合格。728候補中157候補が事前の誤差上限を超え、score最大差0.447470（上限0.025）、明確な順位逆転3対（必要値0）を独立読戻しでも確認した。結果は[直接モデルv1検証記録](body-fitness-direct-model-results-20260929.md)。同hop出生への接続には進めない。保存済み入力に対してHarmonicの時変dampingだけを加える[単一介入診断](body-fitness-direct-damping-diagnostic-20260929.md)も実施したが、平均誤差は縮んでも未達候補157→160、最大score差0.447470→0.474818となり、進行条件は満たさない。旧参照を再取得せず閾値と未達結果を保持した。

全体回帰は1407成功・0失敗・58 ignore、最新単体5件と標準Clippyも成功した。一方、その終了後に凍結binaryで取得した資源screenは0/16成功。690＋21候補の準備込み1体約1.4–4.5 ms、16体約22–73 msで、16体はhop全体の10.667 msも超える。最大容量の別観察1体19.636 msは事前上限がないため合否を付けない。次の設計では、全候補ごとの全bin走査と候補ごとの身体生成を含めて仕事量を見直すとともに、旧解析モデルへの近似と出生で守る機能の条件を分ける。現比較基準は移動greedy由来であり、出生の音楽的妥当性を直接規定したものではない。閾値の緩和でv1を採用したり、全候補×時間×lane×bandの大計算を別名で再導入したりしない。

[部分音群と疎な採点の設計草案](body-fitness-partial-groups-draft-20260929.md)から、[v2の取得前契約](body-fitness-partial-groups-contract-20260929.md)を固定して隔離実装・取得を行った。実子の部分音を最大32群へまとめ、減衰の平均量と身体ごとに準備するmotion分布で評価し、候補ごとの全bin配列の構築・積分を省く。位相干渉・解析窓・ADSR等は再現しない新しい代理モデルであり、名目帯域外の成分をmotionで再投入しない近似も明記した。

[v2検証記録](body-fitness-partial-groups-results-20260929.md)では内部式等のunit10件は成功したが、旧728候補中159件が誤差上限を超え、最大score差0.397169、明確な順位逆転3対が残った。費用条件は10/20成功。K16/U1の16体は約4.2 ms、motion付きで約7.4 msだったが、K64/U9は約61–70 ms、候補依存Landscape身体は約23–55 msでhop全体を超えた。v1とv2は環境Cや実身体が異なる条件を含むため、厳密な速度比とは扱わない。本番接続へ進めず、保存分布の誤差寄与と、入力最大576 lane・候補ごとの身体生成に残る仕事量を切り分ける。閾値・負荷・失敗結果は変更しない。

取得器の監査では、旧v1のLandscapePeaksはModal指定でも実身体がHarmonicだったと判明した。単独取得器のModal factory未登録によるfallbackであり、保存728入力は変更せず実kindで比較した。旧費用の同条件を実Modalの証拠として扱わない。v2の費用は通常runtimeと同じ登録を行い、生成kindを検査して取得した。

v2の最大誤差例を独立再構成すると、弱い1320–2640 Hzの四群を880 Hz群へ移す規則により、その評価周波数が1078.769 Hzへ動く経路を確認できた。また、群化前の最大576 lane処理が毎候補残り、power順位の身体単位共有も未達だった。次案は実在成分の周波数支持と、候補依存処理の仕事量を先に定義する。v2の係数調整や合否基準変更で採用へ進める工程にはしない。

続く[描画参照の採点位置監査](../../target/body-fitness-support-20260929/reference-semantics-audit.md)では、旧peakの `u_erb` は再配分後重心だが、主観強度を置く `bin_idx` は選択済み局所極大のままだと確認した。[採点位置だけの単一介入診断](body-fitness-anchor-readout-results-20260929.md)では、群・質量・motionを共有してanchorへ戻す変更を隔離実装した。全728候補の先行再現・独立検算は成功したが、描画参照との差は93/728候補で上限を超えた。v2から85候補回復・19候補新失敗、最大score差0.572654、level差0.267519、明確な順位逆転4対で、最大差と逆転はv2より悪化した。旧最大例のscore差も0.075178で上限を超える。解析器の選択binと部分音のanchorは同一ではなく、採点位置変更だけを本番方式にしない。資源の再測定・本番接続は行わず、v1/v2の判定と対象負荷を保持する。

[実Tone/FFT入力の認証区間診断](../../target/body-fitness-real-interval-20260929/evidence-report-v2.md)は、固定728候補・52,416 frameを一回抽出し、入力だけを読む区間producerを2 processで実行した。両数学streamはbyte一致。独立Cは全frame・全band・1,952,417 roundの包含・確定ラベル・履歴・完全参照bit一致を検査し、旧728最終出力も再現した。Fractionは全728候補のframe 0・固定4band・保存全round、113,912記録・1,025,208区間pairを通過。登録陰性8件は全件で期待した拒否理由とexit 2を確認した。これはband区間とpeak入口三条件の有限診断であり、prominence・mass・scoreを返す新しい身体モデルではない。

[保存streamの独立集計](../../target/body-fitness-real-interval-20260929/reduction/result-v1/reduction.json)では、prefix項の省略は22,488,537,019/41,505,662,016（54.181854%）。省略分の78.675319%はguard項で、公開bandの項省略は26.333210%、公開band×frameの79.156663%は全項計算まで進んだ。全bandが完全prefixの4 frameではhistoryが未確定のまま残った。同じharmonic_spread身体のframe 52・bin686/687が4環境で反復したものであり、4つの独立音響反例ではない。算術範囲unknownは0。[52 case別集計](../../target/body-fitness-real-interval-20260929/case-summary-v1.csv)と準備・28種の演算counterを保持する。追加のnorm・bound・丸め費用を含む総費用削減の根拠はなく、性能取得・通常runtime接続へ進まない。旧精度失敗、資源不合格、同hop要件は不変。[部分式再使用の数学整理](../../target/body-fitness-real-interval-20260929/reuse-dependency-contract-v1.md)は条件付き数値同値までで、実装・counter登録の変更や新取得は未実施。

未完診断を含む隔離worktreeの原状は、作者の指示により `body-fitness-direct-model` ブランチの `0f0f4f8` へ152ファイルを無修正退避した。`target/` と1 MB超ファイルはコミット対象外で、[一覧と照合記録](../../target/body-fitness-support-20260929/checkpoint-report.md)を保存した。その後の診断修正はレビュー指摘4件を解消し、対象unit・標準Clippy・all-targets checkを通過した。全Rust回帰は初回・再実行とも非同期準備test1件で失敗（1,315成功・1失敗・61 ignore）し、単独再実行だけ成功した。対象request限定の試験用記録を追加した3回目も同じ判定で、30秒時点に新jobの132候補中115候補の計算が完了し、応答送信前であることを確認した。期限・候補数・並列規則は変更していない。[進捗監査](../../target/body-fitness-support-20260929/full-test-trace-audit.json)で候補計算区間へ局所化したが、根本原因は未確定。試験用記録の最終sourceでは単独検査・fmt・標準Clippy・all-targets checkが成功し、全suiteの追加反復はしていない。3回の全体失敗を保持し、診断修正と試験用記録は未コミットのまま残す。

非同期出生の[最新隔離版time4-job-probe](../../.worktrees/body-fitness-nsgt-time4-job-probe/docs/design-notes/body-fitness-nsgt-time4-job-probe-results-20260928.md)は、実ActiveJobの全690候補計算と30秒内3出生の登録条件を通過した。数値・性能・配送の照合と残条件は下記に記す。先行する[協調worker](../../.worktrees/body-fitness-birth-cooperative/docs/design-notes/body-fitness-birth-cooperative-results-20260928.md)が広域ON条件で1出生と翌hopの自己除去receiptを成立させた。全690候補の準備は約19.85秒で、30秒内3出生は未達である。[NSGT連続index群](../../.worktrees/body-fitness-nsgt-runs/docs/design-notes/body-fitness-nsgt-runs-results-20260928.md)は全密度のbyte一致を保ったが、固定全表のABBA比較でCPU時間が約39%増え、不採用とした。後続の4帯域AVX2 SIMDも保存PCM720 frame・全690候補密度のbyte一致を保ったが、全表ABBAのCPU時間が約10〜12%増え、不採用とした。3出生試験への進行条件を満たさず、30秒paced取得は行っていない。以下の旧取得の出生0や未測定という記述は、その取得時点の状態を指す。

[I11の因果履歴接続](../../.worktrees/i11-causal-group-join/docs/roadmap/temporal-dcc/i11-causal-group-join-results-20260928.md)では、重複音条件で自己群除外を各取得1判断観測した。続く原点記録版では同じ8条件を再取得し、旧版との全WAV・全判断一致を確認した。全112判断の消費snapshot・現行Voice状態・実出生の原点照合が成立し、身体Record・Binding・CDF・群履歴までの限定対応は重複音条件の二反復で各1判断成立した。同一scene・seed・同一判断の再現であり、独立した2条件の正例ではない。残る110判断はBinding欠測等でunknown、整合エラー0。原登録§4.5はBindingなしなら自己群を除外しない仕様なので、110判断をすべて成立例へ変えることは要件ではない。物理的source帰属と保持中Toneのrouteは未証明だが、完全な自声同定を新たな必須条件へ追加しない。[主担当の照合記録](../../target/parallel-continuation-20260928/i11-origin-complete-verification.json)に全112判断・5,256 producer hopの分類を保持した。旧[事後照合](../../target/i11-complete-join-20260928/results-v4.md)の完全対応0・snapshot直接一致14件は、間引き報告だけを使った旧取得の結果として保持する。進行中の固定432件は既登録の監査まで完了し、新しい性能候補や反復取得には進まない。残条件は[原登録との仕分け](../../target/i11-requirement-realignment-20260928/read-review.md)に沿って扱う。

三消費者を交差させた最新の有限比較は[実枯渇から親選択・出生への64本](../../.worktrees/body-fitness-selection/docs/design-notes/body-fitness-selection-results-20260927.md)である。同一PopulationのHarmonic／短寿命Sine／Modal founderと非遺伝の固定Sine子について、移動・代謝・出生の点／身体8条件、初期f0割当2通り、通常振幅／孤立固定音RMS減衰の32セルを各2回取得した。全64本がexit0で、条件内WAV・決定的記録が一致した。a/bは決定性検証であり、独立した科学的反復とは数えない。

各軸16対の共通入力を照合し、移動では全対でtarget・実Hz・energyの差、代謝では全対でscore・level・energyの差を確認した。代謝から更新後親energyへの作用は、出生basisによる重複を除いた固有8対すべてで親選択確率へ伝わったが、実際の選択親は同じだった。出生basisの直接比較では16対中10対で実子Hzが変わり、6対は変わらなかった。全取得でtriggerの代謝枯渇から一出生、翌hopの子の自己除去receiptまでを検算した。親2声と子はFinish時点で生存しており、その生存時間は右打切りである。全1354テスト、fmt、標準Clippy、all-targets checkを通過した。

先行する[出生なし有限コホートv2](../../.worktrees/body-fitness-cohort/docs/design-notes/body-fitness-cohort-results-20260927.md)は32本すべてで枯渇からVoice退役・renderer tail終了を観測した別比較であり、生存変化の向きは条件によって異なった。今回も音色固有の一方向の優位や一般的な長期選択は立証していない。版履歴、失敗、監査範囲、証拠索引は各結果文書と[並列継続記録](parallel-continuation-20260927.md)に保持する。

現統合版actionの[30秒持続試験8条件](../../.worktrees/body-fitness-action-sustained/docs/design-notes/body-fitness-action-sustained-results-20260928.md)は取得・監査を終えた。事前固定checkerでは6/8合格、Ready消費拒否の終端計上漏れを別版で訂正した取得後監査では7/8合格。ON全4条件で実消費・有効率・反復変更後の期限内回復を確認したが、split-onは資源採取1回欠測により不合格を維持する。主担当が最終索引1,445項・626,659,830 byteの全サイズとhashを照合した。実deviceと作者受入は未判定である。

代謝の[有界非同期配送28条件](../../.worktrees/body-fitness-async-metabolism/docs/design-notes/body-fitness-async-metabolism-results-20260928.md)も取得した。全28条件で配送検査を通過し、ON14条件は全評価機会・Ready履歴・密度積分・energy更新を独立検算した。energy stage差は全件0 ULP。固定条件のlifecycle有効率は99.36–99.79%だが、動的HzではModal単独約0.067%、action併用4声のsource別21.96–34.54%で、高有効率を示していない。固定checkerは8 OFF条件で異なるRSSカウンタの厳密大小比較に失敗し、F-4／D-4の4 pairは全寿命最大RSS増分256 MiBを超えた。主判定は20/28 case・2/14 pair通過で、補助的な数値正例と資源条件の未達を区別する。Rust全1,368テストは通過した。最終索引1,493項・1,693,002,199 byteは主担当も全サイズとhashの一致を確認した。

取得器を修正した新しい資源再取得は、[action8/8条件](../../.worktrees/body-fitness-action-resource/docs/design-notes/body-fitness-action-resource-results-20260928.md)と[非同期代謝28/28条件・14/14組比較](../../.worktrees/body-fitness-async-resource/docs/design-notes/body-fitness-async-resource-results-20260928.md)で事前固定監査を通過した。旧取得の不合格判定は変更していない。actionは全ON条件の有効率・反復変更後の回復を再確認し、稼働中の資源欠測は0だった。非同期代謝は終了後readbackだけを縮小し、同じ入力と全寿命予算で最大RSS増分2.898 MiB、ON絶対最大64.219 MiB、CPU増分最大0.813 coreとなった。全195,104 lifecycle評価と385 onset batchの数値経路を検算し、energy stage差は全件0 ULP。固定pitch条件の有効率は99.467–99.787%、固定Hzのbrightness二変更は99.360%。Rust全1,369テストも通過した。これは音声deviceを使わないpaced通常runtimeの隔離取得であり、実deviceの受入ではない。

非同期出生・respawnの新36条件は全取得exit0、全条件2813 hop・予算超過0・underflow0で取得を終え、登録済みの後検査も完了した。[postrun-v2](../../.worktrees/body-fitness-nsgt-time4-job-probe/target/time4-next-acceptance-v1/postrun-v2/postrun-results.json)はoracle 18、science 36、resources 1、失敗0、`pass=true`、SHA-256 `92c6ab7b27d2030bf9674ea64e6e19b7c97fcd065667383fb8dbe8bed34ed2c6`。今回の確認は保存結果JSON・終了status・入力hashの照合であり、全DSPの独立再実行ではない。旧登録の診断合格であって、出生期限の実用合格ではない。[待ち時間集計](../../target/time4-birth-latency-scope-20260928/on18-v1.json)ではON18件の全26予約が実出生し、未観測0・整合エラー0だった。初期20出生の予約からの遅延は中央値3.451秒、最大24.651秒、respawn 6出生は中央値2.501秒、最大2.560秒（試験sample時計）。即時出生や実用上の許容待ち時間を満たしたという判定ではない。動的Hzの低有効率は適用限界として残すが、非同期代謝の原登録は可変pitchに率の下限を設けていない。28条件の配送・数値・資源合格と、動的生態比較へ使える高有効率が未実証である点を分ける。高有効率を後付けの必須条件にせず、補間改善や新性能探索は開始しない。前者の[密度線形補間の有限数値試験](../../.worktrees/body-fitness-dynamic-density/docs/design-notes/body-fitness-dynamic-density-results-20260928.md)では、5 family・6格子幅の全30組が登録誤差条件を満たさず、通常runtimeへ接続しない。最細0.78125 centでも解析peakのbin切替を両端密度の混合で再現できなかった。 同版の全1,371 Rust検査とfmt・標準Clippy・all-targets checkは通過し、主担当が最終索引482現物・269,522,077 byteを全照合した。数値未達を保持して同単位を固定し、[元のexact 72-frame計算の費用分解](../../.worktrees/body-fitness-exact-cost/docs/design-notes/body-fitness-exact-cost-results-20260928.md)は29固定入力・16反復対を測定し、全464対の密度・mass bitsと旧oracle交差11入力が一致した。計時leaf総和の88.337%がNSGT、9.625%がTone描画だった。全1,371 Rust検査を通過し、最終索引944現物の全hashを主担当も照合した。通常runtimeの高有効率とは分けてNSGT内部も29入力・464対で測定し、全数値bits一致を確認した。NSGT内の92.105%がsparse積和とsmoothingを含むband loop、7.013%がFFTだった。同版の全1,372 Rust検査も通過したが、改修後の速度向上はまだ測定していない。後者は子の身体生成seed予約と実出生時計を分離して、最新環境・親energyによる選択へ有界非同期準備を接続する隔離実装である。通常runtimeの受入は両者とも未完了である。新取得でもModal単独は5/7,504（約0.067%）、action併用4声はsource別15.67–28.12%であり、高有効率は未達である。三消費者の決定的offline経路に加えて代謝の通常runtime明示ON経路を検査した。非同期出生・respawnは隔離版で実装し、追加診断を含む最終sourceの全1,382 Rust検査が通過した。旧封印sourceから再構築したrendererとのOFF対照は全9 sceneでWAVがbyte一致した。これは代謝・actionもOFFの音声不変対照である。[9 scene・36本の通常配送結果](../../.worktrees/body-fitness-async-birth/docs/design-notes/body-fitness-async-birth-results-20260928.md)は固定し、資源検算は36条件・18比較組すべて通過した。科学判定はON18条件すべて未達で、16条件・20出生は次hop receipt未利用可能、広域2条件は出生0だった。数値整合性のエラーはなく、非空16本の4,004密度を再生成照合したが、通常出生経路の受入とは区別する。固定pitchの成功を可変pitchの高有効率へ読み替えない。全版は隔離実験であり、main採用・既定変更は行っていない。F5の実device・作者受入とF6の音色遺伝は別工程として残す。

[NSGT pack候補](../../.worktrees/body-fitness-nsgt-pack/docs/design-notes/body-fitness-nsgt-pack-results-20260928.md)はgatherをscalar loadとlane packへ置き換えた。固定synthetic FFT・786帯域の局所計測では、全432 passの出力一致を保ち、CPU中央値が旧scalar比約30%短縮した。別隔離版は全1,402 Rust検査、format、標準Clippy、all-targetsを通過し、固定release buildを完了した。続く保存PCM720 frameは旧協調workerと全byte一致し、固定690候補表の旧新新旧4process比較も全密度・oracle byte一致を保った。CPU新旧比は0.71834／0.72956で、両組とも事前基準0.95以下を満たした。続く30秒paced試験では2出生・翌hop receipt2件が成立し、取得済み1382密度がoracleと一致した。全2813 active hopで予算超過・underflowは0だったが、3番目の全表は終了時pendingで、登録3出生条件は未達だった。固定全表の速度改善と通常runtimeの受入れを分けて保持する。3予約の代表密度表を共有する案は、同Hzでも個体別Recipeと密度が異なるため現契約では採らない。 後続の8帯域版はstandaloneの数値一致を通したが、局所CPU中央値の4帯域比は0.96564で、本体組込みを保留した。続くactive-index案は4帯域のままmetadataを64→48 byteへ縮め、native/AVX2最小の4,748 laneでbit一致を確認した。しかし[局所計時](../../target/nsgt-pack-active-index-preparation-20260928/timing-acquisition-v1/result.json)のCPU中央値比active-index/pack4は0.9687309で、事前条件0.95以下に届かなかった。全432行と23入力assetの取得後照合は成功したが、本体実装には進まない。さらに4lane共通prefixのmask/blendだけを省く候補も、native/AVX2最小の数値照合を通過したが、[一回の局所計時](../../target/nsgt-pack-common-prefix-preparation-20260928/timing-acquisition-v1/result.json)は旧pack比0.96498251で登録0.95以下に未達だった。全432行と32入力assetは整合し、再計時・本体組込みは行わない。動的Hzは[peak前power補間の三点probe](../../target/body-fitness-dynamic-prepeak-preparation-20260928/registration-v1.md)を別隔離版で準備し、全1,402 Rust検査とfmt・Clippy・all-targetsを通過した。release binaryと11資産を固定した一回の三点probeは、旧density補間がピーク位置を外したSineの登録一点で4精度条件を通過した。分布L1は2.4326e-6、ERB fitness mass相対差は2.0371e-4、score差は1.4901e-8、level差は5.9605e-8で、主担当の独立再計算も一致した。これは現行guard付き解析に対する単一照会の診断結果である。旧診断はguardなしだが、この一点では旧新のdirect scanと両端node各690 bin、spectral mass、Hz、Recipe hashが実測bit一致し、参照値変更で合格化したものではない。別周波数・全音色でのguard影響は未検証である。全音色・全照会、7504 stepのenergy再演、通常runtimeの受入れは未検証である。

次のNSGT候補は4 time-frameをSIMD laneへ割り当て、係数の読取りを共有する方式である。standaloneはnativeと最小AVX2で6系列×4 frame×786帯域のbit一致を確認した。転置scratchは524,288 byte、疎積和loop内のFMA・gather・shuffleは0である。[固定planの単回計時](../../target/nsgt-time-batch-preparation-20260928/timing-acquisition-v1/result.json)では転置込みCPU中央値がpack4の1,111,329 nsに対して541,730 ns、比0.4874614となり、事前条件0.95以下を通過した。全432測定のdigestと39資産の取得後照合も一致した。これは合成FFT入力の局所結果である。代表72 frameへの隔離統合は親517 sourceのうち4件を変更し、coreの786帯域×72 frame、共有frontendの72 frame、3身体の平均scan/massのbit一致と取消のfocused検査を通過した。全1,405 Rust検査、fmt・標準Clippy・all-targetsとrelease buildを通過した。主担当は517 sourceの現物と保存コピー、build原本と固定binaryの同hashを確認した。保存PCM720 frameは単回取得・既存checkerを通過し、10 query全JSONLが旧packと旧cooperative双方にbyte一致した。ただし全690用に準備したABBA入口が旧計算関数を直接呼び、time4 workerを通らない点を取得前に発見した。計時は未実施のまま停止し、実ActiveJobを通す同一診断入口を旧新の別検証コピーへ追加し、旧版1,402件・新版1,405件の全Rust検査を完了した。両版のsource不変とテスト原本を主担当も照合した。続いて両版のreleaseを固定し、新binaryの保存PCM720 frameが旧pack/cooperative双方とbyte一致した。実ActiveJobを通る単回ABBAは全690候補のoracle JSON・密度が固定baselineとbyte一致し、CPU比0.5125372／0.5255896で事前速度条件を通過した。同じ新版の30秒I-W4-on-aも、3表各690候補の準備、3出生、翌hop receipt3件、全2,813 hopの予算超過0・underflow0で登録条件を通過した。主担当は全2,073 oracle行の保存密度・recipe・周波数と出生時刻を独立照合した（[独立記録](../../target/parallel-continuation-20260928/time4-job-paced-independent-review-v1.json)）。これはseed7・初期出生1条件・模擬sinkの成功であり、出生hopの自己PCM3件はすべて無音だった。OFF相対資源、親energy経路、他scene、実deviceと作者受入は未判定で、F4全体の完了や既定採用を意味しない。別系統のpeak前補間30点版は全1,402 Rust検査とfmt・Clippy・all-targetsを通過した。固定releaseの単回30点取得は全child exit0・整合失敗0だったが、登録精度は17合格・13不合格となった。主担当が全120品質値を独立再計算し一致を確認した。最細格子の選択5点は通過したものの、Modalの途中格子幅では分布L1が0.07009935で不合格となり、単調な精度改善は示していない。30点screen全体は不合格とし、全照会・energy再演・通常runtimeの合格へ一般化しない。別検証として幅96/1536の全prior_actual 2,030点を固定した版は、診断probe 1件だけを変更し、全1,402 Rust検査を通過した。環境欠測20点を分母へ残す。source・release・正式planを固定し、主担当が2,062参照資産と全2,030点の順序を照合した。単回取得は全子exit0・整合失敗0で完了したが、score適格2,010点の四条件合格は1,810点、200点不合格だった。環境欠測20点は分布L1とmassの二条件だけを別分母で通過した。主担当の独立再計算8,080指標は全値一致した。最細幅1536でも適格1,005点中14点が不合格であり、通常runtimeには接続しない。

I11の保持率キャッシュ案は、201 horizonの保持率をconstructorで事前計算する。standaloneの96状態・57,888予測値は旧式とbit一致したが、[単回32process計時](../../target/i11-energy-retention-preparation-20260928/timing-acquisition-v1/summary.json)の成熟forecast CPU中央値比は0.9803835で、事前条件0.95以下に届かなかった。旧・候補双方の状態構築とdigest処理を含むprocess CPUの局所検査であり、productionへ組み込まない。旧I11資源判定も変更しない。 続く別案は成熟forecastのlagを外loop、horizonを内loopにして各horizon内の加算順を保つ。96状態・57,888値のbit一致後、固定24processのCPU中央値比0.9177970で局所0.95以下を通過した。constructor・resetは不変更であり、通常構造体を用いる別隔離実装は1ソース変更に限定した。実historyの全horizon bit境界検査、fmt・標準Clippy・all-targets、全1,204 Rust検査と固定releaseを通過した。36条件の機能回帰は全取得・固定4後検査に合格し、主担当の独立再計算でもNone全12 WAV、10種学習record列、共通候補11,386件の内容が一致した。候補の旧側だけ191件・新側だけ323件は別集計する。到来ONを含む8条件も全取得・登録21後検査・4対応照合を通過し、旧WAV全8件、112判断・5,256 producer、同じModal判断のA/B反復による限定対応2件・unknown110件を再現した。これで[機能回帰36＋8条件](../../target/i11-energy-loop-production-results-20260928/results.md)は完了した。新432条件は全取得exit0・整合エラー0で完了し、主比較の到来ON対Noneは13/16条件合格・3条件不合格だった。主担当の[独立再集計](../../target/i11-energy-loop-resource-432-independent-review-20260928/independent-reduction.json)は48組のA/A floorと全16比較で登録監査に一致した。own処理は全16条件で費用判定を通過し、同時計の予算超過は0。残る不合格はpopulationの中央値またはp99、一条件では高速方向のsynthesis p99差も含む。後半はユーザー指示による出生36条件との並行取得であり、改善幅をloop変更だけへ帰属しない。旧432条件の判定は維持し、新結果の不合格を理由とする追加性能探索は行わない。

I11は現モデルを固定したNone／到来OFF／到来ONの[432本比較](../../target/i11-2-resource-preparation-20260928/result-v1.md)を完了した。全run exit0・整合エラー0だが、主比較の到来ON対Noneは5/16条件のみ合格、全体不合格である。到来knownは2,508判断中369件、CDF確率を使った候補は8,487件。待ちを除く処理の予算超過hopは0でも、reportありのown_usと一部のreportなしのpopulation_usが登録許容差を超えた。144主比較profileの数値集約は主担当も独立再計算して一致を確認した。旧I11-1の2条件未達とは入力モデル・条件が異なるため、旧判定は変更しない。report経路にはarrival有効時だけの有界writer、population処理にはCDF窓確率の重複計算削減を別隔離版で実装した。書込失敗時のflush循環待ちも反例で修正し、統合1,204 Rust検査、format、標準Clippy、all-targetsを通過した。固定binaryの[32process小規模費用比較](../../target/i11-resource-cost-screen-acquisition-20260928/result-v1.md)も全child exit0・整合エラー0で完了した。report付き8対のown中央値・p99とprocess wallはすべて短縮したが、process CPU比は0.865–1.007で混在した。modal条件では背景candidate energyの完了件数が125から117へ減り、同じ全仕事量の効率改善とは判定しない。reportなしではp99が3対で増加した。[新版の機能回帰](../../target/i11-resource-cost-regression-results-20260928/results.md)も完了した。元12scene×3条件は36取得・4後検査が成功し、None全12 WAVと10種学習record、共通候補11,329件が一致した。到来ONを通る原点8条件も旧全WAV・全判断が一致し、112判断・5,256 producer、限定対応2件・unknown110件を再現した。元36条件はarrivalが既定falseなので両回帰を区別する。同じ432件設計による[新版資源取得](../../target/i11-resource-cost-432-preparation-20260928/result-v1.md)も全子process exit0・整合エラー0で完了した。登録C/Aは10/16合格・6不合格、補助C/Bは13/16合格。own_usは全16条件で通過し、残る不合格はpopulation_usだけである。Sine-flow4のreport有無2件はON側が速い方向の差、Harmonic-flow16のp99有無とModal-flow16のreport中央値・no-report p99の4件は増加方向だった。対称絶対差の登録基準を変更せず6件とも不合格に保持する。全432 profile・AA48組の統計と判定は別実装でも一致した。48 report補遺と終了までのwait4 CPU/RSS 432件も整合したが、背景仕事量の違いと資源受入れ未達を残す。

追加の[population区間診断108本](../../target/i11-population-breakdown-acquisition-20260928/result-v1.md)は全child exit0、54組・18条件・診断32,994 hopを欠測なく取得した。6区間と残差の和は全hopで外側時間と一致し、独立集計702統計値も一致した。ただし候補worker件数がU/Dの45/54組で異なり、reportの非時刻record列は27/27組で不一致だった。生存Voice数・render後Tone数は全組で一致した。reportの判断IDを対応付けた事後検査では2,988判断の選択5字段とonset許可も全一致し、record値1,334件の差は主にfootprintの要求・受信時刻だった。ただし背景workerの件数差を残すため、同仕事量の純粋な観測追加費用とは判定しない。区間別p99ではenergy context・判断処理の区間が大きいが、旧432の不合格の直接原因とは同定していない。旧資源判定10/16合格・全体不合格は維持する。

I11の[同窓入力診断](../../.worktrees/i11-window-inputs/docs/roadmap/temporal-dcc/i11-window-inputs-results-20260927.md)と部分hop補遺は固定した。実Bindingは単独条件で2判断成立したが共有群への割当はなく、この旧取得での通常自己群除外正例は0だった。既存rawの追加探索では全sourceと部分groupのenergy量差、prototype経由と直接比較の距離差を分離したが、モデル・尺度・gateや実消費経路は変更していない。
[同じ群配分をprivate身体へ適用した三座標診断](../../target/i11-group-projection-20260928/results-v2.md)では、Aの三座標完備92組すべてがgroupと一致し、Bは87組すべて不一致、Cは67組中66一致だった。全a/b結果は一致した。Aは同じ入力と配分の構成上の整合性であり、物理的source帰属や通常自己群除外の正例ではない。三座標と旧六座標の分母を分け、Cの支持欠測を保持した。
I11-1の正式費用判定は14/16合格、統合25条件はModalの記憶参照0により全体不合格を維持する。
[身体評価の参照結果](body-aware-fitness-results-20260926.md)と[統合検証](body-fitness-integration-validation-20260926.md)を参照する。
調査基準: `06a4772c43d06b41b44753bb0be891f23e16b93e`。F0調査時の読み取りで remote main も同じ commit と確認した。

## 1. 目的と今回の判断

Voice の移動・代謝・出生先を、自分の基音だけでなく、自分の部分音と環境の相性で評価できるようにする。
自己音の除去と候補身体の評価を別々に検証し、その後で同じ評価器を三つの消費者へ接続する。
音色遺伝と合成方式の拡張は、その評価器が成立した後の単位とする。

推奨は、現在の時間構造 DCC の取得を継続し、この計画を独立した音色・生態の工程として進めることである。
現在の A4 に新しい必須条件を追加しない。身体入力の契約と参照実験は先行できるが、通常動作への採用は
現在の A4 後を基準とする。A4 前の実装は隔離した実験版までとし、A4 の対象版に含める場合は、正式取得の前に
対象版と影響する受入条件を改訂する。既存の取得へ後から混入させない。

以下の F0–F6 は本計画内の作業番号であり、既存の I/R/A 行ではない。レビューで使った T1–T4 は
既存の時間認知 T1/T2 と衝突するため、ここでは「自己アンカー」「音色選択」等の検査名で呼ぶ。

## 2. 現在進行中の作業との関係

現在地の参照元は [milestones §5](../roadmap/temporal-dcc/milestones.md#terrain-handoff-20260924) と各実装記録である。
Org の案件 `project-lwrk-conchordal` の WORKDIR もこの checkout を指す。検索時点では対応する Org タスクファイルに
個別の作業見出しはなく、本計画から新しい TODO や状態の写しは作らない。

| 作業 | 確認できた状態 | この計画との関係 |
|---|---|---|
| I10 身体と帰結入力 | 09-21に固定 development 素材への限定付きで完了。資源・作者・実機受入とは別 | Tone、搬送波予測、身体世代、route、自己音採取を再利用する。I10 全体をやり直す前提にはしない |
| I11-1 代表 onset footprint | 現基準06a4772で同一record修正後36本を再取得。§5.4a／b・§5.5(a)を通過し、§5.9 gapも現版で再取得・一致。§5.7は専有取得384本exit 0だが、改訂登録の直接処理欄が14/16合格・2/16不合格 | 現版の第1段技術完了とは扱わない。footprint生成条件・配送と周波数評価を同時に変えない |
| I11-2 / I12b | I11-2のCDF数値・実消費・局所的なON/OFF差を隔離版で確認。後続のreport-only版ではON241判断を正確な消費snapshotから独立再計算し監査エラー0、旧10本の音声・provenance以外の判断と一致。さらに固定周期刺激の通常取得2回で、各28判断中19判断の複数群平均を検算した。I12bの封印版・取得器・結果・稼働はF0で確認できなかった | I11-2の自己群除外・全体資源は未完。静音bus0のBindingが自己群候補となる境界へroute・同一消費Recordの必要条件を追加。ON269判断の監査エラー0、旧11音声・判断不変、全1171テスト成功。通常自己群正例は0。I12b完了の代用にはしない |
| I4 参照と研究検索の分離 | 隔離版の第一段階36本、第二段階の固定区間照合・独立producer・実消費、memory=Noneの12条件回帰を通過。source capsule保存済み | 共有runtime、自己予測、識別・時刻の配線変更が競合する。確定した隔離版の配線へF3を統合し、結合後の検査を別に行う |
| T1 マスキング地形 / T2 変調地形 | 09-24に A4 対象へ追加。09-29に作者がI11-3=T1、I11-4=T2の管理番号を承認。原典モデル・係数・許容差・取得登録は未承認。音色Phase 4を登録準備へ含める | 既存身体の立ち上がり・減衰・再励起だけで、平均スペクトルをそろえた時間的まとまりの対照を定める。音量・onset密度・変調・既存meter出力を記録し、聴取差・解析の検出差・行動へ戻った差を別に報告する。実施は音色Phase 3後という既存順序を維持し、現在のF2へ取得を追加しない。T2やmeterを本計画で置換せず、新検出機構は別計画とする |
| I13 作者操作 | 草案。地形の意味が定まる前の係数公開は保留 | 新しい適応度の選択は最初は実験用に限定する。時間 mode に周波数評価の意味を重ねない |
| R2 / R3 / A1–A4 | R2 正式取得は I4・I10 再検証と T1/T2 地形を含む最終構成を待つ。A1 は版ごとに取得 | 新評価を含める場合だけ資源・感度・可聴性・作者採用を追加する。旧結果を新版の合格へ転用しない |
| 音色設計、励振の統一、遺伝 | [timbre.md](timbre.md) の damping と比率を使う近似 LOO は実装済み。励振の統一・音色遺伝は未完 | 評価器は現行の加算・モーダルで作る。励振方式の変更を前提にしない。遺伝は F6。音色側の工程とF6の正本は[音色・合成の実装計画](../superpowers/plans/2026-09-29-timbre-synthesis.md)（2026-09-29） |

### F0開始時に存在した記録の不一致

F0開始時の[I11-1旧末尾](../roadmap/temporal-dcc/i11-onset-comparison.md)は、同一 record 修正後の再取得と再判定を未実施としていた。
一方、09-24の後続計画には旧 I11-1/2 の限定技術完了を前提にした記述がある。
また `r2-preflight.md` が参照する `i11-2-results.md`、新しい計画が参照する台帳の
`two-tier-rule` / `body-beat-time-constants` アンカーは調査基準の checkout に存在しない。
この計画では、後続計画が述べる方針と、手元で検証可能な完了証拠を分ける。未収録の結果を失敗・未実装とも認定しない。
F0でsource・結果・稼働状態を確認した。その後の再取得と§5.7の未達は上の現在地と後続の結果欄を優先する。

### 作業順と競合

09-24の主工程案は「I12b の取得・終結と基準版固定 → I4 第一・第二段階と影響する I10 再検証」。
その後の正式 R2 は T1/T2 地形と影響回帰を含む版を待つ。T1 の A1 は T2 地形前の版を固定し、
T2 後に影響を再確認するという [現行の順序](../roadmap/temporal-dcc/a1-i11-audition.md)を維持する。

この工程と並行できるのは、本計画の契約整理、既存記録の確認、別環境での参照実験である。
取得中の I12b が確認された場合、その終結後に別 worktree・別 `CARGO_TARGET_DIR` で F1/F2 の数値取得へ進む。共有 runtime へ入る F3 は
I4 の配線変更と直列化する。T1 担当とは F1 の前に出所・周波数・時刻・単位だけを照合し、T1 の実装完了は待たない。
同じ機器で時間測定中は、別 worktree であっても cargo・test・render を競合させない。

### 2026-09-26 の実行時監査による更新

[F0 監査](body-aware-fitness-f0-audit.md)で、現 checkout には I12b の封印済み実行版・登録取得器・結果を確認できず、
I11-2 も到来入力の実消費を確認できなかった。ホスト側のプロセス確認でも当該取得の稼働を認めない。
したがって、過去の引継ぎにある「I12b 取得中」を現在の稼働事実とせず、その終了を待つ工程を現状へ機械的に適用しない。
I12b を完了扱いにする変更ではない。別機器の未収録成果の照合と、実装・登録の不足は未完として残す。

現在の source `06a4772` を保持し、I11-1 の同一 record 修正後36本を新規取得先で再取得した。
登録判定の結果と検査器の旧前提の修正は [F0 監査](body-aware-fitness-f0-audit.md) に記録した。§5.9は現版の再取得で元の集計と一致した。§5.7の費用は他の計算と競合しない専有時間帯で取得し、下記のとおり2組の不合格を残した。
取得・判定終了後に `.worktrees/body-aware-fitness` の隔離版で F1/F2 の最小数値試験を実施し、別の `.worktrees/i4-reference` で I4 第一段階の分離・回帰検査を開始した。
F3 の共有 runtime 接続は引き続き I4 の配線確定と直列化し、A4 の範囲・既定動作は変更しない。

I4第一段階では36本すべてのWAVと登録した10種類の学習recordが一致し、
temporal_observationも事前指定の実時間診断欄を除いて一致した。候補recordの共通keyは一致したが、
片側だけに存在するkeyやworkerの完了時刻・計数の差は残る。全reportのbit一致とは呼ばない。
この入力名の `None` はI11 onset footprintの設定であり、memoryは有効である。
I4第二段階の `memory=None` 回帰は別の基準入力で行う。

F1/F2では通常48 kHz解析の9場面まで数値参照を通し、固定候補による自己アンカーの応答を追加取得した。
自己除去後の地形を持つことと、それを候補生成・移動へ期限内に渡すことは別である。
共有runtimeの接続前の課題は [引継ぎ](body-aware-fitness-runtime-handoff.md) にまとめた。

ALIFE 論文の評価式比較も独立した研究工程とする。影響メモは `~/wrk/conc-paper-2026/notes/2026-09-26-body-aware-fitness-impact.md`。
`.worktrees/alife-fitness/conc-paper-2026` と隣接する `conchordal` v0.3.0 に基準を固定する。
宣言6部分音の質量を使う論文モデルと、実音を前処理した密度を使う F1/F2 は区別し、結果を相互の合格へ転用しない。
論文本文・既存の実験結果を上書きせず、比較の登録と結果は別の研究出力へ保存する。

## 3. 再利用するものと、分けるもの

| 既存箇所 | 再利用する責務 | この計画で追加する責務 |
|---|---|---|
| `life/sound/tone.rs`、`oscillator_bank.rs`、`modal_engine.rs`、`bank_forecast.rs` | 同じ身体・励振・包絡から実際に鳴る音を作る経路。静的条件の予測可能範囲 | 候補基音に対する代表身体の周波数質量。動的条件を静的予測が扱えるとは仮定しない |
| `life/schedule_renderer.rs`、`core/temporal_expectation.rs::OwnSoundHistory` | source 単位の routing 後 PCM と、解析前に自己 PCM を引く境界 | NSGT と SpectralFrontEnd の自己除去参照。既存の3帯域履歴だけから Log2Space scan を復元しない |
| `life/action_candidates/footprint.rs` | recipe、身体世代、時刻、代表条件、期限・unsupported の規則 | 現在の16要素は時間区間の energy。周波数 scan として転用せず、必要な周波数出力を別に定義する |
| `core/stream/analysis.rs`、`landscape_spectral.rs`、R/H kernels | 実音解析・前処理・R/H/C の数値計算 | offline の比較と runtime が同じ数値経路を使う。Python に本体を再実装しない |
| `life/pitch_core.rs`、`voice.rs`、`community/frequency.rs`・`respawn.rs` | 移動・代謝・出生の消費境界 | 同じ身体評価器の score/level を消費する。出生時は子の身体を確定してから評価する |

`Voice::render_spectrum` は旧来の宣言振幅投影であり、scheduled Tone の実状態を表す根拠には使わない。
配線の入口として無条件に採用せず、F1 で Tone 由来の参照と照合して使える部分だけを再利用する。
I10 の実指令条件付き予測、I11 の代表 onset 予測、今回の代表周波数分布は別の量である。
識別や生成コードを共有しても、精度の証拠とキャッシュ内容を共用したとは扱わない。

## 4. 初版の数値契約

### 4.1 二つの身体入力

自己除去には「解析対象区間で実際に habitat へ出た、その source の全 Tone の PCM」を使う。
停止後の残響、重なった発音、pitch glide、実励振、route を含め、同じ source の現在の基音一つで代用しない。
presentation-only の音を habitat から引かない。退役した親の残響を、新しい子の自己音として除去しない。

候補の相性には「その基音・身体で鳴る代表音の周波数分布」を使う。現在休符中だから probe がゼロになる設計は採らない。
初版は実際の vitality や release gain を代表振幅へ混ぜず、代表励振・hold・観測区間を固定して形の相性を測る。
現在の放射量に対する総負荷、状態依存の音色、身体内部分音同士の粗さは別の拡張として残す。
代表条件は I11 の recipe から適用可能な部分を再利用し、4秒打切りや16分割を無条件に周波数側の仕様にしない。

候補基音ごとに Tone 側の部分音比・振幅・減衰と Nyquist 制約を適用し、絶対周波数で聴感補正を行う。
補正済みの固定 scan を単に横へずらすだけにはしない。帯域端で失った質量・支持率を保存し、範囲外の部分音を
端のビンへ clamp しない。音色・基音ごとの in-band 正規化が端への逃避を作るかは F2 の対照で検査する。

### 4.2 自己除去の正解参照

参照は自己 source を除いた PCM を解析する。混合音の `subjective_intensity` から単独音の同量を引かない。
ピーク抽出、power 圧縮、NSGT の干渉項があるため、その減算は一般に成立しない。

offline では、同じ順序で他 source だけを再合成する独立参照と、mix minus own PCM の経路を比較する。
f32 の加算・減算順序による差と、音源の消し違いを区別する。NSGT の窓・平滑化、SpectralFrontEnd の履歴も一致させる。
出生前に自己音が存在しない区間では、その時点の共有解析状態を複製できる。途中開始の場合は必要な履歴を replay し、
履歴不足を無音で埋めない。停止後も tail と解析状態が消えるまで同じ source を追跡する。

ここでの LOO は音源を除いた音響入力の評価であり、「その個体が過去から存在しなかった生態系」の再演ではない。
初版の habituation は同時点の共有履歴を維持し、自己除去した raw C に既存の erosion を適用する。
単独個体の除去でゼロになる検査は前処理密度と raw R/H を対象にし、履歴による状態まで消えたとは主張しない。

### 4.3 最初に接続する評価式

初版は既存 C の意味を保つ身体平均とする。候補基音を x、候補から作る subjective-intensity の
周波数密度を q_i(x,b)、ERB 幅を Δu_b とし、正規化した bin 質量を w_i(x,b) とする。
q は固定した代表観測区間の各 frame に前処理を適用してから時間平均する。平均 power を一度だけ圧縮する計算とは区別する。

```text
w_i(x,b) = q_i(x,b) Δu_b / Σ_b q_i(x,b) Δu_b
fitness_score_i(x) = Σ_b w_i(x,b) C_minus_i_eff(b)
fitness_level_i(x) = sigmoid(beta * (fitness_score_i(x) - theta))
```

これは既存 C の身体平均という作者規則であり、Sethares 不協和度そのものとは呼ばない。
総質量ゼロは unsupported とし、欠測を NaN やゼロ score で代用しない。有効な計算結果としての score 0 は保持する。
Log2Space 長の不整合は既存の hard assert、支持欠落・期限切れは明示状態として区別する。
score/level と支持・時刻を plain struct で返す小さい crate 内評価器を、実際に三消費者で共有する。
新しい汎用 trait、別 crate、UI 専用評価器は作らない。

旧F2で挙げた、raw R の身体積分 → 飽和写像 → H と合成する案は、別の目的関数の検討案として保存する。
今回の身体分布の契約定義と同時には進めない。再開する場合は、両案の相違が生じる反例と、
環境・身体の質量正規化、HR の集約順序を記録し、式・尺度・代謝閾値を別に改訂する。初版の評価器へ混ぜない。
現行の R は正規化環境密度と方向性 kernel を使うため、どちらの案も対称な実振幅積モデルとの厳密等価を前提にしない。

### 4.3a 参照方式と本番用身体表現の境界（2026-09-29）

上の式と全候補×72 frameのTone合成・NSGTは、固定した代表発音モデルの数値参照である。身体評価の意味を検算するため保存するが、本番で出生要求後に同じ全候補計算を終える要件にはしない。候補身体のRecipeが確定した後の密度計算は環境から独立する一方、Landscape依存modeではRecipeの生成自体が環境に依存する。この二段階を分ける。

次の一案は、実子のBodySnapshot/Recipeにある確定済みの部分音・mode情報から、その場で評価に用いる身体分布を作る方式。既存の静的射影を主観強度密度と同一視しない。位相・包絡・時間変化・解析器応答・聴感変換の扱い、単位・正規化・帯域支持、対応外条件を先に契約化する。新方式を採るなら、参照密度との違いだけでなくscore/level、候補選択、代謝へ接続する場合のenergy作用を取得前の別基準で判定する。旧参照の合格・不合格を新方式の判定へ流用しない。詳細と未決項目は[再出発記録](body-fitness-runtime-restart-20260929.md)に置く。

### 4.4 環境と評価方式は直交させる

環境側の自己除去と、候補側の点／身体平均を独立に比較できるようにする。
`LeaveSelfOutMode` 一つに両者を押し込まない。実験では「旧／正解参照の環境 × 点／身体評価」の四条件を使う。
旧 `ExactScan` は歴史的対照として基準版に保存し、新実装を同名で黙って置換しない。
公開面の改名・削除は F5 で扱い、alpha 方針に従って不要な互換 alias は追加しない。

共有 Landscape、spawn density、ListenerTwin の C を個体別 fitness で上書きしない。
`pred_*` と `perc_*` は由来の区別に維持する。`generator_model.rs` の共有 C 予測も身体 fitness と同一視せず、
予測を使う消費者がある箇所は F3/F4 で同じ身体評価を適用するか、共有地形の予測として残すかを明記する。

## 5. 実装単位と終了条件

| 単位 | 変更・成果物 | 主な配置 | 終了条件・依存 |
|---|---|---|---|
| F0 現在地と本番契約 | 版・結果・稼働の旧監査と、今回のQ1–Q6静的確認は完了として保存する。予定Spawn・予告なしSpawn・respawnの生成hop、初発音、翌hop観測、対象負荷と失敗時動作を分ける | 本書と[再出発記録](body-fitness-runtime-restart-20260929.md) | 許容遅延を旧30秒取得窓から流用しない。同hop生成を設計目標とし、同時出生数・全hop費用・CPU/RSS・未準備/失効の扱いを試作前に固定する |
| F1 自己除去と身体の参照 | 実Toneと自己除去PCMの参照、同音他者・route・tail・帯域端の既存有限結果を保存する | 既存の`life/sound/`、`schedule_renderer.rs`、`core/stream/analysis.rs`と参照記録 | 新方式のために参照取得を一律にやり直さない。影響する入力差だけを反例として追加する |
| F2 身体評価と本番用表現 | 4.3の式・四条件比較と72 frame参照を保存。実子の確定部分音・modeから身体分布を作る一案の式、単位、代表状態、省略情報、支持と誤差を別契約にする | 既存評価器・BodySnapshot/Recipe・小さい隔離試作 | まず数値取得前に仕事量・許容差・失敗状態を定義。候補集合・環境・占有・選択式を固定して身体分布だけを比較し、結果後の閾値緩和をしない。本番用の身体表現は、[音色・合成の実装計画](../superpowers/plans/2026-09-29-timbre-synthesis.md)のTask 2（身体モジュール契約）の自己モデル面として共同で定める。参照との事前登録した誤差比較は、同計画D6の照合規則にあたる（2026-09-29） |
| F3 移動への接続 | 固定pitchの持続・資源合格、可変Hzの低有効率という既存結果を保存。採用する同じ身体分布を移動へ使う範囲を定める | `pitch_core.rs`、`pitch_controller.rs`、`runtime/mod.rs` | 身体・音高・route変更、失効、欠測の理由と有効率を対象負荷で判定。旧worker配送合格を新方式の実時間合格としない。I11性能探索へ戻らない |
| F4 代謝・親選択・出生 | 既存の有限因果鎖と旧非同期代謝・出生結果を保存。新身体分布を使う場合、最新環境・占有・更新後親energyで同じ意味の選択を行う | `voice.rs`、`metabolism_policy.rs`、`community/frequency.rs`・`respawn.rs` | 予定Spawn・予告なしSpawnはaction処理hop、respawnは死亡確認のcleanup hopでVoice生成する設計目標。初発音と翌hop自己除去を別時刻で記録。冷cache、未準備、失効、連続出生、追加Hz、部分失敗を含めて期限・選択影響を判定する。[音色・合成の実装計画](../superpowers/plans/2026-09-29-timbre-synthesis.md)のPhase 2（音色側の受入）を合流する。同じ基音・位置・環境で身体だけを交換する対照、放射量の統制、無音・高域除去の対照、親選択確率までの因果経路をF2〜F4の登録条件に含める（2026-09-29） |
| F5 採用・操作・統合検証 | 必要最小のRhai/API・report/UIと実装済み文書を同期する | `scenario/`、`scripting/`、`viewdata.rs`、必要なUI、技術文書 | 全hop資源・実device・試聴・作者受入を別判定。旧36条件の数値合格と本番採用を区別し、既定変更は個別判断する |
| F6 音色遺伝と合成比較 | F4/F5の成立範囲後の独立研究工程として停止する | 親形質・子body構築、研究assay | 親形質が子身体を変えるときは親選択前の単一子表を前提にしない。再開時に遺伝と準備時点の契約を別に定める。正本は[音色・合成の実装計画](../superpowers/plans/2026-09-29-timbre-synthesis.md)のPhase 5。開始条件はF4/F5の成立範囲、身体モジュール契約の実装（同計画Phase 3）、全身評価の再受入。Phase 3の移行は本計画の評価器が使う代表描画の入口を保ち、時期を本計画と合わせる |

F1/F2 の数値実装は Rust を正本とし、小さい独立計算との照合を残す。時間計測、全負荷取得、長期生態実験は分ける。
F3 は実験接続であり、F4 未完の状態を「音色への選択が実装された」と呼ばない。
参照実験では既存の候補生成・spacing・占有制約を保ち、評価器の作用を切り分ける。本番の候補探索まで固定しない。探索変更が必要なら、身体評価で有利な候補を取り逃す影響を別に検証する。次の小試作では同じ候補集合を用いて身体分布だけを比較する。

### F3 の実時間方針

実音 LOO と72 frame身体密度は参照として保持する。本番用身体表現は別契約の一案を先に定義し、
候補ごとの実音合成・NSGT再解析を出生hopへ持ち込まない。環境側の自己除去ではsource、route、
body generation、解析epoch、支持終端を照合し、同じ有効な環境snapshotを一判断内で使う。
候補構築後に得た音を過去の判断へ使わず、決定的なoffline配送と通常runtimeの期限を区別する。

予定Spawnと予告なしSpawnはaction処理hop、respawnはcleanup hop内のVoice生成を設計目標とする。
最初の非ゼロ音、出生hopの自己PCM、翌hopの自己除去receiptは生成時刻から分けて記録する。
約10 hopの延期や旧30秒の取得窓を許容遅延へ転用しない。出生だけの局所処理時間でなく、分析・生態更新・
描画・配送を含むhop全体の残余、同時出生数、CPU/RSSを事前に定める。冷cache、環境・身体変更、
選択後の局所Hz、連続出生、保持量と準備CPUまで対象に含める。同hop生成という条件の下で、
出生処理に割り当てるCPU時間と対象負荷の数値上限はまだ確定していない。

通常対応を約束する入力での未準備・失効は同hopで計算する。対応外入力、容量超過、期限超過、batchの部分失敗は
状態とID/counter消費・再試行の扱いを先に規定して明示する。点評価への黙った切替、古い環境での抽選、
秒単位の延期を合格としない。継続的なF3/F4比較では欠測・明示fallbackの率と選択への影響を記録し、
特定音色で評価欠落が多い状態を生態的選択と取り違えない。自己除去を `subjective_intensity - self` に戻さない。

## 6. 検証と失敗時の判断

| 検査 | 固定する条件と反例 | 判定 |
|---|---|---|
| 自己除去 | 単独の Sine/Harmonic/Modal、同音他者、上部部分音だけの重複、detune、route、発音重なり、tail、出生・退役、解析窓の境界 | 自分だけなら raw 入力が消える。他者は残る。独立した他者-only 再合成との差を許容内にする |
| 身体と衝突 | 同じ f0 で上部部分音だけを衝突／非衝突へ移す。H 単独、R 単独、合成 C も分ける | 期待する部分音対の寄与を直接計算と照合する。合成 C の順位は H/R 競合を含めて説明する |
| 明るさと自己アンカー | 同一外部音・候補集合・移動費用・温度・乱数状態。固定の放射 RMS 対照と通常 gain の条件を分ける | score 誤差、候補順位、受理率、距離を測る。平坦ならその条件の実害を否定するだけで、旧 LOO の構造的誤りは取り消さない |
| 音色への選択 | 遺伝前の固定異種集団で、点／身体評価、移動／生存／全接続を比較 | 生存時間・エネルギー変化・出生先・有効評価率を比較。遺伝実装後も単一音色への収束を必須条件にしない |
| 非調和音程・音域 | H/R 除去、ratio candidate 有無、tessitura gravity、range、帯域端の支持、固定／移動包絡を区別 | 音程峰だけで H の改変を決めない。音域分化を欠いた結果も、試したモデル・条件の結論として残す |
| 時間構造との結合 | 既存 T1/T2 設定を固定した身体評価の介入と、時間側だけの介入を分ける | flux・meter・onset・arrival を追跡。アタックや帯域外成分を一律に不可視として固定しない |
| 本番用身体分布 | 実子のSine/Harmonic/ModalとLandscape依存mode、同じ候補・環境・占有・選択式。位相・包絡・帯域端・選択後Hzの反例を含む | 旧72 frame参照との差、支持、score/level、順位と選択を事前の別基準で判定。代謝へ接続する段階でenergy作用も判定する |
| 出生時刻と資源 | 予定・未予告Spawn、respawn、冷cache、環境・身体変更、連続出生、batch部分失敗。全hopの既存負荷を含む | 対象範囲で既存出生機会hop内のVoice生成とCPU/RSS上限を確認。初発音・翌hop観測は別記し、未準備等の明示失敗を成功へ数えない |

旧参照の数値許容と合否はその登録のまま保存する。新表現の意味・許容差、対照の予想、必要な分岐・有効評価率は新しい取得前に定め、結果取得後に合格するよう緩めない。旧time4のpostrun合格は保存入力の数値・資源判定であり、同hop出生や新表現の実時間成立を示さない。
一つの失敗を隠すために H テンプレート、励振、音色遺伝を同時追加しない。目的関数が失敗したら F2 へ戻り、
誤差・資源が失敗したら環境供給へ戻る。未実装の新合成器を現在の評価器の完了条件にしない。

Rustを変更した単位では、変更に対応する内部・境界・実消費者の検査に加え、リポジトリ必須の
`RUST_BACKTRACE=1 cargo test -- --nocapture` を標準出力・標準エラー込みで保存し、同じ shell invocation で終了コードを記録する。
format、lint、target 検査は変更範囲と repository 規則に従う。commit の依頼がある場合は必ず Clippy を通す。
変更も未解決の懸念もない状態で全検査や全負荷取得を反復しない。

## 7. 文書・公開境界と次の一単位

現在の technote は、mainの点評価と基音binだけを消す `ExactScan` を実装済み範囲として説明している。
今回の計画を public technote の実装済み機構として載せない。現況同期は現在の事実を示す文書作業、
F5 の同期は採用実装を説明する作業として区別する。生成された Rhai 参照は registry から再生成する。
試聴用 WAV は `conchordal-render` だけで作り、楽器の disk-write 経路は追加しない。

v1/v2と採点位置の単一介入診断は取得・保存済みであり、旧精度・費用の未達は変更しない。現在の次工程は、上の現行工程に示したSource5精度取得の判断、完全な一次結果に対する独立SavedReader、標準・pattern・最大容量の資源観察である。代表描画、元の解析モデルと許容差、全候補と選択後Hzの仕事量、対象負荷、未準備・部分失敗時の状態を保持する。本番方式の採用や通常runtimeへの接続は、その証拠と独立した同hop出生・通常負荷の条件がそろった後に判断する。生態系側の身体種別分岐や内部値読取を増やして解決しない。

その契約に未定義の重み・単位・上限がなく小さい隔離試作が可能になった時だけ、一案を試す。旧36条件の一括再取得、NSGT局所高速化、I11性能探索、音色遺伝へ自動で広げない。

### 2026-09-26 21時の進捗

F1/F2は実PitchCoreの10,240判断とglide・route・tailまで参照検査を拡張し、独立検算を通した。
F3aは最大4 sourceの解析状態を実装し、通常48 kHz・実Toneで独立した他者-only解析との一致を確認した。
F3の全体完了ではなく、次は代表身体計算の費用調査と有界workerの配送契約である。

I4は隔離版の `bounded_reference` producerと実消費経路が動作し、memory=Noneの独立した12条件回帰も
旧版と一致した。その後、全検査とstage2 source capsuleの保存を完了した。I11-2も隔離版でCDFの実消費とON/OFF差を確認したが、
作者による既定採用・A4範囲の変更は行っていない。

ALIFEの20 seed × 8条件比較は完了し、継承とランダム出生の差は点評価・身体評価の双方で残った。
両評価方式の差の差の95%区間は0を含み、効果の増強・等価性は主張しない。結果・方法・旧集計の
終端時刻との違いを論文案件の影響メモへ保存した。

合成方式の順位を支える六つの断定は[一次資料点検](body-fitness-synthesis-review-20260926.md)を参照する。
mode lockingの条件、亜臨界Hopfの安定枝、位相と時間経路、音域と文化的調律の論証を区別し、
身体評価の改修だけから新しい合成器の不要性を結論しない。

### 2026-09-26 21時台後半の実装待ち行列

I11 §5.7の正式費用取得は基準版06a4772の専有窓で終了した。A/A 3 passとbody／noneの計384本は
すべてexit 0。改訂登録の `population_us`・`synthesis_us`・`own_us` の組合せは14/16合格で、
harmonic-flow-16のreport有りpopulation中央値とreport無しsynthesis p99の2組は不合格。
`own_us` は16/16合格したが、§5.7全欄合格やI11-1第1段の技術完了とはしない。
生の `elapsed_us`、返却遅延・欠測源とA/A floorは
[I11-1の現版結果](../roadmap/temporal-dcc/i11-onset-comparison.md) と [F0監査](body-aware-fitness-f0-audit.md)に保存した。
I11 §5.9の欠落条件は同基準版で再取得し、旧集計と一致した。

この間にF3bの有界配送、F3cの通常runtime観測、代表密度だけを計算する経路、候補基音の事前準備を
隔離版へ静的実装した。これらは以前のF3a全検査の対象外であり、動作済みとは扱わない。
専有解除後、数値一致・queueと時計・実Voice接続の順に検査する。代表密度の新旧費用比較と、
持続励振した身体による自己除去費用は、それぞれ取得前条件を別文書へ固定した。

I4・I11-2・身体評価の結合は[隔離統合の検査順](body-fitness-integration-validation-20260926.md)に従う。
共有ファイルの機械的統合が衝突なく済んでも、結合動作や資源の合格には数えない。

続いて[F3dの一判断接続](body-aware-fitness-f3d-decision-registration.md)を取得前登録し、隔離版へ静的実装した。
先に実Tone由来の候補密度を準備し、後から受理した一つの自己除去環境で身体score表を作る。
実Voiceのpitch gateでRNG状態・target・周波数座標・候補全集合を照合してから採点表を消費する。
実Voiceの目標変化とcommit、候補欠落など四つの拒否例を検査する構成。専有取得時点では未実行で、
取得解除後の検査結果を別記録で確認する。
一般の非同期予約・失効処理とF4の代謝・出生は、この一判断の試験後に残る。

### 2026-09-26 22時台の検査結果

F3bの実worker・queue・時計、密度専用経路と旧経路の全bin bit比較、F3cの実Voice観測、
候補の事前準備、F3dの実Voice一判断を通過した。F3dは68候補の身体score表を81回読み、
440 Hzから466.163635 Hzへ目標と身体周波数が変化した。旧Cを書き換えた対照も同じ結果と乱数状態になった。
候補欠落、乱数・target・周波数座標の不一致は、実乱数を追加消費する前に拒否した。
全cargo testは22:44:09 JSTにexit 0、formatと標準Clippyも通過。all-targets Clippyは
既存テスト箇所17件が残り、新規箇所の警告はない。19 sourceと全ログは
`.worktrees/target-body-fitness/f3d-source-capsule-20260926/` に保存した。

I11-2の診断理由追加版は全テストと9条件の旧音声・主要記録・成功CDF・共通候補比較を通過した。
自己群識別では、アクセント頻度の標準化に使う分散ゼロが多数のprototype未割当を生むと確認した。
これは各descriptorの再計算であり、判断が消費したsnapshotの完全再演ではない。
割当閾値や参照prototypeを結果に合わせて変更せず、別のモデル改訂として扱う。

統合v3は3入力・49ファイルを同期し、全体検査へ進んだ。F3b〜dは明示offline配送と限定条件での接続であり、
一般の非同期予約・失効、有効評価率、実時間費用、F4の代謝・出生の完了を意味しない。

### 2026-09-26 23時台の次段階

代表密度の専用経路は旧full解析とbit一致を保ち、一候補72 hopの中央値を約42 msから12 ms台へ短縮した。
持続励振した4 sourceの自己除去Processorは1 hopあたり中央値2.960 ms、p95 3.388 msだった。
これは候補準備全体や通常runtimeの総費用ではない。[結果と取得範囲](body-aware-fitness-results-20260926.md)を参照する。

統合版の25条件では旧成果との22条件比較と両ONの2条件を通過したが、両ON Modalの記憶参照0により
全体は不合格だった。新規lint3件を直した第四版は全1228テストと標準lintを通過し、影響4条件も第三版と一致した。
この構文修正でModalの不達を解消したとはしない。[統合記録](body-fitness-integration-validation-20260926.md)に
失敗と、初回lintのcache再利用による検査漏れの訂正を残した。

次は[F3eの消費前照合](body-aware-fitness-f3e-preparation-draft.md)、
[F4aの成人一更新](body-aware-fitness-f4a-metabolism-registration.md)、Modal記憶参照の全hop診断を
隔離して進める。F4aは代謝入力の接続検査であり、出生・親選択や長期の音色選択を含まない。
出生側は[F4bの一機会案](body-aware-fitness-f4b-birth-draft.md)を作成した。
まずRandom respawnの候補評価から実生成までを対象とし、子のid・身体と採点時の条件、
出生拒否時のcounter消費、死亡個体の残響を含む共有環境の対応を確認する。これは取得前の案である。

F3e第一段階は実Voiceのgate内に消費前照合を接続し、14種類の単独失効と4組の拒否優先順、
seed 1／4・1／4 sourceの計10 Voiceで身体表の消費を確認した。
全suiteは1205成功・0失敗・40 ignore。詳細と保証境界は
[第一段階の結果](body-aware-fitness-f3e-phase1-results-20260926.md)を参照する。
次の配送段階では計算中・待機・完成済みの保持量、未完時の旧判断、commit後の再予約を検査する。

Modalの記憶欠落は全hop診断で、追跡群が96 hopの保存窓に届かない仕組みを確認した。
最大90 hopで退役し、保存時の鮮度・無観測拒否に達する前に終わっていた。
診断追加の3素材比較は既存音声・記録と一致したが、元の不合格は維持する。
上流の群退役理由の調査と、身体評価の実装は別々に継続する。

### 2026-09-27 0時台の統合状況

統合第五版にF3e第一段階、F4aの成人代謝・実発声費用、記憶診断第二段階を取り込み、
1232テスト・標準Clippy・全target checkを通過した。通常releaseの影響4条件も第四版の
音声と登録対象の行動記録に一致した。[統合記録](body-fitness-integration-validation-20260926.md)を参照する。
F4aは試験専用の限定接続であり、通常runtimeの既定動作を変更していない。

Modalの退役205件は追跡容量による交代だった。固定96 hopに届かない群を退役時に短い区間として
保存する案を、別の試験専用モデルとして検証中。元の不合格と既定モデルを変更しない。
F3eは配送順と消費時環境の照合、F4bは実際の子の生成と出生採点の一致へ進んだ。
出生試験の初回設定には音源ID衝突、release前の音をtailとする誤記、共有地形の再計算欠落があり、
初回結果を無効として保存した。修正版の取得と、その修正の根拠を別に記録する。

F3e第二段階v2は、代表密度の準備と受理済み環境による採点を分離し、完成済み要求の置換も修正して、
全1209テストと標準Clippyを通過した。[修正版の結果](body-aware-fitness-f3e-phase2-v2-results-20260927.md)に
レビュー前版と修正内容を残した。これはまだ第五版には含まれない。
次は固定した仮想完了遅延を用い、連続gateで準備結果が捨てられ続ける条件を調べる。
queue保持量が有界でも、身体評価の有効率が正になるとは限らない。

退役時に短い区間を保存する18条件の探索では、Modal全3 seedで参照在庫と学習creditが正になった。
音声・参加判断は全9組で一致した。取得前のOFF全report一致条件は非同期配送の差を含めて不成立のまま残し、
既定採用や元の25条件の合格へ読み替えない。[探索結果と事後診断](body-fitness-partial-memory-results-20260927.md)を参照する。

第六版ではF3e第二段階v2、[F4b Random出生](body-aware-fitness-f4b-results-20260927.md)、
試験専用の部分区間保存を統合し、全1240テストと通常releaseの影響4条件を通過した。
F4bの代表recipeと実発声条件の違いは[独立レビュー](body-aware-fitness-f4b-source-review-20260927.md)に反映した。

[配送第三段階](body-aware-fitness-f3e-phase3-results-20260927.md)は、仮想完了遅延とgate周期・初期位相で
採用機会が変わることを確認した。休止後の限定回復ではfresh環境・現在habituation版の表を実Voiceが読んだ。
これは実時間性能や通常runtimeの有効評価率の取得ではない。第六版にはまだ含めない。
次の[F4c](body-aware-fitness-f4c-hereditary-registration.md)は、成人二体の実energy更新から
既存の親抽選とHereditary候補比較、子の系譜へつなぐ一機会検査である。

第七版へ配送第三段階も取り込み、全1242テストを通過した。通常releaseのbinaryは第六版とSHAが一致した。
Hereditary一機会の試験は統合せず、全体未達として記録する。身体score/levelが親へ届いても、
Sustain templateのendurance/rechargeが未設定で代謝係数がゼロならenergyは変わらない。
次回は非ゼロ代謝係数とenergy変化の検査を取得前に明示する。今回のseedや係数を結果後に調整して合格へ変更しない。

### 今回の上限付き実行の保存状態

2026-09-27 00:32 JST時点で7日使用率79%を確認し、指定上限80%以内で新規実装を終えた。
開始済みの検査と証拠保存を閉じ、commit・pushはしていない。

- 統合の保存版は `.worktrees/integration-fitness` の第七版。1242テストと標準Clippyが通過し、
  通常releaseは第六版と同一binaryである。mainの通常動作には採用していない。
- [F4cの不合格結果](body-aware-fitness-f4c-results-20260927.md)とsourceは別capsuleに保存した。
  親energyの不変に加え、登録した失効負例群と試験専用の点C対照は未取得である。
  通常Hereditary経路の旧版比較はWAV・非timing2319記録が一致したが、代謝作用の証明とは別である。
- 論文メモと20 seed比較は `~/wrk/conc-paper-2026/notes/2026-09-26-body-aware-fitness-impact.md` に保存済み。
  遺伝群とランダム群の差は両評価方式で維持した。効果の増大・等価性・音色進化は示していない。

再開時の優先項目は、F4cの非ゼロ代謝係数を明示した新登録、候補準備の連続失効を避ける設計と
実時間の有効評価率、出生の残る入口である。通常採用の前にはA4対象版と影響する取得条件を改訂する。
I11資源の2条件未達、元のModal記憶参照不合格、部分区間保存の厳格OFF比較不合格を、
今回の単体検査や診断で解消したとは扱わない。

### 2026-09-27 11時台の再開: F4c v2と統合第八版

[新登録](body-aware-fitness-f4c-v2-registration-20260927.md)でendurance 10秒、recovery 1秒、dissonance penalty 1を取得前に固定した。親energyの実変化と数式参照、更新後energyによる親抽選、身体による子候補採点、点C対照、六種類の失効拒否を通過した。固定seedの選択親は変わらず、親重みの正規化確率が変わったことを区別して記録した。

隔離版1205テストに続き、統合第八版1243テスト・標準Clippy・全target checkを通過した。通常release binaryは前版と異なったため固定4条件を再取得し、WAVと登録対象記録の一致を確認した。source・検査・取得の対応は[統合記録](body-fitness-integration-validation-20260926.md)の第八版へ集約する。mainの通常動作への採用、commit、pushは行っていない。

次の単位は候補準備の連続失効を避ける設計と実時間の有効評価率、出生の残る入口である。今回のHereditaryは固定身体の一機会であり、音色遺伝や長期選択へは拡張しない。I11資源の2条件、元のModal記憶参照、部分区間保存の厳格OFF比較の未達は維持する。

### 2026-09-27 11時台後半: 配送第四段階、settle出生、通常観測

[F3e第四段階](body-aware-fitness-f3e-phase4-results-20260927.md)では、候補密度の準備中は提案だけを延期し、Voiceの時間進行を維持する試験政策を実装した。独立レビューで延期時間の二重計上を発見し、全経過時間を次の提案に一度だけ反映して蓄積を0へ戻した。直接表との一致と失効時のfallbackを確認した。

専有枠のrelease live-paced取得は1/4 sourceを各938 hop、各一回実行した。固定Sine・440 Hzの各sourceで窓内5回の身体評価消費があり、deadline missとF3b失効は0だった。消費頻度は各sourceで約0.5回/秒であり、全hopの判断に身体評価が届いた結果ではない。通常runtimeの非同期行動配線、変動身体、実device全体へは一般化しない。

[F4d](body-aware-fitness-f4d-results-20260927.md)ではRandomとHereditaryのsettle補助候補を身体score/levelで選び、実子条件と独立選択・RNGの一致、最低levelによる拒否を確認した。候補生成自体は点地形依存である。PeakBiased、初回spawn、長期生態と音色遺伝は未接続。

[通常runtime観測](body-fitness-runtime-observation-results-20260927.md)は明示ONの48 kHz/512 sample・最大4 sourceに限定した。通常renderのON/OFF音声一致、容量超過拒否、途中出生と退役のidentityを検査した。最後の受理値と今hopの欠測を分けて報告し、queue飽和・出力脱落・解析設定変更では停止する。最初のhopのworker起動費用と実deviceの資源受入は未測定。

統合第九版の全suiteは1251成功・0失敗・44 ignore。fmt、標準Clippy、全target check、release buildと既定OFFの固定4条件回帰を通過した。source、ログ、生データの対応は[統合記録](body-fitness-integration-validation-20260926.md)に保存した。mainの通常動作への採用、commit、pushは行っていない。次は延期中の可変pitch・身体・routeの失効規則と、測定した低い判断頻度を踏まえた候補密度準備費用の削減を先に扱い、その後に通常runtimeの行動入力を接続する。

### 2026-09-27 12時台: 有界密度cache、glide照合、全bin出生評価

[F3e5の登録](body-aware-fitness-f3e5-cache-registration-20260927.md)と[結果](body-aware-fitness-f3e5-cache-results-20260927.md)に示すとおり、同じ身体・candidate pitch・解析構成の代表密度だけを再利用するcacheを追加した。各sourceのentryは256件かつ1 MiB以下で、環境scoreとhabituationは保存しない。初回132 miss、同じ候補の再要求は132 hit、RNGだけ進めた要求は129 hitと3 missだった。source・Recipe・世代・解析構成・epoch・観測窓の失効と、実際のLRU除去・byte上限を検査した。

release専有取得は1/4 sourceを各938 hop、各一回実行した。実身体評価消費は1 sourceで196回、4 sourceで194/193/195/194回。前回の各5回から増え、全条件でdeadline missとF3b失効は0だった。cache entryの最大確保量は727,040 byte/source。解析templateやReady表を含むprocess全体のメモリ上限ではない。固定Sine 440 Hz・代表発音条件の結果であり、通常runtime全体や変動身体には外挿しない。

[glideの別登録](body-fitness-glide-reuse-registration-20260927.md)では、候補密度がcurrent pitchへ依存しない一方、移動費用は現gateで再計算されることを独立監査した。local/non-ratio候補で過剰なcurrent pitch一致条件を取り除き、8 hopの実glide後も直接再準備と判断・RNGが一致した。身体・Recipe・source・route・control・target・RNG・epoch・space・habituationの照合は保持した。実unison更新とbrightness更新の拒否も維持する。候補密度の数値検査と制御fixtureによる検査を[結果](body-fitness-glide-reuse-results-20260927.md)で区別した。

[F4e](body-aware-fitness-f4e-results-20260927.md)は、parent不在のPeakBiasedで380–520 Hzの全44 binを身体評価してから11 peakを抽出し、非ゼロ半径の局所探索32点も実子の身体で評価した。独立抽選・RNG、生成子identity、低高閾値、重複防止、四種類のstale拒否を通過した。この例では中心440 Hzが局所最大で、局所探索による周波数移動は起きていない。

統合第十版は1258成功・0失敗・45 ignore、標準Clippy等と通常release固定4条件回帰を通過した。次は有界準備と現gate採点を通常runtimeの明示ON行動入力へ接続する。初回Field spawnの全range身体評価、parentありPeakBiased、global/ratio候補、可変身体の非同期追随は残る。既定変更・長期選択・音色遺伝・作者採用は別の未完条件である。mainの通常実装への採用、commit、pushは行っていない。

### 2026-09-27: 通常runtimeの明示ON行動入力（第十一版）

[登録](body-fitness-runtime-action-registration-20260927.md)と[結果](body-fitness-runtime-action-results-20260927.md)に従い、通常配線に持続worker、source別有界cache、現在環境での再採点、strictな一回消費を接続した。待機・拒否ではpitch提案だけを保留し、発音・寿命・時間更新を継続する。同じhopの別substepで旧point評価へ戻らない。既定OFFとAir-Gapを維持した。

最初の全suiteでRecipe fixtureの5件が失敗し、実Voiceから代表Recipeを構成する形へ修正した。失敗ログを保持したうえで、最終全suiteは1263成功・0失敗・45 ignore。通常renderの最後の定期記録frame 48では1声で10回、4声で各10回の消費を確認した。cacheは各1 MiB以内、観測経過時間は許容4800 sample以内。ON二回の再現性とOFF互換性を確認したが、短いSine条件ではON/OFF WAVも同一だった。

次は非Sine/可変身体での通常runtime判断、実時間の非同期配線全体の費用と追随、初回Field spawnおよびparentありPeakBiasedを扱う。第十版の固定Sine専有実測を第十一版の実時間合格に読み替えない。mainへの採用、commit、pushは行っていない。

### 2026-09-27: 非Sine移動、初回出生、通常runtime実時間取得（第十二版b）

[統合検証](body-fitness-ecology-validation-20260927.md)の隔離版は1271成功・0失敗・46 ignore。通常renderの非Sine身体・control変更、試験専用の初回Field出生5件、実時間のSine/Drone 1声/4声・ON/OFF 4条件を検証した。Drone位相の毎hop失効、at配置による音高Lock、Seqの1秒退役の失敗記録を保持し、それぞれの改訂後の結果を分離した。

次は非Sine/身体変更の非同期実時間追随と長時間の資源・延期率、その後に残る出生入口と通常runtimeの代謝・出生配線を扱う。今回の消費回数は定期report時点、実時間合格は10秒・模擬出力先に限定する。mainへの採用、commit、pushは行っていない。


### 2026-09-27: 通常offline代謝、出生残余、非Sine実時間未達（第十三版b）

[統合記録](body-fitness-metabolism-validation-20260927.md)の隔離版は1285成功・0失敗・47 ignore。通常renderの身体代謝はseed 7の固定条件でframe 48 energyがOFF 0.78194094、ON 0.7596431となり、各二回の決定性を確認した。この条件のWAVはON/OFF同一。現在基音、score/level、実onset、gate分離、cacheと出所/時刻の拒否を検査した。初回Field出生の残余11件、parentありPeakBiasedの実親選択から子出生までを試験専用経路で確認した。

非Sine・身体/control変更の専有10秒取得はOFF合格・ON不合格。全窓2声、underflow 0、hop予算超過0でも、身体変更後の新世代消費はframe 522となり、登録期限frame 432を超えた。旧世代pendingが終わるまで新世代を準備しない配線と、新世代132候補×72 hopの冷計算が残った。後の回復46/40消費を期限内合格へ読み替えない。最初のsuite後のClippy指摘を修正し、全suiteを再実行した。失敗した実時間取得はsource/binary/rawを固定した。

次は旧計算の取消、submit/完了/破棄時刻の診断と冷計算費用を別登録で扱う。通常出生の次案は、offline専用評価器をWorkerStateから出生機会だけCommunityへ渡し、約64 hopの環境形成後にField子1声を実出生させる単位。代謝同時接続には新生児の初回共有環境と次hopの自己除去環境の遷移、respawn後の無音self PCM slot準備が必要。現版の途中出生拒否はそれらの検証まで維持する。mainへの採用、commit、pushは行っていない。


### 2026-09-27: 旧準備の取消、通常offline出生、冷計算の未達（第十四版）

[統合検証](body-fitness-recovery-validation-20260927.md)は1299成功・0失敗・48 ignore、標準Clippy等を通過した。旧版を独立にコンパイルした48 kHz/512の12条件×72 frame比較で、PCMと代表解析のscan/massがbyte一致。通常offline初回Field出生は独立flagで接続し、候補中心と実jitter後の別Tone・直接積分を照合した。既定spacingにより有効候補が一つだけになった初回試験の失敗を残し、spacing 0の別登録で選択作用を検査した。このsceneでは旧評価とも出生位置・音声が同じであり、位置の差は実証していない。

[実時間取得](body-fitness-recovery-results-20260927.md)はOFF合格・ON不合格。取消serial 14はframe 300→301で応答し、旧待ちを解消した。新身体serial 15はframe 302境界に投入、135候補の冷計算に1,905,687 µsを要し、消費frame 480で期限432に届かなかった。全窓2声、underflow 0、hop予算超過0、cache/age上限は維持。後の55/28回消費や前版より早い回復を期限内成功へ読み替えない。

次は候補順・RNG・72 hop・数値結果を維持する冷計算短縮を、処理内訳と有界並行準備の資源上限から別登録する。通常出生と代謝の同時接続、respawnと新生児の自己除去観測遷移、非同期代謝、長期生態、音色遺伝は残る。第十四版は保存単位であり、全体完了ではない。mainの通常実装への採用、commit、pushは行っていない。


### 2026-09-27: 冷候補の有界並行準備と期限内復帰（第十五版b）

[統合検証](body-fitness-cold-parallel-validation-20260927.md)は1303成功・0失敗・48 ignore。冷cache・重複なしの2〜256候補だけを、利用可能CPU数8以上で2laneに分ける。補助threadはsourceごとに最大1、4 sourceで計算thread最大8。元の順序でcacheを確定し、取消時に並行部分結果を残さない。全候補bit、現環境score、実Voice判断・RNGを同版の逐次参照と照合した。

[専有取得](body-fitness-cold-parallel-results-20260927.md)の非Sine ON/OFF、Sine 1/4 source ON/OFFは全6条件成功。非Sineはframe 432で新世代消費12回（旧世代分を含む累計）を確認し、control変更後も回復した。frame 912の消費はHarmonic 62回、Modal 55回。初回冷jobの2lane計算は約0.98/0.99秒。ただし身体変更後の最初の冷jobは定期記録で上書きされ、初回消費の範囲だけを確認した。候補数・非同期軌道の異なる旧取得との速度倍率は主張しない。

全条件で938 hop、出力不足0、hop予算超過0。4 source ONの最大hop時間は2.348105 ms。cache上限1 MiB/sourceは維持したが、解析template・stage結果・出力等を含むprocess全体のmemory上限ではない。初回compileと通常lib import属性の不備は失敗ログを残して修正した。第十三版b・第十四版の未達はその版の結果として保持する。

次は通常offline出生と代謝の結合、自己除去観測への遷移を独立登録する。全体完了、実device受入、main採用、commit、pushは行っていない。


### 2026-09-27: 通常offlineの出生と代謝の結合（第十六版d）

[統合検証](body-fitness-birth-metabolism-validation-20260927.md)は1311成功・0失敗・48 ignore。出生前の全source集合とobserver batchを完全一致で検査し、dispatch後は既存集合に実子一声が加わったことを確認する。出生hopでは既存声は自己除去、子だけ主解析の共有環境を読み、翌hopは両者自己除去へ移る。出生時と次hopのidentity・出所・支持時刻・energyをそれぞれ一回報告する。通常OFFの毎hop allocationは増やしていない。

[登録sceneの結果](body-fitness-birth-metabolism-results-20260927.md)では、frame 65の出生時に両環境のsupportとdecisionが33280 samplesで一致し、frame 66は2source・support/decision 33792 samplesを確認した。44binから選ばれた子の最終基音は440.7801818847656 Hz。frame 96のenergy差は0.00114297で事前閾値1e-6を超えた。各mode内のWAV・対象reportは再現し、mode間のWAVも同じだった。

最初の試験引数型の不備、ゼロPCM試験の時計進行漏れ、通常RhaiがSpawnへ付けるcrowding初期設定の誤拒否、Recipe hashの配列型の試験誤認を、失敗source/log/rawを残して修正した。seed・scene・係数・比較閾値は変更していない。新生児の無音自己PCMは有効な観測入力として受理し、欠落と区別する。

次は通常offlineの親ありrespawn。cleanupはそのhopの代謝・phonation収集後なので、子の初回代謝を出生hopに捏造せず、自己PCM slotを準備して翌hopのSourceRemovedへつなぐ必要がある。固定一機会の数値登録と実装はこれから行う。F4/F5/F6全体、実device・長時間・作者受入の完了ではない。main採用、commit、pushは行っていない。


### 2026-09-27: 通常offlineの親付きrespawnと翌hop自己除去（第十七版f）

[統合検証](body-fitness-offline-respawn-validation-20260927.md)は1322成功・0失敗・48 ignore。第十六版dの410 sourceを照合して引き継ぎ、最終414 sourceを固定した。親poolはphonation・lifecycle後の実energyを使い、予定子のmetadataと身体Recipeから全bin・peak・局所候補・最終Hzを評価して、実子と照合する。出生hopは子の代謝receiptを生成せず、描画した512 samplesのゼロ自己PCMを確認し、翌hopのSourceRemovedへ接続した。

[取得結果](body-fitness-offline-respawn-results-20260927.md)では最初の死亡frame143/Voice3、選択親Voice1、子Voice4を確認。ON親energyは0.30982375、OFFは0.34690475。ON子448.337738 Hz、OFF子448.98578 Hz。各mode内の二回はWAVと対象recordが一致した。子の翌hopはframe144/support73728、energy0.99550247、Finishはframe150で二度目の予測死亡frame154より前だった。高閾値拒否ではspawn_counterだけ進み、子ID/memberを消費しない。OFF/ONは代謝と出生評価の結合比較である。

通常render、記録済peak候補の独立RNG再選択、別環境fixtureの直接Tone積分、production境界への局所fault注入の証拠を区別した。Clippy指摘、entry数の試験誤認、非PeakBiasedの未取得RNG診断、Randomのparent_idに対する回帰試験の誤期待は、失敗source/logを保持して修正した。seed・scene・数値閾値は維持した。

次は[二度目の単独respawn](body-fitness-repeated-respawn-draft-20260927.md)。通常action/observationとの同時接続、異種身体の比較、音色遺伝、同ID再利用、長期生態、現版の実時間性能・実device・作者受入は未完了。main採用、commit、pushは行っていない。


### 2026-09-27: 二度の親付きrespawnと第一子の親参加（第十八版b）

[統合検証](body-fitness-repeated-respawn-validation-20260927.md)は1327成功・0失敗・48 ignore。永久的な一回限りの状態を機会連番、死亡sourceの完全identity、決定sample、spawn sequenceへ変更し、前機会の候補と親poolを次へ持ち越さない。初期founder以外の出生sampleと退役sourceも保持する。実装に二回固定の上限は加えない。

[固定sceneの結果](body-fitness-repeated-respawn-results-20260927.md)は死亡frame143/ID3と154/ID2、親pool1/2→1/4、実子4→5を記録した。ONの第二親は第一子4、更新後energy0.95054436。第二子は系譜generation2、基音400.000763 Hzで、出生hopはreceiptなし・512ゼロPCM、翌frame155の自己除去receiptでenergy0.99524164へ更新した。Finishはframe162/sample82944。各mode二回のWAV・対象record一致、全binからの独立peak抽出、親抽選と局所選択の再計算を照合した。

v18初回はrender4回exit0だがcleanup後source reportのbirth_sample欠落でintegration失敗。v18bで報告とruntime照合に出生sampleを追加し、条件を緩めず再検査した。失敗版source/log/binary/rawを保持する。一般の同hop重なり、未注入fault、異種身体と全消費者の比較は未完了であり、F4完了とは呼ばない。main採用、commit、pushは行っていない。

### 2026-09-27 19時台: 異種founderと三消費者の通常offline比較

[三消費者結果](../../.worktrees/body-fitness-three-consumer/docs/design-notes/body-fitness-three-consumer-results-20260927.md)を固定した。同一PopulationのHarmonic／Modal／Harmonicと固定Sine子について、移動・代謝・出生の点／身体8条件を各2回取得した。全founderと子の実PitchModeはFreeで、登録22hopの間に同hop重複を含む2出生を確認した。初版と不要な通常SpawnのVecを除いた修正版の全16本でWAV・決定記録が一致した。計時入りの身体観測も、処理時間以外の全フィールドが一致する。

各軸を単独で変えた4組すべてで、移動はframe15にVoice3の実Hzが896.065735／876.593689、代謝はframe1にVoice1のenergyが0.990424633／0.990422726へ分岐した。出生はframe19に同じ更新後親pool・選択親・候補Hzを使い、子Hzが463.478485／462.141846となった。局所出生scoreの最大値は独立再計算と一致する。親energyは抽選重みを変えたが、今回の選択親は変わらなかった。

最終源版は422ファイルと固定binaryを保存し、全1336テスト成功・0失敗・48 ignore、19:40:49 JST exit0。fmt、標準Clippy、all-targets check、英日mdbook buildも成功した。主担当はsource・binary・16入力・32 raw hash、初版との比較、全test集計とlog/status複写一致を独立確認した。v1 binary現物はbuild先の上書きで残っておらず、取得時hash照合の記録とsource/rawだけが保存されている。v2 binaryは専用pathへ固定済み。

次は同じ固定sceneと8条件で、確率移動の補正score・乱数drawと代謝全substepの入力をreport-onlyで記録し、独立再計算を補う。現在のreportにない全環境scan・RNG状態・substep値を推定で補わない。長期生存・生態的選択、身体とf0／放射量の交絡、資源・実device・作者受入、F5/F6は残る。main採用、commit、pushは行っていない。

### 2026-09-27: 三消費者traceの独立検算と有限コホートへの継続

[trace v2の結果](../../.worktrees/body-fitness-three-consumer/docs/design-notes/body-fitness-three-consumer-trace-v2-results-20260927.md)を固定した。同じ16取得について、移動176判断、代謝8272 substep、親pool 64 memberを独立再計算し、検算対象の数値誤差は0だった。身体候補のraw積分はbest/baselineを独立検算し、その他の身体候補では保存されたprepared rawを条件入力としている。適応状態全体、完全なRNG再構築、未発生のtieとphonation onsetを通常取得で実証したとは扱わない。

計測追加時に生じたNaN時のMetropolis draw省略とtie drawのf64からf32への型変化を修正し、局所反例で検査した。最終424 source、固定binary、122-entry取得索引を保存し、全1341テスト成功・0失敗・48 ignore（20:53:43 JST exit0）、fmt、標準Clippy、all-targets checkを確認した。16 WAVは計測前の版とbyte一致し、trace v1/v2の2976 recordも計時項目を除いて一致する。失敗版source・raw・test logは保持した。

次の単位は出生なし有限コホートである。隔離した `.worktrees/body-fitness-cohort` は固定trace v2の424ファイルをhash照合して開始する。異種2 founder、移動／代謝の4条件、身体への初期f0割当交換、通常振幅と孤立固定音RMS校正の対照を事前登録する。energy 0、retrigger停止、Idle、Voice cleanup、renderer源別tail終了、observer退役を別々に記録し、Finishで未観測のendpointは右打切りとする。source 0、同hop複数退役、pending action退役、tail帰属の局所検査を先行する。これは長期選択・出生結合・F5/F6の代替ではなく、登録と実装が進行中の次工程である。


### 2026-09-27: 出生なし有限コホートv2の確定

[有限コホート結果](../../.worktrees/body-fitness-cohort/docs/design-notes/body-fitness-cohort-results-20260927.md)を固定した。別PopulationのHarmonic／Modal founderに対し、移動・代謝の点／身体4条件、初期f0割当2通り、通常振幅／孤立固定音RMS減衰を各a/bで取得した。全32本exit0、16セルのWAV・決定的report反復一致、16組の単一軸比較を確認した。全64 source観測で枯渇・retrigger停止・Idle・Voice cleanup・observer退役・renderer tail完了をFinish前に観測し、右打切りは0だった。主比較300hopとFinish後の追加WAV描画は分離した。

代謝106,740 lifecycle eventを独立f32再演し、energy stageの最大差0 ULP。実scan・densityからのscore/mass再計算は登録窓24,420 eventに限定した。onsetは0件。生存変化の向きは身体・初期f0・振幅対照で変わり、身体評価の一方向の利益や音色固有の生存優位は未立証である。孤立RMS校正は主実験の混合・移動音声の音圧等量性を示さない。

校正Tone IDと既存test helperを取り違えたv1を不合格として保持し、修正版は実batchからmetadataを記録して別取得した。最終全Rust suiteは1347成功・0失敗・48 ignore、fmt・標準Clippy・all-targets check成功。継承元factorialの8 WAV・既存4064 recordも一致した。主担当が最終1583項索引の全サイズ・hashを照合した。索引SHA-256は `17d6c9aa4cf3bf7c567c4bbce588ae25c844af25b74f9c224f3b445b980ca875`。

この有限コホート単位は完了だが、親選択・出生先評価・継承との同時接続、長期の音色選択、実時間性能・実device・作者受入、F5/F6は残る。main採用、commit、pushは行っていない。


### 2026-09-28: 実枯渇から親選択・固定子出生への有限64本を確定

[selection結果](../../.worktrees/body-fitness-selection/docs/design-notes/body-fitness-selection-results-20260927.md)は8採点条件×初期f0交換×通常／孤立RMSの32セル各a/b、全64本exit0である。移動・代謝の各16対応対で実Hzまたはenergyへの作用を確認した。代謝による親確率差は出生basisの重複を除く固有8対すべてで成立したが、実親choice差は0。共通入力を照合した出生basisの16直接対は実子Hz差10、同じHz6だった。triggerの通常代謝枯渇から一出生・翌hop自己除去までを確認し、両親と子の生存時間は全取得で右打切りとした。

source431項、全1354テスト成功・0失敗・48 ignore、fmt／標準Clippy／all-targets check合格。[1000項の証拠索引](../../.worktrees/body-fitness-selection/target/body-fitness-selection-validation/validation-index.json)のSHA-256は `4ee4964e223402f217edd508336fc0d2e2b9f11692908b3976eb745f45d41237`で、主担当が全現物のサイズ・hashを照合した。比較は有限offline・親の音色を継承しない固定子に限る。次は有界非同期配送での実Hz identity・欠測fallback・有効率と、現統合版actionの持続稼働・全資源を優先する。通常採用、実device、作者受入、F6継承は未実施のまま維持する。


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

### 2026-09-28: 協調workerの単一出生、NSGT連続index群の不採用

両隔離版の必須検査が完了した。協調workerは全suite 1400成功・失敗0・ignored54、NSGT連続index群は1399成功・失敗0・ignored54。各版の保存10 PCM・720frameも再取得と独立検算を通過し、不一致0・未解決0だった。

協調workerの固定I-W4-on-a（30秒、3個体、各690候補）では、ticket1の全690候補とexact-Hz追加1候補の計691行がoracleと一致し、実出生1件・自己PCM1件・次hop受領1件を得た。実出生はsample1004032、20.917333秒。全表のworker computeは19.848551秒、追加1候補は28.824msで、追加要求から実出生まで3072sampleだった。Finish時に残り2表は未Readyであり、3出生/30秒の受入れは依然不合格。単ON資源は通過したが、36case・OFF対照との相対資源と親energy因果経路は未確認である。結果SHA `4b5ff5da00cba3734fffa968b648d02fdd89bf89497057a56c3747ceb87c3ea8`。

NSGT連続index群は同じ690候補を旧/新/新/旧で測定し、全Recipe・density binary/JSONは一致したが、whole-process CPU比が1.388133/1.397176へ悪化した。事前基準≤0.95を満たさず、不採用とする。境界検査が少なくなるという実装上の変化を性能改善の根拠にしない。退行原因は未確定。結果SHA `e10d6efdec99f39a22f437931a4dd3de15a438ff62800832c1bb7b95ad27d0e1`。

次の主課題は全690×72frameの計算費用である。協調workerは追加候補のqueue待ちを解消する候補として保持するが、まだmainへ採用しない。NSGT連続index群と旧u32版の退行を保存し、同じ意味・積和順・身体依存性を保つ別の費用対策を、固定外側測定で検証する。詳細と封印先は `parallel-continuation-20260927.md` の最新記録を参照する。

上の「次の主課題」は2026-09-28時点の原文を保存した履歴であり、[現行工程](#現行工程2026-10-02)と9月29日の[再出発記録](body-fitness-runtime-restart-20260929.md)に置換済み。全表費用の追加最適化には進まない。
