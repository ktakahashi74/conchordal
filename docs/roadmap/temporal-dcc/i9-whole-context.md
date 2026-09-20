# I9: 長期の聴取応答（研究拡張）と旧全曲文脈の検査記録

2026-09-13の著者判断により、本体の必須工程から研究拡張へ移管し、現方式の自動継続を終了した。
状態は保留・再設定待ち。I8の技術完了を維持し、I9完了、O16合格、R/A受入には変更しない。
工程は[milestones.md §1.1](milestones.md#11-採用スコープと認知機構の選定2026-09-13)、
第一原理の採否は[technote-ledger §9.3.55](../../design-notes/technote-ledger.ja.md#i9-scope-decision)に従う。

## 研究拡張としての再設定条件

対象となる聴取応答と認知神経機構、抽象化する状態・更新、生成側／ListenerTwinの利用先、
具体的な音への作用と比較課題を先に定める。現graph・DTW・保持則・whole headは工学的候補であり、
部品の動作や人の評定への適合だけで神経機構との対応を確立したとはしない。
全曲構成はScenarioの作者操作に属し、Twinには実際に聞こえた証拠だけを渡す。
全曲記憶、人間の包括的な複製、全曲完了感の推定を楽器の必須機能にしない。

有限作品の評定を選ぶ場合は聴取区間と終端条件を定める。終端を持たない継続運転を選ぶ場合は、
有界な状態で保持する影響と資源・鮮度・復旧の検査を定め、予定曲長やEOFを必要としない。
旧30分素材・O16は特定の長期能力の比較条件として残し、全用途への要件にはしない。
本体I4/I8が使う記憶の正しさ・損失・資源検査は移管後も必要である。

以下は移管前の経過と未達条件を保存した記録である。「次」「実行中」「終了条件」は各記録時点の意味を持ち、
現在の再開指示にはしない。保存したRust・数値参照・capture・失敗結果を削除せず、旧runの完走を新しい認知的採用へ転用しない。


## 接続した診断

`temporal_whole`を明示した場合に、phraseとsectionの同じ観測終端から独立したordinal headを評価する。
4 scalar、4 missing indicator、切片を含む9係数、4 cutpoint、5カテゴリprior、独立temperatureを持つ。
入力支持率4値の算術平均だけをheadの支持率に使い、supported distributionへのtemperature適用後にpriorへbackoffする。
全欠測ならtemperatureに関係なくpriorそのものを返す。係数未適合の診断であり、生成・停止制御へ入力しない。

phraseの終止・継続headには、priorを混ぜる前のsupported expected ratingを追加した。
sectionのreturn-given-exitは固定音響重みで集約し、支持率を別に保持する。
第3入力のretrieved continuationのremaining-support fractionは、stage-1割当を伴う因果的producerが未接続のため欠測のまま。
時間経過だけで期待が満たされた扱いにしない。

最新のobserver packet versionは17、phrase snapshotは5、Recall modelは5。UIとreportへ入力値・支持率・元観測終端を渡す。
EOFは最後に取り込んだ音響から得たwhole snapshotを保持し、追加無音・EOF特徴・memory reinforcementを使わない。
epoch変更は古いsnapshotを破棄する。

## 30分の実音probeと全保存参照

`observation/replay.rs::thirty_minute_acoustic_capacity_probe`は、実波形を48 kHz、hop 512、nfft 2048の
RtNSGT、通常observer、phrase、section、wholeへ順に渡す。1,800秒、168,750 hopを省略せず処理する。
bus 1の単独probeであり、登録済み演奏・両busの運用負荷・聴取受入を代用しない。

8秒の素材Aを開始0、144、804、1,740秒で再出現させ、間に周波数・変調周期の異なる音と休止を置く。
区間指定は波形生成と検査queryの選択だけに使い、推論へ渡さない。参照queryは各帰還の開始2秒後以降、最初にdispatchされたものを固定する。
結果を見て都合のよいqueryへ差し替えない。

初回256件bankの証拠:

- `target/temporal-dcc/i9-probe-1800-256/result.json`: 4,489新規episode、4,233退避、21,083発行query、20,998受領query。
- `target/temporal-dcc/i9-captured-1800/`: 同じ音声の再実行と、全4,489 episodeの元descriptor・元終端・利用可能時刻・provenanceを保存。
  元ongoing creditは612件、同一creditの異なる終端は最大30件。帰還窓のdispatch queryは171／250／250件。
- PCM word checksumは両実行とも`508683abd063f28c`。これはf32 word列の再現性検査であり、暗号学的hashではない。
- capture JSONLは116,595,376 bytes、SHA-256 `9bbf0f98d84ecd737373f9b5a6796f4bbf282f7c020c0745d8897ad8a94fef68`。
  ソースhash一覧、capture集計、10秒ごとのprofile、最終snapshotを同じディレクトリへ保存した。

`memory/reference.rs`のoffline参照は、有限archiveの全episode、全knot開始位置、登録した4つ以下の変形refinement、
full-width DTWを調べる。query時点で未利用・別epoch・重複する元支持は除く。初期Aの対象は元開始8秒未満、元終端11秒以下。
新たな生成・再生音を参照descriptorへ混ぜない。全探索はquery条件付きの有限候補空間についての参照であり、任意の連続変形の大域最適性ではない。

| 固定query終端（秒） | 利用可能episode数 | cost≤1の初期A参照episode数 | 現行256件bankに残存 | 全保存bank＋従来候補16件で到達 |
|---|---:|---:|---:|---|
| 146.048 | 431 | 41 | 0 | 0。候補境界同点により曖昧 |
| 806.005333 | 2,004 | 36 | 0 | 0。候補選別による脱落 |
| 1,742.005333 | 4,324 | 41 | 0 | episode 30へ到達 |

`roomy-reference.json`に全保存参照と256件bankの差、`full-bank-reference.json`に保存損失を除いた従来検索の結果を保存した。
これらの実行testが終了コード0でも、容量・検索の合格を意味しない。現在の結果は不合格を示す。

## 容量拡張の実装と再検査

8,192 episodeまで明示設定できるようにし、受領controllerとcoarse query cacheも同じ容量で確保する。
256件を越えた最初のreplayは、100秒付近でcache側の固定上限に当たり失敗した。
`target/temporal-dcc/i9-capacity-8192/`はこの失敗時点の記録であり、完成結果ではない。

cacheのcell・validity・approximation領域を事前確保し、pack・受領時のコピーでは同じ領域を再利用する。
slotの範囲、generation、重複、original query時刻、replay conflictの検査を保持する。
8,192件目のslotと古いgenerationを混同しないこと、繰返しpack・cache置換で格納先pointerが変わらないことを検査する。

これは以前のM0固定native layoutを変更する。64-bit環境でsnapshot本体136 bytes、heap領域は
`24*N + 16*ceil(N/64)` bytes、合計は256件で6,344 bytes、8,192件で198,792 bytes。
以前の6,272-byte固定recordを現在のnativeサイズとして引用しない。queryごとのcoarse vector、controller、descriptor、
provenanceなどの領域とcopy trafficはこの値に含めない。8,192は検査用の設定上限であり、リアルタイム運用の採用容量ではない。

修正後の実replayは`target/temporal-dcc/i9-capacity-8192-v2/`に分離する。
容量を増やしても既知の候補16件の損失は解消していないため、再実行の完走だけではI9を完了にしない。

修正後は1,800秒、168,750 hopを完走。4,424 episodeを保持し退避0、queryは発行・受領とも21,062件。
PCM word checksumは256件bank時と同じ。wall timeは271.331秒だが、他の検査と同時実行しており、
単独実時間負荷の合格指標には使わない。bank変更はその後の推論・query・commitmentへ影響するため、
256件時のqueryを新bankの実queryとして転用せず、新しいcaptureでも全保存参照を実行した。

| 新captureの固定query終端（秒） | 全保存参照が支持する初期A episode数 | 実際の検索で到達した初期A episode |
|---|---:|---|
| 146.048 | 41 | 28, 29, 31, 32, 33, 34, 35, 36, 9 |
| 806.005333 | 41 | なし |
| 1,742.005333 | 41 | 25, 21, 20, 18, 19, 17 |

新captureの`full-bank-reference.json`に結果を保存した。中央queryでは初期Aの最上位coarse候補が
0始まり849位となり、16件に届かない。開始位置間隔を4から2／1 knotへ細かくするoffline比較でも
最上位は1,014／1,076位であり、単なるanchor密度増加では解決しない。これらはcoarse候補順位の比較であり、
新しい検索方式の採用・DTW到達・帰還合格を意味しない。次の実装単位は、短い先頭照合による候補選別を
全queryの対応と比較し、元の支持・時刻・候補上限・未解決質量を維持したindex修正を検証すること。

### この時点の全体検査

Rust 982成功、失敗0、30ignore。`cargo test exit=0 @ 2026-09-13T17:26:45+09:00`。
通常Clippy、全target check、`git diff --check`は通過した。全test出力は`test_report.txt`、
同一shellの終了コードは`test_status.txt`に保存した。30分probeとoffline全探索はignore対象を明示実行したもので、
通常testの成功数には加算しない。snapshot、source hash、各検査ログへの参照は
`target/temporal-dcc/i9-checkpoint-20260913/verification.json`へ集約する。commitは作成していない。

## Query全体を使う索引への修正

先頭8点だけのcoarse順位と、query全体を使う順位を同じ保存済みqueryで比較した。
全観測knotの時間対応を使うと、旧索引で849位だった初期Aが最上位へ上がった。
8／16点の観測支持時間quantileを使う比較でも3帰還すべてで最上位となった。
この結果に基づき、runtimeでは16点を採用する。半数8点の比較結果は
`i9-capacity-8192-v2/quantile-query-index-reference.json`へ保持する。

`memory/index.rs`はqueryの有効な観測支持時間を16等分し、各区間の中点を含む元knotを使う。
anchorは以前と同じ4 knot間隔、各episode最大32箇所。元の先頭8点から推定したtransposition／tempoを使って
queryの点を元のreference時刻へ写し、共通する有効座標だけを線形補間する。gapや欠測knotを跨いで補間しない。
対応する支持がない点には1の索引損失を与え、支持時間で平均する。この値は候補選別用であり、
照合の支持・矛盾・変形判定へ新しい証拠として渡さない。最終DTW、cost閾値、1/64 refinement、band 16は維持する。
変形が未同定でも有効な音響座標の索引照合は許す。未同定変形はmaskを保ち、変形なしの確定根拠にしない。

候補16件の境界で同点群が分断される場合、その群全体を保持対象から外す。
残る候補は16件未満になり得る。失われた候補数と同点群の大きさは、query単位・累計の診断へ明示する。
同点群の一部をID順で採用したり、候補境界から離れた対応まで曖昧扱いにしたりしない。
これは候補の非保持とその計数を定義するもので、未実装のstage-1 availability／joint assignmentを
確率として補ったことにはしない。元のMR1先頭照合は`ordered_prefix`というtest側の比較controlへ分離した。
元の固定fixtureはこのcontrolで照合し、新しいruntime索引には別の回帰検査を置いた。

新索引＋保存済みqueryでは、3帰還とも初期Aへ到達した。
`i9-capacity-8192-v2/runtime-index-reference.json`の到達episodeは、順に
`[24,26,27,22,23,28,29,31,32,33,34,35,36]`、`[25,30]`、`[25,21,20,18,19,17,30]`。
いずれも同点境界群5件を非保持とした。同じ先頭を持つ16個の妨害候補の後ろに正しい後続を置く別fixtureでも、
旧索引の脱落、新索引の到達、bank順序への不変性を検査した。

masked値へ0を掛けてNaNを混入させる経路、1 knotだけのqueryでtempoを捏造しない境界も検査した。
全体検査は一度、短い実音queryを過剰に除外する回帰を検出した。その除外条件を修正し、
`render_binary::temporal_observation_preserves_audio_and_finishes_both_buses`を再通過した。
observer packetは14、Recall modelは4。UIとreportに候補脱落の累計を追加した。

両busの新しい実音replayは`i9-query-wide-bus0/1`へ保存する。この最初の実行は短query回帰の修正前のbinaryである。
採用候補となる修正後の実行は別の`i9-query-wide-v2-bus0/1`へ保存し、各binaryのsource hashを固定した。
最終captureの同じqueryに対する全保存参照が終わるまで、実replayでの帰還合格とはしない。

最初の新索引版は両busとも30分を完走し、5,841 episode、退避0となった。これを短query修正後の証拠には転用しない。
修正後の全体検査はRust 985成功、失敗0、30ignore。
`cargo test exit=0 @ 2026-09-13T17:50:15+09:00`、通常Clippy・全target check・diff check通過。
最新のsource hash、検査ログ、進行中replayのハンドルは
`target/temporal-dcc/i9-index-checkpoint-20260913/verification.json`に保存する。
進行中の同じハンドルを再確認し、観測timeoutだけを理由にreplayを再起動しない。

## 修正後の実音replayと同一query参照

短query修正後の`i9-query-wide-v2-bus0/1`は両busとも正常終了した。各168,750 frame、30分、
5,841 episode、退避0。音声word checksumは両方`508683abd063f28c`。
各`runtime-index-reference.json`の全保存・全anchor参照も終了し、事前に固定した帰還queryで
次の到達を確認した。件数は別々のepisode候補であり、独立した音楽素材の数ではない。

| query終端（秒） | 参照が支持した初期A候補 | 実際のbounded検索で到達（両bus共通） |
|---|---:|---|
| 146.048 | 38 | 27, 29, 30, 21, 22, 31, 32, 33, 34, 35, 36, 37, 38 |
| 806.005333 | 33 | 23, 24, 25, 26, 28 |
| 1742.005333 | 38 | 23, 24, 25, 26, 28 |

wall timeはbus 0が635.201秒、bus 1が635.764秒。同時実行・他検査の影響を含むため、
専有実行の実時間性能とは扱わない。この結果は合成probeの保存・索引の到達検査であり、
後から追加したgraphのedge保持や登録development素材の合格へ転用しない。

## 対応graphと長い未確定phraseの所有

通常Recallに、保存episodeをnodeとする対応候補graphを接続した。sourceは元のgroup/generation、
occurrence/ongoing credit、支持区間、seal時刻、支持重み、元のcontext所属を保持する。
edgeは元query ID、実際に照合した部分区間、受領時刻、過去episode/generation、cost、変形と曖昧性を保持する。
queryの部分区間がsourceに含まれるときだけ結び、短いqueryの成功をsourceの未照合部分へ拡張しない。
retirementはnodeと入出辺を除去し、epoch/generationの再使用へ支持を移さない。
これらは未校正の対応候補であり、stage-1の割当確率や確定した帰還ではない。

最初のgraphは未確定queryを2秒で捨てたため、30秒phraseの初期の対応をseal前に失った。
`a_long_source_keeps_early_correspondence_until_its_endpoint_is_sealed`で修正前の失敗を再現した。
修正後は直近256 queryの記録に加え、進行中prefixと0.5秒の確定待ち終端が対応を保持する。
所有の上限は7群×15 prefix＋512 pending endpoint＋更新前に届く1 completionの618件。
同じgroup/credit/startの履歴をまとめ、source所有がなくなるまで16対象の対応候補を保持する。
より長い支持の同一対象へ更新しても、元query時刻・支持はそのqueryの値のまま保つ。
短い終端への正確な帰属には直近query記録も使用する。容量超過・edge枝刈り・所有退役を別々に報告する。
直近記録の期限切れ件数は、長い所有側にも残る対応の喪失件数ではない。

whole snapshotは最後の音響時点のgraph診断を保持し、EOF後のtransport flushで書き換えない。
graphは既存phrase/sectionや音声生成へフィードバックしない。observer packetは15。
30秒の実音smokeは84 node・824対応候補、owner容量超過0、音声checksum
`b3f709ef2fb357ad`、EOF前後のwhole snapshot一致を確認した。
graph専有の事前確保量は8,192 episode設定で20,164,328 bytes。
これはgraphのstruct/Vec領域の値であり、allocator overheadやdescriptor、matcher、snapshotのコピーを含まない。
node側24回、未確定prefix側371回のedge枝刈りが発生した。16辺の容量採用はまだ検証済みではない。

修正後の30分replayを`target/temporal-dcc/i9-owner-bus0/1`へ保存し、実行前のsource hashを固定した。
このreplayとedge圧力を確認してから次の容量判断へ進む。第3入力と比較controlは引き続き未接続。

この両bus replayも正常終了した。各5,841 node・86,828対応候補、episode退避0、所有容量超過0。
wall timeは633.620／633.316秒。保存episode・query・実照合結果のcaptureは、上記の新索引版と
busごとに全バイト一致した。`reference-reuse.json`にcaptureと元の全探索参照のhashを保存し、
同一入力・同一bounded出力の3帰還到達を維持したと確認する。graph辺の受入へは転用しない。
node側79,656回、prefix側163,477回のedge枝刈りが発生した。これらは枝刈り操作の回数であり、
失った一意な帰還数ではない。10秒間隔の観測最大値は12 prefix・185 prefix辺・59 pending query。
毎hopの最大値とは扱わない。5,841確定終端は612の元ongoing sourceに由来し、最大36終端/source。
完全なstrength/interference lifecycleのreinforcement回数は未供給なので、0でなく未実装として記録した。

## 固定した研究用scenarioと実ファイルreplay

既存登録素材の指定がないため、技術検証用に`samples/research/temporal_long_returns.rhai`を追加した。
seedは2026091309、30秒cellを60個、最初の12秒motifを150／810／1740秒で再提示する。
直前のmotif終端からの介在は138／648／918秒。harmonic Voiceの周波数移動、二声の重なり、
releaseと休止を含み、専用の同期Voiceは置かない。構成ラベルは音響observerへ渡さない。
条件は`i9-development-material.json`に保持し、人間T7の検証素材や作者採用とは区別する。

compile-onlyでFinish=1800秒、493 event・681 action・1 marker、sampleのcompile/seed検査を通過。
`conchordal-render`の実出力はmono PCM16、48kHz、86,400,512 sample（1800秒＋1 hop）。
WAV hashは`4c4256020f648304c3ae65cf70f814e340e822c0e575b235fa0665e3a19e5a42`。
初出と3帰還の各12秒に非ゼロの音を確認した。実renderは既定のnfft=16384。
自動生成TOMLのnfft=2048はコメントであり設定変更にはなっていなかったため、入力を変更せず、
同じcompiled AppConfigで解決した全設定を`effective-render.toml`として別途保存した。
生成前に固定した入力・source hashと、解決後の設定・実音の証拠は
`target/temporal-dcc/i9-development-material/verification.json`から参照する。

ignored replayはこの実WAVを逐次読み、既存のRtNSGT／observerへ通せるようにした。
`CONCHORDAL_I9_WAV`指定時は固定material情報を必須とし、帰還選択と参照対象の範囲を
captureの`material.json`へ保存する。通常推論へは音声だけを渡す。
全体実行ではrenderer終端の1 hopも読み、EOF向けの音声を追加しない。
30秒と165秒のsmokeが通過した。165秒runは最初の帰還後164秒時点のgraph全1,079 nodeを
header／node／complete付きで保存した。以後も各帰還終端＋2秒でgraph全nodeを保存する。
最初の固定queryではbounded検索から初期素材への支持を確認できず、全探索参照による切り分けが必要。
都合のよいqueryへの差し替えは行わない。

実ファイルの全168,751 frame、両bus replayを`target/temporal-dcc/i9-development-bus0/1`で実行中。
run-inputsとsource hashを実行前に固定した。完了後は同じcaptureの全探索参照を実行し、
保存・索引・graphの損失を区別する。smokeや最初のsynthetic probeをこの新素材の全曲合格としない。

## 比較controlの接続完了

`temporal_whole.controls`で3系統の独立した未校正headを明示できる。設定・数値処理・通常observer・
UI/reportへ接続し、全曲headや生成へフィードバックしない。各headは固有の係数、cutpoint、prior、
temperatureを持つ。`feature-manifest.json`の`i9_whole_completion_controls`に入力定義を登録した。

| control設定 | scalarとmissing indicator | 切片を含む係数 |
|---|---|---:|
| `closure_only` | prior／temperature前のsupported closure expected rating、欠測1値 | 3 |
| `gap_energy_2s` | 直近2秒の既知低energy比、元hop RMSの物理時間平均、端点log2-RMS傾き、欠測3値 | 7 |
| `elapsed_only` | epoch開始から現在の音響観測終端までの秒数、欠測1値 | 3 |

gap/energyはepochと元group取得開始で窓をclipする。低energy判定は既存phraseと同じ
`2^raw_log2_RMS <= 0.01 * temporal_gesture.rms_reference`。
平均は元group energyの平方根を物理支持時間で加重し、平均log-RMSや窓energyの平方根へ置き換えない。
各座標は90%以上の物理支持を要求する。傾きは実際の窓の両端hopが支持されるときだけ、
clip後hop中心時刻の差で元log2-RMSの差を割る。欠測端点を都合のよい内側のhopで置き換えない。

群ごとの入力からまず5カテゴリ分布を計算し、既存の音響重みと支持された経路質量で混合する。
入力を平均してから非線形headへ入れない。temperatureは支持された混合分布へだけ適用し、
未支持質量は未変換の各control priorへ渡す。group別の入力・coverage・座標・重み・未変換分布も公開する。
経過時間controlは観測clockだけを使い、取得欠落で時計を縮めない。全controlは最後のwhole snapshotに入り、
EOF後に再計算されない。これらの比較器が実装されたことと、人間T7評定に対する適合・比較合格は別である。

物理支持のclip、端点欠測、既知無音、2群の非線形混合、temperature／prior、設定の形と往復を検査した。
実音NSGTのepoch／EOF検査と、両bus・診断off/on・report有無のWAV不変性も通過した。
新素材30秒replayでも入力・control分布を確認し、元の音声checksumと213 episodeを維持した。
最終検証記録は`target/temporal-dcc/i9-controls-checkpoint-20260913/verification.json`。

## development素材で確認した検索損失とgraphの変形選択

全replayの完走前に、既に保存された同一queryを固定して参照比較を進めた。
`CONCHORDAL_I9_REFERENCE_PROBE=0/1/2`は登録済み3 queryのうち指定した一つだけを評価し、
結果を`runtime-index-reference-probe-N.json`へ保存する。既定の全3件検査とは別の成果物であり、
単独queryの成功を全曲合格へ置き換えない。第2帰還のcaptureは、実行中ファイルの先頭から
固定queryまでを完全な行で保存し、prefix hashと元ファイルを同じディレクトリへ記録した。

| 固定query | 同一query参照が支持する初期素材 | 現行16候補で到達 | 256候補 | 512候補 | 1,024候補 |
|---|---:|---:|---:|---:|---:|
| 6991、152.032秒、最初の帰還 | 59 | 0 | 5 | 37 | 59 |
| 36687、812.010667秒、第2帰還、両bus一致 | 48 | 0 | 0 | 5 | 5 |

この2 queryでは参照支持対象が全て元bankに残っていた。したがって到達0は退避ではなく検索損失である。
16点索引で最初の初期素材はそれぞれ161位、335位。32／64／128候補でも到達しなかった。
第2帰還の512候補は3,125,158 DP cell、1,024候補は6,116,662 cellを使用した。
並行処理中の一query観測はそれぞれ約70ms、106msであり、全負荷の実時間保証ではない。
候補数だけを増やすこの反実仮想は、同じquery、全保存bank、元の16幅DTWと最大4変形を使う。
runtimeの候補数はまだ16のまま。transport／section／graphの上限も含む採用と再検証が必要である。
第3帰還と30分完走の結果はまだこの比較に含まれない。

証拠は`target/temporal-dcc/i9-development-first-return/`および
`i9-development-prefix-reference-bus0/1/`の単独query参照に保存した。
全replayは25分地点で10,085 episode、1,893退避に達した。8,192件はこの素材の全保存容量を満たさない。

graphでは、同じepisodeの複数変形から最小costだけを選ぶと、DTW帯域端に触れた曖昧な候補が
別の有効な変形候補を隠す欠落を回帰テストで再現した。非曖昧な変形を先に比較し、
その中で最小costを保存する。全候補が曖昧なら曖昧のまま保持し、queryの元支持は延ばさない。
保存済み第1帰還窓には該当するquery／episode対が19件、queryとして9件あった。
`graph-refinement-selection-review.json`は行選択の反実仮想であり、graph全体の再実行ではない。
修正前の失敗と修正後の成功を記録し、後続queryによる上書き・edge容量損失は別の残件として保持する。

この単位の最終検査はRust 994成功・失敗0・30ignore、通常Clippy・全target check・差分空白検査通過。
終了コードは2026-09-13 19:35:10 JSTに保存した。ログ、反実仮想の入力・出力参照とsource hashは
`target/temporal-dcc/i9-search-loss-checkpoint-20260913/verification.json`に保持する。
候補上限の改訂では、`memory::CANDIDATES`だけでなくtransportの64 relation、Recallの16×4 cache、
sectionの64 score／16 identity scratch、graphの16 edgeを同時に確認する必要がある。
未測定の後段切捨てを隠した候補数増加では終了条件を満たさない。

## 全曲baselineの完走と候補容量の実接続

`i9-development-bus0/1`の旧8,192件／16候補runは両方終了コード0で完走した。
WAVの168,751 hop、1,800.010667秒を全て読み、checksumはともに`c09ab0f267d6eb25`。
11,888 episodeを保存し3,696件が退避した。並行実行の壁時計は2,995.117秒／2,992.933秒。
同じcaptureの全3 query参照も両busで終了コード0となり、値が一致した。
第3 queryはID 77600、1,742.026667秒、参照支持の初期素材59件は全てbankから退避していた。
全保存bankに対する同一query検索では32候補から5件へ到達した。最初の2帰還は検索、第3帰還は
保存容量による損失と区別できる。これはO16不合格のbaselineであり、旧runを採用結果へ変更しない。

この結果を受け、`temporal_memory.candidates`を任意の明示設定として接続した。
未指定は凍結した16候補baseline、指定範囲は1〜1,024。episode上限は16,384へ拡張した。
元DTWの帯域16と最大4変形、比較用memory modelの候補数は変更しない。
transportは全relationを所有sliceとして運び、最後のrelationまで検証した後だけticketを消費する。
Recallの群別・pending cacheは設定候補数×4を事前確保してswapする。
sectionのscore・対応・identity bufferも同じ上限で再利用し、best/gapの集計は異なる2 episodeの最大値だけを
保持する。局所sectionのbeam 16は別の上限であり、その枝刈り質量をunknownに残す契約は維持する。

graphのnode／未確定ownerも設定候補数の辺を保持する。snapshotにある16辺は表示previewであり、
保存上限とは分けた。queryとownerの辺bufferはpoolへ戻して再利用し、node退役時も回収する。
元queryの重複除外は事前確保したkey scratchを整列して検索する。全recent queryの全edgeを、
各owner edgeごとに走査する増幅を避ける。`allocated_bytes`には辺buffer、poolのpointer容量と
key scratchを含める。test用graph captureは全node edgeに加え未確定prefix ownerも出力する。

80件のtransport全体検証と末尾の世代不一致による原子的拒否、32候補を通常Recall経由で受領する検査、
previewを越える80辺の保存と末尾targetの退役、query bufferの反復再利用を確認した。
1,024候補／16,384件bankの165秒WAV replayも終了コード0となった。
最初の固定query 6991で、同一query参照が支持する初期素材59件全てに到達した。
壁時計117.706秒、1,073 episode。検索結果が下位の診断に入るため、旧165秒runの1,085件とは異なる。
そのため旧runのquery以後の状態を、新設定の結果として流用しない。
このsmokeはprefix ownerの全出力を追加する前に開始しており、その全出力の証拠には使わない。

最終検査はRust 997成功・失敗0・30ignore、通常Clippy・全target check通過。
`cargo test exit=0 @ 2026-09-13T19:52:32+09:00`。
記録は`target/temporal-dcc/i9-capacity-propagation-20260913/verification.json`へ保存する。
全体検査の最初の失敗は、設定上限の旧8193不正値を残したテストによるものだった。
不正値を新上限外の16385へ更新し、候補0／1025の拒否も確認して、全体検査を再実行した。

採用候補を16,384件／512候補／512辺として、実入力と全Rust source hashを開始前に固定し、
`i9-development-wide512-bus0/1`で両busの全replayを開始した。source manifest SHA256は
`706b5fb46b04651ff4d28647d01a5e92df8e3425b4244432445042df3844836f`。
元WAVを全て読み、完走後にそのrun自身の固定queryと全保存参照、graph owner／edgeを照合する。
設定の接続完了を、容量採用やI9完了とはしない。

新512候補runの最初のcheckpointでは、固定query 6991のoccurrence 18794に対応するsealed nodeも
未確定ownerも残っていなかった。`first-query-source-review.json`に記録した。
これは元phrase候補の支持・確定・退役を調べる入口であり、辺の容量損失と即断しない。
検索到達から全曲graph保持までを同一視せず、元sourceの所有と残った支持を次に照合する。

## 取得時計と対応証拠の保持（2026-09-13 20:33）

通常observerにbus共通の取得時計を接続した。整数sample区間の取得unionを数え、無音とNSGT warmupも
取得済みとして扱う。同epochの配送欠落はmissing秒へ一度だけ加算し、source epoch変更時は時計を新設する。
EOFは時計を進めない。128行・hop 512ではrecord領域12,288 bytes、structとscratchを含む所有領域12,480 bytes。
古いmask退避時のprefix区間は累積missingと物理時間から上下限を出す。履歴退避を新たな欠落音へ置き換えない。
Python GapClockと独立sample unionに対する4条件・229操作・916 prefix照合、重複・矛盾入力の原子的拒否、
実observerの欠落／epoch／EOFを検査した。30秒WAV smokeの212 episodeは旧512候補runの同じ入力prefixと完全一致した。
時計はstrength／interference更新の基盤であり、それ自体では保持モデルの接続完了ではない。

同一targetへのgraph対応は、非曖昧・低costを先に選び、同品質なら長い観測区間を保持するよう修正した。
以前の「query終端が新しければ置換」は、途中まで一致した証拠を後続の不一致や帯域端候補で失っていた。
nodeと未確定ownerで同じ順序を用い、辺容量超過時も曖昧な低cost候補を有効な対応より優先しない。
保持するedgeは元query・元の照合区間を保ち、source全体が一致したという主張へ延長しない。
これは対応候補の保存方針であり、後続入力が一致するという予測ではない。

入力・source hashを開始前に固定し、16,384件／512候補で165秒WAV replayを完走した。
実入力164.992秒・1,081 episode・壁時計120.88秒。旧512候補runの同じprefixと全episode captureが一致し、
164秒checkpointの1,075 source recordも全て一致した。取得時計とgraph保持規則の変更で、これらの保存元は変化しなかった。
同じquery 6991に対する全探索参照は初期素材59件を支持し、bankに59件、512候補検索から37件へ到達した。
この最初の帰還の結果を、後続2帰還や全体容量の合格へ転用しない。

checkpointで既存の工学的条件（非曖昧・cost≤1）を満たすsource-target対は255,235から296,827へ増えた。
新規保持44,144対、失った対2,552。最初の帰還区間から初期素材への対応は4,644から4,780対となり、
対応を持つsourceは両方92件で脱落なし。ただし188対を新たに保持し52対を失っており、全ての辺の保存は保証しない。
`graph-quality-comparison.json`に同一source比較と差分を保存した。工学的対応条件は校正済みの認識精度ではない。

固定queryのsource所有を通常経路のtest captureで追跡した。query 6991・credit 18794は152.064秒時点でongoing、
152.2986667秒で消失し、その間も以後も対応する確定待ち・確定endpointはなかった。groupは154.3146667秒まで残った。
このsourceのgraph node欠如は、観測した範囲では未確定の所有消失であり、辺容量による損失とは分類しない。
24件の状態変化を`source-bus1-epoch0.jsonl`へ保存した。支持を失ったsourceを合格目的で新規episodeに作り直さない。

observer packet version 19、Recall model version 5。全Rustテスト1,001成功・失敗0・30ignore、
通常Clippy・全target check通過。`cargo test exit=0 @ 2026-09-13T20:33:38+09:00`。
検証記録は`target/temporal-dcc/i9-graph-quality-20260913/verification.json`、実音・全探索・graph比較は
`target/temporal-dcc/i9-graph-quality-first-return/`。
source manifest SHA256は`c257cc4abf4f0d96b98c07a3916a657992ebc31836809a1d76624b7b98b281f7`。
先に開始した両busの30分runはversion 17のまま継続中であり、この修正の全曲実行結果とは扱わない。

## 保持モデルのRust数値準備と確定順序（2026-09-13 20:54）

`recall/retention.rs`へ既存Python EpisodeRetention／InterferenceWindowの数値契約を移した。
このmoduleはまだ`cfg(test)`であり、通常Recallのstage-1割当・保持更新・退避には接続していない。
単体の完成をI9の接続完了へ読み替えない。生成済みfixtureは`tests/fixtures/temporal_cognition/retention.json`、
生成元は`scripts/generate_temporal_retention_fixtures.py`。

実装した範囲は、元観測時刻の一回限りの確定更新、正の支持による明示admission、上限付きstrengthと
上限を掛けないmembership、再出現後の競合分だけの干渉、欠測coarse costによる干渉区間、
取得時計の上下限を使ったavailability、upper log availabilityによる退避候補、固定した事前記憶の認識確率である。
同一writeの再送・不正順序・矛盾支持・時計に先行する照会・容量不足を検査した。
識別番号・割当・coarse receiptは一回分の事前確保bufferに正規形で保存し、digestの衝突に再送判定を依存させない。
割当／coarse行は一意なhandle順で上限付きとする。元queryとの適合を保証するstage-1 producerは後続の接続責務である。

4条件・56操作の各段階で全metadata、availability、認識上下限、固定assay、退避候補とrate状態をPython参照へ照合した。
rate行容量72／144／288／512では各900イベントを、全履歴から時間範囲・保持容量・slot birthで選ぶ独立集計と照合した。
期限切れが古いgenerationを新しいslotから減算しないこと、overflowのunverifiedが履歴失効後も残ることを検査した。
独立rate検査の最初の閾値3.0は、小数増分の加算順序による丸めで境界を跨いだため不一致になった。
検査の閾値を到達可能な0.01単位の総和の間にある3.125へ変更した。実装とPython参照の厳密な超過判定は変えていない。
生の失敗出力も保存した。認識では、真のavailabilityゼロを数値floorへ持ち上げず、極端なscoreでもlog-sum-expを使う。
Nativeのmetadata本体は112 bytes／slotで、Pythonの256-byte padded rowとは区別する。
staging・receipt・rate・score scratch等の所有領域は別であり、runtime全体の容量採用結果とはしない。

通常経路の確定候補は挿入時に元の終端・開始・occurrence順へ整列するよう修正した。
複数groupから300／100／200 msの終端が同じcutで届くfixtureで、100／200／300 ms順の確定sequenceを確認した。
観測の配信順を、認知的な経過時間へ置き換えない。これはstage-1保持へ渡す前の通常runtimeの変更である。
30秒WAV smokeは29.9946667秒・212 episodeで完走した。同一occurrenceを使った旧版比較で、
全212件の元の音響証拠・支持・descriptorは一致した。順序整列で34件のepisode ID／generation／確定sequenceが変わった。
そのためcapture全体はbit一致ではなく、旧版のIDに基づく容量参照を新しい版へ流用しない。

全Rustテスト1,006成功・失敗0・30ignore、通常Clippy・全target check・差分空白検査通過。
Python参照の27テストも通過。`cargo test exit=0 @ 2026-09-13T20:54:55+09:00`。
observer packet version 20、Recall model version 5。source manifest SHA256は
`5320b851ea803554bf95c882d8538de6123e306216aced990ac4438a83864d10`。
検査記録は`target/temporal-dcc/i9-retention-numerics-20260913/verification.json`、実音比較は
`target/temporal-dcc/i9-retention-chronology-smoke/prefix-comparison.json`。

次は数値moduleを通常経路へ接続する。必要なのは、確定したstage-1のepisode割当と不明／未割当支持を保存するproducer、
同じ元queryのcoarse receipt、完全な確定sequence、bankとgraphを同時に退役させる経路、
認識のepisode別割当と未解決continuationへの接続である。現runtimeのoldest-original-end退避はまだこの数値モデルに置換していない。
r_maxは数値fixtureの明示値であり、新development素材からの校正値はまだない。
先行するversion 17の両bus長尺runは22分20秒時点で9,099 episode・退避0まで進み、継続中である。

## 旧方式で残った終了条件（保存・自動実行しない）

1. 拡張bank・新索引の実replayと同じqueryの参照比較は両busで通過。graph所有修正後の実行結果も確認し、証拠を区別する。
2. 候補境界同点・途中開始位置・候補選別による帰還損失を修正し、別の素材でも再検査する。保存容量、edge、indexの損失を区別する。
3. episode graphの全曲保持・edge損失を検査し、未解決continuationと第3入力をstage-1 retrieval割当に基づく因果的producerから供給する。比較controlの実装は接続済み。
4. 新しい固定長尺素材の全replay・同一query参照を完走し、両busのcommitment-rate、reinforcement／metadataとの差、退避年齢、edge pressure、runtime／memoryを記録する。全strength/interference lifecycleの未供給を0件へ置き換えない。
5. O16の容量・index採用根拠をtechnote-ledger §9.3.55へ反映し、必須の全体検査後にI9終了を判断する。

R/Aの係数適合、人間の全曲評定、作者採用、公開受入は別工程として残る。
