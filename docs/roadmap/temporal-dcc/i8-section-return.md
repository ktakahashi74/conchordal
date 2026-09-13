# I8: sectionと帰還

現在の再開先（2026-09-13の範囲改訂）: I8初期診断版の技術完了を維持する。
I9は本体の必須工程から研究拡張へ移管・保留した。以下の「次はI9」は過去の記録であり、
現在の実行指示にはしない。[マイルストーン§1.1・§5](milestones.md)の対象機構と生成作用の選定へ進む。

2026-09-13。前回は週次使用率37%から51%まで進めて停止した。
今回はユーザーの「I8を完了させて」に基づく再開・完了記録。未commitの既存変更を保持した。

## 現在地

ユーザーの「I8を完了させて」に基づき再開した。I8初期診断版の技術実装・検証を完了した。
確定phrase記憶、committed fallback、局所section分岐、内部展開と帰還、想起焦点を通常observerとUI/reportへ接続した。
最新の実装・検査範囲は末尾の「I8初期診断版の完成検査」を参照する。

## 週次使用率51%での前回停止記録

**週次使用率51%を確認して作業停止。** 確認通知は2026-09-13T03:26:05.625Z。
ユーザーの停止条件を満たした。I8は部分実装であり、技術完了へは変更しない。
停止時の証拠と最終検査は`target/temporal-dcc/weekly-stop-20260913/`へ保存する。

I8は進行中。§9.5の数値参照、phraseの親経路と元の末尾特徴に続き、初期文脈の区間統計consumerを
実音observerとUI/reportへ接続した。`[temporal_section]`を明示した場合だけ有効になる。
section境界の分岐、帰還への文脈復帰、共有context更新はまだ実装していない。初期文脈の予測を
これらの完成とは扱わず、snapshotに`initial_context_only=true`を保持する。

## 実装済みの数値部分

- 各640バイトの累積・区間記録。54個のf64統計と6個のending記述、32バイトpadding、16個のu64 bookkeeping。
- 対応5値、順序遷移25値、articulation/participation 9値。直近2/4/8区間の境界外を結ばない集計。
- 新しい物理観測の一度だけの加算、欠測時間とfractional assignmentの分離、区間membershipの一度だけの適用。
- 受領済みaccentの元時刻による帰属とsequence watermark。元section以前のaccentを加算せず、再配送も加算しない。
- 終端記述の共通有効座標とcached対応によるexact/transformed/contrast/unmatched/unresolvedの分類。
- 82入力と独立した83係数hazard、3出口の係数ごとのsum-to-zero制約。差分はraw単位で作り、欠測は標準化後ゼロへ補完する。
- 予測ではelapsedだけを進める。候補episodeを重複除去したbest/gapは元音響支持から0.5秒以上ならmaskする。

既存Python参照から作ったfixtureを用い、342時点の累積・ring記録を全バイト照合した。
分類60例と48件のhead予測も照合した。手計算検査には境界前のpredecessor、旧recordの退避後再配送、
欠測、世代継承、accentの遅延、一度だけの加算、丸め後の共通分母、無効入力時の非更新を含む。

## I7の積分修正

I8でも使うduration hazardを調べ、`a=20,b=-40,d=0..1200秒`で旧Rustのpanel倍増判定が
冒頭に集中するhazardを見逃すことを回帰テストで再現した。修正前の失敗ログは
`target/temporal-dcc/i8-section-port-20260913/conchordal-i8-gap-regression-before.log`。

共有`hazard.rs`へ四階微分のBernstein上限と丸め余裕を移し、最大64評価、許容差
`1e-9+1e-7*integral`の範囲で確定できなければunknownを返す。600条件で既存Python参照と
支持/未解決判定が一致し、支持された397条件の積分値も許容差内で一致した。
この修正はI7の実行経路にも適用済み。判定不能を「境界なし」へ変換しない。

## 検証

2026-09-13 10:45:47 JST: 全Rustテスト958成功・失敗0・28ignore。
`test_report.txt`と`test_status.txt`に必須の全出力・同一shellでの終了コードを保存した。
通常Clippyと全target checkは通過。全target Clippyは既存テストコードの13警告で失敗した。
内訳は`life/temporal_participation.rs`の不要な整数cast 12件と`runtime/mod.rs`テスト内の
`manual_is_multiple_of` 1件であり、今回の差分からの警告ではない。
検証ログ・各source/fixtureのhashを`target/temporal-dcc/i8-section-port-20260913/`へ保存した。
この検証時の全`src/**/*.rs` source tree hashは
`afcca3b9a64a760e16e1e341735772544826ee96cccd67246ec06511c29d7c5d`。
受け渡し単位の終了時は週次使用率44%。現在の確認値は末尾に記録する。

## 次の接続と残件

1. 初期文脈の履歴継承と実音consumerは接続済み。articulation/timeの共同所属、複数provisional span ownership、遅延accent配分とsealingを接続する。
2. ongoing phraseの0.5秒supportによるcue選択と安定したoccurrence identityのquery接続は実装済み。確定済みphraseが供給されたときのfallback、phrase由来の記憶bank、全selector比較を接続する。
3. 明示した暫定acoustic scoreを最終stage-1 recognitionと共有し、登録比較・fitへ進める。現在の診断値を学習済みscoreとは扱わない。
4. section候補・新規文脈/帰還/対比・想起焦点を共通UI/reportへ接続する。欠測・epoch・退役・EOFを境界や帰還にしない。
5. 同じ局所音の初出/介在後の帰還、内部展開/切替、音声一致の実行検査。I8完了と、R/Aの適合・聴取・最終受入を分ける。

private prepared-commitの遅延適用、full joint beam、全負荷・長尺容量、R/Aはこの数値部品の検査では証明していない。

## 続く受け渡し単位

phrase snapshot version 2、observer packet version 8。新しい候補は親index、更新前のgroup時刻、
閉じたforegroundと末尾6特徴を持つ。New/Overlap/Inactiveだけが元区間を閉じ、Reinterpretと
欠測は新たな確定区間を作らない。二段のgap/observation更新でも親を直前の公開経路へ遡って保つ。

末尾特徴は最大2秒、epoch/group/foreground開始でclipし、90%の物理支持を要求する。
既存の割当energyとスペクトルのcentroid/central varianceを使い、energy重みのmoment結合で
pooled spectrumの最初の2座標を得る。RMSはlog値を逆変換せず元energyから計算する。
同じclip窓は経路間で抽出結果を共有する。32条件のPython参照で値とmaskを照合した。
後の境界hopや、元cutでは未入手だったaccentは閉じた末尾を書き換えない。
phraseのaccent bankには元のadmission recordを保存し、密度は件数ではなく重みの和を用いる。

Recallは16 episode × 最大4変形の既存照合結果を、二つの事前確保heap bufferへ保持する。
受領検証後にbufferをswapし、`latest_matches()`は元query metadataと不変の借用を一緒に返す。
座標別二乗残差/件数、motion/intervalの残差/件数、matched/missing/edit数は同じDTWから取得する。
追加DTWを起動しない。新しい計算、期限切れ、episode退役による結果の混同を検査した。

全候補を共通snapshotに直接詰める試行はobserverのstack overflowを再現したため取り除いた。
候補本体をownerのheapへ移し、observer作業状態もheapへ置いた。その後、通常workerの大型snapshot
値コピーによるstack overflowも実際のhabituationテストで再現した。workerの保持配列をBox、
UI snapshotの配列をArcへ変更し、reportは保持値を借用する。UIへの新しいArcは既存のUI更新間隔で
作り、毎音声hopには割り当てない。stack上限は変更していない。失敗していたhabituation4件は通過した。

この受け渡し単位の検証は2026-09-13 11:16:04 JSTに完了。全Rustテスト964成功・失敗0・28ignore、
通常Clippy・全target check通過。ログは`target/temporal-dcc/i8-handoff-20260913/`。
全Rust source tree hashは`faa867c8a91e4a51361abcdfeb829e30e78954ec3a381f8af94d1d7db1d6a565`。
この受け渡し時点ではI8全体を完了に変更せず、次のconsumer単位へ進んだ。


## 初期文脈consumer

observer packet version 9。`temporal_section`はphrase設定を必須とし、82個の平均・標準偏差、
83個のhazard係数、各83個の3出口係数を明示する。3出口は係数ごとに和ゼロを要求する。
endingの6座標用scaleと、下記の暫定acoustic score用scale/係数も明示する。既定の学習済み値はない。

最大7群、既存phraseの最大15経路を追跡する。各経路の累積記録と直近4区間ringを親indexに従って
事前確保scratchへコピーし、累積活動は新しいgroup観測とadmitted accentを一度だけ加算する。
群の初期化時に確保した二組のpath/ring bufferを再利用する。代表経路の39/39/82入力を公開し、
1秒先のhazard/出口は保持phrase経路の質量で混合する。代表入力を平均して非線形headへ入れない。
EOFでは最後の入力時刻`observed_end_sample`を維持し、将来予測をunknownへ移す。

現在の実音入力はassignment、admitted accent、integer/word admission union、resolved-only overlap。
articulationの経路共同所属と二つのtiming dispersionは未供給なので欠測のまま保持する。
phraseのforeground ownershipだけを消費する初期consumerであり、全provisional spanの共同ownershipや
遅延sealingを実装したとは扱わない。閉じた区間への遅延accent再配分、§9.3のprivate prepared commitとの
接続は残る。対応分類も、経路上の対応が未確定ならunresolvedに保つ。最近傍costの低さだけで
exact/transformedを割り当てない。

既存Recallの受領済みcacheを借用し、同じgroup・foreground開始・支持endpointに一致するprefixだけを
暫定診断に使う。別の固定長assay窓を同じprefixとみなさない。新規DTWは起動しない。
ただし§9.5の0.5秒supportによる単一cue選択は未接続であり、この一致検査でその要件を満たしたとはしない。
`prefix_support_queries`は開始と支持が一致した既存queryの使用件数であり、完全なcue選択の成功率ではない。

この初期scoreの14入力は、既存DTWの10座標の`√(二乗残差和 / observed cue steps)`、
支持された隣接対のmotion/interval RMS、cue step数で割ったinsert/delete件数。各座標の支持がなければ
欠測とし、明示された平均へ補完して標準化する。係数15個の先頭は切片。
識別された変形ごとに計算し、同一episodeをまとめてbestと2位との差を作る。availabilityやsection ageは
加えない。これは未適合の初期score layoutであり、最終stage-1認識headとの共有・登録比較は残る。

実音NSGTからのobserver検査、parent lineageと暗黙gap、欠測座標、重複receipt拒否、EOFとepoch切替を検査した。
renderの既存比較へsection設定を追加し、診断off/onおよびreport有無でWAV全バイト一致を確認した。
この接続単位は2026-09-13 11:38:18 JSTに全Rustテスト966成功・失敗0・28ignore、
通常Clippy・全target check通過。実音NSGTの1群で支持が一致したqueryの利用は7件。
必須ログ、全Rust sourceのhash、quota通知は`target/temporal-dcc/i8-consumer-20260913/`へ保存した。
全Rust source tree hashは`cc9153a4865995d7b65573e8da4fa4443a62e1cf1b86919b69c85da355f40d7a`。
最新の週次使用率は46%（通知時刻2026-09-13T02:39:00.037Z）。50%超の停止条件は未達、goalは継続中。

この単位の次にcue選択へ進んだ。当時のobserverはmemory→phrase→sectionの順なので、
受領済み結果の取り込みと新query発行を分け、今回のphrase prefix選択を同じmatcherへ渡す必要がある。
全prefixのdescriptorはboundedなSpanで保持し、0.5秒窓だけをqueryとして切り出さない。
固定assayの記憶とphrase occurrenceの記憶を同一identityへ押し込まない。追加matching passを増やさず、
選択・受領identity・元の音響支持・fallbackを有限fixtureと実音で検査する。


## ongoing phrase cueの実接続

observer packet version 10、Recall transport model version 2。section有効時は受領済み記憶の取り込みを
先に行い、gesture/phrase/sectionを更新し、選ばれたcueを既存schedulerへ渡す。
新たなmatcher passは追加していない。この経路のcadenceは登録値100msを要求する。
section無効の既存assay経路と、その50msのrender比較条件は保持した。

群ごとに保持phrase候補のforegroundをstable creditで重複除去し、各occurrenceの全prefixを
既存の128-knot Spanへ入れる。同じcreditのReinterpret aliasは二重に蓄積しない。
異なるstart/endpointを持つaliasは拒否する。群退役でdescriptorとselection履歴を破棄し、
同じslotや番号の再使用で別generationの支持を継承しない。
現在保持された最大15 occurrenceだけを追跡し、descriptor以外には0.5秒分のcanonicalな
assignment支持を保持する。prefix本体を0.5秒へ切り縮めない。odd sample rateでも窓境界の
半sampleを丸めず、選択用積分は正確に0.5秒に保つ。

選択はpositiveなstage-1 weighted seconds、最後の有効支持endpoint、span開始、低いcreditの順。
phrase候補のposterior massをcue重みへ掛けない。0.5秒内に正の支持がなければongoing cueなし。
確定済みphrase ledgerはまだ存在しないため、provisionalに閉じた候補や固定assay episodeを
確定済みphrase fallbackと呼び替えていない。このfallbackは未接続のまま。

Frozen query headerにoccurrenceとsupport identityを渡し、受領snapshotへ選択時刻・支持量・元の
prefix範囲を保持する。queryが待機している間に別のcueが選ばれても、受領結果のmetadataを
現在の選択へ差し替えない。epoch内でassay/phrase cueの所有方式を切り替える呼び出しも拒否する。
sectionのretrieval入力は、この選択証拠を持つ受領済みqueryだけから群ごとに一度計算し、
各section経路へ同じstage-1入力を渡す。いまの各foregroundと開始が一致する経路だけへ
別々に入力する旧暫定処理を取り除いた。元の音響支持からの0.5秒失効判定は維持する。

Pythonの既存`select_section_cue`から生成した80時点で選択ID・weighted seconds・最終支持endpointを照合。
競合する長短prefix、同点、posterior massの逆転、alias、ゼロassignment、欠測窓を含む。
別の実行検査で30秒のwhole prefixを128 knot以内に保持し、記憶へ渡る元の開始・終了とcreditを確認した。
退役generation、衝突alias、odd sample rateも検査した。

連続実音の既存I7 assay検査は予測照合30回とclosure支持を維持する。
新cue経路は約3.2秒の休止を挟んだ再出現の実音で検査し、予測照合12回・支持一致query29件を得た。
この時点の保存bankは引き続き短いI4 assay intervalであり、whole prefixが保存された短い続きより
長くなるとclosureの根拠がなくなる。検査終端でclosure支持0・unknown1を保ったことも確認した。
これは長いphrase記憶やsection帰還推論の完成ではない。対応する記憶供給と継続先の不足を明示する。
renderの診断off/on・report有無のWAV全バイト一致も通過した。

2026-09-13の最終検証は全Rustテスト970成功・失敗0・28ignore、通常Clippy・全target check通過。
終了記録は`cargo test exit=0 @ 2026-09-13T11:58:34+09:00`。証拠は`target/temporal-dcc/i8-cue-20260913/`へ保存した。
全Rust source tree hashは`691b43c9f892d773e80ccfd992827f3500cdbf3228710dd9ac3e053d95bee783`。
最新の週次使用率は48%（2026-09-13T03:01:00.699Z）。50%超の停止条件は未達でgoalを継続する。
次の有効な検証は、ターゲット音声と直前の休止を同一にした初出/再出現比較。
それに続いて、正本§9.3の確定条件を満たすphrase occurrence ledgerを供給し、
長いphrase記憶とcommitted fallbackを接続する。単なる最良pathや最近傍候補を確定済みと扱わない。


## 初出・再出現の一致条件と群別cache

ターゲットは48kHzの実音をNSGTから通す。両条件ともframe 300..599の休止と
600..659のターゲットを同じ値にし、frame 0..299で同じ音を先に聞いたかだけを変える。
この対照ではphrase hazardのretrieval係数を両方ともゼロへ固定し、cue候補を変える
直近の記憶フィードバックを除いた。他の設定は共通。既存の連続assay検査と長い再出現検査も残す。

一致を直接検査した範囲は184,320個のf32音声sample、3群それぞれの10個のraw音響特徴、
選ばれたcueの開始・終了・選択支持量、およびsection入力の先頭80成分の値と欠測mask。
浮動小数点値はbit列で照合した。群とoccurrenceの番号は以前の経験によって異なるため、
条件間の数値signatureから除き、物理支持と音響値で整列する。各runのquery内ではstable identityを
別の受領検査で束縛している。

初出では3群ともretrievalの2成分が欠測で、予測照合は0回。再出現では3群とも2成分が有効で、
予測照合は18回。未適合のmatch係数はゼロなので、有効score/gapの値は0である。
支持の有無の差を確認したのであり、このゼロを高い音楽的認識scoreとは扱わない。
section分類・想起焦点の完成、科学的妥当性、聴取受入をこの対照から主張しない。

この検査の途中で、Recallがbus全体の直近queryしか保持せず、別群の結果が到着すると
まだ有効なreceiptを押し出す欠落を特定した。2群を交互に照合する回帰検査で修正前の失敗を保存した。
修正後は各resolved groupが16 episode×最大4変形の受領済みcacheを持つ。
worker共通のpending bufferと、正しい受領先groupのbufferだけをswapする。
群退役ではそのcacheを破棄し、期限切れとepisode退役は各群のcacheへ適用する。

共通snapshotには7群のquery metadata/最良候補だけを渡す。全変形と残差の配列本体は
ownerのheapに保持し、共通snapshotへ複製しない。`latest`は表示用に受領済みgroup queryから導出し、
phraseは`latest_for(group)`、sectionは`matches_for(group)`で元のgroupを指定する。
`retained_matches`は有効な全群cacheの合計になった。observer packet versionは11。
数値matcherとquery cadenceを変えず、照合passも追加していない。

cache bufferの上限は最大7群の受領済みbufferとworker共通pending bufferの8組で、各64変形。
DP、query snapshot、既存Report、descriptorやheap overheadはこの数には含めない。
全O04の容量・コピー量・最悪実行時間検査は引き続き未完了。

最終全検証はRust 971成功・失敗0・28ignore、通常Clippy・全target check通過。
終了記録は`cargo test exit=0 @ 2026-09-13T12:19:05+09:00`。証拠を`target/temporal-dcc/i8-return-20260913/`へ保存した。
全Rust source tree hashは`af76ff8bdf55ccfca6496f5fabaf78c4f14f7c711018f343b92e852a5f887f98`。
最新の週次使用率は50%（2026-09-13T03:21:15.876Z）。50%を超えていないため停止条件は未達。goalは継続中。
次の主な実装残件は、§9.3の0.5秒lagとjoint supportを満たす確定phrase occurrenceの供給。
それを長いphrase記憶・committed fallback・section分岐へ接続する。既存I4の固定assay bankを
完成したphrase記憶と見なさず、provisionalな最良pathを確定済みに読み替えない。


## 停止直前の確定identity監査

確定phraseを既存`OccurrenceLedger`へ渡す前に、identityの未接続を確認した。
実際のI7分岐を20 hop進める検査で、同じongoing credit=1から、異なる元終端を持つ13候補が得られた。
最初の3候補はsample支持[0,100]、[0,200]、[0,300]。0.5秒のcommit lag内で競合する。
各候補のendingはそれぞれの元のheard endpointと一致する。

一方、Python正本の`OccurrenceLedger.stage`は同じoccurrence IDに対する異なる元recordを拒否し、
`reinterpretation must retain the original support/descriptor`を要求する。したがって、
`Foreground.credit`をそのまま確定occurrence IDとする接続は成立しない。
`credit`にはこの境界をEnglishコメントで明記し、Rust回帰検査を保存した。
ここで新たなledger接続や確定判定は実装していない。

再開時は、進行中prefixの共有credit、競合する終端support、確定occurrenceのidentityを分ける。
同じ物理観測のrenewal creditを保全し、異なる終端を機械的に全部別occurrenceとして二重に書かない。
どの元終端を固定するかと、後続の変更をsealed metadata revisionへ送る条件を§9.3に沿って確定し、
0.5秒間のjoint path support・未保持mass・遅延accent所有を結ぶ。これがcommitted fallbackと
長いphrase bankの前提になる。section分岐・帰還焦点、全O04、R/Aはその先の残件。

停止時の全検査はRust 972成功・失敗0・28ignore、通常Clippy・全target check通過。
終了記録は`cargo test exit=0 @ 2026-09-13T12:27:47+09:00`。必須の全出力、終了コード、quota抽出scriptと51%通知を
`target/temporal-dcc/weekly-stop-20260913/`へ保存した。全Rust source tree hashは
`9589a15afed154b5e3c034e01a92390a2ea6567ecfd25ffad740bc5fb58ed6e9`。ユーザー指定の停止条件を満たして作業を止めた。I8全体の完成とは区別する。


## I8初期診断版の完成検査

I/R/A工程のI8終了条件は、実区間記述から継続・対比・変形した帰還・想起焦点・診断までの接続である。
各Iが後続全モデルの完成を要求しないという実行マイルストーン§2の区分に従い、以下は未校正の初期診断版として検査する。

### 確定phraseと元の証拠

`section/commitment.rs`に、共有ongoing creditと異なる終端候補を区別する台帳を接続した。
確定occurrence IDはongoing query IDと別の番号空間に置く。保持phrase親経路から終端所属を継承し、
新たな同一終端の分岐を重複なく合流する。音響割当秒を元の区間長で割り、残った経路質量を一度だけ掛ける。
失われた質量を再配分しない。別終端への書込みでも、同じ観測の分岐質量を重複して使わない。

期限は元の終端から0.5秒。奇数sample rateでも期限後の半sampleを丸めて証拠に含めない。
締切を跨ぐ処理では最後の期限内の支持を封印し、処理時刻・元観測時刻・lagの欠測秒を別に保存する。
後のjoint context cellの変化は厳密な0.25超で原occurrenceを一度だけrevisionに数え、元の支持・descriptorを変更しない。
切り捨て済みの祖先を、新たな解釈の証拠として再構成しない。

最初の音響hop内でphraseが閉じた場合、元の末尾が未保存になる欠落も修正した。
元の音響が利用可能になった同じcutで末尾を抽出し、それ以後のhopから再計算しない。
遅延accentは元のforeground ring所有へ一度だけ反映し、境界accentを隣接二spanへ二重計上しない。
0.5秒後の受領はlate supportとして計上し、封印済みイベント数と末尾を変更しない。

### 記憶とcue

section有効時のbank供給を固定長I4 assayから確定phraseへ変更した。section無効時のassay検査経路は保持する。
長いphraseも128-knotのwhole descriptorのまま保存し、30秒の区間が一度だけ保存される実行検査を追加した。
記憶の公開可能時刻はseal時刻、保持経過時間の起点と音響出典は元の観測時刻にする。
保存順がgroup間の元終端順と異なる場合も、元観測終端で索引を整列する。

ongoing cueがなければ、同じ有効generationの最新の支持された確定phraseを選ぶ。
そのqueryは元の音響時刻を保持し、fallbackを読んだ時刻で0.5秒の鮮度を更新しない。
照合は既存100ms scheduler／16 episode／最大4変形を再利用し、section独自のDTWを追加しない。
同じ14成分の明示acoustic scoreをretrieval入力と帰還候補の変形選択に使う。cutoff tie、変形限界、欠測を帰還根拠にしない。

### 局所sectionと想起焦点

各phrase親の条件付きsection経路から継続・新規文脈・対比・帰還を提案し、群ごとに計16候補を保持する。
枝刈りの質量はunknownとして残し、再正規化しない。各経路が自分の640-byte累積recordと4-record ringを持ち、
非線形headへ平均した履歴を入力しない。継続では累積とrecentを保持し、新sectionでは新しい区間統計を開始する。

phrase終端の元のcontext所属を条件付き経路から投影し、最大16文脈の支持と共に確定記憶へ保存する。
帰還は受領済み照合が指す過去phraseのcontextにだけ接続する。対応する元query ID、episode、変形を診断へ渡す。
同じcontext内の対応は内部展開として現在の区間統計を保ち、別contextへの対応は帰還候補にする。
現在の観測が欠測なら新しい境界・帰還を作らず、EOFは最後の観測状態を保って将来予測をunknownにする。
Scenarioのsection名は入力に使わない。

observer packet version 12、phrase snapshot version 3、Recall transport model version 3。
UI/reportは初期文脈限定の表示から、文脈候補・質量・関係・想起出典・確定数・損失・revisionの表示へ更新した。

### 検査範囲と後続担当

- 登録39／82成分、640-byte record、342時点のbyte照合、48 head予測と従来のhazard参照検査を維持。
- 競合終端・別ID・共有credit保存、元descriptorの固定、期限越え・欠測lag・奇数rate・revision閾値を検査。
- whole-prefixの選択と30秒記憶、fallback、query出典、各群cache、retirement／epoch／EOFを検査。
- 一致した局所実音の初出／再出現で、原音・cue・音響10成分・非retrieval80成分のbit一致を維持した。
  確定phrase bankでは再出現3群中2群が有効retrievalを得た。支持のない群へ成功を補わない。
- 強く帰還を提案する明示試験係数でも実音を通し、介在休止より前のepisodeとcontextへ戻る候補と元query IDを確認。
  この試験係数も候補の微小な質量も、適合済みの認識精度・音楽的確信を表さない。
- 通常renderで診断off/on・report有無の音声一致、両bus終端を検査する。

この初期経路のspan所有はI7が供給するforeground経路に対応する。full shared/local stage-3更新、
articulationのjoint path状態と二つのtiming dispersionの供給、全provisional spanの重複所有、
private prepared commit・完全なstrength/interference lifecycleは後続の統合・適合の対象として保持する。
未供給の6座標は欠測のまま扱い、独立した周辺posteriorの積や平均で埋めない。これらの後続全モデルの義務を削除しない。

累積/ringは各640 bytes、局所候補は群ごとに16。cueは最大15 ongoing prefix/group、
pending endpointはbus当たり最大512、sealed descriptorは最大256、各descriptorは最大128 knot。
局所提案scratchと祖先・descriptor・queryのコピー量は別に発生する。これはR2の両bus・64 Voice・
全処理・最悪負荷の合格ではない。I9の30分全曲replay、I12の階層間相互更新、R1–R5、A1–A4は未完了。

最終検査とsource hashは`target/temporal-dcc/i8-completion-20260913/verification.json`へ保存する。

最終結果: Rust 979成功・失敗0・28ignore、通常Clippy・全target check通過。
cargo test exit=0 @ 2026-09-13T15:56:44+09:00。全Rust source tree hash: `583b3b779548da28c882991ceaf22645135cde6bd40c9e0e8da3519cceaa3056`。
I8の初期診断版を技術完了に変更する。次の技術マイルストーンはI9「全曲文脈」。commitは作成していない。

## 現行ツリーでの再検証

2026-09-13 18:28:51 JST、ユーザーの「I8を完了させて」に基づき完了条件と現行実装を再確認した。
section本体と配下の計7 Rustファイルは、上記I8完了時のsource hashと一致した。
追加の実装修正は不要だった。後続I9の変更を含む現行ツリーで、全Rustテスト988成功・失敗0・30ignore、
通常Clippy・全target check・差分空白検査が通過した。必須の全テスト出力と同一shellの終了コード、
現行source hashを`target/temporal-dcc/i8-revalidation-20260913/verification.json`と同ディレクトリに保存した。
I8初期診断版の技術完了を維持する。この再検証はI9や後続全モデル・R/Aの完了を意味しない。

## 後続変更を含む現行ツリーの再検証

2026-09-13T20:25:18+09:00、I8の完了条件と後続変更を含む現行ツリーを再検証した。
全Rustテスト999成功・失敗0・30ignore、通常Clippy・全target check・差分空白検査が通過した。
登録数値参照、確定phraseと帰還出典、30秒phrase保存、通常renderの診断有無による音声一致を含む。
追加のソース修正は行っていない。

直前の全体検査にはlistenerの報告ファイル破損・消失による2件の失敗があった。
個別再検査と今回の全体検査では再現しなかった。原因確定・修正済みとは扱わず、失敗時の出力も保存した。
検証出力・同一shellの終了コード・現行source hashは
`target/temporal-dcc/i8-current-revalidation-20260913/verification.json`と同ディレクトリに保存した。
I8初期診断版の技術完了を維持する。I9・後続全モデル・R/Aはこの検証の完了対象に含めない。
