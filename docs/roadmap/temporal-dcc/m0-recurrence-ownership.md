# M0: 音響group世代ごとのaccent・周期・grouping所有

2026-09-12。既存の音響Frontendに、有界accent ledger、周期推定、grouping inventoryの所有を接続した。
`cfg(test)`内の数値経路であり、本番workerやbeam／context全体の実装ではない。
[source・検査・費用の記録](m0-recurrence-ownership.json)を保存する。M0全体は未完了である。

## 一hopの更新順序

入力はcanonicalな音響終端、実際の受信時刻、元NSGT窓のsource／availability終端である。
将来のavailabilityを受信済みと扱わず、後退した受信時刻も拒否する。通常の入力エラーはepochの失敗を固定し、
以前のsnapshotを隠す。Log2 scanの長さなど、既存のprogramming invariantはhard assertionのままである。

まずFrontendが元の配分handle付きのgroup energyと四hop accentを作る。各accentはその更新前handleのledgerへ
一度だけ配送する。周期bankの物理expiryとgrouping cadenceを進め、旧ownerの証拠frameを保存してから、
退役・出生によるslotのbindingを変える。新しいgroupはcount、weight、bank、pair-grid、groupingを空から始める。
親の同hopのaccentを子へ付け替えず、旧groupの最終累積値を新groupで上書きしない。

frameは三種類を区別する。`evidence_groups`は更新前ownerとそのhopのeligibility／既知association、
`next_owners`は更新後の保持handle、`newborn`は空の履歴から始めた新世代である。
superseded parentが音響slotに残る間は元の履歴を保ち、次hopから音響associationを無効にする。
retirement後のcommitted contextやpending occurrence ledgerへの移管は、今後のbeam／memory接続に残る。
今回のframe保存を、その長期所有まで完了した証拠とはしない。

residualは独立したmixture-tag付きledgerを持つ。周期推定・groupingのpoolには入れない。
欠測では物理時間によるbank失効を進めるが、累積creditを消さず、無音退役も進めない。
既知の低energyが既存Frontendの退役条件を満たした場合にだけ、音響groupの退役へ反映する。

## 有界storageと世代reset

resolved group用に七組のEstimator／Inventoryと、固定snapshot領域を構築時に確保する。
pair、accent、slot順序のbufferは出生・再利用時にも再確保せず、既存領域をclearして使う。
最初のinline配置では、二instanceとepoch再開を含む通常test threadがstack overflowした。
slot poolと固定snapshotを構築時のheap確保へ移し、`RUST_MIN_STACK`を増やさずに検査した。
この修正はhopごとの新しいheap確保を導入しない。

groupingのcadenceは全体epochを基準に保つ。新世代のactivation sampleを別に記録し、
最初のrefreshで出生前のslotを「飛ばした更新」と数えない。これは過去のgroupingやperiodの継承ではない。
Estimator reset後はpair flags、点、順序、確率、peak、credit、容量欠落markerが空であり、
旧ownerの配送は新世代に受け付けない。公開scopeの拡大は、このFrontend内consumerが使うcrate内部に限る。

## 所有と容量の検査

360-hopの単一peak・振幅変化fixtureでは、30個の異なる(group,event) accentを一度ずつ配送した。
容量8に対して容量制約のあるframeが158件、周期あり326 group-hop、groupingあり248 group-hopとなった。
出生は一回で、最初のsnapshotに旧creditは入らなかった。新規pairの77加算を診断値に残し、
同時刻の不要な二回目のrefreshで更新費用を消さない。

split fixtureは親のcreditを凍結し、子が空から始まる条件を確認する。これは空きslot内に収まるため、
容量交換の検査は別の七peak＋残差の1,200-hop入力で行った。
149回の容量退役・同一hop slot交換について、旧ownerの最終frameと新ownerの空cacheを照合した。
大きな欠測ではbankが空になってもhandleとcreditを保ち、その後の既知無音では退役し、復帰は新世代となった。
異なるbus、epoch再開、遅延availability、wrapしたbankのresetも検査した。

## 実NSGTと一定振幅対照

48 kHz、hop 512、nfft 2048で、三つのPCMを各480 hop処理した。500 Hzのcarrierに24 hop周期・二hop幅の
振幅変化（0.01→0.04）を付けた条件、無音、振幅0.01一定のcarrierである。
同じPCM bufferをコピーして、緩い診断用尺度と既存testの基準尺度へ与えた。
diagnosticはsalience deviations 0.2／threshold 0.05、基準はdeviations 1／threshold 1である。
どちらも今回のために適合した尺度ではなく、人評価用の凍結・校正を済ませた尺度でもない。

| 尺度・入力 | accent数 | 周期ありgroup-hop | groupingありgroup-hop | 0.256秒候補ありgroup-hop |
|---|---:|---:|---:|---:|
| diagnostic・パルス | 431 | 912 | 811 | 892 |
| diagnostic・無音 | 0 | 0 | 0 | 0 |
| diagnostic・一定振幅 | 320 | 824 | 808 | 261 |
| 基準・パルス | 79 | 894 | 736 | 876 |
| 基準・無音 | 0 | 0 | 0 | 0 |
| 基準・一定振幅 | 0 | 0 | 0 | 0 |

group-hopはgroupごとのframe数であり、480 hopを超え得る。accent数は全groupとresidualの合計であり、
振幅パルス数や知覚されたevent数ではない。診断用尺度では一定振幅にも目標周期候補が出たため、
候補の存在だけをパルス回収の証拠にできない。基準尺度はこの小さな対照を区別したが、
幅広い素材・routing・欠測・階層の回収、下流fit、知覚的妥当性の合格とはしない。

## 残る条件

全group／contextのbeam所有、original occurrence creditの長期保持、articulation／arrival等の全特徴、
joint inference、素材固定後の回収・fit・校正、全二bus／64 Voice／deviceを含むO04、
人・機関手続き・素材・分割・pilot・powerの条件は残る。今回の数値接続でこれらのgateを解除しない。

## 最終検証と費用

全Rust検査は881成功・0失敗・26 ignoredである。新しい通常検査は八件で、releaseの関連133検査も成功した。
所有trace、149回の容量交換、六条件のNSGT traceはdebugとreleaseで同じ集計となった。
通常clippyは`-D warnings`で成功し、全targetでは既存の`unnecessary_cast`／`manual_is_multiple_of`のみを除外した。
format、diff、Zola buildも成功した。仕様本文と既存のfit座標は変更していない。

release単体測定は600 hopのwarmup後、6,000 hopを処理した。最大七ownerを実際に保持し、
全測定入力に対してcache reset七回、groupingあり45,416 group-hopを観測した。
Frontend、ledger、period、grouping、固定snapshotの作成を含み、中央値40.55 µs、p99 7.240 ms、最大8.960 msだった。
groupingを100 ms cadenceで一斉更新するhopの費用も含む。

Recurrence本体は82,744 bytes、構築時のslot pool payloadは180,544 bytes、frame pool payloadは80,224 bytesである。
別途、Estimatorのpair／accent／順序bufferとFrontendのscan bufferを持つ。これらの型layoutを
全workerのstack上限・RSSへ読み替えない。NSGT、beam、context、両bus同時動作、64 Voice、deviceを含む
全O04の40 ms p99 worker予算とhop余裕は、別途測定・判定が必要である。
