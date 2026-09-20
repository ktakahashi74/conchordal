# M0: 同じ周期候補からgroupingを提案する

2026-09-12。Rustのaccent ledger／period estimatorを入力とし、整数groupingとcyclic wordの
音響admission、100 msのinventory更新、上位16候補の保持を接続した。`cfg(test)`内の数値実装である。
本番worker、全group lifecycle、beam／context、継続仮説への接続は未完了であり、M0全体は完了していない。
[source・fixture・検査・費用の記録](m0-grouping-inventory.json)を保存する。

## 入力と因果境界

Inventoryは一つのresolved group generationに属し、既存Estimatorのledgerと最大八個のpeakを参照する。
別group／epochを受け付けず、residual groupには作らない。独立したperiod探索や旧2／3／4 bankを持ち込まない。
配送前の右側文脈を持つaccentはEstimatorで保留され、その入力によって発行済みinventoryを更新しない。

cadenceは`floor(10*(sample-epoch_start)/sample_rate)`とする。同じslotでは以前のsnapshotを保持し、
時刻が跳んだ場合は現在のbankから一度だけ再計算する。飛ばしたslot数を別に記録する。
44,101 Hzのように100 msが整数sampleにならない条件でも、丸めた周期を反復加算してdriftを作らない。
snapshotには実際のrefresh時刻、元のsource／availability終端、保持accentの物理span、容量eviction終端を残す。
新しい呼出し時刻で以前の音響支持を若返らせない。

## 候補のadmission

整数長Lは2–16で、D=L*Pとする。t0+D、t0+2Dのそれぞれ前後0.1D内から、前の選択anchorより後にある
最寄りaccentを選ぶ。同距離は早いtimestamp／IDとする。第二の探索窓を広げず、第一の実測anchorへ移動しない。
その後、二つの実測反復を別々に検査し、それぞれduration誤差10%以下、元の取得・association支持90%以上を求める。
整数候補では余分なintervening accentを許す。P=0.5 s、L=4、元anchorから1.9／3.9 sの正例も採用される。

word長は2–8、stepは登録済み九値、総長は16 reference beats以下である。次の2n個の連続区間を使い、
最初のn区間を絶対時間差の最小値でquantizeする。同値なら小さいstepを選ぶ。次のn区間も同じ順序のstepへ
各10%以内で一致し、二つのtotal durationと観測支持の検査を通る必要がある。総長一致だけでは採用しない。
余分なaccentを暗黙に飛ばさず、全start anchorの走査によってrotationを別候補として保持する。

全候補は選択された異なる端点weightの総和で順位を付ける。同値はperiod bin、kind（integer、word）、
長さ／word symbols、元のaccent ID列の辞書順である。同一period・shape・anchor列を重複保持せず、上位16件を残す。
全候補の大きな動的配列を作らず、固定16枠への挿入で同じ順位を得る。連続timing residualと端点weightの平均も残す。

この総和規則には、同weightなら端点の多いwordが有利になる性質がある。等間隔の接続fixtureでは
上位16件が全てwordとなり、採用可能な整数候補が押し出された。2／3／4限定対照ではその整数候補を確認した。
これは現在の宣言された順位規則による結果であり、平均weightへ変えたり知覚的な優劣へ読み替えたりしない。
下流のgrouping評価と候補数の感度比較に残す。

## 独立比較と対照

Python参照は全accentから線形探索で最寄り境界を選び、全採用候補を収集してからdedup／sortする。
Rustのbinary search、固定16枠、cadence cacheを再利用しない。54条件・1,321入力accentを作り、
37,948件の候補採用と605件の保持候補について、shape、anchor、順位、coverage、残差、支持量、元時刻を照合した。
通常条件に加え、支持89／90%、二反復の誤差相殺、第二探索窓、同距離tie、余分なaccent、rotation、
period不在、2^60 sampleのepoch、容量64／128／256を含む。

development対照は別flagで表す。strict integerは余分なintervening accentを禁止し、
one-skip wordは基準候補に加えて次の2n+2端点の内側から一つだけaccentを除き、隣接二区間を結合する。
省略IDを保存し、追加case数を別計上する。`grouping_inventory_234`は同じ周期候補で整数2／3／4だけを使い、wordを使わない。
許容差0.05／0.10／0.20をfixtureで比較し、quantization tie専用fixtureには0.5を使う。
数値一致はこれらの対照のdevelopment採用や下流fit合格を意味しない。

## 費用と未完了条件

容量128・八周期では一refresh当たり最大15,360 integer cases、7,168 word casesである。
binary searchとstep比較、順位比較、採用件数を実測に含める。one-skip対照の追加casesは基準の上限に含めない。
保持span不足と容量evictionを表示し、到達できない長周期を「拍節なし」の証拠としない。
仕様9.2のgrouping費用に残っていた50 msを、既存の40 ms p99 worker予算へ同期した。
完成計画のpair-grid上限も前工程で確定した710,157 bin訪問／busへ同期した。予算を緩めていない。

全grouping／word感度比較と再fit、beam／contextの所有、omission時の継続仮説、実波形からの回収、
全二bus／64 Voice／deviceを含むO04、人・素材・分割・pilot条件は残る。

## 最終検証

全Rust検査は873成功・0失敗・25 ignoredである。今回の通常検査七件はreleaseでも成功し、
独立参照の採用・保持件数はdebugと一致した。fixture再生成はbyte一致、通常clippyは`-D warnings`で成功した。
全targetのclippyは既存の`unnecessary_cast`／`manual_is_multiple_of`だけを除外して成功した。
format、diff、Zola buildも確認した。

releaseの単体費用測定は容量128・八周期・非ゼロの飽和bankを使い、100回warmup後に6,000 refreshを測った。
中央値1.024 ms、p99 1.071 ms、最大1.571 msだった。一refreshで15,360 integer cases、7,168 word casesを走査し、
7,859件を採用、boundary search 126,682比較、step 159,390比較、順位125,003比較を行った。
全候補を先に確保せず、Proposalは544 bytes、返却Viewは8,928 bytes、Inventory本体は8,984 bytesである。
これらは型layoutであり、worker全体のstack上限・RSSではない。DSP、pair-grid、beam、二worker、64 Voice、deviceを
含む全O04の費用または合格としては扱わない。
