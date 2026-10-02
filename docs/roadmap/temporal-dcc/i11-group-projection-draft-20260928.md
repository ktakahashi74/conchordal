# I11: 身体と音響群の比較単位を揃える次の診断案

2026-09-28。以下は取得前の設計草案。有限再解析は別途[三座標投影結果v2](../../../target/i11-group-projection-20260928/results-v2.md)へ固定した。通常自己群除外の正例は依然として0であり、この診断をI11受入へ数えない。

## 現在の証拠から変える問い

[通常入力の診断](../../../.worktrees/i11-window-inputs/docs/roadmap/temporal-dcc/i11-window-inputs-results-20260927.md)では、private身体全体とshared部分群を同じ窓で集計しても一致しなかった。別途固定した[union解析](../../../target/i11-union-boundary-20260927/results.md)では、単独Aの非ゼロ626 hopにおいて全8 slotのunion scanとprivate scanが一致した。これは全体量と部分量の比較が不一致の一因であるという証拠であり、個別groupが特定の物理sourceを表す証拠ではない。

次は閾値0.25を緩めたり、既存holdoutでmedoidを再学習したりせず、「同じgroupへの周波数配分をprivate身体にも適用すると、同じ種類の量を比較できるか」を調べる。現在の固定model、body→medoid→group判定、CDF、既定音声動作は変更しない。

## 比較する量

同じbus・Log2Space・支持hopに、private sourceの非負energy scan（記録 `spectral_scan`）を `P_s(k)`、shared mixtureの記録powerを `P_mix(k)` と置く。private scanは既に全sourceのPCM energyへ正規化されており、生のNSGT powerではない。以下の質量比ではこの一様な倍率が消える。shared frontendがそのhopで実際に使ったgroup配分を `w_g(k)` とする。新しい群分けをprivate側へ独立に走らせて、似た群を事後選択する方法は使わない。既存のownerは距離2 bin以内のpeakについて `(距離, peak bin, slot)` の最小値を採用し、なければ残余slot 7となる。`w_g(k)` はこのownerへのone-hotではなく、`assignment.rows[owner].weights[g]` の8群への配分である。同じ記録と規則から再構成する。

候補量は `Q_s,g(k) = w_g(k) P_s(k)`。private身体とshared群を比較する際には、この部分scanと、sharedが記録したpower scanとassignmentから再演したgroup scanを並べる。group scan自体はrawに保存されていないため、再演値を記録済みのgroup energy・centroid・spreadへ先に照合する。全身体scanと一群のscanを同じ型として扱わない。全groupの配分がpartitionとなるhopでは、`Σ_g Q_s,g = P_s` の保存を検査する。残余slotも分母から落とさない。 sharedのpowerが0のbinはnativeの再演で参照されなくても、private側のpowerが正ならowner rowを必要とする。そこが欠測ならprivate投影はunsupportedとし、0配分で補完しない。

音声energyをscan質量に配分する既存規則に合わせ、候補の部分energyを

`E_s,g = E_s × Σ_k Q_s,g(k) / Σ_k P_s(k)`

とする。支持されたPCMで `E_s=0` かつscan質量も0なら、energyは既知の無音0として残し、centroidとspreadは未定義とする。無音を自己群の正例には使わない。scan質量0だが音声energyが非ゼロの不整合、支持不足、group/gridの識別欠落はunsupportedとし、0で埋めない。centroidとspreadも同じ部分scanから計算し、まずこの三つの座標だけを明示的に比較する。残りの時間特徴やaccent履歴をshared側から転記して、六座標の一致を作らない。それらの再構成にはprivate部分streamの履歴とreset規則が別途必要になる。

このenergy配分はsourceの音声energyを既存スペクトル配分規則で分けた記述量である。重畳Bでは位相干渉と解析・平滑化があるため、`E_s,g / E_mix,g` を物理的な寄与率や所有確率とは呼ばない。差分powerの切り詰めによる所有率も導入しない。

## 最初の有限診断

封印済みA/B/Cの全a/b rawを用いる再解析を、元の前向き登録とは別に固定する。Aだけから良いwindowを選ばない。現在のBinding成立判断、Bindingなし判断、空群、支持欠落、reset、左端部分hopをすべて分類する。

1. 既存shared scan・energy・centroid・spreadの再計算を先に通す。native再現に失敗したhopへ新投影の比較値を付けない。
2. 同じ `w_g` をprivate scanへ適用し、partition保存、三座標の直接差、全体量との比較から何が変わったかを全対象について保存する。
3. Aで入力が同一のhopの部分量一致は構成上の整合性検査として扱う。新しいsource識別成功には数えない。
4. Bではtarget PCMがAと同一のprefixと、それ以後にtarget自体も変化した区間を分ける。後者をmixtureだけの介入とは解釈しない。
5. Cのbody generation変更と支持欠落を保持する。旧generationの投影を新Recordの欠測へ流用しない。

private rawには独立した周波数中心列やgrid fingerprintがない。今回のgrid整合の根拠は、同じcoreからshared/private解析器を構築する固定sourceの系譜、全scan長、sharedの中心列の不変性、元private座標の再演照合までとする。録画のみでprivateとsharedの全周波数中心が独立検証されたとは呼ばない。

実装前にsource、元raw/index、射影script、対象集合、許容誤差を固定する。既にA/B/Cとunion結果を見ているので、この再解析を未見holdoutや前向き正例とは呼ばない。新しい自然場面への確認は別入力で必要となる。

## 通常自己群除外までに残る設計

群への投影が整合しても、現在の全身体prototypeの学習対象と一致したとは限らない。投影後の分布を既存medoidへ通した結果は探索として分け、modelの自動置換はしない。

一つのVoiceが複数groupへ寄与し、一つのgroupに複数Voiceが重なる場合を許す必要がある。単一の `Voice → prototype → group` 対応だけで全自己群を同定できるとは仮定しない。groupのmerge/split、group handle再利用、身体generation、bus、支持窓、公表時刻を明示して対応を無効化する必要がある。

さらに「似た部分音」と「自分だけに由来する群」は異なる。hardな自己群除外へ進む前に、同時刻の自己除去音声などの別の因果的証拠で、他者由来の残存を扱う規則が必要になる。自己除去したpowerを単純なpower差で捏造せず、既存SourceRemoved経路を使える範囲と支持・遅延の契約を先に確認する。混合・曖昧・欠測はunknownとして残し、対応しやすい例だけを自己群と呼ばない。

この草案で実装を許す最初の範囲は、封印済みrawの三座標投影診断だけである。新binding model、SourceRemovedを使う通常runtimeの対応、実際のCDF除外は、その診断後に別の契約と検査を設ける。
