# M0: bundleからのproposal生成と親の固定

2026-09-12。状態: 相関・配分snapshotからの候補生成を、前段の持続・競合reducerに接続した。
`cfg(test)`内のRust数値部品であり、M0全体や本番group lifecycleの完了ではない。
[sourceと結果](m0-proposal-generation.json)、[仕様の版対応](m0-proposal-registration-sync.json)を保存する。
前段の[供給キーreducer](m0-proposal-persistence.md)の入力責務のうち、候補生成と親の固定を今回実装した。

## 因果順序と固定する情報

`Generator`は同じbus／epochの相関結果、現在の正規化配分、現在も適格な親group handleを受け取る。
相関の終端と配分の終端、canonical hop、handleの重複・bus・epoch・世代、配分の正規化を検査する。
適格親は現在のgroup inventoryに含まれる必要がある。相関に支持されたtrajectoryには現在配分の行が必要である。

現在の相関からcomplete-linkで最終bundleを作る。親の初期割当には内部に保存した**直前hopの配分**だけを使う。
現在配分の最大値から過去の親を推定し直さない。前hopでresolved groupが唯一最大の場合だけ親を解決し、
同点・residual最大・未観測・すでに適格性を失った親は未解決とする。

持続中のキーごとに、現在支持される最大8 trajectoryのformer-parent bindingを保持する。
同じhandleが続く間は、そのキーのbindingを現在配分で上書きしない。
新たに支持されたhandleは、その直前hop配分からbindingを定める。支持を失ったhandleのbindingは持ち越さない。
一時的な配列位置はIDに使わず、並べ替え後もhandleで照合する。
bindされた対象親の正確なmember集合が変われば旧キーを止め、新規キーは現在から判定を始める。
mergeでは、別の親のmemberが同じbundleに追加されたという理由だけでは、対象二親のキーを止めない。

frame欠測またはhopの飛びで持続をリセットする。欠測後の最初のframeは、新規split／mergeに使える
直前hopの親配分を持たない。現在配分からのbirth判定は可能だが、欠測を挟んでカウンタを引き継がない。
bus／epoch切替時には新しいGeneratorを構築する。ここで旧epochからのgenerationを移植しない。

## birth・split・merge

birthは正確な最終bundleをキーにし、現在の`u_B >= 0.5`で候補化する。
splitは二つの最終bundleの全memberが同じformer parentを持つ場合に限る。
全cross-pairを確認して、全て既知かつ最大相関が0.2以下の場合に候補化する。
欠測pair一つだけでも判定を落とし、そのhopの持続を継続しない。

mergeは一つの最終bundle内にある二つのformer-parent subsetをキーにする。
二つの異なる最終bundleをmergeの前提にしない。
全cross-pairが既知かつ最小相関が0.8以上であることを確認する。
現在の支持量でreducerが順位を付けるが、それは固定したformer parentを置き換える処理ではない。

受理済み・棄却済みのキーはreducerから除かれ、bindingも対応するpendingキーだけを次hopへ残す。
必要条件を失った旧キーと、新規生成で同じ形になったキーを同一の継続として誤って扱わない。
入力エラーの際は内部の前hop配分と持続状態を更新しない。

## 21 mergeキー上限の補足

同じ親pairが複数の最終bundleにsubsetを持つ場合、各subsetを独立に持続させるだけでは
`C(7,2)=21`という上限の根拠が不足する。さらに、異なるhopで始めたキーが異なる親割当を固定し得る。
二つの4-member bundleと7親を使う独立Python反例では、二時点の割当から24個の異なるキーが生じる。
これは音源に関する知見ではなく、候補inventoryの構造的な反例である。

仕様9.2に、**全bundleを通じてunordered parent pairごとに一つの持続キーを保持する**規則を追加した。
現在のキーが固定member集合と必要条件を満たす間は、それを続ける。
続けられるキーがなければ、その親pairの候補のうち最小cross相関が最大のものを選び、
同点はソートしたmember union、親に付随するmember集合の辞書順で決める。選ばれたキーは1 hopから始まる。
この規則はデータに合わせた調整ではなく、有界実装のための明示的な工学規則である。

上記反例にこのselectorを適用した独立参照の候補数は、初期frameから0、12、15となる。
この反例そのものを追加のRust実行結果とは呼ばない。
Rust側は、複数bundleに同じ親pairがある条件、異なるscore、同点、入力順交換、
持続中に別subsetのscoreが上回る場合を直接検査し、別の864終端の独立参照とも照合する。

仕様変更はこの9行の追加だけで、その他のbyteは前段archiveと同一である。
新SHA256は`64a8608a8192292d12c3fec4e1fedfc23480ceb9fa288bbaed1bd640d0dcb757`。
20登録の`specification.sha256`を同期した。過去のsource archive、実行結果、対応SHAは書き換えない。
feature版30の既存layout・fit条件も変更しない。

## 検証と費用

独立参照は集合ベースのcomplete-link、辞書による固定binding・候補・持続状態、有理数の配分計算を使う。
36系列・864終端で最終bundle、候補キー、score、支持量、継続数、受理順、競合数、pendingのhop数と
全parent bindingを照合した。候補1,659件、継続753件、受理394件、競合244件が一致した。
入力は模擬の係数graphと配分snapshotであり、実音源のgroup回収を示すものではない。

別の接続検査では、実際の`Window`に互いに逆向きの二つのlog-envelope系列を入れた。
8組のpaired支持を満たすまではsplitを作らず、支持を満たしてから3 hop後に受理した。
その他、親の時刻、親同点への変化、親世代の喪失、全pairの確認、集合変化、欠測と時刻の飛び、
最大28 split／21 merge、birth併存、入力エラー時の状態不変を検査した。

Rust全体は826件通過・20件ignore・失敗0。今回の9検査は全て通過し、ignoreしたresource probeは
releaseで別途実行した。通常Clippyと、既存の二種類の警告だけを除外したall-target Clippyも通過した。

releaseでは100回warmup後、各条件で6,000回更新した。相関・配分入力の構築を計時外に置き、
complete-link、配分検査、支持量、親binding、キー選択、持続・順位・競合処理を計時した。
split＋birth条件はp99 28.32 µs、merge＋birth条件はp99 14.18 µsだった。
各条件の4,500 frameはそれぞれ36／22候補、残り1,500 frameは次の親snapshotを準備する候補0 frameである。
全6,000 frameが最大inventoryだったとは扱わない。Generatorは68,664 bytes、出力は58,256 bytesで、
これを全workerのRSSやstack peakへ読み替えない。

相関窓の算出、ridgeと配分の算出、NSGT、group mutation、全workerと実機出力はこのprobeに含まれない。
O04全負荷の合格ではない。残る実装はgroupの世代発行、親のsupersede、容量不足時の扱い、
観測支持による退役、beam／記憶との接続、実際の二bus経路への統合である。
M0の全入力登録、development尺度、実音回収、全O04、素材・分割・fit・予測安定性・収集条件も残る。
