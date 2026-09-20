# M0: proposalキーの持続判定と競合処理

2026-09-12。状態: 外部から与えた候補キーに対するRust参照部品を追加した。
M0完了や本番の音群生成ではない。[sourceと結果](m0-proposal-persistence.json)を保存する。
前段は[相関窓とcomplete-link](m0-grouping-window.md)。今回も`cfg(test)`内に置く。

## 入力と保証の境界

`Key`はkind、former-parentのbus／epoch／generation、二つまでの正確なmember集合を保持する。
member集合はtrajectoryのgeneration handleで表し、一時的な配列位置を使わない。
birthは親なし・一集合、splitは親一つ・二集合、mergeは異なる親二つと各親に付随する集合である。
splitの集合順、mergeの親と付随集合の組の順を正規化する。mergeのA/Bを交換しても同じキーだが、
親A・Bへのmember割当を交換したキーとは区別する。重複member、空集合、異なるbus／epochを拒否する。

`former_parent`は与えられた正規化配分の唯一最大のresolved groupを返す。
resolved同士の同点、residualとの同点、residual最大は未解決とする。
キーを作る時点では**前hopの配分**を渡す必要がある。今回のhelper自体はsnapshotの時刻を保持しない。
固定したキーを再利用する間、現在の配分が変わっても親を上書きしない。
親・member世代や集合を変更した別キーは、新しい持続判定を始める。

候補生成器はまだ実装していない。呼出側が、前hopの配分、最終bundle、親の存続と適格性を照合して
キーを生成・継続する責任を持つ。splitでは二つの最終bundleの全memberが同じformer parentを持つこと、
mergeでは一つの最終bundle内に二つのformer-parent subsetがあることを保証する必要がある。
cross-pair全数が既知であることと、その最大値（split）／最小値（merge）の計算も呼出側の責任である。
今回の数値APIはその極値を受け取るため、単独でこれらの条件を証明するものではない。

## 支持量と持続

現在の配分snapshotから、各distinct memberについてresidualを含む最大の正規化配分を一度ずつ加算する。
member handleで行を検索するので、配分のrow順を変更しても支持量は変わらない。
birthの`u_B`はresidual配分の総和を全配分の総和で割り、0.5以上で候補に残す。
splitは最大cross相関が0.2以下、mergeは最小cross相関が0.8以上で残す。
birthの`u_B`をsplit／mergeに適用しない。必要な相関unknown、member行なし、親handleなしでは候補を落とす。
不正な正規化配分や相関範囲は入力エラーとする。

`Tracker`はcanonical hop終端、完全観測フラグ、その時点で条件を満たす候補だけを受け取る。
候補に配分snapshotの終端を保存し、別時点のsnapshotの再利用を拒否する。
同じキーが連続する完全観測hopに現れた場合のみカウンタを増やす。基準は3 hop。
欠測frame、飛ばしたhop、必要条件の不成立による候補消失、member／親世代の変更は継続しない。
持続長は明示的な引数で与え、1／3／6 hopの部品検査を行う。development上の採用判断とは別である。

## 順位と競合

3 hopに達した候補を、現在の支持量の降順、merge→split→birth、ソートしたformer-parent handle、
ソートしたmember handleの順で処理する。同一union・親列まで等しい供給キーには、付随集合の順で
決定的な最終tie-breakを行う。通常の最終bundleから生成される候補では、この追加tieは発生しない。

受理済み候補とtrajectoryか親groupを共有する後続の成熟候補は棄却し、そのカウンタを消す。
未成熟のキーは成熟候補の競合処理に参加しない。受理キーもカウンタから取り除く。
次hopで再び同じキーを供給した場合は1から始まる。
ここでの「受理」はreducerの結果であり、group slot確保や音群生成の成功ではない。
実際のgroup mutation、容量不足時の扱い、generation割当・親のsupersede・退役は次段に残る。

前回／今回のキーは57枠（birth 8、split 28、merge 21）の固定配列を使う。
候補全体のdistinct trajectoryは8、親groupは7までとし、各kindの上限、重複キー、
異なるbus／epoch、不正な時刻を状態更新前に拒否する。非重複の受理結果は最大8件となる。
更新中のheap確保はない。固定配列のコピー、線形検索、sorting、検査もresource probeの範囲に含む。

## 検証と残る作業

独立Python参照は、有理数による配分・支持量、辞書によるカウンタ、集合による競合判定を使う。
24系列・576終端で、条件通過、score、支持量、受理順、棄却件数、残存キーとhop数を照合した。
受理684件、競合棄却1,900件が一致した。別のRust境界検査は、親同点、A/B交換、handle変更、
閾値境界、row順、現在配分の変化、欠測、候補消失、時刻の飛び、親だけを共有する競合、
kindより支持量が優先される場合、容量と不正入力時の状態不変を含む。

57枠を全て使う検査は、外部から供給するキーの容量上限の検査である。
この全キー集合が一つの実際の相関・former-parent配置から同時に生成されると主張しない。
release probeも、供給済みキーの持続・順位・競合処理だけを測定する。100回のwarmup後に6,000回更新し、
median 15.62 µs、p99 19.02 µs、最大44.601 µsだった。Trackerは49,736 bytes、結果は6,928 bytes、
候補一件は864 bytes。更新中の前回・今回・次回の固定配列とコピーは計算時間に含めるが、
これらの型サイズを全workerのRSSやstack peakと読み替えない。
候補生成、支持量計算、相関窓、NSGT、音群変化、全worker、実機出力を含むO04の合格とは区別する。

Rust全体は817件通過・19件ignore・失敗0。ignoreした今回のresource probeはreleaseで別途通過した。
通常のClippy、既存の二種類の警告だけを除外したall-target Clippy、format検査も通過した。

候補生成器による親の初期固定と継続、bundleと全cross-pairの検査、groupの容量・generation・退役、
実音と本番経路への接続が残る。M0の全O03/O04/O09、development比較、素材・分割・回収・予測安定性等も未完である。
