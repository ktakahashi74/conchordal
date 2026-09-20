# M0: group世代・容量・観測退役

2026-09-12。状態: 受理proposalを音響groupの7 resolved slotへ反映するRust数値部品を追加した。
前段の[候補生成器](m0-proposal-generation.md)との接続を検査し、[sourceと結果](m0-group-lifecycle.json)を保存する。
`cfg(test)`内の実装であり、本番二bus経路、beam、記憶・occurrence ledgerの統合は未完である。
M0全体の完了とは扱わない。

## prepareとcommit

`prepare`は保存済みgroup参照から現hopの分率配分を計算し、現在のtrajectory記述子と
更新予定のgroup参照を所有する`Prepared`を返す。この時点ではlifecycle状態を書き換えない。
候補生成器はその配分を読み、group energyも同じ配分から計算する。
`commit`は同じhopの受理候補とgroup energyを受け取り、観測時計とgroup topologyを更新する。

準備元hop、配分のgroup inventory、終端、候補の終端・支持量・scoreを照合する。
重複消費する受理候補、不正なenergy、古いPreparedを拒否する。
世代番号枯渇を含むエラーでは、状態を部分的に更新しない。
受理候補の3 hop持続と順位は前段reducerの責務であり、lifecycleが未判定の候補を成熟させるものではない。

返却する現hop配分は、lifecycle前のgroup handleと重みをそのまま保持する。
新しい子groupに同じhopを再配分しない。したがって新groupが初めて現hopの音を聴いたという
二重の支持・occurrence creditを、この部品が生成することはない。
beamやoccurrence ledger自体はまだ持っていないため、これだけでそれらの会計全体の正しさは証明しない。

## 世代と参照

bus／epoch内のresidual group handleを世代1に固定し、resolved groupへは世代2以降を単調に発行する。
slot番号とidentityを分け、retire・容量eviction後のslot再利用でも旧handleを復活させない。
birthは子一つ、splitは二つ、mergeは一つを新しい世代で作る。
子の参照は正確なtrigger member集合から作り、各memberの`r_jg=1`、観測終端は現hopとする。
この参照を使う配分は次hopから始まる。

nonsuperseded groupはprepareで作った現在の正配分memberへ参照を更新する。
split／mergeを採択した親は、現hopでrefreshする前の参照に戻してineligibleにする。
親はdormant slotまたは返却するsuperseded／retired記録に残り、子との平均表現にはしない。
返すのは音響groupの参照snapshotであり、旧local beamやcommitted episodeの保存・検索は次段の責務である。

## 容量と失敗

splitは子二つ分の置き場を確保できるか、親をsupersedeする前に確認する。
free slotに加えて、dormant slotと、今回supersedeする親slotを置換候補に数える。
不足なら`Capacity`を記録し、親をineligibleにせず、世代番号も消費しない。
birthも同じ容量規則に従う。mergeでは二親をsupersedeした後の一子分を確保する。

実際の配置は空slotを番号順で使い、不足分だけdormant slotをevictする。
優先順位は最後に閾値より大きな既知energyを持った終端が古い順、同点はgroup handle順である。
新規seedはその終端を初期値に持ち、次hopから観測でactive／dormantを更新する。
capacity retirementは計算上の損失として返し、観測による退役と区別する。
容量棄却で現hopの固定配分を再計算せず、新しいresolved groupは作らない。

観測退役を先に評価するので、そのhopで親が退役した候補は`ParentUnavailable`となる。
有効な別の候補は引き続き処理する。どのslot／親が消えたかを診断記録に残す。

## 観測時計

各groupについて既知の低energyを示すcanonical hopだけ、hop長をinactive sample数へ加える。
正のactive evidenceならカウンタを0に戻す。energy unknownはactive／dormant状態と既存カウンタを保つ。
時刻が飛んでも、後から届いた一hop分を超えてinactive時間を足さない。
superseded groupは以後の配分対象にならず、既知のinactive evidenceで同じ時計を進める。

閾値到達でslotを退役させ、回復時は新世代を作る。基準2秒と1／4秒の数値検査を行った。
`retirement_samples`と`inactive_energy_max`は明示的な設定である。
energyはgroupへ配分したmono mean-squareの単位で渡す。検査の`1e-6`は模擬fixtureの条件であり、
development音源から適合・固定した本番のlow-energy閾値ではない。この尺度とcallerのenergy接続は残る。

## 検証

独立したPython slot／観測時計参照と12系列・768終端を照合した。
birth 541件、容量棄却15件、退役468件が一致し、全slotの世代・trajectory・dormancy・
最後のactive終端・inactive sample数と次世代番号を確認した。
この参照はbirth・退役・evictionを扱う。split／mergeと参照更新の算術を別の独立参照で検査したとは言わない。

Rustの直接検査は、birth、split／mergeの新世代と親参照、8番目のbirthの棄却、
二子を置けないsplitでの親維持、dormant eviction、1／2／4秒退役、欠測、回復、閾値境界、
親の同hop退役、古い準備・不正energy・世代番号枯渇時の状態不変を含む。

配分、実際のPearson窓、候補生成器、lifecycleを接続した模擬envelope traceも検査した。
8組の支持を用意し、正相関から逆向きのenvelopeへ移る条件で、100 msにbirth、140 msにsplitを受理した。
各成立hopの配分は旧groupを参照し、子groupは次hopから配分対象になった。
これは連続した数値部品の因果順序の検査であり、実WAVのgroup回収やsource identityの検証ではない。

最初の参照更新fixtureは0.1 octave／10 msのglideを与え、次には離れた二子への両方の配分を期待していた。
どちらも指定したcontinuity尺度では支持されなかった。glideを0.001 octave／10 ms、二子の間隔を
0.002 octaveへ修正し、意図した有効支持の範囲で同じ期待値を検査した。
実装の距離式・閾値・参照更新順は変えていない。失敗ログは保存した。

resource probeは、7 group × 8 member × 8 trajectory × 2 slopeの配分と通常commit、
別に3件のsplitから6子を作り6件capacity retirementを生じるcommitを測る。
後者は同じ準備済みfixtureのreplayで、fixture reset・prepare・候補生成を計時外に置く。
これらはNSGT、proposal生成、bin別energy生成、beam／context保存、全workerと実機出力を含むO04ではない。

全体のRust検査は835件通過・21件ignore・失敗0。今回の9検査を含み、ignoreしたresource probeは
releaseで別途通過した。通常Clippy、既存の二種類の警告だけを除外したall-target Clippyも通過した。
各probeは100回warmup後に6,000回を測定した。prepare＋通常commitは全て896距離評価を含み、
p99 25.15 µs、最大46.12 µsだった。3 splitのcommitはp99 2.82 µs、最大6.83 µsだった。
Lifecycleは6,280 bytes、Preparedは10,840 bytes、返却結果は23,640 bytes。これらの型サイズは
worker全体のRSSやstack peakの測定ではない。

残る実装は、beamの初期状態、旧contextとoccurrence creditの所有、ridgeからgroupまでの実音接続、
採用するdevelopment尺度の固定と全O04である。M0の全入力・比較、実音回収、素材・分割・fit・
予測安定性・収集条件も引き続き未完である。
