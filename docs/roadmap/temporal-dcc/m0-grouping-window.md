# M0: 相関窓とcomplete-link bundle

2026-09-12。状態: group lifecycleの入力となるRust数値部品を追加。
M0全体の完了ではない。[sourceと結果](m0-grouping-window.json)に対応を保存する。
前段は[音群配分](m0-group-assignment.md)。今回も`cfg(test)`内の部品であり、
本番のridge・grouping・行動経路へ接続した結果ではない。

## 物理時間と相関

入力はgroup配分前のtrajectoryのlog-envelope `ell_i`、bus／epoch／generation handle、
canonical hop終端のsample番号である。原観測支持を持つ値だけを保存し、欠測値・飛ばしたhopを
時間的に詰めない。別generationの履歴を引き継がず、問い合わせ対象は最新frameで支持されたhandleに限る。

相関窓は直前0.25秒、epoch開始でclipする。少なくとも90%の既知paired sample支持と
8組のpaired hopを別々に満たす必要がある。窓端で一部だけ重なる最古hopは、重なったsample数
だけをcoverageへ数えるが、相関には一つの等重みpaired hopとして入れる。
half／double窓と4／16組の検査も登録した。sample単位の窓長は呼出側が確定して渡す。

Pearson相関は、二系列の平均と中心化した平方和・積和からf64で計算する。
大きな共通offsetで精度を落とさないよう、最初のpaired値を引いた座標で平均と二回目の中心化を行う。
エネルギー加重やgroup配分後のenvelopeへ置換しない。
片方の分散ゼロ、paired数不足、coverage不足では係数をunknownにする。
paired数とsample支持は別に残すので、係数unknownを支持ゼロや反相関へ読み替えない。

保存bufferは`ceil(window_samples/hop)`枠を初期化時に確保し、更新中に拡張しない。
上限128枠を超える設定、または窓内に最低paired数が収まらない設定は拒否する。
48 kHz／hop512では0.125／0.25／0.5秒が12／24／47枠となる。

## bundle形成

呼出側が選んだ現在の適格な最大8 handleをsingletonから始める。
二bundle間の全cross-member相関が既知かつ0.8以上の場合だけ結合候補とし、
その最小相関をscoreに使う。最大scoreを選び、同点はソートしたhandle列の組を辞書順で比較する。
毎回この規則を繰り返し、候補がなくなれば止める。

A–BとB–Cの相関が強くても、A–Cが閾値未満なら三者を一つにしない。
欠測pairも結合を禁じるが、反相関の証拠にはならない。
結果はソート済みhandle集合として返し、呼出側の一時slot番号を持続IDに使わない。
最大7結合と196 cached correlation readの上限を検査する。

## 独立参照と入力精度

独立した100桁Decimal参照で96窓を検査し、うち75窓が既知の相関を持つ。
別の集合ベース実装と480配置を照合した。3 memberでは、各edgeが
None／0.5／0.8／0.9の全64構成と全6入力順を試し、8 memberの96配置を追加した。
任意の係数graphはアルゴリズム検査用であり、実際の音源分離や人のgroupingを示さない。

最初の参照照合では、共通offsetが1e12のcase 13に不一致が出た。
JSONの数値として読んだ入力のうち296値が、参照側の元のbinary64と一致していなかった。
fixtureのwindow入力をIEEE754 bit表現で固定すると、演算式・期待値・1e-12の許容差を
変更せずに全条件が一致した。数値JSONは可読表示として残し、bit表現を正本にした。
旧fixture・生成器・失敗ログを保存し、参照関数のASTと全数値条件・期待値が不変と照合した。

## 検証範囲と残る工程

Rust全体は809件通過・18件ignore・失敗0。今回の9検査は、epoch clip、90%境界、
部分hop、欠測・世代変更、分散ゼロ、buffer再利用、非推移性、同点、容量、独立参照を含む。
通常のClippyと、既存の二種類の警告だけを除外したall-target Clippyも通過した。

releaseでは8系列の非ゼロenvelopeを変化させ、全28相関が支持される入力を各6,000回更新した。
push・相関計算・complete-linkを含むp99は、0.125／0.25／0.5秒窓で8.66／13.47／24.84 µs。
frame bufferは3,936／7,872／15,416 bytes、Window headerは104 bytes、相関結果は2,328 bytes、
bundle結果は2,064 bytesだった。これらはNSGT、他のlifecycle、配送、全worker、RSS、
実出力機器を含むO04の証拠ではない。

次は、former parentの固定、正確なmember集合に基づく三hop持続、birth／split／mergeの
proposal競合、group世代・退役・容量処理。callerによるtrajectoryの適格性判断、
linear-RMSとのdevelopment比較、実音接続、全O04、素材・分割・回収・予測安定性等も残る。
