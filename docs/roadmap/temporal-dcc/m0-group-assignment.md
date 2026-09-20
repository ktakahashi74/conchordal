# M0: trajectoryから音群への配分

2026-09-12。状態: O03／O09のRust数値部品を追加。M0全体は未完了。
[sourceと検証結果](m0-group-assignment.json)を保存する。
この音群部品は[ridge数値準備](m0-ridge-preparation.md)と同じく`cfg(test)`内にあり、
development尺度の実測固定、本番接続、全group lifecycleはまだない。

## 保存memberとの照合

各音群は最大8個の保存memberを参照する。memberはtrajectoryのhandle、原観測sample時刻、
log2周波数、最大2傾き、log-envelope、各座標の欠測、および保存時の所属重みを持つ。
最大7音群に現在の8 trajectoryを照合し、残差を含む8列へ配分する。
残差trajectoryはpitchがなくてもenvelopeで照合できる。

距離はridgeと共有する数値関数を使う。異なる音群・memberの保存時刻を同じと仮定せず、
それぞれの原時刻から外挿とsecantの可否を求める。各memberの最大2傾きから最小距離を選び、
距離1以内なら`r_jg * exp(-distance^2)`を候補にする。音群内ではその最大値を採用する。
member数で平均したり、周波数重心へ縮約したりしない。同点の参照は小さいmember handle、
続いて小さいslope indexを選び、選んだ原時刻と距離も出力へ残す。

残差のraw重み`exp(-1)`を加えて一度だけ正規化する。
同じscoreの音群は同じ分率を受け、片方を勝者へ変換しない。
使える参照がない観測済みtrajectoryは残差100%となる。一方、現在のtrajectoryが欠ける場合、
または周波数とenvelopeが両方欠測の場合は、所属行全体をunknownにする。

## 更新順序と境界

すべてのtrajectoryの配分を保存済みmember集合に対して計算してから、音群の参照を更新する。
適格な各音群に現在の正の所属重みがあれば、そのmember集合と新しい重みで参照を置き換える。
正の所属がなければ以前の参照を維持する。superseded等の非適格な音群は照合にも更新にも使わない。

`Group::seed`は、birth／split／mergeの呼出側が選んだ正確なmember集合を重み1で保存する。
保存したものは次hopの参照であり、同じ時刻の参照を読もうとすると拒否する。
新しいgroup generationの発行、proposal競合の解決、相関・持続条件・退役等の全lifecycleは
この部品の機能ではない。dispatcherによるframeの一度だけの配送と版固定も、本番接続時の条件として残る。

配分snapshotには、計算時のtrajectory handle、group handleと列の対応、frame終端を保存する。
後から同じslotへ別の音群を置いても、過去の配分の意味を変えない。
group／memberのbus・epoch・ID重複、非有限値、重み、未来または非canonicalな原時刻を、
参照更新より前に検査する。途中までの参照更新は残さない。

bundleの未配分率は、選んだmemberの有効な所属行だけから
`sum(residual_weight) / sum(all_group_weights)`を求める。
分母ゼロはunknownであり、他のtrajectoryやbus全体のエネルギーで分母を増やさない。

## エネルギー配分

Log2 binのtrajectory所属は、接続済みの[frame内ピーク抽出](m1-trajectory-input.md)と
同じ2-bin半径・最近ピークの関数を共有する。全所属が既知なら、各binについて
`E_gb = E_bus * s_b/sum(s) * a_owner,g`を計算する。
8音群を合計するとそのbinのエネルギーを回復し、全bin・音群の合計はbus energyを保存する。

8本の出力scanは呼出側の事前確保bufferへ書く。入力と全出力の長さをLog2Space境界で
hard assertする。mono energy不明、または正エネルギーのbinのowner所属がunknownなら、
音群エネルギー全体の結果をunknownとして返し、bufferを更新しない。
呼出側は結果の支持を確認せずに古いbufferを読んではならない。

既知のmono energyゼロは全音群の既知ゼロとなる。
正のmono energyに対してスペクトル質量ゼロなら、仕様のfallbackとしてscalar energyを
残差音群へ保持し、スペクトル形状を欠測にする。この場合のゼロで埋めたscanを観測済み形状やfluxに使わない。

## 検証と資源

独立した100桁Decimal参照の64配置・419有効所属行を照合した。
重み付きmember最大値、異なる原時刻、両傾き、欠測、非適格group、重み・参照IDを含む。
fixtureの再生成はbyte一致した。既存のridge 160条件、ピーク配分106条件も維持する。

Rust検査では、候補なしとmissingの区別、等分配、重心／平均を使わない反例、全行計算後の更新、
seedの次hop限定、superseded・未観測時の参照維持、envelope-only、距離閾値、
残差raw重みの半分／倍、bundle未配分率の0.5境界、最大896比較を確認した。
エネルギーのbin別・全量保存、unknownとゼロ、入力／出力scan境界も検査する。

今回の全体Rust検査は800件通過・17件ignore・失敗0。通常のClippyと、既存の二種類の
警告だけを除外したall-target Clippyが通過した。追加の警告除外はない。

資源probeは、周波数・envelopeが変わり、傾きと残差が非ゼロの構成入力を使う。
最大配分構成では6,000更新で5,376,000距離評価となり、入力検査・最大値選択・正規化・参照更新を含める。
別probeは129／513／2049 binの8本の出力scanを再利用し、bin別エネルギー配分を各6,000回測る。
詳細な時刻・byte数・元のログはJSONに保存する。test harnessの並列性や他の処理を隔離した測定ではない。
最大構成の配分・参照更新p99は31.161 µs、2049-binのエネルギー配分p99は18.36 µsだった。
保存memberのOption枠は104 bytes、Groupは864 bytes、Assignmentは4,080 bytes。
8本の2049-bin出力は131,136 bytesであり、これらをqueueやRSSを含む全体費用と扱わない。

この数値部品の費用は、NSGT、全lifecycle、beam・関係記憶、配送・copy・allocator・実機出力を含む
O04の60秒warmup・600秒同時負荷を証明しない。機関手続き、素材・分割、回収・予測安定性等も未達である。
