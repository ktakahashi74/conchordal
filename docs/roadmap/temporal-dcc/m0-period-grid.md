# M0: 観測されたaccent pairから周期候補を作る

2026-09-12。Rustの四hop detector、順序付きaccent ledger、incrementalなperiod pair-gridを
数値経路として接続した。`cfg(test)`内の実装であり、本番worker、全group lifecycle、grouping提案器へは
まだ接続していない。[sourceと検査の記録](m0-period-grid.json)を保存する。M0全体は未完了である。

## 元の観測支持

各groupのStreamは、observedなcanonical hopの取得sample数を、associationが既知の場合だけ累積する。
同じsampleをNSGT窓の重なりによって複数回数えない。部分取得は取得分だけを足し、欠測・association不明は
足さない。group generationを変えたら累積をリセットし、親の値を継承しない。
正のmono energyと未知spectral shapeの組では、resolved groupのassociationも不明として渡す。
接続検査では、修正前にこの一hop分の512 samplesを過計上した。修正後は元のcounterと一致する。
既知ゼロenergyの完全な観測は、この欠測とは区別する。

accentには中央hop終端での累積値`observed_prefix`を付ける。検出を確定する右側hopや配送時刻の値で
置き換えない。同じIDのcounterを変えた再送もprovenanceの矛盾として拒否する。
counterはevent時刻以下、eventまでの三つの完全取得raw区間以上でなければならず、
隣接admission間の増分は非負で、その原event間のsample数を超えてはならない。

pairのcoverageは二つの元終端のcounter差を経過sample数で割った値である。
整数の比較で90%以上を判定する。同時刻の二点は正の周期を支持しない。
後でgapが生じても、既に保存したpairの支持を再解釈しない。
既知の無音は取得支持を保ち、未観測の音声を既知の無音へ補完しない。

既存Pythonの局所detector／ledger参照はこのglobal prefixを計算しない。
新しい独立period fixtureは元の整数counterを入力として使い、RustのStreamからのcounter生成は別の連結検査で確認する。
参照の共通部分を検査したことと、この追加provenanceまで旧参照が供給したことを混同しない。

## 固定cacheと周期候補

基準は0.125–4秒、1/48 octave刻みの241 binsである。最大128 accentsの8,128 pairsを保持する。
各pairを32-byteの実layoutへ保存する。二つのu16 slot、base bin、support flags、五つのf32寄与、
四byteのpaddingを含み、paddingと範囲外の寄与はゼロにする。

三角kernelをf64で評価し、各寄与を一度だけf32へ丸め、その値の正確なf64変換を加算する。
期限切れ・容量evictionでは保存した同じ値を減算し、slot再利用前に全incident pairを一度だけ除く。
新しいaccentとのpairだけを計算し、保持中のpairを再評価しない。
1,024採用ごとに、保持しているcacheからsumを再構築する。

f64加減算の小さな負のdriftは診断値として残し、正部分を正規化する。
非ゼロ寄与を持つcacheが無くなったらgridを明示的にゼロへ戻し、残差から架空の周期を作らない。
この空cacheへの遷移時のclearは、定期的なrebuildと区別する。

正のplateauが外側の隣接値を厳密に超える場合だけpeakとする。plateau内では最小periodのbinを選び、
端では存在する側の隣接値だけを比較する。全面uniformにはpeakが無い。
supportの降順・periodの昇順で並べ、1/24 octave以上離れた最大八個を保持する。
最初のpeakが`P_best`であり、未知時にperiodを補作しない。等間隔accentの複数周期候補を、唯一の知覚拍とは解釈しない。

## 数値比較と反例

独立Python参照は全241 binsでkernelを評価し、各比較点で保持pairを再集計する。
四つの容量条件に2,952入力を作り、512／1,024／2,048採用ごとのrebuildを含む273比較点を照合した。
通常fixtureではf64加減算driftの最大絶対値が約7.1e-15、f32丸めによる全f64再計算との差が約1.91e-6で、
peak順序は一致した。これらの最大値はその有限入力内の値であり、一般的な誤差上限ではない。

近接同点の別fixtureでは、weight 0.5と0.500000001がf32で同値となった。
全f64なら最上位は1秒、丸めたcacheでは同点規約により0.5秒となる。
また、極小weightでは積のf32 underflowによってperiodがunsupportedとなり得る。
この変化を許容済みの音楽的・認知的安定性とはしない。実波形、全grouping admission、下流fitでの影響は残る。

## 費用と残務

peak候補は最大121、separation検査は最大121×8である。
実装が行う正規化の二passと空cache clearを数え直し、仕様9.2の保守的なbin訪問上限を
一group・十hopあたり101,451、七groupで710,157へ修正した。peak pass／sortは別勘定である。
同節の旧50 ms表記も、既存preflightの40 ms p99 worker予算へ揃えた。予算の緩和ではない。

最終sourceの全Rust検査は866成功・0失敗・24 ignoredである。今回追加した通常検査は八件で、
releaseでもperiod七件とassociation回帰一件が成功した。Pythonの既存accent／descriptor検査は計53件成功した。
通常clippyは`-D warnings`で成功し、全targetでは既存の`unnecessary_cast`と`manual_is_multiple_of`だけを除外した。
format、fixtureのbyte一致、Zola buildも確認した。

release単体測定は各容量で飽和後6,000更新を行い、ledger、pair更新、peak生成、Viewコピーを含む。
rebuildは頻度が低いため、呼出し全体のp99とは別に最大値を示す。全pairの一括expiryも別に測った。

| accent容量 | 更新p99 (µs) | rebuild呼出し最大 (µs) | 全expiry (µs) | pair payload (bytes) |
|---|---:|---:|---:|---:|
| 64 | 3.83 | 7.64 | 7.07 | 64,512 |
| 128 | 3.53 | 15.73 | 24.14 | 260,096 |
| 256 | 5.34 | 62.731 | 100.971 | 1,044,480 |

Estimatorのinline本体は16,776 bytes、返却Viewは2,336 bytesである。表のpair payloadに加え、
accent保存は順に9,216／18,432／36,864 bytes、slot順序は128／256／512 bytesを使う。
これらは型と確保payloadの集計であり、プロセスRSSやworker全体のstack計測ではない。
DSP、beam、二bus worker、64 Voice、実audio deviceを含む全O04の合格とはしない。

構築後のpair更新・expiry・peak生成は既存bufferを再利用し、Viewは固定サイズの値として返す。
period範囲・grid解像度・物理windowの全感度比較、下流の再fit、grouping／cyclic-word提案、
beam／context所有、二bus／64 Voiceを含む全O04、本番接続、全M0の人・データgateは未完了である。
