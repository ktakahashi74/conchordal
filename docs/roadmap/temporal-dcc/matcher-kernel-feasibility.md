# M0 matcherのnative入力・anchor・DTW／経路復元実験

状態: 数値kernelの実験。既存Python参照、設計メモ、演奏経路は変更していない。
全O04およびM0の終了条件は未達である。

`scripts/temporal_matcher_kernel.rs`を独立した共有libraryへコンパイルし、
`scripts/evaluate_temporal_matcher_kernel.py`から呼ぶ。追加のPython packageは使わない。
Rust側へ移したのは、coarse anchor探索、元時刻でのband検索、座標残差、二行DPとparent記録である。
通常のfloat入力の検証と転送もnativeへ移した。特殊な型の処理、math.fsumと最終結果構築はPython側に残る。最良anchorの選択・残差標本もnative側で扱う。
全query比較ではnative driverを使い、検索適格性・cache・順位・cutoff・refinement・全出力を既存matcherと照合する。
通常のimportで使う既存matcherやquery schedulerの経路は変化しない。

## DTW部分の規約と検査

- cue／referenceそれぞれ最大128 knot、座標数10。既定band半径16とunbanded対照を扱う。
- 元時刻で最寄りの位置を選び、同距離なら早い位置を選ぶ。index比による時間置換は行わない。
- 共通して観測された座標だけを使う。全欠測のpairは費用0かつ証拠0。
- 費用の完全同点は対角、挿入、削除、早い終点の順に選ぶ。f64の演算順序を維持し、fast-mathを使わない。
- 700件の無作為なmask・不等間隔・変形・penalty・band、全1,024 mask、最大寸法とscratch再利用、
  境界・欠測・gap・極端な有限値・不正入力、queryの候補同点とcacheを追加9検査で照合した。
- 全出力のfloatをbinary64へ変換し、型・bit・経路・診断・検索順位・元時刻を比較する。
  既存33検査もnative DTWを通して実行した。ここには独立した短経路の全列挙との50件の比較が含まれる。
- 最初の失敗は、全欠測なら挿入を選ぶというテストの誤った期待値だった。対角費用0、挿入費用1なので対角を選ぶ。
  PythonとRustは最初から一致しており、期待値だけを修正した。失敗ログを保存した。

## 最初のDTW移植の測定結果

| 有効座標 | Python参照の中央値 | native DTW経由の全query中央値 | 短縮率 |
|---|---:|---:|---:|
| 1 | 269.573 ms | 152.566 ms | 43.4% |
| 10 | 426.318 ms | 189.972 ms | 55.4% |

結果全体の保存とbinary来歴を追加した最終driverで再測定した値を採用した。
初回の147.463／185.216 msも保存し、速い値への差替えはしない。
追加9検査、既存33検査のnative実行、Python全837検査（31.111秒）、Rust746通過・12件ignoreを確認した。
登録参照は388検査のままで、この実験を本番または既存matcherの置換とは扱わない。
両入力とも40 msを超え、当時はanchor探索、入力検証、Pythonとの値の変換が残る費用だった。

## 先行するanchor／DTW版

| 有効座標 | Python参照 | 旧DTW版 | 新anchor版 | 旧DTW版からの短縮 |
|---|---:|---:|---:|---:|
| 1 | 270.968 ms | 147.211 ms | 74.521 ms | 49.4% |
| 10 | 425.345 ms | 184.874 ms | 94.553 ms | 48.9% |

続いてcoarse anchor探索を一括するRust kernelと、記述子を一括変換するadapterを追加した。
8,192 anchorの中央値・残差を評価し、詳細診断は各episodeの最良anchorで既存参照から生成する。
元時刻、mask、f64加算順、中央値の比較数、変形・同点・候補数は維持した。
16実験検査には500組の全anchorと90本の全queryを含み、既存33検査もnative経由で通過した。
全Python844検査（29.404秒）とRust746通過・12件ignoreを確認した。
9回の順序無作為化比較で、全queryのPython参照／旧DTW版／新anchor版の中央値は、
1座標270.968／147.211／74.521 ms、10座標425.345／184.874／94.553 msだった。
旧DTW版から49.4%／48.9%短縮したが、40 msは超える。54回の結果比較では型・f64 bit・経路・診断が一致した。
固定ctypes payloadは53,008 bytes、二行DP配列は2,064 bytes、anchor標本配列は128 bytesである。
Python object・変換中のlist・結果・allocator・ABI由来のstack増分・RSSは別途必要である。
初回候補は未選択anchorのRMS overflowを隠し、巨大な時刻差で参照が残すNaN診断を拒否した。
極端な算術では元参照へ戻す経路を加え、両反例を保持した。判定はf64上限と最大8標本から導く数値保護であり、認知的な閾値ではない。
初回auditのfixtureが一方の差を区別しなかった履歴も保存した。NaN診断の再現は、その入力を本番で許容する判断ではない。
既存Python参照・feature版29・設計メモ・演奏経路は同一。次は残る入力検証・変換と、実機・両worker・64 Voiceの全O04へ進む。
根拠は`m0-native-anchor-20260911`。fit、人のgateを含むM0全体は未完了である。


## 先行する入力検証・直接転送版

| 有効座標 | Python参照 | 旧anchor版 | native検証＋Python転送 | native検証＋直接転送 |
|---|---:|---:|---:|---:|
| 1 | 272.587 ms | 73.531 ms | 63.259 ms | 23.433 ms |
| 10 | 426.370 ms | 95.701 ms | 86.193 ms | 37.477 ms |

入力の数値検証とCPythonからの直接転送をnative実験へ追加した。通常のfloat／list／dictを
GIL保持のPyDLLで固定bufferへ転送し、特殊な型とエラー順序は既存Python経路で扱う。
数値kernelはPython非依存のまま、転送APIだけを`python-binding`で有効にする。
26実験検査は1,200件の検証比較、320件の転送byte比較、型fallback・参照解放・buffer末尾を含む。
既存33 matcher検査と全Python854検査（32.151秒）、Rust746通過／12件ignoreも確認した。
同じ256 episode×128 knot・cue128・16候補を保つ9回の四版比較で、旧anchor版→直接転送版の
全query中央値は1座標73.531→23.433 ms、10座標95.701→37.477 msだった。
72回の全出力比較で型・f64 bit・経路・診断が一致した。単体中央値が40 msを下回っても、両worker・64 Voiceのp99合格は未証明である。
固定ctypes payloadは67,344 bytes。DP配列2,064、anchor標本128、検証metadata56、転送pointer配列144 bytesは個別の配置であり、
Python object・結果・allocator・全stack・RSSはこの合計に含まれない。既存Python参照・feature版29・設計メモ・演奏経路は維持した。
根拠は`m0-native-input-validation-20260911`。consumerの処理量、実際の生成入力と同時負荷、fit、人のgateは残る。

profile上の主な残存費用はsubsequence_dtwの経路復元・診断構築である。
profileには計測負荷があるため、表の実時計中央値とは分ける。

同じ入力を用いたtail診断では、各caller 10回warmup後に200回を測定した。
1座標のp99は単独23.460 ms／2 caller 57.322 ms、10座標は単独49.625 ms／2 caller 87.469 msだった。
2 callerは独立scratchを持つPython threadで、各200組の呼出しが重なった。GILは共有する。
全1,260回で出力の型・f64 bit・経路・診断を照合した。照合とbarrier待機は計測区間の外に置いた。
この条件でも40 msを超えるため、単体中央値を根拠に実時間採用しない。
次は残る経路復元・診断計算と構築をnative側へ移し、同時呼出しを再測定する。
これは200回の合成入力診断であり、実際の両worker・64 Voice・6,000周期のO04ではない。

## 先行する経路復元・診断版

| 有効座標 | Python参照 | 直接転送版 | 経路復元版 |
|---|---:|---:|---:|
| 1 | 287.025 ms | 23.744 ms | 16.484 ms |
| 10 | 442.111 ms | 38.226 ms | 21.222 ms |

経路復元、座標別の逆順残差加算、隣接pairのmotion／interval診断をnative側へ移した。
`math.fsum`と最終結果構築はPython側に残し、`**2`の丸め・overflow、特殊gap型の遅延評価は元参照と照合する。
追加検査には乗算とpowが異なる固定反例と320個の広い桁、例外後の再利用、欠落gapの参照順序を含む。
最初のconfig配置ずれは修正し、五構造体のサイズをbuffer使用前に照合する。旧候補と失敗ログを保存した。
全30実験検査、既存33 matcher検査、全Python858検査（32.917秒）、Rust746通過／12件ignoreを確認した。
9回の三版比較で、入力転送版→経路復元版の全query中央値は1座標23.744→16.484 ms、
10座標38.226→21.222 msだった。54回の全出力比較で型・f64 bit・経路・診断が一致した。
単独／2 callerのp99は1座標17.303／42.039 ms、10座標32.859／50.261 ms。
各callerのwarmup10回＋200回、全1,260回の出力一致を確認したが、2 callerの40 ms予算は未達である。
固定ctypes payloadは73,152 bytesで、経路1,536・診断配列2,048・座標残差と件数120 bytesをこの内数として計上する。
Python object・結果・allocator・stack・RSSと全O04は別に残る。既存参照・feature版29・設計メモ・演奏経路は同一。
次は残るanchor診断を扱う。根拠は`m0-native-traceback-20260911`。実生成入力・consumer処理量・fit・人のgateも未達である。

このhostのpow参照は[CPythonのfloat_pow](https://raw.githubusercontent.com/python/cpython/3.14/Objects/floatobject.c)を確認した。
出力中のmath.fsumは元と同じ標本順でPythonへ渡し、単純和へ置換しない。

## 先行する最良候補診断版

| 有効座標 | Python参照 | 経路復元版 | 最良候補診断版 |
|---|---:|---:|---:|
| 1 | 269.281 ms | 15.650 ms | 13.588 ms |
| 10 | 426.453 ms | 20.710 ms | 18.567 ms |

最良anchorの選択・比較件数集計・残差標本をnative側へ移した。最良候補だけの標本を元順序で保持し、
powの丸めとPythonのmath.fsumを維持する。8,192 anchor・16候補・62変形と全診断は削減していない。
500組の全anchor比較を最良候補・総比較数・境界件数・RMSへ拡張し、同点・全欠測・再利用・元順序を検査した。
32実験検査、既存33 matcher検査、全Python860検査（32.544秒）、Rust746通過／12件ignoreを確認した。
最終9回の三版比較で、経路復元版→今回版の全query中央値は1座標15.650→13.588 ms、
10座標20.710→18.567 ms。54回の全出力比較で型・f64 bit・経路・診断が一致した。
単独／2 callerのp99は1座標13.594／28.807 ms、10座標30.016／44.354 ms。
各callerのwarmup10回＋200回、全1,260回の出力を照合した。1座標の2 callerは40 ms以内だが、10座標は未達である。
coarse呼出しの引数追加に合わせABI照合の入口をv2とし、旧adapter／新libraryと逆組合せの双方を呼出し前に拒否した。
固定ctypes payloadは73,296 bytesで、最良候補診断144 bytesを含む。anchor用の元標本・最良標本・sort配列は計320 bytes。
Python object・結果・allocator・全stack・RSSは別である。既存参照・feature版29・設計メモ・演奏経路は同一。
次はquery単位で転送と呼出しを束ね、所有・元cut・拒否順序とコピー費用を検査する。
根拠は`m0-native-anchor-diagnostics-20260911`。実生成入力・consumer処理量・全O04・fit・人のgateは残る。


## 先行するquery一括処理版

| 有効座標 | Python参照 | 最良候補診断版 | query一括処理版 |
|---|---:|---:|---:|
| 1 | 270.524 ms | 13.811 ms | 11.355 ms |
| 10 | 426.467 ms | 18.482 ms | 15.380 ms |

query単位の所有bufferと三つのnative呼出しを実装した。全query中央値は直前版から
1座標13.811→11.355 ms、10座標18.482→15.380 msへ短縮した。
2 callerのp99は30.454／39.974 msだが、10座標は400回中4回が40 ms超過、最大44.548 msで、余裕は十分でない。
固定ctypes bufferは6,997,808 bytes／caller、最大queryの数値転送は5,526,528 bytes。
元ID・世代・cut・scalesと全出力を維持し、54回の全query比較、1,260回の同時呼出し比較、42回の資源監査で照合した。
38実験検査、既存33 matcher検査、全Python866検査、Rust746通過／12件ignoreを確認した。
入力転送は別の段階計測で6.286／7.371 msを占める。次は所有済みpacked表現からの転送を検討する。
根拠は`m0-native-query-batch-20260911`。登録参照・feature版29・設計メモ・演奏経路は維持し、
実生成入力・consumer処理量・全O04・fit・人のgateは未完了のまま残す。

一queryにつき、GIL保持の一括転送、GILを解放する粗探索、GILを解放する候補精査／DTWの三呼出しで処理する。
8,192 anchor、16候補、62変形と全診断を維持した。順位の同点規則、Pythonのpow／math.fsum、経路も照合した。
元のworker入力はdetachedで呼出し中不変とする。転送後は入力を参照せず、必要なmetadataとboundsも切り離す。
並行更新中のbankからのsnapshot作成はこの処理の責務ではない。非正規型・大きなinventory・極端な数値は
元参照を全体で実行し、後の入力検査が先の例外を上書きしないようにする。fallbackの費用を実時間合格に含めた試験は未実施である。

固定payloadは直前73,296 bytesから6,997,808 bytesへ増えた。新規query領域6,924,512 bytesのうち、
数値入力が5,526,528 bytes、64件分のDTW出力が1,323,008 bytesを占める。
最大256×128 knotと128-knot cueは各一度の転送で32,896 knotとなる。数値表現168 bytesは登録済み320-byte形式と区別する。
queryごとのPython一時・結果のtracemalloc peak差は1,545,464／1,558,856 bytes。RSS観測、固定領域内訳、
段階別20回の実時計、ABI不一致二組の拒否を`resource-audit.json`に記録した。Python/C heap・全stack・allocatorの上限は未証明である。
初回測定11.351／15.317 msと対応sourceは`initial-measurement/`に保存し、最終値と区別する。

## 現在のpacked入力版

| 有効座標 | 辞書一括版 | 所有済みpacked入力 | 全blobコピー込み |
|---|---:|---:|---:|
| 1 | 11.416 ms | 5.042 ms | 5.718 ms |
| 10 | 15.158 ms | 7.221 ms | 7.831 ms |

320-byteの元記述子を単一の所有済みimmutable入力とし、辞書へ展開せずnativeで照合する経路を追加した。
全query中央値は既存の辞書一括版11.416／15.158 msから、
packed入力5.042／7.221 msへ短縮した。毎回の全コピーを含めると5.718／7.831 msだった。
コピーなし／あり・1／2 caller・1／10座標の8条件で全2,400測定が40 ms以内、全2,520出力が一致した。
2 callerのp99はコピーなし21.580／22.513 ms、コピーあり8.944／11.087 ms。順序・条件の異なる測定からコピーの高速化効果は推定しない。
固定ctypes bufferは7,001,928 bytes／caller。元payload10,526,720 bytesを保持し、コピー条件では同量を追加コピーする。
数値projectionは5,526,528 bytesを書き、借用pointerをreturn前に失効させる。原支持cut・世代・coverageの除算順序・全出力を維持した。
43実験検査、既存33 matcher検査、全Python871検査（32.693秒）、Rust746通過／12件ignoreを確認した。
72回の全query比較、84回の資源監査も全出力一致。根拠は`m0-native-packed-input-20260911`。
次はpacked exportからschedulerのdispatch・matcher・receiptまでを接続する。登録参照・feature版29・設計メモ・演奏経路は同一。
実生成入力・consumer処理量・全O04・fit・人のgateは未完了のまま残す。

元320-byte列からcoverageを除算後に判定し、168-byteの数値Knotへ写す。u64のepoch／generationは整数のまま比較する。
借用pointerの抽出だけGILを保持し、projection・粗探索・精査／DTWの三処理はGILを解放する。
Python参照とnativeが同じPackedKnotsを読めるため、更新されないcacheや二重のvalue表現に依存しない。
後の壊れたrecordで先の例外を隠さないよう、native失敗時は参照の逐次accessへ戻す。

実query slotとepisode bankのexportに対して、元metadata・全view・cut・export後の独立性を検査した。
ただし既存schedulerのtakeは元の辞書exportを続けており、実dispatch／receiptへの接続は次の工程である。
コピー条件は全immutable blobとmetadataの新規所有を計時に含む。元bank登録や旧辞書export作成の費用は比較に含まない。
従来10座標fixtureのmoment不整合をそのままcacheとして使わず、matcherの全観測値・mask・元時刻・IDが等しいpacked素材を生成した。
この素材生成はtiming外であり、実音声生成・group推定の代わりではない。

GILを保持する借用は約0.003〜0.005 ms、projectionは約0.43〜0.52 ms（独立した段階計測）。
tracemalloc peak差はコピーなし1,545,536／1,558,928 bytes、コピーあり12,198,265／12,211,657 bytesだった。
元source blob、コピーの所有者、全queue、Python結果、C／ABI／allocator／RSSは固定bufferと分けて計上する。
8条件は独立した閉ループ測定であり、6000周期・両実worker・64 Voice・実出力機器を含むO04の合格ではない。

## 計測の範囲

全queryの計測は入力検証・変換・検索・traceback・診断・結果構築を含む。
素材生成と既存episodeの保持は含まない。256 episode × 128 knot、cue 128 knot、候補16、
8,192 anchor、62変形を維持した。1座標入力は179,392 DP cell、10座標入力は245,024 cellを評価する。
現行比較は二条件それぞれ9回、Python参照・辞書一括版・packed・コピー込みpackedの四条件の順を無作為化した。
最初のDTW実験のprepared native call計測は、既に詰めた最後の一変形だけを繰り返す局所診断である。
これを全queryや全workerの時間と扱わない。profileは計測負荷を含む別記録である。

現在の固定ctypes payloadは7,001,928 bytes／instanceであり、元320-byte blobとそのコピーは別である。Rustの二行DP配列2,064 bytes、anchor標本／sort配列320 bytesを別に使う。
Python object、可変長の結果、変換時の一時値、allocator、ABI／compiler由来のstack増分、RSSは別途残る。
既存出力の`dp_payload_bytes`はPython参照との比較用の論理配置であり、native物理配置を示さない。
instanceは一workerから逐次使用する。二worker同時実行、64 Voiceとの負荷、実機音声、fitと人のgateは未検査である。

## 再実行

現在のsourceに対応するbuildは以下。過去archiveへ上書きせず、新しい出力先を指定する。

```bash
rustc --edition 2021 --crate-type cdylib -D warnings -O \
  --cfg 'feature="python-binding"' scripts/temporal_matcher_kernel.rs \
  -o /tmp/libtemporal_matcher.so
rustc --edition 2021 --crate-type cdylib -D warnings -O \
  scripts/temporal_matcher_kernel.rs -o /tmp/libtemporal_numeric.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/conchordal-context-scores-venv/bin/python \
  -m unittest discover -s tests -p 'test_evaluate_temporal_matcher_kernel.py' -v
```

計測driver・全query結果・profile・検査ログ・二binary・sourceは
`target/temporal-dcc/m0-native-packed-input-20260911/`へ保存した。
ABI相互誤接続とコピー／allocationの測定は`resource-audit.py`に含む。
比較は先行する`m0-native-query-batch-20260911`と、その入力生成用の
`m0-matcher-kernel-20260911`、`m0-matcher-reference-20260911`に依存する。
`packed-inputs.py`が元観測値との一致を検査し、毎回の新規blobコピーを行う。
native単体検査を先に実行し、対応するsource hashと終了記録を確認する。
再計測は別directoryへ分ける。保存directoryはGit対象外なので、別checkoutへの引継ぎには実体も必要である。
次は実dispatch／receiptの接続・全費用を測る。consumer処理量・実生成入力・全O04・fit・人のgateも未達である。
