# M0: 音響frontendの連結

2026-09-12。状態: trajectory分割、ridge継続、group配分、相関窓、候補生成、group更新を
一つのRust数値経路へまとめた。[sourceと検査結果](m0-acoustic-frontend.json)を保存する。
前段は[group lifecycle](m0-group-lifecycle.md)。今回も`cfg(test)`内であり、稼働中の
instrument／observer workerへ接続した実装ではない。M0全体は未完である。

## 入力と処理順序

入力は同じsource hopのLog2 power scanとmono mean-squareの組、または観測欠測である。
callerがbus／epoch、完全なNSGT支持、scanとmono energyの対応を保証する。
Frontend自身は固定したbus／epochのsample clock、scan長、finite／nonnegative値を検査するが、
sliceだけから外部での原音・bus対応を検証するものではない。

有効な組をtrajectoryへ分割し、peak順にridgeへ渡す。peakの強度順位が入れ替わっても、
ridgeの安定handleとpeak／配分rowの対応を保つ。bin別energyを計算する時点まで、このrow順を維持する。
groupのprepareで得た固定配分を、energy計算と候補生成器の双方へ渡す。
候補を処理した後にlifecycle commitを行い、現hopを子groupへ再配分しない。

全てのfrozen尺度、相関窓、持続長、退役時間、low-energy閾値は明示的なConfig入力とする。
未指定のdevelopment尺度を、本番用の適合済み値であるかのように補完しない。
今回の検査は明示した模擬尺度を使う。本番のfit版とsource frameの受け渡しは次段に残る。

## residualとゼロenergy

residual trajectoryとresidual groupは異なる役割である。
内部のresidual trajectoryはbus／epoch内でgeneration `u64::MAX`を予約する。
ridge allocatorは発行前のchecked incrementに失敗するため、この値を発行しない。
residual groupはlifecycleのgeneration 1を使い、resolved groupの世代とは別のslot役割を持つ。
これらの内部handleを、外部表示用の数値ID形式として採用したわけではない。

既知のゼロenergyも、log-envelopeのfloor値として相関窓と配分入力へ保存する。
一方、現在のbundle候補へ渡すのは`E_i > 0`のtrajectoryだけとする。
これは厳密なゼロの除外であり、適合した可聴閾値ではない。
無音を観測したという支持と、無音だけから新groupを作ることを混同しない。

欠測hopではridge、相関窓、親snapshotへ欠測を渡し、resolved group energyもunknownとする。
古いscan bufferを保持していても、現在値として公開しない。
正のmono energyとゼロspectral massの組では、既存のfallbackに従いscalar energyをresidualへ残す。
spectral shapeはunknownであり、resolved energyのゼロ値を既知の無音として退役時計へ渡さない。
mono energyが既知ゼロなら、全groupのゼロenergyは観測された値として扱える。

## energy snapshotと失敗

energy scanは8本を初期化時に確保し、hop更新中に増やさない。
読み出しviewには原hop終端と**更新前のgroup handle列**を付ける。
同じhopでslotが新しい子groupへ使われても、その子のenergyであると読み替えない。
spectral shapeがunknown、またはepochが失敗状態なら、scan viewを返さない。

通常の入力・kernelエラーを受けたepochは失敗状態にし、その後のhopを受理しない。
一部の数値部品がすでに進んでいても、その部分状態を成功frameとして公開しない。
新たに処理するには新しいFrontendを構築する。これは全連結部品を巻き戻すtransactionではない。
scan長の不一致は従来どおりreleaseでもhard assertとなるプログラミングエラーであり、
unknown値や通常の欠測へ変換しない。

## 検証

合成scanで、binごとのenergy保存、総energy保存、成立hopの旧handle列、buffer再利用、
peak強度順位の交換、無音でのproposal不発、欠測・未知spectral shapeでの退役抑制、
既知無音による退役、epoch失敗と再構築、scan長境界を検査した。

別の検査では、構成したPCM hopを実際の`RtNsgtKernelLog2`へ入れた。
48 kHz、hop512、nfft2048、145 binsで、bus0は2 Hz AMを持つ500＋750 Hz音、bus1は既知無音である。
各240 hopのうち、NSGT窓が満ちるまでの3 hopを欠測として扱い、残り237 hopをFrontendへ渡した。
成功したgroup更新はbus0で4件、bus1で0件。group energy総和とmono mean-squareの最大差は
`3.469446951953614e-18`だった。音声の保存やinstrumentのdisk-write経路は追加していない。

これは実際のNSGT実装と数値Frontendの接続検査である。
独立した音源回収oracle、録音済みdatasetの回収、作者による試聴、予測安定性の合格を意味しない。
beam、episode／occurrence所有、公開API／UIとの接続も検査していない。

全体のRust検査は842件通過・22件ignore・失敗0。今回の7検査を含む。
同じ構成PCM→NSGT検査をreleaseでも実行し、完全支持数、group更新数、最大energy差を含む
出力要約がdebug検査と一致した。これは一つの構成traceのbuild profile比較であり、全モデルの
予測安定性を代用しない。通常Clippyと、既存の二種類の警告だけを除外したall-target Clippyも通過した。

## 部分的な費用と観測された変動

単一bus、7 peakと正のresidualを持つscanで100回warmup後、6,000 hopを測定した。
spectral scanとmono入力の構築を計時外に置き、以降のFrontend全体を計時した。
193／769 binsでp99は30.55／43.281 µsだった。現在trajectoryは各hopで8本であるが、
ridge比較は49回／hopであり、登録上限98回を埋めた測定ではない。
group比較も全hopで最大896回を埋める条件ではない。

193-bin条件では測定区間に成功更新603件、棄却65件、退役703件があった。
769-bin条件ではwarmup後のこれらの件数は0だった。
この入力はbin当たりのbackground power等を固定した模擬scanなので、同じPCMを異なる解像度で
解析した比較ではない。尺度もdevelopmentデータから適合していない。
この違いをgroupingの安定性・不安定性の一般的な結論や、音楽的採用判断に拡張しない。
処理費用・状態遷移の記録として保存し、回収と安定性は別のsource固定assayで調べる。

Frontend headerは77,648 bytes、返却結果は84,640 bytes。
energy scanのbufferは193／769 binsで12,352／49,216 bytesである。
WindowやLog2Spaceの別buffer、全workerのRSS、stack peakを含むサイズではない。
NSGT、beam／記憶、queue、二bus並列実行、実出力機器を含む全O04も未検証である。

次はgroupごとのdescriptor／accent、beam初期状態、旧contextとcreditの所有を接続し、
尺度と実音sourceを固定した回収・安定性assayへ進む。
本番observerへの接続、全入力登録、全O04、素材・分割・fit・収集条件を含むM0全体は未完である。
