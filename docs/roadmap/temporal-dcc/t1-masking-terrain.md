# T1 マスキング地形の実装前登録草案

2026-09-24。状態：CASP 2008を作者が選択済み、実装前登録を具体化中、未凍結。作者の「全て推奨で、実装」により、担当IはI11-3、T2はI11-4、旧60本は保管して必要時の手順確認だけに使う方針を確定した。モデル選択前の比較は以下に残す。未記入の数値を暗黙の既定値にして実装を始めない。

## 1. 対象と今回の終了点

[二層規則](../../design-notes/technote-ledger.ja.md#two-tier-rule)と[milestones](milestones.md#two-tier-terrain-scope)に従い、T1の経験的な重なり費用を、ERB帯域での同時・前方マスキングを使う費用へ置き換える。合成器由来の代表footprintを使い、学習済みfootprintへの置換はA4後に残す。T2のaccent・周期性の再設計、I4の研究検索、I12bの封印条件は変更しない。

選択用草案は独立Astraレビューで「作者判断へ提示: 可」となり、作者はCASP 2008を選択した。次の終了点は、原典係数、入力・比較量、時計・配送、判定値を具体化し、実装前レビュー「可」の後に登録を凍結して、許可された実装へ進むことである。「選択資料として可」と「実装前登録を凍結してよい」は別の判定である。

## 2. 原典候補の比較

以下は2026-09-24に確認した範囲である。第一・第二候補は同時／前方マスキングを同じ枠組みで扱う。第三候補は時間窓による別系統であり、それ単独で本登録全体を満たす候補とはしない。計算費用はどれもこの実装で未測定である。

| 候補 | 形と主要パラメータ | 確認した適用範囲と残る条件 |
|---|---|---|
| **CASP：Jepsen・Ewert・Dau 2008（推奨案）** | 外・中耳変換、非線形DRNL、hair-cell変換、二乗、adaptation、150 Hz低域通過、変調filterbank、内部雑音、検出器。DRNLの圧縮指数は0.25、周波数別係数は同論文Appendix Table Iの改訂版を使う | 同時・スペクトル・前方マスキングを評価。レベル依存の圧縮と周波数選択性を含む。全係数・離散化・入力校正の転記照合、ERB格子との対応、任意の合成音への適用は未完了 |
| **Dau・Püschel・Kohlrausch 1996** | 線形の120 segment wave-digital基底膜filter、半波整流＋1 kHz低域通過、5段adaptation（5／50／129／253／500 ms）、8 Hz低域通過、内部雑音、template検出器 | Part IIで同時・前方等を評価。原版は主に信号周波数のchannelを使う。レベル依存の蝸牛圧縮を持たず、任意のERB多帯域集約には別の対応検証が必要。1997後継のgammatone版と混同しない |
| **Oxenham・Moore 1994の時間窓系** | 振幅の圧縮べき則、移動時間窓、出力S/Nの判定。要旨にある適合指数は0.5〜0.7 | 短い6 kHz信号と前後の広帯域masker、その加算を扱う。要旨とDOIを確認したが、本文の窓係数・判定値は未確認。同時マスキングの周波数モデルを追加して一式を選び直す必要があり、即時の凍結候補ではない |

第一候補を推奨する理由は、音量と帯域を変える本用途で、蝸牛の非線形性を初めから扱うモデルを基準にできるためである。これは工学上の推薦であり、音楽的有効性や実時間動作の確認ではない。簡単に通る検査へ縮める目的で、原典にない帯域energy則へ置換しない。第二候補を選ぶ場合は線形前段の適用限界を受け入れ、第三候補を選ぶ場合は不足する原典と同時マスキングモデルの確認から始める。

原典と確認箇所：

- Jepsen, Ewert & Dau (2008), *A computational model of human auditory signal processing and perception*, JASA 124, 422–438。[DOI 10.1121/1.2924135](https://doi.org/10.1121/1.2924135)、[著者所属機関の本文](https://backend.orbit.dtu.dk/ws/portalfiles/portal/4295299/Dau.pdf)。要旨、§II、§III、§IV、Appendixを確認。Table Iは2001年の基礎DRNLの全係数表を置き換えるものではない。選択後には参照先も含めて係数の対応表を完成する。
- Dau, Püschel & Kohlrausch (1996), *A quantitative model of the “effective” signal processing in the auditory system. I. Model structure*。[DOI 10.1121/1.414959](https://doi.org/10.1121/1.414959)、[本文](https://pure.tue.nl/ws/files/1471028/622101.pdf)。§I.B–E、Appendixと脚注1・3を確認。templateはmasker＋閾上信号の内部表現からmaskerの内部表現を引いて作り、信号単独の応答では代用しない。
- 同著者 (1996), *II. Simulations and measurements*。[DOI 10.1121/1.414960](https://doi.org/10.1121/1.414960)、[要旨](https://pubmed.ncbi.nlm.nih.gov/8655794/)、[本文](https://pure.tue.nl/ws/files/1307552/622104.pdf)。短いforward masker等での不一致も残る。原典との一致と、人の結果への完全一致を同義にしない。
- Oxenham & Moore (1994), *Modeling the additivity of nonsimultaneous masking*。[DOI 10.1016/0378-5955(94)90014-0](https://doi.org/10.1016/0378-5955(94)90014-0)、[著者所属機関の要旨](https://experts.umn.edu/en/publications/modeling-the-additivity-of-nonsimultaneous-masking)。本文未確認であり、要旨にない窓係数は補わない。
- ERBの基礎文献はGlasberg & Moore (1990), *Derivation of auditory filter shapes from notched-noise data*。[DOI 10.1016/0378-5955(90)90170-T](https://doi.org/10.1016/0378-5955(90)90170-T)、[要旨](https://pubmed.ncbi.nlm.nih.gov/2228789/)。要旨とDOIまで確認。ERB座標の採用だけで、任意のfilterを原典の聴覚filterと同一とはしない。

## 3. 現入力の不足と供給契約案

現行の代表footprintは総energyをpeakで規格化した16点であり、帯域配分は静的な3帯域profileである。外部予測も3帯域のenergyである（`src/life/action_candidates/footprint.rs`、`src/core/temporal_expectation.rs`、`src/life/temporal_participation.rs`）。この入力から帯域内の位相・波形・絶対音圧・過去のadaptation状態を復元できない。平方根を取って波形と呼んだり、3帯域をERBと改名したりして原典対応を主張しない。

モデル選定後に固定する入力は、少なくとも次の組である。

1. **代表自声。** 合成器recipe、sample rate、代表kick・seed・hold・modulator、振幅基準、支持長から代表PCMまたは同じ聴覚モデルの内部応答を作る。現在の規格化energyとは新しい入力契約になる。代表化と支持長の打切りは作者規則であり、実kickや実際の聴取音を予知するものではない。`body`／`proxy`比較は、carrier・校正・窓・masker・初期状態を共通にし、登録した包絡／身体応答だけを交換する案とする。旧16点proxyを波形入力へ無条件に流用しない。
2. **観測済み外部音。** habitat PCMから、同じsample時計・同じrouteの既知の自声PCMを差し引く。その後に選択した聴覚モデルへ入れ、前方マスキングの状態を更新する。非線形処理をした二つの出力を後から差し引く手順とはしない。観測終端以降の実PCMを候補選択へ先読みしない。
3. **将来の外部刺激。** 現予測はenergyだけなので追加の刺激構成則が必要である。第一案は、既存の短期反復予測に支持がある外部PCM区間を、登録した位相位置と周期で延長するものである。周期・素材区間・補間・継目処理・有効範囲を記録し、支持のない予測は不明とする。これは原典モデルの入力を作る作者規則であり、観測波形でも原典による将来予測でもない。選定後の入力仕様と参照検査で確定する。異なる刺激構成則の結果を同一モデルの検証へ混ぜない。
4. **音圧と帯域。** digital amplitude、Pa、SPLの対応、peak／RMSの区別、外・中耳変換を固定する。CASP本文の振幅1と最大100 dB SPLの対応はモデルの基準であり、本機のスピーカーやヘッドホンの校正値ではない。技術検査は宣言した仮想音圧基準で行い、実試聴の機器・音量との対応はA1で別に記録する。ERB中心・範囲・密度とモデルchannelの対応、帯域外の扱いは未凍結。DRNLを線形ERB filterへ置換する場合は別の近似になる。
5. **状態と支持。** filter・adaptation・template・予測素材の状態を、入力履歴とモデル版へ結ぶ。連続状態の更新、起動時の初期状態、履歴欠落、sample rate／route変更、身体世代交換、支持長の終了を区別する。前方マスキングを現在のenergy一枚から計算しない。履歴打切りの誤差、初期化後の適格化条件は検査と共に凍結する。

ERB方向の内部配列を既存のLog2Space `_scan` と取り違えない。ERB配列にはその座標を示す名前を使う。既存UI等へ `_scan` として出す場合だけLog2Spaceへ写像し、長さの境界検査と補間誤差を登録する。モデル内部の変調filterbankはT1の検出計算に限る。CASPを選んでもT2のaccent・周期モデルが自動的に決まるわけではない。

## 4. 比較量と行動への接続

以下は原典外の作者規則層の実装候補である。16点gain探索による閾値差の案は、素朴な反復計算量が大きいため、代表自声PCMそのものを`G=1`とした**固定gainの検出可能性低下**へ置き換える。候補onset `a`、自声`S`、外部masker`M`を固定し、`masker＋自声`と`maskerのみ`を同一履歴から別状態で計算する。quiet側も`自声`と`無音`を同一quiet履歴から別状態で計算する。音楽候補用のtemplate gainは代表PCM比`10^(10/20)`、同時に原典課題の閾上刺激レベルは各原典条件の指定値を使い、混同しない。[kernel契約](../../../target/handoff-20260924/phase0/casp-kernel-contract.md)の正規化したmasker別template、期待相関`d_M,d_0`、`h=ln(d_0/d_M)`、`cost=max(0,h)`を計算する。これは検出閾値・JNDではない。`d_M<=0`等を0費用へ丸めずcensored/unknownとして数える。maskerの有無で履歴状態を共有・破壊しない。

原典が検証したtone／maskerに対する量と、任意の代表合成音・多帯域への一般化は分ける。全対象ERB×変調×時間の正規化templateを一度作り、その内積を先に集約する。診断用にERB別の内積・template norm・床率・対象標本数を残し、ERB別費用を後から平均しない。`h<0`はmasked側の増益として符号付き生値に残し、参加費用だけ0とする。既知の無音、不明、帯域外、打切りを0費用へ一括変換しない。結合係数とcensor/適格率の採用基準は、T1全体の実装前凍結と独立レビューに残す。音を聞いてから遡及調整しない。

この費用で既存のonset候補を比較する。通常のT1有効・完全packetでは、現行`src/life/temporal_participation.rs`の旧`overlap`加算項 `coupling × 6 × overlap_sensitivity × overlap/(own_total×footprint_mass)`だけを、`coupling × 6 × overlap_sensitivity × cost`へ置換する。context距離、displacement、arrival、候補順は保持する。新項は現行の`base_cost`へ加え、選択後の既存skip閾値`best_base_cost > 1+skipped_cycles`にも従う。packet欠落・unknown時は全23候補を同じ旧overlap項へ戻し、その機会をT1非適格と記録する。`cut`は新項の選択・skipへの加算だけを切り、診断計算は維持する。新body/proxy/cutの既存FootprintChoiceは全て`body`に固定し、unknown/lateで旧overlapへ戻る際も同じ旧footprint・H・arrival・displacement・skip入力を使う。新proxy腕を旧config `proxy`へ変更しない。可能なら一つのworker packetへ新body/proxy両費用を載せ、armによる配送負荷差を避ける。`None`では新経路に入らず既存bitを保つ。この係数6の保持は既存設定の重みを別の無次元費用へ接続する作者則であり、旧新費用の同じ数値尺度を保証しない。尺度の作用とskip率は全12条件で測る。「受け渡し」が聞こえることはA1で検査する。

重いfilter・状態複製・検出は有界workerで行う。Voice側が受け取るのは、固定した候補集合と入力識別に対応する評価packetである。計算中に観測や候補集合が進んだ場合、古い値を新しい集合へ転用しない。旧footprintと同じ低頻度cacheで足りるとは仮定しない。将来刺激と履歴が変われば無効化が必要になる。容量、共有範囲、要求率、締切、破棄・回収の費用はR2へ渡す。

## 5. I11-1から継承する検査

正本は[I11-1 §5と後続の限定宣言](i11-onset-comparison.md)である。旧版の完了や共通record照合を、新モデルの同一packet再評価が完了した証拠にしない。現行版の基準はI12bの終結後に固定する完全なsource、binary、設定と入力であり、現在のHEADだけでは未commitの封印sourceを表せない。

| 検査 | 新しい登録へ継承する条件 |
|---|---|
| 入力一致 | 一回の選択は同じ観測終端・masker素材・代表recipe・内部状態snapshot・モデル版・校正・候補集合で比較する。reportへその識別を残し、独立参照が消費した値を再計算する。一候補だけ支持が不足した場合、候補間で比較量の意味が変わる使用を避け、登録した共通適格性規則で全候補から当該項を外す |
| 状態介入 | 固定した入力packetで`body`／`proxy`を交換し、登録した自声応答以外は固定する。別に、先行masker履歴のみを変更した対で前方マスキングを検査する。最初の費用差、選択分岐、生成音まで追跡する。全条件が不明や無分岐でも成功になる判定は置かない |
| 接続除去 | `masking-connection-cut`は入力受領・聴覚モデル・状態更新・費用の診断計算を維持し、参加判断への加算だけを切る。他項、候補、乱数消費を保つ。最初の分岐以降に入力が変わる閉ループの差と、同一状態での介入を分ける |
| 時計と期限 | 観測終端、要求、計算完了、受領、失効、候補onsetの各sample時計を別に記録する。未来の受領、stale、世代不一致、支持外の候補は不使用。予約後でも選択hop開始前の同一機会packetは使い、選択確定後の到着では再選択しない。offlineの決定的配送とinstrumentの実時間配送は別に検査する |
| 欠測・世代 | absent／有効／stale／再受領、途中の身体・モデル・route・sample rate変更、queue drop、置換、再送を検査する。不明は既知の無音と分ける。fallbackはその理由と利用率を記録し、全件fallbackで技術完了にはしない |
| off／None | 地形を無効にした新版と封印基準版のWAV、I10学習record10種をbit比較する。新しい設定自体がNoneの場合と、既存I11明示設定のまま新地形だけ無効の場合を含む。同一凍結packetの候補決定的fieldもbit比較し、実時間・資源欄とliveのdrop／片側欠落は明示して別集計する |
| 作用と分母 | 登録12条件はsine-hold／sine-flow／harmonic-flow／modal-flow × 4／16／64 Voice。主作用のharmonic／modal-flow 6条件、各sine群3条件を保持する。欠測・無作用で分母を縮めない。旧body/proxyのsine無差は新モデルやcutへ継承せず、期待を比較ごとに事前固定する |
| 代表と実音 | 実発行したonset条件と代表条件の新しい帯域応答／マスキング量の差を、全登録機会に対する適格数と共に報告する。実条件の未来情報を選択時へ戻さない。旧16点energyの分布と、新モデルの誤差を同じ量として集計しない |

初回技術検査には、モデル入力と検出器の有効経路が通る既知刺激の正例を、生成runとは別に置く。生成runでも事前に選ぶ主作用条件で適格な消費、費用・選択の差、WAV差を要求する。適格率と最初の分岐の数値基準は、実装を見て通りやすい条件だけを選ぶ前に凍結する。全条件で差を要求するか、主条件中の最低数を要求するかは検査設計として独立レビューで確定する。

## 6. 原典照合・資源・A1の別々の判定

**原典照合。** 選択した版のfilter、非線形変換、状態更新、template、検出を段階ごとに独立参照と照合し、同時masker、masker終了後のgap、レベルと周波数の変化を含める。CASPなら§IIIのtone-in-noise、spectral masking、forward maskingの刺激を候補とし、元論文の条件と予測曲線を登録する。図の読取り誤差、原典実装との差、数値離散化誤差、人の結果との差を分ける。原典に残る不一致を隠す再適合はしない。参照実装の版・利用条件・source SHAと、数値許容は実装前に固定する。参照コードを未確認のまま「原典実装と一致」としない。

**入力近似。** 仮想音圧、代表化、未来刺激の構成、ERB格子、状態履歴の打切り、固定gain費用への写像を別に検査する。原典モデルの再現に通っても、近似した未来刺激の正しさまでは証明しない。対照では凍結したPCMによる参照値と、選択時点までの情報から作った値を比較し、未知率と誤差を残す。

**資源。** 技術検査は機構固有の処理・配送が使えることを示す。[R2](r2-preflight.md)では、PCMの保持と複写、聴覚状態の容量、workerの要求・queue・cache、候補別評価とtemplate・固定gain検出、返却と回収、hop、report/UIを含めて測り直す。4／16 Voice等の短期検査を64 Voiceや長時間の合格へ流用しない。I4第二段階・I10再検証後、T1／T2を統合した版が最初の正式R2対象である。技術完了をR2や通常採用の合格にしない。

**A1。** [A1 §0](a1-i11-audition.md)の対象版はT1地形後・T2地形前である。旧body対新bodyは変更の記述、新版内body対proxyは自声応答介入、新版内body対cutは接続除去として扱う。48本・4 session各12試行は未凍結案である。モデルが決まっても、proxy構成、比較量、対照期待の同一性確認を省かない。T1対象版を固定すればT2の作業とT1 A1を並行できる。T2後の回帰で証拠を引き継げない範囲は再試聴する。I12bのA1は別登録であり、作者の回答や採用は代理しない。

## 7. 選択後の一単位と凍結条件

作者はCASP 2008を選択し、実装を指示した。次の一単位は、その版の原典係数表と入力・出力・比較量を埋め、独立レビューに出せる登録へ仕上げることである。通常の技術選択について承認を取り直さない。実時間費用や可聴性は、モデルの選択だけで確認済みにはならない。

凍結までに埋める欄は、選択モデル・原典版・参照実装、全係数と単位、ERB格子、仮想音圧、代表PCM／proxy、未来刺激構成、履歴と初期化、検出・集約・結合、候補の共通支持、packetと締切、容量、入力行列と正例、精度・作用・適格率の判定値である。これらは技術案としてまとめてレビューし、通常の実装選択を一項目ずつ作者へ戻さない。T行・三条件・完了範囲を変える必要が生じた場合だけ、影響を示して作者判断へ戻す。

取得仕様は入力source・binary・設定・検査規則のSHAを固定し、独立Astra「可」の後に凍結する。取得後にWAVと結果のSHAを確定し、提示前登録を別にレビュー・凍結する。未生成WAVのSHAを取得開始条件にはしない。source変更はI12b終結と基準版固定後の別worktree・別`CARGO_TARGET_DIR`に限る。

## 8. 選択後に具体化した検討資料

原典の全DRNL係数、AMT後年再構成との違い、外・中耳の公開伝達表、adaptation、変調filter、内部雑音の較正課題を[原典照合メモ](../../../target/handoff-20260924/phase0/casp-spec.md)と[kernel数値契約案](../../../target/handoff-20260924/phase0/casp-kernel-contract.md)へ記録した。AMT関数本文をrepositoryの実装へ複写せず、公開数値と原典の式を独立実装する方針である。2008年原典の未公開FIR tapとの同一性、後年再構成の原典との同一性を主張しない。モデル選択済みであることと、全数値契約の凍結は別である。

現在の候補生成は最早候補に近づいてから23点を作るため、非同期workerの返却値にその場の現在時刻・候補gridとの完全一致を要求する契約では間に合わない。[計算・配送案](../../../target/handoff-20260924/phase0/t1-computation-contract.md)では、T1有効時だけ機会を前もって予約し、観測snapshotとgridを固定して最早候補hopの前を期限にする案を具体化した。off/Noneでは旧経路を保つ。予約後の通常観測と、身体・route・sample rate等の無効化条件を区別する。期限内率が0でも合格する規則にはしない。

§4は固定gain費用を実装候補にした。masker側のtemplateは必ず `R(M+G_ref*S)-R(M)`、quiet側は `R(G_ref*S)-R(0)` をそれぞれ正規化する。信号単独のtemplateをmasker条件へ転用しない。共通iid白色雑音分散が、異なる単位normのtemplateでも期待相関の標準化比から相殺する条件と、原典の3AFC閾値・JNDとは異なる出力であることをkernel案へ明記した。原典刺激の検査、費用結合、支持・期限・資源上限、作用の判定は依然として実装前レビューと凍結の対象であり、この追記をT1完了としない。

### 8.1 kernel単位の入出力と先行終了点

独立Astraで部分凍結可となった単位は**48 kHzの単声道PCM→CASP前段の時間×30末梢channel×各channelの変調channel表現**である。[部分凍結記録](../../../target/handoff-20260924/phase0/casp-kernel-freeze.md)は対象行・仕様SHA・審査SHAを固定した。数値fixtureの合格は未実施である。[kernel契約](../../../target/handoff-20260924/phase0/casp-kernel-contract.md)の数式・係数・取得byte SHA、外耳22点・中耳31点、FIR設計規則、IHCの一次1 kHz LP、参照実装由来の二乗前50 dB、adaptationの5状態とMU出力、150 Hz LP、変調bankを一つの版として扱う。中心周波数は100 Hzから1 ERB-rateずつ、8000 Hzを超えない30点（最後は約7220.468 Hz）である。二FIRの各256標本は**群遅延metadata**であり、live判断時刻を512標本戻さない。観測終端までに消費した入力だけから状態を作り、未知未来を零flushしない。FIR/IIR/charge/complex状態と入力時計を同一checkpointへ保存し、分割処理・複製分岐・resetを照合する。PCM値1は仮想2 Pa peakだが、実habitat混合の`|PCM|>1`を自動clampしない。検査済みSPL域外は別表示とし、原典で較正済みと呼ばない。

kernelの合格資料はimpulse・周波数表全節点、BF250/500/1000/1500/4000 Hzの純音・レベル走査と4 kHz channelへの異周波数入力、静音→step→offsetのadaptation、2/5/10/16.667/150 Hz変調の独立再帰照合、床値率、段別誤差を含む。原典Fig.2との図読取は別の対応診断として残し、未公開tapを含む原典数値との一致をkernel単位の合格条件にしない。`BF=1500 Hz`を直接指定した境界fixtureは実格子外でも行う。刺激長・位相・許容はkernel版で標本単位に固定し、独立Astraがこの**実装前仕様**を凍結可と判定した。原典Fig.3のJNDは雑音・検出器較正が要るため**後続gate**とする。SourceForge 1.6.0 tagの外耳表・中耳表・adaptation Cはdevelopとbyte一致を確認した。2008著者のFIR tapは未公開であり、独立版の一致と原典tapの同一性を混同しない。kernel fixtureが将来合格しても、その終了点は**kernel単位の数値照合**であり、T1費用や通常経路の完了ではない。

### 8.2 全T1への接続と未凍結境界

kernel後に必要な通常経路は、観測済みhabitat PCMから既知自声を同じsample時計・routeで除いた外部履歴、支持付き未来PCM、実body recipe由来の代表自声、body/proxy/cutが共有する機会IDと候補23点、masker別template・費用、worker返却packetである。既存`Tone::prediction_sine`／`prediction_bank`と`BankForecast::lane_at`はcarrier・位相・lane状態への候補入力だが、rendererの完全PCM bit同値は未証明である。proxyは共通carrier・位相・初期状態を保ち、登録した身体応答の総振幅包絡だけを介入する。bodyとproxyの実音誤差を独立照合し、既存forecastをそのままbody真値へ置換しない。必要な連続観測masker履歴を一度処理してcheckpointを作り、候補ごとに履歴を再計算しない。未来`M+S`だけはDRNL/adaptationが非線形なので候補別に通す。全候補の共通支持が欠けた機会はT1項を一機会単位で外し、unknown理由を記録する。`None`と既知静音を混ぜず、off/Noneの封印版bit同値を保つ。[計算・配送案](../../../target/handoff-20260924/phase0/t1-computation-contract.md)のprospective opportunity予約、最早候補前deadline、世代・周期・route・model版の無効化を通常接続で検査する。期限は64 Voiceの実測に基づき、全staleを合格にしない。代表自声の実音誤差、未来外部PCMの予測誤差、期限内完全packet率、結合係数、主作用6条件と全12条件の受入数値はまだ測定・凍結していない。これらの全T1登録、通常接続、独立レビュー、作用・資源検査を終えるまでI11-3技術完了と呼ばない。I12b再取得・基準固定を先に終えた後、別worktreeで実装する。

### 8.3 通常接続の未凍結技術案

[接続仕様案](../../../target/handoff-20260924/phase0/t1-integration-spec.md)では、代表`Tone`の同一renderからbody PCMとdry backend・control gainをtapで得て、body返値のbit同値を保つ。proxyは同じbody PCMへ**正の時間gainだけ**を掛け、50 msのdry振幅包絡を最大64倍の正gainで平坦化し、全支持窓RMSをbodyと合わせる。carrier、符号・零交差、位相、modulator、ADSR、seed、masker、CASP初期状態は共通である。介入は身体応答の**総振幅包絡**に限定し、身体の全スペクトル応答を除去したとは呼ばない。既存`prediction_sine`／`prediction_bank`は位相・lane診断へ使えるが、body PCMの代替にしない。代表対実音、proxyの側波帯・peak、各12条件の有効率を計測してから全T1版を確定する。

未来maskerは最後の実測sample`e`までの同一route・epochの3周期PCMを要求し、次の`e+1`から整数周期で延長する。周期は0.25–2秒、継目は先頭480標本のraised-cosineで前周期末尾から繋ぐ。波形相関・RMS比・8秒支持上限は接続案の暫定値であり、holdout誤差・適格率を測って全T1審査前に固定する。自声は候補onsetから登録した`D<=4秒`だけ入力し、必要な未来PCMが一候補でもない機会は全有効候補のT1項を揃えて外す。既知静音、unknown、`None`を分ける。

[計算・配送案](../../../target/handoff-20260924/phase0/t1-computation-contract.md)では、T1有効時だけ将来の選択hop`q`を整数sampleで予測し、`q`を使った`earliest`と23欄を予約時に固定する。未予約なら現行経路、予約済packetの期限切れなら凍結grid上の旧overlapへ一機会単位で戻す。固定gain費用は`R(M+G_ref S),R(M),R(M+S)`の同期streamからdot/normを蓄積し、quietは別の3枝を使う。原典の閾値探索・JNDとは区別する。`lead`・worker数・必要容量は、通常経路の変更前に凍結kernelと同値費用を使う独立bench harnessの入力・計測規則を固定し、当チームの競合計算を避けて見積もる。その測定値、共通取得長、適格率の数値判定を全T1実装前登録へ記入し、独立レビュー可・版凍結後に通常経路へ接続する。接続後の実packet期限内率と64 Voice資源は別途実測し、事前benchをlive成立の証拠にしない。詳細な順序は[事前測定の境界](../../../target/handoff-20260924/phase0/t1-prereg-measurement-order.md)に記す。

既存の6秒本体＋tail素材では、3周期PCMの最大6秒履歴、CASP状態のwarmup、先行予約leadを満たした後に評価機会が残らない条件がある。新しいT1取得入力は既存12条件のVoice構成を維持し、warmupと作用評価窓を**取得前**に固定して十分な適格機会を設ける。同じ延長入力を旧body、新body、proxy、cutへ渡し、旧短素材の全fallbackを成功扱いしない。延長秒数はkernelとlead計測後の技術判断として全T1 prereviewで固定し、未取得の結果に合わせて条件を選別しない。

### 8.4 kernel完了と独立benchの進行（2026-09-24）

§8.1のkernel単位を実装・検証し、独立completion review3で「限定技術完了: 可」となった。30聴覚channelの0 Hz枝を含む271出力、登録したtone／level／AM刺激の各標本・段・全状態、非有限値、checkpointを照合した。対象source、凍結契約、fixture目録、全体Rust検査、独立レビューを[`casp-kernel-completion.json`](../../../target/handoff-20260924/phase0/casp-kernel-completion.json)に固定した。§8.1の未実施という記述は、この完了前の登録時点を示す。

独立benchの局所設計もレビュー「可」となり、kernel完了を結び付けた実装前freezeに従ってtest-only harnessを作成中である。未改変Toneから先に固定した3 body×2 hold×64 IDの384 PCM条件について、hook追加後のcapture無効・有効の両返値が、同じrelease条件で元の全f32標本とbit一致した。参照生成binaryと比較binaryを別に保存した。先行したdev profile比較の1 ULP差は失敗記録として保持する。この検査は登録した代表入力だけを対象とし、任意の実Voiceの再現性、proxy妥当性、費用計算の性能、期限内配送の成立を含まない。bench全体の検査・独立審査・排他的な時間計測は、この時点では未了である。
