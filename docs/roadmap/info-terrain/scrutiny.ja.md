# 機構選択規則による審査：予測精度と驚きの地形（「響きの豊かさ」の候補）草案 第6稿

> 記録：2026-10-21 に `target/info-terrain-20261020/` から移した審査の草案（第6稿、独立レビュー6回のうち第6回で「可」）。本文は実装と実測の前に確定した文面のままで、移した後に書き換えていない。冒頭の「状態」は作成時のものである。経過・結果・判断は[台帳 §9.3.56](../../design-notes/technote-ledger.ja.md#info-terrain)に記録した。本文中の旧パスは次のように読み替える：`draft.ja.md` → `scrutiny.ja.md`、`target/info-terrain-20261020/registration-*.ja.md` → 同じディレクトリの同名ファイル、`results-a1/` → `results/a1/`、`target/info-terrain-20261020/results/exploratory_dimensions.md` → `results/exploratory_dimensions.md`。

状態：草案。ledger・technote にはまだ入れない。2026-10-20、作者は第9節の 1〜3 を推奨どおりに決めた（主張の範囲を候補の対応に限る、精度と驚きの二つの scan とし PIR を退ける、A では原型と拡張の両方を並べて実測する）。これに伴い、原型の細部の照合（第9節の 4）も行う。B は未判定。実装・検証・採用はいずれも記録しない。文献の照合は書誌情報と要旨の水準であり、争点となる箇所についても本文の細部は確認していない（照合日 2026-10-20）。

## 1. 背景と問い

Flora の暫定の力学（replicator＋自前の疲労）は、固定した地形で静止した和音に縮退した（`docs/design-notes/flora.md`、Implementation Findings、2026-10-20）。二つの丘の 60 秒後の疲労は h ≈ 0.80 と h ≈ 0.667、適応度は両方とも g ≈ 0.200 に揃った。オフラインの render でも、sine 構成は数秒で 3×55 Hz と 5×55 Hz に落ち着いた。これは特定の疲労則がもたらした混合平衡であり、生成側の時間発展が一般に外付けでしかありえないことを示すものではない。

ただし DCC の地形は、安定性（R/H から C）と、それを削る順応（habituation、AdaptationContext）しか表していない。予測に基づく価値、つまり聴き手にとって音のどこに知覚上の新しさがあるかを表す共有量はない。ledger §9.2 のギャップは句と場面の水準の期待を指しており、本審査の量がそれを埋めるわけではない。本審査が問うのは、より短い時間尺度で、スペクトルの予測に関わる知覚量を DCC 側に置けるかである。

## 2. 審査する機能

- **A. 観測。** 聴き手側の予測から、bin ごとの二つの `_scan` を作る。精度の指標（原型版は履歴分散、拡張版は予測残差に基づく）と、現在の入力の驚きの指標である（第5節）。既定 off、受動的（report/UI）とする。いずれも「対応を検討する候補の指標」であり、聴き手の知覚量そのものとは主張しない。
- **B. 生成への結合。** Flora が A を読む。将来は個体も読む。

規則上、A は条件が欠けていても、既定 off の受動的な診断として置ける（ledger、機構選択規則）。ただしこの例外は、量の認知的な妥当性を保証しない。また A の状態を B が読んだ時点で、その経路にはもう例外を使えない。B は三条件の新規審査を要する。

## 3. 対象とする聴取応答

観測できる応答として、向きを区別して定義する。

- **R1a 逸脱応答と順応。** 予測された入力への応答は減り、予測されない入力（逸脱）は大きな応答を引き起こす。
- **R1b 精度の追跡。** 規則性を持つ系列では、ランダムな系列よりも持続的な活動が大きい。その大きさは入力の予測可能性（精度）に従う。向きは R1a と逆である。
- **R1c 予測不能性の知覚判断。** 聴き手は音響系列の予測不能性を一貫して評定し、その評定は音響の surprisal 指標で追える。
- **R2 テクスチャの要約。** 長い定常テクスチャでは時間的な細部への到達が弱まり、統計的な要約が優勢になる。
- **R3 快。** 快は不確実性と驚きの組み合わせに依存する。

**提案。** 対応を検討する対象応答を、R1b（精度）と R1a・R1c（驚き）とする。それぞれに候補の指標を一つずつ割り当てる。指標と応答の対応は、仮説として扱う。R2 は表現の選び方（時間平均の統計を扱う）への示唆にとどめ、「情報を運ばない」の根拠には使わない。R3 は B の結合の形を選ぶ動機として扱い、主張しない。

## 4. 条件1：応答と機構

### R1a（逸脱と順応）

- ネコ A1 のニューロンは、同じ音でも稀に提示されるときに強く応答する。周波数と振幅の逸脱で見られ、当該実験の範囲では視床で見られなかった。SSA は MMN の単一ニューロン相関の候補とされる（Ulanovsky, Las & Nelken 2003, *Nat Neurosci* 6:391–398）。
- SSA は数百 ms から数十秒の複数の時間尺度を同時に持つ。短期と長期の刺激履歴を使う線形モデルで、A1 の応答をよく近似できる（Ulanovsky, Las, Farkas & Nelken 2004, *J Neurosci* 24:10440–10453）。
- MMN の予測符号化による解釈は、順応説とモデル調整説を統一する提案である（Garrido, Kilner, Stephan & Friston 2009, *Clin Neurophysiol* 120:453–463、総説）。
- roving oddball 課題で単試行の MMN 振幅の変化を比べると、自由エネルギー原理に基づく近似ベイズ学習のモデルが、変化検出や順応のモデルよりよく説明した。著者は、MMN が感覚の規則性の近似ベイズ学習を反映すると結論している（Lieder, Daunizeau, Garrido, Friston & Stephan 2013, *PLoS Comput Biol* 9:e1002911、被験者 8 名）。この比較で支持されたのはモデル調整の族であり、Shannon の驚きを近似する novelty detection の族ではない。本草案の驚きの指標が R1a に対応するという主張は、Lieder を根拠にできず、仮説にとどまる。

### R1b（精度）

- 規則的な音列は、周波数を揃えたランダム列より大きな持続的 MEG 応答を引き起こす。持続的な活動の変化は、入力の期待精度（予測可能性）と相関する。一次聴覚皮質、海馬、下前頭回の相互作用が関与する（Barascud, Pearce, Griffiths, Friston & Chait 2016, *PNAS* 113:E616–E625）。

### R1c（予測不能性の判断）

- 合成した音響系列 300 本を聴き手 195 名が予測不能性について評定した。スペクトルの変動に由来する予測不能性は Shannon surprisal と SSM novelty が、リズムの不規則さは ACF surprisal が追った。多くの指標で、スペクトル変動に最も効く窓はおよそ 1 秒に収束した。Bayesian surprise も比較されたが、その成績は要旨にない（Anikin 2026, *Behav Res Methods* 58, doi:10.3758/s13428-026-03153-3）。

### R2・R3（主張しない）

- テクスチャ：McDermott, Schemitsch & Simoncelli 2013（*Nat Neurosci* 16:493–498）。
- 快：Cheung et al. 2019（*Curr Biol*, doi:10.1016/j.cub.2019.09.067）は不確実性と驚きの交互作用。Gold et al. 2019（*J Neurosci* 39:9397–9409）は中間的な複雑さへの選好と「学習への報酬」の解釈。どちらも、和音と旋律という記号列の上のモデルによる。逆 U 字は、交互作用とは別の主張である。いずれの結果も聴き手の経験・個人差・刺激集合に依存しうるので、別の音楽様式や合成音への一般化は示されていない。
- 報酬系：Salimpoor et al. 2011（*Nat Neurosci* 14:257–262）。
- 総説：Chmiel & Schubert 2017（*Psychol Music* 45:886–909）は、57 研究の 87.7 % を区分的な逆 U 字と両立と分類している。これは全個人に逆 U 字が成り立つことの証明でも、機構の同定でもない。

### 評価

- **R1a、R1b、R1c には一次研究の証拠がある。**
- **R1a と R1b の機構は候補として特定されている。** R1a は A1 の SSA と MMN の生成過程、R1b は精度を追跡するネットワーク（一次聴覚皮質、海馬、下前頭回）である。R1c は行動上の証拠だけで、神経機構は提示されていない。
- **本機能との対応は確立していない。**
  - SSA と MMN の証拠の中心は、純音の oddball 系列である。
  - Barascud の刺激はトーン・ピップの系列である。
  - Anikin は合成系列で、評定であり、神経応答ではない。
  - 持続する複合音のスペクトルの連続的な変化に、そのまま一般化できることは示されていない。
- 条件1の判定は、応答ごとに分ける。R1a と R1b は「候補機構あり、本機能の指標との対応は未確立」、R1c は「行動上の証拠あり、機構未提示」とする。

## 5. 条件2：計算モデル

### 退けた候補：PIR

予測情報率（PIR）は、既知のモデルの下の離散値の決定論的な列では 0、i.i.d. 列でも 0 になる（Abdallah & Plumbley 2009, *Connection Science* 21:89–117）。連続値で雑音のない決定論的な過程では、相互情報量が特異になり、0 とは言えない。しかし定常なガウス AR(1) 過程（係数 a ≠ 0）では、学習が完了した後も `½ log(1 + a²) > 0` のままである。定常な有色ノイズ（たとえば Flora の残差帯域）は恒常的に正の値を得るので、「静止したノイズへの縮退」を防げない。PIR は学習の進み具合とも別の量である。予測情報（Bialek, Nemenman & Tishby 2001, *Neural Comput* 13:2409–2463）も、構造の複雑さの尺度であって、快や新しさの尺度ではない。IR（Dubnov, McAdams & Reynolds 2006, *JASIST* 57:1526–1536）は、決定論的な周期で高くなる。

### 比較対象の暫定案：二つの候補の指標

**驚きの指標。** 既存モデルの採用を優先し、Anikin 2026 の指標を原型とする。

- **原型。** 同論文の Shannon surprisal は、ガウス密度をその最大値で割ったものの負の対数である。自然対数で書くと `S = (y − μ)² / (2 s²)` になり、`μ` と `s²` は分析窓の中の履歴の平均と分散である（同論文の式 4。独立レビューが本文で照合した内容による。草案の執筆者は本文を読んでいない）。正の分散の下で値は 0 以上であり、0 になるのは現在値が履歴平均と一致するとき（`y = μ`）に限る。周期的な列のように予測できる変化でも、履歴平均から外れれば正の値をとる。
- **変更点**（原型と区別して記録する）:
  1. 入力は bin ごとの NSGT 対数パワー `y_t = ln(P_t + ε)` とする。原型のスペクトル表現の細部は未照合。
  2. 分析窓は、指数重みの履歴として実装する。窓の長さ、履歴のカーネル、更新の間隔の三つは別々のパラメータとする。Anikin の約 1 秒は分析窓の長さについての結果であり、`SpectralHistory` の 8 つの年齢の重みに直接は移せない。
  3. 評価は bin ごと、hop ごとに行う。原型は系列全体の評定との対応を示しただけで、瞬時の bin 値の妥当性は示していない。その対応を本指標に拡張しない。
- **時刻の境界。** `μ_t` と `s²_t` は時刻 `t − 1` までの履歴から計算し、`y_t` を含めない。評価を済ませてから更新する。分散の床 `s²_min` を置く。
- **拡張案（authored）。** 線形予測（RLS、忘却係数 `λ`）で予測値 `μ̂_t` を作り、分母には予測残差の二乗平均 `m̂²_t`（残差二乗の EMA）を使う。`S′ = (y − μ̂)² / (2 m̂²)` とする。RLS は残差の平均が 0 であることを保証しないので、`m̂²` は分散ではなく二乗平均（`E[e²] = Var(e) + E[e]²`）である。分散を意図する場合は、同じ重みと整合した初期化のもとで、二乗平均から残差平均の EMA の二乗を差し引く（`v̂ = EMA(e²) − [EMA(e)]²`）。平均だけを置き換えて原型の履歴分散 `s²` を残すと、予測残差の尺度にはならないので、分母も合わせて定義し直す。たとえば振幅一定の交互列は、履歴分散が正でも、直前値の符号を反転するだけで完全に予測できる。周期的な構造を取り込めるが、原型からは離れる。原型に従う版と拡張版は、別の候補として比べる。

**精度の指標。** 候補ごとに定義が異なる。

- **原型版。** 履歴分散の逆数 `π_t = 1 / s²_t`。履歴の中の散らばりの尺度であり、予測誤差の尺度ではない。
- **拡張版。** 予測残差の二乗平均の逆数 `π′_t = 1 / m̂²_t`。
- どちらも参照値 `π_ref` に対して `[0, 1]` に写す（`precision_state = π / (π + π_ref)`）。
- いずれも、選んだ表現と予測についての尺度である。Barascud の持続応答がこの指標を表すという証拠はない。R1b に対応するかどうかは、検討すべき仮説である。

**bin ごとの近似。** 各 bin を、その bin 自身の履歴だけで評価する。他の bin が持つ予測情報や、和音としての共同変化は落とす。名前は「bin ごとの自己履歴に条件づけた指標」とし、スペクトル全体の情報量とも、候補音の価値とも同一視しない。

**忘却。** PPM-Decay（Harrison, Bianco, Chait & Pearce 2020, *PLoS Comput Biol* 16:e1008304）は、記憶の減衰を予測モデルに入れる先例として挙げる。数値の根拠ではない。

### 縮退時の振る舞い（未検証の仮説）

いずれも、上の表現と指標のもとで立てる仮説であり、導いたものではない。驚きは時点ごとに変動するので、平均、分布、時間平滑した値を区別して測る。

- **一定の音。** 驚きは小さい。精度は高い。
- **定常のノイズ。** 白色の PCM でも、NSGT の重なった窓と平滑化された履歴によって bin のパワーは時間相関を持ちうる。小振幅のノイズや予測しやすい有色ノイズでは残差が小さく、精度が高くなりうる。「ノイズは低く評価される」は検証すべき仮説である。
- **学習済みの周期。** 原型（履歴平均）では、周期的な変化がそのまま驚きとして残りうる。拡張案（線形予測）では、驚きが小さくなりうる。
- **規則の中の逸脱。** 精度が高い文脈で驚きが大きくなる。

### 既存の実装との関係

- core 既存の帯域エネルギー予測（`HistoryEnergyPrediction`）は 3 帯域の IIR クロスオーバー（200 Hz / 3000 Hz、`src/core/stream/dorsal.rs`）で、Log2Space に整列していない。周辺分散と共有の残差統計も持たない。したがって、そのまま代用することはできない。ridge・RLS の更新式という部品は流用できる。
- ledger の表の「短期の再帰期待と habituation」の行には「core 採用は既存実装に限る」とある。本機能はこれと別の表現なので、既承認の扱いにはせず、新規の審査とする。

### 既存モデルとの比較（未了）

- Anikin 2026 の指標（R package soundgen に実装）を、本文と実装に当たって確かめる必要がある。特徴空間、窓、Bayesian surprise の成績が対象である。今回見つかった soundgen の `getSurprisal` の説明は、実験的な版のものだった（rdrr.io、tatters/soundgen）。内容は、bark 尺度で等間隔の 64 チャネルの聴覚スペクトログラム（step 20 ms、窓 2 s）の上で、チャネルごとに自己相関に基づく驚きを計算し、振幅で重みづけして平均するものである。2026 年の論文で成績のよかった Shannon surprisal の実装は、まだ特定できていない。
- 採用できれば、条件2の「既存モデルの採用を優先する」を満たす最も強い経路になる。
- Abdallah & Plumbley の後続研究（ガウス過程、オンライン推定）とも比べてから、本構成を選ぶ理由を確定する。

### 工学規則として未確定の値

`ε`、`s²_min`、`τ_s`、`λ`、`π_ref`、特徴の年齢の重み、更新の周期（hop ごと）、欠測（解析の取りこぼし）の扱い、学習の開始期間。登録の前にすべて明記する。

**評価。** 条件2は未充足とする。驚きの指標は Anikin 2026 の原型を採用でき、変更点も明示した。ただし数値、更新式、初期化、欠測の扱いが未確定である。精度の指標は工学規則の新しい表現にあたる。判定は、指標を原型どおりに採るか拡張するかでも変わる。どちらの場合も、人の評定に当てはめることはしない。

## 6. 条件3：判断

- **A.** 作者の試聴と、測定値との照合で判断できる。比較する素材は、純音、定常の白色雑音と有色雑音、決定論的な周期、規則の中の逸脱、Flora の三構成、既存のサンプル。被験者からの新たなデータ収集は要らない。
- **B.** 結合した生成の可聴差は、A の観測だけでは確かめられない。B を実装してからの試聴が必要である。「作者の判断で完結できる」ことと、「試聴して採用した」ことは別に記録する。

## 7. 二層分類

| 部分 | 原理層 | authored 層 |
|---|---|---|
| 驚き（原型版） | 原理層の資格は自動では成立しない。Anikin の式 4 を原典との対応を照合して採用した場合に限り、その式の部分が既存の刺激→知覚モデルにあたる。 | 入力表現、窓の実装、床、bin・hop ごとの評価 |
| 精度（原型版：履歴分散の逆数） | — | 写像、`π_ref`、窓の実装 |
| 驚き・精度（拡張版：RLS の予測と残差の二乗平均） | — | 予測器、忘却、残差二乗の EMA、写像の全体 |
| 予測器 | — | 特徴（8 尺度）、線形の形、`λ`、`τ_s`、床、`π_ref` |
| 地形化 | — | bin への割り当て、時間平滑、報告の周期 |
| 生成への結合（B） | — | 結合の形（例：精度 × 驚き）と係数。C の `a, b, c, d` と同じ扱い |

authored 層の値は、認知定数とも当てはめた値とも呼ばない。

## 8. 判定案

- **A（観測）。** 既定 off の受動的な診断として置ける。三条件の判定は、条件1は応答ごとに「候補機構あり、対応未確立」（R1a・R1b）と「行動証拠のみ」（R1c）、条件2は未充足（値と更新則が未確定）、条件3は可。
- **B（生成への結合）。** 今回は判定しない。精度と驚きの指標は、それぞれ R1b と R1a・R1c への対応を検討する候補である。それらの組み合わせを生成の価値にすること（例：「予測できる文脈の中の驚き」が大きい所へ音を置く）は、知覚量ではなく生成方針である。動機は Cheung 2019 の交互作用だが、それを主張しない。結合の形は authored 層とし、A の実測と B の試聴の後に判定する。

B を審査するときには、次の区別が要る。

- A が評価するのは、現在の入力である。
- 候補の音を置いた後の驚きを比べる生成方針には、「候補の音をそこに置いたら、聴き手にとってどれだけ驚きになるか」という反実仮想の評価が必要になる。これは A とは別の量である。現在の観測値にそのまま反応する生成方針もありうるので、反実仮想の評価は B の一般的な必須条件ではない。
- 鳴っていない bin も、無音を観測し続ければ無音として予測可能になる。忘却が働いても、その場所の観測された驚きが自動的に戻るわけではない。
- したがって「鳴り続けた場所は価値が下がり、音はよそへ移り、価値は戻る」という性質は、A からは導けない。生成方針と候補の評価をどう定義するかに依存する。

## 9. 作者が決める点

1. 主張する対象応答を R1a・R1b・R1c とし、R2 は表現への示唆、R3 は動機にとどめるか。
2. 量を、精度と驚きの二つの scan とするか（PIR は退ける）。
3. 驚きの指標を Anikin 2026 の原型（履歴平均）のまま採るか、線形予測に拡張するか。原型なら条件2が強くなり、拡張なら周期的な構造を扱える。
4. soundgen の現行の実装と論文の本文を照合し、原型の細部（スペクトル表現、窓）を確定するか。
5. 地形を habitat 由来にするか、presentation 由来にするか。既存の habituation には、生態系用と ListenerTwin 用の二系統（`hab_ecology` / `hab_listener`）という先例がある。
6. B の判定を、A の実測の後にするか。

## 10. 照合した文献

- Abdallah, S., & Plumbley, M. (2009). Information dynamics: Patterns of expectation and surprise in the perception of music. *Connection Science*, 21, 89–117.
- Anikin, A. (2026). Measuring surprisal in sound sequences. *Behavior Research Methods*, 58. doi:10.3758/s13428-026-03153-3
- Barascud, N., Pearce, M. T., Griffiths, T. D., Friston, K. J., & Chait, M. (2016). Brain responses in humans reveal ideal observer-like sensitivity to complex acoustic patterns. *PNAS*, 113(5), E616–E625.
- Bialek, W., Nemenman, I., & Tishby, N. (2001). Predictability, complexity, and learning. *Neural Computation*, 13(11), 2409–2463.
- Cheung, V. K. M., Harrison, P. M. C., Meyer, L., Pearce, M. T., Haynes, J.-D., & Koelsch, S. (2019). Uncertainty and surprise jointly predict musical pleasure and amygdala, hippocampus, and auditory cortex activity. *Current Biology*. doi:10.1016/j.cub.2019.09.067
- Chmiel, A., & Schubert, E. (2017). Back to the inverted-U for music preference: A review of the literature. *Psychology of Music*, 45(6), 886–909.
- Dubnov, S., McAdams, S., & Reynolds, R. (2006). Structural and affective aspects of music from statistical audio signal analysis. *JASIST*, 57(11), 1526–1536.
- Garrido, M. I., Kilner, J. M., Stephan, K. E., & Friston, K. J. (2009). The mismatch negativity: A review of underlying mechanisms. *Clinical Neurophysiology*, 120(3), 453–463.
- Gold, B. P., Pearce, M. T., Mas-Herrero, E., Dagher, A., & Zatorre, R. J. (2019). Predictability and uncertainty in the pleasure of music: A reward for learning? *Journal of Neuroscience*, 39(47), 9397–9409.
- Harrison, P. M. C., Bianco, R., Chait, M., & Pearce, M. T. (2020). PPM-Decay: A computational model of auditory prediction with memory decay. *PLoS Computational Biology*, 16(11), e1008304.
- Lieder, F., Daunizeau, J., Garrido, M. I., Friston, K. J., & Stephan, K. E. (2013). Modelling trial-by-trial changes in the mismatch negativity. *PLoS Computational Biology*, 9(2), e1002911.
- McDermott, J. H., Schemitsch, M., & Simoncelli, E. P. (2013). Summary statistics in auditory perception. *Nature Neuroscience*, 16(4), 493–498.
- Salimpoor, V. N., Benovoy, M., Larcher, K., Dagher, A., & Zatorre, R. J. (2011). Anatomically distinct dopamine release during anticipation and experience of peak emotion to music. *Nature Neuroscience*, 14, 257–262.
- Ulanovsky, N., Las, L., & Nelken, I. (2003). Processing of low-probability sounds by cortical neurons. *Nature Neuroscience*, 6(4), 391–398.
- Ulanovsky, N., Las, L., Farkas, D., & Nelken, I. (2004). Multiple time scales of adaptation in auditory cortex neurons. *Journal of Neuroscience*, 24(46), 10440–10453.

第1稿から外した文献とその理由：

- Singer et al. 2018（*eLife* 7:e31557）。予測学習によって受容野の特性を再現した研究で、本機能の機構の同定にはならない。
- Itti & Baldi 2009（*Vision Res* 49:1295–1306）。Bayesian surprise が注意を引くことを示したが、視覚であり、本審査の対象応答ではない。
- Gottlieb & Oudeyer 2018（*Nat Rev Neurosci*, doi:10.1038/s41583-018-0078-0）。学習進捗を好みとして扱う総説で、聴覚の水準の証拠ではない。
