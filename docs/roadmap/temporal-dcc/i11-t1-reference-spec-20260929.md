# I11-3 / T1：Dau 1996の原典核と参照実装の境界

2026-09-29。[入力対応草案](i11-t1-model-mapping-20260929.md)の原典不足を補う読取仕様。**式(A5)とStrube 1985の係数定義は原PDF画像で確認できた。ただし、Dauが使用したStrubeの設定、離散適応実装、音圧校正、雑音分散まで一意には復元できていない。** 本書はモデル採用・係数凍結・実装・数値取得の承認ではない。

## 1. 出典と版

| 略号 | 一次資料・確認範囲 |
|---|---|
| D96-I | Dau, Püschel, Kohlrausch, JASA 99, 3615–3622。[出版社版PDFをTU/eが公開](https://pure.tue.nl/ws/files/1471028/622101.pdf)、[DOI](https://doi.org/10.1121/1.414959)。§I.B–C、Appendix、脚注。冊子p.3621 / PDF第8ページのA5–A7と、p.3617のFig.2を画像確認した |
| D96-II | 同著者、JASA 99, 3623–3631。[出版社版PDFをTU/eが公開](https://pure.tue.nl/ws/files/1307552/622104.pdf)、[DOI](https://doi.org/10.1121/1.414960)。§I、§II.A–Bと図説明。図の点列を数値化していない |
| S85 | Strube, *A Computationally Efficient Basilar-Membrane Model*, Acustica 58(4), 207–214。[EAA公式公開プレビュー](https://dael.euracoustics.org/bin/EAA/v3.20/quickview?id=58387)。公開指定の通常download手順で全文を取得。§2–3、pp.208–209、Fig.2–3と無番号係数式をPDF画像で確認した |
| AMT16 | [AMT公式配布案内](https://amtoolbox.org/download.php)から[1.6.0のsource archive](https://sourceforge.net/projects/amtoolbox/files/AMT%201.x/amtoolbox-1.6.0.zip/download)を取得し、必要なsourceだけ読んだ。Dau自身を著者欄に含む後年実装であり、1996年の実験プログラムではない。実行・インストールしていない |

公開資料の作業cacheは `target/i11-t1-primary-20260929/`。原文全文や画像は本書へ転載しない。S85のPDFは機関表紙を含み、冊子p.209はPDF第4ページである。以下の `AMT16/...:line` はcache中の `amt160-source/amtoolbox-1.6.0/` を起点とする。資料hashと取得元は同cacheの `manifest.json` に残す。

## 2. 原典の核と、埋めてはいけない不足

分類は **原典**＝当該論文に明記、**後年**＝AMT16で確認、**不足**＝今回の資料ではDau原実装を特定できない、の三つである。

| 段階 | 原典で確定した核 | 後年実装・不足 |
|---|---|---|
| 波形と基底膜 | D96-I §I.B.1 p.3616：Strube型の線形wave-digital filter、120出力、実験では信号周波数に対応する1出力を評価 | S85の空間分割・係数は下記。D96-Iは減衰値・終端条件・出力gain・チャンネル選択法を列挙していない。120はERB等間隔120帯域という指定ではない |
| 整流と1 kHz LP | D96-I p.3616：半波整流の後に1 kHz low-pass | 原論文の次数・離散係数・初期状態は未特定。AMT16 `common/ihcenvelope.m:150` は半波整流、`butter(1,2000/fs)`、`defaults/arg_ihcenvelope.m:28` は1段。これを1996原係数と認定しない |
| 適応5段 | D96-I p.3617 Fig.2：各段の出力をLPし、入力の除数へ戻す。定常出力は入力の32乗根。p.3621脚注1とD96-II p.3627の実使用時定数は5, 50, 129, 253, 500 ms | 原論文の定常式と時定数は離散更新の順序・初期値を指定しない。AMT16の具体的再帰を§4に区別して示す |
| floorとMU | D96-I p.3617：適応入力の定数下限。定常入力0–100 dBに対応する適応出力の両端を0–100 model unitsへ再尺度化する。これは適応出力に対する線形変換であり、入力dBに対する厳密な直線ではない | 原実装の下限振幅とfilterbank出力の音圧換算は未特定。AMT16の`minspl=0`、振幅1を100 dBとする内部換算は後年実装の規約である |
| 適応後LP | D96-I p.3617：8 Hz、時定数20 ms | AMT16 `models/dau1996.m:125` は `a=exp(-1/(0.02 fs))`、`b=1-a`。これは20 msを離散化した後年係数で、8 Hzを厳密なデジタル−3 dB周波数に合わせた指定ではない |
| 内部雑音 | D96-I pp.3617–3618：長い60 dB SPL信号の約1 dB弁別閾で分散を校正し、その後固定。Appendix p.3621：標本間無相関、Gaussian、同分散 | 数値分散、校正信号の完全な指定、内部表現の標本時計は未特定。校正手続があることと数値定数が既知であることを分ける。AMT16の`dau1996`は雑音・検出器を実装していない |
| templateとcriterion | D96-I §I.C p.3618、脚注2 p.3621：閾上信号の内部差を単位energyへ正規化し固定。3IFCの70.7%正答を閾値に使う。Appendix A5–A7は§3 | 閾上レベル・比較窓・離散内積の時間刻みと雑音分散を一組で特定する必要がある。sample rateだけを変えて同じσを使えるとは限らない |

## 3. Strube係数とA5：画像で読めた参照核

S85 §2 p.208では区間長Dに対して `Lsn=2ρD/An`、`Ln=Mn/(bnD)`、`Cn=bnD/Sn`、`Rn=Vn/(bnD)`。ρは液体密度で、ここで独自の数値を補わない。§3 p.209の係数式は次の通りである。`n`は基底膜位置、`fs`はこのfilterを更新するsample rateである。

```text
Zn    = 2 fs Ln + 1/(2 fs Cn) + Rn
Wn    = 1 / (1/Zn + 1/(W(n-1) + 2 fs Lsn))
αRn   = Rn/Zn
αLCn  = (4 fs² Ln Cn - 1)/(4 fs² Ln Cn + 1)
αsn   = W(n-1)/(W(n-1) + 2 fs Lsn)
αpn   = Wn/(W(n-1) + 2 fs Lsn)
```

これは係数定義であり、Fig.3の全状態更新式を転記した実装仕様ではない。原図では前向きの波をn増加順、後向きをn減少順で更新し、出力は横方向体積速度 `In=(an−bn)/Zn`。変位を得る場合は時間積分が別途必要である。波の変数は計算上の定義で、実際の蝸牛進行波そのものと同一視しない。

| S85 p.209の設定 | Dau原実装との未解決点 |
|---|---|
| `N=120`、基底膜長32 mm、`D≈0.267 mm`。場所依存は `F(n)=F0 exp(F′(n−1)D)`。孤立LC共振は `26.28 kHz exp(−0.16(n−1)D)` | 約5区間/Barkという説明でありERB scanではない。実際の電流共振は減衰にも依存し、孤立LC共振と一致しない |
| `b0=0.08 mm`、`A0=1 mm²`、`M0/b0=0.77 mg/mm³`、`V0/b0=(63 または13)×10³ mg/(mm³ s)`、`S0/b0=21×10⁹ mg/(mm³ s²)`。指数係数は `b′=.05`、`A′=0`、`M′−b′=−.05`、`V′−b′=−.21`、`S′−b′=−.37`（すべてmm⁻¹） | S85自身に二つの減衰設定がある。D96の選択が今回の本文では特定できない。小さい減衰では電流共振が約0.96倍になるというS85の値も、その設定に限る |
| 入力終端 `W0=√(Ls1/C1)`。頂端は抵抗終端から誘導終端も試している。通常24 kHz、pulse実験50 kHz | D96-II p.3624の30 kHzは**音響刺激生成**のrate。S85の24/50 kHzをD96へ引き継いだと推定しない。D96内のfilter時計・終端・再標本化は未特定 |
| 双一次変換でアナログfが `(fs/π) atan(πf/fs)`へ写る（S85 p.208） | 現runtimeのrateへ変更する場合、係数再計算だけで中心周波数の対応が保たれると扱わない。700 Hz未満のBark尺度との不一致も原著自身が指摘する |

D96-I p.3621式(A5)は次の通り。`eν`は観測内部差の標本、`sν`は期待信号、`σ²`は内部雑音分散、`l`は尤度比である。

\[
\ln l(e)=\frac{1}{\sigma^2}\left\{\sum_\nu e_\nu s_\nu-\frac12\sum_\nu s_\nu^2\right\}. \tag{A5}
\]

単位energyのtemplateでは相関のnoise分散がσ²となる。信号あり相関の平均をdとすると、m区間強制選択の正答確率は原式(A6–A7)で求められる。

\[
P_m(\mathrm{correct})=\int_{-\infty}^{\infty}\phi(x-d/\sigma)\Phi(x)^{m-1}\,dx,
\qquad \Phi(x)=\int_{-\infty}^{x}\phi(u)\,du.
\]

`φ`は標準正規密度。これは条件付きGaussianモデルの検出核であり、onset費用や音楽的な良さを定義しない。既知の内部差・template・σ・標本時計を与えれば参照計算を定義できるが、現I11入力からそれらが得られるとの主張ではない。

## 4. AMT16が補う離散仕様と、原典との差

[公開コード説明](https://www.amtoolbox.org/amt-1.6.0/doc/models/dau1996_code.php)と配布sourceの `models/dau1996.m:78` はDau、Jepsen、Søndergaardを著者欄に置く。一方 `mex/comp_adaptloop.c:1` はEwertの1999–2004年実装をMajdakが2016年に適応したと記す。**著者の関与は版の同値性の証明ではない。**

| 確認箇所 | 読み取った後年実装・扱い |
|---|---|
| `models/dau1996.m:44`、`:114` | Strubeの代わりにgammatoneを使い、原モデルとして不正確である旨を実装自身が記す。Strube再現の代用品にはしない |
| `models/dau1996.m:106`、`defaults/arg_adaptloop.m:20`・`:24` | 単独`adaptloop`既定は`limit=10`だが、`dau1996`経路は`adt_dau1996`を読み`limit=0`にする。1997型overshoot制限を加えたというwrapperの説明文だけを根拠に、現在の呼出しを制限ありと判定しない |
| `mex/comp_adaptloop.c:56`・`:62`・`:130` | 段jで `aj=exp(−1/(τj fs))`、初期状態 `qj=m^(1/2^j)`。毎標本、`y0=max(x,m)`、`yj=y(j−1)/qj_old`、`qj_new=aj qj_old+(1−aj)yj`。除算が状態更新より先にある |
| `mex/comp_adaptloop.c:88`・`:135` | 5段後のMUは `100(y5−m^(1/32))/(1−m^(1/32))`。これはコードの式であり、D96本文から唯一に導いた離散仕様ではない |
| `common/adaptloop.m:115`・`:134`、`defaults/arg_adaptloop.m:21` | `minspl=0`を振幅1＝100 dBの規約で線形振幅へ変換する。現PCMの1を100 dB SPLとする校正は別途必要。物理Pa単位の既定規約と混ぜない |
| `mex/comp_adaptloop.c:221`、`models/dau1996.m:130` | C入口は呼出しごとにinit/set/run/freeする。wrapperはfilter状態を外部に返さない。連続streamでhopごとに呼ぶだけでは履歴保持の参照にならない |

ここで示した再帰はC経路の読解である。MATLAB fallback、Octave、依存library間の数値同値性は未検証。後年参照の試験を登録するなら、AMT版・選択経路・引数・依存版を固定し、D96再現と別の比較名にする。

## 5. 原典再現試験の最小条件案と不足

以下は**未実施の登録候補**で、許容値や取得開始を決めていない。

| 最小試験 | 固定すべき入力・期待値の出所 | 現時点の入口 |
|---|---|---|
| 検出核の単独検査 | 同じ有限e、s、σ、内積規約に対してA5の相関と尤度比の順序、A6のchance値`1/m`・信号増加方向を確認。template energyと校正を独立記録 | 式は確認済み。有限fixture・数値積分法・許容を先に登録すれば技術作業を分離できる。生態系や原図の再現とは呼ばない |
| Strube単独参照 | pp.208–209の係数、Fig.3全更新、減衰・終端・ρ・fs・出力単位を列挙し、インパルス/定常応答の独立参照を決める | 係数は得られたが、D96が選んだ設定とgolden出力は未入手。現在の式表だけで完了しない |
| 適応状態 | floor定常状態、段ごとの定常32乗根関係、全長処理と状態を保持した分割処理、同じcutからの候補分岐が観測状態を汚さないこと | 原著の動的goldenは未入手。AMT16 C経路との比較なら「後年実装一致」である。hopごとのresetを許容しない |
| 同時条件 | D96-II p.3624 Fig.1：300 ms frozen noise（20–5000 Hz、77 dB SPL）、5 ms・1 kHz Hanning信号。位置を変える | 元のfrozen-noise標本・信号位相、数値閾値、全前処理設定が未入手。新seedで原図の各点一致を要求しない |
| 前方条件 | D96-II pp.3626–3627 Fig.7：200 ms背景、10 ms・1 kHz信号。横軸は背景offsetから信号offsetまで。40 ms点ならonset gapは30 ms | 同じ未入手項目に加え、prehistory/reset、比較窓、校正σを固定する。横軸が正というだけでは非重複ではない |

不足の理由は、S85本文自体の取得不能ではなく、**S85の複数設定とD96実行版の対応・原著プログラム/刺激が今回の公開資料で特定できないこと**である。原論文は後年codeへの恒久リンクを持たず、AMT16は前段を置換している。無限に探索を延長せず、この不足を次の判断へ渡す。GM05は前回の[AES抄録](https://aes.org/publications/elibrary-page/?id=13391)以上の全文を今回確認していない。Dauの不足からGM05の前方適合や軽量性が証明されるわけではない。

## 6. 現入力への影響と、次の有限一単位

現PCM入口は [src/runtime/mod.rs:3003](../../../src/runtime/mod.rs#L3003)、自己除去PCMは [src/core/temporal_expectation.rs:248](../../../src/core/temporal_expectation.rs#L248)。一方、未来背景は [同:780](../../../src/core/temporal_expectation.rs#L780)のenergy予測、代表自声は [src/life/action_candidates/footprint.rs:283](../../../src/life/action_candidates/footprint.rs#L283)の正規化energyであり、Strube入力波形への同値変換は得られていない。今回の原典補完はこの入力不足を解消しない。

**次の一単位：AMT16 C適応核だけの独立比較登録を作る。** limit=0の後年離散再帰を固定し、独立な状態保持実装との比較、初期floor・定常入力・分割処理・候補forkと観測状態の分離を有限fixtureで定める。これは全Dau pipeline、Strubeや音圧校正、心理物理閾値の再現ではない。実行前に入力・数値許容・分母・終了点を登録し、モデル採用とは分ける。原Dau再現の路線を選ぶ場合には、初稿で候補としたStrube Fig.3全状態更新の追加復元へ戻る。その場合も不明なDau設定を推定で埋めない。

この比較登録の具体化は追加のモデル採用判断なしで進められる。D96再現を求めるか、AMT16等の変更モデルを独立比較候補にするか、現入力契約を波形まで広げるかは作者判断に残す。T2は前処理・適応・8 Hz LPのどこを共有するか未決とし、Phase 4へはモデル版、波形・尺度・状態cut・時計・欠測を引き継ぐ。既存のT1→T2およびI12b/I4統合順序は[対応草案§5](i11-t1-model-mapping-20260929.md#5-t2--phase-4へ渡す最小の記録)を維持する。

検証：原PDFの式・図の画像照合、公式配布sourceの静的読取、文書の参照確認のみ。原典再現、近似補完、係数凍結、刺激・性能測定は行っていない。

同日の[独立読取レビュー](../../../target/i11-t1-primary-20260929/independent-review.md)でA5/A6、S85係数・時計、AMT呼出しと再帰、現入力境界を照合し、資料hash16件が一致した。MUの再尺度化を明確化し、先行対応草案から本書への案内を追加した。原Dauの全pipeline取得へは進まず、AMT16 C適応核だけの独立比較登録を次の有限単位とする。Strube全更新の追加復元は、原Dau再現の路線を選ぶ場合に扱う。

## 後続の1チャンネルPCM参照検査（2026-09-29）

作者はAMT16変更版を名前付き参照候補として検証する路線と、未来背景PCM欠測時はunknownとして新T1を適用しない扱いを承認した。[入力契約と判断記録](../../../target/i11-t1-model-decision-20260929.md)に範囲を記載した。原Dauの厳密再現、モデル本体の採用、runtime接続をこの承認に含めない。

fs=48 kHz、fc=1 kHzのcomplex allpole gammatone→半波整流・1 kHz Butterworth→検証済みC適応核→20 ms LPをsource式から実装し、[取得前登録と単回結果](../../../target/i11-t1-pcm-reference-20260929/registration.json)を保存した。4刺激、143360要素の段階別比較、適応終状態20要素、chunk/fork 90比較・402560要素、状態検査18件が事前条件を満たし、exit 0となった。原版MATLABでの実行ではない。

| 同じ入力を渡した段階比較 | 最大絶対差 |
| --- | --- |
| Gammatone: 4次多項式／4段cascade | 4.809994633495074e−12 |
| 半波整流 | 0 |
| IHC: direct／transposed | 5.421010862427522e−19 |
| 適応: 原C／独立Python | 9.094947017729282e−13 MU |
| 20 ms LP: direct／transposed | 0 |

登録SHA-256は `6d49b97f253b738c2e45d61d666b9c199d848f613291b3d380e84ac1c3fb9620`。統括は60個の保存rawのhash・長さ・有限性と、段階別全出力の誤差を[独立再集計](../../../target/i11-t1-pcm-reference-20260929/independent-reduction.json)した。状態・chunk/forkは実装と保存判定を確認した範囲で、独立に全実行し直したとは扱わない。

二つの全chain間では最大約2.7244e−7の最終出力差があり、これは非合否の診断値である。段階別許容を非線形適応の後段へ流用せず、全chain同士の同値性を宣言しない。chunk/forkのbit一致は各chain内の状態保持を示す。単チャンネルの回路比較から検出閾、同時／前方の聞こえの差、マスキング地形、複数帯域、T2、通常runtimeの成立を認定しない。
