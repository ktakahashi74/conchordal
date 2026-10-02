# I11-3 / T1：Dau 1996と現入力の対応草案

2026-09-29。原典対応と入力適合性の判断資料。I11-3=T1、I11-4=T2という番号とDau 1996の優先照合は作者承認済み。モデル採用、係数、独自写像、登録凍結は未承認である。今回は読解と文書だけを行い、実装・刺激作成・取得・性能調整は行わない。

**現I11の3帯域energyと16区間footprintを、そのままDauモデルの入力とする根拠はない。** 波形入口は存在するが、原典前処理、候補自声の波形、未来背景、音圧尺度、適応状態の所有を別途定める必要がある。優先照合を継続できることと、現入力で実装可能であることを分ける。

## 1. 読んだ一次資料と未確認部分

- **D96-I**：Dau, Püschel, Kohlrausch, *A quantitative model of the “effective” signal processing in the auditory system. I. Model structure*, JASA 99, 3615–3622。[TU/e公開の出版社版](https://pure.tue.nl/ws/files/1471028/622101.pdf)、[DOI](https://doi.org/10.1121/1.414959)。§I.B–C、§II、Appendix・脚注を本文抽出で確認した。
- **D96-II**：同著者、*II. Simulations and measurements*, JASA 99, 3623–3631。[TU/e公開の出版社版](https://pure.tue.nl/ws/files/1307552/622104.pdf)、[DOI](https://doi.org/10.1121/1.414960)。§I、§II.A–Bと対応する図説明を本文抽出で確認した。以下のページは冊子ページで、PDF先頭の機関表紙を含まない。
- **GM05**：Glasberg–Moore 2005、JAES 53, 906–918。[AES原論文の抄録](https://aes.org/publications/elibrary-page/?id=13391)。出版社ページは本文ダウンロードに会員・購読を要求し、今回は全文未取得。信号・背景のexcitation patternから部分ラウドネスを計算する比較候補に留め、前方マスキングの適合、式、費用は未判定とする。

本対応草案の初回読取時点では、図の点列の数値化、Strube 1985の原式・係数、原著者の実装・刺激ファイルは未確認だった。D96-Iの式(A5)は今回の本文抽出で式本体が欠落したため転記しない。図版の画像照合も未完である。下表の式は読めた本文に対応する範囲だけを本書の記号で表したものであり、離散実装の仕様ではない。

同日後続の[参照仕様](i11-t1-reference-spec-20260929.md)でA5の画像、Strube係数、AMT16の適応実装を確認した。原Dauの具体的設定・校正・元刺激の不足は残る。上の未確認記述は初回の調査範囲であり、現在の補完状態は参照仕様を正とする。

## 2. 原典から必要になる量と、現在ある量

`R(x)`を原典の内部表現、`M`を背景波形、`T`を検出対象波形とする。原典由来の処理、数値実現の選択、行動への作者規則を別欄に置く。

| 原典の段階・出典 | 原典から確認した内容 | 現入力との対応・不足 | 分類と次判断 |
|---|---|---|---|
| 波形前処理：D96-I §I.B.1、p.3616、Fig.1 | Strube型の線形基底膜モデルは120出力。原実験では信号周波数のチャンネルを評価し、半波整流、1 kHz low-passを通す | 現NSGT powerも3帯域平均energyも波形・位相を保持しない。ERB多帯域地形への対応は未照合 | 原典前処理の係数を確認する。別filterbankへの置換・多帯域集約は、原典と同値と示すまでは変更として扱う |
| 非線形適応：D96-I §I.B.2、p.3617、Fig.2 | 5段の除算feedback。定常時は1段で `O=√I`、n段で `O=I^(1/2^n)` | 帯域ごとの適応状態が必要。現在のenergy履歴やhabituationを同じ状態とみなす根拠はない | 動的更新式、初期値、下限、精度、サンプル周期の実現を照合する。定常式だけから逐次更新を自作しない |
| 適応の版：D96-I p.3621脚注1、D96-II §II.B、p.3627 | 実使用時定数は `5, 50, 129, 253, 500 ms`。旧等間隔列の376 msを50 msへ変更 | 現コードへの採用値ではない。異なる記述の数列を混ぜない | 原典再現の候補値。Conchordalの係数凍結・正当化とは別 |
| 内部表現：D96-I §I.B、pp.3616–3618 | 適応後に8 Hz low-pass、内部雑音。モデル単位MUと音圧レベルの対応を持つ | producer recordの`energies: [f64; 16]`は未正規化値を保持するが、消費者の`BodyFootprint.power`はpeak正規化され、元のレベルを保持しない。どちらもデジタル振幅から音圧尺度への規則は未定 | 正規化、尺度、雑音分散の数値対応を確認する。任意振幅をdB SPLと呼ばない |
| templateと検出：D96-I §I.C、p.3618、Appendix pp.3621–3622 | 閾上信号から `ΔR=R(M+T)−R(M)` を作り単位energyへ正規化。実試行の差との相関を使う。式(A1–A4)は独立Gaussian標本の尤度、(A6–A7)は強制選択の正答確率 | 現onset候補は聴取試行ではなく未来の行動。対象templateと背景参照、観測区間の取り方が欠ける | 原典検出器の出力を参加費用へ直結しない。templateの作り方の変更も明記する |

原典I §II（p.3620）はチャンネル間処理などの限界を述べる。原典II §II.A.3（p.3625）では信号長依存に不一致がある。候補として使う理由は同時・前方を扱うことであり、T1全条件や時間的まとまりの説明が既に成立したからではない。

以下は2026-09-29に読んだ**main上のsource**との照合であり、隔離energy-loop版の機能・資源証拠を更新しない。リンク表示の `path:line` はこの読取時点の位置である。

| 現在の入力・処理 | 読取根拠 | 原典対応への意味 |
|---|---|---|
| habitat PCMとVoiceごとの自己除去PCM | [src/runtime/mod.rs:3003](../../../src/runtime/mod.rs#L3003)、[src/core/temporal_expectation.rs:248](../../../src/core/temporal_expectation.rs#L248)。後者は `habitat_mix−own_habitat` を波形で作る | 過去の背景波形を処理する入口の候補。ただし生態系内部の自己除去という出所を保持し、聴覚的な自声同定の証明とはしない |
| 3帯域energyと予測 | [src/core/stream/dorsal.rs:42](../../../src/core/stream/dorsal.rs#L42)、[src/core/temporal_expectation.rs:780](../../../src/core/temporal_expectation.rs#L780)、[同:938](../../../src/core/temporal_expectation.rs#L938)。200/3000 Hz crossover、約10 ms窓、局所履歴の予測 | ERB filterbank出力ではない。未来の帯域energyから背景PCMを一意に復元できない。予測背景の代替生成は独自仮定になる |
| 代表自声の16区間 | [src/life/voice.rs:1074](../../../src/life/voice.rs#L1074)、[src/life/action_candidates/footprint.rs:283](../../../src/life/action_candidates/footprint.rs#L283)。凍結`ToneEnergy`を`project_window`へ渡し、coherent energyをpeakで割る | 受領recordは時間energyで、ERB方向のscanでも原典波形でもない。代表描画を正とする身体契約に合わせ、波形供給と既存近道の照合を別途定める。身体種類の分岐を新設しない |
| 音響解析のpower | [src/core/stream/analysis.rs:76](../../../src/core/stream/analysis.rs#L76) | NSGT powerを保持するが、それを原典の半波整流前信号と呼べない。公開する地形が`*_scan`なら既存Log2Space契約を守り、ERB内部表現との変換を明記する |
| 現費用と欠測処理 | [src/life/temporal_participation.rs:661](../../../src/life/temporal_participation.rs#L661)、[src/core/temporal_expectation.rs:762](../../../src/core/temporal_expectation.rs#L762) | 現費用は `own*other/(own+other+ε)` の和と独自係数。予測地平後は平均energyへ延長する。いずれもDauの検出器やマスキング状態ではなく、そのまま名称変更しない |

## 3. 時計・状態・候補分岐の境界案

この表は新T1の**提案**であり、現在のI11登録を変更しない。必要な状態は有界にするが、長い履歴を一定秒で切る値や、未知から復帰する時間はまだ選ばない。

| 境界 | 現在確認できたこと | 新T1で登録する案・残る選択 |
|---|---|---|
| 観測と判断 | [src/runtime/mod.rs:2084](../../../src/runtime/mod.rs#L2084)で判断、同:2103で描画、同:2110で観測更新。forecastは[観測済み・利用可能な時計](../../../src/core/temporal_expectation.rs#L486)を持つ | `[observed_start, observed_end)`、`available_at`、`decision_at`、`candidate_onset`、実際の`applied_at`を分ける。判断より後のPCMを候補評価へ戻さない。更新はPCMのsample clockで行い、wall時計や候補数で進めない |
| 共通履歴と分岐 | [src/runtime/mod.rs:2488](../../../src/runtime/mod.rs#L2488)は共有forecastからVoice用の外部energy viewを作る | 同一Voiceの全候補が同じ観測cutと適応状態から分岐する案。各分岐の`M`と`M+T`を局所状態で進め、捨てる候補も選ばれた仮想結果も観測状態へcommitしない。共有状態は後で届く実PCMだけで進める |
| 自己除去と状態所有 | [src/core/temporal_expectation.rs:228](../../../src/core/temporal_expectation.rs#L228)は出生時に外部履歴を継承し、その後Voice別PCMを処理する | Voice間まで同じ適応状態を共有できるとは限らない。自己除去を採るなら非線形処理前に行う。混合音の適応状態から自声状態を引く案は採らない。既存自声を含む聴覚状態を使う案との選択は未決 |
| 未来背景と候補波形 | 未来予測はenergyだけ。代表recordもPCMを配送しない | 原典再現では既知刺激PCMを使う。実候補へ接続する際の未来背景、代表波形、位相、振幅、支持窓は別契約が必要。既知未来を使うoracle検査と因果runtimeを区別する |
| 配送と期限 | [src/life/action_candidates/energy.rs:1011](../../../src/life/action_candidates/energy.rs#L1011)は受領時刻を付与。[src/life/temporal_participation.rs:217](../../../src/life/temporal_participation.rs#L217)のstaleはidentity差。[footprint.rs:254](../../../src/life/action_candidates/footprint.rs#L254)のcomputed時刻は実測費用換算 | identity・観測cut・実受領・モデル版を照合する。`received_at≤decision_at`を要求する案。現staleを新しい時刻期限と同一視しない。TTL、最大遅延、予測地平、容量、超過時処理は未決 |
| 未知・無音・off | [src/core/temporal_expectation.rs:805](../../../src/core/temporal_expectation.rs#L805)は欠落を無音と区別。現footprintは[absent/stale/unsupported](../../../src/life/temporal_participation.rs#L543)を区別 | 観測無音と未知を分ける。入力不完全時は新T1項を不適用として理由を返す案で、0マスキングと断定しない。旧費用へ戻すかは未決。新T1 offと既存`None`の対象版・bit一致範囲を別登録する |
| 有界状態 | 原典のfilter/適応状態と、候補評価用の有限区間は別物 | 固定次元の再帰状態は長期履歴の全保存を要しない。ただし有限秒でresetする近似とは同じでない。保持する波形窓、帯域数、候補数、worker、コピー量を列挙し、R2へ渡す。今回性能を測らない |

**行動への写像は作者規則である。** 検出出力を「聞こえやすさ」等の費用入力に使う場合も、方向、尺度、clamp、結合係数、候補間正規化、skipへの作用を一つずつ明示する。既存overlapの係数6や正負の選好を原典由来としない。検出確率、候補選択の因果差、作者の可聴性判断は別の成果にする。

## 4. 次の数値登録へ渡す条件候補

下記は条件選択案で、取得指示ではない。**原典の数値再現**、**同一入力での状態・接続介入**、**作者A1**を別の行列にする。原典の閾値を再現しても参加選択や可聴性の成立とはしない。

| 対象 | 原典条件と候補 | 登録前に埋める不足・比較の目的 |
|---|---|---|
| 同時マスキング | D96-II §II.A.1、p.3624、Fig.1：300 msの20–5000 Hz frozen noise、77 dB SPL、5 ms・1 kHz Hanning信号。信号位置を変える条件を候補とする | 原著の固定noise波形・位相列は今回未入手。別seedの雑音で図の各点を一致させる検査にはできない。原刺激と数値基準を得るか、独立参照との同一新刺激比較に目的を限定する。後者は原著図の再現ではない |
| 前方マスキング | D96-II §II.B、pp.3626–3627、Fig.7：200 msの同帯域背景、77 dB SPL、10 ms・1 kHz信号。横軸は**背景offsetから信号offsetまで**。正の横軸だけで非重複としない | 完全非重複の代表候補はoffset差40 ms。信号onsetは背景offsetの30 ms後となる。背景末尾付近の比較も別に保持する。波形、図の数値、許容、reset/初期化、観測窓は未確定 |
| 原典の制約確認 | D96-II §II.A.3、p.3625、Fig.3の信号長依存 | 原典が外す傾向を勝手な補助則で直して原典再現としない。拡張する場合は別モデル差分として判断する |
| T1生成接続 | [A1新版案 §0.1](a1-i11-audition.md)の旧body対新body、新版body対proxy、新版body対接続cut | 旧新版は変更記述、新版内は因果比較。適応状態だけを同じ入力から介入する検査を別に置く。共通record・背景・候補・時計を最初の分岐まで揃える。新T1 off/旧`None`、期限・欠測・容量の検査を登録してから取得する |

D96-II §I（p.3624）の刺激サンプルレートは30 kHz。現runtimeのrateへの変更は別の離散化照合になる。Fig.7の横軸やfrozen-noiseの位相を失ったenergy入力への縮約は、単位変換ではない。原典値・独立参照値・人の測定値を混ぜず、数値許容と使用する比較値は取得前に固定する。

## 5. T2 / Phase 4へ渡す最小の記録

| 受け手 | 記録・契約の候補 | 現単位との境界 |
|---|---|---|
| I11-4 / T2 | 音響bus、帯域中心・帯域幅・モデル版、適応前後の包絡の区別、振幅/energy/MUの単位、sample rate・更新周期、観測cut・利用可能時刻・欠測mask、背景と自声の出所 | T1後段の8 Hz low-pass出力だけをT2入力と決めない。どの段から分岐するか、共有状態か別状態かはT2モデル選択時に判断する。accent・周期・Hazard/CDFへの写像を原典から導いたと扱わない |
| 音色Phase 4 | 刺激版・身体/描画契約、attack・decay・再励起、平均スペクトルと残差、level、onset密度、変調、既存meter出力。共有partial/帯域、順序・priming等の統制条件 | [音色計画Phase 4](../../superpowers/plans/2026-09-29-timbre-synthesis.md#phase-4-temporal-contrasts-through-existing-paths)に従い、聴取差・解析差・行動差を別出力にする。平均スペクトル一致だけでまとまりを分離したとしない。新検出機構は別計画 |

T1 A1はT1地形後・T2地形前の版を固定する。T2後に継承できない条件は再検査し、I11-2/I12b比較を新版で再登録する。共有sourceの統合はI12b終結・基準版固定→I4配線→body-aware F3→音色Phase 3の順序を守り、Phase 4実施はその後に置く。文書読解をその待ち条件にはしない。旧60本は保管し、必要時の非受入手順確認に限る。

## 6. この草案を受けて必要な判断

| 次の判断 | 推奨する扱い | まだ選べないこと |
|---|---|---|
| Dau入力への適合 | [後続参照仕様](i11-t1-reference-spec-20260929.md)で確認できた原典核・後年実装・不足を分ける。現energy入力だけでの置換案は引き続き保留。次はAMT16の適応核だけの独立比較登録を具体化し、原Dau再現と別に扱う | S85の係数確認だけでDauの設定まで既知とすること、未来energyから任意の波形を作り原典入力と呼ぶこと |
| 適応状態と作者規則 | 自己除去の位置、候補branch、未来背景、音圧尺度を具体案として独立レビューへ渡す | モデル・独自写像・既定値・数値許容の凍結。今回の調査優先承認はそれらを含まない |
| GM05との比較 | Dauへの追加入力・状態が具体化した後、GM05全文を入手して同じ欄を埋める | 抄録だけで前方条件や軽量性を認定すること。不足を独自減衰モデルで穴埋めすること |

検証範囲：一次資料の本文・出典位置と現sourceの読取照合、文書の参照先確認のみ。Dauの数値再現、入力の同値性、現runtimeへの実装可能性は未検証である。本草案を独立レビューへ渡し、この単位を終了する。次の取得には進まない。

2026-09-29の独立読取レビュー（別担当 `/root/anchor_review`、Astra high）では、原典の時定数・刺激軸と現sourceを確認し、重大な誤記や無断gate変更は認めなかった。上記のproducer/consumerのレベル保持の違いと、次の原典補完単位の終了条件を明確化した。未確認部分を明示した判断資料として利用できるが、数値取得の入口ではない。
