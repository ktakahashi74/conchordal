# I13 作者規則層の係数と操作経路――実装前登録草案

日付: 2026-09-24。状態: 草案、未レビュー、未凍結。I12b候補の取得・終結前の文書作業である。ここでの「公開候補」はAPI決定でも既定値変更でもない。対象は[本体範囲](milestones.md#13-機構選択規則による本体範囲の縮小2026-09-20同日のレビューを反映)のT1、T2、T4短期反復期待、I4有界因果参照、I5、I10、I11、I12bの実消費者と、ALife側の参加判断に限る。研究拡張の検索・評定head・旧dispositionを戻さない。

## 判定の境界

[二層規則](../../design-notes/technote-ledger.ja.md#two-tier-rule)に従い、閾値・尺度を持つ工学則、独自の数値近似、学習則と学習率を作者規則層とする。学習で得た重み、記憶内容、観測したaccent・周期・身体記述値は状態であり、設定係数ではない。原理層と呼べるのは物理量の厳密計算、または原典との対応を確認した定量知覚モデルの部分に限る。現行T1の重なり比、T2のsalience・周期ヒストグラム・hazardは作者規則層である。T1マスキング、T2変調スペクトルへの置換後も、原典対応を確認するまでは新しい原理層パラメータと断定しない。計算モデルの根拠と、その機能を本体に置く根拠は別である。

以下の範囲・既定は現在のコードまたは登録入力を示す。「明示必須」は全体の既定が`None`であり、明示した場合だけ値を要する意味である。設定可能という事実は、日常の作者操作として推薦する判断とは別である。

| 群と所有者 | 係数・意味、単位／範囲／既定 | 現在の面と分類 | I13の扱い |
|---|---|---|---|
| T1／ALife参加判断 `TemporalParticipation` | `coupling` は参照追従と各費用項の強さ、無次元`0..1`、Populationの結合指定に由来する。`overlap_sensitivity` は重なり回避の符号と強さ、`-1..1`、現行Voiceで`0.8`。費用の`6`、自声／外部energyの比、予測窓と候補配置は内部則 | `coupling` は既存Rhaiのリズム指定から到達。`0.8`と`6`は内部固定。地形置換で重なり比の意味が消える。`src/life/phonation_engine.rs:674`、`src/life/temporal_participation.rs:448-452,701-749` | `coupling`を既存の作者操作として説明する候補。`overlap_sensitivity`と`6`を新規公開しない。マスキング版で保つ「回避／接近」という作者意図と、置換される現行計算を分けて再登録する |
| T1 footprint／供給側 | `footprint=body|proxy`、明示時既定`body`。span内3点Simpson等の近似・分割幅、最大16点、現行重なり量の正規化は実装／精度条件 | `footprint`は`[temporal_onset_comparison]`で公開済み。近似は内部固定。`src/config.rs:306-340`、台帳§9.3.55 | `footprint`は診断・比較用の既存設定として残す。合成器由来footprintをT1地形の入力とし、学習済みfootprintへの置換はA4後。数値分割を作者操作にしない |
| T2／観測accent・周期 | `accent_means/deviations` は二特徴の標準化尺度、明示必須、登録入力`[0,0]`／`[1,1]`。salience閾値`1.0`、3 hop局所最大、周期窓32秒／128件、核半幅1/24 octave、grouping許容`0.1`は現行工学則 | 尺度は`[temporal_acoustic]`等の明示TOML、残りは主に内部固定。`src/config.rs:351-365,405-412`、`src/temporal_cognition/accents/periods.rs:6,135-153`、台帳§9.3.55 | 地形置換対象。現行数値を恒久作者操作として公開しない。変調モデル選定・原典対応・新しい出力尺度を別登録で決める |
| T2／到来供給とALife消費 | `model`、18 hazard係数、8特徴のmeans/deviations、`horizon_sec`は`[temporal_period]`の明示必須値。主登録の`c0=-2,c10=4`は未適合診断値。`arrival_weight`は到来確率`P`から費用`coupling × weight × (1-P)`へ結ぶ無次元係数、明示比較時既定`1.0`、検証範囲`0..=4`。`arrival`既定false | hazard入力は明示TOML、`arrival`／`arrival_weight`は`[temporal_onset_comparison]`で公開済み。`src/config.rs:316-340,471-499,683-687`、`src/life/temporal_participation.rs:740-751` | `arrival_weight`は実消費者の作者規則候補。ただしT2変調による`P`の尺度・解釈を確認してから恒久操作を決める。hazard係数を通常UIへ展開しない |
| I5 gesture／診断 | 状態間遷移`softplus(C·x)`のCは`[temporal_gesture]`の`4×4×11`係数、登録入力は全0、率は約`ln 2/s`。経路質量減衰、handoff規則も内部則 | 明示TOMLは診断入力で、通常生成の消費者なし。`src/config.rs:458-467`、台帳§9.3.55 | 作者操作に昇格しない。状態や診断結果を、生成作用と混同しない |
| T4短期反復／外部予測 | 直近3周期の混合重み・忘却／減衰、履歴予測RLSのridge`0.01`・特徴数57。重みは観測から更新される状態 | 内部固定。`src/core/temporal_expectation.rs:8-11,21-80`、`src/core/history_prediction.rs:6-8,370-405` | 現時点で公開しない。学習重みを作者が設定する係数と呼ばない |
| I4有界参照／保持 | 保持の`tau_sec,kappa,strength_max,r_max,no_memory_bias`、episode容量・span等は`[temporal_memory]`で明示必須。私有参照の容量上限16、共有参照の同時数16、支持窓2秒は内部境界。検索パラメータは分離待ち | TOMLは現状`recall`全体の診断設定であり、研究検索との分離未了。`src/config.rs:423-453`、`src/temporal_cognition/reference_inventory.rs:9,118-135`、`milestones.md:95-97` | I4分離後、保持・減衰と有界参照だけ再棚卸し。検索係数を本体の作者操作へ戻さない |
| I10私有身体予測 | descriptor／energyのNLMS学習率`0.1`、重み範囲`-8..8`。`[temporal_private_trace]`の`tau_sec,kappa,strength_max`は明示必須の保持尺度。学習済み重み・traceは状態 | 学習則は内部固定、trace尺度はTOMLで診断用公開。`src/life/self_prediction/descriptor.rs:220`、`energy.rs:187`、`src/config.rs:343-349` | 学習率と重みを通常作者操作にしない。実音支持・epochと保持状態の表示契約を優先 |
| ALife参加の形・記憶 | 二乗Hellinger距離を`coupling`倍、文脈記憶更新率`0.1`、周期追従の緩和則`2×固有周期`、`flow_depth`は既存リズム形状`0..1` | `flow_depth`と結合はScenario側で公開済み。距離・更新・緩和則は内部固定。`src/scenario.rs:281-330`、`src/life/temporal_participation.rs:413-445,701-721,912-925` | 既存リズム操作を説明する。内部更新率やH距離の係数を増設しない。これはT3評定headの復活ではない |

この表は全DSP定数の一覧ではない。和声R／Hの内部核、`MeterNetwork`、研究検索はI13の公開選定から外す。内部値の感度はR3登録で扱い、公開と同義にしない。

## 最小操作案と解決契約

新規の作者操作を決める前に、既存の`temporal_mode("off"|"observe")`、Populationの`coupling/flow_depth`、明示TOMLの`temporal_onset_comparison`を一つの利用経路として整える。日常操作の第一候補は既存の参加傾向と、成立したI11-1／I11-2の消費者を有効化する最小の選択である。生のhazard係数・学習率・窓長を一括config化しない。T1/T2地形のモデルと尺度が未選定の間、係数の新名称、範囲、効用を固定しない。`None`／off既定は維持する。

現在のmode型は`Off|Observe`だけで、Rhaiの`participate`は起動前に拒否する（`src/scenario.rs:27-34`、`src/scripting/engine.rs:536-573`）。[初期操作契約](author-interface-contract.md#設定の所有者と解決)と[生成API参照](../../rhai_book/src/reference/api.md#temporal_mode)はobserveの新しい生成作用をゼロと説明する。一方、後日追加された明示`[temporal_onset_comparison]`はmodeとは別に各Voiceへ渡される（`src/runtime/mod.rs:2485-2494`）。現行コードの`Participation`は外部予測が適格で、onset候補が開き、係数が非ゼロなら費用を変え得る（`src/life/temporal_participation.rs:701-751`）。従って「observe単独」「observe＋明示TOML」「off＋明示TOML」を区別した説明が必要である。これは静的な経路確認であり、全組合せの実走結果やUI一致をまだ示さない。

実装前に固定する対応表は、各操作について次の順に一行で結ぶ。①RhaiとTOMLの入力・所有者・重複／優先規則、②解決済みmodeと設定、③無効理由（既定off、未実装、未設定、入力不適格、支持不足、期限切れ等）、④実際に消費した候補と作用の有無、⑤同じ要約からのreport／UI表示、⑥作者が編集して再実行できる記述。UIの更新間引きやreport接続が候補選択を変えない。`participate`を受理する前に、そのmodeが何を有効化し、明示TOMLとどう重なるかをこの表で確定する。旧dispositionの未校正gain群は対象にしない。

## 実装前の回帰・検査案

1. `None`／offと、Rhai `observe`だけの基準を固定し、同一seed・同一入力で生成音、決定record、学習recordを比較する。旧版から増えた診断欄は別集計にし、I11既存登録のbit一致条件を勝手に緩めない。
2. mode×明示TOML×Voice参加方策を小さな行列で検査する。設定解決、未知key／重複／`participate`拒否、`arrival=false`、有効な`arrival=true`、予測欠落・期限切れを含め、費用項の有無と理由を候補recordから確かめる。I12b封印版の取得条件は変更しない。
3. 同じ解決済み設定でreport有無・GUI有無を比較し、生成と学習の独立性、report／UIの同じ状態・無効理由・鮮度を確認する。実際の操作→編集→再実行を実行sampleと日英API参照へ反映し、生成物はregistryから更新する。
4. T1マスキング版とT2変調版が決まった後、その版の係数を再分類する。既存`coupling`等の意味と単位を確認し、公開候補の小範囲感度をR3の別登録へ渡す。A1／A2の作者採用、R2／A3、A4をI13の文書・契約検査で代用しない。

開始条件はI12b取得・終結と基準版固定、sourceを触る場合の別worktree・別`CARGO_TARGET_DIR`、独立Astraの実装前登録レビュー「可」である。本文はそのレビュー前の草案であり、実装・検査・I13完了を宣言しない。
