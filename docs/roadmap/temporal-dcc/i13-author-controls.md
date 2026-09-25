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

新規の作者操作を決める前に、既存の`temporal_mode("off"|"observe")`、Populationの`coupling/flow_depth`、明示TOMLの`temporal_onset_comparison`を一つの利用経路として整える。日常操作の第一候補は既存の参加傾向と、成立したI11-1／I11-2の消費者を有効化する最小の選択である。生のhazard係数・学習率・窓長を一括config化しない。T1のモデルはCASP 2008を選択済みだが、実装版の尺度は未確定である。T2のモデルと尺度も未確定であり、両地形の係数の新名称、範囲、効用をここで固定しない。`None`／off既定は維持する。

### 終了点と実行可能な一往復

[I13の終了条件](milestones.md#2-技術実装の完了点)は、成立した機能で「Scenario指定→演奏→状態・無効理由の確認→編集・再実行」を通し、Rhai・UI・文書・実行sampleを同期することである（`milestones.md:129-133,157`）。[W0の初期契約](author-interface-contract.md#試聴と受入)も、M8–M9で実行sample、生成API参照、既定off、資源と音の回帰を要求する。説明文の同期や`participate`未実装の正確な記述だけではI13を完了しない。

最小の一往復は、既存の実行可能なRhai操作（例：Populationの`entrained()`／`entrainment(value)`）と、明示`[temporal_onset_comparison]`を持つ検査Scenarioで行う。前者は参加傾向の作者指定、後者はI11比較の診断入力として所有者を分ける。支持された同一入力で、指定→解決設定→適格な候補費用→選択→実際の発音を識別し、UI／reportで理由と時刻を確認し、指定を編集して再実行する。設定差だけ、または候補費用差だけを聴こえる効果の成立と呼ばない。`entrained()`／`entrainment(value)`は既存API（`src/scripting/engine.rs:257-259`、`src/scripting/docs.rs:716-752`）である。新しい`participate` modeやdisposition gainをこの一往復のために実装しない。

### modeとTOMLの解決表（現行動作を保持する案）

現在のmode型は`Off|Observe`だけで、`participate`はRhaiの起動前に拒否される（`src/scenario.rs:27-34`、`src/scripting/engine.rs:536-573`）。明示`[temporal_onset_comparison]`はmodeで遮断せず各Voiceへ渡る（`src/runtime/mod.rs:1120-1130,2485-2494`）。したがって両者は現行実装では独立した指定であり、TOMLがmodeを上書きする、またはoffがTOMLを無効化するという優先規則を作らない。`temporal_mode`の重複・時刻付き指定、未知mode、未知TOML keyは既存の起動前拒否を維持する。

| Rhai mode | onset比較TOML | 起動時の解決設定 | 候補判断時の実作用・無効理由 |
|---|---|---|---|
| 省略／`off` | `None` | 共有temporal observerなし。I11比較は従来の64点代理経路 | 新しい比較設定なし。既定生成と記録のbit一致を維持する |
| `observe` | `None` | 両busの共有observerを起動。I11比較は従来経路 | mode単独の新しいI11費用なし。観測の受信／欠測は別の状態である |
| 省略／`off` | 明示`Some` | 比較設定はVoiceへ渡る。mode offでは共有body worker／到来CDFを起動しない | `Some`の16点代理経路が働き得る。body指定は`proxy(absent)`等へ退避し、到来は入力不在なら未知。候補・音声の不変を一律には主張しない |
| `observe` | 明示`Some` | Voiceへ比較設定を渡し、条件が満たされれば共有body／到来を配線 | bodyの有効支持、proxy理由、到来の既知／未知、`arrival=false`／weight 0、coupling 0、候補期限を判断ごとに区別する |
| `participate` | 任意 | 起動前エラー。解決済みmodeは存在しない | 未実装・未校正。実行可能sampleに載せない |

`Some`の16点代理・body退避は`src/life/temporal_participation.rs:529-594`、費用適用の条件は同`:701-751`にある。[W0](author-interface-contract.md#設定の所有者と解決)の「observeで新しい生成作用ゼロ」は、後日追加された明示比較TOMLがない**mode単独**の場合へ限定して同期する。現在のmode/TOML優先関係を将来変更するなら、別の作者判断とoff／None回帰を要する。

### 設定、作用、表示の最小契約

起動時にはRhai mode、TOMLの有無と明示値、body capture／到来CDFの配線可否を一度解決する。設定済み・worker起動・有効な入力受領・候補費用へ適用・選択変更・発音は異なる状態である。決定時の理由は実消費者の判断点で確定し、`body`／`proxy(setting|absent|stale|unsupported)`、到来の既知／未知、外部予測・候補窓・期限、coupling 0などを事実どおり示す。bodyの`stale`はrecipe／世代のidentity不一致であり、到来の期限切れと同じ時計判定にしない。未知を既知0へ変えない。決定機会がなければ作用は「未観測」であり「無効」と断定しない。

現在の詳細`ParticipationDecision`はreport接続時だけ収集され、23候補を含む（`src/runtime/mod.rs:2485-2494,2609-2625`、`src/life/temporal_participation.rs:65-104`）。これをGUI用に常時生成・複製しない。実消費者が決定時に、Voice ID／generation、epoch、decision sample tick、選択onset tick、設定・入力の適格性、適用した費用項と理由を持つ**固定サイズの最新判断要約**を、report有無と独立に更新する案とする。実際の発音は選択と同義にせず、同じVoice世代・選択識別子に対応する発音tickが確認されたときだけ後から記す。未発音・退役・epoch更新ではその状態を保持または失効させる。各Voiceの有界な最新要約をworkerがUI更新時に小さいsnapshotへ集め、間引き後のsnapshotには取得tick／epochを付ける。reportは同じ要約を別の軽量recordとして出せるが、詳細traceの有無は従来どおりreport専用である。

小さい解決済み設定要約と判断要約をreportとUIの共通入力にする。UIは更新間引き後の古いsnapshotを現在の支持と表示しない。reportの既存record型・既存fieldの意味と順序は維持する。追加の設定・判断recordが必要なら既存recordへfieldを差し込まず、明示`Some`または`observe`時だけ別型・schema版付きで出す案とし、`off+None`には出さない。既存`None` runの音声、決定・学習record、report schemaの固定比較を保つ。GUIなしでもreportから同じ設定・作用理由を得る。診断の有無で生成・学習を変えず、hopへの全候補frameコピーや表示専用の閾値を加えない。

## 実装前の回帰・検査案

1. `off+None`と`observe+None`の同一seed・同一入力で、音声、決定・学習record、既存report schemaを固定基準と比較する。新しい設定recordは`off+None`へ出さず、I11のbit一致条件を緩めない。
2. mode二値×TOML `None/body/proxy`×到来指定の最小行列を検査する。重複・未知key・`participate`拒否、`arrival=false`／weight 0、coupling 0、body支持欠落・世代違い・期限切れを含める。設定と判断の適格性、費用の非ゼロ、選択、発音を別欄で照合する。
3. 支持された検査ScenarioでRhaiの参加指定から実作用までを一往復させ、Rhai指定または明示TOMLを編集して再実行する。同一状態介入の入力差分、決定record、音声を保存する。同一入力・同一配送列でreport有無×GUI有無の四条件を比較し、生成と学習の一致、GUI単独での理由表示、判断要約と詳細reportの対応を検査する。間引き、旧epoch、Voice退役、選択後未発音、report schema／順序の不変も検査する。実行sampleと日英API参照へ反映し、生成APIはregistryから更新する。
4. 選択済みCASP 2008を使うT1マスキング実装版の尺度と、T2変調地形のモデル・尺度が定まった後、その版の係数を再分類し、既存`coupling`の意味と単位を確認する。公開候補の小範囲感度はR3の別登録へ渡す。A1／A2の作者採用、R2／A3、A4をI13の文書・契約検査で代用しない。

開始条件はI12b取得・終結と基準版固定、sourceを触る場合の別worktree・別`CARGO_TARGET_DIR`、独立Astraの実装前登録レビュー「可」である。本文はそのレビュー前の草案であり、実装・検査・I13完了を宣言しない。
