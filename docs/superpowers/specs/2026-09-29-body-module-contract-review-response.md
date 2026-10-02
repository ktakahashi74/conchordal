# Body module contract: response to the independent review

2026-09-29。対象は[身体モジュール契約の修正稿](2026-09-29-body-module-contract.md)。以下の所見処置は作者判断前の履歴である。後続の[作者判断](2026-09-29-body-policy-author-decisions.md)によりA1、A2のOff方針と試作T60、A3の費用帰属を採用した。A3容量、B5、A4承認、Phase 3開始は未完のままである。

## レビューの来歴と現在の状態

[Claude原文](2026-09-29-body-module-contract-claude-review.md)は変更していない。SHA-256は `7288f5374d888cb8d5e6d46b4bf47abe0db79a1339d5da8122c1bf2fbe8af805`。実モデルは `claude-opus-5-5[1m]`。作者が承認した固定6文書だけをtoolsなしで読み、リンク先・実装・hash・描画・試験・資源・試聴は検証していない。判定は「作者方針判断へ条件付き可、A4採用未準備」。本応答はAstraによる所見処置であり、修正稿の独立再reviewではない。

body-aware B1–B7の協議は記録済み。初期のOAuth失敗と自動承認reviewによる送信拒否は履歴であり、その後の固定文書送信の承認とreview受領を現在の状態とする。今回の送信承認、I11-3/4の番号、旧60本の用途に関する承認は、T60・容量・Off・pitch・契約本文の採用を意味しない。試聴も未実施。

## 15所見の処置

「修正済み」は草案の記述についての状態であり、推奨挙動の採用や実装を意味しない。節名は修正稿の見出しに対応する。

| 所見 | 処置 | 根拠・修正箇所・残る境界 |
|---|---|---|
| 1 pitch対象 | 修正済み／A1判断 | `Command table` にopen handleのみをUpdate対象とする推奨を記載。Off済みtailは除外。閉鎖後の新発声は新On、open handleへの再励振はre-kick。Updateによる状態数増加は防ぐが、発声間の旧音高tail重畳は残る。その帰結をA1に明示した。 |
| 2 Off係数 | 修正済み／A2判断 | `Randomness and fluctuation` に、実効Off直前のmotion offsetと励振依存spectral balanceを凍結する案を推奨として追加。即時復帰・別途登録するrampも選択肢として記載。位相・周波数・balance・自由減衰の境界検査を追加。元の「offset加算をやめる」だけの記述は置換した。 |
| 3 乱数・merge | 修正済み／A2内の提案 | source共通modulation、Tone初期化、Tone別独立noiseを別domainにした。mergeは同じ座標・位相基準、初期／現状態と入力の厳密な加算、独立control・乱数の保存が必要。位相が異なるだけで線形状態の厳密な加算を常に禁止する理由はないため、単なる「初期位相一致」より状態写像と検証条件を明確にした。merge実装は必須としない。 |
| 4 確率量の参照 | 修正済み／B5判断 | `Per-consumer reference, fallback and gate` の全行に実現値を明記。追加予測consumerも固定乱数実現値との比較を提案。期待値モデルを使う場合は別purpose・ensemble・推定誤差・行動上の意味の登録が必要で、現数値を流用しない。ownerと作者の合意は未取得。 |
| 5 標本化率 | 修正済み／module仕様未完 | 入力を「標本時刻で指定する無次元level」へ訂正。身体ごとにrate域、noise/drive/kickの正規化と不変量を宣言・検査し、域外はunsupportedとした。Sine/Harmonic式はModalの根拠にならない。Modalの具体式とrate集合は実装前に担当者が書く必要がある。 |
| 6 容量・lowering | 修正済み／A3判断 | `Core guarantees, dispatch and bounds` にopen数＋保持tail数≤16のenvelope登録を追加。最大open期間、burst、発声頻度、gainに依存するtail保持時間、死亡source分を含める。T60×頻度だけでは十分でない。例として単位振幅の1e-6到達は2 T60。容量適合は未証明で、制限域・容量・loweringの選択をA3へ残した。 |
| 7 F2開始条件 | 定義を修正／gate緩和案は不採用 | B7でidentity合意とproduction表現の決定を区別。表現・producer・全登録域・参照版・検証状態・費用経路を決定記録に要求。既存のF2依存とI12b固定→I4配線→body-aware F3→音色Phase 3の順序を保持。「高速F2前提を除く」は既採用gateの変更になり得るため本応答では採用しない。作者がrender-only移行を明示承認する選択肢だけ残す。 |
| 8 費用・死亡tail | 修正済み／A3判断 | habitat送出PCMのみを費用基準とする提案。生存中は所有Voiceへ課金し、死亡後は退役source世代へ記録するが親子へ転嫁しない。死亡tailは環境とrenderer資源に残る。presentation分を含める／死亡前予約・子への転嫁は別政策であり未採用。 |
| 9 Off時刻 | 修正済み | close→update→Onは実効action tickへ適用。要求tickはrelease ramp開始、実効Offはその終端。終端同tickのUpdateは閉鎖済みhandleへ作用しない。 |
| 10 unsupported pitch | 修正済み／A1内の提案 | source Updateは対象open handleを先に全検査し、unsupportedなら部分変更せず拒否・報告。既存targetを保持し、暗黙の保留・reset・crossfadeなし。次の新発声のpitch要求は別にfresh-On能力を検査する。身体種別ではなくcapabilityで判定する。 |
| 11 suite回数 | 修正済み | [anchor結果](../../design-notes/body-fitness-anchor-readout-results-20260929.md)と照合し、計装なし2回＋計装付き1回の計3失敗へ訂正。最終計装整理後のsourceでは全suite未反復、単独成功は全体成功ではない。 |
| 12 1 ms / 8 ms | 修正済み | [direct-model契約](../../design-notes/body-fitness-direct-model-contract-20260929.md)の登録上限と明記。実測最大値でも実測残余hop予算でもない。 |
| 13 Hz補間 | 修正済み | 共有表現のbin位置決めはLog2Spaceのlog2写像。Hz線形は、そのbin中心間の重みとして登録された場合のみ許可。凍結診断の歴史的意味は変更しない。 |
| 14 文書同期・review | 修正済み | [plan](../plans/2026-09-29-timbre-synthesis.md)と[timbre](../../design-notes/timbre.md)を協議済み・初回review受領・修正稿再review未実施へ更新。Task 2の別モデルreview特則と作者のClaude送信承認を適用し、一般Astra指定や過去のAstra plan reviewとは分離。AGENTS.mdは変更しない。 |
| 15 無音clock費用 | 修正済み／A2内の提案 | 無励振区間のlogical clock継続と実仕事量を分離。counter-addressable過程や厳密なbounded jump等、飛ばした標本数に依存しないadvanceを要求し、resume費用も計上。状態付きnoiseを単に停止・再開して同一と言わない。具体的生成器は未選定。 |

## 作者へ残す決定

以下は推奨付きの選択表であり、無回答を採用とは扱わない。個々の数値や付帯選択は分けて決定できる。所有者が詰める正規化式、work/memory見積り、数値取得登録の作成は、この表の作者方針を代行するものではない。

| 決定 | 修正稿の推奨 | 別案と帰結 |
|---|---|---|
| A1 pitchと発声の単位 | open handleだけ状態移転。閉鎖後は新On、open再励振はre-kick。unsupported Updateは原子的拒否。旧tailとの異音高重畳を許す。 | 全sourceのtailも再調律するとOff後の自由状態保持を変更する。crossfadeならtailの出力加工・費用・有界な同時状態数を別規定する。どちらも本文のA1変更が必要。 |
| A2 drive/free応答 | Sine/Harmonic自由T60 0.5 s。実効Off時のoffsetとspectral balanceを凍結。source共通modulationとTone別noise、無音時計は有界費用。rateごとのtransferは身体が宣言する。 | 0.5 s以外は値を指定する。係数の即時復帰は段差を許し、ramp復帰は長さとtail内決定論的変化を登録する。凍結案ではtailがdetuneを保持する。Task 1の揺らぎ水準の試聴判断とは独立。 |
| A3 容量・退役・費用 | 64 live Voiceを含む登録負荷と通常出生機会を維持。退役世代を含むsource総容量は必要枠算定後に決め、旧64総枠案は保留する。16 Tone/sourceもopen＋保持tailへの適合が未実証。576 lane、queue 16、1e-6、60 s、5 msは未採用案。habitat分を現ownerへ課金、死亡後は旧世代への記録のみ。 | 既存負荷・出生を縮めず、必要容量とloweringを具体化する。死亡後tail費用を予約／親子へ課すなら別の代謝政策となる。容量適合とdeadline通過を分け、数値承認だけで成功としない。 |
| B5 追加許容差と参照量 | 新consumerは固定実現値との照合を提案。energy `1e-10 + 0.001*abs(render)`、正規化temporal power 0.001、spectral L1・mass相対0.01は各ownerと作者が用途ごとに決める。 | 確率的期待値が必要なら別purposeとensemble/誤差定義を先に登録する。現F2閾値は不変。数値の根拠・用途が未合意のconsumerは未承認のまま残す。 |
| F2開始条件 | 現gate維持。identity合意だけで表現決定済みとしない。通常body-aware OFF維持、近道合格なしという結果を保持する。 | 作者がrender-only移行を選ぶ場合のみ、F2依存の変更、無効のruntime消費者、縮小したexitと後日の受入作業を明記する。I12b→I4→body F3→音色Phase 3の統合順序は維持する。高速モデルの全域実現可能性は未証明。 |

A4は、これらの記録、B1–B7とreview所見の解決を揃えた完成稿の採用である。A4未準備の根拠はA1–A3・B5・F2入口が未決であることであり、独立再review未実施だけではない。二度目のClaude reviewを自動的な新gateとはしない。追加reviewが必要な場合の範囲は、変更内容と未解決の懸念から判断する。今回の送信承認は固定した旧packetに限られ、修正稿の追加送信は含まない。既存の隔離prototype許可とmain統合許可、契約採用とruntime受入も分ける。

## A3の容量案に対する必要条件（統括による技術補足）

提案した64 source世代を、64 live Voiceに加えて死後tailも扱える容量と解釈してはならない。64世代がすべて発音中で、1 Voiceがまだ無視できないtailを残して死亡し、同じ時刻に子が発音を開始する場合、63生存世代＋1退役世代＋1新世代＝65枠が必要となる。旧世代の枠を子へ再利用するとtailの所有・自己除去・乱数identityが変わり、tailを切ると自由応答の契約が変わる。どちらも容量の実装詳細としては扱えない。

したがって64という数値は最終容量への推奨を保留し、**登録済みlive数と出生を維持したまま、退役世代を含む必要枠を算定する**。時刻tの必要枠は「実際に発音状態を持つlive世代数＋未廃棄の退役世代数」。退役tail保持の上限をT、任意のT長区間の退役件数の上限をR(T)と登録できるなら、`N_live,max + R(T)` が保守的な準備容量となる。境界時刻で廃棄と出生をどちらから処理するかも固定する。R(T)が未確定のまま64や128で足りるとしない。

同様に16 Tone/sourceは、全open handleと保持tailの合計に対する案であり、開いた発声数だけの上限ではない。上の世代数とTone数の二つを別々に数え、必要メモリ・block仕事量を計算する。出生の延期、通常Onの拒否、負荷域の縮小によって適合させる案は今回採らない。数値の承認だけで容量適合やdeadline通過を認定しない。これは数え上げによる反例であり、新しい負荷取得や方式採用ではない。

## B5の数値が保証する範囲（統括による技術補足）

追加許容差は未採用のまま、以下の包含関係だけを確認した。これは取得結果や数値の正当化ではない。既存Sine試験の許容差があることも、全身体・全consumerへ同じ値を採用する根拠にはならない。

- 時間footprintは各窓の平均二乗energyを求めてから最大値で正規化する。正本の最大値を `M>0`、全窓の絶対誤差上限を `δ<M` とすると、正規化後の最大絶対誤差は実数上で `2δ/(M−δ)` 以下となる。したがってenergyの相対0.001だけから、正規化powerの0.001を導くことはできない（絶対項なしでもこの十分上界は約0.002002）。純相対誤差だけでも、正本 `(1,1)` と予測 `(1.001,0.999)` は各相対誤差0.001だが、正規化後の第2窓の差は `0.002/1.001 > 0.001` となる。現在の草案が両方を別に検査する理由である。
- 絶対項 `1e-10` は小さい正のenergyに対する正規化誤差を制限しない。例として正本 `(1e-12, 1e-12)` と予測 `(1e-10, 1e-12)` は提案中の窓energy条件を満たすが、最大値正規化後は `(1,1)` と `(1,0.01)` で差0.99となる。無音と既知ゼロの完全一致、正規化後の直接比較、未支持の区別を残す。小振幅を後から除外して通過させない。
- 同じ固定地形 `C` に対する二つの正規化mass `p,q` なら、score差は `(max C−min C) × ||p−q||₁ / 2` 以下となる。L1≤0.01だけでは一般の地形でF2のscore≤0.025を保証しない。実数の例 `p=(0.5,0.5)`、`q=(0.505,0.495)`、`C=(-10,10)` ではL1=0.01、score差=0.1となる。これは数式の反例であり、新しい取得条件ではない。地形とbetaを固定していないconsumerへ、既定係数だけの上界を流用しない。
- sigmoidの実数上の傾きは最大 `abs(beta)/4`。beta=2ならscore差0.025からlevel差0.0125を導けるが、一般のbetaやf32丸め込みの同一保証ではない。登録済みF2の直接score・level・順位検査を残す。
- LOOでは自己除去によって地形自体を再計算する。片方が `C`、もう片方が `Ĉ` なら、上の固定地形上界に加えて `max|Ĉ−C|` の項が必要となる。身体massのL1だけでは、非線形な解析・peak選択・R/Hを経た地形差を囲めない。完全な自己除去PCMと解析結果との比較を、部分音表の検査に置き換えない。

以上は実数上の代数境界と小さな反例の検算で、実renderer、乱数実現値、丸め、資源、人の聞こえを検証していない。B5の次の判断では、各consumerの参照量・正規化・ゼロ・support・使用地形を先に固定し、数値をownerと作者が選ぶ。現契約の提案値を承認済みに変更しない。

[独立数量レビュー](../../../target/timbre-contract-numeric-review-20260929.md)で上記A3/B5の数え上げと代数境界を確認した。旧64総枠の推奨を契約本文・決定表でも保留へ揃え、16 Toneのrate条件を平均値ではなく任意区間の新状態生成arrival上界とした。数値採用や取得は行っていない。

## この単位の検証範囲

編集は契約、plan、timbre設計、本文の4文書のみ。原reviewのhash一致、相対リンクの対象の実在、変更差分に空白エラーがないことを確認した。Rust/source変更、cargo、render、性能取得、追加Claude送信、試聴、commitは行っていない。新しいgate数値・runtime証拠は取得していない。
