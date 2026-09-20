# M0: 評定入力と順序回帰の数値準備

日付: 2026-09-12。状態: O03の評定入力を追加登録し、O09の数値検査を実施。
M0、M7、T3、適合・校正の完了ではない。

## 登録した入力

`feature-manifest.json`を`m0-registration-30`へ進めた。以前のlayoutと数式は変えず、
メモ§9.2と§10.2にあるgroove／参加欲求の入力を具体的な順序付きIDへ展開した。
二つの評定対象は入力を共有し、係数・cutpoint・校正を独立に持つ。

| layout | 入力数 | 内容 |
|---|---:|---|
| `groove_desire_109` | 109 | 定数、二つの群間history、群内periodic history、8窓のdensity、word entropy／二乗／surprise、grouping supportと各mask |
| `groove_desire_no_within_81` | 81 | 群内historyの28入力を除いた同じmap |
| `groove_desire_complexity_5` | 5 | 定数、entropyと二乗、それぞれのmask |
| `groove_desire_oscillator_beat_only_5` | 5 | issue-timeのbeat log-frequency／confidenceとmask、定数 |
| `groove_desire_meter_existing_234_13` | 13 | 上記にsubdivision／measureのconfidenceとratio/4、それぞれのmaskを追加 |

109入力は定数の後に、raw値数`[14,14,14,8,3,1]`の順で各blockの値、同じ順序のmaskを置く。
raw値は計54個。historyごとの14値は二mode×4値と、family・history weight・全modeに対する
residual dispersion・mode count・overflow・coverageの6値である。存在しないmodeや
支持不足はmissing、支持された非該当座標のゼロはknownとして区別する。
densityの窓順は0.125、0.25、0.5、1、2、4、8、16秒で固定する。

Rustの`groove_features`は、上流が支持を検査済みのraw値と凍結した54組の平均・標準偏差を受ける。
標準偏差のfloorは既定の1e-6、欠測値は標準化後の0、missing indicatorは1とする。
非有限値・負の標準偏差・表現範囲を超える算術を拒否する。オンラインの再標準化や学習は行わない。
これはfeatureの組立てであり、history・word・density・groupingの実音抽出器ではない。

## 五段階順序回帰

メモ§10.1のproportional-odds式を`ordinal_log_probabilities`へ実装した。
入力は線形予測子と厳密に昇順の4 cutpoint、出力は5カテゴリのlog probability。
定数slotの係数は0に固定し、自由cutpointに位置を持たせる適合契約も登録した。
この関数は適合・正則化選択・stage-4校正を行わない。

中間カテゴリを丸め済みCDFの減算で求めると、飽和したtailや狭いcutpoint間隔がゼロになる。
Rustではlog sigmoidと元のcutpoint幅の`expm1`を使う。共通予測子を引いた後のlogit差から
幅を復元しない。独立参照はPython Decimalの750桁でCDFを直接計算し、隣接差のlogを取る。
予測子±1000、1e-12のcutpoint幅、固定seedの48条件を含む全56条件を照合する。
これは工学的な入力域の数値検査であり、極端な人間の評定を観測した結果ではない。

入口は`src/temporal_cognition/ratings.rs`。現在の呼出し元はcrate testのみで、
実音のobserver・生成作用・聴取者評定へは未接続である。生成器は
`scripts/generate_temporal_rating_fixtures.py`、固定参照は
`tests/fixtures/temporal_cognition/ratings.json`。sourceと結果は
[m0-rating-preparation.json](m0-rating-preparation.json)に記録する。

検証結果はRust全体769件通過・13件ignore・失敗0、既存の一般数値参照58件とsection参照43件が通過。
通常Clippy、all-target check、fmt、Zola build、fixtureの決定的な再生成も通過した。
全target Clippyの既存二分類13件は残り、その二分類を許容した検査で他の警告がないことを確認した。
追加前のHEADにある全raw feature・layout・headはJSON値として同一である。

## 残る条件

onset-concentration-onlyの厳密な入力定義、history/mode数・bin・window・smoothing等の
感度比較の全layoutと採否規則、実音producerと欠落／capacityのfixture、有限fit予算・
lambda候補・solver・weight、O07のloss／calibration基準、O12の予測安定性は未完である。
平均・標準偏差、回帰係数、cutpoint、校正値は未適合のまま保持する。
旧`m0-registration-29`に固定した照合モデル・過去の数値実験を、新しい入力版へ自動昇格しない。
