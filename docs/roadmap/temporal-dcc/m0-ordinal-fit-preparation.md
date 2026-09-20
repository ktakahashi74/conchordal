# M0: 評定の混合・有限fit準備

日付: 2026-09-12。状態: O03／O09／O12の評定用数値部品とfit手順を追加。
全stageのfit表、実データ適合、M0の完了ではない。
前段は[評定入力の準備](m0-rating-preparation.md)、手順は[ordinal-fit-plan.json](ordinal-fit-plan.json)、
今回のsourceと結果は[m0-ordinal-fit-preparation.json](m0-ordinal-fit-preparation.json)に保存する。

## 支持・混合・temperature

Rustの`project_rating`は、最大1024件の重み付きpath-localカテゴリ分布を受ける。
重みは上流で確定したshared context・音群の音響支持・conditional pathの積であり、
この関数は全音群のCartesian productを作らない。未列挙・未実装の重みはunknownに残す。
支持されたheadの質量を`s`、物理的な観測coverageを`e`とし、`r=e*s`を報告する。

出力は、支持された混合を正規化してtemperatureを適用した分布と、固定された訓練priorの
`r : 1-r`混合である。先に個々のpathをtemperature変換したり、priorを変換したりしない。
全unknownやcoverageゼロではpriorだけを返し、支持ゼロを保持する。
確率のlog空間で計算するため、full supportの小さいtail確率をCDFのゼロ差へ潰さない。
形状・有限性・非負重み・確率和・coverageを検査し、1e-12を超える質量不整合は拒否する。
この許容差は最大1024項のf64加算の保護であり、欠落質量を既知へ配り直す規則ではない。

独立参照はDecimalの確率空間で直接混合・べき変換し、40条件を照合した。
既存の750桁のordinal link参照56条件は維持し、混合参照は100桁で計算する。
入力fixtureの版をv2とし、旧版と旧sourceは前段のarchiveに残す。

## 評定fitの有限手順

`scripts/temporal_ordinal_fit.py`はPython／NumPy／SciPyによる統計fitであり、実音抽出器ではない。
凍結済みのpath-local特徴・重み・coverageと個々の五段階回答を受け、混合後の回答NLLの平均に
非定数係数のL2罰則を加える。支持不足の回答も同じ分母と訓練priorによるlossに残す。
解析勾配は各pathの回答への寄与率を使う。pathごとのloss平均を目的関数へ代入しない。
受け取った配列は独立した変更不可コピーにする。

正則化候補は`1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100`。一jobにつき開始点は一つ、
L-BFGS-Bの上限は400 iteration・1000評価・300秒。時間は完全な目的関数評価の前後で検査し、
一評価の超過も未完として記録する。追加restartや上限延長は含めない。
cutpointは基点と三つの正のsoftplus増分で表し、数値guardへの接触は未完とする。
solver成功、有限値、勾配基準、guard非接触を満たした場合だけ`fit_complete`を返す。
係数、実際のcutpoint、内部パラメータ、prior、loss、評価回数と停止理由を保存する。
これらの限界値は数値処理の登録であり、人の評定から得たパラメータではない。

訓練priorは回答数に各カテゴリ0.5を加えたもの。held-out評価には訓練側の固定priorを明示して渡す。
評価回答からpriorを作り直さず、評価用として準備した入力を訓練fitへ渡す操作も拒否する。
stage 2のtemperatureは1。stage 4の別分割による校正は未実装である。

## 分割と検査範囲

素材familyと参加者をそれぞれ五つに分け、25個の交差cellをscoring対象とするテンプレートを登録した。
各cellの訓練からは、評価する素材familyまたは参加者を共有する回答の和集合を除外する。
同じ番号の五cellだけを評価して残りを捨てない。検査では各回答を一度だけ評価し、両軸の
訓練・評価の重複をゼロにした。空の交差cellは黙って除かず、割当未成立として拒否する。
一headのテンプレートは7候補×25 foldの175 fitと、選択後のfull refit一つ。
stage 3の別の64-job上限を変更するものではない。

検査ではCDF直接計算によるloss、有限差分による解析勾配、既知カテゴリ比率の回収、
評価回数・時間超過、all-unknown、入力コピー、held-out priorを確認した。
別途、構成した109列・200回答・400 path行で一つのfitを実行し、28評価で数値収束した。
29回答はcoverageゼロのままlossへ残した。この行列は算術検査用の共変量で、音響抽出値でも
100組／条件のO12予測安定性検査でもない。回収・認知妥当性・作者の採用を示さない。

今回のRust全体検査は772件通過・13件ignore・失敗0。評定に関する6検査を含む。
Pythonの追加8検査も通過した。完全なテスト出力と終了値、構成入力・fit結果、
登録手順とsourceのハッシュは今回のarchiveとmanifestへ保存する。

実際の素材・参加者の割当、上流のfold別scalingとOOF特徴の由来、全head・対照・感度比較の
job ID一覧、実行時間とcore時間、校正、予測安定性は残る。fold番号だけでは上流の漏洩隔離を証明できない。
全データ準備・失敗job・数値guard超過を含む費用を測るまで、jobテンプレートを実行可能な全fit計画と扱わない。
