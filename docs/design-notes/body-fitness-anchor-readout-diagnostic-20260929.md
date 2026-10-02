# 部分音群v2の採点位置：単一介入診断

2026-09-29。数値取得前の登録。[主計画](body-aware-fitness-plan.md)のF2内で、[採点位置の静的監査](../../target/body-fitness-support-20260929/reference-semantics-audit.md)が指摘した一つの差を調べる。身体footprintの正本は代表描画とする。v1/v2の失敗、対象負荷、音色計画Task 2・Phase 3と既存I系列の順序は変えない。

## 1. 問いと唯一の介入

旧描画参照は、peak質量を再配分しても主観強度を選択済み局所極大の `bin_idx` に置く。v2は再配分後のERB重心を採点位置に使う。この違いが保存された誤差へどう寄与するかを調べる。

v2の身体準備と群形成をそのまま実行し、保持群のpower、anchor、重心、motion標本を読み出す。次の二経路を72標本の直接列挙（motionなしは既存の1点）で計算する。

```text
control: f_g = erb_to_hz(erb_sum_g / P_g)
intervention: f_g = anchor_hz_g
f_gj = f_g + f0 * delta_j
M_g = (P_g / ref_power)^e
bin_mass_b = sum_g mean_j[M_g * I_inband(f_gj) * interpolation_b(f_gj) * A_b^e]
score = sum_b bin_mass_b * C_b / sum_b bin_mass_b
level = sigmoid(beta * (score - theta))
```

controlの重心には既存v2と同じ、丸め誤差だけを補正する分析帯域clampを適用する。補間はLog2Space中心間でHzに対して線形、端の短区間は端bin値、分析帯域外は寄与ゼロ。保持群、power、5%/−50 dB選別、最大32群、部分音・unison、減衰平均、motion profile、代表時間、A-weighting、聴感指数、環境C、候補集合は同じとする。名目帯域外の事前除外というv2の近似も本介入では変更しない。全partialの支持保存を新しい必須条件へ加えない。

## 2. 凍結入力と実装境界

- 旧描画参照と実Recipe：`target/body-fitness-direct-model-20260929/semantic-v1.jsonl`、SHA-256 `fb604ba3e83ad598a4ed9c4ed741cee44b6da33435cfdf86e6ee81dead3711e1`。
- v2出力：`target/body-fitness-partial-groups-20260929/semantic-v2.jsonl`、SHA-256 `f895342626cc19463e7722fdd41915e0f42344e9fba6b3b01b9d85b087607d09`。
- 13身体条件×2基音×7候補×4環境＝728候補。環境・case・基音ごとの104組すべて7候補を照合する。`landscape_peaks` は凍結入力の実Harmonicとして扱い、Modal入力へ差し替えない。
- 既存v2producerは変更しない。隔離worktreeの取得器に `cfg(test)` の診断子moduleだけを追加する。新しい `BodyKind` 分岐を作らず、`footprint.rs`・`energy.rs`・`temporal_cognition/body.rs`・`self_prediction` は変更しない。既存の身体準備を呼ぶだけで、新たな身体内部値からの生成式を生態系へ加えない。
- 波形・NSGTの再取得は行わない。比較入力は設計に使用済みであり、holdoutではない。

## 3. 先行照合と観測

各候補で既存v2producerを再生し、凍結v2のmass・score・level・正規化bin massとの一致を確認する。さらにcontrolの直接列挙を同じv2出力へ照合する。score/level絶対差は1e-6以下、mass相対差と正規化bin massのL1差は1e-5以下。先行照合が失敗した場合は、その候補の介入差を解釈しない。欠測・重複・unsupportedは成功扱いにせず、全行の状態を残す。

出力は入力hash、候補identity、環境C・格子、保持/除外群のanchor・重心・power、motion標本、control/interventionの非正規化bin mass・score・level・mass、旧描画と凍結v2の参照値を含む。群とmotionは両経路で共有した同一値とし、介入関数はそれらを変更しない。

独立checkerは群とmotionから両経路のbin massを再構成し、Cとの内積とsigmoidを検算する。これは描画自体や群形成を独立再現する検査ではない。既存のv2群形成unitと保存入力の照合を併用する。最小unitは、anchor=重心の一致、非平坦Cで採点位置だけを変えた差、一定C、motionの分析帯域境界と有効massゼロを扱う。

## 4. 判定と終了条件

旧描画への比較値は既存登録のまま、score最大絶対差0.025以下、level最大絶対差0.0125以下、旧score差0.1以上の候補対の厳密な順位逆転0とする。同点を逆転へ混ぜず、同点化も別記する。728候補の未達数、最大差、case別差、v2からの回復/新失敗、最大誤差例の変化をすべて記録する。新しい値を見て閾値・群数・係数・候補・負荷を変えない。

本単位の終了は、実装の内部検査、全入力の先行照合、単一介入の全728件の記録と独立検算、必要なRust回帰の結果保存まで。旧描画の比較が未達なら性能取得・本番接続へ進めない。比較を通っても、v2の大きい身体・候補依存生成の資源未達は解消していないので、この介入だけを本番へ接続しない。資源の再測定は本単位に含めない。

代表描画の基準を替える判断、音色Phase 3の移行、未統合のmeter/valuation変更、I11の再取得、T1/T2刺激作成、音色遺伝は本単位に含めない。
