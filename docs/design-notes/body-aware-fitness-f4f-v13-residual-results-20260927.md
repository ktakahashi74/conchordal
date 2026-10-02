# F4f 第十三版: 初回 Field spawn 残余試験結果（2026-09-27）

## 対象と取得

[固定登録](body-aware-fitness-f4f-v13-residual-registration-20260927.md)に従い、`.worktrees/body-fitness-metabolism` の `cfg(test)` オフライン注入を検査した。変更したsourceは `src/life/community/actions/f4f_offline.rs` の試験helperと試験だけである。通常runtimeの初回出生へ身体評価を配線した結果ではない。

`CARGO_TARGET_DIR=/home/shafi/lwrk/conchordal/target/body-fitness-metabolism-build-20260927 cargo test --lib life::community::actions::f4f_offline::tests:: -- --nocapture` は11件成功、0件失敗、1226件filter。破損入力の負例で表示されるpanicは各試験が捕捉した。最初の取得試行は並行編集のruntime live試験の引数不足、次の試行はrespawn試験のprivate importでコンパイル前に止まった。両者の修正後、登録したfixtureの数値や合否閾値を変えずに取得した。

## 取得できた境界

| 対象 | 結果 |
| --- | --- |
| Dissonance/Edge | 各targetのPeakとDensityで、全候補身体の準備から2子の出生まで通過。Peakの最初の選択binはfitness levelの最小値または0.5への最短距離と一致。Densityの最初の選択binは、それぞれの身体平均massを用いた独立 `WeightedIndex` と一致。 |
| 実身体とrho | 440 Hz＋466.16 Hzの二音環境でSine対Harmonic（rho 0）の候補massが変わり、正規化抽選確率の全変動距離は **0.056282911**。Harmonic固定のrho 0対3はfield scoreと候補fitness scoreを保ったまま候補massが変わり、全変動距離は **0.101807393**。各条件の実出生binは、その条件のmass列を使う独立 `WeightedIndex` と一致。 |
| 狭いclamp範囲 | 220.0–220.1 Hzと408.0–420.0 HzのPeak/Densityで、全候補slotのbin番号とclamp周波数を照合。各条件の2子は指定range内に出生。 |
| 最終spacing | 380–520 Hz、Density、指定0.25 ERB、tension 0.4の連続jitter後、2子の実周波数間距離は **0.852322578 ERB**。 |
| runtime event | 注入経由の2子について、時刻、population ID、voice ID、member index、最終周波数bit、parentなし、世代0、Initial理由を実Voice・receiptと照合。 |
| 全占有fallback | spacing 100 ERB、seed 7–14の後続子は全bin占有時も準備済みrange内に出生。選択binは8例で `[108, 109, 113, 120, 123, 126, 127, 128]` の8種類。 |
| 既定OFFとGap/Uniform | 注入contextなしのConsonanceは同じseedで周波数bitを再現。Gap/Uniformの注入準備は明示拒否し、注入なしの旧経路では同じseedの結果を再現して指定range内に出生。 |

body/rhoの全変動距離は候補mass列から求めた抽選確率ベクトルの距離であり、反復出生による経験的頻度ではない。固定seedでの実出生binと独立抽選を照合した。全占有の8例も分布推定ではなく、単一点へ退化しない境界確認である。

## 残る範囲

今回もbin中心の身体評価でbinを選び、連続jitter後の確定子を別評価・照合した。確定周波数を直接最適化した結果ではない。Gap/Uniformは身体注入対象外。通常runtimeの初回出生配線、非同期費用、長期の選択効果、任意の身体・環境・rangeに対する保証は未取得。full suiteとtimingはこの作業では実行していない。
