# F4f 初回 Field spawn の身体評価: 焦点試験結果（2026-09-27）

## 対象と取得条件

[固定登録](body-aware-fitness-f4f-initial-registration-20260927.md)の第十二版 ecology 隔離 worktree を対象に、`cfg(test)` のオフライン注入を初回 `Action::Spawn` へ接続した。通常 runtime への出生評価配線は含まない。fixture は Community seed 7、48 kHz、512 samples/hop、frame 64、population ID 7、子ID 2・3、380–520 Hz、80–4000 Hz・48 bins/octave の Log2Space、440 Hz Sine を72 hop解析した共有環境、72-hop代表身体、epoch 37。子template は440 HzのHarmonic、brightness 0.7、inharmonic 0、unison 1を使った。

## 取得できた結果

`CARGO_TARGET_DIR=/home/shafi/lwrk/conchordal/target/body-fitness-ecology-build-20260927 cargo test --lib life::community::actions::f4f_offline::tests:: -- --nocapture` は5件通過、失敗0件。破損入力を拒否する負例のpanicはテストが捕捉した。`cargo check --lib` と `cargo clippy --lib -- -D warnings` も通過した。

| 条件 | 実際に確認した事柄 |
| --- | --- |
| Consonance Peak、spacing 0、tension 0 | 範囲内の全候補bin slotを実子仕様の代表身体で準備し、2子を出生させた。少なくとも1子の最終周波数bitが選択bin中心と異なる。 |
| Consonance Density、spacing 0.25 ERB、tension 0.4 | 同じ全候補準備と2子の出生、連続jitter後の最終周波数を確認した。候補の `consonance_mass` と `fitness.level` が異なる例を確認した。 |
| Densityの重みと乱数 | spacing 0、tension 0の候補mass列を独立の `WeightedIndex` に渡し、最初の子の選択binが一致した。選択後のRNG probeは点地形の既存経路と一致した。 |
| Densityのfallback | 全候補massを試験境界で0にした条件と、spacing 100 ERBで後続子が全bin占有となる条件で、2子とも380–520 Hz内に出生した。 |
| 失効・破損 | epoch、有効field level、Log2Space中心、template body、子ID、候補bin周波数を個別に変えた6条件は出生前に拒否された。拒否後のVoice数は0、spawn counterは1。 |
| 最終子照合 | 子ID、世代0、body snapshot、周波数bit、代表Recipeのidentityを候補中心と別に再評価した。runtime eventは2件の発生数まで確認した。 |

合成2binの単体検査では、同じ身体重みと `H=0.8` に対し、`rho=0` 相当の平均massは0.8、`rho=1` 相当は0.5になった。この検査はmass算術の対照であり、実際の解析環境の `rho` 設定を切り替えて出生確率を測った結果ではない。

## 未取得と解釈限界

今回のF4f注入試験は Consonance のPeak/Densityだけを通した。Dissonance/Edgeの身体重み分岐は実装されているが、出生結果の試験は未取得。Gap/Uniformは身体注入の対象外で、`Context::prepare` が拒否する。`frequency.rs` の既存試験は狭いclamp範囲、解析域外、Uniform、占有・zero mass fallbackを点地形経路で調べるものであり、F4f注入経由の検証として数えない。

実bodyや実環境の `rho` を変えてmassと抽選確率がどう変わるか、同一fitness scoreでmassだけが異なる対照、spacing 0.25 ERBでの最終子間距離、全占有時の分布、通常分岐の同一seed既定OFF対照、runtime eventの各属性は未取得。候補bin中心で計算した重みを選択に使い、連続jitter後の実子fitnessは照合用に別計算した。最終周波数そのものを最適化した結果ではない。epoch拒否は注入context内の受理値を変えた試験であり、通常runtimeの解析更新との連動は検査していない。

`cargo clippy --tests -- -D warnings` はF4f以外の既存・並行編集ファイルの21件で失敗した。full suite、実時間費用、音響差はこの結果記録では未取得。
