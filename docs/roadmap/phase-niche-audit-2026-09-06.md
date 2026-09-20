# 拍内の位相ニッチと発音経路の監査

2026-09-06。対象は現行Sample 08のseed 21・42への作者回答と、現在のworktreeである。

## 作者の判断

作者は両方について「最初の拍に揃ってゆく」「最終的に一つの拍に縮退」と報告した。
その縮退は音楽的でないが、意図的なら問題ないという条件も示した。
現行2音源への音楽的な受入は未達として扱い、単に同期したことを成功とはしない。
変化時刻は未指定であり、内部指標から補わない。

回答の原文と2 WAVのSHA-256は
`target/stage2-evaluation/2026-09-05-current08-gate/listening-feedback-2026-09-06.json`に保存した。
標準比較の`listening.csv`と`author-listening-provenance.json`も更新した。
生成時の`audition-plan.json`と前日の完了監査は、その時点の記録として保持する。

## 残っている機構と切れている接続

| 部分 | 現在の実装 | 意味 |
|---|---|---|
| 個体固有の位相差 | `src/life/articulation_core.rs`の`KuramotoCore.phase_offset`。生成時に−π〜πから選び、目標を`theta.phase + phase_offset`とする | 初期位相とは別の、個体ごとに保持する位相差が残っている |
| 個体内周期からの発音 | `src/life/voice.rs`は`PhonationMode::Hold`の場合だけautonomous attackを有効にする | Sample 08のGated発音では、この位相差を持つ周期が直接onsetを発行しない |
| 実際の発音時計 | `.entrained()`→`CoupledTimingSpec`→`PhonationClockConfig::Coupling`→`CouplingClock` | この時計は`KuramotoCore.phase_offset`を受け取らない |
| 発音位相の引力先 | `CouplingClock`の`target_frac = delta.phase / TAU + microtiming` | Sample 08は両Populationとも`microtiming=0`。個体ごとの引力先の差がない |
| 初期のばらつき | `CouplingClock::new`で初期`phase`をランダム化 | 生成直後の同時発音を避けるが、保持する位相ニッチではない |
| 社会的なタイミング重み | Coupling経路は`TimingField::rebuild_flat`を使い、env/socialのゲートを迂回する | 社会的な密度重みを拍内の棲み分けに使う経路ではない |
| 位相と代謝の結合 | Gated onsetは`apply_phonation_onset`へ入り、recharge multiplierは1 | 個体の位相誤差によるrewardはこの発音経路では適用されない |
| `avoid_neighbors()` | `PopulationSpec::set_crowding`がpitch controlを変更 | 音高の混雑回避であり、拍内位相の混雑回避ではない |
| 同期表示 | `entrain_aligned_phases`は個体内位相から`phase_offset`を引いて集計 | offsetを尊重した内部同期の指標であり、実onsetの位相分布とは別 |

したがって、個体固有の位相差の骨格は残っているが、現在のSample 08の発音経路では位相ニッチとして働いていない。
同じ周期を共有することと、拍内の同じ位置で全個体が発音することが、この経路では分離されていない。
後者へ寄る引力はコードで確認でき、作者の聴取報告と整合する。聴取した縮退の全要因を定量的に分解した結果ではない。

## 履歴

- `451f3ab`（2025-12-31）で個体ごとの`phase_offset`と、その差を保つKuramotoの目標位相を追加している。
- `379dcbd`（2026-03-07）でGated時のautonomous attackを無効化した。二重発音と余分なエネルギー消費を避ける変更である。
- `a15100f`（2026-06-10）で現在の`CouplingClock`を導入した時点から、目標は共通beatと`microtiming`である。

今回の小節アクセント追加以前からある構造であり、追加アクセント0にしても解消しない。
この履歴から確認できるのは個体固有の位相差である。位相空間の混雑を知覚して、各個体が生息域を自律選択・学習する
完成した機構が過去に存在したかまでは、この監査では確認できていない。作者の設計の記憶と区別して保持する。

## 次の設計と検証

監査直後には、個体の`phase_offset`を実onsetへ再接続する修正を提案した。
その後、作者から拍と位相表現自体の再考を求められたため、この修正を確定した実装方針とはしない。
[拍・位相・時間的ニッチの再設計](../design-notes/rhythm-temporal-niche.md)に、
知覚の参照枠、個体の身体、実際の発音判断を分ける提案を記録した。

固定offsetの再接続は、共通周期と異なる発音位置の両立を調べる対照として残す。
一方、一次PLVが複数の規則的な発音位置を相殺する観測側の限界も切り分ける。
周期の予測、個体の相対関係、実onset分布、音楽的な受入を別々に検証する。
意図したユニゾンも成立し、初期のランダム化や一律な位相反発だけを自律ニッチと呼ばない。
具体的な実験順序と未確定の学習・報酬設計は再設計メモを参照する。

今回の作業は診断と試聴記録の反映であり、Rust実装は変更していない。
既存の`cargo test --lib coupling_clock -- --nocapture`は4件通過したが、内容は自由走行・rate上限・
beat周波数への結合・発音抑制中の時計継続であり、複数個体の位相ニッチを保持する検査ではない。
新しい原理上のギャップは日英設計台帳（旧技術ノート第9章）にも記録する。
