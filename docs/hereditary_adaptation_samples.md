# Hereditary Adaptation サンプル比較 (E6)

`samples/04_ecosystems/hereditary_adaptation_*.rhai` は、論文 (arXiv:2603.25637)
実験 **E6 — Hereditary Adaptation** を live assay に圧縮して再生するサンプル群。
6 変種で「遺伝の有無」「選択圧の有無」「選択の正確さ (Approx LOO)」の 3 軸を直交化している。

## 軸

| 軸 | 実装上の切替 |
| --- | --- |
| **Respawn 起源** | `respawn_peak_bias()` (heredity) / `respawn_random()` (random) |
| **選択圧** | `continuous_recharge_rate(0.10)` + `survival_signal(0.30, 0.80)` を入れる (selection) / 0.0 + 未指定 (only) |
| **選択の正確さ** | `selection_approx_loo(true)` を付加 (approx_loo) / 付加なし (baseline) |

## 6 変種

| # | ファイル | Respawn | 選択圧 | Approx LOO |
| --- | --- | --- | --- | --- |
| 1 | `hereditary_adaptation_heredity_only.rhai` | peak_bias | — | — |
| 2 | `hereditary_adaptation_heredity_selection.rhai` | peak_bias | ✓ | — |
| 3 | `hereditary_adaptation_heredity_selection_approx_loo.rhai` | peak_bias | ✓ | ✓ |
| 4 | `hereditary_adaptation_random_only.rhai` | random | — | — |
| 5 | `hereditary_adaptation_random_selection.rhai` | random | ✓ | — |
| 6 | `hereditary_adaptation_random_selection_approx_loo.rhai` | random | ✓ | ✓ |

## 対照関係

- **1 vs 4 / 2 vs 5 / 3 vs 6**: 遺伝 (peak_bias) が寄与しているかを比較する軸。
- **1 vs 2 / 4 vs 5**: 選択圧が寄与しているかを比較する軸。
- **2 vs 3 / 5 vs 6**: Approx LOO による選択の先鋭化が差を生むかを比較する軸。
- **1 (heredity only) と 4 (random only)**: 選択圧なしベースライン。
- **3 (heredity+selection+approx_loo)**: 完全な進化的収束設定。

## 共通条件

- `seed(20260330)`、初期個体数 16、`respawn_capacity(16)`、`random_log(55.0, 880.0)` に配置
- `anchor` は 220 Hz の harmonic drone で **`.mute()`** — 聴者バスには出さず、perceptual bus (他 voice の harmonicity 場) にのみ貢献する参照音源
- `assay_voice` は metabolism=0.12、action/recharge/dissonance_cost=0
- 時間スケールは E6b の step 比率 (`512/48000` s × 12000 ステップ) を live 倍速 1.0 で再生
- intro / outro 各 0.5 s

## 期待される可聴差 (論文再現)

- **1, 4** (選択圧なし): ランダム拡散。可聴な収束なし。
- **2, 5** (選択+遺伝/ランダム): 遺伝ありの方が協和帯域に収束しやすい。
- **3, 6** (+Approx LOO): 選択の近似改善により、peak_bias 側は収束がより鋭く、random 側はまだ揺らぐ。

## 参照

- 論文: arXiv:2603.25637 実験 E6 / E6b
- コード: `src/life/population.rs` (respawn / survival), `src/life/pitch_core.rs` (hill-climb)

## 関連サンプル群

- `temporal_scaffolding_{off,shared,scrambled}.rhai`: 論文 Temporal Scaffolding 軸の 3 変種
  (`set_scaffold_off / shared / scrambled` × `rhythm_coupling_vitality` + `rhythm_reward(attack_phase_match)`)
