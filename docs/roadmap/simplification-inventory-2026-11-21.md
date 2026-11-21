# Conchordal 引き算・構造整理 棚卸し台帳（2026-11-21）

対象基準：`origin/main` `dee5a91d` およびローカル checkout（2026-11-21 11:30 JST 実測）。  
本台帳は [実行指示書](simplification-directive-2026-11-21.md) の Stage 0〜Stage 4 の入力正本である。破壊的操作はまだ行っていない。

---

## 1. ローカル `main` vs `origin/main` 差分監査（Stage 0）

ローカル `main` は `04c6ba7e` に留まっており、`origin/main`（`dee5a91d`、5 commit 先行：部屋・距離・英語デモ統合済み）から遅れている。ローカルの未 commit 4ファイルの正体は次の通り：

| ファイル | `origin/main`（`dee5a91d`）との比較結果 | 推奨処置 |
|---|---|---|
| `src/ui/plots.rs` | **`origin/main` と完全一致（差分 0）** | そのまま `origin/main` 同期で解消 |
| `src/viewdata_web.rs` | **`origin/main` と完全一致（差分 0）** | そのまま `origin/main` 同期で解消 |
| `src/ui/windows.rs` | **`origin/main` が上位互換**（`end_request: Option<&Arc<AtomicBool>>` がブラウザデモ用に導入済み。ローカル側は旧シグネチャ） | `origin/main` を採用 |
| `scripts/worker-shell.el` | **唯一の純ローカル差分**（`conchordal-worker-model` を `"gpt-6.2-sol"` に固定する 2026-11-18 変更） | patch 退避または `origin/main` 上へ再適用・commit |

### 稼働中 worktree（未 commit・触る前に作者／統括で決着）
- `.worktrees/body-phase2-20261121`（`work/body-phase2-20261121`、tracked 44 / untracked 2）：DR-39（`motion` 修正・連続化）＋ DR-41（`resonance`・`attack_noise`）。全試験 1,927 通過済み、**作者試聴待ち**。
- `.worktrees/room-stage1-20261119`（`work/room-stage1-20261119`、tracked 6 / untracked 3）：DR-40（単位 E：研究版 Line 距離対照）。
- `.worktrees/dr1-consolidate-20261119`（`work/dr1-consolidate-20261119`、tracked 30 / untracked 1）：DR-1〜23 の変更はすでに `3180847b` で `origin/main` に統合済み。残差分の有無だけ照合して退避・閉鎖可。

---

## 2. 全 206 worktree の実測分類（総容量 143 GB）

| 分類 | 件数 | 内容 | 推奨処置 |
|---|---:|---|---|
| **clean & merged** | **24** | 変更ゼロ、HEAD は `origin/main` の祖先 | 参照チェック後、worktree 即削除可（`hr2-range-20261107` 単独で 26 GB） |
| **clean & unmerged** | **44** | 変更ゼロ、研究ブランチ上に commit 済み | ブランチ名を残したまま worktree だけ削除可（git 履歴は保全） |
| **dirty (untracked のみ) & unmerged** | **3** | ログ・WAV 等のみ | 証跡の要否を確認し、不要なら worktree 削除 |
| **dirty (tracked 変更あり) & merged** | **122** | HEAD はマージ済みだが未 commit 変更あり | 上記稼働中 3 本を除き、差分が `origin/main` に吸収済みか確認。未吸収コードのみ `archive/<name>` ブランチへ退避 commit 後に worktree 削除 |
| **dirty (tracked 変更あり) & unmerged** | **13** | 未マージ研究ブランチ＋未 commit 変更 | 研究ブランチへ WIP 退避 commit 後に worktree 削除 |

- **ディスク内訳**：`.worktrees/*/target` だけで **90 GB**（全体 143 GB の 63%）。
- **容量上位 worktree**：`hr2-range-20261107`（26G）、`i4-reference`（7.3G）、`ph1-span-gap-20261024`（6.9G）、`flora-self-exclusion-20261024`（6.9G）、`body-fitness-dynamic-prepeak-prior-actual`（4.6G）。
- **clean & merged 24本一覧**（最初に安全に削除候補となる組）：
  `conchordal-baseline`, `main-source`, `source`, `conchordal`, `canon-20261108`, `ci-fix-20261108`, `doc1-records-20261102`, `doc5-v7-range-20261105`, `docs-artpath-20261106`, `fa1-flora-amplitude-20261102`, `felt-beat-20261117`, `fi1-read-20261107`, `hr2-range-20261107`, `hv1-harmonicity-preference-20261101`, `hv11-read-20261102`, `hv13-arrival-lanes-20261102`, `hvint-20261102`, `ipacq-20261102`, `ipacq-ref-20261102`, `ipacq2-new-20261102`, `ipint-rebase-20261101`, `ug3-read-20261102`, `vr1-rearticulate-roughness-20261102`, `wp4-numerics-20261106`。
- **注意（`AGENTS.md` 契約）**：削除前に `.orchestration/` と `docs/` から参照されているパス、および取得済み証跡（WAV/TSV/JSON）の有無を確認し、証跡は残す。

---

## 3. Rhai API（186 エントリ）と `config.rs` opt-in 機能の 4分類原案（Stage 1）

分類基準：
- **A（コア維持）**：楽器の中核・デモ・主要作曲に必要。
- **B（研究隔離）**：学術検証・アッセイ用に残すが、Cargo `research` feature または明確な隔離層へ押し出す。
- **C（廃止候補）**：役目を終えた旧互換・未接続・過剰な微調整ノブ（git 履歴に保全）。
- **D（12/25 後再判定）**：マニフェスト1周年改訂（動き・文化/経験・時間構造の位置づけ）と直結するため保留。

### 3.1 Rhai API 概要（`origin/main` `src/scripting/docs.rs`、計 186 エントリ）
トップレベルサンプル 18 曲（`samples/*.rhai`）およびデモ 2 曲（`samples/demo/*.rhai`）での使用状況で実測：
- `Core`: 128 エントリ（うちトップ・デモ未使用 **42**）
- `Experimental`: 23 エントリ（Flora/Rain 系を中心に活発に使用）
- `Tuning`: 26 エントリ（うちトップ・デモ未使用 **25**。使用中は `move_cost` のみ）
- `Research`: 9 エントリ（うちトップ・デモ未使用 **9 / 9**）

| 対象群 | 具体的な語・機能 | サンプル/デモ使用 | 原案分類 | マニフェスト依存 | 理由 |
|---|---|---:|---|---|---|
| **Research 語（9語）** | `scaffold_off`, `scaffold_shared`, `scaffold_scrambled`, `temporal_mode`, `pitch_core`, `move_cost_time_scale`, `control_update_mode`, `observe_felt_beat`, `syncopate` | 0 / 0 | **B（隔離）** / 一部 **C** | なし | すべて研究・アッセイ専用。通常作曲面に置く理由がない。`scaffold_*` / `control_update_mode` / `move_cost_time_scale` は不要なら C |
| **Tuning：発音微調整（12語）** | `once`, `pulse`, `while_alive`, `adaptive_duration`, `pulse_lock`, `social`, `duration_range`, `duration_curve`, `shorten_on_drop`, `rhythm_freq`, `rhythm_coupling_vitality`, `rhythm_reward` | 0 / 0 | **C（整理・削減候補）** / 一部 **B** | なし | 現行デモ・18サンプルのどれにも使われない旧リズム/発音微調整群。テスト専用でないか監査して削減・統合 |
| **Tuning：音高探索微調整（11語）** | `landscape_weight`, `neighbor_step_cents`, `move_cost_exp`, `tessitura_gravity`, `window_cents`, `top_k`, `sigma_cents`, `random_candidates`, `global_peaks`, `ratio_candidates`（※`move_cost` はデモ使用のため **A**） | 0 / 0（`move_cost` のみデモ 1） | **B（隔離）** / **C** | なし | 内部アルゴリズムのチューニング定数が直接 Rhai に露出しており、`pitch_core.rs` の肥大要因。`move_cost` だけ Core 昇格を検討 |
| **Tuning：その他（3語）** | `crowding_target`, `viability_scope`, `set_global_coupling` | 0 / 0 | **B** / **C** | なし | 実行サンプル未使用。必要性を確認して整理 |
| **Experimental：Flora / Rain 主力** | `flora`, `fade`, `tone`, `residual`, `respond_to_onsets`, `rain`, `rate`, `sway`, `hit`, `width`, `ring`, `follow_meter`, `leave_drops_out`, `objects_follow_drops`, `objects`, `object_modes`, `impact_modes`, `object_decay` | デモ主力 | **A（コア維持・昇格候補）** | なし | 現在のデモを支える主要表現。安定語は Core へ昇格検討 |
| **Experimental：その他（2語）** | `remember_pitch_set` (PS-1), `phonate_when_viable` | 0 / 0 | **D（12/25後）** / **B** | あり（記憶） / なし | PS-1 は記憶・文脈の整理待ち。`phonate_when_viable` は使用有無を確認 |
| **Core 未使用：基本・構成語** | `saw`, `square`, `noise`, `site`, `edge`, `density`, `tension`, `room`, `seed`, `parallel`, `endless`, `steady`, `repeat`, `intrinsic_rate`, `tie`, `respawn_random`, `respawn_hereditary`, `power_modes`, `custom_modes`, `landscape_peaks_modes`, `jitter`, `set_roughness_aversion`, `set_harmonicity_preference`, `set_pitch_objective` | 0（一部デモ/他フォルダ使用） | **A（コア維持）** | なし | 作曲語彙の直交セット・エンドレス・部屋・モード基本語。整理対象外 |
| **Core 未使用：候補層・和声拡張（14語）** | `listener_profile`, `expressive_section`, `dwell`, `phrase_boundary`, `follow_tonal_context` (TC), `melodic_continuity` (ME), `follow_own_sequence` (SQ), `tail_roughness_aversion`, `harmonic_rhythm`, `observe_harmonic_relations`, `observe_tonal_context`, `set_support_focus_preference`, `set_support_conflict_preference`, `set_related_novelty_preference`, `follow_harmonic_relations`, `follow_harmonic_continuation` | 0 / 0 | **D（12/25後再判定）** | **あり** | 既定 0 で本体に堆積している最大領域。旋律・時間構造・Layer 3 の位置づけをマニフェスト改訂で整理してから、残す軸を絞る |

### 3.2 `src/config.rs` opt-in / 研究フラグ棚卸し

| 設定項目（`AppConfig` 他） | 現行既定 | 原案分類 | マニフェスト依存 | 理由 |
|---|---|---|---|---|
| `birth_surrogate` | `false` | **C（廃止候補）** | なし | 旧 body モデルの比較用フラグ。役目を終えていれば削除可 |
| `render_prototype` | `false` | **C（廃止候補）** | なし | body-fitness 試作の残留フラグ（`modal_engine.rs` の clamp 超過 panic 誘発歴あり） |
| `body_metabolism` | `None` | **B（隔離）** / **C** | なし | 点経路と並存する opt-in。採否を確定して単一化 |
| `temporal_meter_t2` + `temporal_t2` | `false` / `None` | **B（研究隔離）** | なし（2026-10-14 終了判断済み） | `src/temporal_cognition/`（23,165 行）の大きな割合を占める opt-in 経路 |
| `temporal_ridge` / `temporal_acoustic` / `temporal_period` / `temporal_body` / `temporal_body_prototypes` / `temporal_onset_comparison` | すべて `None` | **B（研究隔離）** / **C** | なし | 6 種の診断・プロトタイプ設定。通常楽器の `AppConfig` を肥大させている |
| `analysis.felt_beat` | `None` | **B（研究隔離）** | なし | FB-1〜15 の研究観測 |
| `room`（`off` / `presentation` / `both`） | `None`（`off`） | `off`/`presentation` = **A**、`both` = **B** | なし（効果2の採否は別） | 出力ステレオ部屋は本採用済み、`both` は研究用 |
| `dcc.coupling_strength` | `0.0` | **D（12/25後）** | **あり** | マニフェストの DCC 第2段階に直結 |
| `psychoacoustics.use_incoherent_power` | `false` | **C（廃止候補）** | なし | 現行前段への移行後も必要か確認 |

---

## 4. コード構造負債マップ（Stage 2〜Stage 4）

### 4.1 God Struct（Stage 3 対象）
- **`Landscape`**（`src/core/landscape.rs`、約 45 フィールド）：純粋地形（R/H/C/慣れ/S6）＋住人別観測（`loo`, `flora_observations`, `tonal_observations`, `tonal_context`, `pitch_set`, `spectral_history`）＋任意和声拡張（`support*`, `harmonic_relations*`, `harmonic_recorder`）が1つの `Clone` 構造体に同居。
- **`Voice`**（`src/life/voice.rs`、40 フィールド）：`support_*`（6個）・`harmonic_*`（5個）・`sequence_control`・`participation_pitch`・`voice_adsr` 等がフラットに並ぶ。
- **`Flora`**（`src/life/flora.rs`、40 超フィールド）：バイオマス本体＋`harmonic_*`（7個）＋`support_*`（8個）＋`RefCell`（`support_scratch`, `#[cfg(test)] support_records`）。
- **`Community`**（`src/life/community.rs`）：後付け機能用の `RefCell` 4箇所（`support_scratch`, `harmonic_spawn`, `harmonic_spawn_weights`, `harmonic_spawn_candidates`, `birth_surrogate`）。
- **`WorkerState`**（`src/runtime/worker.rs`、40 超フィールド）：再生状態・UI配信・解析同期・研究観測・テストプローブが混在。

### 4.2 本番コード巨大ファイル（Stage 4 対象、テスト除く実測）
| ファイル | 本番行数 | 全体行数 | 主な分割軸 |
|---|---:|---:|---|
| `src/life/pitch_core.rs` | 2,905 | 5,229 | 候補生成 / スコアリング / 決定・更新 / テスト |
| `src/life/voice.rs` | 2,340 | 2,977 | ライフサイクル・制御 / 発音・参加 / 身体連携 |
| `src/life/flora.rs` | 2,134 | 4,409 | バイオマス力学 / レーン・合成 / 拡張評価 |
| `src/life/community.rs` | 1,445 | 2,261 | 既に `community/` 分割済み、残る責務と `RefCell` 解消 |
| `src/life/schedule_renderer.rs` | 1,231 | 3,094 | コマンド適用 / ミキシング / バス・LOO 捕捉 |
| `src/life/temporal_participation.rs` | 1,177 | 2,884 | 予測記録 / 参加判定 / テスト |
| `src/scripting/docs.rs` | 2,355 | 2,355 | オーナー別または tier 別の定義ファイル分割 |

### 4.3 `#[cfg(test)]` の本番侵食上位（Stage 2 対象）
`src/core/auditory_frontend.rs`（58箇所）、`src/life/pitch_core.rs`（24箇所）、`src/life/sound/oscillator_bank.rs`（21箇所）、`src/runtime/mod.rs`（19箇所）、`src/life/voice.rs`（16箇所）、`src/core/temporal_expectation.rs`（16箇所）、`src/life/sound/tone.rs`（13箇所）、`src/runtime/wiring.rs`（12箇所）、`src/life/schedule_renderer.rs`（12箇所）、`src/life/body_metabolism.rs`（12箇所）、`src/core/history_prediction.rs`（12箇所）、`src/life/temporal_participation.rs`（11箇所）。
