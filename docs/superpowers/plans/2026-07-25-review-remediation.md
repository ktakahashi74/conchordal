# コードレビュー指摘の是正 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2026-07-25 の全体レビューで確定した、ドキュメントの失効・worker_loop の割当違反・死んだコード・空虚なテストを是正する。

**Architecture:** 4フェーズを順に実行する。Phase 1 はドキュメントのみ（src 非変更）、Phase 2 は実行時性能、Phase 3 は削除と集約、Phase 4 はテストの実効性回復。フェーズ間に依存はないが、Phase 1 を先に完了させることで以降のフェーズが正しい規約の下で走る。

**Tech Stack:** Rust 2024 edition / cargo / clippy / rhai 1.24

## Global Constraints

- コメントはすべて簡潔な英語。コード内に日本語を書かない（AGENTS.md:76）
- `f32` を優先。`worker_loop` 内でヒープ確保しない（AGENTS.md:77）
- YAGNI 厳守。2つ目の具体的用途が現れるまで抽象化を追加しない（AGENTS.md:81）
- 実際に起こりうるエラーだけを扱う（AGENTS.md:82）
- ヘルパは3箇所以上で使われる場合のみ切り出す（AGENTS.md:84）
- alpha フェーズにつき後方互換を既定で保持しない（AGENTS.md:227）
- コミットはユーザーが明示的に要求したターンのみ（AGENTS.md:231）
- コミット前に必ず `cargo clippy -- -D warnings` を通す（AGENTS.md:232）
- 各タスク完了時に `cargo test --all-targets` を実行し、603 passed / 0 failed を維持する
- `web/` には触れない（AGENTS.md:216）

## 方針決定（ユーザー承認済み 2026-07-25）

- **命名規約**: 規約を実態に合わせる（コードの全面リネームはしない）。`err_*` は実装ゼロのため規約から削除する。
- **エアギャップ**: AGENTS.md の記述を実態に合わせるのみ。強制機構やテストは追加しない。

---

## Phase 1: AGENTS.md の実態同期

`CLAUDE.md` は `AGENTS.md` への symlink。編集対象は `AGENTS.md` のみ。src 変更なし。

### Task 1: エアギャップ記述を実態に合わせる

**Files:**
- Modify: `AGENTS.md:66-73`

**Interfaces:**
- Consumes: なし
- Produces: なし（ドキュメントのみ）

- [ ] **Step 1: 現状の記述が虚偽であることを再確認**

Run: `grep -n "debug_assertions" src/cli.rs src/app.rs`
Expected: 出力なし（exit 1）。`cli.rs`・`app.rs` に `debug_assertions` は存在しない。

- [ ] **Step 2: AGENTS.md:69 を書き換える**

現在の行:
```markdown
  The `#[cfg(debug_assertions)]` guards in `cli.rs` and `app.rs` enforce this.
```

置換後:
```markdown
  This is upheld by absence, not by a guard: `cli.rs` exposes no output-path
  flag, and `main.rs` never calls `run_render`. There is no regression test —
  if you add a disk-write path to the `conchordal` binary, nothing will stop you.
```

- [ ] **Step 3: 変更を確認**

Run: `sed -n '66,75p' AGENTS.md`
Expected: 新しい3行が表示され、`debug_assertions` の語が消えている。

- [ ] **Step 4: 事実確認**

Run: `grep -n "run_render" src/main.rs`
Expected: 出力なし。`main.rs` は `run_render` を呼ばない（記述の正しさを担保）。

### Task 2: Rhai Spawn API 節を現行 API に差し替える

**Files:**
- Modify: `AGENTS.md:148-151`（Consonance Variants 5）
- Modify: `AGENTS.md:162-165`（Rhai Spawn API 節）

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 旧 API が削除済みであることを再確認**

Run: `grep -rn "consonance_density_pmf\|SpawnStrategy::ConsonanceDensity" src/ --include=*.rs`
Expected: 出力なし（exit 1）。

Run: `grep -n "consonance_density" src/scripting/tests.rs`
Expected: `removed_pre_040_api_names_are_not_registered` テスト内の1行がヒット。削除済みAPIとして登録拒否がテストされている。

- [ ] **Step 2: AGENTS.md:148-151（Consonance Variants の5番）を書き換える**

現在:
```markdown
5. `consonance_density_pmf`
- Definition: normalized PMF from density mass; uniform fallback on all-zero totals.
- Implementation: global PMF cache in `src/core/landscape.rs`.
- Usage: `SpawnStrategy::ConsonanceDensity` and Rhai spawn API.
```

置換後:
```markdown
5. `consonance_density_pmf`
- Definition: normalized PMF from density mass; uniform fallback on all-zero totals.
- Implementation: `build_consonance_density` in `src/core/landscape.rs`.
- Usage: currently exercised only by inline tests. Spawn placement goes through
  `SpawnStrategy::Field` (see the Rhai Spawn API section), not through this PMF.
```

- [ ] **Step 3: AGENTS.md:162-165（Rhai Spawn API 節）を全面差し替える**

現在:
```markdown
## Rhai Spawn API
- `consonance_density_pmf(min_freq, max_freq)` builds `SpawnStrategy::ConsonanceDensity`.
- Spawn sampling is range-local in `Population`: it builds local masses with occupancy masks and normalizes in-range.
- If range-local total mass is zero, fallback stays in-range and remains well-defined (unoccupied-uniform first, then full-range uniform if all occupied).
```

置換後:
```markdown
## Rhai Spawn API
Placement builders take `(min_freq, max_freq)` and produce a `Placement`
(`src/scripting/engine.rs`), which lowers to `SpawnStrategy` in `src/scenario.rs:575`.

- `consonance(lo, hi)` / `dissonance(lo, hi)` / `edge(lo, hi)` / `gap(lo, hi)` /
  `random(lo, hi)` build `SpawnStrategy::Field` with `FieldTarget::Consonance` /
  `Dissonance` / `Edge` / `Gap` / `Uniform` respectively (`src/scenario.rs:551`).
- `line(lo, hi)` builds `SpawnStrategy::Linear`.
- Modifiers: `.peak()` / `.density()` set `FieldSampling` (`src/scenario.rs:566`;
  `Density` is the default), `.tension(t)` sets the tension degree in `[0,1]`
  (Consonance target only), `.count(n)` and `.range(min_mul, max_mul)` shape the batch.
- Spawn sampling is range-local in `Community` (`src/life/community.rs`): it builds
  local masses with occupancy masks and normalizes in-range.
- If range-local total mass is zero, fallback stays in-range and remains well-defined
  (unoccupied-uniform first, then full-range uniform if all occupied).
```

- [ ] **Step 4: 書籍側と矛盾しないことを確認**

Run: `cargo test --test rhai_book_accuracy 2>&1 | tail -5`
Expected: PASS（AGENTS.md はこのテストの対象外だが、記述した API 名が実在することの傍証になる）

Run: `grep -rn "consonance(\|dissonance(\|edge(\|gap(" docs/rhai_book/src/reference/api.md | head -5`
Expected: 生成済み API リファレンスに同じ関数名が存在する。

### Task 3: 存在しないファイル参照と Consonance Variants の欠落を直す

**Files:**
- Modify: `AGENTS.md:140`, `AGENTS.md:147`（`population.rs` → `community.rs`）
- Modify: `AGENTS.md:133-151`（`_eff` 系3変種の追記）

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: リネーム済みであることを再確認**

Run: `ls src/life/population.rs 2>&1; ls src/life/community.rs`
Expected: 前者は "No such file"、後者は存在。

- [ ] **Step 2: `population.rs` 参照2箇所を置換**

`AGENTS.md:140` の `src/life/population.rs` を `src/life/community.rs` に置換。
`AGENTS.md:147` の `src/life/population.rs` を `src/life/community.rs` に置換。

- [ ] **Step 3: `_eff` 系3変種を Consonance Variants 節の末尾（現在の5番の後）に追記**

```markdown
6. `consonance_field_score_eff` / `consonance_field_level_eff` / `consonance_density_mass_eff`
- Definition: variants 1, 2 and 4 after habituation erosion is applied.
- Implementation: `src/core/landscape.rs:99,101,103`; driven per hop from
  `src/runtime/mod.rs` when `[psychoacoustics.habituation]` is enabled.
- Usage: these are the views actual consumers read. `src/listener_twin/mod.rs:182,188`
  reads `consonance_field_level_eff`, not the un-eroded variant 2.
- With habituation disabled (the default) the `_eff` views equal their base variants.
```

- [ ] **Step 4: 記述の正しさを確認**

Run: `grep -n "consonance_field_score_eff\|consonance_field_level_eff\|consonance_density_mass_eff" src/core/landscape.rs | head -3`
Expected: 99, 101, 103 行目にフィールド定義がある。

Run: `grep -n "consonance_field_level_eff" src/listener_twin/mod.rs`
Expected: 182, 188 行目でヒット。

### Task 4: Config Keys 節を全セクションに拡張する

**Files:**
- Modify: `AGENTS.md:153-160`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 現行の config セクションを確認**

Run: `grep -n "pub [a-z_]*: [A-Za-z]*Config" src/config.rs`
Expected: `AppConfig` が `audio` / `analysis` / `psychoacoustics` / `dcc` / `playback` を持つ。

- [ ] **Step 2: Config Keys 節に habituation を追記する**

現在の節の末尾（`AGENTS.md:160` の後）に追記:

```markdown
- `[psychoacoustics.habituation]`
- `enabled` (default: `false`), `satiation_sec` (`5.0`), `recovery_sec` (`8.0`),
  `ref_drive` (`0.25`)
- When `enabled = false` the `_eff` consonance views equal their base variants.

Other top-level sections exist and are defined in `src/config.rs`:
`[audio]`, `[analysis]`, `[dcc]`, `[playback]`. They are not enumerated here —
read `src/config.rs` for their keys and defaults.

Note: `AppConfig` uses `#[serde(default)]` throughout, so a misspelled TOML key is
silently ignored and the default is used. There is no unknown-key rejection.
```

- [ ] **Step 3: 既定値が正しいことを確認**

Run: `grep -n "default_satiation_sec\|default_recovery_sec\|default_ref_drive\|default_enabled" -A 2 src/config.rs | grep -E "5\.0|8\.0|0\.25|false"`
Expected: `5.0` / `8.0` / `0.25` / `false` がそれぞれ出力される。

### Task 5: 命名規約節を実態に合わせる

方針 A（承認済み）: コードをリネームせず、規約を実態に合わせる。

**Files:**
- Modify: `AGENTS.md:98-122`（Terminology 節）
- Modify: `AGENTS.md:172-180`（Suffix convention 節の例）

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 実態を計測して記録する**

Run:
```bash
for p in err_ pred_ perc_ r_pot_scan h_pot_scan c_score_scan c_density_scan c_level_scan; do
  echo "$p: $(grep -rc "$p" src/ --include=*.rs | awk -F: '{s+=$2} END{print s+0}')"
done
```
Expected: `err_: 0` / `pred_: 86` / `perc_: 29` / `r_pot_scan: 17` / `h_pot_scan: 10` / `c_score_scan: 7` / `c_density_scan: 8` / `c_level_scan: 5`

- [ ] **Step 2: Axis A から `err_*` を削除する**

`AGENTS.md:105` の行を削除:
```markdown
- **error** (`err_*`): `err_* = perc_* - pred_*`.
```

`AGENTS.md:122` の行も削除:
```markdown
- `err_c_field_level_scan = perc_c_field_level_scan - pred_c_field_level_scan`
```

- [ ] **Step 3: Axis A に適用範囲の但し書きを足す**

`AGENTS.md:107` の `` `perceptual` is reserved for this axis only. `` の直後に追記:

```markdown
The `pred_` / `perc_` prefixes are used only where both origins coexist and must be
distinguished — chiefly the WorldModel layer (`src/life/generator_model.rs`,
`src/listener_twin/`). Kernel-layer scans that have only one possible origin carry no
prefix (`r_pot_scan`, `h_pot_scan`, `c_score_scan`). Do not add a prefix that
distinguishes nothing.
```

- [ ] **Step 4: Axis B に `_state_` の定義を追加する**

`AGENTS.md:115`（`energy` の行）の後に追記:

```markdown
- **state** (`*_state_*`): a bounded `[0,1]` view of a potential, normalized against a
  reference (`src/core/psycho_state.rs`). Distinct from `level`, which is a sigmoid of a
  score.
```

- [ ] **Step 5: 実在しない正準例を実在するものに差し替える**

`AGENTS.md:117-121` の例のリストを置換:

現在:
```markdown
Potential/representation is orthogonal to pred/perc:
- `pred_h_pot_scan`, `pred_h_state_scan`
- `perc_r_pot_scan`, `perc_r_state_scan`
- `perc_c_field_score_scan`, `perc_c_field_level_scan`
- `perc_c_density_scan`, `perc_c_field_energy_scan`
```

置換後（すべて実在する名前）:
```markdown
Potential/representation is orthogonal to pred/perc. Real examples from the tree:
- unprefixed kernel scans: `r_pot_scan`, `h_pot_scan`, `c_score_scan`,
  `c_level_scan`, `c_density_scan`, `c_energy_scan`
- state views: `r_state01_scan`, `h_state01_scan` (`src/core/psycho_state.rs:80,93`)
- prefixed, where both origins coexist: `pred_c_field_level_scan`,
  `perc_c_field_level_scan`, `perc_habituation_state_scan`
```

- [ ] **Step 6: Suffix convention 節の例も実在するものに差し替える**

`AGENTS.md:177-180` の Example ブロックを置換:

現在:
```markdown
Example:
- `perc_r_state_scalar`
- `pred_c_field_level_scan`
- `perc_c_field_score_scan`
```

置換後:
```markdown
Example:
- `r_state01_scan` (Log2Space-aligned)
- `pred_c_field_level_scan` (prefixed because a `perc_` counterpart exists)
- `loudness_mass` (a scalar summary; `_scalar` is only needed when a `_scan` of the
  same name would otherwise be ambiguous)
```

- [ ] **Step 7: 差し替えた名前がすべて実在することを確認**

Run:
```bash
for n in r_pot_scan h_pot_scan c_score_scan c_level_scan c_density_scan c_energy_scan \
         r_state01_scan h_state01_scan pred_c_field_level_scan perc_c_field_level_scan \
         perc_habituation_state_scan loudness_mass; do
  c=$(grep -rc "$n" src/ --include=*.rs | awk -F: '{s+=$2} END{print s+0}')
  [ "$c" -gt 0 ] && echo "OK $n ($c)" || echo "MISSING $n"
done
```
Expected: 全行が `OK`。`MISSING` が1つもない。

### Task 6: End-of-Task 手順の `$?` バグと残りの細かい乖離を直す

**Files:**
- Modify: `AGENTS.md:44-63`（手順）
- Modify: `AGENTS.md:7`（nsgt FFT variant）
- Modify: `AGENTS.md:23-24`（`web/` の位置）と構造説明の欠落
- Modify: `AGENTS.md:217`（`samples/README.md` の craft rules）
- Modify: `AGENTS.md:219-224`（生成物の手編集禁止）

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: `$?` バグを再現確認**

Run: `bash -c 'true' ; bash -c 'echo "exit=$?"'`
Expected: `exit=0`。別プロセスなので直前のコマンドの終了コードは伝わらない。手順書が2つのコードブロックに分かれている限り `$?` は cargo test を指さない。

- [ ] **Step 2: 手順を単一ブロックに統合する**

`AGENTS.md:44-59` の2ブロックを1ブロックに置換:

```markdown
### 1. Run cargo tests and record status

Run tests with full output and backtraces enabled, and record the exit code in the
same shell invocation — `$?` does not survive a separate shell.

```bash
set -o pipefail
( RUST_BACKTRACE=1 cargo test -- --nocapture ) 2>&1 | tee test_report.txt
echo "cargo test exit=$? @ $(date -Iseconds)" > test_status.txt
```
```

`### 2. Record test status` の見出しと、その下の重複したコードブロックを削除する。
その下の3つの箇条書き（`test_report.txt must contain...` 以降）は残す。

- [ ] **Step 3: nsgt の記述を実態に合わせる**

Run: `ls src/core/nsgt*.rs`
Expected: `nsgt_kernel.rs` と `nsgt_rt.rs` の2つのみ。

`AGENTS.md:7` を置換:

現在:
```markdown
    - `nsgt*.rs`: Non-Stationary Gabor Transform implementations (RT, FFT, and Kernel variants).
```

置換後:
```markdown
    - `nsgt_rt.rs` / `nsgt_kernel.rs`: Non-Stationary Gabor Transform (streaming RT
      variant and offline kernel variant). An older FFT variant is retired to `attic/`.
```

- [ ] **Step 4: 構造説明の欠落と `web/` の位置を直す**

`AGENTS.md:23-24` の `web/` エントリを `src/` 配下のリストから削除し、代わりに `src/` リストの末尾に実在モジュールを追記:

```markdown
  - `scripting/` (Rhai engine, API registration, generated docs).
  - `runtime/` (thread wiring, worker_loop, scenario execution).
  - `listener_twin/` (listener-side model).
  - `scenario.rs` / `config.rs` / `dcc_coupler.rs`: scenario IR, TOML config, DCC coupling.
  - `bin/`: `render.rs` (conchordal-render) and `gen_rhai_defs.rs`.
```

`Entrypoints:` の行の後に追記:
```markdown
- `web/` (Zola site sources) lives at the repository root, not under `src/`.
```

- [ ] **Step 5: `samples/README.md` の記述を実態に合わせる**

Run: `wc -l samples/README.md; head -5 samples/README.md`
Expected: 44行程度のサンプル索引であり、craft rules ではない。

`AGENTS.md:217` の末尾の文を置換:

現在:
```markdown
Compositional craft rules live in `samples/README.md`.
```

置換後:
```markdown
`samples/README.md` is an index of the études, not a rules document.
```

- [ ] **Step 6: 生成物の手編集禁止と research サンプルのシード方針を明記する**

`AGENTS.md:219`（`docs/rhai_book` の行）の後に追記:

```markdown
- `rhai-defs/conchordal.d.rhai` and `docs/rhai_book*/src/reference/api.md` are generated
  by `cargo run --bin gen_rhai_defs`. Never hand-edit them; `tests/rhai_defs_in_sync.rs`
  fails if they drift from the registry.
- `samples/` top-level études must not pin a seed; `samples/research/` assays must pin one.
  This is enforced by `tests/sample_seed_policy.rs`.
```

- [ ] **Step 7: 全体を通しで確認**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: すべて `ok`。0 failed。ドキュメント lint テスト（`rhai_book_accuracy` / `samples_no_legacy_keys` / `sample_seed_policy`）が壊れていないこと。

- [ ] **Step 8: 残った乖離がないか再走査**

Run:
```bash
grep -o '`src/[a-z_/]*\.rs`' AGENTS.md | tr -d '`' | sort -u | while read p; do
  [ -e "$p" ] || echo "MISSING: $p"
done
```
Expected: 出力なし。

Run: `grep -n "debug_assertions\|consonance_density_pmf(\|population.rs\|err_" AGENTS.md`
Expected: 出力なし（すべて是正済み）。

---

## Phase 2: worker_loop の割当違反と実行時性能

計測トレース: `runtime/mod.rs:636 spawn_analysis_worker` → `core/analysis_worker.rs:37` → `AnalysisStream::process`。1オーディオホップにつき1回。現状およそ40確保/ホップ。

### Task 7: `tone.rs` の VecDeque を事前確保にする

AGENTS.md:77「Avoid allocations in the audio thread (`worker_loop`)」の直接違反。
`schedule_renderer.render` は `worker_loop`（`runtime/mod.rs:1581`）から呼ばれる。

**Files:**
- Modify: `src/life/sound/tone.rs:155`（`VecDeque::new()`）

**Interfaces:**
- Consumes: なし
- Produces: なし（内部フィールドの初期容量のみ）

- [ ] **Step 1: 呼び出し経路を再確認**

Run: `grep -n "schedule_renderer.render" src/runtime/mod.rs`
Expected: 1581行目でヒット。`worker_loop`（1218行目定義、loop は 1308-1774）の内側。

- [ ] **Step 2: 現状の初期化を確認**

Run: `sed -n '150,160p' src/life/sound/tone.rs`
Expected: `pending_updates: VecDeque::new(),` が見える。

- [ ] **Step 3: 事前確保に変更**

`src/life/sound/tone.rs:155` を置換:

```rust
            // Pre-sized so the worker_loop insert path never reallocates.
            pending_updates: VecDeque::with_capacity(16),
```

- [ ] **Step 4: テストとリントを実行**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:|error" | tail -5`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 8: `runtime/mod.rs:1401` の二重確保を除去する

`Arc::from(vec.clone())` — `Arc::from` 自体が `Vec` をコピーするので `.clone()` は無駄。

**Files:**
- Modify: `src/runtime/mod.rs:1401`

**Interfaces:**
- Consumes: `observe_consonance_field_level(Tick, Arc<[f32]>)`
- Produces: なし

- [ ] **Step 1: 呼び出し先のシグネチャを確認**

Run: `grep -n "fn observe_consonance_field_level" -A 3 src/life/generator_model.rs`
Expected: 第2引数が `Arc<[f32]>` であること。

- [ ] **Step 2: `.clone()` を除去**

`src/runtime/mod.rs:1401` を置換:

現在:
```rust
                    Arc::from(current_landscape.consonance_field_level_eff.clone()),
```

置換後:
```rust
                    Arc::from(&current_landscape.consonance_field_level_eff[..]),
```

- [ ] **Step 3: 型が通ることを確認**

Run: `cargo check --all-targets 2>&1 | tail -5`
Expected: エラーなし。`Arc<[f32]>: From<&[f32]>` が効く。

- [ ] **Step 4: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 9: `stream/analysis.rs` の二重コピーを除去する

`env.to_vec()` した直後に `envelope.clone()` で同じデータを2回コピーしている。

**Files:**
- Modify: `src/core/stream/analysis.rs:62-68`

**Interfaces:**
- Consumes: `RtNsgtKernelLog2::process_hop(&mut self, &[f32]) -> &[f32]`
- Produces: なし

- [ ] **Step 1: 現状を確認**

Run: `sed -n '60,72p' src/core/stream/analysis.rs`
Expected: `env.to_vec()` と `envelope.clone()` の2つの確保が見える。

- [ ] **Step 2: 借用期間の制約を確認**

`process_hop` は `&mut self` を取り `&[f32]` を返すため、返り値を保持したまま `self.spectral_frontend` を触れない。
`self.last_landscape.nsgt_power` へ直接コピーし、以降はそのスライスを読む形にする。

- [ ] **Step 3: 二重コピーを1回に減らす**

置換前:
```rust
        let envelope: Vec<f32> = {
            let env = self.nsgt_rt.process_hop(audio);
            env.to_vec()
        };
        self.last_landscape.nsgt_power = envelope.clone();
        let dt_sec = audio.len() as f32 / self.params.fs;
        let spectral_frame =
            self.spectral_frontend
                .process_nsgt_power(&envelope, dt_sec, &self.params);
```

置換後:
```rust
        {
            // Copy once into the owned snapshot buffer; the borrow of nsgt_rt ends here.
            let env = self.nsgt_rt.process_hop(audio);
            let dst = &mut self.last_landscape.nsgt_power;
            if dst.len() == env.len() {
                dst.copy_from_slice(env);
            } else {
                dst.clear();
                dst.extend_from_slice(env);
            }
        }
        let dt_sec = audio.len() as f32 / self.params.fs;
        let spectral_frame = self.spectral_frontend.process_nsgt_power(
            &self.last_landscape.nsgt_power,
            dt_sec,
            &self.params,
        );
```

- [ ] **Step 4: 借用エラーが出た場合の対処**

`self.last_landscape.nsgt_power` の不変借用と `self.spectral_frontend` の可変借用が衝突する場合、
`process_nsgt_power` に渡す直前で `std::mem::take` は使わない（`Landscape` の状態を壊す）。
代わりに `AnalysisStream` に `scratch_env: Vec<f32>` フィールドを追加し、そこにコピーしてから
`nsgt_power` へは `copy_from_slice` する。フィールド追加は `AnalysisStream::new` で
`vec![0.0; space.n_bins()]` として初期化する。

- [ ] **Step 5: コンパイルとテスト**

Run: `cargo check --all-targets 2>&1 | tail -5`
Expected: エラーなし。

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

- [ ] **Step 6: 確保が減ったことを目視確認**

Run: `sed -n '58,80p' src/core/stream/analysis.rs`
Expected: `to_vec()` と `clone()` がこのブロックから消えている。

### Task 10: `pitch_core.rs` の Voice ごとカーネル再構築をやめる — **見送り (2026-07-25)**

**判断: スコープ外とする。**

実行前の調査で、この経路が**デフォルトでは走らない**ことが判明した:

- `exact_loo_consonance_score_scan`（`src/life/pitch_core.rs:1020`）は冒頭で
  `if !matches!(mode, LeaveSelfOutMode::ExactScan) { return None; }` と早期 return する。
- `LeaveSelfOutMode` の `#[default]` は `ApproxHarmonics`（`src/life/control.rs:192`）。
- `ExactScan` は Rhai で `"exact"` / `"exact_scan"` / `"scan"` を明示指定した場合のみ
  有効になる（`src/scripting/mod.rs:51`）。オプトイン機能。

したがって LUT 1601 要素の再構築は既定の再生経路では発生しない。
これを直すには `Landscape` に `Arc<RoughnessKernel>` フィールドを追加し、
`roughness_kernel_params` への代入6箇所（`stream/analysis.rs:31,100,164`、
`runtime/mod.rs:374,385`）すべてにキャッシュ無効化を通す必要がある。
`Landscape::new` の呼び出しは50箇所以上ある。

デフォルト無効な経路のために、この規模の構造変更を**計測なしに**入れるのは
AGENTS.md:81 の YAGNI に反する。`ExactScan` を常用するシナリオが実際に現れ、
かつプロファイルで律速だと確認できた時点で再検討する。

<details>
<summary>元の計画（保留）</summary>

`RoughnessKernel::new(params, 0.005)` は `half_width_erb=4.0 / 0.005` → 長さ1601のLUTを毎回構築し、
`eval_kernel_delta_erb`（`exp()` を含む）を1601回呼ぶ。これが Voice の pitch 判断ごとに走る。
対処するなら `Landscape` に `Arc<RoughnessKernel>` / `Arc<HarmonicityKernel>` を持たせ、
params 変更時のみ再構築する。`KernelParams` / `HarmonicityParams` には `PartialEq` が無いため、
変更検知にはまず derive の追加が要る。

</details>

### Task 10b: （旧 Task 10 の内容・保留）

`RoughnessKernel::new(params, 0.005)` は `half_width_erb=4.0 / 0.005` → 長さ1601のLUTを毎回構築し、
`eval_kernel_delta_erb`（`exp()` を含む）を1601回呼ぶ。これが Voice の pitch 判断ごとに走る。

**Files:**
- Modify: `src/life/pitch_core.rs:1045-1052`
- Modify: `src/core/landscape.rs`（キャッシュ用フィールドの追加）

**Interfaces:**
- Consumes: `RoughnessKernel::new(KernelParams, f32) -> RoughnessKernel`,
  `HarmonicityKernel::new(&Log2Space, HarmonicityParams) -> HarmonicityKernel`
- Produces: `Landscape` にキャッシュ済みカーネルへのアクセサ

- [ ] **Step 1: 再構築コストを確認**

Run: `grep -n "EXACT_LOO_ROUGHNESS_ERB_STEP" src/life/pitch_core.rs`
Expected: 24行目に `0.005`、1050行目で使用。

Run: `grep -n "half_width_erb: 4.0" src/core/roughness_kernel.rs`
Expected: 51行目。LUT長 = `2 * ceil(4.0 / 0.005) + 1 = 1601`。

- [ ] **Step 2: 呼び出し頻度を確認**

Run: `grep -n "propose_target" src/life/pitch_controller.rs src/life/pitch_core.rs | head`
Expected: `pitch_controller.rs:197` から Voice ごとに呼ばれる経路が見える。

- [ ] **Step 3: キャッシュの置き場所を決める**

`Landscape` は `roughness_kernel_params` と `harmonicity_params` と `space` を既に持つ。
これらが不変である限りカーネルも不変なので、`Landscape` に構築済みカーネルを持たせる。
`Landscape` は毎フレーム `clone()` されている（Task 11 で解消予定）ため、
カーネル本体を `Arc` で包んで clone コストを O(1) にする。

`src/core/landscape.rs` の `Landscape` に追加:
```rust
    /// Kernels for exact leave-one-out evaluation, built once per parameter change.
    /// Arc so that cloning a Landscape snapshot stays O(1).
    pub loo_roughness_kernel: std::sync::Arc<RoughnessKernel>,
    pub loo_harmonicity_kernel: std::sync::Arc<HarmonicityKernel>,
```

- [ ] **Step 4: 初期化と再構築のタイミングを実装する**

`Landscape::new`（および params 更新経路）でカーネルを構築する。
既存の params 更新箇所を特定するため、まず調査する:

Run: `grep -n "roughness_kernel_params\|harmonicity_params" src/core/landscape.rs`

params が更新される関数の末尾でカーネルを再構築する。`EXACT_LOO_ROUGHNESS_ERB_STEP` は
`src/life/pitch_core.rs:24` から `src/core/landscape.rs` へ移す（カーネル構築側が持つべき定数）。

- [ ] **Step 5: `pitch_core.rs` を書き換える**

置換前（`src/life/pitch_core.rs:1045-1052`）:
```rust
    let harmonicity_kernel = HarmonicityKernel::new(&landscape.space, landscape.harmonicity_params);
    let (h_blended, _) =
        harmonicity_kernel.potential_h_from_log2_spectrum(&density_loo, &landscape.space);
    let roughness_kernel = RoughnessKernel::new(
        landscape.roughness_kernel_params,
        EXACT_LOO_ROUGHNESS_ERB_STEP,
    );
```

置換後:
```rust
    let (h_blended, _) = landscape
        .loo_harmonicity_kernel
        .potential_h_from_log2_spectrum(&density_loo, &landscape.space);
    let roughness_kernel = &landscape.loo_roughness_kernel;
```

- [ ] **Step 6: 数値が変わっていないことを確認**

このタスクは純粋な計算結果キャッシュであり、出力は変わってはならない。

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。特に `src/life/tests.rs` と `tests/habituation_field_assay.rs` が通ること。

- [ ] **Step 7: 決定性の回帰確認**

Run: `cargo test --test habituation_field_assay 2>&1 | tail -10`
Expected: PASS。`habituation_is_deterministic` が通ること（別プロセス2回実行の厳密一致）。

- [ ] **Step 8: リント**

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 11: `erb_grid` の毎フレーム再計算を除去する

同一値が1フレームに3回計算されている。`landscape_spectral.rs:19-20` に既にキャッシュがある。

**Files:**
- Modify: `src/core/roughness_kernel.rs:274`（`potential_r_from_log2_spectrum_density`）
- Modify: `src/core/stream/analysis.rs:83`
- Modify: `src/core/landscape_spectral.rs`（`erb`/`du` のアクセサ公開）

**Interfaces:**
- Consumes: `erb_grid(&Log2Space) -> (Vec<f32>, Vec<f32>)`
- Produces: `potential_r_from_log2_spectrum_density_with_grid(&self, density, space, erb, du)` —
  既存の `peak_extraction.rs:125` の `_with_grid` 命名に倣う

- [ ] **Step 1: 再計算箇所を確認**

Run: `grep -rn "erb_grid(" src/ --include=*.rs`
Expected: `landscape_spectral.rs`（キャッシュ済み）、`stream/analysis.rs:83`、
`roughness_kernel.rs:274`、`pitch_core.rs:1052` がヒット。

- [ ] **Step 2: `roughness_kernel.rs` に `_with_grid` 変種を追加**

既存の `potential_r_from_log2_spectrum_density` の本体から `erb_grid` 呼び出しを取り除き、
`erb: &[f32], du: &[f32]` を引数で受ける `_with_grid` 版を作る。
既存関数は `_with_grid` を呼ぶ薄いラッパとして残す（テストが既存シグネチャを使っているため）。

- [ ] **Step 3: `SpectralFrontEnd` の `erb`/`du` を読めるようにする**

`src/core/landscape_spectral.rs:19-20` のフィールドに `pub(crate)` アクセサを足す:
```rust
    pub(crate) fn erb_grid_cached(&self) -> (&[f32], &[f32]) {
        (&self.erb, &self.du)
    }
```

- [ ] **Step 4: `stream/analysis.rs` の呼び出しをキャッシュ経由にする**

`compute_potentials` 内の `let (_erb, du) = erb_grid(space);` を
`self.spectral_frontend.erb_grid_cached()` に置き換える。
`compute_potentials` が `&mut self` を取るため借用衝突が起きる場合は、
`erb`/`du` をローカルにコピーせず、`compute_potentials` の引数として渡す形に変える。

- [ ] **Step 5: 数値不変を確認**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。特に `roughness_kernel.rs` のインラインテスト23件。

- [ ] **Step 6: 再計算が消えたことを確認**

Run: `grep -rn "erb_grid(" src/core/stream/analysis.rs src/core/roughness_kernel.rs`
Expected: `roughness_kernel.rs` のラッパ1箇所のみ。ホットパスからは消えている。

### Task 12: `Landscape::clone()` の毎フレーム18確保を削る — **見送り (2026-07-25)**

**判断: スコープ外とする。費用対効果が見合わない。**

`Arc<Log2Space>` 化で削れるのは 18 確保中 2 確保（約11%）にすぎない。残る 14 本の
`Vec<f32>` はチャネル送信のために所有権が必要で、これを消すにはバッファプールと
返却経路の設計が要る（`crossbeam` 越しの往復）。

一方コストは大きい。`Landscape::new` の呼び出しは 50 箇所以上あり、
`space: Log2Space` → `Arc<Log2Space>` の変更は `landscape.space.clone()` を書いている
すべての箇所で型が変わる。deref coercion で通る箇所と通らない箇所が混在する。

加えて、この確保は解析スレッド上で起きており `worker_loop` ではない。
Phase 1 で AGENTS.md:77 にその区別を明記した通り、ここは no-allocation 規約の
対象外である。本気で削るなら別計画としてバッファプールを設計する。

<details>
<summary>元の計画（保留）</summary>

`stream/analysis.rs` の `self.last_landscape.clone()`。`Landscape` は `Vec<f32>` 14本 +
`Log2Space` 内2本。`Arc<Log2Space>` 化が第一段、バッファプール化が本命。

</details>

### Task 12b: （旧 Task 12 の内容・保留）

`stream/analysis.rs:78` の `self.last_landscape.clone()`。`Landscape` は `Vec<f32>` 14本 + `Log2Space` 内2本。
全確保の約45%を占める単独最大の要因。

**Files:**
- Modify: `src/core/stream/analysis.rs:78`
- Modify: `src/core/landscape.rs`（`Log2Space` の共有化）

**Interfaces:**
- Consumes: `AnalysisStream::process(&mut self, &[f32]) -> Landscape`
- Produces: 同一シグネチャを維持（呼び出し側 `analysis_worker.rs:37` を変えない）

- [ ] **Step 1: 確保数を確認**

Run: `grep -c "pub [a-z_]*: Vec<f32>" src/core/landscape.rs`
Expected: 14。

Run: `grep -n "pub space: \|struct Log2Space" -A 6 src/core/landscape.rs src/core/log2space.rs | head -20`
Expected: `Landscape` が `Log2Space` を値で持ち、`Log2Space` が内部に `Vec` を2本持つ。

- [ ] **Step 2: `Log2Space` を `Arc` 共有にする**

`Log2Space` はフレーム間で不変。`Landscape` のフィールドを `Arc<Log2Space>` に変えるだけで
clone あたり2確保が消える。まず影響範囲を調べる:

Run: `grep -rn "\.space\b" src/ --include=*.rs | wc -l`

`&landscape.space` として渡している箇所は `Arc` の `Deref` で変更不要。
`Landscape` を構築している箇所のみ修正が要る。

- [ ] **Step 3: 残る14本の Vec について判断する**

呼び出し側 `analysis_worker.rs:37` はチャネルへ送るために所有権が要る。
バッファプール化は `crossbeam` チャネル越しの返却経路が必要になり、YAGNI に反する規模。
`Arc<Log2Space>` 化（2確保削減）までを本タスクの範囲とし、14本の Vec は
「解析スレッドは worker_loop ではない」ため許容する。

**この判断を AGENTS.md に記録する**（Phase 1 Task 1 で触れた「規約と実態の一致」の原則）:
`AGENTS.md:77` の DSP Efficiency 項に追記:
```markdown
  The no-allocation rule applies to `worker_loop` and the cpal callback. The analysis
  worker (`core/analysis_worker.rs`) runs on its own thread and may allocate; it
  currently allocates ~40 times per hop, dominated by the `Landscape` snapshot it
  sends over the channel.
```

- [ ] **Step 4: 変更を適用してコンパイル**

Run: `cargo check --all-targets 2>&1 | tail -10`
Expected: エラーなし。エラーが出た箇所は `Landscape` 構築側なので、`Arc::new(space)` を足す。

- [ ] **Step 5: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 13: harmonicity の二重計算を除去する — **見送り (2026-07-25)**

**判断: スコープ外とする。等価な置換ではない。**

`drive_and_apply_habituation`（`src/runtime/mod.rs:437`）は `worker_loop` の中で
`apply_pending_landscape_update`（`:1381`）の**直後**に呼ばれる。この更新関数は
`lparams` を書き換えうるため、params が変わったフレームでは:

- 現行コード（`:447`）は**新しい** `lparams.harmonicity_kernel` で再計算する。
- `landscape.harmonicity` は analysis スレッドが**古い** kernel で計算した値。

したがって参照への置換は1フレーム分の遅れを持ち込み、params 変更時に挙動が変わる。
削減できるのは habituation 有効時（既定は無効）の Vec×3 + O(N·harmonics) のみで、
正確性を落とす対価に見合わない。現状維持とする。

<details>
<summary>元の計画（保留）</summary>

`stream/analysis.rs:98-101` と `runtime/mod.rs:447-449` が同一入力・同一カーネルで
harmonicity を2回計算している。params が変わらない定常状態では確かに重複だが、
上記の理由により単純な参照置換はできない。解くなら params 更新時に analysis 側へ
再計算を要求し、結果が届くまで habituation の駆動を1フレーム遅らせる設計が要る。

</details>

### Task 13b: （旧 Task 13 の内容・保留）

`stream/analysis.rs:98-101` と `runtime/mod.rs:447-449` が同一入力・同一カーネルで
harmonicity を2回計算している。後者は habituation 有効時のみ発火（既定は無効）。

**Files:**
- Modify: `src/runtime/mod.rs:443-452`

**Interfaces:**
- Consumes: `Landscape::harmonicity`（既に計算済みのフィールド）
- Produces: なし

- [ ] **Step 1: 入力が同一であることを確認**

Run: `sed -n '443,455p' src/runtime/mod.rs`
Run: `sed -n '96,104p' src/core/stream/analysis.rs`
Expected: どちらも `subjective_intensity` と同じ `space` を `potential_h_from_log2_spectrum` に渡している。

- [ ] **Step 2: 既存フィールドで代替できるか確認**

Run: `grep -n "pub harmonicity" src/core/landscape.rs`
Expected: `Landscape` に harmonicity のスキャンが既に格納されている。

- [ ] **Step 3: 再計算を参照に置き換える**

`runtime/mod.rs:447-449` の `potential_h_from_log2_spectrum(...)` 呼び出しを
`landscape.harmonicity` の参照に置換する。値が一致しない場合（`_eff` 侵食の順序差など）は
置換せず、この Task をスキップして理由を記録する。

- [ ] **Step 4: habituation を有効にして数値一致を検証**

Run: `cargo test --test habituation_field_assay 2>&1 | tail -10`
Expected: PASS。この経路は habituation 有効時のみ走るため、このテストが唯一の検証手段。

- [ ] **Step 5: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

---

## Phase 3: 死んだコードの削除と重複の集約

`lib.rs` が `pub mod core` / `pub mod life` のため `dead_code` lint が効かず、未参照 pub API が隠れている。

### Task 14: 0..1 クランプの3実装を統一する

挙動が割れているため、これはバグ源。`float.rs:2 clamp01_finite` は `±Inf → 0.0`、
`psycho_state.rs:23 sanitize01` は `+Inf → 1.0`。`psycho_state.rs:18 clamp01` は中身のないエイリアス。

**Files:**
- Modify: `src/core/psycho_state.rs:18`（`clamp01` の削除）
- Modify: `src/core/float.rs:2`（`clamp01_finite` の挙動統一）
- Modify: `src/life/articulation_core.rs`, `src/life/metabolism_policy.rs`（呼び出し側）

**Interfaces:**
- Consumes: なし
- Produces: `sanitize01(f32) -> f32` を唯一の 0..1 クランプとする

- [ ] **Step 1: 3実装の挙動差を確認**

Run: `sed -n '1,30p' src/core/float.rs`
Run: `sed -n '15,30p' src/core/psycho_state.rs`
Expected: `clamp01_finite` が `is_finite()` で弾いて `0.0` を返し、`sanitize01` が
`+Inf → 1.0` に飽和させる差が見える。

- [ ] **Step 2: 差を固定する失敗テストを書く**

`src/core/psycho_state.rs` のテストモジュールに追加:
```rust
    #[test]
    fn sanitize01_saturates_infinities() {
        assert_eq!(sanitize01(f32::INFINITY), 1.0);
        assert_eq!(sanitize01(f32::NEG_INFINITY), 0.0);
        assert_eq!(sanitize01(f32::NAN), 0.0);
        assert_eq!(sanitize01(0.5), 0.5);
    }
```

- [ ] **Step 3: テストを実行して現状の挙動を確認**

Run: `cargo test --lib sanitize01_saturates_infinities 2>&1 | tail -10`
Expected: PASS（現行 `sanitize01` の挙動を固定するテストなので通る）。通らなければ実装を先に読む。

- [ ] **Step 4: 中身のないエイリアス `clamp01` を削除**

Run: `grep -rn "clamp01\b" src/ --include=*.rs`
Expected: `psycho_state.rs:18` の定義と `:101` の使用のみ。使用箇所を `sanitize01` に置換して定義を削除。

- [ ] **Step 5: `clamp01_finite` の利用者を確認して統一する**

Run: `grep -rn "clamp01_finite" src/ --include=*.rs`
Expected: `float.rs` の定義、`articulation_core.rs`、`metabolism_policy.rs` の3箇所。

利用者が2箇所しかないため、`clamp01_finite` を削除して `sanitize01` に寄せる。
ただし `articulation_core.rs` / `metabolism_policy.rs` で `+Inf` が渡りうるかを先に確認する:
Run: `grep -n "clamp01_finite" -B 3 src/life/articulation_core.rs src/life/metabolism_policy.rs`

`+Inf` が入力されうる経路であれば、飽和（`1.0`）のほうが `0.0` より安全。置換して問題ない。

- [ ] **Step 6: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 15: `finite_or` と `sanitize_nonnegative_finite` を `float.rs` に集約する

`finite_or` はバイト単位で同一の実装が2箇所。`sanitize_nonnegative_finite` は3実装。
AGENTS.md:84「3箇所以上で使われる場合のみ切り出す」を満たす。

**Files:**
- Modify: `src/core/float.rs`（集約先）
- Modify: `src/core/roughness_kernel.rs:68`（削除）
- Modify: `src/core/harmonicity_kernel.rs:334`（削除）
- Modify: `src/core/landscape.rs:426`（削除）
- Modify: `src/core/mode_pattern.rs:523`（削除）
- Modify: `src/core/consonance_kernel.rs:112`（`sanitize_finite` を `finite_or` に統合）

**Interfaces:**
- Consumes: なし
- Produces: `float::finite_or(f32, f32) -> f32`, `float::sanitize_nonnegative_finite(f32, f32) -> f32`

- [ ] **Step 1: 重複を確認**

Run: `grep -rn "fn finite_or\|fn sanitize_nonnegative_finite\|fn sanitize_finite" src/ --include=*.rs`
Expected: `finite_or` が2箇所、`sanitize_nonnegative_finite` が3箇所、`sanitize_finite` が1箇所。

- [ ] **Step 2: 実装が同一であることを確認**

Run: `grep -n "fn finite_or" -A 8 src/core/roughness_kernel.rs src/core/harmonicity_kernel.rs`
Expected: 本体が同一であること。差があれば集約前に差を潰す。

- [ ] **Step 3: `float.rs` に集約版を用意する**

`src/core/float.rs` に `pub(crate) fn finite_or(x: f32, fallback: f32) -> f32` が既にあるか確認し、
なければ追加する。既にある場合はそれを使う。

- [ ] **Step 4: 各ファイルのローカル定義を削除して `crate::core::float::` を使う**

各ファイルで `use crate::core::float::finite_or;` を追加し、ローカル定義を削除する。

- [ ] **Step 5: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし（`dead_code` が出たら削除漏れ）。

### Task 16: 未参照の pub API を削除する

**Files:**
- Modify: `src/core/erb.rs`（`ErbSpace` 型を丸ごと削除）
- Modify: `src/core/log2space.rs:19,70,80,86,196`
- Modify: `src/core/landscape.rs:249,255,260`
- Modify: `src/core/nsgt_kernel.rs:97,419`
- Modify: `src/core/nsgt_rt.rs:183,215`
- Modify: `src/core/timebase.rs:31,44`
- Modify: `src/core/harmonic_ratios.rs:17`
- Modify: `src/core/peak_extraction.rs:295`
- Modify: `src/core/psycho_state.rs:13`

**Interfaces:**
- Consumes: なし
- Produces: なし（削除のみ）

- [ ] **Step 1: 各シンボルが本当に未参照であることを個別に確認**

削除前に必ず1つずつ確認する。テスト・examples・docs も含めて検索する:
```bash
for s in fold_to_octave_near Log2SpaceSpec hz_to_log2 log2_to_hz delta_log2 \
         evaluate_pitch_level_log2 consonance_field_score_at consonance_field_level_at \
         NsgtBand analyze_envelope current_envelope reconfigure_smoothing \
         peaks_to_delta_density HarmonicityRef ceil_to_hop_tick min_lead_ticks; do
  c=$(grep -rn "\b$s\b" src/ tests/ examples/ docs/ --include=*.rs --include=*.md 2>/dev/null | \
      grep -v "pub fn $s\|pub struct $s\|fn $s\|struct $s" | wc -l)
  echo "$s: $c refs"
done
```
Expected: すべて `0 refs`。1つでも参照があればそのシンボルは削除対象から外す。

- [ ] **Step 2: `ErbSpace` 型の削除可否を確認**

Run: `grep -rn "ErbSpace" src/ tests/ examples/ --include=*.rs | grep -v "^src/core/erb.rs"`
Expected: 出力なし。`erb.rs` の外から一切参照されていない。

Run: `grep -rn "hz_to_erb\|erb_to_hz\|erb_bw_hz\|erb_grid" src/ --include=*.rs | grep -v "^src/core/erb.rs" | wc -l`
Expected: 1以上。これらの自由関数は生きているので**残す**。削除するのは `ErbSpace` 型と
そのメソッド（`to_erb` / `to_hz` / `is_empty` / `freqs_hz` / `index_of_freq` / `new` / `len`）のみ。

- [ ] **Step 3: 1シンボルずつ削除し、都度コンパイルする**

まとめて削除しない。1つ削除 → `cargo check --all-targets` → 次、を繰り返す。
削除したシンボルのテストも同時に削除する（テストだけ残すとコンパイルエラーになる）。

- [ ] **Step 4: `dead_code` を検出できる状態を作る**

削除後、隠れた未使用が残っていないか一時的に確認する:
```bash
sed -i 's/^pub mod core;/pub(crate) mod core;/' src/lib.rs
cargo check --all-targets 2>&1 | grep "never used" | head -20
git checkout src/lib.rs
```
Expected: 出力された `never used` を追加の削除候補として記録する。
`git checkout` で `lib.rs` を必ず戻すこと（`pub mod core` は外部公開の意図があるため）。

- [ ] **Step 5: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。テスト総数は削除したテストの分だけ減る。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 17: `a_weighting.rs` を `utils.rs` に統合し、残骸を削除する

**Files:**
- Delete: `src/core/a_weighting.rs`
- Modify: `src/core/utils.rs`
- Modify: `src/core.rs`（モジュール宣言）
- Delete: `attic/core/harmonicity_kernel_templatematch.rs`, `attic/core/nsgt_fft.rs`
- Delete: `examples_tmp/`（空ディレクトリ）

**Interfaces:**
- Consumes: `utils::a_weighting_gain(f32) -> f32`
- Produces: `utils::a_weighting_gain_pow(f32) -> f32`

- [ ] **Step 1: `a_weighting.rs` の中身を確認**

Run: `cat src/core/a_weighting.rs`
Expected: 3行程度の関数1つで、`utils.rs` を import している。

- [ ] **Step 2: 利用者を確認**

Run: `grep -rn "a_weighting_gain_pow\|a_weighting::" src/ --include=*.rs`

- [ ] **Step 3: `utils.rs` に関数を移し、モジュールを削除**

`a_weighting_gain_pow` を `src/core/utils.rs` に移動。`src/core.rs` から
`pub mod a_weighting;` を削除。利用側の `use` を書き換える。

Run: `rm src/core/a_weighting.rs`

- [ ] **Step 4: `attic/` の扱いを決める**

`attic/core/` の2ファイルは git 管理下にあるが、どこからもビルドされない退避コード。
AGENTS.md（Task 6 Step 3 で更新済み）が「An older FFT variant is retired to `attic/`」と
参照しているため、**削除せず残す**。参照が文書に書かれた時点で意図的な保管物になる。

`examples_tmp/` は空で untracked。削除する:
Run: `rmdir examples_tmp`

- [ ] **Step 5: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 18: 過剰な `pub` を絞る

ファイル内でのみ使われている `pub` を private / `pub(crate)` にする。

**Files:**
- Modify: `src/core/consonance_kernel.rs:103`, `src/core/landscape.rs:339,352`,
  `src/core/log2space.rs:129,178,186`, `src/core/nsgt_kernel.rs:340,348`,
  `src/core/mode_pattern.rs:501`, `src/core/psycho_state.rs:105`,
  `src/core/roughness_kernel.rs:106,160`

**Interfaces:**
- Consumes: なし
- Produces: なし（可視性のみ変更）

- [ ] **Step 1: 各シンボルの参照範囲を確認**

```bash
for s in sigmoid01_stable recompute_consonance_field recompute_consonance_density_mass \
         bin_pos_of_freq bandwidth_hz bandwidth_hz_at time_ref_sample_in_frame \
         modal_table_ratios roughness_ref_from_r_pot_scan eval_kernel_delta_erb \
         build_kernel_erbstep; do
  echo "=== $s ==="
  grep -rn "\b$s\b" src/ tests/ examples/ --include=*.rs | grep -v "fn $s"
done
```
Expected: 各シンボルが定義ファイル内からのみ参照されている。他ファイルから参照があるものは
`pub(crate)` に留め、定義ファイル内のみのものは `pub` を外す。

- [ ] **Step 2: 1つずつ可視性を下げてコンパイルする**

`pub fn` → `fn`（同一ファイル内のみ）または `pub(crate) fn`（クレート内他ファイルから使用）。
`NsgtKernelLog2::analyze` は `tests/` から参照される可能性があるため Step 1 の結果に従う。

- [ ] **Step 3: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

---

## Phase 4: 空虚なテストの実効性回復

603 テストが通っているが、通っていることが何も保証していないものがある。

### Task 19: `tests/scripts/*.rhai` のマーカー検証を実効化する

19本すべてが `TEST_OK: <stem>` を1行目で print するため、
`run_script_samples.rs:44` のマーカー照合は「1行目に到達した」以上を検証していない。

**Files:**
- Modify: `tests/scripts/*.rhai`（19本）
- Modify: `tests/run_script_samples.rs`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 現状を確認**

Run: `head -3 tests/scripts/stable_order_same_time.rhai`
Expected: 1行目に `print("TEST_OK: ...")` がある。

Run: `grep -n "TEST_OK" tests/run_script_samples.rs`
Expected: 44行目付近でマーカーを照合している。

- [ ] **Step 2: マーカーをスクリプト末尾に移動する**

19本すべてで `print("TEST_OK: <stem>")` を**ファイル末尾**に移す。
これによりマーカーの存在が「スクリプトが最後まで実行された」ことの証明になる。

```bash
for f in tests/scripts/*.rhai; do
  stem=$(basename "$f" .rhai)
  grep -v "TEST_OK: $stem" "$f" > "$f.tmp"
  echo "" >> "$f.tmp"
  echo "print(\"TEST_OK: $stem\");" >> "$f.tmp"
  mv "$f.tmp" "$f"
done
```

- [ ] **Step 3: 移動が正しく行われたか確認**

Run: `tail -2 tests/scripts/stable_order_same_time.rhai`
Expected: 最終行が `print("TEST_OK: stable_order_same_time");`。

Run: `head -3 tests/scripts/stable_order_same_time.rhai`
Expected: 1行目に `TEST_OK` が**ない**。

- [ ] **Step 4: テストが依然として通ることを確認**

Run: `cargo test --test run_script_samples 2>&1 | tail -10`
Expected: PASS。全19本が最後まで実行されている。

- [ ] **Step 5: 意図的に壊して検証が効くことを確かめる**

一時的に1本のスクリプトの中程に `throw "boom";` を挿入し、テストが FAIL することを確認する。
確認後、必ず元に戻す。

Run: `cargo test --test run_script_samples 2>&1 | grep -E "FAILED|test result"`
Expected: FAIL する。これでマーカーが実効的になったことが証明される。その後 `git checkout` で戻す。

### Task 20: 名前が内容を裏切っているスクリプトを是正する

**Files:**
- Modify または Rename: `tests/scripts/stable_order_same_time.rhai`,
  `tests/scripts/rhythm_timing_validation.rhai`, `tests/scripts/note_polyrhythm_3_2.rhai`,
  `tests/scripts/tag_selector_ops.rhai`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 4本の中身を確認**

Run: `for f in stable_order_same_time rhythm_timing_validation note_polyrhythm_3_2 tag_selector_ops; do echo "=== $f ==="; cat tests/scripts/$f.rhai; done`
Expected: それぞれ6-8行で、名前が主張する検証を行っていない。

- [ ] **Step 2: 各スクリプトを名前どおりの内容にする**

`note_polyrhythm_3_2.rhai` の例（現状は両方 `.count(1)` で 3:2 になっていない）:
```rhai
// 3:2 polyrhythm: two populations whose gate counts form the ratio.
let a = consonance(200.0, 400.0).count(3);
let b = consonance(400.0, 800.0).count(2);
place(sine().rhythm(metric(3)), a);
place(sine().rhythm(metric(2)), b);
wait(4.0);

print("TEST_OK: note_polyrhythm_3_2");
```

正確な API 名は `docs/rhai_book/src/reference/api.md` を参照して確定すること。
Run: `grep -n "metric\|count\|place" docs/rhai_book/src/reference/api.md | head -20`

`tag_selector_ops.rhai` は tag セレクタを実際に使う内容にする。
`stable_order_same_time.rhai` は同時刻イベントの順序が安定であることを、
2回実行して同じ順序になることで示す — これはスクリプト単体では検証できないため、
`tests/` 側の Rust テストに移すか、スクリプト名を実態に合わせて改名する。

- [ ] **Step 3: 名前を実態に合わせる選択肢**

内容を実装するコストが高い場合、名前を `smoke_` 接頭辞に改名して
「これは煙テストであり仕様検証ではない」ことを明示する。
どちらを選んだかを `tests/scripts/README.md`（なければ作成）に1行記録する。

- [ ] **Step 4: テスト**

Run: `cargo test --test run_script_samples 2>&1 | tail -10`
Expected: PASS。

### Task 21: `log2space_scan_invariants.rs` を release でも効くテストにする

12検証すべてが `assert_scan_len`（実体は `debug_assert_eq!`）のため、
`cargo test --release` で全部 no-op になる。

**Files:**
- Modify: `tests/log2space_scan_invariants.rs`

**Interfaces:**
- Consumes: `Log2Space::assert_scan_len`
- Produces: なし

- [ ] **Step 1: `assert_scan_len` の実体を確認**

Run: `sed -n '55,70p' src/core/log2space.rs`
Expected: `debug_assert_eq!` を使っている。

- [ ] **Step 2: release で no-op になることを確認**

Run: `cargo test --release --test log2space_scan_invariants 2>&1 | tail -10`
Expected: PASS するが、実際には何も検証していない。

- [ ] **Step 3: アサーションのないテストにアサーションを足す**

`tests/log2space_scan_invariants.rs:28-44` はアサーションが1つもない。
全ベクタを同一長 `n` で作っているため関数内の assert も発火しない。
異なる長さを渡して panic することを検証する形に書き換える。

- [ ] **Step 4: `#[should_panic]` に `expected` を付ける**

`tests/log2space_scan_invariants.rs:48` の `#[should_panic]` を
`#[should_panic(expected = "scan length")]` にする。
正確な期待文字列は `assert_scan_len` のメッセージから取る:
Run: `grep -n "debug_assert_eq!" -A 3 src/core/log2space.rs`

- [ ] **Step 5: debug と release の両方で実行**

Run: `cargo test --test log2space_scan_invariants 2>&1 | tail -5`
Expected: PASS。

Run: `cargo test --release --test log2space_scan_invariants 2>&1 | tail -5`
Expected: PASS。ただし `debug_assert` 依存のテストは release でスキップされるため、
`#[cfg(debug_assertions)]` を付けて「release では走らない」ことを明示するか、
`assert_scan_len` を `assert_eq!` に格上げするかを選ぶ。

F2 は「debug_assert is acceptable; tests must cover it」と定めているため、
テスト側に `#[cfg(debug_assertions)]` を付けて意図を明示するのが規約に忠実。

### Task 22: 貫通しうるアサーションを締める

**Files:**
- Modify: `tests/habituation_field_assay.rs:114-120`
- Modify: `tests/render_binary.rs:121`
- Modify: `tests/listener_twin_validation_fixture.rs:46,201`
- Modify: `tests/control_rate_dt_invariance.rs:77`
- Modify: `tests/phonation_rate_is_time_based.rs:24`
- Modify: `tests/terrain_predictor_tau_depends_on_rhythm.rs:41-42`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: `habituation_field_assay.rs` に空データガードを足す**

他2テストにある `!s.is_empty()` ガードがこのテストだけ無い。
`assert_eq!(0, 0)` で空虚に合格する経路を塞ぐ:
```rust
    assert!(
        !series_a.is_empty(),
        "habituation series is empty — the report format likely drifted"
    );
```

- [ ] **Step 2: `render_binary.rs:121` を無音検出できるようにする**

現状 `reader.duration() > 0` のみで、全サンプル0の無音WAVでも合格する。
サンプルの絶対値の最大が0より大きいことを検証する:
```rust
    let peak = reader
        .into_samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s.unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(peak > 0, "rendered WAV is digital silence");
```

- [ ] **Step 3: `listener_twin_validation_fixture.rs` の `unwrap_or(0.0)` を厳格化**

JSONキーがドリフトすると「欠落 = 0」で `< 1.0e-6` 系のアサーションが自動的に通る。
`unwrap_or(0.0)` を `expect("field <name> missing from fixture")` に変える。

- [ ] **Step 4: `control_rate_dt_invariance.rs:77` に前進検証を足す**

fine と coarse の一致だけを見ているため `update_articulation` が no-op でも合格する。
状態が実際に変化したことを先に確認する:
```rust
    assert_ne!(
        fine_final, initial_state,
        "articulation did not advance — the invariance check would be vacuous"
    );
```

- [ ] **Step 5: `phonation_rate_is_time_based.rs:24` の許容幅を締める**

rate 1.0Hz × dt=1/12s × 24 gate なので onsets は決定的に 2。
`(1..=3).contains(&onsets)` を `assert_eq!(onsets, 2)` にする。
FAIL した場合は決定的でない要因があるということなので、その原因を先に調べる。

- [ ] **Step 6: `terrain_predictor_tau_depends_on_rhythm.rs` に clamp 域外のケースを足す**

4ケース全部が clamp 境界（`generator_model.rs:28` の 2 と 8）に落ちており、
テスト名が主張する theta/delta 比率依存を一度も通っていない。
clamp されない中間の比率（例: theta/delta = 4）のケースを1つ追加する。

Run: `sed -n '20,40p' src/life/generator_model.rs`
Expected: clamp の範囲を確認し、その内側に落ちる入力値を選ぶ。

- [ ] **Step 7: 全テストを実行**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:|FAILED" | tail -5`
Expected: 0 failed。締めたアサーションで FAIL が出た場合、それは**実バグの発見**なので、
テストを緩めるのではなく実装を直す。

### Task 23: `Community::new()` の非決定シードを塞ぐ

`src/scripting/tests.rs` の `Community::new` 18箇所が `set_seed` を呼んでおらず、
`rand::random::<u64>()` のまま走っている。本番経路は seed をログ出力して replay 可能なので
実害はテスト側のみだが、決定性が呼び出し規律だけに依存している。

**Files:**
- Modify: `src/life/community.rs:322`
- Modify: `src/scripting/tests.rs`（18箇所）
- Modify: `tests/body_envelope_propagates_to_tone_adsr.rs:37`

**Interfaces:**
- Consumes: `Community::new(Timebase) -> Community`
- Produces: 同一シグネチャ（既定 seed を固定値に変更）

- [ ] **Step 1: 現状を確認**

Run: `grep -n "seed: rand::random" src/life/community.rs`
Expected: 322行目。

Run: `grep -c "Community::new(" src/scripting/tests.rs; grep -c "set_seed" src/scripting/tests.rs`
Expected: 18 と 0。

- [ ] **Step 2: 本番経路が必ず上書きすることを確認**

Run: `grep -n "set_seed" src/runtime/mod.rs`
Expected: 946 と 1122。`run_headless` と `run_render` の両方で `scenario.seed` を設定している。

- [ ] **Step 3: 既定を固定値にする**

`src/life/community.rs:322` を置換:
```rust
            // Deterministic default. Production paths always override this via
            // set_seed(scenario.seed); a fixed default keeps tests reproducible
            // instead of silently depending on rand::random().
            seed: 0,
```

- [ ] **Step 4: 既定変更でテストが落ちないか確認**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:|FAILED" | tail -5`
Expected: 0 failed。落ちるテストがあれば、それは非決定シードに依存していた証拠なので、
そのテストに明示的な `set_seed` を足す。

- [ ] **Step 5: `ScriptContext` 側も同様に確認する**

Run: `grep -n "with_seed(random" src/scripting/mod.rs`
Expected: 1166行目。こちらは `ScriptContext::default` の経路。

同じ理由で固定値にするか判断する。ただし `scenario.seed` が「フレッシュな seed」を
意図的に引く仕様（`cli.rs:31` の `--seed` が override）であれば、
`ScriptContext` の `random()` は**仕様どおり**なので変更しない。
Run: `grep -n "fresh scenario seed" src/cli.rs src/bin/render.rs`
Expected: 「Override the fresh scenario seed」とあり、フレッシュ引きが意図的だとわかる。
この場合 Step 5 は変更なしで終える。

- [ ] **Step 6: リント**

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。

### Task 24: `samples_no_legacy_keys.rs` の誤ヒットを直す

`"create("` は `recreate(` に、`"linear("` は `nonlinear(` に誤ヒットする。
同ファイル `:106` の `banned_identifiers` は境界判定しており非対称。

**Files:**
- Modify: `tests/samples_no_legacy_keys.rs:59,84`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 誤ヒットが現実に起きていないことを確認**

Run: `grep -rn "recreate(\|nonlinear(" samples/ docs/ tests/scripts/ 2>/dev/null`
Expected: 出力なし。現状は潜在バグに留まっている。

- [ ] **Step 2: 境界判定を入れる**

`create(` と `linear(` の照合を、直前の文字が識別子文字でないことの確認付きにする。
既存の `banned_identifiers`（`:106`）が使っている境界判定ロジックを再利用する。

Run: `sed -n '100,120p' tests/samples_no_legacy_keys.rs`
Expected: 境界判定の実装が見えるので、同じ方法を `create(` / `linear(` にも適用する。

- [ ] **Step 3: 誤ヒットしないことをテストで固定する**

`tests/samples_no_legacy_keys.rs` にユニットテストを追加:
```rust
#[test]
fn boundary_matching_does_not_flag_superstrings() {
    assert!(!is_legacy_hit("let x = recreate(1);", "create("));
    assert!(!is_legacy_hit("let y = nonlinear(2);", "linear("));
    assert!(is_legacy_hit("let z = create(3);", "create("));
    assert!(is_legacy_hit("let w = linear(4);", "linear("));
}
```
`is_legacy_hit` に相当するヘルパが無い場合は、境界判定を関数に切り出してから書く。

- [ ] **Step 4: テスト**

Run: `cargo test --test samples_no_legacy_keys 2>&1 | tail -10`
Expected: PASS。

### Task 25: `sample_seed_policy.rs` のハードコード件数をやめる

`assert_eq!(sample_count, 12)` はサンプルを追加するたびに落ちる。

**Files:**
- Modify: `tests/sample_seed_policy.rs:28`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 現状を確認**

Run: `sed -n '20,35p' tests/sample_seed_policy.rs`
Expected: `assert_eq!(sample_count, 12)` がある。

- [ ] **Step 2: 件数固定をやめ、ポリシー自体を検証する**

このテストの目的は「トップレベルは無シード、research は固定シード」の強制であり、
件数ではない。件数アサーションを下限チェックに変える:
```rust
    assert!(
        sample_count > 0,
        "no top-level samples found — the glob or the directory layout changed"
    );
```

- [ ] **Step 3: テスト**

Run: `cargo test --test sample_seed_policy 2>&1 | tail -10`
Expected: PASS。

---

### Task 26: 残りの小さな乖離を回収する

**Files:**
- Modify: `src/ui/viewdata.rs:54`（`PlaybackState` に `Copy`）
- Modify: `src/runtime/mod.rs:1304,1726`（不要な `.clone()` の除去）
- Modify: `AGENTS.md:87-89`（Testing Policy に第3のパターンを追記）

**Interfaces:**
- Consumes: なし
- Produces: `PlaybackState: Copy`

- [ ] **Step 1: `PlaybackState` が fieldless であることを確認**

Run: `sed -n '54,60p' src/ui/viewdata.rs`
Expected: `NotStarted` / `Playing` / `Finished` の3バリアント、フィールドなし。

- [ ] **Step 2: `Copy` を派生させる**

`src/ui/viewdata.rs:54` を置換:
```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
```

- [ ] **Step 3: 不要になった `.clone()` を外す**

Run: `grep -n "playback_state.clone()" src/runtime/mod.rs`
Expected: 1304 と 1726。両方の `.clone()` を削除する。

- [ ] **Step 4: Testing Policy を実態に合わせる**

`src/life/tests.rs`（1049行/20件）と `src/scripting/tests.rs`（2501行/88件）は
「同ファイル内 `#[cfg(test)]`」でも `tests/` でもない第3のパターンで、
規約が想定していない形態にテストの大半（108件）が集中している。
方針 A（規約を実態に合わせる）に従い、この形態を規約に追記する。

`AGENTS.md:89` の後に追記:
```markdown
- **Crate-internal test modules** (`src/<module>/tests.rs`, declared with
  `#[cfg(test)] mod tests;`) are used where a suite is too large to inline but needs
  access to private APIs across a module tree — currently `src/life/tests.rs` and
  `src/scripting/tests.rs`. Prefer one of the two forms above; reach for this only when
  the suite genuinely needs crate-internal visibility.
```

- [ ] **Step 5: テストとリント**

Run: `cargo test --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。

Run: `cargo clippy --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 警告なし。`clippy::clone_on_copy` が出たら消し漏れ。

---

## 実行結果 (2026-07-25)

**完了**: Task 1-9, 11, 14-19, 21, 22, 24, 26

**見送り（本文に理由を記録）**: Task 10, 12, 13

**実行中に判断を変更したもの**:

- **Task 20（名前が内容を裏切るスクリプト）— 未着手**。Task 19 のマーカー末尾化を検証した際、
  Rhai の `throw` は非ゼロ終了するため `run_script_samples.rs:33` の status チェックが既に
  捕捉することが分かった。マーカー位置の追加検出力は限定的（「最後まで到達した」表明の価値のみ）。
  4本のスクリプトを名前どおりの内容にするか `smoke_` へ改名するかは、Rhai API の
  意味を伴う判断なので別途扱う。
- **Task 23（`Community::new` の非決定シード）— 未着手**。本番経路は seed をログ出力して
  `--seed` で replay 可能（`runtime/mod.rs:912-913`）であり、既定値を `0` に変えると
  `scripting/tests.rs` の18テストが暗黙に依存している乱数列が変わる。影響を測ってから行う。
- **Task 25（`sample_seed_policy.rs` の件数固定）— 変更しない**。`assert_eq!(sample_count, 12)`
  は「エチュードは 01-12 の12曲」という設計の表明であり（`samples/README.md` の
  "Twelve small demonstrations" と一致）、13本目を足したときに気付くための意図的なガード。
- **Task 18 の副産物**: `pub` を外したことで `Log2Space::bandwidth_hz` /
  `bandwidth_hz_at` が本番未到達（テスト専用）だと判明し、関連テスト3本とともに削除した。
  NSGT は窓から実測した `enbw_hz` を使っており、この解析式とは別物で重複ではない。
  `pub` が `dead_code` lint を隠していた実例。

**最終状態**: debug 603 passed / release 601 passed / 0 failed、
`cargo clippy --all-targets -- -D warnings` 警告0、`cargo fmt --all --check` 差分なし。

## 外部レビュー (codex) とその対応 (2026-07-25)

2巡実施。1巡目8件、2巡目6件の指摘。すべて事実確認のうえ対応した。

### 発見された実バグ (計画外)

**解析スレッド死亡時に worker がハングする** — `src/runtime/mod.rs`

`merge_latest_analysis_results` / `merge_latest_listener_analysis_results` は
`while let Ok(..) = rx.try_recv()` で受信しており、チャネルの「空」と「切断」を
区別していなかった。解析スレッドが最初の結果を送る前に panic すると:

- `last_analysis_frame` は `None` のまま
- `analysis_ok(frame_idx > 0, None, _)` は常に `false`（`:151-159`）
- 待機ループの脱出条件は `analysis_ok` か `exiting` のみで、切断では誰も `exiting` を立てない
- → 200µs sleep で無限ループ。`run_headless` の join に到達せず、プロセスがハングする

実証: 解析スレッドに panic を注入したところ、修正前は `exit=124`（タイムアウト）、
修正後は `exit=1`。

対応: 両関数の受信を `match` に変えて `TryRecvError::Disconnected` を区別し、
戻り値で切断を返す（引数追加だと8個になり `too_many_arguments` に触れるため戻り値へ）。
呼び出し側は切断を検出したら stderr に出力して `exiting` を立て、ループを抜ける。

**`run_headless` がスレッドの panic を握りつぶす** — `src/runtime/mod.rs:1010`

`let _ = handle.join()` で3スレッドすべての join 結果を捨てていた。worker が panic しても
`exit=0` で終了するため、`tests/run_script_samples.rs` は死んだ実行を成功と判定していた。
`run_render` 側には既に `join_render_thread` による panic 伝播があり、headless だけが
抜けていた。同関数を `join_thread` にリネームして共用し、panic 時は stderr 出力のうえ
`exit(1)`。worker への panic 注入で `exit=1` を実証。

### ドキュメントの誤り (今回の修正で新たに入れたもの)

- worker_loop の割当記述 — 行番号誤り（1599-1600 → 1610-1611）、`:1412` の地形 Arc の
  欠落、網羅的でない旨の明示。「予算であって絶対条件ではない」と書き直した
- Rhai Spawn API — `Placement` の定義位置（`src/scripting/mod.rs:941`）、
  `consonance(root)` 1引数形式、`at(freq)`、`.spacing(erb)` の欠落を補った
- Consonance Variants — variant 2 だけ直して 4 を直し忘れていた。また「実消費は `_eff`」は
  behavior/listener/spawn には当てはまるが UI には当てはまらない（`src/ui/windows.rs:708,830`
  は base を読む）。用途を behavior 系と UI/diagnostics に分けて記述
- `src/core/landscape.rs:238` — 削除した `evaluate_pitch_level_log2` を指す doc コメント

### テストの補強

- `tests/log2space_scan_invariants.rs` — 「every scan」と書きながら `_eff` 系3本と
  `perc_habituation_state_scan` を検査していなかった。12→16本に
- `tests/control_rate_dt_invariance.rs` — 前進検証に等価性の `tol`(1e-2) を流用していた。
  正しく小さく前進する実装を落とすため `moved_eps`(1e-6) に分離
- `src/life/sound/tone.rs:155` — 「never reallocates」は嘘（17件目で再確保）。
  「予算であって上限ではない」に修正

### 残存ギャップ (未対応・記録のみ)

`tests/scripts/*.rhai` の `TEST_OK` マーカーは、末尾へ移してもシナリオの完走を保証しない。
Rhai は `ScriptHost::load_script` で worker 起動前に同期評価されるため、マーカーは
「スクリプト評価が最後まで到達した」ことしか示さない。worker panic は上記の修正で
`exit=1` になったが、`stop_flag` による早期終了は正常 join・`exit=0` のままなので、
Finish 到達前でもマーカーと成功 status が揃いうる。厳密にやるなら worker から
終了理由（ScenarioFinished / Stopped）を返し、テストで ScenarioFinished のみ成功とする。

---

## 本計画のスコープ外

レビューで挙がったが、本計画では扱わないと決めたもの。理由を残す。

- **`#[allow(clippy::too_many_arguments)]` 23箇所の引数構造体化** — 機械的だが広範な変更で、
  今回是正する実バグより優先度が低い。AGENTS.md:81 の YAGNI に照らし、
  引数が実際に増え続けている関数が現れた時点で個別に対応する。
- **`src/ui/windows.rs` / `src/ui/plots.rs` のテスト追加** — 描画非依存の純粋関数
  （`format_time`, `split_widths`, log2軸座標変換）は確かに未検証だが、
  UI 層のリグレッションは目視で捕捉されている。Phase 4 で締めた
  「通っているのに何も保証しないテスト」の是正を優先する。
- **`src/runtime/mod.rs`（1960行 / `#[test]` 3件）のカバレッジ拡充** — 最大の穴だが、
  `pub(crate)` のため統合テストから到達できず、拡充には可視性設計の変更が要る。
  別計画とする。
- **内部型 `Community` と Rhai 公開名 `Population` の二重化** — `engine.rs:9` の
  `register_type_with_name::<PopulationHandle>("Population")` は、
  Population を canonical ontology name として英語で保つ方針（`docs/rhai_book_ja/README.md`）
  と整合しており、意図的な対応付けと判断した。Phase 1 Task 3 で
  AGENTS.md の内部ファイル参照のみを是正する。

---

## 完了時の検証

全フェーズ完了後に一度だけ実行する。

- [ ] **Step 1: 全テスト**

```bash
set -o pipefail
( RUST_BACKTRACE=1 cargo test --all-targets -- --nocapture ) 2>&1 | tee test_report.txt
echo "cargo test exit=$? @ $(date -Iseconds)" > test_status.txt
```
Expected: 0 failed。`test_status.txt` に正しい終了コードが記録される
（Phase 1 Task 6 で修正した手順そのものの動作確認になる）。

- [ ] **Step 2: リント**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: 警告なし。

- [ ] **Step 3: フォーマット**

Run: `cargo fmt --all --check`
Expected: 差分なし。差分があれば `cargo fmt --all` を実行する。

- [ ] **Step 4: ドキュメント乖離の再走査**

```bash
grep -o '`src/[a-z_/]*\.rs`' AGENTS.md | tr -d '`' | sort -u | while read p; do
  [ -e "$p" ] || echo "MISSING: $p"
done
grep -n "debug_assertions\|consonance_density_pmf(\|population.rs\|err_" AGENTS.md
```
Expected: どちらも出力なし。

- [ ] **Step 5: release ビルドの健全性**

Run: `cargo test --release --all-targets 2>&1 | grep -E "^test result:" | tail -3`
Expected: 0 failed。debug_assert 依存のテストが release で無意味になっていないこと
（Task 21 の成果確認）。
