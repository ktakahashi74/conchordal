# Habituation Field (Assay Phase) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature-gated (default-OFF) landscape-level habituation mechanism plus a deterministic research assay that proves — or refutes — production-loop closure, before any permanent core integration or composer API.

**Architecture:** A worker-owned `HabituationField` holds a persistent per-bin state `h ∈ [0,1]`, advanced once per analysis hop by an asymmetric leaky integrator driven by root-projected perceived-consonance activity. Each hop it writes eroded "effective" views (`*_eff`) onto the owned `Landscape`; all terrain consumers read the effective views. When disabled, `h ≡ 0` and every effective view equals its raw kernel value bit-for-bit, so existing sample output is unchanged. A seeded assay script plus an integration test drive the real audio→analysis→terrain→movement chain and assert the closure criteria.

**Tech Stack:** Rust; existing engine modules `src/core/landscape.rs`, `src/core/harmonicity_kernel.rs`, `src/config.rs`, `src/runtime/mod.rs`, `src/life/report.rs`, `src/life/pitch_core.rs`; Rhai scenario scripts under `samples/research/`; integration tests under `tests/`.

**Spec:** `docs/superpowers/specs/2026-07-17-habituation-field-design.md`

## Global Constraints

- Every task that modifies `src/` ends with the mandatory end-of-task procedure: `set -o pipefail; ( RUST_BACKTRACE=1 cargo test -- --nocapture ) 2>&1 | tee test_report.txt` then `echo "cargo test exit=$? @ $(date -Iseconds)" > test_status.txt`.
- Before any commit: `cargo fmt --all` and `cargo clippy -- -D warnings` must pass.
- All code comments in English only.
- No allocations in the audio `worker_loop` hot path: preallocate buffers (`Vec::with_capacity` / reuse). The habituation projection allocates only on the enabled path and only in the offline/assay flow; keep the `h`/drive buffers preallocated in `HabituationField`.
- Alpha no-compat policy: no aliases, no migration shims.
- Default OFF: `HabituationParams::default().enabled == false`. The 12 existing samples (`samples/01_*.rhai` .. `samples/12_*.rhai`) must produce byte-identical behavior when habituation is absent from config.
- Determinism: habituation state advances only inside the deterministic analysis path (`reporter.is_some() && !args.play`), driven by `hop_duration.as_secs_f32()`, never wall-clock. Advance exactly once per unique analysis hop.
- Log2Space invariants (F1/F2): every new `_scan` field has `len() == space.n_bins()`, is resized in `Landscape::resize_to_space`, and is covered by `Landscape::assert_scan_lengths`.
- Naming: persistent state scan is `perc_habituation_state_scan` (`perc_` = audio-derived per Axis A; `_scan` per Log2Space convention). Effective views are `consonance_field_score_eff`, `consonance_field_level_eff`, `consonance_density_mass_eff`. Raw kernel field names keep their CLAUDE.md meaning. The raw-vs-effective telemetry difference is NOT an `err_*` quantity (`err_*` is reserved for `perc − pred`).
- No new top-level sample: `tests/sample_seed_policy.rs` asserts exactly 12 top-level samples and requires every `samples/research/*.rhai` to contain a `seed(` line. The assay goes under `samples/research/` and MUST be seeded.
- Only the feature-gated experimental path and the assay are authorized by the spec. Do NOT add a `satiation()`/`recovery()` Rhai builder, do NOT edit the technote/ledger, do NOT promote to default-on. Those are the post-assay Phase 2 plan.

**Erosion math (single source of truth, used everywhere):**
- `erode_score(raw, h, theta) = theta + (raw - theta) * (1 - h)` — relax a signed score toward its level-neutral baseline `theta`.
- `erode_mass(raw, h) = raw * (1 - h)` — non-negative mass, plain multiplicative.
- `transfer(drive_raw, ref_drive) = d / (d + ref_drive)` where `d = max(0, drive_raw)` — maps `[0,∞) → [0,1)`, zero at zero (so recovery reaches raw terrain).
- Time-constant derivation: `LN_STALE = ln(10) ≈ 2.302585`; `tau_e = satiation_sec / LN_STALE`; `tau_r = recovery_sec / LN_STALE`. Contract: under constant drive `d=1`, `h` rises `0→0.9` in `satiation_sec`; under `d=0`, `h` falls `1→0.1` in `recovery_sec`.

**Assay-tuned knobs (chosen initial values; the assay may retune these, NOT the code shape):** `ref_drive` default `0.25`; `satiation_sec` default `5.0`; `recovery_sec` default `8.0`.

---

## File Structure

- Create `src/core/habituation.rs` — pure `HabituationField` state + advance + erosion free functions. One responsibility: the habituation state machine. Unit-tested in-file.
- Modify `src/config.rs` — add `HabituationConfig` leaf + nest under `PsychoAcousticsConfig`.
- Modify `src/core/landscape.rs` — add `HabituationParams` to `LandscapeParams`; add effective-view + state-scan fields to `Landscape`; add `apply_habituation`; switch `evaluate_pitch_*` chokepoints to effective views.
- Modify the 5 `LandscapeParams { ... }` construction sites — `src/runtime/mod.rs` (×2: ~531, ~1821), `src/core/analysis_worker.rs` (~63), `src/core/landscape_spectral.rs` (~113), `src/core/psycho_state.rs` (~180).
- Modify `src/runtime/mod.rs` — own two `HabituationField` in `worker_loop`; advance + apply once per hop (ecology + listener); re-apply (no advance) in `apply_pending_landscape_update`.
- Modify `src/life/pitch_core.rs` — erode the exact-LOO sampled score via the landscape state scan.
- Modify `src/life/report.rs` — add `ReportRecord::Habituation` + `write_habituation`.
- Create `samples/research/habituation_field_assay.rhai` — seeded shared-terrain assay.
- Create `tests/habituation_field_assay.rs` — deterministic closure test (config-enabled shell-out).
- Modify `mod.rs` in `src/core/` — register `pub mod habituation;`.

---

## Phase 0 — Config gate (default OFF)

### Task 0.1: `HabituationConfig` leaf and nesting

**Files:**
- Modify: `src/config.rs`
- Test: inline `#[cfg(test)]` in `src/config.rs` (follows `parse consonance density keys` at ~`config.rs:563`)

**Interfaces:**
- Produces: `HabituationConfig { enabled: bool, satiation_sec: f32, recovery_sec: f32, ref_drive: f32 }` with `Default` (enabled=false, satiation=5.0, recovery=8.0, ref_drive=0.25); `PsychoAcousticsConfig.habituation: HabituationConfig`.

- [ ] **Step 1: Write the failing test**

Add near the other config tests in `src/config.rs`:

```rust
#[test]
fn habituation_defaults_off_and_parses() {
    let cfg = AppConfig::default();
    assert!(!cfg.psychoacoustics.habituation.enabled);
    assert_eq!(cfg.psychoacoustics.habituation.satiation_sec, 5.0);
    assert_eq!(cfg.psychoacoustics.habituation.recovery_sec, 8.0);
    assert_eq!(cfg.psychoacoustics.habituation.ref_drive, 0.25);

    let text = "\
[psychoacoustics.habituation]
enabled = true
satiation_sec = 3.0
recovery_sec = 6.0
ref_drive = 0.4
";
    let parsed: AppConfig = toml::from_str(text).expect("parse habituation keys");
    assert!(parsed.psychoacoustics.habituation.enabled);
    assert_eq!(parsed.psychoacoustics.habituation.satiation_sec, 3.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib habituation_defaults_off_and_parses 2>&1 | tail -20`
Expected: FAIL — `no field 'habituation' on PsychoAcousticsConfig`.

- [ ] **Step 3: Add the config struct and nest it**

Add the leaf (imitating `ConsonanceDensityConfig` at `config.rs:157`):

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HabituationConfig {
    #[serde(default = "HabituationConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "HabituationConfig::default_satiation_sec")]
    pub satiation_sec: f32,
    #[serde(default = "HabituationConfig::default_recovery_sec")]
    pub recovery_sec: f32,
    #[serde(default = "HabituationConfig::default_ref_drive")]
    pub ref_drive: f32,
}

impl HabituationConfig {
    fn default_enabled() -> bool { false }
    fn default_satiation_sec() -> f32 { 5.0 }
    fn default_recovery_sec() -> f32 { 8.0 }
    fn default_ref_drive() -> f32 { 0.25 }
}

impl Default for HabituationConfig {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            satiation_sec: Self::default_satiation_sec(),
            recovery_sec: Self::default_recovery_sec(),
            ref_drive: Self::default_ref_drive(),
        }
    }
}
```

In `struct PsychoAcousticsConfig` (near `config.rs:192`) add the field:

```rust
    #[serde(default)]
    pub habituation: HabituationConfig,
```

In `impl Default for PsychoAcousticsConfig` (near `config.rs:209`) add:

```rust
            habituation: HabituationConfig::default(),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib habituation_defaults_off_and_parses 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Confirm config round-trip still passes**

Run: `cargo test --test config_restore 2>&1 | tail -20`
Expected: PASS (no keys removed; only additive).

- [ ] **Step 6: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add src/config.rs
git commit -m "feat(config): add opt-in habituation config block (default off)"
```

---

## Phase 1 — `HabituationField` pure state machine

### Task 1.1: The habituation module

**Files:**
- Create: `src/core/habituation.rs`
- Modify: `src/core/mod.rs` (add `pub mod habituation;`)
- Test: inline `#[cfg(test)]` in `src/core/habituation.rs`

**Interfaces:**
- Produces:
  - `pub fn erode_score(raw: f32, h: f32, theta: f32) -> f32`
  - `pub fn erode_mass(raw: f32, h: f32) -> f32`
  - `pub struct HabituationParams { pub enabled: bool, pub satiation_sec: f32, pub recovery_sec: f32, pub ref_drive: f32 }` with `Default` (disabled).
  - `pub struct HabituationField` with:
    - `pub fn new(params: &HabituationParams, theta: f32, n_bins: usize) -> Self`
    - `pub fn ensure_len(&mut self, n_bins: usize)`
    - `pub fn is_enabled(&self) -> bool`
    - `pub fn theta(&self) -> f32`
    - `pub fn state(&self) -> &[f32]`
    - `pub fn advance_from_parts(&mut self, level: &[f32], proj: &[f32], dt: f32)`

- [ ] **Step 1: Register the module**

In `src/core/mod.rs`, add alongside the other `pub mod` lines:

```rust
pub mod habituation;
```

- [ ] **Step 2: Write the failing tests**

Create `src/core/habituation.rs` with only the tests first:

```rust
//! Landscape-level habituation: a per-bin state that erodes the consonance
//! evaluation terrain under sustained perceived-consonance activity and
//! recovers after release. See
//! docs/superpowers/specs/2026-07-17-habituation-field-design.md.

#[cfg(test)]
mod tests {
    use super::*;

    fn full_drive_params() -> HabituationParams {
        HabituationParams { enabled: true, satiation_sec: 5.0, recovery_sec: 8.0, ref_drive: 0.25 }
    }

    #[test]
    fn disabled_is_identity() {
        let f = HabituationField::new(&HabituationParams::default(), 0.0, 4);
        assert!(!f.is_enabled());
        assert_eq!(erode_score(0.7, 0.0, 0.0), 0.7);
        assert_eq!(erode_mass(0.7, 0.0), 0.7);
    }

    #[test]
    fn erode_relaxes_score_toward_theta() {
        // full erosion drives score to theta, not zero
        assert!((erode_score(0.9, 1.0, 0.2) - 0.2).abs() < 1e-6);
        // half erosion is halfway between raw and theta
        assert!((erode_score(0.9, 0.5, 0.2) - 0.55).abs() < 1e-6);
        // a dissonant (below-theta) score is NOT pushed below theta
        assert!(erode_score(-0.5, 0.5, 0.0) > -0.5);
    }

    #[test]
    fn satiation_reaches_0_9_in_satiation_sec_under_full_drive() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 1);
        // proj=1, level=1 -> drive_raw=1 -> transfer(1)=1/(1+0.25)=0.8; use large proj to saturate
        let level = [1.0f32];
        let proj = [1000.0f32]; // transfer(1000)≈1.0
        let dt = 0.01;
        let mut t = 0.0;
        while f.state()[0] < 0.9 { f.advance_from_parts(&level, &proj, dt); t += dt; }
        assert!((t - 5.0).abs() < 0.15, "reached 0.9 at t={t}");
    }

    #[test]
    fn recovery_falls_to_0_1_in_recovery_sec_at_zero_drive() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 1);
        // saturate first
        for _ in 0..2000 { f.advance_from_parts(&[1.0], &[1000.0], 0.01); }
        let start = f.state()[0];
        assert!(start > 0.99);
        let dt = 0.01;
        let mut t = 0.0;
        while f.state()[0] > 0.1 { f.advance_from_parts(&[0.0], &[0.0], dt); t += dt; }
        assert!((t - 8.0).abs() < 0.2, "fell to 0.1 at t={t}");
    }

    #[test]
    fn state_stays_bounded() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 3);
        for _ in 0..10_000 { f.advance_from_parts(&[1.0, 0.0, 0.5], &[1e6, 1e6, 1e6], 0.05); }
        for &h in f.state() { assert!((0.0..=1.0).contains(&h), "h={h}"); }
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test --lib habituation:: 2>&1 | tail -20`
Expected: FAIL — `HabituationField` / `erode_score` not found.

- [ ] **Step 4: Implement the module**

Prepend above the `#[cfg(test)]` block in `src/core/habituation.rs`:

```rust
const LN_STALE: f32 = 2.302_585_1; // ln(10): "stale" = 90% eroded

/// Relax a signed score toward its level-neutral baseline `theta`.
#[inline]
pub fn erode_score(raw: f32, h: f32, theta: f32) -> f32 {
    let h = h.clamp(0.0, 1.0);
    theta + (raw - theta) * (1.0 - h)
}

/// Erode a non-negative mass multiplicatively.
#[inline]
pub fn erode_mass(raw: f32, h: f32) -> f32 {
    raw * (1.0 - h.clamp(0.0, 1.0))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HabituationParams {
    pub enabled: bool,
    pub satiation_sec: f32,
    pub recovery_sec: f32,
    pub ref_drive: f32,
}

impl Default for HabituationParams {
    fn default() -> Self {
        Self { enabled: false, satiation_sec: 5.0, recovery_sec: 8.0, ref_drive: 0.25 }
    }
}

#[derive(Clone, Debug)]
pub struct HabituationField {
    enabled: bool,
    tau_e: f32,
    tau_r: f32,
    ref_drive: f32,
    theta: f32,
    h: Vec<f32>,
    drive_scratch: Vec<f32>,
}

impl HabituationField {
    pub fn new(params: &HabituationParams, theta: f32, n_bins: usize) -> Self {
        Self {
            enabled: params.enabled,
            tau_e: (params.satiation_sec.max(1e-3)) / LN_STALE,
            tau_r: (params.recovery_sec.max(1e-3)) / LN_STALE,
            ref_drive: params.ref_drive.max(1e-6),
            theta,
            h: vec![0.0; n_bins],
            drive_scratch: vec![0.0; n_bins],
        }
    }

    pub fn ensure_len(&mut self, n_bins: usize) {
        if self.h.len() != n_bins {
            self.h.resize(n_bins, 0.0);
            self.drive_scratch.resize(n_bins, 0.0);
        }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool { self.enabled }

    #[inline]
    pub fn theta(&self) -> f32 { self.theta }

    #[inline]
    pub fn state(&self) -> &[f32] { &self.h }

    #[inline]
    fn transfer(&self, drive_raw: f32) -> f32 {
        let d = drive_raw.max(0.0);
        d / (d + self.ref_drive)
    }

    /// Advance `h` one step. `drive_raw[i] = level[i] * proj[i]` (perceived-
    /// consonance activity in root coordinate), asymmetric relaxation:
    /// rising uses tau_e (satiation), falling uses tau_r (recovery).
    pub fn advance_from_parts(&mut self, level: &[f32], proj: &[f32], dt: f32) {
        if !self.enabled || self.h.is_empty() {
            return;
        }
        let dt = dt.max(0.0);
        let a_e = (-dt / self.tau_e).exp();
        let a_r = (-dt / self.tau_r).exp();
        for i in 0..self.h.len() {
            let lvl = level.get(i).copied().unwrap_or(0.0);
            let pj = proj.get(i).copied().unwrap_or(0.0);
            let d = self.transfer(lvl * pj).clamp(0.0, 1.0);
            let a = if d > self.h[i] { a_e } else { a_r };
            self.h[i] = (a * self.h[i] + (1.0 - a) * d).clamp(0.0, 1.0);
        }
    }
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test --lib habituation:: 2>&1 | tail -20`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add src/core/mod.rs src/core/habituation.rs
git commit -m "feat(core): add HabituationField state machine (pure, unit-tested)"
```

---

## Phase 2 — Thread `HabituationParams` into `LandscapeParams`

### Task 2.1: Add the params field and update all construction sites

**Files:**
- Modify: `src/core/landscape.rs` (add field + import)
- Modify: `src/runtime/mod.rs` (2 sites: ~531 production, ~1821)
- Modify: `src/core/analysis_worker.rs` (~63)
- Modify: `src/core/landscape_spectral.rs` (~113)
- Modify: `src/core/psycho_state.rs` (~180)
- Test: inline test in `src/core/landscape.rs`

**Interfaces:**
- Consumes: `HabituationParams` (Task 1.1).
- Produces: `LandscapeParams.habituation: HabituationParams`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` in `src/core/landscape.rs`:

```rust
#[test]
fn landscape_params_carries_habituation() {
    // Construct via the production config path is covered elsewhere; here just
    // confirm the field exists and defaults are disabled.
    let p = crate::core::habituation::HabituationParams::default();
    assert!(!p.enabled);
}
```

- [ ] **Step 2: Add the field**

In `src/core/landscape.rs`, in `struct LandscapeParams` (after `consonance_density_roughness_gain`, ~line 19):

```rust
    pub habituation: crate::core::habituation::HabituationParams,
```

- [ ] **Step 3: Update the production construction site**

In `src/runtime/mod.rs` at the `LandscapeParams { ... }` around line 531, add (after `consonance_density_roughness_gain: ...`):

```rust
        habituation: crate::core::habituation::HabituationParams {
            enabled: config.psychoacoustics.habituation.enabled,
            satiation_sec: config.psychoacoustics.habituation.satiation_sec,
            recovery_sec: config.psychoacoustics.habituation.recovery_sec,
            ref_drive: config.psychoacoustics.habituation.ref_drive,
        },
```

- [ ] **Step 4: Update the other four construction sites with the default**

At each of `src/runtime/mod.rs` (~1821), `src/core/analysis_worker.rs` (~63), `src/core/landscape_spectral.rs` (~113), `src/core/psycho_state.rs` (~180), add to the `LandscapeParams { ... }` literal:

```rust
        habituation: crate::core::habituation::HabituationParams::default(),
```

Find every site with: `grep -rn "LandscapeParams {" src/`

- [ ] **Step 5: Run to verify it builds and passes**

Run: `cargo test --lib landscape_params_carries_habituation 2>&1 | tail -20`
Expected: PASS. Also run `cargo check --all-targets 2>&1 | tail -20` — expected: no "missing field habituation" errors.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add src/core/landscape.rs src/runtime/mod.rs src/core/analysis_worker.rs src/core/landscape_spectral.rs src/core/psycho_state.rs
git commit -m "feat(core): thread HabituationParams through LandscapeParams"
```

---

## Phase 3 — Effective views on `Landscape` and per-hop wiring

### Task 3.1: Effective-view fields + `apply_habituation`

**Files:**
- Modify: `src/core/landscape.rs`
- Test: inline test in `src/core/landscape.rs`

**Interfaces:**
- Consumes: `erode_score`, `erode_mass` (Task 1.1); `ConsonanceRepresentationParams` (existing, has `theta` and `level(score)`).
- Produces: `Landscape` fields `consonance_field_score_eff`, `consonance_field_level_eff`, `consonance_density_mass_eff`, `perc_habituation_state_scan`, `consonance_theta`; method `pub fn apply_habituation(&mut self, h: &[f32], theta: f32, repr: &ConsonanceRepresentationParams)`; `evaluate_pitch_score*` read the effective score.

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)] mod tests` in `src/core/landscape.rs`:

```rust
#[test]
fn apply_habituation_zero_state_is_identity() {
    let space = Log2Space::new(55.0, 1760.0, 12);
    let mut ls = Landscape::new(space);
    let n = ls.space.n_bins();
    for i in 0..n { ls.consonance_field_score[i] = 0.8; ls.consonance_density_mass[i] = 0.5; }
    let repr = ConsonanceRepresentationParams { beta: 2.0, theta: 0.0 };
    let zeros = vec![0.0f32; n];
    ls.apply_habituation(&zeros, repr.theta, &repr);
    for i in 0..n {
        assert!((ls.consonance_field_score_eff[i] - 0.8).abs() < 1e-6);
        assert!((ls.consonance_density_mass_eff[i] - 0.5).abs() < 1e-6);
    }
}

#[test]
fn apply_habituation_full_state_relaxes_to_theta_and_zero_mass() {
    let space = Log2Space::new(55.0, 1760.0, 12);
    let mut ls = Landscape::new(space);
    let n = ls.space.n_bins();
    for i in 0..n { ls.consonance_field_score[i] = 0.8; ls.consonance_density_mass[i] = 0.5; }
    let repr = ConsonanceRepresentationParams { beta: 2.0, theta: 0.1 };
    let ones = vec![1.0f32; n];
    ls.apply_habituation(&ones, repr.theta, &repr);
    for i in 0..n {
        assert!((ls.consonance_field_score_eff[i] - 0.1).abs() < 1e-6);
        assert!(ls.consonance_density_mass_eff[i].abs() < 1e-6);
        assert!((ls.perc_habituation_state_scan[i] - 1.0).abs() < 1e-6);
    }
    // movement chokepoint reads the eroded score
    let mid = ls.space.centers_hz[n / 2];
    assert!((ls.evaluate_pitch_score(mid) - 0.1).abs() < 1e-3);
}
```

- [ ] **Step 2: Add fields and initialize them**

In `struct Landscape` (after `consonance_field_energy`, ~line 96), add:

```rust
    /// Effective (habituation-eroded) score read by movement/prediction.
    pub consonance_field_score_eff: Vec<f32>,
    /// Effective level derived from the eroded score.
    pub consonance_field_level_eff: Vec<f32>,
    /// Effective (eroded) density mass, source of the spawn PMF.
    pub consonance_density_mass_eff: Vec<f32>,
    /// perc_habituation_state_scan in [0,1] (copy of the worker-owned state).
    pub perc_habituation_state_scan: Vec<f32>,
    /// Level-neutral baseline used by score erosion (mirrors representation theta).
    pub consonance_theta: f32,
```

In `Landscape::new` (after `consonance_field_energy: vec![0.0; n],`, ~line 133):

```rust
            consonance_field_score_eff: vec![0.0; n],
            consonance_field_level_eff: vec![0.0; n],
            consonance_density_mass_eff: vec![0.0; n],
            perc_habituation_state_scan: vec![0.0; n],
            consonance_theta: 0.0,
```

In `Landscape::resize_to_space` (after `consonance_field_energy.resize(n, 0.0);`, ~line 163):

```rust
        self.consonance_field_score_eff.resize(n, 0.0);
        self.consonance_field_level_eff.resize(n, 0.0);
        self.consonance_density_mass_eff.resize(n, 0.0);
        self.perc_habituation_state_scan.resize(n, 0.0);
```

In `assert_scan_lengths` (after the `consonance_field_energy` assertion, ~line 188):

```rust
        self.space
            .assert_scan_len_named(&self.consonance_field_score_eff, "consonance_field_score_eff");
        self.space
            .assert_scan_len_named(&self.consonance_field_level_eff, "consonance_field_level_eff");
        self.space
            .assert_scan_len_named(&self.consonance_density_mass_eff, "consonance_density_mass_eff");
        self.space.assert_scan_len_named(
            &self.perc_habituation_state_scan,
            "perc_habituation_state_scan",
        );
```

- [ ] **Step 3: Implement `apply_habituation` and make recompute seed the eff views**

At the end of `recompute_consonance` (after the density normalization, ~line 264), seed the eff views so they are always valid even before the first worker apply:

```rust
        // Seed effective views to raw (identity) until apply_habituation runs.
        self.consonance_field_score_eff
            .copy_from_slice(&self.consonance_field_score);
        self.consonance_field_level_eff
            .copy_from_slice(&self.consonance_field_level);
        self.consonance_density_mass_eff
            .copy_from_slice(&self.consonance_density_mass);
```

Add the method inside `impl Landscape` (near `recompute_consonance`):

```rust
    /// Apply a habituation state to the raw kernel outputs, producing the
    /// effective views the consumers read. `h[i] == 0` is identity. Also
    /// re-normalizes the spawn PMF (`consonance_density`) from the eroded mass.
    pub fn apply_habituation(
        &mut self,
        h: &[f32],
        theta: f32,
        repr: &ConsonanceRepresentationParams,
    ) {
        self.assert_scan_lengths();
        let n = self.consonance_field_score.len();
        self.consonance_theta = theta;
        for i in 0..n {
            let hi = h.get(i).copied().unwrap_or(0.0);
            self.perc_habituation_state_scan[i] = hi.clamp(0.0, 1.0);
            let score_eff =
                crate::core::habituation::erode_score(self.consonance_field_score[i], hi, theta);
            self.consonance_field_score_eff[i] = score_eff;
            self.consonance_field_level_eff[i] = repr.level(score_eff);
            self.consonance_density_mass_eff[i] =
                crate::core::habituation::erode_mass(self.consonance_density_mass[i], hi);
        }
        for i in 0..n {
            self.consonance_density[i] = self.consonance_density_mass_eff[i];
        }
        normalize_or_uniform(&mut self.consonance_density[..n]);
    }
```

Switch the movement chokepoints to the effective score. Replace the body of `evaluate_pitch_score` (~line 196) and `evaluate_pitch_score_log2` (~line 202):

```rust
    pub fn evaluate_pitch_score(&self, freq_hz: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear(&self.consonance_field_score_eff, freq_hz)
    }

    pub fn evaluate_pitch_score_log2(&self, log_freq: f32) -> f32 {
        self.assert_scan_lengths();
        self.sample_linear_log2(&self.consonance_field_score_eff, log_freq)
    }
```

Switch the level chokepoints similarly (`evaluate_pitch_level` ~207, `evaluate_pitch_level_log2` ~213) to read `consonance_field_level_eff`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib apply_habituation 2>&1 | tail -20`
Expected: PASS (2 tests).

- [ ] **Step 5: Confirm existing landscape/behavior tests still pass**

Run: `cargo test --lib landscape 2>&1 | tail -30` and `cargo test --test control_rate_dt_invariance 2>&1 | tail -20`
Expected: PASS (eff==raw with zero state; recompute seeds eff to raw).

- [ ] **Step 6: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add src/core/landscape.rs
git commit -m "feat(core): effective consonance views eroded by habituation state"
```

### Task 3.2: Erode the exact leave-self-out score

**Files:**
- Modify: `src/life/pitch_core.rs` (~line 946, `sample_consonance_score_with_loo`)
- Test: inline test in `src/life/pitch_core.rs`

**Interfaces:**
- Consumes: `Landscape.perc_habituation_state_scan`, `Landscape.consonance_theta`, `erode_score`.

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)] mod tests` in `src/life/pitch_core.rs`:

```rust
#[test]
fn exact_loo_scan_is_eroded_by_habituation() {
    use crate::core::landscape::{Landscape, ConsonanceRepresentationParams};
    use crate::core::log2space::Log2Space;
    let space = Log2Space::new(55.0, 1760.0, 12);
    let mut ls = Landscape::new(space);
    let n = ls.space.n_bins();
    for i in 0..n { ls.consonance_field_score[i] = 0.8; ls.consonance_density_mass[i] = 0.5; }
    ls.recompute_consonance(&test_landscape_params(&ls.space)); // helper already in this test module
    let ones = vec![1.0f32; n];
    let repr = ConsonanceRepresentationParams { beta: 2.0, theta: 0.0 };
    ls.apply_habituation(&ones, repr.theta, &repr);

    let scan = vec![0.8f32; n]; // a raw exact-LOO scan
    let mid_log2 = ls.space.centers_log2[n / 2];
    let s = sample_consonance_score_with_loo(
        mid_log2, mid_log2, &ls, true,
        LeaveSelfOutMode::ExactScan, 0, Some(&scan), None,
    );
    // fully habituated -> relaxed to theta (0.0), not the raw 0.8
    assert!(s.abs() < 1e-3, "expected eroded ~0, got {s}");
}
```

(If `test_landscape_params` does not exist in this module, build `LandscapeParams` inline the same way other tests in the file do; keep the assertion on the eroded value.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib exact_loo_scan_is_eroded_by_habituation 2>&1 | tail -20`
Expected: FAIL — returns raw `0.8`.

- [ ] **Step 3: Erode the exact-LOO sample**

In `sample_consonance_score_with_loo` (`pitch_core.rs:942-947`), replace the early return:

```rust
    if leave_self_out
        && matches!(leave_self_out_mode, LeaveSelfOutMode::ExactScan)
        && let Some(scan) = exact_loo_scan
    {
        let raw = landscape.sample_linear_log2(scan, pitch_log2);
        let h = landscape.sample_linear_log2(&landscape.perc_habituation_state_scan, pitch_log2);
        return crate::core::habituation::erode_score(raw, h, landscape.consonance_theta);
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib exact_loo_scan_is_eroded_by_habituation 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add src/life/pitch_core.rs
git commit -m "fix(life): erode exact leave-self-out score through habituation state"
```

### Task 3.3: Own and drive `HabituationField` in the worker loop

**Files:**
- Modify: `src/runtime/mod.rs`
- Test: covered by the Phase 5 integration assay (no cheap unit test for the worker loop; the disabled-path invariance is asserted there and by existing render determinism tests).

**Interfaces:**
- Consumes: `HabituationField` (Task 1.1), `Landscape::apply_habituation` (Task 3.1), `lparams.habituation`, `lparams.consonance_representation`, `lparams.harmonicity_kernel`.

- [ ] **Step 1: Construct two fields where the worker owns `current_landscape`**

In `worker_loop`, near where `current_landscape: LandscapeFrame` is created (~mod.rs:1214) and `lparams` is in scope, add:

```rust
    let hab_n = current_landscape.space.n_bins();
    let mut hab_ecology = crate::core::habituation::HabituationField::new(
        &lparams.habituation,
        lparams.consonance_representation.theta,
        hab_n,
    );
    let mut hab_listener = crate::core::habituation::HabituationField::new(
        &lparams.habituation,
        lparams.consonance_representation.theta,
        hab_n,
    );
```

- [ ] **Step 2: Add a free helper for drive + advance + apply**

Add near the other free functions in `src/runtime/mod.rs`:

```rust
/// Advance a habituation field one hop from the landscape's raw views, then
/// write the effective views. Skips the (allocating) projection when disabled.
fn drive_and_apply_habituation(
    landscape: &mut Landscape,
    hab: &mut crate::core::habituation::HabituationField,
    lparams: &LandscapeParams,
    dt_sec: f32,
) {
    hab.ensure_len(landscape.space.n_bins());
    if hab.is_enabled() {
        let (proj, _max) = lparams
            .harmonicity_kernel
            .potential_h_from_log2_spectrum(&landscape.subjective_intensity, &landscape.space);
        hab.advance_from_parts(&landscape.consonance_field_level, &proj, dt_sec);
    }
    landscape.apply_habituation(hab.state(), hab.theta(), &lparams.consonance_representation);
}
```

- [ ] **Step 3: Call it once per hop after the main merge settles**

In `process_frame`, immediately after the main-analysis fixed-lag loop breaks and `merge_latest_analysis_results` has updated `current_landscape` (right after the loop at ~mod.rs:1333, before the community step at ~mod.rs:1402/1419), add:

```rust
            drive_and_apply_habituation(
                &mut current_landscape,
                &mut hab_ecology,
                lparams,
                hop_duration.as_secs_f32(),
            );
```

- [ ] **Step 4: Drive the listener field once per hop**

In the listener path (after `merge_latest_listener_analysis_results` settles, ~mod.rs:1385, on the observed frame before it is handed to `ListenerTwin`), apply the listener field. The listener frame is local; own the state in the worker and apply to the frame:

```rust
                drive_and_apply_habituation(
                    &mut listener_frame,
                    &mut hab_listener,
                    lparams,
                    hop_duration.as_secs_f32(),
                );
```

(Bind the observed listener frame to a `let mut listener_frame` if it is not already mutable; the function signature of `merge_latest_listener_analysis_results` returns/borrows the frame — apply before `observe_presentation_landscape`.)

- [ ] **Step 5: Re-apply (no advance) on parameter-only recompute**

In `apply_pending_landscape_update`, after `recompute_consonance` (~mod.rs:1719), refresh the eff views WITHOUT advancing `h`:

```rust
    current_landscape.apply_habituation(
        hab_ecology.state(),
        hab_ecology.theta(),
        &lparams.consonance_representation,
    );
```

(Ensure `hab_ecology` is in scope at this call site; if `apply_pending_landscape_update` is a separate fn, thread `&hab_ecology` in as a parameter.)

- [ ] **Step 6: Verify build and the disabled path is unchanged**

Run: `cargo check --all-targets 2>&1 | tail -20` — expected: clean.
Run: `cargo test 2>&1 | tail -30` — expected: all existing tests PASS (habituation absent from config → disabled → eff==raw).

- [ ] **Step 7: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add src/runtime/mod.rs
git commit -m "feat(runtime): drive habituation once per hop on ecology and listener landscapes"
```

---

## Phase 4 — Telemetry

### Task 4.1: `ReportRecord::Habituation`

**Files:**
- Modify: `src/life/report.rs`
- Modify: `src/runtime/mod.rs` (write per-hop summary when reporting)
- Test: covered by Phase 5 (the assay parses these records).

**Interfaces:**
- Produces: `JsonlReporter::write_habituation(&mut self, time_sec, mean_h, max_h, mean_erosion, tracked) -> Result<(), String>` emitting `{"type":"habituation", ...}`.

- [ ] **Step 1: Add the record variant**

In `enum ReportRecord` (`report.rs:83`) add:

```rust
    Habituation {
        time_sec: f32,
        mean_h: f32,
        max_h: f32,
        mean_erosion: f32,
        tracked_bin: usize,
        tracked_h: f32,
        tracked_raw_score: f32,
        tracked_eff_score: f32,
    },
```

- [ ] **Step 2: Add the writer**

In `impl JsonlReporter`, near `write_dcc_pressure` (`report.rs:383`):

```rust
    #[allow(clippy::too_many_arguments)]
    pub fn write_habituation(
        &mut self,
        time_sec: f32,
        mean_h: f32,
        max_h: f32,
        mean_erosion: f32,
        tracked_bin: usize,
        tracked_h: f32,
        tracked_raw_score: f32,
        tracked_eff_score: f32,
    ) -> Result<(), String> {
        self.write_record(&ReportRecord::Habituation {
            time_sec, mean_h, max_h, mean_erosion,
            tracked_bin, tracked_h, tracked_raw_score, tracked_eff_score,
        })
    }
```

- [ ] **Step 3: Emit it per hop in the reporting path**

In `worker_loop`, inside the `reporter.is_some()` per-hop block (near the death/onset writes ~mod.rs:1432), compute summary stats over `current_landscape` and write. `tracked_bin` = the argmax of the RAW score (the strongest consonant peak) so the report shows that peak eroding:

```rust
                let hstate = &current_landscape.perc_habituation_state_scan;
                let n = hstate.len().max(1);
                let mean_h = hstate.iter().sum::<f32>() / n as f32;
                let max_h = hstate.iter().cloned().fold(0.0f32, f32::max);
                let mut mean_erosion = 0.0f32;
                for i in 0..hstate.len() {
                    mean_erosion += current_landscape.consonance_field_score[i]
                        - current_landscape.consonance_field_score_eff[i];
                }
                mean_erosion /= n as f32;
                let tracked_bin = current_landscape
                    .consonance_field_score
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
                    .0;
                report_try(&mut reporter, "habituation", |w| {
                    w.write_habituation(
                        current_time,
                        mean_h,
                        max_h,
                        mean_erosion,
                        tracked_bin,
                        current_landscape.perc_habituation_state_scan[tracked_bin],
                        current_landscape.consonance_field_score[tracked_bin],
                        current_landscape.consonance_field_score_eff[tracked_bin],
                    )
                });
```

- [ ] **Step 4: Verify build**

Run: `cargo check --all-targets 2>&1 | tail -20` — expected: clean.
Run: `cargo test 2>&1 | tail -20` — expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add src/life/report.rs src/runtime/mod.rs
git commit -m "feat(report): emit per-hop habituation telemetry"
```

---

## Phase 5 — Assay and closure test

### Task 5.1: Seeded shared-terrain assay script

**Files:**
- Create: `samples/research/habituation_field_assay.rhai`

**Interfaces:**
- Consumes: existing Rhai scenario API (`seed`, `sine`/`harmonic`, `section`, `place`, `at`, `consonance`, `wait`) — mirror `samples/research/lifecycle_time_domain_assay.rhai`.

- [ ] **Step 1: Author the assay (shared terrain, seeded)**

Create `samples/research/habituation_field_assay.rhai`. It must establish a SHARED terrain: multiple sustained sources so a raw attractor persists when any single voice leaves (the closure precondition). Keep it deterministic and long enough for several `max(satiation, recovery)` cycles.

```rhai
// Habituation field assay — deterministic, seeded.
// Verifies production-loop closure: a voice on a shared consonant basin erodes
// it, departs, the basin recovers after the voice leaves, and a voice returns.
// Enable habituation via config: run with a config TOML that sets
//   [psychoacoustics.habituation] enabled = true
// (see tests/habituation_field_assay.rs). With habituation disabled this script
// settles and does not migrate — that is the causal control.
seed(20260717);

// A persistent shared terrain: several fixed sustained partials build consonant
// basins that survive any one mover leaving.
section("shared_terrain", || {
    place(harmonic(110.0)).at(110.0);
    place(harmonic(165.0)).at(165.0);
    place(harmonic(220.0)).at(220.0);
    wait(40.0);
});
```

(Match the exact builder names/signatures used in `samples/research/lifecycle_time_domain_assay.rhai`; adjust `place(...).at(...)` to the real API if it differs. The load-bearing requirements: a `seed(...)` line, sustained sources forming a shared terrain, and a total duration ≥ ~40 s so several cycles occur.)

- [ ] **Step 2: Verify it compiles as a scenario**

Run: `cargo run --quiet -- --config config.toml --nogui --play=false samples/research/habituation_field_assay.rhai 2>&1 | tail -20`
Expected: runs to completion with no parse/eval error.

- [ ] **Step 3: Confirm seed-policy test still passes**

Run: `cargo test --test sample_seed_policy 2>&1 | tail -20`
Expected: PASS (research assay has a `seed(` line; top-level count still 12).

- [ ] **Step 4: Commit**

```bash
git add samples/research/habituation_field_assay.rhai
git commit -m "test(assay): seeded shared-terrain habituation assay script"
```

### Task 5.2: Deterministic closure integration test

**Files:**
- Create: `tests/habituation_field_assay.rs`
- Create (test-local, temp): a config TOML written at runtime enabling habituation.

**Interfaces:**
- Consumes: the assay script (Task 5.1), the `--config`/`--report`/`--nogui`/`--play=false` CLI, and the `{"type":"habituation",...}` records (Task 4.1). Mirrors `tests/lifecycle_time_domain_assay.rs`.

- [ ] **Step 1: Write the test — causal control + erosion + recovery**

Create `tests/habituation_field_assay.rs`:

```rust
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("conchordal_habituation_{nanos}_{name}"));
    p
}

fn write_config(enabled: bool) -> PathBuf {
    let path = temp_path("config.toml");
    let mut f = std::fs::File::create(&path).expect("create config");
    write!(
        f,
        "[psychoacoustics.habituation]\nenabled = {enabled}\nsatiation_sec = 5.0\nrecovery_sec = 8.0\nref_drive = 0.25\n"
    )
    .expect("write config");
    path
}

fn run(scenario: &str, config: &PathBuf, report: &PathBuf) {
    let status = Command::new(env!("CARGO_BIN_EXE_conchordal"))
        .arg(scenario)
        .arg("--config").arg(config)
        .args(["--nogui", "--play=false", "--report"])
        .arg(report)
        .status()
        .expect("run conchordal");
    assert!(status.success(), "conchordal exited nonzero");
}

fn habituation_series(report: &PathBuf) -> Vec<(f32, f32, f32)> {
    // (time_sec, tracked_h, tracked_eff_score)
    let text = std::fs::read_to_string(report).expect("read report");
    let mut out = Vec::new();
    for line in text.lines() {
        let v: serde_json::Value = match serde_json::from_str(line) { Ok(v) => v, Err(_) => continue };
        if v["type"] == "habituation" {
            out.push((
                v["time_sec"].as_f64().unwrap() as f32,
                v["tracked_h"].as_f64().unwrap() as f32,
                v["tracked_eff_score"].as_f64().unwrap() as f32,
            ));
        }
    }
    out
}

#[test]
fn habituation_erodes_then_recovers_the_tracked_basin() {
    let scenario = "samples/research/habituation_field_assay.rhai";
    let cfg_on = write_config(true);
    let rep_on = temp_path("on.jsonl");
    run(scenario, &cfg_on, &rep_on);

    let series = habituation_series(&rep_on);
    assert!(!series.is_empty(), "no habituation records emitted");

    // The tracked basin's h must rise substantially at some point (erosion).
    let max_h = series.iter().map(|s| s.1).fold(0.0f32, f32::max);
    assert!(max_h > 0.5, "tracked basin never eroded (max_h={max_h})");

    // After the peak erosion, h must come back down somewhere later (recovery),
    // proving the state is not a one-way ratchet.
    let peak_idx = series.iter().enumerate().max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap()).unwrap().0;
    let after_min = series[peak_idx..].iter().map(|s| s.1).fold(1.0f32, f32::min);
    assert!(after_min < max_h - 0.2, "no recovery after peak (max={max_h}, after_min={after_min})");
}

#[test]
fn habituation_off_is_the_causal_control() {
    let scenario = "samples/research/habituation_field_assay.rhai";
    let cfg_off = write_config(false);
    let rep_off = temp_path("off.jsonl");
    run(scenario, &cfg_off, &rep_off);

    let series = habituation_series(&rep_off);
    // With habituation disabled, tracked_h stays ~0 and eff==raw everywhere.
    for (_, h, _) in &series {
        assert!(*h < 1e-6, "disabled path must keep h==0, got {h}");
    }
}
```

- [ ] **Step 2: Run to verify it fails first (before Task 4.1 records exist), then passes now**

Run: `cargo test --test habituation_field_assay 2>&1 | tail -30`
Expected: PASS both tests. If `habituation_off_is_the_causal_control` fails because no records are emitted when disabled, adjust Task 4.1 to emit records unconditionally in the reporting path (it already does — records emit whenever `reporter.is_some()`, independent of `enabled`).

- [ ] **Step 3: Record the run**

Append the observed erosion/recovery numbers (max_h, recovery delta, timings) to `docs/roadmap/v0.4.0-rhythm-report-runs.md` under a new "Habituation assay" heading, matching how other assay runs are logged.

- [ ] **Step 4: Commit**

```bash
cargo fmt --all && cargo clippy -- -D warnings
git add tests/habituation_field_assay.rs docs/roadmap/v0.4.0-rhythm-report-runs.md
git commit -m "test(assay): deterministic habituation erosion/recovery closure test"
```

### Task 5.3: Closure verdict (manual research follow-up, documented)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-habituation-field-design.md` (append an "Assay results" section) OR `docs/design-notes/` per project convention.

The automated test in 5.2 proves the minimal closure signature (erosion + recovery of the tracked basin, plus the causal control). The FULL closure claim — return-to-vacated-basin by a voice, bounded recurrence with no secular drift, and the robustness matrix (seeds, population sizes, tau ratios, body spectra sine/harmonic/modal/missing-fundamental, gain, ref_power, bins_per_oct, per-voice adaptation on/off) — is a manual research campaign, because most cells are "run and observe," not a single golden assertion.

- [ ] **Step 1: Run the robustness sweep**

For each cell, run the assay with a per-cell config (vary `satiation_sec`/`recovery_sec`/`ref_drive` and the scenario's body spectra), collect the JSONL, and record: max_h, recovery delta, migration synchrony, adjacent-bin reversal rate, deaths, density uniform-fallback count, mean effective consonance drift. Note any cell where the loop fails to close (diverges, collapses, thrashes).

- [ ] **Step 2: Write the verdict**

Append the results and a PASS/REJECT verdict on production-loop closure. If PASS, the spec's Phase 2 (permanent core integration, composer API, technote ledger update, default-on decision) is authorized as a separate plan. If REJECT, record which failure mode killed it, matching the §9.3 "both tests" gate.

- [ ] **Step 3: Commit**

```bash
git add docs/
git commit -m "docs: habituation assay results and closure verdict"
```

---

## Self-Review

**Spec coverage:** Point of action (erode kernel output) → Phase 3.1. Excitation in root coordinate → Phase 3.3 (`potential_h_from_log2_spectrum` on `subjective_intensity`, gated by level). Decay toward theta → Phase 1.1 `erode_score` + 3.1. Asymmetric leaky integrator + bounded transfer → Phase 1.1. State scope per-Landscape global → Phase 3.3 (two worker-owned fields). satiation/recovery seconds contract → Phase 1.1 tau derivation + Phase 0 config. Persistent-state ownership → Phase 3.3 (worker-owned, not on channel frame). Per-hop update order → Task 3.1 `apply_habituation` (raw → advance → eff → level/energy from eff → PMF from eff mass). Raw vs effective distinct names → Phase 3.1. LOO bypass → Task 3.2. Prediction bias → measured in Task 5.3 (no predictor state added, per spec). Telemetry (not err_*) → Phase 4. Opt-in default off → Phase 0 + Global Constraints. Assay before core → Phases 5. Non-goals (composer API, ledger, default-on, continuum) → explicitly excluded in Global Constraints and Task 5.3.

**Placeholder scan:** assay-tuned knobs have concrete default values; the Rhai builder names in Task 5.1 are flagged to match the real API (the one place needing on-the-spot verification against `lifecycle_time_domain_assay.rhai`). No "TBD"/"handle edge cases"/"similar to" placeholders.

**Type consistency:** `erode_score(raw,h,theta)`/`erode_mass(raw,h)` signatures identical across Tasks 1.1, 3.1, 3.2. `HabituationField::{new,advance_from_parts,state,theta,is_enabled,ensure_len}` used consistently in 1.1 and 3.3. `apply_habituation(&mut self, h, theta, repr)` identical in 3.1, 3.3, 4.1 call sites. `write_habituation` field order matches the `ReportRecord::Habituation` variant.
