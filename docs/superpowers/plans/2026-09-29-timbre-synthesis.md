# Timbre and Synthesis Implementation Plan

> 2026-10-07: the remaining work, gates and integration order in this document are superseded by [the current plan](../../roadmap/plan-current.md). This document is kept as history and reference.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry the timbre design (a Voice's body under a listener model) from its adopted principles to working code, in an order where each step is testable and none presumes a later one.

**Architecture:** The listener model is the fixed law (ledger §9.3.56: listener-model layer 1 fixed, layer 3 set by the composer, layers 4–5 modeled or emergent). Bodies follow a state-space grammar (excitation drives a persistent state that radiates), with parameters free of material law. Synthesis methods are replaceable body modules; the core computes everything the ecology evaluates from radiated sound, and a body's self-model is only a verified fast path. A body feature becomes heritable only through a named pathway: a mutation direction, the detector that sees it, and the behavior consumer it reaches.

**Tech Stack:** Rust engine (`src/life/sound/`, `src/life/`, `src/core/`, `src/runtime/`), Rhai scenarios, `examples/timbre_probes.rs` for audition.

**Spec:** `docs/design-notes/timbre.md`; listener-model layers and parameter ledger in `docs/design-notes/technote-ledger.md` §9.3.56.

## Global Constraints

- Every task that modifies `src/` ends with: `set -o pipefail; ( RUST_BACKTRACE=1 cargo test -- --nocapture ) 2>&1 | tee test_report.txt; echo "cargo test exit=$? @ $(date -Iseconds)" > test_status.txt` in one shell.
- Before any commit: `cargo fmt --all` and `cargo clippy -- -D warnings` pass. Commit only when the user asks.
- Code comments in English only.
- Hot-path allocation budget as listed in `AGENTS.md`; reuse scratch buffers in hop paths.
- Alpha no-compat policy: no aliases or migration shims.
- Sample outputs may change when the core spec changes; bit-identity of `autumn_cycle` or any étude is not required (user, 2026-09-29). Each such change states which outputs change and why.
- Determinism: everything that reaches the habitat bus is ecological input. Under a fixed seed it must be reproducible, including stochastic synthesis: seed it and advance its state deterministically. Nondeterminism is tolerated only on presentation-only output.
- The composer sets listener-model layer 3, the initial conditions of what the ecology evolves, and the macro form (`AGENTS.md`, ledger §9.3.56). No new composer-facing timbre parameters (timbre.md non-goals).
- Core and body modules (timbre.md, "Core and body modules"): new ecology code does not branch on body kind; a body self-report is used only where it matches the body's render in the conformance tests.
- Whole-body fitness (F1–F5) is owned by `docs/design-notes/body-aware-fitness-plan.md`. Timbre heredity (its F6) is planned here as Phase 5; that plan points here.
- Three questions are always recorded separately: did the synthesizer emit the intended waveform; did the analysis detect the difference and return it to behavior; can a person hear it. Timbre is judged by ear.
- A verified causal path is an engineering pass. The sign and size of a selection difference is a research result, never a pass condition.

## Status

Done: energy-driven `damping` in the scheduled harmonic path; body-ratio leave-one-out in `ApproxHarmonics`; production-meter onset drive limited to habitat-routed Voices (2026-09-29); `set_roughness_k` replaced by the valuation-side `set_roughness_aversion` (2026-09-29), with one validation path (`LandscapeParams::set_roughness_aversion`: non-finite ignored, clamped to `[0, 100]`), recomputation on any stored change, and the worker's kernel kept in step with its own parameters. These code changes live on branch `timbre-valuation-meter` (commit `a3a56fd`, worktree `.worktrees/timbre-valuation-meter`; 1,165 tests and standard Clippy pass there), not in main, because Astra sessions work in the shared main checkout and its pre-commit hook stages every tracked change. They merge at the next integration point after I12b closes and the base version is fixed. Registered I11 and body-aware conditions do not change output under them: no registered scene routes a Voice to the presentation bus alone, and the default aversion reproduces the configured kernel bit-exact.

In progress elsewhere: whole-body fitness (body-aware fitness plan, isolated worktrees).

Task 1 steps 1–5 completed on 2026-09-29 in `.worktrees/timbre-valuation-meter` (uncommitted example change at this checkpoint). The 23 blind stimuli are in that worktree's `target/timbre_probes/`, with a blind file/hash index in `fluct_sha256.txt`. All are mono PCM16, 48 kHz, three seconds; PCM RMS ranges from 0.099987 to 0.099990, and the maximum realized modulation RMS error is 0.20 cents against the 0.25-cent check. Example check, release generation, example Clippy, formatting and the full Rust suite passed. The author has not auditioned them; the condition key remains undisclosed. Step 6 and adoption are pending.

## Decisions (2026-09-29)

- **D1, raw consonance coefficients:** not exposed to scenarios. Composer-facing valuation stays `set_roughness_aversion`, `set_pitch_objective` and the meter priors (ledger §9.3.56). `set_roughness_aversion` acts as aversion only for kernels with `b <= 0` and `b + c <= 0` (the defaults); otherwise it scales the configured roughness terms.
- **D2, transmission:** the child copies the body of the parent respawn already selects (linked with pitch, success-biased). Linkage is a model assumption, not a defect. Hitchhiking is measured by the controls in Phase 5 before any copying from a non-parent.
- **D3, learning listener-model layer 3 in performance:** not scheduled. Reconsidered only after Phase 5 shows generational change and a runaway measure is registered; co-drift of valuation and trait is necessary but not sufficient evidence of runaway, so the registration needs a causal control of mutual amplification.
- **D4, source attribution:** implemented last (Phase 6); its interface is checked early. Until a perceptual pathway exists, no feature is heritable through that pathway. The gate is per pathway, not per parameter: a decay may be heritable through spectrum and not through grouping.
- **D5, micro-fluctuation:** split in two.
  - Audition decides only whether irregular fluctuation should be present while a body is driven, and at what level (Task 1). The result is a candidate setting, not a physical or cognitive argument.
  - Where fluctuation lives (excitation or body state), who owns its state, and what the tail keeps are design decisions of Task 2 for this body family. The contract distinguishes "no new stochastic drive after Off" from "no trace of fluctuation in the tail": phase and amplitude acquired while driven stay in the tail.
  - The engine's `motion` is almost pure vibrato: 5 Hz at relative depth 0.02·`motion` (24.5 cents RMS at 1.0), while its pink-noise jitter is 0.03–0.1 cents RMS at 1.0 (measured from the implementation). Task 1 therefore does not use `motion`.
- **D6, core and body modules:** from received excitation to radiated sound is a replaceable body module; hearing, evaluating and deciding excitation is core. Everything the ecology evaluates is computed by the core from radiated sound; a body's self-model (ratios, footprint, energy forecasts) is a fast path used only where the conformance tests show it matches the body's render. A body without a self-model runs on the render-derived path, offline first.
- **D7, scope of Task 2 and Phase 3:** Task 2 becomes the body module contract (excitation, state, genotype, optional self-model, what the core guarantees). Phase 3 delivers a conformance harness and migrates `Sine`, `Harmonic` and `Modal` behind the contract, removing the closed `BodySnapshot`/`AnyBackend` enumeration and the body-kind branches in ecology code.

- **D8, Task 2 and body-aware F2:** the self-model face of the body module contract is designed together with F2's production representation (the direct model and its partial-group successor). One representation serves both; F2's reference comparison with pre-registered tolerances is D6's verification rule applied.
- **D9, Phase 2 folded into body-aware F2–F4:** the timbre-specific acceptance (body swap, controlled emitted level, silence and high-cut controls, causal path to parent-selection probability) is added to the F2–F4 registrations instead of registering a separate acquisition.
- **D10, Phase 4 folded into T1/T2:** the temporal grouping contrasts are registered within the T1 masking / T2 modulation terrain work of the temporal DCC track.
- **D11, placement:** today's code changes sit on branch `timbre-valuation-meter` (see Status).

## Dependencies

- Task 1 runs now, in parallel with everything; its audition can share a session with the A1 audition.
- Task 2 runs now as a document; its self-model face is drafted with body-aware F2 (D8).
- Phase 2 runs inside body-aware F2–F4 (D9): a baseline on the current synthesizers, then re-acceptance on the Phase 3 synthesizers before Phase 5.
- Phase 3 integrates only after: Task 2 approved; F2's representation decided; I12b closed and the base version fixed; I4's wiring change landed (temporal DCC order). A prototype in an isolated worktree may start earlier; integration is serialized with those plans because they change the same files (`schedule_renderer.rs`, `voice.rs`, `pitch_core.rs`, `temporal_cognition/body.rs`, footprint and energy code) and their registered comparisons depend on the current render path.
- Phase 4 runs inside T1/T2 (D10), after Phase 3. A trait whose observability is first established there stays non-heritable until then; it is not a precondition for heritability through spectrum.
- Phase 5 after Phase 2 re-acceptance and body-aware F4/F5.
- Phase 6: the interface to the temporal DCC attribution is checked at the start of Phase 5; implementation acceptance comes last.
- Builds, tests and renders for this plan do not run during another plan's timing acquisition.

Phases 2–6 get their own task-level plan when they start. Their sections below fix scope, files and exit criteria so later plans cannot drift.

---

### Task 1: Fluctuation audition stimuli

Gives the author a blind, equal-loudness comparison for D5's audition half. The stimuli bypass the engine on purpose: `motion` is vibrato there, and the question is perceptual. No `src/` change.

The operational definition is narrow. "Irregular" means frequency modulation with a 1/f spectrum limited to 0.2–12 Hz (slow flutter), not cycle-to-cycle jitter in the voice-science sense. "Periodic" means 5 Hz sinusoidal FM at the same RMS in cents. The pair stimuli test whether independent versus shared irregular FM changes how two tones group. A pair at ratio 1.799 with six partials has no shared partials (minimum separation about 130 cents), so a change there cannot come from beating of coincident partials. A fifth, whose partials coincide, is kept as the ecological example; a difference there is not attributable to modulation coherence alone (Carlyon 1991). Neither result isolates a grouping mechanism.

**Files:**
- Modify: `examples/timbre_probes.rs`

**Interfaces:**
- Consumes: `FS` and the existing probe loop in the same file.
- Produces: `target/timbre_probes/fluct_01.wav` … `fluct_23.wav` in shuffled order, and `target/timbre_probes/fluct_key.txt` (file, condition, realized cents RMS per tone).

- [x] **Step 1: Imports**

Change the `rand` import line to:

```rust
use rand::{RngExt, SeedableRng, rngs::SmallRng};
```

and add below the other `use` lines:

```rust
use std::f32::consts::{SQRT_2, TAU};
```

- [x] **Step 2: Split the WAV writer so stimuli keep their equal-RMS scaling**

Replace the existing `write_wav` with:

```rust
/// Normalize to 0.9 peak and write a mono 16-bit WAV. Returns the pre-normalize peak.
fn write_wav(dir: &Path, name: &str, samples: &mut [f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    if peak > 0.0 {
        let gain = 0.9 / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
    write_pcm16(dir, name, samples);
    peak
}

/// Write a mono 16-bit WAV as given (clamped to [-1, 1]).
fn write_pcm16(dir: &Path, name: &str, samples: &[f32]) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: FS as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let path = dir.join(format!("{name}.wav"));
    let mut writer = WavWriter::create(&path, spec).expect("create wav");
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}
```

- [x] **Step 3: Add the stimulus generator above `fn main`**

```rust
/// Fluctuation for stimulus-level audition. These bypass the engine, where
/// `motion` is almost pure 5 Hz vibrato; the question here is perceptual.
#[derive(Clone, Copy)]
enum Fluctuation {
    None,
    /// FM with a 1/f spectrum limited to 0.2-12 Hz (slow flutter, not
    /// cycle-to-cycle jitter).
    Irregular { cents_rms: f32 },
    /// 5 Hz sinusoidal FM with the same RMS in cents.
    Periodic { cents_rms: f32 },
}

const CONTROL_RATE_HZ: f32 = 1000.0;
const WARMUP_SEC: f32 = 5.0;
const STIM_SEC: f32 = 3.0;
const STIM_RMS: f32 = 0.1;

fn normalize_unit_rms(x: &mut [f32]) {
    let n = x.len().max(1) as f32;
    let mean = x.iter().sum::<f32>() / n;
    x.iter_mut().for_each(|v| *v -= mean);
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
    if rms > 0.0 {
        x.iter_mut().for_each(|v| *v /= rms);
    }
}

/// Zero-mean, unit-RMS irregular trajectory at the control rate: Kellet pink
/// noise, one-pole low-pass at 12 Hz and high-pass at 0.2 Hz, with a warm-up
/// discarded so the filters start in steady state.
fn irregular_series(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let warm = (WARMUP_SEC * CONTROL_RATE_HZ) as usize;
    let a_lp = (-TAU * 12.0 / CONTROL_RATE_HZ).exp();
    let a_hp = (-TAU * 0.2 / CONTROL_RATE_HZ).exp();
    let (mut b0, mut b1, mut b2) = (0.0f32, 0.0f32, 0.0f32);
    let (mut lp, mut hp, mut prev) = (0.0f32, 0.0f32, 0.0f32);
    let mut out = Vec::with_capacity(len);
    for i in 0..warm + len {
        let white: f32 = rng.random_range(-1.0..1.0);
        b0 = 0.99765 * b0 + white * 0.099_046;
        b1 = 0.963 * b1 + white * 0.296_516_4;
        b2 = 0.57 * b2 + white * 1.052_691_3;
        let pink = b0 + b1 + b2 + white * 0.1848;
        lp = a_lp * lp + (1.0 - a_lp) * pink;
        hp = a_hp * (hp + lp - prev);
        prev = lp;
        if i >= warm {
            out.push(hp);
        }
    }
    normalize_unit_rms(&mut out);
    out
}

fn cents_trajectory(fluct: Fluctuation, len: usize, seed: u64) -> Vec<f32> {
    match fluct {
        Fluctuation::None => vec![0.0; len],
        Fluctuation::Irregular { cents_rms } => irregular_series(len, seed)
            .into_iter()
            .map(|x| x * cents_rms)
            .collect(),
        Fluctuation::Periodic { cents_rms } => {
            let phase0 = (seed % 1000) as f32 / 1000.0 * TAU;
            (0..len)
                .map(|i| cents_rms * SQRT_2 * (phase0 + TAU * 5.0 * i as f32 / CONTROL_RATE_HZ).sin())
                .collect()
        }
    }
}

/// Sustained harmonic tones (partials 1/n^1.2), 50 ms fades, scaled to
/// `STIM_RMS`. `trajs[k]` is tone k's cents trajectory, common to its
/// partials. Also returns each tone's realized cents RMS over the unfaded span.
fn render_additive(f0s: &[f32], partials: usize, trajs: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let n = (STIM_SEC * FS) as usize;
    let fade = (0.05 * FS) as usize;
    let mut out = vec![0.0f32; n];
    let mut realized = Vec::with_capacity(f0s.len());
    for (&f0, cents) in f0s.iter().zip(trajs) {
        let mut phases = vec![0.0f32; partials];
        let (mut sum_sq, mut count) = (0.0f64, 0usize);
        for (i, s) in out.iter_mut().enumerate() {
            let pos = i as f32 * CONTROL_RATE_HZ / FS;
            let j = pos as usize;
            let c = cents[j] + (cents[j + 1] - cents[j]) * (pos - j as f32);
            if i >= fade && i < n - fade {
                sum_sq += f64::from(c * c);
                count += 1;
            }
            let f = f0 * (c / 1200.0).exp2();
            let env = (i.min(n - 1 - i) as f32 / fade as f32).min(1.0);
            let mut acc = 0.0f32;
            for (p, phase) in phases.iter_mut().enumerate() {
                let h = (p + 1) as f32;
                *phase = (*phase + TAU * f * h / FS).rem_euclid(TAU);
                acc += phase.sin() / h.powf(1.2);
            }
            *s += env * acc;
        }
        realized.push((sum_sq / count.max(1) as f64).sqrt() as f32);
    }
    let rms = (out.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
    if rms > 0.0 {
        out.iter_mut().for_each(|s| *s *= STIM_RMS / rms);
    }
    (out, realized)
}

struct Stimulus {
    label: String,
    f0s: Vec<f32>,
    partials: usize,
    trajs: Vec<Vec<f32>>,
}

fn fluctuation_battery() -> Vec<Stimulus> {
    let len = (STIM_SEC * CONTROL_RATE_HZ) as usize + 2;
    let none = || cents_trajectory(Fluctuation::None, len, 0);
    let irregular = |cents, seed| cents_trajectory(Fluctuation::Irregular { cents_rms: cents }, len, seed);
    let mut out = vec![Stimulus {
        label: "single none".into(),
        f0s: vec![220.0],
        partials: 12,
        trajs: vec![none()],
    }];
    for cents in [3.0f32, 8.0] {
        for seed in 1..=3u64 {
            out.push(Stimulus {
                label: format!("single irregular {cents}c seed{seed}"),
                f0s: vec![220.0],
                partials: 12,
                trajs: vec![irregular(cents, seed)],
            });
        }
        out.push(Stimulus {
            label: format!("single periodic-5Hz {cents}c"),
            f0s: vec![220.0],
            partials: 12,
            trajs: vec![cents_trajectory(Fluctuation::Periodic { cents_rms: cents }, len, 1)],
        });
    }
    // Tone A keeps its trajectory across the coherence conditions; only tone B changes.
    for (tag, ratio) in [("noshare-1.799", 1.799f32), ("fifth", 1.5)] {
        let f0s = vec![220.0, 220.0 * ratio];
        out.push(Stimulus {
            label: format!("pair {tag} none"),
            f0s: f0s.clone(),
            partials: 6,
            trajs: vec![none(), none()],
        });
        for seed in 1..=3u64 {
            let a = irregular(8.0, seed);
            out.push(Stimulus {
                label: format!("pair {tag} shared 8c seed{seed}"),
                f0s: f0s.clone(),
                partials: 6,
                trajs: vec![a.clone(), a.clone()],
            });
            out.push(Stimulus {
                label: format!("pair {tag} independent 8c seed{seed}"),
                f0s: f0s.clone(),
                partials: 6,
                trajs: vec![a, irregular(8.0, seed + 100)],
            });
        }
    }
    out
}
```

- [x] **Step 4: Render the battery at the end of `main`, before the final `println!` lines**

```rust
    let mut battery = fluctuation_battery();
    let mut order_rng = SmallRng::seed_from_u64(0xF1C7_0DE5);
    for i in (1..battery.len()).rev() {
        let j = order_rng.random_range(0..=i);
        battery.swap(i, j);
    }
    let mut key = String::from("file\tcondition\trealized cents rms per tone\n");
    for (i, stim) in battery.iter().enumerate() {
        let name = format!("fluct_{:02}", i + 1);
        let (samples, realized) = render_additive(&stim.f0s, stim.partials, &stim.trajs);
        write_pcm16(dir, &name, &samples);
        let realized: Vec<String> = realized.iter().map(|c| format!("{c:.2}")).collect();
        key.push_str(&format!("{name}\t{}\t{}\n", stim.label, realized.join(",")));
    }
    std::fs::write(dir.join("fluct_key.txt"), key).expect("write key");
    println!(
        "  fluct_01..fluct_{:02}: fluctuation stimuli, equal RMS, shuffled.",
        battery.len()
    );
    println!("      listen blind first; conditions are in fluct_key.txt");
```

- [x] **Step 5: Build, render, check the key**

Run: `cargo check --examples && cargo run --release --example timbre_probes && cat target/timbre_probes/fluct_key.txt`
Expected: 23 stimuli; realized RMS within about 0.25 cents of 3.00 or 8.00 for fluctuating tones and 0.00 for static ones (a check run gave 3.00–3.03 and 7.80–8.08).

- [ ] **Step 6: Author audition and record**

The author deferred this audition on October 7; this step remains incomplete. The author listens blind, then reads the key, and records in `docs/design-notes/timbre.md` (micro-fluctuation bullet) a candidate: whether irregular fluctuation should be present while a body is driven, at what RMS, and whether it differs from periodic FM at equal RMS; and whether independent fluctuation separates the non-sharing pair. Nothing in `src/` changes on this basis alone; Phase 3 re-checks the candidate on the real synthesizer (drive, stop, tail, re-excitation).

---

### Task 2: Body module contract

A written contract, reviewed before any code, because Phases 3–5 depend on it. It covers the excitation and state semantics and the boundary between core and body modules (D6, D7).

Draft prepared and revised on 2026-09-29: [body module contract](../specs/2026-09-29-body-module-contract.md). Body-aware B1–B7 responses are recorded. The [independent Claude review](../specs/2026-09-29-body-module-contract-claude-review.md), actual model `claude-opus-5-5[1m]`, read only the author-authorized six-document packet without tools. Its verdict is conditional readiness for author policy decisions, not readiness for A4 adoption. The [response](../specs/2026-09-29-body-module-contract-review-response.md) records fifteen dispositions. The later [author decision](../specs/2026-09-29-body-policy-author-decisions.md) adopts A1, A2's Off policy with 0.5 s T60 as a prototype value, and A3's cost attribution. The October 7 decision also adopts C5A's purpose-specific realized-PCM references and numerical criteria, plus the single-round closed-amplitude law for a new Sine/Harmonic prototype version. Remaining capacity/module-domain details and full A4 adoption are pending; Phase 3 remains unstarted.

The October 3 decision also adopts natural certified closed tails (C1), an initial
48 kHz-only new-module family (C3), and addressed SplitMix64 white excitation with
at most sixteen source-modulation components (C4). These choices need no repeat
approval. Standalone Rust checks now cover the exact C4 key/white/source-driver
law and the now-adopted single-round scalar tail evaluator; see the
[C4 implementation record](../../../target/addressed-noise-native-20261004/note.md)
and [scalar arithmetic record](../../../target/closed-amplitude-native-20261004/note.md).
The [Modal native checks](../../../target/modal-native-certificate-20261004/note.md)
observe actual coefficients and scalar/SIMD rounding, while the
[initialization checks](../../../target/addressed-init-coupling-20261004/note.md)
supply the addressed key to the existing Modal coupling algorithm.
The October 7 adoption selects `RN32(a_close * (4193097/2^22)^N)` for the new
prototype version, with `N = checked(t-c)+1` and uniform zero exponent 669702.
It covers finite nonnegative binary32 amplitudes and all u64 exponents; scalar
zero alone does not establish body disposal. The isolated opt-in native
`cfg(test)` prototype `sine-harmonic-closed-scalar-native-v1` now implements
explicit immediate effective Off for Sine/Harmonic at 48 kHz with indefinite
native hold. Its six focused native cases, fmt, Clippy and final full test pass
(1,172 passed, 0 failed, 36 ignored). Production body/handle integration,
complete input and capacity domains, consumer conformance and
full A4 remain open. Author audition is explicitly deferred.

The October 7 precision-scope decision keeps the adopted consumer thresholds
and closed-amplitude law. An exact shared compiler's intermediate bit equality
is evidence for that implementation, not an additional admission gate for all
predictive fast paths. Reordered, vectorized or approximate arithmetic may use
the existing combined error budget and is judged end to end against the original
reference, coverage, supported-domain and resource requirements. Preserve the
observation path, discrete ownership/clock/support contracts and frozen controls.
Reuse the completed scalar certificate and arithmetic checks for native
integration; do not repeat the universal sweep or add analogous proofs to other
bodies. See the contract's [precision scope](../specs/2026-09-29-body-module-contract.md#precision-scope-adopted-on-october-7)
and the [author decision record](../specs/2026-09-29-body-policy-author-decisions.md).

**Files:**
- Create: `docs/superpowers/specs/2026-09-29-body-module-contract.md`
- Modify: `docs/design-notes/timbre.md` (link from Priority 3)

- [x] **Step 1: Write the contract with a table and one chosen behavior plus its test per question**

1. For `ToneCmd::On`, `Off` and `Update`: the change to input, state and output, as a table. Excitation stop, state disposal and audible silence are separate events.
2. Two body kinds: decaying bodies (free response ends on its own) and self-sustaining bodies (sound only while driven, with the ecology paying for the drive). How an oscillator body (`Sine`, `Harmonic`) rings down once excitation stops; today it sounds indefinitely after one impulse.
3. Unified meaning of excitation across bodies: input units, what a finite drive means, what the free response is. Each body keeps its own transfer characteristic and output level; unification must not flatten them. `DriveMode` in `src/life/sound/any_backend.rs`; `sine_impulse_boost` goes.
4. Conditions under which re-excitation superposes with the ringing state: linearity, unchanged coefficients, output gain, and per-`Tone` release handling.
5. Pitch or coefficient change of a sounding body. Options to compare and one to adopt as the author's policy: transfer the state to the new coefficients, keep the old state ringing (today: a separate `Tone`, so one individual can sound two pitches), or crossfade. Continuity of a source is not the same as monophony.
6. Fluctuation (D5): whether it is an excitation input or part of body state; who owns its random state (Voice or `Tone`; today per `Tone` via `modal_phase_seed(source_id, onset, tone_id)`); what happens to that state across silences; seeded, deterministic advancement on habitat output.
7. Where ADSR survives: excitation shaping only; click prevention and output gain as output processing.
8. Lifetime and resources: state owner and disposal on Voice death, silence and coefficient change; bounds on live states and per-hop work for long tails and repeated excitation; no new hop-path allocation.
9. Which existing tests and probes must pass unchanged, which outputs are expected to change, and why.
10. The module's three faces and their signatures: genotype (construction from the founder vocabulary, mutation, capture for heredity), renderer (excitation in, audio block out, free response, disposal), optional self-model (spectral footprint at a pitch, band energy forecast, own ratios).
11. What the core guarantees to every body: seed and random advance, excitation meaning, per-block resource bound, no hop-path allocation, routing and output processing outside the body.
12. Dispatch: built-in bodies stay statically dispatched on the hot path; research bodies enter through the registry. Which trait objects are allowed where, and the per-note allocation budget at `Tone` creation.
13. Reference frequency: what the commanded pitch means for inharmonic bodies (the anchor each body maps to its spectrum). Unpitched bodies stay open until one is tried.
14. Render-derived fallback for each current self-model consumer, and the conformance tolerance each fast path must meet: `action_candidates/footprint.rs`, `action_candidates/energy.rs`, `temporal_cognition/body.rs`, `self_prediction`, `voice.rs`/`modal.rs` (`project_spectral_body`), `pitch_core.rs` (ratio leave-one-out).
15. Coordination with the body-aware fitness work, whose evaluator renders representative `Tone`s from the current structures: the migration keeps an equivalent representative-render entry point and lands at a point agreed with that plan.

- [ ] **Step 2: Independent review** by a model other than the drafter (Claude when Astra drafts), then the author's approval. Record the verdict in the spec. The spec's section "Points to settle with body-aware" is answered by the body-aware thread before approval.

Initial review is received; this combined step remains unchecked because author
adoption is pending. Task 2's explicit other-model requirement and the author's
specific Claude authorization govern this review; they are a task-specific
requirement preserved under `AGENTS.md`'s Sol 6.1 primary routing (author
instruction, 2026-09-30). No new external send is authorized by recording the
response. Remaining A3 capacity/module details,
C5A consumer-domain/conformance work and the F2 entry interpretation
must be resolved before A4; the later adopted policies are not awaiting repeat approval.

---

### Phase 2: Timbre acceptance on whole-body fitness

Folded into body-aware F2–F4 (D9). This section states what those registrations must carry. Baseline on the current synthesizers; re-acceptance after Phase 3.

**Files (expected):** `samples/research/timbre_collision_assay.rhai` (seed pinned, per `tests/sample_seed_policy.rs`), a test under `tests/`.

**Exit (engineering):**
- Body-swap control: at the same fundamental, position and environment, exchanging only the body changes the evaluation, and the change propagates to energy and to parent-selection probability. The path is traced; a single run need not change the selected parent.
- Bodies are compared at controlled emitted level. Zero-level and high-cut controls record both the benefit and the cost of silence, so the report can say what is being optimized.
- The averaging approximation is stated: H is relational, and pointwise averages of C over partial positions approximate it.
- A fixed-seed regression test and a multi-seed research assay are kept separate.

**Re-acceptance after Phase 3:** radiated sound, candidate representation, self-exclusion and the selection path are re-checked against the new excitation and decay, since a mean partial table may not represent them.

**Finite baseline evidence (2026-09-29):** one registered 60-cell acquisition on the current-body reference tree completed with 48 positive-mass evaluations and 12 zero-PCM controls. The independent saved-artifact audit checked 2,980,800 frame-density values and the intervention → density → score/level → energy → parent-probability path. The equal-RMS body contrast changed that probability; changing the sampled parent was not required. The four research seeds share one fixed acoustic contrast, so they are not four independent acoustic conditions. The final full Rust suite, standard Clippy and formatting passed; earlier test failures remain recorded. See the [evidence report](../../../target/timbre-phase2-controls-20260929/evidence-report.md) and [independent audit](../../../target/timbre-phase2-controls-20260929/independent-audit-v1.json).

This does not close Phase 2. All 12 zero-PCM controls in that original acquisition reached the existing `NoInBandMass` boundary before metabolism or parent selection; their original benefit, cost and parent probability remain unevaluated. The later author-approved limited reference now evaluates the same saved conditions as described below. Author audition, ordinary-runtime acceptance, F2 accuracy/resources, the proposed research script and post-Phase-3 re-acceptance remain open. The test-only reference work introduced no production adoption or Phase 3 start.

The [silence proposal](../../../target/timbre-silence-policy-proposal-20260929.md) is [author-approved for a limited reference implementation](../specs/2026-09-29-body-policy-author-decisions.md): for a verified fully zero representative-radiation condition, retain existing basal and incurred action costs, give neither acoustic recharge nor an extra dissonance charge, and retain parent participation through remaining energy. Score and level stay undefined. The rule does not classify ordinary rests, missing support or nonzero out-of-band sound as silence. The separate `timbre-silence-reference` worktree now implements this rule through the actual Voice lifecycle and onset paths under `cfg(test)`. One saved-input replay retained all 48 positive-mass results exactly and evaluated all 12 silent conditions; the independent checker passed both groups. In this fixture, silent final energy is 0.34950000047683716 and parent probability is 0.3478373399094321. Seven focused tests, the full Rust suite (1,212 passed, 42 ignored), standard Clippy and all-targets check passed. See the [limited-reference evidence](../../../target/timbre-silence-reference-20260929/evidence-report.md). The original acquisition remains unchanged. This is not ordinary-runtime adoption and does not remove the discontinuity between exact silence and very small positive mass.

### Phase 3: Implement the body module contract

Starts after Task 2 is approved, at a point agreed with the body-aware fitness plan. Files follow the contract: expected `src/life/sound_body.rs`, `src/life/sound/events.rs` (`BodySnapshot`, `BodyKind`), `src/life/sound/tone.rs`, `src/life/sound/oscillator_bank.rs`, `src/life/sound/modal_engine.rs`, `src/life/sound/any_backend.rs`, `src/life/modal.rs`, `src/life/schedule_renderer.rs`, the self-model consumers listed in Task 2 question 14, `tests/body_conformance.rs` (new), `examples/timbre_probes.rs`.

The existing F2-representation dependency and integration order remain:
I12b closure/base freeze → I4 wiring → body-aware F3 → timbre Phase 3.
Contract B1 request/evidence identity agreement alone does not settle the
production representation. B7 specifies the required decision record and keeps
runtime accuracy/resource/same-hop admission separate; no fast candidate has
passed that combined gate. Ordinary runtime retains default body-aware OFF and
point evaluation until admission, without treating fallback on a failed opt-in
request as success. Claude's proposal to permit migration without the fast-F2
dependency remains an explicit author amendment option, not a plan change made
by the review response. A render-only option would still retain the ordered
integration point and disabled body-aware runtime consumers, with reduced exit
and later acceptance work stated. Existing isolated-prototype permission does
not authorize integration; this Task 2 revision starts no prototype.

**Exit:**
- A conformance harness in `tests/body_conformance.rs` runs every registered body through: contract responses (drive, stop, tail, re-excitation) with finite output; reproducibility under a seed; agreement of each provided self-model with the body's render within the Task 2 tolerance; the per-block resource bound; genotype capture round-trip and mutation staying in range.
- `Sine`, `Harmonic` and `Modal` pass it. Emitted level matches the cost charged for it (no waveform identity required). The pitch-change policy adopted in Task 2 holds.
- Ecology code no longer branches on body kind or reads body internals; the per-sample `articulate_wave` path, which has no production caller, is removed.
- A minimal test body with no self-model, added only in its own module plus its registration, passes the harness and runs an offline render. This is the proof that a new synthesis method can be tried without touching the ecology.
- Task 1's candidate is re-auditioned on the real synthesizer.

### Phase 4: Temporal contrasts through existing paths

Folded into the T1/T2 terrain work (D10). This section states what those registrations must carry.

**Files (expected):** `samples/research/temporal_grouping_contrast.rhai`, an assay report under `docs/design-notes/`.

**Exit:** stimuli from existing bodies (attack, decay, re-excitation only). Beyond matched mean spectra, level, onset density, modulation and the existing meter output are recorded, since matched spectra alone do not isolate grouping. The report lists, separately, what a person hears, what the analysis detects, and what returns to behavior. Adding a detecting mechanism is a separate plan; "at most one mechanism" limits the work, it is not a claim of sufficiency.

### Phase 5: Heredity of timbre (body-aware F6)

Start condition: body-aware F4/F5 established range, Phase 3 done, and Phase 2 re-acceptance passed. This section is the canonical plan for F6.

**Files (expected):** `src/life/community.rs` (`ParentCandidate` gains the body genotype), `src/life/community/respawn.rs`, `src/core/mode_pattern.rs` (ratio mutation via `jitter_cents`), a spectral-slope perturbation, a research assay with pinned seed.

**Exit (engineering):**
- Children copy the selected parent's persistent body traits, never ringing state or random sequences.
- Heritability is granted per pathway (mutation direction, detector, consumer). Ratios, spectral slope and decays are heritable through the spectral pathway (R/H to movement, metabolism, respawn); nothing is heritable through grouping (D4).
- Hitchhiking controls: pitch-fixed body swap; a run with timbre evaluation disabled; several seeds. The direct contribution of a timbre trait is measured, not inferred from correlation with fitness.

**Research readout (not a pass condition):** the research question in timbre.md, read as a distribution over seeds and initial conditions, and as C's verdict under this listener model.

### Phase 6: Consume listener-side attribution

Starts when the temporal DCC track delivers source attribution from presented audio; the interface is checked at the start of Phase 5.

**Exit:** grouping features (onset coherence, common modulation) enter the observability map with their consumers, including behavior under uncertain or wrong attribution (fusion, split, ambiguity). Generator IDs and true ratios are used only for evaluation. Only then may a feature become heritable through grouping.

## Integration checks for the 2026-09-29 code changes

Covered by tests now: the aversion weight reaches the worker's kernel and field arrays on update without touching `roughness01`/`harmonicity01`; validation (non-finite ignored, clamping, small changes recompute); bit-exact default over all bins; density falls back to uniform when aversion zeroes all mass; presentation-only onsets are excluded from the meter's onset sum.

The four additional checks are implemented and validated in the isolated
`.worktrees/timbre-controls-tests-20261002-v1` overlay: the Rhai setter changes
an actual pitch proposal without changing sensation; `drive_production_meter`
excludes presentation-only onsets; a near-silent habitat Voice still drives the
meter through its commanded accent; and fixed-seed runs reproduce emitted
habitat audio and controls through the real `process_hop` and `AnalysisStream`
feedback path, with a feedback-cut contrast. The final ordinary suite passed
1,169 tests, with zero failures and 36 ignored (2026-10-02). The source and primary
report were rechecked on 2026-10-05; no test rerun was needed.

These are isolated ordinary checks, pending integration at the existing agreed
point. The near-silent fixture stays above `Voice::AMP_EPS` and emits nonzero
audio; it does not establish onset behavior at exact zero. The closed-loop check
compares audio values and ordered control observations; it is not a
cross-platform bit-pattern or device-timing acceptance result.

## Coordination (2026-09-29)

- `docs/roadmap/temporal-dcc/milestones.md` §5 records this plan, the constraints on I-series code, and the T1/T2 fold.
- `docs/design-notes/body-aware-fitness-plan.md` records D8 (F2 row), D9 (F4 row) and F6 ownership.
- `AGENTS.md` (main checkout) carries the composer rule and the body-module rule; sessions started before 2026-09-29 have not read them and need the instruction.
- Running sessions are told explicitly (I11, body-aware, T1/T2); the records above are what later sessions read.
- Execution (routing updated by the author, 2026-10-02): one orchestrating session runs this plan together with I11 and body-aware (`AGENTS.md`, "Multi-agent Orchestration"). `gpt-6.1-sol` is primary: `high` for fully specified execution and `xhigh` for design, contracts, interpretation and reviews. `gpt-6-astra` with `xhigh` is reserved for a concrete unresolved design or review problem, with the reason recorded on the orchestration board before dispatch. Task 1 uses the isolated `.worktrees/timbre-valuation-meter`; Task 2 writes the shared spec in the main checkout. Task 2's separate other-model review and author-adoption requirements remain. The self-model face of Task 2 is settled with body-aware through the spec's section "Points to settle with body-aware", which the orchestrator routes.
- Concurrency (adopted 2026-09-29): pass/fail timing measurements take the exclusive window of `scripts/timing_lock.sh`; others run no cargo, tests or renders meanwhile; I-series work has priority for the window; integration into main is one at a time in the order I12b closure and base freeze, I4 wiring, body-aware F3, this plan's Phase 3.

## Not scheduled

- Learning listener-model layer 3 during a performance (D3; ledger §9.3.56).
- Reading roughness as alarm and salience, a possible material for tension; belongs to a tension-design revision.
- Exposing raw consonance coefficients (D1).

## Review record

- 2026-09-29, user: D8–D11 adopted (coordination with body-aware F2–F4 and T1/T2, code on a branch).
- 2026-09-29, user: D6 and D7 adopted (core and body modules; Task 2 widened to the body module contract, Phase 3 delivers the conformance harness and the migration). Task 2's independent review follows Step 2 (Claude when Astra drafts); the Astra plan review below is a separate review of the plan.
- 2026-09-29, Task 2: body-aware B1–B7 coordination and initial Claude independent review received. The fixed six-document transmission was authorized; contract adoption was not. All fifteen findings are mapped in the response, with A1–A3, extra consumer tolerances/references and F2 entry decisions pending. Revised-draft independent re-review and author adoption remain undone. No Phase 3 start or gate relaxation follows.
- 2026-09-29, Astra (gpt-6-astra, high; no file access, material embedded): **conditional pass**. Two major findings, both addressed here: R1, the determinism constraint contradicted itself (habitat synthesis is ecological input); R2, D5 conflated an audition question with a contract decision. Medium findings addressed: Phase 3 invalidates the Phase 2 pass (re-acceptance added); hitchhiking needs causal controls; heritability per pathway, not per parameter; monophony removed from the Phase 3 exit and made a Task 2 option; excitation unification must not flatten bodies; stimulus confounds (depth, band, shared partials, phase, loudness, order, priming) redesigned; aversion validation on every entry point, recomputation on any stored change, kernel version skew between worker and analysis frames fixed and tested; aversion's sign condition documented; `motion`'s jitter measured (0.03–0.1 cents RMS, not about one cent); success of a causal path separated from the sign of a result; the composer rule now covers founder bodies; F6 ownership unified. Minor: the bit-exact claim is now tested over all bins; the valuation action is renamed `Action::UpdateLandscape`. Full answer: `~/tmp/astra-timbre-20260928/review2_answer.md` (local).

- 2026-10-07, user: C5A purpose-specific realized-PCM references and criteria adopted; new-version Sine/Harmonic single-round closed-amplitude prototype law adopted. Author audition deferred. Actual conformance, capacity, full A4 and Phase 3 entry remain open; the original integration order is preserved.
- 2026-10-07, user: preserve necessary strict precision and improve unnecessary requirements. Keep original F2 and C5A criteria, the finite-zero scalar law and discrete/scientific controls. Intermediate float bits and evaluation order are not universal predictive admission gates; bounded-error alternatives use the original combined output budget. Reuse completed exact evidence without universal reruns. Existing acquisitions, live Sources and integration order remain unchanged.
