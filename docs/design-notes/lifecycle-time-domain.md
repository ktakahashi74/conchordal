# Lifecycle Time-Domain Re-parameterization and the Brain Axis

Status: Implemented 2026-07-10. Phase C verdict: keep the three articulation
cores; retain Seq as the fixed-authored-event life architecture. The original
plan was externally reviewed before implementation; outcomes are recorded at
the end of each phase below.
Scope: two coupled core-design questions, sequenced as one program:
(A/B) replace the rate-based lifecycle knobs with time-domain parameters,
and (C) decide the fate of the three-variant articulation-core enum
("brain" axis) by a strict equivalence assay. Assays precede all core
changes; the measurement framework built for A is reused by C, which is
why the lifecycle work goes first.

## Problem

**Lifecycle knobs.** The ecology surface asks composers to hand-balance
7–8 coupled rates (`metabolism`, `initial_energy`, `energy_cap`,
`action_cost`, `recharge_rate`, `viability_rate`, `dissonance_cost`) with
no perceptual meaning — `metabolism(0.055)` says nothing about how long a
voice survives. Every ecology sample (08, 10, 12) carries the same
imitated numeric block; it is the steepest usability cliff in the
composing API. A script-side preset facade was considered and rejected:
it diverges the vocabulary between API, core config, and telemetry. The
fix belongs in the model, per the project's culture (rhythm redesign,
`harmonic_tension` removal).

**Brain axis.** `AnyArticulationCore` has three variants: `Entrain`
(KuramotoCore — vitality, metabolism economy), `Seq` (fixed-duration
timer, field-deaf, no metabolism), `Drone` (immortal slow sway). The
question is whether `Seq`/`Drone` are degenerate configurations of the
living core (→ unify, delete the enum) or genuinely distinct life
architectures (→ keep; the doc fix framing brain as the articulation-life
axis, orthogonal to the rhythm coupling continuum, is already done).

## Phase A — Lifetime assay (measurement before model change)

Goal: establish the nominal mapping between the current rate knobs and
observed survival time, with error bands, using the JSONL report
(`Death { lifetime_sec, first_k_mean }`) as the observable-death instrument
and a deterministic assay probe for the earlier energy-depletion instant.

- **Nominal contract.** Define `starve_sec` as the energy-depletion time
  at zero field fit, with no attacks, continuous recharge forced to zero,
  and background death disabled. Under the current model its closed-form
  baseline is
  `initial_energy / (metabolism_rate * (1 + dissonance_cost))`.
  Record energy depletion and observable death separately because envelope
  decay delays `is_alive() == false` (`metabolism_policy.rs`,
  `articulation_core.rs`); the current `Death` report contains only the
  latter. Actual dynamics also include windowed continuous recharge,
  per-attack cost and consonance recharge, and
  rhythm-reward-multiplied recharge. The parameter is a contract under
  stated conditions, not a promise about every performance.
- **Stratification.** Sweep the knob space and report error bands by
  phonation regime (`sustain` versus re-attacking), coupling preset
  (`metric` / `entrained` / `flow`), attack density, consonance trajectory,
  and respawn setting.
- **Background death** (`respawn_background_death_rate`) can dominate
  nominal starvation; exclude it or label it in every measurement.
- **`energy_cap = 1` pre-checks.** Before fixing the cap as
  normalization, verify the two consumers of raw energy/cap:
  vitality `(energy / energy_cap)^0.5` (articulation_core.rs) and
  hereditary respawn parent selection, which weights by raw energy
  (population/respawn.rs). Fixing the cap makes parent fitness
  comparable across groups but removes "larger cap as lineage
  advantage"; confirm no sample or assay depends on that.
- **Determinism.** Fixed seeds, fixed control-update mode, golden
  summary outputs — the assay must be replayable, not auditioned.
- Deliverable: `samples/research/` assay script(s) + a closed-form nominal
  mapping (rates ⇔ endurance/recovery seconds, with discrete attack effects
  reported separately) and measured error bars. The assay stays in the tree
  as the regression instrument for Phase B.

### Phase A outcome

- `samples/research/lifecycle_time_domain_assay.rhai` is the fixed-seed
  regression instrument. Configured endurances of 2, 4, and 8 seconds produced
  energy-depletion times of 2.000, 4.001, and 8.001 seconds and observable
  deaths at 2.005, 4.011, and 8.011 seconds in the reference run. Error stayed
  within one control step plus floating-point accumulation.
- The assay exposed an existing telemetry bug: `lifetime_ticks` counted
  substeps but was multiplied by hop duration, inflating lifetime eightfold in
  this configuration. Reports now derive observable lifetime from birth/death
  frame distance and accumulate energy-depletion time directly in seconds.
- `energy_cap` was not an independent life parameter. Vitality was its only
  normalization consumer; hereditary parent selection is group-local, so it
  never compares differently configured caps across groups. Moreover, runtime
  recharge did not actually clamp raw energy to the advertised cap. Energy is
  now genuinely clamped to `[0,1]`, making vitality and hereditary fitness
  comparable without a composer-facing cap.
- Closed-form tests cover zero-fit endurance, fit-dependent penalty shaping,
  full-signal recovery, discrete attack cost/recharge, and control-step error.
- Stratification does not require separate endurance maps. Phonation/coupling
  presets affect survival only through their observed attack/recharge trace;
  respawn policy acts after death, while background turnover is an independent
  hazard. These remain report dimensions around one nominal contract rather
  than family-specific parameter definitions.

## Phase B — Core re-parameterization (time domain)

- `LifecycleConfig` stores time-domain parameters:
  - `endurance_sec` — nominal survival at zero fit, no attacks
    (replaces the metabolism/initial_energy pair as the primary dial). With
    normalized initial energy and cap, derive
    `basal_cost_per_sec = 1 / (endurance_sec * (1 + dissonance_penalty))`
    so changing the penalty shape does not silently change the zero-fit
    endurance contract.
  - optional `recovery_sec` — time to refill normalized energy from 0 to 1
    at a continuous-recharge signal of 1, with basal drain and attacks
    disabled; when present, derive
    `continuous_recharge_per_sec = 1 / recovery_sec`. Absence disables
    continuous recharge without a numeric sentinel. This parameter does not
    govern the discrete per-attack economy.
  - `consonance_viability(low, high)` — unchanged (already perceptual,
    level-domain).
  - `energy_cap` fixed to 1 as normalization (pending A's pre-checks).
- Runtime rates are derived **once** in `AnyArticulationCore::from_config`,
  the current lifecycle-to-runtime boundary during `Voice` construction.
  The core keeps integrating rates; the *stored and reported* vocabulary
  is seconds.
- API follows the core 1:1:
  - Core tier: `endurance(sec)`, optional `recovery(sec)` (naming TBD), the
    viability window, and the respawn family. Both time values must be finite
    and strictly positive when configured.
  - Mechanism Tuning: dimensionless normalized-energy fractions and shape
    controls. `attack_cost_fraction` and `attack_recharge_fraction` are
    fractions of the normalized capacity applied per full-strength attack;
    `dissonance_penalty` shapes basal drain. These discrete attack terms are
    intentionally independent of `recovery_sec`.
- Telemetry reports both the configured nominal endurance and the
  observed `lifetime_sec`, so the composing loop closes: "endurance 8 s,
  measured 2 s in section IV" is a direct read from the report.
- Alpha no-compat policy: old keys and verbs removed, no aliases;
  curated samples rewritten; generated docs regenerated; technote
  Appendix A (key system parameters) updated in the same change.

### Phase B outcome

- The public lifecycle surface is now `endurance(seconds)`, optional
  `recovery(seconds)`, `attack_cost_fraction(value)`,
  `attack_recharge_fraction(value)`, `consonance_viability(low, high)`, and
  `dissonance_penalty(value)`.
- `initial_energy`, `energy_cap`, `metabolism`, `recharge_rate`, `action_cost`,
  `viability_rate`, and `dissonance_cost` were deleted without aliases. Energy
  begins at 1 and is clamped to the normalized domain.
- `AnyArticulationCore::from_config` derives runtime rates once. JSONL death
  records report `configured_endurance_sec`, `energy_depletion_sec`, and
  observable `lifetime_sec` separately.
- Curated samples, research assays, Rhai definitions, the API book, and both
  technote languages use the time-domain vocabulary.

## Phase C — Brain-axis equivalence assay

Reuses Phase A's measurement framework. "Equivalent" means **observable
equivalence across every contract**, not audible similarity:

- `ArticulationSignal` time series (amplitude, active, relaxation,
  tension) under identical seeded rhythm/landscape/phonation traces.
- `is_alive()` / death timing, including release-to-idle.
- `apply_phonation_onset()` side effects (Seq resets its timer —
  articulation_core.rs; a degenerate Kuramoto must reproduce this).
- Render modulator output: the enum has distinct `EntrainPulse` /
  `SeqGate` / `DroneSway` render variants.
- Telemetry: PLV is `None` for non-entrain cores.
- Population consequences: hereditary respawn weights parent energy for
  `Entrain` only; other cores contribute 0.0.

Verdict is two-level:

1. **Exact behavioral equivalence** across all of the above → unify the
   core (enum deleted, 3 cores → 1 core + config), following the
   `harmonic_tension` precedent: deletion justified by proven
   redundancy.
2. **Audible-only equivalence, or divergence** → keep the enum as three
   honest life architectures; the brain doc fix completes the work.

**Guard — the disguised enum.** If unification requires flag
proliferation (`hard_ttl`, `sway_lfo`, `death_mode: Never`, ...), the
unified core is a disguised enum and the unification is rejected: that
is config archaeology, not simplification.

**Seq audition.** Independently of equivalence, audition `brain("seq")`
in one research sample as a compositional intent (a fixed authored event
amid living material). If no role emerges, prune it directly as a v0.5
candidate — do not launder the deletion through unification.

### Phase C outcome

- `brain_axis_contract_assay_rejects_exact_unification` compares onset side
  effects, signal behavior, death rules, and render-modulator variants. The
  cores diverge observably: Seq resets a hard timer and emits `SeqGate`; Drone
  is immortal and emits `DroneSway`; Entrain owns normalized energy, envelope
  release, PLV, and `EntrainPulse`.
- Verdict: **keep `AnyArticulationCore` and all three variants**. Reproducing
  these contracts in one core would require the rejected disguised enum.
- `samples/research/brain_seq_authored_event.rhai` establishes Seq's concrete
  composing role: a fixed authored event inside living material. Events placed
  at 2, 4, and 6 seconds each died after 1.003 seconds while the Entrain
  population continued; the offline render completed with the three distinct
  insertions. Seq is retained.

## Deferred (recorded here on purpose)

The brain enum is plausibly a discrete stand-in for a **life-coupling
continuum**: terrain (coupling 0, immortal) ⇔ living voice (coupling 1,
mortal), with interpolations such as partially field-coupled vitality
and *mortal terrain*. Do not build it now. The concrete second use case
arrives with the habituation field (technote §9.2 — terrain that erodes
under sustained excitation is exactly a mortal drone). Revisit this note
when that work starts.

## Non-goals

- Script-side preset facade over the rate knobs (rejected: vocabulary
  divergence between API, core, and telemetry).
- Scenario-level state queries / `wait_until` (held separately; not part
  of this program).
