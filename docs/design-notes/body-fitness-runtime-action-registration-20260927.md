# Body fitness normal-runtime action registration (2026-09-27)

Status: implementation and functional validation, isolated `body-fitness-action` worktree. This registration precedes normal-render acquisition. Main source adoption and real-time performance acceptance remain separate.

## Contract

`body_fitness_action = true` enables source-removed observation and representative-body local non-ratio pitch scoring. Default false preserves the existing path. Supported analysis dimensions are 48 kHz, 512 samples, at most four simultaneously observed sources. Unsupported candidates or unavailable environments defer pitch proposals; articulation, elapsed time, and lifecycle continue. No instrument audio disk path is added.

Four persistent candidate workers are created during wiring. Input/output capacity is one each. Each worker caches candidate densities, with 256-entry and 1 MiB entry-storage bounds. Candidate recipes are reconstructed from the actual Voice's body, envelope, and representative modulator, with fixed amplitude 1, hold 48,000 samples and 72 observation hops. These are representative onsets, not predictions of actual emitted audio. Candidate recipe identities omit only the live base frequency; each candidate frequency remains exact.

Source-removed environments must have the complete current source set, matching identities, support no later than the decision, and age at most 4,800 samples. Fresh ecology habituation is applied before scoring. Ready densities are retained until consumption, scored again each hop, and validated against current source, body generation, route, control, target, RNG, analysis space, and habituation version. Successful or rejected consumption cannot fall back to point scoring in another control substep of that hop.

Offline render explicitly blocks for candidate completion for deterministic functional verification. Instrument mode polls without waiting. This registration makes no real-time deadline or throughput claim.

## Registered checks

- Config defaults off; invalid sample rate/hop rejected.
- Normal `conchordal-render`: seeded short Sine scene with one and four voices, local non-ratio hill-climb; actual prepared decision consumption must be positive.
- Missing flag versus explicit false: identical audio and ordinary behavior records.
- Two identical ON runs: deterministic audio and action counters, excluding wall-clock diagnostics.
- Voice-level strict validation refusal, recovery, once-only consumption across substeps, and default-off behavior.
- Observer identity, freshness, and source-capacity refusal remain covered; add decision-boundary tests if needed.
- Final cargo fmt, standard Clippy, all-target check, full cargo test with backtraces/nocapture and true exit record. No parallel timing acquisition.

Initial spawn, parent-present PeakBiased birth, broader candidate modes, long-run performance, and author adoption are not completion claims of this step.

## Additional default-off release regression (before acquisition)

After the implementation is frozen, rebuild the normal release renderer and repeat the four v10 regression cases against their frozen v6 comparators: I4 bounded Sine and both-on Sine/Harmonic/Modal, four voices each. Reuse the existing scripts, configs, comparator and record-type set without changing acceptance criteria. Pin the current source and binary hashes in a new v11 plan before execution. This is an output regression, not a throughput/deadline measurement. Modal's prior zero memory evidence remains an insufficiency even if the output comparison passes.
