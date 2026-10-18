# Render-derived leave-self-out conformance

`src/life/pitch_loo_conformance_tests.rs` ports the B-1 reference into the current
crate. It subtracts independently rendered own PCM from the mixture **before
NSGT**, checks the result against independently rendered other-only PCM, and
compares three retained methods: the original harmonic subtraction (`approx`),
single-bin removal (historically `exact`), and the T-31 spectral-ownership
candidate. These labels identify the compared implementations, not accepted
contracts.

The ignored test covers 15 families and 84 scenes. It keeps B-1's seven fixed
candidates, actual Tone-derived body weights, greedy decision, paired Metropolis
seeds, and native candidate generation. Both fixed and generated candidates use
the production selection function. The T-31 projection calls
`Voice::project_loo_power` with the fixture's body controls and an active unit
articulation signal. It does not claim coverage of live articulation history or
ownership/analysis latency. Rendering uses the default Tone path; the optional
phase3 renderer is outside this fixture.

Run with the repository's required Cargo wrapper and resource limits:

```sh
"$CARGO" test --offline --lib rendered_loo_conformance -- --ignored --nocapture
```

`T32_CASE` JSON lines contain all scene/method witnesses, including scores,
paired decisions and current-position metabolism differences. `T32_COMPLETE`
contains the per-family verdicts. `T32_PILOT=1` selects only sine 440 Hz with
another sine at 466 Hz; this checks producer operation, not family acceptance.

A family passes only when all scenes meet the existing score/level gates
(0.025 / 0.0125), have no rank reversal at a reference gap of at least 0.1, and
all paired selected targets agree. Point and body-weighted values are checked
separately. There is no numerical metabolism acceptance threshold in B-1;
its recorded differences remain descriptive.

The first current-source acquisition rejected all three methods for eleven
families. The T-31 candidate passed only `adsr_slow`, `harmonic_inharmonic`,
`harmonic_spread`, and `modal_spread`. In ten rejected families the candidate
passed the fixed-candidate value gates but changed native Metropolis selections;
`motion` also failed the value gates. No production method was adopted by this
acquisition. The test deliberately fails after emitting the complete matrix if
any family has no conforming method. Do not turn that assertion into a passing
regression test for the rejected behavior. It remains ignored in ordinary test
runs until the unresolved method decision is implemented and accepted.
