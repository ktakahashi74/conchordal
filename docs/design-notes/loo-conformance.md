# Render-derived leave-self-out conformance

## Current PCM contract and composer-selected pitch view (VR-1)

Production removes the complete Voice habitat PCM, including release tails,
before the causal auditory front end. The default remains unchanged:
`tail_roughness_aversion(0)` uses `C(H_external, R_external)` for pitch as on
main `105e3cde`. The shared observation and environment recovery always retain
that excluded view. The author adopted `.tail_roughness_aversion(w)` as an
initial/live Population **Author rule**, finite [0,1], on 2026-11-02 evening.
For eligible held-Tone Off/On replacements only, positive w evaluates
`C(H_external, (1-w)*R_external + w*R_mixed)` at the same frame and grid with the
existing a/b/c/d and sigmoid. One reproduces the prior accepted VR-1 choice;
zero bypasses view construction and retains the original audio, events and RNG
path. No roughness difference is treated as additive self-roughness. Mixed R
includes existing tails and the outgoing Tone, without harmonic self-support.
Invalid values warn and use the existing sanitize01 rule (finite clamp, +infinity
1, -infinity/NaN 0). With self-exclusion false, positive w is inert and warns once
per Population. No listener-profile default enables the word.

Hearing roughness belongs to sensation; choosing whether to avoid it belongs to
the composer's disposition (DCC §5). In VR-2, the author described Undergrowth B
as “クリアで濁りがすくない”. This is attributed author audition, not a universal
preference or a new sensation law.

The candidate uses the existing final target projection and application rules:
unchanged targets do not create an On; Lock, inclusive `rearticulate_above` Glide,
GateSnap, unavailable onsets and independent deferred participation retain their
previous field. Shared harmonic opportunities use their own legal Off/On gate.
Missing/gap/reset observations preserve the existing fallback and do not reuse
stale view storage. Identity includes Voice generation. The scratch scan is built
only at a positive-weight eligible proposal, retains capacity, and does not mutate a shared LOO
snapshot. Habituation, support terms, private adaptation, candidate generation,
RNG, temperature, release, routing and Flora are unchanged.

At weight one, `rearticulation_mixed_roughness_matches_independent_pcm` independently renders an
external source, the outgoing Tone, zero or two old tails, and a new Tone. It
checks actual overlap, habitat/presentation routing, full PCM exclusion, the
same-history H/R reference, the tails-only counterfactual, fixed and native
candidate decisions, generation mismatch and reset/missing observations. The
registered score/level gates remain 0.025 / 0.0125, with no reversal at reference
score gap 0.1 and the existing 1e-6 score tie rule. The new fixture's maximum
errors are 0.000238 / 0.000113. Without existing tails, the independently
calculated outgoing-Tone control changes 11 of 21 fixed candidate pair rankings
and 5 of 48 native decisions; it was not replaced by a chosen tie case.

The earlier weight-one acquisition used the unchanged HV-10 ref/demo, seed 1128749181177961461, 354.016 seconds; the
old/new-Tone primary pair-seconds fell from 5705.451 to 5319.563 and from 5992.331
to 5758.752. Both existing amplitude-floor sensitivities and the overlap-time
normalized measure passed per scene. These are registered operational criteria,
not audibility thresholds or evidence that all beating disappeared. The full
B-1 matrix and environment consumer references remain separate controls.

The composer-control checks add exact default/explicit-zero WAV bytes, native
events and pitch RNG continuation, including a separate baseline build at
105e3cde; intermediate weights check monotonic R in both directions, unchanged
external H and shared snapshots, validation warnings and live patches. Previous
ABBA evidence applies to the identical weight-one endpoint. Zero retains the old
path, so the author requested no new timing acquisition. Intermediate weights
have implementation checks; no separate performance or audition claim is made.

## Historical comparators and their registered gates

`src/life/pitch_loo_conformance_tests.rs` ports the B-1 reference into the current
crate. It subtracts independently rendered own PCM from the mixture **before
the auditory front end**, checks the result against independently rendered other-only PCM, and
compares three retained methods: the original harmonic subtraction (`approx`),
single-bin removal (historically `exact`), and the T-31 spectral-ownership
candidate. These labels identify the compared implementations, not accepted
contracts.

The test covers 15 families and 84 scenes. It keeps B-1's seven fixed
candidates, actual Tone-derived body weights, greedy decision, paired Metropolis
seeds, and native candidate generation. Both fixed and generated candidates use
the production selection function. The T-31 projection calls
`Voice::project_loo_power` with the fixture's body controls and an active unit
articulation signal. It does not claim coverage of live articulation history or
ownership/analysis latency. Rendering uses the default Tone path; the optional
phase3 renderer is outside this fixture.

Run with the repository's required Cargo wrapper and resource limits:

```sh
"$CARGO" test --offline --lib rendered_loo_conformance -- --nocapture
```

`T32_CASE` JSON lines contain all scene/method witnesses, including scores,
paired decisions and current-position metabolism differences. `T32_COMPLETE`
contains the per-family verdicts. `T32_PILOT=1` selects only sine 440 Hz with
another sine at 466 Hz; this checks producer operation, not family acceptance.

A family passes only when all scenes meet the existing score/level gates
(0.025 / 0.0125), have no rank reversal at a reference gap of at least 0.1, and
satisfy the consumer decision rule below. Point and body-weighted values are
checked separately. There is no numerical metabolism acceptance threshold in B-1;
its recorded differences remain descriptive.

## Consumer ties (T-33)

The production acceptance comparison already uses a score tie width of `1e-6`.
When all seven reference candidate scores fit within this width (`max - min <=
1e-6`), any selected candidate is an acceptable decision. The shortcut must then
also have `max - min <= 1e-6`. For native point selection the same width must also contain every score actually
queried by the paired native proposals. Only when both reference sets are tied
and both shortcut sets are flat is selected-target identity waived. The numerical score/level and rank gates
still apply. Point and body-weighted ties are evaluated independently. In every
other scene all paired fixed-candidate decisions and native point decisions must
agree as before. This rule changes neither the production scorer nor its RNG.
Witness metrics retain both fixed-candidate spreads and tie flags, the native
score extrema, and all original decisions, including permitted differences.

Rejudging the saved T-32 witnesses with this rule accepts all 78 non-motion scenes
(14 families). Of the 84 scenes, 28 have tied point and body reference scores; 26
of those belong to the non-motion families. The six motion scenes still fail the
numerical gates. These are results for the saved source, not acceptance of a
motion repair.

`T33_MOTION_ONLY=1` selects the six motion scenes for a bounded repair check;
`T33_TIES_ONLY=1` selects the 28 solo scenes to check native tie coverage.
The subset selectors cannot be combined. The full acquisition still contains 84
scenes and 252 method witnesses. Acceptance now requires the **same T-31 method**
to pass every family; selecting a different method per family is forbidden.
The test emits the entire requested matrix before asserting acceptance and runs
in the ordinary test suite. A failed acceptance assertion must not be changed
into a passing regression test for the rejected behavior.

## Adopted projection and finite conformance

The T-33 acquisition passed all 15 families and 84 scenes using the same
spectral-ownership method. Its 28 tied scenes passed the flatness requirement;
the other 56 scenes retained all paired fixed and native decisions. The fixture's
motion value remains `0.9`, as in B-1; this is not a sweep of every motion setting.

The default Tone oscillator applies an additive common frequency offset to every
partial: `delta_hz = f0 * 0.02 * motion * sin(phi)`, with `phi` advancing at 5 Hz.
For phase averaged over a cycle, the frequency offset has the arcsine distribution.
For a partial at `f`, deviation `d` and bin edges `lo, hi`, its power fraction is
`(asin(clamp((hi-f)/d,-1,1)) - asin(clamp((lo-f)/d,-1,1))) / pi`.
The amplitude projection adds the square root of this fraction times the partial
amplitude because the ownership path squares amplitudes. Bin edges are halfway
between Log2Space centers. No spread coefficient or quadrature count is fitted.
At zero deviation the ordinary static projection applies.

Static partials now occupy the nearest bin, matching the front end's representation
of each resolved peak. Keeping the former two-bin interpolation left a false
other-source share beside the true peak in the 1800 Hz unison motion scene.
The periodic spread alone passed five of six motion scenes; aligning the common
static projection to peak bins passed all six. Both changes belong to body
projection; the ecological ownership formula has no family branch.

The cycle average is a fast approximation to the finite-window spectrum, not an
exact reconstruction of FM sidebands, pink jitter, or the current Tone phase.
Its adequacy here is established by the unchanged PCM/NSGT consumer comparison
on the registered finite windows. Live articulation histories, different analysis
settings and the opt-in phase3 renderer remain outside this acquisition.

Production pitch movement exposes only `leave_self_out(bool)`. Historical
single-bin removal remains compiled only for tests and the `loo-diagnostic`
feature, so the retained three-way conformance comparison stays reproducible.
The public mode selector and its enum/configuration patches are removed.

Environment viability uses the same spectral-ownership score at the current
pitch as movement. T-34 applies T-33's existing `1e-6` tie width to differences
between absolute errors: a candidate error increase no larger than this width
is equal, not worse. This is the existing tie rule, not a new tolerance.
Rejudging the saved T-33 values passes all 84 scenes for point score/level and
body score/level. The largest point-score increase is `6.092173e-14`; the other
three quantities never increase. No new acquisition was needed for that decision.

The ownership snapshot is prepared when either pitch self-exclusion or
`viability_scope("environment")` needs it, including stationary Voices and both
pitch cores. The score is cached with movement within the hop. Missing ownership
retains the observed field. The ordinary mixed-field level used for basal drain
and telemetry, and the optional PCM body-metabolism evaluator, keep their existing
semantics; the environment scope selects the continuous-recharge score.
The obsolete harmonic-count API and production partial-score subtraction are
removed. Historical subtraction remains only in tests or `loo-diagnostic`.
The conformance suite exercises the lifecycle consumer at the current pitch
against the same PCM self-removal reference and retained approximate control.

The full T-33 matrix was acquired once. A later focused check of the 28 tied
scenes recorded all native scorer extrema, closing the coverage gap between the
seven fixed candidates and the larger native candidate set. All 28 passed;
the largest native shortcut spread was `1.102e-12`, and every native reference
spread was zero. The scores, decisions and metabolism values already present
in the full acquisition were unchanged. No second full matrix was acquired.
