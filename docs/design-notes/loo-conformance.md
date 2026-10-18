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
The test emits the entire requested matrix before asserting acceptance and stays
ignored in ordinary test runs. A failed acceptance assertion must not be changed
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

Environment viability keeps its legacy partial-score subtraction. At the current
position, the new point **level** error is no worse in all 84 scenes, but its point
**score** error is greater in 15 scenes, with tiny positive residuals where the old
method was zero or closer to zero. The adopted criterion requires no worsening
in every scene and provides no extra tolerance for this comparison. Thus viability
is not switched, and `leave_self_out_harmonics` remains used for that consumer.

The full T-33 matrix was acquired once. A later focused check of the 28 tied
scenes recorded all native scorer extrema, closing the coverage gap between the
seven fixed candidates and the larger native candidate set. All 28 passed;
the largest native shortcut spread was `1.102e-12`, and every native reference
spread was zero. The scores, decisions and metabolism values already present
in the full acquisition were unchanged. No second full matrix was acquired.
