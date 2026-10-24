# Flora: a Field-Form Inhabitant

Status: implemented, with beat opportunities for every Flora, tempo-derived pace and finite retirement, 2026-10-24. The adopted design is draft 14: the rendering
of draft 12 and the consumer–resource dynamics of State and Dynamics. Current paths
and the historical branch are listed in [Reference Implementation](#reference-implementation).
The DCC route for the drive of temporal evolution, a listener-side information terrain,
is closed (technote ledger §9.3.93); the dynamics stay authored.
Scope: a new composition mode in which sound is a spectral mass that grows over the
consonance landscape, instead of a set of Individuals that each carry a fixed body.
Flora coexists with Populations in the same habitat and couples to them only through
sound.

## Problem

An Individual today carries a body fixed at spawn: a partial set (`modes`) plus
`brightness` / `spread` / `unison`. The terrain decides where the fundamental goes and
when the Voice sounds; it does not decide what the sound is made of. The timbre note
already names this gap: "A voice's timbre is chosen by the composer from a preset
vocabulary" (`docs/design-notes/timbre.md`). Even `landscape_density_modes()` picks a
fixed number of partials once, at spawn.

The terrain also suggests a different form. Per bin it states how favourable it would
be to add sound there, given what is sounding:

- the roughness kernel is documented as "the potential roughness increase from adding a
  unit pure tone" (`src/core/roughness_kernel.rs`);
- the harmonicity kernel is a common-root projection on the log2 axis
  (`src/core/harmonicity_kernel.rs`).

An Individual is a particle climbing this terrain. Flora instead lets a spectral
distribution evolve on it. Partials, their spacing and their succession emerge instead
of being chosen from a preset.

Flora adopts the terrain as an **authored local fitness**. It does not claim that the
terrain is the gradient of a spectrum-wide objective, and it claims no monotone
improvement of any objective. Growth on that fitness is limited by a slowly renewing band
resource, so that the spectrum keeps succeeding itself instead of settling (State and
Dynamics).

Flora is an additional composition mode, not a replacement. It must be usable alone and
together with Populations in one scenario.

## Concept

- **Populations** are mobile fauna: discrete Individuals that move, sound and die.
- **Flora** is vegetation in the same habitat: a mass that grows where the terrain is
  favourable, spreads, exhausts the band resource where it grows, and withers.

Name: `flora`. The word `field` is avoided because it is already taken by
`consonance_field_*`, `SpawnStrategy::Field`, `FieldTarget`, `FieldSampling` and the
habituation field.

## Coupling Contract

All Flora shares the composer's `temporal_basin` and `meter_stability` priors
with Voice metric timing. Every Flora uses the private clock described below.
Only the habitat audio-only meter supplies beat evidence;
internal commands never supply a beat. Ecological pace uses the private clock's
confidence-weighted frequency, excluding its phase correction. A stale, unavailable
or unconfirmed observation uses the basin center, or 2 Hz without a basin.


The principle requires causal, local temporal evidence with uncertainty, and separates
prediction, action and observed consequences. It does not require universal beat
synchronization or make an unpredicted onset defective. The author chooses the existing
T1 rule for every Flora, without an opt-out: the Population metric CouplingClock with
coupling 0.95 has a synchronization tendency, not a neutral response. Native pace and
ecological preferences remain authored embodiment. The clock reads tactus only;
bar selection, phrase structure and onset-overlap preferences are not implemented.

Flora's actual habitat sound may confirm a beat and then support its own opportunities.
This acoustic feedback is allowed. Priors and internal commands alone cannot supply
confirmation; absence of confirmation in Flora-only audio is not guaranteed.

1. **Only through sound.** Flora's audio enters the buses selected by its routing. The
   habitat bus feeds NSGT and the landscape; Populations perceive Flora there. Their
   audio reaches the same landscape, and Flora perceives them there. There is no direct
   state coupling in either direction. This is the contract Voices already follow.
2. **Flora reads the base view before habituation.** It reads
   `consonance_field_level` and `harmonicity01` per hop and, at placement,
   `consonance_density_mass`. Habituation does not modify `harmonicity01`.
   Consumer–resource depletion and recovery already express fatigue of a habitat band.
   Reading the eroded view would count that fatigue twice, including habituation
   driven by Flora's own sound. The adopted dynamics constants were selected on the
   reference branch without habituation. The 2026-10-22 diagnostic (F-7) isolated the
   extra erosion, and the author adopted the base view after listening that day.
   Global habituation remains enabled by default: Voices and the listener still read
   eroded views, and Flora's habitat audio still drives habituation.
3. **Flora perceives itself in C/H.** Base consonance and harmonicity include its
   own sound. Only the external resource-consumption input excludes its actual
   habitat PCM. Self-interaction in C/H shapes the pattern:
   - its own roughness lowers fitness within a critical band of each sounding peak;
   - its own harmonicity raises fitness and seeds at common-root relatives.

   It is not what keeps Flora moving: in the reduced model the succession continues on
   a fixed terrain.
4. **Routing** follows `send()` as for Populations; the default is both buses.
   - `send(habitat_bus)`: a living, unheard terrain. The ecology responds to it; the
     listener does not hear it.
   - `send(presentation_bus)`: heard decor. Flora still reads the landscape, but its own
     sound is absent from it, so it cannot perceive itself. The succession continues on
     the terrain made by others.
5. **Self-sound separation of Voices.** Flora is part of every Voice's surroundings.
   The renderer builds each Voice's own-sound history from the Voice's own body and the
   whole habitat buffer (`sound.history.process`, `src/life/schedule_renderer.rs`).
   Flora is therefore added to the habitat buffer before that call. It never enters a
   Voice's own body or own-habitat buffer (see Rendering).
   Pitch and environment self-exclusion subtract that Voice's actual habitat PCM
   from the same mixed hop before NSGT and the nonlinear analysis pipeline.
   The result therefore includes habitat-routed Flora, including residual sound.
   Its frame, Voice ID and generation travel together; both consumers reuse the
   same observation and the ordinary shared habituation field.
6. **Predictive terrain.** The `GeneratorModel` (`src/life/generator_model.rs`)
   extrapolates the perceived `consonance_field_level_eff` history it is fed in
   `apply_landscape_updates` (`src/runtime/mod.rs`). Habitat-routed Flora reaches that
   prediction indirectly, through the perceived history. Flora publishes no explicit
   forecast of its own future spectrum.
7. **ListenerTwin** receives the presentation chunk, which contains presented Flora.
   Nothing is added on the listener side.
8. **Invariance.** Flora's empty render/dispatch path preserves the four frozen
   no-Flora cases in Tests 1. The independently adopted Voice self-exclusion
   PCM self-removal and silence-floor changes intentionally affect existing no-Flora
   samples that use self-exclusion (the Changelog's Unreleased section distinguishes
   the A-to-B comparison from earlier changes); the four hashes are
   not a claim that every older sample output is unchanged. The shared floor also
   applies to general analysis without Flora or self-exclusion consumers, so even
   those paths have no general invariance guarantee for tiny positive inputs.

## Closed-Loop Caveats

These are recorded as caveats, not solved by the design:

- **Delayed self-perception.** Flora perceives its own sound through the analysis path:
  - the worker keeps the landscape at most `MAX_LANDSCAPE_LAG_FRAMES` (= 1) hops behind
    generated audio (`src/runtime/mod.rs`);
  - NSGT band integration adds further delay, most in low bands.

  The neutral growth time scale (`1/rate` = 0.125 s) is above this, and the reduced
  model is insensitive to up to 213 ms of added delay. That does not prove stability,
  because the loop is discontinuous (next item). Stability is checked by the closed-loop
  assays in Tests, including a low-register extent and two overlapping Flora.
- **External power also has analysis latency.** Each Flora subtracts only its own
  habitat PCM before FFT. Initial windows and post-gap windows are unknown; an
  available old consumption is a timestamped hold approximation. The resource law
  can suppress total target mass as well as move peaks. Low occupied fractions alone
  do not demonstrate audible succession; absolute target mass and five-second
  Jensen–Shannon distance must be read together.
- **Voice self-exclusion follows PCM subtraction.** Each reserved Voice has an
  independent analysis history for `mixed habitat PCM - own habitat PCM`.
  At birth, before the first own sound, it copies the mixed stream's complete
  state. Populations with a later enabling patch are reserved from birth too.
  Gaps reset both histories and invalidate the result for the common one-window
  warmup. New geometry requires fresh analysis storage; old-geometry observations
  are discarded. IDs and generations prevent observations from surviving replacement.
  The existing shared near-silence floor and all consumer criteria are unchanged.
  B-1's 84 cases and all 40 Flora cases, including nearby habitat residual sound,
  are assertions. These finite fixtures are regression coverage, not a universal
  floating-point theorem about arbitrary scenes. Case records remain available
  under `FLORA_LOO_REPORT_DIR`.
- **Latent mass versus sounding spectrum.** The biomass `v` is latent. What sounds is
  the tonal mass of held peaks, plus the residual when it is on.
  The design narrows the gap three ways:
  - the cell decomposition renders both tonal and residual mass, so diffuse mass is heard
    rather than dropped;
  - tonal candidacy has hysteresis;
  - consumption is driven by all biomass, so biomass that does not sound cannot
    accumulate unchecked.

  A change of dominant peak within a cell can still switch the acoustic input. This is
  the main closed-loop risk, and the assays measure it as flapping.
- **A future `pred` self-term.** If delayed self-perception has to be replaced by
  Flora's own state, Flora's perceived contribution must first be removed from the
  terrain it reads. Otherwise it would be counted twice. That requires per-source
  separation of Flora's C/H landscape, which the consumption-only power observer
  does not provide and which remains out of scope.
- **Temporal observation.** With `temporal_mode("observe")`, temporal observation sees
  Flora's audio. Births and deaths are slewed (see Rendering) so that steady Flora is
  not a train of attacks. Whether the existing onset detector stays quiet on Flora
  alone is measured, not assumed.

## State and Dynamics

Adopted by the author on 2026-10-21. The reasons, the reduced-model evidence and the
closed-loop measurements are in [Dynamics Design Record](#dynamics-redesign) and
Implementation Findings. Every part of it belongs to the authored-rule tier.

**State.** Each Flora instance holds:

- an **extent** `[lo_hz, hi_hz]` kept in Hz, and its bin range on the current Log2Space
  (runtime default 55–8000 Hz, 96 bins/oct);
- a biomass scan `v[x] ≥ 0` (`biomass_mass_scan` in the reference implementation): mass
  per bin, zero outside the extent. Its total is not conserved and is never
  renormalized; silence is a legitimate state;
- a resource scan `u[x] ∈ [0, 1]` (`resource_state_scan`);
- an amplitude `A(t) = amp × fade_gain(t)`;
- the cells and lanes (see Rendering);
- the consumption kernel and fixed-capacity scratch (see Allocation).

Flora reads `g = consonance_field_level` and `H01 = harmonicity01` per hop on its
extent, and `consonance_density_mass` at placement, all before habituation erosion.

**Ecological time.** `f` is the private CouplingClock velocity before phase correction.
Without fresh confirmed evidence it returns to the tempo-basin center (default 2 Hz).
The adopted authored mapping is `p = f / (2 Hz)`. Advance the entire consumer-resource
system in `dt_ecological = p * dt_physical`: growth, diffusion, seed rain, resource
recovery and both forms of consumption use this same time. Thus default growth is
`4 f`, renewal is `20 / f` seconds and consumption is `0.6 / f` seconds. The ratios
come from the adopted 8/s, 10 s and 0.3 s at the existing neutral 2 Hz, not a claim
about neural rhythms or a new bar/phrase model. Substeps are recomputed each hop
from ecological time with the same positivity bounds.

**Update.** Each hop runs `n` equal substeps, `dt = p * dt_hop / n`, with the
terrain held for the hop. `n` is the smallest integer with `rate * dt <= 0.05`
and `D * dt <= 0.25`, with fixed neutral `rate = 8` and D in bins² per ecological
second. It is recomputed each hop, including after a pace or space change.
Each substep:

```
v ← v + dt · [ rate · v · (g · u · v / (v + v_a) − k)     growth, Allee term, metabolism
             + D · ∇²v                                     spread
             + u · (ι_bg + ι · H01) ]                      seed rain
c_b[x] = Σ_y Ĝ(x, y) · v[y] · bins_per_oct / c_ref
z[y] = p_external[y] / 0.00045
c_e[x] = Σ_y Ĝ(x, y) · z[y] · bins_per_oct / c_ref
c[x] = c_b[x] + c_e[x]
u ← u* + (u − u*) · exp(−a · dt),  a = 1/τ_r + c/τ_c,  u* = (1/τ_r) / a
```

- The biomass update reads only old values. The spread has reflecting ends:
  `v[lo−1] := v[lo]`, `v[hi+1] := v[hi]`.
- `Ĝ` is a Gaussian in ERB-rate (σ = 0.5 ERB, truncated at ±3σ) with each row
  normalized to 1 over the extent, so `c` measures density rather than the number of
  bins. It is built at placement and on a space change.
- Consumption is computed from the biomass after its update, then the resource is
  updated exactly for that consumption, which keeps it in `[0, 1]` for any `c ≥ 0`.
- Biomass consumption includes latent mass even while muted or presentation-only.
  External consumption uses the actual mixed habitat PCM minus this Flora's own
  rendered habitat PCM, including fades, retiring tails and residual sound. Other
  Flora remain audible to this observer. Resources remain per instance.
- The fixed reference `P_ref = 0.03²/2 = 0.00045` maps the default nominal sine's
  mean square at unit unsaturated target mass to biomass 1. This is an authored
  acoustic unit, not a fitted auditory constant; changing or muting this Flora's
  amplitude does not change it. Playback gain and guards are downstream.
- The observer reuses the mixed stream's pre-birth raw window and power integration.
  For the shared rectangular PCM window, `sum(window²) = N`. One-sided FFT power
  is `|X[k]|²/(N sum(window²))`, doubled except at DC and Nyquist. Each supported
  FFT bin is distributed with `|K_i[k]|² / sum_l |K_l[k]|²`; unsupported power is
  excluded. Sparse weights are shared per geometry. Band integration reuses the
  existing NSGT coefficients. This is separate from `nsgt_power`, and the smoothed
  sum is not the instantaneous hop RMS.
- External observation is fixed within a hop. Before the first full valid window,
  resource updates wait while biomass and rendering continue. Gaps hold the last
  valid consumption with its observation frame; they never manufacture silence.
  Geometry changes discard the old external scan and wait for a valid new scan.
  Reports distinguish current observations, held observations and initial unknown.

**Why it moves.** A bloom depletes its resource and collapses; seeds can then establish
at harmonic positions whose resource is still fresh. Turnover occurs in the finite
reference and native observations below. This is a mechanism for succession, not a
proof that no static state is stable.

**Constants and defaults.** All are authored and none is fitted to listeners. The
initial values came from the reduced-model reference setting (Dynamics Design Record).
On 2026-10-23 the author adopted the establishment study's recommendation to lower the
fixed Allee density from 0.576 to 0.144 mass per octave. The same day, the author
adopted the recommendation to raise the default `rate` from 4/s to 8/s, retaining
`renewal` at 10 s; the other constants are retained. The finite comparison appears
under Implementation Findings. This is an authored default, not an optimum, and
the final listening choice remains with the author.

- Rhai: scene-wide `temporal_basin`, with neutral center 2 Hz if omitted.
  Internal neutral coefficients: growth 8/s and resource recovery `τ_r = 10 s`.
  Physical-time coefficients follow the common pace above; no independent rate/renewal knobs.
- Internal: metabolism `k = 0.3`; consumption time `τ_c = 0.3 s`; reference density
  `c_ref = 10` per octave; Allee half-saturation `v_a = 0.144` mass per octave; seed rain
  `ι = 0.48` and background seed `ι_bg = 9.6·10⁻⁴` per octave per second; initial total
  biomass `v_0 = 0.5`; spread σ = 10 cents/√s, `D = σ²/2` in cents²/s.
- Units: `v` is mass per bin. `v_a`, `ι`, `ι_bg` and `c_ref` are densities per octave,
  divided by `bins_per_oct` where a per-bin value is needed; `D` is converted with
  `(bins_per_oct / 1200)²`.
- Level scale: `c_ref`, `v_a`, `ι`, `ι_bg` and `v_0` scale together without changing the
  dynamics; scaling `c_ref` alone removes the Allee effect and gives a haze.

The new Allee value is two halvings of the previous 0.576 reference, below the
0.219–0.252 mass/oct limits diagnosed at two vacant positions with neighbours held
fixed. It is a coarse authored choice, not an optimum or a uniquely derived auditory
constant. The local calculation and its limitations are recorded below. Choosing
0.288 instead of 0.144 remains an audition choice, not an implementation gate.

**Non-negativity.** The substep rule implies `dt · (rate · k + 2D) ≤ 1`, under which `v`
stays non-negative in exact arithmetic. An implementation may guard f32 rounding with
`max(0)`, with a `debug_assert!` that the pre-guard value is not below `−1e-6 · max v`.

**Boundedness is not proven.** Ingredients for a bound: the spread conserves `Σ v`; the
seed input per bin is at most `(ι_bg + ι) / bins_per_oct` per second; the per-bin growth
rate is at most `rate · (u − k)`; and `Ĝ(x, x) > 0`, so a bin's own biomass drives its
resource below `k` once it is large enough. These do not yet bound a bin that keeps
receiving seeds and spread from its neighbours. Until a bound is derived, a unit test
checks boundedness on a flat terrain at the largest rate, and the closed-loop assays
record the largest total biomass.

**Seeding.** At placement:

- `consonance(...)`: the shape of base `consonance_density_mass` restricted to the extent,
  with a uniform fallback when the restricted total is zero (the fallback rule of
  range-local spawn in `Community::decide_frequency`);
- `random(lo, hi)`: uniform on the extent.

Each in-extent bin is multiplied by `0.9 + 0.2·ξ`, with `ξ` uniform in `[0, 1)` from a
deterministic RNG seeded by the scenario seed and the Flora id, so offline renders with a
pinned seed are reproducible. Then the weights are divided by their total and multiplied
by `v_0`, in that order, so tiny weights cannot overflow; if the perturbed total is zero,
the biomass is uniform with total `v_0`. The resource starts at `u = 1`.

**Space reconfiguration.** On a Log2Space change:

1. Recompute the bin range from the Hz extent, intersected with the new space.
2. Resample `v` as a density per octave and convert back to mass per bin; resample `u`
   by log2 interpolation and clamp it to `[0, 1]`. Nothing is renormalized. A bin newly
   inside the extent starts with `v = 0` and `u = 1`.
3. Rebuild `Ĝ`, recompute `n` and the cells. Tonal lanes are not bound to cells: they
   keep their Hz positions and modes, and tracking resumes against the new cells.
4. Residual lanes are rebuilt for the new cells. Each old residual lane with `r > 0`
   moves to an orphan list as a retiring lane, holding its Hz position until it frees
   (within 80 ms). The lane storage is reallocated with room for the new lanes plus the
   orphans and any carried tonal lanes. A reconfiguration while orphans remain carries
   them over again.

If no bin of the new space falls inside the extent, the Flora goes dormant: all its lanes
retire and are kept until they free, and it stays silent until a later space has at
least one bin inside its extent. It then reseeds uniformly with total `v_0`, `u = 1` and
no current lanes. This allocates, but only on reconfiguration (off the steady-state
path, like the space clones in `merge_latest_analysis_results`).

**F1/F2.** Every Flora function that accepts or returns a `_scan` asserts its length
with `Log2Space::assert_scan_len_named`: the biomass and resource scans, the landscape
inputs, the ERB-rate scan and the scratch scans passed between helpers.

## Rendering

Rendering turns the biomass into sound in three stages each hop. In this section and in
Tests 3, `p` denotes the biomass scan `v`; the decomposition is scale-free, since
candidacy is relative to `max p` and masses are excesses over the cell floor: a fixed **cell** allocation, a
**tonal atom** per cell, and an optional **residual layer**. Two choices are made per
Flora:

- `.tone("sine" | "resonant")`, default `"sine"`;
- `.residual(bool)`, default `false`; `.residual(true)` enables the band-noise layer.

v1 audits three configurations:

1. **sine + residual** (explicit `.residual(true)`);
2. **sine without residual** (default since the author’s 2026-10-22 audition);
3. **resonant + residual** (explicit `.residual(true)`).

The earlier peak-tracking allocation is not implemented. It is kept in Appendix A as
the fallback, in case cell boundaries or the coarse low register prove audible.

### Cells

**Partition.** The cells are an authored allocation rule.

- The extent is cut into cells 1 ERB wide in ERB-rate,
  `E(f) = 21.4 · log10(1 + 0.00437 f)` (Glasberg–Moore), starting at the extent's
  lower edge.
- A last remnant narrower than 0.5 ERB merges into the previous cell. An extent
  narrower than 0.5 ERB is a single cell.
- A bin belongs to the cell whose half-open interval `[E_k, E_{k+1})` contains its
  centre's ERB-rate. The last cell is closed, so a bin exactly at the extent's upper edge
  belongs to it.
- A cell that receives no bin is merged into its lower neighbour; an empty first cell
  merges into its upper neighbour. This happens only at very coarse `bins_per_oct`.
- Cells are recomputed only at placement and on space reconfiguration. The default
  runtime space holds at most about 31 cells.

Cells bound how many tonal partials can be offered, one per ERB of extent. They do not
by themselves keep partials apart: two components in adjacent cells can sit close to a
shared boundary. The minimum-separation rule below handles that case.

**Decomposition**, per cell and hop:

- **Excess.** `p_min_c` is the minimum of `p` over the cell, and the excess is
  `x[i] = p[i] − p_min_c`. The in-cell floor of the biomass (the seed bank) is subtracted this way and
  never sounds.
- **Local maxima** follow Appendix A's plateau rule on the whole extent, so a slope
  rising into the neighbouring cell is not a peak. A plateau unit belongs to the cell
  of its centre bin (the left centre for even widths).
- **Dominant peak.** The cell's highest local maximum; ties go to the lower bin. A cell
  with none has no candidate.
- **Position and width.** A three-point fit on `ln max(p, 1e-12)` around the peak bin
  gives the offset `δ` (Appendix A, clamped to `[−0.5, 0.5]`) and the curvature
  `c₂ = ln p₋ − 2 ln p₀ + ln p₊`.
  - The width is `σ = sqrt(−1/c₂)` bins, clamped to `[0.5, 4]`.
  - The fallback is `δ = 0`, `σ = 1` when `c₂ ≥ −1e-9`, for a plateau of width ≥ 2
    (its centre is used), and at an extent end.
- **Tonal window.** The bins within `2σ` of the peak bin that also lie in the peak's
  basin (between its neighbouring local minima, Appendix A) and in the cell. A second
  peak in the cell lies beyond a local minimum, so it is outside the window.
- **Candidacy** has hysteresis. The peak is a `κ_in` candidate when
  `p_peak ≥ κ_in · max p` (`κ_in = 0.02`). A peak held by a current lane remains a
  `κ_out` candidate while `p_peak ≥ κ_out · max p` (`κ_out = 0.01`).
- **Window excess.** `W` is the sum of the excess over the tonal window. It is computed
  before the tonal update, which compares candidates by `W`.
- **Masses.**
  - After the tonal update, `m_t = W` if a current lane holds the cell's dominant peak,
    and `m_t = 0` otherwise.
  - That covers: no peak, candidacy failed, minimum separation suppressed it, or a birth
    is waiting.
  - The residual mass is `m_r = (cell excess) − m_t ≥ 0`. Mass that does not sound as a
    tone therefore sounds as residual (or not at all when the residual is off), and no
    mass is silently dropped.
- **Residual position.** The excess-weighted mean log2 of the residual bins, clamped
  to the cell. When `m_r = 0`, the previous position is held (initially the cell
  centre).
- **Residual dead band.** `r*_res = sqrt(max(0, m_r − m_dead))`, with
  `m_dead = 0.001`. It is continuous, so the residual never gates on and off.
- **Target sum.** `S = Σ_c m_t + [residual on] · Σ_c max(0, m_r − m_dead)`. When
  `S > 1`, every target amplitude is multiplied by `1/√S`, so `Σ r*² ≤ 1`. This is the
  definition of the rendered target, not a loudness guarantee: `amp` is the nominal
  target-amplitude scale.

### Lanes

There are two kinds of lane.

- **Tonal lanes** form one pool of `2 × N_cells` lanes, not bound to cells. Candidates
  number at most `N_cells`, one per cell, so `current ≤ N_cells`. Since
  `current + retiring ≤ 2 × N_cells` (the pool size), births wait only when retiring
  lanes fill the rest of the pool. Lanes carried over a space reconfiguration are held
  separately (see Space reconfiguration).
- **Residual lanes**, one per cell, bound to that cell.

All lanes are allocated at placement.

**Flora amplitude.** `A_s(t) = amp_s(t) × fade_gain(t)`, where `amp_s` follows `.amp`
patches with a linear 80 ms ramp, and `fade_gain` is the placement and release ramp
(Scenario IR and Runtime).

**Lane state:**

- a unit complex rotator `z`;
- a rotation step `w`, refreshed every 64 samples from the target log2 frequency, which
  is interpolated linearly in log2 across the hop;
- a complex envelope `e` (see Atoms);
- a relative amplitude `r`, its target `r*`, and its power `q = r²`;
- a mode: free, current or retiring.

**Tonal update.** At a local crossing, or once per hop during continuous fallback,
decompose current biomass and visit the whole pool in fixed stages. Between crossings,
supported tones track and retain their windows as described in Default beat allocation.

1. **Track.** Current lanes, in descending previous `r*` (ties to the lower lane
   index), each claim the nearest unclaimed dominant peak of any cell within 50 cents
   in log2. The peak must be a `κ_out` candidate; ties in distance go to the lower bin.
   A lane can therefore follow its peak across a cell boundary with no handoff, since
   the oscillator is the same. A current lane that claims nothing retires: `r* = 0`,
   and its position is held.
2. **Separate.** Current lanes are scanned in ascending position. Whenever two
   neighbours are closer than 0.5 ERB, the one whose claimed peak has the smaller `W`
   retires (ties: the higher lane index). The scan repeats until no pair conflicts. A peak released
   this way is marked suppressed for this hop.
3. **Birth.** Unclaimed, unsuppressed `κ_in` candidates are taken in descending window
   excess (ties to the lower bin).
   - A candidate within 0.5 ERB of a current lane is skipped.
   - Each remaining candidate takes the lowest-index free lane at `r = 0`, with `z`
     from the seeded RNG, and `e = 1` (`sine`) or one noise draw (`resonant`, see
     Atoms).
   - With no free lane, the candidate waits through the finite retirement tail.
     Births are retried at later opportunities, or each hop during fallback. A given
     candidate is born only if it still qualifies and wins the order.
4. **Targets.** Each current lane sets its target position to its peak's and
   `r* = sqrt(m_t)`.

**Known consequences.**

- A peak that jumps more than 50 cents in one hop is a retirement plus a birth. The old
  lane retires through the finite metabolic tail. The new lane is born at an allocation
  opportunity, once a lane is free and the candidate wins the order. Re-onset at the new position therefore has no
  general upper bound: it grows with the hop length and with contention for lanes.
- Peaks that cross within 50 cents can swap lanes.
- Two candidates in adjacent cells closer than 0.5 ERB: only one sounds as a tone, and
  the other's window mass goes to its cell's residual.

**Residual lanes** are never born or retired. Each one's `r*` comes from its cell's dead
band, and its position glides to the residual position. Its bandwidth is `ERB(f)` at
that position.

**Slew (current lanes).** A fixed target is reached within 80 ms, independently of
`A_s`. Retirement uses the finite tail below.
A retiring tonal lane frees at 0.

- `sine` tonal lanes slew amplitude: `|Δr|` per sample is at most `1 / (0.08 · fs)`.
- `resonant` tonal lanes and residual lanes slew power: `|Δq|` per sample is at most
  `1 / (0.08 · fs)`, with `r = sqrt(q)`.
- In both cases the step is clipped so that the target is reached exactly.

The slew variable is relative amplitude `r` for sine, or relative power `q = r²`
for resonant, before the output scale `A_s = amp_s × fade_gain`. A birth from zero
reaches a fixed target in `ceil(r* × 0.08 × fs)` samples for sine and
`ceil(r*² × 0.08 × fs)` for resonant. Thus `r* = 0.5` takes 40 ms or 20 ms,
and `r* = 1` takes 80 ms, even when `amp = 0.03`. Multiplying the slew time by
output amplitude would confuse relative and output amplitude. These are fixed-target
arrival times; placement fade, live amplitude ramps, stochastic envelopes and detected
acoustic onset are separate quantities.

No two lanes ever carry the same signal (there is no handoff), so no statement about
summed amplitude during transitions is needed. Each lane's own ramp is the guarantee.

**Ramps per sample.** As in draft 6, every ramp is evaluated at each sample time with
its end values pinned, and the output is their product: the lane slew, `amp_s`'s 80 ms
ramp and `fade_gain`.

### Finite retirement

Retirement starts immediately but its amplitude follows the mass-decay
scale. Unsupported biomass has `v' = -rate * p * k * v`, so amplitude has decay
coefficient `lambda = rate * p * k / 2` (`k = 0.3`). For a finite continuous tail,
use the existing dead-mass floor `r_floor = sqrt(0.001)` and the authored law
`r_next = max(0, (r + r_floor) * exp(-lambda / fs) - r_floor)`, `q = r*r`.
This is a shifted exponential, not exact biomass tracking: the floor supplies an
extinction drain, with zero reached after `log(1 + r0/r_floor)/lambda` seconds
for constant pace. It avoids an arbitrary long-release duration. The macro fade
still bounds total release, including explicit zero fade. Ordinary attack/target
slews remain unchanged.

The fixed `2 * cells` tonal pool is retained. Longer retirement occupies its slots;
a candidate waits if none is free. Retiring tails own no pending reservation and
are not reintroduced as candidates. Their actual habitat PCM remains in the
self-exclusion capture, including the tail. Per-sample processing adds bounded
scalar work and no allocation; no lane or scratch pool grows at retirement.


### Default beat allocation

Every Flora uses the former T1 allocation rule. `FloraSpec.metric` and
`flora().metric()` have been removed. Wiring reserves the shared habitat acoustic
MeterNetwork only when the scenario contains a Flora placement. In that run it observes from the start, before any Flora is
placed. The MeterNetwork receives flux and the selected event/continuous recurrence
path with the scene's enculturation priors. It receives neither internal Voice
onset commands nor Flora allocation commands. Each Flora owns only its private
CouplingClock, using the Population metric rule (0.95 attraction, independent
phase, tempo-basin center or 2 Hz initial rate). The meter observation is frozen
at the hop start; its audio interval ends at that start. Positive beat confidence
is the existing model's confirmation signal, not a new confidence threshold.
Unknown, stale or unconfirmed observations use continuous allocation.

Biomass and resource integration remain continuous. Between crossings, current
lanes track nearby supported candidates. Ordinary replacement retirement becomes
pending; each such lane retains its last tonal window and targets only that
window's current excess. One bounded pending birth slot per cell records the
candidate bin and first waiting tick, revalidated every hop. A changed/disappeared
candidate is dropped. A lane's generation identifies each reuse of its slot.

Reservations are a union of the current tonal windows: each bin's excess belongs
to at most one tone, and is subtracted from the residual once. Empty windows,
out-of-range support, release and minimum-separation conflicts force immediate
retirement through the finite metabolic tail. At a local crossing, decomposition and
allocation are reevaluated in existing W order, tie breaks and pool capacity.
Born/current/pending/retiring describe lifecycle and waiting status; pending
retirement remains a sounding current lane until the opportunity or support loss.
The render loop applies the transition at its sample tick, including births
inside the hop. Retirement reaches finite zero through the shifted exponential above.
Fallback discards queued candidates and reevaluates current biomass without
resetting the clock or replaying the queue. Actual envelopes may change before a
crossing as support falls, and slew delays acoustic onset detection.

Flora reports include the acoustic observation end tick, flux, full meter support,
active/fallback state, local phase, bounded pending slots, drops, and transition
ticks, generations, waiting samples and forced reasons. Retirement events labeled
`crossing` or `fallback` identify allocation timing, not an exhaustive cause
classification: they can include support loss or separation conflicts handled by
the tonal update. Counts selected by these labels do not exclude all support-loss
retirements. Reporting allocation is outside the allocation-free renderer contract.
Every Flora report includes acoustic timing and ecological pace. Scenes without
Flora allocate no Flora meter.

### Atoms: one kernel

Every lane outputs `y = A_s · r · Re(e · z)`.

**Rotator.**

- `z ← z · w` each sample, and `|z|` is renormalized once per hop.
- This is the exact-quadrature rotator of `OscillatorBank` (`src/life/sound/oscillator_bank.rs`),
  not the magic circle of `ResonatorBank`: the magic circle's elliptic orbit adds
  amplitude modulation while the frequency glides, and the ellipse grows with frequency.
- The loop is written for Flora rather than shared with `OscillatorBank`, whose type
  presumes harmonic lanes. This is its second use, below the three-place extraction rule.

**Envelope.**

- `sine` tonal lanes: `e ≡ 1`.
- `resonant` tonal lanes and residual lanes use a complex AR(1) recursion driven by
  proper white noise:

  ```
  e ← a · e + sqrt(1 − a²) · n,    a = exp(−π B / fs)
  ```

  - `n` is proper white noise with unit second moment: zero mean, real and imaginary
    parts uncorrelated with variance 1/2 each (so `E|n|² = 1`, `E[n²] = 0`), and
    independent across samples and lanes. Uniform parts make it proper, not circularly
    symmetric.
  - `e` is initialized from one such draw. That sets the second moment, `E|e|² = 1`, from
    the first sample, and the recursion keeps it at 1; the initial draw does not come
    from the recursion's stationary distribution.
  - **Output power.** With `C = A_s r`,
    `E[y²] = ½ E[C² |e|²] + ½ Re E[C² e² z²]`. Suppose the pair `(C, z)`, with
    `|z| = 1`, is independent of `e` and `a` is fixed. Properness gives `E[e²] = 0`, and
    conditionally `E[y² | C, z] = C² / 2`: the power of a sinusoid of amplitude `C`.
    For fixed deterministic `C` and `z` this is simply `E[y²] = C² / 2`.
  - These statements hold only for that exogenous case. In Flora, `a` is updated
    once per hop from `σ`, and `σ` depends on the sound through the closed loop. Then
    `E|e′|² = 1 + E[a² (|e|² − 1)]`, and the last term need not vanish. `C` and `z` can
    also correlate with `e` through the loop. The unit second moment, the output power
    and the PSD formula are therefore nominal in closed loop.
  - The discrete power spectrum of `e` is `(1 − a²) / (1 + a² − 2a cos ω)`. For
    `B ≪ fs` this is close to a Lorentzian of FWHM `B`, and the recursion is close to
    the continuous Ornstein–Uhlenbeck process. These are approximations, stated as such.
  - Carried by the carrier, this is the baseband form of a resonator with
    `T60 = ln(1000) / (π B)` driven by white noise: the timbre note's
    excitation × resonance.
- **Resonant bandwidth:** `B = clamp(f · (2^{σ / bins_per_oct} − 1), 0.5 Hz, 0.5 · ERB(f))`.
  The field's local peak width becomes the tone's bandwidth.
- **Residual bandwidth:** `B = ERB(f)`.

**Noise.**

- One splitmix64 stream per Flora, seeded by the scenario seed and the Flora id.
- Draws are made in a canonical order: per sample, noisy lanes in ascending index, the
  real part then the imaginary part.
- Each draw is a uniform variate scaled to variance 1/2.
- Each hop's draws are written in that order to a per-Flora scratch buffer, sized
  `2 × noisy lanes × hop` and allocated at placement (reallocated only on a hop or space
  change). Both the scalar and the SIMD path read from it, so they consume identical
  noise.
- Renders with a pinned seed are reproducible.

**SIMD.** Lanes are processed in `f32x8` blocks under `simd-wide`, with a scalar
fallback. Both paths use the same `mul_add_fast` policy as `OscillatorBank`.

**Cost** per hop:

- at most `3 × N_cells × hop` lane-samples, about 93 lanes (12 `f32x8` blocks) on the
  default space;
- `2 × noisy lanes × hop` uniform draws;
- `O(N_extent · n)` for the dynamics and `O(N_extent)` for the decomposition;
- `O(N_extent × w)` for the consumption kernel, truncated at ±3σ.

**Known limits.**

- One tonal partial per cell. A second peak in the cell sounds as part of the residual
  band.
- The low cells are wide in log terms: about 7 semitones at 55 Hz.
- The residual lane is a single 1-ERB band at a centroid, so the residual's shape within
  the cell is not reproduced.

**Mixing.** The renderer calls each live Flora inside the render pass, as follows:

- the call comes after the voice sample loop has filled `buf_presentation` and
  `buf_habitat`, and before the own-sound histories run `sound.history.process(...)`
  (`src/life/schedule_renderer.rs`);
- Flora adds its hop to `buf_presentation` and `buf_habitat` according to its routing;
- it never writes a Voice's own-sound buffers.

`render_with_prediction_matches` gains a `&mut` parameter for the live Flora. In
`render_and_route_audio` that is a borrow of `state.pop`'s Flora, disjoint from the
other borrows there. `RenderFrame` stays read-only. Flora is not a Voice: it emits no
`PhonationBatch`, owns no own-sound history, and is not body-captured. Output limiting
is unchanged.

## Scenario IR and Runtime

- **Actions** (in `src/scenario.rs`):
  - `PlaceFlora { flora_id, spec: FloraSpec, extent_hz: (f32, f32), seeding: FloraSeeding }`
  - `UpdateFlora { flora_id, patch: FloraPatch }`
  - `ReleaseFlora { flora_id, fade_sec }`

  Actions within one event apply in order, so an update followed by a release at the
  same time updates and then releases.
- **Ownership and dispatch.** `Community` owns the Flora instances. It is already the
  container the `Conductor` dispatches actions to (`Community::apply_action`), so the
  conductor signature is unchanged.
- **Hop order.** Inside `advance_population`, after `Conductor::dispatch_until` and
  before the voices advance:
  - a Flora placed this hop seeds from the current landscape;
  - every live Flora runs its substeps.

  The landscape is current here because `apply_landscape_updates` has already run.
  Rendering follows in `render_and_route_audio`.
- **Fade.** `fade_gain` is a linear ramp, so it ends in finite time:
  - placement: 0 → 1 over `fade`;
  - release: from its current value → 0 over `fade_sec`.

  A release during a fade-in starts from the current gain. When the gain reaches 0 the
  Flora is removed and its lanes are freed.
- **Scenario end.** When the conductor is done, the runtime currently clears the
  Voices and calls `schedule_renderer.shutdown_at`, which note-offs started tones
  (`src/runtime/mod.rs`, `src/life/schedule_renderer.rs`). At the same point, every
  Flora that is not already releasing starts its release with its own `fade`, from its
  current gain. That includes a Flora in the middle of its fade-in. Only a Flora whose
  release ramp is already running keeps that ramp. The run finishes when the renderer is idle and no
  Flora remains: `finished_now` gains that second condition. `abort_requested` stops
  Flora with everything else.
- **Release tail.** `ScriptContext::max_release_tail` (`src/scripting/mod.rs`) also
  counts each `ReleaseFlora` event's time plus its fade. A Flora still live at script
  end does not extend the scripted duration; it fades after `Finish`, as started tones
  ring out after `shutdown_at`.
- **Allocation.** Every per-instance buffer is allocated when `PlaceFlora` is applied,
  as `Spawn` allocates Voices:
  - the biomass and resource scans, the update scratch, the per-bin ERB-rate scan and
    the consumption kernel;
  - the cell index arrays;
  - the `2 × N_cells` tonal pool and the `N_cells` residual lanes;
  - the noise scratch buffer, `2 × noisy lanes × hop`.

  Flora's own steady-state advance/render/capture path, including continual
  births and retirements, adds no allocations. Runtime PCM ownership, conditional
  Flora analysis publication and GUI/report snapshots are separate allocation paths;
  `AGENTS.md` records their measured conditions and counts. Placement, reconfiguration
  and reporting remain exceptions to the per-instance steady-state claim.

## Rhai Surface

Use scene-wide `temporal_basin(lo, hi)` to set the prior for all Flora pace.
Every Flora follows confirmed acoustic beats and schedules lane opportunities,
with continuous allocation at the basin center when evidence is unavailable.
The `metric`, independent `rate` and `renewal` spec methods are removed;
there is no timing opt-out or live pace override. Existing assay migrations and their changed interpretation
are listed in `samples/research/README.md`.

```rhai
let moss = flora()
    .amp(0.03)        // nominal target-amplitude scale: Σ a² ≤ amp²
    .tone("sine")     // tonal atom: "sine" or "resonant"
    .residual(false)  // default; true renders diffuse mass as band noise
    .fade(2.0);       // fade-in at place, fade-out at release

let sun = place(harmonic().brain("drone").sustain(), at(110.0));
let m = place(moss, consonance(110.0, 1760.0));  // the range is the extent
wait(12.0);
m.amp(0.015);
release(m);
```

- Defaults: `amp` 0.03, `fade` 2 s, `tone("sine")`,
  `residual(false)`, both buses.
- `flora()` returns a `FloraSpec`, with `.amp`, `.tone`,
  `.residual`, `.fade` and `.send` (same `Bus` / `BusSet` values as Populations).
  `.tone` accepts only `"sine"` and `"resonant"`; any other string is a script error.
- **Validation.** Every numeric argument, on the spec and on handle patches, must be
  finite both as given and after conversion to `f32`. Overflow to infinity in the
  conversion is an error. Out-of-range values are script errors, not clamps:
  - `amp` outside `[0, 1]`;
  - `fade` outside `[0, 60]` s.

  The upper bounds keep every ramp finite on a musical time scale.
- `place(FloraSpec, Placement)` returns a `Flora` handle.
  - The handle method `.amp` records a pending patch. As for
    Populations, the patch is emitted as `UpdateFlora` at the next `wait` / `flush`,
    or at scope end.
  - `release(Flora)` marks the release pending, as `release_population` does
    (`src/scripting/mod.rs`). At the next `wait` / `flush`, or at scope end, the pending
    patch is emitted first and then `ReleaseFlora` with the spec's `fade`, in one event.
    So `m.amp(x); release(m); wait(t);` updates and then releases. Releasing twice, or
    releasing after scope end, warns and is ignored, as for Populations.
- **Scopes.** A Flora created inside any scoped block (`section(...)` and the other
  verbs that push a scope, `src/scripting/engine.rs`) is recorded in that scope. At
  scope end its pending patch is emitted and it is released, in the same event and
  order as Populations (`ScriptContext::pop_scope`). Nested scopes release inner Flora
  at the inner end.
- **Accepted placements** are judged by value, because `Placement` cannot tell an
  explicit default from an omitted one (`src/scripting/mod.rs`).
  - Accepted: `SpawnStrategy::Field` with target `Consonance` or `Uniform`, built by
    `consonance(lo, hi)`, `consonance(root).range(a, b)` or `random(lo, hi)`.
    `count == 1`, `sampling == Density`, `tension == 0` and the default spacing are
    all accepted, so `.count(1)`, `.density()` and `.tension(0)` are harmless.
  - Rejected: anything else is an error at script evaluation (scenario construction),
    with a message naming the offending part. That covers `at`, `line`,
    `dissonance` / `edge` / `gap`, `count != 1`, `.peak()`, `tension != 0` and a
    non-default spacing.
  - An extent narrower than 1/96 octave (one bin of the default space, checked at
    script evaluation independently of the runtime resolution), or outside the existing
    placement bounds, is also an error.
- **Docs.** FnDoc entries in `src/scripting/docs.rs`; regenerate
  `rhai-defs/conchordal.d.rhai` and `api.md` with `gen_rhai_defs`; add a concept page
  to `docs/rhai_book` and its Japanese adaptation, whose examples
  `tests/rhai_book_examples.rs` executes.

## Expected Behaviour (hypotheses to audition)

The current 0.144 Allee setting has finite T0 measurements in Implementation Findings,
including the comparison that selected default rate 8/s. The establishment study's
rate-4 observations below are retained as such; they do not predict rate 8. The
historical consumer–resource reference used 0.576. The following sound descriptions
remain audition hypotheses:

- **Flora around a drone.** Vacant-position target mass increases in the measured
  scenes, but does not become the majority in the long drone: 58.0–64.2% of target
  mass still lies near its eight partials. Positions near the fifth and above the
  highest sounding partial can establish. This is not a promise of broad repertoire
  or tracking within a fixed delay after a retune.
- **Number and clarity.** The short mature sample17 window has target participation
  ratio 3.51–4.45; the 90-second drone has 6.55–7.09. These are effective target counts,
  not a count of clearly heard tones. A few clear positions is an audition hypothesis,
  especially in the longer scene.
- **Turnover.** The fixed drone retains distribution changes and lane births and
  retirements in its final 30 seconds. Its mature-window JS5 is lower than at 0.576;
  the new setting does not establish uniformly faster succession or perpetual motion.
- **Lone Flora and rate.** Self-related succession and general tempo control by `rate`
  remain hypotheses for the new setting. Earlier reference runs support them at
  0.576. The later rate-4/rate-8 comparison covers only sample17 and the long drone,
  not lone Flora or a rate sweep.
- **Degenerate regimes.** Earlier reference ablations produced haze when biomass far
  exceeded `v_a`, a frozen chord with fast renewal (about 1 s), and calm low-register
  recurrence. Their boundaries have not been remeasured at 0.144.

## Tier and Claims

Every part of Flora's dynamics belongs to the authored-rule tier of the two-tier
reading (technote ledger §9.3.57):

- the use of the terrain as fitness;
- the consumer–resource form (growth with the Allee term, metabolism, spread, seed rain,
  consumption and renewal) and its constants;
- the cell partition, the decomposition, the lane rules, the atoms' bandwidth maps
  and every constant above.

Flora's inputs, the R/H terrain, keep their own classification. Documentation must not
describe Flora's behaviour as grounded in auditory cognition.

## Alternatives Considered

- **Partials as Individuals.** A Population of sine Voices, each a partial. This keeps
  per-Voice phonation, lifecycle and identity machinery for what is meant to be a mass.
  Neither conservation of total energy nor births at arbitrary positions follows from
  that machinery.
- **Continuous `landscape_*_modes`.** The deferred "dynamic intonation" in the timbre
  note. It retunes a fixed-count body per Voice, which keeps the Individual as the unit.
- **Replicator–mutator with fatigue** (draft 12). Implemented on the reference branch; it
  settles to a static chord (Dynamics Design Record).
- **A listener-side information terrain as the drive** (surprise and precision, an
  expected-free-energy composite). Examined under the mechanism-selection rule and not
  adopted (technote ledger §9.3.93).

## Original v1 exclusions

The following list is historical v1 scope. Beat opportunities are now the default,
and reports also record lifecycle changes and temporal state. Phrase prediction,
bar selection and explicit Flora forecasts remain excluded.

- **Onsets or pulses** (temporal attacks), and any coupling to the production meter,
  arrival hazard or participation. The boom-and-bust of the dynamics is spectral, not an
  onset regime.
- **An explicit Flora forecast** for the `GeneratorModel`, and a `pred` self-term.
- **Further atoms.** Transients, and a grain atom whose source is `p` or the terrain
  (see the end of this note's discussion record).
- **GUI.** No visualization of the biomass or the resource.
- **Reports.** The only report record is one `flora` record per second per Flora, written
  only under `--report` (the report path's existing allocation exemption). It carries what
  the assays need: lane births and retirements, tonal and residual power, the latent
  peaks and the rendered partial set.
- **Temporal DCC acceptance.** No change to the acceptance steps or their material.

## Tests

The default path retains a saved candidate/rhythm replay against unconfirmed fallback, sample-interior birth and
expired-candidate tests, retained-window mass accounting, unknown fallback and
clock continuity, and zero-allocation turnover for both atoms. Runtime tests feed
identical PCM with different internal onset commands through both recurrence
paths and compare the acoustic meter exactly. Closed-loop silence, drone,
ordinary multi-Voice and explicit synchronization scenes compare transition
phase, waits/drops/forced retirements and PCM envelopes without treating phase
concentration as musical value. Non-opt-in output retains the F-10 WAV control.


Unit and integration tests assert. Assays render offline, record their measurements,
and are judged by listening together with those records.

1. **Invariance.**
   - Four short, pinned-seed offline cases in `tests/flora_invariance.rs` cover sine,
     harmonic and modal bodies, plus a moving harmonic case with habituation enabled.
     They are focused scenarios, not full top-level sample renders.
   - `tests/fixtures/flora_invariance_hashes.txt` stores the pre-Flora WAV hashes for
     each captured target-feature combination. A matching combination compares all
     four hashes; an uncaptured combination checks render completion without claiming
     bit identity. The baselines are captured before Flora, never refreshed by the
     implementation under test. These frozen cases retain their own 2048-sample
     configuration; the current measurement campaigns below use the default window.
   - Structurally, a scenario with no Flora passes an empty slice to the renderer, and
     Flora's code adds no arithmetic to the voice path.
2. **Dynamics (unit, fixed synthetic terrain).**
   - **Closed-form update.** With uniform biomass (no spread) and constant `g`, `H01`
     and `u`, one hop matches the update computed independently, for biomass and
     resource.
   - **Non-negativity and extent.** At `rate = 8/s` and a large hop, with the substep
     rule active, `v ≥ 0` everywhere and zero outside the extent over many hops.
   - **Resource range.** `u` stays in `[0, 1]`, relaxes to 1 with no biomass, and falls
     under heavy consumption.
   - **Boundedness on a flat terrain** (level 0.5 and `H01 = 0` everywhere) at
     `rate = 8/s`, from initial totals 0.05 and 5: the total biomass stays finite and
     below 20 (the reference test's bound) over 120 s of hops. This is a regression check
     on those conditions, not a general boundedness guarantee.
   - **Spread.** The reflecting spread conserves `Σ v` when growth and seeds are zero.
   - **Consumption kernel.** Truncated at ±3σ, rows normalized to 1, local to the extent.
   - **Substeps.** The count follows the rule for `rate` 4 and 8 at the default hop.
   - **Seeding.** Reproducible for a fixed seed, restricted to the extent, total `v_0`
     up to f32 rounding (the reference tests use an absolute tolerance of `1e-6`), uniform
     fallback for an all-zero shape, finite for tiny weights next to zero bins.
3. **Rendering (unit).** Algebraic checks are exact. Statistical checks are deterministic
   under their fixed seeds. Each tolerance sits at least 5 standard errors from the
   expected value. Under a normal approximation, a correct implementation run with a fresh
   seed fails a single check with probability below `6 · 10⁻⁷`, and all of them together
   below `5 · 10⁻⁶`.
   - **Partition.**
     - The cells tile the extent in 1-ERB steps, and a remnant under 0.5 ERB merges.
     - An extent under 0.5 ERB is one cell.
     - Every bin belongs to exactly one cell, and a bin on a boundary goes to the upper
       cell.
     - A bin at the extent's upper edge belongs to the last cell.
     - An empty first cell merges upward and an empty later cell merges downward.
   - **Decomposition.**
     - A slope rising into the neighbouring cell yields no candidate.
     - A plateau that straddles a boundary belongs to its centre bin's cell.
     - Equal-height maxima resolve to the lower bin.
     - A second peak beyond a local minimum stays outside the tonal window.
     - A flat `p` yields zero excess, so the floor is silent.
     - With `m_r = 0`, the residual position holds.
   - **Mass accounting.** In each case below, `m_t + m_r` equals the cell excess and the
     non-sounding tonal mass appears in `m_r`. The cases are separate:
     - no candidate;
     - candidacy failed;
     - suppressed by separation;
     - birth waiting for a free lane;
     - residual off.
   - **Target sum.** With `S` the defined sum, `Σ r*² = min(S, 1)`:
     - held tonal masses, plus residual above the dead band when residual is on;
     - held tonal masses only, when residual is off;
     - with `S > 1`, tonal and residual targets are scaled by the same `1/√S`.
   - **Tracking.**
     - A peak drifting 5 cents per hop keeps its lane, with phase continuous at hop
       boundaries.
     - A peak that crosses a cell boundary back and forth keeps the same lane, with no
       retirement.
     - A jump over 50 cents retires the lane and births another.
     - A peak whose `p_peak / max p` oscillates between 0.012 and 0.03 keeps its lane.
   - **Separation.**
     - Two adjacent current lanes under 0.5 ERB: the smaller retires.
     - A chain of three cells with mutually close peaks resolves deterministically to a
       conflict-free set, identical for any processing order of the input cells.
   - **Births.**
     - With the pool exhausted by retiring lanes, a candidate waits.
     - It is born at the first hop with a free lane, in the specified order.
     - With a hop longer than 80 ms, the old lane is silent within 80 ms but the new lane
       starts only at the next hop boundary. The test asserts that timing, not an 80 ms
       re-onset.
   - **Slew and ramps.**
     - `|Δr|` (sine) and `|Δq|` (noisy lanes) per sample never exceed their bounds.
     - Every target, including 0, is reached within 80 ms. This holds after `.amp(0)`
       and with a hop longer than 80 ms.
   - **Atoms, algebraic.**
     - A `sine` lane has `|e| = 1` exactly.
     - One envelope step equals `a·e + sqrt(1−a²)·n` for the logged `n`.
     - The noise draws follow the canonical order.
     - `resonant` bandwidth follows `σ` within its clamps.
   - **Atoms, statistical.** 1024 independent seeded lanes at `B = 0.5 Hz` and at
     `B = ERB(1 kHz)`, each run for 10 s:
     - The ensemble mean power, after the first sample, lies within 5 % of
       `(A_s r)² / 2`. That is over 6 standard errors at `1/√(1024 · max(1, πBT))`.
     - **Autocorrelation.** The estimator is
       `ρ̂(k) = Re Σ_lanes Σ_t e_{t+k} ē_t / Σ_lanes Σ_t |e_t|²`, with both sums over the
       same range `t ∈ [0, T − k)` and a fixed `a`. Its value at the integer
       lags `k = 1` and `k = round(fs / (10 B))` matches `a^k` within 0.06. That is at
       least 5 standard errors at `B = 0.5 Hz`, where the effective sample count is about
       `1024 · πBT ≈ 1.6 · 10⁴`.
     - No separate spectral-width test. For an AR(1) process, the algebraic step check
       and the autocorrelation `a^k` together determine the spectrum.
     - Every tolerance above is fixed before the test is first run.
   - **SIMD.** The `f32x8` path and the scalar path read the same noise buffer and render
     10 s of output. The difference must satisfy both
     `‖y_simd − y_scalar‖₂ ≤ 1e-5 · ‖y_scalar‖₂` and
     `max |y_simd − y_scalar| ≤ 1e-6 · A_s`. An absolute bound is used because a relative
     bound diverges near zero crossings.
   - **Reconfiguration.**
     - A reconfiguration while lanes are retiring keeps the orphans until they free and
       keeps every ramp within 80 ms.
     - A second reconfiguration carries the orphans over.
     - A new space with no bin inside the extent makes the Flora dormant, even if the Hz
       intervals overlap. A later space with at least one such bin resumes it.
   - **F2.** Each Flora function that takes a `_scan` panics on a wrong length, naming
     the scan.
   - **Edge cases.** These each produce the specified decomposition, with no NaN:
     - a uniform `p`;
     - an end-bin peak;
     - `p` with exact zeros;
     - no lane sounding.
4. **Scripting.**
   - `place(flora(), ...)` emits the expected actions.
   - Each rejected placement raises its message.
   - Each out-of-range, non-finite or `f32`-overflowing spec or patch value raises its
     message.
   - The accepted harmless defaults pass.
   - A pending patch is emitted at `wait`.
   - An update and a release at the same time apply in order.
   - `m.amp(x); release(m); wait(t);` emits the update before the release, in one event.
   - A nested scope releases the inner Flora at the inner end.
   - `release` uses the spec's fade, and a double release warns.
   - `max_release_tail` includes `ReleaseFlora`.
5. **Runtime.**
   - **Lone Flora.** A scenario with only Flora runs and finishes.
   - **End of script.** A Flora not released by script end fades after `Finish`, and the
     run finishes after the fade.
   - **Release during fade-in** starts from the current gain.
   - **Finish during fade-in.** A Flora still fading in when the conductor finishes
     starts its release, reaches gain 0, and the run finishes.
   - **Long hop.** With a hop longer than 80 ms, ramps still complete within 80 ms because
     they are evaluated per sample. That covers a born lane's rise, a retirement, and an
     `.amp` ramp. Waiting for a birth itself has no bound (Lanes).
   - **Allocations.** With the `profile-alloc` feature, counted between
     `begin_allocations` and `finish_allocations` (`src/runtime_profile.rs`):
     - around Flora's own advance and render, over many hops including continual births,
       retirements and boundary crossings, zero allocations;
     - separately, the worker-total count for a Flora scenario is recorded. It is not
       attributed to Flora, since Flora changes what the ecology does.
   - **Routing.** A `send(presentation_bus)` Flora is absent from the habitat chunk and
     present in the presentation chunk, which is what reaches the ListenerTwin. A
     `send(habitat_bus)` Flora is the reverse.
   - **Self-sound separation.** One Voice plus a habitat-routed Flora: the habitat
     buffer passed to the Voice's own-sound history contains Flora; the Voice's own
     body buffer does not.
   - **Space change.** A Log2Space reconfiguration resamples the biomass and the resource
     and keeps the Hz extent. An extent that leaves the space goes dormant and resumes.
6. **Closed-loop assays (offline, recorded).** All campaigns (`closed_loop`,
   `mixing_*`, `succession`) use the default analysis window (16384 samples, hop 512)
   and default-enabled habituation, and retain their resolved configuration. A
   2048-sample window biases harmonicity and level peaks 2–3 bins low (Implementation
   Findings). Each scenario runs in all three rendering configurations and records, per
   second, lane births and retirements, flapping events (a lane retired within 80 ms of
   its birth), the latent biomass peaks, tonal and residual power, the total biomass and
   the rendered partial set. The rendered partial set is a proxy, not a PCM spectrum:
   every lane with power in the last hop, tonal and residual, as its Hz position and its
   hop-mean deterministic power `(A_s r)²`; envelope noise, bandwidth and interference
   between lanes are excluded. The spectrum, distance and lattice measures below are
   computed on that proxy. The `succession`
   campaign computes from them, after a 20 s warm-up: the Jensen–Shannon distance
   between rendered spectra 1, 5 and 20 s apart; recurrence; the participation ratio;
   the lattice fraction (±25 cents of multiples of `f0/2`), before and after a retune;
   lane changes and flapping per second; the levels; the largest total biomass. The
   scenarios, each with a pinned seed:
   - a 110 Hz drone with Flora on `consonance(110, 1760)`, retuned to 146.83 Hz;
   - the same with Flora muted for 20 s and restored;
   - the drone with five seeking Voices;
   - habitat-only and presentation-only routing;
   - `rate = 8/s`;
   - a 55 Hz drone with Flora on 55–220 Hz;
   - two overlapping Flora.

   There are no pass thresholds; the measurements are judged together with audition.
7. **Mixing (integration, offline).** Tuning and evaluation are separate.
   - **Tuning** uses a 110 Hz harmonic drone with a Flora on `consonance(110, 1760)`.
   - **Evaluation** uses a 146.83 Hz drone with `consonance(146.83, 2349.3)`. It runs
     with three pinned seeds that tuning never used, and its margins are fixed before
     its first run.
   - **Measures**, each on both the latent biomass and the rendered spectrum:
     - the fraction within 0.5 ERB of the drone's partials;
     - the fraction within ±25 cents of the drone's common-root lattice (multiples of
       f0/2).
   - **Pass:** the first fraction is below, and the second above, the same fractions
     under uniform mass on the extent, by the fixed margins.
   - The near-partial measure needs the redesign noted in Implementation Findings
     before margins are registered, and the default analysis window.

## Deliverables When Implemented

- `src/life/flora.rs` (state, dynamics, cells, lanes, atoms).
- The invariance suite and the `closed_loop`, `mixing` and `succession` assays with their
  research samples (pinned seeds).
- Scenario IR, Rhai registration, generated defs, the Rhai Book page (EN + JA).
- A top-level étude in `samples/` (no pinned seed) mixing a Population with Flora, and
  its line in `samples/README.md`.
- `CLAUDE.md`'s module list gains `life/flora.rs`.
- The technote is updated only when the implementation lands.

## Implementation Findings

### Historical ecological pace and finite retirement, 2026-10-23

The old slew was a maximum slope, not an 80 ms duration: a sine lane at amplitude
`r0` reached zero in `ceil(0.08 * fs * r0)` samples. At the existing hysteresis
threshold, low-amplitude tones could disappear within one hop. Retirement in both
T0 and T1 used the metabolic amplitude scale and a finite extinction drain.
At that revision both modes advanced the entire ecology at pace `p = f/2`: T0 uses only the scene's
normalized tempo-basin center (2 Hz if absent), while T1 follows the same
confidence-weighted CouplingClock velocity as timing opportunities, with basin
fallback on stale or unknown observations. The neutral 2 Hz retains the adopted
ecological coefficients. The first implementation applied pace and finite tails
only to T1; the subsequent extension also changed T0. T0 retained continuous lane
allocation, not the old fixed ecological time or the old maximum-slope tail.
Articulation retirement and ecological recovery are authored embodiment, not
inferred phrase lengths. The composer supplies only the tempo prior; no new
authored ratio is exposed as a listener-state control.


### Historical default rate 8/s comparison, 2026-10-23

After the author requested the recommendation, the default changed from rate 4/s
to 8/s, retaining renewal 10 s. The native comparison used the same post-T1 code,
continuous T0 allocation, Allee density .144 mass/oct, seed 21, sine, residual off,
Flora amp .035 and Population amp .05. Config +4.5 dB and scene +7.5 dB playback
gain were common. The two scenes were sample17 and a 90-second 110 Hz drone;
only rate differed between the two versions of each scene.

Occupied fraction and JS5 use the definitions and mature windows in the
establishment study below. Holding time runs from a positive-target tonal lane's
birth to retirement start, excluding its fade. Medians use completed episodes
before macro release (2≤t<32 s for sample17, 2≤t<92 s for the drone); surviving
lanes remain right-censored.

| Scene | Rate / renewal | Occupied fraction | JS5 | Median holding (s) | Completed / censored |
|---|---|---:|---:|---:|---:|
| 17 | 4 / 10 | .670503 | — | 9.221 | 22 / 12 |
| 17 | 8 / 10 | .297313 | — | 7.051 | 24 / 8 |
| drone90 | 4 / 10 | .589451 | .302567 | 12.059 | 28 / 12 |
| drone90 | 8 / 10 | .213873 | .534416 | 7.616 | 37 / 8 |

The higher rate reduces occupied fraction and shortens holding in both measured
scenes. It also changes quantity: mature total target increases from .143164 to
.955215 for sample17 and .539877 to .756518 for the drone. These are design masses,
not perceived loudness, and the change is not just a rescaling of time. The value
8/s is the existing supported upper bound and the existing rate-8 assay's setting,
not a fitted optimum. This one-seed comparison does not establish generality or a
listening verdict. At the time, explicit `.rate(4)` selected the earlier growth
rate, and research assays pinned it. That API has since been removed, together
with independent `renewal`. Today's `temporal_basin(1, 1)` gives `p = 0.5`, matching
only the old growth coefficient 4/s. It gives renewal 20 s and consumption 0.6 s
instead of the old 10 s and 0.3 s, also scales diffusion and seed rain, and uses
the new finite tail. It does not restore these historical conditions or their
waveforms. The numbers above remain observations of the earlier implementation;
the current sound's final audition remains with the author.

### Lower fixed Allee density, 2026-10-23

The author adopted 0.144 mass/oct as the fixed establishment reference. The native
comparison used continuous T0 allocation before the metric extension, unchanged
sample17 and a 90-second 110 Hz drone, seeds 21–23, rate 4/s, renewal 10 s, residual
sound off, Flora amp .035 and Population amp .05. Config +4.5 dB and scene +7.5 dB
playback gain were common to all versions. These are retained design observations,
not a new acquisition with T1 or a listening verdict.

Occupied fraction is the snapshot-mean share of target mass within ±25 cents of the
eight drone partials. Vacant target and total target are absolute design mass before
Flora amp, not perceived loudness. PR is inverse target concentration. JS5 is the
square-root Jensen–Shannon distance (base 2), after 0.25 ERB row-normalized smoothing,
at a five-second lag. One prescribed snapshot per second is used; lifecycle rows do
not duplicate it. The mature windows are 22≤t<26 s for sample17 and 22≤t<92 s for the
drone. The short window cannot supply JS5.

| Scene / seed | Allee mass/oct | Occupied fraction | Vacant target | Total target | PR | JS5 |
|---|---:|---:|---:|---:|---:|---:|
| 17 / 21 | .576 | .744376 | .005922 | .023141 | 9.939038 | — |
| 17 / 21 | .144 | .670503 | .047591 | .143164 | 4.004526 | — |
| 17 / 22 | .576 | .769634 | .005784 | .025102 | 9.544027 | — |
| 17 / 22 | .144 | .789110 | .044110 | .206821 | 4.447641 | — |
| 17 / 23 | .576 | .721149 | .005999 | .021503 | 10.926361 | — |
| 17 / 23 | .144 | .561748 | .093222 | .213571 | 3.512130 | — |
| drone / 21 | .576 | .823405 | .007144 | .082155 | 6.032428 | .337192 |
| drone / 21 | .144 | .589451 | .229287 | .539877 | 6.703420 | .302567 |
| drone / 22 | .576 | .793823 | .018111 | .092441 | 5.092415 | .360587 |
| drone / 22 | .144 | .580494 | .239168 | .547551 | 6.552970 | .312159 |
| drone / 23 | .576 | .827650 | .007254 | .083404 | 5.801425 | .344621 |
| drone / 23 | .144 | .641829 | .214478 | .555448 | 7.091063 | .295076 |

The lower density recovers target quantity and increases vacant target, without
uniformly reducing occupied fraction: sample17 seed22 becomes more occupied. Long
drone PR increases, so fewer simultaneous effective tones is not established there.
The alternative 0.288 restores the seed21 drone's total target to .409429 with PR
5.003700, but sample17 vacant target is only .006645 against .005922 at .576. This is
why 0.144 was recommended first; it does not establish an optimal sound.

In the last 30 seconds (62≤t<92 s), the .144 seed21 drone has vacant target .199395,
total .488345, PR 6.632030 and JS5 .276040, with seven births and seven retirements.
The other two seeds also retain distribution changes. The mature-window fifth-region
target is .1015/.1109/.0975 for seeds 21–23, and target above the highest of the eight
partials is .1047/.1097/.0974. These finite observations do not prove persistent
succession, majority-vacant occupation, perceptual clarity or T1 behaviour. The
reduced model's high-register fraction must not be substituted for these native values.

### Heard habitat consumption, 2026-10-23

Native T0 renders compared the biomass-only implementation at `05abeb0` with
external power consumption, using unchanged sample 17, seeds 21–23, and a 90-second
110 Hz drone made by extending its first held segment. Reporting-only additions on
the baseline leave sample17 seed21 WAV bytes identical to the uninstrumented build.
Both use the default +4.5 dB playback gain and the sample's existing +7.5 dB gain.
In this earlier comparison the reference amplitude is 0.03 and seed rain, Allee
(0.576 mass/oct), rate and renewal are unchanged.

The occupied fraction is the share of Flora target mass within ±25 cents of the
eight sounding drone partials. The active mature sample17 interval is 22–26 seconds
(Flora age at least 20 seconds); the long-drone interval is 22–92 seconds. Absolute
target mass is before the Flora amplitude factor. PR is the inverse concentration
of lane target masses. JS5 is the square-root Jensen–Shannon divergence in bits,
after the existing 0.25 ERB row-normalized display smoothing, at a five-second lag.
The native summaries sample once per second, retaining lifecycle changes separately.

| Scene/window | Consumption | Occupied fraction | Target total | Vacant target | PR | JS5 |
|---|---|---:|---:|---:|---:|---:|
| 17 seed21, 22–26 s | biomass only | .95182 | .13612 | .00330 | 2.446 | — |
| 17 seed21, 22–26 s | biomass + external | .74438 | .02314 | .00592 | 9.939 | — |
| 17 seed22, 22–26 s | biomass only | .93064 | .10186 | .00492 | 2.590 | — |
| 17 seed22, 22–26 s | biomass + external | .76963 | .02510 | .00578 | 9.544 | — |
| 17 seed23, 22–26 s | biomass only | .92573 | .09865 | .00465 | 2.716 | — |
| 17 seed23, 22–26 s | biomass + external | .72115 | .02150 | .00600 | 10.926 | — |
| 90 s drone, 22–92 s | biomass only | .87294 | .41651 | .07444 | 2.699 | .52113 |
| 90 s drone, 22–92 s | biomass + external | .82341 | .08216 | .00714 | 6.032 | .33719 |

The four-second active mature sample17 window cannot provide a five-second lag.
Across 22–32 seconds, including drone release and the following silence, its JS5 is
.51361/.37652 for biomass-only/external at seed21, .52438/.35083 at seed22 and
.54148/.29119 at seed23. Those transitions are not evidence for stationary-scene
succession. In the long drone's final 30-second window, JS5 remains .44055 with
external consumption, with 16 births and 15 retirements between the endpoint
snapshots. The reduced model's weak *stationary* polyphony did not recur here.

This does not remove the repertoire problem. The external version is weaker and
has more simultaneous lanes; its long-drone occupied fraction remains high, and
its absolute vacant target is lower than the baseline. At 80 seconds, the vacant
1100.75 Hz bin has `v_crit = .00595`, actual biomass `.00104`, and seed supply
`.00128 mass/bin/s`; at 164.81 Hz these are `.00680`, `.00122`, and `.00170`.
These are local, instantaneous seed-free threshold diagnostics, not a proof of
future establishment in the coupled system. This comparison predates the lower
fixed Allee density adopted above; it did not itself change seed or Allee rules.

The declared real-time set (top-level samples 01–17 plus sample08 with one Flora)
completed 59,144 hops at 48 kHz / hop512 in the default release profile: zero deadline
misses, zero queued analysis hops, and no harness disturbance flag. The largest
case used 12 Voice observers and one Flora observer. This is a bounded measurement
on the acquisition machine, not a guarantee for arbitrary populations or hardware.

The instrument still has no audio-file output. Audition copies come from the offline
renderer with one common static gain; separately normalized copies are auxiliary.
Numerical implementation and these bounded render observations do not imply an
author's listening verdict.


The entries dated 2026-10-20 and 2026-10-21 come from the historical reference
implementation (Reference Implementation); the first two concern the replaced
draft-12 dynamics. The main remeasurement is recorded separately below.

**2026-10-20, the provisional dynamics settle to a static chord.**

- **Setup.** The fixed-terrain alternation test (Tests 2) uses two hills at 1.0 and
  0.6, both sounding, with `rate = 2/s` and `fatigue = 6 s`.
- **Observation.** The hills do not cross within 24 s. At 60 s they hold 0.495 and
  0.251 of the mass.
- **Measured state at 60 s.**
  - Fatigue: `h ≈ 0.80` on the high hill and `h ≈ 0.667` on the other.
  - Fitness: `g ≈ 0.200` on both.
- **Cause.** The fatigue drive is normalized by its maximum (`s = v / max v`), so every
  sounding peak tires in proportion to its share. Fitness equalizes across the sounding
  peaks, and the replicator rests at that mixed equilibrium.
- **Consequence.**
  - Draft 12's alternation hypothesis (its Expected Behaviour) does not hold for these
    dynamics.
  - This is the frozen-chord degeneracy that motivated the open DCC question (a drive
    of temporal evolution from a listener-side information terrain).
  - The test is kept, marked `#[ignore]`, as the reference case for the dynamics
    redesign. No parameter was tuned to make it pass.

**2026-10-20, the first mixing measurement is confounded.**

- **Setup.** Tests 7, run as measurement only; the margins stay unregistered and the test
  stays ignored. The evaluation used a 146.83 Hz drone and three seeds, measured on
  Flora-only PCM and on latent `p`.
- **Results.** The two fractions below are given as latent `p` / rendered spectrum.
  - Lattice fraction (±25 cents of multiples of `f0/2`):
    - uniform: 31 %;
    - sine without residual: 37 % / 83 %;
    - sine + residual: 39 % / 74–75 %;
    - resonant + residual: 39 % / 47 %.

    Concentration on the drone's common-root lattice is clear in the rendered sound.
  - Near-partial fraction (within 0.5 ERB of the drone's partials):
    - uniform: 65 %;
    - all three configurations: 65–73 %.

    This is not below uniform.
- **Why the second measure cannot decide.**
  - Above roughly the 8th harmonic, the drone's partials are spaced closer than 1 ERB, so
    ±0.5 ERB windows cover almost the whole axis. That is why uniform mass already
    scores 65 %.
  - The common-root lattice contains the drone's partials, so concentration on the
    lattice raises the near-partial fraction too.
  - The "avoids the critical bands of the drone's partials" hypothesis is therefore
    untested, not refuted.
- **Next.** Before margins are registered, the measure needs to be redesigned. Restrict
  it to the resolved harmonics, and count mass near a partial but off it (between 25
  cents and 0.5 ERB).

**2026-10-21, a 2048-sample analysis window biases the terrain low.**

- **Observation.** With `[analysis] nfft = 2048`, the configuration the Flora assay
  helper wrote for every campaign until now, a 110 Hz harmonic drone gives NSGT power
  peaks at the right bins but harmonicity and consonance-level peaks 2–3 bins low:
  107.6, 215.3 and 322.6 Hz instead of 110, 220 and 330 Hz. With the default 16384
  they coincide. The bias appears before any Flora is placed.
- **Consequence.** Flora follows the biased terrain, so the earlier closed-loop and
  mixing campaigns measured it 25–37 cents flat of the drone's lattice (a beating halo).
  Their recorded numbers, including the lattice fractions above, carry this bias.
- **What changed.** The assay helper now takes the analysis section per campaign; the
  earlier campaigns keep 2048 for comparability, the succession campaign uses the
  default. The repository's `config.toml` uses the default, so the instrument is not
  affected. Where the shift arises in the spectral front end (peak extraction from broad
  low-band lobes) is not yet diagnosed.

**2026-10-21, the consumer–resource dynamics keep moving in closed loop.**

- **Setup.** `succession` in `tests/flora_assays.rs`: eight research samples
  (`samples/research/flora_succession_*.rhai`, seeds 43001–43008) in the three
  configurations, offline with the full NSGT path at the default window. Metrics are
  those of the reduced model, computed on the per-second rendered partials after a 20 s
  warm-up (`target/flora-assays/succession/summary.md`), on the rendered-partial proxy
  of Tests 6.
- **Drone at 110 Hz with a retune to 146.83 Hz.** In all three configurations the
  dominant partials keep changing (5 s JS distance 0.41–0.46) with a participation
  ratio of 3.1–4.2. Recurrence is 0–0.11. Before the retune 0.92–1.00 of the rendered
  power lies on the drone's lattice; from 15 s after it, 0.80–0.995 lies on the new
  one. The largest total biomass is 1.7.
- **Flicker.** In the drone sample, lane births plus retirements average about 1/s (at
  most 7 in a second) and nothing flaps; across all runs flapping stays at or below
  0.015/s. Under the biased window the same samples flapped up to 4/s.
- **Mute and restore.** With `amp` at 0 for 20 s the output is silent and the ecology
  runs on; after restoring, the succession resumes at the same level.
- **A moving Population, routing, overlap.** With five seekers, with habitat-only or
  presentation-only routing, and with two overlapping Flora, the 5 s JS distance stays
  at 0.29–0.41.
- **`rate = 8/s`.** Faster (0.68–0.72), with about 3 lane changes per second.
- **Low register (55–220 Hz over a 55 Hz drone).** Calmer: 0.18–0.36, and the sine-only
  configuration mostly recurs (0.79).
- **Adoption.** The author adopted the dynamics as implemented on 2026-10-21. No audition
  record is kept here, and the levels are not tuned.

<a id="dynamics-redesign"></a>

### main での再測定、2026-10-22

The current main-layout implementation was measured from engine revision `6cbf951`;
this acquisition precedes the separate ownership follow-up. The campaign was not
repeated after the adopted redistribution, f64/floor and Flora-only stream changes;
its numbers remain measurements of that earlier producer. The adopted dynamics and
rendering constants were unchanged. All 45 campaign renders used 48 kHz, nfft 16384,
hop 512 and default-enabled habituation (5 s satiation, 8 s recovery, reference drive
0.25). Every output retains the fully resolved config and the compiled Flora settings.
The three configurations are sine + residual, sine only, and resonant + residual.
A bounded six-second real-producer run in each configuration first checked the report
schema, applied settings, residual presence/absence and aggregation. Only then were
the 24 succession, nine closed-loop and twelve mixing renders acquired.

The table gives ranges over the three configurations. Overlap pools the two Flora;
the saved summary also retains each Flora separately. JS distances use the base-two,
0.25-ERB Gaussian partial proxy on a 0.05-ERB grid, after 20 s of Flora age, with
non-silent endpoints. Recurrence uses an earlier spectrum at least 5 s away and JS
at most 0.1, as a descriptive definition, not an acceptance threshold. Main windows
are [22,102) s for drone/mute and [22,88) s otherwise. Maximum biomass is the largest
**one-second recorded** total over the whole run, including warm-up and release;
it does not bound unobserved between-record peaks.

| Scene | JS 1 s | JS 5 s | JS 20 s | Recurrence | PR | Lane changes/s | Flap/s | Maximum biomass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drone | 0.046–0.067 | 0.177–0.231 | 0.407–0.512 | 0.080–0.293 | 4.724–8.487 | 1.013–1.876 | 0.000–0.100 | 1.041–1.112 |
| mute | 0.033–0.043 | 0.138–0.151 | 0.245–0.309 | 0.536–0.679 | 3.696–5.658 | 0.262–0.812 | 0.000 | 1.073–1.331 |
| population | 0.036–0.044 | 0.164–0.182 | 0.340–0.377 | 0.525–0.639 | 3.727–5.815 | 0.212–1.032 | 0.000–0.015 | 1.044–1.191 |
| habitat_only | 0.036–0.046 | 0.166–0.184 | 0.340–0.378 | 0.557–0.639 | 3.682–5.870 | 0.243–1.074 | 0.000–0.060 | 1.064–1.224 |
| presentation_only | 0.035–0.041 | 0.163–0.184 | 0.344–0.379 | 0.557–0.639 | 3.666–5.496 | 0.243 | 0.000 | 1.106 |
| rate8 | 0.080–0.091 | 0.336–0.364 | 0.469–0.538 | 0.475–0.607 | 5.462–7.158 | 1.303–2.425 | 0.000–0.015 | 3.216–3.294 |
| low | 0.047–0.047 | 0.190–0.224 | 0.441–0.471 | 0.279–0.443 | 2.296–4.186 | 0.000–0.030 | 0.000 | 1.073–1.257 |
| overlap | 0.031–0.044 | 0.134–0.147 | 0.204–0.257 | 0.492–0.689 | 6.390–10.863 | 0.425–3.864 | 0.000–0.349 | 2.324–2.634 |

Retune lattice fractions are measured against 110 Hz in [22,42) s, then against
both 110 and 146.83 Hz in [57,102) s (15 s after retune):

| Configuration | Lattice before | Old lattice after | New lattice after | Maximum biomass |
|---|---:|---:|---:|---:|
| sine_residual | 0.8055 | 0.1890 | 0.6655 | 1.0431 |
| sine_only | 1.0000 | 0.1877 | 0.9811 | 1.0415 |
| resonant_residual | 0.7901 | 0.1862 | 0.6596 | 1.1121 |

Main-window mean tonal/residual powers below are deterministic lane-power proxies,
not PCM powers. Envelope noise, bandwidth and inter-lane interference are excluded;
combined overlap rows pool powers without cross terms. The full records retain
per-second levels and lane-change maxima as well as the means above.

| Scene | Tonal power | Residual power |
|---|---:|---:|
| drone | 1.030e-04–1.033e-04 | 0.000e+00–4.497e-05 |
| mute | 8.812e-05–8.925e-05 | 0.000e+00–2.139e-05 |
| population | 1.080e-04–1.091e-04 | 0.000e+00–3.001e-05 |
| habitat_only | 1.073e-04–1.091e-04 | 0.000e+00–3.018e-05 |
| presentation_only | 1.074e-04–1.074e-04 | 0.000e+00–2.547e-05 |
| rate8 | 8.834e-05–8.890e-05 | 0.000e+00–1.230e-05 |
| low | 1.199e-04–1.675e-04 | 0.000e+00–4.374e-05 |
| overlap | 1.635e-04–1.720e-04 | 0.000e+00–4.625e-05 |

During mute [42,62) s, 19/20 sampled proxy spectra are silent in every configuration;
the first sampled transition is included. Mean tonal power falls from
1.011–1.052e-4 before mute to 9.904e-8–1.004e-7 during it, then is
1.227–1.235e-4 in [67,102) s. Biomass remains positive during mute
(0.124–0.162 across the sampled records). No exact equality of pre/post levels is
claimed.

These observations differ from the historical succession campaign: drone JS at 5 s
is 0.177–0.231 versus 0.41–0.46, and rate-8 JS is 0.336–0.364 versus 0.68–0.72.
The drone's pre-retune lattice fraction is 0.790–1.000 versus 0.92–1.00; the new
lattice afterwards is 0.660–0.981 versus 0.80–0.995. Drone participation is
4.724–8.487 versus 3.1–4.2. Mean flapping reaches 0.100/s around the drone and
0.349/s in combined overlap; the old campaign reported at most 0.015/s. These are
observed differences, not failed acceptance thresholds. Main's defaults, engine base
and integration differ from that branch; the acquisition does not isolate a single
cause. The older closed-loop/mixing campaigns at nfft 2048 remain historical data;
they are not relabelled as current default-window measurements.

Mixing remains **measurement only**. Tuning seed 41101 and the previously used
seeds 42101–42103 were remeasured with the unchanged masks and measurement window
[18,34) s. The evaluation-labelled files are repeated measurements, not a new
held-out acceptance test. Ranges over the three evaluation seeds are:

| Configuration | Near fraction latent / PCM | Lattice fraction latent / PCM | Uniform near / lattice |
|---|---:|---:|---:|
| sine_residual | 0.685–0.692 / 0.725–0.752 | 0.647–0.674 / 0.745–0.771 | 0.648 / 0.312 |
| sine_only | 0.684–0.692 / 0.855–0.903 | 0.645–0.672 / 0.986–1.000 | 0.648 / 0.312 |
| resonant_residual | 0.677–0.688 / 0.727–0.733 | 0.651–0.653 / 0.686–0.696 | 0.648 / 0.312 |

The PCM measure uses Flora-only output, one-second Hann windows and 65536-point
FFTs, conditioned on the extent. Margins remain null and every `accepted` field is
null. Neither these fractions nor successful execution decide the avoidance
hypothesis: the near-partial mask still overlaps the lattice and covers much of the
extent. A decision would require the separate metric redesign, discriminating
controls and a justified margin registered with unused seeds. No parameter was tuned
from these results, and no author audition or real-time-load acceptance is claimed.

**2026-10-22, habituation suppresses the adopted succession.** The F-7 comparison
kept the front end fixed and separated global habituation from residual rendering.
Removing habituation recovered concentrated, changing partials and higher biomass;
scan inspection located the growth difference between base and eroded field levels.
Residual removal separately brought the sound closer to the author's intended Flora.
The F-7 measurement and scan reports retain the numerical values and windows. After
listening, the author adopted base-field growth and seeding for Flora, with residual
disabled by default. Voice/listener habituation, the seed-rain harmonicity and the
consumer–resource constants remain unchanged. This does not adopt a wider harmonic
repertoire or a new diffuse-biomass law.

## Dynamics Design Record

Record of how the consumer–resource dynamics were chosen. The proposal went through four
Astra reviews (draft 4 acceptable), was implemented on the reference branch, checked in
closed loop (Implementation Findings), and adopted by the author on 2026-10-21; the
levels are not tuned. On 2026-10-22 the author adopted heard habitat power as an
additional resource consumer; on 2026-10-23 the fixed 0.03 reference amplitude was
adopted. Later on 2026-10-23 the author adopted a fixed Allee density of 0.144
mass/oct for establishment, retaining the same law and the existing biomass
consumption, seed, renewal, cell and lane rules. C/H still include self-sound. A listener-side information
terrain as the drive did not pass its registered checks (technote ledger §9.3.93).

### Why the baseline settles

- On a fixed terrain the pure replicator term raises the mean fitness. With mutation,
  spread and fatigue the full update makes no such claim, but the measured outcome is a
  mixed equilibrium: fatigue is driven by `s = v / max v`, so every sounding peak tires
  in proportion to its share and fitness equalizes across them (Implementation
  Findings).
- The closed loop behaves the same way. In the `flora_sine_only` render of
  `samples/13_flora.rhai` (the info-terrain measurement, seed 5), Flora holds 15–16 weak
  partials on the drone's lattice. When the drone moves from 110 Hz to 146.83 Hz at
  14 s, the rendered partials stay at 110, 165, 220, 275, 330, 440, … Hz.
- Crowding, deeper fatigue or a minimum-mutation condition would each suppress one
  failure. None of them gives the system a reason to keep moving.

### The adopted update

Specified in State and Dynamics, with the rendering change under Rendering (target sum)
and the contract changes under Coupling Contract items 2–4.

### Local establishment barrier with seed and diffusion

For one interior bin, hold the terrain, resource and neighbouring biomass fixed.
Let `r = rate`, `q = g u`, `a = v_a / bins_per_oct`, and
`s = u (ι_bg + ι H01) / bins_per_oct`. Splitting diffusion into inflow and outflow gives

```text
dv/dt = r v [q v/(v+a) − m] + S
m = k + 2D/r
S = s + D(v_left + v_right)
S_crit = r a (sqrt(q) − sqrt(q−m))²,  q > m
```

`S_crit` is the maximum net loss along this fixed local curve. If `S > S_crit`, its
low-biomass establishment barrier disappears; equality leaves a tangent equilibrium.
For `q ≤ m` there is no high-biomass growth branch of this form, so the formula is not
extrapolated there. Reflecting ends change the outflow count. In the coupled system
`u`, `g` and neighbouring biomass evolve, so this is not a general establishment or
stability guarantee.

At 80 s in the seed21 native drone with external consumption and the old .576 density,
`r = 4/s`, `D = .32 bins²/s`, `k = .3`, hence `m = .46`. The diagnostics are:

| Vacant position Hz | q | Seed mass/bin/s | Diffusive inflow | Barrier without spread | Barrier with fixed neighbours | Limiting Allee mass/oct |
|---|---:|---:|---:|---:|---:|---:|
| 164.81 | .564542 | .001702 | .000221 | .001348 | .004397 | .251840 |
| 1100.75 | .602505 | .001282 | .000170 | .001228 | .003815 | .219104 |

Seed alone exceeds the no-spread barrier, but seed plus inflow stays below the barrier
including diffusive outflow. The last column is
`bins_per_oct S / [r (sqrt(q) − sqrt(q−m))²]`. A half step to .288 remains above both
limits; the second halving to .144 is below them. These two sampled local conditions
motivate the coarse comparison, not a unique or optimal value. The authored rate,
renewal and seed law were not changed, and no position-dependent compensation was added.

### Evidence from a reduced closed-loop model

[`flora-reduced-model/`](flora-reduced-model/README.md) ports the harmonicity and
roughness kernels and the consonance level from `src/core`. It renders the dominant peak
of each 1-ERB cell (sine only, no residual) over 110–1760 Hz and lets Flora hear itself
and an optional 8-partial drone one hop later. NSGT, the residual layer, lanes and
Populations are omitted. Metrics are taken on the rendered spectrum after a 20 s
warm-up, sampled every 0.501 s; their definitions are in the README, and every number
below is in `results/experiments.jsonl` or `results/sequences.txt`.

**Validation of the model.** The implemented baseline (revision 1 of the model)
reproduces the lattice-locked weak chord of the render above: 15–16 weak partials on the
drone's lattice and a JS distance of 0.02 between states 5 s apart. Unlike the render,
it partly follows a retune (0.82 of the rendered power on the new lattice from 15 s
after the move), so the model is less sticky than the full renderer.

**Historical 2026-10-21 reference setting** (before the .144 adoption). `rate = 4/s`, `τ_r = 10 s`, `τ_c = 0.3 s`, `k = 0.3`,
`c_ref = 10`, `v_a = 0.576`, `ι = 0.48/s`, `ι_bg = 9.6·10⁻⁴/s` (per octave),
`v_0 = 0.5`.

- **Drone at 110 Hz, 180 s.** An initial bloom reaches the ceiling at 3–6 s and
  collapses by 15 s, visiting 165, 550, 1101, 1318 and 1760 Hz. From about 20 s, 110 Hz
  (the drone's own fundamental) persists, and 220, 440, 330 and 659 Hz take turns above
  it, each for about 6–12 s, in varying order. The 5 s JS distance is 0.40, the
  participation ratio 3.0, and the mean target sum 0.38 of the ceiling. 36 % of frames
  lie within JS 0.1 of a frame at least 5 s earlier: the order varies, but the
  repertoire is small. The character is a slow succession over the drone's lower
  harmonics, not a dense texture.
- **Retune.** With the drone moving to 146.83 Hz at 40 s (120 s run), the time-mean
  power fraction on the new lattice from 55 s on is 0.9994 (0.25 also on the old
  lattice, through shared positions). After the move, 147 Hz persists and 220, 294, 440,
  587, 880, 1175 and 1760 Hz take turns. Over this run the 5 s JS distance is 0.48 and no frame
  recurs.

**Ablation** (the retune run, one change each):

| Change | Outcome | JS 5 s | Participation | New lattice |
|---|---|---|---|---|
| none | moving | .48 | 3.2 | .9994 |
| no Allee term | haze | .05 | 11.0 | .95 |
| consumption about 3·10⁹ times weaker (`τ_c = 10⁹ s`) | biomass reaches 5·10⁹, haze | .17 | 9.4 | .88 |
| renewal 1 s | frozen | .003 | 9.5 | .58 |
| uniform seed rain, same total before the resource weighting | silent (mean target sum < 0.001) | — | — | — |
| uniform seed rain, same total after the resource weighting | silent (mean target sum < 0.001) | — | — | — |
| seed rain from Flora's own harmonicity | moving | .41 | 3.2 | .65 (old 1.00) |
| Flora does not hear itself (fixed terrain) | moving | .48 | 2.8 | .9995 |
| consumption by rendered power (revision 1) | biomass reaches 8·10³², haze | .24 | 8.3 | .88 |

- In this setting, removing the Allee term, the consumption, the slow renewal or the
  harmonic placement of seeds each loses the succession. Each change is made at the
  reference values without retuning the others, so this shows what the reference
  depends on, not that no other combination works.
- The succession does not need self-perception: on the drone's fixed terrain it still
  occurs. Seeding from what is heard, rather than from Flora's own harmonicity, is what
  lets it follow the drone.
- In revision 1, the baseline at `rate = 4/s` gave a haze (JS 0.09 on biomass,
  participation 11.2): a faster rate alone does not produce the succession.

**Breadth.** A 3⁴ grid around the reference with the drone at 110 Hz, 90 s each:
`rate` 3, 4, 6; `τ_r` 6, 10, 15 s; `v_a` × 1, 1.67, 3.33; `ι` × 0.5, 1, 2. 29 of 81
settings are labelled moving, 30 frozen, 19 recurrent and 3 haze.

- No setting with `v_a` × 3.33 moves (19 frozen, 8 recurrent).
- `τ_r = 6 s` is mostly recurrent or frozen (13 and 10 of 27); `τ_r` 10 and 15 s give 14
  and 12 moving.
- `rate` 3, 4 and 6 give 7, 11 and 11 moving; `ι` × 0.5 gives 5, mostly frozen (17).

**Robustness and numerics.**

- Five seeds: 5 s JS distance 0.460–0.461.
- Lone Flora without a drone, 180 s: 0.46, no recurrence, participation 3.7.
- Drone at 220 Hz: 0.48. Drone at 82.41 Hz with the extent from 55 Hz: 0.35.
- 10 or 20 hops (107 or 213 ms) of added perception delay: 0.484–0.485, against 0.485
  without.
- Two substeps per hop instead of the one the rule gives: 0.483. The rule gives two at
  `rate = 6/s`. The model does not clip the biomass; the smallest value in each of the
  122 runs was positive (smallest 1.07·10⁻⁸).
- **Resolution.** At 48 bins/oct the regime is calmer (0.24, participation 6.1). The
  model's tonal window is ±2 bins, so it widens in cents at coarse resolution; the
  implementation's window must be checked at the runtime resolution.
- **Rate** sets the tempo: at 2, 3, 4 and 6/s the 5 s JS distance is 0.10, 0.26, 0.48 and
  0.74. At 2/s 61 % of frames recur; at 6/s the 1 s distance reaches 0.22.
- **Level.** Scaling `c_ref`, `v_a`, `ι`, `ι_bg` and `v_0` together from 0.1 to 2 times
  the reference leaves the 5 s JS distance at 0.45–0.46 and the participation at
  2.7–2.8, while the mean target sum goes from 0.038 to 0.76. Scaling `c_ref` alone by
  10 gives a haze (0.08, participation 9.1), because the biomass then far exceeds `v_a`.
  Scaling the power of Flora's partials in its own terrain by 0.03 to 1 (a power factor
  on self-perception, not a measured level ratio to the drone) gives 0.44–0.46.
- **Boundedness.** The largest total biomass is 3.4 in the reference and 6.8 at twice its
  scale. On a flat terrain (no drone, no self-perception, `g = 0.5`) biomass started at
  0.05 or 1.0 never exceeds its initial total and dies out within 120 s, with `c_ref` at 1
  or 10. Finite runs do not replace a bound.

**Limits.** Sine only: the residual configurations are untested. The model omits NSGT
smearing and its band-dependent delay, lane births, slews and the minimum separation,
moving Populations and routing, and its labels use fixed reading thresholds, not
registered criteria. Whether the succession survives the full renderer, and whether it
sounds rich rather than restless or monotonous, are questions for closed-loop renders and
audition.

### Before adoption

*Run on 2026-10-21 by the `succession` campaign; results in Implementation Findings. The
author adopted the dynamics the same day.*

Closed-loop offline renders with the full NSGT path, with the metrics above computed on
the rendered spectrum and lane counts from the `flora` report record:

- the reference over 110–1760 Hz with the drone of `samples/13_flora.rhai`, including its
  retune, at the runtime resolution;
- a low-register extent (55–220 Hz);
- threshold flicker: lane births and retirements per second;
- sine + residual and resonant + residual;
- two overlapping Flora;
- a moving Population, and habitat-only and presentation-only routing;
- muting (`amp` to 0) and restoring Flora mid-run;
- `rate` 4 and 8.

### Changes from draft 12

Applied on the reference branch on 2026-10-21: `.fatigue(sec)` became `.renewal(sec)`;
the `rate` range widened to `[0.05, 8]` with default 4/s; `amp` is documented as the
nominal target-amplitude scale; the ignored alternation test was replaced by the
`succession` campaign; the samples moved to the new surface.

### Open decisions

1. **Heard-sound consumption adopted and implemented, 2026-10-23.** Per-Flora
   resources consume latent biomass plus the power of all other habitat sound.
   Self-removal occurs on actual PCM, with fixed `P_ref = 0.00045`; the existing
   extent-local 0.5 ERB kernel applies to both terms. This adoption does not select
   new seed or Allee rules, nor guarantee persistent succession.
2. **Residual default resolved by author audition, 2026-10-22:** off, with
   `.residual(true)` retained for explicit use and the residual-on assays. A separate
   treatment of diffuse biomass remains unadopted.
3. **Establishment recommendation adopted, 2026-10-23.** D1 lowers the fixed Allee
   density rather than changing seed rain. D2 selects 0.144 mass/oct, two halvings of
   .576 and below the two local .219–.252 diagnostics. The implemented reference is
   .144; choosing .288 instead remains an author audition choice. This is not an
   optimum claim. Occupied-position bias, long-scene polyphony and perceived clarity
   remain sound-design questions; the finite improvement is not a new acceptance
   requirement for perpetual succession or a reason to expose an internal knob.
4. **Default-rate recommendation adopted, 2026-10-23.** Rate 8/s replaces 4/s;
   renewal remains 10 s. In the two seed-21 scenes, occupied fraction and median
   holding time decrease; the long drone's JS5 increases (Implementation Findings).
   The value is the existing supported upper bound, not an optimum. The final
   author audition remains open; the previous rate can be restored with one value.
5. **Temporal participation adopted for every Flora, 2026-10-24.** The author
   selected the existing T1 rule without an opt-out. See Rendering for audio-only
   observation, bounded pending state, sample crossings and immediate support-loss
   exceptions. This is an authored participation rule, not uniquely implied by the
   principle of temporal evidence. Unknown, stale and unconfirmed evidence retains
   continuous allocation without resetting the clock.

The proposal went through four independent Astra reviews (drafts 1–3 conditionally
acceptable, draft 4 acceptable as a proposal, with the checks before adoption still open;
`target/flora-redesign-20261021/review-1.md` to `review-4.md`).

## Reference Implementation

A complete implementation of this specification exists on branch
`research/flora-info-terrain-20261021`, on a base that main has since left behind
(it branched before main's runtime, scripting and life refactors and carries an
unmerged temporal checkpoint). It was not merged as is. It remains a historical
code and measurement reference. The current implementation follows main's refactored layout, listed below.

**Commits.** `eb13308` adds the draft-12 Flora (and the passive information-terrain
diagnostics, which stay on the branch); `ceddde5` replaces the dynamics by the
consumer–resource update and moves the Rhai surface to `.renewal`; `bbf42ce` holds this
note's evidence.

**Where things are on the branch.**

- `src/life/flora.rs`: state, dynamics, cells, decomposition, lanes, atoms, the SIMD
  path, the report record, and the inline unit tests (about 2400 lines).
- `src/scenario.rs`: `FloraSpec`, `FloraPatch`, `FloraSeeding`, `FloraTone` and the
  `PlaceFlora` / `UpdateFlora` / `ReleaseFlora` actions.
- `src/scripting/{mod,engine,docs,tests}.rs`: the builders, validation, patches,
  scopes, release tail and FnDocs. On main these concerns are split across
  `spec.rs`, `placement.rs`, `context.rs` and `engine.rs`.
- `src/life/community.rs`, `src/life/community/actions.rs`: ownership and dispatch.
- `src/runtime/mod.rs`: hop order, routing, scenario end and the report hook. On main
  these live in `worker.rs`, `analysis_sync.rs` and `phases.rs`.
- `src/life/schedule_renderer.rs`: Flora mixed into the habitat buffer before the
  Voices' own-sound history.
- `src/life/report.rs`: the per-second `flora` record.
- `tests/flora_invariance.rs` with `tests/fixtures/flora_invariance_hashes.txt`;
  `tests/flora_assays.rs` (`closed_loop`, `mixing_*`, `succession`).
- `samples/13_flora.rhai`, `samples/research/flora_*.rhai`,
  `docs/rhai_book*/src/concepts/flora.md`.

**Current main layout (2026-10-22).**

- `src/life/flora.rs` (including its private ERB partition): state, dynamics,
  rendering, report data and internal tests; pure Flora IR types
  and actions remain in `src/scenario.rs`.
- `src/life/community/actions.rs` dispatches placement, updates and release;
  `src/runtime/phases.rs` advances and drains Flora, and `src/runtime/analysis_sync.rs`
  handles space changes. `src/life/schedule_renderer.rs` mixes its sound into the buses.
- `src/runtime/worker.rs` publishes the per-second records through `src/life/report.rs`.
- `src/life/schedule_renderer.rs` captures each reserved Voice's actual habitat PCM.
  `src/runtime/worker.rs` transfers one recycled source packet with the mixed hop.
  `src/core/stream/loo.rs` subtracts PCM and retains the shared `AnalysisStream`
  pipeline independently per Voice. `src/core/analysis_worker.rs` publishes their
  H/R observations together with the mixed Landscape; `analysis_sync.rs` revalues
  them with the current C kernel. `Community` binds scores by ID and generation,
  and pitch/environment consumers apply the shared habituation once.
  The production ownership projection, peak-mass redistribution and Flora-only
  analysis stream have been removed. The old estimate remains test-only as a
  historical comparator for the registered non-worsening checks.
- `src/scripting/{spec,placement,context,engine,docs}.rs` and `src/scripting/tests/`
  hold the builders, strict validation, positioned diagnostics and API tier registry.
  Flora-specific vocabulary is Experimental; the authored-rule classification is a
  separate axis. Generated definitions are in `rhai-defs/` and both Rhai Books.
- `tests/flora_invariance.rs`, `tests/flora_assays.rs`, their fixtures and
  `samples/research/flora_*.rhai` hold regression checks and opt-in measurements.
  `FLORA_ASSAY_OUTPUT_DIR` selects a persistent acquisition directory outside `target/`.
- `samples/17_flora.rhai` is the unseeded composition; the English and Japanese
  concept pages are `docs/rhai_book*/src/concepts/flora.md`.

**What to carry over unchanged.** The cell partition and decomposition, the lane
tracking and births, the atom kernel and its SIMD path, and their unit tests do not
depend on the runtime layout.

**Pitfalls found on the branch.**

1. Run assays with the default analysis window. With `nfft = 2048` the harmonicity and
   level peaks sit 2–3 bins low, and Flora then beats against the drone.
2. Seed by perturbing, then dividing by the total, then scaling to `v_0`; normalizing
   before the perturbation shifts the total by up to ±10 %, and scaling before dividing
   overflows for tiny weights.
3. Scale the target amplitudes by `1/√S`, not by `1/S`.
4. At 96 bins/oct `harmonicity01` is one bin wide around each partial, so seeds land on
   exact bins and level peaks are one bin wide.
5. Main enables `[psychoacoustics.habituation]` by default; the branch's measurements
   ran with it disabled. At the initial main integration Flora read the eroded views;
   the 2026-10-22 comparison above motivated the switch to base views.
6. Fixed-terrain unit tests need a `Landscape` whose level and `harmonicity01` are set,
   not only the level.

**Verification on main.** The unit tests of Tests 2 and 3, the invariance suite, and the
`succession` campaign at the default window. For comparison, the branch measured on the
rendered-partial proxy a 5 s JS distance of 0.41–0.46 around the drone, 0.92–1.00 of
the power on the lattice before a retune and 0.80–0.995 on the new one afterwards, and
flapping at or below 0.015/s; these are measured values, not thresholds. Differences that follow from main's defaults (habituation) are recorded, not
tuned away.

This specification, consolidated for implementation on main, went through two Astra
reviews (draft 1 conditionally acceptable, draft 2 acceptable;
`target/flora-redesign-20261021/spec-review-1.md` and `spec-review-2.md`).

## Appendix A: Peak-Tracking Allocation (not implemented)

*Written for draft 12: `p` is the PMF there, its `Σ m ≤ 1` and the fatigue kernel refer to
the replaced dynamics. It is kept only as the fallback allocation.*

Retained from draft 6 as the fallback allocation. The plateau rule and the three-point
position fit here are also used by the cell decomposition.

**Peak picking**, on `p` within the extent, once per hop:

- **Plateaus.** A run of equal values is treated as one unit. It is a peak if it is
  strictly higher than the units on both sides; at an extent end, only the inner side
  counts. A plateau spanning the whole extent is not a peak. A uniform `p` is silent.
- **Basins** partition the extent without overlap. They are cut at local minima, and a
  minimum bin (or the left bin of a minimum plateau) belongs to the basin on its left.
  Each basin contains exactly one peak, whose mass `m_k` is the basin's sum of `p`.
- **Candidacy has two thresholds.** A peak enters as a candidate if
  `p_peak ≥ κ_in · max p` (`κ_in = 0.02`). A peak already held by an active slot stays
  a candidate while `p_peak ≥ κ_out · max p` (`κ_out = 0.01`).

**Position.** The candidate position is three-point parabolic interpolation on
`ln max(p, 1e-12)`:

```
δ = (ln p₋ − ln p₊) / (2 (ln p₋ − 2 ln p₀ + ln p₊))
```

- `δ` is clamped to `[−0.5, 0.5]`.
- The fallback is `δ = 0` when the denominator is `≥ −1e-9` (not strictly concave),
  for a plateau of width ≥ 2 (its centre is used), and for an extent end.

The result is a continuous log2 position, converted to Hz.

**Amplitude.**

- **Relative amplitude.** Each slot carries a relative amplitude `r` in `[0, 1]` and a
  target `r*`. Its output amplitude is `A_s(t) · r`.
- **Flora amplitude.** `A_s(t) = amp_s(t) × fade_gain(t)`, where `amp_s` follows `.amp`
  patches with a linear 80 ms ramp.
- **Target.** A selected peak of mass `m_k` sets `r* = sqrt(m_k)`, with no
  renormalization over the selected set.
- **What is guaranteed.** The steady target satisfies `Σ r*² = Σ_selected m_k ≤ 1`, so
  `Σ (A_s r*)² ≤ A_s²`, with equality when the selected basins hold all the mass. This
  is a statement about target amplitudes:
  - the waveform's mean power also has cross terms between partials;
  - transitions add or remove power.

  It is not a loudness guarantee.

**Partial bank.**

- **Capacity:** `2 × partials` slots.
- **States:** free, active, retiring. A slot records its position, and its retire time
  while retiring.
- **Slew.** `r` moves toward `r*` at a fixed slope of `1 / 80 ms`. It therefore reaches
  its target, or exactly 0, within 80 ms whatever `A_s` is, including `A_s = 0`. A
  retiring slot frees when `r = 0`.

**Selection and tracking.** Once per hop, in this fixed order:

1. **Peaks.** Compute peaks, basin masses, positions and both candidacy flags.
2. **Continue.** Active slots, in descending previous `r*` (ties to the lower slot
   index), each claim the nearest unclaimed peak within 50 cents in log2 that is a
   candidate under `κ_out`. Ties in distance go to the lower bin. An active slot that
   claims nothing becomes retiring.
3. **Revive.** While fewer than `partials` slots are active: retiring slots, in
   descending current `r` (ties to the lower slot index), each claim the nearest
   unclaimed `κ_in` candidate within 50 cents (ties to the lower bin) and become active
   again. Their `r` slews from its current value.
4. **Fill.** While fewer than `partials` slots are active and a free slot exists, the
   unclaimed `κ_in` candidate with the largest mass (ties to the lower bin) takes the
   lowest-index free slot, starting at `r = 0`. If no slot is free, the candidate is not
   selected this hop and competes again on the next.
5. **Swap.** Repeat while all three conditions hold:
   - a free slot exists;
   - the largest unclaimed `κ_in` candidate (ties to the lower bin) has mass above
     `1.25 ×` the smallest active slot's mass (ties to the higher slot index);
   - that active slot was not claimed in step 3 or 4 of this hop.

   The smallest active slot becomes retiring, and the candidate takes the lowest-index
   free slot.
6. **Targets.** Active slots set `r* = sqrt(m_k)` and move their target position to
   their peak's. Retiring slots set `r* = 0` and hold their position.

**Invariants** of this order:

- At most `partials` slots are active.
- A tracked peak can lose its slot in four ways:
  - it loses `κ_out` candidacy;
  - it moves more than 50 cents in one hop;
  - a slot earlier in step 2's order claims it first (a claim conflict, possible when
    peaks move toward each other);
  - it is out-massed by a factor 1.25.
- **Revival.** A peak that returns is revived without its `r` reaching 0 only if all
  of these hold when it returns:
  - its slot is still retiring (a slot retiring from `r` frees after `80 ms · r`, which
    can be well under 80 ms);
  - fewer than `partials` slots are active;
  - the peak is within 50 cents and a `κ_in` candidate;
  - no earlier retiring slot claims it.
- **Free slots.** A slot that retires and is not revived frees within 80 ms of the
  retirement. A full bank therefore never drops a slot's state, and the first free slot
  appears within 80 ms of the oldest retirement that is not revived.
- **Births run at hop boundaries.** Selection runs once per hop. A slot freed by a
  retirement that is not revived reaches a selection pass within 80 ms plus one hop of
  that retirement. Only that time is bounded. Whether a given waiting candidate is born
  then depends on the selection conditions at that pass:
  - the active cap in step 4;
  - step 4's priority among candidates;
  - the 1.25 factor in step 5.

  For example, with `partials = 1`, one active slot and a candidate of at most `1.25 ×`
  its mass, the candidate waits indefinitely. There is no general upper bound on a
  candidate's wait.

**Synthesis.**

- **Ramps per sample.** Every ramp is evaluated at each sample time, with its end
  values pinned exactly: `r`'s slew, `amp_s`'s 80 ms ramp, and `fade_gain`.
- **Output** is the product `amp_s · fade_gain · r` at that sample. Interpolating the
  product linearly between hop ends would not match the ramps, since a product of
  linear ramps is not linear, and on a long hop it would miss the 80 ms deadline.
- **Frequency.** Log2 frequency is interpolated linearly per sample within the hop.
  Each slot keeps a phase accumulator, so frequency changes are phase-continuous.

All in `f32`.

**Known limits.**

- Two peaks that cross within 50 cents can swap slots.
- A peak that jumps further than 50 cents in one hop is a retire plus a birth.

**What is bounded.** Each slot's own ramp, up or down, completes within 80 ms. A freed
slot reaches the next selection pass within 80 ms plus one hop of its retirement. A
re-birth of a particular peak is not bounded in time: it must also win the selection
conditions above. The level lost while it waits is not bounded either. These are
audible artefacts of tracking, accepted for v1.

**Cost** per hop:

- the partial bank: at most `2 × partials × hop` sinusoid evaluations;
- the dynamics: `O(N_extent · n)`;
- peak picking: `O(N_extent)`;
- matching: `O(slots × peaks)`;
- the fatigue kernel: `O(slots × w)`, where each slot's Gaussian is truncated at ±3σ
  and `w` is that window in bins. Without truncation it would be
  `O(slots × N_extent)`.

`slots = 2 × partials`.
