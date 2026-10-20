# Flora: a Field-Form Inhabitant

Status: Design, draft 12 (2026-10-19). Nothing here is implemented.
The dynamics (State and Dynamics) are a provisional baseline. Where the drive of
temporal evolution should come from is an open DCC question: an information terrain on
the listener side is under consideration. The baseline is implemented first so that it
can be heard. The rendering (cells, atoms, residual) is independent of that question.
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
terrain is the gradient of a spectrum-wide objective. Accordingly, no monotone
improvement of any objective is claimed.

- Under a fixed terrain, the replicator term alone raises the mean fitness,
  `d⟨g⟩/dt = rate · Var_p(g)`.
- The mutation and spread terms can lower it, so even a fixed terrain gives no
  monotone claim for the full update.
- Once Flora's own sound changes the terrain, the sign of the change is undetermined.

Flora is an additional composition mode, not a replacement. It must be usable alone and
together with Populations in one scenario.

## Concept

- **Populations** are mobile fauna: discrete Individuals that move, sound and die.
- **Flora** is vegetation in the same habitat: a mass that grows where the terrain is
  favourable, spreads, tires where it has sounded, and withers.

Name: `flora`. The word `field` is avoided because it is already taken by
`consonance_field_*`, `SpawnStrategy::Field`, `FieldTarget`, `FieldSampling` and the
habituation field.

## Coupling Contract

1. **Only through sound.** Flora's audio enters the buses selected by its routing. The
   habitat bus feeds NSGT and the landscape; Populations perceive Flora there. Their
   audio reaches the same landscape, and Flora perceives them there. There is no direct
   state coupling in either direction. This is the contract Voices already follow.
2. **Flora reads the eroded view.** It reads `consonance_field_level_eff` per hop and,
   at placement, `consonance_density_mass_eff`. With global habituation disabled (the
   default) these equal their base variants.
3. **Flora perceives itself.** There is no leave-one-out for Flora. Self-interaction is
   the pattern-forming mechanism:
   - its own roughness lowers fitness within a critical band of each sounding peak;
   - its own harmonicity raises fitness at common-root relatives.
4. **Routing** follows `send()` as for Populations; the default is both buses.
   - `send(habitat_bus)`: a living, unheard terrain. The ecology responds to it; the
     listener does not hear it.
   - `send(presentation_bus)`: heard decor. Flora still reads the landscape, but its own
     sound is absent from it, so it cannot perceive itself. Its peaks then follow only
     the terrain made by others. This is a valid composition choice, stated here
     because it disables self-organization.
5. **Self-sound separation of Voices.** Flora is part of every Voice's surroundings.
   The renderer builds each Voice's own-sound history from the Voice's own body and the
   whole habitat buffer (`sound.history.process`, `src/life/schedule_renderer.rs`).
   Flora is therefore added to the habitat buffer before that call. It never enters a
   Voice's own body or own-habitat buffer (see Rendering).
6. **Predictive terrain.** The `GeneratorModel` (`src/life/generator_model.rs`)
   extrapolates the perceived `consonance_field_level_eff` history it is fed in
   `apply_landscape_updates` (`src/runtime/mod.rs`). Habitat-routed Flora reaches that
   prediction indirectly, through the perceived history. Flora publishes no explicit
   forecast of its own future spectrum.
7. **ListenerTwin** receives the presentation chunk, which contains presented Flora.
   Nothing is added on the listener side.
8. **Invariance.** A scenario that places no Flora renders bit-identically to the
   current build. Existing samples and the temporal DCC acceptance material contain no
   Flora and are unaffected.

## Closed-Loop Caveats

These are recorded as caveats, not solved by the design:

- **Delayed self-perception.** Flora perceives its own sound through the analysis path:
  - the worker keeps the landscape at most `MAX_LANDSCAPE_LAG_FRAMES` (= 1) hops behind
    generated audio (`src/runtime/mod.rs`);
  - NSGT band integration adds further delay, most in low bands.

  The default growth time scale (`1/rate` = 2 s) is far above this. That comparison
  alone does not prove stability, because the loop is discontinuous (next item).
  Stability is checked by the closed-loop assays in Tests, including a low-register
  extent and two overlapping Flora.
- **Latent mass versus sounding spectrum.** `p` is latent. What sounds is the tonal
  mass of held peaks, plus the residual when it is on.
  The design narrows the gap three ways:
  - the cell decomposition renders both tonal and residual mass, so diffuse mass is heard
    rather than dropped;
  - tonal candidacy has hysteresis;
  - fatigue is driven by the actual sounding amplitudes, not by latent mass.

  A change of dominant peak within a cell can still switch the acoustic input. This is
  the main closed-loop risk, and the assays measure it as flapping.
- **A future `pred` self-term.** If delayed self-perception has to be replaced by
  Flora's own state, Flora's perceived contribution must first be removed from the
  terrain it reads. Otherwise it would be counted twice. That requires per-source
  separation of the landscape, which does not exist and is out of scope.
- **Temporal observation.** With `temporal_mode("observe")`, temporal observation sees
  Flora's audio. Births and deaths are slewed (see Rendering) so that steady Flora is
  not a train of attacks. Whether the existing onset detector stays quiet on Flora
  alone is measured, not assumed.

## State and Dynamics

Each Flora instance holds:

- an **extent** `[lo_hz, hi_hz]` kept in Hz, and its bin range on the current Log2Space
  (runtime default 55–8000 Hz, 96 bins/oct, `src/runtime/mod.rs`);
- `density_scan` `p`: a PMF over bins, `Σ p = 1`, zero outside the extent;
- `fatigue_state_scan` `h` in `[0, 1]`;
- `fitness_level_scan` `g` in `[0, 1]`;
- `sounding_drive_scan` `s` in `[0, 1]`: the rendered spectrum mapped to bins;
- an amplitude `A(t) = amp × fade_gain(t)`;
- the cells and lanes (see Rendering);
- fixed-capacity scratch (see Allocation).

**Update.** The hop length is a configuration value (`[analysis] hop_size`), so
`dt_hop = hop / fs` is not assumed. Each hop runs `n` equal substeps,
`dt = dt_hop / n`. The terrain is held fixed within the hop. `n` is the smallest
integer that satisfies all three conditions:

- `rate · dt ≤ 0.05`;
- `μ · dt ≤ 0.05`;
- `D · dt ≤ 0.25` (D in bins²/s).

`n` is recomputed only when `rate`, the hop or the space changes. At the defaults and
the default hop, `n = 1`.

Each substep:

```
g[x]  = level_eff[x] · (1 − h[x])                          fitness level
ḡ     = Σ_x p[x] · g[x]
q[x]  = p[x] · (1 + dt · rate · (g[x] − ḡ))                replicator
q     = (1 − μ·dt) · q + μ·dt · u                           mutation toward uniform u on the extent
p[x]  = q[x] + dt · D · (q[x−1] − 2 q[x] + q[x+1])          spread; reads q only (old values)
        reflecting ends: q[lo−1] := q[lo], q[hi+1] := q[hi]
p     = p / Σ p
```

Then, once per hop, after rendering:

```
h ← HabituationField relaxation with drive d[x] = s[x]
```

**Why each step stays well-defined.**

- **Replicator.** `g − ḡ ∈ [−1, 1]`, so `rate·dt ≤ 0.05` keeps every factor in
  `[0.95, 1.05]`. `p` stays non-negative, and the step conserves `Σ p` exactly.
- **Mutation.** A convex combination for `μ·dt ≤ 1`. It keeps every in-extent bin at
  least `μ·dt / N_extent` after renormalization, so mass can appear at a new consonant
  position without travelling there (a jump).
- **Spread.** With reflecting ends and `D·dt ≤ 1/2`, it is a non-negative, mass-
  conserving stencil. It lets a peak glide when the fitness gradient moves.
- **Renormalization** only removes rounding drift.

These conditions guarantee a well-defined PMF. They do not rule out slow oscillation
of the closed loop (see Closed-Loop Caveats).

**Spread constant.** Specified as a standard deviation σ (cents/√s), so
`D = σ² / 2` in cents²/s, converted to bins² with the current `bins_per_oct`.
Default σ = 10 cents/√s gives `D ≈ 0.32` bins²/s at 96 bins/oct.

**Fatigue.** It reuses `HabituationField`'s asymmetric relaxation and transfer
`d / (d + ref_drive)` (`src/core/habituation.rs`) as a per-instance state. It is driven
by what Flora actually sounded, not by latent mass, and it is not the global
`[psychoacoustics.habituation]` field, which stays untouched.

- **Drive.** A proxy for what sounded: each lane's deterministic power `(A_s · r)²`,
  averaged over the samples of the hop (see Rendering). Lanes that rose, retired or
  freed during the hop count for the samples they sounded. The envelope's random
  fluctuation is excluded. The fixed-coefficient unit second moment (Atoms) is the
  nominal reference. Cross terms between lanes are not
  included: two sine lanes at nearby frequencies interfere, and this proxy ignores it.
  - The contribution is spread over the bins by a Gaussian in ERB-rate around the lane's
    current position (σ = 0.5 ERB, `unit_gaussian` in `src/core/float.rs`, truncated at
    ±3σ), and summed into `v[x]`.
  - Then `s = v / max v`. If `max v = 0` (no lane sounding, `amp = 0`, a fully faded or
    dormant Flora), `s = 0` and every bin relaxes toward recovery.
  - Fatigue therefore covers about a critical band around each sounding partial, not
    just its two bins. Without that, a peak would creep onto fresh bins of the same
    terrain hill instead of leaving it. A silent latent peak does not tire.
- **Disabling.** `fatigue = 0` disables fatigue: `h` is cleared to 0 and is no longer
  updated, so fitness is `level_eff` alone. Re-enabling starts from `h = 0`. A change
  between two positive values keeps `h` and changes only the time constants.
- With drive at most 1 and `ref_drive = 0.25`, `h ≤ 0.8`. Fatigue can lower a sounding
  peak's fitness to at most `0.2 ×` its level.
- Fatigue therefore makes mass move from a sounding peak at level `L₁` to a silent
  position at level `L₂` only if `L₂ > 0.2 · L₁` (strictly, only if `L₂ · (1 − h₂)`
  exceeds the eroded `g₁`). Weaker alternatives are never reached through fatigue. A
  fixed point remains a legitimate outcome.

**Provisional defaults** (authored, to be set by audition):

- `rate = 0.5/s`;
- `fatigue = 6 s` satiation, with recovery `1.6 ×` satiation (the global default ratio,
  5 s / 8 s);
- `ref_drive = 0.25` (the global default);
- `μ = 0.02/s`, σ = 10 cents/√s;
- `fade = 2 s`, `tone = "sine"`, `residual = true`.

**Seeding.** At placement, `p` is seeded from the target:

- `consonance(...)`: the current `consonance_density_mass_eff` restricted to the extent
  and normalized, with a uniform fallback when the restricted total is zero (the same
  fallback rule as range-local spawn in `Community::decide_frequency`);
- `random(lo, hi)`: uniform on the extent.

A multiplicative perturbation then breaks symmetry: `p[x] ← p[x] · (1 + 0.1·ξ[x])` with
`ξ[x]` drawn uniformly from `[−1, 1]`, followed by renormalization to `Σ p = 1`. `ξ` comes
from a deterministic RNG seeded by the scenario seed and the Flora id, so offline renders
with a pinned seed are reproducible. Fatigue starts at 0.

**Space reconfiguration.** On a Log2Space change:

1. Recompute the bin range from the Hz extent, intersected with the new space.
2. Resample `p` and `h` from the old bins by log2 interpolation. Set every bin outside
   the new range to 0, clamp `h` to `[0, 1]`, and renormalize `p`. If the resampled total
   is 0, use the uniform PMF on the range.
3. Recompute `n` and the cells. Tonal lanes are not bound to cells: they keep their Hz
   positions and modes, and tracking resumes against the new cells.
4. Residual lanes are rebuilt for the new cells. Each old residual lane with `r > 0`
   moves to an orphan list as a retiring lane, holding its Hz position until it frees
   (within 80 ms). The lane storage is reallocated with room for the new lanes plus the
   orphans and any carried tonal lanes.
5. The fatigue drive maps every lane's Hz position onto the new bins. The F1/F2 checks
   apply to the new scans.

A reconfiguration while orphans remain simply carries them over again.

If the intersection is empty, or no bin of the new space falls inside the extent, the
Flora goes dormant. All its lanes retire and are kept
until they free, and it stays silent until a later space has at least one bin inside
its extent. It then reseeds uniformly with `h = 0` and no current lanes. This allocates,
but only on reconfiguration (off the steady-state path, like the space clones in
`merge_latest_analysis_results`).

**F1/F2.** Every Flora function that accepts or returns a `_scan` asserts its length
with `Log2Space::assert_scan_len_named`. That covers `p`, `h`, `g`, `s`, the landscape
inputs, and the scratch scans passed between helpers.

## Rendering

Rendering turns `p` into sound in three stages each hop: a fixed **cell** allocation, a
**tonal atom** per cell, and an optional **residual layer**. Two choices are made per
Flora:

- `.tone("sine" | "resonant")`, default `"sine"`;
- `.residual(bool)`, default `true`.

v1 audits three configurations:

1. **sine + residual** (default);
2. **sine without residual** (control);
3. **resonant + residual**.

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
  `x[i] = p[i] − p_min_c`. The flat floor left by mutation is subtracted this way and
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
- **Target sum.** `Σ r*² = Σ_c m_t + [residual on] · Σ_c max(0, m_r − m_dead) ≤ 1`.
  This is the definition of the rendered target, not a loudness guarantee.

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

**Tonal update.** Once per hop, after the decomposition, over the whole pool in fixed
stages:

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
   - With no free lane, the candidate waits. A lane frees within 80 ms of retiring, and
     births are retried every hop in this order. Only the time to the next free lane is
     bounded (80 ms plus one hop). A given candidate is born only if it still qualifies
     and wins the order.
4. **Targets.** Each current lane sets its target position to its peak's and
   `r* = sqrt(m_t)`.

**Known consequences.**

- A peak that jumps more than 50 cents in one hop is a retirement plus a birth. The old
  lane retires within 80 ms. The new lane is born only at a hop boundary, once a lane is
  free and the candidate wins the order. Re-onset at the new position therefore has no
  general upper bound: it grows with the hop length and with contention for lanes.
- Peaks that cross within 50 cents can swap lanes.
- Two candidates in adjacent cells closer than 0.5 ERB: only one sounds as a tone, and
  the other's window mass goes to its cell's residual.

**Residual lanes** are never born or retired. Each one's `r*` comes from its cell's dead
band, and its position glides to the residual position. Its bandwidth is `ERB(f)` at
that position.

**Slew.** Each lane reaches its target, or exactly 0, within 80 ms, whatever `A_s` is.
A retiring tonal lane frees at 0.

- `sine` tonal lanes slew amplitude: `|Δr|` per sample is at most `1 / (0.08 · fs)`.
- `resonant` tonal lanes and residual lanes slew power: `|Δq|` per sample is at most
  `1 / (0.08 · fs)`, with `r = sqrt(q)`.
- In both cases the step is clipped so that the target is reached exactly.

No two lanes ever carry the same signal (there is no handoff), so no statement about
summed amplitude during transitions is needed. Each lane's own ramp is the guarantee.

**Ramps per sample.** As in draft 6, every ramp is evaluated at each sample time with
its end values pinned, and the output is their product: the lane slew, `amp_s`'s 80 ms
ramp and `fade_gain`.

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
    and the PSD formula are therefore nominal in closed loop. The fatigue drive
    uses the nominal power `(A_s r)²` as a proxy in both cases.
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
- `O(lanes × w)` for the fatigue kernel, truncated at ±3σ.

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
  Rendering, and the fatigue update from the rendered spectrum, follow in
  `render_and_route_audio`.
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
  - the scans `p`, `h`, `g`, `s`, the spread scratch `q`, and the per-bin ERB-rate scan for
  the fatigue kernel;
  - the cell index arrays;
  - the `2 × N_cells` tonal pool and the `N_cells` residual lanes;
  - the noise scratch buffer, `2 × noisy lanes × hop`.

  The steady-state hop path, including continual births and retirements, adds no
  allocations. The allocation census in `CLAUDE.md` therefore does not change.

## Rhai Surface

```rhai
let moss = flora()
    .amp(0.03)        // target level: Σ a² ≤ amp²
    .rate(0.5)        // growth response, 1/s
    .fatigue(6.0)     // satiation seconds; 0 disables fatigue
    .tone("sine")     // tonal atom: "sine" or "resonant"
    .residual(true)   // render diffuse mass as band noise
    .fade(2.0);       // fade-in at place, fade-out at release

let sun = place(harmonic().brain("drone").sustain(), at(110.0));
let m = place(moss, consonance(110.0, 1760.0));  // the range is the extent
wait(12.0);
m.amp(0.015);
release(m);
```

- `flora()` returns a `FloraSpec`, with `.amp`, `.rate`, `.fatigue`, `.tone`,
  `.residual`, `.fade` and `.send` (same `Bus` / `BusSet` values as Populations).
  `.tone` accepts only `"sine"` and `"resonant"`; any other string is a script error.
- **Validation.** Every numeric argument, on the spec and on handle patches, must be
  finite both as given and after conversion to `f32`. Overflow to infinity in the
  conversion is an error. Out-of-range values are script errors, not clamps:
  - `rate` outside `[0.05, 4]`;
  - `amp` outside `[0, 1]`;
  - `fatigue` outside `[0, 600]` s;
  - `fade` outside `[0, 60]` s.

  The upper bounds keep every ramp finite on a musical time scale.
- `place(FloraSpec, Placement)` returns a `Flora` handle.
  - Handle methods `.amp`, `.rate` and `.fatigue` record a pending patch. As for
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
  - An extent narrower than one bin, or outside the existing placement bounds, is also
    an error.
- **Docs.** FnDoc entries in `src/scripting/docs.rs`; regenerate
  `rhai-defs/conchordal.d.rhai` and `api.md` with `gen_rhai_defs`; add a concept page
  to `docs/rhai_book` and its Japanese adaptation, whose examples
  `tests/rhai_book_examples.rs` executes.

## Expected Behaviour (hypotheses to audition)

These are hypotheses about the dynamics, not guarantees:

- **Lone Flora.** It organizes into sparse peaks spaced beyond a critical band and biased
  toward common-root relations. With fatigue, mass moves among positions whose level
  exceeds the eroded level of the sounding peaks, giving a slow succession of spectra.
  It may also settle.
- **Flora around a drone.** Mass concentrates on the drone's harmonic lattice and avoids
  the critical bands of its partials, forming a halo.
- **Degenerate regimes.**
  - Under a fixed terrain with fatigue off, the system settles to a fixed point. In
    closed loop no such claim is made.
  - With `μ` or σ too large, it blurs into a haze.
  - With a high `rate`, one peak takes all the mass.

The defaults aim between these regimes. An offline sweep chooses them by listening.

## Tier and Claims

Every part of Flora's dynamics belongs to the authored-rule tier of the two-tier
reading (technote ledger §9.3.55):

- the use of the terrain as fitness;
- the replicator–mutator form, the spread, the fatigue drive;
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

## Out of Scope for v1

- **Onsets or pulses** (an excitable regime), and any coupling to the production meter,
  arrival hazard or participation.
- **An explicit Flora forecast** for the `GeneratorModel`, and a `pred` self-term.
- **Further atoms.** Transients, and a grain atom whose source is `p` or the terrain
  (see the end of this note's discussion record).
- **The dynamics' redesign.** A consumer–resource drive from a listener-side
  information terrain is an open DCC question. It is not part of v1.
- **GUI.** No visualization of `p` / `h`.
- **Reports.** The only report record is one `flora` record per second per Flora, written
  only under `--report` (the report path's existing allocation exemption). It carries what
  the assays need: lane births and retirements, tonal and residual power, the latent
  peaks and the rendered partial set.
- **Temporal DCC acceptance.** No change to the acceptance steps or their material.

## Tests

Unit and integration tests assert. Assays render offline, record their measurements,
and are judged by listening together with those records.

1. **Invariance.**
   - A representative set of renders, made offline with deterministic analysis and a
     pinned seed, is bit-identical to references captured from the pre-Flora build.
   - The set holds one top-level sample per body backend (sine, harmonic, modal), plus
     one `samples/research` assay with habituation enabled.
   - The references are captured once before the change; the suite compares against
     stored hashes.
   - Structurally, a scenario with no Flora passes an empty slice to the renderer, and
     Flora's code adds no arithmetic to the voice path.
2. **Dynamics (unit, fixed synthetic terrain).**
   - `p` stays a PMF: non-negative, summing to 1, zero outside the extent, every
     in-extent bin positive.
   - These hold for a large hop and a high `rate`, where `n > 1` substeps are needed.
   - The spread conserves mass at both reflecting ends.
   - **Common setup for the next two tests.**
     - The space is 96 bins/oct, and the extent is the 96 bins from 440 Hz to 880 Hz.
     - A fixed level scan is 0 on the extent, except two flat-top hills of exactly 4
       bins each, on bins `[31, 35)` and `[71, 75)`. Each is centred in its own cell, and
       the two cells (1 and 3) are not adjacent.
     - A hill's share is the sum of `p` over its 4 bins.
     - Initial `p` is uniform with no perturbation.
     - `rate = 2/s`, `fatigue = 6 s`, `tone = "sine"`, `residual = true`.
     - Observed for `10 ×` the satiation time.
     - The fatigue drive comes from the lanes' deterministic power, with rendering in the
       loop.
   - **Alternation.** Hills at 1.0 and 0.6, default spread:
     - by `1 ×` the satiation time, the 1.0 hill holds more than half the mass;
     - at some hop before `4 ×` the satiation time, the 0.6 hill's share exceeds the
       1.0 hill's.
   - **No alternation.** Hills at 1.0 and 0.1, with the spread disabled (`D = 0`): at
     every hop, the 0.1 hill's share `m₂` is at most the 1.0 hill's share `m₁`.
     Derivation, which does not depend on discrete symmetry:
     - `h ≤ 0.8` everywhere, because `h` starts at 0 and relaxes toward a transfer value
       of at most `1 / 1.25`.
     - So every bin of hill 1 has `g ≥ 0.2` and every bin of hill 2 has `g ≤ 0.1`. Hence
       `ḡ₁ > ḡ₂` for the mass-weighted means within the hills.
     - Mutation adds the same absolute mass to both hills, because they have equal bin
       counts. In continuous time,
       `d ln(m₁/m₂)/dt = rate (ḡ₁ − ḡ₂) + μ (w/N) (1/m₁ − 1/m₂)`, whose second term
       vanishes at `m₁ = m₂`.
     - The ratio therefore starts at 1 and cannot fall below it. The step sizes
       (`rate·dt ≤ 0.05`) keep the discrete update on the same side.
   - **Fixed point.** With fatigue off, `p` converges (the change per second falls below
     a set tolerance).
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
   - **Target sum.** `Σ r*²` equals the defined sum:
     - held tonal masses, plus residual above the dead band when residual is on;
     - held tonal masses only, when residual is off.
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
   - **Fatigue drive.** It equals the hop-mean of `(A_s r)²`, including a lane that frees
     mid-hop.
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
     - all lanes silent (`s = 0`).
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
   - **Fatigue toggle.** `fatigue(0)` clears `h`, and re-enabling starts from 0.
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
   - **Space change.** A Log2Space reconfiguration resamples `p` and `h` and keeps the Hz
     extent. An extent that leaves the space goes dormant and resumes.
6. **Closed-loop assays (offline, recorded).** Each scenario runs in all three
   rendering configurations. Each records, per second:
   - lane births and retirements, flapping events;
   - latent-`p` peak positions;
   - tonal and residual power;
   - the rendered partial set.

   The scenarios:
   - a lone Flora on 110–1760 Hz;
   - a lone Flora on 55–220 Hz (the low bands with the most analysis delay);
   - two overlapping Flora.
7. **Mixing (integration, offline).** Tuning and evaluation are separate.
   - **Tuning** uses a 110 Hz harmonic drone with a Flora on `consonance(110, 1760)`.
   - **Evaluation** uses a 146.83 Hz drone with `consonance(146.83, 2349.3)`. It runs
     with three pinned seeds that tuning never used, and its margins are fixed before
     its first run.
   - **Measures**, each on both the latent `p` and the rendered spectrum:
     - the fraction within 0.5 ERB of the drone's partials;
     - the fraction within ±25 cents of the drone's common-root lattice (multiples of
       f0/2).
   - **Pass:** the first fraction is below, and the second above, the same fractions
     under uniform mass on the extent, by the fixed margins.

## Deliverables When Implemented

- `src/life/flora.rs` (state, dynamics, cells, lanes, atoms).
- Scenario IR, Rhai registration, generated defs, the Rhai Book page (EN + JA).
- A top-level étude in `samples/` (no pinned seed) mixing a Population with Flora, and
  its line in `samples/README.md`.
- `CLAUDE.md`'s module list gains `life/flora.rs`.
- The technote is updated only when the implementation lands.

## Implementation Findings

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
  - The alternation hypothesis in Expected Behaviour does not hold for these dynamics.
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

## Appendix A: Peak-Tracking Allocation (not implemented)

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
