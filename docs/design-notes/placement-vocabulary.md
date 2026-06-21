# Placement Vocabulary: Orthogonal Field Targets

Status: Design + implementation (v0.4).
Scope: how a voice *enters* the frequency field (placement) — making the
placement surface orthogonal and comprehensive over compositional intent.

This is the placement-side companion to `voice-movement-redesign.md` (which is
about where a voice *moves*). Placement and movement are independent axes: where
you enter and where you go are separate choices, and the API must let them cross
freely.

## Problem: the placement surface was truncated, not designed

An earlier refactor (`894aa95`, "refactor for new scripting syntax") consolidated
a flat analytic enum (`SpawnMethod`: harmonicity, low_harmonicity,
harmonic_density, zero_crossing, spectral_gap, random_log_uniform) down to a
composer-facing set — but kept only the *consonance* pole:

- `peaks` (consonance maximum), `density` (consonance-weighted cloud),
  `random` (log-uniform), plus geometric `at` / `line`.

That conflated two different things: what the autonomous voices *do* by default
(seek consonance — the instrument's thesis) and what the *composer* is allowed to
ask for (the whole field). The field has structure — consonance maxima, the
dissonant pole, the metastable boundary between them, empty registers — and only
the consonance pole was addressable at placement. Dissonance was reachable only
indirectly (`harmonic_tension` strains the *whole* terrain; a movement objective
is per-voice behavior), never as a targeted entry. That is a real gap: tension,
clusters, color, register-filling are first-class compositional intents.

YAGNI governs implementation, not a language surface. An API is designed for
orthogonality and comprehensiveness over the intent space.

## The design space: two orthogonal generators

Placement (relative to the field) is the cross product of:

- **target** — which region of the field: `consonance`, `dissonance`, `edge`
  (the consonance/dissonance boundary), `gap` (empty register), `uniform`
  (ignore the field).
- **sampling** — how the target is realized: `peak` (the deterministic
  extremum) or `density` (a stochastic cloud weighted by the target).

|              | `.peak()` (extremum) | default `.density()` (cloud) |
|--------------|----------------------|------------------------------|
| `consonance` | most consonant point | consonance cloud             |
| `dissonance` | most dissonant point | dissonance cloud             |
| `edge`       | C ≈ midpoint         | boundary band                |
| `gap`        | emptiest register    | gap-weighted cloud           |
| `uniform`    | — (degenerate)       | **`random`** (log-uniform)   |

The single degenerate cell (`uniform` has no extremum) is the only hole — a sign
the model is tight. `random` is therefore not a geometric special case but the
field-agnostic corner of the same sampler; log-uniformity is automatic because
the field lives in Log2Space.

## One surface, no doubling

There is no separate "canonical" generator alongside shorthand verbs. The target
is the **constructor name** and the sampling is a chainable **modifier** with a
default — the same builder idiom as the rest of the API (`harmonic().amp()...`).

```rhai
consonance(a, b)            // consonance cloud (density is the default)
consonance(root).peak()     // consonant extrema around a root
dissonance(a, b)            // dissonance cloud
dissonance(a, b).peak()     // most dissonant point
edge(a, b)                  // boundary band
gap(a, b)                   // fill empty registers
random(a, b)                // uniform cloud (.peak() is degenerate, ignored)
at(hz) / line(a, b)         // geometry: authored coordinates, no field measure
```

Rule, one line: **no modifier ⇒ `density` (a population/cloud, the common case);
`.peak()` opts into the deterministic extremum.** Discrete consonant peaks (the
old `peaks`) are therefore `consonance(root).peak()`, not the default — the
default is "a population weighted toward the target", which is what placing
`count > 1` voices wants.

`consonance` takes either a root (1 arg, with `.range(lo_mul, hi_mul)` multiples)
or an absolute range (2 args). The other field targets take an absolute range;
root-relative multiples are a harmonic-series convenience specific to consonance.

## Mapping from the old surface

| old            | new                          |
|----------------|------------------------------|
| `peaks(root)`  | `consonance(root).peak()`    |
| `density(a,b)` | `consonance(a,b)`            |
| `random(a,b)`  | `random(a,b)` (unchanged)    |
| `at` / `line`  | unchanged                    |
| (removed) `low_harmonicity` | `dissonance(...).peak()` |
| (removed) `zero_crossing`   | `edge(...).peak()`       |
| (removed) `spectral_gap`    | `gap(...).peak()`        |

`peaks` and `density` are removed as verb names (no compatibility alias — alpha
policy); unregistered calls fail at scenario build, so migration is enforced by
the compile-only sample test.

## Implementation

`SpawnStrategy::Field { target, sampling, min_freq, max_freq, min_dist_erb }`
(plus `Linear`, `RejectTargets`). `decide_frequency` resolves per target:

- **peak**: deterministic argextremum over the range —
  consonance `argmax(consonance_field_level)`, dissonance `argmin`, edge
  `argmin |level - 0.5|`, gap `argmin(subjective_intensity)`.
- **density**: a non-negative per-bin mass, normalized to a PMF (the existing
  `ConsonanceDensity` machinery, generalized) —
  consonance `consonance_density_mass`, dissonance `1 - level`, edge
  `1 - 2|level - 0.5|`, gap `max_intensity - subjective_intensity`, uniform `1`.
  Occupied bins are masked; an all-zero range falls back to in-range uniform.

The density mass definitions for the restored targets are deliberately minimal
and may be tuned; they are the v0.4 starting point, not a frozen choice.

## Open

- `consonance(root).span(...)` for a root-relative *non-multiple* window, if a
  piece needs it (YAGNI until then).
- Whether `edge` should track the level midpoint (used here) or the gradient of
  the consonance field; revisit if the boundary band reads wrong on audition.
