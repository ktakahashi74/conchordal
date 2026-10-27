# Mechanism evidence and choices

Conchordal gives each API term evidence labels in the [API reference](../reference/api.md)
and editor hover. A label explains the grounding of a mechanism, or says that the
term sets placement, an initial body or macro form. It is information, not another
setting. You can start with the existing defaults and choose more specific controls
only when a piece needs them.

## Reading the labels

| Label | Meaning |
|---|---|
| Principle | An adopted mechanism with primary evidence, an identified mechanism and a published model used as specified by its source, without requiring new human data collection. A label on an input function does not certify every equation or constant in the surrounding implementation. |
| Candidate (source) | A published formal model whose brain mechanism is unidentified, or whose parameters are task-specific fits. The source names the hypothesis; additions outside it receive separate author-rule labels. |
| Candidate (source match pending) | The published-model part has been identified, but source agreement is unfinished. This is not an accepted candidate. |
| Author rule | An authored mechanism without a corresponding published formal model, including selected forms, thresholds and values. Using sensory evidence does not turn an action rule into a principle. |
| Not a mechanism (kind) | Placement, initial body, macro form, routing, operation or data. Follow the referenced input and behavior terms for their mechanism labels. |

One term can have several labels. For `metric()`, beat perception is a principle
input, the Large neural-resonance oscillator form and Large & Jones (1999) gains
are a candidate part with source matching pending, and participation and default
coupling are author rules. `consonance()` is a placement operation: it reads H/R
and authored consonance valuation, then applies a selection rule. It does not
make the whole operation a sensory principle.

`follow_own_sequence(w)` combines the PPM-Decay candidate (Harrison and colleagues,
2020) with authored interval encoding and generation coupling. Its existing source
fixtures cover distributions, information content and learning state; they do not
establish R API or RNG equivalence. The term remains opt-in with weight 0 by default.
See [Rhythm](rhythm.md) for its supported execution paths and resource limits.

`phrase_boundary(w)` combines the LBDM candidate (Cambouropoulos, 2001) with
authored finite-sequence completion, a six-onset online adaptation, birth-fixed
caps and gain on one later legal Sine onset. Its implementation checks do not
reproduce the original program or corpus results. It defaults to weight 0 and is
available in the opt-in tonal profile, while the default sensory profile leaves it off. See [Rhythm](rhythm.md) for support and
response expiry.

Evidence is independent of **Core / Experimental / Tuning / Research**, which
organize how the API is used and disclosed. Experimental does not mean Candidate,
and Core does not mean Principle.

## What operates by default

Principles operate by default. The instrument also retains its existing authored
base behavior: movement cost, tessitura gravity, consonance valuation, habituation
constants, participation, metabolism and lifecycle rules. For example, movement
cost remains 0.5 and resolved tessitura gravity remains 0.1. These values are author
choices, not consequences of sensory principles. Adding labels changes no sound,
value, route or control.

Under the current policy, **new candidate and author-rule axes start neutral**.
This does not switch off the existing base. `dwell`, `rest_on_habituation`,
`follow_own_sequence` and `phrase_boundary` are examples of optional weights
whose defaults are 0.
Defaults come first, then presets, then individual controls: simple pieces should
not require a stack of parameters.

### The listener profile

Write `listener_profile("tonal");` before scene actions or scopes to select a
bundle; `listener_profile("sensory");` removes its inherited weights. Omitting the
word still selects **sensory**, preserving the existing base sound. Changing that
default requires separate author adoption; the bundle is available for audition,
not already adopted.

| Mechanism | Tonal weight | Evidence |
|---|---:|---|
| `follow_own_sequence` | 0.05 | Candidate: Harrison et al. (2020), PPM-Decay; author rules for interval encoding and scoring |
| `melodic_continuity` | 0.00076066292 | Candidate: von Hippel (2000), text-consistent AR(1); author rules for prefix estimation, scoring and replacement |
| `phrase_boundary` | 1 | Candidate: Cambouropoulos (2001), LBDM; author rules for online boundaries and gain |
| Final section arch / ritard | 1 / 1 | Candidates: Todd (1992), Friberg–Sundberg (1999); author rules for interval assignment, gain floor and clock mapping |

These are the SQ-1, ME-1i, PX-1i and PE-1i demonstration weights. They make
mechanism differences available to audition; they are not author-adopted values
and will be updated after adoption. `listener_profile` itself is labelled **Not a
mechanism (profile selection)**; the constituent terms retain their own labels.

ME's weight matches the displaced old terms' mean absolute score contribution.
`follow_tonal_context` is not inherited: the table costs about 1 GB and 80 seconds
to create, and approximation acceptance is limited to [0.25,1], based on
measurements at 0.25/0.5/1. These are not word limits; outside that range acceptance
is unverified and no extra warning is added.

Explicit Population weights override the profile, including explicit zero.
Inheritance checks the final Voice form at placement, independently for each
mechanism, and silently skips unsupported Voices. Individual axes still default
to zero. Existing author-rule base behavior remains, except that a known enabled ME
prediction replaces movement cost and fixed-center gravity for that decision.

Section expression uses the last completed `section`, only when it ends at the
scene's score cursor and the entire scene satisfies existing PE-1 support.
Without a suitable section it does nothing. Flora, sustained participation and
other unsupported PE-1 scenes are silently skipped as a whole; this does not
extend PE-1 to those Voices. Any explicit `expressive_section`, including weights
0/0, takes priority over inherited expression. Runtime delivery and duration
checks remain in force. See [Rhythm](rhythm.md) for each mechanism's limits.

## What the composer chooses

The composer chooses the use and weight of hypotheses and style-dependent priors
in layer-3 cognitive coordinates, as well as initial conditions and macro form.
The labels do not expose internal memory integration or learning dynamics (layer 4),
or change front-end sensation, roughness or harmonicity (layer 1). The configuration
evidence table in the API reference includes research settings so their status is
visible; it does not make those settings composing controls. Raw consonance
valuation coefficients also remain outside the Rhai surface.

## What acceptance means

Principles and candidates require numerical agreement with their source, the
registered causal difference and real-time operation on the declared load.
Candidate acceptance checks that a hypothesis has been implemented correctly; it
does **not prove that the hypothesis is true**. Principles have no subjective
acceptance gate.

Author rules require implementation checks of default agreement, causal difference
and real-time operation, together with the author's listening. A bundled default
listener preset receives one author audition; listening does not rank parameter
values. Composer-control upper bounds follow failure or effect saturation, not
computational cost. These acceptance conditions do not change simply because a
label appears in an editor.

`melodic_continuity` separates the von Hippel (2000) candidate from the author rules for estimation, self execution and generation coupling. Its default is zero.
