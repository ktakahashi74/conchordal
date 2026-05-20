# Listener Twin Design

Status: Phase 1 implemented; later phases are design proposal
Scope: future conchordal architecture, DCC coupling, tension/resolution model
Date: 2026-05-19

This memo records the proposed design before implementation. It intentionally
does not define a public scripting API yet.

This memo is the implementation-facing companion to
`docs/design-notes/dcc-neurocognitive-hierarchy.md`. That hierarchy memo states
the broad DCC position across time scales. This memo narrows the first
implementation target to a listener-side view of Layers 1, 3, and 4: acoustic
feature stability, local pitch/field affordance, and phrase-like tension or
closure signals.

## Problem

conchordal currently has two partly separate ideas:

- `Landscape`: a frequency-space terrain derived from harmonicity, roughness,
  and consonance.
- ALife population dynamics: distributed voices that adapt, move, synchronize,
  and survive in that terrain.

This has been enough while demos were mostly about local voice behavior. It is
not enough for DCC tension/resolution, because the relevant coupling target is
not only the generated field. The target is how a musical listener hears the
current presentation.

In other words, the DCC counterpart of conchordal is not another voice. It is a
digital twin of the listener state that the system keeps synchronized with the
audible output.

## Name

Use `ListenerTwin`.

Reasons:

- It says whose state is being modeled.
- It matches the future direction where real listener feedback can update the
  same state.
- It avoids overloading `Landscape`, which is a material used by the twin, not
  the twin itself.
- It avoids `WorldModel` / `Memory` ambiguity. Those names imply a subject, but
  did not say whose world or whose memory.

`AuditoryTwin` remains a plausible lower-level name, but it is narrower. The
planned object is not only auditory feature tracking; it is listener-side state
tracking for DCC.

## Core Semantics

`ListenerTwin` models the listener-side state of the audible presentation.

It should observe the presentation bus, not the hidden habitat bus.
Hidden anchors, scaffold tones, and internal control fields must not directly
contribute to listener tension unless they are actually audible.

The implementation should make this boundary difficult to violate. Prefer an API
where `ListenerTwin` receives presentation-derived analysis output, not a generic
`Bus` selector. If a builder or input type is added, it should not allow
subscribing `ListenerTwin` to `habitat_bus`.

The twin is global by design, but this does not conflict with distributed ALife:

- ALife voices are the distributed generative dynamics.
- `ListenerTwin` is the shared listener-side coupling surface.
- DCC coupling happens between the distributed population and this shared
  listener-state surface.

The twin must not directly command voices. It should expose state and pressure
signals. A later coupling layer can translate those signals into weak,
distributed selection or movement pressure.

## Relationship To Landscape

`Landscape` stays as the terrain representation.

`ListenerTwin` may contain or derive several landscapes:

- presentation landscape: derived from actually audible output.
- predicted presentation landscape: derived from NoteBoard / population intent.
- error landscape: difference between presentation-derived and predicted state.
- memory summaries: phrase, event, and longer-timescale listener context.

The first implementation should only need the presentation landscape. Prediction
and memory can be added after the first real use appears.

For Phase 1, pitch-contour expectation is approximated by local field affordance:
reachable improvement in the presentation-derived consonance field. This is a
temporary Layer-3 approximation, not a full pitch-contour prediction model.

## First State Model

The minimal state should be small and explicit:

```text
ListenerState {
  time_sec,
  attention_level,
  neural_rhythms,
  stability_level,
  resolvability_level,
  tension_level
}
```

Recommended meanings:

- `stability_level`: how stable / consonant the current audible presentation is.
- `resolvability_level`: whether nearby audible states offer a plausible
  improvement.
- `tension_level`: unstable now, but with an available path toward improvement.
- `attention_level`: presentation-derived onset / spectral-flux salience.
- `neural_rhythms`: presentation-derived listener-side rhythm model.

Low or absent presentation evidence should not be reported as instability.
When the presentation-derived intensity mass is effectively zero, the first
implementation reports neutral stability, zero resolvability, and zero tension.

Derived values should not be stored in Phase 1:

- `instability_level = 1.0 - stability_level`

Delayed values:

- `closure_level`: stable state with low unresolved pressure.
- `surprise_level`: mismatch from the listener twin's short-term expectation.

Do not include delayed values in the first report format. They should be added
only after a sample needs them and their input signals are clear.

`attention_level` and `neural_rhythms` are output state in the first
implementation. They must not act as direct generation commands.
They must also remain listener-side: do not feed generator vitality,
population energy, or habitat-only scaffold state into these values.
If rhythm persistence needs a gain, derive it from ListenerTwin's own
presentation-derived attention state, not from generator-side vitality.

For harmonic tension, the clean first definition is:

```text
tension_level = (1.0 - stability_level) * resolvability_level
```

This is not the same as:

```text
tension = current_instability
        + resolution_affordance
        + voice_leading_pressure
        + expectation_pressure
```

The additive form is better understood as a list of possible components. It
mixes different meanings unless each term is normalized and kept separately.
For the first design, keep the components visible and define the harmonic
tension scalar narrowly.

## Resolvability

`resolvability` should not hard-code `V7 -> I`.

It should ask a more general listener-side question:

```text
From the currently sounding region, is there a nearby, more stable region that
the listener can hear as a plausible continuation?
```

The first approximation can be:

```text
resolvability = reachable_stability_gain(current_audio_landscape)
```

Where reachable gain is the best local improvement in `consonance_field_level`
after subtracting a movement cost. This makes dominant-like resolution possible
when the consonance landscape and the current audible tones create that affordance,
without encoding a tonal grammar rule.

Use a simple log-frequency cost first:

```text
candidate_gain =
  candidate_stability_level
  - current_stability_level
  - movement_cost_per_oct * abs(candidate_log2 - current_log2)

resolvability_level = clamp(best_positive_gain / gain_scale, 0, 1)
```

For a multi-bin audible region, compute this per active bin and average by
presentation-derived intensity. Keep `movement_cost_per_oct` and `gain_scale`
internal constants until reports show that they need to be configurable.

Later versions can add idiom-specific expectation, learned cadence statistics,
or listener feedback, but those should be separate layers.

## Responsibility Boundary

`ListenerTwin` is responsible for:

- observing the audible presentation state.
- maintaining short-term listener state.
- exposing stability, resolvability, and tension.
- later, keeping listener memory and expectation state.
- later, exposing closure, surprise, and attention after their inputs are clear.

It is not responsible for:

- rendering audio.
- spawning voices.
- deciding exact pitches.
- owning population dynamics.
- defining public scenario verbs.

`Landscape` remains responsible for:

- log-frequency terrain representation.
- harmonicity, roughness, and consonance scans.
- field/density transformations.

`Population` / voice logic remains responsible for:

- individual movement and adaptation.
- survival, density, social pressure, and synchronization.
- converting external pressure into local behavior.

A future `DccCoupler` should be responsible for:

- converting `ListenerState` into weak ALife pressures.
- keeping those pressures distributed rather than issuing global commands.
- making the feedback strength configurable.

## Runtime Placement

Recommended data flow:

```text
voices/population
  -> presentation_bus
  -> analysis_worker
  -> presentation Landscape
  -> ListenerTwin
  -> ListenerState
  -> DccCoupler
  -> weak distributed ALife pressure
```

The existing habitat analysis path should remain separate:

```text
habitat_bus
  -> analysis_worker
  -> habitat Landscape
  -> voice terrain / UI
```

This distinction matters. If `ListenerTwin` reads the habitat bus,
hidden scaffolds can create fake listener tension. If it reads the presentation
bus, only audible structure affects listener-side state.

## Bootstrapping

Moving listener entrainment to the presentation bus creates a real cold-start
issue. A population can no longer lock to an inaudible scaffold before any sound
has reached the listener. This is the right DCC semantics, but the generator
still needs a startup path.

The first implementation keeps two rhythm surfaces:

- generator rhythm: derived from `habitat_bus`; used by current
  generator timing and prediction.
- listener entrainment: future presentation-derived fast path inside
  `ListenerTwin`; initially report-only.

When listener entrainment later feeds generation, use a confidence-weighted
blend. Start with generator rhythm, then shift pressure toward listener
entrainment only after the listener-side fast path has stable phase confidence.

## Migration From Existing Generator Model

The old mixed `WorldModel` name has been removed. The remaining generator-side
state lives in `src/life/generator_model.rs` as `GeneratorModel`.

| Existing surface | Current role | Target |
|---|---|---|
| Old `WorldModel::percept_landscape` | Nominal perceptual landscape slot. It was not the active prediction/control path. | Removed from generator state. A future stored presentation landscape belongs inside `ListenerTwin`. |
| `TerrainPredictor` and `predict_consonance_field_level_*` | Habitat terrain forecast used by population scheduling near the next gate. | Kept in `GeneratorModel`. Do not move this predictor into `ListenerTwin`. |
| `next_gate_tick_est` | Rhythm-derived timing estimate used by population phonation scheduling. | Kept in `GeneratorModel` for now. It is generator timing state, not listener tension state. |
| `DorsalMetrics` | Low-level rhythm flux metrics from the habitat path. | Removed from model storage; pass to UI/report as generator diagnostics. A listener rhythm feature must be presentation-derived. |
| `last_pred_next_gate` / prediction cache | Cache for population's next-gate habitat prediction. | Kept in `GeneratorModel`; it supports generation, not listener reporting. |

The clean target is:

- generator-side forecast: supports population movement and scheduling.
- listener-side twin: reports the state of what is actually heard.
- DCC coupler: converts listener-side state into weak distributed pressure.

## DCC Coupler Interface

The coupling layer should be narrow when it is added:

```text
DccCoupler {
  coupling_strength: f32 in [0, 1]
}
```

Semantics:

- `0.0`: listener state is report/UI only; generation is unchanged.
- `1.0`: listener pressure is fully applied by whatever pressure mapping is
  chosen.

Do not add a no-op code skeleton before the first concrete pressure consumer.
The interface belongs in the design now; the module belongs with the first
actual coupling implementation.

## Reporting

The first implementation should expose `ListenerState` in reports before using
it to control behavior. This gives a way to inspect whether tension, closure,
and surprise match what is heard.

Expected JSONL event:

```json
{
  "type": "listener_state",
  "time_sec": 1.25,
  "analysis_lag_frames": 2,
  "attention_level": 0.31,
  "theta_hz": 5.84,
  "theta_mag": 0.42,
  "theta_alpha": 0.74,
  "stability_level": 0.72,
  "resolvability_level": 0.64,
  "tension_level": 0.18
}
```

Do this before wiring `ListenerState` into generation. If the reported state is
not musically legible, coupling it back into ALife will only hide the problem.

## Phase 1 Implementation

Implemented in:

- `src/listener_twin/mod.rs`
- `src/runtime/mod.rs`
- `src/life/report.rs`

Current behavior:

- `ListenerTwin` is passive.
- It reads presentation-derived audio for fast attention / rhythm state.
- It reads presentation-derived analysis for stability / resolvability /
  tension.
- It reports `listener_state` JSONL records when `--report` is enabled and
  exposes the latest state to the GUI.
- It does not change population movement, pitch, rhythm, spawning, or audio
  rendering.
- It keeps `GeneratorModel` and its habitat `TerrainPredictor` separate.

## Implementation Phases

### Phase 1: Passive Twin

- Add a small `listener_twin` module.
- Compute `ListenerState` from presentation-derived `Landscape`.
- Compute listener-side `attention_level` and `neural_rhythms` from
  presentation audio.
- Report `listener_state` in headless reports.
- Show the latest listener state in the GUI when the presentation analysis path
  is active.
- Include analysis latency telemetry: generated frame id, analysis frame id, and
  `analysis_lag_frames`.
- Verify that habitat-only hidden scaffold changes do not affect
  `listener_state`.
- Do not change audio generation yet.

### Phase 2: Validation Samples

- Add or revise one tension/resolution sample so the report can be compared
  with what is heard.
- The sample should not hard-code `V7 -> I`; it should make the landscape
  affordance audible.
- Check that hidden anchors do not affect reported listener tension.

### Phase 3: Weak DCC Coupling

- Add a narrow coupling layer that reads `ListenerState`.
- Convert high tension / high resolvability into small distributed pressure.
- Keep the coupling weak enough that voice dynamics remain ALife-like.

### Phase 4: Public Bus Naming

- Public scripting uses `habitat_bus` and `presentation_bus`.
- Do not keep a `field_bus` alias.
- GUI should show `Auditory salience` and `Neural rhythm` only from
  ListenerTwin, not from habitat-derived generator state.
- The Listener Twin mandala should keep the listener-side delta/theta rhythm
  display and layer compact visual cues for attention, stability,
  resolvability, and tension. Numeric details can live below the time-series
  plots.

### Phase 5: Memory And Expectation

- Add short-term expectation only after passive state is useful.
- Add `surprise_level` and `closure_level` only after their inputs are explicit.
- Add phrase/event memory only after a concrete sample needs it.
- Keep `Memory` scoped as `ListenerMemory` or inside `ListenerTwin`, never as an
  unqualified global `Memory`.

## Non-Goals

- Do not implement tonal grammar as fixed rules.
- Do not make `dominant -> tonic` a scripted transition.
- Do not expose public scenario API until the internal state is reportable and
  audible.
- Do not merge listener state into `Landscape`.
- Do not let hidden habitat structures count as heard tension.
- Do not move the existing habitat `TerrainPredictor` into
  `ListenerTwin`.

## Open Questions

- Should the first `resolvability` use only `consonance_field_level`, or also
  roughness and harmonicity components separately?
- Should `closure` be derived from low tension, high stability, expectation
  satisfaction, or a separate phrase-boundary model?
- What report plots are needed to compare heard tension with `ListenerState`?
