# Routing and the Listener Twin

## Two buses

Each voice contributes to two independent mono buses:

- **presentation bus** → cpal output / offline render / UI metering (the work
  as presented)
- **habitat bus** → NSGT analysis → landscape (what the ALife ecology responds
  to)

By default both buses receive the voice. Use `send()` when a voice should feed
only one side, or combine buses with `|`:

```rhai
// Reference anchor: sensed by the ecology, absent from the presented sound.
let anchor = harmonic().brain("drone").send(habitat_bus);

// Presented decor that does not influence the ecology.
let decor = sine().send(presentation_bus);

// Explicitly both (the default).
let normal = harmonic().send(habitat_bus | presentation_bus);

place(anchor, at(110.0));
place(decor, at(880.0));
place(normal, consonance(110.0).peak().count(3));
wait(4.0);
```

Keeping the buses separate is a composition tool. A hidden anchor on
`send(habitat_bus)` can shape how the population organizes without ever
becoming audible, and presented decor on `send(presentation_bus)` can be heard
without perturbing the ecology.

## The Listener Twin

conchordal keeps a `ListenerTwin`: a listener-side model of the *presented*
sound only. It never reads the habitat bus, so hidden scaffolds cannot create
fake listener tension. It reports five state values:

- `stability_level`: how stable / consonant the current audible sound is.
- `resolvability_level`: whether a nearby audible state offers a plausible,
  more stable continuation.
- `tension_level`: `(1 - stability) * resolvability` — unstable now, but with
  a reachable path toward improvement.
- `attention_level`: presentation-derived onset / spectral-flux salience.
- `neural_rhythms`: presentation-derived listener-side rhythm (delta/theta).

There is no scripting verb for the twin. It is observed, not commanded: when
you run with reporting enabled it emits `listener_state` records, and the GUI
shows the same state. Use it to check whether the tension you hear matches
what the twin reports before coupling it back into generation.

## DCC: coupling the twin back

That optional coupling is **DCC**, configured in `conchordal.toml`, not in
script:

```toml
[dcc]
# Listener pressure is report/UI-only by default.
# coupling_strength = 0.0
# max_exploration_bonus = 0.10
```

- `coupling_strength` (`0.0`–`1.0`, default `0.0`): at `0.0` the twin is
  report/UI-only and generation is unchanged. Above `0.0` it applies
  `tension_pressure = tension_level * resolvability_level * coupling_strength`
  as a transient pitch-exploration bonus only. It never sets target pitches or
  changes rhythm synchronization.
- `max_exploration_bonus` (default `0.10`): ceiling on that transient bonus.

Raise `coupling_strength` gradually, and only after the reported
`listener_state` looks musically legible.
