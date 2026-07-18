# Population — A Persistent Unit of Voices

The shortest useful mental model of conchordal is not a note followed by
another note. It is a population definition becoming a persistent population of living
voices:

```text
PopulationSpec + Placement --place()--> Population --> Voice(s)
                                                |
                                                `-- later generations

Community = every Population sharing the runtime terrain
```

A **PopulationSpec** is a reusable pre-placement definition. It combines the
founder Voice defaults with policy that belongs to the population as a whole,
such as lifecycle, viability, and respawn. A **Placement** says where and how
many founder voices enter. `place()` combines them and immediately returns a
**Population**: one stable handle for the placed population, not one handle per
Voice. The runtime **Community** is the aggregate of all Populations that share
the Landscape.

```rhai
let population_spec = harmonic()
    .amp(0.035)
    .sustain()
    .respawn_capacity(6);

let population = place(
    population_spec,
    consonance(90.0, 900.0).count(6)
);
wait(3.0);
release(population);
```

Here `population` controls six founder Voices together. Reports distinguish
their shared `population_id` from each individual `voice_id`. If a Voice dies
and is replaced, its `voice_id` and `generation` change; the Population and its
`population_id` do not.

## The placement boundary

`place()` is the only transition from definition to runtime. Configure all
initial-only properties on the `PopulationSpec` first. Once placed, a
`Population` exposes only live patches and release; it cannot be turned back
into a specification or have its founder policy rewritten.

```rhai
let spec = harmonic()
    .brightness(0.4)
    .brain("entrain")
    .endurance(8.0);

let population = place(spec, consonance(220.0).count(3));
population.amp(0.03); // Live patch at the current script time.
flush();              // Emit pending live patches without advancing time.
wait(2.0);
release(population);
```

`wait(seconds)` also emits pending live patches, then advances the script
cursor. Neither `wait()` nor `flush()` creates a deferred or draft Population:
the founder Voices were scheduled by `place()` itself.

The [API Reference](../reference/api.md) marks `PopulationSpec` methods as
initial-only and `Population` methods as live-patchable.

## Five independent questions

A PopulationSpec answers several independent questions. Keeping them separate
prevents one musical decision from being mistaken for another.

| Question | Main controls | Meaning |
|---|---|---|
| What sounds? | `sine`, `harmonic`, `modal`, `brightness`, `modes` | The founder Voice body and spectrum. |
| What kind of life? | `brain(name)`: `entrain`, `seq`, `drone` | Whether articulation participates in ecology, follows an authored life, or persists as terrain. |
| When does it sound? | `sustain`, `repeat`, `metric`, `entrained`, `flow` | Phonation and onset timing. |
| Where does pitch go? | Placement, `anchor`, `seek_consonance`, `temperature` | Founder entry position and later movement. |
| How does the population persist? | `endurance`, `recovery`, viability, respawn | Voice energy, death, and Population turnover. |

Calls on different rows compose. Calls on the same axis are generally
last-write-wins; the API Reference identifies the exact builder behavior.

## Articulation life: `brain`

`brain(name)` selects how a Voice lives while it is sounding:

- `brain("entrain")` is the default living articulation. It can respond to
  consonance and rhythmic fit through the metabolism and lifecycle controls.
- `brain("seq")` is an authored event with a fixed life. It ignores field
  viability and metabolism.
- `brain("drone")` is undying until explicitly released. It is useful for
  terrain anchors and other persistent material.

This is separate from phonation timing. In particular:

> `brain("entrain")` chooses a kind of life; `.entrained()` chooses medium
> coupling of repeated onsets to the shared meter.

The similar names describe different axes and may be used together.

```rhai
let colony = harmonic()
    .brain("entrain")
    .entrained()
    .cycles(2)
    .seek_consonance()
    .endurance(8.0)
    .recovery(4.0)
    .consonance_viability(0.30, 0.80);

place(colony, consonance(90.0, 900.0).count(5));
wait(8.0);
```

The brain does not choose the body, Placement, or pitch strategy. A drone may
be audible or habitat-only; a living Voice may be anchored or moving; the same
modal body may use any articulation life.

## Phonation and duration

Phonation answers two questions: when an onset occurs, and how long that onset
remains open.

- `sustain()` holds while the Voice is alive.
- `repeat()` selects repeated phonation with defaults.
- `metric()`, `entrained()`, and `flow()` select regions of the shared-meter
  coupling continuum and imply re-attacking behavior.
- `cycles(n)` expresses duration in rhythmic cycles.

The lower-level `once()`, `pulse(rate_hz)`, `while_alive()`, and adaptive
duration controls are available when the presets do not express the intended
gesture. Start with the presets; use explicit timing only when the piece needs
it.

## Release is not death

`release(population)` is a terminal script decision: later patches on that
Population are ignored, its current Voices enter their release envelopes, and
the Population closes. Ecological death is a runtime
result: one living Voice exhausts its energy, after which the Population's
respawn policy may replace it. A `section` or `play` scope also releases the
Populations it created when the scope ends.

The next chapter explains
[Voice and Landscape — Sound–Environment Feedback](ecological_loop.md).
