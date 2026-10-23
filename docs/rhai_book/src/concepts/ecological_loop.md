# Voice and Landscape — Sound–Environment Feedback

A Voice routed to the habitat bus changes the Landscape. The changed
Landscape then affects placement, movement, and viability. This chapter traces
that runtime feedback path and the energy, death, and respawn processes it
supports.

```text
Voice bodies
    |-- presentation bus --> listener / Listener Twin
    |
    `-- habitat bus --> Landscape --> placement, movement, viability
                              ^                         |
                              `------ new sound <------'
```

By default every voice feeds both buses, so the audience and the ecology share
the same physical event. Routing can split them deliberately; see
[Routing and the Listener Twin](routing.md).

## From sound to terrain

The Landscape analyzes the habitat bus in log-frequency space and computes:

- **roughness potential**: interference and beating within critical bands;
- **harmonicity potential**: periodicity and support for virtual roots;
- **Consonance Field**: the combined terrain used to evaluate candidate
  frequencies.

A voice changes the spectrum on the habitat bus, which changes these scans.
That is why `consonance(...).peak()` does not return a fixed scale degree: its
answer depends on what is sounding now.

## Score, level, mass, and density

Several representations are derived from the same terrain. They are related,
but they are not interchangeable.

| Representation | Range | Used for |
|---|---:|---|
| potential | kernel-dependent | Raw roughness and harmonicity output. |
| field score | unbounded real value | Comparing positions, hill-climb, and placement tension. |
| field level | `0..1` | Bounded behavior and viability signals. |
| density mass | non-negative | Weight before normalization for stochastic placement. |
| density / PMF | sums to 1 in the selected range | Sampling a density cloud. |

`.peak()` selects an extremum. The default density placement samples a PMF, so
several voices can form a cloud around supported regions instead of collapsing
onto one bin. `tension(degree)` targets a field-score step below the strongest
peak. A viability window reads a bounded or environment-relative fit signal.

The suffixes in reports preserve the same distinction: for example,
`mean_c_field_score` and `mean_c_field_level` are different quantities.

## Placement is not movement

Placement answers where a voice enters. Pitch behavior answers what happens
after entry.

```rhai
let fixed_strain = harmonic()
    .amp(0.035)
    .sustain()
    .anchor();

let resolving_strain = harmonic()
    .amp(0.035)
    .sustain()
    .seek_consonance()
    .glide(0.4);

section("two responses to dissonance", || {
    place(fixed_strain, dissonance(140.0, 900.0).count(3));
    place(resolving_strain, dissonance(140.0, 900.0).count(3));
    wait(6.0);
});
```

Both Populations enter a dissonant region. One holds it; the other treats it as a
starting point for resolution.

## Evaluate the environment, not the self

A voice contributes energy to the field it later evaluates. Without care, a
strong voice could appear viable merely because it hears its own footprint.
`consonance_viability()` therefore enables environment-relative evaluation by
default: the system removes the Voice's actual habitat PCM before
analysing the remaining sound and judging its fit.

This is a survival rule, not a routing rule. The voice may still feed the
habitat bus and reshape the terrain for every other voice. Use
`viability_scope("total")` only when the intended question is explicitly
whether the voice fits the total field including itself.

Pitch movement has its own switch, `leave_self_out(true)`. Both consumers read
one synchronized observation of the mixed habitat PCM minus that Voice's actual
habitat PCM. The remainder passes through the same causal auditory filterbank, hop statistics, frequency reassignment,
critical-band compression, density smoothing and H/R/C pipeline as
the mixed field. Habitat-routed Flora, including residual sound, stays in the
remainder. All bodies use this method; there is no mode selector.

Observers are reserved from birth for populations that use either consumer,
including a later `leave_self_out(true)` patch. Before birth their state is the
mixed stream's state. Voice identity and generation bind the observation to its
consumer. At the analysis-worker boundary, gaps reset the mixed and per-Voice
histories; valid observations resume after the common analysis window fills if
further PCM arrives. The field runtime currently ends the run when disconnected
or invalidated observations leave no usable frame within its fixed one-hop lag
limit. It does not keep generating warmup hops for automatic recovery.

Environment viability reads the current pitch even without pitch movement. The
ordinary mixed-field level used for basal drain and telemetry is unchanged.

## Energy, death, and replacement

The ecological lifecycle belongs to `brain("entrain")`. Its energy is
normalized to `0..1`:

1. `endurance(seconds)` establishes the nominal zero-fit lifetime.
2. Each attack spends `attack_cost_fraction`.
3. A consonant attack can restore up to `attack_recharge_fraction`.
4. `recovery(seconds)` enables continuous recovery; the viability window
   determines how much of that recovery is available at the current pitch.
5. At zero energy the voice dies and enters its release tail.
6. If a respawn policy exists, the Population may create a replacement.

```rhai
let settlement = consonance(70.0, 1100.0).spacing(0.8);

let ecology = harmonic()
    .brain("entrain")
    .entrained()
    .cycles(2)
    .seek_consonance()
    .endurance(8.0)
    .recovery(4.0)
    .attack_cost_fraction(0.017)
    .attack_recharge_fraction(0.70)
    .consonance_viability(0.32, 0.82)
    .respawn_consonance()
    .respawn_capacity(8)
    .respawn_settle(settlement);

place(ecology, consonance(70.0, 1100.0).count(8));
wait(20.0);
```

Respawn policies answer different compositional questions:

- `respawn_random()` creates no parent lineage. Candidates come from the
  Population's original Placement and are weighted by the current scene score;
  it is not uniform random placement.
- `respawn_hereditary(sigma_oct)` selects a living parent by energy, proposes
  offspring near it, and keeps the candidate with the best current Field level.
- `respawn_consonance()` selects an energy-weighted living parent and chooses
  among high field-score peaks with a bias around that parent.
- `respawn_capacity(n)` bounds how many living members the Population maintains;
  without it, the founder count is the capacity, and an explicit value cannot
  be lower than that founder count.
- `respawn_settle(placement)` adds candidates from that Placement; the
  respawn policy's own baseline still contributes one candidate.

Respawn preserves the `Population` and its `population_id` while individual
`voice_id` and `generation` values change. The report stream makes that
turnover observable.

## Runtime feedback and human revision

Two different processes are involved:

- **Runtime feedback** is automatic: sound reshapes the terrain, which changes
  voice behavior and survival.
- **Human revision** happens between runs: run, listen, inspect a report,
  revise the scenario, and run again.

The practical run-and-revise procedure is described in
[Performance](../tutorial/observing.md).
