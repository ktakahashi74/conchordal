# Samples

This is an ordered set of twelve small demonstrations. Run and read them to
see the instrument's main capabilities one at a time. They are API and
behavior samples, not musical works.

```bash
cargo run --release -- samples/01_a_single_voice.rhai
```

1. **A Single Voice** — one voice appears, holds its breath, and leaves.
2. **Constellation** — six ways to enter: a line, the peaks, the density, the
   strain, the gaps, and chance.
3. **Gravity** — the same root under two suns; the peaks listen to what is
   sounding, not to a chart.
4. **Tension** — the voices settle onto consonance, grow restless and stray,
   then cool and settle again.
5. **Settling** — scattered voices glide to where the field can hold them.
6. **Bells** — struck bodies; the last bell lets the field choose its
   partials.
7. **Heartbeat** — no external scaffold is imposed; the population locks into a
   shared pulse.
8. **Murmuration** — a flock drifts into step, never commanded.
9. **Rain** — time without a beat, falling along the field.
10. **Generations** — voices live, starve, and are reborn where harmony can
    hold them.
11. **Autumn Cycle** — a directed harmony; the season turns and comes home.
12. **Emergence and Resolution** — everything at once, bent into a single
    arc.

Samples 1–6 walk the consonance terrain (placement, gravity, tension,
movement, timbre); 7–9 walk the rhythm continuum one region at a time;
10 closes the loop into life; 11–12 combine several mechanisms. All samples
are compile-checked by the test suite, so they always match the current API.
Top-level samples intentionally do not call `seed(...)`: each run starts from
a fresh scenario seed. Research assays keep fixed seeds so comparisons remain
reproducible.

One craft rule holds across the path: **scaffolding is inaudible or
embodied**. Terrain anchors sing to the habitat bus, not to the audience,
unless the drone is itself the subject, as with the suns of *Gravity*. Pulse
carriers get resonant bodies and lives of their own, or the pulse is left to
condense from the colony.

## Research assays

`samples/research/` holds comparison fixtures — heredity/selection ablations,
external-scaffold rhythm controls, and mechanism studies. They study the
instrument rather than play it, and are not part of the path.

## Offline rendering

The `conchordal` instrument never writes audio to disk — performances are
ephemeral by design. For offline WAV rendering use the separate
`conchordal-render` binary, which shares the core engine but is not the
instrument.
