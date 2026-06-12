# Études

The samples are a book of twelve études — small pieces, in order. Played and
read one after another, they are the instrument: each is written around one
of its capacities, and the script itself is the score.

```bash
cargo run --release -- samples/01_a_single_voice.rhai
```

1. **A Single Voice** — one voice appears, holds its breath, and leaves.
2. **Constellation** — four ways to enter: a line, the peaks, the density,
   chance.
3. **Gravity** — the same root under two suns; the peaks listen to what is
   sounding, not to a chart.
4. **Tension** — the director leans on the terrain until it strains, then
   lets the gravity come home.
5. **Settling** — scattered voices glide to where the field can hold them.
6. **Bells** — struck bodies; the last bell lets the field choose its
   partials.
7. **Heartbeat** — no clock is imposed; a pulse forms because the population
   drives one.
8. **Murmuration** — a flock drifts into step, never commanded.
9. **Rain** — time without a beat, falling along the field.
10. **Generations** — voices live, starve, and are reborn where harmony can
    hold them.
11. **Autumn Cycle** — a directed harmony; the season turns and comes home.
12. **Emergence and Resolution** — everything at once, bent into a single
    arc.

Études 1–6 walk the consonance terrain (placement, gravity, tension,
movement, timbre); 7–9 walk the rhythm continuum one region at a time;
10 closes the loop into life; 11–12 are directed by the composer. All études
are compile-checked by the test suite, so they always match the current API.

## Research assays

`samples/research/` holds comparison fixtures — heredity/selection ablations,
external-scaffold rhythm controls, and mechanism studies. They study the
instrument rather than play it, and are not part of the path.

## Offline rendering

The `conchordal` instrument never writes audio to disk — performances are
ephemeral by design. For offline WAV rendering use the separate
`conchordal-render` binary, which shares the core engine but is not the
instrument.
