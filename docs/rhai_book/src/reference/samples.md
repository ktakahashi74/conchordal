# Samples

Sixteen samples in two groups, kept together in `samples/`. Samples 1–12
introduce the instrument's capabilities one at a time: API and behavior
demonstrations, not musical works. Samples 13–16 are longer scenes written
as compositions combining those capabilities. Run and read both groups.

```bash
cargo run --release -- samples/01_a_single_voice.rhai
```

## Capability demonstrations (1–12)

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
7. **Heartbeat** — a dedicated downbeat Voice supports an explicit
   synchronization demonstration.
8. **Murmuration** — a flock drifts into step, never commanded.
9. **Rain** — time without a beat, falling along the field.
10. **Generations** — voices live, starve, and are reborn where harmony can
    hold them.
11. **Autumn Cycle** — a directed harmony; the season turns and comes home.
12. **Emergence and Resolution** — a colony heats up and settles, with rhythmic
    input from its own onsets and no dedicated beat carrier.

## Composed scenes (13–16)

13. **Three Windows** — three lines attack on pitch adoption and leave by their own doors.
14. **Stone and Wire** — two metrical voices turn while an independent line crosses.
15. **Folded Air** — a companion rests and returns, changing articulation across sections.
16. **Open Court** — a denser crowd fills the court around two independent paths.

## Reading and running

Samples 1–6 walk the consonance terrain (placement, gravity, tension,
movement, timbre). Samples 7–9 introduce three timing intentions: coupling
to a shared beat (`metric()`), participation informed by heard sound without
targeting shared-beat phase (`entrained()`), and non-metric clustered
renewal (`flow()`). Sample 10 closes the loop into life; 11–12 combine
several mechanisms. Samples 13–16
combine adoption-driven attacks, metrical voices with a crossing line, a resting
companion with changing articulation, and a denser crowd. All samples
are compile-checked by the test suite, so they always match the current API.
Top-level samples intentionally do not call `seed(...)`: each run starts from
a fresh scenario seed. Research assays keep fixed seeds so comparisons remain
reproducible.

Terrain anchors generally feed the habitat bus without being heard; in
*Gravity*, the sounding drone is itself the subject. *Heartbeat* includes a
dedicated beat carrier because synchronization is the demonstration's subject.
The other top-level samples have no dedicated beat carrier.

## Research assays

`samples/research/` holds comparison fixtures — heredity/selection ablations,
external-scaffold rhythm controls, and mechanism studies. They study the
instrument rather than play it, and are not part of the path.
Dedicated synchronization support is limited to demonstrations or assays
where explicit synchronization is essential. Sending that support only to
the habitat bus does not make it suitable for other samples.

## Offline rendering

The `conchordal` instrument never writes audio to disk — performances are
ephemeral by design. For offline WAV rendering use the separate
`conchordal-render` binary, which shares the core engine but is not the
instrument.
