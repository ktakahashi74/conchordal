# Flora — Spectral Growth

Flora grows spectral biomass across a frequency extent. Growth consumes a slowly
renewing band resource, and a patch needs a minimum density to sustain itself.
New growth appears where the heard sound is harmonic and the resource is fresh.
Silence is a legitimate state. These dynamics are authored composing rules, not
claims about auditory cognition or a guarantee of continuing succession.

Population and Flora couple only through sound: habitat audio feeds the Landscape,
and Flora reads its base field level before habituation and its harmonicity.
Resource depletion and recovery already supply Flora's band fatigue; Voices and the
listener still read the habituated terrain. There is no direct exchange of their
state. With both buses selected,
Flora also hears its own sound through this analysis path. `send(presentation_bus)`
removes its sound from the habitat while keeping it audible; `send(habitat_bus)` does
the reverse. Resource consumption follows biomass, including biomass that does not
sound. The ecology continues on the available terrain without self-perception.

`flora()` builds a reusable FloraSpec. `place(spec, placement)` returns a Flora that
can receive amplitude, growth-rate and resource-renewal updates. A Population's Voices and
Flora can share the same habitat, as in this example:

```rhai
let sun = place(harmonic().brain("drone").sustain().amp(0.04), at(110));
let moss = flora().amp(0.03).renewal(10).fade(0.5)
    .tone("sine").residual(false).send(habitat_bus | presentation_bus);
let m = place(moss, consonance(110).range(1, 16));
wait(2);
m.amp(0.015);
release(m);
wait(1);
release(sun);
```

`consonance(lo, hi)` seeds from the base consonance density mass before habituation inside the extent,
with a uniform fallback when that mass is zero. `random(lo, hi)` seeds uniformly.
Both require density sampling, count 1, zero tension and the target's default spacing
(1 ERB for consonance, 0 for random). Explicit default modifiers are accepted by value.
Other placement targets, `at`, `line`, `.peak()` and non-default modifiers raise script
errors. Bounds must lie within 1–20000 Hz and span at least 1/96 octave, one bin of the
initial Log2Space. Later space changes can make a Flora dormant until its extent
contains bins again.

| Method | Accepted values | Default | Live update |
|---|---|---|---|
| `amp` | 0–1 | 0.03 | Yes |
| `rate` | 0.05–8 per second | 4 | Yes |
| `renewal` | 0.5–600 seconds; band resource recovery time | 10 | Yes |
| `fade` | 0–60 seconds | 2 | Before placement |
| `tone` | `"sine"`, `"resonant"` | `"sine"` | Before placement |
| `residual` | Boolean; diffuse mass as band noise | `false` | Before placement |
| `send` | Bus or BusSet | Both buses | Before placement |

`residual(true)` enables the optional band-noise layer before placement.

`amp` is the nominal target-amplitude scale: a ceiling, not a guaranteed level.
Fades, slews and retiring tones can affect the transient output.

Numeric arguments must be finite before and after conversion to f32. Invalid values
raise errors during script evaluation; they are not clamped.

Live methods collect a pending patch. The next `wait`, `flush` or scope end emits
it as UpdateFlora. `release(m)` marks release pending; the patch precedes ReleaseFlora
in the same event, using the spec's fade. Further updates and repeated releases on that
handle warn and are ignored. Scoped verbs release the Flora they create, with nested
Flora released at the inner scope's end. Explicit release fades extend the scripted
duration when needed. Flora still live at script end fades after Finish.

The Flora API is Experimental. Its dynamics are authored composition rules; this API tier is separate from claims about auditory cognition.
