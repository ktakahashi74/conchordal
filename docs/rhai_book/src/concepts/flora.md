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
the reverse. The ecology continues on the available terrain without self-perception.

Resource consumption includes latent biomass, even when it does not sound, and the
power of other habitat audio. Each Flora removes its own actual habitat PCM before
measuring that external power. The base consonance and harmonicity used for growth
and seeding still include its own habitat sound. The fixed power reference is the
mean square of a sine with amplitude 0.03 (`0.03²/2 = 0.00045`); changing the Flora's
amplitude does not change that reference. Missing observations do not count as
silence: resources retain their initial state before the first valid observation,
and later gaps retain the last observed external consumption.

`flora()` builds a reusable FloraSpec. `place(spec, placement)` returns a Flora that
can receive amplitude updates. A Population's Voices and
Flora can share the same habitat, as in this example:

```rhai
let sun = place(harmonic().brain("drone").sustain().amp(0.04), at(110));
let moss = flora().amp(0.03).fade(0.5)
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
| `fade` | 0–60 seconds | 2 | Before placement |
| `tone` | `"sine"`, `"resonant"` | `"sine"` | Before placement |
| `residual` | Boolean; diffuse mass as band noise | `false` | Before placement |
| `metric()` | Beat opportunities for lane changes | Off (T0) | Before placement |
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

## Beat opportunities

`flora().metric()` opts into T1: new tonal lanes and ordinary replacements wait
for the Flora's next local beat crossing. The default T0 updates allocation every
hop. Biomass and resources continue to evolve every hop in both modes.

Like Population `metric()`, Flora uses a private CouplingClock with confidence-weighted
rate and phase attraction (coupling 0.95), an independent initial phase, and the
scene's tempo-basin center as its starting rate (2 Hz without a basin). Its shared
meter reads only habitat audio: flux and the selected acoustic recurrence path,
with the same enculturation priors. Internal Voice onset commands and Flora lane
commands never enter this meter. This is separate from the production meter used
by Population timing.

A waiting tone owns only the current excess in its last tonal window. That mass
is removed once from the residual. Empty windows, out-of-range tones, separation
conflicts and release start retirement immediately. New assignments occur at the
sample crossing within the hop; births retain their amplitude slew while retiring
tones use the metabolic tail described below. Thus neither
all endings nor detected acoustic onsets are quantized to the beat.

Before beat confirmation, or while the observation is unavailable, allocation uses
T0. Pending candidates are discarded and current biomass is reconsidered; the
private clock is not reset and stored candidates are not replayed as a burst.

## Pace and release

All Flora uses `temporal_basin(lo, hi)` as the composer's tempo prior. T0 keeps
its center frequency (2 Hz without a basin) and never follows heard beats.
Metric Flora follows fresh, confirmed beats with confidence-weighted coupling
0.95 and otherwise returns to the center. For frequency f, ecological time
advances by f/2: growth is 4f/s, resource recovery is 20/f seconds and consumption
is 0.6/f seconds. Diffusion and seed rain scale together. Phase correction only
changes metric opportunities. This authored rule does not model phrase duration.

All lanes retire with a finite metabolic amplitude tail. The existing macro
`fade` bounds release; `fade(0)` ends immediately. The fixed lane pool can postpone
births while tails occupy slots. The independent `rate` and `renewal` methods
have been removed. Set the scene's tempo prior instead.

Sample 17 uses continuous T0 and `temporal_basin(2, 4)`: its fixed 3 Hz center is
1.5 times neutral 2 Hz. The sustained scene has no beat carrier. This is a
compositional choice, not a new default or a measured optimum.
