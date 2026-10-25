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
state. By default, each Flora removes its own actual habitat PCM before evaluating the base
consonance and harmonicity used for growth and seed rain, as well as external
resource power. Other Flora remain audible. Within-band competition still comes
from the resource consumption kernel acting on latent biomass.
`send(presentation_bus)` keeps its audio out of the habitat, so that Flora reads
mixed C/H directly without an H/R observer. `send(habitat_bus)` affects the habitat
without direct presentation. The shared beat meter still hears all habitat audio.

Set `flora().leave_self_out(false)` before placement to use mixed C/H including
its own sound, without an H/R observer. External resource power still excludes
its own PCM. Voice defaults to `false`, while Flora defaults to `true` because
one Flora has many partials whose own harmonic lattice can sustain its growth
and seed positions. In the undergrowth-v2 observation, that lattice persisted
without a ground drone; choose `false` to allow this self-support, without a
guarantee that the pitch stays fixed. Presentation-only Flora uses mixed C/H
with either setting.

Resource consumption includes latent biomass, even when it does not sound, and the
power of other habitat audio. The fixed power reference is the
mean square of a sine with amplitude 0.03 (`0.03²/2 = 0.00045`); changing the Flora's
amplitude does not change that reference. Missing observations do not count as
silence: resources retain their initial state before the first valid observation,
and later gaps retain the last observed external consumption. Growth observations
start from the mixed pre-birth history. Analysis invalidation ends the runtime,
following its existing mixed-analysis rule; own-excluded C/H never falls back to
mixed C/H. A geometry change discards old growth/seed scans until new observations
are available. Reports show the growth and seed values actually consumed.

`flora()` builds a reusable FloraSpec. `place(spec, placement)` returns a Flora that
can receive amplitude and pace updates. A Population's Voices and
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
with a uniform fallback when that mass is zero. This placement value is mixed:
no own sound exists yet. `random(lo, hi)` seeds uniformly.
Both require density sampling, count 1, zero tension and the target's default spacing
(1 ERB for consonance, 0 for random). Explicit default modifiers are accepted by value.
Other placement targets, `at`, `line`, `.peak()` and non-default modifiers raise script
errors. Bounds must lie within 1–20000 Hz and span at least 1/96 octave, one bin of the
initial Log2Space. Later space changes can make a Flora dormant until its extent
contains bins again.

| Method | Accepted values | Default | Live update |
|---|---|---|---|
| `pace` | 1/64–4 | 1 | Yes |
| `amp` | 0–1 | 0.03 | Yes |
| `fade` | 0–60 seconds | 2 | Before placement |
| `tone` | `"sine"`, `"resonant"` | `"sine"` | Before placement |
| `residual` | Boolean; diffuse mass as band noise | `false` | Before placement |
| `leave_self_out` | Boolean; exclude own sound from growth and seed C/H | `true` | Before placement |
| `send` | Bus or BusSet | Both buses | Before placement |

`residual(true)` enables the optional band-noise layer before placement.

`amp` is the nominal target-amplitude scale: a ceiling, not a guaranteed level.
Fades, slews and retiring tones can affect the transient output.

`amp` and `fade` must be finite before and after conversion to f32. Invalid values
raise errors during script evaluation. `pace` instead warns and clamps to its
supported bounds. NaN and negative infinity use the lower bound; positive infinity
uses the upper bound. The bounds were acquired at 48 kHz, hop 512 and the default
grid: the lower bound protects finite retirement at 0.5 Hz and the upper bound
keeps full-extent advance below one observer's cost at 4 Hz.

Live methods collect a pending patch. The next `wait`, `flush` or scope end emits
it as UpdateFlora. `release(m)` marks release pending; the patch precedes ReleaseFlora
in the same event, using the spec's fade. Further updates and repeated releases on that
handle warn and are ignored. Scoped verbs release the Flora they create, with nested
Flora released at the inner scope's end. Explicit release fades extend the scripted
duration when needed. Flora still live at script end fades after Finish.

The Flora API is Experimental. Its dynamics are authored composition rules; this API tier is separate from claims about auditory cognition.

## Beat opportunities

Every Flora uses beat opportunities: with fresh confirmed evidence, new tonal lanes
and ordinary replacements wait for the Flora's next local clock crossing. Biomass
and resources continue to evolve every hop. There is no timing opt-out;
`flora().metric()` has been removed.

Like Population `metric()`, Flora uses a private CouplingClock with confidence-weighted
rate and phase attraction (coupling 0.95), an independent initial phase, and the
scene's tempo-basin center as its starting rate (2 Hz without a basin). Its shared
meter reads only habitat audio: flux and the selected acoustic recurrence path,
with the same enculturation priors. Internal Voice onset commands and Flora lane
commands never enter this meter. The meter runs from scene start, before Flora is
placed, and includes Flora's actual habitat sound. That sound may confirm and sustain
a beat. This is separate from the production meter used by Population timing.

A waiting tone owns only the current excess in its last tonal window. That mass
is removed once from the residual. Empty windows, out-of-range tones, separation
conflicts and release start retirement immediately. New assignments occur at the
sample crossing within the hop; births retain their amplitude slew while retiring
tones use the metabolic tail described below. Thus neither
all endings nor detected acoustic onsets are quantized to the beat.

Before beat confirmation, or while the observation is unknown or stale, allocation
runs continuously each hop. Pending candidates are discarded and current biomass is reconsidered; the
private clock is not reset and stored candidates are not replayed as a burst.

## Pace and release

All Flora uses `temporal_basin(lo, hi)` as the composer's tempo prior, starting at
its center frequency (2 Hz without a basin).
Every Flora follows fresh, confirmed beats with confidence-weighted coupling
0.95 and otherwise returns to the center. `flora().pace(k)` sets an authored
multiplier relative to that beat; a placed handle also accepts `m.pace(k)`. The
default is 1. For frequency f, ecological time advances by k*f/2: growth is
4kf/s, resource recovery is 20/(kf) seconds and consumption is 0.6/(kf) seconds. Diffusion and seed rain scale together. Phase correction only
changes metric opportunities. This authored rule does not model phrase duration.

All lanes retire with a finite metabolic amplitude tail. The existing macro
`fade` bounds release; `fade(0)` ends immediately. The fixed lane pool can postpone
births while tails occupy slots. The independent `rate` and `renewal` methods
have been removed. `pace(k)` scales their shared time and divides the retirement
tail duration by k at fixed f. Birth slew and macro fade retain physical seconds;
pace does not change the private beat clock or its opportunities.

Sample 17 uses `temporal_basin(2, 4)`: its 3 Hz prior is 1.5 times neutral 2 Hz.
The sustained scene has no explicit beat carrier; any confirmed acoustic beat can
still change Flora's pace. The basin is a compositional choice, not a new default
or a measured optimum.
