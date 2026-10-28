# Rain: Impacts Drawn from Distributions, and the Objects They Strike

Status: experimental v1 on `work/rain-20261027`, for the author's audition. Not merged. Sample 18 (*Rain on Glass*) uses it.

## Problem

The author asked whether two axes form a musical expression:

1. Texture as a distribution of sound, not bound by consonance.
2. Short resonant tones combined with that texture.

Rain is the inspiration, not the target. Reproducing rain is not the goal.

The offline prototypes in `.orchestration/units/TEX-1/` gave three findings that hold beyond rain:

- **Exact repetition sounds artificial.** In A, 28 Voices each repeated one fixed strike. The author heard the same repetition. A2 drew every strike fresh and sounded more natural. Auditory listeners learn exactly repeated sounds quickly, and they hear textures through their time-averaged statistics. A texture source therefore keeps its statistics fixed and draws every event anew. A Voice keeps a fixed body, so it cannot do this.
- **Common onset binds a tone to the texture.** In C3, objects were struck by the drops that landed on them, at the same instant and with level following the drop. The author heard it as one world. In C2, objects were struck independently and the author heard a separate layer. Both are expressions. Grouping by common onset is the auditory scene-analysis principle that separates them.
- **One material reads as drops; scattered modes read as crumpling.** In C4, strikes were band-limited noise bursts, and the author found them unnatural. The first Rust impacts (M0/M1) placed six modes at random within half an octave, decaying in 6 ms, and the author heard them as crumpling ("くしゃくしゃ"), not rain. A2 and C3, heard as natural and rain-like, struck one inharmonic pattern (wine glass) every time, with 60-cent mode offsets, decays of 25-50 ms and a soft onset. Listeners judge material from a body's mode structure and damping. When every event has a new random structure, each event is a different object, and a run of them sounds like something being crumpled. When every event shares one structure, the events group into one material struck many times. Each event then has a momentary pitch, but the pitch changes every time, lasts tens of milliseconds and is never sustained, so the texture carries no melody or harmony.

## Concept

Rain is a third composition mode, beside Populations and Flora. Each Rain has:

- a stream of impacts over a frequency extent, every impact drawn from distributions;
- optional resonant objects that belong to the Rain and ring when an impact lands on them.

The coupling contract matches Flora's. Rain, Flora and Populations affect each other only through radiated sound and the Landscape. Voices are never struck by Rain. The separate-layer expression (C2) is a Rain placed beside a Population.

## Impacts

- **Timing.** A Poisson process at the composer's `rate` (impacts per second). There is no beat. The process runs by time rescaling: a unit-exponential budget is drawn after each impact and the current rate is subtracted per sample, so it stays Poisson while the rate changes, and live `rate()` patches take effect at once.
- **Sway.** `sway(octaves)` lets the rate fluctuate as `rate · 2^{sway · g(t)}`, where `g` has unit variance and a 1/f spectrum (the author's proposal after S1). `g` sums nine equal-variance AR(1) processes with time constants 0.4 s · 2^k, k = 0..8 (0.4 s to 102 s), which gives 1/f between about 0.0016 and 0.4 Hz. The fastest stays below the pulse range (about 0.5 Hz and up), so the sway is heard as changing intensity rather than a beat. States start from their stationary distribution, so the sway is present from placement. The swayed rate is capped at the work bound of 2000 per second. Default 0, range [0, 4]; beyond four octaves the rate spends most of its time at zero or at the bound. Struck objects follow the swayed rate, since strikes are a fraction `hit` of impacts.
- **Band position.** At each impact the centre frequency is drawn from the placement's distribution over the extent. The distribution is rebuilt each hop from the Landscape with the spawn-placement density masses (`field_density_mass`):
  - `random(lo, hi)`: log-uniform, independent of the Landscape;
  - `consonance(lo, hi)`: follows the eroded consonance density mass;
  - `dissonance`, `edge` and `gap`: follow the same masses as Voice placement.
  - When the in-range mass is zero, the distribution is log-uniform.
- **Material.** `impact_modes(pattern)` sets the struck body with the ModePattern vocabulary; the default is `modal_table("wine_glass")`. Its ratios are taken once at placement, at the extent's geometric centre, and at most eight are used.
- **Strike.** Mode `k` sits at `centre · ratio_k · 2^{u_k}`, where `u_k` is drawn uniformly from ±`width` for every impact. Mode `k` has level `0.85^k` with a 3 dB spread and a random phase. The first mode decays with time constant `τ = 25 ms · (3.2 kHz / f_c)^{1/2}` with a log-normal spread of 0.4, and mode `k` with `τ / ratio_k^{1/2}`. Each impact starts with a raised-cosine onset of 0.5 to 1.5 ms; a hard start adds a broadband click. Level is log-normal with a 5 dB spread, capped at +9 dB. These constants reproduce the A2 prototype (25 ms over 1.5-7 kHz and 50 ms over 0.25-1.5 kHz); they are authored, not parameters of auditory cognition.
- **Width.** `width(octaves)` is a composer control in [0, 5], live-patchable. The default 0.05 (60 cents, as in A2) keeps every impact recognisably the same material. Wider ranges scatter the modes until the material dissolves into crackles; 0 repeats the pattern exactly. The upper bound covers the audible band from any centre.
- **Pool.** 64 preallocated impact slots. A new impact takes a free slot, or the one closest to its end.

## Objects

- **Count.** `objects(n)`, from 0 to 16; the default is 3.
- **Pitch.** At placement each object takes a peak of the eroded consonance field level inside the extent, with peaks at least 1 ERB apart. Pitched sound follows consonance even when the impacts are free. Missing peaks are filled by log-uniform positions in the extent.
- **Drift.** Each hop an object proposes the best eroded consonance score within one semitone of its sounding pitch, inside the extent and at least 1 ERB from the other objects. The score subtracts the Voice movement cost (1 per octave: coefficient 0.5 over the 2 s integration window), and a move must gain more than the Voice threshold 0.1. The pitch objective sign applies as it does to Voices. The object moves to its proposal only when it is next struck, so the new pitch starts with a strike and the rate of movement follows `hit`. `anchor()` holds every object at its placement pitch; the author found the drifting version more interesting (M0/M1), so drift is the default. Mode ratios stay fixed from placement; modes that reach 0.45 fs are muted, not dropped, so a retuned bank keeps its state.
- **Body.** `object_modes(pattern)` uses the ModePattern vocabulary; the default is `modal_table("vibraphone_1")`. Mode `k` has gain `0.5^k` and T60 `object_decay · ratio^{-0.7}`. The default `object_decay` is 2 s.
- **Strikes.** Each impact lands on an object with probability `hit` (default 0.06, about 2.4 strikes per second at the default rate, matching the C3 audition). The object is chosen uniformly. The impact also sounds, so the common onset is preserved. The drive is scaled per object so the ring's onset peak matches the impact's level. Common onset is what binds the two, so their onsets are made equally loud. Longer `object_decay` still adds presence.

## Rendering and runtime

- Rain renders after Voices and Flora, into the buses its `send()` selects. This happens before the self-sound history reads the habitat bus, so Voices hear Rain as external sound.
- Advance, impact triggering and rendering allocate nothing in steady state.
- Placement and a Log2Space change may allocate.
- RNG: `SmallRng` seeded from the scenario seed and the Rain id.
- Release ramps the gain to zero over `fade`. A finished Rain is removed. The run waits for Rain tails, as it does for Flora.

## Why impacts do not form consonance

An impact's pitch lasts tens of milliseconds and changes with every impact, so the harmonicity impacts leave in the Landscape is brief and scattered; they mostly add roughness. The Landscape's memory is short (density smoothing of 10 ms, hop windows of tens of milliseconds), and habituation is negative feedback that erodes sustained peaks. A run of impacts therefore does not build a consonance structure by itself (TEX-1 N0/N1: no change in octave shares or spectral flatness). Consonance comes from pitched sound with slow inertia, which is the objects' role. Impacts use Landscape distributions based on level or roughness, such as `gap`, and `width` moves them toward pitch when the composer wants it.

## Measurements (TEX-1 M0/M1, before the impact material)

Four objects, no drone, impacts on `gap(150, 5000)`, 60 s, seeds 21-24. Over all object pairs, the mean distance to the nearest of unison, 6/5, 5/4, 4/3, 3/2, 5/3 and the octave fell from 66 cents at placement to 42 cents over the last 30 s. Uniformly random intervals give 60 cents. Two seeds reached near-just chords (seed 21 ended at 180, 359, 719 and 960 Hz, every pair within 2 cents); two moved little. These numbers describe the mechanism; they do not decide whether it is heard as music. Listening to seed 21, the author found the drifting objects (M1) more interesting than fixed ones (M0) and heard the impacts as crumpling, which led to the impact material above. On the same scene and seed, the author heard the material impacts (P1) as natural and the scattered-mode impacts (M1) as artificial rather than rain.

## Sample 18 and the S1 trace

Sample 18 places a Rain with four objects on `gap(150, 5000)` with `sway(1)`, adds an unheard field (a habitat-only harmonic drone on D), turns the field up a fourth, then releases it and thins the rain. An object-pitch trace of the same scene without sway (S1, seed 21) showed the objects approaching the new harmony within a few seconds of the turn, one landing on G4 within 2 cents, and then loosening as habituation eroded the field. Before the turn they barely approached the field. The author wanted to take S1 further.

With `sway(1)` (S2, the sample as written), the author reported a very strong cognitive effect: pleasure, and the body sways by itself, although the rain has no beat. S3 added three quiet sustained voices seeking the same field, to make the unheard field audible; the author found them unnatural. They share neither the impacts' onsets nor the rain's sway, so they are heard as a separate overlay (the C2 finding) rather than part of the rain world. The sample keeps the S2 form.

## Population sway (experimental)

After the 1/f rain, the author asked what the finding implies for Conchordal. Two alternatives were tried and rejected first:
- Multi-timescale landscape habituation as a bottom-up source. It is on `research/multiscale-habituation-20261027`, and it left the onset fluctuation index unchanged.
- An offline loudness sway. The author rejected it, because changing only the output level changes nothing the ensemble does.

Rain's sway changes the event process itself, so the Voice analog changes participation density.

- `PopulationSpec.sway(octaves)`, Initial, range [0, 4], default 0.
- Each Population that sways owns one process `g` from the same 1/f source as Rain (`src/life/sway.rs`), seeded from the scene seed and the Population id.
- At every otherwise eligible onset opportunity, in every timing mode, a Voice sounds with probability `min(1, 2^(depth · g))` on a private draw and otherwise lets the opportunity pass. This is the same gate a refused onset uses, so participation clocks resolve it as a skip.
- The composer's density is the ceiling; the sway thins it. While `g` is positive nothing changes, so a Population whose slow components stay positive for a whole short piece does not thin. Sample 16's far crowd did this with seed 21.
- Pitch, loudness and sustained notes are not changed. With depth 0 no draw is taken, so behavior is bit-identical to before.

## Caveats and exclusions in v1

- Rain reads the mixed Landscape, including its own sound. A Landscape-following Rain can reinforce itself. Flora's self-exclusion observer is not mirrored.
- v1 has no reports, GUI or beat participation.
- Objects hear their own sound and each other's. Habituation erodes a peak an object sustains, so objects keep wandering instead of settling. Two objects that reach an octave or fifth can move together as a locked pair.
- Rain is Experimental. The technote is not updated until the author accepts the expression.
