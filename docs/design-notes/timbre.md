# Timbre: Phenotype, Excitation × Resonance, State Leakage

Status: Design (v0.5 direction). Consequences 1–2 are implemented (energy-driven
`damping` in the scheduled harmonic path; body-ratio LOO in `ApproxHarmonics`);
3–5 remain open.
Scope: give timbre the same kind of *necessity* that pitch and rhythm already
have — grounded in the ecology, in the physics of sounding bodies, and in the
life state of the voice — without inventing mechanisms conchordal does not
already have.

## Problem

The Manifesto's closing question — "why must it be this sound?" — has an
answer for pitch (the consonance terrain) and for rhythm (the emergent meter),
but not for timbre. A voice's timbre is chosen by the composer from a preset
vocabulary (`modes` × `brightness` × `spread` × `unison`): exactly the
symbol-selection paradigm the Manifesto rejects.

The current state (v0.4/0.5):

- Three backends with fractured excitation semantics: `Sine` has no drive
  concept at all (`sine_impulse_boost` fakes an attack by lifting output
  gain), `Harmonic` takes deterministic drive, `Modal` takes noise-modulated
  drive (`DriveMode` in `life/sound/any_backend.rs`).
- ADSR is an external gain curve folded into `ArticulationSignal.amplitude`,
  not a property of excitation and body.
- Timbre is static: fixed at spawn from `BodySnapshot`, patchable only from
  the scenario. The one designed state→timbre path — `TimbreGenotype.damping`,
  which should thin upper partials as energy decays — is inert in the
  authoritative scheduled-tone path: `build_harmonic_lanes` builds lanes once
  with `energy = 1.0` (`life/sound/oscillator_bank.rs`), and render-time drive
  applies only a scalar gain. (The legacy `HarmonicBody` render path does pass
  `signal.amplitude` into `harmonic_gain`, but live tones are built from
  `BodySnapshot` and never take it.)
- Perception does not use the body either: leave-one-out analysis subtracts an
  assumed harmonic series (`ApproxHarmonics`) or only the fundamental bin
  (`ExactScan`) — a voice judges the terrain with a generic template, not with
  its own partial set (`life/pitch_core.rs`).
- The ledger already points here: "heredity of timbre" sits at Horizon
  (technote §9.1) — hereditary respawn exists as assays but inherits only
  pitch, and `ParentCandidate` stores no timbre genotype. §9.2 assigns the
  micro layer (jitter, breath, beating) to the **body**, but today the body's
  micro layer moves only amplitude and pitch (vibrato/jitter), never the
  spectrum.

What is already right, and must be preserved: the ecology *hears*
habitat-routed timbre. The habitat bus carries the actual rendered voice sum
through the NSGT into R/H/C, so each habitat-routed voice's partials shape the
terrain everyone else reads. And `landscape_density_modes` /
`landscape_peaks_modes` already let a body's partials be *chosen from* the
terrain at spawn.

## Principle

Timbre is the voice's **body**. Necessity for a body comes from three sources:

1. **Ecology — timbre is a phenotype.** The fixed-point requirement (technote
   §9.3) read from the timbre side: the terrain's shape is a function of what
   the population radiates. One partial set should play three ecological roles
   at once — it *radiates* (deforms the terrain everyone reads: the production
   side of the loop), it *perceives* (where consonance is felt should depend
   on the voice's own partials — today's generic-template LOO is a gap, not a
   fact), and it *identifies* (the stable trait by which a listener tracks an
   individual while its pitch moves). A phenotype is the substrate of
   selection — which makes heredity and variation the natural next step, not
   a feature bolted on.
2. **Physics — phonation is excitation × resonance.** Sounding bodies —
   voices, strings, bars, most biological sound production (larynx × vocal
   tract, syrinx) — are well modeled as a source filtered by a resonant body.
   Conchordal already has both halves: energy and breath (metabolism,
   ArticulationCore) are the excitation; the SoundBody is the resonance.
   Envelope should be a *consequence* of excitation shape plus body decay — a
   bell does not "release"; its excitation stops and its body rings down.
3. **Expression — timbre leaks life state.** Vocal effort flattens spectral
   tilt (the Lombard effect); arousal raises brightness and jitter; depleted
   energy thins the upper partials. These are physiologically motivated
   mappings, not arbitrary knobs, and they are what §9.2 means by the body
   owning the micro layer.

A fourth, structural principle bounds the space:

4. **Two timbre domains.** The terrain (NSGT → R/H) represents the harmonic
   lattice and spectral envelope — position, slope, density, spread of
   partials. Transients, phase, and noise texture have no separate
   representation; they surface only through spectrum and roughness. So
   *ecological timbre* (`modes`, `brightness`, `spread`, `unison`) is the
   ecology's language — the target of selection, heredity, and state
   coupling — while *presentation timbre* (attack transients, noise, space)
   is the audience's language, mirroring the habitat/presentation dual bus.
   Investment goes to the former until the ListenerTwin closes a perceptual
   loop that can hear the latter.

## Consequences (priority order)

1. **Re-activate state→timbre coupling where it was already designed.** Drive
   `harmonic_gain`'s `energy` argument at render time so `damping` does what
   it was designed to do. The energy signal must be the *excitation* state
   (drive envelope / metabolic energy), not `ArticulationSignal.amplitude` —
   the latter folds in ADSR, gate, and output gain, and would turn release
   into a spectral mute instead of a ring-down. No new mechanism — the same
   reuse discipline as the tension design.
2. **Perceive with your own body.** Let `ApproxHarmonics` leave-one-out
   subtract the voice's actual mode ratios instead of a generic harmonic
   series. This aligns the perception side of the phenotype with the
   production side; the data pieces (ratio storage, subtraction machinery)
   exist, but `PitchCore` does not yet receive the body's ratios — the work is
   the wiring.
3. **Unify excitation semantics.** One source–filter story across bodies:
   impulse = strike, continuous drive = breath/bow, with the same meaning for
   every body. Sine becomes the single-mode limit of a resonant body,
   dissolving the boost hack — but sustain semantics must be specified first:
   oscillator tones currently persist indefinitely after one impulse, while a
   modal mode rings down unless driven, so the unification defines sustained
   tones as *continuously breathed* (excitation, not infinite T60). ADSR moves
   conceptually to the excitation side. `OscillatorBank` may survive as an
   efficiency implementation, but `Modal`'s excitation × resonance is the
   conceptual mainline.
4. **Heredity of timbre.** Extend hereditary respawn so offspring inherit the
   parent's timbre genotype (`ratios`, `brightness`) with mutation — the
   `jitter_cents` machinery is reusable as a mutation operator for ratios
   (brightness needs its own perturbation), but a genotype capture path is
   missing today (`ParentCandidate` stores only
   id/frequency/energy/generation). Timbre moves from "a preset the composer
   picks" to "a solution the ecosystem finds" — the Manifesto's
   placement→discovery shift extended to the timbre axis. The ledger row
   stays Horizon until this lands.
5. **Couplings that must earn their way in.** Two attractive extensions are
   deliberately deferred behind explicit justification, per the
   mirror-dualism lesson (both tests: a perceptual mechanism exists, and the
   production loop closes):
   - *Tension→brightness/jitter.* Physiologically motivated, but it is a new
     cross-domain coupling, not free plumbing: DCC pressure today reaches
     only the pitch-search temperature, and render-side tension is
     NeuralRhythms-derived. Making a pitch-tension signal also alter timbre
     needs its own argument.
   - *Dynamic intonation* — partials continuously pulled toward terrain peaks
     (the continuous version of `landscape_*_modes`).

## Non-goals

- No new composer-facing timbre parameters. The vocabulary (`modes`,
  `brightness`, `spread`, `unison`, `motion`) already spans the ecologically
  visible space.
- No presentation-timbre machinery (transient designers, noise layers,
  spatialization) until a perceptual loop can hear it.
- No per-voice timbre micromanagement from scenarios; state coupling and
  heredity are the ecology's job.
