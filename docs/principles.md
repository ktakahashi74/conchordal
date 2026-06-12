# Principles

Version-independent knowledge about Conchordal: findings from perception
science and from the system's own behavior that any implementation must
respect. This document sits between the Manifesto (the declaration of intent)
and the technote (the as-built description): the Manifesto says *why*, the
technote says *how it currently works*, and this file says *what must be true
regardless of version*. Entries accumulate; they are corrected, not rotated.

Format per entry: the knowledge, its consequence for Conchordal, and the
current implementation status (which may lag the principle).

---

## P1. The landscape is a model of perception, not of preference

Consonance and meter are *percepts*, constructed by the auditory system —
not properties of notation, ratio tables, or schedules. Sensory dissonance
(critical-band interference) and tonal fusion (virtual pitch) are separable
mechanisms with quantitative models; pulse and meter are self-sustaining
oscillatory percepts that need a few cycles of evidence, persist through
gaps, and re-lock under tempo drift.

**Consequence**: every musical capability enters Conchordal as a model of the
*listener's mechanism*, never as a rule about the *material*. Interval names,
scales, and time signatures must not appear anywhere in the engine.

**Status**: implemented on both axes — R/H/C kernels (technote §3) and the
emergent meter (technote §4).

## P2. Adaptation is the third perceptual mechanism (largely unimplemented)

Human temporal cognition layers experience in three windows:

1. **The perceptual present (~3 s, upper bound ~8 s)**: within it, no change
   is needed; texture itself carries. (Pöppel's integration window, echoic
   memory.)
2. **The prediction window (3–8 s)**: musical phrases live here. Boredom is
   the operational state of a prediction engine with nothing left to update
   (Huron): beyond ~8 s with no *perceptible* change, attention releases.
3. **The scene window (15–30 s)**: event-segmentation boundaries must arrive
   at this scale or the mind wanders (Zacks; Berlyne's information-rate
   inverted U).

Beneath all three: habituation. Auditory cortex adapts to repeated spectra
(stimulus-specific adaptation); an unchanging percept literally fades from
salience and recovers after withdrawal.

**Consequence**: consonance is a *meal, not a place*. A terrain that models
perception must devalue what has been sounding (habituation as erosion, time
constant ≈ the prediction window) and let it recover after release. Stasis
avoidance then belongs to the ecology, not to authored pacing. The three
windows assign a division of labor:

- micro (< 3 s) = the **body** (jitter, breath, beating);
- meso (3–8 s) = the **ecology** (adaptation-driven movement, life and death);
- macro (15–30 s) = the **scenario** — the one layer the Manifesto explicitly
  assigns to the human director, who owes the listener a boundary at this
  scale and cannot delegate it.

**Status**: fragments only. Per-agent boredom/familiarity (`PerceptualContext`)
is the agent-side preview; the ListenerTwin's tension/attention reporting and
the (default-off) DCC pressure path are the feedback channel that would close
the loop. A landscape-level habituation field is future work; until then the
"no >8 s of imperceptible change" rule is a *diagnostic* for the etude path,
not a property of the system.

## P3. Perceptual symmetries must survive the production loop

The mirror-dualism episode generalizes. A perceptual mechanism can be mirrored
in software (undertone projection), but Conchordal is a *closed loop*: the
landscape is computed from what the population actually radiates, and every
physical body radiates overtones. A configuration is musically real only if it
is a **fixed point of the perception–production loop**: agents attracted to a
position must radiate spectra that reinforce the very structure that attracted
them. Overtone attractors self-reinforce; undertone attractors cannot — which
is why the predicted minor-mode emergence failed and was abandoned, while the
field-level mirror survives as a harmonic-tension dial.

**Consequence**: before predicting any emergent structure from a terrain
operation, check both sides: (a) is there a perceptual mechanism for it, and
(b) does the loop close on the production side? A terrain shape with no
self-reinforcing radiation is a tension device, not an attractor.

**Status**: analysis recorded in technote §3.3.3; applies to any future
terrain operation.

## P4. Scaffolding is inaudible or embodied

Infrastructure leaks destroy musicality. A terrain-anchor drone held by fiat
reads as a hand pinning the world still; a fixed-amplitude, click-enveloped
"beat voice" reads as a metronome regardless of how emergent the meter behind
it is. The Manifesto's inhabitants *are* the music; the environment must be
felt through them.

**Consequence**: in the listening path, terrain anchors route to the habitat
bus (heard by the ecosystem, not the audience) unless the drone is itself the
dramatic subject; no voice may have "being the beat" as its only musical
content — pulse carriers get resonant bodies and lives, or the pulse simply
condenses from the colony. Feature-explanation samples (`samples/research/`)
are exempt: there the scaffolding *is* the subject.

**Status**: applied to the etude path (2026-06); enforced editorially, not
mechanically.
