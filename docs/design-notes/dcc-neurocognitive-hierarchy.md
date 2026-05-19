# DCC Neurocognitive Hierarchy

Status: design memo
Scope: future conchordal architecture and AI-facing design context
Date: 2026-05-18

This memo records a working design position for DCC in conchordal. It is meant
as a reference for future agents and design discussions, not as a completed
implementation spec.

## Core Position

DCC should ground musical structures in auditory neurocognitive structures, but
it should not reduce every structure to a slower neural rhythm.

The shared principle is survival in an auditory-cognitive environment. The
mathematical structure should change by time scale:

- Short rhythm: phase survival.
- Melody: predictive pitch-contour survival.
- Phrase: boundary and closure survival.
- Section and form: memory-attention event survival.

In other words, DCC means each musical time scale is connected to an appropriate
auditory representation, prediction, memory, and segmentation mechanism. It
does not mean every layer is a theta/delta oscillator.

## Recommended Layers

1. Acoustic / auditory features.
2. Rhythm entrainment layer.
3. Pitch-contour prediction layer.
4. Phrase boundary / closure layer.
5. Section event-state layer.
6. Form-level memory-attention graph.

## Layer Model

| Layer | Musical role | Neurocognitive basis | Mathematical structure |
|---|---|---|---|
| Acoustic / auditory features | Sound evidence | Cochlear/log-frequency analysis, roughness, harmonicity, pitch salience | Log-frequency scans, salience fields, posterior-like scores |
| Rhythm entrainment | Beat, meter, attack timing, groove | Timing prediction, auditory-motor coupling, low-frequency phase alignment | Phase oscillators, Kuramoto-style coupling, `S^1`, torus `T^n`, PLV, beat posterior |
| Pitch-contour prediction | Melody and local variation | Pitch, relative pitch, contour, interval expectation, melodic surprise | Stochastic trajectory over pitch/context state, vector field on pitch space, predictive distribution |
| Phrase boundary / closure | Phrase start, continuation, cadence-like closure, reset | Auditory event segmentation, boundary detection, expectation reset, closure | Progress variable `rho in [0, 1]`, boundary hazard, closure potential, semi-Markov duration |
| Section event state | A/B/A', development, contrast, recurrence | Long-timescale event model, novelty, habituation, attention shift | Hybrid dynamical system, latent section state, semi-Markov model, attractor graph |
| Form memory-attention graph | Long form, recall, transformation, large-scale contrast | Memory cueing, attention, recurrence, similarity/novelty tracking | Graph of motifs/phrases/sections with similarity, contrast, transformation, and recurrence edges |

## Evidence Strength

Strong:

- Short rhythm, beat, and meter are well served by oscillator, phase-locking,
  PLV, beat posterior, and auditory-motor coupling models.
- Pitch and melody have strong auditory cortex support: pitch encoding, missing
  fundamental, relative pitch, contour, pitch change, and melodic expectation.
- Phrase boundaries are observed as event-boundary-like responses distinct from
  simple note onsets.

Important caution:

- Beat-frequency neural responses do not prove intrinsic entrainment by
  themselves. Repeated evoked responses can mimic beat-frequency activity. Use
  oscillator models pragmatically, and compare against evoked-response baselines
  where that distinction matters.

Moderate:

- Melodic expectation can be modeled as probabilistic prediction, for example
  `surprise_t = -log P(note_t | context_t)`.
- Melody memory, repetition, and variation recognition should be modeled as
  motif memory, contour memory, recurrence detection, and variation distance.
  Do not reduce them simplistically to hippocampus.
- Tension/release can be treated as interaction among prediction, uncertainty,
  surprise, and reward.

Theoretical extension:

- Section and long form can cautiously borrow from event segmentation, temporal
  receptive windows, hippocampal-cortical memory integration, narrative
  understanding, attention, novelty, and habituation.
- This is plausible for DCC but less directly proven in music than the rhythm,
  pitch, and phrase evidence.

Weak or speculative:

- Directly mapping sections, chapters, or long musical form to infra-slow neural
  rhythm is weak.
- Western tonal assumptions are a risk. Cadence, tension/release, interval
  expectation, and closure should be idiom-specific learned statistics rather
  than fixed tonal grammar.

## Design Implications

### Rhythm

Use phase oscillators for attack timing, beat locking, meter, groove, and local
pacing:

```text
dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j - theta_i) + input_i(t)
```

Outputs can include:

- beat phase
- meter phase
- PLV
- tempo posterior
- timing prediction error

This layer should remain local and short-timescale. It should not be stretched
to explain long-form musical structure by itself.

### Melody

Treat melody as a predictive trajectory, not as a low-frequency oscillator and
not as a fixed pitch sequence:

```text
melody_state_t = F(
    pitch_t,
    interval_t,
    contour_t,
    tonal_or_idiom_context_t,
    melodic_memory_t
)
```

Outputs can include:

- next-pitch distribution
- expected interval
- contour continuation probability
- melodic surprise
- variation distance

Generation controls should be phrased in cognitive terms:

- expectedness
- contour stability
- interval novelty
- motif similarity
- register drift

### Phrase

Treat phrase as boundary, closure, and reset:

```text
closure_t = f(
    melodic_stability,
    rhythmic_gap,
    harmonic_resolution,
    entropy_reduction,
    motif_completion
)

boundary_hazard_t = sigmoid(g(closure_t, pause_t, surprise_t, duration_t))
```

A boundary should do more than mark a time point. It should compress or reset
context, consolidate phrase memory, and generate the next phrase goal.

### Section

Treat section as a listening-side event state, not as a fixed external timeline:

```text
z_k = section_state
d_k = dwell_time
n_k = novelty
a_k = attention_gain
r_k = recurrence_target

P(z_{k+1} | z_k, boundary_k, novelty_k, recurrence_k, attention_k)
```

The section state should be influenced by phrase boundaries, novelty,
habituation, recurrence cues, and attention shifts.

### Long Form

Represent long form as a memory-attention graph:

- Nodes: motifs, phrases, sections.
- Edges: similarity, contrast, transformation, recurrence.
- State: current attention focus, memory cue strength, retrieval strength.

Useful form-level operations:

- introduce motif
- repeat motif
- vary motif
- contrast section
- recall previous section
- delay resolution
- increase novelty
- restore familiarity

## Practical Rule

Use the same survival-space principle across layers, but do not force the same
mathematical object onto every layer.

- Rhythm survives by phase fit.
- Melody survives by predictive pitch-contour fit.
- Phrase survives by boundary and closure fit.
- Section and form survive by event-memory and attention fit.

## Reference Anchors

- Rhythm / meter: Nozaradan et al. 2011, J Neurosci, DOI
  `10.1523/JNEUROSCI.0411-11.2011`; Doelling & Poeppel 2015, PNAS, DOI
  `10.1073/pnas.1508431112`; entrainment caution: Damsma et al. 2025.
- Pitch: Bendor & Wang 2005, Nature, DOI `10.1038/nature03867`; Patterson et
  al. 2002, Neuron, DOI `10.1016/S0896-6273(02)01060-7`.
- Relative pitch / contour: Trainor et al. 2002, J Cogn Neurosci, DOI
  `10.1162/089892902317361949`; Sankaran et al. 2024, Science Advances, DOI
  `10.1126/sciadv.adk0010`.
- Melody memory: Jacobsen et al. 2015, Brain, DOI `10.1093/brain/awv135`;
  Esfahani-Bayerl et al. 2019, Cortex, DOI `10.1016/j.cortex.2018.12.023`.
- Phrase boundary: Feng et al. 2022, Scientific Reports, DOI
  `10.1038/s41598-022-13710-3`; Teng et al. 2024, J Neurosci, DOI
  `10.1523/JNEUROSCI.1331-23.2024`.
- Tension/release: Lehne et al. 2014, SCAN, DOI `10.1093/scan/nst141`; Cheung
  et al. 2019, Current Biology.
- Event segmentation and long timescales: Zacks & Swallow 2007, DOI
  `10.1111/j.1467-8721.2007.00480.x`; Lerner et al. 2011, J Neurosci, DOI
  `10.1523/JNEUROSCI.3684-10.2011`; Baldassano et al. 2017, Neuron, DOI
  `10.1016/j.neuron.2017.06.041`; Barnett et al. 2024, Neuron, DOI
  `10.1016/j.neuron.2023.10.010`.
