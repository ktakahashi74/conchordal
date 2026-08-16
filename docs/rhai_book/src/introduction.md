# Introduction

[日本語版](/ja/docs/rhai/)

Conchordal is a bio-acoustic instrument for generative composition. Rhai is
its scenario language: it directs the instrument without reducing it to note
scheduling. The central idea is shaping a perceptual consonance field, then
letting populations of voices move, survive, and reorganize inside it.

A scenario script defines **PopulationSpecs**, places each one into the field
as a stable **Population**, and shapes the terrain its **Voices** inhabit.
Together those Populations form the runtime **Community**. Harmony emerges from
psychoacoustics (roughness and harmonicity); rhythm emerges from coupled
oscillators on a shared meter the Community itself drives. The script is a
director, not a sequencer.

Conchordal v0.4.0 is an alpha release for researchers and developers who want
to examine these concepts directly. Features are incomplete and may be
unstable; composers and creators should wait for the beta release. The model
is intentionally described in its own terms rather than hidden behind common
music-production vocabulary.

## How this book is organized

- **Tutorial** gets a first sound out, wires up your editor, and covers running,
  recording analysis, and replaying a performance:
  [Quick Start](tutorial/quick_start.md),
  [Editor Setup](tutorial/editor_setup.md), and
  [Performance](tutorial/observing.md).
- **Concepts** builds the model from a single living voice to the complete
  ecological and temporal structure:
  [Population — A Persistent Unit of Voices](concepts/voice_life.md),
  [Voice and Landscape — Sound–Environment Feedback](concepts/ecological_loop.md),
  [Consonance Field — A Terrain for Evaluating Pitch](concepts/consonance.md),
  [Rhythm](concepts/rhythm.md), and
  [Routing and the Listener Twin](concepts/routing.md), followed by
  [Timeline and Structure](concepts/timeline.md).
- **Reference** contains the generated
  [API Reference](reference/api.md) and the
  [Curated Samples](reference/samples.md) listening path. The reference is
  split into four tiers: the **Core API** (enough for every curated sample),
  **Experimental** (candidate Core verbs under audition), **Mechanism
  Tuning**, and **Research Controls**. Start with Core and ignore the rest
  until a piece demands it. Registered signatures and tier membership are
  generated and checked against the engine; explanatory prose is maintained
  in the documentation registry.

Every `rhai` code block in this book is executed against the real script
engine by the test suite. This checks that the examples compile and run with
the current engine; it does not by itself prove that every explanation is
semantically correct.
