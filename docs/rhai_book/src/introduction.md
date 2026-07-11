# Introduction

Conchordal is a research-composer scripting surface. The central idea is not
note scheduling. The central idea is shaping a perceptual consonance field,
then letting populations of voices move, survive, and reorganize inside it.

A scenario script defines **PopulationSpecs**, places each one into the field
as a stable **Population**, and shapes the terrain its **Voices** inhabit.
Together those Populations form the runtime **Community**. Harmony emerges from
psychoacoustics (roughness and harmonicity); rhythm emerges from coupled
oscillators on a shared meter the Community itself drives. The script is a
director, not a sequencer.

Conchordal v0.4.0 is aimed at research composers who want to work with these
concepts directly. It is not trying to hide the model behind common
music-production vocabulary.

## How this book is organized

- **Tutorial** gets a first sound out, wires up your editor, and shows how to
  inspect a performance:
  [Quick Start](tutorial/quick_start.md),
  [Editor Setup](tutorial/editor_setup.md), and
  [Observing a Performance](tutorial/observing.md).
- **Concepts** builds the model from a single living voice to the complete
  ecological and temporal structure:
  [the Life of a Voice](concepts/voice_life.md),
  [the Ecological Loop](concepts/ecological_loop.md),
  [the Consonance Field](concepts/consonance.md),
  [Rhythm](concepts/rhythm.md), and
  [Routing and the Listener Twin](concepts/routing.md), followed by
  [Timeline and Structure](concepts/timeline.md).
- **Reference** is the complete, generated
  [API Reference](reference/api.md) — it is produced from the engine's
  registered scripting surface, so it cannot drift — and the
  [Curated Samples](reference/samples.md) listening path. The reference is
  split into three tiers: the **Core API** (enough for every curated sample),
  **Mechanism Tuning**, and **Research Controls**. Start with Core and ignore
  the rest until a piece demands it.

Every `rhai` code block in this book is executed against the real script
engine by the test suite, so the examples are guaranteed to run.
