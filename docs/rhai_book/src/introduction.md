# Introduction

Conchordal is a research-composer scripting surface. The central idea is not
note scheduling. The central idea is shaping a perceptual consonance field,
then letting populations of voices move, survive, and reorganize inside it.

A scenario script defines **Materials** (voice templates), places them into the
field as **Participants**, and shapes the terrain they live on. Harmony emerges
from psychoacoustics (roughness and harmonicity); rhythm emerges from coupled
oscillators on a shared meter the population itself drives. The script is a
director, not a sequencer.

Conchordal v0.4.0 is aimed at research composers who want to work with these
concepts directly. It is not trying to hide the model behind common
music-production vocabulary.

## How this book is organized

- **Tutorial** gets a first sound out and your editor wired up:
  [Quick Start](tutorial/quick_start.md),
  [Editor Setup](tutorial/editor_setup.md).
- **Concepts** explains the three pillars of the model:
  [the Consonance Field](concepts/consonance.md),
  [Rhythm](concepts/rhythm.md), and
  [Routing and the Listener Twin](concepts/routing.md).
- **Reference** is the complete, generated
  [API Reference](reference/api.md) — it is produced from the engine's
  registered scripting surface, so it cannot drift — and the
  [Curated Samples](reference/samples.md) listening path.

Every `rhai` code block in this book is executed against the real script
engine by the test suite, so the examples are guaranteed to run.
