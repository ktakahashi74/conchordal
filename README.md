# Conchordal

**A bio-acoustic instrument for generative composition.**

> **Note:** This software is in a **Research Alpha Phase**. It is an open laboratory for researchers and artists exploring the direct link between computational audio and human auditory cognition.

## Concept: Emergence over Composition

Conchordal is a computational ecosystem where sound is treated as a living organism.

It does not rely on fixed grids—no equal temperament, no metronomic time. Instead, it simulates a **physiological environment** based on human auditory perception. Within this environment, autonomous "Individuals" struggle, coexist, and evolve.

Their survival depends on finding **Spectral Consonance** (minimizing sensory roughness) and establishing **Virtual Pitch** (maximizing harmonic stability). The resulting music is not composed by a human; it *emerges* from the physical interactions of sound itself.

## The Architecture

The system models a dynamic feedback loop across three layers, unifying Pitch (Space) and Rhythm (Time) under biological principles:

### 1. The Landscape (The Cognitive Environment)

The environment represents the structure of human hearing in both frequency and time domains:

* **Spectral Potential (Pitch):** Using a Non-Stationary Gabor Transform (NSGT) on a Log2 axis—simulating the **Cochlear Tonotopic Map**—it calculates potentials corresponding to physiological mechanisms:
    * **Roughness (R):** Simulates **Basilar Membrane Interference** (Critical Bands).
    * **Harmonicity (H):** Simulates **Neural Phase-Locking** (Temporal Periodicity detection).
    * **Consonance (C):** Simulates **Cognitive Integration**. The resultant fitness terrain ($C = H - R_{norm}$).

* **Neural Rhythms (Time):** Instead of a grid, rhythm emerges from entrainment to simulated **Neural Oscillations (Brainwaves)**:

	* **Delta Band (~0.5-4Hz):** Governs the macroscopic **Pulse** and musical **Phrasing**.

    * **Theta Band (~4-8Hz):** Dictates **Articulation** and syllabic grouping.

    * **Alpha Band (~8-13Hz):** Influences **Texture** and timbral fluctuation (e.g., unison detuning).

    * **Beta Band (~13-30Hz):** Controls micro-timing (groove) and ensemble tightness.
	

### 2. The Population (The Collective)

Sound is not a singular event but a mass phenomenon. The **Population** manages the aggregate state of all active agents. It represents the "species" or "society" of sound that inhabits the Landscape, handling the density, diversity, and collective spectral footprint that feeds back into the environment.

### 3. The Individual (The Agent)

The atomic unit of the system. Each **Individual** is an autonomous entity:
* **Proprioception:** It senses the Landscape's spectral potentials and synchronizes its internal clock to the environmental Neural Rhythms.
* **Metabolism:** It consumes energy to sustain articulation.
* **Autonomy:** It makes local decisions—drifting away from dissonance (segregation) or locking onto harmonic peaks (fusion)—without a central conductor.

## The Role of the Artist: Scenarios as Macro-Structure

While the *micro-structure* (harmony, rhythm, articulation) emerges autonomously, the **macro-structure** (the timeline and narrative arc) is crafted by the artist.

Using **Rhai** scripts, the creator acts not as a composer of notes, but as a **Director of Ecosystems**. Through the scenario file, you define:

* **Phases:** The sectional progression of the piece (e.g., "Genesis", "Conflict", "Resolution").
* **Interventions:** Injecting new populations or altering environmental constants (e.g., changing the system's "temperature" or consonance sensitivity).
* **Constraints:** Setting boundaries within which the system evolves.

This allows for the creation of structured "works" where the overall form is intentional, but the momentary details are emergent.

## Technical Stack

Built in **Rust** for lock-free concurrency and real-time safety.

* **DSP Kernel:** High-performance, SIMD-friendly convolution for psychoacoustic evaluation.
* **Log-Frequency Space:** All calculations occur in a perception-aligned logarithmic coordinate system.
* **Scripting:** Scenario definitions using **Rhai**, allowing dynamic experimentation with ecosystem parameters.
* **Visualization:** Real-time psychoacoustic monitoring via `egui`.

## Getting Started

### Prerequisites

* Rust (latest stable)
* ALSA dev headers (Linux only: `libasound2-dev`)

### Installation & Run

Run in release mode to ensure the DSP thread meets real-time deadlines.

```bash
git clone [https://github.com/ktakahashi74/conchordal.git](https://github.com/ktakahashi74/conchordal.git)
cd conchordal
cargo run --release -- samples/harmonic_density.rhai
```

### Experimentation

Define the ecosystem's initial conditions using Rhai scripts (see samples/).

```Rust
// Example: Spawning a population that seeks harmonic density
let life_config = #{
    type: "sustain",
    initial_energy: 1.0,
    metabolism_rate: 0.05,
    envelope: #{ attack_sec: 0.1, decay_sec: 0.2, sustain_level: 0.5 }
};

let method = #{ 
    mode: "harmonic_density", 
    min_freq: 200.0, 
    max_freq: 1200.0, 
    temperature: 0.7 
};

// Spawn 5 individuals into the population
spawn_agents("swarm", method, life_config, 5, 0.2);
```

### Contributing

We invite engineers and artists who are exploring the frontiers of Auditory Scene Analysis and Computational Creativity. Check the Issue Tracker for open research topics.

### License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).

### Author

Created by Koichi Takahashi <info@conchordal.org>
