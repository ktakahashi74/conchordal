+++
title = "Technical Note: The Physics of Conchordal"
description = "A deep dive into the psychoacoustic algorithms, logarithmic signal processing, and artificial life strategies powering the Conchordal ecosystem."
template = "page.html"
[extra]
source_commit = "272f31a"
author = "Koichi Takahashi"
last_updated = "2026-01-19"
source_version = "0.2.0"
source_snapshot = "2026-01-19T00:00:00+09:00"
generated_by = "claude-opus-4-5-20251101"
+++

# 1. Introduction: The Bio-Acoustic Paradigm

Conchordal represents a fundamental divergence from established norms in generative music and computational audio. Where traditional systems rely on symbolic manipulation—operating on grids of quantized pitch (MIDI, Equal Temperament) and discretized time (BPM, measures)—Conchordal functions as a continuous, biologically grounded simulation of auditory perception. It posits that musical structure is not an artifact of abstract composition but an emergent property of acoustic survival.

This technical note serves as an exhaustive reference for the system's architecture, signal processing algorithms, and artificial life strategies. It details how Conchordal synthesizes the principles of psychoacoustics—specifically critical band theory, virtual pitch perception, and neural entrainment—with the dynamics of an autonomous ecosystem. In this environment, sound is treated as a living organism, an "Individual" possessing metabolism, sensory processing capabilities, and the autonomy to navigate a hostile spectral terrain.

The emergent behavior of the system is driven by a unified fitness function: the pursuit of Consonance. Agents within the Conchordal ecosystem do not follow a pre-written score. Instead, they continuously analyze their environment to maximize their "Spectral Comfort"—defined as the minimization of sensory roughness—and their "Harmonic Stability," or the maximization of virtual root strength. The result is a self-organizing soundscape where harmony, rhythm, and timbre evolve organically through the interactions of physical laws rather than deterministic sequencing.

This document explores the three foundational pillars of the Conchordal architecture:

1.  **The Psychoacoustic Coordinate System**: The mathematical framework of `Log2Space` and ERB scales that replaces linear Hertz and integer MIDI notes.
2.  **The Cognitive Landscape**: The real-time DSP pipeline that computes Roughness ($R$) and Harmonicity ($H$) fields from the raw audio stream.
3.  **The Life Engine**: The agent-based model governing the metabolism, movement, and neural entrainment of the audio entities.

# 2. The Psychoacoustic Coordinate System

A critical innovation in Conchordal is the rejection of the linear frequency scale ($f$) for internal processing. Human auditory perception is inherently logarithmic; our perception of pitch interval is based on frequency ratios rather than differences. To model this accurately and efficiently, Conchordal establishes a custom coordinate system, `Log2Space`, which aligns the computational grid with the tonotopic map of the cochlea.

## 2.1 The Log2 Space Foundation

The `Log2Space` struct serves as the backbone for all spectral analysis, kernel convolution, and agent positioning within the system. It maps the physical frequency domain ($f$ in Hz) to a perceptual logarithmic domain ($l$).

### 2.1.1 Mathematical Definition

The transformation from Hertz to the internal log-coordinate is defined as the base-2 logarithm of the frequency. This choice is deliberate: in base-2, an increment of 1.0 corresponds exactly to an octave, the most fundamental interval in pitch perception.

$$ l(f) = \log_2(f) $$

The inverse transformation, used to derive synthesis parameters for the audio thread, is:

$$ f(l) = 2^l $$

The coordinate space is discretized into a grid defined by a resolution parameter, `bins_per_oct` ($B$). This parameter determines the granularity of the simulation. A typical value of $B=48$ or $B=96$ provides sub-semitone resolution sufficient for continuous pitch gliding and microtonal inflection. The step size $\Delta l$ is constant across the entire spectral range:

$$ \Delta l = \frac{1}{B} $$

### 2.1.2 Grid Construction and Indexing

The `Log2Space` structure pre-calculates the center frequencies for all bins spanning the configured range $[f_{min}, f_{max}]$. The number of bins $N$ is determined to ensure complete coverage:

$$ N = \lfloor \frac{\log_2(f_{max}) - \log_2(f_{min})}{\Delta l} \rfloor + 1 $$

The system maintains two parallel vectors for $O(1)$ access during DSP operations:

*   `centers_log2`: The logarithmic coordinates $l_i = \log_2(f_{min}) + i \cdot \Delta l$.
*   `centers_hz`: The pre-computed linear frequencies $f_i = 2^{l_i}$.

This pre-computation is vital for real-time performance, removing the need for costly `log2` and `pow` calls inside the inner loops of the spectral kernels. The method `index_of_freq(hz)` provides the quantization logic, mapping an arbitrary float frequency to the nearest bin index.

## 2.2 Constant-Q Bandwidth Characteristics

The `Log2Space` inherently enforces a Constant-Q (Constant Quality Factor) characteristic across the spectrum. In signal processing terms, $Q$ is defined as the ratio of the center frequency to the bandwidth: $Q = f / \Delta f$.

In a linear system (like a standard FFT), $\Delta f$ is constant, meaning $Q$ increases with frequency. In `Log2Space`, the bandwidth $\Delta f_i$ of the $i$-th bin scales proportionally with the center frequency $f_i$. This property mimics the frequency selectivity of the human auditory system, where the ear's ability to resolve frequencies diminishes (in absolute Hz terms) as frequency increases. This alignment allows Conchordal to allocate computational resources efficiently—using high temporal resolution at high frequencies and high spectral resolution at low frequencies—without manual multirate processing.

## 2.3 The Equivalent Rectangular Bandwidth (ERB) Scale

While `Log2Space` handles pitch relationships (octaves, harmonics), it does not perfectly model the critical bands of the ear, which are wider at low frequencies than a pure logarithmic mapping suggests. To accurately calculate sensory roughness (dissonance), Conchordal implements the Equivalent Rectangular Bandwidth (ERB) scale based on the Glasberg & Moore (1990) model.

The `core/erb.rs` module provides the transformation functions used by the Roughness Kernel. The conversion from frequency $f$ (Hz) to ERB-rate units $E$ is given by:

$$ E(f) = 21.4 \log_{10}(0.00437f + 1) $$

The bandwidth of a critical band at frequency $f$ is:

$$ BW_{ERB}(f) = 24.7(0.00437f + 1) $$

This scale is distinct from `Log2Space`. While `Log2Space` is the domain for pitch and harmonicity (where relationships are octave-invariant), the roughness calculation requires mapping spectral energy into the ERB domain to evaluate interference. The system effectively maintains a dual-view of the spectrum: one strictly logarithmic for harmonic templates, and one psychoacoustic for dissonance evaluation.

# 3. The Auditory Landscape: Analyzing the Environment

The "Landscape" is the central data structure in Conchordal. It acts as the shared environment for all agents, a dynamic scalar field representing the psychoacoustic "potential" of every frequency bin. Agents do not interact directly with each other; they interact with the Landscape, which aggregates the spectral energy of the entire population. This decouples the complexity of the simulation from the number of agents ($O(N)$ vs $O(N^2)$).

The Landscape is updated every audio frame (or block) by the Analysis Workers. It synthesizes two primary metrics:

*   **Roughness ($R$)**: The sensory dissonance caused by rapid beating between proximal partials.
*   **Harmonicity ($H$)**: The measure of virtual pitch strength and spectral periodicity.

Both metrics are normalized to the $[0, 1]$ range before combination. The net Consonance ($C$) is computed as a signed difference, then rescaled:

$$ C_{signed} = \text{clip}(H_{01} - w_r \cdot R_{01},\; -1,\; 1) $$

$$ C_{01} = \frac{C_{signed} + 1}{2} $$

where $w_r$ is the `roughness_weight` parameter (default 1.0). Individual agents maintain their own perceptual context (`PerceptualContext`) which tracks per-agent boredom and familiarity, providing additional score adjustments during pitch selection.

## 3.1 Non-Stationary Gabor Transform (NSGT)

To populate the `Log2Space` with spectral data, Conchordal uses a custom implementation of the Non-Stationary Gabor Transform (NSGT). Unlike the Short-Time Fourier Transform (STFT), which uses a fixed window size, the NSGT varies the window length $L$ inversely with frequency to maintain the Constant-Q property derived in Section 2.2.

### 3.1.1 Kernel-Based Spectral Analysis

The implementation in `core/nsgt_kernel.rs` utilizes a sparse kernel approach to perform this transform efficiently. For each log-frequency band $k$, a time-domain kernel $h_k$ is precomputed. This kernel combines a complex sinusoid at the band's center frequency $f_k$ with a periodic Hann window $w_k$ of length $L_k \approx Q \cdot f_s / f_k$.

$$ h_k[n] = w_k[n] \cdot e^{-j 2\pi f_k n / f_s} $$

These kernels are transformed into the frequency domain ($K_k[\nu]$) during initialization. To optimize performance, the system sparsifies these frequency kernels, storing only the bins with significant energy.

During runtime, the system performs a single FFT on the input audio buffer to obtain the spectrum $X[\nu]$. The complex coefficient $C_k$ for band $k$ is then computed via the inner product in the frequency domain:

$$ C_k = \frac{1}{N_{fft}} \sum_{\nu} X[\nu] \cdot K_k^*[\nu] $$

This "one FFT, many kernels" approach allows Conchordal to generate a high-resolution, logarithmically spaced spectrum covering 20Hz to 20kHz without the computational overhead of calculating separate DFTs for each band or using recursive filter banks.

### 3.1.2 Real-Time Temporal Smoothing

The raw spectral coefficients $C_k$ exhibit high variance due to the stochastic nature of the audio input (especially with noise-based agents). To create a stable field for agents to sample, the `RtNsgtKernelLog2` struct wraps the NSGT with a temporal smoothing layer.

It implements a per-band leaky integrator (exponential smoothing). Crucially, the time constant $\tau$ is frequency-dependent. Low frequencies, which evolve slowly, are smoothed with a longer $\tau$, while high frequencies, which carry transient details, have a shorter $\tau$.

$$ y_k[t] = (1 - \alpha_k) \cdot |C_k[t]| + \alpha_k \cdot y_k[t-1] $$

where the smoothing factor $\alpha_k$ is derived from the frame interval $\Delta t$:

$$ \alpha_k = e^{-\Delta t / \tau(f_k)} $$

This models the "integration time" of the ear, ensuring that the Landscape reflects a psychoacoustic percept rather than instantaneous signal power.

## 3.2 Roughness ($R$) Calculation: The Plomp-Levelt Model

Roughness is the sensation of "harshness" or "buzzing" caused by the interference of spectral components that fall within the same critical band but are not sufficiently close to be perceived as a single tone (beating). Conchordal implements a variation of the Plomp-Levelt model via convolution in the ERB domain.

### 3.2.1 The Interference Kernel

The core of the calculation is the Roughness Kernel, defined in `core/roughness_kernel.rs`. This kernel $K_{rough}(\Delta z)$ models the interference curve between two partials separated by $\Delta z$ ERB. The curve creates a penalty that rises rapidly as partials separate, peaks at approximately 0.25 ERB (maximum roughness), and then decays as they separate further.

The implementation uses a parameterized function `eval_kernel_delta_erb` to generate this shape:

$$ g(\Delta z) = e^{-\frac{\Delta z^2}{2\sigma^2}} \cdot (1 - e^{-(\frac{\Delta z}{\sigma_{suppress}})^p}) $$

The second term is a suppression factor that ensures the kernel goes to zero as $\Delta z \to 0$, preventing a single pure tone from generating self-roughness.

### 3.2.2 Convolutional Approach

Calculating roughness pairwise for all spectral bins ($N^2$ complexity) is computationally prohibitive for real-time applications. Conchordal solves this by treating the Roughness calculation as a linear convolution.

1.  **Mapping**: The log-spaced amplitude spectrum from the NSGT is mapped (or interpolated) onto a linear ERB grid.
2.  **Convolution**: This density $A(z)$ is convolved with the pre-calculated roughness kernel $K_{rough}$.

$$ R_{shape}(z) = (A * K_{rough})(z) = \int A(z-\tau) K_{rough}(\tau) d\tau $$

The result $R_{shape}(z)$ represents the raw "Roughness Shape" at frequency $z$. To convert this to a normalized fitness signal, Conchordal applies a physiological saturation mapping.

### 3.2.3 Physiological Saturation Mapping

Raw roughness values from the convolution have unbounded range. Rather than hard-clamping, Conchordal uses a saturation curve that models the compressive nonlinearity of auditory perception. This mapping converts reference-normalized roughness ratios to the $[0, 1]$ range.

**Reference Normalization**: The system maintains reference values $r_{ref,peak}$ and $r_{ref,total}$ representing "typical" roughness levels. The reference-normalized ratios are:

$$ x_{peak}(u) = \frac{R_{shape}(u)}{r_{ref,peak}} $$

$$ x_{total} = \frac{R_{shape,total}}{r_{ref,total}} $$

**The Saturation Parameter**: The parameter `roughness_k` ($k > 0$) controls the saturation curve's shoulder. The reference ratio $x = 1$ maps to:

$$ R_{ref} = \frac{1}{1+k} $$

Larger $k$ reduces $R_{01}$ for the same input ratio, making the system more tolerant of roughness.

**Piecewise Saturation Mapping**: The normalized roughness $R_{01}$ is computed from the reference-normalized ratio $x$ as:

$$
R_{01}(x; k) = \begin{cases}
0 & \text{if } x \leq 0 \\
x \cdot \frac{1}{1+k} & \text{if } 0 < x < 1 \\
1 - \frac{k}{x+k} & \text{if } x \geq 1
\end{cases}
$$

This function is continuous at $x = 1$ (both branches yield $\frac{1}{1+k}$) and saturates asymptotically to 1 as $x \to \infty$. The piecewise structure ensures linear response for low roughness (preserving sensitivity) while compressing extreme values (preventing saturation).

**Numerical Safety**: The implementation handles edge cases robustly:

*   $x = \text{NaN} \to 0$
*   $x = +\infty \to 1$
*   $x = -\infty \to 0$
*   Non-finite $k$ is treated as $10^{-6}$

Agents seeking consonance actively avoid peaks in the $R_{01}$ field.

## 3.3 Harmonicity ($H$): The Sibling Projection Algorithm

While Roughness drives agents away from dissonance (segregation), Harmonicity ($H$) drives them toward fusion—the creation of coherent chords and timbres. Conchordal introduces a novel algorithm termed "Sibling Projection" to compute this field. This algorithm approximates the brain's mechanism of "Common Root" detection (Virtual Pitch) entirely in the frequency domain.

### 3.3.1 Concept: Virtual Roots

The algorithm posits that any spectral peak at frequency $f$ implies the potential existence of a fundamental frequency (root) at its subharmonics ($f/2, f/3, f/4 \dots$). If multiple spectral peaks share a common subharmonic, that subharmonic represents a strong "Virtual Root".

### 3.3.2 The Two-Pass Projection

The algorithm operates on the `Log2Space` spectrum in two passes, utilizing the integer properties of the logarithmic grid:

1.  **Downward Projection (Root Search)**: The current spectral envelope is "smeared" downward. For every bin $i$ with energy, the algorithm adds energy to bins $i - \log_2(k)$ for integers $k \in \{1, 2, \dots, N\}$.

    $$ Roots[i] = \sum_k A[i + \log_2(k)] \cdot w_k $$

    Here, $w_k$ is a weighting factor that decays with harmonic index $k$ (e.g., $k^{-\rho}$), reflecting that lower harmonics imply their roots more strongly than higher ones. The result `Roots` describes the strength of the virtual pitch at every frequency.

2.  **Upward Projection (Harmonic Resonance)**: The system then projects the `Roots` spectrum back upwards. If a strong root exists at $f_r$, it implies stability for all its natural harmonics ($f_r, 2f_r, 3f_r \dots$).

    $$ H[i] = \sum_m Roots[i - \log_2(m)] \cdot w_m $$

**Emergent Tonal Stability**: Consider an environment with a single tone at 200 Hz.

*   **Step 1 (Down)**: It projects roots at 100 Hz ($f/2$), 66.6 Hz ($f/3$), 50 Hz ($f/4$), etc.
*   **Step 2 (Up)**: The 100 Hz root projects stability to 100, 200, 300, 400, 500... Hz.
    *   300 Hz is the Perfect 5th of the 100 Hz root.
    *   500 Hz is the Major 3rd of the 100 Hz root.

Thus, without any hardcoded knowledge of Western music theory, the system naturally generates stability peaks at the Major 3rd and Perfect 5th relationships, simply as a consequence of the physics of the harmonic series. An agent at 200 Hz creates a "gravity well" at 300 Hz and 500 Hz, inviting other agents to form a major triad.

### 3.3.3 Mirror Dualism: Overtone vs. Undertone

The implementation in `core/harmonicity_kernel.rs` includes a profound parameter: `mirror_weight` ($\alpha$). This parameter blends two distinct projection paths:

*   **Path A (Overtone/Major)**: The standard "Down-then-Up" projection described above. It creates gravity based on the Overtone Series, favoring Major tonalities.
*   **Path B (Undertone/Minor)**: An inverted "Up-then-Down" projection. It finds common overtones and projects undertones. This is the theoretical dual of Path A and favors Minor or Phrygian tonalities (the Undertone Series).

$$ H_{final} = (1-\alpha)H_{overtone} + \alpha H_{undertone} $$

By modulating `mirror_weight`, a user can continuously morph the fundamental physics of the universe from Major-centric to Minor-centric, observing how the ecosystem reorganizes itself in response.

# 4. The Life Engine: Agents and Autonomy

The "Life Engine" is the agent-based simulation layer that runs atop the DSP landscape. It manages the population of "Individuals," handling their lifecycle, sensory processing, vocalization timing, and audio synthesis.

## 4.1 Overview: The Individual Architecture

The `Individual` struct (`life/individual.rs`) is the atomic unit of the ecosystem. Its architecture is based on a **Control-Driven Design** where behavior is parameterized through hierarchical control structures.

### 4.1.1 Core Components

| Component | Type | Responsibility |
|-----------|------|----------------|
| `base_control` | `AgentControl` | Original control parameters set at spawn |
| `effective_control` | `AgentControl` | Current active control (may differ from base after updates) |
| `articulation` | `ArticulationWrapper` | Rhythm, gating, envelope dynamics |
| `pitch_ctl` | `PitchController` | Pitch targeting with integrated perceptual context |
| `phonation_engine` | `PhonationEngine` | Note timing, clock sources, social coupling |
| `body` | `AnySoundBody` | Sound generation (waveform synthesis, spectral projection) |

### 4.1.2 The AgentControl Hierarchy

The `AgentControl` struct (`life/control.rs`) serves as the central configuration interface for all agent behavior:

```
AgentControl
├── body: BodyControl
│   ├── method: BodyMethod (Sine | Harmonic)
│   ├── amp: f32
│   └── timbre: TimbreControl
│       ├── brightness: f32
│       ├── inharmonic: f32
│       ├── width: f32
│       └── motion: f32
├── pitch: PitchControl
│   ├── mode: PitchMode (Free | Lock)
│   ├── freq: f32
│   ├── range_oct: f32
│   ├── gravity: f32
│   ├── exploration: f32
│   └── persistence: f32
├── phonation: PhonationControl
│   ├── type: PhonationType (Interval | Clock | Field | Hold | None)
│   ├── density: f32
│   ├── sync: f32
│   ├── legato: f32
│   └── sociality: f32
└── perceptual: PerceptualControl
    ├── enabled: bool
    ├── adaptation: f32
    ├── novelty_bias: f32
    └── self_focus: f32
```

### 4.1.3 Fixed vs. Mutable Properties

A critical design principle is the distinction between **fixed** and **mutable** properties:

| Property | Mutability | Set At | Description |
|----------|------------|--------|-------------|
| `fixed_body_method` | Immutable | Spawn | `Sine` or `Harmonic` — cannot change after spawn |
| `fixed_phonation_type` | Immutable | Spawn | Phonation type — cannot change after spawn |
| `amp`, `freq`, `timbre.*` | Mutable | Runtime | Can be updated via `apply_update()` |
| `brain`, `metabolism`, `adsr` | Immutable | Spawn | Must be set before spawn via scripting API |

This separation ensures that fundamental architectural decisions (synthesis method, phonation model) are determined at birth, while expressive parameters (amplitude, frequency, timbre) can be modulated in real-time.

The Individual acts as an integration layer: it orchestrates lifecycle (metabolism, energy), coordinates cores via control-plane signals (`PlannedPitch`), and manages state transitions without coupling the cores directly.

## 4.2 SoundBody: The Actuator

The `SoundBody` trait (`life/sound_body.rs`) defines sound generation capabilities. Two implementations exist:

### 4.2.1 SineBody

A pure sine tone oscillator. Minimal parameters:
- `freq_hz`: Fundamental frequency
- `amp`: Amplitude
- `audio_phase`: Current oscillator phase

### 4.2.2 HarmonicBody and TimbreGenotype

Synthesizes a complex tone with a fundamental and configurable partials. The `TimbreGenotype` struct encodes the timbre DNA:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | `HarmonicMode` | `Harmonic` (integer multiples: $1, 2, 3...$) or `Metallic` (non-integer: $k^{1.4}$) |
| `stiffness` | `f32` | Inharmonicity coefficient; stretches partial series via $f_k = k \cdot (1 + \text{stiffness} \cdot k^2)$ |
| `brightness` | `f32` | Spectral slope; partial amplitude decays as $k^{-\text{brightness}}$ |
| `comb` | `f32` | Even harmonic attenuation (0–1); creates hollow timbres |
| `damping` | `f32` | Energy-dependent high-frequency decay; higher partials fade faster at low energy |
| `vibrato_rate` | `f32` | LFO frequency (Hz) for pitch modulation |
| `vibrato_depth` | `f32` | Vibrato extent (fraction of frequency) |
| `jitter` | `f32` | $1/f$ pink noise FM strength; adds organic fluctuation |
| `unison` | `f32` | Detune amount for chorus effect (0 = single voice) |

**Spectral Projection**: Both bodies implement `project_spectral_body()`, which writes their energy distribution back to the `Log2Space` grid for Landscape computation. This enables the system to "see" each agent's spectral footprint.

## 4.3 The Behavioral Core Stack

Behavior is organized into specialized cores, each handling a distinct aspect of agent behavior.

### 4.3.1 ArticulationCore (When/Gate)

Defined in `life/articulation_core.rs`. Manages rhythm, gating, and envelope dynamics. Three variants selected via the `brain` parameter in scripts:

| Variant | Script Name | Description | Key Parameters |
|---------|-------------|-------------|----------------|
| `KuramotoCore` | `"entrain"` | Kuramoto-style coupling to `NeuralRhythms` | `lifecycle`, `rhythm_freq`, `rhythm_sensitivity` |
| `SequencedCore` | `"seq"` | Fixed-duration envelope | `duration` (seconds) |
| `DroneCore` | `"drone"` | Continuous tone with slow amplitude sway | `sway` (modulation depth) |

**ArticulationWrapper**: Wraps the core with a `PlannedGate` struct that manages fade-in/fade-out transitions when pitch changes occur. The gate value (0–1) multiplies the amplitude, ensuring smooth transitions.

**ArticulationSignal**: The output of articulation processing:
- `amplitude`: Current envelope level
- `is_active`: Whether the agent is currently sounding
- `relaxation`: Modulation signal for vibrato/unison expansion
- `tension`: Modulation signal for jitter intensification

### 4.3.2 PitchController (Where)

Defined in `life/pitch_controller.rs`. The `PitchController` integrates pitch targeting with perceptual context into a unified component:

```
PitchController
├── core: AnyPitchCore (HillClimb algorithm)
├── perceptual: PerceptualContext (boredom/familiarity)
├── target_pitch_log2: f32
├── integration_window: f32
└── perceptual_enabled: bool
```

The controller exposes pitch behavior through `PitchControl` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `Free` | `Free` (autonomous movement) or `Lock` (fixed frequency) |
| `freq` | 220.0 | Center frequency (Free) or locked frequency (Lock) |
| `range_oct` | 6.0 | Maximum exploration range in octaves |
| `gravity` | 0.5 | Tessitura gravity (0 = none, 1 = strong pull to center) |
| `exploration` | 0.0 | Random exploration probability (0–1) |
| `persistence` | 0.5 | Resistance to movement when satisfied (0–1) |

**Scoring**: Each candidate pitch is evaluated as:

$$ \text{score} = C_{01} - d_{\text{penalty}} - g_{\text{tessitura}} + \Delta s_{\text{perceptual}} $$

The `TargetProposal` output includes `target_pitch_log2` and a `salience` score (0–1) reflecting improvement strength.

## 4.4 PerceptualContext: Subjective Adaptation

Defined in `life/perceptual.rs` and integrated into `PitchController`. Models per-agent habituation and preference, preventing agents from "getting stuck" at locally optimal positions.

The context maintains two leaky integrators per frequency bin:
- **h_fast**: Short-term exposure (boredom accumulator)
- **h_slow**: Long-term exposure (familiarity accumulator)

Controlled via `PerceptualControl` in the agent's `AgentControl`:

| Control Parameter | Mapped To | Description |
|-------------------|-----------|-------------|
| `enabled` | — | Enable/disable perceptual adaptation |
| `adaptation` | `tau_fast`, `tau_slow` | Time constants for decay (0 = fast, 1 = slow) |
| `novelty_bias` | `w_boredom` | Weight of boredom penalty |
| `self_focus` | `rho_self` | Self-injection ratio |

Internal parameters (derived from control):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_fast` | 0.5 s | Time constant for boredom decay |
| `tau_slow` | 20.0 s | Time constant for familiarity decay |
| `w_boredom` | 1.0 | Weight of boredom penalty |
| `w_familiarity` | 0.2 | Weight of familiarity bonus |
| `rho_self` | 0.15 | Self-injection ratio (how much the agent's own position contributes) |
| `boredom_gamma` | 0.5 | Curvature exponent for boredom ($h_{\text{fast}}^\gamma$) |
| `self_smoothing_radius` | 1 | Spatial smoothing radius for self-injection |
| `silence_mass_epsilon` | 1e-6 | Threshold for detecting silence |

**Score Adjustment**:

$$ \Delta s = w_{\text{familiarity}} \cdot h_{\text{slow}} - w_{\text{boredom}} \cdot h_{\text{fast}}^{\gamma} $$

This creates a dynamic where agents are drawn to familiar regions but pushed away from over-visited locations.

## 4.5 PhonationEngine: Timing and Vocalization

Defined in `life/phonation_engine.rs`. The PhonationEngine governs *when* an agent vocalizes, managing note onsets, durations, and coordination with other agents.

### 4.5.1 Phonation Types

The phonation behavior is selected via `PhonationControl.type` in the agent's control:

| Type | Script Name | Description |
|------|-------------|-------------|
| `Interval` | `"decay"` | Probabilistic onset with decay envelope |
| `Clock` | — | Clock-synchronized onset |
| `Field` | `"grain"` | Granular, field-responsive phonation |
| `Hold` | `"hold"` | Sustain once per lifecycle; ignores density/sync/legato |
| `None` | — | Silent agent (no phonation) |

### 4.5.2 PhonationControl Parameters

The `PhonationControl` struct provides high-level control over phonation behavior:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `type` | `Interval` | — | Phonation model (see above) |
| `density` | 0.5 | 0–1 | Note density / onset rate |
| `sync` | 0.5 | 0–1 | Synchronization to theta rhythm |
| `legato` | 0.5 | 0–1 | Note duration (0 = staccato, 1 = legato) |
| `sociality` | 0.0 | 0–1 | Social coupling strength |

### 4.5.3 Social Coupling

The `SocialConfig` enables agents to respond to the vocalization density of the population:

| Parameter | Description |
|-----------|-------------|
| `coupling` | Strength of social influence (0 = independent) |
| `bin_ticks` | Temporal resolution for density measurement |
| `smooth` | Smoothing factor for density trace |

**SocialDensityTrace** (`life/social_density.rs`): Tracks the recent onset density of the population, allowing agents to synchronize or avoid crowded moments.

Global coupling can be set via the scripting API using `set_global_coupling(value)`.

## 4.6 Lifecycle and Metabolism

Agents are governed by energy dynamics modeled on biological metabolism. The `LifecycleConfig` (`life/lifecycle.rs`) defines two modes:

### 4.6.1 Decay Mode

Models transient sounds (plucks, percussion):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_energy` | 1.0 | Starting energy pool |
| `half_life_sec` | — | Exponential decay half-life |
| `attack_sec` | 0.01 | Attack ramp duration |

Energy evolves as: $E(t) = E_0 \cdot e^{\lambda t}$ where $\lambda = \ln(0.5) / t_{1/2}$

### 4.6.2 Sustain Mode

Models sustained sounds with metabolic feedback:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_energy` | 1.0 | Starting energy pool |
| `metabolism_rate` | — | Energy drain per second |
| `recharge_rate` | 0.5 | Energy gain rate (consonance-dependent) |
| `action_cost` | 0.02 | Energy cost per vocalization |
| `envelope` | — | ADSR config (`attack_sec`, `decay_sec`, `sustain_level`) |

**Breath Gain Feedback**: The `breath_gain` parameter (set at spawn via `breath_gain_init`) determines how much consonance contributes to energy recovery. An agent in a dissonant region "starves" while one in a consonant region "feeds."

This creates Darwinian pressure: **Survival of the Consonant**. Musical structure emerges because only agents that find harmonic relationships survive to be heard.

## 4.7 Pitch Retargeting and Control Flow

Agents move through frequency space to improve fitness. The control-rate update flow is orchestrated by `tick_control()`:

```
tick_control(dt_sec, rhythms, landscape, global_coupling)
└── update_articulation(dt_sec, rhythms, landscape, global_coupling)
    ├── update_pitch_target(rhythms, dt_sec, landscape)
    │   └── pitch_ctl.update_pitch_target(...)
    ├── update_articulation_autonomous(dt_sec, rhythms)
    │   └── articulation.update_gate(&planned, rhythms, dt_sec)
    └── tick_articulation_lifecycle(dt_sec, rhythms, landscape, global_coupling)
        └── articulation.process(consonance, rhythms, dt_sec, global_coupling)
```

### 4.7.1 Pitch Mode: Free vs. Lock

The `PitchControl.mode` determines pitch behavior:

| Mode | Behavior |
|------|----------|
| `Free` | Autonomous pitch exploration via hill-climbing algorithm |
| `Lock` | Fixed frequency; gate always open; no pitch search |

When `mode` is `Lock` (set automatically when using `.freq()` or `.place()` in scripts), the agent immediately snaps to the target frequency without fade transitions.

### 4.7.2 The Hop Policy (Free Mode)

In `Free` mode, pitch movement uses discrete **hops** rather than continuous portamento:

1. **Proposal**: `PitchController` evaluates candidates and returns a `TargetProposal` with target pitch and salience.

2. **Gate Coordination**: `update_articulation_autonomous()` constructs a `PlannedPitch`:
   - `target_pitch_log2`: Next intended pitch
   - `jump_cents_abs`: Distance to target (cents)
   - `salience`: Improvement strength (0–1)

3. **Fade-out**: Gate closes when `jump_cents_abs > 10` (movement threshold)

4. **Snap**: When gate < 0.1, `body.set_pitch_log2()` updates to target

5. **Fade-in**: Gate reopens, new pitch sounds

**Ordering**: On snap, pitch updates *before* consonance evaluation, ensuring the Landscape score reflects the actual sounding frequency. These timing-sensitive transitions are guarded by regression tests.

### 4.7.3 Live Updates via apply_update()

The `apply_update(&ControlUpdate)` method allows runtime modification of mutable parameters:

| Updatable | Field | Constraint |
|-----------|-------|------------|
| Amplitude | `amp` | Clamped to [0, 1] |
| Frequency | `freq` | Clamped to [1, 20000] Hz; forces `mode = Lock` |
| Brightness | `timbre_brightness` | Clamped to [0, 1] |
| Inharmonic | `timbre_inharmonic` | Clamped to [0, 1] |
| Width | `timbre_width` | Clamped to [0, 1] |
| Motion | `timbre_motion` | Clamped to [0, 1] |

**Important**: Updates cannot change `body.method` or `phonation.type`. These are validated by `ensure_fixed_kinds()` and will return an error if mismatched.

# 5. Temporal Dynamics: Neural Rhythms

Conchordal eschews the concept of a master clock or metronome. Instead, time is structured by a continuous modulation field inspired by Neural Oscillations (brainwaves). This is the "Time" equivalent of the "Space" landscape.

## 5.1 The Modulation Bank

The `NeuralRhythms` struct manages a bank of resonating filters tuned to physiological frequency bands:

*   **Delta (0.5–4 Hz)**: The macroscopic "pulse" of the ecosystem. Agents locked to this band play long, phrase-level notes.
*   **Theta (4–8 Hz)**: The "articulation" rate. Governs syllabic rhythms and medium-speed motifs.
*   **Alpha (8–12 Hz)**: The "texture" rate. Used for tremolo, vibrato, and shimmering effects.
*   **Beta (15–30 Hz)**: The "tension" rate. High-speed flutters associated with dissonance or excitement.

## 5.2 Vitality and Self-Oscillation

Each band is implemented as a Resonator, a damped harmonic oscillator. A key parameter is `vitality`.

*   `Vitality = 0`: The resonator acts as a passive filter. It only rings when excited by an event (e.g., a loud agent spawning) and then decays.
*   `Vitality > 0`: The resonator has active gain. It can self-oscillate, maintaining a rhythmic cycle even in the absence of input.

This creates a two-way interaction: The global rhythm drives the agents (entrainment), but the agents also drive the global rhythm (excitation). A loud "kick" agent spawning in the Delta band will "ring" the Delta resonator, causing other agents coupled to that band to synchronize.

**Input Source**: The `NeuralRhythms` are driven by the `DorsalStream` (§6.2.4), which extracts rhythmic energy from the audio signal via 3-band flux detection. The DorsalStream runs synchronously on the main thread, feeding `(u_theta, u_delta, vitality)` into the oscillator bank each frame.

## 5.3 Kuramoto Entrainment

The `entrain` ArticulationCore uses a Kuramoto-style model of coupled oscillators.

$$ \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i) $$

In Conchordal, the "coupling" $K$ is to the global `NeuralRhythms` rather than directly to every other agent (Mean Field approximation).

*   **Sensitivity**: Each agent has a sensitivity profile determining which bands (Delta, Theta, etc.) it listens to.
*   **Phase Locking**: Agents adjust their internal articulation phase to match the phase of the resonator.

This results in emergent synchronization. Agents spawned at random times will gradually align their attacks to the beat of the Delta or Theta bands, creating coherent rhythmic patterns without a central sequencer.

# 6. System Architecture and Implementation Details

Conchordal is implemented in Rust to satisfy the stringent requirements of real-time audio (latency < 10ms) alongside heavy numerical analysis (NSGT/Convolution). The architecture uses a concurrent, lock-free design pattern.

## 6.1 Threading Model

The application creates two primary thread contexts:

1.  **Audio Thread (Real-Time Priority)**:
    *   Managed by `cpal` in `audio/output.rs`.
    *   **Constraint**: Must never block. No Mutexes, no memory allocation.
    *   **Responsibility**: Iterates through the `Population`, calling `render_wave` on every active agent, mixing the output, and pushing to the hardware buffer. It reads from a read-only snapshot of the Landscape.

2.  **Analysis Worker (Background Priority)**:
    *   Defined in `core/analysis_worker.rs`.
    *   **Responsibility**: Unified processing of spectral analysis (NSGT), Roughness (ERB convolution), and Harmonicity (Sibling Projection) in a single dedicated thread.
    *   **Input**: Receives time-domain audio hops via `hop_rx: Receiver<(u64, Arc<[f32]>)>`.
    *   **Output**: Sends complete `AnalysisResult = (frame_id, Landscape)` tuples via `result_tx`.
    *   **Parameter Updates**: Receives `LandscapeUpdate` messages via `update_rx` for runtime parameter changes.

The **Main/GUI Thread** runs the `egui` visualizer and the `Rhai` scripting engine, handling user input and scenario execution.

### 6.1.1 Analysis Worker Architecture

The Analysis Worker (`core/analysis_worker.rs`) implements a producer-consumer pattern:

```rust
pub fn run(
    mut stream: AnalysisStream,
    hop_rx: Receiver<(u64, Arc<[f32]>)>,
    result_tx: Sender<AnalysisResult>,
    update_rx: Receiver<LandscapeUpdate>,
)
```

**Key Design Principles**:

1. **No Hop Skipping**: The worker drains the backlog but processes *all* queued hops in order. NSGT-RT maintains an internal ring buffer assuming time continuity; dropping hops creates broadband artifacts ("mystery peaks").

2. **Sequential Processing**: Each hop is processed in order to preserve the per-hop `dt` used by normalizers:
   ```rust
   for hop in &hops[1..] {
       analysis = stream.process(hop.as_ref());
   }
   ```

3. **Non-Blocking Result Delivery**: Uses `try_send()` to avoid blocking on the result channel.

## 6.2 Data Flow

The unified analysis pipeline simplifies the data flow compared to the previous multi-worker design:

```
Audio Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Analysis Worker Thread                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ AnalysisStream (core/stream/analysis.rs)    │   │
│  │  ├── RtNsgtKernelLog2 (NSGT spectrum)       │   │
│  │  ├── SpectralFrontEnd (normalization)       │   │
│  │  ├── RoughnessKernel (ERB convolution)      │   │
│  │  └── HarmonicityKernel (Sibling projection) │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
    │
    ▼ (frame_id, Landscape)
Main Thread
    │
    ▼
Population (agents read Landscape for fitness)
```

### 6.2.1 AnalysisStream Processing

The `AnalysisStream` (`core/stream/analysis.rs`) performs the complete analysis pipeline per hop:

1. **NSGT Spectrum**: `nsgt_rt.process_hop(audio)` → power envelope
2. **Spectral Frontend**: Normalization and subjective intensity computation
3. **Roughness Computation**:
   - Level-dependent roughness strength (total/max/p95 modes)
   - Level-invariant roughness shape (normalized density)
   - Ratio-to-state mapping via `roughness_ratio_to_state01()`
4. **Harmonicity Evaluation**: `harmonicity_kernel.potential_h_from_log2_spectrum()`
5. **Landscape Assembly**: All metrics combined into a single `Landscape` snapshot

### 6.2.2 Landscape Output Fields

The resulting `Landscape` contains:

| Field | Description |
|-------|-------------|
| `nsgt_power` | Raw NSGT power envelope |
| `roughness` | Per-bin roughness strength (level-dependent) |
| `roughness01` | Normalized roughness (0–1 range) |
| `roughness_scalar_*` | Aggregate metrics (total, max, p95, raw, norm) |
| `harmonicity` | Per-bin harmonic potential |
| `consonance01` | Combined consonance metric |
| `subjective_intensity` | Perceptual loudness density |
| `loudness_mass` | Integrated loudness |
| `rhythm` | `NeuralRhythms` state (theta, delta, beta oscillations) |

### 6.2.3 Runtime Parameter Updates

The `LandscapeUpdate` struct allows runtime modification of analysis parameters:

| Field | Description |
|-------|-------------|
| `mirror` | Harmonicity mirror weight (0 = overtone, 1 = undertone) |
| `limit` | Harmonicity parameter limit |
| `roughness_k` | Roughness tolerance scaling |

Updates are applied via `stream.apply_update(upd)` before processing each batch.

This unified architecture ensures that the audio thread always sees a consistent snapshot of the physics, while the single analysis worker maintains time continuity for accurate spectral analysis.

### 6.2.4 DorsalStream: Rhythm Extraction

The `DorsalStream` (`core/stream/dorsal.rs`) handles fast rhythm extraction and motor synchronization, running **synchronously on the main thread** for low-latency response. It provides the input signal that drives the `NeuralRhythms` modulation bank (§5).

**Processing Pipeline**:

1. **3-Band Crossover Flux Detection**:
   - Low band: < 200 Hz
   - Mid band: 200–3000 Hz
   - High band: > 3000 Hz
   - Positive flux summed across bands

2. **Non-linear Neural Activation**:
   ```
   drive = tanh(raw_flux × 500.0)
   u_theta = clamp(drive, 0, 1)
   ```
   High gain + tanh saturation detects ambient shifts

3. **Envelope Smoothing**:
   - Delta envelope with τ = 0.6s time constant
   - Provides `u_delta` modulation signal

4. **RhythmEngine Update**:
   - Feeds `(u_theta, u_delta, vitality)` to oscillator bank
   - Returns updated `NeuralRhythms` state

**Output** (`DorsalMetrics`):

| Field | Description |
|-------|-------------|
| `e_low` | Low band energy |
| `e_mid` | Mid band energy |
| `e_high` | High band energy |
| `flux` | Positive flux sum (onset strength) |

**Vitality Parameter**: Controls self-oscillation energy of the rhythm section (0–1). Higher values allow rhythms to persist even during silence.

## 6.3 The Conductor: Scripting with Rhai

The Conductor module acts as the interface between the human artist and the ecosystem. It embeds the [Rhai](https://rhai.rs/) scripting language, exposing a high-level API for controlling the simulation.

### 6.3.1 Core Concepts

The scripting API is built around three fundamental types:

| Type | Description |
|------|-------------|
| `SpeciesHandle` | Blueprint/template for agent configuration |
| `GroupHandle` | Reference to a live group of spawned agents |
| `SpawnStrategy` | Algorithm for determining spawn frequencies |

### 6.3.2 Species and Presets

**Built-in Presets** (available as global variables):

| Preset | Body Method | Timbre |
|--------|-------------|--------|
| `sine` | Sine | Pure tone |
| `harmonic` | Harmonic | Default timbre |
| `saw` | Harmonic | brightness=0.85, width=0.2 |
| `square` | Harmonic | brightness=0.65, width=0.1 |
| `noise` | Harmonic | brightness=1.0, motion=1.0, width=0.35 |

**Species Builders** (chainable methods on `SpeciesHandle`):

| Method | Description |
|--------|-------------|
| `derive(species)` | Clone a species for modification |
| `.amp(value)` | Set amplitude (0–1) |
| `.freq(value)` | Set frequency (Hz); sets `PitchMode::Lock` |
| `.timbre(brightness, width)` | Set timbre parameters |
| `.brain(name)` | Set articulation type: `"drone"`, `"seq"`, `"entrain"` |
| `.phonation(name)` | Set phonation type: `"hold"`, `"decay"`, `"grain"` |
| `.metabolism(rate)` | Set energy consumption rate |
| `.adsr(a, d, s, r)` | Set envelope parameters |

### 6.3.3 Group Lifecycle

**Group Creation**:
```rhai
let g = create(species, count);  // Returns GroupHandle (Draft state)
```

**Group State Machine**:

```
   create()          flush()/wait()         release()/scope exit
Draft ────────────────► Live ──────────────────► Released
  │                                                 │
  │ (scope exit without flush)                      │
  ▼                                                 ▼
Dropped ◄───────────────────────────────────────────
```

| State | Description |
|-------|-------------|
| `Draft` | Created but not spawned; can modify all parameters |
| `Live` | Spawned and active; can only modify mutable parameters (amp, freq, timbre) |
| `Released` | Marked for removal with fade-out |
| `Dropped` | Never spawned (generates warning) |

**Group Builders** (chainable methods on `GroupHandle`):

| Method | Draft | Live | Description |
|--------|-------|------|-------------|
| `.amp(value)` | ✓ | ✓ | Set/update amplitude |
| `.freq(value)` | ✓ | ✓ | Set/update frequency (clears strategy if Draft) |
| `.timbre(b, w)` | ✓ | ✓ | Set/update timbre |
| `.place(strategy)` | ✓ | ✗ | Set spawn frequency strategy |
| `.brain(name)` | ✓ | ✗ | Set articulation type |
| `.phonation(name)` | ✓ | ✗ | Set phonation type |
| `.metabolism(rate)` | ✓ | ✗ | Set metabolism rate |
| `.adsr(a, d, s, r)` | ✓ | ✗ | Set envelope |

### 6.3.4 Spawn Strategies

Strategies determine how frequencies are assigned when spawning multiple agents:

| Strategy | Constructor | Description |
|----------|-------------|-------------|
| Consonance | `consonance(root_freq)` | Pick highest consonance within multiplier range of root |
| Consonance Density | `consonance_density(min, max)` | Weighted-random based on consonance in freq range |
| Random Log | `random_log(min, max)` | Uniform distribution in log-frequency |
| Linear | `linear(start, end)` | Linear interpolation across agents |

**Consonance Strategy Modifiers**:
- `.range(min_mul, max_mul)`: Set multiplier range (default: 1–4); spawn freq = root × [min_mul, max_mul]
- `.min_dist(erb)`: Set minimum separation in ERB (default: 1.0)

### 6.3.5 Timeline Control

| Function | Description |
|----------|-------------|
| `create(species, count)` | Create a new group of `count` agents from `species`; returns `GroupHandle` in Draft state |
| `wait(sec)` | Commits drafts, advances cursor by `sec` seconds |
| `flush()` | Commits drafts without advancing time |
| `release(group)` | Marks group for removal with fade |

### 6.3.6 Scopes and Scenes

**Scene**: Named section with automatic cleanup:
```rhai
scene("intro", || {
    let g = create(sine, 3);
    flush();
    wait(2.0);
});  // g is automatically released here
```

**Play**: Scoped execution without scene marker:
```rhai
play(|| {
    let g = create(sine, 1);
    flush();
    wait(1.0);
});
```

**Parallel**: Execute multiple closures simultaneously:
```rhai
parallel([
    || { create(sine, 1); wait(0.5); },
    || { create(sine, 1); wait(1.0); }
]);  // Cursor advances to max end time (1.0)
```

### 6.3.7 World Parameters

| Function | Description |
|----------|-------------|
| `seed(value)` | Set random seed for reproducibility |
| `set_harmonicity_mirror_weight(value)` | Adjust overtone/undertone balance (0–1) |
| `set_global_coupling(value)` | Set global social coupling strength |
| `set_roughness_k(value)` | Set roughness tolerance parameter |

### 6.3.8 Example Script

```rhai
seed(12345);

// Define species
let anchor = derive(sine).amp(0.4).phonation("hold");
let voice = derive(harmonic).amp(0.2).timbre(0.7, 0.1);

scene("exposition", || {
    // Create anchor drone
    let a = create(anchor, 1).freq(220.0);
    flush();
    wait(1.0);

    // Spawn voices on harmonic series
    let strat = consonance(220.0).range(1.0, 4.0).min_dist(0.8);
    for i in 0..4 {
        create(voice, 1).place(strat);
        wait(0.5);
    }

    // Modulate physics
    set_harmonicity_mirror_weight(0.5);
    wait(2.0);
});
```

**Scenario Parsing**: Scenarios are loaded from `.rhai` files via `ScriptHost::load_script()`. This separation allows users to compose the "Macro-Structure" (the narrative arc, the changing laws of physics) while the "Micro-Structure" (the specific notes and rhythms) emerges from the agents' adaptation to those changes.

# 7. Case Studies: Analysis of Emergent Behavior

The following examples, derived from the `samples/` directory, illustrate how specific parameter configurations lead to complex musical behaviors using the current scripting API.

## 7.1 Case Study: Polyrhythmic Interaction (`samples/02_mechanisms/rhythmic_sync.rhai`)

This script demonstrates polyrhythmic emergence through parallel timelines.

```rhai
let click = derive(sine)
    .amp(0.4)
    .phonation("decay")
    .adsr(0.01, 0.1, 0.0, 0.2);

parallel([
    || {
        for i in 0..8 {
            create(click, 1).freq(60.0);
            wait(0.5);
        }
    },
    || {
        for i in 0..6 {
            create(click, 1).freq(120.0);
            wait(0.666);
        }
    }
]);
```

**Analysis**:
1. **Species Definition**: A `click` species is derived from `sine` with decay phonation and short ADSR envelope.
2. **Parallel Execution**: Two independent timelines run simultaneously—one at 60 Hz with 0.5s intervals (4:4), another at 120 Hz with 0.666s intervals (3:3).
3. **Emergence**: The 4-against-3 polyrhythm creates a 12-beat cycle without explicit synchronization. The `parallel()` function advances the cursor to the maximum child end time.

## 7.2 Case Study: Mirror Dualism (`samples/04_ecosystems/mirror_dualism.rhai`)

This script explores the structural role of the `mirror_weight` parameter.

```rhai
let anchor = derive(sine).amp(0.4).phonation("hold");
let voice = derive(sine).amp(0.2).phonation("hold");

scene("Mirror Dualism", || {
    create(anchor, 1).freq(261.63);
    flush();
    wait(0.8);

    set_harmonicity_mirror_weight(0.0);
    for i in 0..4 {
        let strat = consonance(261.63).range(1.0, 3.0).min_dist(0.9);
        create(voice, 1).place(strat);
    }
    wait(1.5);

    set_harmonicity_mirror_weight(1.0);
    for i in 0..4 {
        let strat = consonance(261.63).range(0.8, 2.5).min_dist(0.9);
        create(voice, 1).place(strat);
    }
    wait(1.5);
});
```

**Analysis**:
1. **Setup**: An anchor drone at C4 (261.63 Hz) with `phonation("hold")` for sustained tone.
2. **State A (Overtone/Major)**: `set_harmonicity_mirror_weight(0.0)`. Voices spawn on the harmonic series using `consonance()` strategy. The system favors overtone relationships—agents cluster around E4 and G4, forming a C Major triad.
3. **State B (Undertone/Minor)**: `set_harmonicity_mirror_weight(1.0)`. The landscape inverts to undertone projection. Agents find stability at different intervals, creating a Phrygian/Minor texture.
4. **Scoped Cleanup**: The `scene()` automatically releases all groups when it exits.

## 7.3 Case Study: Drift and Flow (`samples/04_ecosystems/drift_flow.rhai`)

This script validates live parameter updates and frequency drift.

```rhai
let anchor = derive(sine).amp(0.6).phonation("hold");
let slider = derive(sine).amp(0.4).phonation("hold");
let swarm = derive(sine).amp(0.15).phonation("hold");

scene("Drift Flow", || {
    let a = create(anchor, 1).freq(65.41);
    let s = create(slider, 1).freq(138.59);
    flush();
    wait(1.0);

    s.freq(220.0);
    flush();
    wait(1.5);

    release(s);
    wait(0.5);

    for i in 0..5 {
        let strat = consonance(130.0).range(1.0, 4.0).min_dist(1.0);
        create(swarm, 1).place(strat);
        wait(0.6);
    }

    a.freq(87.31);
    flush();
    wait(1.0);
});
```

**Analysis**:
1. **Initial State**: Anchor at C2 (65.41 Hz), slider at C#3 (138.59 Hz)—a dissonant minor 9th.
2. **Live Update**: `s.freq(220.0)` demonstrates runtime frequency modification. The slider smoothly transitions to A3, creating a consonant relationship with the anchor.
3. **Release**: `release(s)` marks the slider for fade-out removal.
4. **Swarm Spawning**: Five agents spawn sequentially on the harmonic series of 130 Hz with minimum 1.0 ERB separation.
5. **Anchor Movement**: `a.freq(87.31)` shifts the anchor to F2, demonstrating that even "anchor" agents can be dynamically repositioned, causing the swarm to reorganize around the new root.

# 8. Conclusion

Conchordal successfully establishes a proof-of-concept for Bio-Mimetic Computational Audio. By replacing the rigid abstractions of music theory (notes, grids, BPM) with continuous physiological models (`Log2Space`, ERB bands, neural oscillation), it creates a system where music is not constructed, but grown.

The technical architecture—anchored by the `Log2Space` coordinate system and the "Sibling Projection" algorithm—provides a robust mathematical foundation for this new paradigm. The use of Rust ensures that these complex biological simulations can run in real-time, bridging the gap between ALife research and performative musical instruments.

Future development of Conchordal will focus on spatialization (extending the landscape to 3D space) and evolutionary genetics (allowing successful agents to pass on their `TimbreGenotype`), further deepening the analogy between sound and life.

# Appendix A: Key System Parameters

## A.1 Core Analysis Parameters

| Parameter | Module | Unit | Description |
| :--- | :--- | :--- | :--- |
| `bins_per_oct` | `Log2Space` | Int | Resolution of the frequency grid (typ. 48-96). |
| `sigma_cents` | `Harmonicity` | Cents | Width of harmonic peaks. Lower = stricter intonation. |
| `mirror_weight` | `Harmonicity` | 0.0-1.0 | Balance between Overtone (Major) and Undertone (Minor) gravity. |
| `roughness_k` | `Roughness` | Float | Saturation parameter for roughness mapping. Default: $\approx 0.4286$. |
| `roughness_weight` | `Landscape` | Float | Weight of roughness penalty in consonance calculation. Default: 1.0. |
| `vitality` | `DorsalStream` | 0.0-1.0 | Self-oscillation energy of the rhythm section. |

## A.2 AgentControl Parameters

The `AgentControl` hierarchy provides the primary configuration interface for agents.

### A.2.1 BodyControl

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `method` | `Sine` | — | `Sine` or `Harmonic` (immutable after spawn) |
| `amp` | 0.18 | 0–1 | Output amplitude |

**TimbreControl** (nested in `BodyControl`):

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `brightness` | 0.6 | 0–1 | Spectral slope (higher = brighter) |
| `inharmonic` | 0.0 | 0–1 | Inharmonicity coefficient (stiffness) |
| `width` | 0.0 | 0–1 | Unison/chorus detune amount |
| `motion` | 0.0 | 0–1 | Jitter/vibrato depth |

### A.2.2 PitchControl

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `mode` | `Free` | — | `Free` (autonomous) or `Lock` (fixed frequency) |
| `freq` | 220.0 | 1–20000 Hz | Center frequency (Free) or locked frequency (Lock) |
| `range_oct` | 6.0 | 0–6 | Maximum exploration range in octaves |
| `gravity` | 0.5 | 0–1 | Tessitura gravity strength |
| `exploration` | 0.0 | 0–1 | Random exploration probability |
| `persistence` | 0.5 | 0–1 | Resistance to movement when satisfied |

### A.2.3 PhonationControl

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `type` | `Interval` | — | `Interval`, `Clock`, `Field`, `Hold`, or `None` |
| `density` | 0.5 | 0–1 | Note onset density/rate |
| `sync` | 0.5 | 0–1 | Synchronization to theta rhythm |
| `legato` | 0.5 | 0–1 | Note duration (0 = staccato, 1 = legato) |
| `sociality` | 0.0 | 0–1 | Social coupling strength |

### A.2.4 PerceptualControl

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `enabled` | true | — | Enable/disable perceptual adaptation |
| `adaptation` | 0.5 | 0–1 | Time constants (0 = fast adaptation, 1 = slow) |
| `novelty_bias` | 1.0 | 0–∞ | Weight of boredom penalty |
| `self_focus` | 0.15 | 0–1 | Self-injection ratio |

## A.3 Timbre Parameters (TimbreGenotype)

Internal timbre representation used by `HarmonicBody`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `mode` | `Harmonic` | `Harmonic` (integer multiples) or `Metallic` (non-integer). |
| `stiffness` | 0.0 | Inharmonicity coefficient (maps from `timbre.inharmonic`). |
| `brightness` | 0.6 | Spectral slope (maps from `timbre.brightness`). |
| `comb` | 0.0 | Even harmonic attenuation. |
| `damping` | 0.5 | Energy-dependent high-frequency decay. |
| `vibrato_rate` | 5.0 Hz | Vibrato LFO frequency. |
| `vibrato_depth` | 0.0 | Vibrato extent (maps from `timbre.motion * 0.02`). |
| `jitter` | 0.0 | Pink noise FM strength (maps from `timbre.motion`). |
| `unison` | 0.0 | Detune amount for chorus effect (maps from `timbre.width`). |

## A.4 Scripting API Quick Reference

### A.4.1 Species Presets

| Preset | Body | Brightness | Width | Motion |
| :--- | :--- | :--- | :--- | :--- |
| `sine` | Sine | — | — | — |
| `harmonic` | Harmonic | 0.6 | 0.0 | 0.0 |
| `saw` | Harmonic | 0.85 | 0.2 | 0.0 |
| `square` | Harmonic | 0.65 | 0.1 | 0.0 |
| `noise` | Harmonic | 1.0 | 0.35 | 1.0 |

### A.4.2 Brain Types (Articulation)

| Script Name | Core | Description |
| :--- | :--- | :--- |
| `"entrain"` | `KuramotoCore` | Kuramoto-style neural rhythm coupling |
| `"seq"` | `SequencedCore` | Fixed-duration envelope |
| `"drone"` | `DroneCore` | Continuous tone with slow sway |

### A.4.3 Phonation Types

| Script Name | Type | Description |
| :--- | :--- | :--- |
| `"hold"` | `Hold` | Sustain once per lifecycle |
| `"decay"` | `Interval` | Probabilistic onset with decay |
| `"grain"` | `Field` | Granular, field-responsive |

### A.4.4 World Parameter Functions

| Function | Parameter | Description |
| :--- | :--- | :--- |
| `seed(value)` | — | Set random seed |
| `set_harmonicity_mirror_weight(v)` | `mirror` | Overtone/undertone balance (0–1) |
| `set_global_coupling(v)` | `coupling` | Global social coupling strength |
| `set_roughness_k(v)` | `roughness_k` | Roughness tolerance

# Appendix B: Mathematical Summary

**Consonance Fitness Function:**

$$ C_{signed} = \text{clip}(H_{01} - w_r \cdot R_{01},\; -1,\; 1) $$

$$ C_{01} = \frac{C_{signed} + 1}{2} $$

**Roughness Saturation Mapping** (from reference-normalized ratio $x$ to $R_{01} \in [0,1]$):

$$
R_{01}(x; k) = \begin{cases}
0 & \text{if } x \leq 0 \\
x \cdot \frac{1}{1+k} & \text{if } 0 < x < 1 \\
1 - \frac{k}{x+k} & \text{if } x \geq 1
\end{cases}
$$

where $k$ is `roughness_k` (default $\approx 0.4286$). The function is continuous at $x=1$ and saturates to 1 as $x \to \infty$.

**Harmonicity Projection (Sibling Algorithm):**
$$ H[i] = (1-\alpha)\sum_m \left( \sum_k A[i+\log_2(k)] \right)[i-\log_2(m)] + \alpha \sum_m \left( \sum_k A[i-\log_2(k)] \right)[i+\log_2(m)] $$

**Roughness Convolution:**
$$ R_{shape}(z) = \int A(\tau) \cdot K_{plomp}(|z-\tau|_{ERB}) d\tau $$