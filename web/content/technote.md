+++
title = "Technical Note: The Physics of Conchordal"
description = "A deep dive into the psychoacoustic algorithms, logarithmic signal processing, and artificial life strategies powering the Conchordal ecosystem."
template = "page.html"
[extra]
source_commit = "404a3a1"
author = "Koichi Takahashi"
last_updated = "2026-08-16"
source_version = "0.4.0"
source_snapshot = "2026-08-16T13:25:15+09:00"
+++

# 1. Introduction: The Bio-Acoustic Paradigm

Conchordal represents a fundamental divergence from established norms in generative music and computational audio. Where traditional systems rely on symbolic manipulation—operating on grids of quantized pitch (MIDI, Equal Temperament) and discretized time (BPM, measures)—Conchordal functions as a continuous, biologically grounded simulation of auditory perception. It posits that musical structure is not an artifact of abstract composition but an emergent property of acoustic survival.

This technical note serves as an exhaustive reference for the system's architecture, signal processing algorithms, and artificial life strategies. It details how Conchordal synthesizes the principles of psychoacoustics—specifically critical band theory, virtual pitch perception, and neural entrainment—with the dynamics of an autonomous ecosystem. In this environment, sound is treated as a living organism, a "Voice" possessing metabolism, sensory processing capabilities, and the autonomy to navigate a hostile spectral terrain.

The emergent behavior of the system is driven by a unified fitness function: the pursuit of Consonance. Agents within the Conchordal ecosystem do not follow a pre-written score. Instead, they continuously analyze their environment to maximize their "Spectral Comfort"—defined as the minimization of sensory roughness—and their "Harmonic Stability," or the maximization of virtual root strength. The result is a self-organizing soundscape where harmony, rhythm, and timbre evolve organically through the interactions of physical laws rather than deterministic sequencing.

This document explores the four foundational pillars of the Conchordal architecture, mirroring the Manifesto's two perceptual axes:

1.  **The Psychoacoustic Coordinate System**: The mathematical framework of `Log2Space` and ERB scales that replaces linear Hertz and integer MIDI notes.
2.  **The Frequency Axis — the Auditory Landscape**: The real-time DSP pipeline that computes Roughness ($R$) and Harmonicity ($H$) fields from the raw audio stream.
3.  **The Temporal Axis — the Emergent Meter**: The coupled-oscillator model that forms a metrical percept from the ecosystem's own onsets.
4.  **The Life Engine**: The agent-based model governing the metabolism, movement, and entrainment of the audio entities that inhabit both terrains.

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

# 3. The Frequency Axis: The Auditory Landscape

Music cognition treats consonance not as a property of notation but as a *percept* with two distinct components. **Sensory dissonance** is a bottom-up sensation: when two partials fall within the same critical band on the basilar membrane, their interference is heard as beating and roughness—a phenomenon present for any listener, musical training or not. **Tonal fusion** is a structural percept: partials standing in harmonic relation are bound by the auditory system into a single tone with a single (possibly *virtual*) pitch, heard as one voice rather than many. What musicians call consonance is the integration of the two—low roughness *and* strong fusion. The small-integer ratios celebrated since Pythagoras are properties of the *stimulus*; the harmony itself is constructed in the listener.

Both components have quantitative models. Helmholtz located dissonance in beating; Plomp and Levelt measured its dependence on critical bandwidth, yielding the roughness curve that peaks at roughly a quarter of a critical band and vanishes beyond it. Fusion runs through Stumpf's *Tonverschmelzung* to Terhardt's virtual pitch: the auditory brainstem, phase-locked to temporal fine structure, matches incoming partials against harmonic templates and infers the fundamental that best explains them—even when that fundamental is physically absent. The two mechanisms are separable: a sound can be smooth yet unfused, or fused yet rough.

Conchordal adopts this account literally. The system runs a cochlear front end on its own sound, computes a Roughness field and a Harmonicity field over the entire frequency axis, and integrates them into Consonance—a terrain whose peaks are where a new tone would *fuse* and whose valleys are where it would *grate*. Nothing in the system knows an interval name or a ratio; agents feel only this terrain. Section 4 applies the same move to time: just as harmony is computed from a model of the cochlea rather than imposed, meter will be computed from a model of beat perception rather than scheduled.

The "Landscape" is the central data structure in Conchordal. It acts as the shared environment for all agents, a dynamic scalar field representing the psychoacoustic "potential" of every frequency bin. Agents do not interact directly with each other; they interact with the Landscape, which aggregates the spectral energy of the entire population. This decouples the complexity of the simulation from the number of agents ($O(N)$ vs $O(N^2)$).

The Landscape is updated every audio frame (or block) by the Analysis Worker. It synthesizes two primary metrics:

*   **Roughness ($R$)**: The sensory dissonance caused by rapid beating between proximal partials.
*   **Harmonicity ($H$)**: The measure of virtual pitch strength and spectral periodicity.

Both metrics are normalized to the $[0, 1]$ range. The rest of this chapter follows the analysis pipeline in order: Section 3.1 builds the log-frequency spectrum (NSGT), Sections 3.2 and 3.3 derive the Roughness and Harmonicity fields from it, and Section 3.4 integrates the two into the Consonance terrain that agents actually climb.

## 3.1 Non-Stationary Gabor Transform (NSGT)

To populate the `Log2Space` with spectral data, Conchordal uses a custom implementation of the Non-Stationary Gabor Transform (NSGT). Unlike the Short-Time Fourier Transform (STFT), which uses a fixed window size, the NSGT varies the window length $L$ inversely with frequency to maintain the Constant-Q property derived in Section 2.2.

### 3.1.1 Kernel-Based Spectral Analysis

The implementation in `core/nsgt_kernel.rs` utilizes a sparse kernel approach to perform this transform efficiently. For each log-frequency band $k$, a time-domain kernel $h_k$ is precomputed. This kernel combines a complex sinusoid at the band's center frequency $f_k$ with a periodic Hann window $w_k$ of length $L_k \approx Q \cdot f_s / f_k$.

$$ h_k[n] = w_k[n] \cdot e^{-j 2\pi f_k n / f_s} $$

These kernels are transformed into the frequency domain ($K_k[\nu]$) during initialization. To optimize performance, the system sparsifies these frequency kernels, storing only the bins with significant energy.

During runtime, the system performs a single FFT on the input audio buffer to obtain the spectrum $X[\nu]$. The complex coefficient $C_k$ for band $k$ is then computed via the inner product in the frequency domain:

$$ C_k = \frac{1}{N_{fft}} \sum_{\nu} X[\nu] \cdot K_k^*[\nu] $$

This "one FFT, many kernels" approach generates a high-resolution, logarithmically spaced spectrum without the cost of calculating a separate DFT for every band or using recursive filter banks. The current instrument runtime constructs a 96-bin-per-octave analysis space from 55 Hz to 8 kHz; `Log2Space` itself supports other positive, ordered bounds for experiments and tests.

### 3.1.2 Real-Time Temporal Smoothing

The instantaneous band powers $p_k = |C_k|^2$ exhibit high variance due to the stochastic nature of the audio input (especially with noise-based agents). To create a stable field for agents to sample, the `RtNsgtKernelLog2` struct wraps the NSGT with a temporal smoothing layer.

It implements a per-band leaky integrator (exponential smoothing). Crucially, the time constant $\tau$ is frequency-dependent. Low frequencies, which evolve slowly, are smoothed with a longer $\tau$, while high frequencies, which carry transient details, have a shorter $\tau$.

$$ y_k[t] = (1 - \alpha_k) \cdot p_k[t] + \alpha_k \cdot y_k[t-1] $$

where the smoothing factor $\alpha_k$ is derived from the frame interval $\Delta t$:

$$ \alpha_k = e^{-\Delta t / \tau(f_k)} $$

This models the "integration time" of the ear. It is only the first perceptual stage: the smoothed power is subsequently converted into a sparse subjective-intensity density before the Roughness and Harmonicity kernels read it.

### 3.1.3 From NSGT Power to Subjective Intensity

`SpectralFrontEnd` (`core/landscape_spectral.rs`) does not pass the smoothed NSGT power directly to the Landscape kernels. For each hop it:

1.  Converts per-bin power to power density using the Log2Space-aligned ERB cell widths.
2.  Extracts significant spectral peaks, preserving their integrated mass rather than treating every analysis bin as an independent partial.
3.  Applies an A-weighting **power** gain and the compressive exponent `loudness_exp` to each peak mass.
4.  Scatters the resulting masses back onto a sparse Log2Space-aligned subjective-intensity density and applies a second leaky normalization with time constant `analysis.tau_ms`.

The resulting `subjective_intensity` scan—not raw $|C_k|$ or raw NSGT power—is the common input to the Roughness and Harmonicity computations below. Its integral is retained separately as `loudness_mass`.

## 3.2 Roughness ($R$) Calculation: The Plomp-Levelt Model

Roughness is the sensation of "harshness" or "buzzing" caused by the interference of spectral components that fall within the same critical band but are not sufficiently close to be perceived as a single tone (beating). Conchordal implements a variation of the Plomp-Levelt model via convolution in the ERB domain.

### 3.2.1 The Interference Kernel

The core of the calculation is the Roughness Kernel, defined in `core/roughness_kernel.rs`. This kernel $K_{rough}(\Delta z)$ models the interference curve between two partials separated by $\Delta z$ ERB-rate units. The default Sethares core peaks near a quarter ERB-rate unit; the reference-normalization stimulus uses a separation of 0.25 ERB.

The implementation uses a Sethares difference-of-exponentials core, a small directional masking asymmetry, a center-suppression dip, and an optional neural Gaussian component. With $u=|\Delta z|/\kappa$:

$$ S(\Delta z) = G\,\max(0, e^{-bu} - e^{-cu})\,A(\Delta z) $$

$$ M(\Delta z) = \left(1-e^{-\Delta z^2/(2\sigma_{sup}^2)}\right)^p, \qquad
N(\Delta z)=e^{-\Delta z^2/(2\sigma_n^2)} $$

$$ K_{rough}(\Delta z)=(1-w_n)\,S(\Delta z)M(\Delta z)+w_nN(\Delta z) $$

Here $A(\Delta z)$ is the exponentially decaying positive/negative-side asymmetry controlled by `mix_tail` and `tau_erb`. The default $w_n=0$ selects the cochlear Sethares path. $M(0)=0$ suppresses self-roughness for a single pure tone; raising $w_n$ deliberately fills that dip with the neural component.

### 3.2.2 Convolutional Approach

Calculating roughness pairwise for all spectral bins ($N^2$ complexity) is computationally prohibitive for real-time applications. Conchordal solves this by treating the Roughness calculation as a linear convolution.

1.  **Mapping**: The Log2Space-aligned subjective-intensity density from Section 3.1.3 is mapped onto a linear ERB grid.
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

1.  **Downward Projection (Root Search)**: The current spectral envelope is "smeared" downward. In scatter form: every bin $i$ with energy adds evidence to the bins $i - \log_2(k)$ for integers $k \in \{1, 2, \dots, N\}$ (its candidate roots). The implementation uses the equivalent gather form—each bin collects evidence from the positions where its harmonics would lie:

    $$ Roots[i] = \sum_k A[i + \log_2(k)] \cdot w_k $$

    On the log-frequency grid, $\log_2(k)$ is a constant bin offset (non-integer offsets are interpolated). $w_k$ is a weighting factor that decays with harmonic index $k$ (e.g., $k^{-\rho}$), reflecting that lower harmonics imply their roots more strongly than higher ones. The result `Roots` describes the strength of the virtual pitch at every frequency.

2.  **Upward Projection (Harmonic Resonance)**: The system then projects the `Roots` spectrum back upwards. If a strong root exists at $f_r$, it implies stability for all its natural harmonics ($f_r, 2f_r, 3f_r \dots$).

    $$ H[i] = \sum_m Roots[i - \log_2(m)] \cdot w_m $$

**Emergent Tonal Stability**: Consider an environment with a single tone at 200 Hz.

*   **Step 1 (Down)**: It projects roots at 100 Hz ($f/2$), 66.6 Hz ($f/3$), 50 Hz ($f/4$), etc.
*   **Step 2 (Up)**: The 100 Hz root projects stability to 100, 200, 300, 400, 500... Hz.
    *   300 Hz is the Perfect 5th of the 100 Hz root.
    *   500 Hz is the Major 3rd of the 100 Hz root.

Thus, without any hardcoded knowledge of Western music theory, the system naturally generates stability peaks at the Major 3rd and Perfect 5th relationships, simply as a consequence of the physics of the harmonic series. An agent at 200 Hz creates a "gravity well" at 300 Hz and 500 Hz, inviting other agents to form a major triad.

## 3.4 Consonance: Integrating the Fields

With $R_{01}$ and $H_{01}$ in hand, Consonance is derived in two layers: a **Consonance Kernel** that fuses the two observables into a single fitness score, and a set of **representation transforms** that reshape that score for its different consumers in the Life Engine (Section 5).

**Layer 1 — Consonance Kernel (bilinear family):**

$$ C_{score} = a \cdot H_{01} + b \cdot R_{01} + c \cdot H_{01} R_{01} + d $$

Default coefficients: $a = 1.0$, $b = -1.35$, $c = 1.0$, $d = 0.0$. Because $b < 0$, roughness acts as a penalty; because $c > 0$, high harmonicity attenuates that penalty (the interaction term $c \cdot H_{01} R_{01}$ partially cancels $b \cdot R_{01}$ when $H_{01}$ is large). The bilinear family subsumes the earlier $\alpha H - wR$ formulation as the special case $c = 0$.

**Layer 2 — Representations:**

| Name | Formula | Range | Meaning |
| :--- | :--- | :--- | :--- |
| $C_{score}$ | $aH + bR + cHR + d$ | $(-\infty,+\infty)$ | raw fitness from the kernel |
| $C_{level01}$ | $\sigma(\beta(C_{score} - \theta))$ | $[0,1]$ | base sigmoid level; habituation drive and un-eroded view |
| $C_{density\_mass}$ | $\max(0,\;H_{01}(1 - \rho R_{01}))$ | $[0,+\infty)$ | base density mass before erosion |
| $C_{density\_pmf}$ | $\text{normalize}(C_{density\_mass}^{eff})$ | $[0,1],\;\Sigma=1$ | normalized global view retained in `Landscape` |
| $C_{energy}$ | $-C_{score}$ | $(-\infty,+\infty)$ | energy for minimization |
| $C_{score}^{eff}$ | $\theta+(C_{score}-\theta)(1-h)$ | $(-\infty,+\infty)$ | habituation-eroded score read by movement and prediction |
| $C_{level01}^{eff}$ | $\sigma(\beta(C_{score}^{eff}-\theta))$ | $[0,1]$ | habituation-eroded level read by behavior and the listener |
| $C_{density\_mass}^{eff}$ | $(1-h)C_{density\_mass}$ | $[0,+\infty)$ | habituation-eroded mass read by density placement |

where $\sigma(x) = 1/(1+e^{-x})$, $\beta$ controls sigmoid steepness (default 2.0), and $\theta$ is the sigmoid threshold (default 0.0). The density mass uses a separate $\rho$-kernel with coefficients $a{=}1, b{=}0, c{=}{-}\rho, d{=}0$; $\rho$ (`consonance_density_roughness_gain`, default 1.0) controls how strongly roughness suppresses density placement.

Spawn does **not** sample the global $C_{density\_pmf}$. A `SpawnStrategy::Field` first slices the requested frequency range, derives target-specific local mass from the effective views, masks occupied bins, and normalizes only that local vector. A zero-mass range falls back to an unoccupied-uniform distribution and then to full-range uniform if every bin is occupied. Peak placement reads $C_{score}^{eff}$ deterministically. The global PMF remains a normalized Landscape representation but is currently exercised only by core tests.

There is also a distinct, per-Voice adaptation mechanism. `AdaptationContext` tracks fast boredom and slow familiarity over the shared **fundamental-occupancy** field and adds a candidate-specific score adjustment during pitch selection. It coexists with the Landscape-level habituation below: adaptation is agent-relative memory; habituation is a shared perceptual erosion field.

## 3.5 Landscape-Level Habituation

Consonance is not treated as an inexhaustible place. When `[psychoacoustics.habituation]` is enabled, `HabituationField` maintains a per-bin state $h_i\in[0,1]$ for both the ecology's habitat analysis and the `ListenerTwin`'s presentation analysis. The raw drive combines the base consonance level $L_i$ with a fresh common-root projection $P_i$:

$$ d_i^{raw}=L_iP_i, \qquad d_i=\frac{d_i^{raw}}{d_i^{raw}+r_{ref}} $$

The state relaxes asymmetrically toward that drive:

$$ h_i(t+\Delta t)=a h_i(t)+(1-a)d_i, \qquad a=e^{-\Delta t/\tau} $$

with $\tau=\tau_e$ while the drive is rising and $\tau=\tau_r$ while it is falling. `satiation_sec` and `recovery_sec` are defined as the times to reach 90% erosion and 90% recovery respectively, so the implementation uses $\tau_e=T_s/\ln 10$ and $\tau_r=T_r/\ln 10$. The effective score relaxes toward the neutral sigmoid threshold $\theta$, while density mass is eroded multiplicatively, as shown in the table above.

The default is `enabled=false`, with `satiation_sec=5.0`, `recovery_sec=8.0`, and `ref_drive=0.25`. Disabled habituation is an exact identity: every effective view equals its base view bit-for-bit. When enabled, movement, prediction, metabolism, respawn, density placement, and the listener read effective views; the UI terrain and diagnostics intentionally retain the un-eroded base views.

# 4. The Temporal Axis: The Emergent Meter

Music cognition distinguishes three layers of musical time. **Rhythm** is the surface: the actual pattern of onsets as they occur. **Pulse** (the tactus) is the perceived regular beat a listener taps along to—already an inference, since the surface rarely contains it explicitly. **Meter** is the hierarchical organization of that pulse: nested periodicities (subdivision, beat, measure) with alternating strong and weak positions. Crucially, meter in this sense is a *percept*, not a notation. A time signature is an instruction; meter is what a listener's brain constructs from the sound—and constructs even from sound that was never notated.

This percept has well-studied dynamics. It takes a few cycles of evidence to establish (*beat induction*); once established it persists through syncopation, gaps, and silence, with events heard *against* the induced grid rather than destroying it; and it is plastic, re-locking when the input tempo drifts. Neural resonance theories (Large and colleagues) explain these properties mechanistically: populations of neural oscillators entrain to acoustic onsets, and the self-sustaining dynamics of the oscillation—not the stimulus itself—carry the pulse across interruptions.

Conchordal adopts this account literally. It eschews the concept of a master clock or metronome; time is structured by an **emergent meter**—a coupled limit-cycle oscillator network implementing the neural resonance model—that *listens* to the ecosystem's own onsets and forms a metrical percept from them. Voices, in turn, entrain their onset timing to this perceived pulse with per-voice coupling strengths (Section 5.4). Rhythm is therefore a closed perception–action loop: nothing schedules a beat; a beat condenses out of the population's behavior and then attracts it. This is the temporal mirror of the harmonic landscape (Section 3): just as consonance is not imposed but computed from a model of the cochlea, meter is not imposed but computed from a model of beat perception.

## 4.1 The Meter Core: A Forced Limit-Cycle Oscillator

The `MeterNetwork` (`core/meter.rs`) maintains a beat oscillator as a forced Hopf normal form—the canonical equations of a system at the threshold of self-sustained oscillation—integrated in polar coordinates $(r, \varphi)$:

$$ \dot{r} = \alpha r + \beta r^3 + F_a\, s(t) \cos\varphi $$
$$ \dot{\varphi} = \omega - F_p \frac{s(t)}{r} \sin\varphi $$

With $\alpha > 0$ and $\beta < 0$ the unforced system has a stable limit cycle of radius $\sqrt{-\alpha/\beta} = 1$: the beat is **self-sustaining** and coasts through gaps in the input (the persistence regime of beat induction). The drive $s(t)$ is a rectified onset signal combining spectral flux—the frame-to-frame increase in spectral energy, a generic onset detector—extracted by the `DorsalStream` (`core/stream/dorsal.rs`, a 3-band crossover flux detector) with the population's own phonation onset strengths—a low-latency auditory–motor reinforcement path.

The oscillator's natural frequency is plastic. A Hebbian learning rule shifts $\omega$ to reduce the phase error to the stimulus:

$$ \dot{\omega} = -\eta\, s(t) \sin\varphi $$

Random input (a renewal process: independently drawn inter-onset intervals, with no preferred phase) averages to zero net shift, so the beat does not chase noise; periodic input pulls $\omega$ toward the stimulus rate within the beat band (0.5–4 Hz). On top of the beat, the network tracks an entrained **subdivision** and a slow **measure** subharmonic (ratios 2, 3, 4 against the unwrapped beat count), giving a three-level metrical state (`MeterState`).

**Perception vs production.** The runtime maintains two meter instances. The *production meter* runs in the worker thread on the habitat bus and drives all voice behavior. The *perception meter* lives inside the `ListenerTwin` (`listener_twin/`), which analyzes the presentation audio exactly as an audience member would hear it; its beat confidence feeds the UI, the headless report, and the Direct Cognitive Coupling (DCC) pressure path.

## 4.2 Beat Confidence: Phase-Locking Value

The meter does not merely track a beat—it knows *how much* beat there is. Each detected onset deposits a unit phasor (a unit complex vector at the current beat phase) into leaky accumulators; the length of their resultant is a phase-locking value (PLV): 1.0 when all onsets land at the same phase, near 0 when phases are scattered. Confidence is the PLV gated by a *presence* term that requires roughly four accumulated onsets of evidence before it saturates—matching the psychological observation that beat induction needs a few cycles—and decays in silence. Scattered onset phases keep the resultant low, so confidence cannot be fabricated by density alone.

## 4.3 From Meter to Modulation: NeuralRhythms

`NeuralRhythms::from_meter_state` (`core/modulation.rs`) projects the metrical state onto the modulation bands consumed by voice behavior:

*   **Delta** ← the beat (tactus): phase and tempo of the pulse. Its phase drives `env_open`, a cosine gate that sharpens the articulation envelope toward the downbeat as confidence rises and leaves it open when the beat is uncertain.
*   **Theta** ← the subdivision: the note-rate band that the breath oscillator (Section 5.5) locks to.
*   Band precision (`alpha`) equals beat confidence; prediction error (`beta`) is its complement.

## 4.4 Composer Priors: Shaping the Temporal Terrain

The director can bend the terrain the pulse forms on—never schedule it—via `MeterShaping` (set from Rhai):

*   `meter_stability(v)` — attractor depth in $[0,1]$. Scales the entrainment forcing and frequency learning, and lowers the presence threshold (a top-down prior that commits with less evidence). Because forcing acts only in the stimulus direction, random input still cancels: stability cannot fabricate a beat.
*   `temporal_basin(min_hz, max_hz)` — a tempo prior. The beat frequency is seeded at the basin's center, gently pulled toward it (a weak restoring rate), and its Hebbian learning is confined to the band. The basin shapes *where* a pulse settles; onset entrainment within the basin still does the work.

These are the temporal analogue of the consonance-field operations: soft priors on an emergent process, in keeping with the Manifesto's rejection of imposed grids.

# 5. The Life Engine: Agents and Autonomy

The "Life Engine" is the agent-based simulation layer that inhabits the two terrains established above—the consonance landscape (Section 3) and the emergent meter (Section 4). It manages the population of "Voices," handling their lifecycle, sensory processing, and actuation (audio synthesis).

## 5.1 The Voice Architecture

The `Voice` struct (`life/voice.rs`) is the atomic unit of the ecosystem. It is composed of several components:

*   An `AnySoundBody` actuator (synthesis backend).
*   An `ArticulationWrapper` (wrapping an `ArticulationCore`).
*   A `PitchController` (wrapping a `PitchCore`).
*   A `PhonationEngine` that manages note-level timing and command queuing.
*   An optional `ToneAdsr` envelope for attack-decay-sustain-release shaping.
*   Lifecycle and metabolism tracking (energy, age, perceptual context).

The Voice itself acts as an integration layer, managing the control-plane signals that coordinate the components without coupling them directly.

### 5.1.1 The SoundBody (Actuator)

The `BodyMethod` enum defines three synthesis body types, each projecting a distinct spectral footprint onto the Landscape:

*   **`Sine`**: A pure sine tone via a single oscillator. Minimal spectral interference; useful as anchors and calibration probes.
*   **`Harmonic`**: A complex tone with a `TimbreGenotype` governing its partial structure. Parameters include:
    *   `stiffness`: Inharmonicity coefficient (stretching the partial series).
    *   `brightness`: Spectral slope (decay of higher partials).
    *   `comb`: Even harmonic attenuation.
    *   `damping`: Energy-dependent thinning of upper partials. A slow spectral-energy envelope tracks the excitation (fast attack, ~0.5 s release, floored so ringing notes stay audible) and darkens the tone as drive fades.
    *   `vibrato_rate` / `vibrato_depth`: LFO-based pitch modulation.
    *   `jitter`: 1/f pink noise FM strength for organic fluctuation.
    *   `unison`: Detuned copy amount for chorus-like thickening.
    *   `mode`: Harmonic (integer multiples) vs. Metallic (non-integer ratios).
*   **`Modal`**: Resonator-based synthesis via `ModalEngine`, supporting arbitrary mode frequency ratios and decay times. Mode patterns can be specified as harmonic, odd harmonics, power-law, stiff string, or custom ratios.

Sound generation is dispatched through the `AnyBackend` enum:

*   **`Oscillator(OscillatorBank)`**: A struct-of-arrays layout for cache-efficient additive synthesis. Handles `Sine` and `Harmonic` bodies. Pitch refresh occurs every 64 samples; motion/vibrato refresh every 8 samples.
*   **`Resonator(ModalEngine)`**: A Damped Modified Coupled Form resonator bank. Handles `Modal` bodies. Mode coefficients are rebuilt every 64 samples on pitch change.

The `HarmonicBody` allows for the evolution of timbre. An agent with high stiffness might find survival difficult in a purely harmonic landscape, forcing it to seek out unique "spectral niches" where its inharmonic partials do not clash with the population.

### 5.1.2 The Core Stack

Behavior is split into three focused cores plus the `PhonationEngine`, each defined in a separate file:

*   **ArticulationCore (When/Gate)** — `life/articulation_core.rs`: Manages gating and envelope dynamics. Three variants exist:
    *   `KuramotoCore`: Coupled "breath" oscillator with a normalized energy/vitality model, rhythm coupling modes (`TemporalOnly`, `TemporalTimesVitality`), rhythm reward (metabolism bonus for phase match), and autonomous attack capability. It entrains its envelope to the meter-derived rhythm bands (Section 4.3). Energy is fixed to the 0–1 domain; the stored composing parameters are nominal `endurance_sec` and optional `recovery_sec`, while runtime rates are derived once at voice construction.
    *   `SequencedCore`: Fixed-duration gate patterns.
    *   `DroneCore`: Sustained output with optional sway modulation.

*   **PitchCore (Where)** — `life/pitch_core.rs`: Proposes the next target in log-frequency space. Two implementations:
    *   `PitchHillClimbPitchCore`: Local search over the consonance terrain. Parameters: `neighbor_step_log2`, `tessitura_gravity`, `landscape_weight`, `move_cost_coeff`, `move_cost_exp`, and a single search `temperature` (0 = greedy settling; higher accepts downhill moves via a Metropolis rule). Occupancy/crowding: `crowding_strength`, `crowding_sigma_cents`. Leave-self-out analysis supports `ApproxHarmonics` and `ExactScan` modes.
    *   `PitchPeakSamplerCore`: Probabilistic peak sampling with `window_cents`, `top_k`, `sigma_cents`, `random_candidates`, and the same search `temperature` (softmax over candidates).

*   **PhonationEngine** — `life/phonation_engine.rs`: Manages note-level command scheduling. Issues `ToneCmd` (On, Off, Update) to the `ScheduleRenderer`. Uses a gate grid (`ThetaGrid`) for onset bookkeeping and note-off placement. Configuration is via `PhonationSpec`:
    *   **When**: `Once` (single trigger), `Pulse { rate_hz, sync, social }` (repeated triggers on the adaptive gate clock), or `Coupled(CoupledTimingSpec)` — the rhythm-family continuum in which a per-voice phase oscillator entrains to the shared emergent meter (Section 5.4).
    *   **Duration**: `WhileAlive`, `Gates(n)`, `Field { hold_min_theta, hold_max_theta, curve_k, curve_x0, drop_gain }`.

### 5.1.3 The Sound Pipeline

Audio rendering is handled by `ScheduleRenderer` (`life/schedule_renderer.rs`), which maintains a `HashMap<ToneKey, RoutedTone>` of active tones, each routed to one of two buses (Section 6.2): the **habitat bus** (analyzed as the landscape's environment) and the **presentation bus** (what the audience hears).

The `Tone` struct (`life/sound/tone.rs`) combines:

*   A backend (`AnyBackend`: `OscillatorBank` or `ModalEngine`).
*   An optional `RenderModulator` for articulation envelope shaping.
*   An ADSR envelope: linear attack ramp, exponential decay to sustain level, constant sustain, linear release ramp.
*   Smoothed pitch and amplitude transitions with configurable time constants.
*   Continuous drive for sustained excitation.

The processing flow proceeds as follows: the `PhonationEngine` emits `ToneCmd` commands; the `ScheduleRenderer` creates, updates, or releases `Tone` instances accordingly; each `Tone` renders through its backend with ADSR shaping; the results are mixed per bus.

### 5.1.4 Control-Plane Signals: Planned and Error

The Voice coordinates its cores through two orthogonal signals rather than direct coupling:

*   **Planned**: The PitchCore proposes a target (`TargetProposal`), and the Voice maintains the "planned" state—next target frequency, expected jump distance, and salience. This represents the agent's *intention*.
*   **Error**: The Voice computes the discrepancy between the SoundBody's current pitch and the planned target (signed cents, absolute cents). This represents the *result* of prior actions and is available for observation or future extensions (e.g., adaptive articulation). Importantly, the PitchCore does not read the error signal—search remains decoupled from feedback.

This separation keeps each core focused: PitchCore explores the landscape, ArticulationCore shapes the envelope, and the Voice orchestrates timing and state transitions.

## 5.2 Lifecycle and Metabolism

Agents in Conchordal are governed by energy dynamics modeled on biological metabolism. The `LifecycleConfig` defines two modes of existence:

*   **Decay**: A transient articulation whose envelope decays with a configured half-life. This models plucks and percussion without exposing a redundant energy scale.
*   **Sustain**: A living articulation with energy normalized to $[0,1]$.
    *   **Endurance**: `endurance_sec` is the nominal energy-depletion time at zero field fit, with no attacks or recovery. The derived basal rate is $1/(T_e(1+p_d))$, where $T_e$ is endurance and $p_d$ is `dissonance_penalty`; this keeps zero-fit endurance fixed while allowing good fit to extend life.
    *   **Continuous recovery**: Optional `recovery_sec` is the time to refill energy from 0 to 1 at a viability signal of 1, with drain and attacks disabled. The derived rate is $1/T_r$.
    *   **Attack economy**: `attack_cost_fraction` and `attack_recharge_fraction` are fractions of normalized capacity applied per full-strength attack; consonance scales the recharge term.
    *   **Rhythm Reward**: An optional `MetabolismRhythmReward` multiplies the recharge contribution for phase-matched attacks, configured via `rho_t` and the `AttackPhaseMatch` metric.

Energy depletion disables retriggering and starts the envelope tail. Reports therefore distinguish configured endurance, energy-depletion time, and observable lifetime.

This mechanic creates a Darwinian pressure: **Survival of the Consonant**. Agents in dissonant (low $C_{level01}^{eff}$) regions starve—energy depletes, amplitude fades, and they die. Agents in consonant (high $C_{level01}^{eff}$) regions thrive—they maintain or gain energy, allowing them to sing louder and live longer. With Landscape habituation enabled, a formerly supportive region can lose effective value under sustained activity and recover after withdrawal, so survival depends on both acoustic fit and recent perceptual history.

## 5.3 Pitch Retargeting Logic

Agents are not static; they move through frequency space to improve their fitness. The execution layer applies a retarget gate (a zero-crossing of the meter-derived theta band, Section 4.3, plus an integration window) and then asks the PitchCore to propose the next target. Candidate evaluation reads the habituation-eroded score; exact leave-self-out recomputes the raw self-subtracted score and then applies the same local erosion before comparison.

### 5.3.1 Pitch Application Modes

Two modes govern how a new pitch target is applied:

*   **GateSnap**: Discrete hop at note boundaries. The pitch snaps to the new target at note onset, so each note sounds a single stable frequency. Ordering matters: on the sample where the snap occurs, the pitch is updated *before* consonance is evaluated, ensuring the Landscape score reflects the agent's actual sounding frequency.
*   **Glide**: Smooth continuous pitch transition with a configurable time constant $\tau$. The SoundBody interpolates exponentially toward the target frequency, producing portamento effects. Suited for drone-like Voices or slow melodic movement.

For `seek_consonance()` voices the mode is resolved automatically from the phonation timing unless the script chooses explicitly: sustained voices (`once()`) glide, while re-attacking voices (pulse or coupled timing) snap at onsets.

### 5.3.2 Crowding and Leave-Self-Out

The crowding system prevents agents from collapsing to identical frequencies. Both crowding (mutual repulsion) and pitch adaptation read one shared occupancy field: a Gaussian penalty centered on each occupied fundamental (raw f0), with width set by `crowding_sigma_cents` (default 60) and strength by `crowding_strength`. A pairwise split bias further prevents frequency degeneracy. (An octave-equivalence/chroma term — `octave_avoidance` — was prototyped and removed: as a flat penalty decoupled from the consonance potential it had no stable operating point, only a cliff from octave fusion to forced tension. See the placement/voice-movement design notes; a field-derived avoidance is deferred to the occupancy-unified-drive redesign.)

When evaluating landscape fitness, an agent can subtract its own spectral contribution via leave-self-out analysis. Two modes are supported:

*   **`ApproxHarmonics`**: Fast approximation using ~24 cent Gaussian subtraction. When the body defines explicit mode ratios, the voice subtracts its *own* partial set; bodies without ratios fall back to the integer harmonic series.
*   **`ExactScan`**: Full ERB grid scan for precise spectral subtraction.

These timing-sensitive transitions and crowding evaluations are guarded by regression tests to prevent subtle breakage.

## 5.4 Onset Timing: The Coupling Continuum

Onset timing for repeated phonation is generated by the `CouplingClock` (`life/phonation_engine.rs`): a per-voice phase oscillator that emits an onset at every integer crossing of its phase. Its effective rate blends an intrinsic renewal rate with the shared beat:

$$ f_{eff} = (1 - \ell)\, f_{int} + \ell\, f_{beat}, \qquad \ell = \kappa \cdot c $$

where $\kappa$ is the voice's coupling strength (`entrainment`, 0–1) and $c$ is the meter's beat confidence (Section 4.2)—so a voice can only lock as strongly as the meter is believed. A phase pull drags the oscillator's crossings toward the beat phase (optionally offset by `microtiming`):

$$ \dot{\phi} = f_{eff} \left(1 + \ell K \,\mathrm{err}(\phi_{beat} - \phi)\right) $$

where $\mathrm{err}(\cdot)$ is the wrapped phase error in cycles ($[-0.5, 0.5]$) and $K$ a fixed pull gain chosen so the rate factor stays positive—the phase always advances, only faster or slower.

This single mechanism spans the rhythm-family continuum selected by the Rhai presets:

*   $\kappa \to 0$ (**flow**): a free renewal process. The inter-onset intervals are drawn from a clustered renewal distribution (`flow_depth` controls cluster/gap probability), producing rain-like non-metric texture.
*   medium $\kappa$ (**entrained**): the voice locks loosely, and only as the meter gains confidence—synchronization *emerges over time*.
*   $\kappa \to 1$ (**metric**): the shared beat is a deep attractor; the voice reads as a stable pulse.

Each onset carries a strength set by the voice's `rhythm_role`—beat 1.0, subdivision 0.7, accent 2.5, texture 0.85—and these strengths feed back into the production meter's drive. A recurring accent therefore drives the meter harder, allowing a downbeat (and eventually a measure) to be *induced* by the population rather than declared. There is no externally imposed grid anywhere in this loop.

## 5.5 The Breath Oscillator: Kuramoto Articulation

Independent of onset scheduling, the `KuramotoCore` ArticulationCore entrains each voice's *envelope* (its breath) to the meter-derived theta band, using a mean-field Kuramoto phase step:

$$ K_{eff} = \omega_{target} \cdot K_{global} \cdot s_\theta \cdot |\theta_{mag}| \cdot \theta_\alpha \cdot g_{env} \cdot a_{env} $$

where $\omega_{target}$ is the theta band's angular frequency, $K_{global}$ the scene-wide coupling gain (`set_global_coupling`), $s_\theta$ the voice's theta sensitivity, $|\theta_{mag}|$ and $\theta_\alpha$ derive from beat confidence, and $g_{env}$, $a_{env}$ are the envelope gate and amplitude. Helper functions `kuramoto_k_eff()` and `kuramoto_phase_step()` are exposed for external simulation (paper experiments). The energy/vitality subsystems interact with this coupling:

*   **Rhythm Coupling Modes**: `TemporalOnly` (pure phase coupling) or `TemporalTimesVitality { lambda_v, v_floor }` (healthy agents synchronize more strongly).
*   **Rhythm Reward**: an optional `MetabolismRhythmReward` (`rho_t`, `AttackPhaseMatch`) grants a metabolic bonus for phase-matched onsets, linking rhythmic conformity to survival.
*   **Autonomous Attack**: self-triggered attacks when envelope-gate and confidence thresholds align.

# 6. System Architecture and Implementation Details

Conchordal is implemented in Rust to satisfy the stringent requirements of real-time audio (latency < 10ms) alongside heavy numerical analysis (NSGT/Convolution). The architecture uses a concurrent, lock-free design pattern.

## 6.1 Threading Model

The application creates four primary thread contexts, plus the GUI event loop:

1.  **Audio Thread (Real-Time Priority)**:
    *   Managed by `cpal` in `audio/output.rs`.
    *   **Constraint**: Must never block. No Mutexes, no memory allocation.
    *   **Responsibility**: Pops mono samples from a lock-free ring buffer and copies them to all output channels. A `Limiter` (soft-clip or peak-limiter) is applied in-place on the interleaved output.

2.  **Analysis Thread (Background Priority)**:
    *   Defined in `core/analysis_worker.rs`, running `AnalysisStream` from `core/stream/analysis.rs`.
    *   **Responsibility**: Receives habitat-bus hops (time-domain chunks), runs the NSGT to produce a log2 power spectrum, then computes *both* the Harmonicity field (Sibling Projection) and the Roughness field (ERB-domain convolution) in a single pipeline.
    *   **Update Cycle**: When analysis is complete, it sends the updated Landscape snapshot back to the worker thread via a bounded SPSC channel.

3.  **Listener-Analysis Thread**:
    *   Runs the `ListenerTwin` perception pipeline on presentation-bus hops.
    *   **Responsibility**: Models what an audience member perceives—including its own habituation field and the perception meter's beat confidence—for the UI, the headless report, and the DCC pressure coupler.

4.  **Worker Thread (Simulation Loop)**:
    *   Named `"worker"` in `app.rs`.
    *   **Responsibility**: Runs the main simulation loop. Each iteration: merges analysis results into the current Landscape, dispatches Conductor events, advances the Population (pitch retargeting, articulation, metabolism), renders audio via `ScheduleRenderer` (which processes `PhonationBatch` vectors of `ToneCmd` and maintains the `Tone` pool), drives the production `MeterNetwork` from habitat flux and the population's own onsets, and pushes mono samples into the ring buffer for the audio thread.

5.  **App/GUI Thread (Main)**:
    *   Runs the `eframe`/`egui` visualizer.
    *   **Responsibility**: Handles user input, visualizing the Landscape (`ui/plots.rs`), and displaying simulation metadata. It receives `UiFrame` snapshots from the worker thread via a bounded channel.

## 6.2 Data Flow

To maintain data consistency without locking the audio thread, Conchordal uses a multi-channel update strategy for the Landscape. Rendered audio is split across two buses: the **habitat bus** (the environment the ecosystem senses) and the **presentation bus** (what the audience hears). A drone can be routed to the habitat bus only—shaping the landscape without being presented.

1.  The **Worker Thread** renders audio per bus and sends each habitat hop to the **Analysis Thread**; presentation hops go to the **Listener-Analysis Thread**.
2.  The **Analysis Thread** runs the NSGT + subjective-intensity + Roughness + Harmonicity pipeline and sends the resulting raw `Landscape` snapshot back.
3.  The **Worker Thread** merges the analysis result into the current `LandscapeFrame`, recomputes the base Consonance representations, advances the ecology's `HabituationField`, and writes the effective views. A separate habituation state is advanced on listener-analysis results.
4.  The **Worker Thread** drives the production `MeterNetwork` with the habitat onset flux (`DorsalStream`) combined with the population's own phonation onset strengths; the resulting `MeterState` is projected into `landscape.rhythm` via `NeuralRhythms::from_meter_state`.
5.  The `Community` evaluates the effective Landscape for pitch selection, metabolism, spawn, respawn, and Voice lifecycle.
6.  When DCC coupling is enabled, `ListenerTwin` tension and resolvability produce a bounded temperature bonus that feeds the Voices' pitch search. The default coupling strength is zero, so this path is behaviorally inert unless explicitly enabled.
7.  The `PhonationEngine` emits `ToneCmd` batches; the `ScheduleRenderer` creates, updates, or releases `Tone` instances accordingly and renders audio through ADSR-shaped backends.
8.  Rendered presentation audio is pushed into a lock-free ring buffer consumed by the **Audio Thread**.

This decoupled architecture ensures that the audio thread always sees a consistent stream of samples, even if the analysis thread lags slightly behind real-time. The analysis thread processes all hops in-order to maintain NSGT time continuity.

## 6.3 The Conductor: Scripting with Rhai

The Conductor module acts as the interface between the human artist and the ecosystem. It embeds the [Rhai](https://rhai.rs/) scripting language, exposing a tiered API for controlling the simulation.

The API follows four explicit lifetimes: a **PopulationSpec** is a reusable pre-placement definition, a **Population** is the stable population identity created by `place()`, a **Voice** is one living member, and the runtime **Community** aggregates all Populations sharing the Landscape. The authoritative, always-current reference is the Script Reference book (`docs/rhai_book`, published under `/docs/rhai/`); this section summarizes the conceptual tiers only. `Species` remains reserved for a future hereditary/speciation model rather than being used as a synonym for a configuration object.

### 6.3.1 PopulationSpec Configuration

A PopulationSpec begins with a preset and is refined through method chaining. It contains founder Voice defaults together with population-level lifecycle, viability, and respawn policy:

**Presets**: `sine()`, `harmonic()`, `saw()`, `square()`, `noise()`, `modal()`. `variant(parent)` clones an existing PopulationSpec for modification.

**Body**: `amp(v)`, `freq(v)`, `brightness(v)`, `spread(v)`, `unison(n)`, `modes(pattern)`, `adsr(a,d,s,r)`, `send(bus)` (habitat/presentation routing).

**Pitch**: `anchor()` (hold the voice at its pitch; implied by `freq(hz)`), `seek_consonance()` (climb the consonance terrain; the apply mode is then resolved automatically—sustained voices glide, re-attacking voices snap at onsets—unless overridden via `pitch_apply_mode("gate_snap"|"glide")`), `pitch_core("hill_climb"|"peak_sampler")`, `glide(v)`, `landscape_weight(v)`, `neighbor_step_cents(v)`, `tessitura_gravity(v)`, `temperature(v)` (one search-stochasticity knob shared by both cores; 0 settles greedily — this *is* the movement-tension knob: a hot search keeps a voice restless and straying off consonance, cooling it resolves), `move_cost(v)`, `proposal_interval(sec)`, `global_peaks(n)`, `ratio_candidates(n)`, plus peak-sampler knobs (`window_cents`, `top_k`, `sigma_cents`, `random_candidates`).

**Crowding**: `avoid_neighbors(strength)` (default sigma) / `avoid_neighbors(strength, sigma_cents)`, `crowding_target(same, other)`, `leave_self_out(bool)`, `leave_self_out_mode("approx"|"exact")`, `leave_self_out_harmonics(n)`.

**Brain/Phonation**: `brain("entrain"|"seq"|"drone")`, `sustain()`, `repeat()`, `once()`, `pulse(rate)`, `pulse_lock(depth)`, `social(coupling)`; duration via `while_alive()`, `cycles(n)`, `adaptive_duration()`, `duration_range(min,max)`, `duration_curve(k,x0)`, `shorten_on_drop(gain)`.

**Rhythm (the coupling continuum, Section 5.4)**: presets `metric()`, `entrained()`, `flow()` select a region of the continuum—no Hz argument, since tempo belongs to the director's `temporal_basin`. Fine control: `entrainment(v)` (lock strength 0–1), `rhythm_role("beat"|"subdivision"|"accent"|"texture")`, `microtiming(v)`. Breath-level coupling: `rhythm_freq(v)`, `rhythm_coupling_vitality(lambda_v, v_floor)`, `rhythm_reward(rho_t, "attack_phase_match")`.

**Lifecycle/Viability**: `endurance(sec)`, optional `recovery(sec)`, `attack_cost_fraction(v)`, `attack_recharge_fraction(v)`, `consonance_viability(low, high)`, `dissonance_penalty(v)`, and `phonate_when_viable()` (withhold the first onset until the viability window opens).

**Respawn**: `respawn_random()`, `respawn_hereditary(sigma_oct)`, `respawn_consonance()`, `respawn_capacity(n)` (maximum living membership; founder count by default and never lower than it), `respawn_settle(placement)`, `respawn_min_c_level(v)`, `respawn_background_death_rate(v)`.

### 6.3.2 Mode Patterns

Modal synthesis mode patterns are specified via constructor functions with optional modifiers:

*   `harmonic_modes()`, `odd_modes()`, `power_modes(beta)`, `stiff_string_modes(stiffness)`, `custom_modes(ratios)`, `modal_table(name)`, `landscape_density_modes()`, `landscape_peaks_modes()`.

Modifiers: `.count(n)`, `.range(min, max)`, `.spacing(d)`, `.gamma(g)`, `.jitter(cents)`, `.seed(s)`.

### 6.3.3 Placements

Placements determine the founder Voices' initial frequency allocation when a PopulationSpec enters the ecosystem. A field-relative placement names a target — `consonance`, `dissonance`, `edge` (the consonance/dissonance boundary), `gap` (low-intensity registers) — realized as a density cloud by default (`.density()`) or a deterministic extremum with `.peak()`; `consonance(root)` takes a harmonic window, the others an absolute `(min, max)` range. Field-agnostic placements are `random(min, max)` (log-uniform) and the geometric `at(freq)` / `line(start, end)`. Modifiers: `.count(n)`, `.range(min_mul, max_mul)`, `.spacing(d)` (minimum ERB distance), and Consonance-only `.tension(degree)`, which targets an in-range field-score step below the strongest peak.

### 6.3.4 Population Placement and Live Control

*   `place(population_spec, placement)`: Immediately schedules founder Voices at the current cursor and returns their stable Population.
*   `release(population)`: Terminally closes the Population and marks its current members for fade-out; later patches are ignored.

`place()` is the sole definition/runtime boundary; there is no public draft Population. Initial body, behavior, lifecycle, and respawn methods exist only on PopulationSpec. Population exposes only live patches—such as pitch, amplitude, and timbre updates—and release. A Population retains its `population_id` while member `voice_id` and generation values change through death and respawn.

### 6.3.5 Control Flow

*   `wait(sec)`: Emits pending live patches, then advances the timeline cursor.
*   `flush()`: Emits pending live patches without advancing the timeline.
*   `seed(n)`: Sets the random seed for reproducible runs.
*   `section(name, callback)`: Marks a named scene boundary; Populations created within the callback are automatically released when the section ends.
*   `play(callback)`: Executes a scoped block—Populations created inside are released on exit.
*   `parallel([callbacks])`: Runs multiple blocks concurrently (timeline branches), advancing the cursor to the latest endpoint.

### 6.3.6 Director Operations

Scene-global terrain shaping, on both axes:

*   **Harmonic terrain**: `set_roughness_k(v)`, `set_pitch_objective("consonance"|"dissonance")`.
*   **Temporal terrain**: `meter_stability(v)`, `temporal_basin(min_hz, max_hz)` (Section 4.4).
*   **Interaction**: `set_global_coupling(v)` scales agent interaction strength.

**Scenario Parsing**: Scenarios are loaded from `.rhai` files. This separation allows users to compose the "Macro-Structure" (the narrative arc, the changing laws of physics) while the "Micro-Structure" (the specific notes and rhythms) emerges from the agents' adaptation to those changes.

# 7. Case Studies: Analysis of Emergent Behavior

The following examples, derived from the `samples/` directory, illustrate how specific parameter configurations lead to complex musical behaviors.

## 7.1 Case Study: The Rhythm Family Continuum (`samples/07_heartbeat.rhai`, `08_murmuration.rhai`, `09_rain.rhai`)

Three sibling samples demonstrate that one coupling mechanism (Section 5.4) spans qualitatively different temporalities:

1.  **Metric** (`07_heartbeat.rhai`): voices declare `metric()` with an accent role on the downbeat voice. Their onsets drive the shared production meter; the meter's confidence rises; high coupling pulls every onset into the now-deep attractor. A legible pulse appears—yet there is no external master clock, only per-Voice coupling clocks and a `temporal_basin` telling the terrain *where* a tempo may settle.
2.  **Entrained** (`08_murmuration.rhai`): medium coupling plus vitality coupling and attack reward. Synchronization is not immediate; it *emerges* over tens of seconds as confidence accumulates, and degrades if the colony weakens—rhythmic coherence is tied to ecological health.
3.  **Flow** (`09_rain.rhai`): near-zero coupling with high `flow_depth`. Onsets follow a clustered renewal process—rain on a roof—non-metric by construction, while pitch behavior still rides the consonance field.

## 7.2 Case Study: Drift and Flow (`samples/research/drift_flow.rhai`)

This file is retained as an early, compact historical fixture; it no longer validates autonomous hop-based drift.

1.  **Explicit patch**: A sustained anchor enters at C2 (65.41 Hz), while a second sustained Voice enters at C#3 (138.59 Hz). The script then patches that Voice directly to 220 Hz with `population.freq(220.0)` and releases it.
2.  **Field query**: Five one-Voice swarms are placed in sequence with `consonance(130.0).peak().range(1.0, 4.0)`, demonstrating repeated deterministic queries against the live field.
3.  **Terrain change**: The anchor is finally patched to F2 (87.31 Hz), changing the field after those placements.

The fixture therefore demonstrates live Population patches plus repeated field-relative placement. Autonomous settling is demonstrated by `samples/05_settling.rhai`; the old claim that `drift_flow.rhai` itself produces boredom-driven, endless hopping was stale and has been removed.

## 7.3 Case Study: Emergence and Resolution (`samples/12_emergence_and_resolution.rhai`)

The closing integration sample combines the full stack as a single directed arc. The scenario controls a temperature arc—heating the colony's search so it grows restless and strays off consonance, then cooling it to settle (harmonic tension as search temperature)—plus a register transposition, while everything else emerges: a metric heartbeat (an accent-role voice driving a deep beat attractor), a living colony that locks to the *same* emergent beat while climbing toward consonance, and a non-metric flow shimmer appearing only at the tension peak. The colony's survival through the dissonant peak (consonance-gated viability plus consonance-biased respawn) demonstrates resolution enacted by the ecosystem rather than scheduled note by note.

# 8. Conclusion

Conchordal establishes a foundation for Bio-Mimetic Computational Audio. By replacing the rigid abstractions of music theory (notes, grids, BPM) with continuous physiological models (`Log2Space`, ERB bands, neural oscillation), it creates a system where music is not constructed, but grown.

The paper "Conchordal: Emergent Harmony via Direct Cognitive Coupling in a Psychoacoustic Landscape" (arXiv:2603.25637) validated the psychoacoustic landscape as an effective ALife terrain through controlled experiments demonstrating self-organization, selection, synchronization, and hereditary accumulation. These results confirm that the Roughness-Harmonicity-Consonance pipeline and the Kuramoto entrainment model produce musically coherent emergent behavior under a range of initial conditions.

Version 0.4.0 integrates the paper findings into the instrument itself and completes the temporal half of the architecture: the fixed rhythm filterbank is replaced by an emergent meter (a forced limit-cycle oscillator with Hebbian tempo learning and PLV confidence), voice timing is unified on a single coupling continuum spanning metric, entrained, and flow families, and the composer's temporal control is reduced to terrain priors (`meter_stability`, `temporal_basin`) that shape where a pulse forms without ever scheduling one. A dual-bus design separates the habitat (what the ecosystem senses) from the presentation (what the audience hears). The release also includes opt-in Landscape habituation and a simulated `ListenerTwin` feedback path that can raise pitch-search temperature; both default to behaviorally inert settings.

The technical architecture—anchored by the `Log2Space` coordinate system and the "Sibling Projection" algorithm—provides a robust mathematical foundation for this paradigm. The use of Rust ensures that these complex biological simulations can run in real-time, bridging the gap between ALife research and performative musical instruments.

Chapter 9 closes the document by auditing the distance between the Manifesto's commitments and this implementation: what is discharged, what remains open, and what the implementation taught back.

# 9. Manifesto Correspondence and Open Problems

The Manifesto declares commitments; this chapter audits them. Each row of the ledger maps a commitment onto the mechanism that discharges it—or onto the gap where none yet does—so that the distance between declaration and implementation stays explicit. Findings that flowed *backward*, where implementation results revised the Manifesto's mechanism-level sketches, are recorded separately (Section 9.3): the principles stand; the sketches are corrigible.

## 9.1 The Ledger

| Manifesto commitment | Mechanism | Section | Status |
| :--- | :--- | :--- | :--- |
| Generation without symbolic intermediaries | `Log2Space` + landscape; no note names, scales, or time signatures anywhere in the engine | §2–4 | Implemented |
| Frequency-axis terrain (cochlea/brainstem models) | Roughness, Harmonicity, Consonance kernels | §3 | Implemented |
| Temporal-axis terrain (neural oscillation) | Emergent meter: forced limit cycle, Hebbian tempo learning, PLV confidence | §4 | Implemented; mechanism sketch revised (9.3). Measure-level accent *production* remains open |
| Landscape variability (culture, individual, unknown principles) | `roughness_k`, consonance kernel coefficients, optional habituation erosion | §3.4–3.5, §6.3.6 | Partial — cultural tuning systems not yet absorbed |
| Adaptation and expectation | per-Voice `AdaptationContext` + ecology/listener `HabituationField` | §3.5, §5 | Partial — adaptation is implemented; explicit phrase/scene expectation remains open (9.2) |
| Acoustic life: perception, metabolism, autonomy | The Voice: distinct articulation-life cores plus normalized energy, time-domain endurance/recovery, and viability | §5 | Implemented |
| Population: niches, symbiosis, terrain deformation | Crowding, respawn, the closed loop | §5 | Implemented |
| No central conductor | Local perception only; the meter emerges from the population's own onsets | §4–5 | Implemented (temporal scaffolding remains an explicit experiment) |
| Scenario as macro direction | Director terrain operations | §6.3.6 | Implemented; grounded by the scene window (9.2) |
| DCC stage two: biosignal closed loop | `ListenerTwin` pressure can feed pitch-search temperature when `[dcc]` coupling is enabled | §4.1, §6.2 | Simulated loop implemented and off by default; physical biosignal loop remains open |
| Dissolution of roles; spatial landscapes; heredity of timbre; other domains | Hereditary respawn exists as assays | — | Horizon |

## 9.2 Implemented Adaptation and the Remaining Expectation Gap

Human temporal cognition layers experience in three windows. Within the **perceptual present** (~3 s, upper bound ~8 s) no change is needed; texture itself carries. In the **prediction window** (3–8 s)—where musical phrases live—boredom is the operational state of a prediction engine with nothing left to update: beyond roughly eight seconds without a *perceptible* change, attention releases. And at the **scene window** (15–30 s), segmentation boundaries must arrive or the mind wanders. Beneath all three sits habituation: auditory cortex adapts to repeated spectra (stimulus-specific adaptation), so an unchanging percept literally fades from salience and recovers after withdrawal.

For a system whose terrain *is* a model of perception, the consequence is structural: **consonance is a meal, not a place**. The current implementation now realizes that claim at two levels. Each Voice's `AdaptationContext` keeps fast and slow traces over the fundamental-occupancy field, producing boredom and familiarity adjustments during pitch choice. At the shared-environment level, the optional `HabituationField` devalues sustained perceived-consonance activity and recovers after withdrawal (Section 3.5). The ecology and `ListenerTwin` keep separate states because the habitat and presentation buses can contain different sounds.

The remaining gap is **expectation**, not the absence of adaptation. No phrase-level predictor yet represents what event should occur next, and no scene-level mechanism autonomously creates or evaluates segmentation boundaries. The division of labor therefore remains: the **body** owns the micro layer (jitter, breath, beating), the **ecology** owns the meso layer (adaptation-driven movement, life and death), and the **scenario** owns the macro layer. `ListenerTwin` tension and resolvability can already close a simulated feedback loop by adding pitch-search temperature when `[dcc].coupling_strength > 0`; the default is zero, and connection to a physical listener's biosignals remains future work. With habituation disabled, "no more than ~8 s without perceptible change" remains a useful sample diagnostic rather than a system invariant.

## 9.3 Upstream Revisions

Implementation results have revised the Manifesto's mechanism-level sketches while confirming its principles:

*   **The four-band table → an emergent metrical hierarchy.** The Manifesto sketches fixed delta/theta/alpha/beta bands with assigned musical roles. Building that taught otherwise: a fixed filterbank is an imposed grid in disguise. What survives the perception research is the principle—neural oscillation structures musical time—realized as a self-organizing beat–subdivision–measure hierarchy with confidence (Section 4).
*   **Perceptual symmetry → the production-loop fixed-point requirement.** A terrain operation is ecologically meaningful only when a perceptual mechanism exists and the agents it attracts radiate spectra that reinforce it. Current tension controls therefore read the ecology-built terrain instead of warping it: movement tension is pitch-search *temperature*, and placement tension is a *relative consonance level*. See `docs/design-notes/tension.md`.
*   **Rate archaeology → observable time contracts.** Exposing raw energy pools and per-second rates made the composer balance dimensions that the ecology could derive. The implementation assay showed that the independent contract is nominal endurance, not `initial_energy` or `energy_cap`: energy can be normalized to $[0,1]$, while zero-fit drain is derived from endurance and the dissonance shape. Continuous recovery remains a separate time contract because per-second recovery and per-attack recharge have different dimensions. The same assay rejected collapsing `Entrain`, `Seq`, and `Drone`: their onset resets, death rules, telemetry, and render modulators are observably different, so one configured core would only hide the enum behind flags.
*   **Ambiguous object nouns → lifetime-bearing ontology.** `Material`, `Participant`, and a public draft Population each required prose exceptions because none named the lifetime it represented. The implementation now makes the transition explicit: `PopulationSpec + Placement --place()--> Population`, whose living members are Voices; all Populations sharing the Landscape form the Community. Population policy belongs to PopulationSpec, live control belongs to Population, and `place()` schedules founders immediately. `Species` is reserved until heredity and speciation give it an actual biological invariant. The naming is therefore guarded by observable identity: a Population survives member death and respawn, while a Voice and its generation do not.

# Appendix A: Key System Parameters

| Parameter | Module | Unit | Description |
| :--- | :--- | :--- | :--- |
| `bins_per_oct` | `Log2Space` | Int | Resolution of the frequency grid (typ. 48-96). |
| `sigma_cents` | `HarmonicityParams` | Cents | Width of harmonic peaks. Lower = stricter intonation. |
| `roughness_k` | `LandscapeParams` | Float | Saturation parameter for roughness mapping. Default: $(1/0.7) - 1 \approx 0.4286$ (so $x=1$ maps to $\approx 0.7$). |
| `kernel.a` | `ConsonanceKernel` | Float | Harmonicity coefficient (default 1.0). |
| `kernel.b` | `ConsonanceKernel` | Float | Roughness coefficient (default -1.35; negative penalizes roughness). |
| `kernel.c` | `ConsonanceKernel` | Float | Interaction coefficient (default 1.0; positive attenuates roughness penalty at high harmonicity). |
| `kernel.d` | `ConsonanceKernel` | Float | Bias term (default 0.0). |
| `beta` | `ConsonanceRepresentationParams` | Float | Sigmoid steepness for $C_{level01}$ (default 2.0). |
| `theta` | `ConsonanceRepresentationParams` | Float | Sigmoid threshold for $C_{level01}$ (default 0.0). |
| `consonance_density_roughness_gain` | `LandscapeParams` | Float | $\rho$ in density kernel $H(1-\rho R)$ (default 1.0). |
| `habituation.enabled` | `HabituationParams` | Bool | Enable Landscape erosion; default false, making effective views identical to base views. |
| `habituation.satiation_sec` | `HabituationParams` | Seconds | Time to reach 90% erosion under sustained unit drive (default 5.0). |
| `habituation.recovery_sec` | `HabituationParams` | Seconds | Time to recover 90% after drive withdrawal (default 8.0). |
| `habituation.ref_drive` | `HabituationParams` | Float | Half-saturation drive in the habituation transfer function (default 0.25). |
| `loudness_exp` | `LandscapeParams` | Float | Compressive exponent applied to A-weighted peak power in the subjective-intensity front end (default 0.23). |
| `tau_ms` | `LandscapeParams` | Milliseconds | Leaky normalization time constant after peak extraction (`[analysis].tau_ms`). |
| `stability` | `MeterShaping` | 0.0-1.0 | Beat attractor depth (`meter_stability`): scales entrainment forcing and tempo learning. |
| `basin_hz` | `MeterShaping` | Hz pair | Tempo prior region (`temporal_basin`): seeds and confines beat-frequency learning. |
| `coupling` | `CoupledTimingSpec` | 0.0-1.0 | Per-voice lock strength onto the shared beat (`entrainment`). |
| `flow_depth` | `CoupledTimingSpec` | 0.0-1.0 | Renewal clustering of free-running onsets (0 = regular). |
| `microtiming` | `CoupledTimingSpec` | cycles | Signed beat-phase offset of the lock target. |
| `temperature` | `PitchHillClimbPitchCore` / `PitchPeakSamplerCore` | Float ≥ 0 | Search stochasticity shared by both pitch cores; 0 settles greedily, higher explores. Musically the movement-tension knob (high = restless/straying off consonance, 0 = settled). |
| `crowding_strength` | `PitchHillClimbPitchCore` | Float | Strength of frequency-space crowding avoidance. |
| `crowding_sigma_cents` | `PitchHillClimbPitchCore` | Cents | Width of crowding penalty Gaussian (default 60). |
| `leave_self_out` | `PitchHillClimbPitchCore` | Bool | Whether to subtract own spectral contribution during evaluation. |
| `endurance_sec` | `LifecycleConfig::Sustain` | Seconds | Nominal energy-depletion time at zero field fit, with attacks/recovery disabled. |
| `recovery_sec` | `LifecycleConfig::Sustain` | Optional seconds | Time to refill normalized energy 0→1 at a full viability signal. |
| `attack_cost_fraction` | `LifecycleConfig::Sustain` | Normalized energy / attack | Capacity spent by a full-strength attack. |
| `attack_recharge_fraction` | `LifecycleConfig::Sustain` | Normalized energy / attack | Maximum capacity restored by a fully consonant attack. |
| `dissonance_penalty` | `LifecycleConfig::Sustain` | Float ≥ 0 | Extends well-fit lifetime while preserving the zero-fit endurance contract. |
| `dcc.coupling_strength` | `DccConfig` | 0.0-1.0 | Strength of ListenerTwin tension feedback; default 0.0 (behaviorally off). |
| `dcc.max_temperature_bonus` | `DccConfig` | Float ≥ 0 | Maximum pitch-search temperature added by listener pressure (default 0.10). |
| `attack_step` | `KuramotoCore` | Float | Envelope attack step size. |
| `decay_rate` | `KuramotoCore` | Float | Envelope decay rate. |

# Appendix B: Mathematical Summary

**Consonance Kernel (bilinear):**

$$ C_{score} = a \cdot H_{01} + b \cdot R_{01} + c \cdot H_{01} R_{01} + d $$

**Consonance Level (sigmoid representation):**

$$ C_{level01} = \frac{1}{1 + e^{-\beta(C_{score} - \theta)}} $$

**Consonance Density Mass ($\rho$-kernel):**

$$ C_{density\_mass} = \max(0,\; H_{01}(1 - \rho R_{01})) $$

**Landscape habituation and effective views:**

$$ d_i=\frac{L_iP_i}{L_iP_i+r_{ref}}, \qquad
h_i(t+\Delta t)=a h_i(t)+(1-a)d_i, \quad a=e^{-\Delta t/\tau_{rise/fall}} $$

$$ C_{score,i}^{eff}=\theta+(C_{score,i}-\theta)(1-h_i), \qquad
C_{density\_mass,i}^{eff}=C_{density\_mass,i}(1-h_i) $$

$$ C_{level01,i}^{eff}=\frac{1}{1+e^{-\beta(C_{score,i}^{eff}-\theta)}} $$

**Roughness Saturation Mapping** (from reference-normalized ratio $x$ to $R_{01} \in [0,1]$):

$$
R_{01}(x; k) = \begin{cases}
0 & \text{if } x \leq 0 \\
x \cdot \frac{1}{1+k} & \text{if } 0 < x < 1 \\
1 - \frac{k}{x+k} & \text{if } x \geq 1
\end{cases}
$$

where $k$ is `roughness_k` (default $\approx 0.4286$). The function is continuous at $x=1$ and saturates to 1 as $x \to \infty$.

**Time-domain lifecycle** (normalized energy $E\in[0,1]$):

$$ b = \frac{1}{T_e(1+p_d)}, \qquad
\Delta E_{basal} = -b\,[1+p_d(1-C_{level01}^{eff})]\,\Delta t $$

$$ \Delta E_{recovery} = \frac{V(C^{eff})}{T_r}\,\Delta t, \qquad
\Delta E_{attack} = -f_c + f_r C_{level01}^{eff}m_r $$

where $T_e$ is endurance, $T_r$ is recovery, $p_d$ is the dissonance penalty,
$V(C^{eff})$ is the viability-window signal, $f_c/f_r$ are attack cost/recharge
fractions, and $m_r$ is the bounded rhythm-reward multiplier.

**Harmonicity Projection (single-path Sibling Algorithm):**

$$ Roots[i]=\left(\sum_k k^{-\rho}A[i+\log_2(k)]\right)^{\gamma} $$

$$ H[i]=\max\left(0,\;\sum_m m^{-\rho}Roots[i-\log_2(m)]
-(1-w_{diag})\left(\sum_{k=1}^{K}k^{-2\rho}\right)A[i]\right) $$

The implementation then applies the optional absolute-frequency TFS gate and, by default, peak-normalizes the scan.

**Roughness Convolution:**
$$ R_{shape}(z) = \int A(\tau) \cdot K_{rough}(z-\tau) d\tau $$
