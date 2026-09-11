+++
title = "Technical Note: The Physics of Conchordal"
description = "A deep dive into the psychoacoustic algorithms, logarithmic signal processing, and artificial life strategies powering the Conchordal ecosystem."
template = "page.html"
[extra]
source_commit = "a5447e1"
author = "Koichi Takahashi"
last_updated = "2026-09-08"
source_version = "0.4.0"
source_snapshot = "a5447e1 plus local runtime and research changes, 2026-09-08"
revision_scope = "Runtime continuity, placement bounds, measure accents, habituation observations, DCC and Chapter 9; not a complete re-audit"
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

With `--report`, an additional `habituation_scan` record at each generated second carries
`fmin_hz`, `bins_per_octave`, `n_bins`, and the habitat's `state_scan`, `raw_score_scan`,
and `eff_score_scan`. The arrays share fixed Log2Space coordinates and are serialized
from borrowed slices. This permits a fixed region to be followed through withdrawal;
the older per-hop `tracked_bin` remains the moving raw-score maximum.

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

**Perception vs production.** The runtime maintains two meter instances. The *production meter* runs in the worker thread on the habitat bus and drives all voice behavior. The *perception meter* lives inside the `ListenerTwin` (`listener_twin/`), which analyzes presentation audio; its beat confidence feeds the UI and headless report. Direct Cognitive Coupling (DCC) pressure uses presentation-derived consonance tension, not beat confidence directly.

## 4.2 Beat Confidence: Phase-Locking Value

The meter does not merely track a beat—it knows *how much* beat there is. Each detected onset deposits a unit phasor (a unit complex vector at the current beat phase) into leaky accumulators; the length of their resultant is a phase-locking value (PLV): 1.0 when all onsets land at the same phase, near 0 when phases are scattered. Confidence is the PLV gated by a *presence* term that requires roughly four accumulated onsets of evidence before it saturates—matching the psychological observation that beat induction needs a few cycles—and decays in silence. Scattered onset phases keep the resultant low, so confidence cannot be fabricated by density alone.

## 4.3 From Meter to Modulation: NeuralRhythms

`NeuralRhythms::from_meter_state` (`core/modulation.rs`) projects the metrical state onto the modulation bands consumed by voice behavior:

*   **Delta** ← the beat (tactus): phase and tempo of the pulse. Its phase drives `env_open`, a cosine gate that sharpens the articulation envelope toward the downbeat as confidence rises and leaves it open when the beat is uncertain.
*   **Theta** ← the subdivision: the note-rate band that the breath oscillator (Section 5.5) locks to.
*   Delta/Theta precision (`alpha`) equals beat confidence; prediction error (`beta`) is its complement.
*   **Measure** carries its own phase, frequency and confidence, with `measure_ratio` identifying the detected grouping. Its phase origin follows the observed strong-onset phasor. No detected grouping yields a neutral, zero measure band; the optional `measure_accent` strength coupling consumes this band.

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
    *   **When**: `Once` (single trigger), `Pulse { rate_hz, sync, social }` (repeated triggers on the adaptive gate clock), or `Coupled(CoupledTimingSpec)` — explicit synchronization or bodily acoustic participation (Section 5.4).
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

## 5.4 Onset Timing: Synchronization and Participation

`RhythmRelation` distinguishes synchronized timing (`metric()`) from acoustic
participation (`entrained()` and `flow()`). Both issue sound through the same
PhonationEngine command path.

Participation separates causal habitat observation from the body's due time.
The observer projects fully observed recurrence cycles onto a roughly four-second
horizon at 20 ms spacing. Recurrence contrast fades toward the adaptive baseline
with a six-second time constant; energy beyond the horizon uses the observed mean.
Future queries never enter the observation history. Known self habitat PCM is subtracted
from the observed mixture before energy analysis. The external history is projected
using both the previous shared recurrence projection and a local history predictor. Matching actual outcomes fit their bounded mixture for subsequent participation requests (9.3.37).

The action policy compares sounding near the due time, waiting up to one participation
period, and omitting this event. Candidate cost combines squared displacement in
participation periods, distance from experienced external energy shapes, and a
three-band overlap integral. Context memory and candidate forecasts refer to the same
two observation windows after an onset (9.3.35). A 64-point envelope estimate includes the duration
rule's hold and the Voice's ADSR release. Omitting costs one plus the number of
consecutive voluntary omissions; an emitted onset resets that count. This cost
is an experimental action preference, not a new metabolic reward. It prevents
bounded acoustic costs from prescribing permanent silence, but does not establish
musical quality. The envelope omits additional body/modulator decay, and the
energy proxy is not an auditory masking or consonant-fusion model.

Experience memory updates only after an emitted onset and both matching acoustic windows. The intrinsic period, acoustically
adapted participation period, and sound duration have separate roles. An entrained Voice
selects a supported recurrence near its intrinsic scale, weighted by a Gaussian
affinity with a half-octave width. Coupling and recurrence support control the
adaptation rate, with two intrinsic periods setting the time constant. Unavailable
references relax the cadence toward the intrinsic pace. These support values and
constants are engineering choices, not validated human beat confidence.

Cadence updates preserve the remaining fraction of the Voice's own cycle and
never target a shared onset phase. Each observation is used once; future evidence
and changes to already selected actions are rejected. Chosen waits do not train
the cadence. Flow keeps the intrinsic scale for clustered renewal instead of
adapting it to recurrence. Acoustic context still affects waiting and omission,
so executed intervals can change. Switching to flow preserves a committed
candidate and restores the intrinsic scale for subsequent intervals. It also
clears the cadence-observation timestamp, so returning to entrained timing does
not count time spent in flow as elapsed adaptation time.

Hold lengths and their overlap preview use the configured intrinsic period,
independently of cadence adaptation. Thus `cycles(n)` does not stretch a sound when
the acoustic recurrence slows; flow retains its existing threefold hold multiplier.
ADSR release remains in seconds. Later tempo observations do not reschedule an
existing note-off. Candidates respect the actual-onset 200 Hz ceiling, and amplitude
and viability gates still decide whether to emit an onset.

This separation follows the author's report that the fixed-hold comparison sounded
more natural. The earlier version adapted both cadence and hold length and was
judged closer to random in Sample 08. The new report is a scoped observation, not
acceptance across all seeds or a proof of autonomous temporal niches. Forecasts
are built only for pending decisions. Observed external contexts now feed subsequent
candidate comparisons (9.3.35). Sustained-note termination, causal effect estimation
and action-value learning remain open. Earlier device timings retain their
original version and duration scope (Section 9.3).

Observed external RMS history, indexed by past age and known coverage, is also frozen
with the decision-time evidence (9.3.36). The current cost still uses two-window
experienced context; history-conditioned relationship retrieval/prediction is a separate connection.

For explicit synchronization, onset timing is generated by the `CouplingClock` (`life/phonation_engine.rs`): a per-voice phase oscillator that emits an onset at every integer crossing of its phase. Its effective rate blends an intrinsic renewal rate with the shared beat:

$$ f_{eff} = (1 - \ell)\, f_{int} + \ell\, f_{beat}, \qquad \ell = \kappa \cdot c $$

where $\kappa$ is the voice's coupling strength (`entrainment`, 0–1) and $c$ is the meter's beat confidence (Section 4.2)—so a voice can only lock as strongly as the meter is believed. A phase pull drags the oscillator's crossings toward the beat phase (optionally offset by `microtiming`):

$$ \dot{\phi} = f_{eff} \left(1 + \ell K \,\mathrm{err}(\phi_{beat} - \phi)\right) $$

where $\mathrm{err}(\cdot)$ is the wrapped phase error in cycles ($[-0.5, 0.5]$) and $K$ a fixed pull gain chosen so the rate factor stays positive—the phase always advances, only faster or slower.

This phase-attraction mechanism serves synchronized timing. Increasing coupling
for an entrained/flow participant instead increases its response to acoustic
context; it does not switch the participant into this clock.

Each onset carries a strength set by the voice's `rhythm_role`—beat 1.0, subdivision 0.7, accent 2.5, texture 0.85—and these strengths feed back into the production meter's drive. A recurring accent therefore drives the meter harder, allowing a downbeat (and eventually a measure) to be *induced* by the population rather than declared. There is no externally imposed grid anywhere in this loop.

The optional `measure_accent(amount)` couples detected measure structure back to onset strength.
Its multiplier is `1 + 0.35 * amount * confidence * cos(phase)`, evaluated at each actual onset.
The measure phase is measured relative to the circular mean of the observed accents, rather than
the arbitrary origin of the accumulated beat count. An undetected measure produces no modulation;
the default amount is zero. This path schedules no new onset and sets no beats-per-measure value.
Production `rhythm_observation` records expose the measure phase, frequency, confidence and ratio
used by the voices; listener observations remain separate.

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
6.  When DCC coupling is enabled, `tension_pressure = tension_level * coupling_strength` produces a bounded temperature bonus that feeds the Voices' pitch search. Listener tension already includes resolvability; the coupler applies no second factor. The default coupling strength is zero, so this path is behaviorally inert unless explicitly enabled.
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

**Rhythm (Section 5.4)**: `metric()` selects shared-beat attraction; `entrained()` and `flow()` select acoustic participation. `entrainment(v)` changes the influence within that intention. `microtiming(v)` offsets the synchronized target only. `temporal_basin` shapes the production meter, while participation retains a bodily prior. Articulation controls remain separate: `rhythm_freq(v)`, `rhythm_coupling_vitality(lambda_v, v_floor)`, and `rhythm_reward(rho_t, "attack_phase_match")`.

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

## 7.1 Case Study: Three Timing Intentions (`samples/07_heartbeat.rhai`, `08_murmuration.rhai`, `09_rain.rhai`)

The three samples now exercise the separate intentions described in Section 5.4.

1. **Metric** (`07_heartbeat.rhai`): explicit synchronization to the production beat. The dedicated downbeat Voice remains because synchronization is the subject of the demo.
2. **Entrained** (`08_murmuration.rhai`): bodily participation using acoustic context. The author rejected unintended final unison in the previous implementation; the revised audio is under evaluation. Gated attacks use ordinary consonance recharge, without a phase-match reward.
3. **Flow** (`09_rain.rhai`): clustered bodily renewal with weak acoustic participation. Pitch still follows the consonance field.

Samples 08 and 09 do not gain a dedicated synchronization accompaniment.

## 7.2 Case Study: Drift and Flow (`samples/research/drift_flow.rhai`)

This file is retained as an early, compact historical fixture; it no longer validates autonomous hop-based drift.

1.  **Explicit patch**: A sustained anchor enters at C2 (65.41 Hz), while a second sustained Voice enters at C#3 (138.59 Hz). The script then patches that Voice directly to 220 Hz with `population.freq(220.0)` and releases it.
2.  **Field query**: Five one-Voice swarms are placed in sequence with `consonance(130.0).peak().range(1.0, 4.0)`, demonstrating repeated deterministic queries against the live field.
3.  **Terrain change**: The anchor is finally patched to F2 (87.31 Hz), changing the field after those placements.

The fixture therefore demonstrates live Population patches plus repeated field-relative placement. Autonomous settling is demonstrated by `samples/05_settling.rhai`; the old claim that `drift_flow.rhai` itself produces boredom-driven, endless hopping was stale and has been removed.

## 7.3 Case Study: Emergence and Resolution (`samples/12_emergence_and_resolution.rhai`)

The closing integration sample combines a habitat-only harmonic root, a metric colony, and a temporary non-metric flow Population. It has no dedicated beat-carrier Voice. The colony's own onsets drive the shared production meter, while the hidden sustained root supplies a pitch reference to the Landscape.

The colony enters at 2.3 s. At 11.7 s, the Scenario raises its search temperature and shifts the root upward; flow enters at 15.0 s. At 20.3 s, the Scenario cools the search, explicitly enables Glide, returns the root, and reduces flow amplitude. Flow is released at 23.6 s, followed by a further colony amplitude reduction at 24.9 s and the remaining releases. These are scheduled macro operations; individual pitch searches, onsets, viability, and consonance-biased respawn remain local processes.

“Tension” and “resolution” name the intended arc, not a verified perceptual outcome. The resulting harmonic development and closure require listening judgments for the rendered conditions. The separate flow-amplitude comparison and its author judgments are recorded in `docs/roadmap/flow-amplitude-audition-2026-09-05.md`.

# 8. Conclusion

Conchordal establishes a foundation for Bio-Mimetic Computational Audio. By replacing the rigid abstractions of music theory (notes, grids, BPM) with continuous physiological models (`Log2Space`, ERB bands, neural oscillation), it creates a system where music is not constructed, but grown.

The paper "Conchordal: Emergent Harmony via Direct Cognitive Coupling in a Psychoacoustic Landscape" (arXiv:2603.25637) validated the psychoacoustic landscape as an effective ALife terrain through controlled experiments demonstrating self-organization, selection, synchronization, and hereditary accumulation. These results confirm that the Roughness-Harmonicity-Consonance pipeline and the Kuramoto entrainment model produce musically coherent emergent behavior under a range of initial conditions.

Version 0.4.0 integrates the paper findings into the instrument itself and completes the temporal half of the architecture: the fixed rhythm filterbank is replaced by an emergent meter (a forced limit-cycle oscillator with Hebbian tempo learning and PLV confidence), the release initially unified voice timing on a single coupling continuum spanning metric, entrained, and flow families (the subsequent participation redesign is recorded in Sections 5.4 and 9.3), and the composer's temporal control is reduced to terrain priors (`meter_stability`, `temporal_basin`) that shape where a pulse forms without ever scheduling one. A dual-bus design separates the habitat (what the ecosystem senses) from the presentation (what the audience hears). The release also includes opt-in Landscape habituation and a simulated `ListenerTwin` feedback path that can raise pitch-search temperature; both default to behaviorally inert settings.

The technical architecture—anchored by the `Log2Space` coordinate system and the "Sibling Projection" algorithm—provides a robust mathematical foundation for this paradigm. The use of Rust ensures that these complex biological simulations can run in real-time, bridging the gap between ALife research and performative musical instruments.

Chapter 9 closes the document by auditing the distance between the Manifesto's commitments and this implementation: what is discharged, what remains open, and what the implementation taught back.

# 9. Manifesto Correspondence and Open Problems

The Manifesto declares commitments; this chapter audits them. Each row of the ledger maps a commitment onto the mechanism that discharges it—or onto the gap where none yet does—so that the distance between declaration and implementation stays explicit. Findings that flowed *backward*, where implementation results revised the Manifesto's mechanism-level sketches, are recorded separately (Section 9.3): the principles stand; the sketches are corrigible.

## 9.1 The Ledger

| Manifesto commitment | Mechanism | Section | Status |
| :--- | :--- | :--- | :--- |
| Generation without symbolic intermediaries | `Log2Space` + landscape; no note names, scales, or time signatures anywhere in the engine | §2–4 | Implemented |
| Frequency-axis terrain (cochlea/brainstem models) | Roughness, Harmonicity, Consonance kernels | §3 | Implemented |
| Temporal-axis terrain (neural oscillation) | Emergent meter: forced limit cycle, Hebbian tempo learning, PLV confidence | §4, §5.4 | Metric retains explicit synchronization. Entrained/flow use a bounded acoustic observer and bodily participation policy through normal Voice rendering (9.3). Device load checks and the author's no-collapse report for Sample 08 apply to their recorded versions; later changes require separate author judgments. Relation persistence across all conditions remains open. Measure-accent coupling is opt-in |
| Landscape variability (culture, individual, unknown principles) | `roughness_k`, consonance kernel coefficients, optional habituation erosion | §3.4–3.5, §6.3.6 | Partial — cultural tuning systems not yet absorbed |
| Adaptation and expectation | Per-Voice `AdaptationContext`, ecology/listener `HabituationField`, passive ListenerTwin interval observations and acoustic participation memory | §3.5, §5 | Partial: short-term recurrence, energy forecasts updated from matching observed windows, and experienced context memory are connected to habitat participation (9.3.32–9.3.35). Continuous relational histories and action-conditioned acoustic hypotheses remain research work (9.3.8–9.3.31). Ordinary samples do not provide scorable intervals. Cognitive retention/interference, stream identity, phrase/scene expectation and action-value learning remain open. |
| Acoustic life: perception, metabolism, autonomy | The Voice: distinct articulation-life cores plus normalized energy, time-domain endurance/recovery, and viability | §5 | Implemented |
| Population: niches, symbiosis, terrain deformation | Crowding, respawn, the closed loop | §5 | Implemented |
| No central conductor | Local perception only; the meter emerges from the population's own onsets | §4–5, §7 | Implemented; dedicated beat carriers and temporal scaffolds are restricted to explicit synchronization demonstrations or assays (sample 07 among the public samples) |
| Scenario as macro direction | Director terrain operations | §6.3.6 | Implemented as authorial composition; distinct from internally perceiving and remembering long structure (9.3.55) |
| Temporal structure grounded in auditory cognition | Interactions among articulation, groove, beat/meter, phrase, repetition/variation, section and whole-piece context | §9.3.55 | Baseline design reviewed; subsequent M0 numerical and model-boundary revisions have no new external review. Replacement contracts and experiments are registered, not implemented or passed. Short-time participation/history partly implemented; persistent relations, boundary/closure and long-context effects on generation remain incomplete |
| DCC stage two: biosignal closed loop | `ListenerTwin` pressure can feed pitch-search temperature when `[dcc]` coupling is enabled | §4.1, §6.2 | Simulated loop implemented and off by default; physical biosignal loop remains open |
| Cognition coupled to the sound actually presented | Separate presentation analysis; missing hops invalidate observations, reset NSGT history and suspend DCC pressure until a complete window is available | §6.2 | Implemented; physical-device overload validation remains open |
| Music as a living performance | The instrument exposes no audio-file output; the separate `conchordal-render` binary supports offline study | §6 | Implemented as a binary boundary |
| Dissolution of roles; spatial landscapes; heredity of timbre; other domains | Hereditary respawn exists as assays | — | Horizon |

## 9.2 Implemented Adaptation and the Remaining Expectation Gap

Adaptation, expectation, and segmentation need distinct mechanisms. The former division here into a roughly three-second perceptual present, a 3–8-second prediction window, and a 15–30-second scene window was neither derived from DCC nor fitted to cognitive tasks in this implementation. Duration alone does not prescribe boredom, phrase length, or the time to change a scene. Sections 9.3.1–9.3.5 distinguish constraints from cognitive tasks, candidate mathematical mechanisms, and numerical windows. Scenario timing remains an authorial choice.

The design metaphor **consonance is a meal, not a place** corresponds to two mechanisms that change the action environment with occupancy history. Each Voice's `AdaptationContext` keeps fast and slow traces over the fundamental-occupancy field, producing boredom and familiarity adjustments during pitch choice. At the shared-environment level, the optional `HabituationField` devalues sustained perceived-consonance activity and recovers after withdrawal (Section 3.5). The ecology and `ListenerTwin` keep separate states because the habitat and presentation buses can contain different sounds. Implementing these update rules does not identify human boredom or attentional change.

The remaining gap is **expectation**, not the absence of adaptation. No phrase-level predictor yet represents what event should occur next, and no scene-level mechanism autonomously creates or evaluates segmentation boundaries. In the current implementation, the **body** owns the micro layer (jitter, breath, beating), the **ecology** owns the meso layer (adaptation-driven movement, life and death), and the **scenario** provides macro direction. This describes the present division of labor, not a principle excluding long temporal structure from DCC (9.3.55). `ListenerTwin` tension and resolvability can already close a simulated feedback loop by adding pitch-search temperature when `[dcc].coupling_strength > 0`; the default is zero, and connection to a physical listener's biosignals remains future work. The roughly eight-second value used by existing sample diagnostics is also an assay setting, not a universal boredom threshold or a system invariant.

`ListenerTwin` currently measures local improvement potential in the presentation-derived
consonance field. Its search holds that field fixed; it does not predict a moved Voice's
new field or remember a home root or preceding phrase. The existing
`tension_level = (1 - stability_level) * resolvability_level` therefore does not measure
homecoming or phrase closure. A more stable presentation can still have higher tension
when nearby improvement potential increases. This expectation gap remains open.

An offline comparison on 2026-09-06 estimated pitches from monophonic waveforms
and predicted the next continuous log-frequency interval from the preceding two
intervals. The baseline used the same history without order conditioning. Across
six seeds and 60 conditions, repeated contours yielded 1.338–2.275 bits/event of
mean predictive gain. Timbre and amplitude changes retained similar gains, but
repetition of an altered contour was also predictable: mean gain did not locate
the change. The initial change-candidate rule detected the partial alteration in
only one of six seeds. At that stage the research script was not connected to the production
ListenerTwin, generation, or a closure decision. Conditions and limits are recorded
in `docs/roadmap/phrase-expectation-evidence-2026-09-06.md`.

The author subsequently reported that all presented sounds matched their labels.
Calibrating errors only for previously supported contexts then detected all six
change conditions from three unused seeds within one to three observed notes,
with no false candidates in 384 evaluation events across unchanged, timbre-only,
amplitude-only, and pause controls. These checks use stimulus-generation records;
they do not establish correspondence with perceived phrase boundaries or closure.

Mixture controls exposed a short-lag autocorrelation failure: nearby tones were
assigned an intermediate frequency. A research option now checks agreement over
multiple periods and combines their period estimates. It left 144 nearby-tone and
inharmonic-mixture events without a pitch estimate while retaining 360 single-tone,
harmonic, and weak-interference events. Missing audio does not count as a rest or
a new attack on reconnection, and no interval is scored across it. The revised
66-condition assay passed; separating continuously overlapping melodic streams
remains unimplemented. Another 30 monophonic conditions retained observation
accuracy but included a delayed shuffle detection. The frozen previous observer
yielded the same candidate times, exposing a generalization limit of the local
predictor.

The limited observer has since been ported to Rust and attached to ListenerTwin
when `--report` is enabled. It reads post-render presentation audio and sample
indices, emitting `listener_contour` records for waveform periodicity, local
prediction error, silence, and missing input. `time_sec` identifies when evidence
became available; it does not imply that a pitch persisted from `onset_sec`.
No values feed back into the existing ListenerState, DCC, or generation. Audio and
FFT buffers are reused, with at most 128 transitions and 128 calibration errors.
A 350-tone check recorded zero Rust allocation requests inside the observer after
construction. Validation passed 710 Rust tests, 90 frozen Python audio conditions,
and 12 device conditions on MOTU M2. With report enabled, DCC 0, and seed 21,
the 10-second measurement windows had maximum hop p99 4.938 ms against a
10.667 ms budget, with zero missing output frames or callback errors.

However, normal Samples 07, 08, 09, and 12 at three seeds each yielded zero scored
intervals in all 12 renders, even though their WAVs matched the previously adopted
version exactly. Continuously overlapping sound did not satisfy the observer's
silence-separated monophonic assumption. These checks establish passive wiring
and its measured cost; useful expectation in normal performances remains open,
alongside observing changes within continuous mixtures, closure, and generation coupling.

A subsequent offline assay compared prediction of normalized log-frequency power
mass in overlapping audio, without assigning source pitches. It used non-overlapping
85.3 ms trailing windows at 120 ms intervals and at most 128 transitions. Two-window
context was compared with a history average, one-window context, and persistence of
the immediately preceding distribution. After 54 development conditions, another
54 conditions at newly selected seeds retained the coefficients. Overlapping
repetition gained from ordered context, but timbre changes exposed a limit, and
transposition did not consistently beat the one-window control. In the same 12
ordinary sample recordings, 1,643 windows after ten seconds were scored. All 12
conditions beat the history-average control but lost to both one-window context
and persistence. This candidate therefore remains research-only. Spectral mass
is not melodic contour, execution is not acceptance, and replaying the same
recordings does not provide independent samples. The next comparison must specify
the prediction horizon and distinguish when a change occurs from its direction.
See `docs/roadmap/phrase-expectation-evidence-2026-09-06.md`.

An offline first-passage comparison subsequently separated timing from direction.
It froze the origin spectrum, defined a change by Hellinger distance at least
0.25, and predicted its first observed crossing within 0.36, 0.72, or 1.44 seconds,
or no crossing before the deadline. Direction described the log-frequency mass
center, with a 0.025-octave tolerance. Ten joint outcomes kept the time probabilities
consistent. Labels entered bounded memory only at the final deadline; incomplete
horizons were censored, and end of file supplied no ending label. These thresholds
define an assay, not universal perceptual boundaries. Across 45 development and
45 additional-seed conditions, all 36 steady/gain/glide/redistribution/noise control
checks passed. The normal recordings supplied 1,499 scored origins after ten seconds,
of which 1,498 crossed before 1.44 seconds. Origins share future samples and are not
independent events. No normal condition improved timing against all controls,
including immediate change magnitude. Direction improved against all controls in
one condition by only 0.000137 bits over one-frame context. Additional-seed repeated
stimuli also lost to one-frame context, so the score is not a measure of musical
repetition. This candidate remains research-only. The next representation must
distinguish local fluctuation from a sustained change of acoustic state before it
can be linked to phrase or closure.

The next offline observer compares two adjacent collections of twenty spectra
(2.4 seconds each). A Gaussian kernel on square-root spectral mass gives an
empirical distribution distance: the biased MMD in
[Gretton et al. (2012), equation 5](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf),
divided by the square root of two. This distinguishes alternating from simultaneous
occupancy even when their average spectra agree. A second channel measures
transport of mean mass along the log-frequency axis, retaining register distance.
The mean-spectrum Hellinger distance is a separate comparator. These are acoustic
features, not source pitches, sequence memory, or phrase labels.

Fixed thresholds produced false candidates at a new seed. The revised observer
compares all 126 balanced allocations of ten four-frame blocks; empirical
quantiles and distance floors set hysteresis thresholds. These correlated blocks
do not supply an iid significance test or a calibrated false-alarm probability.
Three consecutive high comparisons on either channel announce a candidate;
six low comparisons on both channels establish or recover operational stability.
Initially the state is unestablished. Missing evidence resets it and censors an
open difference episode; an already-varying input does not establish a new onset.
Settling means that this window comparison is low, not that the sound has returned
to an earlier identity or reached musical closure.

After 54 development conditions, the frozen implementation passed 27 conditions
at three unused seeds: twelve intended sustained transitions were detected after
1.52–2.52 seconds, with no extra candidates in those controls. Mean spectra alone
missed the alternating-to-simultaneous changes; the distribution channel alone
missed one register control in each seed group. The same twelve normal recordings
remain much less conclusive: six never established combined stability. At the last
usable comparison five others had open differences, later censored when evidence
became unavailable, and one was stable. No phrase labels or prediction gains
were established. This observer remains offline. A condition-hidden audition now
asks whether acoustic changes correspond to a new musical grouping; synthetic
control agreement cannot settle that author judgment or justify feedback.

The author subsequently heard A as unchanged, B as suggesting a change that did
not persist, and C as changing into a continuously sounding state without motion.
C deliberately replaced alternating tones with a fixed four-tone mixture. This
supports the reported distinction in that presentation; it does not establish a
musical phrase or acceptance of the destination. The next control retains the
same frequencies and event trajectory but changes from alternating single tones
to alternating neighboring pairs. A short excursion supplies the negative
control. The observer and its thresholds remain fixed; the new comparison asks
whether a destination can retain a recognizable state while motion continues.
The moving destination and its brief excursion passed across seven seeds,
including three newly selected seeds and the original presentation seed. That
additional campaign also missed two existing register-change controls, so the
observer's general usefulness remains unestablished. These failures were retained;
the new audition evaluates the destination's flow without adopting the observer.

The author then reported unchanged flow with added overlap. The stimulus retained
the event trajectory and changed concurrent tones; this is evidence of a heard
texture change with preserved flow in that presentation. It supplies no positive
example of a new flow or phrase, and the reply expresses no preference. Retain it
as a texture-only control. The next candidate comparison holds overlap and mean
interval constant while changing interval order; that manipulation is not itself
a definition of musical flow.

An offline interval-order comparison now fixes timbre, note length, attack count,
mean interval and the interval multiset while changing `short-short-long-long`
to `short-long-short-long`. Whole-sequence cyclic rotation is an unchanged-order
control. Native acoustic flux candidates include releases and energy ripples in
some controls, so the research observer requires rising energy at both trailing
20 ms and 40 ms scales. Raw candidates remain available. This observes energy
attacks, not individual sources; equal-energy spectral changes can be missed.

The frozen observer matched all 2730 attacks in 24 development and 24 additional
conditions at unused seeds, with maximum latency 18.7 ms. It forecasts the next
log inter-onset interval from the previous two, before observing that interval,
against one-interval, unordered-history and persistence controls. All six order
changes produced prediction-error candidates; the thirty unchanged-order,
rotation, loudness, timbre and overlap controls produced none. The simpler
one-interval predictor wins after the switch to alternation. Across the same
twelve normal recordings, 313 intervals were scored after the first ten seconds,
but only three recordings beat all controls. These are exploratory gains, with
no phrase labels or established listener correspondence. The observer remains
offline; an equal-energy, equal-count A/B audition tests whether the author hears
the changed order. Both clips last 17.819 seconds and contain 49 attacks.

Generalizing the frozen predictor to five interval patterns and three unused
seeds preserved all 7833 attack observations across 120 controls. Nine of fifteen
order changes produced candidates. Three long-run changes were missed despite
eligible thresholds; three asymmetric changes lacked calibrated thresholds during
the first three seconds. The report now separates unavailable error judgments
from eligible checks with no candidate. A fixed number of repetitions does not
ensure enough supported contexts for calibration.

Twelve normal recordings at three additional seeds yielded 297 scored intervals,
but only 58 were eligible for error judgments. Before observation, the next-step
screen required positive gains against all three controls and at least sixteen
scored intervals at every seed of a sample. None of the four samples passed.
This is an engineering screen, not a test of statistical significance or musical
acceptance. The interval predictor will not be integrated into the runtime on
this evidence. Its synthetic observation success cannot replace usefulness in
normal performances or the author's judgment of the intended musical change.

The author then reported that only B changed. The presentation hashes were
verified, and the reply was retained as a discrimination of interval order with
fixed timbre, count and energy. It establishes no preference, phrase identity or
closure. The next observation hypothesis retains frequency position alongside
the order of acoustic changes, before pooling them into one interval sequence.
The current three coarse bands alone cannot supply that distinction: every
accepted attack in the three new Sample 12 recordings was in the middle band.
This motivates a comparison, not a diagnosed cause of the prediction failures.

The next offline observer retains spectral-component trajectories and local
power rises. It uses trailing 85.3 ms Hann windows at 10 ms strides, interpolates
spectral peaks into log frequency, and follows at most 32 local ridges. A rise
requires three consecutive observations and increased power across two adjacent
three-frame averages. Harmonic partials remain separate components; ridge IDs
are observation identifiers. Missing input resets continuity, and an already
sounding initial component supplies no rise merely because observation began.

After development controls, 27 conditions at unused seeds matched all 591
component rises within 70 ms. Fixed-frequency error was at most 0.317 cents in
those controls. In three weak-overlay comparisons, the scalar energy observer
covered only one of 25 attack times; the frequency-local observer covered all 25.
It also retained a continuous glide without repeated rises. The same twelve
normal recordings yielded trajectories, but these are not separated melodies
or newly demonstrated predictive gains.

Resolution and identity controls delimit this representation. Two sustained
tones at 330 and 338 Hz produced one ridge and 21 power rises; two crossing tones
produced five ridge IDs and six rises. Stationary noise also produced a few
rises. These outcomes motivated the name `rise`, replacing the initial `attack`
label without changing numerical results. A rise can reflect interference or
newly resolved energy without a source beginning to sound. Timing and frequency
must remain available to the next prediction comparison, and simultaneous rises
must not acquire an arbitrary source order. The instrument is unchanged.

The next offline comparison groups rises within a fixed 30 ms window, preserving
each group's power-normalized frequency distribution. It forecasts a normalized
joint density over the next log interval and log frequency. Two-group timing and
frequency context is compared with timing context alone, one group, an unordered
history and persistence, all using the same observations. At most 128 transitions
are retained. Forecasts are issued after the current group closes and before the
next group starts; missing input and EOF censor pending evidence. Scores separate
the time marginal from frequency conditional on time.

After 18 development conditions, 18 conditions at unused seeds matched all 1170
controlled groups. Frequency order added information beyond timing-only context,
and interval order remained observable. Replaying the author's existing A/B
matched 49 groups in each clip; only B's post-change time loss increased. These
are acoustic and predictive checks, not additional listening judgments.

Across the same twelve normal recordings, 2207 groups were scored after ten
seconds. Joint gains over timing-only context were 0.316–0.795 bits/group, but
gains over unordered history were only −0.00122 to 0.00216. Every recording's
median conditional-context weight was below 1.57e-8: the nominally ordered model
mostly fell back to its unordered distribution. No sample passed the
prespecified screen requiring positive gains against all four controls at all
three seeds. The candidate remains offline.

A subsequent factorial comparison separates absolute versus relative frequency
configuration from two versus three context groups. Relative coordinates use
the mean of the last group's eight log-frequency quantiles as a reference,
without treating it as source pitch. Historical targets are aligned to the
current reference for every learned baseline, including unordered history.
The persistence baseline and broad prior stay identical. Three observed groups
can retain two relative movements; comparing both context lengths prevents a
longer history from masquerading as a coordinate advantage.

After development, 33 controlled recordings at unused seeds matched 2145 groups.
On the single-component pattern moving through the register, relative three-group context gained
1.351–1.415 bits/group over its equally aligned unordered baseline. Yet twelve
new normal recordings supplied 2258 scored groups without any of the four
variants passing the all-three-seed screen. Relative two-group gains over
unordered history ranged from −0.01171 to 0.00696 bits/group; three-group gains
were −0.000374 to 0.0000457. Context support remained scarce. The candidate stays
offline; neither coordinate alignment nor longer aggregate history resolves the
normal-recording limitation demonstrated here.

The next design question concerns grouping before sequential prediction.
[Elhilali et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2673083/)
showed different streaming percepts for synchronous and asynchronous components
despite similar time-averaged tonotopic responses, and proposed temporal coherence
as a grouping criterion. [Barascud et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4747708/)
studied sensitivity to acoustic regularities with an online statistical observer;
that does not validate the grouping of arbitrary polyphonic input. Our inference
is to compare cross-component temporal relations and multiple local histories
before requiring the whole mixture to recur as one aggregate sequence. These
studies motivate a comparison; they do not prove a Conchordal source separator,
phrase model or remedy for the measured prediction failure.

A research-only temporal-relation observer now tests one part of that proposal.
It projects ridge power onto 113 regular log2-frequency bins from 55 to 7040 Hz,
using a normalized Gaussian of width 0.075 octave. The difference between causal
40 ms and 200 ms amplitude filters supplies modulation features. A trailing
2-second covariance window produces signed correlations between sufficiently
energetic, occupied and modulated channel peaks, reported every 100 ms. The
observer keeps bounded state; its Python recording collectors do not. This is
an engineering diagnostic inspired by temporal coherence, not a reproduction of
the cited neural model or a measured real-time implementation.

Twenty-four controlled recordings at three unused seeds passed the frozen
observation criteria. Common activity gave median correlations of 0.999–1.000,
alternation −0.441 to −0.407, and differing periods −0.008 to 0.003. Constant
tones remain energetically present without providing modulated pairs. Missing
input resets the filters and requires another complete window. A common gain
modulation also creates positive correlation between sustained components, so
correlation cannot identify a Voice or a number of sources. Glide, vibrato,
beating and noise remain ambiguity diagnostics, without source-count labels.

Reusing the twelve normal recordings from the coordinate comparison yielded
718 available reports out of 720 in the 4–10 second interval. A separately
specified audit used complete, nonoverlapping 2-second feature windows outside
the initial and final margins. Within each window, 99 surrogates independently
circularly shifted each selected modulation channel, retaining its values and
circular autocorrelation while changing relative alignment. The number of
pairs with correlation at least 0.8 exceeded the surrogate 95th percentile in
28 of 143 available windows (150 complete windows). Nine recordings supplied
such windows; all three Sample 09 recordings supplied none. This is a
descriptive alignment comparison, without independent-window or corrected
significance claims. It establishes neither sequence-prediction gain nor
perceptual stream identity. Within-channel timing remains necessary for the
author's interval-order A/B; cross-channel relations supplement that evidence.
The observer remains offline, and local-history prediction, phrase, closure and
generation coupling remain open.

Local envelope prediction was then compared without assigning streams. Each
forecast targets the same 113 filtered amplitudes 300 ms later. Centered ridge
regression uses only completed examples, capped at 128, and begins with 32.
The comparisons use twenty observations per channel, two per channel, or two
observations of five adjacent bins with coefficients shared across bins.
Within each variant, sorted histories remove temporal order while retaining
values and regression dimension. One-observation regression, historical mean
and persistence supply additional controls. Missing input discards pending
forecasts and learned examples. This remains a bounded research model, without
a native real-time cost measurement.

The same twelve normal recordings supplied 2097 scored deadlines per variant.
Twenty-observation histories lost to one-observation regression in every
recording. Shared neighborhood regression improved on that control by
0.95–8.53%, but its squared-error reduction against equally shared sorted
histories was −0.04 to 2.06%. No variant met the screen of at least 5% reduction
against every control at all three seeds. These are development recordings;
there was no new held-out normal rendering. All three variants' full predictions
were subsequently replayed exactly.

After freezing the implementation, 42 controlled recordings at unused seeds
were compared under all three variants. The long history reduced error by at
least 89.10% against every control in the 21 prespecified repetition cases.
However, shuffled-interval recordings also retained gains over unordered
history. Their envelopes retain temporal constraints even when interval order
changes. Relative improvements on near-constant signals additionally need
their absolute error scale reported: the initial ratio-only screen counted
roughly 1e-11-scale errors as useful. A stricter signal-energy-relative variation
requirement excludes those cases without changing any normal-recording failure.
Shared neighborhood regression also retains bias on steady input. These
findings do not establish musical order, phrase or closure, and the candidate
stays offline. The next design review must preserve the author's distinction
between interval-order change and acoustic texture, rather than equating
envelope forecast accuracy with musical organization.

## 9.3 Upstream Revisions

*   **Trace model choices to cognition before treating them as DCC models.**
    DCC requires acoustic generation to be grounded in auditory cognition.
    Acoustic input, causal prediction and bounded storage alone do not establish
    that correspondence. The recent regression and interval-memory comparisons
    use engineering choices for model families, history lengths and thresholds;
    they have not been derived from a specified cognitive mechanism. A subsequent
    variable-memory diagnostic recovered candidates in all fifteen previously
    used interval-change cases, but that result does not supply the missing
    derivation. First specify the cognitive phenomenon and empirical constraints
    on representation, retention, interference, updating and time scale. Then
    select a mathematical approximation and distinguish its parameters from
    computational ceilings. DCC alone does not uniquely determine every number,
    and an ideal-observer benchmark does not identify the brain's implementation.

*   **Acoustic predictability does not identify musical order.** Filtered
    envelopes can retain predictive constraints after interval shuffling.
    Preserve a matched order-removal control and the author's independently
    heard distinction. Report absolute error scale beside relative gains;
    improvement on minute stationary fluctuations is insufficient evidence
    for a musical unit or generation coupling.

*   **Shared modulation is a relation, not source identity.** A common gain
    change can correlate distinct components. Constant energetic channels and
    missing input cannot supply the same evidence as independent modulation.
    Report availability explicitly, retain within-channel order, and test any
    proposed grouping through subsequent predictive usefulness rather than
    interpreting channel counts or correlations as Voices or phrases.

*   **Coordinate invariance and perceptual grouping are separate requirements.**
    Relative frequency can preserve a moving pattern while an aggregate mixture
    still supplies no supported sequential context. Establish which temporal
    relations survive mixing before assigning one sequence or a fixed number of
    streams. Shared activity is evidence to test, not a Voice identifier. A
    coordinate advantage needs equally transformed baselines.

*   **Report whether predictive context was actually supported.** A conditional
    model can outperform a weak baseline while almost always falling back to an
    unordered distribution. Compare against that fallback and expose context
    support before attributing gains to sequence memory. Joint timing/frequency
    gains also need their time and conditional-frequency contributions stated.

*   **Component energy rises are not source onsets.** Localizing energy in
    frequency can expose weak activity masked in a scalar amplitude trace, but
    interference, resolution limits and ridge crossings still affect the event
    stream. Preserve these distinctions when constructing temporal expectations.
    Tracking a spectral ridge does not establish a Voice, a melodic stream or a
    listener's musical unit.

*   **Observe temporal order separately from spectral change.** A flux peak can
    arise without a new energy attack, while an interval distribution can remain
    unchanged when the order changes. State what each observation retains and
    loses before attributing a prediction gain to musical organization. More
    context is not automatically better: the altered sequence can be simpler.
    Neither an error candidate nor the end of an audio file establishes closure.
    Report unavailable judgments separately: a missing calibrated threshold is
    not evidence that a transition did not occur.

*   **State persistence must allow internal motion.** A static destination makes
    a sustained change easy to demonstrate, but the author heard that destination
    as motionless. Compare transitions between two moving textures as well.
    Low difference between adjacent collections does not require unchanging
    frames, and successful change detection does not establish ongoing musical
    flow, an ending, or the value of the destination. In the subsequent moving
    comparison the author heard the same flow with added overlap: a change in
    simultaneous occupancy and a change in temporal organization therefore need
    separate controls. A detector of the former must not be reported as a
    detector of the latter.

*   **A change of acoustic statistics is a candidate representation, not a musical
    boundary.** Averaging can erase the difference between alternating and
    simultaneous sounds, while an unordered collection erases their sequence.
    Preserve frequency geometry where displacement matters, and expose the
    information each representation discards. Require observed prior stability
    before labeling a transition onset. A passing synthetic control does not
    resolve the normal samples in which that prerequisite remains unavailable.

*   **An almost-certain change is a weak test of musical expectation.** Moving from
    a short snapshot forecast to a longer first-passage horizon can make a
    no-change baseline easy to beat. Require information beyond empirical change
    frequency and immediate motion, and score timing and direction separately.
    Frequent texture fluctuations do not themselves identify a new musical unit.
    Censoring and repeated forecast origins must remain visible in the evidence.

*   **Sequence memory must add information beyond acoustic continuity.** A model
    can beat a history-only average while losing to the prediction that the current
    sound simply continues. Compare the same future evidence against persistence
    and a current-state-only model before attributing gains to order. Extracting
    features from continuous mixtures repairs observability, but does not by
    itself establish a useful prediction target or justify intervention.

*   **A successful solo control can rely on an assumption absent in normal music.**
    Segmenting sounds by silence produced just one acoustic episode in ordinary
    samples with continuous overlap. Event counts and successful execution do not
    establish musical prediction. Preserve the existing flow while redesigning
    which changes can be observed within the mixture. Added beat carriers,
    forced silence, or generator-side individual labels do not establish the
    missing perceptual capability.

*   **Waveform periodicity is distinct from the pitch of an identified source.**
    Nearby-tone interference can produce strong short-lag agreement. Even a
    multiple-period check cannot distinguish one missing-fundamental sound from
    several sources producing identical samples. Generator Voice IDs must not
    fill this gap in an audio-only ListenerTwin. State the feature and its
    availability limits, then separately test melodic or phrase identification.
    Likewise, absent samples are not evidence of silence: break continuous
    context at a detected gap and do not label sound already present on
    reconnection as a newly observed attack.

*   **Errors under initial ignorance differ from violations of learned expectations.**
    The initial contour predictor pooled large startup errors into calibration and
    missed later changes. Restricting calibration to forecasts made with sufficient
    prior contextual support improved change detection without changing the
    predictive densities. The support condition and error threshold are research
    settings, not established human learning counts or perceptual confidence.

*   **Predictive gain, change detection, and perceived closure are separate tests.**
    Learning the repetition after a change can restore mean predictive gain;
    that average does not locate the change or a phrase boundary. A midstream
    rest and an input truncated during that same silence yield identical gap
    candidates from their shared past. Future reentry and end-of-file must not
    supply evidence to the observer. Correspondence with the author's perceived
    boundaries and closure needs its own evaluation.

Current listening decision (2026-09-06): the author reported no collapse in response
to the current Sample 08 check. The invitation specified seed 21 and seconds 18–24;
the answer did not restate seed or playback range. Together with the earlier
naturalness response for 08, slight A preference for 09, and preserved flow in 12,
this supports adopting the current separation of cadence and hold length.
Absence of collapse in this audition does not establish persistent relations or
call-and-response across all conditions. The observations below retain their
original versions and comparison scopes.

Implementation results have revised the Manifesto's mechanism-level sketches while confirming its principles:

*   **A response through shared beat tracking is not necessarily a change in relative participation.** In a fixed-position control with three Voices and three seeds, withdrawing one Voice changed the remaining onset times while their assigned beat positions persisted. The shared beat itself responds to sound, so changed timing alone does not establish autonomous niches. The current participation policy changed relative positions in seed 42, but seed 1 did not exhibit stable positions under this within-cycle measure. This does not exclude non-periodic relations. Test common-position collapse, distinct-position persistence, and contextual relation changes separately from musical acceptance. These 18 conditions used a research renderer with metric holds tied to intrinsic periods; the production metric duration rule remains unchanged. See `docs/roadmap/rhythm-perception-evidence-2026-09-06.md`.

*   **Different temporal behaviors need not share one cadence-adaptation rule.** After a positive response to acoustic cadence with fixed holds in Sample 08, the author found baseline A slightly more natural in the flow-based Sample 09. A control without cadence adaptation exactly reproduced A for all three Sample 09 seeds. Improvement has therefore not been established across these works and timing modes; this is not a general prohibition on cadence adaptation in flow. Sharing acoustic perception and adapting every Voice's pace to a recurrence remain separate design decisions. The author subsequently reported that Sample 12 preserved its flow. The current presets retain cadence adaptation for entrained timing, while flow uses intrinsic renewal with acoustically informed waiting and omission.

*   **Cadence adaptation does not require duration adaptation.** The previous policy already separated a configured intrinsic prior from an executed-interval participation period; describing that distinction as an intrinsic-prior bug was inaccurate. Removing interval learning alone left the three Sample 08 seeds bit-identical. Adding acoustic cadence while scaling hold lengths changed the sound, and the author preferred the previous flow. A subsequent control retained acoustic cadence but fixed hold lengths to the intrinsic period; the author reported that it sounded more natural. The current implementation follows that separation. Seed, playback range, and an explicit preference over the baseline were not specified, so this is not full musical acceptance. Changing holds also changes predicted overlap and later acoustic feedback. Earlier short and 132-second statistics do not establish this new version's musical quality. The three-band event representation still does not identify individual sound objects or their relations. The temporal-niche goal includes different positions in a common flow, with call-and-response as one possible relation. See `docs/roadmap/rhythm-perception-evidence-2026-09-06.md`.

*   **A choice of onset time is not yet a choice of participation.** After the normal Sample 08 comparison, the author reported no perceived difference. The policy used in that comparison searched within 60 ms of each bodily due time; it did not choose to omit that event or end a sustained sound in response to its surroundings. All three tested seeds retain intervals near 0.5 seconds, while Sample 08 holds notes for roughly 1–1.5 seconds before release. That overlap proxy read only 0, 40, and 80 ms after a candidate, rather than the consequences across that sound's duration. These are confirmed limits of the action representation and comparison; they do not establish the perceptual cause of the author's result. Changed onset records and successful runtime tests therefore do not establish audible temporal niches. The subsequent implementation distinguishes acting now, waiting, and omission, and integrates an envelope estimate across the sound's persistence. A three-Voice short/long comparison with intrinsic-pacing controls and one Voice resting and returning tested that mechanism across three seeds. In all six participating conditions, the two remaining Voices changed their onset sequences after the withdrawal; intrinsic controls did not. Intervention and continuous-control audio were identical before withdrawal. This establishes acoustic responsiveness in those cases, but the author heard independently changing timing, without a handoff. The policy has no representation of which acoustic event a Voice answered or what followed its action; stronger omission or retiming alone is not the next acceptance criterion. Silence or uniform spacing alone is not success. The accepted separation of perception, body, and action remains the design direction; this particular policy has not passed musical acceptance. See `docs/roadmap/rhythm-perception-evidence-2026-09-06.md`.

On 2026-09-06, the author accepted the direction of separating temporal perception,
bodily readiness, and onset decisions. An audio-only campaign has now observed 42
stimuli under two priors. In the alternating and syncopated baseline conditions,
onset detection succeeds while beat confidence stays low; 1.4 and 2.6 Hz pulse
inputs also leave the unshaped estimate near its 2 Hz initialization over these
40-second trials. This identifies limits of the tested acoustic path, not causal
shares of Sample 08's closed-loop collapse.

An offline predictor compares delayed acoustic-event histories against an adaptive
event-rate model, scoring each hop before observing it. A switching mixture of
recurrence candidates now has a bounded Rust research implementation. It matches
the Python predictor on the archived observations and has been evaluated on 24
new audio inputs. Fixed observation windows make its evidence independent of
audio delivery chunks; missing audio invalidates history rather than fabricating
observed silence. It is now connected to the production habitat stream. Recurrence periods are not unique beat/tactus labels,
and model weights are not perceptual confidence. The comparison includes a small
loss on renewal input; it does not establish universal superiority or musical
acceptance. See `docs/roadmap/rhythm-perception-evidence-2026-09-06.md` for inputs,
results, limitations, and the remaining Voice and runtime work.

A first action-policy experiment remembers the acoustic features predicted around
its own participation times and searches near its bodily due time. Against fixed
audio, this changes actions without selecting a shared period, but some shuffled
inputs still concentrate those actions. Remembering a short shared context is
therefore insufficient evidence of temporal niches.

A subsequent 48-condition research campaign synthesizes short decaying tones and
feeds their actual audio back into the observer. It predicts three-band energy
overlap after subtracting a prediction of the actor's own synthesized sound. A
solitary actor then has no predicted competitor and does not start avoiding its
own repetition. Actor visit order also leaves actions and audio unchanged. In the
three-actor same-band conditions, overlap avoidance reduces or retains already
small late energetic overlap relative to context memory alone. One withdrawal/
return condition increases overlap, so this is not a universally improving rule.
The proxy does not establish perceptual masking or consonant fusion, and the
research synth was a separate test path. The subsequent production integration is described below.

The normal renderer now retains each participant's body audio and its habitat
contribution in reusable buffers. Both align to the observer's fixed windows,
including known silence before a mid-window birth. Retiring a participant removes
its history without attributing its remaining Tone tails to a new occupant of the
buffer slot. The mixed audio is unchanged by this tracking. PhonationEngine
separates candidate selection from acceptance; blocked candidates do not learn.
Tests cover shared-phase and hop-size independence of onset and note-off times,
flow renewal, the 200 Hz bound, and live clock updates. A first normal Sample 08
render completed with all twelve participants sounding. A subsequent campaign rendered Sample 08 with participation and zero coupling,
and Sample 09, for three seeds each. Device checks for seed 21 confirmed active
MOTU M2 routing, no generated-hop underruns or callback errors, and hop p99 values
of 1.410 ms (08) and 4.350 ms (09). These checks cover the sampled populations
and short performances; they do not establish larger/longer operation or musical acceptance.

The legacy `AttackPhaseMatch` reward applies to autonomous articulation attacks.
Gated execution disables those attacks and recharges emitted onsets with the
ordinary consonance multiplier of 1. No new metabolic reward was added to make
participation appear successful. Shared modulation and pitch-control paths still
need to be considered when interpreting the resulting audio.

*   **Self-prediction belongs in the counterfactual action model.** Shared sound includes the actor. Treating all predicted energy as a competitor would also penalize the actor's own expected sound. Applying the same temporal projection to shared and self-produced energy separates this effect in the research loop; it neither grants access to another actor's hidden plan nor schedules a common pulse. Power subtraction is approximate when waveforms interfere. The result is a constraint on the next production integration, not proof of musical acceptance or a replacement for auditory masking and fusion models.

*   **Dedicated synchronization support is a musical choice.** A dedicated beat carrier changes the onset evidence driving the shared meter. Routing it only to the habitat bus preserves that influence. It therefore belongs only where explicit synchronization is essential to the demonstration or assay; sample 12 now relies on its colony's onsets. A sustained pitch-reference anchor has a different role. Successful synchronization alone does not establish the musical value of a repeating accompaniment.
*   **Detected grouping needs an observed phase origin.** A measure ratio and confidence do not identify its strong beat. Subtracting the accumulated onset phasor angle from the unwrapped beat phase aligns emphasis with observed accents. The opt-in multiplier changes onset strength; later timing and survival can still change through feedback. A uniform-onset feedback control does not invent a measure. Audible grouping remains an author judgment.
*   **Shared tempo does not require a shared onset position.** In the pre-redesign Sample 08, the author hears both seeds 21 and 42 converge to one beat position and rejects that collapse unless it is intentional. The older Kuramoto core retains an individual `phase_offset`, but Gated production disables its autonomous attacks. Actual onsets come from `CouplingClock`, which targets the shared beat plus the script-level `microtiming`, zero for every Voice in this sample. Random initial phase does not preserve an individual phase niche. The Coupled path also bypasses social timing weights, and `avoid_neighbors()` acts on pitch. The disconnected path is confirmed, but reconnecting a fixed offset is only a proposed control, not a complete adaptive-habitat design. Nor does this establish that adaptive phase-space habitat selection was once fully implemented. See `docs/roadmap/phase-niche-audit-2026-09-06.md`.
*   **A temporal reference and a decision to sound are different contracts.** The adopted direction separates sound-derived timing expectations, a Voice's bodily readiness, and its choice to sound, sustain, delay, or rest. Phase can encode a periodic relation without becoming a symbolic score or a mandatory state for every Voice. A temporal niche is a context-dependent relation to surrounding sound, not a permanently assigned beat position. Shared analysis may supply evidence; it must not force all actions to the same phase. Intentional unison remains valid, and compulsory phase repulsion is not a general musical objective either. The observation and counterfactual action experiments above implement parts of this proposal; entrained/flow production now implements this separation. It does not replace the author's macro-level Scenario. See `docs/design-notes/rhythm-temporal-niche.md`.
*   **Onset concentration is not general rhythmic predictability.** The current beat confidence uses the first circular moment of detected onset phases. Equal weighted evidence at phases 0 and π cancels, even when the two positions form a repeatable pattern. The existing subdivision detector can capture higher moments, but `CouplingClock` reads beat confidence for its lock strength. The cancellation is a property of the statistic, not a measured causal decomposition of Sample 08; a doubled-tempo interpretation can also be legitimate. The proposed evaluation separates period hypotheses, relative onset relations, and actual presentation audio. It must test repeating omissions and multiple onset positions as well as a single pulse. A phase-corrected internal order parameter or maximal phase dispersion cannot establish musical success. This distinction is consistent with [missing-pulse evidence](https://pmc.ncbi.nlm.nih.gov/articles/PMC5490067/); [phase-inference models](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009025) offer one candidate treatment of uncertainty and missing events, not a settled neural implementation.
*   **Recovery and autonomous return require different evidence.** Fixed-coordinate scans directly demonstrate stimulus withdrawal and recovery; a flat mean state cannot establish recovery of a particular basin. In the 27-condition comparison on 2026-09-06, disabling only the mobile Population's adaptation removed all return candidates across the nine habituation-off, normal-recovery and slow-recovery cases, with temperature and crowding preserved. The compiled settings differed only in that flag; initial spawns matched and all five Voices survived and emitted late onsets. Returns in this fixture therefore depend on per-Voice adaptation; shared Habituation alone has not demonstrated recurrence. With adaptation enabled, normal recovery yielded more candidates with a preceding state decline, but total returns were not monotonic in recovery time. State decline, recovered basin rank and actual return remain distinct requirements, so autonomous closure stays PARTIAL. Sustained references must also outlive the measurement interval: pitch anchoring alone does not guarantee this. See the 2026-09-06 addition to `docs/roadmap/stage2-evidence-2026-09-05.md`.
*   **Bus separation does not freeze shared modulation.** Drone amplitude still reads the shared rhythm, so a presentation-only Drone is not an exact fixed-input control. The DCC research test uses fixed-duration SeqGate references in its scenario IR and verifies identical presentation WAVs and listener observations across gains. With the default maximum temperature bonus of 0.1, the tested gains show no clear resolution benefit; default coupling remains zero.
*   **The four-band table → a revisable temporal model.** The Manifesto sketches fixed delta/theta/alpha/beta bands with assigned musical roles. The current implementation uses an adaptive beat–subdivision–measure hierarchy with confidence (Section 4). Its exact oscillators, ratio candidates, and confidence statistic are modeling choices, not established universal neural machinery. Frequency bands or numeric phase coordinates are not inherently imposed scores; the design question is how evidence and local action interact. [Tempo-dependent neural responses can be reproduced by both oscillator and evoked-response models](https://academic.oup.com/cercor/article/35/9/bhaf258/8263582), so beat-frequency activity alone does not select the mechanism. The temporal-niche proposal preserves cognition-grounded generation while reopening those choices.
*   **Perceptual symmetry → the production-loop fixed-point requirement.** A terrain operation is ecologically meaningful only when a perceptual mechanism exists and the agents it attracts radiate spectra that reinforce it. Current tension controls therefore read the ecology-built terrain instead of warping it: movement tension is pitch-search *temperature*, and placement tension is a *relative consonance level*. See `docs/design-notes/tension.md`.
*   **Rate archaeology → observable time contracts.** Exposing raw energy pools and per-second rates made the composer balance dimensions that the ecology could derive. The implementation assay showed that the independent contract is nominal endurance, not `initial_energy` or `energy_cap`: energy can be normalized to $[0,1]$, while zero-fit drain is derived from endurance and the dissonance shape. Continuous recovery remains a separate time contract because per-second recovery and per-attack recharge have different dimensions. The same assay rejected collapsing `Entrain`, `Seq`, and `Drone`: their onset resets, death rules, telemetry, and render modulators are observably different, so one configured core would only hide the enum behind flags.
*   **Ambiguous object nouns → lifetime-bearing ontology.** `Material`, `Participant`, and a public draft Population each required prose exceptions because none named the lifetime it represented. The implementation now makes the transition explicit: `PopulationSpec + Placement --place()--> Population`, whose living members are Voices; all Populations sharing the Landscape form the Community. Population policy belongs to PopulationSpec, live control belongs to Population, and `place()` schedules founders immediately. `Species` is reserved until heredity and speciation give it an actual biological invariant. The naming is therefore guarded by observable identity: a Population survives member death and respawn, while a Voice and its generation do not.
*   **Sound-body range and perceptual coverage are different contracts.** Initial placement must respect its requested Hz bounds, including fallback sampling. An anchored Voice can retain a frequency outside the current analysis band but inside the synthesizer range. A Field placement with no overlap uses a log-uniform fallback inside the request; it makes no consonance claim there. Subsequent free pitch search still operates on the available Landscape.
*   **Missing audio is missing evidence.** Joining the audio before and after a dropped hop creates a signal that was never presented. Listener recovery therefore resets the spectral history, invalidates old observations and suppresses pressure until a continuous analysis window is available. The same unexpected gap ends deterministic runs, whose fixed-lag schedule cannot advance through recovery.
*   **Intensity density must be integrated before listener aggregation.** The spectral front end stores ERB density $d_i$ on Log2Space bins. Listener weights are intensity masses $w_i=d_i\,\Delta u_i$, where $\Delta u_i$ is the ERB integration width. Stability and per-bin best reachable gain use $\sum_i w_i x_i/\sum_i w_i$; the silence/evidence check uses the same total mass $\sum_i w_i$. Correcting the earlier density-weighted aggregation removes an unintended bin-width bias. It changes neither the consonance coefficients nor the tension formula, and adds no memory or expectation model. The implementation contract is recorded in `docs/design-notes/listener-twin.md`.

### 9.3.1 Temporal Memory: Cognitive Constraints and Observation Contract

Status: a design contract, not an accepted cognitive model. The target is remembering
the order of unequal acoustic intervals and forming expectations about subsequent
events. The author's interval-order A/B supports a heard change, not a phrase boundary.
The following primary studies constrain different tasks; their parameters cannot be
pooled into one universal memory length.

| Evidence | Constraint and limit |
|---|---|
| [Teki and Griffiths (2014)](https://doi.org/10.3389/fpsyg.2014.01329), interval reproduction | Precision deteriorated with load across one to four intervals. Test graded fidelity, not only an all-or-none item cutoff. This explicit recall task does not identify automatic sequence memory or a forgetting time constant. |
| [Andreou, Griffiths and Chait (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4389832/), sequence-offset MEG | Responses to non-arriving tones depended on temporal context; making rapid nonisochronous sequences relevant improved coding. Evaluate waiting without a new onset. Neural offset responses do not establish musical closure or a universal attention gain. |
| [Harrison et al. (2020)](https://doi.org/10.1371/journal.pcbi.1008304), PPM-Decay | Pattern length and presentation rate constrain memory models separately. Its 15-item buffer and 3.5-second half-life were manually fitted to rapid pitch sequences, not derived constants. The model assumes discrete symbols; extracting those from audio is outside its scope. Its parameters therefore do not transfer directly to our interval observations. |

**Representation choice.** The proposed mathematical object is an ordered auditory
trace with observation times, retrieval uncertainty, and a distribution over the time
of the next observable event. This is an inference from the requirements, not a neural
implementation established by those studies. Absolute elapsed time and sequence
position must remain separately accessible. Retaining relative intervals must not
remove the rate information needed to test memory limits. A detected spectral rise is
an acoustic observation, not automatically a Voice, note, or memory item. Do not feed
the observer the stimulus labels `short`/`long`, a known alphabet, or Scenario events.
The symbolic PPM model can be a comparison, but cannot supply this missing front end.

**Waiting contract.** Let `T` be the next observable onset's elapsed time from the
last onset, and let `S(u) = P(T > u | H)` be the survival probability of a forecast
issued from available history `H`. With continuous, valid observation and no new onset
through `u`, the probability of an onset in the following interval is

$$
P(u<T\leq u+h\mid T>u,H)=1-\frac{S(u+h)}{S(u)},\qquad S(u)>0.
$$

The evidence supplied by continued nonarrival is the increment
`-log2(S(u+h)/S(u))`, not an error that must wait for a subsequent onset. Evaluate
arrival and nonarrival consistently without counting already scored waiting twice.
This probability identity introduces no cognitive time constant. It still requires
a justified predictive distribution. If that distribution assigns nonzero mass to
no future event, retain it; do not normalize a finite forecast horizon to certainty.
An exhausted or unsupported forecast is unavailable, not proof of closure.

Observed nonarrival requires sufficient detection support. Unobservable events,
missing audio, and silence are different conditions. Missing input censors the active
forecast at the last valid observation; EOF adds no terminal event. Preserve evidence
already obtained before censoring. Keep the issued forecast immutable for scoring;
subsequent memory updates may change later forecasts, not the earlier prediction.

**Memory and numerical limits.** Context span, learned-example retention, retrieval
fidelity, and storage capacity are distinct quantities. The next comparison must
separate a finite-buffer hypothesis from gradual loss of precision, and separate
elapsed-time effects from intervening-event effects. It must not simply rename the
old 6/128/32 limits. Candidate retrieval laws and their fitted parameters remain open.
A common decay multiplier cancels when weights are normalized:

$$
\frac{a(u)w_j}{\sum_k a(u)w_k}=\frac{w_j}{\sum_k w_k},\qquad a(u)>0.
$$

Therefore, decaying stored weights alone need not reduce predictive certainty.
Retrieval noise, precision loss, or declining support relative to a prior are competing
mechanisms to test, not three additions automatically required together.

| Quantity | Required justification |
|---|---|
| Retention duration and interference sensitivity | Fit an identified auditory task, report uncertainty, and test transfer to interval order. Neither the runtime hop nor a convenient window identifies them. |
| Context size and retrieval precision | Specify what counts as an item and compare load/rate effects. A cycle length in an assay does not identify human capacity. |
| Observation step and prediction horizon | Quantify timing resolution, latency, and retained tail probability. These are numerical choices, not phrase durations. |
| Maximum stored traces | State the truncation rule and compare against larger limits while holding the cognitive parameters fixed. Resource saturation must be reported rather than interpreted as forgetting. |

**Current counterexample and implementation order.** A reproducible probe of the
archived variable-memory model used 81 synthetic onset observations followed by 60
seconds of valid no-onset observations at 10 ms steps. It produced no waiting report;
history, context, model evidence and the pending forecast were unchanged. This exposes
missing clock-time and nonarrival mechanisms, not a measured human forgetting curve.
The probe and source hashes are retained under
`target/phrase-expectation/2026-09-07-dcc-temporal-contract/`.

The waiting contract is implemented offline in `scripts/evaluate_temporal_waiting.py`,
using the existing engineering interval distribution as a test input. It reports
conditional onset/nonarrival likelihoods, preserves pre-gap evidence, censors missing
input and the recording end, and marks numerically unresolved tails unavailable.
Seven tests cover probability partition, accumulated likelihood, causal prefixes,
waiting, missing input, initial silence and invalid/tail inputs. The full Python suite
passes 157 tests. Replaying two existing acoustic observation streams scores 61 onsets
and 2440 no-onset windows each, with no unresolved model scores. These are software
checks, not cognitive calibration or new listening evidence.

Next compare retrieval mechanisms with event count and elapsed time varied
separately, observed silence versus missing data, recurrence after interruption, and
the existing order-change versus texture controls. Fit cognitive parameters against
the specified task before assessing normal recordings; label cross-task extensions
explicitly. Waveform access must remain causal. Boundary/closure requires additional
evidence and remains a separate output. Generation coupling follows useful passive
observation and the existing load checks; it must preserve the accepted temporal
participation behavior. This contract does not close Stage 2, Stage 3, or Stage 4.

### 9.3.2 Reconstructing the Published Memory Constraint

The [PPM-Decay analysis archive, v0.2.0](https://github.com/pmcharrison/pdec-analysis-2/tree/v0.2.0)
provides responses and model summaries for the rapid pitch-sequence task.
`scripts/evaluate_memory_constraints.py` reconstructs its participant exclusions,
trial flags, participant/block STEP correction, and within-condition outlier rule.
The 12,000 original trials yield 23 participants and 4,439 retained RAND-REG hits,
matching the paper's reported count. All six condition counts also match those of
the archived unmodified PPM output. This is a reanalysis of the original task, not
a new interval-memory experiment.

Using the archive's mean of participant medians, the six published model configurations
have the following errors against the six reconstructed condition means. These are
comparisons of already fitted, archived outputs; we did not rerun the models or fit
Conchordal parameters.

| Archived configuration | RMSE, additional tones after the first full cycle | Excluded model lags |
|---|---:|---:|
| Original PPM | 7.844 | 0 |
| Exponential decay | 5.782 | 0 |
| Retrieval noise added | 3.030 | 22 |
| 5-item buffer | 3.376 | 25 |
| 10-item buffer | 2.539 | 15 |
| 15-item buffer | 0.973 | 22 |

The parameter sets were tuned on these participants and several parameters change
between configurations. Their ranking supports a comparison of memory mechanisms;
it is neither independent validation nor an isolated causal effect of buffer size.
Invalid model lags were omitted by the upstream code, so report those omissions
beside the conditional errors. Its fixed context-order bound of four and the fitted
15-item buffer are different quantities; the buffer is not a 15-event predictive
context. The analysis specifies `ppm` v0.1.1, which was retrieved separately from
the current package source and checked against Git blob hashes.

**The measurement unit belongs to the model contract.** At equal 0.5-second cycle
duration, the paired contrast between 20 × 25 ms and 10 × 50 ms is +6.579 additional
tones, but −0.0483 seconds. These are different response variables, not contradictory
measurements. A single contrast in tone counts cannot by itself identify a duration
cutoff. Retain both units, specify the detection/readout rule, and evaluate the full
rate-by-load pattern before identifying a memory mechanism.

**Aggregation also matters.** The archived fitting code uses participant medians;
the paper describes means, and its plotting code uses a different group summary.
Both participant-median and participant-mean routes were retained. The 15-item model
has the smallest conditional RMSE in either route, but neither reconstruction's
participant-bootstrap interaction interval exactly reproduces the published interval.
Do not claim full statistical reproduction from matching trial counts. The bootstrap
resamples participants, not trials, and its NumPy draws differ from the original R run.

The new parser/analysis has four tests; the full Python suite passes 161 tests.
Source snapshots, verified downloads, both aggregate routes and complete contrasts
are stored in `target/phrase-expectation/2026-09-07-auditory-memory-evidence/`.
The numerical reference is now compared below. This evidence motivates a memory-model
comparison, not a production default of 15 items or 3.5 seconds.

### 9.3.3 Numerical Reference and the Meaning of Memory Duration

`scripts/ppm_decay_reference.py` implements an offline reference of
[`ppm` v0.1.1](https://github.com/pmcharrison/ppm/tree/v0.1.1). It retains the original
buffer-expiry rule, trace decay, retrieval noise and context interpolation. Its input
is a discrete symbol and its presentation time; parameters are explicit and none is
a Conchordal default. Stored traces remain unbounded, as in the original research
implementation. Cognitive retrieval limits and allocated storage are separate.

The original C++ numerical methods were compiled with R bindings and serializers
replaced by standalone adapters. The first original trial in each of 24 conditions
for participant 1 was compared under all six published configurations, alongside
four synthetic boundary cases. Across 148 cases and 19,808 predictions, the maximum
absolute probability difference was `1.55e-15`, and the maximum information-content
difference was `3.91e-14` bits, within the declared `2e-12` tolerance. The Python model
received each original C++ retrieval-noise increment. This establishes numerical
agreement for those cases, not equality of random generators, the full R API,
participant reaction-time readout, or independent cognitive validation.

**Decay half-life is not total retention time.** In the archived Experiment 3
configurations, the item buffers have a `buffer_length_time` of 1,000 seconds.
The 15-item model's 3.5-second half-life starts after a trace leaves the buffer.
Direct deterministic-weight queries to both implementations show that a single
stored symbol retains weight 1 after 60 seconds without intervening symbols. If
15 other symbols arrive first, its weight drops to 0.6 at item expiry and to 0.3
3.5 seconds later. This is a property of that configuration, not an observation of
human quiet retention. The paper's illustrative Figure 10 uses a two-second time
limit; the archived continuous-tone task does not distinguish these limits when
item expiry occurs first. Neither value transfers automatically to musical pauses.

**Prediction identity and arrival time are different targets.** This reference
predicts the identity of a symbol at its supplied presentation time. It does not
forecast when that symbol will arrive or score its non-arrival. An interval extension
must preserve the issued-forecast contract in §9.3.1: later arrival times cannot
retroactively change the forecast being scored. Acoustic event encoding, continuous
interval uncertainty, quiet retention and intervening-event interference remain
explicit cross-task assumptions to test. Numerical agreement alone does not select
a model of phrase, closure or listener-to-generator coupling.

Seven regression tests cover upstream vectors with paired noise, time/item expiry,
the original exact-boundary distinction between learning and prediction, causal
identity prediction, invalid-input atomicity and the quiet-retention example.
The full Python suite passes 168 tests. Inputs, executable hashes, complete outputs
and the standalone adapters are archived in
`target/phrase-expectation/2026-09-07-ppm-reference/`.

### 9.3.4 Continuous Interval Retrieval: Implemented Hypothesis and Limits

`scripts/evaluate_interval_retrieval.py` now tests a continuous extension. A completed
interval is represented by its log duration and the time at which it became
observable. No short/long labels, score events or known pattern periods are supplied.
Each past episode contains a context and its observed successor. Time and intervening
intervals independently determine its exit from a strong buffer; thereafter its
retrieval weight decays. Contexts that could not fit inside that buffer are excluded.

For context order `k`, continuous cue similarity multiplies the retained episode
weight to give `a_j`. The successor density is interpolated with the shorter-context
density:

$$
q_k(y)=\frac{\sum_j a_j K_{\sigma_y}(y-y_j)+\alpha q_{k-1}(y)}
{\sum_j a_j+\alpha},\qquad y=\log_2(T),\quad \alpha>0.
$$

The lowest-order fallback is an explicit broad prior. Consequently, fading all
episode weights reduces their support against that prior instead of cancelling
under normalization. These continuous cue kernels and prior competition are new
modeling hypotheses, not numerical equivalents of symbolic PPM-Decay or an identified
neural mechanism. They do not yet reproduce the interval-reproduction study's
load-dependent precision, primacy and recency results.

All parameters must be supplied. The development comparison varied buffer capacities
4/8 intervals, time limits 1/4 seconds, half-lives 1/4 seconds and cue widths 0.07/0.14
in log2 duration: 16 settings. The other explicit choices were context orders 0–4,
post-buffer weight 0.6, outcome width 0.07, prior mean −1 and width 2 in log2 seconds,
and prior mass 1. These values form a sensitivity experiment, not cognitive estimates
or production defaults. Storage remains unbounded and independent of the modeled
buffer capacity. No real-time port has been made.

The original acoustic observer was run on the exact 49-onset A/B waveforms heard by
the author. All 49 onsets were observed in each, within 6.29–16.50 ms of the supplied
stimulus times. Prediction uses only those acoustic observations; stimulus times are
used afterward to check the observer. Prechange outputs matched for every setting.
During the first three seconds after the change, B's complete waiting-plus-arrival
loss exceeded A's by 0.006–1.090 bits per interval for the highest interpolated order.
The same model exceeded its marginal prediction on later B intervals by 1.017–1.247
bits per interval. The sensitivity range matters: the pair does not identify a
memory duration, calibrate a change detector, or select a preferred setting.

**An unavailable negative control is not a correct negative.** The author described
the moving-texture comparison as unchanged flow with added overlap. Its observation
path supplied no scoreable intervals in the 9–22 second listening window, for either
the steady control or the moving texture. Therefore this interval model has not
demonstrated the intended distinction across both kinds of sound. Continuous texture
needs an observation and memory representation appropriate to its audible movement;
changing interval-memory parameters cannot repair absent observations.

Issued forecasts remain immutable. Waiting and arrival are scored jointly; missing
input censors the previous forecast and clears episodes whose intervening-event count
is unknown. Earlier scores remain. EOF supplies no closure evidence. Ten new tests
cover these contracts, continuous matching, time/item retention, prior fallback,
proportional time rescaling and unresolved tails. The full Python suite passes 178
tests. Plans, full outputs, exact input identities and both observation checks are in
`target/phrase-expectation/2026-09-07-interval-retrieval/`. Cognitive parameter
identification, continuous-texture coverage, bounded implementation and generation
coupling remain open.

### 9.3.5 Contract for Perception, Memory and Action under DCC

The design review separates three origins of model choices. Auditory representations,
retention and expectation require perceptual evidence. An individual's intrinsic pace,
sound body and activity preferences belong to the authored ecology. Analysis windows,
storage bounds and scheduling rates are numerical choices. A neural-band name does
not establish the origin of a time constant. The Manifesto's fixed assignment of musical
roles to four neural bands remains a revisable mechanism sketch, not the contract below.

| Contract | Required state and update | Current implementation and gap |
|---|---|---|
| P: Continuous auditory evidence | Preserve timestamped frequency/amplitude trajectories and the distinction between observed constancy, undetected activity and missing input. Onsets and relation summaries are derived views. Keep the ordered evidence that a window mean or covariance discards. | Landscape, DorsalStream and research component tracking supply parts. Log2-frequency history is now connected to native shared-habitat and presentation observations (9.3.44). Component modulation is observable on the author's sustained-texture controls; perceptual grouping and continuity remain unvalidated. |
| M: Retained relationships and expectations | Retain uncertain histories and candidate relationships, with separate elapsed-time and interference effects. Issue predictions from available evidence and keep scored forecasts immutable. A supported periodic relation can supply a phase coordinate; phase is optional. | Recurrence forecasts, actual-energy feedback and two-window experience memory have partial production connections. Continuous RMS history with past age and known coverage is also attached to decision-time evidence (9.3.36). Three-band compression loses some spectral configurations and ordering distinctions, so it is not sufficient evidence for relationship memory (9.3.43). History-conditioned energy prediction and observed comparison with recurrence now feed participation, but longer horizons retain regressions (9.3.37). Matching outcomes and unscored counts are now audited at each actor's real issuance contexts (9.3.38). Fine-band histories, associations, credit and cross-band prediction remain research mechanisms (9.3.8–9.3.14). Native-history relational forecasts and retained recurrent contexts also remain research candidates without uniform benefit (9.3.45–9.3.46). Conditional acoustic candidate retention and revision do not establish relational memory (9.3.25–9.3.31). Cognitive retention, interference, updates from non-arrival, stream identity, boundaries and closure remain open. Analysis windows and the two-window offset are not phrase duration. |
| A: Situated action and observed consequence | Evaluate the perceptual consequences of a local action, beginning with sound/wait choices. Combine these with the individual's body and ecological preferences. Record selected, executed and acoustically observed outcomes separately. | TemporalParticipation separates selection, execution and two subsequent observed context windows, updating memory used in later candidate comparisons (9.3.35). Shared continuation weights learn from matching-lead actual windows, while known self PCM is removed before observing the surroundings (9.3.32–9.3.34). Owned-state sound/wait predictions and selected-outcome revision remain research work (9.3.21–9.3.31). Experienced context does not identify intervention effects or action values. Long-horizon reliability, local calibration and action-value learning remain open. |

The action environment can use observed roughness/harmonicity and supported temporal
relationships without first assigning a definitive stream identity. Generator knowledge
of its own intended action belongs to prediction, while ListenerTwin observations remain
grounded in presentation audio. Correlation is neither a Voice identifier nor a calibrated
probability of perceptual grouping. Missing coverage contributes no invented certainty.
Do not collapse consonance, synchrony and predictive accuracy into one universal reward:
an unchanging signal can be perfectly predicted. The authored ecology must still admit
fusion, alternation, independent activity and intentional unison. Scenario retains the
composition's large-scale direction.
Internal representations must also perceive and retain long context, repetition,
variation and return, and make them available to local action. P/M/A applies across
timescales; the contract alone does not implement their cognitive models (9.3.55).

**First enforced boundary.** `GeneratorModel` now returns no terrain prediction for
queries before the latest observation or beyond its declared horizon, including when
only one observation exists. Previously, the horizon capped the damping exponent while
the extrapolation distance continued growing, and a single observation could be returned
indefinitely. An unsupported next gate now supplies no prediction and the existing
phonation fallback is neutral. Observations that resume in range can restore prediction.
The current theta/delta-based horizon remains an engineering rule, explicitly documented
as such; enforcing its boundary does not validate its cognitive interpretation.

Two regressions failed before this correction and pass afterward. Full Rust validation
passes 712 tests with seven ignored; Clippy and release builds pass. Current Samples
08, 09 and 12 at seed 21 produce byte-identical WAVs before and after the change. This
is a bounded regression comparison, not a new audition or runtime-load validation.

Existing continuous observers were also applied to the exact steady and moving-texture
controls. All 130 reports per waveform in the 9–22 second listening window supplied
component relations, with 3–4/4 selected channels and no channel truncation. This repairs
the availability problem of an onset-only input for this representation, but does not
prove the author's judgment of unchanged flow. The next memory contract must retain
the underlying ordered trajectories and their uncertainties, rather than treating those
correlations as recognized streams. Specification and checks are archived in
`target/phrase-expectation/2026-09-07-dcc-state-contract/`.

### 9.3.6 Preserve Observation Order before Choosing Memory Duration

A temporal model must receive the observed trajectory independently of thread scheduling.
The existing analysis worker processed each audio hop, but published only the final
Landscape in a backlog; the listener receiver also kept only its newest queued snapshot.
Consequently, listener habituation depended on how observations were batched for delivery.
A regression with 24 queued audio hops delivered one observation. A second regression
showed that batching sound and its following silence could discard all of the sound's
effect on habituation. Changing a retention time cannot repair this loss of evidence.

The generator's current-state path and the listener's observation path now have explicit
delivery policies. Listener analysis publishes each valid Landscape in order, and the
runtime observes every received frame before exposing the latest ListenerState. Each
observed hop, including observed silence, advances habituation by `hop / fs`; neither
empty polling nor an input-gap notification invents auditory exposure. The listener's
existing 64-hop input queue and four-result output queue remain bounded. A receive pass
processes only its initial queued batch. Analysis backlog collection is likewise capped
at its initial queue length, so an active producer cannot extend a batch indefinitely.
Ordered publication can wait for its consumer; closing that consumer releases the worker.
Real-time input overflow still invalidates listener evidence and requires NSGT refill.

Tests compare all 24 per-frame spectra with direct sequential analysis at output capacities
1, 4 and 24, and verify shutdown while publication is backpressured. Receiving observations
in batches of 1, 3 or 24 now produces bit-identical habituation states. Gap/refill checks
cover both delivery policies. The full Rust suite passes 714 tests, with seven ignored;
Clippy passes. This establishes observation-order and update-clock contracts, not a
continuous-relationship memory model. Retention under unknown input, perceptual grouping,
boundary and closure remain open. Plans and checks are stored in
`target/phrase-expectation/2026-09-07-listener-observation-order/`.

The release build passes. All before/after renders explicitly enable reporting and
therefore listener analysis. Samples 08, 09 and 12 at seed 21 retain identical WAVs and
listener/pressure/population traces. Two probes with habituation and DCC enabled were
each rendered twice per binary; both repeats and both versions agree on those outputs.
These fixed-lag render comparisons preserve the ordinary behavior while the forced
backlog regressions exercise the correction. They do not establish real-time device
performance, cognitive validity or a musical improvement.

### 9.3.7 Parallel Histories for Regularity and Cross-channel Relations

[Sollini et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9411505/) manipulated
within-band predictability and across-band coherence during a precursor, then measured
subsequent tone detection. Our design inference is to retain individual continuity and
relations between components in parallel. Grouping must not be a prerequisite for all
sequence memory. Regular individuals at different time positions and components sharing
an irregular trajectory require separate descriptions, rather than a common synchrony
or predictability value.

**Input representation.** A reconstruction of the background timing adds those two
crossed controls. Nine 0.8-second conditions at three seeds were observed both fresh and
after 2.24 seconds of actual silence. Both observers read the same saved PCM16. The
prelude aligns the 240-sample research and 512-sample native hops; it is not a cognitive
retention time. The assay uses 24 kHz, uncalibrated levels and no target tone. It does
not reproduce human detection thresholds.

The existing narrow-ridge observer detected no components in any of the 54 conditions,
including the primed cases. Native NSGT channels contained energy, but their different
window lengths and delays produced negative uncorrected correlations even for synchronous
regular pips. A spectrum suited to roughness and harmonicity therefore cannot be directly
interpreted as temporal coherence. The Rust evidence exporter processes complete hops
only, leaving the final partial hop pending instead of adding silence. Its full-FFT-window
flag does not describe individual filter latency, runtime eligibility or perceptual
confidence.

A research observer now uses the fourth-order, one-ERB
[Hohmann gammatone model](https://doi.org/10.5281/zenodo.20745033).
For center frequency \(f_j\) and sample rate \(f_s\), its transfer function is

\[
E_j=24.7+f_j/9.265,\quad
r_j=\exp\!\left[-\frac{2\pi E_j}{(5\pi/16)f_s}\right],\quad
a_j=r_j e^{2\pi i f_j/f_s},\qquad
H_j(z)=\frac{2(1-r_j)^4}{(1-a_jz^{-1})^4}.
\]

It reports 1 ms RMS envelopes of the complex responses on 113 Log2-aligned channels,
without requiring a ridge or onset. Constant signals and observed silence remain
observations; an input gap is a separate record and resets the filter approximation.
Unknown prehistory is not observed silence. Order and bandwidth come from the reference
model, while grid density and reporting interval are numerical choices. Filter decay
belongs to auditory preprocessing, not musical memory or phrase duration. This linear
approximation is not a reproduction of auditory-nerve responses or individual hearing.

The authors' unchanged C++ all-pole calculation was exercised with an independently
written coefficient and RMS wrapper. Across 24/48 kHz, two grids and three inputs,
12 conditions and 90,270 values agreed within \(5.17\times10^{-15}\) absolute error.
This does not reproduce the complete Matlab software or synthesis path. After freezing
the implementation, 54 conditions at three unused seeds passed all 18 comparisons
separating regularity from shared variation. In the 250–500 ms window, the ranges across
the five channels and signal/flanker pairs were:

| Input | Within-band correlation at a 50 ms lag | Signal/flanker zero-lag correlation |
|---|---:|---:|
| Regular and synchronous | 0.998–1.000 | 0.459–0.897 |
| Regular individuals with displaced relative timing | 0.998–1.000 | −0.503–−0.114 |
| Shared irregular trajectory | 0.170–0.619 | 0.453–0.898 |

The frequencies and lag are known-stimulus diagnostics, not inferred beats. The two
exact texture controls for which the author heard unchanged flow also yielded continuous
envelopes at every one of 13,001 observations in 9–22 seconds. Availability does not
establish flow identity or grouping. Production phonation and DCC coupling are unchanged.
Plans, PCM, numerical comparisons and full channel observations are retained in
`target/phrase-expectation/2026-09-07-grouping-cue-pcm/`. Uncertain memory of continuity
and relative lags, boundary/closure and action updates from observed consequences remain
open.

### 9.3.8 A Candidate Temporal History with Separate Observation Coverage

A research history now receives continuous envelopes using the temporal representation
proposed by [Shankar and Howard (2012)](https://sites.bu.edu/tcn/files/2015/12/ShankarHoward12-NeuralComp.pdf).
The proposal applies a Post inverse to a Laplace representation of past input, with
temporal spread proportional to elapsed time. No common period is required. This is a
candidate history representation, not a reduction of melody, phrase and form to one
mechanism.

For present time \(t\), past delay \(d\geq0\), preferred response delay \(\tau>0\)
and Post order \(k\), equations 2.2 and 2.3 imply the kernel

\[
s=k/\tau,\qquad
K_{\tau,k}(d)=\frac{s^{k+1}d^k e^{-sd}}{k!},\qquad
T_j(t,\tau)=\int_0^\infty K_{\tau,k}(d)f_j(t-d)\,\mathrm{d}d.
\]

Its integral is one, its mean delay is \(\tau(k+1)/k\), and its standard deviation
is \(\tau\sqrt{k+1}/k\). Increasing \(k\) sharpens temporal position. This width
is neither note duration, a firing period, nor the lifetime of a retrieved item.

Our numerical realization uses an equivalent \(k+1\)-stage continuous-time cascade,
not the paper's proposed neural inverse circuit or finite-difference implementation.
Envelopes are approximated as constant within each observation interval, and the state
transition is integrated analytically. Partitioning a constant interval leaves its
result unchanged. Small input coefficients are computed with a positive series to
avoid losing known contributions through complement cancellation.

Observation coverage is an additional contract. With \(q=1\) for observed intervals
and \(q=0\) for missing ones, the implementation separately retains \(K*(qf_j)\)
and \(K*q\). These are the known input contribution and the observed fraction of the
kernel, respectively. Coverage is neither perceptual confidence nor a probability
distribution for the missing sound.

| Input status | Known contribution | Observation coverage |
|---|---|---|
| Observed silence | Advance existing history without adding sound | Include the observed silent interval |
| Missing input | Advance existing history without inventing sound | Exclude the missing interval |

A gap does not erase earlier evidence. The known contributions can agree while their
coverage differs. Reading a report does not advance time, and EOF is not a boundary or
closure event.

The example explicitly uses \(k=12\), also used in the paper's illustrations, and
eight logarithmically spaced delays from 0.125 to 16 seconds. These are not parameters
fitted to the author or an auditory task. All 1 ms observations advance the state;
reports are emitted every 10 ms. The state for 113 channels plus coverage contains
11,856 scalars, or 94,848 bytes. Coefficients and work arrays require additional space;
this is not a real-time performance claim. Retained state size does not grow with
performance duration.

Checks cover analytic step responses, independent kernel quadrature, interval partition,
joint time scaling, opposite earlier order with equal present activity and totals,
and silence versus missing input. The implementation also processed 199,318 observations
from 58 existing audio inputs. In six pairs with common endings, the last envelopes
were identical while their different precursors remained represented. Author A/B histories
agreed up to the first different PCM sample at 6.544041… seconds; the first differing
report was at 6.55 seconds. This verifies causal updating, not a detection latency or
phrase recognition. Applying the history to the two texture controls does not classify
the unchanged flow heard by the author.

All 201 Python tests pass. Rust sources and standard binaries remain unchanged from
the preceding section's verification. Inputs, outputs, explicit parameters and checks
are stored in `target/phrase-expectation/2026-09-07-temporal-history/`. The next section
addresses within- and cross-channel associations. Item interference and competitive retrieval,
identification of temporal precision and range, predictive utility, boundary/closure and
generation coupling remain open.

### 9.3.9 Associations from Temporal History and Recall Issued before Input

The research implementation now includes the associative rule in section 2.4,
equations 2.11–2.14 of [Shankar and Howard (2012)](https://sites.bu.edu/tcn/files/2015/12/ShankarHoward12-NeuralComp.pdf):

\[
M_{i,\tau,j}(t)=\int_0^t f_i(u)T_j(u,\tau)\,\mathrm{d}u,
\qquad
p_i(t)=\sum_j\int M_{i,\tau,j}(t)T_j(t,\tau)g(\tau)\,\mathrm{d}\tau.
\]

Here \(M\) accumulates co-occurrence of history and subsequent input; \(p\) is recalled
activity. We use \(g=1\), as in the paper's behavioral applications. Equal weights on
log-spaced age cells would implement a different measure, so recall uses trapezoidal
weights in seconds. The finite age domain and quadrature remain numerical approximations.

For piecewise-constant input, the implementation analytically integrates the history
throughout each learning interval. It does not multiply the endpoint history by the
entire interval. Subdividing a constant-input interval preserves learned associations
and integrated coverage. The preceding history implementation's endpoint states are unchanged.

The missing-input contract extends the original rule. A gap adds no association and
does not erase previously learned weights. Observed silence adds no sound co-occurrence
but increases known coverage. The additional per-age quantity
\(\int q(t)(K_\tau*q)(t)\,\mathrm{d}t\) records joint observation coverage of the current
input and history. Neither the weights nor recall are divided by this quantity.

Recall is serialized before reading the next input's kind, timestamp or content.
Its target is the next 1 ms observation interval, not an identified cognitive forecast
horizon. The later observation is a separate outcome. A discovered gap censors an issued
forecast; EOF leaves it pending instead of inventing silence or closure. Contributions
with \(i=j\) and \(i\ne j\) are reported separately. Cross-band association does not imply
cross-Voice association: multiple components of one sound can also contribute.

Checks cover independent time quadrature, partition invariance, age-quadrature convergence,
cue/successor learning, missing input, and immutable issued recall. In the ideal impulse-pair
limit with \(g=1\), recall peaks at \(kD/(k+1)\), rather than exactly at the trained delay
\(D\). Finite-pulse controls at 0.25, 1 and 4 seconds approach that location. A history
cell's age and the peak of associative recall are therefore different quantities.

The implementation processed 110,998 observations from 12 existing audio inputs.
Of 11,110 issued recalls, 11,100 have observed outcomes and ten remain pending at EOF.
Four fresh/silent-primed pairs have identical known recall contributions and different
coverage. Author A/B recall agrees throughout the common PCM prefix; the first numerical
difference is at 6.55 seconds. The two texture inputs also agree through their common
12-second prefix. These are causal update checks, not flow-change or flow-identity decisions.

The audio run explicitly uses \(k=12\), 65 cells over 0.0125–16 seconds and 113 bands.
This numerical domain spans the existing short pulses and longer contexts; it does not
identify cognitive retention. Association storage has 829,985 scalars, or 6,639,880 bytes,
independent of performance duration. History, scratch space and output logs are additional.
All 211 Python tests and 714 Rust tests pass, with eight Rust tests ignored. Production
Rust and standard binaries are unchanged. Inputs, complete outputs and checks are stored
in `target/phrase-expectation/2026-09-07-temporal-associations/`.

The additive rule has no forgetting, competition or weakening by counterevidence.
Declining temporal precision of history does not specify the lifetime of acquired
associations. Scaling input amplitude by \(a\) scales \(M\) by \(a^2\) and recall by
\(a^3\): recall is neither probability nor predicted amplitude and cannot directly supply
DCC pressure. The next section examines a candidate for cue competition. Counterevidence,
calibrated arrival probabilities, cognitive parameter identification, utility in ordinary
performances, phrase/closure, and learning perceived consequences of local action remain open.

### 9.3.10 Cue Competition and the Limits of Rate Predictions

[Goh, Ursekar and Howard (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8944185/)
propose credit learning to limit double-counting of overlapping cues. We independently
implemented its calculation as a research component. Inputs are histories of identified,
counted events and pretrained pairwise rates. Continuous auditory amplitudes from the
preceding sections have not been reinterpreted as event counts or connected to this component.

Given baseline event rates \(\Lambda_i\), cue credit \(C_{ij}\), and history projected
\(\delta\) into the future without adding new events, equations 2.2–2.3 compute

\[
P_i(\delta;t)=\Lambda_i\exp\!\left[
\sum_j\int C_{ij}(\tau)\widetilde T_j(\tau;t+\delta)\,\mathrm d\tau\right].
\]

For the prediction just before a cue, \(P^-\), and the cue's pairwise prediction, \(m\),
equation 2.7 updates \(G=e^C\) by \(G\leftarrow(1-\alpha)G+\alpha m/P^-\).
A stronger prior prediction reduces the additional credit assigned to the cue. This is
an arithmetic update of the gain, not an arithmetic interpolation of log gains.

Our implementation computes this update in the log domain. It keeps extreme finite
log rates without exponentiating intermediate ratios. Projected histories, pairwise log
rates, baseline rates, learning rate and finite age grid are explicit inputs, not fitted
cognitive parameters. The unit-event mass within the numerical domain is returned and
is not silently normalized at its edges. Tests cover neutral credit, simultaneous-cue
geometric means, reduction of redundant credit, fixed-delay analytic predictions, grid
convergence and immutable outputs.

We compared numerical components against the authors' [public code](https://gitlab.com/varunu/leaky-integrator-credit-learner)
at commit `9d0f2122b41f227fd9be6fa7dfd3dabf1d89a829`, using identical input tensors.
Both calculations receive analytic unit-event kernels and histories projected by the
reference implementation. Across 432 conditions, 64,800 prior and cue-derived rate
values agree within a maximum relative difference of \(7.79\times10^{-14}\). This
does not reproduce the inverse circuit, first-event initialization, gradient clipping,
the complete training protocol or the paper's behavioral demonstrations.

Gain updates differ in 79 conditions. The reference adds a fixed \(10^{-20}\) to the
denominator and evaluates \(G\leftarrow G+\alpha(r-G)\). The correction matters for
extremely small prior rates; at \(\alpha=1\), subtraction can also round a small positive
target gain to zero. The latter occurs in 199 elements, with none in the new implementation.
The initial failed comparison and all outputs are retained. A separate grid comparison
shows decreasing inverse-approximation error from 17 to 200 cells. A projection assay
using the published tutorial's domain and 200 cells still has relative L2 difference
about 0.170 against the analytic kernel. This does not invalidate the whole paper or
establish an error rate across all settings. The existing cascade history is unchanged.

Two counterexamples constrain further use. First, the normalization of cue-derived
rate \(m\) matches the fixed-delay analytic solution but does not calibrate arbitrary
rates. For \(k=8\), a constant pairwise rate of 0.05 gives approximately 1.7757 times
that rate even over a wide domain. Second, one event on every trial and either zero
or two nearby events with equal probability both have mean count one, but probabilities
of any occurrence of one and 0.5. With the two events separated by 1 ms, the pairwise
prediction curves differ by less than \(1.99\times10^{-6}\) in relative L2 norm.
Mean rates alone do not determine how to interpret nonarrival. Section 5.2 of the paper
also states limitations concerning event counts and joint statistics.

The model assumes stationary statistics and recommends learning credit after the pair
associations stabilize. Cue competition is therefore distinct from revising a relationship
after an omitted outcome or acquiring a new context. Next steps must specify observable
events in continuous sound and address expectations conditional on counts, nonarrival
and context, together with relationship updating. Phrase, closure and musical value do
not follow automatically from this calculation.

All 219 Python tests and 714 Rust tests pass, with eight Rust tests ignored. Production
Rust, standard binaries and the preceding auditory, history and association implementations
are unchanged. There is no new audition. Reference provenance, numerical inputs, complete
outputs, counterexamples and verification are stored in
`target/phrase-expectation/2026-09-07-credit-assignment/`.

### 9.3.11 Continuous observations and context revision

The credit component in 9.3.10 requires identified, counted events. Calling
each envelope frame an onset would make the reporting cadence determine the
event count. The design instead distinguishes four levels:

| Unit | Meaning and boundary |
|---|---|
| Continuous observation | Timestamped band envelopes and trajectories, including steady and decreasing energy. Frames are not counted musical events. |
| Acoustic-event hypothesis | Possible onset, offset, or rearticulation, with observation coverage and identification uncertainty. Envelope rises do not establish Voice onsets; a count distribution for arbitrary overlaps is still missing. |
| Statistical context | A candidate relationship explaining how observations continue, and the interval over which it applies. Creating a hypothesis is not detecting an event. |
| Musical phrase or closure | A separate inference involving temporal relationships, grouping and fulfilled expectations. Acoustic changes, prediction errors and statistical resets are not boundary ground truth. |

[D-REX (Skerritt-Davis and Elhilali, 2021)](https://engineering.jhu.edu/lcap/data/uploads/pdfs/jneurometh2021_skerritt.pdf)
maintains competing contexts for continuous-valued features. Whether temporal
dependence is represented changes its interpretation of the same trajectory.
This motivates updating expectations without first segmenting every sound into
discrete events; it does not establish source separation or musical phrases.

A research implementation now maintains independent Bayesian autoregressions
over context hypotheses for each band. Mathematical references are
[online changepoint inference](https://arxiv.org/abs/0710.3742) and its
[autoregressive extension](https://proceedings.mlr.press/v80/knoblauch18a/knoblauch18a.pdf).
This is not a D-REX port. Each context predicts the next value from declared
lags, using a normal–inverse-gamma prior on coefficients and variance. Integrating
those parameters gives a Student predictive density; mixing over contexts
gives the forecast. The observed likelihood then updates context weights and
sufficient statistics. Separate band contexts avoid a forced global reset, but
do not explain grouping. The cross-band histories and associations in 9.3.8–9.3.9
remain distinct mechanisms.

With declared reset rate \(\lambda\), the prior probability of a reset over the
elapsed \(\Delta t\) since the last learned observation is
\(h=1-\exp(-\lambda\Delta t)\). This is a model assumption, not a universal
DCC-derived value. Posterior reset weight depends on the likelihood; the constant
prior itself is not a change-detection probability. Missing input leaves sufficient
statistics intact and contributes elapsed time to the next reset prior. Lags must
refill with observed values before scoring resumes. A reset spanning missing time
has an interval of possible occurrence, not an invented timestamp.

The computation budget drops the least-supported contexts separately for each
band and reports discarded posterior mass. It is not a memory-age cutoff: an old,
supported context can survive. Small per-update discarded mass does not guarantee
small error across an entire sequence. Forecasts are saved before reading the next
input. Observed low energy and zeros are learned; missing input and EOF do not
supply unobserved outcomes. Outputs are likelihoods in declared observation units,
not onset-arrival probabilities. Continuous densities can exceed one, yielding
negative log losses. The initial real-valued candidate assigned mean negative-domain
mass of 0.304–0.403 across bands and times in the four author recordings. This failure
is preserved.

The envelope model now separates an exact-zero atom from a log-Student positive
density. Each context has a Beta prior for positive-value rate \(\pi\). Its
likelihood is \(\pi\,t_\nu(\log y;\mu,s)/y\) for \(y>0\), and \(1-\pi\)
for zero. Positive/zero counts summarize observation frames, not sound arrivals.
Zeros update rate and context weights without supplying fictitious log values to
regression; positives use their actual logarithms with no threshold or epsilon floor.
Exact numerical zero is not perceptual silence. Forecast locations describe log
positive values. Log-Student tails have no finite positive-amplitude mean, so these
locations are not exponentiated into an expected amplitude. Using interval probabilities
to evaluate the perceived consequences of actions remains a separate question.

Enumerating every segmentation of short sequences and integrating each segment's
regression in batch agrees with online predictive densities and context weights
for zero, one and two lags. Checks also cover observed zero versus missing input,
independent features, time-unit scaling, pruning, immutable forecasts, aggregation,
and right censoring. Exhaustive segmentation also agrees with integrated Beta and
regression evidence for sequences containing zeros. The zero mass plus the positive
density integrates to one. Declared assay choices are 50 ms aggregation, zero/one/four
lags, 0.2/s reset rate, 32 hypotheses, explicit regression priors, and 0.5 prior counts
for each of positive and zero. None is fitted to a cognitive task. Replaying the same
four author and twelve ordinary recordings over three candidates scores 27,094 targets,
including 2,733,757 positive and 327,865 zero band observations. Comparisons use common
targets after the largest lag warmup. One lag improves log loss over the marginal
model on all twelve ordinary recordings; four lags do not improve uniformly over one.
Short-term acoustic continuity can explain this gain, so it is not evidence for musical
order and does not authorize a production connection. Inputs, failures and results are
in `target/phrase-expectation/2026-09-07-context-expectation/`.
Relating continuous predictions to event counts, nonarrival,
and musical expectations still requires an observation-to-event model. Prediction
error is not a universal reward for Voices; learning perceived consequences of
local actions remains separate.

### 9.3.12 Concurrent covariation, past predictive information, and continuity of a stream

Predicting a band's next observation from another band's past is distinct from hearing
concurrently varying bands as a group. [Elhilali et al. (2009)](https://engineering.jhu.edu/lcap/data/uploads/pdfs/neuron2009_elhilali.pdf)
compare synchronous and alternating tones and propose a model using temporal coherence
at multiple rates. Common modulation can therefore be a grouping cue; checking a
predictive confound must not discard the original covariation. Conversely,
[Skerritt-Davis and Elhilali (2021)](https://engineering.jhu.edu/lcap/data/uploads/pdfs/jneurosci2021_skerritt.pdf)
study feature-specific expectations and subsequent integration in pitch/timbre or
pitch/location sequences without a particular cross-feature correlation. Those feature
dimensions are not cochlear frequency bands. Our design inference is to retain
covariation, predictive information, and perceptual grouping separately. This is not a
port of either paper's model.

The research predictor in 9.3.11 now accepts past observations from other bands as
covariates. For each target, four candidates issue forecasts before the next observation
and are scored on that same observation:

| Candidate | Past observations supplied to the regression |
|---|---|
| own | Target band |
| partner | Target and partner bands |
| background | Target band and outside-band envelope summary |
| both | Target band, outside-band summary, and partner band |

The outside summary uses RMS envelopes from bands outside ±one ERB of each target
center. It is an observed covariate, not an identified common cause or the physical
total power of independent channels. Using the zero atom and positive density of
9.3.11, the two reported partner gains are

\[
\Delta_{j\to i}(t)=\log_2\frac{q_{\mathrm{partner}}(y_i(t))}{q_{\mathrm{own}}(y_i(t))},\qquad
\Delta_{j\to i\mid b}(t)=\log_2\frac{q_{\mathrm{both}}(y_i(t))}{q_{\mathrm{background}}(y_i(t))}.
\]

A positive gain means improved prediction under these inputs, conditioning variables,
and candidate models. It is neither causal influence nor a source label or grouping
probability. Pairs share observations, so their gains cannot be summed as independent
evidence. Missing covariates or model misspecification can leave spurious conditional gains.

Seven positive synthetic sequences across three seeds provide controlled comparisons.
When the outside bands encode the constructed common variable, the raw partner gain
of 1.061–1.075 bits/target falls to 0.0032–0.0054 after conditioning. Direct dependence
on the past partner retains 1.374–1.409 bits/target. Removing a relationship halfway
through a sequence changes its conditional gain from 1.097–1.171 in the earlier
evaluation interval to −0.0044–−0.0020 in the later interval. A positive whole-recording
mean does not establish that the relationship persists now. Synchronous unpredictable
variation with contemporaneous correlation 0.953–0.959 yields nearly no past-partner
gain. This is a counterexample to treating past prediction as concurrent coherence,
not evidence against perceptual grouping by synchrony.

**The supported lag range belongs to the evidence.** The one-lag candidate misses a
constructed two-lag dependence. A follow-up fixes two nested history sets, lag one
alone and lags one plus two, on all 21 existing synthetic inputs. The latter recovers
1.373–1.430 conditional bits/target in the two-lag case; independent controls remain
at −0.0074–−0.0051. It also adds the target's and background's second lag, so this
does not isolate a change to the partner coefficient alone. The 50 ms observation
step, lag set, reset rate of 0.2/s, 32 hypotheses, and regression priors are numerical
probe settings. The result neither identifies a human memory horizon of 100 ms nor
allows absent gain to rule out relationships beyond the declared history.

**Observation cadence is not a cognitive time range.** A 50 ms sequence is sampled
at 20 Hz and cannot represent the entire 2–32 Hz modulation range used by the 2009
coherence model. Passing constructed positive envelopes with adjusted amplitude and
phase at 8 Hz and 12 Hz through the actual RMS/log1p aggregation gives 80 coarse
observations agreeing within \(2.00\times10^{-15}\); both appear to vary at 8 Hz.
The 800 observations at 5 ms retain the separate 8 Hz and 12 Hz variations. These
are constructed envelopes, not reconstructed PCM or auditory-filter responses.
They demonstrate information lost by aggregation. Coherence observation needs a
separate path from the retained fine envelopes, with channel delays, temporal
resolution, window boundaries, and gaps checked explicitly. A modulation filter's
phase coordinate is also distinct from a Voice's position within a beat.

Four existing author recordings and twelve ordinary recordings are scored over
28 pairs and 56 directions from eight specified frequency bins. This diagnostic
subset is not an inferred source partition. The scores have not reproduced the
author's distinction between changed interval order and added overlap with unchanged
flow. All 37 initial conditions replay exactly with the original NumPy installation.
An independent SciPy calculation checks 123,256 likelihood values, with maximum
log-likelihood difference below \(6.09\times10^{-13}\). The 506,352 own-only values
match the previous independent-band model exactly. All candidate forecasts agree
over the common-input prefixes. The passing checks comprise 236 Python tests and
714 Rust tests with eight ignored; the 101 production Rust source files and standard
binaries are unchanged since the preceding verification. Plans, outputs, replay
checks, lag follow-up, and aggregation counterexample are retained in
`target/phrase-expectation/2026-09-07-predictive-relations/`.
Perceptual grouping and continuity, cognitive parameters, phrase/closure, learning
perceived action outcomes, and production coupling remain open.

### 9.3.13 Fine-envelope coherence and peripheral phase response

A research temporal-coherence observer now consumes the 1 ms envelopes directly,
avoiding the aggregation ambiguity in 9.3.12. Its mathematical reference is the
gamma-sine/Hilbert pair in [Elhilali et al. (2009)](https://engineering.jhu.edu/lcap/data/uploads/pdfs/neuron2009_elhilali.pdf):

\[
g(u)=u^2 e^{-3.5u}\sin(2\pi u),\qquad
h_{r,\phi}(t)=r g(rt)\cos\phi+r\widehat g(rt)\sin\phi.
\]

It uses characteristic rates 2, 4, 8, 16, and 32 Hz and six phases separated by
\(\pi/3\). This is neither an execution of the authors' code nor a reproduction of
neural responses. The Hilbert pair has future-side support. The finite approximation
uses four positive cycles and one negative cycle, aligning all rates to a common
0.5 s availability delay. The real seed is centered before an at-least-eightfold FFT
extension, and the cropped complex kernel also has its DC removed. These are
numerical approximation choices. Every 1 ms observation is retained; reports occur
every 5 ms. The slowest rate needs 2,501 observations and the fastest 626. Each rate
stays unavailable until its full support is observed, including after gaps. Unknown
prehistory and EOF are never filled with silence. `support_start_sec` denotes the
oldest envelope report timestamp used, not the support of the underlying PCM.

For a complex response \(z_{ir}\), summing the six phase products gives
\(C_{ij}^{(r)}=3\operatorname{Re}(z_{ir}\overline{z_{jr}})\), with diagonal
\(P_{ir}=3|z_{ir}|^2\). Outputs retain each rate's products and powers, their sums,
and signed inner products normalized by the diagonals. This is an inner product
over filter-phase directions, not a correlation in a fixed two-second window.
Input differences and cumulative FIR coefficients annihilate observed constants
exactly without an activity threshold. Zero power leaves normalization undefined,
not evidence of no relationship. A tiny modulation can still have a large normalized
value, so power must accompany it. The instantaneous full Gram matrix has rank at
most ten; that rank cannot count sources among 113 bands.

The finite transfer functions were compared with the analytic Fourier transform of
the gamma-sine seed. At 141 frequencies spanning 0.25–2 times each characteristic
rate, the maximum complex positive-frequency error is below 0.632% of the reference
peak and negative-frequency leakage below 0.501%. Increasing support and FFT extent
is checked separately. Eight positive cycles and two negative cycles reduce the
former below 0.199%, with a one-second common delay. These numerical errors and delays
do not identify cognitive integration times or memory horizons. Streaming updates
also match direct convolution and explicitly calculated six-phase products.

Eleven constructed eight-second envelope cases across three seeds cover constants,
silence, synchronous and alternating regular variation, shared and independent
irregular variation, two groups, gain scaling, and a known delay. Shared irregular
envelopes give normalized inner product one without requiring past predictability.
The 8 Hz/12 Hz envelopes that became identical after coarse aggregation in 9.3.12 are
also distinguishable. These checks do not validate the entire auditory front end,
source separation, or perceptual grouping.

**Fine sampling does not remove peripheral delay.** A control includes the current
gammatone front end and eight carriers from 110 to approximately 4,186 Hz modulated
in phase at five rates. Each carrier is processed separately, and their mixture is
processed as PCM. For the mixture, the lowest/highest-band mean normalized product
is 0.9805 at 2 Hz, 0.7112 at 8 Hz, 0.0813 at 16 Hz, and −0.8344 at 32 Hz. Physically
synchronous modulation can therefore produce a negative value after this front end.
At 32 Hz the phase-equivalent delays are approximately 14.53 ms in the lowest band
and 1.75 ms in the highest. The low-band delay is approximately 17.50 ms at 2 Hz;
a single constant time shift need not align all rates. The faster SciPy four-stage
filter calculation matches 2,000 values from the existing sample recurrence within
6.25e-17. This distortion must not become evidence for independent sources or the
author's perception of separate streams.

The four existing author recordings have been processed, but the twelve ordinary
recordings are deferred until the peripheral timing problem is addressed. Common
input prefixes give identical results by availability time. The first differences
appear at 6.545 s for author A/B and 12.005 s for the texture pair. Their filter
reference times are 6.045 s and 11.505 s; these are not anticipatory detections.
Predictions and actions may consume the observation only at the later availability
time. The reference time is neither a forecast issue time nor an event boundary.
The next step is to model and align the peripheral phase response, checking detuned
carriers and mixtures before combining these cues with predictive information and
the author's flow distinctions. Production phonation and DCC coupling are unchanged.
The passing checks comprise 243 Python tests and 714 Rust tests with eight ignored.
Plans, inputs, outputs, and the peripheral counterexample are preserved in
`target/phrase-expectation/2026-09-07-temporal-coherence/`.

### 9.3.14 Peripheral Phase Compensation and Flow Beyond Change Magnitude

The distortion in 9.3.13 motivated a fixed phase correction derived from the known
gammatone poles. For pole magnitude \(q_c\), PCM sample rate \(f_s\), and envelope
modulation frequency \(\nu\), the four-stage centered-carrier baseband response and
phase target are

\[
B_c(\nu)=\left(\frac{1-q_c}{1-q_c e^{-i2\pi\nu/f_s}}\right)^4,
\qquad A_c(\nu)=\frac{\overline{B_c(\nu)}}{|B_c(\nu)|}.
\]

This factor multiplies the gamma-sine/Hilbert frequency response before cropping
to the previous support. There is no inverse-amplitude compensation, but finite
cropping and DC removal still alter gain. The filter is fixed before reading audio;
it neither searches for signal-dependent delays nor mixes channels. Raw and aligned
outputs retain separate filter-reference and availability times, with the same
0.5-second delay. This is an engineering derivation, not evidence that the nervous
system performs this correction. The
[AMT Hohmann delay stage](https://www.amtoolbox.org/amt-1.6.0/doc/modelstages/hohmann2002_delay_code.php)
supports waveform resynthesis; its delays and phase factors are not adopted as
cognitive envelope-alignment times.

Ten previous controls and 132 additional conditions at three seeds cover carrier
detuning up to ±0.5 ERB, mixtures, intentional 6 ms timing differences, and independent
modulation. In the centered-carrier 32 Hz synchronous mixture, the low/high-band mean
inner product improves from −0.8344 to 0.9925. Across the tested synchronous mixtures,
detunings, rates, seeds, and 28 pairs, the aligned minimum is 0.8548: compensation is
not universal. With an intentional 6 ms difference at 32 Hz, separately processed
low/high carriers retain an inner product of 0.3531, compared with the ideal phase
value of about 0.3564. Finite-filter responses are checked independently, rather than
treated as exact ideal phase rotations.

Independent FFT convolution checks 10,735,200 raw/aligned complex values across all
142 inputs with maximum error below 1.12e-18. Both signs of the exact sinusoidal
steady response independently check 252,000 detuned/delayed envelope values with
maximum error below 2.27e-13. These establish numerical behavior, not stream accuracy.

The four saved author recordings and twelve ordinary recordings then supplied
453,109 fine-envelope observations over 113 bands. Coherence uses the unchanged
28 diagnostic pairs of eight bands. Channel independence permits this projection;
raw author responses exactly match the corresponding prior full-grid responses.
The actual 24 kHz author and 48 kHz ordinary sample rates are retained without
resampling. Reaggregation to 50 ms matches the previous prediction inputs within
2.89e-15. There are 8,258 shared-availability reports, including 6,546 from ordinary
audio. Coherence-reference and prediction issue/target times remain distinct: equal
availability does not mean evidence about the same event time.

**Change magnitudes and the author's actual tasks must remain distinct.**
The first PCM differences occur at 6.5440417 seconds in the interval-order A/B and
12.0000417 seconds in the texture comparison. Both raw and aligned cues preserve
their identical-input prefixes, first differing at available times 6.545 and
12.005 seconds. This is neither anticipation nor phrase-boundary detection.
Post windows begin after another 2.501 seconds, accounting for the longest finite
support and its oldest RMS interval; peripheral IIR history is not thereby removed.
The maximum absolute difference between power-weighted pair inner products is
0.0343 for the interval-order A/B, but 0.5428 for the auxiliary steady/moving_texture
contrast. Maximum absolute differences of directionwise mean conditional
prediction gain are respectively 0.00414 and 0.2378 bits. A single change-magnitude
threshold on either summary that detects the former also detects the latter.
However, the interval-order answer does not establish a flow-identity judgment.
The moving_texture answer was attributed to the immediately preceding candidate,
whose presented comparator was the previous C; steady was an auxiliary comparator
added later. Reinspection of the questions and response records on 2026-09-08
withdraws their interpretation as shared flow-change labels. The numerical ordering
stands, but it establishes neither success nor failure at identifying flow (9.3.15).

The next DCC state must retain candidate relations and their order while expressing
what continues through changes in level or overlap. Maximizing coherence, minimizing
prediction error, or increasing acoustic change must not become a universal musical
reward. Continuation and change hypotheses need to be issued from the same past
evidence and checked against subsequent sound. These observers currently supply
evidence; flow identity, cognitive parameter identification, phrase/closure, learning
local action consequences, and production coupling remain open. Python 247 tests
and Rust 714 tests pass, with eight Rust tests ignored. Production Rust and standard
binaries are unchanged from the preceding verification; sound and DCC coupling are
unchanged. Plans, controls, ordinary audio, author comparisons, and figures reside in
`target/phrase-expectation/2026-09-07-peripheral-alignment/`.

### 9.3.15 Bodies, Excitation, Relational History, and Perceptual Tasks

**Fix the task supported by each author answer before evaluating a model.** The
numbers in 9.3.14 were reproduced, but reinspection on 2026-09-08 corrects their
interpretation. The interval-order A/B asked whether the timing changed to another
ordering. "Only B changed" answers that question, not source replacement, flow
identity, phrase, or closure. The moving_texture question asked whether flow
continued after the change; the answer reported unchanged flow and increased
overlap. That answer named neither a label nor a time, so it is attributed to the
immediately preceding candidate. Its presented comparator was the previous C
(texture_change), not the steady auxiliary comparator subsequently used in the
numerical analysis. These are not positive and negative labels for one common
flow-change task. Unanswered attributes remain unknown.

[BASS, Cusimano et al. (2024)](https://mcdermottlab.mit.edu/papers/Cusimano_etal_2024_listening_w_generative_models.pdf)
provides a reference for separating shared source properties from individual
excitations and assessing hypotheses through resynthesis. It restricts events
within a source to nonoverlap and reports over-segmentation of components with
different decay rates. Its natural-sound priors, numerical discretization, and
expensive sequential inference are not DCC-derived constants or a real-time
implementation. These assumptions should not be transplanted directly into
Conchordal's resonant bodies.

**The next state contract is a design, not an implemented audio inference system.**

| State | Retained information | Distinctions |
|---|---|---|
| Body candidate \(B_j\) | Continuous frequencies, relationships between modes, component-specific decay, and variation in timbre | Not a Voice ID, note name, or fixed instrument label |
| Excitation \(u_j(t)\) | Energy supply, repeated or sustained drive, candidate timing | A new excitation need not create a new source |
| Sounding state \(z_j(t)\) | Vibration retained through subsequent excitation | Excitation offset differs from sound extinction; acoustic decay differs from forgetting |
| Attribution hypotheses | Single/multiple-source, continuation/replacement explanations, uncertainty, and unexplained residual | Inferred sources differ from population individuals; the best retained candidate is not certain perception |
| Relations and order \(R\) | Ordered arrivals, intervals, delayed dependencies, repetition and variation over attribution candidates | Not instantaneous correlation, a shared beat, source identity, or phrase identity |

A body can retain excitation through a state equation such as
\(z_{j,n+1}=A(B_j)z_{j,n}+G(B_j)u_{j,n}\). Existing `ResonatorBank` and `ModeParams`
provide a concrete forward reference. This does not authorize giving the listener
the generator's true parameters. Listener candidates must come from observed
presentation audio. Compare \(\Phi(\sum_j y_j)\), the auditory transform of the
summed candidate waveforms, with observation; summing nonlinear transformed
envelopes \(\sum_j\Phi(y_j)\) does not generally yield the same observation. Acoustic oscillator
phase may belong to the body state; it is distinct from beat phase and does not
require every individual to occupy a shared beat coordinate.

A frozen scalar build of the current resonator checked the forward assumptions.
Four modes at 277, 421, 633, and 911 Hz, with T60 values 0.4, 1, 2.5, and 5 seconds,
were excited at zero and 0.3 seconds. Re-exciting the same body agrees with the sum
of separate excitation responses within 2.24e-7. Creating a fresh body at the second
excitation discards the old tail, producing error energy about 9.88% of the reference
after that point. One four-mode body and four one-mode bodies yield exactly the same
72,000 PCM samples. Their generator body count is not identifiable from that sound
alone. An independent eigendecomposition of the state-transition matrix checks
216,000 values within 2.74e-7. These are constructed physical counterexamples, not
source inference or reproduction of human grouping.

Parameter origins remain distinct. Body T60 describes acoustic response, with
\(r=10^{-3/(T60\,f_s)}\) in the current coefficient compiler. Inferring body decay
does not identify cognitive retention. Relational retention, interference, and
retrieval require separate constraints from matching cognitive tasks. Analysis
cadence, finite approximation support, and search capacity are computational
choices. Weights after candidate pruning are conditional on the explored set;
they are not probabilities over unproposed explanations or human confidence.

Implementation begins by comparing re-excitation of a continuing body with adding
another body, as acoustic predictions issued from the same past. True mode counts,
excitation times, and Voice IDs belong only to evaluation fixtures. Identical
predicted waveforms must receive identical data evidence; prior preferences remain
separate. Relational order is then retained without equating its change with source
replacement. Coherence may guide proposals, but correlations and predictive gains
derived from the same sound cannot be multiplied as independent likelihoods.
Finally, generator choices such as sounding or waiting should be evaluated through
the same acoustic observations, comparing frozen pre-action predictions with actual
outcomes. No single maximization of consonance, synchrony, or prediction accuracy
serves as a universal musical reward.

This checkpoint completes task attribution, physical forward counterexamples, and
the state design. Inferring bodies, excitation, and attribution from unknown
mixtures, relational expectations, cognitive parameters, and learning local action
consequences remain unimplemented or unintegrated. There are no new author judgments,
production sound changes, or DCC coupling changes. Plans, primary material, probes,
and audit results are in `target/phrase-expectation/2026-09-08-perceptual-state-design/`.

### 9.3.16 Audio-Only Modal Candidates and the Boundary of a Validated Forecast

The state contract in 9.3.15 now has an acoustic candidate implementation in
`scripts/evaluate_modal_continuation.py`. It estimates poles from observed PCM with
the [Hua–Sarkar (1990) matrix pencil method](https://intra.ece.ucr.edu/~yhua/MPM.pdf),
then fits complex amplitudes by least squares. The observer receives no true frequencies,
mode count, excitation times, or Voice IDs. A sum of damped oscillations is an acoustic
hypothesis, not an inferred number of perceptual bodies or a source assignment.

Four forecasts compare continuing the previous fit, updating its state while retaining
its poles, adding components to that explanation's residual, and selecting a fresh pole
set from current evidence. Updating the state can be written as an amplitude/phase
increment to the continuing vibration. It does not identify an excitation time or mechanism.
Residual component count is not added source count. These alternatives do not classify
source identity or musical flow.
The preceding fit supplies the comparison reference; persistent identities, reappearance
and disappearance of earlier body hypotheses are not yet tracked.

The numerical configuration fits 40 ms of PCM, selects candidates against the following
already-observed 10 ms, and forecasts the next unseen 10 ms. Forecasts are issued every
50 ms. Candidate pole counts are 0, 2, 4, 8, 12 and 16, with a 96-sample pencil and a
relative singular-value cutoff of 1e-10; zero is a silence candidate. A conjugate pair
can represent one real sinusoid, not one body. Native 24/48 kHz input rates are retained.
These windows, ranks and numerical precision settings are approximation choices,
not cognitive memory durations derived from DCC.

**Preserve the model that was validated.** The first implementation selected rank on
past holdout audio, then refitted coefficients. On ordinary audio, the refit could create
growing poles absent from the validated model and produce explosive forecasts. The
corrected implementation freezes poles and amplitudes after validation. A further
boundary excludes growing estimates from selection as unforced passive bodies, while
retaining their validation errors and growth diagnostics. A separately declared undamped
candidate preserves angular frequencies and fits constant-amplitude oscillations. This
is a competing hypothesis, not a silent correction of the estimated damping. Actual
growth, sustained drive and changing pitch require excitation or time-varying body models;
their unexplained effects remain model mismatch.

Issued forecasts are immutable. A gap clears the acoustic candidate and starts a new
contiguous observation segment. Observed zero PCM remains evidence; missing PCM does not
become silence. Forecast targets extending beyond EOF remain unresolved rather than
receiving zero error or invented silence.

Two sample rates and three previously unused seeds cover 36 conditions of decay,
re-excitation, added frequency, glide, noise and PCM16 quantization. Every issued future
window in the noiseless decay conditions has mean squared waveform error below 1e-15.
The reduced eigenproblem agrees with the full truncated-pseudoinverse eigenproblem to
within 1.81e-15 in pole distance. Poles inferred from actual unchanged Rust resonator
output agree with its state-transition eigenvalues to within 2.05e-9. Identical PCM from
one four-mode body and four single-mode bodies produces identical forecasts. Glide and
unmodeled excitation retain prediction errors. These are acoustic and numerical checks.

The 12 ordinary and four author-related recordings also compare the frozen summed
waveforms through the existing causal auditory filter. Each candidate, actual future,
and zero-input baseline starts from the same observed filter state. A last-envelope
persistence baseline is included. Squared error across 113 Log2 bands at 1 ms resolution
is a descriptive loss, not a calibrated perceptual likelihood or a judgment of flow.
There are 9,058 issued forecasts and 9,040 windows shared by all four candidates.
The first forecast in each recording has no preceding body; two EOF targets remain
unresolved. Across the 7,134 ordinary-audio windows, every modal alternative has higher
aggregate error than envelope persistence in each of the 12 recordings. Fresh-fit error
is 1.221–2.115 times persistence error. For moving_texture the fresh-fit ratio is 0.264;
this acoustic prediction improvement does not turn the distinct author tasks in 9.3.15
into shared flow-change labels. Actual-audio envelopes match the previous saved values
within 2.85e-16. All 254 Python tests and 714 Rust tests pass (eight Rust tests ignored).
The 101 production Rust files and standard binaries remain unchanged.
This implementation reaches audio-only acoustic proposals. Persistent body/excitation
attribution, uncertainty, relational expectation, cognitive parameters, local
action-consequence learning, phrase/closure and production integration remain open.
Artifacts, including failed earlier versions, are retained under
`target/phrase-expectation/2026-09-08-modal-continuation/`.

### 9.3.17 Driven Acoustic Forecasts and Power Lost by Averaging Waveforms

**Separate the preceding failure from numerical capacity.** At fixed times of 2, 8,
16 and 22 seconds in the 12 ordinary recordings, 48 windows compare pencil width and
rank caps. The original width 96 and cap 16 give future waveform error 0.996 times zero
prediction; raising the cap to 64 still gives 0.996. Width 384 with cap 64 gives 0.539,
and fitting error falls from 0.840 to 0.165 times its zero baseline. At 48 kHz these
widths span 2 and 8 ms. This diagnostic does not establish an optimal width or whole-run
improvement, but the previous failure cannot be attributed solely to missing drive or glide.

`scripts/evaluate_driven_acoustic_state.py` adds a research candidate separating an
acoustic response from innovations. Reflection coefficients are estimated from forward
and backward prediction errors using [Burg (1975), chapter II-C, equation II-64](https://sepwww.stanford.edu/data/media/public/oldreports/sep06/06_09.pdf).
An all-pole response is retained in lattice form, avoiding expansion into a high-order
polynomial. Recent observed samples update its backward-error state. Freezing response
coefficients after validation is compatible with incorporating the newest known audio
into the sounding state.

Within a candidate, \(x_n=-\sum_k a_kx_{n-k}+\epsilon_n\), with independent
\(\epsilon_n\sim N(0,q)\). This is a conditional acoustic distribution, not an
identification of the physical driving mechanism. Harmonic uses deterministic drive
input; Modal permits noisy drive. Estimated \(q\) can also absorb unexplained pitch,
timbre or observation-precision effects and is not the generator's `sustain_drive` value.

The model fits 40 ms, compares candidates by joint density of the following already-known
10 ms, and freezes a mean and response for the next unseen 10 ms. Every 50 ms it compares
fresh fits, retained coefficients and variance, and retained coefficients with updated
variance. Up to four distinct response candidates survive by comparison, without a fixed
age expiry. Capacity pruning is not cognitive forgetting. Orders 0/4/8/16/32/64, windows
and retained capacity are numerical settings, not source counts or DCC-derived memory
durations. PCM16 quantum \(\Delta=1/32768\) supplies a variance floor \(\Delta^2/12\).
The continuous Gaussian density is not exact PCM-bin probability, and observation noise
has not been separated from drive uncertainty.

**Distinguish waveform means from expected auditory quantities.** For acoustic response
\(h\), auditory filter \(g_c\), and mean output \(\mu_c[n]\) computed from the same
observed filter state, the conditional future power is

\[
E|Z_c[n]|^2=|\mu_c[n]|^2+q\sum_{j=0}^{n}|(g_c*h)[j]|^2.
\]

Filtering only the predicted mean waveform omits the second term. Waveform responses
are composed before conversion into 113 auditory bands, then expected power is averaged
over each 1 ms report. Its square root is a second-moment RMS quantity, not expected
envelope amplitude. Power and envelope errors are reported separately; better acoustic
prediction is not recognition of musical flow.

An independent quadratic objective agrees with reflection estimates across 18 conditions
within 3.67e-15. Independent convolution agrees on 6,780 power values within 7.81e-18.
Actual selected order-64 responses also match a separate state-transition matrix within
4.63e-12. The overprediction below is therefore not dismissed as lattice arithmetic failure.

The existing 16 recordings yield 9,058 forecasts and 9,040 common scored windows. After
implementation freeze, the same archived renderer, scenarios and configuration generate
12 ordinary recordings with new seeds 184101, 184117 and 184139, yielding 7,146 forecasts
and 7,134 common windows. The archived and current render binaries have different file hashes,
but re-rendering all twelve additional recordings with the current binary produced identical
whole-WAV SHA-256 hashes. This verifies those scenario/config/seed combinations;
this is not current-standard-production acceptance. In both sets of 12 ordinary recordings,
expected power beats persistence in 10 recordings, and adding variance improves over mean
power in 11. The square-root quantity beats envelope persistence in all 12 of each set,
but the different metrics must not be collapsed. Held-out Rain seed 184101 has power error
3.810 times persistence: at 7.9 seconds, a retained response with updated variance strongly
overpredicts power. Independent drive innovations and transition-time candidate selection
remain incomplete models.

Empirical coverage of marginal 95% PCM intervals is 94.45–95.30% in the new ordinary
recordings, versus 86.9–90.5% in the existing author-related recordings. This does not
establish uniform calibration. Coefficient uncertainty, a joint perceptual likelihood and
human confidence are not represented by these intervals. Reusing an old response also
does not establish persistent perceptual source identity.

All 260 Python and 714 Rust tests pass (eight Rust tests ignored). The 101 production Rust
files remain unchanged. Glide trajectories, intermittent versus sustained drive, perceptual
source attribution, relational/cognitive expectations, action-consequence learning,
phrase/closure and production integration remain open. Evidence is retained under
`target/phrase-expectation/2026-09-08-driven-acoustic-state/`.

### 9.3.18 Continuous-Phase Trajectories and the Separation of Parameters from Vibration State

`scripts/acoustic_trajectory.py` adds PCM-only acoustic trajectory candidates. The local
signal family follows [Neri, Depalle and Badeau (2021), equation 1](https://www.dafx.de/paper-archive/2021/proceedings/papers/DAFx20in21_paper_30.pdf).
With time relative to the fitting-window center,

\[
x(t)=\operatorname{Re}\sum_m b_m\exp\{d_mt+i2\pi(f_mt+\tfrac12 v_mt^2)\}.
\]

Here \(v_m\) is frequency change in Hz/s, giving instantaneous frequency \(f_m+v_mt\).
Integrated phase remains continuous during a glide; \(b_m\) is complex amplitude.
Positive local amplitude growth is not an unforced passive body. Adopting this signal
family does not identify a uniquely DCC-derived cognitive model.

The estimator is time-domain least squares, not the paper's Bayesian EM algorithm.
Residual FFT peaks propose components; Gauss–Newton directions, line search and exact
linear-coefficient solves refine frequency, chirp rate and local amplitude change.
The numerical choices are 40 ms fit, 10 ms known validation, 10 ms future forecast,
counts 0/1/2/4/8/16 (controls cap at four), and at most 30 optimizer iterations.
They are neither perceived source counts nor cognitive memory durations. Generator
frequencies, glide rates, excitation times, Voice IDs and true counts never enter inference.

**Separate signal-family comparisons from initialization.** The first implementation
initialized stationary and gliding fits independently and reached different local minima.
The current implementation also initializes each glide count from the same-count stationary
fit. The two starts compete on training loss before observed holdout selection. Tests keep
the stationary solution available within the nested family; they do not establish a global
optimum. For Murmuration seed 3141593, the glide waveform error/zero ratio falls from 7.758
to 0.328. Initial and revised outputs are preserved separately. Optimization failure must
not be attributed solely to an inadequate signal family.

Four alternatives are compared: retained continuation, a complex-state update on the
retained trajectory, a fresh stationary fit, and a fresh glide fit. Observed validation
waveform error selects the candidate; future observations cannot rewrite issued forecasts.
A trajectory leaving \((0,f_s/2)\) becomes unavailable, not a silence prediction. Selection
is not a perceptual-attribution probability, and a state update does not identify reexcitation.

Forty-two conditions across two sample rates and three seeds reproduce stationary,
gliding and crossing synthetic continuations. Unobserved reexcitation, added components
and direction changes cause errors; subsequent evidence updates the candidate. Identical
PCM prefixes produce identical forecasts. Packetization, gaps and forecast immutability
are tested. These are acoustic controls, not evidence of human stream identity at a crossing.

The ordinary-audio assay uses the previous 12 recordings and the additional 12 recordings.
It targets 10 ms starting at 2, 8, 16 and 22 seconds: 96 ordinary target windows, plus a
separately labeled diagnostic window at 7.9 seconds in the known Rain failure. The four-way
ordinary comparison has 94 common available windows; retained trajectories leave the
represented frequency range in two windows. The diagnostic is excluded from recording-level
aggregates. Predictions and actual PCM pass through the same observed 113-band filter state.
Waveform and band-power errors are scored separately. Selected power beats persistence in
11/12 existing and 10/12 additional recordings, but the largest ratios remain 1.503 and 1.883.
Even after revised initialization, the standalone glide waveform error beats the stationary
candidate in only 1/24 recording-level aggregates. These sparse windows are not whole-performance
scores, musical improvement, or a new author acceptance judgment.

At the known Rain 7.9-second window, actual band-power arrays match the preceding assay
exactly. The driven predictor has power error 2059.59 times persistence; the selected
stationary trajectory has 0.11346 times persistence. This single-window comparison differs
from the preceding whole-recording ratio of 3.810. It does not establish a general solution
or attribute the improvement to glide. These point forecasts do not yet incorporate the
stochastic drive covariance from 9.3.17.

Independent recursive oscillators reproduce saved ordinary-audio candidates within 1.10e-12
absolute error. All 266 Python tests pass. The 101 production Rust files remain unchanged;
the last Rust record has 714 passing tests and eight ignored. Current evidence is in the
`*-initialized` files and directories under
`target/phrase-expectation/2026-09-08-acoustic-trajectory/`.

**The next state-update boundary.** Trajectory parameters must be separated from vibration
state updated by observations. In the 9.3.18 implementation, complex-amplitude updates stopped at the training boundary
and do not assimilate excitation in the subsequent known validation interval. This representation
still needs the distinction made in 9.3.17: fixed coefficients can coexist with an updated
observed state. Connecting trajectory continuation, intermittent/sustained drive and observation
error within a common predictive distribution remains open, as do persistent perceptual
attribution, relational expectations, cognitive parameters, action-consequence learning,
phrase/closure and production integration.

### 9.3.19 Conditional Amplitude and Residual-State Updates from the Latest Audio

`scripts/acoustic_posterior.py` is a research implementation that freezes trajectory frequency,
frequency rate, and log-gain rate, together with AR residual coefficients and innovation variance,
while updating linear amplitudes and residual state from the latest PCM. The linear Gaussian
posterior uses the weight-space calculation in
[Rasmussen and Williams (2006), section 2.1](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
Zero prior precision is admitted only when the observed design identifies every amplitude direction.
This is neither an identified cognitive prior nor a normalized posterior over candidate models.

**Amplitude and residual state are correlated.** With real trajectory basis \(B\) and amplitudes
\(\beta\), the observed residual is \(r=x-B\beta\). Changing amplitudes also changes the residual
state constrained by past audio. Let \(K\) map past AR state into the future, \(L\) propagate
innovations, \(m_\beta,P_\beta\) be the conditional amplitude mean and covariance, and \(q\) the
innovation variance. Our conditional forecast is

\[
M=B_{future}-K B_{past},\qquad
\mathbb E[x_{future}\mid x_{past}]=Kx_{past}+Mm_\beta,
\]
\[
\operatorname{Cov}[x_{future}\mid x_{past}]=MP_\beta M^\mathsf T+qLL^\mathsf T.
\]

Adding \(B_{future}P_\beta B_{future}^\mathsf T\) independently would lose the constraint from
observed audio. Raw PCM and basis AR states are retained separately, and QR updates incorporate
new evidence. The first \(p\) samples, for AR order \(p\), are conditioned on rather than scoring
an invented zero prehistory. Contiguous observation packetizations agree within numerical tolerance;
gaps and reversed input are rejected before mutation. Issued forecasts remain immutable. The
predictive mean and covariance are mapped to band power using the same already-observed auditory
filter state as the target audio.

**Numerical identifiability matters.** The initial version normalized columns only by their whitened
norms. At a quarter of the sample rate, a stationary sinusoid is annihilated by \(x[n]+x[n-2]\).
The initial implementation amplified the remaining roundoff and incorrectly admitted an amplitude
posterior. The corrected version scales by the original basis and checks both relative singular
values and a roundoff floor referenced to that original signal. The counterexample is a regression
test. These thresholds are numerical choices, not auditory discrimination thresholds.

Candidates use 40 ms of fitting evidence and 10 ms of known validation before forecasting a new
10 ms target. Residual orders are 0, 4, 8, 16, 32, and 64; ordinary-audio component counts are
0, 1, 2, 4, 8, and 16. Conditional validation density selects the candidate, whose state is then
conditioned through validation. Fixed-amplitude controls either select their own best candidate or
use exactly the updated model's trajectory and AR parameters. These windows, orders, and variance
floor are computational settings, not cognitive parameters derived from DCC.

Across 42 synthetic controls, known reexcitation reduced future waveform error by approximately
6.5–10.4% relative to the matched fixed-amplitude control. This does not identify reexcitation;
added components and direction changes also remain incompletely handled. Six pairs with a future,
not-yet-observed reexcitation had identical forecast hashes for all three variants before that change.
The unseen event remains a subsequent prediction error.

All three variants were available in 96 ordinary windows: 10 ms at 2, 8, 16, and 22 seconds in each
of 24 recordings. The known Rain window at 7.9 seconds is one separate diagnostic. Averaging the four
windows per recording, updated band-power prediction beat persistence in 11 of 12 existing and
11 of 12 additional recordings. It beat matched fixed amplitudes in all 24 recordings, but the
independently selected fixed-amplitude control in only 17. Additional sample 12, seed 184139, had
power-error ratios to persistence of 369.18 for matched fixed amplitudes, 2.806 after updating, and
0.200 for independently selected fixed amplitudes. State updates do not resolve candidate-selection
failure or overprediction. Recording-average coverage of nominal marginal 95% intervals ranged
from 68.3% to 96.1%; uncertainty is not calibrated. These short fixed windows establish neither
whole-performance quality nor recognition of musical flow.

Independent dense Gaussian calculations verified pre/post-update means, covariance, and density.
Three real-audio diagnostic windows were also recomputed at 80 decimal digits using a direct AR
polynomial, recurrent complex oscillators, and normal equations. Maximum mean absolute error was
below 4.25e-10, variance relative error at twelve specified positions below 3.79e-11, and joint
log-density difference below 5.77e-9. Those numerical errors do not explain the observed
overprediction. Inputs, timestamps, hashes, and error aggregates were checked for 417 saved forecasts.
All 274 Python tests passed. The 101 Rust files remain unchanged from the preceding check;
the recorded Rust result is 714 passing tests and eight ignored tests.

Only conditional linear-amplitude uncertainty is integrated. Trajectory parameters and \(q\)
remain point estimates; the model does not infer intermittent versus sustained excitation,
observation noise, or perceptual source attribution. Hard candidate selection is not a source
probability, and short-term acoustic prediction is not by itself cognitive expectation. Evidence is
under `target/phrase-expectation/2026-09-08-acoustic-posterior/`; `controls-rank` and `audio-rank`
are current, while the initial `controls` remain preserved. Relational expectation, cognitive
parameters, action-consequence learning, phrase, closure, and production integration remain open.
Next, the action contract in 9.3.5 must connect a candidate-time prediction, confirmed execution,
and an observed acoustic outcome by time. Perfect source separation is not a prerequisite, but a
subsequent prediction error alone cannot establish that the actor caused the change.

### 9.3.20 Joining an Issued Onset to Subsequent Acoustic Evidence

The production report now implements the action-to-observation association in the 9.3.5 contract.
When `ParticipationClock` issues an onset, it freezes an available whole-habitat continuation
forecast and associates it with the first complete observation window at or after that onset.
Rejected candidates issue no record. The window uses the existing approximately 10-ms grid of
`AcousticTemporalExpectation`; no new cognitive timescale or reward is introduced. The observed
forecast boundary, command issue time, exact onset sample, and target-window bounds remain separate.
The forecast can have been retained since candidate planning; it does not necessarily incorporate
every observation available at the later command-issue time.

After the habitat audio is actually processed, its three-band energy for the same window is
reported as `participation_outcome`. The saved forecast precedes self-exclusion: a decision view
with the actor removed must not be scored directly against the whole habitat. Pending records
live outside the individual, so a target can still be observed after that individual disappears.
Skipped input gives `input_gap`; an unresolved target at input termination gives `end_of_input`.
Both carry null observation values. Observed silence instead gives `observed` with zero energy.

Forecast records are retained only with reporting enabled, and cover Participation onsets with an
available forecast. They do not cover every phonation type or establish learning. Buffers reuse
their capacity through drain/retain; runs without a report incur no prediction-record allocations.
Voice IDs label generator-side commands, not inferred auditory sources, and are not fed into the
acoustic observer.

**The association exposed an existing timestamp error.** `onset.time_sec` used the start of the
processing hop instead of the onset's actual sample position. The control's first onset was at
sample 7459, while the old report represented sample 7168. The corrected report derives seconds
from `onset_tick` and also exports integer `onset_frame` for exact joins. Shared scaffold phase is
evaluated at that actual onset time. Across the four comparisons, old timestamps were up to
511 samples, approximately 10.65 ms at 48 kHz, early. Old report-derived IOI and phase statistics
retain this hop quantization and must not be reinterpreted as sample-accurate measurements.
The correction does not alter archived WAVs or author audition responses.

A routing control and samples 08, 09, and 12 at seed 21 were rendered before and after the change.
All four WAVs were identical. The new outcome counts were 10, 202, 731, and 166: all 1109 records
joined an issued onset to the exact target observation window. Existing report rows also matched,
excluding onset records, rhythm summaries, execution timings, and the new outcome records.
The control's first nonzero PCM was at sample 7463, following its onset command at 7459. Rendering
with and without reporting also produced identical WAVs. Regression checks cover input gaps,
unfinished targets, rejected candidates, and immutable predictions. All 718 Rust tests passed,
as did Clippy, all-target checking, and the release renderer build. Evidence is stored under
`target/phrase-expectation/2026-09-08-action-outcomes/`.

**Association is not action-consequence learning.** The saved quantities are a pre-action
continuation forecast and subsequent audio, not a prediction conditioned on adding the selected
action. That prediction must include the actor's known acoustic body, chosen onset or waiting,
routing, and already-sounding state. The control actor sends only to presentation and contributes
nothing to the habitat, yet another habitat source changes its post-action observations. Treating
the discrepancy directly as the actor's contribution, success, or reward would therefore be wrong.
`observed` means the window was available, not that the onset was audible, preferred, or causally
identified. Next, generator-side action-conditioned forecasts should be checked against sounded
and waiting branches from the same history. Learning from those outcomes, cognitive parameters,
relational expectation, phrase, and closure remain open.

### 9.3.21 Owned-State Branch Forecasts and Updates After Observed Outcomes

Action forecasts include both existing ringing and the new command. The research-only
`ScheduleRenderer::fork_source` clones the specified individual's Tones, preserving phase,
resonators, smoothing, RNG state, pending updates and triggers, and routing. It does not copy
other individuals' Tones or self-sound observers. The caller resumes at the next unrendered
sample and supplies future owned commands. Advancing a fork does not consume live state;
waiting does not erase ringing. This allocating path is not connected to RT operation.
Audio export is confined to the offline `examples/action_prediction.rs`; the instrument gains
no recording path.

**Known owned state and inference about the environment have different evidential status.**
Subtract the known owned waveform from the observed mixed waveform, then fit the environmental
forecast of 9.3.19 to that observed residual. Add each predicted owned future waveform to its
mean, preserving conditional covariance. This control assumes exact owned body parameters and
routing; it is not auditory separation of an unknown source. Other individuals' identifiers,
body states, and future commands never enter inference.

Do not add band powers as though coherent components were independent. Equal-frequency
components can reinforce or cancel. Pass the predicted mixture through the same observed
auditory filter state and propagate its mean and covariance to expected squared magnitude.
An ablation deliberately omits the cross term between the environmental and owned contributions.
An exact opposite-phase unit control leaves only residual noise in the proper forecast, while
the sum of two powers retains a large component.

The initial 72 cases crossed three bodies, two routes, 24/48 kHz, three seeds and the presence
of an additional external onset. All 36 pairs had identical mixed past, owned past and owned
future branches; changing only the external future left issued forecasts identical. The
predicted sound-minus-wait waveform agreed with the difference between actual mixed branches
to less than 2.43e-8. Presentation-only routing had zero habitat effect. These are checks with
known bodies, fixed future rhythm parameters and nonreactive other sources, not social
response or learned causal effects.

**The initial continuation control also contained a release.** Its `continues` label meant
no additional onset, but the existing external Tone entered release at 155 ms. That schedule
was unavailable to inference at the 150 ms issue time. Preserve the original audio and label,
but do not describe the condition as stationary continuation. Three additional, unused seeds
crossed `held` beyond 170 ms, `release` at 155 ms, and `release_and_onset` at the same time.
These 108 cases formed 36 triples with identical evidence through 150 ms and identical initial
forecasts. Identical past evidence cannot distinguish those unobserved future commands.

In the additional experiment, execute the `sound` branch and read only its 150–160 ms prefix.
Subtract its known executed contribution and issue a new environmental continuation for
160–170 ms. Keep the previously issued forecast unchanged and retain the ringing of the
irreversible earlier action. Neither the unchosen wait outcome nor samples after 160 ms enter
this update. A regression changes both withheld portions and obtains identical initial and
updated forecasts. This updates an acoustic state estimate; it does not learn action effects
or an action policy.

Compare both forecasts on the same 160–170 ms target. Coverage below is a condition-average
of pointwise waveform intervals, not musical confidence. The 36 cases include body/routing
controls sharing an environment; they are not 36 independent listening experiments.

| Environment | Lower waveform MSE | Lower band-power MSE | Nominal 95% coverage: frozen → updated |
|---|---:|---:|---:|
| held | 36/36 | 36/36 | 60.4% → 80.0% |
| release | 18/36 | 19/36 | 44.0% → 88.5% |
| release_and_onset | 12/36 | 22/36 | 2.5% → 21.7% |

Mean waveform MSE fell from 1.20e-5 to 4.38e-7 for held sound, but rose from 3.56e-5 to
6.83e-5 for release and from 7.37e-3 to 7.62e-3 for release plus onset. Increased coverage
alone establishes neither an improved mean nor calibrated uncertainty. Refitting with new
observations is insufficient for excitation/release changes. Continuation, changed drive or
amplitude of an existing body, and additional components need competing hypotheses whose
support is updated from already observed outcomes. Repeatedly refitting one local model is
not adopted as DCC's cognitive memory or learning mechanism.

Evidence and forecast hashes, timing, waveform differences and aggregates were checked for
all 180 cases. Separately from the production four-stage auditory recurrence, convolution with
its closed-form impulse response checked three bands of each initial branch, with maximum
power difference below 1.39e-17. All 719 Rust tests passed, with eight ignored; all 278 Python
tests, Clippy, all-target checking, and release binary/example builds passed. Artifacts are in
`target/phrase-expectation/2026-09-08-action-conditioned/` and its `release-diagnostic/` child.

The 40 ms fit, 10 ms validation/re-observation, 20 ms initial horizon, candidate orders and
variance floor are numerical choices. Inputs are f32; the inherited floor is not the actual
quantization error of PCM16 observations. Cognitive retention parameters, perceptual body
attribution, relational expectations, phrase/closure, consequence learning and production
integration remain open. No new author audition judgment is added.

### 9.3.22 Separating Proposed Frequency Support from Amplitude Evolution

Retain alternative explanations of observed sound before selecting one forecast.
[Pieszek et al. (2013)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0053634)
reported violation-related ERP responses to contradictory predictions from auditory probability
and visual cues. That task does not derive this implementation's acoustic families, candidate
counts, selection rule or windows. Likewise,
[Marchand (2012), section 2](https://dafx12.york.ac.uk/papers/dafx12_submission_35.pdf), separates
amplitude and phase evolution in nonstationary partials; the implementation below is neither
that estimator nor an identified cognitive mechanism.

Research module `acoustic_hypotheses.py` compares continuation of a conditional state, changed
complex amplitudes on retained frequency support, and additional support proposed from residual
PCM. Evidence through 150 ms proposes the retained support; subsequently observed sound-action
outcomes update the candidates. True external body identities and excitation/release times are
not inputs. Duplicate support with unidentifiable individual coefficients is rejected as rank
deficient, not counted as another perceptual source.

**Periodic structure can reside in the residual model.** The first prototype reused the explicit
partials of the model selected by conditional density. An AR-only model can predict periodic
sound while containing zero explicit partials. Its apparent amplitude-change comparison then
merely refitted shorter residual models. The revision proposes explicit support using waveform
continuation error on the already-known 10 ms following the old 40 ms fitting block, separately
from scoring the complete residual-plus-signal forecast. Empty support produces no amplitude-change
candidate. This engineering proposal does not identify a unique physical or perceptual decomposition.

**Amplitude-change hypotheses must not silently import another envelope evolution.** The next
version retained exponential gains estimated when proposing additional partials from short PCM.
One selected example implied amplitude multipliers of approximately 5039 and 807 over 10 ms.
The revised amplitude-change and added-support candidates retain estimated frequency/phase
trajectories but fit constant or affine complex amplitudes anew:

\[
x(t)=\operatorname{Re}\sum_k\left(a_k+b_k u(t)\right)e^{i\phi_k(t)}.
\]

The constant case sets \(b_k=0\); \(u\) is a numerical coordinate scaled to the fitting interval,
not a cognitive clock. Complex coefficients also permit local carrier-phase correction; this phase
is distinct from position within a beat. The continued-state
candidate retains its original decay and residual state. Changed bases are fitted and validated
before issue; already-issued predictions are not clipped afterward. Replaying the same extreme
case reduced waveform MSE relative to the refit baseline from about 13.78 million times to 1.063
times. This repairs extreme extrapolation, but does not beat that baseline in this case.

Candidate fitting precedes a common observed selection block. Frequency support is not refitted
on that selection block; complex coefficients and residual states are subsequently conditioned
on it. Keep each family's best forecast and all candidate records, without converting density
ranks into probabilities of perceptual objects. Families have unequal numbers of candidates;
maximum validation density is not family posterior probability. Save forecasts and evidence
coordinates/hashes before opening actual future targets.

Both the exponential-envelope version and the separated phase/amplitude version were tested on
three seeds unused for that version, 108 controls, and two observation times. The current version
uses seeds 184907/184913/184927. Each row below compares its selected forecast to the previous
global refit on the same target window, across 36 controls.

| Observed duration / target | Environment | Lower waveform MSE | Median MSE ratio | Nominal 95% coverage |
|---|---|---:|---:|---:|
| 10 ms / 160–170 ms | held | 36/36 | 0.180 | 79.8% |
| 10 ms / 160–170 ms | release | 36/36 | 0.053 | 61.9% |
| 10 ms / 160–170 ms | release_and_onset | 6/36 | 1.153 | 21.2% |
| 20 ms / 170–180 ms | held | 6/36 | 1.333 | 44.3% |
| 20 ms / 170–180 ms | release | 36/36 | 0.006 | 88.5% |
| 20 ms / 170–180 ms | release_and_onset | 24/36 | 0.956 | 27.4% |

The 10 ms observation uses 8 ms for adaptation and 2 ms for selection; the 20 ms observation
uses 10 ms each. Candidate adaptation starts are 150/152/154/156 ms; the actual 155 ms release
time is withheld. Observation amount, fit/selection lengths and target time all differ between
the row groups, so this does not isolate the causal effect of observing longer. Body/routing
controls share environments and are not independent listening experiments.

Release forecasts improve in every control at both observation times. However, held-sound
forecasts mostly lose to the global refit after 20 ms, and the mean error for release plus onset
exceeds the baseline at both times. The latter's worst MSE ratio after 20 ms is 3.327. Nominal
95% intervals remain uncalibrated. Selection on a short observed block does not guarantee longer
continuation; coefficient uncertainty alone does not represent uncertain selection or unmodeled
changes. This candidate does not replace the production behavior or the refit baseline.

Tests cover direct regression/covariance agreement, nonidentifiability of duplicate support, and
forecast invariance when unchosen outcomes and still-unobserved future samples are changed.
All 719 Rust tests passed with eight ignored; all 281 Python tests, Clippy and all-target checking
passed. Inputs, failed prototypes, both fixed-version evaluations, source snapshots and independent
checks are stored in `target/phrase-expectation/2026-09-08-acoustic-hypotheses/`.

These are local acoustic approximations. PCM error is not the completion criterion for DCC
memory or musical relational understanding. Predicted perceptual consequences and relational
expectations still need comparison with observed outcomes. Cognitive parameters, persistent
perceptual attribution, phrase/closure, consequence learning and production integration remain
open. No new author audition is added.

### 9.3.23 Mapping Sound/Wait Forecasts through the Same Observed Auditory State

Research sound/wait forecasts now pass through the same NSGT, peak extraction, intensity smoothing,
roughness, harmonicity and consonance-field computation as ecological audio. The test-only
`runtime::perceptual_action_assay` uses the production configuration builder and `AnalysisStream`.
It adds neither a second Python R/H implementation nor instrument recording or production action selection.

**Preserve history at the branch.** The FFT ring, band smoothing and peak-intensity history are copied
from observed state. Known samples pending within the 512-sample hop are prepended to each branch.
Only complete hops are processed, without filling incomplete tails with silence. Outputs retain input
and processed endpoints, pending counts, the FFT container and startup zeros. The container is neither
per-band perceptual latency nor cognitive memory duration. Forecasts never advance observed state.
At 24/48 kHz, all branch snapshots match complete-history replay bit for bit.

The earlier 160–170 ms forecasts reach no new update with the default 512-sample hop at either rate.
For 170–180 ms, the first update contains only 16 or 32 forecast samples. This assay observes one second
of audio and issues 50 ms branches. The final scored update is at 1.045333… seconds at both rates,
without startup zeros. These durations specify the assay, not cognitive retention or phrase length.
The research generator's hold and gate follow the issue time to avoid unintended silence during the
longer history. Its original defaults preserve all 252 compared PCM files exactly.

**Expected perceptual values differ from perception of the mean waveform.** Each forecast supplies
32 trajectories preserving correlated amplitude and residual uncertainty. All use the same observed
auditory state, and sound/wait share environmental draws. Opposite-polarity waveforms have a silent
mean while each retains roughness and harmonicity: `E[g(X)]` cannot generally be replaced by `g(E[X])`.
Sampling moments match an independent conditional-state matrix calculation. These draws still describe
only the fixed acoustic model, without integrating model selection or unrepresented changes.

The assay covers seeds 184957, 184961 and 184967, three bodies, two routes, 24/48 kHz and three
environmental conditions: 108 cases. Every predictive trajectory and auditory output is saved before
either realized branch is analyzed. Across 36 environmental triples, prior evidence and issued
trajectories are identical. For the 18 habitat-routed cases in each environment, the table counts lower
squared error in the predicted sound-minus-wait difference than a zero-difference control. Log2Space
bins receive equal weight; perceptual axes are not collapsed into a universal reward.

| Environment | Roughness state | Harmonicity state | Consonance field level |
|---|---:|---:|---:|
| Held | 17/18 | 17/18 | 17/18 |
| Release | 17/18 | 16/18 | 17/18 |
| Release plus new onset | 16/18 | 15/18 | 16/18 |

The remaining 54 presentation-only action controls have exactly zero predicted and observed habitat
differences. The observation bus is habitat, not the presentation-only ListenerTwin. Other actors do
not react in these controlled branches; this does not establish learned causal effects during interaction.

Averaging perceptual outputs reduces mean roughness error relative to processing only the mean
waveform in each environment. Harmonicity and consonance show the reverse. Correct marginalization
does not guarantee lower error on individual outcomes. Coverage of the 32-draw central 90% quantile
interval is approximately 96–98% for held environments, 37–60% for release and 33–57% with a new onset.
These are descriptive bin/branch averages, not independent listening trials or calibration evidence.
The 16-versus-32-draw mean difference remains a numerical diagnostic, not cognitive variability.

All 721 Rust and 284 Python tests and all-target Clippy/check pass. Nine Rust tests remain ignored
by the normal suite; this exporter passes separately for prediction and actual observation. Inputs,
distributions, samples, native results, source and checks are under
`target/phrase-expectation/2026-09-08-perceptual-consequences/`.
This connects acoustic forecasts to existing perceptual quantities and matched evaluation. Next,
the selected action's prediction and actual evidence must be retained at matching times and updated
while distinguishing environmental changes from owned effects. Local consequence learning, production
integration, persistent perceptual attribution, relational expectation, cognitive parameters, phrases
and closure remain open. No new author audition or musical acceptance is inferred.

### 9.3.24 Updating Residual Variance from the Selected Action's Actual Sound

The previous distribution updated amplitude state while holding residual variance fixed.
Research `AcousticPosterior` now optionally retains and updates unknown variance. Trajectory
and AR response remain fixed; this does not learn source attribution or action value.
Variance conventions follow [Murphy (2007), sections 6 and 10.2](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf);
the correlated regression application below is derived and checked separately.

After conditioning on the initial AR samples, let `n` be the whitened observation count
and `k` the amplitude dimension. With the explicit reference prior
`p(amplitude,q) ∝ 1/q`, residual sum of squares `RSS` gives
`q | observed ~ InverseGamma((n-k)/2, RSS/2)`. This conditions on the selected trajectory
and AR response without integrating their estimation or selection uncertainty. Positive
residual energy and a finite variance mean are required; otherwise this option is unavailable.
The prior does not establish a human memory mechanism or neural implementation.

Updates subtract known owned sound from the chosen actual mixture in the waveform domain.
Noncontiguous inputs are rejected, and each prediction is saved before its observation enters
the model. Integrating variance produces a joint Student prediction. One variance draw scales
the whole correlated amplitude/residual trajectory, rather than independent samplewise scales.
Means, covariances, joint densities and conditional updates match independent state-matrix
calculations; batch and 1 ms chunked updates also agree.

Each of the same 108 controls supplies two independent chosen-action episodes, sound and wait:
216 episodes. Fixed-variance Gaussian and variance-learning controls start with identical means
and covariances. All initial forecasts are saved first. Each episode then observes only its
selected branch's first 10 ms and forecasts the remaining 40 ms. Updated distributions and
native habitat perceptual outputs are saved before opening the remaining target. Another
action's outcome belongs to a separate state copy. Modifying the unchosen action or the
unobserved suffix leaves that episode's updates and issued forecasts unchanged in regression.

Amplitude updates and mean waveforms remain identical between controls; PCM squared error
therefore matches in every episode. Learned variance changes relative to its initial value
as shown below. Joint density of the same future sound improves in all 216 episodes, without
establishing calibration. Coverage is the average over bins and episodes for a 32-draw central
90% quantile interval; the displayed range spans R/H/C. Correlated bins, routes and action pairs
are not independent listening trials.

| Environment (72 episodes each) | Median variance ratio | Fixed-variance coverage | Learned-variance coverage |
|---|---:|---:|---:|
| Held | 1.046 | 84.6–90.6% | 95.4–97.0% |
| Release | 2.047 | 17.4–49.5% | 19.6–51.2% |
| Release plus new onset | 2238 | 8.3–41.9% | 51.1–58.5% |

The largest new-onset variance ratio is approximately 10007. Intervals widen and average
harmonicity/consonance errors decrease, but roughness error increases. The comparison includes
both variance updating and a different predictive family; it does not isolate their individual
effects. Inflating variance does not acquire missing frequency structure or context. Indefinitely
retaining the same trajectory/response is not adopted as cognitive long-term memory.

The 54 presentation-only pairs retain exactly identical updated forecasts and actual habitat
results across actions. Initial forecasts match across 72 environmental triples. Subtraction of
habitat-routed own sound retains float32 mixing roundoff: the maximum external-wave difference
across two actions sharing the same environment is 1.12e-8, not evidence of another actor reacting.
The observation bus remains habitat, not ListenerTwin presentation or biological feedback.

The default fixed-variance path preserves hashes for six prior cases, 384 sampled trajectories
and 12 means. All 721 Rust tests (nine ignored), 289 Python tests and all-target Clippy/check pass;
native exports pass separately in both stages. Evidence is under
`target/phrase-expectation/2026-09-08-chosen-outcome-learning/`. Production action selection,
sound and author auditions are unchanged. Next, structural changes and added components need
comparison from observations rather than absorbing every mismatch into residual variance.
Local action-value learning, production integration, persistent perceptual attribution, relational
expectation, cognitive parameters, phrases and closure remain open.

### 9.3.25 Retaining Conditional Structural Predictions

A new research `AcousticModelBank` retains continued-state, changed-amplitude and
added-component candidates. `propose_hypotheses` now separates proposal construction
from the common comparison block; the preceding fixed-variance family-winner API
is preserved. Each candidate updates amplitude, residual state and residual variance.
Candidate support and AR response remain fitted conditional estimates.

[Hoeting et al. (1999), section 1](https://sites.stat.washington.edu/www/research/online/hoeting1999.pdf)
provides the mixture and within/between-model uncertainty formulas. Here the candidates
are already conditioned on observed proposal data `H`. Before reading comparison
block `B`, each available family receives equal total mass, divided over its admitted
candidates. With this declared `prior_j`, the subsequent update is
`w_j ∝ prior_j * p_j(B | H)` and the future distribution is
`p(Y | H,B) = sum_j w_j * p_j(Y | H,B)`. Its variance includes both each candidate's
variance and the squared displacement of its mean from the mixture mean.
This is a conditional predictive weighting scheme, not full integrated evidence over
support/AR fitting or a probability of perceived source identity. Family masses,
within-family enumeration and proposal windows are experimental choices, not DCC-derived
cognitive parameters. Duplicating candidates with conserved total prior mass preserves
predictions; arbitrary enumeration changes need not do so.

The bank keeps finite log weights even when their exponentials underflow, allowing
later evidence to restore support. No low-weight candidate is pruned. An issued mixture
is immutable. Sampling chooses one candidate for an entire correlated trajectory;
a product of samplewise mixtures would describe a different joint distribution.

The same 108 native controls supply 216 independent chosen-action episodes. Only the
chosen first 8 ms is used for adaptation; proper finite candidates and their prior
weights are saved before the next 2 ms enters comparison. Remaining 40 ms predictions
and native habitat R/H/C outputs are saved before the actual target is opened.
The controls are the continued variance-learning model and the largest-weight single
candidate. All share the observed auditory state and pending PCM. Native results use
its final complete hop, not a padded target. These reused conditions are not new held-out
inputs, biological observations or independent listening trials.

The table compares mixture against continued state. PCM classifications use a relative
1e-6 tolerance. The error ratio is the ratio of episode-mean squared errors, not the
mean of individual ratios. Coverage averages 32-draw central 90% intervals across bins
and episodes; the range spans R/H/C. Correlated observations and finite sampling prevent
interpreting these counts or intervals as calibrated perceptual success.

| Environment (72 episodes) | PCM improved / tied / worse | Mean PCM error ratio | Mixture R/H/C coverage |
|---|---:|---:|---:|
| Held | 48 / 24 / 0 | 0.182 | 83.6%–90.7% |
| Release | 60 / 12 / 0 | 0.294 | 36.7%–61.6% |
| Release plus onset | 12 / 12 / 48 | 1.717 | 53.2%–62.1% |

For release plus onset, mean R/H/C squared-error ratios are 0.213 / 0.204 / 0.209
relative to continued state. Yet mean PCM error increases to 1.717 times the control.
Thus waveform and native auditory errors disagree in this comparison. Improved R/H/C
prediction does not establish stream recognition, relational expectation or musicality.
The same future joint density improves in 60 release and 60 new-onset episodes, with
12 ties each; held sound has 12 improvements, 24 ties and 36 losses (absolute 1e-7
tolerance). Lower mean error does not itself establish probabilistic calibration.

Mixture against the largest-weight single candidate has mean PCM ratios of
0.934 / 1.003 / 0.971 across the three environments. Its native mean errors are not
uniformly smaller, and Monte Carlo draws need not match between these distributions.
A separate advantage from averaging remains unestablished. Of 216 episodes, 108 put
more than 99% weight on one candidate. Only 1–69 candidates per episode are available;
52–114 proposals fail numerical/proper-posterior checks. New-onset single-candidate
winners comprise 12 added-component, 48 changed-amplitude and 12 continued-state cases.
These family names are not identified causes; residual AR state can carry periodicity.

The future-window diagnostic finds new-onset PCM error worsening especially at
20–40 ms. The mean affine-candidate weight there underflows to zero, so the observation
does not identify affine-amplitude extrapolation as its cause. The short observation's
support, fixed phase/AR dynamics and prediction horizon need further examination.
The 40 ms scoring window is not adopted as a cognitive retention or phrase duration.

All 54 presentation-only action pairs have identical updated distributions and native
habitat observations. All 216 continued controls match the preceding experiment's
PCM error and joint density within numerical tolerance. Independent weight normalization,
real Student batch/chunked updates, immutable forecasts, and separate adaptation,
comparison, future and unchosen-action I/O boundaries pass. The six legacy family-winner
records and forecasts match exactly after refactoring. All 721 Rust tests (nine ignored),
294 Python tests and all-target Clippy/check pass; native exports complete for every
predicted and actual episode. Evidence is under
`target/phrase-expectation/2026-09-08-structural-predictions/`.

The next work must address supported acoustic dynamics and prediction horizon while
retaining separate native perceptual outcomes and uncertainty. Local action-value
learning, persistent perceptual attribution, relational expectation, cognitive parameters,
phrases, closure and production integration remain open. No production sound, author
audition or commit is added.

### 9.3.26 Refitting Frequencies and Non-Growing Carrier Gains

An optional `refitted_passive` family now estimates frequencies and logarithmic gains
directly from the chosen adaptation PCM. It uses fixed-frequency carriers with gains
`g <= 0`: `r = exp(g/fs)` gives non-growing individual modes, including the undamped
limit. Modal frequency, damping and linear gains are distinct parameters; see
[Maestre et al. (2017), section 2](https://dafx.de/paper-archive/2017/papers/DAFx17_paper_95.pdf).
The code uses projected time-domain Gauss–Newton trials with linear amplitudes re-solved
before loss evaluation. It does not reproduce that paper's frequency-band optimizer,
claim a global minimum, or clip an already fitted pole while retaining its old amplitude.

This is an explicit physical hypothesis about unforced carriers. Their superposition
need not have monotone sample amplitude, residual AR drive remains stochastic, and
other families remain available. The fixture includes attack/release and body motion;
its external synthesis settings are audit metadata, never inference inputs. Carrier
phase here does not define a musical beat or an individual's participation phase.
Decay values are fitted acoustic parameters; candidate counts, observation windows and
iteration limits remain numerical choices, not derived cognitive retention durations.

The previous 8 ms adaptation, 2 ms comparison and 40 ms target boundary is preserved.
The same 108 controls supply 216 independent chosen-action episodes. Available families
receive equal total prior mass before comparison. A separate `retained` forecast
restricts and renormalizes the bank to the old families, recovering the preceding
mixture at the same issue and target time. This comparison combines fresh fitting,
constrained dynamics and an additional family; the sole effect of passivity is not isolated.
The controls are reused data, not new held-out inputs or listening trials.

| Environment (72 episodes) | PCM improved / tied / worse | Mean PCM error ratio | Mean R/H/C error ratios |
|---|---:|---:|---:|
| Held | 12 / 60 / 0 | 0.9994 | 1.000 / 1.000 / 1.000 |
| Release | 36 / 36 / 0 | 0.6299 | 0.983 / 1.000 / 0.998 |
| Release plus onset | 60 / 0 / 12 | 0.6900 | 0.019 / 0.097 / 0.051 |

Ratios compare the extended mixture with the old-family mixture. PCM classifications
use relative tolerance 1e-6; ratios divide episode-mean squared errors. New-onset
R/H/C errors decrease markedly, but its PCM error increases in 12 episodes, reaching
1.603 times the control in the worst case. Joint future density improves in 204 episodes
and ties in 12 (absolute 1e-7 tolerance). These scores do not establish source identity,
relational expectation, musicality or probabilistic calibration.

Uncertainty remains inadequate. For release, R/H/C central-90% coverage changes from
61.6 / 36.7 / 52.2% to 55.2 / 26.1 / 44.5%. For release plus onset it changes from
62.1 / 53.2 / 58.7% to 65.0 / 42.2 / 54.7%; intervals narrow. Frequencies, gains and AR
responses are still conditional point estimates, not integrated parameter uncertainty.
Held-sound native scores from the 32 draws match exactly, although the mixture distributions
do not. Small-weight alternatives can affect exact joint density without appearing in
this finite native ensemble. Mean-error improvement alone cannot settle uncertainty quality.

The new family has mean weight 0.000354 / 0.176891 / 0.661154 across the three environments,
and supplies the maximum-weight candidate in 0 / 12 / 48 episodes. All 16,416 admitted
new candidates have nonpositive fitted gains; total available counts range from 73 to 149.
These are predictive weights and numerical models, not identified causes or body counts.

Known two-mode decay and unseen continuation are recovered at 24/48 kHz. Growing-input
fits distinguish constrained amplitude re-solving from posthoc clipping. The old-family
subset matches the preceding PCM error/density within numerical tolerance in all 216
cases; 54 presentation-only pairs match across actions. Existing six family-winner
forecasts remain bit-exact with the option disabled. Adaptation/comparison/future and
unchosen-action boundaries still pass. All 721 Rust tests (nine ignored), 297 Python
tests and all-target Clippy/check pass, as do all 432 native prediction/actual exports.
Evidence is under `target/phrase-expectation/2026-09-08-passive-dynamics/`.

Next, continuous frequency/decay/AR estimation uncertainty and the native contribution
of low-weight alternatives need explicit treatment. Cognitive parameters, sustained
perceptual attribution, relational expectation, phrases, closure, local action-value
learning and production integration remain open. Production sound, auditions and commits
are unchanged.

### 9.3.27 Conditional Predictions with Continuous Frequency and Decay Uncertainty

`scripts/acoustic_parameters.py` integrates frequency and decay conditional on a fixed
mode count and AR response. Proper amplitude and variance priors keep marginal densities
finite when nearby or duplicate modes leave amplitudes unidentified. For whitened observations
\(z\), real cosine/sine design \(X_\eta\) and coefficients \(c\), use
\(c\mid q\sim N(0,q\lambda^{-1}I)\) and \(q\sim\operatorname{InvGamma}(a_0,b_0)\).
The physical amplitude origin is fixed at the observed block start; the prior does not
apply to numerically normalized coefficients. The inverse-Gamma convention follows
[Murphy (2007)](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).
Condition on the initial \(p\) samples and score subsequent AR innovations.

\[
V=\lambda I+X_\eta^T X_\eta,\qquad m=V^{-1}X_\eta^Tz,\qquad
a=a_0+n/2,\qquad
b=b_0+\tfrac12\left(\lVert z-X_\eta m\rVert^2+\lambda\lVert m\rVert^2\right).
\]

An augmented QR implements this calculation. The amplitude/variance-integrated marginal
likelihood compares continuous parameters \(\eta\); their retained posterior states give
a mixture of conditional Student forecasts. One parameter state generates a whole sampled
trajectory, rather than a different state per PCM sample. Mode labels are exchangeable,
not perceived source identities.

The declared priors are uniform in log2 frequency over 20–8000 Hz and logarithmic gain
over −1000–0 /s, with \(\lambda=10^{-4},a_0=2,b_0=10^{-6}\). These are acoustic modeling
choices and numerical settings, not DCC-derived cognitive constants. The earlier observed
8 ms adaptation and 2 ms comparison select a passive candidate's count, AR response and
observed suffix. Frequency/decay inference uses that observed suffix and predicts the same
40 ms future. Count, AR response and suffix-selection uncertainty remain conditional.

Symmetric reflected Gaussian and uniform proposals use
[Metropolis updates](https://doi.org/10.1093/biomet/57.1.97). Four initial chains include
one prior draw. Step sizes adapt only during 512 discarded warmup sweeps; subsequent steps
are fixed, retaining 256 states per chain at two-sweep intervals. Repeated states retain
their multiplicity in the 1024-state empirical mixture. Rank-normalized split/folded R-hat
and bulk ESS follow [Vehtari et al. (2021)](https://arxiv.org/abs/1903.08008), applied to
frequency-ordered parameters and marginal log likelihood. R-hat < 1.01 and bulk ESS ≥ 400
provide an initial screen, not a guarantee about tails or predictive functions. Episodes
failing that screen remain in the comparison.

Two controls use identical amplitude/variance priors: fixed parameters at the highest
marginal density visited, and the original fitted parameters. The visited point is not a
proven global MAP. The preceding structural mixture remains another control; changing
priors must not be attributed to parameter integration alone. When the residual AR already
represents periodic sound, broad explicit frequency/decay uncertainty can coexist with
almost unchanged predictions. Chain agreement, parameter identifiability and predictive
accuracy require separate checks.

One-mode predictive means and variances are checked against independent two-dimensional
quadrature. Proper marginal likelihood and Student predictions are checked against dense
matrix calculations. Sequential evidence additivity, unobserved amplitude directions,
mode permutation and future/unchosen-action information boundaries are covered. Batched
lattice columns retain the arithmetic order of individual columns; batch size is a memory
budget for computation, not cognitive capacity.

The existing 108 inputs supply two chosen-action episodes each, totaling 216. Ratios
below divide episode-mean squared errors of integrated parameters by the best visited
point using the same proper priors.

| Environment (72 episodes each) | PCM improved / tied / worse | Mean PCM MSE ratio | Passing initial exploration screen |
|---|---:|---:|---:|
| Held | 36 / 0 / 36 | 1.3266 | 36 / 72 |
| Release | 36 / 0 / 36 | 1.0290 | 48 / 72 |
| Release plus onset | 0 / 0 / 72 | 1.0392 | 48 / 72 |

PCM classifications use relative tolerance 1e-6. Mean errors increase relative to the
point control, so this configuration does not establish a benefit from integration.
Against the preceding full structural mixture, mean PCM MSE ratios are 234.36 / 2.679 /
0.651, but priors and count/AR selection also differ. Native R/H/C mean-error ratios
against that mixture increase to approximately 31424 / 25050 / 25377 for held sound,
3255 / 295 / 543 for release, and 108 / 18 / 39 for release plus onset. This pathway
is not adopted as a replacement for the preceding mixture. Native quantities use
32 trajectory draws; low-mass contributions and Monte Carlo error remain unresolved.

The initial exploration screen passes in 132/216 episodes; maximum R-hat is 1.1413 and
minimum bulk ESS is 20.1. For the largest disagreement, unchanged priors with 2048 warmup
sweeps and 1024 retained states per chain at four-sweep intervals reduce maximum R-hat
to 1.0289 and increase minimum ESS to 156.0, still failing the screen. This diagnoses
exploration length rather than providing a fresh held-out accuracy test. In the one-mode
pilot, decay standard deviation remains about 98% of its prior value while predictions
change little. A fixed AR response can also represent periodicity, so explicit-mode
estimation alone cannot demonstrate acquisition of an acoustic body.

The preceding mixture's metrics and all component forecast hashes match exactly in
216 episodes; 54 presentation-only pairs also match. All 721 Rust tests (nine ignored),
304 Python tests and all-target Clippy/check pass. All 757 native input files match
the earlier manifest. The 104 Rust/Cargo production files remain unchanged from the
preceding checkpoint. No author audition, production sound/action selection or commit
is added. Native evidence is under `target/phrase-expectation/2026-09-08-parameter-uncertainty/`.
Low-mass native contributions, AR-response uncertainty and empirically supported priors
remain open, as do cognitive retention times, stream identity, relational expectation,
phrases, closure, action-value learning and production integration.

### 9.3.28 Conditional Driven Vibration State without a Duplicate Resonant Residual

The explicit component and residual AR in 9.3.27 can explain the same periodicity.
For a 441 Hz component with log gain −17/s and an AR(2) with matching poles,
component amplitude and retained residual state compensate. At four amplitude
precisions from 1e−8 to 1e4, adding the component changes conditional log density,
forecast mean and variance by exactly zero. Extra amplitude uncertainty does not
identify an additional acoustic body.

`scripts/acoustic_body.py` instead retains one vibration state with ongoing drive.
For each mode's two real quadratures, the conditional acoustic hypothesis is

\[
dx_j=(g_j I+2\pi f_j J)x_j\,dt+\sqrt{q d_j}\,dW_j,
\qquad y_n=\sum_j[1\;0]x_{j,n}+v_n,\quad v_n\sim N(0,q).
\]

Here \(J\) generates planar rotation, \(g_j\leq0\) is damping, and \(d_j\geq0\)
is continuous quadrature drive intensity relative to measurement variance.
Independent isotropic drive is an acoustic hypothesis, not an identified excitation
mechanism or perceptual body. There is no second resonant AR state. Over
\(h=1/f_s\), the transition is a rotation scaled by \(e^{g_jh}\), and normalized
process covariance is \(d_j\operatorname{expm1}(2g_jh)/(2g_j)\,I\), with limit
\(d_jhI\) at zero damping. Linear SDE discretization and filtering follow
[Särkkä and Solin (2019), §§6.2 and 10.6](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).

The initial state has prior \(x_0\mid q\sim N(0,q\lambda^{-1}I)\), with
\(q\sim\operatorname{InvGamma}(a_0,b_0)\). Conditional Kalman and conjugate scale
updates issue a joint Student forecast containing retained vibration, future drive
and measurement uncertainty. One scale draw applies to an entire trajectory.
Issued forecasts own frozen copies of state. Known owned sound is added as a waveform,
then the sum passes through the same native auditory analysis state. New observations
retain existing vibration; predictions do not advance observed analysis history.

A dense covariance built independently from the continuous-time kernel verifies
conditioning, marginal density, forecast moments and joint density. Zero drive
recovers the proper-prior passive amplitude model. Composing 48 kHz transitions
and drive covariances matches a 24 kHz step. This checks latent discretization,
not equivalence of auditory or cognitive inference across sampling rates.
Trajectory draws retain observation-induced cross-mode covariance and dependence
from the common unknown scale. Splitting one mode into two equal-frequency modes
with half the initial covariance and drive preserves its prediction: body count
and perceptual stream identity remain unidentifiable from those observations.

An optional `driven_body` plan reuses passive modal parameters and fitting-start
candidates from the observed 8 ms adaptation. Duplicate residual AR orders do
not duplicate body prior mass. Declared drive intensities are 0, 100 and 10,000/s,
with \(\lambda=10^{-4},a_0=2,b_0=10^{-6}\). Candidate weights update only on the
subsequent common 2 ms block. State-conditioning scores on the fitting interval
are not treated as integrated structural evidence. Frequencies and damping remain
point estimates; the continuous-parameter integration of 9.3.27 is not transferred.
Forecasts and 32 whole trajectories over the following 40 ms are saved before actual
future audio is scored. Tests alter future and unchosen-action audio without changing
issued predictions, while changing comparison evidence changes subsequent predictions.

The comparison completed all 216 episodes from 108 reused inputs and two separately
chosen actions. Mean squared-error ratios against the previous structural mixture are:

| Environment | PCM | Roughness | Harmonicity | Consonance |
|---|---:|---:|---:|---:|
| Held | 0.7961 | 0.0000600 | 0.0000309 | 0.0000425 |
| Release | 1.8849 | 0.4143 | 0.6337 | 0.6181 |
| Release and new onset | 0.8017 | 1.5418 | 0.3486 | 0.6389 |

PCM improves/worsens in 68/4, 24/48 and 36/36 episodes respectively. Joint predictive
density is worse in all 216. This is retained as a research state model, not adopted
as a production replacement. Against the same bodies' zero-drive mixture, mean PCM
ratios are 1.00006, 0.9350 and 0.1605, with joint density improving in every episode.
Body parameters, drive ratios and the measurement model remain uncalibrated.

Prior-scale sensitivity reuses three environments × two actions at sine/habitat/24 kHz/
seed 184957, multiplying the inverse-Gamma scale by 0.01 or 100. The larger scale gives
approximately 0.463 times the release PCM error but 5.78–5.90 times its perceptual error.
The smaller scale improves joint density in these six episodes; this diagnostic reuse
is not a selection of a default prior.

Held-condition driven mass is 0.000381–0.001908. Nevertheless all 72 × 32 = 2,304
sampled trajectories are bit-identical to those drawn from the zero-drive subset.
The probability of no driven choice in 32 draws at those masses is about 94.1–98.8%.
Identical perceptual scores against the passive subset therefore include a finite-draw
omission, not evidence of identical predictive distributions. Next, component-weighted
perceptual integration and its numerical uncertainty must preserve admitted alternatives'
contributions. Retaining a candidate does not ensure that a downstream sample uses it.

The previous mixture's metrics and all component forecasts match in all 216 episodes;
54 presentation-only action pairs and one pilot replay also match exactly. Rust has
721 passing tests with 9 ignored; Python has 311 passing tests; the site builds.
All 104 production Rust/Cargo files match the previous checkpoint. No new audition
or commit is added. Evidence is under `target/phrase-expectation/2026-09-08-driven-body-state/`.
Cognitive retention, relational expectation, phrases, closure, local action-value
learning and production integration remain open under the 9.3.5 contract.

### 9.3.29 Component-Weighted Perceptual Integration

In 9.3.28, retained candidates can disappear from a finite set of predictive draws.
`scripts/perceptual_sampling.py` stratifies trajectories by acoustic candidate and
integrates each candidate with its original mass. This applies
[Owen's stratified estimator, §8.4](https://artowen.su.domains/mc/Ch-var-basic.pdf)
to a discrete conditional acoustic mixture; it does not learn new model weights.

For candidate weight \(w_k\), count \(n_k\), transformed trajectory value \(f_{ki}\)
and retained set \(S\), compute \(\hat\mu_S=\sum_{k\in S}w_k\bar f_k\).
Do not renormalize retained mass. Omitted mass \(m\) contributes between zero and
\(m\) to expectations of observables in \([0,1]\). This applies to roughness and
harmonicity states and consonance levels, not unbounded PCM or raw potentials.

Within-candidate sample variances estimate numerical variance as
\(\sum_k w_k^2s_k^2/n_k\). Independently, bounded independent draws give the
coordinatewise Hoeffding radius
\(r=\sqrt{\tfrac12\sum_k(w_k^2/n_k)\log(2/\alpha)}\).
The numerical interval clips \([\hat\mu_S-r,\hat\mu_S+m+r]\) to \([0,1]\).
It concerns integration of a fixed predictive model, not coverage of future
observed audio or simultaneous coverage of every frequency bin. Floating-point
error remains separate.

After allocating a minimum to each retained candidate, increase counts until
\(\sum_k w_k^2/n_k\leq1/N\). Minimum allocations to small masses must not starve
dominant candidates. This keeps the bounded-observable worst-case variance bound
no worse than \(N\) ordinary draws, without guaranteeing smaller actual variance.
The target, minimum and omitted-mass tolerance are numerical settings. Each model's
sampler still preserves whole-trajectory correlation and shared scale uncertainty.
The acoustic bodies and predictive distributions themselves are unchanged.

An optional `perceptual_sampling` plan adds this path to the existing comparator.
Ordinary draws and their predictive quantiles remain available. Separate
`stratified_fields` and `integration-moments.npz` record weighted means, omitted
mass, numerical standard errors and intervals. Error at the midpoint of the
uncomputed tail contribution is explicitly named `midpoint_mse`; numerical bounds
do not replace predictive quantiles. A weight-0.001 candidate omitted by 32 ordinary
draws still contributes under stratification. Tests cover omitted mass, log-weight
underflow, single components, numerical variance and the future-information boundary.

The completed primary comparison reuses three bodies × three environments × two actions
at 24 kHz, habitat, seed 184957: 18 episodes. The three sine environments also receive two stratified draw-seed
replications and a target-512 numerical reference, six episodes each. The ordinary draw seed stays fixed. Main settings are target 32,
minimum 4 and omitted mass 1e−10; the reference uses 512, 16 and 1e−12.
An earlier pilot used minimum allocation without the dominant-component variance
guard; its source and results are preserved separately. Main allocations use 32–72
trajectories, with omitted mass at most 5.23e−11. Driven candidates receive draws in
every held case. All 54 primary/replication/reference forecasts preserve the previous
predictive distributions, ordinary sampled waveforms and ordinary-sampling metrics exactly.

For the body mixture, ratios of mean error against actual future auditory fields,
stratified over ordinary estimates, are:

| Environment | Roughness | Harmonicity | Consonance |
|---|---:|---:|---:|
| Held | 1.2965 | 1.3257 | 1.3299 |
| Release | 0.9544 | 0.9001 | 0.9211 |
| Release and new onset | 0.3830 | 1.2510 | 0.8372 |

This compares numerical estimates of the same conditional expectation, not a change
to the acoustic model. The target-512 reference uses 512–670 trajectories. Mean squared
deviations from that finite reference, divided by the ordinary estimate's deviation,
are 1.3983/1.8666/0.00618 across the three stratified seeds for roughness,
1.3934/1.8759/0.00650 for harmonicity and 1.3962/1.8705/0.00631 for consonance.
The ordinary seed is fixed, so this is not general evidence of reduced variance.
Maximum estimated reference standard errors remain 5.84e−5, 1.97e−4 and 6.41e−5.
Conservative Hoeffding radii are about 0.239–0.240 in the main run and 0.060 in the
reference; they do not establish convergence of small field differences.

Retained candidates now receive draws; deliberately omitted mass is explicitly reported
without renormalization. Numerical convergence and predictive calibration remain distinct open issues.
Rust has 721 passing tests with 9 ignored; Python has 315 passing tests. Evidence is under
`target/phrase-expectation/2026-09-08-perceptual-integration/`. Predictive calibration,
relational memory, phrases, closure and production action selection remain open.
This section forecasts continuation after observing 10 ms following the chosen action;
it does not test pre-action choice value. Next, branch sound/wait from the same
pre-action evidence and share environmental predictive trajectories to estimate their
perceptual difference. Keep prospective prediction, learning from the chosen outcome,
and offline evaluation of unchosen branches separate.

### 9.3.30 Prospective Action Branches and the Limits of Chosen-Outcome Updates

The shared environmental trajectories of 9.3.23 now connect to the retained candidates and
perceptual integration of 9.3.25–9.3.29. `scripts/evaluate_prospective_actions.py` fits on the last
8 ms before a common 2 ms comparison block ending at action issue. It samples 50 ms of external
continuation from that evidence and adds each owned sound/wait waveform to the same trajectories.
Both branches start from the same observed native analysis state. This implements common random
numbers ([Owen, §8.6](https://artowen.su.domains/mc/Ch-var-basic.pdf)). Signed differences are
integrated directly over `[-1,1]`, retaining omission mass and numerical error separately.
An exactly zero paired effect remains exactly zero numerically.

After the initial predictions are sealed, each prescribed chosen episode clones the original
models independently and observes only its selected 10 ms actual prefix. State, weights and
variance update before issuing the remaining continuation. Neither unchosen audio nor the
selected later outcome is read for evaluation until those predictions are sealed. Initial and
updated estimates target the same complete native hop, about 45.33 ms after action issue.
The two choices are separate research interventions, not actions selected by learned values.
The 8/2/10/50 ms windows are numerical assay settings, not identified cognitive or phrase lengths.

All 108 reused conditions completed: three bodies, two routes, 24/48 kHz, three future environments
and three seeds. The three environments have identical pre-action evidence within each of 36
groups. Candidate forecasts, draws, native predictions and integrated effects remain identical;
only the native contract's case ID differs. All 216 isolated chosen continuations reproduce the
same actual target when the observed prefix and actual tail are analysed separately. All 54
presentation conditions, whose owned action stays outside the ecological bus, have exactly zero
predicted and actual effects. Metamorphic tests separately change later selected outcomes,
unchosen outcomes and the selected prefix to check their access boundaries.

A zero-effect baseline alone confounds environmental prediction with knowledge of one's own
waveform. An exploratory control added after the primary campaign therefore keeps the exact
owned future and full observed analysis history while setting only the external future to zero.
It fits no parameters and reuses 36 matched-group representatives (24 distinct evidence sets).
The table divides mean squared
error of the predicted action effect by that control's error, over 18 habitat cases per environment.

| Environment | Predictor | Roughness | Harmonicity | Consonance field |
|---|---|---:|---:|---:|
| Held | Structural mixture | 0.002775 | 0.001298 | 0.001559 |
| Held | Vibration/drive mixture | 1.619e−5 | 8.948e−7 | 2.419e−6 |
| Release | Structural mixture | 0.03896 | 0.04909 | 0.05226 |
| Release | Vibration/drive mixture | 0.04189 | 0.05592 | 0.05948 |
| Release plus new onset | Structural mixture | 4.332 | 0.2253 | 0.3363 |
| Release plus new onset | Vibration/drive mixture | 4.506 | 0.2425 | 0.3587 |

Environmental prediction helps in held and release conditions, but both models lose to this
control on roughness after a new onset. Chosen-outcome updates are also uneven. For the
vibration/drive mixture, updated errors in held conditions are about 0.196–0.285 of the original
forecast errors. With a new onset, they increase to about 19.60–22.30 for wait and 4.02–10.93 for
sound, at the same future target. Updating states and weights cannot introduce a new frequency
structure into the pre-action candidates. Next, candidates must be proposed from selected actual
evidence and compared with retained candidates on the same subsequent observations. This must
distinguish missing candidate structure from effects of state and variance updates.

An exploratory reuse of the 18 prior 24 kHz, habitat, seed 184957 episodes in 9.3.29
matches the actual history through 10 ms and the actual target exactly. Post-action proposal
errors divided by the present retained-candidate update errors are 0.07671/0.07610/0.07685
for held, 0.1922/0.2165/0.1982 for release and 0.1742/0.5373/0.3790 for release plus onset,
in roughness/harmonicity/consonance order. Individual new-onset cases still worsen.
State priors are also refreshed and integration seeds differ; this does not isolate candidate
addition as the cause of the improvement.

The initial 216 forecasts use 44–86 draws, with omitted mass at most 9.99e−11. Sharing the
environment does not uniformly reduce numerical variance. Across the 18 distinct habitat
observed states, estimated paired/independent variance ratios for roughness, harmonicity and
consonance are 1.009/1.001/1.002 for the structural mixture and 0.890/0.924/0.909 for the
vibration/drive mixture. These are estimates from one draw set, not replicated sampling variances.
The signed-effect Hoeffding radii remain conservative at about 0.473–0.480. Numerical convergence
and predictive calibration remain open.

Reconstructing 36 saved body banks verifies the shared-helper change: all 54 prior forecasts and
810 default-range integration arrays are identical. Rust passes 721 tests with nine ignored;
Python passes 318. Evidence is in `target/phrase-expectation/2026-09-08-prospective-actions/`.
External sources do not respond to owned actions in these fixtures, so they do not validate
social response learning or the author's perception. Production sound and action choice remain
unchanged. Relational memory, phrases, closure, cognitive parameters, local action-value learning
and production integration remain open.

### 9.3.31 Selected-Evidence Proposals and Separate Retention/Reset Controls

`scripts/acoustic_revision.py` and `scripts/evaluate_selected_revision.py` implement controls
for the deterioration after updating in 9.3.30. **All 108 conditions and 216 prescribed
selected-action episodes completed. Fresh proposals have a useful range, but short-comparison
weights are not yet justified for production adoption.**

The inferred pre-action candidates are reconstructed from saved parameters and actual history,
and must reproduce the previously issued forecast exactly. After observing the selected 8 ms
adaptation, every alternative predicts the same subsequent 2 ms comparison and then the
remaining 40 ms target.

| Alternative | Operation |
|---|---|
| retained | Keep and update vibration state, the common variance posterior and candidate weights |
| state_reset | Reset only the vibration-state prior at action issue, then observe the same 8 ms |
| state_scale_reset | Also reset the common variance prior |
| uniform_reset | Keep the identical state_scale_reset component predictions, replacing only the weights with uniform mass after adaptation |
| fresh | Fit frequency, decay and component-count candidates to the selected adaptation or a suffix; condition every state on the full 8 ms, using uniform weights before comparison |
| revision | Retain both retained and fresh groups, assign their declared 1:1 mass before comparison, then update on the same subsequent 2 ms |

The uniform_reset and fresh controls share state/scale priors and state-conditioning intervals;
their inferred structure and admitted candidate sets differ. Fresh includes a zero-resonance
measurement-noise candidate. Revision keeps tiny weights in log space and declares group mass
before seeing comparison outcomes. New candidates provide predictions conditional on fitted
parameters; adaptation performance is not treated as posterior probability of a new structure.
These resets do not identify cognitive forgetting. Neither the 8/2/40 ms windows nor group masses
are derived cognitive-retention parameters.

Comparison audio is read only after candidate predictions are recorded; evaluation audio is read
only after native predictions are recorded. Unchosen actual audio is never used. Changing the
comparison changes later predictions while leaving already proposed candidates unchanged.
Five added tests cover this access boundary, separate state/scale resets, weight-only changes,
inference of a frequency absent from retained support, tiny log mass and group updates, and the
zero-mode candidate. All 323 Python tests pass. Production Rust is unchanged; all 721 Rust tests
pass with nine ignored.

All 216 episodes reproduce their pre-action model and match the actual history and complete
native target in 9.3.30. The weight-only control preserves every component comparison density.
Splitting the chosen update into 8 ms and 2 ms differs from a single 10 ms update by at most
1.71e−14 in forecast means and 6.63e−17 in variances. The table divides fresh mean squared
errors by retained errors over 36 habitat episodes per environment.

| Environment | PCM | Roughness | Harmonicity | Consonance field |
|---|---:|---:|---:|---:|
| Held | 0.3475 | 0.06119 | 0.07498 | 0.06612 |
| Release | 0.5369 | 0.4019 | 0.5087 | 0.5002 |
| Release plus new onset | 1.0838 | 0.2232 | 0.1712 | 0.2156 |

Reset effects vary by environment. With a new onset, state reset raises the three native errors
to about 1.035–1.036 of retained; resetting scale as well raises them to 1.515–1.570. Uniform
weights reduce them to 0.738–0.799. Improvement cannot uniformly be attributed to a state reset
or to new frequencies alone. Fresh joint density on the unobserved PCM improves in every held
and new-onset episode, but worsens in all 36 release episodes.

**Additional distributional audit.** Mean errors and PCM densities do not fully evaluate native
perceptual distributions. After the primary comparison, `perceptual_scores.py` scores the saved
draws with the negative orientation of the energy score in
[Gneiting and Raftery (2007), equation 22](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
Each perceptual field is a frequency vector, scored separately:

\[
L(P,y)=E\,d(X,y)-\tfrac12 E\,d(X,X'),
\qquad d(x,y)=\sqrt{\operatorname{mean}_{b}(x_b-y_b)^2}.
\]

Independent within-component pairs exclude self-pairs and use denominator \(n(n-1)\).
Original component masses are not renormalized. For omitted mass \(\epsilon\) and retained mass
\(M\), the omitted contribution lies in \([-\epsilon(2-\epsilon)/2,\epsilon]\). Replacing one draw
changes the estimator by at most \((1+M)w_j/n_j\), giving a separate numerical interval through
the [bounded-differences inequality](https://arxiv.org/pdf/1212.5796).
This audit does not change any issued forecast or model weight.

The table gives mean energy loss divided by retained loss; lower is better. Entries are in
roughness/harmonicity/consonance order.

| Environment | fresh | revision |
|---|---|---|
| Held | 0.2745 / 0.3076 / 0.2870 | 0.7258 / 0.7982 / 0.7543 |
| Release | 0.6222 / 0.5705 / 0.5894 | 0.6222 / 0.5705 / 0.5894 |
| Release plus new onset | 0.2778 / 0.2463 / 0.2693 | 0.2778 / 0.2463 / 0.2693 |

Fresh improves all 36 held episodes and 32 release episodes for each field. With a new onset,
it improves 30–31 and worsens five or six. Revision improves the held mean but worsens 20–21
individual episodes. The 2 ms comparison prefers the fresh group in only six of 36 held cases,
although fresh has better density and perceptual distribution scores on the subsequent 40 ms
in every held case. Short-interval fit is not a general measure of longer-horizon reliability;
the evaluated representation and the prediction horizon actually used must remain explicit.

Four new tests cover distinct joint distributions with equal means/marginals, exhaustive
discrete expectations, omitted mass and tiny distances. All 327 Python tests pass. The 1,296
forecasts use 32–88 draws with omitted mass at most 6.83e−11. Conservative score radii remain
about 0.472–0.480, and seeds are reused across conditions. Case averages alone do not establish
numerical convergence or predictive calibration. Evidence is in
`target/phrase-expectation/2026-09-08-selected-model-revision/`.

The next production boundary is distinct. TemporalParticipation currently adds the predicted
context to participation memory when an onset executes. Actual outcomes are joined to reports
but do not update action choice. Integration needs an explicit route retaining predicted intent,
execution and observed consequences, then updating performance on the same forecast target.
Relational memory, phrases, closure, cognitive parameters, local action-value learning and
production integration remain open.

### 9.3.32 Updating Continuous Energy Forecasts from Outcomes at the Same Lead

`AcousticTemporalExpectation` now connects a candidate for delayed energy-forecast
learning to the production path. This updates a continuation forecast, not the
value of a selected action. An error in a forecast without the new action must
not count as that actor's success or failure. Participation memory still stores
predicted context; action-conditioned outcome learning remains open.

For each acoustic band and lead, let \(r\) be recurrence energy, \(p\) persistence of
the last observed energy at issuance, and \(y\) the matching future observation.
Only pairs whose target windows have ended by issuance enter the cumulative fit:

\[
A=\sum(r-p)^2,\qquad B=\sum(r-p)(y-p),\qquad
w=\operatorname{clip}(B/A,0,1),\qquad \hat y=wr+(1-w)p.
\]

This convex combination minimizes past square loss. When \(A=0\), no distinguishing
evidence exists and the existing \(r\) is retained. The coefficient is neither
cognitive confidence nor musical reward. The estimator has no forgetting model;
adaptation to environmental change remains unresolved. Forecasts are issued on
the observation clock, independently of queries and reporting. Saved weights
and candidates stay frozen. Observed silence supplies evidence; missing input
clears pending forecasts and fit statistics without manufacturing silence.
Self-exclusion uses the same weights as the shared projection. Subtracting band
energies remains an approximation, not exact separation of waveform interference.

The existing table covers 10 ms windows beginning 0–4 seconds ahead at 20 ms
spacing. Intermediate queries interpolate; sustained queries beyond the table
retain the observed-mean fallback. These are computational boundaries, not
cognitive retention times or phrase lengths. Pending storage is bounded.

Replaying the pre-change presentation WAVs of Samples 08/09/12, seed 21,
independently reconstructed all 212,670 exported weights bit-exactly as f32 using
only outcomes available at their issuance. Energy square-loss ratios relative
to the existing recurrence forecast were:

| Sample | Lead 0 ms | Lead 80 ms | Lead 4 sec |
|---|---:|---:|---:|
| 08 | 0.2443 | 0.6056 | 1.0914 |
| 09 | 0.3317 | 0.5552 | 1.0607 |
| 12 | 0.3917 | 0.9794 | 1.2521 |

Short forecasts improve, while all three 4-second forecasts worsen. Separating
leads does not establish long-horizon reliability. This is recorded-audio replay,
not an action-conditioned counterfactual comparison or author audition. Production
rerenders change onset times in 08/09; 12 retains its onset sequence and WAV
bit-exactly. The connection affects action, but musical adoption is unassessed.

All 724 Rust tests and Clippy pass; ten research assays remain ignored. Added
checks cover delayed leads/bands, a batch minimization objective across ring
wraps, and continuous energy improvement without new detected onsets. Existing
chunking, query, gap and self-exclusion checks also cover the new forecast.
The constructed observation/forecast path allocates no additional heap memory;
pending storage occupies 1,947,256 bytes. Evidence is in
`target/phrase-expectation/2026-09-08-energy-outcome-feedback/`. Cognitive parameters,
relational memory, phrases/closure and local action-value learning remain open.

A device check used four harmonic entrained Voices, three seconds of warmup
and an eight-second measurement, with reporting off/on. Both retained four live
Voices, zero measurement-window missing output frames and zero callback errors.
Hop p99 was 0.461/0.497 ms against a 10.667 ms budget. Reporting confirmed repeated
onsets from every Voice. ALSA used its default output; a separate PipeWire check
showed MOTU M2 as the default sink, without continuous route or hardware-xrun
verification. The initial sandbox attempt could not connect to the audio server;
the device check succeeded outside that restriction. This does not revalidate
all densities, timbres or long performances.

### 9.3.33 Keeping Pending Candidate Information Separate from Delayed Outcomes

The 4-second degradation occurred in the middle of performances as well as at
their ends. Fitting only completed pairs allowed large switches based on a small
initial set while longer outcomes were still pending. The 9.3.32 update now
separates candidate information known at issuance from outcomes available only
when their target windows end.

Write the change from recurrence as \(x=p-r\), with observed residual \(e=y-r\).
For each band and lead, let \(I_t\) contain issued candidates and \(O_t\) those
whose targets have completed:

\[
G_t=\sum_{s\in I_t}x_s^2,\qquad b_t=\sum_{s\in O_t}x_se_s,\qquad
\beta_t=\operatorname{clip}(b_t/G_t,0,1),\qquad
\hat y_t=r_t+\beta_t(p_t-r_t).
\]

Zero \(G_t\) uses zero \(\beta_t\). This minimizes completed residual square loss
plus \((\beta x_s)^2\) for unresolved candidates. Many pending disagreements
restrain departures from recurrence. Issuance does not update \(b_t\), and unknown
residuals are not stored as observations. Missing input still differs from
waiting for a future target.

[Qiu et al. (2025)](https://arxiv.org/pdf/2506.07595) likewise distinguish known
features and delayed targets in online regression. Here that distinction is
applied to a constrained departure from a baseline forecast. The implementation
does not use their positive regularization or label-range clipping, so their
regret theorem is not claimed. It also identifies no cognitive forgetting time.
The 4-second table, observation clock, self-exclusion and participation reward
remain unchanged.

For development seed 21, 4-second loss ratios to the initial recurrence model
fall from 1.0914 to 1.0164 in 08, 1.0607 to 1.0577 in 09, and 1.2521 to 1.0661
in 12. Two additional seeds, 192001/192013, were declared before comparison and
rendered with the pre-change executable. Replaying those common recordings gives:

| Sample | Completed-only fit | Pending-candidate fit |
|---|---:|---:|
| 08 | 1.0692–1.0751 | 1.0079–1.0346 |
| 09 | 1.0696–1.1837 | 1.0016–1.0307 |
| 12 | 1.0569–1.1828 | 1.0284–1.0320 |

These ranges cover the two additional seeds and share the initial recurrence
model denominator. All six improve over the completed-only fit, but superiority
over recurrence remains unestablished. Some intermediate leads worsen relative
to the completed-only fit. All 638,010 exported coefficients across nine recordings
were independently reconstructed bit-exactly as f32 using only information
available at issuance.

All 725 Rust tests and Clippy pass; ten research assays remain ignored. Checks
cover pending features without fabricated target correlations, rescaling physical
units, and an independent objective across storage-ring wraps. Evidence is in
`target/phrase-expectation/2026-09-08-delayed-energy-fit/`. This improves the
continuation predictor, not action-conditioned value learning. Cognitive retention,
relational memory, phrases and closure remain open.

### 9.3.34 Removing Known Self PCM Before Observing the Surroundings

Before connecting local outcome learning, the decision view needed a correction.
Subtracting self band power from mixture power leaves a cross term:
\(E[a+b]-E[a]=E[b]+2E[ab]\), with self signal \(a\) and external signal \(b\).
Two identical signals make the neighbor appear three times as energetic; opposite
signals produce a negative estimate that the old clamp turns into silence.
A 440 Hz regression reproduced the threefold error before this change.

`OwnSoundHistory` now subtracts the actor's actually rendered, habitat-routed PCM
from the actual habitat mixture, then analyzes the residual with a stateful
`DorsalStream`. It uses no other Voice's private state or labels. Known self
removal does not establish perceptual source identification; floating-point
mixing and subtraction error also remains.

Tracking starts before the actor sounds. Shared pre-birth energy history, filter
state and the unfinished observation window are copied, since all earlier habitat
sound is external. They are not replaced with zeros. All living Voices are tracked,
including those that later switch into participation. Retired-source tails belong
to the surroundings of a new actor. Observation does not alter PCM synthesis or
presentation routing.

Projection uses the shared recurrence candidates and delayed-fit coefficients
from 9.3.33; the external observed mean supplies queries beyond the table. Those
coefficients were fitted to mixture observations, so this is not separately
calibrated local prediction. Onset contrast, cadence reference, action costs and
sound-duration rules are unchanged. The correction supplies better grounded
external observations for subsequent memory and outcome updates; action-conditioned
learning, cognitive retention, relational memory, phrase and closure remain open.

Checks cover reinforcement/cancellation, birth inside an observation window,
presentation-only self routing, and input chunking. All 726 Rust tests pass, with
10 research assays ignored; Clippy and release builds pass. The dedicated allocation
test now includes self observation and verifies zero heap allocations after construction.

Renders of 08/09/12 at seeds 192001/192013 change 08's onset timing in both cases
(counts 200→200 and 202→205); 09 and 12 retain identical onset sequences and WAVs.
There is no new author audition. Device-free sustain/entrained loads with 16 harmonic
Voices and reporting off/on retain all 16 Voices during four seconds after a
two-second warmup. Worker hop p99 ranges from 1.400 to 1.752 ms, with no over-budget
hops; reporting confirms all 16 entrained Voices sounded. This does not test device
callbacks or underruns, and the load runs themselves do not instrument allocations.
Evidence is stored in `target/phrase-expectation/2026-09-08-local-acoustic-context/`.

### 9.3.35 Distinguishing Predicted Context from Experienced Context

Previously, resolving an onset stored the onset-probability contrast predicted before
it sounded. Execution does not establish that the predicted context occurred, and
observed energy cannot simply be substituted into probability-contrast coordinates.
Both memory and candidate comparison now represent external band energy in two windows.

The windows are the first complete observation-grid windows starting at or after the
onset and onset plus 80 ms. Their start/end frames and three-band predictions are frozen
when the candidate is selected. Only an executed onset enters the pending queue. Rejected
or omitted actions and unresolved windows do not update memory. Once both windows have
ended, matching external observations from 9.3.34 update the six-component memory:

\[
\mu_{n+1}=\mu_n+0.1(c_n-\mu_n).
\]

The first complete actual context initializes memory. Stable energy and silence are
observations; expired or off-grid history is unavailable rather than zero-filled.
Issued predictions remain unchanged, and reporting does not enable or disable learning.
Contexts unresolved when an actor ends do not enter that actor's memory.

Candidate energy and remembered energy are separately normalized by their six-component
mass and compared by squared Hellinger distance. This measures energy allocation over
bands and windows, not perceptual confidence. Both numerically silent contexts have
zero distance; one silent context has distance one. Absolute overlap retains its separate
cost. No common onset phase or universal consonance/prediction-accuracy reward is added.

This connects the surroundings actually encountered during a chosen onset to subsequent
participation. It does not identify the action's causal effect or the unchosen outcome.
Action-conditioned mixture prediction and action values remain separate gaps. The 80 ms
offset, coarse bands and 0.1 update rate are engineering choices, not identified cognitive
retention. The rate operates over executed contexts rather than elapsed-time forgetting.

Pending capacity is reserved at construction from the existing maximum onset rate,
observation delay and configured hop span. `participation_context` reports the frozen
forecast, target windows, actual context and updated memory. Existing
`participation_outcome` retains the shared-continuation-versus-mixture comparison.
The evaluator now checks both record types and their observation-window consistency.

Two distinct actual waveforms following identical predictions leave memory untouched until
both windows finish, then yield different memories and subsequent candidate positions.
Checks cover duplicate uptake, unavailable evidence and bounded allocation-free pending
updates. All 729 Rust tests (ten ignored), Clippy and the allocation test pass. The
integration check retains identical WAVs with reports disabled/enabled; 53 Python
tests covering the parser and its existing consumers pass.

Six comparisons cover samples 08/09/12 with seeds 192001/192013; their baselines exactly
reproduce the preceding candidate WAVs. Onset counts change by 200→202 / 205→200 for 08,
702→702 / 694→694 for 09, and 602→602 / 624→623 for 12. Every onset sequence and WAV
changes. All 2,142 observed contexts join executed onsets, and an independent reconstruction
from actual contexts matches all 12,852 memory components exactly in f32. No context falls
outside observed history; uptake lags the second completed window by at most 10 ms.
There are no full-scale output samples. This compares the combined representation and
observed-update change, not an isolated effect of actual feedback.

Four device-free runs cover 16 harmonic Voices with entrained/flow and reports off/on,
using two seconds of warmup and a four-second measurement. All retain 16 Voices with
zero over-budget hops; hop p99 ranges from 1.464 to 2.518 ms against a 10.667 ms budget.
Report-enabled runs demonstrate repeated onsets from every Voice. These runs do not
measure device callbacks/underruns or whole-worker allocations. No new author judgment
applies to these candidate recordings. Evidence is stored in
`target/phrase-expectation/2026-09-08-observed-participation-context/`.
Relational memory, cognitive retention/interference, phrase/closure and musical acceptance
remain open.

### 9.3.36 Retaining the Order Before a Decision as Continuous History

Two-window context describes the surroundings at the onset and 80 ms later. It does
not encode the preceding order. Reversing a low/high pair and then providing the same
silence yields exactly zero energy in both existing context windows at a decision at
0.8 seconds. The current context distance cannot distinguish these inputs. This gap
also respects the distinctions among covariance, directed predictive information and
stream identity in 9.3.12–9.3.14; adding covariance alone would not establish identity.

The continuous history from 9.3.8 is now connected to production habitat observation
and each Voice's external waveform observation. `AuditoryHistory` uses a 13-stage
cascade equivalent to the gamma kernel from the Post inverse of
[Shankar–Howard (2012)](https://sites.bu.edu/tcn/files/2015/12/ShankarHoward12-NeuralComp.pdf).
Inputs are the square roots of the three observed band energies, treated as constant
RMS features over each window, with analytically integrated state transitions. Temporal
precision broadens with age. Eight explicit age modes from 0.125 to 16 seconds and
order 12 match the existing research comparison; these settings are not fitted cognitive
retention or phrase durations. This does not reproduce the proposed neural circuit.

Known-input contribution and observed kernel coverage remain separate. Observed silence
adds coverage; missing input does not. A forward input gap advances existing history
without clearing it or filling an incomplete window with invented RMS. Rewound input
starts a new history. At birth, pre-birth habitat history is inherited as shared observed
background, followed by updates excluding known self PCM. It is not a personal memory
of experiences before the actor existed.

`decision_external_history` freezes this RMS history and its coverage at the latest
observed frame available to the candidate. Later actual context cannot rewrite it.
Only actual audio feeds history; planned onsets do not. Its structure, clock, missing
evidence and self-exclusion are verified before connecting history-conditioned retrieval
and prediction. Current onset costs and the two-window experience memory remain in use.
A history representation alone does not select a response to another stream.

Checks cover constant-input partitioning, joint scaling of time and ages, independent
gamma-kernel quadrature, reversed order, silence versus missing input, inherited history
and waveform self-exclusion. At 24/44.1/48 kHz, the reversed-order controls have identical
zero two-window contexts but pre-decision history differences of 0.01982–0.01984. These
are RMS representation differences, not perceptual discrimination rates. The six controls
and two seeds of 08/09/12 yield 18,906 observations and 604,992 components compared with
the existing double-precision Python model. Maximum absolute f32 error is below 3.97e−5;
relative error with a 1e−8 denominator floor is below 4.65e−5. Evidence is under
`target/phrase-expectation/2026-09-08-continuous-participation-history/`.
All 732 Rust tests (eleven ignored), Clippy, the allocation-free observation/forecast/
self-exclusion check and 53 Python consumer tests pass. Six ordinary sample renders
retain exactly the preceding WAVs, onset records and existing context updates, while
adding 2,142 decision histories. The integration test also retains report-off/on WAV
identity. Four device-free 16-Voice harmonic entrained/flow runs with reports off/on
retain all Voices and have zero over-budget hops. With two seconds of warmup and a
four-second measurement, hop p99 is 1.479–2.520 ms against a 10.667 ms budget. Reporting
runs demonstrate repeated onsets from every Voice. These checks do not measure device
callbacks/underruns or whole-worker allocations, and add no author musical judgment.
Cognitive retention/interference, persistent relational attribution, phrase/closure and
action-value learning remain open.

### 9.3.37 Predicting from Prior History and Comparing Issued Candidates with Sound

Continuous RMS history now conditions future external band-energy predictions. Each Voice learns from actual PCM after removing its known self waveform. Candidates are issued before their targets and compared only after the matching observation windows finish. This is conditional acoustic prediction, not an identified causal effect of acting on another Voice or learned relational value.

**History predictor.** The 57 features comprise an intercept, current three-band RMS, eight ages of three-band RMS, eight coverage values, and 21 directed adjacent-age band contrasts. For older and newer RMS vectors \(o,n\), each band pair \(a<b\) contributes
\((o_a n_b-o_b n_a)/(\|o\|^2+\|n\|^2+10^{-12})\).
Reversing order reverses its sign; proportional common-gain changes contribute zero. The floor is numerical, not perceptual confidence. Ridge regression predicts a correction to current measured energy.

For issued features \(x_i\), current energy \(e_i\), and a matching future target \(e_{i,h}\), the exact estimator is

$$
G=0.01I+\sum_{i\in\mathrm{issued}}x_i x_i^T,\qquad
b_h=\sum_{i\in\mathrm{completed}(h)}x_i(e_{i,h}-e_i),\qquad
\widehat e_h=\max(0,e_t+x_t^T G^{-1}b_h).
$$

Pending designs penalize corrections. They do not supply invented zero-energy observations; this also differs from ordinary completed-only ridge. The fixed ridge and cumulative statistics are engineering estimator choices, not identified cognitive retention or interference. Forward gaps cancel pending matches while preserving learned statistics; rewind resets them.

**Candidate comparison.** The initial additive 36-feature model worsened one of two development cue conditions. Directed contrasts improved both, but unconditional replacement worsened sample 12 energy predictions at 100/200 ms to roughly 1.6–1.9 times the previous error. That replacement is rejected.

The current candidate retains both the previous external recurrence forecast \(a\) and the history forecast \(b\). Separately for each lead and band, completed forecasts and actual targets \(y\) fit
\(w=\operatorname{clip}_{[0,1]}\{\sum(b-a)(y-a)/\sum(b-a)^2\}\), yielding \((1-w)a+wb\). Without comparison evidence, \(w=0\). Both sums use completed matches only. This weight is neither probability nor action value. Local participation requests publish candidate pairs; repeated requests at one observation step are idempotent. Shared report queries do not publish local pairs. Performance is conditional on issued contexts, not proven invariant to issuance frequency.

**Evidence and limits.** Fourteen recordings supply 95,106 completed windows. Independent batch matrix solves agree on 1,008 ridge components with maximum absolute error below 8.63e−10. Independent completed-pair sums also reproduce 5,734,530 mixture-weight components and the corresponding mixed predictions, issued every 100 ms. Registered fresh seeds 7301/7309 and seed 7313 at 48 kHz/amplitude 0.2 give cue-time squared-energy error ratios 0.690/0.700/0.673 against the previous forecast over their last sixteen episodes. Dominant-band matches are 10/8/7 of sixteen: reduced energy loss is not perceptual order classification. Shuffled cue/response controls are retained.

Across six existing ordinary recordings, immediate-window error ratios are 0.889–1.001, while four-second ratios are 1.022–1.315. Long-horizon superiority and universal improvement remain unestablished. These are standalone observations of recorded presentation audio, issued at 100 ms intervals; they do not measure each live Voice's external observations and decision-dependent issuance frequency.

An actual-audio integration test trains on a common past, reverses only the final cue order, and queries in equal current silence. Of 81 combinations of body pace, duration, band and readiness, 27 change their selected onset with the mixed forecast; 26 of these were unchanged with recurrence alone. Zero coupling preserves every paired onset. The first single-body trial changed prediction but not action; that failed trial is retained alongside the broader body comparison. This demonstrates a prediction-to-choice path, not author-audible handoff.

Six ordinary rerenders cover samples 08/09/12 and two seeds. Sample 08 onset counts change 202→199 and 200→203; both sample 09 WAVs remain identical. Sample 12 retains 602/623 onsets while its WAVs change. Independent reconstruction matches all 12,852 f32 memory components from 2,142 observed contexts.

All 737 Rust tests pass with twelve ignored, as do Clippy, the allocation-free observation/forecast/self-exclusion test, fourteen Python report tests, and report-off/on WAV identity. Four device-free runs cover sixteen harmonic Voices, entrained/flow and reports off/on. After two seconds of warmup, four seconds retain all Voices with zero over-budget hops; hop p99 is 1.651–2.594 ms against 10.667 ms. These checks do not measure device callbacks/underruns or whole-worker allocations. Evidence is retained under `target/phrase-expectation/2026-09-08-history-conditioned-expectation/`. Cognitive retention/interference, stream/relationship identity, phrase/closure, action-value learning and musical acceptance of this candidate remain open.

### 9.3.38 Scoring Each Actor's Issued Forecast against Its Subsequent Surroundings

The recorded-audio comparison in 9.3.37 passed presentation audio through one observer and issued forecasts every 100 ms. Production instead observes each Voice's surroundings after subtracting known self PCM, and issues forecasts when participation requires them. Those inputs and issuance conditions differ; recorded-audio ratios are not each actor's measured performance.

`HistoryEnergyPrediction` now retains the mixed forecast computed with issuance-time weights alongside frozen recurrence and history candidates. Later fitted weights never reconstruct the past mixed candidate. When a matching actual window finishes, all three are scored against the same external band energy. Diagnostic leads are nominally 0/100/200/500/1000/2000/4000 ms, recorded as integer observation-window offsets. Squared errors are retained per lead/band with completion counts. These diagnostics do not feed action choice or coefficient updates.

Unique local issuance is counted as well. Repeated requests at one observed context count once. An issuance without a newly completed actual window is recorded with zero completed counts and zero errors. Missing input, retirement of the actor's observer and performance end never supply invented silence or correctness/failure labels. Issued minus completed counts at each lead expose the unscored range.

`local_prediction_error` drains deltas after each actual render, before the next hop can retire tracking. Reporting cannot change forecasts, actions or context memory. A test compares drained and undrained prediction trajectories. Report checks cover observation-window spans, completion-count bounds, issuance-only records, and zero error where no target completed.

Six existing ordinary recordings and two order controls yield 37,506 observed windows. Independent reconstruction from frozen candidates and subsequent sound checks 230,508 squared-error components for 25,612 completed lead-target pairs from 3,750 issuances. Observations, history and all three candidate predictions remain identical to the 9.3.37 exports. The first integration check incorrectly required a score after 4.9 seconds, although the last available target finishes exactly at 4.9 seconds. That arbitrary endpoint requirement was replaced by complete accounting of all 0/200 ms targets and censored four-second targets in the fixture; the failed log remains.

**Production-path comparison.** Six ordinary rerenders score the forecasts actually issued by each actor against its subsequent local observations. The table gives ratios of summed squared errors across actors and bands, not averages of actor ratios. All three candidates face the same realized mixed-policy audio. This is not a behavioral experiment in which recurrence alone generated a different performance.

| Sample / seed | Issued | Completed at 4 s | Immediate error ratio | At 200 ms | At 4 s |
|---|---:|---:|---:|---:|---:|
| 08 / 192001 | 511 | 415 | 0.915 | 1.073 | 1.052 |
| 09 / 192001 | 756 | 604 | 0.858 | 0.953 | 1.027 |
| 12 / 192001 | 192 | 116 | 0.968 | 1.084 | 0.978 |
| 08 / 192013 | 507 | 411 | 0.984 | 1.043 | 1.044 |
| 09 / 192013 | 743 | 594 | 0.935 | 0.937 | 1.036 |
| 12 / 192013 | 207 | 117 | 0.994 | 0.944 | 0.975 |

Of 2,916 issuances, 2,257 have completed four-second targets and 659 remain unscored. Across all issuances, production-path four-second ratios are 0.975–1.052. Restricting issuance to eight seconds onward, as in the earlier standalone assay, yields 0.975–1.140: including the initial period masks some deterioration. Even after matching the cutoff, inputs, issuance frequency and realized audio differ; the standalone maximum regression of 31% is not directly the local result. Aggregation still hides actor differences. Sample 12, seed 192013, Voice 12 worsens about 40% at 100 ms across 23 targets. Voice 14 in that condition worsens about 29% at one second across seventeen targets. Its largest excess-error window is 18.78–18.79 seconds, concentrated in the middle band. At issuance at 17.78 seconds, only five one-second comparison targets had completed for that actor. Sparse comparison evidence is the next diagnostic target, not an established causal explanation. Per-actor and time-local results are retained.

All six WAVs, onset records, prior context records and prior outcome records remain identical before and after this instrumentation. All 738 Rust tests pass with twelve ignored, as do Clippy, the post-construction allocation-free check, fifteen Python tests and report-off/on WAV identity. Four device-free workloads cover sixteen harmonic Voices, entrained/flow and reports off/on. After two seconds of warmup, four seconds retain all Voices with zero over-budget hops and hop p99 of 1.673–2.673 ms against 10.667 ms. Device callbacks/underruns and whole-worker allocations were not measured.

Evidence is under `target/phrase-expectation/2026-09-08-local-forecast-outcomes/`. Band-energy squared error is not calibrated as relational recognition or musical value. Cognitive retention/interference, stream identity, phrase/closure, causal action consequences/value and musical acceptance remain open.

### 9.3.39 Pending Candidate Geometry Does Not Uniformly Improve Local Mixing

The sparse comparisons identified in 9.3.38 motivate a specific estimator comparison, not a new cognitive lifetime. Before inspecting its scores, we registered a shadow mixture that adds the known squared displacement between recurrence `a` and history `b` at every unique issuance, while updating the residual cross-product only after the matching actual target `y` arrives:

$$
 w_{\mathrm{pending}}=\operatorname{clip}_{[0,1]}
 \frac{\sum_{i\in\mathrm{completed}}(b_i-a_i)(y_i-a_i)}
 {\sum_{i\in\mathrm{issued}}(b_i-a_i)^2}.
$$

The denominator includes the current issuance; a zero denominator gives weight zero. Compared with the current completed-only denominator, uncompleted candidate differences penalize correction away from recurrence. They are not invented silent observations, calibrated uncertainty or cognitive interference. Forward gaps preserve these accumulated design statistics while cancelling unmatched targets; rewind resets them. Seven diagnostic horizons retain the issued shadow predictions without returning them to policy.

An independent replay on the same eight stored inputs reconstructs the shadow errors from frozen forecasts and actual targets: 37,506 windows and 76,836 additional squared-error components. All previous observations, history, forecast values and report fields match. Ordered cue continuations retain error ratios 0.691 and 0.673 against recurrence, versus 0.690 and 0.673 for the current mixture. These are reused controls, not fresh validation or perceptual recognition.

Production shadow renders preserve all six WAVs and all previous non-wall-timing report values. The initial identity check incorrectly required measured `hop_timing` execution/wait durations to match; that failed check is retained. Only its three wall-time fields are excluded from the corrected identity comparison. The 2,916 local issuances and 2,257 completed four-second targets are unchanged. For issuances at or after eight seconds, the four-second squared-error ratio of shadow to current mixture is:

| sample / seed | Pending / current |
|---|---:|
| 08 / 192001 | 0.9784 |
| 09 / 192001 | 0.9879 |
| 12 / 192001 | 1.0090 |
| 08 / 192013 | 0.9914 |
| 09 / 192013 | 0.9957 |
| 12 / 192013 | 1.0240 |

The previously concentrated sample-12/192013 errors improve: Voice 12's 100 ms ratio to recurrence falls from 1.403 to 1.059, and Voice 14's one-second ratio from 1.291 to 1.233. However, that Voice's two- and four-second errors increase by about 18% relative to the current mixture, although both remain below recurrence. Global attenuation can reduce harmful corrections while also suppressing useful continuation. Pooled improvement does not identify when a correction is reliable.

Both sample-12 conditions fail the pre-registered no-regression requirement for late four-second loss. The candidate is not adopted. Its temporary runtime scoring and schema extensions are removed, restoring the four affected production/test files byte-for-byte. The tested experiment, binaries, native exports, actual local reports, comparison scripts and failed check remain in `target/phrase-expectation/2026-09-08-pending-local-mixture/`. The experimental full suite passed 739 tests with twelve ignored; the restored implementation passed 738 with twelve ignored. No new behavior or author acceptance is claimed. Next work must distinguish context-dependent predictive support from indiscriminate shrinkage; this result does not select a retention time, prove action value, or complete stream identity, phrase or closure.

### 9.3.40 Issuance Context, Actual Request Time and Completed Local Matches

The rejected global attenuation in 9.3.39 cannot tell which contexts support a useful correction. `local_prediction_match` now records each completed local target together with the frozen recurrence/history/mixed candidates, issued history weight, number of comparisons completed before issuance, and the already-retained 57 input features. The raw-history ring supplies a borrowed context before its slot can be overwritten; an unavailable context remains explicitly absent. Callbacks pass matches through observation and rendering to the existing reporter, with no growing queue or new steady-state allocation. The diagnostic count and records never enter fitting or policy. Pending metadata remains bounded by the existing comparison ring.

The record distinguishes `forecast_observed_frame`, the end of the last complete analysis window, from `requested_frame`, the audio cursor at the first request for that observation state. Repeated requests preserve the first request and frozen candidates. Actual target boundaries are derived from the completed waveform window, rather than assuming model step zero equals audio frame zero. Birth within a partial window, multiple completed windows per call, fragmented input, ring wrap and cancelled targets are tested. The nominal zero-horizon target is the next analysis window: its prefix can already have been rendered when requested. It is not an entirely future waveform interval relative to request time. In the six ordinary cases, request minus observation boundary reaches 448 frames at 48 kHz, or 9.333 ms within a 480-frame window.

Eight reused native inputs retain 37,506 observation windows. An independent join verifies 25,612 matches and 1,459,884 issuance-feature components against previously saved observations and frozen queries; all previous values and score totals match. Six actual local runs add 19,127 matches and reconstruct 172,143 squared-error components exactly to numerical tolerance. Their 1,090,239 feature components include 2,142 selected contexts matched to the existing decision-history records. Prior completion counts are checked against only targets completed by each request. All six WAVs and previous non-wall-timing report fields are unchanged.

A diagnostic comparison was specified before inspecting the traces. The target is signed normalized prediction advantage `(recurrence error - mixed error)/(their sum)`, zero when both errors are zero; this is not action reward. Context distance uses the joint normalized 27 current/history RMS components, coverage divided by `sqrt(8)` and directed contrasts divided by `sqrt(21)`. These scales are analytic conventions, not perceptual calibration. For each actor and horizon, only earlier completed targets can contribute. Predictors are the mean of their advantages, the last completed advantage, and the advantage of the nearest earlier context; nearest ties prefer the latest completed target.

Across the six cases and seven horizons, nearest-context retrieval improves on the past mean in none of the 42 strata. For requests at or after eight seconds, there are 14,729 eligible comparisons and 793 without a completed predecessor. The nearest context is the last completed context in 8,818 comparisons, about 60%. Its pooled squared error is 1.780 times the past mean and 0.966 times the last-result predictor; it beats the latter in 28 of 42 strata. This narrowly tested one-neighbor geometry does not establish useful conditional reliability. It neither disproves contextual memory in general nor identifies a relationship or stream. No retrieval rule is connected to participation, and these reused data are not fresh validation.

The final Rust suite passes 740 tests with twelve ignored; Clippy, sixteen Python report tests, report-off/on WAV identity and the allocation-instrumented observation/match callback test pass. Sixteen harmonic Voices remain active in entrained/flow workloads with reports off/on, after two seconds of warmup and over four measured seconds. All four cases have zero over-budget hops; p99 is 1.699–2.731 ms against 10.667 ms. Report-enabled runs retain 57/146 onsets from all Voices. This is device-free evidence, not hardware callback/underrun or whole-worker allocation acceptance. Renderer profiling includes own-sound observation and enabled match reporting.

Evidence is retained in `target/phrase-expectation/2026-09-08-local-context-matches/`. Next work must separate known candidate displacement and issued weight from the not-yet-observed residual, rather than treating the nearest past relative error as reliable context support. Cognitive retention/interference, persistent relationships, stream identity, causal action value, phrase/closure and musical adoption remain open.

### 9.3.41 Known Prediction Differences and Unobserved Local Residuals

Let $a$ be the frozen recurrence forecast, $m$ the current mixture, and $y$ the later observed energy in the same target window. The correction $\delta=m-a$ is known at issuance, while the residual $r=y-a$ is not. Their squared-loss difference is $\|\delta\|^2-2\delta\cdot r$. This research comparison separates the known term from the unknown residual instead of retrieving one previous relative-error outcome as in 9.3.40. It predicts three-band energy residuals, not causal action effects or value.

Four controls predict the physical loss difference: the mean of previous loss differences, the mean residual vector inserted into the identity, a geometry-conditioned residual ridge, and a geometry-plus-auditory-context residual ridge. Geometry contains six recurrence/history energy values jointly L2-normalized, followed by three issued history weights. Context adds the same 56 normalized auditory features as 9.3.40. The two regressions use 9/65 features, a separate unpenalized intercept, centered cumulative sufficient statistics and fixed ridge penalty 1.0. These are analytic choices, not perceptual distances, cognitive retention lifetimes or calibrated confidence.

Only target windows completed by the request time can train each actor/horizon model. All controls use the same available-context training set; missing context and no previous training remain unavailable. Query interfaces receive no current target outcome. A prospective estimator comparison selects the current mixture when predicted loss difference is negative, otherwise recurrence; unavailable comparisons retain the current mixture. All scores concern the same realized current-policy audio, not the sound or behavioral outcomes of a changed participation policy. Zero correction is known equality, so nonzero-correction strata are scored separately. All seven reported horizons and whole-run/requests at or after eight seconds remain in scope, including the nominal zero-horizon timing qualification in 9.3.40.

The comparison was registered before analysis, using six existing cases for development and six fresh cases from samples 08/09/12 at unused seeds 192029/192041 for validation. Fresh audio uses the unchanged renderer and configuration. Independent reconstruction from source reports verifies scores and estimator selections for 19,127 reused and 19,420 fresh matches, 38,547 in total. Eighty-four selected requests, one per case/horizon, also verify both regressions against 168 independent centered batch fits; the maximum absolute residual-prediction difference is below 2.12e-18. Tests cover future-outcome mutation, exact completion boundaries, missing context, common energy scaling and known zero correction.

Fresh nonzero-correction strata contain 16,336 whole-run and 13,876 late requests, all with earlier training outcomes. Context-conditioned loss-difference prediction has squared error 0.9894/0.9871 times geometry-only and 0.9673/0.9611 times the mean-residual control. However, it has 1.3853/1.3334 times the error of directly averaging previous loss differences. Its selected forecast energy loss is also 1.00170/1.00478 times the unchanged current mixture. These are pooled nonzero-correction scores, not gains in cognition or musical quality.

In late four-second strata, sample 08 at seed 192029 worsens selected loss by about 0.115% relative to geometry-only. Both sample 09 seeds worsen relative to the current mixture by about 3.80%/2.78%. The registered requirement of no case-level worsening fails, so no selection policy is adopted. The modest pooled benefit of adding context does not establish general conditional reliability or recognition of a musical relationship. No post-validation coefficient tuning or horizon shortening is performed.

The Python evaluation suite passes 338 tests. Only research Python and documentation change; production Rust remains the verified 9.3.40 version. There is no new hardware or author-listening acceptance. Evidence is retained in `target/phrase-expectation/2026-09-08-prediction-residual-reliability/`. Next distinguish overall residual prediction accuracy from evidence specifically along the current candidate difference, and test when changing the estimator selection is warranted. Cognitive retention/interference, persistent relationships, stream identity, action value, phrase/closure and musical acceptance remain open.

### 9.3.42 Directly Learning the Candidate Loss Difference

The residual regressions in 9.3.41 minimize squared error of a three-band residual vector, while estimator selection is scored on the scalar difference between candidate losses. To isolate this target difference, two controls retain the same 9/65 input features, fixed ridge penalty 1.0, centered cumulative statistics and unpenalized intercept, but learn the actual scalar loss difference directly. No new features, retention lifetime or tuning grid is added. For a constant candidate correction, loss difference is a fixed linear transformation of the residual plus a constant, so the two regressions agree; a test verifies this. Changing candidate corrections make the training objectives different even with the same inputs.

All four earlier controls remain unchanged. The two new models train on the same available-context targets completed by each request. Identical candidates have equal loss for every possible outcome, so an available new prediction returns that known zero difference. No target is added to training before completion. Primary comparisons remain nonzero-correction strata; no previous training or missing context remains unavailable. Scalar outputs are not scored as three-band residual predictions.

The preceding twelve cases are now development data. Six fresh validation cases were registered from samples 08/09/12 at unused seeds 192053/192067, retaining all seven horizons and whole-run/requests at or after eight seconds. The previous four controls and every previous record field remain exactly unchanged on the 38,547 earlier matches. Independent reconstruction checks scores and selections for 57,659 matches including 19,112 fresh ones. At 126 requests, one per case/horizon, all four regressions also agree with 504 independent batch fits. Maximum absolute differences are below 6.02e-18 for residual vectors and 7.28e-22 for scalar loss differences.

Fresh nonzero-correction strata contain 16,290 whole-run and 14,001 late requests, all with earlier training outcomes. Ratios below are whole/late; values below one indicate smaller error than the named comparator.

| Direct loss-difference model | Loss-difference prediction SSE / direct past mean | Loss-difference prediction SSE / corresponding residual regression | Selected forecast energy SSE / current mixture |
|---|---|---|---|
| Candidate geometry only | 1.0986 / 1.1072 | 0.8261 / 0.8995 | 0.99852 / 0.99282 |
| Candidate geometry and context | 1.0943 / 1.1025 | 0.8306 / 0.9098 | 0.99797 / 0.99228 |

Directly matching the training target reduces loss-difference prediction error relative to the vector regressions in these comparisons, but does not beat the direct past mean. Both new selectors also worsen late four-second loss for sample 12 at seed 192053 by about 7.10% against current mixture, over 26 nonzero-correction matches. Both fail the registered requirements to beat direct mean loss prediction and avoid case-level late four-second worsening. Neither is adopted. Pooled selected-forecast improvements on unchanged-policy audio do not establish the effects of an unexecuted participation policy or musical improvement.

The Python evaluation suite passes 341 tests and both language versions build. Production Rust and the renderer remain the verified 9.3.40 version, with no new hardware test, author listening acceptance or commit. Evidence is retained in `target/phrase-expectation/2026-09-08-direct-loss-comparison/`. This comparison clarifies the learning target for estimator choice; it does not implement cognitive relationship memory. Next inspect the existing continuous-observation and relationship-candidate research, comparing the information retained by three-band energy history with the information needed to distinguish persistent relations. Regression tuning cannot stand in for a missing memory representation. Cognitive retention/interference, stream identity, action value, phrase/closure and musical acceptance remain open.

### 9.3.43 Spectral Configurations and Order Lost at the Temporal Memory Input

Before further estimator tuning, this audit asks what the retained observation can distinguish. Its target is the 10-ms three-band energy input used by temporal participation; the frequency-side Landscape retains other observations. Earlier research already supplies fine auditory envelopes, component trajectories, coherence and acoustic-body candidates (9.3.8–9.3.31). They are reused here as evidence observers, not newly declared stream recognizers.

**Different waveforms, identical ideal observations.** Frequency responses are derived from the production 200/3000-Hz one-pole filters. Six disjoint groups of three frequencies complete integer cycles in 10 ms. Each group's basis satisfies linear constraints making both filter states at the period boundary, and the first/last waveform samples, zero. A nullspace of the matrix of three-band energies plus total waveform power gives two nonnegative power mixtures A/B with disjoint spectral supports and equal measurements. Filters are not reset between windows: either period inherits the common state. Ideal steady A, steady B, ABAB and AABB sequences therefore have identical energy observations.

Three construction seeds at 24/48 kHz, four conditions each, were registered and saved as 24 PCM16 signals of 4.8 seconds. Each 0.4-second construction segment contains forty analysis periods. Frequency responses, time-domain recurrences and zero-padded FFT convolution agree; maximum ideal energy discrepancy is below 1.74e-17. This is a constructive representation counterexample with aligned analysis and switching boundaries, not a claim of identical observations for arbitrary clock origins or ordinary signals.

PCM16 quantization and native f32 arithmetic leave small differences. Maximum relative difference between matched three-band vectors is 4.84e-5, about 0.00484%; native eight-age histories differ by at most 2.81e-5 relatively. Numerical closeness is not exact equality or human indiscriminability. On the same decoded PCM, existing 113-band envelopes retain normalized energy-shape distances of 0.950–0.975 for steady A/B and 0.451–0.463 for ABAB/AABB after one second. These are descriptive distances, not perceptual probabilities or recognition rates. Both reporting intervals are 10 ms, but the frontends also use different filter families; this is not an isolated channel-count intervention.

**Retained order after a common present.** A supplementary check was registered after this comparison and reuses its controls. The final 0.4 seconds of decoded ABAB/AABB PCM are identical. The existing continuous-history model is applied unchanged to both three-band and 113-band envelopes, with the same numerical ages 0.125–16 seconds and order 12. No cognitive lifetime is introduced. At the endpoint, current fine envelopes differ relatively by less than 5.72e-17, while their retained histories differ by 0.539–0.558. Current three-band values are identical; their history differences are only 1.47e-6–9.50e-6, including finite-precision effects. Preserving frequency positions in the earlier evidence can preserve ordering information after the present becomes the same. This does not identify a relationship or perceptual stream.

Independent discrete convolution integrates each gamma kernel over the piecewise-constant observation intervals. Endpoint history differences are below 4.31e-16 and coverage differences below 7.22e-15. At 4.8 seconds of observed input, the eight- and sixteen-second age cells have known coverage only about 0.0327 and 0.000100. A distant age cell is not evidence that the corresponding past was observed, nor a claim of confident long-term memory.

Three existing ordinary WAVs, samples 08/09/12 at seed 192053, are reused with the same PCM for both frontends. Their 9,153 windows bring the total to 20,673. Independent calculations check 62,019 native energy components, 496,152 history components and 165,384 coverage values, with matching fine-observation availability times. Ordinary maximum simultaneous observed components are 20/13/18, with no truncation. These are neither Voice counts nor perceived stream counts, and do not reproduce the author's flow judgments.

The Python evaluation suite passes 346 tests, the existing native history-export assay passes one test, and both technote languages build. Production Rust, participation policy and instrument audio are unchanged, with no new hardware or author-listening acceptance. Evidence and plots are retained in `target/phrase-expectation/2026-09-08-spectral-history-observability/`.

The upstream design revision is to **stop treating three-band energy alone as sufficient evidence for memory of spectral relationships**. Regression or longer retention cannot recover distinctions already mapped to the same ideal input. The requirement is not exactly 113 channels: it is preservation of task-relevant frequency positions, relations, order and missing-evidence status. Next design and verify how this evidence enters the existing production observation path, then connect pre-outcome relationship candidates to subsequent audio. Physical component tracking, source attribution, perceptual streams, phrase/closure and action value remain distinct unfinished requirements.

### 9.3.44 Native Frequency History Before Delivery Coalescing

Following the input limitation in 9.3.43, the native analysis worker now retains continuous history from the existing NSGT observations. RMS at each frequency uses the existing auditory-history kernel independently. Batching three components reuses the numerical operator without pooling frequencies. Coordinates follow the runtime Log2Space, with 690 bins in the configuration tested here. The existing 0.125–16 second age layout and order 12 remain numerical choices, not identified cognitive retention times or auditory resolution.

**Update before delivery and retain availability.** Every analysis hop advances history, including observations whose individual snapshots are coalesced by the generator's latest-only delivery. Ordered listener delivery yields the same history. Initial FFT filling and post-gap refilling contribute unknown input; pre-gap history survives with separately updated known coverage. A filled FFT container does not establish per-band latency or perceptual confidence. The update uses each hop's endpoint NSGT estimate as a numerical observation; it does not recover the within-hop waveform.

Snapshots carry the observed sample endpoint, sample rate, analysis window and frequency coordinates. The generator Landscape and ListenerTwin receive their respective shared-habitat and presentation histories. Approximately 100 ms report intervals keep generated and observed times distinct; only reporting is decimated. Gap notifications invalidate previously available snapshots. Shared habitat history is not local evidence with a Voice's self-produced sound removed. The local three-band participation path and its action policy remain unchanged.

**Independent acoustic checks.** The 24 controls and ordinary 08/09/12 recordings from 9.3.43 were reused for 16,682 native NSGT windows. Independent interval-integrated gamma convolution of the recorded RMS sequence matched 8,804,400 reported history components, with maximum absolute history and coverage differences below 3.52e−7 and 2.47e−5. This checks the numerical history of the same observations, not a new validation of NSGT's auditory interpretation.

The original 0.4-second common PCM tail still left approximately 0.0765–8.15% relative differences in current native RMS. This configuration uses a 16,384-sample FFT container, approximately 0.683/0.341 seconds at 24/48 kHz, plus smoothing. After inspecting that result, a supplement registered a further 2.24 seconds of the same final period, an engineering duration aligned with the analysis hops and construction periods. With a total 2.64-second common PCM tail, all six pairs had identical current NSGT values while their histories retained relative differences of 0.0467–0.0482. An additional 5,940 windows and 3,080,160 history components were independently checked. This demonstrates retained preceding spectral order under a common present, not source identity, perceptual stream recognition or musical value.

Ordinary 08/09/12 at seed 192053 produced bit-identical before/after WAVs and 74,950 identical existing non-wall-timing records. Each bus added 848 history reports. Presentation history also matched same-time PCM16 reanalysis within a maximum absolute difference below 8.76e−6; quantization of the live f32 presentation remains distinct from history integration error.

**Cost and remaining scope.** History updates add no allocations. Each delivered snapshot allocates its array and Arc, two allocations; latest-only delivery does not allocate intermediate history snapshots. Four matching 16-Voice conditions cover entrained/flow with reporting off/on. All voices remained throughout each four-second measurement window, with zero over-budget hops. After-change hop p99 was 2.31–3.83 ms against a 10.67 ms budget, approximately 0.49–0.89 ms above the corresponding before measurements. Reporting cost also increases. These device-free checks do not establish hardware delivery or performance at arbitrary population sizes.

All 744 Rust tests passed, with 12 still ignored; Clippy, the allocation-instrumented history test and two explicit native acoustic assay invocations passed. Evidence is saved under `target/phrase-expectation/2026-09-08-native-spectral-history/`. Next, retained relationship candidates must use this frequency evidence and have their issued expectations matched to later observed sound. Source attribution, cognitive retention/interference, persistent stream identity, phrase/closure, action value and author adoption remain incomplete.

### 9.3.45 Regional Predictions and the Contribution of Partner History

Research module `evaluate_spectral_relations.py` uses the native NSGT history from 9.3.44 to compare predictions of later regional sound. At most three separated energetic peaks and their neighborhoods are selected from approximately the first two seconds of observed power; their locations and an RMS reference are then frozen. These regions summarize observations, not identified Voices or perceptual sources.

The target is `log1p` of regional RMS relative to that reference. Four centered ridge regressions combine the target region's current/history values, an outside-pair frequency summary, and the partner's current/history values. Retained regional features summarize squared retained bin RMS; they are not the history obtained by first calculating regional RMS. The unpenalized intercept, fixed ridge penalty 1, order 12, eight ages and region selection are numerical choices. Cumulative sufficient statistics implement no forgetting, interference or context changes and are not adopted as a completed cognitive memory model.

At 48 kHz and 512-sample hops, the query lead is 48,128 samples, approximately 1.003 seconds. The target FFT container begins after issuance, but NSGT smoothing still retains earlier input. Issued predictions remain fixed; only the matching subsequently observed target updates training. Gaps censor queries, and EOF retains pending outcomes without synthesized silence. The mean of exactly the same completed targets and current persistence are additional baselines.

**Do not attribute current-value benefit to retained history.** Initially, six 36-second constructions at three seeds covered independent variation, approximately 1.003-second delay, common driving, relationship removal at 20 seconds, common gain variation, and static sound. Removal has an abrupt switch at 20 seconds. Ordinary 08/09/12 at seed 192053 were reused. After inspecting these 21 cases, a supplement registered two dimension/penalty-matched comparators replacing only partner history with copies of its current value, plus three approximately 2.005-second-delay controls. This compares models without uniquely separating information from regularization effects. All original four-model predictions and matching outcome records remain unchanged.

The table reports the predeclared 277-to-733 Hz direction, target times 24–36 seconds, and ranges across three seeds. Ratios below one favor the numerator. Full includes own, outside and partner histories; no partner includes own and outside; current partner replaces only the partner's history block with repeated current values.

| Control | Full / no partner | Full / current partner |
|---|---:|---:|
| Approximately 1.003-second delay | 0.583–0.740 | 1.038–1.134 |
| Approximately 2.005-second delay | 0.631–0.775 | 0.635–0.764 |
| Relationship removed at 20 seconds | 1.072–1.180 | 0.891–0.962 |
| Common driving | 0.877–0.938 | 1.030–1.049 |
| Independent variation | 0.997–1.101 | 0.997–1.033 |
| Common gain variation | 1.005–1.031 | 1.003–1.027 |

Partner history supplies additional predictive information for the longer delay, whereas current values perform better for the shorter delay. After removal, cumulative regressions retaining partner information perform worse than the no-partner comparison. Conditional benefit also remains under common driving: the outside summary is not the actual common cause. Neither benefit establishes causal linkage or perceptual stream identity. Static target variances are only approximately 2.3e−11–4.2e−11, so their error ratios do not support musical claims.

Summarizing all directed comparisons in ordinary recordings, late full/no-partner error ratios are 1.086, 0.978 and 0.995 for 08/09/12. Against persistence, however, errors increase to 2.325, 2.749 and 1.464 times. The last scored targets in 08/09 are at 28.416/26.816 seconds, leaving only 42/27 late queries; 12 has 113. Directions share observations and are not independent samples. These results do not support production policy adoption or stream identification.

Across 24 cases, independent reconstruction from native observations checked 7,209,180 issued feature components and 43,692 target components, with maximum absolute differences below 1.34e−15 and 1.78e−15. Another 10,476 independent batch ridge fits matched online predictions within 1.24e−12. Each seed retains 169 identical issued records before relationship removal. All 354 Python tests passed. All 107 Rust/Cargo files and both release binaries match the verified 9.3.44 production version; there is no new production policy, hardware check or author audition.

Evidence is saved under `target/phrase-expectation/2026-09-08-native-predictive-relations/`. Next, candidates for relationship continuation, change and recurrence must have their expectations updated from observed sound. Input-history age, retention of learned relationships and contextual change remain distinct; uniformly weakening all past relationships is not by itself identified cognitive forgetting. Cognitive retention/interference, persistent stream identity, phrase/closure, action value and author adoption remain incomplete.

### 9.3.46 Retaining Relational Contexts and Separating Numerical Decay Tails

Relationship disappearance need not imply erasing all prior learning. [Gershman, Blei and Niv (2010)](https://www.princeton.edu/~yael/Publications/GershmanEtAl2009.pdf) treat conditioning and renewal using latent causes. [Skerritt-Davis and Elhilali (2021)](https://engineering.jhu.edu/lcap/data/uploads/pdfs/jneurometh2021_skerritt.pdf) address continuous auditory expectations, while [Mirea, Shin et al. (2024)](https://nivlab.princeton.edu/wp-content/uploads/sites/938/2025/05/jocn-36-11-2442.pdf) compare temporal priors in an explicit visual classification task. These motivate distinguishing retained contexts; they do not identify the finite model or numerical parameters below.

Research module `relational_contexts.py` compares stationary cumulative learning, renewal into a fresh prior, and recurrent contexts retaining inactive parameter posteriors. Each recurrent assignment-path hypothesis keeps separate regression, variance and zero-rate statistics for three contexts. Unused labels are exchangeable and collapsed by first observed use. A context is not a source or Voice identity.

The recurrent candidate uses a symmetric continuous-time Markov chain with K=3 states and leaving rate lambda=0.2/s. With U the uniform matrix,

\[
T(\Delta t)=U+e^{-\lambda K\Delta t/(K-1)}(I-U).
\]

This permits marginalization over unobserved transitions across query leads and gaps under the specified finite model. The renewal comparison instead retains its current segment with probability exp(-lambda*dt), assigning the remainder to a fresh prior. At most 32 assignment paths are retained, reporting discarded posterior mass. State count, transition rate and path budget are not cognitive retention times. Inactive parameters have no elapsed-time forgetting, and cognitive interference remains unimplemented.

The target y is the same nonnegative regional log1p RMS as in 9.3.45. A beta-Bernoulli atom handles exact zero; positive log(y) uses normal-inverse-gamma regression and a Student density with the 1/y Jacobian. Its positive mean is not finite, so exponentiated locations are not expected amplitudes. Score issued log predictive densities. All candidates fix prior means at zero, intercept precision 0.01, slope precision 1, variance shape 2/scale 0.5, and positive/zero prior counts 0.5 each.

Intervening completed outcomes can change the current posterior after a forecast is issued. Scoring the frozen issued distribution is therefore separate from filtering the new outcome into the latest posterior. Only actual completed targets train the model; gaps and pending EOF outcomes do not supply silence. Adding issued features to the existing 24 inputs preserved all 14,853 original records.

**Continuation, disappearance and recurrence.** Twelve 60-second controls at three new seeds supplement the original 24 cases: persistent positive delay, independence only during 20–40 seconds, a reversed relationship only during 20–40 seconds, and reversal at 20 seconds without return. They were registered before model outcomes. The table gives recurrent-minus-stationary log-density gains in bits per target for the specified 277-to-733 Hz direction; positive favors recurrent contexts.

| Control and target interval | Range across three seeds |
|---|---:|
| Persistent relationship, 44–60 seconds | −0.277 to −0.264 |
| Relationship removed at 20 seconds, 24–36 seconds | −0.254 to +0.201 |
| Return after independence, 44–60 seconds | −0.243 to −0.086 |
| Return after reversal, 44–60 seconds | −0.080 to +0.113 |
| Reversal without return, 44–60 seconds | −0.049 to +0.109 |

Retention and reuse operate, but acoustic benefit is not uniform. The fixed transition rate assigns weight to unused contexts even under stable relationships. Four cases were repeated at 128 paths: focal window gain signs persisted, but values changed by up to approximately 0.053 bits. Maximum mass discarded in one update was 20.24% across the 32-path cases and 17.48% even in the four 128-path cases. This does not establish approximation convergence.

**Large ordinary gains do not establish relational understanding.** For 08/09/12 after 24 seconds, gains averaged over all directions were approximately 112.3/226.7/42.5 bits versus stationary learning. A per-second diagnostic registered after these results located the large differences primarily in extremely small positive decay-tail observations. Recurrent contexts were worse than fresh-prior renewal by approximately 1.066/0.969/0.118 bits. There were 42/27 targets in 08/09, and 115 in 12 through 36.203 seconds. Directions and overlapping analysis windows are not independent evidence.

Hash-verified PCM16 inputs confirm that NSGT temporal smoothing retains a tail after the entire input FFT container becomes zero. At the end of 08/09, selected regions retain the smallest positive f32 subnormal power, approximately 1.40e−45. The current zero-input smoothing update is s <- alpha*s; rounding can arrest decay at this precision. Selected regions in 12 eventually become zero. Adapting a density to such tiny values is not perceptual stream or phrase recognition. The diagnostic adds no threshold, excludes no tail, and has not changed production numerical processing.

The 36 cases score 13,702 targets and retain 361 pending EOF queries. Short-series tests match exhaustive two- and three-context assignment sums with independent batch conjugate evidence. On acoustic outputs, independent calculations check 25,432,698 mixture-density components and 5,844 batch regressions, with maximum differences below 1.14e−13 and 2.74e−11. The 128-path comparison is independently checked as well. Common acoustic prefixes preserve 2,592 identical issued forecasts. All 364 Python tests and all 364 tests under the CI discovery pattern pass. All 107 production Rust/Cargo files and both release binaries remain the verified 9.3.44 version.

Evidence is saved under `target/phrase-expectation/2026-09-08-recurring-relational-contexts/`. Numerical decay stagnation and observation resolution require attention before further interpreting context gains. Then revisit context persistence/switching assumptions and inference approximation. This candidate is not adopted into production action policy. Cognitive retention/interference, persistent stream identity, phrase/closure, action value and author adoption remain incomplete.

### 9.3.47 Numerical decay to zero and the limits of tail likelihood

The numerical fixed point found in 9.3.46 is now removed from `RtNsgtKernelLog2`.
After the existing per-band smoothing operation, a subnormal `f32` power is set
to zero. The boundary is the floating-point representation's smallest positive
normal value, approximately \(1.17549\times10^{-38}\); it is not an auditory
threshold, learned parameter, or phrase boundary. Subnormal power precision is
explicitly relinquished. The arithmetic for normal results is unchanged, and
no global processor floating-point mode, allocation, clock, or analysis-window
change is introduced. This numerical convention does not identify the resolution
of human hearing or of the predictive observation model.

Two regressions failed before the fix. After it, both coherent and incoherent
tone-to-silence analysis preserve the preceding normal-range decay bit-for-bit
and eventually reach zero; a subsequent tone is still observed, and reset still
clears the state. Directly initialized subnormal states also cease to be positive
fixed points.

The exact three PCM16 inputs from 9.3.46 were replayed through the native analyzer.
Across 8,582 hops, 5,104,218 normal power components with no preceding affected
state are unchanged; no normal power component differs anywhere in these inputs.
All 150,327 changed power components are subnormal, with maximum absolute change
below \(1.175\times10^{-38}\). Observed time, frequency coordinates, history
coverage and delivery positions remain equal. The largest change in the retained
RMS history is \(1.388\times10^{-17}\); history is not erased when current power
reaches zero. The last nonzero PCM samples end at 25.649, 24.406 and 31.026 seconds
for 08, 09 and 12. All NSGT bands reach zero by 27.264, 26.048 and 32.608 seconds,
respectively. Before the fix, all three inputs had at least one residual band at
the minimum positive subnormal at EOF. Section 9.3.46 inspected selected regions,
which were already zero in 12; the whole-frequency comparison here is broader.

The same region-selection procedure and 32-path relational-context models were
rerun without retuning, tail exclusions or invented EOF observations. Region and
normalization contracts, the 772 completed queries and 31 pending queries are
unchanged. For targets at or after 24 seconds, the recurrent-over-stationary gains
averaged across directions fall from approximately
112.3/226.7/42.5 to 84.4/179.1/32.3 bits for 08/09/12, but remain large. Recurrent
contexts still lose to fresh-prior renewal by approximately 1.112/1.311/0.106 bits.
Removing the numerical fixed point therefore does not make these scores evidence
of auditory grouping, memory or musical flow. Independent reconstruction checked
1,407,345 density components and 336 batch fits, with maximum differences below
\(1.14\times10^{-13}\) and \(9.08\times10^{-12}\).

The full Rust suite passes 746 tests, with 12 ignored; Clippy and release builds
also pass. The frozen 08/09/12 scenarios, configuration and seed 192053 were
rendered with both release versions. All three WAVs are identical, as are 74,950
records outside spectral history and execution timing. Of 1,696 spectral-history
records, 90 change only in the RMS history values, by at most approximately
\(3.001\times10^{-18}\) in the serialized report. Onsets, Listener state, DCC
pressure, local predictions and participation records are unchanged in these
comparisons. This is a scoped offline regression, not new author listening or
hardware acceptance. The instrument's air-gap remains intact.

The checkpoint is `target/phrase-expectation/2026-09-08-nsgt-decay-precision/`.
The remaining observation-resolution and context-persistence questions must be
addressed before adopting the research model. Cognitive retention/interference,
persistent stream identity, phrase/closure, action value and author acceptance
remain incomplete.

### 9.3.48 Scoring forecast distributions without treating numerical precision as auditory fidelity

Removing the subnormal fixed point does not specify what the listener can
distinguish. [Skerritt-Davis and Elhilali (2021), section 2.4.3](https://engineering.jhu.edu/lcap/data/uploads/pdfs/jneurometh2021_skerritt.pdf)
places observation noise in the modeled input dimension and lets it affect both
predictive uncertainty and the precision of learned statistics. Its perceptual
parameters are compared with listener responses. Changing a final score or a
floating-point cutoff alone does not implement that mechanism.

The positive log-Student forecasts in 9.3.46 have no finite mean in the response
coordinate \(y\), the nonnegative regional `log1p` RMS. Their ordinary CRPS in
that coordinate also has a divergent tail. To audit the frozen forecasts,
`evaluate_context_scores.py` uses the bounded coordinate
\(u_s(y)=y/(y+s)\) and the loss

\[
L_s(F,y)=\int_0^1\left[F_s(u)-\mathbf{1}\{u_s(y)\leq u\}\right]^2\,du.
\]

This is CRPS in the bounded coordinate, equivalently a weighted integral of
binary Brier scores with weight \(s/(y+s)^2\) in the original coordinate.
The general scoring construction follows [Gneiting and Raftery (2007), equations
20 and 49](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
The exact integral is proper and lies in [0,1], with smaller values better. The
scale \(s=1\) is the primary evaluation choice; \(s=0.25,4\) are registered
sensitivity comparisons. These are not auditory thresholds or fitted cognitive
parameters. All scales and original log-density scores remain visible. A point
forecast that simply preserves the regional value at issuance is also scored;
its loss is absolute distance in the same bounded coordinate.

The zero atom and all mixture components are retained. Only forecasts saved at
issuance are scored when their matching target is observed; neither later
posterior weights nor EOF silence are substituted. No context model, feature,
region, normalization, transition rate or action policy is retuned. The main
comparison combines 33 constructed forecast streams from 9.3.46 and the three
corrected ordinary replays from 9.3.47; the three ordinary streams before the
numerical fix remain a separate comparison.

Uniform quadrature on the two sides of the observation proved inadequate for
narrow forecasts, even after doubling the node count. Those initial results and
their implementation are preserved as diagnostics. The revised calculation
uses a 16-node component quantile approximation only to place mixture-quantile
partition boundaries, then integrates the actual Student-mixture CDF or
survival function on each interval. The observation is also a boundary. Eight
Gauss-Legendre nodes per interval are compared with 16 on six registered cases;
adaptive quadrature with component-aware breakpoints supplies an independent
numerical check. Neither the partition nor the quadrature order changes the
predictive distribution or supplies a cognitive time or resolution parameter.

The \(1/y\) Jacobian in the positive density cancels when two models are
compared on the same observation. The large gain is therefore not an extra
Jacobian reward: the models assign different likelihoods to \(\log y\).
Those likelihoods also update the context weights. A bounded evaluation loss
does not remove this dependence from learning; an observation-noise mechanism
must address the represented evidence and its update, not just the displayed
score. Proper prediction scoring does not establish musical or cognitive value.

The final stream audit matched all 1,823,724 score components to the frozen
forecasts and outcomes across 39 streams: 14,474 completed queries and 392
pending at EOF. The main 36 contain 13,702 completed queries and 361 pending;
the remaining three are the before-correction comparison. Independent adaptive
integration checked 478 selected forecasts, not every score. A logit-coordinate
reference removed endpoint warnings from the earlier bounded-coordinate
reference: its largest reported integration error was \(1.17\times10^{-10}\),
with an omitted-tail bound of \(8.50\times10^{-18}\). The maximum primary-score
difference was \(7.28\times10^{-5}\). Doubling the partitioned quadrature order
on six cases changed an individual score by at most \(3.63\times10^{-4}\), and
a late-window mean by at most \(4.08\times10^{-5}\). These checks bound the
observed numerical discrepancies, not the error on all possible forecasts.

At the primary scale, the constructed stable relationship favors stationary
learning: stationary-minus-recurrent loss on the selected direction ranges
from -0.00909 to -0.00846 across three seeds in [44,60). The returned relationship
also favors stationary learning (-0.00806 to -0.00178); reversed-return cases
are mixed (-0.00294 to +0.00529). A positive gain favors recurrence. Some small
gains change sign under the scale sensitivity comparison. None of these
bounded losses is measured in bits or establishes a cognitive memory benefit.

For ordinary sources, the late-window all-direction means at \(s=1\) are:

| Sample | Stationary | Fresh-prior renewal | Recurrent | Current-value persistence |
|---|---:|---:|---:|---:|
| 08 | 0.189375 | 0.116903 | 0.135059 | 0.144183 |
| 09 | 0.265967 | 0.149178 | 0.176394 | 0.184225 |
| 12 | 0.118312 | 0.066981 | 0.074546 | 0.074750 |

There are respectively 42, 27 and 115 completed target queries in [24,40),
with coverage ending at different times. Directions share the same input and
are not independent samples. Renewal outperforms recurrence here; 12's near-tie
against current-value persistence reverses at \(s=0.25\). Better cessation
forecasting does not establish phrase boundaries or a changing musical flow.

A separately registered learning diagnostic holds the prior, covariates and
timing fixed after 20 observations of \(y=1\). Updating with zero and with
\(10^{-48}\) produces different subsequent distributions: under stationary
learning their CDFs at \(y=1\) are 0.534091 and 0.602428. The zero atom updates
without changing the positive regression, whereas every positive observation
updates that regression using \(\log y\). Five independent stationary batch
formulas reproduce this distinction. It is a discontinuity of the noiseless
hurdle observation model, not an identified hearing threshold or an f32 defect.
An observation-resolution mechanism must therefore change the likelihood and
statistical update, as well as the forecast. Neither a post-hoc scoring change
nor an unexplained small-value floor meets that contract.

The seven scoring tests and all 371 Python evaluation tests pass. Rust/Cargo
sources and release binaries remain at the verified 9.3.47 baseline. No new
author audition or hardware acceptance is claimed. Records are saved under
`target/phrase-expectation/2026-09-08-bounded-expectation-scores/`. The context
candidate remains outside production; observation uncertainty, cognitive
retention/interference, stream identity, phrase/closure and action value remain open.

### 9.3.49 Observation uncertainty in a nonnegative regression likelihood

The next research component changes the observation law and its Bayesian
update together. `MagnitudeRegression` in `scripts/uncertain_regression.py`
models a nonnegative observation by

\[
y=\left|x^\top\beta+\eta+\epsilon\right|,\qquad
\eta\sim\mathcal N(0,r^2),\quad\epsilon\sim\mathcal N(0,n^2).
\]

Here \(x\) contains the declared covariates, including an intercept when needed;
\(r\) is residual variation and \(n\) is explicit observation uncertainty.
Both standard deviations are fixed inputs in this component, in the same
units as \(y\). The folding operation is an engineering hypothesis for
nonnegative regional `log1p` RMS, not a fitted hearing mechanism. Noise enters
before taking the magnitude. D-REX's observation-noise argument in 9.3.48
motivates the prediction/update requirement, not this particular likelihood.
This law has a continuous boundary density at zero and no zero atom. Digital
quantization is not separately modeled; its log densities at zero must not be
compared as like-for-like gains against the old atom-plus-density reference.

Given a Gaussian coefficient component \(\beta\sim\mathcal N(m,V)\), the
observation density is the sum of Gaussian densities at \(y\) and \(-y\),
with location \(x^\top m\) and variance
\(x^\top Vx+r^2+n^2\). Each possible sign produces a Gaussian posterior.
Writing its precision as \(\Lambda=V^{-1}\) and information vector as
\(h=\Lambda m\), the update for sign \(s\in\{-1,+1\}\) is

\[
\Lambda'=\Lambda+\frac{xx^\top}{r^2+n^2},\qquad
h'=h+\frac{xsy}{r^2+n^2}.
\]

Sign probabilities use the same likelihood as the forecast. The component
retains the resulting mixture instead of replacing it by a fitted single
Gaussian. The precision is common to all sign histories with the same
observations; identical information vectors are merged by adding their
probability masses. At zero the two identical branches merge while preserving
the factor of two in the boundary density. A declared path budget truncates
the remaining mixture and reports discarded mass. This is a numerical limit,
not a forgetting rule. Issued forecasts own their arrays and do not change
when subsequent observations update the regressor.

Uncertainty within each Gaussian component is reported separately from
residual and observation variance. Differences between component locations
retain further parameter uncertainty; the within-component variance is not
the variance of the entire mixture. Increasing \(n\) both broadens the
observation distribution and reduces the precision increment. However,
\(r^2\) and \(n^2\) enter only through their sum. Swapping residual SD 0.3
and observation SD 0.4 gives identical likelihoods and learned states. Naming
the two terms does not identify perceptual noise from these data.

Eight tests cover exhaustive batch sign-history marginalization, normalized
density and independently integrated CDF, the zero limit, noise-dependent
learning precision, units, immutable issuance, the variance-identification
limit, and numerical pruning. In 42 registered diagnostic conditions, after
20 observations of \(y=1\), updates with zero and with
\(10^{-6},10^{-12},10^{-24},10^{-48}\) gave next-prediction CDF differences
of at most \(1.57\times10^{-11}\) at the five declared positions. The
observation SDs were 0.01, 0.1 and 1, residual SD was 0.1, and prior coefficient
variance was 100. These choices were not fitted to listening judgments.
The corresponding within-component coefficient variances after a zero update
were approximately 0.000481, 0.000952 and 0.048072. An absent observation
preserved the preceding state; a zero observation changed it. At the smallest
noise, zero and a new level of 2 differed by 0.3566 in the largest inspected
next-prediction CDF difference, so the continuity result was not obtained by
disabling learning. Budgets 128 and 512 gave identical inspected CDFs; at most
118 components survived and the reported lost mass was below
\(1.89\times10^{-15}\), consistent with rounding rather than a budget cut.
This diagnostic does not establish adequacy of either budget for general inputs.

All 379 Python evaluation tests passed, followed by the eight focused tests
after clarifying the variance field names. The naming change leaves the 42
diagnostic results unchanged. This component has no context transitions or
production connection yet. It establishes a testable observation/update
mechanism, not a completed relational memory, auditory parameter fit, stream
identity, phrase boundary or musical improvement. The next comparison must
retain the issued-observation protocol while integrating context persistence,
renewal and recurrence, checking the variance assumptions and mixture budget
on the ordinary and constructed inputs. Records are under
`target/phrase-expectation/2026-09-08-uncertain-magnitude-regression/`.

### 9.3.50 Context inference with observation noise, and the limits of its prior

`MagnitudeContexts` connects the observation/update law in 9.3.49 to
stationary learning, fresh-prior renewal and a finite set of recurring
contexts. Each context retains its own Gaussian coefficient state. Uncertainty
about context assignments is distinct from the latent signs introduced by the
folded observation law. The zero-mean prior permits a whole coefficient vector
to change sign without changing any future magnitude distribution. The filter
uses this symmetry and merges identical complete states before truncating
paths; it does not identify signs with relationship contexts. Exhaustive short
sign/context histories verify the marginal evidence, and the stationary
prediction agrees with the unquotiented regressor from 9.3.49. Unused-context
label marginalization and the continuous-time transition rule remain as in
9.3.46. Missing time advances uncertainty without assimilating silence.

The comparison keeps three contexts, change rate 0.2/s, intercept prior
variance 100 and slope variances 1. Residual SD is 0.1; observation SD 0.1 is
primary, with 0.01 and 1 as registered sensitivities. The main path budget is
128. Six frozen feature/outcome streams are used: persistent, returned and
reversed-return controls at seed 4721, and the corrected ordinary 08/09/12
streams at seed 192053. Three noise levels give 18 runs; 512-path comparisons
on the persistent control and 12 give 20 in total. No region, cadence, target
or scenario is retuned. Constructed results below concern one seed and the
previously selected direction from 277 to 733 Hz, not general recurrence or
source-recognition performance.

The bounded scores retain the scales and mixture-quantile partitions of
9.3.48. Two outer component boundaries at eight standard deviations improve
integration of narrow Gaussian mixtures; the full integral, including tails,
remains. The audit matched 7,975 completed queries and 203 pending queries
across the repeated runs, including all 1,004,850 score components. The six
input streams and directed pairs are reused, not independent observations.
Recomputing 180,469,362 positive/negative Gaussian density terms gave mixture
log-density differences below \(1.82\times10^{-12}\). Independent adaptive
integration checked 228 selected recurrent forecasts at scale 1: the largest
primary score error was \(2.47\times10^{-5}\), and at doubled quadrature order
\(6.74\times10^{-6}\). One bounded-coordinate reference warned near an
integration boundary; a separate logit-coordinate reference removed the
warning and differed by \(1.05\times10^{-16}\). These are numerical checks on
selected forecasts, not a global error bound or cognitive calibration.

At observation SD 0.1 and scale 1, the constructed [44,60) focal losses are:

| Relationship | Stationary | Renewal | Recurrent | Current-value persistence |
|---|---:|---:|---:|---:|
| Persistent | 0.037038 | 0.047679 | 0.043498 | 0.073497 |
| Returned | 0.035852 | 0.047491 | 0.046363 | 0.073497 |
| Reversed, then returned | 0.036330 | 0.048342 | 0.042550 | 0.073497 |

Stationary learning also wins these focal comparisons at both other noise
levels. This does not justify adopting recurring contexts. Ordinary [24,40)
all-direction mean losses at the primary settings are:

| Sample | Stationary | Renewal | Recurrent | Current-value persistence |
|---|---:|---:|---:|---:|
| 08 | 0.316840 | 0.270861 | 0.285121 | 0.144183 |
| 09 | 0.435577 | 0.361260 | 0.346200 | 0.184225 |
| 12 | 0.289511 | 0.234842 | 0.254367 | 0.074750 |

Coverage is respectively 42, 27 and 115 completed target queries. All three
context modes lose to current-value persistence in these ordinary late-window
comparisons at every registered noise level. Correcting the zero-update
discontinuity does not by itself yield useful acoustic predictions, let alone
musical flow or phrase recognition.

Numerical context inference has not converged either. At the primary noise
and 128 paths, an update of the partner-inclusive recurring model discards as
much as 0.507 of its posterior mass. At 512 paths, the persistent control and
12 still reach 0.254 and 0.189. Their all-direction recurrent losses change
from 0.049755 to 0.052129 in the persistent [44,60) window, and from 0.254367
to 0.244766 in 12's [24,40) window. Neither result establishes adequate path
capacity. A separate direct-precision check on 48 issued stationary forecasts
from actual inputs found location differences below \(8.01\times10^{-10}\)
and within-component variance differences below \(3.68\times10^{-13}\), even
with a precision condition number near \(6.46\times10^6\). That rounding
error does not explain the much larger prediction losses.

The prior also needs an observational interpretation. At the first issued
ordinary queries, current regional observations average 1.092, 1.214 and
3.315, while the folded-Gaussian prior predictive means average 8.050, 8.052
and 8.125. Reusing the numerical prior precision from the earlier logarithmic
model did not preserve its meaning in the observation coordinate. These
initial discrepancies do not isolate the cause of late-window errors, but
they rule out treating the inherited settings as a cognitively derived prior.
In the current model a new context resets the entire conditional predictor,
including coefficients for self-continuation. The next design should separate
continuation supported by the current acoustic state from a learned relational
correction, and compare priors in the observable coordinate. Neither a new
relationship nor uncertainty about it should automatically imply ignorance of
the currently observed sound. The appropriate continuation law and uncertainty
still require comparison; this is a design requirement, not an accepted repair.

The Python evaluation suite passes 388 tests. The default legacy stream route
also preserved all 1,082 records against the saved pre-change driver on an
actual frozen input.
Rust/Cargo and release binaries remain at 9.3.47; no new author audition or
hardware acceptance is claimed. Records are under
`target/phrase-expectation/2026-09-08-uncertain-relational-contexts/`. This
observation/context candidate remains outside production. Prior grounding,
reliable continuation and inference approximation must be addressed before
adoption; cognitive retention/interference, stream identity, phrase/closure
and situated action value remain open.

### 9.3.51 Continuation conditional on the current sound, with contextual corrections

The candidate in §9.3.50 resets target-continuation coefficients along with relational
coefficients when renewing a context. The research candidate `ContinuationContexts`
conditions its forecast on the target region observed at issuance:

\[
y=\left|o+u^\top\beta+v^\top\delta_c+\eta+\epsilon\right|.
\]

Here \(o\) is current regional `log1p` RMS, \(u\) contains target-region current and
history features, \(v\) describes the environment, and \(\delta_c\) is a contextual
correction. This is a persistence reference, not a requirement that every sound
persist. Observation error in the input features is not inferred as a latent state.
Each feature block has its own intercept and unit Euclidean norm. Independent
zero-mean coefficient priors with covariance \(\sigma_{\mathrm{correction}}^2 I/2\)
make the total initial correction SD explicit in observation units. The reference,
features and prior change together from §9.3.50; this is not an isolated comparison
of the reference alone.

`reset_all` places both coefficient blocks inside each context;
`shared_continuation` shares the continuation block. The latter retains the full
joint Gaussian covariance of shared and contextual coefficients. Renewal replaces
only the relational block by an independent prior, preserving the marginal of
every retained variable. An inactive context may update indirectly through its
correlation with the shared block; that is conditional learning, not forgetting.
A known nonzero offset destroys the sign symmetry used in §9.3.50. Both latent signs
are retained, and only identical complete states are merged.

Exhaustive short sign and two-/three-context histories agree with marginal
likelihoods computed independently from observation covariance matrices. Both
layouts and all three transition modes were checked at correction SDs 0.03, 0.1
and 0.3. Checks also cover renewal marginal preservation, stationary layout
agreement, continuity near zero, unit changes, immutable issuance and missing
input. Extracting shared routing from the two existing models preserves 24,192
compared fields and 491 actual-input records for each old route. All 395 Python
evaluation tests passed.

The registered comparison applied two layouts to six frozen streams,
plus 512-path shared-continuation comparisons on the persistent control and
ordinary Sample 12, for 14 runs. Primary correction, residual and observation
SDs are each 0.1, with three contexts, transition rate 0.2/s and 128 paths. Controls
include current-value point persistence and an unlearned continuation distribution
with the same initial variance. These settings are not identified cognitive priors
or retention times.

In the 12 primary comparisons, sharing continuation reduces renewal/reuse loss in the preselected
direction of the constructed controls, but does not outperform cumulative learning.
Scale-1 all-direction means over 24–40 seconds of ordinary recordings are:

| Sample | Cumulative | Reuse, reset all | Reuse, shared continuation | Unlearned continuation | Current-value point |
|---|---:|---:|---:|---:|---:|
| 08 | 0.181365 | 0.198661 | 0.191950 | 0.156489 | 0.144183 |
| 09 | 0.214407 | 0.243626 | 0.229005 | 0.199833 | 0.184225 |
| 12 | 0.143462 | 0.126814 | 0.134074 | 0.097272 | 0.074750 |

Cumulative loss is the same in both layouts. Sharing is not uniformly beneficial:
reuse loss increases in Sample 12. Renewal also loses to both continuation controls
in all three ordinary cases under both layouts. Mathematical consistency in
separating retained states does not establish predictive gain on this observable
or cognitive validity.

A post-comparison diagnosis separates numerical-zero and positive
observations without excluding targets or refitting. Of 252, 162 and 690 late
direction targets in ordinary 08, 09 and 12, respectively, 98, 70 and 262 are zero.
Zero refers to a selected frequency region, not necessarily silence in the whole
performance. These counts reuse streams and regions rather than independent auditory observations.
Loss is large on zero outcomes, while errors also remain on positive outcomes.

The fixed noise assumption places a structural constraint on part of this error.
Every component has variance at least the residual-plus-observation variance 0.02.
At any nonnegative evaluation point, folded-Gaussian survival is no smaller than
that of the centered component with variance 0.02. Consequently a normalized
mixture also satisfies the following scale-1 score bound at the zero target:

\[
L(0)\geq\int_0^\infty\frac{[2\Phi(-y/\sqrt{0.02})]^2}{(1+y)^2}\,dy
\simeq0.05972765.
\]

An independent integral over the positive half-line confirms the value. The
unlearned continuation distribution has variance 0.03 and scores approximately
0.07164874 when both offset and outcome are zero. This is a consequence of the
registered noise hypothesis, not an auditory silence threshold or a reason to
eliminate uncertainty generally. It does not account for all errors. Expressing
priors in observation units alone does not calibrate observation or future-prediction
variability.

Across 14 runs, 5,598 completed and 142 pending targets and 806,112 score components
were checked. Rebuilding 144,895,816 signed Gaussian density terms gives maximum
mixture log-density difference below \(1.37\times10^{-12}\). Independent adaptive
integration of 160 selected forecasts differs from the primary scores by at most
\(3.43\times10^{-6}\); doubling the nodes reduces the maximum difference to
\(1.73\times10^{-8}\). One reference-integral warning was checked in separate
logit coordinates without warnings, differing from the original reference by
\(6.94\times10^{-18}\). These are not independent recording counts or auditory
calibration results.

Across layouts, 43,884 issued stationary/baseline forecasts and 28,524 stationary
updates are exactly equal. Context inference is not numerically converged.
Increasing 128 to 512 paths changes shared-reuse loss from 0.067491 to 0.067513
in the persistent control's selected direction over 44–60 seconds, and from
0.134074 to 0.136076 in ordinary Sample 12's late all-direction mean. Comparing
issued CDFs at 63 specified bounded-coordinate points gives maximum differences
of approximately 0.316 and 0.320, respectively. These 326,592 evaluated points
do not bound differences over the entire continuous coordinate. Maximum discarded
posterior mass remains approximately 0.205 and 0.132 at 512 paths; small mean-loss
differences alone do not establish that the path budget is sufficient.

This candidate is not adopted in production. Next, distinguish the assumption
of a fixed variance shared by all observations from predictive residual variation
that actual outcomes can calibrate. Do not arbitrarily split observation error
from residual prediction error, or conflate either with elapsed-time/interference
effects on relational memory. Selecting favorable scores by changing numerical
settings does not identify cognitive retention or expectation. Cognitive
retention/interference, stream identity, phrase/closure and situated action-value
learning remain open.
All 107 Rust/Cargo files and both release binaries remain at §9.3.47; no new author
audition or hardware acceptance was added. Evidence is stored under
`target/phrase-expectation/2026-09-08-continuation-conditioned-contexts/`.

### 9.3.52 Learning predictive residual variation while retaining the stationary accumulation limit

The fixed variance floor in §9.3.51 does not identify observation uncertainty.
The research-only `ResidualRegression` treats input features as given and learns
one unknown variance for unexplained target variation. It does not split that
variance into observation and process terms that this likelihood cannot identify.

\[
y=\left|o+x^\top\beta+\sqrt{v}z\right|,\qquad
z\sim N(0,1),\quad\beta\mid v\sim N(m,vV),\quad v\sim IG(a,b).
\]

Integrating the variance gives a folded Student predictive distribution.
The Gaussian/unknown-variance conjugate calculation follows
[Murphy's summary](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf);
the offset regression with latent signs is checked against independent batch
integrals over all short sign histories. Each path retains a coefficient mean,
inverse-Gamma scale and weight. Exact duplicate states merge before path truncation.
Student scale squared, signed-component variance and the folded observation law
remain distinct. Issued predictions are immutable; numerical zero is observed
evidence, whereas missing input supplies no update.

The primary prior uses \(a=3\), \(E[v]=0.02\), and \(V=0.25 I\) for coefficient
models. The two unit-norm feature blocks from §9.3.51 give initial signed predictive
variance 0.03. A coefficient-free continuation control starts at \(E[v]=0.03\),
learns only variance, and has an unlearned copy. Their initial predictive laws match
those of the coefficient models. Matching the previous Gaussian's second moment
does not match its prior law or coefficient/variance dependence. The shape 3 makes
the variance of the variance finite; it is not an identified cognitive sample count.

Repeated zero observations reduce variance without a fixed positive floor, and a
later error increases it. Accumulation nevertheless slows adaptation after a
variance change. Checks cover unit scaling, batch/sequential agreement, positive
definite covariance, delayed outcomes, missing input, EOF, heavy tails and narrow
components. All 403 Python tests pass. Both previous scoring routes also match their
saved implementations exactly on real-input prefixes with four completed targets
and six output records each.

Independent held-out synthetic draws expose prior influence. With true noise SD 0.01,
the coefficient-free posterior mean variance is about 0.000569 after 256 observations
and 0.000131 after 4096. On 10000 held-out draws, the PIT KS distance falls from about 0.396
to 0.0645. For SD 0.1, the 4096-observation variance is about 0.00996 and KS distance 0.00735.
A two-coefficient control recovers weighted means about[0.0763,-0.0492] from truth
[0.08,-0.05] after 512 observations; central 90% coverage on 2000 held-out draws is 0.928.
These are diagnostics of specified generative conditions, not auditory calibration.
Deterministic zero sequences have PIT zero under this continuous law and cannot be
interpreted as uniform-PIT calibration samples.

Eight runs complete: six frozen inputs at 128 paths and 512-path comparisons for the persistent control and ordinary 12.
Mean bounded loss for ordinary audio at 24–40 seconds, scale 1, over all directions is below.
The coefficient-model columns include partner information.

| Sample | Previous stationary Gaussian | Learned coefficients and variance | Learned variance only | Unlearned continuation | Current-value point |
|---|---:|---:|---:|---:|---:|
| 08 | 0.181365 | 0.188441 | 0.179091 | 0.154067 | 0.144183 |
| 09 | 0.214407 | 0.227478 | 0.223778 | 0.197024 | 0.184225 |
| 12 | 0.143462 | 0.182183 | 0.179684 | 0.094689 | 0.074750 |

The coefficient model without partner information also fails to outperform the continuation controls in all three ordinary late windows.
For constructed persistent, returned and reversed-return controls, the designated 277→733 Hz direction at 44–60 seconds
has losses about 0.0459, 0.0458 and 0.0458 with coefficient/variance learning, versus about 0.0520 for variance-only
learning and 0.0735 for point persistence. This does not identify a recurrence-specific memory benefit or a causal source relation.

For numerical-zero targets in ordinary audio at 24–40 seconds, the variance-only
control's mean retained residual variance is about 0.176,0.172 and 1.834 for 08,09 and 12.
Removing a fixed floor does not remove the influence of earlier large variation.
Positive observations also retain errors, so this diagnosis is not a complete causal
explanation. A selected region's zero is not whole-performance silence; directions
that reuse the same sound are not independent auditory observations.

The audit matches 3,221 completed targets, 81 pending targets and 289,890 score components
to actual observations, independently rebuilding 25,509,372 signed Student density terms.
The maximum mixture log-density difference is 4.26e-14. Across 180 selected independent integrals,
the maximum primary-score difference is 1.35e-05, falling to 4.89e-07 at order 32.
Reference warnings are checked again in logit coordinates and agree with the original references within 1e−10;
primary scores remain unchanged.

At 128 paths, ordinary 08 and 09 discard about 0.5 posterior mass in one update.
The persistent control discards about 0.0406, and about 0.0324 remains at 512 paths.
Comparing issued CDFs at 63 specified scale-1 coordinates gives maximum differences about 0.0030 for the persistent
control and 1.37e−7 for ordinary 12. All 5,184 unlearned forecasts match exactly.
This is a grid comparison, not a continuum error guarantee. Ordinary 08/09 budget sensitivity is untested;
convergence of the entire comparison is not established.

This candidate is not adopted in production. Learning variance does not derive a
changing environmental state or cognitive relational retention/interference.
Updates depend on completed observation counts; elapsed time and intervening
information do not implement memory change here. Next, distinguish the state that
predicts changing sound from the memory that retains and retrieves relationships,
and review the existing context candidates against that boundary. Tuning priors or
forgetting constants to these scores would not establish a DCC basis.
Stream identity, phrase/closure, action value and author acceptance remain unresolved.
The 107 Rust/Cargo files and both release binaries remain at §9.3.47. No new audio,
author audition or hardware acceptance is added. Evidence is stored under
`target/phrase-expectation/2026-09-09-learned-residual-uncertainty/`.

### 9.3.53 Evolving forecast deviation and retained coefficients

The research-only `EvolvingResidualRegression` separates retained coefficients from an evolving
forecast deviation. With the frozen issuance offset \(o\) and features \(x\), it uses

\[
y(t)=|o+x^\top\beta+r(t)+\sqrt v\,e|,\qquad dr=\sqrt{qv}\,dW.
\]

It retains the joint coefficient/state covariance and the common unknown variance \(v\), yielding
the folded Student prediction of the preceding section. Evolution without observation increases
state uncertainty while preserving the coefficient marginal and the posterior over \(v\).
An actual observation conditions both. The deviation starts deterministically at zero at the first
issuance time. This is a statistical separation, not cognitive forgetting.
The linear SDE transition follows [Särkkä and Solin (2019), §6.2](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
The folded-observation and unknown-variance extension is checked independently by batch integration
over every short sign history at irregular observation times.

The primary \(q=1\,\mathrm{s}^{-1}\) and sensitivity values 0.1 and 10 are declared numerical hypotheses
about temporal covariance, not identified retention times or human change-detection rates. A fixed
process/residual variance ratio does not identify auditory measurement noise separately from
environmental variation. The FFT container duration does not specify the independent-sample interval
of each smoothed NSGT band. Forecasts report parameter, state, cross-covariance and residual contributions separately.

The default route and \(q=0\), excluding explicit new state metadata, exactly preserve 4,827 records,
58,512 forecasts and 42,786 updates from all six preceding inputs. Issued forecast immutability, units,
missing input, delayed outcomes and EOF are checked; all 407 Python tests pass. On three seeds from
the declared diffusion process, 128 prequential predictions after 256 observations give mean losses
of about 0.00077–0.00083 for coefficients with \(q=1\), versus 0.00262–0.00401 with \(q=0\).
The evolving model's coefficient means are about 0.0786–0.0836 against the generating value 0.08.
This checks that specified process. A separate constant-design counterexample produces the same
64 observations from a coefficient change or a state change: separate variables alone do not identify the cause.

Ten 128-path conditions are complete: the six frozen inputs at the primary setting, plus \(q=0.1,10\)
for the persistent control and ordinary 12. The table shows mean bounded loss over all directions,
24–40 seconds and scale 1; lower is better. The coefficient model includes partner features.
The state-only model also conditions on current-value continuation and learns residual variance.

| Sample | Previous stationary coefficients | Coefficients + state | State only | Unlearned continuation | Point persistence |
|---|---:|---:|---:|---:|---:|
| 08 | 0.188441 | 0.235927 | 0.256376 | 0.154067 | 0.144183 |
| 09 | 0.227478 | 0.313516 | 0.321493 | 0.197024 | 0.184225 |
| 12 | 0.182183 | 0.221825 | 0.225368 | 0.094689 | 0.074750 |

The model without partner features also loses to continuation controls in the ordinary primary windows.
For the constructed persistent, returned and reversed-return conditions, the focal 277→733 Hz direction
at 44–60 seconds gives coefficient-model losses of about 0.0488, 0.0489 and 0.0486. These improve on the
unlearned continuation distribution at about 0.0576, but lose to the preceding stationary model at
about 0.0459, 0.0458 and 0.0458. At \(q=0.1,10\), coefficient-model losses are about 0.2239 and 0.2283
for ordinary 12, and 0.0488 and 0.0556 for the persistent control. These sensitivities do not select
a cognitive time constant or a production value of \(q\). The candidate is not adopted in production.

All 4,065 completed targets, 101 pending targets and 365,850 score components are checked against actual outcomes.
Independent reconstruction covers 18,091,332 signed Student density terms; maximum mixture log-density difference is 4.26e-14.
For 225 selected independent integrals, maximum primary-score error is 1.63e-05, falling to 1.23e-09 at order 32.
Thirteen reference warnings are rechecked in logit coordinates. Nine resulting warnings at nearly coincident
interval boundaries are resolved by bounding omitted interval contributions by at most 1.88e-13.
The tail bound is 8.50e-18, and agreement with the original references is within 1.15e-14.
Initial warning records are preserved and primary scores are unchanged. These reuse six inputs,
not independent auditory trials.

Approximation convergence remains unresolved. A single primary update discards posterior mass of
about 0.5 in 08/09 and 0.106 in the persistent control. Additional 512-path forecasts and updates for
these three inputs are compared at every tenth query and the first query after maximum primary pruning,
at 63 scale-1 coordinates. Across 121,338 selected points, three models and all directions, maximum
CDF differences are about 0.118, 0.00659 and 0.341 for 08, 09 and the persistent control respectively.
All 6,174 unlearned forecasts match exactly. Maximum discarded masses at 512 paths remain about
0.5, 0.5 and 0.0873. These three additional conditions are prediction comparisons only; they are not
included in the ten scored and independently density-audited conditions above. Selected points do
not establish convergence over every query or coordinate.

**Acoustic state and forecast deviation have different meanings.** Equal future sound can have
different deviations when issuance offsets differ. For future values all equal to 1, offsets
\([0.2,0.4,0.8,1]\) give uncorrected deviations \([0.8,0.6,0.2,0]\), whereas offsets all equal to 1 give zeros.
This state is a forecast-origin-conditioned deviation, not acoustic state or stream identity.
The existing `AcousticPosterior` and `DrivenBodyPosterior` separate waveform state and parameters,
but do not hold relational memory. `ContinuationContexts` likewise holds regression hypotheses,
not recognized auditory streams.

The next design separates P-state updated by actual sound, issuance context frozen for each prediction,
and M-state retaining reusable relationships. Additional forecast requests must not create extra
P observations or M experiences. Preserve continuous evidence and its coverage before attributing
state tracking to retention governed by elapsed time or intervening information. Cognitive retention
and interference, stream identity, phrase/closure, situated action value and author acceptance remain
incomplete. The 107 production Rust/Cargo files and both release binaries remain at §9.3.47; no new
audio, author audition or hardware acceptance is added. Evidence is stored under
`target/phrase-expectation/2026-09-09-evolving-state-retained-coefficients/`.


### 9.3.54 Continuous acoustic observations, independent requests and missing input

Inspection of existing paths clarifies the boundary from §9.3.53. Production `SpectralHistory` and
`BandEnergyHistory` already update at acoustic observation boundaries. The regional research
regressions learn completed query targets. That defines their calibration sample, not a count of
continuous auditory experiences.

The research-only `DrivenBodyPosterior` now has `advance_to`, which composes its existing linear
transition and drive covariance across explicitly missing input. It advances vibration mean and
covariance without conditioning on invented measurements. The variance posterior, log evidence
and observed-sample count remain unchanged. The transition and omission of measurement updates
follow [Särkkä and Solin (2019), §§6.2 and 10.6](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
Physical damping can reduce vibration variance even without observation; that is neither cognitive
forgetting nor new evidence. This state concerns waveform vibration, not beat phase or stream identity.

| State or record | Updated by | Not updated by |
|---|---|---|
| Vibration state and its clock | Conditioning on actual sound or explicit missing-time evolution | Another forecast request at the same time |
| Unknown variance, observed count and last observed endpoint | Newly received actual samples | Missing input, forecast issuance or scoring |
| Issued prediction | Frozen at issuance | Later filtering or another request for the same target |
| Prediction outcome | Scored after the complete target waveform arrives | A missing interval or unobserved EOF suffix |

`stream_predictions` consumes contiguous intervals explicitly marked as observed or missing and
updates each actual sample once. Forecast requests have a separate schedule. Overlapping or
repeated predictions do not duplicate the acoustic update. Each prediction is a joint distribution
over a future waveform interval, so an intersecting input gap censors that target. Predictions may
also be requested during missing input; the state clock and last observed endpoint remain distinct.
EOF leaves incomplete targets pending without silence padding and distinguishes future unissued requests.

Sparse-observation filtering is checked against an independent dense continuous covariance
conditioned only on observed indices. Cases cover no modes, no drive, and driven/damped modes;
tests distinguish numerical zeros from missing input, partition missing-time evolution and reject
invalid clocks. Streaming tests cover overlapping predictions, repeated requests, no requests,
different chunk boundaries, gaps, censoring and pending targets. All 411 Python tests pass.

Reused ordinary 08/09/12 audio supplies 8.0–8.5 seconds at its original sample rate and amplitude.
Passive 0/1/2-mode candidates are fitted only to the preceding 40 ms, and the largest available
declared support is retained. Drive ratio 100, initial precision 0.0001 and IG(2, 1e-6) reuse a fixed
numerical hypothesis without selection on these results. Parameters are available before the
forecast interval. Their uncertainty and human cognitive time scales are not identified here.
Joint 20/100-ms forecasts requested every 50 ms are compared with a denser 10-ms schedule plus
exact repeated requests, and with no requests. Input chunks also vary across 512, 257 and 24,000
samples. The comparison is repeated with the 8.20–8.25-second interval explicitly unavailable.

Across 18 conditions, 1,008 issued predictions account for 795 completed targets, 87 targets censored
by missing input and 126 pending targets. Each condition updates 24,000 new actual samples without
a gap or 21,600 with the gap, independently of query count. The preceding 40-ms initialization
contains a separate 1,920 observed samples. In 396 common-target comparisons, maximum differences
are 9.03e-16 for predictive mean and 2.57e-16 for predictive variance. Final-state differences are at
most 9.63e-17 for mean, 1.61e-15 for covariance, 1.41e-13 for the variance scale and 6.04e-10 for
accumulated log evidence, including floating-point summation differences from chunk partitioning.
A separate persisted-ledger reader checks all interval continuity, observed counts, issuance times,
actual targets, censoring and EOF accounting. These comparisons reuse three inputs; they are not independent auditory trials.

This verifies an observation/request contract, not improved prediction accuracy or musical flow.
Long-duration ordinary performance and human perceptual calibration are not tested. The observed
count belongs to a conditional waveform likelihood; it is not cognitive memory strength or retention
duration. No direct transfer of this count into relational memory is implemented.

The next step must determine what continuous P evidence and uncertainty supply to temporal
relationship retention. Acoustic tracking of new sound and retention/interference governed by
elapsed time and intervening information remain distinct mechanisms; issued prediction errors
alone do not provide this input. Cognitive retention/interference, stream identity, phrase/closure,
situated action value and author acceptance remain incomplete. This is research code only: the 107
production Rust/Cargo files and both release binaries remain at §9.3.47, with no new audio or author
audition. Evidence is stored under `target/phrase-expectation/2026-09-09-continuous-acoustic-evidence/`.


### 9.3.55 Temporal DCC across the hierarchy and its connection to local generation

The present task is temporal DCC using the existing harmonic/consonance landscape
as an input. It includes articulation, groove, beat/meter, phrases, repetition and
variation, sections and experience of the whole piece. Work so far concentrated on
short-time participation and acoustic history/prediction; it did not complete the
connection from neurocognitive evidence to representations and interactions across
these functions. `docs/design-notes/dcc-neurocognitive-hierarchy.md` now gives a
design specification with updates, connections and discriminating comparisons.
Improving acoustic prediction does not count as completing temporal DCC.
Actual Fable review round 39 returned no actionable issues on the exact memo
version recorded in `docs/design-notes/dcc-hierarchy-review.md`. The completed
delivery plan, `docs/roadmap/temporal-dcc-completion.md`, starts from the audited
current worktree and covers M0–M9 through all temporal functions and four distinct
acceptance gates. Review acceptance completes this design task, not the future
instrument or its cognitive validation.

**The common principle is retaining temporal relations and returning them to the
environment for generation.** Observed sound supports hypotheses about order,
overlap, grouping, recurrence and transformation. Each retains its supporting
intervals, uncertainty, temporal extent, persistence/revision conditions, expected
continuations and context links. P/M/A applies across these scales. Reusing one
sound through different summaries must not count it as independent evidence.
Internal representation is allowed; Scenario labels and Voice IDs cannot be
passed off as perceived structure.

| Function | Proposed representation and update | Generative consequence and comparison |
|---|---|---|
| Articulation | Continuous trajectories and tentative attack/sustain/release/gap groupings | Phrase context changes sustain/release choices; vary duration/gap relations with onsets held fixed |
| Beat/meter and groove | Uncertain recurrence, relative timing and beat grouping, alongside bodily participation and its variability | Retain different participation positions around a shared beat; assess movement invitation/preference separately from synchrony |
| Repetition/variation | Ordered span relations and correspondence hypotheses, separating retained relations from current transformation | Test recurrence across register, speed, timbre and intervening material against reordered controls |
| Phrases | Internally moving continuation states and contextual boundaries; separate boundary, closure and acoustic change | Change prior context with the local ending fixed; compare with fixed-duration and gap-only models |
| Sections/whole context | Episode memory and recurrence/variation/contrast/unresolved relations; a retrieval/attention graph is proposed | Compare the same local material as new or returning, and test its effect on local participation |

Meter and phrase organization may cross, and concurrent streams may have distinct
boundaries. Duration and gap relations inform phrase hypotheses, while phrase
expectation changes candidate sustain/release consequences. Section memory changes
expectations of return, which affect interpretations of the same continuation or
variation. A higher-level state does not centrally schedule a note sequence. Each
body, with its ecological disposition, participates, overlaps, waits, varies or
declines an opportunity. The action rule and any metabolic value are hypotheses
to compare separately from perception.

There is direct music-specific evidence for long structure.
[Farbood et al. (2015)](https://doi.org/10.3389/fnins.2015.00157) compared fMRI responses
to music scrambled at different structural scales;
[Williams et al. (2022)](https://doi.org/10.1162/jocn_a_01815) compared neural event
boundaries with listeners' segmentation. [Hołubowska et al. (2026)](https://doi.org/10.1111/ejn.70481)
reported phrase tracking with irregular lengths in Bach-derived stimuli. These
motivate comparisons of long context and nonperiodic segmentation. They do not
identify the proposed graph, universal windows or a genre-independent closure law.
The memo distinguishes each study's observations, adopted constraints and limits.

Scenario supplies artistic direction alongside internal perception and memory of
the unfolding work. An instructed return differs from an audible return. Whole-piece
context represents experience so far; future duration, a scheduled ending and a
canonical form are not heard facts. P/M/A is the common contract, not evidence that
each scale's cognitive model has already been implemented.

The selected reference combines a single conditional-state update, ordered span
correspondences with explicit transformations, separate elapsed-time/interference
retrieval, duration-dependent phrase/section transitions, and a separate closure
estimator. These are engineering hypotheses constrained by the cited tasks, not
uniquely derived neural mechanisms. Authored local dispositions act on separate
contextual consequences; cognitive likelihood supplies no new survival reward.
Memo §§9–10 specify fitting, alternatives, resource limits and comparisons for
articulation through whole-piece experience. Parameters and cognitive validity
remain to be identified; selecting an algorithm does not establish its validity.

Generator and listener relation models process habitat and presentation separately;
a Voice also retains its own executed participation. Shared prototypes provide
bounded action-consequence tables, while Voice decisions preserve bodily timing
and multiple relative positions. Unsupported or stale context adds no pressure.
Episode retention is identified against human return recognition, private timing
traces against observed execution, and groove/participation heads against separate
listener judgments. Those tasks cannot substitute for one another. Bounded beam
pruning preserves enumerated mass as unknown, while reporting lost identities;
computation limits and latency targets remain engineering constraints to validate.
The boundary task asks for the earliest grouping change among simultaneous audible
strands, with its exit type. Its factorized first-event calculation retains unresolved
mass; it does not require a common ending or reset local clocks. Ordinal judgments
use a separately declared whole-mixture aggregation. Local overlap and audibility
use only the Voice's own-excluded energy forecast; shared mixture energy is not a
substitute. Configuration changes explicitly restart cognition under a new epoch,
while ordinary acquisition gaps preserve memory with missingness. These are selected
reference policies with numerical comparisons, not established cognitive laws.
A candidate preview carries hypothetical feature support separately from heard
support, while candidate/default differences share the issue-time coverage factor.
Rendered counterfactual continuations must validate this transfer before ordinal
consequences are enabled. T2 also has its own human omission-expectancy judgment;
an oscillator continuing through missing sound is not that judgment. T7 includes
a separate retrospective whole-piece completion judgment and fitted scoring head,
compared with short-closure, recent-energy and elapsed-time controls; the ongoing
return/context and retrospective-completion endpoints remain distinct. An open
ending is valid, and EOF itself never enters the prediction. Before validation,
M5 must size memory/index capacity against complete long-form development replays
and a roomy offline reference, preserving supported return cues at the declared
delays. Validation failures remain failures; a foreseeable capacity artifact must
be resolved before that freeze.

The public Japanese Manifesto currently describes real-time extraction of four
delta/theta/alpha/beta bands as an implemented capability. The audited
`src/core/modulation.rs` derives delta/theta from meter and uses alpha/beta as
precision/error scalars; these are not four extracted neural bands. M0 owns the
public wording reconciliation and records its result here, preserving the
neurocognitive design commitment without overstating the current mechanism.
M0 also freezes a hardware/workload baseline. Runtime acceptance requires 20%
headroom in the processing slice on both simultaneous buses; a benchmark miss
follows the memo's ordered optimization, bounded-model revision and refit path.
These are future implementation gates, not measured performance of this design.
Cross-group handoff is now an explicit bounded gesture view: a two-group union
can be judged as one gesture without merging its acoustic sources or Voice clocks.
Larger unions remain a declared capacity limit to measure before validation.
Grouping judgments distinguish the kind of grouping from its count at the
listener's chosen tactus; half/double beat levels do not automatically imply a
different grouping extent. The reference also separates short-state uncertainty
leaks from phrase/section leaks and tests quiet/gap recovery of committed context.
Named closure/section controls and a finite fitting-compute envelope make these
claims testable without identifying the engineering constants as cognitive laws.
Phrase exits now specify the successor's clock and the older unresolved span:
overlap starts a new foreground while preserving continuation links, and
reinterpretation inherits the heard start. These do not force cognitive closure.
Before human collection, M0 checks numerical feasibility on saturated and expected
loads. A registered reduced-cost fit uses complete-passage probes but accepts
changes only after full-corpus replay and bounded objective-drift checks. The memo
and delivery plan share 22 obligation IDs linking each requirement to its owner
and the gate it blocks; future artifacts remain future work.
T1 treats a definite overlapping gesture as known even if another group remains
unresolved, using the existential probability checked against joint enumeration.
The registered breadth includes gradual tempo drift, and every novel listening
instrument must pass declared pilot-comprehension and usable-response criteria
before its acceptance margin is set. The instrument's audio air-gap remains
separate from perceptual label/private-state isolation.
The reference fixes temporal mode for each performance: passive observation builds
the same new temporal state without action pressure, participation may use it, and
off constructs no new relation state. A new performance begins with empty banks;
an assay state intervention is explicitly distinguished from a live mode switch.
Before stage-1 collection, synthetic parameter recovery must also separate time,
interference and saturation at the planned study size under registered tolerances;
passing this simulator gate does not validate human memory. Every sensitivity
comparison has an advance metric and adoption/limitation rule. Boundary scoring
distinguishes retrospective window-end weights from issued forecasts and accounts
for mid-window history loss separately from the raw first-exit hazard law.
Event-head predictive stability also has a synthetic pre-collection gate. O04
includes tuple composition, grouping refresh and the Voice decision slice with
its own hop CPU headroom, not only relation workers. The initial phrase foreground
remains group-local: cross-strand continuations have a dedicated T5 instrument
and representation-miss report; demonstrated required handoffs reopen M4 before
T5 completion. Public wording reconciliation checks each per-band functional-role
sentence as well as the four-band extraction claim, including the temporal
section's cross-band agent-entrainment sentence; the four correction clauses are
numbered for language-by-language comparison. Recognition and correspondence
have fixed replay/exposure rules and disjoint cohorts. A recognition query cannot
reinforce itself in the pre-query scoring bank. Resource revisions now also order
decision, matcher, gesture-view and pair-grid reductions, each with its required
fidelity/refit gates; epoch, evidence and mode isolation fixtures have named owners.
Gesture-family priors now have their own explicit acoustic-support/window rule,
and body/group association uses a shared six-coordinate observable descriptor.
Groove and desire use the same fixed 109-coordinate input layout, retaining separate
fits and judgments; shared feature definitions do not identify the two experiences.
All 30 section correspondence/transition coordinates now have explicit category,
weight, adjacency and coverage rules. Gap handoffs use observed low-energy entry
after release, and required longer-gap misses reopen the gesture/phrase design.
T2 can report 2–16 and larger equal groups, with a ruler long enough for the
reference period range. The Manifesto audit includes temporal claims outside its
temporal subsection, including the enclosing two-axis-landscape sentence.
Bundle formation and cross-hop persistence now have deterministic member/parent
rules, and milestone IDs are pinned bidirectionally between the specification
and delivery plan. These close implementation choices without supplying evidence
that the resulting grouping matches listeners.
Single-stream microtiming remains required: a separate within-group periodic
interval history supplements inter-group pairs, with added state explicitly
budgeted. T3 compares systematic swing and matched interval-marginal jitter,
separating histogram information from order. Mono energy, downmix and playback
channel conventions now pin the analysis/stimulus relationship. M0 also records
ethics-review status, consent, compensation and individual-response handling
before any human pilot. These are delivery prerequisites, not completed studies.
The acoustic association reference now fixes member-level continuity weights,
residual mass and reference-refresh order, with explicit state/work bounds and
energy-conservation fixtures. Ridge secants, log-RMS Pearson bundle correlation,
accent-bank cap order and the section cue-selection window are also specified.
All period consumers share a peak/separation rule rather than selecting adjacent
high bins. Episode interference uses cached all-bank coarse comparisons, with
explicit unknown increments and commitment costs. Phrase-head inventories must
be fixed before their synthetic gate, and T2's raw-support cutoff excludes the
entire target interval from the issued forecast.
Completed-span ending descriptors anchor at original observed support and clip
their windows at the span's start, preserving short-span contrast and making
the predecessor-window alternative an explicit comparison.
Integer grouping now covers every extent 2–16, including 13, with corresponding
scan bounds. Private-fit freshness follows the current auditory binding and
matcher support, so a Voice's inactivity is not mistaken for stale memory context.
Arrival phase uses a circular encoding; section overlap explicitly measures
assigned resolved-group coexistence. Continuation's phrase-survival consumers
keep their fitted unscaled hazard input, and pilot readiness is distinguished
from the separately frozen validation count-coverage requirement.
Memory controls now name exact no-retrieval, recent-only, time-only and orderless
bag operations, with their information losses, refits and costs. Articulation-rate
controls also have fixed input layouts. The shipped reference temporal default is
off; M8 identifier registration cannot silently change previously accepted behavior.
Integer boundary search now explicitly uses both sides of each predicted time,
split persistence has its own conditions independent of birth support, and the
section periodic fraction includes recurrence of nonuniform cyclic words.
Episode knots now have ten specified acoustic coordinates, a two-hop cadence,
gap/partial-block rules and weighted-error compaction with explicit costs.
Original-time matching retains density-mismatch limits and required comparisons.
All seven action classes have legal timing subsets, and DTW insertion/deletion
penalties have named units, stage-1 selection and sensitivity/freeze rules.
The phrase hazard and exit categorical now share an exact 26-coordinate input
layout with separate fits. Inter-group timing fixes the pre-accent period snapshot,
the pair cache specifies f32 contributions with f64 sums, and all five-level
heads use the same cumulative-logit proportional-odds family. Continuation-control
wording distinguishes unscaled survival input from ordinal output calibration.
Arbitrary live bodies now obtain the common descriptors from their own actually
emitted audio through a separately budgeted private path; newborn and mutated
populations remain inside runtime acceptance. Required non-primary controls have
an owned registration of statistics, clustered intervals, thresholds and failure
consequences. The existing meter and the new proposal inventory restricted to
2/3/4 are distinct controls with explicit grouping/count maps. The Manifesto
sentence audit also names individual entrainment in the emergence section and
sweeps other agent-behavior temporal claims. These are design obligations, not
newly demonstrated cognitive or runtime capabilities.
Promotion of the new participation trace replaces the 2x3 memory contribution
only in local-participation mode. Off and passive modes retain the accepted legacy
generation behavior; a later shipped-default change needs separate author acceptance
and a versioned default/baseline fixture. The phrase heads' 64-input bound is also
explicitly separate from stage 3's coefficient cap and the larger registered heads.

Begin with a relation experiment using repetition, variation, intervening contrast
and return, then carry existing observations through the passive hierarchy and
connect supported relations to local actions. Higher-scale work does not wait for
universally successful waveform prediction. All temporal functions remain required
if a selected reference estimator fails a comparison.

This revision changes design documents only; it adds no generation mechanism, audio
or audition result. Implementation correctness, cognitive-task correspondence,
audible distinction of the intended relation and artistic preference remain separate
acceptance claims. Existing short-time participation/history does not complete the
whole hierarchy.

#### M0 public-claim reconciliation (2026-09-10)

The Japanese and English Manifestos were compared with baseline commit
`ec67b50c303d930a3d4c810333e3742f111c6ac8` and the actual source, then revised.
In `src/core/modulation.rs`, `NeuralRhythms::from_meter_state` maps beat/subdivision
to delta/theta and stores confidence and its complement as alpha/beta. This is not
four-band neural extraction. `TemporalParticipation` and `ParticipationClock`
implement intrinsic-period/sound-duration separation and sounding, waiting or
skipping from short acoustic context; they do not implement the new relational
memory or whole-piece cognition.

The four numbered clauses from memo section 3 retain their IDs. The exact adopted
Japanese text and canonical English are fixed below. The English Manifesto conveys
the same four commitments in its surrounding prose.

| ID | Exact adopted Japanese wording | Canonical English |
|---|---|---|
| 1 | 2026年9月時点の実装では、拍と細分拍の推定からデルタ・シータと名づけた周期信号を作り、アルファ・ベータは推定の確かさと予測誤差の指標を表す数値として扱っている。 | The current implementation derives delta/theta bands from meter estimates and represents alpha/beta as precision/error scalars. |
| 2 | 音響信号や脳信号から四つの神経帯域を抽出しているわけではない。 | It does not extract four neural bands from audio or brain signals. |
| 3 | Conchordalは、これらを支える神経認知的な時間構造とその相互作用を内部で表現することを設計目標とする。 | Representing interacting neurocognitive time structures is a design goal. |
| 4 | 以下の四帯域への機能配分は初期の案であり、再検討の対象である。 | The four-band allocation remains a revisable proposal. |

The sentence-level comparison covers 18 bilingual paragraph pairs, not only the
extraction sentence. `docs/roadmap/temporal-dcc/manifesto-audit.json` stores the old
and new text, dispositions, line locations, and SHA-256 hashes of both Manifestos
and 15 inspected source/specification files. IDs below follow the record's `O02-`
prefix.

| Location / IDs | Disposition |
|---|---|
| `intro`, `dcc_mapping` | Retain autonomous sound agents and DCC principles. Biological feedback was already future work; existing model-to-audio mapping does not complete temporal cognition. |
| `landscape` | Replace completed real-time two-axis terrain with current frequency terrain, short history and meter, plus the temporal design goal. |
| `temporal_mechanism`, `current_mechanism`, `allocation` | Remove four-band extraction and cross-band entrainment as completed mechanisms. State current signal provenance and proposed relational memory, expectation and return to generation. |
| `delta`, `theta`, `alpha`, `beta` | Mark large-scale phrasing, articulation, phrase accents and groove/microtiming allocations individually as revisable proposals. |
| `long_context` | Register order, retrieval, boundaries and closure as future representational and listener-comparison work; do not reduce all scales to slow oscillators. |
| `individual`, `population`, `emergence` | Replace neural entrainment with implemented local participation. Separate frequency niches from future temporal roles; sustained handoff, phrase and whole-piece emergence remain goals. |
| `scenario_scope`, `scenario_controls` | Retain the authored large-scale arc. Describe available period, duration and synchronization controls; coexistence with perceived long context remains future work. |
| `closing`, `name` | Retain autonomous agents, ephemeral performance and the coral metaphor as commitments, not proof of completed temporal cognition. |

This reconciled public source text; it did not deploy the site or demonstrate a
new cognitive mechanism. The memo at that initial reconciliation had SHA-256
`8b88ebcc987e6a9b4a8923f68748469c544387b2ba38c1e1d47f094122d681d9`.
Its description of the old public claims refers to the baseline commit; this
ledger records their correction and the subsequent numerical revision below. M0's remaining obligations are tracked in
`docs/roadmap/temporal-dcc/m0.md`.

#### M0 episode-recovery allocation failure

The first conditional recovery design, `episode-allocation-1`, was registered in
`docs/roadmap/temporal-dcc/episode-recovery-design-v1.json` before simulation. It
crosses delay, observed interfering-content mass, occurrence count, assignment
support and cue quality in 720 cells, with four responses per cell: 2,880 targets
under a candidate 960-listener allocation. No listeners or waveforms were collected.
Acoustic scores are fixed engineering nuisances. Three conditions cover complete
responses, 10% independent missing responses, and weaker cues with competing
episodes and ambiguity-dependent missing responses. These do not yet cover
acquisition gaps or fitted acoustic scales.

All 108 parameter stress points were simulated 100 times in each condition,
yielding 32,400 fits. Of 324 grid-condition points, 234 pass the registered
parameter-wise error criteria and 90 fail. All 81 points with a two-second
retention constant fail; the other nine failures have a twenty-second constant.
The 3,272 failed fits remain infinite errors in acceptance statistics. No criterion
or grid point was removed. `docs/roadmap/temporal-dcc/parameter-recovery.json` links
the complete results, exact runner, seeds, assignment and joint-error figures.
This candidate allocation does not pass O11 and cannot authorize human collection.

The failure exposes a task-design issue rather than establishing a human memory
limit. The shortest registered delay is four seconds before the eight-second
query, so the earliest scoring time is twelve seconds after reinforcement.
At that time, `tau=2, bias=0` and `tau=3, bias=2` give the primary retrieved
episode the same time-plus-bias penalty; later observations must separate them. With `kappa=4, strength_max=3`,
the expected conditional-response log-likelihood ratio over the entire allocation
is only 0.310 nats with clear cues and 0.038 nats in the weak-cue condition.
This is a post-run explanatory diagnostic, not an acceptance threshold. An
independent expected-count oracle recovers the short-retention parameters when
information is sufficient; optimizer correctness alone does not make this
allocation adequate. M0 must revise the observation/task allocation or register
an explicit scoring-model revision and repeat the full gate, retaining the short
retention stress points. The accepted design memo and live instrument remain
unchanged at the end of that first allocation's evaluation.

#### M0 auxiliary-query recovery and remaining private-trace semantics

The revised allocation keeps every primary eight-second cell and adds 1/2/4-second
queries, with shorter delays only where observed intervening spans can commit
before query start. A registered expected-Fisher search compared 4/8/16/32 ratings
per cell before any new simulated responses were drawn. Six, two, zero and zero
stress-condition points respectively failed its planning precision limits, so it
selected sixteen. These planning limits do not replace the unchanged recovery
criteria. The resulting 2,640 cells require 42,240 responses from a candidate
3,840 participants, eleven distinct source families each. Estimated audio alone
averages 32.52 minutes and reaches 44.54 minutes per participant, before instructions,
responses or breaks. This is a material study burden, not an approved recruitment.

`episode-allocation-2` then passed all 324 truth/nuisance points in 32,400 independent
synthetic fits. The full grid, all fit errors, assignment, exact executed runner,
planning search and figures remain linked from `parameter-recovery.json`.
The run took about 62.14 seconds and 488.64 measured simulation/parent CPU seconds;
the latter excludes process startup. This is neither an audio real-time benchmark
nor a human-memory result. The registered engineering cue scores, homogeneous
participant model and absence of acquisition gaps limit the result.

The hierarchy memo and completion plan now include the auxiliary identification
queries while retaining the original primary task and retention equation.
`recognition-instrument.json` fixes the duration-specific questions, no-replay and
bank-copy rules. `control-comparisons.json` registers a primary eight-second transfer
check against fitting eight-second queries alone; its human split, power and final
meaningful margin remain unfinished. Actual short-query correspondence, duration-
specific pilots and recovery after acoustic-scale fitting remain prerequisites.
The old round-39 Fable result applies to its recorded earlier memo hash, not this
numerically motivated revision. The live instrument remains unchanged.

The independent private-trace work has verified fractional executed credit,
missing/retired-reference mass, exact uniform timestamp-difference bin integration,
multiple frozen anchor modes, overflow and conditional normalization. Shared extra
wait/interference multiplies every retained bin by the same factor, which cancels
on normalization; waiting-only probes cannot identify private tau/kappa.
The stateful reference must therefore specify recurrence accumulation and saturation,
the reinforcement/interference order under fractional assignments, and each bin's
observed timestamp support before recovery. Literal independent one-occurrence
strengths never activate a cap above one, so they cannot establish strength-cap
identification. `private-trace-reference.json` records these unresolved semantics.
The subsequent stateful reference resolves accumulation with capped **retained
mass per bin**: decay old mass, add the integral of observed timing credit, cap
the bin, then apply competing-reference interference to both heads. It integrates
the exponential timing moment over known uniform outcome/anchor supports rather
than substituting a mean or delivery time. This explicit private-filter revision
leaves episode-strength renewal unchanged. Tests cover scalar recurrence,
independent timestamp quadrature, head separation, lost support, capacity eviction
and long waits without converting log-space retained mass into an empty trace.

A separate exact observation audit finds a fitting defect. With two phase-bin
masses (0.8,0.2), four quarter-period observation intervals offset by one eighth
period have probabilities (0.4,0.25,0.1,0.25). Their uniform soft-bin assignments
average to (0.65,0.35), so the earlier soft-bin cross-entropy favors a biased
distribution even with infinite data. A likelihood obtained by integrating the
phase density over each observed interval has its population optimum at the
true distribution. The same bias is verified with 32 bins and 512-sample observed
intervals at 48 kHz, including the known partial last interval of a one-second
period. These diagnostics do not replace the full 32/33-bin parameter grid.
The multiple-reference physical-outcome law, including
anchor alternatives, nonperiodic overflow and missingness, still needs completion
before the recovery simulation; independently drawing outcomes for each
reference would fabricate observations. All 32 reference tests pass, but no
private parameter recovery or production-template replacement had passed at that
point. The next revision supplies the missing joint observation law.

Each probe freezes a finite physical opportunity window, its target head and
detector intervals at issue. Within that window, each reference lifts its complete
32/33-bin distribution to physical time, retaining the frozen anchor alternatives.
Nonperiodic overflow occupies the union of the available outside regions. Missing
reference/anchor weight and bins with no legal intersection use a shared uniform
physical-time baseline for this assay, with no live memory/action support. The
reference densities are mixed using the original issue-time weights. Only one
physical outcome is sampled; observed result credit neither changes that frozen
prediction nor creates independent observations for each reference.

The fitting likelihood integrates the mixed density over each known detector
interval and retains missing detection as one additional category. The old
fractional soft-bin loss remains a separate diagnostic. Analytic integration of
affine overlap-length ratios keeps uncertain anchor priors normalized before
mixing; adaptive quadrature and the earlier single-period calculation agree.
The reduced recovery forward and its analytic gradient also match actual
sequential updates of both heads, including fractional credit, missing exposure,
multiple timing modes, overflow and competing references.

The first private recovery allocation selected 1,024 independent probes for each
of 140 exposure cells in each of three nuisance conditions. It retained all 36
tau/kappa/cap truth points and 100 datasets per point. Of 108 condition points,
107 passed; all 73 failed fits count as infinite errors. The failing point is
tau=2 seconds, kappa=1 and cap=3 under ambiguous anchors/missingness. Its 11 failed
fits prevent a finite 90th-percentile error. Every fit was saved, but a NumPy
boolean serialization error prevented the final timing/result JSON. Summaries
were reconstructed without rerunning fits; exact final CPU/wall counters remain
unavailable, and the last captured progress line reports 99.4 seconds. No complete
cost gate is claimed for that run.

The uniform design would require about 45,883 hours per nuisance condition if
every prefix were replayed separately, so it is not an empirical collection plan.
A prospectively registered Fisher optimization retains all 140 exposure cells,
at least two probes per cell and all truth/condition constraints. It selects
9,086 probes per condition, reducing the same replay bound to about 554 hours.
The maximum expected log-parameter standard errors remain below 0.2. A separate
registered nonsmooth fallback refines fits that stop at the retained-mass cap;
it changes neither the forward law nor the recovery tolerances.

The new v2 run passed 106/108 points. Its local Fisher criterion did not separate
a globally different low-cap branch: when every addition saturates a bin, cap
cancels from normalized timing. V3 added a calibrated separation probe against
that branch and required full-rank fitted information. It passed 105/108 points;
the rank check exposed another flat branch where accumulated mass never reaches
the cap. V4 doubled all probe counts and passed 106/108 points, retaining 43 failed
fits, including 40 with deficient rank. These failures remain in all denominators.

V5 fixes a solver error: a higher-loss converged candidate must not suppress
refinement of a lower-loss nonsmooth cap corner. Unresolved lower-loss attempts
and rank-deficient best fits still fail. With independent random outcomes, the
unchanged 50,932-probe allocation again passed 106/108 points. The remaining failed
points have tau=2 seconds, cap=6 and kappa=4/16 under ambiguous/missing observation.
Thus the fitting correction does not establish sufficient experimental information.
A registered v6 allocation search added probes only to four existing late-exposure
cells and profiled the unsaturated alternative over tau/kappa. None of its five
candidate counts met the advance distance criterion at all three stress truths.
The search selected no allocation and launched no new recovery. Finite multistart
profiles are a planning heuristic, not a global certificate. The next revision
must reconsider informative exposure timing; merely meeting a local Fisher
standard-error target is insufficient.

All independent recovery runs retain their exact designs, sources, fit records,
plots and failure counts. Actual acquisition-aligned timing, body/opportunity
feasibility, waveform-derived references and an accepted replay method remain
outstanding. Separately, O03 now registers the section descriptor's exact
39-coordinate order, support-weighted edge/transition statistics, unresolved
mass, four-span ring, periodic-proposal union and resolved-only overlap. The full
82-coordinate head and O12 mixture-first-event simulation were still outstanding
at that point; descriptor registration is not a fitted section model.

V7 changes the exposure schedule: after the original A prefix and independent
delay/interference, it prescribes 1/4/16 B outcomes before a single physical probe.
Both target heads and their actual counterparts use the original credit and
observation rules; elapsed decay continues between every B outcome, including
missing ones. All original A/delay/interference cells remain at B=1. The expanded
420-cell design passes sequential-filter and analytic-gradient checks. A
registered Fisher allocation with maximum log-parameter SE 0.12 selects 14,382
probes per nuisance condition, with a full independent-prefix replay bound of
about 1,034 hours per condition. The complete 10,800-fit recovery passes 104/108
points, retaining 152 failed fits, including 133 with deficient rank. All three
short-tau/high-cap ambiguous points still fail; a fractional-observation point
also narrowly exceeds the cap-error bound. No tolerance is relaxed.

Diagnostic starts near the true parameters find identifiable local optima, but
the retained datasets can favor a lower-loss unsaturated branch. Choosing the
higher-loss identified solution would conceal this failure. Under the ambiguous
prefix at tau=2 seconds, the maximum unsaturated cycle mass is only about 7.8995
per bin: a 0.6 credit split over two modes, one missing outcome in five and
62.5 ms spacing constrain accumulation. The cap=6 stress point lies near that
exposure ceiling. A future revision must examine genuinely informative exposure
support as well as timing; neither the new B count nor local Fisher precision
establishes global identification. These are assay diagnostics, not evidence
for a biological memory constant or an accepted empirical replay plan.

The section-head registration now includes the full 82 covariates and separately
fitted bias (83 hazard coefficients). Three exit logits use coefficient-wise
sum-to-zero constraints, including bias (166 free coefficients). Recent-only
uses 41 covariates plus bias; elapsed-only has two inputs including bias. Raw
recent-minus-cumulative differences precede global development standardization.
The missing-support feature is one minus the observed physical support union's
fraction of the clipped candidate-section window: known inactivity is observed,
overlapping intervals count once, and fractional association weight is not
missing time. Contrast masks require both descriptors; no extra learned mask
coordinates are added. Forecasts advance duration with frozen context, and stale
matcher audio masks the two retrieval coordinates at the existing 0.5 s limit.
Four new numerical assembly checks cover these conventions, future-input rejection
and support partitioning. Full descriptor extraction, lineage/cue selection,
fitted scales and all O12 event-head simulations remain required. The exact
pre/post specification snapshots retain the earlier private/episode clauses
byte-for-byte; the section clarification has no new external review.

V8 retains the 420 ambient exposure cells and adds 72 cells with fully supported
first-prefix observations, preserving multiple modes, periodic missing outcomes,
the weaker later prefix and ambiguous probes. Its Fisher allocation passes the
local precision gate, but even multiplying all added counts by 16 fails the
registered unsaturated-alternative separation check. No v8 recovery is run.
V9 instead preserves every v8 count and minimizes replay cost against successive
profiled-alternative distance constraints, adding observations only in the new
stratum. Three profile rounds select 23,794 probes per condition, with the four
minimum profiled distances all above the unchanged threshold of eight. The
additional counts concentrate in two cells. This finite multistart planning result
is not a global identification certificate. The separate v9 recovery completes
all 10,800 datasets and passes all 108 condition/truth points. Four failed fits
(two rank-deficient, two unresolved lower-loss cases) remain infinite errors;
none are discarded. The maximum condition/parameter median and 90th-percentile
absolute log errors are about 0.09599 and 0.26284, below the unchanged limits.
The run takes about 534.41 seconds and 4,198.61 CPU seconds, excluding worker
startup from CPU accounting. Exact records, source, plots and all point summaries
are verified and retained. About 1,844 hours of independent-prefix replay
per condition also remain an unadopted empirical burden. The fully supported
stratum is explicit: this success does not establish recovery from weak exposure alone.

O12 now has a finite-hop first-event aggregation reference. Independent exhaustive
joint histories verify context/path marginalization, all-group loss before exits,
stable-handle tie ownership, initial unknown, retained first events and the
unconditional event/survival/unresolved identity. Issued forecasts and retrospective
heard-prefix scores carry distinct frozen-snapshot metadata; this interface does
not implement production issuance retention. A separate two-point Gauss–Legendre
reference integrates the duration-dependent hazard over each original observed
hop, applying calibration locally before aggregation. Its registered 135-case
envelope agrees with independent adaptive integration and step doubling within
absolute 1e-9 plus relative 1e-7 error. Nine added checks bring the reference suite
to 51 at that stage. Production history/register lifecycle, all
generating vectors and the full stage-2 fitting/OOF/calibration and 100-dataset
predictive gate remain required. This establishes numerical conventions, not
auditory segmentation or a fitted neural rate.

The section input work now provides six cached-input oracle functions with 21
checks. Original assigned acoustic hops supply the six ending observables, clipped
to epoch, ending generation, span start and the original support endpoint. Cached
correspondence alternatives remain path-local: multiplying separate marginal
category probabilities would invent an unregistered cross-record independence.
The 30 edge/transition coordinates retain unresolved adjacencies, fractional
support and excluded pair weight. The nine activity components retain admitted
nonuniform words, all supported outgoing/within-group timing histories and
resolved-only overlap. Cue selection uses actual supplied generation handles and
returns the whole selected prefix to the reused query; publication time and
availability cannot refresh or replace acoustic match evidence. O03 registers
the shared six-observable order and scaling rules. A contiguous-span fixture
connects these components to the 39/82 layouts. These exhaustive oracles consume
earlier-stage acoustic, detector, matcher and timing caches; they do not implement
those upstream estimators. At that stage, general overlapping-span/ring assembly, real lineage
ownership, the 640-byte record and all finite fitting/predictive-stability gates
remain incomplete. Numerical component checks do not establish heard sections,
cognitive parameter identification or real-time feasibility.

The gap-survival reference now uses bounded panel doubling of the same two-point
quadrature, holding the last supported context fixed. Endpoint evaluations count
toward its 64-evaluation limit. Successive small estimates alone can miss a sharp
initial hazard, so acceptance also requires a fourth-derivative quadrature bound
with a rounding allowance, using Bernstein bounds over the endpoint sigmoid range.
Absolute 1e-9 plus relative 1e-7 tolerances and the failure convention are frozen
in the feature manifest. Of 600 numerical audit cases, 416 return a supported
prior integral within the independent adaptive-quadrature tolerance; 184 exhaust
the conservative budget and remain unresolved. The largest accepted error is
about 0.05756 times tolerance. All cases remain in the archive. Exits during a
missing interval supply unknown-current-state mass, including unobserved multiple
transitions, never observed boundaries or inactivity evidence. Failed integration
leaves survival itself unresolved. Seven additional checks cover analytic cases,
calibration, partitioning, a missed initial hazard and retained unknown mass in
a subsequent observed first-event window. Existing reference functions remain
AST-identical. This brings that suite to 58 checks without changing the recovery
models. The 184 unresolved cases are not successful predictions; fitted-envelope
coverage, live lineage/epoch integration, O04 cost and full O12 stability remain
required. The record is `docs/roadmap/temporal-dcc/event-head-reference.json`.

The section reference now conserves the frozen ownership of each physical hop
fragment and admitted accent across overlapping spans. Each sealed span's scalar
membership multiplies its numerators, denominators and valid durations once;
the recent ring sums those sufficient statistics. Cumulative activity and missing
audio support still count new physical group observations once. Missing clock
records cannot disappear from a span's denominator, and a clipped fragment cannot
use a raw hop that extends beyond the observation cut. Twelve added checks bring
the section suite to 33, including overlapping-span 39/82 assembly, exact
correspondence and hand-summed occupancy for all 20 prefixes of 2/4/8-span rings,
replay handling, section starts and independent generation inheritance.

Each packed section record contains 54 f64 sums, six f64 original ending
descriptors, 32 padding bytes and 128 bookkeeping bytes, fitting the unchanged
640-byte limit. Five records use 3,200 bytes per local path. A prototype using
two f32 values per sum changed the 90% adjacency-validity decision; a saved
counterexample reproduces that failure, and f64 storage agrees with the
exhaustive reference on this case. Lower precision requires renewed validity
checks as well as scalar-error checks. These are payload measurements and
conditional cache-input tests. At that stage provisional ownership and sealing,
late accent delivery, live topology/ancestry and additional state/scratch costs
remained unimplemented or unmeasured. That clarification changes only memo §9.5, preserves
earlier recovery results and has no new external acceptance. Full O12 fitting,
real-time feasibility and cognitive evidence remain required. The exact record
is `docs/roadmap/temporal-dcc/section-feature-reference.json`.

The subsequent accent reference conserves assigned group energy, compares three
saliences using four canonical raw hops, and retains the uncertain middle-hop
event separately from the full supporting-audio endpoint and availability.
Twenty added checks cover fractional energy assignment, all four acquisition
and component masks, sample-support union, spectral-only flux, plateau peaks,
the strict threshold and a causal counterexample. The old cache interface could
include a recent event whose right-context audio was still in the future. All
three section extractors now enforce the complete evidence cut; perturbing only
the right context leaves the original ending descriptor and counts unchanged.

An ordered admission ledger retains at most 128 accents in 32 seconds, preserving
chronological/ID eviction and per-window capacity masks. Its independent totals
and section cumulative delivery path count an accent once even when its event
precedes the current acoustic delta. Replays add nothing, saved snapshots remain
unchanged, and explicit generation continuation preserves totals while resetting
the generation-local delivery watermark in existing bookkeeping slot 15. The
scalar inventory timestamp is the middle-hop endpoint; timing integrations keep
its full uniform interval. This convention and unrun inclusion-time controls are
registered in `docs/roadmap/temporal-dcc/accent-reference.json`. The 20 new checks
plus 33 section and 58 general checks total 111 numerical checks. This does not
finish pending-span ownership/sealing, arbitrary packet reordering, waveform
NSGT/group inference, real ancestry, O12 fitting or O04 runtime measurement. The
memo changes have no new external acceptance or human-listening evidence.

The next finite-input reference implements provisional occurrence commitment.
Stable original occurrence/support identities, ending descriptors and acoustic
payloads survive reinterpretation. Joint path weights are integrated once with
original observed support at the 0.5-second deadline; later processing time is
separate. Sealed writes and uncapped episode/context membership totals cannot be
rewritten by retrieval, duplicate reads or later interpretation. A strict change
above 0.25 in a joint support cell or unassigned support flags the original
occurrence once, measured against its sealed write. Missing lag support remains
uncertain, and late accent receipts can update pending activity without changing
the original ending descriptor. Post-seal receipts report support loss.

Twenty-one additional checks cover exact joint support, missing/unknown mass,
deadline and threshold sensitivities, original-time retention, metadata-only
revision, context eviction, same-cue coarse snapshots and bounded interference
increments. All 12 prefixes of 100 finite streams match independently summed
episode/context support. Conserved overlapping-span accent ownership reaches
the existing 39/82 section interface without counting acoustic support twice;
adjacency uses the original sample-support union. The numerical total is now 132
checks, recorded in `docs/roadmap/temporal-dcc/occurrence-reference.json`.
This oracle is conditional on supplied activity/path inputs and retains finite
audit history exhaustively. Its pending-count guard does not satisfy the
128-byte runtime endpoint layout. Full beam-dependent activity production, global
memory-clock/bank lifecycle, runtime bounds, O12 fitting and human gates remain
incomplete. The earlier recovery laws/results and external-review boundary remain.

The raw-descriptor reference next implements the ten assigned-hop observables,
generation-aligned 1/2/4-hop blocks and bounded moment compression. Twenty-nine
checks cover source/availability cuts, partial-acquisition sample unions, clipped
span supports, missing coordinates, immutable update APIs and generation isolation.
Ten f64 W/mean/error triples and timing/provenance occupy an actual 320-byte payload.
The 128-knot bank plus insertion, partial and gap scratch uses 41,920 payload bytes
per active span, or 85,852,160 for 1,024 independent paths on each of two buses.
Python headers, raw caches, outputs, beam state and matching/edge costs are extra;
this is not the full O04 memory/performance census.

All 54 combinations of six assigned-hop fixtures, 64/128/256-knot caps and 1/2/4-hop
cadence agree with direct uncompressed f64 reconstruction to a maximum absolute
rounding discrepancy of 8.53e-14. Independent moment tests cover 100 finite streams.
At the default cap/cadence a 43.69-second sustain has zero reconstruction error;
a rapid gesture has integrated standardized squared error about 21.676 and maximum
pointwise error about 1.307 standard deviations. A gapped fixture retains missing
weight and its gap flag but loses precise gap location in one merged block.
Compression is a reported approximation, not auditory forgetting or a measured
cognitive timescale. The numerical total is 161 checks, recorded in
`docs/roadmap/temporal-dcc/descriptor-reference.json`. The memo itself is unchanged.
Real waveform/group inference, original-time matching, ongoing query/episode
integration, fitted scales, T4/T6 evidence and the full M0 gates remain incomplete.

The next reference connects detached pending-prefix queries, coarse anchor
selection, original-time subsequence matching and sealed episode descriptors.
Thirty-two checks preserve the live descriptor's cadence, distinguish a trailing
gap from fresh supporting audio, and keep original occurrence age separate from
late delivery availability. Fifty small random inputs agree with independent
exhaustive edit-path enumeration to 13 decimal places. Missing pairs contribute
no residual evidence; unmatched cue/reference steps retain their explicit costs,
and outside-subsequence reference material is free. Pitch-motion and interval
residuals remain separate diagnostics, not new fitted cost terms.

Every eligible episode retains a generation-qualified coarse cost, including
those outside the 16 full-match candidates. The explicit handle adapter reaches
the existing occurrence ledger and section classification; late completion cannot
supply an earlier commitment, and an ambiguous cutoff remains unresolved.
Across 15 small searches, anchor spacing 2/4 recovers the exact, transposed,
faster and masked middle quotation. Spacing 8 misses the transposed quotation
and biases the other three pitch estimates by -0.1875 octave. The failed control
is retained and not adopted; the default's small-fixture success does not establish
broad or compression-density-invariant retrieval. The same-value shuffled control
retains positive cost, without establishing human order recognition.

One 256-episode/128-knot query visits 8,192 anchors, makes 121,349 median
comparisons, evaluates 62 refinements and 179,392 DP cells. The numerical DP
arrays have a maximum 7,824-byte payload at the default dimensions. Its roughly
0.276-second offline query already exceeds the entire worker's 40 ms budget.
This Python reference cannot be installed directly on that real-time cadence;
production kernels and the complete two-worker/64-Voice load still require O04 verification.
Query copies/exports, raw inference, cadence/queues, bounded cache layout,
bank/edge/memory-clock state and allocator costs remain unmeasured or incomplete.
The numerical total is 193 checks; records and the failed sensitivity fixture are
in `docs/roadmap/temporal-dcc/matcher-reference.json`. Memo section 9.3 clarifies
these numerical contracts; full fitting, real audio and human gates remain open.

The next reference adds acquired-sample query scheduling and fixed coarse-cache
storage. Each group retains one pending query and eight completed snapshots;
each bus has one outstanding worker ticket. Missing acquisition and repeated
delivery add no cadence credit. Pending replacement preserves waiting priority
and leaves a running job valid; group retirement and epoch restart invalidate
the result while holding the worker occupied until acknowledgment. This avoids
both stale-generation reuse and starvation caused by discarding every slow job.

Three saved counterexamples expose and fix earlier occurrence-ledger acceptance
of future full-source audio, stale audio refreshed by a trailing gap, and a query
from the wrong generation. The supporting-audio tracker retains the maximum full
source endpoint across observations. Cache receipt must precede the original
commitment deadline: an earlier worker finish cannot backdate late delivery.
Frozen writes remain unchanged after later cache eviction or group retirement.

The new component has 22 checks, alongside three additional occurrence checks
and one source-window check: 219 numerical checks in total. An independent
full-history oracle agrees with eight-snapshot retention across 100 streams of
80 insertions, including ties, older endpoints and replay. Nine virtual schedules
compare 0.05/0.1/0.2-second cadence and three stipulated worker delays. At the
default cadence, a 5 ms per-query delay completes 152 jobs in two seconds; a
276.46 ms delay completes six, with maximum completed support age about 286 ms.
These are simulated delays with empty result scans, not measured performance or
retrieval quality. Pending slots stay at most eight and cache slots at most 64.

The actual fixed buffers use 411,360 bytes for pending/active/query scratch,
401,408 for coarse records, 6,272 for shared coarse scratch and 808 for controller
arrays/occupancy: 819,848 bytes per bus. Scalar/object overhead, temporary capture
and exports, raw inference, bank/edges, memory clock, full ledger and worker
transport remain separate. Records are in
`docs/roadmap/temporal-dcc/query-scheduler-reference.json`. Full O04 performance,
all feature inventories/fits, real waveforms and human gates remain incomplete;
the earlier 40 ms budget failure still stands.

The memory reference now connects sealed occurrences to capped strength, uncapped
membership and elapsed/interference availability bounds. One bus acquisition clock
counts physical missing samples once across groups. Late delivery preserves the
original occurrence time; a recurrence clears its previous interference interval
before adding the same write's competing share. Retrieval and replay add no exposure.
The exact observed-rate window is (t-1,t]; rate overflow preserves actual retention
increments but leaves the gap envelope unverified. Affected availability then has
zero lower bound while retaining known interference in its upper bound.

A saved counterexample showed that the first prototype accepted a backdated assay
copy after observing later clock input. Live updates and new assay copies now honor
both bank and acquisition observation cuts. The pre-target recognition copy retains
only prior eligible episodes and advances elapsed time through the actual query;
query content cannot reinforce or interfere with its own scoring history.

The new component passes 27 checks, bringing the numerical total to 246. Independent
uncompressed sample unions cover 75 streams, exact one-second sums cover 50 streams,
and scalar recurrence updates cover another 50. A 162-condition closed-form grid
combines exposure count, strength cap, recurrence support, missing duration and rate
envelope; maximum absolute error is 2.78e-17. This is numerical agreement, not fitted
memory parameters, human recognition or the O12 predictive-stability gate. The full
Python suite passes 686 checks; existing Rust regression passes 746 with 12 ignored.

Fixed metadata, staging scratch, gap clock and rate buffers total 446,720 bytes per
bus, of which 65,536 metadata bytes already belong to the full bank budget. An assay
copy adds 6,176 bytes. These counts exclude objects, temporary exports, descriptors,
edges and transport. Nine capacity stresses compare 72/144/288 rate rows: overflow
is explicit, and 144 rows do not establish maximum-workload support. The current
Python retention loop also repeats finite competing-support and similarity work;
cached batch updates remain necessary for O04. Complete bank/edge/handle lifetime,
bounded endpoints, actual waveform inference, fitted scales and human gates remain
open. Records are in `docs/roadmap/temporal-dcc/memory-reference.json`; the 40 ms
worker-budget failure has not been resolved. Harmony and production audio are unchanged.

The next M0 container owns episode handles, frozen descriptors, anchors and a
shared-cap edge store. Reservations are bounded and are not heard episodes;
cancelled or evicted handles never return through slot reuse. Recurrence updates
strength and membership while preserving the first admitted ordered descriptor.
Context and supplied correspondence edges share 16 slots. Dropped membership
keeps its denominator, and eviction leaves incoming links explicitly unretrieved.
Splitting or pruning a context path therefore does not assign its past to another
surviving path. A later recurrence can add new support without recovering lost totals.

A saved counterexample exposed another delivery boundary: actual bank receipt at
3.0 s was reported as the ledger's 0.6 s delivery and accepted for a 0.7 s relation
deadline. Bank availability now uses actual receipt, retaining ledger delivery
separately. Ordered links require a retained earlier target and original source/
target times. Eighteen checks cover these contracts; 30 independent membership
streams agree with full sealed joint sums. A capacity audit fills all 256 slots
with 128-knot/32-anchor descriptors, evicts the representative, marks 255 incoming
links unretrieved and reuses its slot for a fresh handle. Full compressed moments
and anchor members survive storage exactly. These synthetic inputs do not establish
long-return recognition or an inferred relation graph.

The implemented default bank is 11,862,016 bytes including retention metadata.
Edge scratch, scales/reservations and additional retention/clock state bring the
fixed partial subsystem to 12,769,616 bytes per bus at the default acquisition
layout. Objects, temporary exports, inference, full endpoint ledger and transport
remain extra. Numerical checks total 264; the full Python suite passes 704 and
Rust regression passes 746 with 12 ignored. Records are in
`docs/roadmap/temporal-dcc/bank-reference.json`. Real relation/ancestry producers,
packed-index worker integration, capacity sensitivity, all fitting and O04/human
gates remain open; the earlier 40 ms budget failure is unchanged.

The pending-endpoint reference now stores actual 128-byte headers with a fixed
deadline heap, free list and two ID indexes. It distinguishes the original
commitment deadline from delayed delivery, freezes one claimed header until
acknowledgment, and rejects stale tickets after slot reuse. Capacity overflow is
unsealed computational loss. Provisional revisions return replaced activity/joint
references without changing the original span or deadline. Their immutable/versioned
payload storage and original-ID production are separate components; the byte-pool
and ownership adapter below now cover storage while inferred producers remain open.

Thirteen checks include 60 independent sorted-order streams with ties and deliberate
hash collisions, tombstone churn, replay, revision and finite-ledger comparisons.
A full-capacity audit retains 65,536 headers, rejects an extra ending, and drains
six 10,240-commit cycles plus one 4,096-commit cycle, preserving original order.
Successful acknowledgments are supplied by this synthetic test; it is not evidence
of actual inferred writes. The fixed buffers use 8,388,608 header bytes, 262,144
each for heap/free arrays and 1,048,576 for indexes: 9,961,472 bytes per bus.
Referenced payloads, claims, receipt state and transport remain additional.

The header record is `docs/roadmap/temporal-dcc/endpoint-reference.json`.

The immutable payload pool now uses fixed 320-byte blocks, 24-byte object metadata,
versioned handles and typed owner counts. A queue entry retains its own four
references; aliases add no owners. Revisions acquire new versions before releasing
old ones. Failed bundle allocation rolls back partial ownership, active claims
keep their bytes alive, and retirement releases pending/active owners as explicit
unsealed loss. A monotone paired ID issuer is provided, but inference must still
identify original spans and propagate their aliases. No byte-pool operation proves
that a real bank/section consumer has committed once.

The initial pool uses 87,818,240 fixed bytes per bus, or 97,779,712 including the
endpoint queue. Other workers, caller inputs, detached copies and object/allocator
overhead are additional. The existing 85,852,160-byte active-descriptor estimate
counts both buses together. Copying 128 knots for every pending endpoint instead
would use 2,684,354,560 bytes per bus for descriptors alone. Sharing in this pool
is explicit reuse of a complete immutable object, not automatic interning or
structural sharing of interior knots.

Sixteen additional checks cover 50 independent finite byte/owner streams, failures,
epoch retirement and actual packed descriptor/SectionRecord round trips. A shared
fixture holds all 65,536 headers with four objects and 132 blocks, then releases all
owners. A distinct-copy fixture exhausts half/base/double block capacities at
992/1,985/3,971 endpoints. Thus bounded storage exposes a sizing failure: actual
retained inference, structural sharing and/or revised capacities must resolve it
before full O04 validation. It does not justify losing required musical histories.
The audit uses synthetic bytes and supplied acknowledgments, not inferred writes.

After the byte-pool step, numerical checks totaled 293; Python passed 733 checks and Rust regression passed
746 with 12 ignored. The new record is `docs/roadmap/temporal-dcc/payload-reference.json`.
Semantic payload codecs, late receipts, all beam producers, consumer atomicity,
genuine runtime-cycle budgets and complete O04 accounting remain required. The full
ledger, fitting and human gates remain open, including the earlier 40 ms failure.

The next storage step shares individual immutable 320-byte knots. Each descriptor
version holds an ordered u32 reference vector, preserving shifted-but-unchanged
knots across compression. A hash match requires complete byte equality, including
all f64 moments, original times, coverage inputs and generations. Root owners and
leaf owners are separate: retaining one version does not multiply its child
references. Partial insertion rolls back acquired leaves, and final root release
frees only leaves no longer referenced by any version.

A split-pool adapter sends descriptor references to shared storage and the other
three payload kinds to the byte pool. Kind and epoch disambiguate equal integer
handles. The existing queue can retain, read and release these roots while
receiving the original packed descriptor bytes. This does not supply semantic
beam/receipt producers or prove an actual bank/section transaction.

Fifteen added checks cover 50 independent finite byte-union/owner streams, forced
hash collisions, tombstone saturation, rollback, namespace routing and retirement.
Eighty versions produced by the actual moment compressor over 640 assigned raw
records round-trip byte-exactly. This preserves existing numeric results; it adds
no new musical inference or acoustic evidence.

The sizing construction retains 64 versions for each of 1,024 fixed paths. Each
path starts with 128 unique knots and changes six knots per later version. The
half capacity of 131,072 leaves stops at 1,024 versions, the initial 262,144-leaf
capacity stops at 22,869, and the 524,288-leaf comparison
holds all 65,536 with 518,144 distinct knots. Descriptor buffers alone total
128,713,216 and 222,036,480 bytes per bus respectively. Auxiliary pools, queue,
active mutable descriptors, other workers and transient allocation remain extra.
The all-new-knot condition still exhausts the initial capacity at 2,048 versions.
These byte constructions establish conditional storage behavior; real path births,
churn, lineage, complete payloads and simultaneous two-bus performance remain
required before accepting capacities. A fixed-path success does not cover them.

Numerical checks now total 308; Python passes 748 checks and Rust regression passes
746 with 12 ignored. The record is `docs/roadmap/temporal-dcc/shared-descriptor-reference.json`.
Complete producer/consumer integration, O04, fitting and human gates remain open.

The next reference delivers a frozen claimed endpoint to the actual episode bank
and a declared list of section histories. Every projection is validated before
bank effects, and a fixed receipt journal preserves original identities, joint
weights and delivery times. Lost bank responses are reconciled against the
completed-write digest; lost section responses use generation, sequence and
original occurrence times, including spans outside the recent ring. Success
booleans without applied state are rejected. Queue acknowledgment and payload
release wait for all declared targets; the combined summary remains unpublished
until completion. Only a transaction without bank effects may be aborted as
explicit loss. A later write then uses the next committed sequence.

The journal uses33,024 fixed bytes per bus: a256-byte current receipt and1,024
target records of32 bytes. Seventeen checks include30 independent streams of20
writes and interrupted bank/section/queue acknowledgments. An audit with1,024
unequal target weights recovers all three lost responses in67 calls, with at most
16 section applications per call, exactly1,024 applications overall, zero error
against independently weighted statistics and all queue payload owners released.
These checks cover call/receipt boundaries under single ownership; they do not
establish process-crash recovery or arbitrary internal consumer atomicity.

The decoded batch, target map, preview copies and hashing are additional storage
and work. Its Python representation alone takes1,248,569 bytes. The first call
validates every target and the implementation hashes the full batch repeatedly:
the maximum call took48.490351997315884 ms, above the40 ms whole-worker budget.
All67 calls took568.5904265847057 ms. These are single-owner diagnostics, not a
simultaneous-worker preflight. The budget16 currently limits section applications
only. Bounded semantic payload ownership, validation scheduling and copy/hash
costs must be resolved before production adoption. Actual beam/lineage/receipt
generation, complete routing and runtime publication remain unimplemented.

Numerical checks now total325; Python passes765 checks and Rust regression passes
746 with12 ignored. The record is `docs/roadmap/temporal-dcc/consumer-reference.json`.
Previous matcher and payload-capacity failures, full O04, fitting and human gates
remain open. No instrument audio or harmony implementation changes in this step.

The consumer now owns immutable input packets and validates section projections
in slices before any bank effect. A local tagged encoding preserves exact f64
bits, signed/unsigned 64-bit identities, tuple joint keys, list windows and masks.
The builder copies each section row independently and cannot finish an incomplete
declared fanout. Packet fingerprints include the original claim; later producer
or decoded-result mutations cannot change the accepted bytes. Each advance
validates or applies at most 16 section projections, while preserving original
times, weights, the publication barrier and lost-response recovery. Private bank
decodes and lookup sets are released after completion or abort. Input encoding
does not itself infer routing or link the queue's joint/lineage/receipt payloads
to the decoded semantics; that provenance remains required.

The journal still uses 33,024 bytes. Initial input limits are 67,108,864 bank
bytes, 16,384 bytes per section row, 1,024 targets and nesting depth 16. These
limits are not an accepted full-model storage census. The same unequal-target
audit now takes 131 calls: 1,024 validations followed by exactly 1,024 section
applications, with all three lost responses recovered and zero weighted error.
Maximum consumer call time is 7.179675158113241 ms; total call time is
141.72862633131444 ms. Packet construction is measured separately at
34.825715236365795 ms total, including 8.344630943611264 ms finalization. The
packet has 273,968 bank bytes and 1,260,544 section bytes, plus the claim/digests
and additional object, allocator and scratch costs. Call counts do not establish
physical-cycle or audio latency; this fixture has only one bank admission.

The legal maximum of 256 bank admissions, each with 128 knots, preserves every
descriptor byte and original membership through the capsule and actual bank.
However, its bank packet alone occupies 26,982,619 bytes. Encoding takes
477.75619593448937 ms, finalization 1,147.6100960280746 ms, consumer decoding
747.8895240928978 ms and application 652.5812409818172 ms: all exceed the 40 ms
whole-worker budget. Expanded descriptor views and unsplit bank processing must
be replaced, while retaining one publication/receipt boundary. Producer work,
all copies and actual arrival latency belong in the renewed O04 measurement.

Fifteen new packet checks include independent wire vectors, f64 bit preservation,
mutation isolation, incomplete/failed construction and 500 seeded native-record
round trips. Together with the 17 updated consumer checks, numerical tests total
340; Python passes 780 and Rust regression passes 746 with 12 ignored. The new
record is `docs/roadmap/temporal-dcc/consumer-packet-reference.json`. Previous
failures, full waveform/inference/fit integration and human gates remain open.


The next revision accepts compact320-byte knots directly, encodes admission
metadata incrementally and shares exactly equal whole descriptors. Preparation
writes only unused knot/anchor slots and edge scratch; published memory stays
unchanged until one final unit. The default budget is8 bank units per call,
separate from16 section rows, with one phase per call. Abort or late validation
failure releases unpublished work. Final bank availability uses the actual final
admission cut while retaining original ledger delivery and acoustic timestamps.

Two full256x128-knot fixtures use shared or entirely distinct real compressed
descriptors. Header/admission/descriptor bytes total186,311 and10,631,111, before
claim/digests and object/scratch costs. Each uses67 consumer calls and514 bank
units; maxima are16.592381987720728 and13.762363931164145 ms. Separate packet
construction takes6.20/11.25ms. Every byte and original1/256 membership matches,
with final publication and complete queue release. Controlled fixture cuts27.0s
and27.65s are not a measurement of audio arrival latency. The unequal1024-target
audit retains all three lost responses and zero weighted error across132 calls.

A separate maximum header with1024 joint cells and4096 relation proposals still
exceeds the budget: its first bank call reaches57.98464617691934 ms. Header
encoding/finalization/detached decoding take23.47/37.68/34.79ms; these remain
unsplit. The bank-only fixture preserves memberships and the shared16-edge cap,
including1024 dropped edges. It has no section fanout or real inferred routing.
Header/proposal slicing and duplicate decode ownership are the next requirement.
The24 bank,19 consumer and19 packet checks pass, giving352 numerical tests;
Python792 and Rust746 pass with12 ignored. The three updated registrations retain
previous failures and freeze sources/audits under m0-bank-staging-reference-20260911.
Full O04, waveform/inference/fit integration and human gates remain open.


The next revision encodes relation proposals independently and shares one private
header decode between consumer and bank. Joint/proposal validation yields every
64 rows; the default eight bank units allow at most 512 relation decodes per call.
All rows remain required before publication. A native dictionary cannot replace
the packet-bound input owner, and public decoded reads stay detached.

The maximum header fixture now takes 5.531514063477516 ms per call across 43 calls,
with exact agreement against the previous frozen bank implementation's memory,
knots, anchors and edges. An additional finite integration sends two chronological
writes through the actual ledger, bank, 1,024 section histories and owned queue.
The second combines 1,024 joint cells, all 4,096 proposals and all 1,024 recipients.
It recovers bank/section/queue response losses, decodes its header once and every
proposal once, and preserves all weights with no duplicate effects or ownership
leaks. Its maximum call is 8.745113853365183 ms, but total work remains
219.30024586617947 ms across 174 calls. The full 256-by-128-knot fixtures remain
separate and still require about 371–374 ms total despite calls below 12 ms.
These are controlled finite inputs, not inferred waveforms, measured physical
arrival/queue latency or simultaneous-worker O04. Numerical tests total 360;
Python 800 and Rust 746 pass (12 ignored). Sources and four audits are frozen
under m0-header-staging-reference-20260911. Earlier matcher, rate-window, capacity,
full fitting and human gates remain open.

A subsequent host-clock audit releases eight synthetic100ms endings into the
actual delivery path, with a100ms period and40ms work-start window. The maximum
256-episode/1024-joint/4096-proposal/1024-section condition retains every weight
and once-only effect, but accumulates0.635–5.224s of deadline-to-publication delay.
Unpaced delivery still accumulates0.284–1.747s. A diagnostic four-slot queue
explicitly rejects four of eight arrivals; all retained payload owners are released.
The bank-call cut precedes the complete consumer barrier by127–201ms in the paced
maximum, so external publication must use a separately sampled completion cut.
The separate profile points to recursive codec work and repeated section assembly;
the next change must reduce aggregate work while preserving validation and original
support. Traces and sources are frozen under m0-delivery-clock-audit-20260911.
This is a finite100Hz synthetic source measurement, not a48kHz audio/device run,
two simultaneous workers, live action-evidence age or full O04. The mathematical
model, instrument and harmonic path are unchanged; M0 remains in progress.

Typed R1 relation records now use107 bytes each; S1 section records have a534-byte
fixed prefix plus the full bounded correspondence map. u64 identities, f64 values,
presence and validity remain separate, and supplied views are not reconstructed
or repaired. The complete maximum inventory now uses438,272 relation bytes and
694,272 section bytes. A frozen-generic-codec comparison preserves all320
cumulative section buffers and1,184 ring buffers byte-for-byte. Five new packet
checks cover literal layouts, masks, invalid storage and200 seeded round trips;
the full Python evaluation suite passes805 tests, including365 numerical checks.
With the unchanged host-clock arrival driver, maximum paced timed work falls33.2%,
and deadline-to-publication delay falls to0.420–3.234s (unpaced0.182–0.937s).
All original weights and once-only effects survive. The queue still grows, and
full128-knot admissions still require about372ms total. This is a representation
improvement, not full O04 readiness. The next target is repeated section preparation,
preserving intervening observations. Sources and comparisons are frozen under
m0-typed-consumer-codec-20260911; model, instrument and harmonic behavior are unchanged.

Section delivery now retains each prepared sealed update once. Applying it merges
only commit-owned statistics into the current cumulative record, preserving later
acoustic observations, acquisition cuts and accent receipts. Zero-weight targets
cannot reuse stale staging data. The private single-owner format adds 1,408 bytes
per recipient; receipt and preparation buffers total 1,474,816 bytes per bus.
This excludes temporary copies, Python objects and other worker/transport state.
A frozen old-algorithm comparison matches all 1,152 cumulative and 4,426 ring
buffers across 2,304 intervening observations, generation inheritance and accent
replays. Six new regressions bring the Python suite to 811 checks (371 numerical).
The unchanged maximum-load arrival driver measures 23.4% less timed work than the
typed-only revision and 0.323–2.324 s deadline-to-publication delay; unpaced delay
is 0.145–0.559 s. The initial 22.4% improvement run is also preserved. Full
inventories and weights remain, but backlog and work-window overruns persist.
Combined consumer work falls to 95.230 ms; full 256-by-128-knot admissions still
take 371.016/366.019 ms. Sources and evidence: m0-section-preparation-20260911.
Remaining preparation/observation/encoding and bank/ledger work, actual audio,
both workers, fit and human gates still prevent full O04 and M0 completion.
The mathematical model, instrument and harmonic path are unchanged.

The next implementation revision batches SectionRecord's contiguous f64 reads
and writes. Each coordinate still computes old + scale * value in the same order;
shared articulation denominators must have identical rounded bytes, including
signed zero. Invalid input or overflow leaves the full record unchanged. The
640-byte record and 1,474,816-byte consumer staging/receipt layout stay the same.
A frozen scalar implementation agrees on all 12,000 updates and 96,000 read
blocks, including 1,099 rejected cases; the 1,152-write interleaving oracle still
matches every cumulative/ring byte. Three regression tests bring the Python
evaluation suite to 814 tests (374 numerical). The unchanged maximum-load arrival
driver measures another 10.2% timed-work reduction, with 0.312–2.031 s
deadline-to-publication delay; unpaced delay is 0.132–0.435 s. Combined consumer
work is 83.532 ms, while full 256-by-128-knot admissions still take about 371 ms.
The 44.858 ms maximum paced cycle and remaining backlog still fail sustained
delivery. Source and evidence: m0-section-record-blocks-20260911. The design memo
is unchanged; this is numerical implementation equivalence and finite performance
evidence, not full O04, waveform inference, fit or human validation.

Activity assembly is now shared by section observation, prepared commits and
occurrence payload staging. One operation validates 29 logical values before
writing their 23 stored f64 values; cumulative observation explicitly includes
elapsed missing-clock support, while sealed spans keep their original owned
window and membership. Rounded shared denominators, signed zero, invalid values
and overflow retain their checks. Adjacent classification reads its predecessor
statistics once, and an ending read loads its mask once. The stored layout and
model are unchanged; 232-byte expanded/184-byte compact activity temporaries and
Python object costs remain outside the fixed-buffer count.
A frozen fieldwise oracle matches all 4,000 activity outcomes, including 1,263
rejections without mutation, and 768 ledger writes with 1,536 pending/sealed
payload byte comparisons and 2,304 section projections. Earlier scalar and
interleaving oracles still agree. Three tests bring the Python suite to 817
checks (377 numerical); the frozen five-component package passes 144 checks.
The identical maximum-paced arrival driver measures 4.6% less work and
0.305–1.919 s deadline-to-publication delay; unpaced delay is 0.128–0.377 s.
The sparse case uses 0.7% more timed work, so this is not an all-case speedup.
The initial assembly-only 3.1% improvement run is also preserved. Combined
consumer work is 78.317 ms; full admissions still take 370.486/367.011 ms.
Backlog and the 44.669 ms maximum paced cycle remain above the required budget.
Sources and evidence: m0-activity-blocks-20260911. Full O04, actual inferred
payload/routing, waveform/fit and human requirements remain unmet.

Sealed section preparation now combines its 31 edge/transition/pair sums in one
248-byte block, retaining every addition, including zero additions that normalize
signed zero or reveal nonfinite old state. The validated original ending bytes
and mask are reused. A frozen comparison covers 4,000 cases: 2,240 exact complete
preparations and cumulative records, 6,748 exact ring buffers, 1,575 no-effects
and 185 unchanged rejections, including 337 predecessor-only preparations.
The full-admission profile then identified scalar descriptor reads as a major
remaining cost. DescriptorKnot snapshots now read 38 f64 values and two u64 IDs
in one operation. Another 4,000-case comparison preserves every nested float bit,
type, mask and rejection outcome, including 236 zero-duration errors; these
invalid-state stress cases do not become newly admissible input.
Three new tests bring the numerical total to 380 and the Python suite to 820;
176 checks pass from the frozen six-component package. The model, record layouts
and 1,474,816-byte consumer buffers remain unchanged. Additional packed temporaries
and Python containers still require complete O04 allocation accounting.
The unchanged maximum-paced driver measures 5.7% less work than the preceding
activity revision, with 0.302–1.722 s deadline-to-publication delay; unpaced delay
is 0.127–0.348 s. Sparse work is 6.6% lower, but the maximum paced cycle grows
from 44.669 to 46.521 ms. Full256-by128-knot admissions fall from 370.486/367.011
to 262.971/266.644 ms, still above budget. Combined consumer work is 70.530 ms.
The sealed-sums-only version, its 5.8% maximum-work reduction and full-admission
profile are retained separately. These finite measurements establish neither
confidence intervals nor sustainable throughput. Evidence: m0-sealed-sums-20260911.
All remaining inference, waveform, fit, human and full O04 obligations stay open.

Bank admission now retains a readonly view of each immutable packed descriptor,
runs the existing source/order/duration/moment/gap checks, and copies the original
validated 320-byte rows into unpublished free slots. It avoids reconstructing
and re-exporting views that came from those same immutable bytes. Expanded native
input still reconstructs every knot and checks all derived views for consistency.
A frozen-bank comparison covers 720 transactions over 1/2/8/128 real compressed
knots: 240 staged packets and 480 mixed native/packed transactions in both orders.
All 1,775 intermediate/final states agree. The 296 successful outcomes preserve
every buffer and receipt; 424 rejections preserve existing memory and reservations
and clear prepared free slots. Hidden nonfinite moments, full-width IDs, source
and final-memory failures are included. Three new tests bring the numerical total
to 383 and the Python suite to 823; 179 frozen component checks pass.
Full256-by128-knot admissions decrease from 262.971/266.644 to 154.701/152.413 ms.
The unchanged maximum arrival case, mostly using one-knot descriptors, measures
0.2% more work and 0.302–1.729 s deadline-to-publication delay, with a 47.354 ms
maximum paced cycle. Unpaced delay is 0.122–0.310 s; sparse work falls 2.8%.
Combined consumer work is 70.600 ms. These finite results improve full-descriptor
admission, not sustained maximum delivery. The model and fixed layouts remain
unchanged; all O04 and other M0 requirements still apply. The initial oracle-fixture
error and its correction to actual compressed-knot counts are also retained.
Evidence: m0-packed-bank-admission-20260911.

Section projection now uses binary lookup on the already sorted, deduplicated
paths owned by each sealed write. It adds no persistent index and preserves
original membership, cuts, cached correspondence, and detached output copies.
An exhaustive-before-image comparison covered 6,882 probes across 36 streams
with 0/1/2/8/64/1,024 paths: 5,658 successful S1 packets were byte-identical,
1,224 failures matched, and all 72 before/after sealed states were unchanged.
Pending revisions, post-seal reinterpretation, sparse full-width IDs, missing
correspondence, and mutation of external views were included. The regression
totals are 386 numerical tests, 826 Python tests, and 109 frozen component tests.
Under the unchanged maximum arrival driver, measured work fell from 1,010.042 to
923.122 ms (8.6%); deadline-to-publication delay was 0.232–1.538 s and the maximum
cycle was 43.721 ms. Sparse work increased 2.9%; capacity-four and unpaced work
fell 8.2% and 7.5%. These are finite measurements, not confidence intervals or
sustained-throughput acceptance. The previous full-bank/header diagnostics were
retained rather than rerun; their drivers have no section-projection recipients.
Model, fixed layouts and the 1,474,816-byte consumer buffers per bus are unchanged.
The 40 ms window, complete kernels, allocation/device workloads, real audio and
other M0 requirements remain unmet. Evidence: m0-section-path-search-20260911.

The generic consumer codec now shares spellings for 71 existing schema keys and
reuses literal/scalar decoding tables. The format and all capacity, depth, type,
UTF-8 and map-key checks remain unchanged. A frozen-before-image comparison
covered 9,774 encodes and 13,312 decodes: 6,724 encoded byte strings and 4,573
decoded values matched exactly, while 3,050/8,739 rejections also matched. The
suite totals are 388 numerical, 828 Python and 84 frozen component tests.
The shared encoded/UTF-8 spellings occupy 1,204/849 literal bytes per module;
Python table, string, object and compiled-format overhead remains additional.
Fixed packet and 1,474,816-byte consumer buffers per bus are unchanged.
The selected finite maximum-arrival measurement used 897.324 ms of timed work,
2.8% below the preceding version, with 0.227–1.509 s deadline-to-publication
delay. Its maximum cycle increased to 47.001 ms, so the 40 ms gate remains unmet.
Sparse, capacity-four and unpaced work fell 9.8%, 4.2% and 3.6%, respectively;
full-bank admissions were 152.868/152.954 ms. All results remain finite diagnostics.
An instrumented profile initially suggested a decoder regression. Removing the
decoder changes was tested and retained as a separate alternative. Fifteen
randomly rotated paired samples without profiling instead favored the original
reuse variant: maximum-header encode medians were 4.964/3.572 ms and decode
medians 7.174/6.277 ms for the preceding/selected implementations. The selected
source and delivery measurement are exactly the already-tested initial variant;
both candidates, the paired evidence and an invocation-path error are retained.
These changes do not establish semantic payload provenance, full O04, fits or
human gates. Evidence: m0-schema-key-codec-20260911.

A standalone Rust experiment now computes original-time band search, coordinate
residuals and the two-row DTW recurrence. It does not replace the registered
Python matcher or the instrument. Nine additional tests cover 700 randomized
inputs, all 1,024 masks, full capacities, scratch reuse and boundary cases;
the existing 33 matcher tests also pass through native DTW, including independent
short-path enumeration. All 837 Python tests pass in 31.111 s, and Rust remains
746 passed / 12 ignored. Nine randomly ordered paired full-query comparisons
preserve 256 episodes of 128 knots, a 128-knot cue, 16 candidates, 8,192 anchors
and 62 transforms. Python/native-adapter medians are 269.573/152.566 ms
with one observed coordinate and 426.318/189.972 ms with ten, reductions
of 43.4%/55.4%. Every output value, path and diagnostic agrees,
including binary64 bits. Both queries still exceed 40 ms. These timings include
Python validation, packing, traceback and diagnostics; prepared native-call
timings are a separate local measurement. Fixed ctypes payloads occupy 41,624
bytes per instance, plus 2,064 bytes for the two DP arrays. Python objects,
temporary results, allocator/ABI stack overhead, RSS and simultaneous real-audio
workloads remain unmeasured. The registered 388 numerical tests, feature version
29 and design memo remain unchanged. The initial measurement and a corrected
test-side expectation for an all-missing path are retained. Full M0 remains open.
Evidence: m0-matcher-kernel-20260911.

The native experiment now batches coarse anchor search and packs descriptor
rows in blocks. All 8,192 anchors retain original-time intervals, masks, ordered
f64 sums, median comparison counts, transformations and ties. Detailed transform
diagnostics use the Python reference at each episode's winning anchor. Sixteen
experiment tests include every anchor in 500 randomized pairs and 90 whole
queries; the existing 33 matcher tests also pass through the native driver.
All 844 Python tests pass in 29.404 s; Rust reports 746 passed / 12 ignored.
Nine randomly ordered three-version comparisons preserve 256 episodes of 128
knots, 128 cue knots, 16 candidates and 62 transforms. Python/DTW-only/new-anchor
whole-query medians are 270.968/147.211/74.521 ms for one coordinate
and 425.345/184.874/94.553 ms for ten. The new version reduces
DTW-only time by 49.4%/48.9%, with matching types, binary64 bits,
paths and diagnostics across all 54 comparisons; both still exceed 40 ms.
Fixed ctypes payloads occupy 53,008 bytes, the two DP arrays 2,064 bytes and the
anchor sample array 128 bytes. Python objects, temporary packing lists, outputs,
allocator/ABI stack overhead and RSS remain additional. Two counterexamples
showed that the initial candidate hid an unused anchor's RMS overflow and
rejected NaN diagnostics produced by the reference under extreme time differences.
Extreme arithmetic now takes the exact reference path. Its guard derives from
the f64 maximum and eight samples, not a cognitive threshold. Both failures and
an initial audit fixture that failed to distinguish one of them are retained.
Reproducing a NaN diagnostic does not adopt that input as production evidence.
The registered Python model, feature version 29, memo and instrument are unchanged.
Input validation/transfer, simultaneous real-device workloads, fit and human
gates remain open. Evidence: m0-native-anchor-20260911.

Native input validation and optional CPython transfer now preserve the original
observation cuts and rejection behavior. Exact builtin floats/lists/dicts transfer
through GIL-held PyDLL into fixed storage; other types retain the Python path.
The default numeric library has no Python dependency. Twenty-six experiment tests
include 1,200 validation cases, 320 byte comparisons, fallbacks, object release
and untouched buffer tails. Existing 33 matcher tests and all 854 Python
tests (32.151 s) pass; Rust reports 746 passed / 12 ignored. Nine four-version
comparisons retain 256 episodes x 128 knots, 128 cue knots and 16 candidates.
Whole-query medians change from 73.531 to 23.433 ms for one coordinate
and 95.701 to 37.477 ms for ten, with exact types, binary64 bits,
paths and diagnostics in 72 comparisons. These single-query medians below 40 ms
do not establish p99 under two workers and 64 Voices. Fixed ctypes payload is
67,344 bytes; DP arrays, anchor samples, validation metadata and transfer pointer
arrays are separate inventories of 2,064, 128, 56 and 144 bytes. Python objects,
outputs, allocator overhead, total stack and RSS remain outside these figures.
The registered model, feature version 29, memo and instrument are unchanged.
Consumer throughput, actual producers, simultaneous workloads, fit and human
gates remain open. Evidence: m0-native-input-validation-20260911.

A subsequent caller-tail diagnostic uses 10 warmup and 200 timed calls per
caller. One-coordinate p99 is 23.460 ms for one caller and 57.322 ms for two;
ten-coordinate p99 is 49.625 ms and 87.469 ms. Two Python threads use separate
native scratch and share the GIL; all 200 pairs overlap. Exact output checks pass
for 1,260 calls and run outside timed intervals after a barrier. This diagnostic
still exceeds 40 ms. Next, move remaining traceback and diagnostic computation
and construction out of Python loops and remeasure concurrent callers. It is
not the real two-worker/64-Voice/6,000-cycle O04 workload.

Native traceback now preserves reverse-order coordinate accumulation and
computes adjacent-pair motion/interval errors. Python retains math.fsum and final
result construction. Tests cover pow-versus-multiplication rounding, 320 wide
magnitudes, overflow, lazy/custom gap evaluation and reuse after fallback. An
initial Config layout mismatch was corrected; all five ABI structure sizes are
checked before buffer use. The initial source and failure log remain archived.
All 30 experiment tests, existing 33 matcher tests and 858 Python tests (32.917 s)
pass; Rust reports 746 passed / 12 ignored. Nine three-version comparisons retain
the full query inventory. Input-transfer/new-traceback medians are
23.744/16.484 ms for one coordinate and 38.226/21.222 ms for ten,
with exact types, binary64 bits, paths and diagnostics in 54 comparisons.
Single/two-caller p99 is 17.303/42.039 ms for one coordinate and
32.859/50.261 ms for ten. All 1,260 outputs agree, but concurrent callers
still exceed 40 ms. Fixed ctypes payload is 73,152 bytes, including 1,536 bytes
for paths, 2,048 for diagnostic arrays and 120 for coordinate sums/counts.
Python objects, outputs, allocator/stack/RSS and full O04 remain additional.
The registered model, feature version 29, memo and instrument are unchanged.
Next: winning-anchor diagnostics. Actual producers, consumer throughput, fit and
human gates remain open. Evidence: m0-native-traceback-20260911.

Native coarse search now selects the winning anchor, aggregates comparison and
boundary counts, and retains only its residual samples in original order. Pow
rounding and Python math.fsum are preserved. All 8,192 anchors, 16 candidates,
62 transformations and complete diagnostics remain. The 500-pair anchor audit
now includes winner selection, totals and RMS; tests cover ties, missingness,
reuse and sample order. All 32 experiment tests, existing 33 matcher tests and
860 Python tests (32.544 s) pass; Rust reports 746 passed / 12 ignored.
Nine three-version comparisons give previous/new full-query medians of
15.650/13.588 ms for one coordinate and 20.710/18.567 ms for ten.
All 54 comparisons retain types, binary64 bits, paths and diagnostics.
Single/two-caller p99 is 13.594/28.807 ms for one coordinate and
30.016/44.354 ms for ten. All 1,260 outputs agree. Two one-coordinate callers
meet 40 ms in this diagnostic, but the ten-coordinate condition still exceeds it.
The added coarse-call argument requires ABI v2; both mismatched old/new pairings
are rejected before buffer calls. Fixed ctypes payload is 73,296 bytes, including
144 bytes for the winning diagnostic. Original/winning/sort sample arrays total
320 bytes; Python objects, outputs, allocator, total stack and RSS are additional.
The registered model, feature version 29, memo and instrument are unchanged.
Next: query-scoped transfer and call batching with bounded ownership, original
cuts, rejection order and measured copies. Actual producers, consumer throughput,
full O04, fit and human gates remain open. Evidence: m0-native-anchor-diagnostics-20260911.

Query-scoped owned buffers replace repeated descriptor transfers. Three native
calls cover transfer, all-episode coarse search, and refinement/DTW. Full-query
medians improve from 13.811 to 11.355 ms (one coordinate) and
18.482 to 15.380 ms (ten). Two-caller p99 is 30.454/39.974 ms;
the ten-coordinate assay still has four of 400 calls above 40 ms, maximum 44.548 ms.
Fixed ctypes storage is 6,997,808 bytes per caller and full-query numeric transfer
writes 5,526,528 bytes. All 54 query, 1,260 caller-tail and 42 resource-audit outputs
agree, including original identities, generations, cuts, scales and diagnostics.
All 38 native, 33 original matcher, 866 Python and 746 Rust tests pass (12 Rust ignored).
GIL-held transfer takes 6.286/7.371 ms in separate stage measurements;
owned packed input is the next target. Evidence: m0-native-query-batch-20260911.
The registered model, feature version 29, design memo and instrument are unchanged.
Actual producers, consumer throughput, full O04, fit and human gates remain open.

Owned immutable 320-byte descriptors now feed native matching without dictionary
expansion. Full-query medians change from 11.416/15.158 ms for the object-batched
path to 5.042/7.221 ms for packed input; copying all blobs each time gives
5.718/7.831 ms. Across eight copy/no-copy, one/two-caller and one/ten-coordinate
conditions, all 2,400 timed calls stay below 40 ms and all 2,520 outputs agree.
Two-caller p99 is 21.580/22.513 ms without copies and 8.944/11.087 ms with copies;
these differently ordered conditions do not establish that copying reduces tails.
Fixed ctypes storage is 7,001,928 bytes per caller. Original blobs occupy
10,526,720 bytes, copied again in the copy condition. Projection writes
5,526,528 numeric bytes. Borrowed addresses are invalidated before owners can die.
All 43 native, 33 original matcher, 871 Python and 746 Rust tests pass (12 Rust ignored).
All 72 query and 84 resource-audit outputs agree. Evidence: m0-native-packed-input-20260911.
Next: connect packed exports to scheduler dispatch, native matching and receipts.
The registered reference, feature version 29, memo and instrument are unchanged;
actual producers, consumer throughput, full O04, fit and human gates remain open.

#### Model evidence and replacement boundaries (2026-09-11)

Evidence for a perceived relation, a neural mechanism, a selected estimator and
its effect on generation are separate claims. Even R/H contain model choices:
[Marjieh et al. (2024)](https://www.nature.com/articles/s41467-024-45812-z)
demonstrate timbre-dependent consonance judgments and compare revised models.
For timing, [Damsma et al. (2025)](https://academic.oup.com/cercor/article/35/9/bhaf258/8263582)
show that both oscillator and evoked-response models reproduce selective beat
enhancement. These findings motivate replaceable estimators; they do not select
Conchordal's DTW, semi-Markov state, graph or ordinal heads as neural laws.

The M0 replacement record, `docs/roadmap/temporal-dcc/model-replacement.md`, and
memo §8.1 now define stable observation/support/time/output semantics, model-owned
representations and exchange units. Runtime and Voice consumers must not depend
on ten-coordinate knots, packed layouts, DTW paths or beams. Coupled memory,
matcher and decoder may be replaced together, and the joint inference scheme is
also a candidate. Preserve cross-scale interaction with the specified causal
ownership; an unchanged output type alone does not justify reusing a fitted scale.

O03/O09/O12 record dependency-based feature replay, affected fits and calibration,
fresh state at run start, incompatible-version rejection and resource remeasurement.
MR1 is a small ordered/orderless exchange before production wiring; MR2 compares
timing models in M2; MR3 checks version/state isolation at the first M1–M2 boundary.
All three are registered and unrun. A weaker cognitive control can test the
exchange boundary without being promoted. Python/Rust numerical equivalence is
an implementation substitution and does not satisfy MR1.

This architecture revision changes no reference equation, feature layout or
capacity and keeps feature version 29. Instrument code and existing harmonic
kernels are unchanged. Prior tests and external reviews retain their saved-version
scope; the in-progress packed worker integration is still unverified. Neither the
replacement experiments nor full M0, fitting or human gates have passed.

## 9.4 Alignment and Extension Sequence

The overall roadmap remains `docs/roadmap/manifesto-alignment-and-beta.md`; the
complete temporal-DCC delivery plan is `docs/roadmap/temporal-dcc-completion.md`.
With the §9.3.55 design review complete, M0 baseline capture and public-claim
reconciliation have started. T1–T7 tasks, controls, the feature manifest, data
splits, acceptance criteria, numerical preflight and synthetic gates remain M0
work, preserving
shared local audio under changed prior context. M1 then wires independent passive
relation instances to the existing ordered analysis paths. The following records existing implementation and audition
outcomes; it does not restrict the next work to short-time prediction. Current-version,
seeded comparisons separate judgments of audible pulse, tension/resolution and form.
Measure-accent production is implemented with a default depth of zero.
The author's preference for B is recorded against seed 1, as requested immediately before the reply;
the reply itself did not specify a seed. B in that pair has no added measure accent, so Sample 08
and the zero default are retained. Audible grouping and preferences for the other seeds remain unconfirmed.
Same-region habituation recovery is measured, while autonomous closure stays
PARTIAL; the practical DCC comparison does not establish a beneficial nonzero gain.
Results and decision boundaries are recorded in `docs/roadmap/stage2-evidence-2026-09-05.md`.
The corrected coupler uses
`tension_pressure = tension_level * coupling_strength`: resolvability is already present
in listener tension and is not applied twice.

Short-term recurrence, band-energy expectation and experienced context memory are
partly connected to the habitat side. ListenerTwin observes presentation audio.
Boundary/closure, persistent stream identity and action-value learning remain open.
Scenario remains the artist's macro direction; phrase and long-form models need their own representations, not merely
slower meter oscillators. Personalization, real biosignal input and long-form memory each
require their own comparison and acceptance conditions. These extensions do not change
the ledger to “implemented” until the corresponding mechanism and evidence exist.

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
