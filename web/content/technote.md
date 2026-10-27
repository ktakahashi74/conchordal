+++
title = "Technical Note: The Physics of Conchordal"
description = "A deep dive into the psychoacoustic algorithms, logarithmic signal processing, and artificial life strategies powering the Conchordal ecosystem."
template = "page.html"
[extra]
source_commit = "ec3c1327"
author = "Koichi Takahashi"
last_updated = "2026-10-25"
source_version = "0.5.0"
source_snapshot = "ec3c1327"
revision_scope = "Flora self-exclusion, beat opportunities and pace; authored rest and dwell; replacement release and conditional NSGT (2026-10-25)"
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

The bin geometry of `Log2Space` has a Constant-Q (Constant Quality Factor) characteristic across the spectrum. In signal processing terms, $Q$ is defined as the ratio of the center frequency to the bandwidth: $Q = f / \Delta f$.

In a linear system (like a standard FFT), $\Delta f$ is constant, meaning $Q$ increases with frequency. In `Log2Space`, the bandwidth $\Delta f_i$ of the $i$-th bin scales proportionally with the center frequency $f_i$. This property mimics the frequency selectivity of the human auditory system, where the ear's ability to resolve frequencies diminishes (in absolute Hz terms) as frequency increases. This describes the output grid and the retained NSGT kernels. The auditory front end uses ERB-dependent filter bandwidths (Section 3.1); output-bin spacing does not set its frequency selectivity.

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

Conchordal adopts this account literally. The system runs a cochlear front end on its own sound, computes a Roughness field and a Harmonicity field over the entire frequency axis, and integrates them into Consonance—a terrain whose peaks are where a new tone would *fuse* and whose valleys are where it would *grate*. Nothing in the system knows an interval name or a ratio; agents feel only this terrain. Section 4 applies the same move to time: just as harmony is computed from a model of the cochlea rather than imposed, meter is computed from a model of beat perception rather than scheduled.

The "Landscape" is the central data structure in Conchordal. It acts as the shared environment for all agents, a dynamic scalar field representing the psychoacoustic "potential" of every frequency bin. Agents do not interact directly with each other; they interact with the Landscape, which aggregates the spectral energy of the entire population. This decouples the complexity of the simulation from the number of agents ($O(N)$ vs $O(N^2)$).

The Landscape is updated every audio frame (or block) by the Analysis Worker. It synthesizes two primary metrics:

*   **Roughness ($R$)**: The sensory dissonance caused by rapid beating between proximal partials.
*   **Harmonicity ($H$)**: The measure of virtual pitch strength and spectral periodicity.

Both metrics are normalized to the $[0, 1]$ range. The rest of this chapter follows the analysis pipeline in order: Section 3.1 builds the auditory subjective-intensity density, Sections 3.2 and 3.3 derive the Roughness and Harmonicity fields from it, and Section 3.4 integrates the two into the Consonance terrain that agents actually climb.

## 3.1 Causal Auditory Analysis

`AuditoryFrontEnd` (`core/auditory_frontend.rs`) converts real habitat PCM into a `Log2Space`-aligned subjective-intensity density. The same processing analyzes presentation audio for ListenerTwin and the actual `mix − own` PCM for Voice self-exclusion. It uses a causal complex filterbank, hop statistics, frequency reassignment and compressive loudness weighting.

### 3.1.1 Filterbank and Hop Statistics

Each channel is a cascade of four complex one-pole filters, with positive pole angle and unity gain at its center. Its bandwidth is $b_c=\tfrac14\,1.0186(24.7+f_c/9.265)$ Hz. Centers lie on the LTFAT ERB-number scale $e(f)=9.2645\ln(1+0.00437f)$ at intervals of 0.1875, anchored at 1000 Hz. The analysis range runs from $55\cdot2^{-7/8}\approx29.989$ Hz (84 bins below 55 Hz on the 96-bin-per-octave grid, near the roughly 30 Hz lower limit of melodic pitch for harmonic complexes) to 8000 Hz, with a margin of 0.375 ERB-number units on either side: 174 channels, from about 24.629 to 8175.326 Hz. All centers must lie below Nyquist; unsupported sample rates fail configuration validation.

For each complete hop of $N$ samples, a channel accumulates energy $E_h=\sum|z[n]|^2$ and lag-one correlation $S_h=\sum z[n]\overline{z[n-1]}$. The first sample uses the previous hop's final output. A channel keeps a boxcar of $n_c=\max(1,\lceil f_s/(N\,0.75\,\mathrm{ERB}(f_c))\rceil)$ hops. Unobserved slots start at zero and the denominator stays fixed while the boxcar fills.

The estimated frequency and reassigned real mean-square power are

$$\hat f_c=\frac{f_s}{2\pi}\arg\sum_h S_h,\qquad d_c=\frac{2|\sum_h S_h|}{n_cN\,G(\hat f_c)},\qquad G(f)=\sum_j|H_j(f)|^2.$$

The coverage function $G$ is tabulated at construction and interpolated. Frequency and power come from the same complex correlation, so the deposited power vanishes as that correlation approaches zero.

### 3.1.2 Critical-Band Weighting and Density

Before compression, reassigned powers form a critical-band context

$$P_{cb}(x)=\sum_j d_j\left[1+\left(\frac{x_j-x}{1.0186\,\mathrm{ERB}(x)}\right)^2\right]^{-4}.$$

The implementation uses sixth-order moments in 1/32-ERB-number buckets within ±4 ERB-number units of each query. For each deposit, it takes $P=\max(P_{cb},d)$ and applies the continuous gate $\gamma=\max(0,1-10^{-6}/(P/du))$, where $du$ is the interpolated Log2Space cell width in ERB units. A zero context produces zero weight. The compressed mass is

$$m=\gamma\frac{d}{P}\left(\frac{A_w(x)P}{P_{ref}}\right)^{\texttt{loudness\_exp}},$$

with A-weighting power gain $A_w$ and default exponent 0.23. Each mass is shared linearly between adjacent log-frequency bins and divided by their cell widths. Outside the end-bin centers it fades to zero over one bin; nonpositive frequencies contribute zero. Out-of-range deposits still contribute to the critical-band context.

S6 leaky smoothing uses `analysis.tau_ms` (default 10 ms). The resulting `subjective_intensity` density feeds both H and R. Its integral remains `loudness_mass`; `roughness` and `roughness01_scalar` retain their existing formulas and types. Their values change with the new input density.

ListenerTwin groups positive bin masses by ascent to local mass maxima before
weighting stability and reachable improvement. Each component contributes its
total mass at its peak-bin field level. Improvement candidates must lie outside
that component, with movement cost measured from the peak. This makes one
component shared between two bins contribute once. Gain scale, reachable range,
movement cost and the product defining tension retain their existing values.

### 3.1.3 Spectral Power, History and Timing

Mixed and listener streams calculate NSGT power and spectral history only when a display, report or temporal observation tap needs them; Flora placements also enable the mixed stream's resource analysis. Without these consumers, NSGT does not advance. Each Flora retains a separate FFT/NSGT-weighted resource observer. Voice and Flora H/R self-exclusion observers calculate the auditory density without running NSGT. `analysis.nfft` controls the retained NSGT window, Flora resource coverage and the publication hold after gaps; `analysis.kernel_align` controls NSGT kernel alignment. `psychoacoustics.use_incoherent_power` selects the NSGT power measure, independently of H/R input.

The audio worker at hop K still reads analysis frame K−1, stamped at that frame's end. Before a run, history is silence. A gap resets filter, previous sample, statistics and density state, and publication waits for the existing `window_samples()` coverage (16384 samples, 341.333 ms at 48 kHz by default). This publication rule is not an accuracy bound on the IIR tail. Clones retain the full causal history. Synchronous body analysis advances filters and hop statistics even on unselected hops; selected updates apply S4–S6 once, with the elapsed-hop interval for S6.

## 3.2 Roughness ($R$) Calculation: The Plomp-Levelt Model

Roughness is the sensation of "harshness" or "buzzing" caused by the interference of spectral components that fall within the same critical band but are not sufficiently close to be perceived as a single tone (beating). Conchordal implements a variation of the Plomp-Levelt model via convolution in the ERB domain.

### 3.2.1 The Interference Kernel

The core of the calculation is the Roughness Kernel, defined in `core/roughness_kernel.rs`. This kernel $K_{rough}(\Delta z)$ models the interference curve between two partials separated by $\Delta z$ ERB-rate units. The default Sethares core peaks near a quarter ERB-rate unit; the reference-normalization stimulus uses a separation of 0.25 ERB.

The implementation uses a Sethares difference-of-exponentials core, a small directional masking asymmetry, a center-suppression dip, and an optional neural Gaussian component. With $u=|\Delta z|/\kappa$:

$$ S(\Delta z) = G\,\max(0, e^{-bu} - e^{-cu})\,A(\Delta z) $$

$$
M(\Delta z) = \left(1-e^{-\Delta z^2/(2\sigma_{sup}^2)}\right)^p, \qquad
N(\Delta z)=e^{-\Delta z^2/(2\sigma_n^2)}
$$

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

Larger $k$ reduces $R_{01}$ for the same input ratio. $k$ belongs to the sensation model and is set only in the configuration; how much roughness is minded is a valuation, changed from scenarios through `set_roughness_aversion` (Section 3.4).

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
    *   300 Hz lies a perfect fifth above the 200 Hz tone (3:2).
    *   500 Hz lies a major tenth above it (5:2): an octave plus a major third.

No interval name appears in the code. The peaks follow from H's integer-ratio templates, a model of periodicity-based fusion; the integer prior is built into the kernel, not discovered by it. An agent at 200 Hz creates a "gravity well" at 300 Hz and 500 Hz, inviting other agents to form a major triad in open position (2:3:5).

## 3.4 Consonance: Integrating the Fields

With $R_{01}$ and $H_{01}$ in hand, Consonance is derived in two layers: a **Consonance Kernel** that fuses the two observables into a single fitness score, and a set of **representation transforms** that reshape that score for its different consumers in the Life Engine (Section 5).

**Layer 1 — Consonance Kernel (bilinear family):**

$$ C_{score} = a \cdot H_{01} + b \cdot R_{01} + c \cdot H_{01} R_{01} + d $$

Default coefficients: $a = 1.0$, $b = -1.35$, $c = 1.0$, $d = 0.0$. Because $b < 0$, roughness acts as a penalty; because $c > 0$, high harmonicity attenuates that penalty (the interaction term $c \cdot H_{01} R_{01}$ partially cancels $b \cdot R_{01}$ when $H_{01}$ is large). The bilinear family subsumes the earlier $\alpha H - wR$ formulation as the special case $c = 0$.

The director's `set_roughness_aversion(w)` multiplies the roughness terms $b$ and $c$, and the density $\rho$ below, by $w$ (default 1, which leaves the configured values bit-exact). $w$ is clamped to $[0, 100]$ and non-finite values are ignored. $w = 0$ makes valuation ignore roughness. Larger $w$ penalizes roughness more when the configured kernel penalizes it at every harmonicity ($b \le 0$ and $b + c \le 0$, true for the defaults). $R_{01}$ itself, as sensed and displayed, does not change.

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

Spawn samples a range-local distribution. By default, a `SpawnStrategy::Field` slices the requested frequency range, derives target-specific local mass from the effective views, masks occupied bins, and normalizes only that local vector. A zero-mass range falls back to an unoccupied-uniform distribution and then to full-range uniform if every bin is occupied. Peak placement chooses the target extremum deterministically, or the nearest field-score target when tension is specified. The global $C_{density\_pmf}$ remains a normalized Landscape representation exercised by core tests.

The root configuration key `birth_surrogate = true` enables optional body-density evaluation for Field placement; it is disabled by default. For each candidate fundamental, a direct model of the selected renderer estimates the body's subjective-intensity density. Density placement integrates that density against the effective target mass over ERB cells, divides by the body's in-band mass, and uses the result as the local birth weight. Peak placement integrates the target score instead. All candidates are evaluated within the same hop; equal body recipes reuse their preparation while occupancy and spacing are updated after each birth. Range-local normalization and the zero-mass fallbacks remain in force. Unsupported bodies and recipes use the existing terrain-only placement (the supported conditions are listed in Section 5.1.1).

The birth option is independent of body-aware metabolism (Section 5.2); both can be enabled together. The excitation renderer supports Field placement and frequency proposals for Sine and the supported Harmonic recipes; Modal retains terrain-only placement. With the legacy renderer, the option also affects supported final respawn evaluations. Random retains 16 proposals and resamples their non-negative scene scores. Hereditary selects one parent and keeps the maximum-level choice among 16 proposals. PeakBiased keeps its terrain-peak proposals, parent bias and local search grid, but uses integrated body scores for its final weights and local search when supported. The integrated score is converted to level with the same configured sigmoid as the terrain; minimum-level rejection therefore retains its scale and threshold. Unsupported policy/body combinations use the existing point evaluation. With the excitation renderer, final respawn selection, local search and minimum-level rejection retain point evaluation because native final-selection distributions have not been conformed.

The author adopted a maximum total-variation distance of 0.1 from the rendered-and-analyzed selection distribution for each supported measured body condition. The value is an author acceptance rule chosen to include the measured Sine result (about 0.095); it is not derived from the psychoacoustic model. For the legacy renderer, the bound was checked only at the measured points in saved Consonance/Density fixtures: 104 groups of seven candidates near 440 and 1800 Hz, at 48 kHz with hop 512, in silence and other-only Sine 440 Hz, Sine 660 Hz and Harmonic 330 Hz environments. These fixtures have no occupancy, spacing or tension. For the excitation renderer, the corresponding 64 Sine/Harmonic groups were checked against native candidate PCM, keeping the saved terrain and realized mode ratios fixed. The guarantee is limited to the supported conditions in those fixtures; it does not extend to arbitrary parameters or environments, Peak placement or final respawn distributions. Legacy final respawn selection has the separate measured scope below.

Legacy final respawn evaluation uses the same author rule of total-variation distance at most 0.1. Saved rendered candidate scores were passed through the existing Random and Hereditary selectors, with the seven measured frequencies repeated cyclically to form 16 proposals; the supported measured conditions reached maxima of 0.095963 and 0, respectively. PeakBiased was checked with 16 terrain-derived proposals from its existing generator and the default parent bias and local search (0.50-semitone radius, 0.05-semitone step), around 440 Hz in the other-only Harmonic 330 Hz environment. At 48 kHz with hop 512 and 72-hop body observations, its complete final distribution included 158 rendered candidate frequencies per body condition; the supported conditions reached a maximum of 0.025876. These results cover those saved measurements and that bounded PeakBiased case, not arbitrary parameters, environments or proposal sets. Conditions exceeding the author rule retain point evaluation; the policy-specific supported domains are listed in Section 5.1.1.

There is also a distinct, per-Voice adaptation mechanism. `AdaptationContext` tracks fast boredom and slow familiarity over the shared **fundamental-occupancy** field and adds a candidate-specific score adjustment during pitch selection. It coexists with the Landscape-level habituation below: adaptation is agent-relative memory; habituation is a shared perceptual erosion field.

## 3.5 Landscape-Level Habituation

Consonance is not treated as an inexhaustible place. When `[psychoacoustics.habituation]` is enabled, `HabituationField` maintains a per-bin state $h_i\in[0,1]$ for both the ecology's habitat analysis and the `ListenerTwin`'s presentation analysis. The raw drive combines the base consonance level $L_i$ with a fresh common-root projection $P_i$:

$$ d_i^{raw}=L_iP_i, \qquad d_i=\frac{d_i^{raw}}{d_i^{raw}+r_{ref}} $$

The state relaxes asymmetrically toward that drive:

$$ h_i(t+\Delta t)=a h_i(t)+(1-a)d_i, \qquad a=e^{-\Delta t/\tau} $$

with $\tau=\tau_e$ when the drive exceeds the current state and $\tau=\tau_r$ otherwise. `satiation_sec` and `recovery_sec` are the times to reduce the discrepancy from the respective equilibrium by 90%, so the implementation uses $\tau_e=T_s/\ln 10$ and $\tau_r=T_r/\ln 10$. The sustained-drive equilibrium depends on drive strength; after complete withdrawal it is zero. The effective score relaxes toward the neutral sigmoid threshold $\theta$, while density mass is eroded multiplicatively, as shown in the table above.

The default is `enabled=true`, with `satiation_sec=5.0`, `recovery_sec=8.0`, and `ref_drive=0.25`. Explicit `enabled=false` is an exact identity: every effective view equals its base view bit-for-bit. When enabled, Voice movement, prediction, metabolism, respawn, Voice density placement, and the listener read effective views; Flora growth and seeding, the UI terrain, and diagnostics retain the base views before erosion. Habituation belongs to the sensation and in-performance listener model; Rhai provides no switch for it. Scientific controls use explicit `enabled=false`.

With `--report`, an additional `habituation_scan` record at each generated second carries
`fmin_hz`, `bins_per_octave`, `n_bins`, and the habitat's `state_scan`, `raw_score_scan`,
and `eff_score_scan`. The arrays share fixed Log2Space coordinates and are serialized
from borrowed slices. This permits a fixed region to be followed through withdrawal;
the older per-hop `tracked_bin` remains the moving raw-score maximum.

# 4. The Temporal Axis: The Emergent Meter

Music cognition distinguishes three layers of musical time. **Rhythm** is the surface: the actual pattern of onsets as they occur. **Pulse** (the tactus) is the perceived regular beat a listener taps along to—already an inference, since the surface rarely contains it explicitly. **Meter** is the hierarchical organization of that pulse: nested periodicities (subdivision, beat, measure) with alternating strong and weak positions. Crucially, meter in this sense is a *percept*, not a notation. A time signature is an instruction; meter is what a listener's brain constructs from the sound—and constructs even from sound that was never notated.

This percept has well-studied dynamics. It takes a few cycles of evidence to establish (*beat induction*); once established it persists through syncopation, gaps, and silence, with events heard *against* the induced grid rather than destroying it; and it is plastic, re-locking when the input tempo drifts. Neural resonance theories (Large and colleagues) explain these properties mechanistically: populations of neural oscillators entrain to acoustic onsets, and the self-sustaining dynamics of the oscillation—not the stimulus itself—carry the pulse across interruptions.

Conchordal uses this structure in an engineering mechanism. It eschews the concept of a master clock or metronome; time is structured by an **emergent meter**—a self-sustaining beat oscillator with adaptive integer-ratio layers—that *listens* to the ecosystem's own onsets and forms a metrical percept from them. Voices, in turn, entrain their onset timing to this perceived pulse with per-voice coupling strengths (Section 5.4). Rhythm is therefore a closed perception–action loop: nothing schedules a beat; a beat condenses out of the population's behavior and then attracts it. This is the temporal mirror of the harmonic landscape (Section 3): just as consonance is not imposed but computed from a model of the cochlea, meter is not imposed but computed from a model of beat perception.

## 4.1 The Meter Core: An Adaptive Integer-Ratio Hierarchy

The default **Events** input and the presentation-side ListenerTwin meter use acoustic onset recurrence. An opt-in **Continuous** input for the existing production meter is selected by the top-level `temporal_meter_t2 = true` (default false). It reads raw habitat T2 directly and shares the beat, fast onset detector, PLV, integer layers, shaping and output state with the Events mechanism. It does not add a second production meter.

`MeterNetwork` (`core/meter.rs`) combines a self-sustaining beat oscillator with acoustic recurrence corroboration. Its amplitude retains the forced Hopf equation, while phase advances freely between detected onsets:

$$ \dot r = \alpha r + \beta r^3 + F_a\,g\,s(t)\cos\varphi, \qquad \dot\varphi=\omega $$

Here $g=1+\mathrm{stability}$, $\alpha=1$ and $\beta=-1$. The unforced radius tends to 1, so the beat continues through gaps. Production drive combines the habitat bus's `DorsalStream` spectral flux with habitat-routed phonation onset strengths. Presentation-only Voices do not directly reinforce this meter.

At detected onset $j$, the meter measures phase error $\epsilon_j$ **before** correcting its prediction. It adjusts the current period $T$ using the preceding error and corrects phase using the current error:

$$
T^+=T^-\left(1+\eta_p g\frac{\sin\epsilon_{j-1}}{2\pi}\right), \qquad
   \varphi^+=\varphi^- -\eta_\varphi g\sin\epsilon_j.
$$

The gains $\eta_\varphi=0.6$ and $\eta_p=0.1$ come from the Large & Jones (1999) simulation. The meter retains detected acoustic onset times for $H=2\times2\,\mathrm{s}+60\,\mathrm{ms}=4.06\,\mathrm{s}$, covering two maximum candidate intervals and the confirmation tail. Three possibly nonadjacent onsets corroborate a lag $p$ when successive spacings agree within $w(p)=\min(0.2p,60\,\mathrm{ms})$, reusing the acoustic participation window. Lags are ranked by $R(p)=M(p)/N$, where $M$ is triangular, chronological one-to-one matching mass between the history and its shift by $p$, and $N$ is the full history's event count, common to all candidates. An already accepted lag remains in this comparison. Histories whose adjacent intervals all agree within the existing window retain the previous adjacent-interval confirmation path. From the corroborated rate $f_o=1/p$, the meter considers $f_r=r f_o$ for $r\in\{1,1/2,1/3,1/4,2,3,4\}$ within the continuous beat band, normally 0.5–4 Hz. Ideal nested event trains give support $S_r=\min(r,1/r)$: shared events divided by their union. Thus a faster candidate cannot win solely because its PLV also aligns with every input onset.

Candidates are ranked by $S_r W(f_r)$. The cultural prior is $W=(1-s)+sP$, with $s=\mathrm{clamp}(\texttt{meter\_stability},0,1)$. Here $P$ falls linearly with log-frequency distance from the preferred center to either band edge. The center is the arithmetic mean of an explicit `temporal_basin`, or 2 Hz in the default band. The default depth 0 gives flat weighting within the band; greater depth favors an in-band tactus nearer the center. With an explicit basin, the center restoring term uses the same depth and is reduced by preceding beat confidence. This is an engineering rule based on neural resonance structure, rather than a numerical transplant of a published oscillator network.

The first adoption anchors phase zero to its onset, clears evidence in the old coordinates, and does not count that onset as evidence for itself. Subsequent corroboration sets the resulting tempo without resetting phase; the event correction and remaining hop advance also update the unwrapped beat count. An established beat survives whole missing beats: an interval compatible with two, three or four current periods retains the established tactus. This resolves the ambiguity in favor of persistence and cannot separately identify a sudden integer-factor slowdown from missing onsets.

Subdivision phases are $2\varphi,3\varphi,4\varphi$; measure phases divide the unwrapped beat count by 2, 3 or 4. Each layer retains separate phase-coherence or accent evidence. Faster-than-band input can therefore be read as subdivision of an in-band beat. Independent noninteger tempi and adaptive inter-oscillator coupling are outside this implementation.

**Perception vs production.** The runtime uses this same mechanism in two instances. The production meter listens to the habitat bus and guides Voice behavior. The `ListenerTwin` meter listens to presentation audio and supplies beat confidence to the UI and report. DCC pressure uses presentation-derived consonance tension, rather than beat confidence directly.

### Continuous habitat recurrence

Continuous input retains the native 30×4 signed hop means without group attribution or component weights. It forms positive rises only from two consecutive, known hops with the same filter reset origin. Within the existing 4.06-second history, native-hop lags compare component-wise minimum and maximum masses; their intersection/union ratio proposes recurrence. A flat profile does not induce a beat. Interior strict peaks may use the fixed local quadratic vertex, reevaluated against the actual linearly interpolated overlap. The accepted lag wins a tie, followed by the shorter grid lag. The existing integer aliases and cultural prior choose the tactus.

T2 proposes recurrence; actual habitat flux and executed onset reinforcement supply phase and intensity. A proposal received at a hop end can be adopted only at a subsequent fast onset. Initial adoption clears old confirmation evidence and contributes no confidence. Once adopted, each issued future source-hop component prediction freezes its lag, source support and target. Only coordinates known in both prediction and later observation contribute, once, to predicted, observed and matching masses. Late evidence retains its source age. Known silence can disagree with a positive prediction; missing evidence contributes no disagreement.

Raw frames are computed once in the analysis thread and sent through a bounded, preallocated queue, independently of the group arrival adapter. Playback does not wait on this queue; overflow records a missing interval. Offline rendering waits for the current source-hop result at the fixed end cut, after fast advancement and before state publication and the next forecast. Physical source time, availability and receipt remain distinct. A source gap restarts the recursive filters and breaks unsupported differences/interpolation, while the accepted meter coordinate and bounded history persist. Configuration, bus, sample-rate, hop or feature-unit changes require an explicit coordinate reset. Filter origin and the group's existing continuity epoch are not the raw configuration epoch.

## 4.2 Beat Confidence: Phase and Pattern Recurrence

The following event-count confirmation describes Events input. Until an interval hypothesis has first been corroborated, the meter accumulates no beat, subdivision or measure evidence: all three confidences and the subdivision/measure ratios are zero (unknown). The oscillator and onset-event corrections continue during this interval. After corroboration, subsequent detected onsets contribute unit phasors at their predicted phases before correction. Their leaky resultant length divided by their leaky count is the phase-locking value (PLV). A single-position history retains $\mathrm{PLV}\times\mathrm{presence}$. Evidence decays with a 2.5-second time constant; measure accents use 6 seconds. Stability lowers the presence thresholds but preserves the confidence ceiling of 1. The initial adoption contributes no phasor. Subdivision evidence combines its harmonic coherence with $\tfrac12(1-\mathrm{beat\ PLV})$, keeping a beat-only input from certifying its harmonics as subdivisions.

After first adoption, each subsequent onset issues a prediction at its then-current basic lag $t_i+p$. Issued targets are not rewritten when tempo changes. Future onsets and targets match one-to-one within $w(p)$, with triangular timing weight. After their windows close, leaky predicted, observed and matching masses give $J=M/(N_p+N_o-M)$. These are fresh post-confirmation masses; the confirming onset supplies neither a prediction nor confidence evidence. If a retained adjacent interval is shorter than $p-w(p)$, the history contains multiple positions and beat confidence reads $\max(\mathrm{PLV},J S_r)\times\mathrm{presence}$. This supports recurrence of a pattern without requiring every event to occupy the same phase. Subdivision and measure evidence keep their existing phase and accent rules.

Confidence describes phase coherence or confirmed pattern recurrence with accumulated evidence. After initial corroboration, a chance cluster in irregular input can still produce a temporary rise; confidence alone does not certify a unique tempo. In silence the oscillator continues while evidence and presence decay. Phase zero remains the predicted onset and is not shifted again by subtracting the mean onset-phasor angle.

For Continuous input, the future component comparisons accumulate matching $M$, predicted $Q$ and observed $O$ with the existing persistence decay. Where $Q+O-M>0$, the recurrence support is $S_R=\rho M/(Q+O-M)$, with the existing integer-alias profile $\rho=\min(r,1/r)$. Beat confidence is $\operatorname{clamp}_{[0,1]}(\max(\mathrm{PLV}_{fast},S_R)\,\mathrm{presence}_{fast})$; undefined support is omitted rather than treated as a negative observation. Initial seed confidence remains zero. The continuous branch uses defined subsequent support rather than the event `multi` qualification. The score does not condition recurrence mass on beat phase and does not certify a correct phase or a posterior probability. Missing fast input advances time without updating the onset baseline or variance; the oscillator and existing evidence decay continue.

## 4.3 From Meter to Modulation: NeuralRhythms

`NeuralRhythms::from_meter_state` (`core/modulation.rs`) projects the metrical state onto the modulation bands consumed by voice behavior:

*   **Delta** ← the beat (tactus): phase and tempo of the pulse. Its phase drives `env_open`, a cosine gate that sharpens the articulation envelope toward the downbeat as confidence rises and leaves it open when the beat is uncertain.
*   **Theta** ← the subdivision: the note-rate band that the breath oscillator (Section 5.5) locks to.
*   Delta/Theta precision (`alpha`) equals beat confidence; prediction error (`beta`) is its complement.
*   **Measure** carries its own phase, frequency and confidence, with `measure_ratio` identifying the detected grouping. Its phase origin follows the observed strong-onset phasor. No detected grouping yields a neutral, zero measure band; the optional `measure_accent` strength coupling consumes this band.

## 4.4 Composer Priors: Shaping the Temporal Terrain

The director can bend the terrain the pulse forms on—never schedule it—via `MeterShaping` (set from Rhai):

*   `meter_stability(v)` — attractor depth in $[0,1]$. Scales amplitude and event correction, and lowers the existing evidence-presence thresholds. It changes how readily the meter commits without raising the confidence ceiling. The same depth directly weights the preferred tempo: higher values favor an integer-ratio reading closer to the basin center, or 2 Hz without an explicit basin. The default is 0 (neutral).
*   `temporal_basin(min_hz, max_hz)` — a continuous tempo-prior band. Seeds the beat at its arithmetic center and confines candidate adoption and frequency adaptation to the band. At stability 0 there is no additional center preference or restoring pull. At higher stability, candidate preference and confidence-weighted restoring grow in direct proportion to the depth.

The basin has a second role: it initializes `metric()` Voices at the same sanitized arithmetic center, independently of stability. Without a basin, their intrinsic starting rate remains 2 Hz. The final basin declaration applies to every metric Population created by `place()` or `create()`, including declarations after `wait()`; it is one scene initial condition, not a timed change. Entrained/steady/flow retain their separate bodily scale, 2 Hz unless `intrinsic_rate(hz)` is supplied, and explicit `pulse(rate_hz)` retains its rate. Initial Voice phases remain independent, and shared-beat coupling stays zero until sound confirms a beat. Respawn reads the Population template rather than a parent's current clock rate.

These operations shape an emergent process without placing a beat grid; the listener's tempo adaptation remains confined to the declared band.

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
    *   `damping`: Energy-dependent thinning of upper partials. A slow spectral-energy envelope tracks the excitation (fast attack, ~0.5 s release) and darkens the tone as drive fades. Its floor bounds the relative partial balance; it does not create a free tail in the excitation renderer.
    *   `vibrato_rate` / `vibrato_depth`: LFO-based pitch modulation.
    *   `jitter`: 1/f pink noise FM strength for organic fluctuation.
    *   `unison`: Detuned copy amount for chorus-like thickening.
    *   `mode`: Harmonic (integer multiples) vs. Metallic (non-integer ratios).
*   **`Modal`**: Resonator-based synthesis via `ModalEngine`, supporting arbitrary mode frequency ratios and decay times. Mode patterns can be specified as harmonic, odd harmonics, power-law, stiff string, or custom ratios.

Sound generation is dispatched through the `AnyBackend` enum:

*   **`Oscillator(OscillatorBank)`**: A struct-of-arrays layout for cache-efficient additive synthesis. Handles `Sine` and `Harmonic` bodies. Pitch refresh occurs every 64 samples; motion/vibrato refresh every 8 samples.
*   **`Resonator(ModalEngine)`**: A Damped Modified Coupled Form resonator bank. Handles `Modal` bodies. Mode coefficients are rebuilt every 64 samples on pitch change.

The default renderer retains the legacy body response. The optional
excitation renderer separates shaped excitation from body radiation (Sections
5.1.3 and 6.1). Sine and Harmonic respond immediately to composer ADSR and the
existing onset transient, with no free radiation after effective Off. Sine retains
its additive onset boost (gain 0.2, maximum 1, time constant 80 ms); Harmonic retains
its 80 ms drive response and onset spectral burst. Modal retains mode-specific
free decay, impulses and noise drive, including radiation after Off. Explicit body
motion remains effective, with zero as its default.

Sine and Harmonic share the legacy f32 partial-rotation kernel, without per-sample
normalization. Pitch and motion retain their 64- and 8-sample refresh clocks.
The carrier has the same floating-point drift as the legacy kernel. Isolated
constant-level holds can match legacy PCM bits, while excitation multiplication
order and handle scheduling can still change the complete waveform. Effective
Off still stops oscillator radiation; Modal keeps its existing free response.

Body parameters change the emitted spectrum and therefore the shared Landscape. Movement samples that terrain at a fundamental frequency. Birth placement and supported final respawn selections have the optional body-density evaluation described in Section 3.4. Metabolism can also use a body's representative density to evaluate its current position against an environment with its own sound removed (Section 5.2). This option is disabled by default. Timbre inheritance is not implemented.

The birth surrogate requires a body factory to declare direct-model conformance for the selected renderer. The current Sine implementation is supported. Harmonic requires zero motion and one active unison copy. With the legacy renderer, Modal requires one active unison copy and brightness exactly 1. Native Modal is unsupported because its radiation is not shaped by output ADSR and persists after Off. A single copy results from zero spread or `unison = 1`. Controls ignored by a body's renderer are ignored in this decision too. Other Modal brightness values, randomized mode recipes used by Harmonic or Modal, unsupported factories and inputs outside the model's bounded domain use the existing terrain-only birth path. A replacement renderer receives no surrogate capability by default. These conditions identify the implemented fast path; its numerical guarantee remains limited to the saved measurements in Section 3.4.

Body-aware metabolism uses the factory's declaration of conformance for the selected renderer and obtains its density by rendering and analyzing a representative Tone with that renderer. It does not apply the birth surrogate's stricter recipe conditions. Factories that do not declare this capability use the existing point-based metabolism. The excitation renderer supports Sine and Harmonic body metabolism; Modal retains point-based metabolism.

Final respawn body evaluation is enabled only with the legacy renderer and adds policy-specific restrictions to this shared body capability. With the excitation renderer, all bodies retain point evaluation at final selection; the supported Sine/Harmonic Field proposals remain body-aware. Random uses point evaluation for Sine and LandscapeDensity recipes. Hereditary uses point evaluation for LandscapeDensity and LandscapePeaks recipes. PeakBiased supports legacy Harmonic bodies with implicit modes or terrain-derived LandscapeDensity/Peaks recipes; Sine, Modal and explicit fixed-ratio recipes keep point evaluation. These runtime domains identify the implemented paths; their numerical guarantees remain limited to the measured conditions in Section 3.4.

### 5.1.2 The Core Stack

Behavior is split into three focused cores plus the `PhonationEngine`, each defined in a separate file:

*   **ArticulationCore (When/Gate)** — `life/articulation_core.rs`: Manages gating and envelope dynamics. Three variants exist:
    *   `KuramotoCore`: Coupled "breath" oscillator with a normalized energy/vitality model, rhythm coupling modes (`TemporalOnly`, `TemporalTimesVitality`), rhythm reward (metabolism bonus for phase match), and autonomous attack capability. It entrains its envelope to the meter-derived rhythm bands (Section 4.3). Energy is fixed to the 0–1 domain; the stored composing parameters are nominal `endurance_sec` and optional `recovery_sec`, while runtime rates are derived once at voice construction.
    *   `SequencedCore`: Fixed-duration gate patterns.
    *   `DroneCore`: Sustained output with optional sway modulation.

*   **PitchCore (Where)** — `life/pitch_core.rs`: Proposes the next target in log-frequency space. Two implementations:
    *   `PitchHillClimbPitchCore`: Local search over the consonance terrain. Parameters: `neighbor_step_log2`, `tessitura_gravity`, `landscape_weight`, `move_cost_coeff`, `move_cost_exp`, and a single search `temperature` (0 = greedy settling; higher accepts downhill moves via a Metropolis rule). Occupancy/crowding: `crowding_strength`, `crowding_sigma_cents`. Leave-self-out pitch evaluation uses a shared body-power ownership estimate.
    *   `PitchPeakSamplerCore`: Probabilistic peak sampling with `window_cents`, `top_k`, `sigma_cents`, `random_candidates`, and the same search `temperature` (softmax over candidates).

*   **PhonationEngine** — `life/phonation_engine.rs`: Manages note-level command scheduling. Issues `ToneCmd` (On, Off, Update) to the `ScheduleRenderer`. Uses a gate grid (`ThetaGrid`) for onset bookkeeping and note-off placement. Configuration is via `PhonationSpec`:
    *   **When**: `Once` (single trigger), `Pulse { rate_hz, sync, social }` (repeated triggers on the adaptive gate clock), or `Coupled(CoupledTimingSpec)` — explicit synchronization or bodily acoustic participation (Section 5.4).
    *   **Duration**: `WhileAlive`, `Gates(n)`, `Field { hold_min_theta, hold_max_theta, curve_k, curve_x0, drop_gain }`.

### 5.1.3 The Sound Pipeline

Audio rendering is handled by `ScheduleRenderer` (`life/schedule_renderer.rs`), which maintains a `BTreeMap<ToneKey, RoutedTone>` of active tones, each routed to one of two buses (Section 6.2): the **habitat bus** (analyzed as the landscape's environment) and the **presentation bus** (what the audience hears).

The `Tone` struct (`life/sound/tone.rs`) combines:

*   A backend (`AnyBackend`: `OscillatorBank` or `ModalEngine`).
*   An optional `RenderModulator` for articulation envelope shaping.
*   An ADSR envelope: linear attack ramp, exponential decay to sustain level, constant sustain, linear release ramp.
*   Smoothed pitch and amplitude transitions with configurable time constants.
*   Continuous drive for sustained excitation.

The `PhonationEngine` emits `ToneCmd` commands; the `ScheduleRenderer` creates,
updates or releases Tones and mixes their radiation per bus. The legacy renderer
keeps its output ADSR. In the excitation renderer, ADSR and articulation shape
the input, with no second ADSR applied to body output. An Off request starts the
specified release ramp; effective Off is its endpoint. Sine and Harmonic then
radiate known zero, while Modal continues its free response. A zero-second
release uses the existing minimum of one sample.

In the excitation renderer, pitch Updates address open handles, including their release interval, and retain
carrier phase or resonator state. Closed handles reject Updates and future kicks;
Modal tails keep their last coefficients. A new On creates a separate Tone that
may overlap retained radiation. Ownership and routing remain attached to each
source generation. Effective Off and disposal are distinct: closed states are
retired when the summed residual bound for their source generation reaches the
renderer threshold. A silent open handle is not disposed merely for being silent.

With body-aware metabolism enabled, either renderer also captures each supported Voice's habitat PCM by `(id, generation)`. Subtracting it from the habitat mix gives that Voice's environment. Presentation-only audio and other generations are excluded from its own signal. The shared mix is analyzed continuously. Each Voice's environmental PCM history is retained; its analysis advances on its turn, accounting for the intervening hops in one update rather than restarting from a finite window.

### 5.1.4 Control-Plane Signals: Planned and Error

The Voice coordinates its cores through two orthogonal signals rather than direct coupling:

*   **Planned**: The PitchCore proposes a target (`TargetProposal`), and the Voice maintains the "planned" state—next target frequency, expected jump distance, and salience. This represents the agent's *intention*.
*   **Error**: The Voice computes the discrepancy between the SoundBody's current pitch and the planned target (signed cents, absolute cents). This represents the *result* of prior actions and is available for observation. Importantly, the PitchCore does not read the error signal—search remains decoupled from feedback.

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

By default, energy updates use the effective consonance level at the Voice's fundamental. Optional body-aware metabolism instead evaluates the current position against the environment after subtracting the Voice's own sound. It averages the effective consonance score over the representative body's subjective-intensity density, weighted by ERB cell widths and normalized by in-band mass, then applies the existing score-to-level sigmoid. The current-position density is retained per Voice and rebuilt when the fundamental, recipe, generation or analysis settings change.

The `[body_metabolism]` configuration section enables this path with `enabled = true`; `updates_per_hop`, `observation_frames` and `representative_hold_sec` must be specified explicitly. Omitting the section or setting `enabled = false` keeps point-based metabolism. With `render_prototype = true`, Sine and Harmonic use native representative Tones and source-PCM subtraction; Modal uses point-based metabolism. The author adopted `updates_per_hop = 2` from comparison results; 2 is a configured update budget, not a constant of the psychoacoustic model.

Each hop updates at most the configured number of supported Voices in round-robin order. Per-Voice slots carry monotonically increasing visit numbers: the smallest numbers are selected first, and both visited Voices and newborns receive new numbers at the back. Other Voices keep their last body score. Until its first evaluation, a newborn uses the existing point score. Unsupported bodies and unavailable evaluations also use the point score; an unknown result is not treated as a known zero. These metabolism scores feed energy updates; life diagnostics and direct phonation gates retain their point evaluations. Metabolism alone does not change final respawn scoring; the independent birth option can enable the supported legacy-renderer body evaluations described in Section 3.4.

This mechanic creates a Darwinian pressure: **Survival of the Consonant**. Poor acoustic fit depletes energy; good fit can maintain or replenish it. The fit is a point level by default and a body-weighted level when the optional evaluation is available. With Landscape habituation enabled, a formerly supportive environment can lose effective value under sustained activity and recover after withdrawal, so survival depends on both acoustic fit and recent perceptual history.

## 5.3 Pitch Retargeting Logic

Agents are not static; they move through frequency space to improve their fitness. The execution layer applies a retarget gate (a zero-crossing of the meter-derived theta band, Section 4.3, plus an integration window) and then asks the PitchCore to propose the next target. Candidate evaluation reads the habituation-eroded score at the candidate fundamental. Self-exclusion recomputes a raw score from the actual remaining habitat PCM, then applies the same local erosion before comparison. Body-aware metabolism leaves these point-based movement evaluations in place. Its energy changes can still affect subsequent movement through the existing energy-dependent behavior.

### 5.3.1 Pitch Application Modes

Three modes govern how a new pitch target is applied:

*   **GateSnap**: Discrete pitch changes at sounded note boundaries. Gated phonation uses the target at onset; sustained participation retains the latest adopted target until a sounded opportunity.
*   **Glide**: Smooth continuous pitch transition with a configurable time constant $\tau$. The SoundBody interpolates exponentially toward the target frequency, producing portamento effects. Suited for drone-like Voices or slow melodic movement.

*   **Rearticulate**: For free-pitch Hold Voices without a participation clock, the default zero boundary releases the old Tone and attacks at each changed target's adoption sample. A positive rearticulate_above boundary instead slides at or below its distance from the last sounded onset.

When a new Tone replaces another at the same sample for the same source and generation, the old Tone decays with its original release duration from its current envelope value.

For `seek_consonance()` voices the mode is resolved automatically from the phonation timing unless the script chooses explicitly: sustained voices (`once()`) glide, while re-attacking voices (pulse or coupled timing) snap at onsets. Explicit sustain with participation also resolves to Glide.

### 5.3.2 Crowding and Leave-Self-Out

The crowding system prevents agents from collapsing to identical frequencies. Both crowding (mutual repulsion) and pitch adaptation read one shared occupancy field: a Gaussian penalty centered on each occupied fundamental (raw f0), with width set by `crowding_sigma_cents` (default 60) and strength by `crowding_strength`. A pairwise split bias further prevents frequency degeneracy. No octave-equivalence (chroma) penalty is applied; octave relations act only through the consonance potential.

`leave_self_out(true)` evaluates the surroundings by subtracting the Voice's
actual habitat PCM from the mixed PCM before analysis. The remainder passes through
the same auditory filterbank, hop statistics, reassignment, critical-band compression, density
smoothing and H/R/C calculations as the mixed field. Habitat-routed Flora, including
residual sound, stays in the remainder. All bodies use this method; there is no
public mode selector.

Each reserved Voice retains its own analysis history. At birth it copies the mixed
stream's state before the first own sound. Populations enabled by a later patch are
also observed from birth. Mixed and own PCM travel in one frame packet, and the
result is bound to Voice ID and generation. At the analysis-worker boundary, a gap
resets the mixed and per-Voice histories; valid publication resumes after the common
analysis window fills if further PCM arrives. The field runtime currently ends the
run when a disconnected or invalidated observation leaves no usable frame within
its fixed one-hop lag limit. It does not generate warmup hops for automatic recovery.
The shared near-silence floor remains in force. The usual shared habituation erosion
is applied once when a consumer samples the resulting score.

`viability_scope("environment")` uses the same observation at the current pitch for
the continuous-recharge score, independently of pitch movement. Basal drain and
telemetry retain their mixed-field level; the optional body-weighted metabolism
evaluator retains its own contract.

These timing-sensitive transitions and crowding evaluations are guarded by regression tests to prevent subtle breakage.

### Retaining the Voice's articulation policy

For free-pitch sustained coupled timing with Rearticulate and a positive movement
boundary, `articulation_persistence(p)` adds a private Base/Connect policy. Base
uses the existing boundary and participation rules. While a Tone lives, Connect
slides all adopted targets and intentionally withholds sounded opportunities,
without a new Tone, onset metabolism or onset-context learning. Small adoptions
and cost skips propose Connect; large adoptions and newly selected sounded
opportunities propose Base. For `0 < p < 1`, an opposing proposal retains the
current policy with probability `p`; a matching proposal does not draw. Each
planned opportunity keeps its decision until execution, independently of waiting
hops. Entering Connect cancels pending pitch without changing last-onset memory;
cancelled adoptions never replay and policy changes alone never sound.

The parameter is initial-only and inherited through templates; replacements
start in Base with fresh performance state and a dedicated random stream.
Default zero and endpoint one bypass this state and retain the existing path.
Live mode changes reset Base; the same mode and boundary-only changes preserve
it, and a zero boundary suspends its path. Finite Off, rest, tails, gaps and
lifecycle constraints take priority, and a legal onset without a living Tone
cannot be vetoed. Retention acts on both policies, so increasing `p` need not
monotonically increase Connect time or reduce attacks. Its duration follows
existing events, not seconds or beats; it is inertia without inferred cadence,
contour, rest or phrase boundaries.

### Extending a living Tone without pending pitch

`tie(p)` is an initial-only probability in free-pitch sustain plus metric/entrained/steady/flow
with GateSnap or Rearticulate. At a selected legal opportunity, a pending target
still attacks; a living Tone without one extends with probability p. Omission
resolves to 1 for sustained metric and 0 for sustained entrained/steady/flow, where costs
can omit opportunities. Explicit 0 retains attacks without tie state or draws;
1 extends eligible Tones
without drawing. Intermediate values use a dedicated individual stream at execution,
never during waiting. Small boundary-limited glides are eligible; small GateSnap
adoptions remain pending and attack. This thins the existing opportunity sequence,
without changing pitch, timing candidates, costs or skip. It is not an overall
legato amount or phrase-boundary mechanism. Connect vetoes before tie; Base uses
it only without pending pitch, leaving policy proposals and retention unchanged.
Finite Off, tails/rest and lifecycle have priority, and no-Tone legal onsets pass.
Templates carry the probability, not performance/RNG state. Builder order is
independent; live Glide suspends tie and returning resumes it. There is no live
probability setter. Inputs follow sanitize01; unsupported final forms warn and ignore.


## 5.4 Onset Timing: Synchronization and Participation

`RhythmRelation` distinguishes synchronized timing (`metric()`) from acoustic
participation (`entrained()`, `steady()` and `flow()`). Both issue sound through the same
PhonationEngine command path.

The presets keep distinct timing relationships. An opportunity is not an attack:
phonation legality, lifecycle, rest and articulation can withhold onsets or extend
a living Tone.

| Preset | Keeps | May change | Does not guarantee | `entrainment(x)` |
| --- | --- | --- | --- | --- |
| `entrained()` | Lifetime-fixed intrinsic bodily rate | Participation period follows supported recurrence weighted by proximity to that scale; costs select near-due timing, waiting up to one participation period, or skipping | Equal onset intervals, persistent differences between Voices' onset rates, or shared beat phase | Acoustic influence on adaptation and timing costs; omitted `0.5` |
| `flow()` | Intrinsic scale for bounded renewal, without recurrence adaptation | Irregular renewal intervals, acoustic timing choices, waiting and skipping | A regular sequence, a beat, or a mean onset interval equal to the intrinsic period | Acoustic timing costs, not renewal-scale adaptation; omitted `0.05` |
| `steady()` | Intrinsic period and later private due-time grid, subject to sample rounding | Sound or skip at due; execution constraints can remove onsets | An attack count, a pitch outcome, or following another Voice's time | Sound-or-skip costs only; omitted `0.5` |
| `metric()` | Confidence-weighted attraction to the confirmed production beat and configured phase offset; independent phase before confirmation | Effective rate and phase; the clock free-runs before confirmation | Beat confirmation itself, exact phase coincidence, or independent noninteger tempo relations | Beat-rate and phase attraction; omitted `0.95` |

Setting `entrainment(0)` removes these acoustic timing influences; flow remains
irregular. The meter supplies one beat and integer-ratio layers. Steady advances missed or illegal
opportunities by whole intrinsic periods. Its initial sustained Hold onset is
immediate, outside the later grid. These are initial-only presets; the last timing
preset wins, with duration resolved separately.

Participation separates causal habitat observation from the body's due time.
The observer projects fully observed recurrence cycles onto a roughly four-second
horizon at 20 ms spacing. Recurrence contrast fades toward the adaptive baseline
with a six-second time constant; energy beyond the horizon uses the observed mean.
Future queries never enter the observation history. Known self habitat PCM is subtracted
from the observed mixture before energy analysis. The external history is projected
using both the previous shared recurrence projection and a local history predictor. Matching actual outcomes fit their bounded mixture for subsequent participation requests.

For entrained/flow, candidate cost combines squared displacement in
participation periods, distance from experienced external energy shapes, and a
three-band overlap integral. Context memory retains three-band external energy in
two post-onset windows (2×3); candidate forecasts use the same windows. The default
64-point envelope proxy includes the duration rule's hold and the Voice's ADSR release.
An optional footprint path uses 16 bins of body power approximated from a frozen
representative-onset recipe, or an envelope proxy. Its worker is independent of the
auditory-cognition observer; missing, stale or unsupported body results use the proxy.
For Sine and Harmonic, the excitation renderer computes these 16 bins from native
representative PCM and normalizes them to their peak; its identity is separate from
the legacy analytic footprint. Native Modal remains unsupported (`renderer-phase3`)
and uses the envelope proxy because its free radiation has different support.
Legacy analytic Tone-energy forecasts remain unavailable in the excitation renderer.
Actual-PCM observation, including T2 and the external/self-sound history predictor, continues
through the rendered bus audio.
Omitting costs one plus the number of
consecutive voluntary omissions; an emitted onset resets that count. This cost
is an action preference, not a metabolic reward. It prevents
bounded acoustic costs from prescribing permanent silence, but does not establish
musical quality. The envelope proxy omits additional body/modulator decay; neither
energy estimate is an auditory masking or consonant-fusion model.

An optional **T2 preprocessing path** supplies accents from the actual audio of
each observed bus. Its principle layer is a named composition of AMT 1.6 auditory
components: 30 gammatone bands on the LTFAT `erbspacebw(80,8000,1,1000)` grid,
half-wave rectification and a 1 kHz inner-hair-cell low-pass stage, five adaptation
loops with `limit=10`, and four modulation outputs: LP 2.5 Hz, signed BP 5 and
10 Hz, and absolute BP $50/3$ Hz. Numerical checks on finite inputs establish
agreement with the corresponding reference realizations. These component
references are described in the AMT implementations of
[gammatone](https://amtoolbox.org/amt-1.6.0/doc/common/gammatone_code.php),
[ihcenvelope](https://amtoolbox.org/amt-1.6.0/doc/common/ihcenvelope_code.php),
[adaptloop](https://amtoolbox.org/amt-1.6.0/doc/common/adaptloop_code.php), and
[modfilterbank](https://amtoolbox.org/amt-1.6.0/doc/common/modfilterbank_code.php).
The analysis worker derives coefficients from the actual sample rate and retains
filter state across hops. An input gap starts a new cold state; support records
include the input prefix on which the recursive filters depend.

Group attribution and event formation are **author rules**, separate from those
auditory components. Existing NSGT assignments supply each group's fraction of
the bus mass in each native band, including residual mass in the denominator.
These fractions condition the whole-bus responses; groups do not have separate
PCM streams or filter banks. The current rule takes signed hop means, freezes
the first hop's fractions over four consecutive hops, and combines the positive
component rises with explicit component weights and gain. A supported local
peak above the explicit threshold creates an accent directly, with an explicit
event-weight gain. It does not require an accent from the other acoustic detector.
The event time and the later time at which it becomes available remain distinct.
The same accent goes to the period estimator and next-arrival model, with credit
retained by its owner and a reset when that owner changes. Missing PCM, unknown
ownership or broken support suppress an event; supported zero remains known zero.

T2 is disabled by default. Its producer runs for raw production recurrence when
`temporal_meter_t2 = true`, or for group events when `[temporal_t2]` is present.
The raw meter input requires none of the group event parameters. Enabling the
group event path requires the acoustic-group and period configurations and
every T2 setting explicitly: `component_weights`, `gain`, `threshold`,
`weight_gain`, `window = "hop_mean"`, `combination = "positive_component_rise"`,
`attribution = "first_hop_mass_fraction"`, and `unknown = "suppress"`.
The free numerical parameters have no selected values or defaults. Reference
agreement establishes the preprocessing behavior; it does not select those
parameters or establish a musical effect.

The policy can additionally use **Periodic next-arrival forecasts** from observed
habitat groups to select a participation candidate. Using the observed period and
last accent, each forecast fixes the next arrival time at issuance and represents
it as a single right-continuous step.
The arrival term is disabled by default; `arrival = true` in
`[temporal_onset_comparison]` enables it with the configured auditory observation
and period estimator. An omitted `arrival_weight` uses 1, placing arrival
mismatch on the same scale as the context mismatch term by convention. Explicit
finite, nonnegative weights remain supported. The existing participation coupling
sets the strength of both terms; the unit weight is not a preprocessing constant.

Each decision uses one common set of active habitat groups with causal forecasts
that cover every valid candidate window. A group is excluded as the acting Voice's
own only when habitat routing, source and body generations, body evidence,
prototype assignment and freshness agree. If this correspondence is unknown, the group is not
excluded as self. For each candidate, the step's increment across its onset
window gives a group arrival probability; their equal-weight mean is $P$. The
additional cost is $\mathrm{coupling} \times \mathrm{arrival\_weight} \times (1-P)$,
favoring candidates near a predicted arrival.

Groups with unavailable periods, uncertain elapsed time, invalid clocks or
insufficient forecast coverage are excluded. An empty common set leaves arrival
unknown and applies no arrival term. A nonempty set with $P=0$ is known zero and
incurs the weighted cost. Omission still uses the minimum candidate cost without
the arrival term, so this term affects candidate selection rather than prescribing
an omission.

Steady uses unchanged costs and weights. Arrival cost cannot change sound versus
skip in its one-candidate policy because the skip comparison excludes it.
Explicit sustain retains the current Tone on skips. All three pitch modes,
movement boundary, tie (omitted zero), persistence and finite-duration rest use
the existing articulation path; no adoption-time candidate is added, even in Glide.
Respawn inherits policy and rate from the template with fresh performance state.

`intrinsic_rate(hz)` sets the founder bodily rate for `entrained()`, `steady()` and `flow()`
before `place()` or `create()`, independently of builder order. The intrinsic
period is `1/hz` and remains fixed during life. This sets a bodily scale rather
than commanding onset intervals or placing a shared beat. The same period feeds
participation, finite note lengths (`cycles`, `duration_range`), `rest(periods)`
and representative overlap hold, so note lengths and rests in seconds change
together. Omission retains the existing 2 Hz path. Replacements read the
Population template with fresh due, plan and policy state, without inheriting
parental cadence. There is no live setter. Final entrained/steady/flow specifications
require finite values in 0.01–200 Hz; invalid values are script errors before
placement. These are execution protection limits, not a musically recommended
range. Other final timing forms, including metric, pulse, repeat and once, warn
and ignore the modifier even if its value is invalid. Flow's renewal intervals
separately retain their 0.035–4 second limits, so extreme rates do not scale
actual intervals proportionally. `rhythm_freq` remains an articulation-core
oscillator control; `proposal_interval` remains a pitch-proposal interval.

Experience memory updates only after an emitted onset and both matching acoustic windows. The intrinsic period, acoustically
adapted participation period, and sound duration have separate roles. An entrained Voice
selects a supported recurrence near its intrinsic scale, weighted by a Gaussian
affinity with a half-octave width. Coupling and recurrence support control the
adaptation rate, with two intrinsic periods setting the time constant. Unavailable
references relax the cadence toward the intrinsic pace. These support values and
constants are engineering choices, not validated human beat confidence.

Cadence updates preserve the remaining fraction of a future, uncommitted cycle
and never target a shared onset phase. Each observation is used once; future evidence
and changes to already selected actions are rejected. Chosen waits do not train
the cadence. Flow keeps the intrinsic scale for clustered renewal instead of
adapting it to recurrence. Acoustic context still affects waiting and omission,
so executed intervals can change. An internal switch to flow preserves a committed
candidate and restores the intrinsic scale for subsequent intervals. It also
clears the cadence-observation timestamp, so returning to entrained timing does
not count time spent in flow as elapsed adaptation time.

Hold lengths and their overlap preview use the configured intrinsic period,
independently of cadence adaptation. Thus `cycles(n)` does not stretch a sound when
the acoustic recurrence slows; flow retains its existing threefold hold multiplier.
ADSR release remains in seconds. Later tempo observations do not reschedule an
existing note-off. Candidates respect the actual-onset 200 Hz ceiling, and amplitude
and viability gates still decide whether to emit an onset.

Forecasts are built only for pending decisions, and observed external contexts feed
subsequent candidate comparisons. Sustained-note termination, causal effect estimation
and action-value learning are not implemented.

Observed external RMS history, indexed by past age and known coverage, is also frozen
with the decision-time evidence. The context-distance term compares external energy in the two experienced windows.

For explicit synchronization, onset timing is generated by the `CouplingClock` (`life/phonation_engine.rs`): a per-Voice phase oscillator that offers an onset opportunity at each integer crossing of its phase. Its effective rate blends an intrinsic renewal rate with the shared beat:

$$ f_{eff} = (1 - \ell)\, f_{int} + \ell\, f_{beat}, \qquad \ell = \kappa \cdot c $$

where $\kappa$ is the voice's coupling strength (`entrainment`, 0–1) and $c$ is the meter's beat confidence (Section 4.2)—so a voice can only lock as strongly as the meter is believed. A phase pull drags the oscillator's crossings toward the beat phase (optionally offset by `microtiming`):

$$ \dot{\phi} = f_{eff} \left(1 + \ell K \,\mathrm{err}(\phi_{beat} - \phi)\right) $$

where $\mathrm{err}(\cdot)$ is the wrapped phase error in cycles ($[-0.5, 0.5]$) and $K$ a fixed pull gain chosen so the rate factor stays positive—the phase always advances, only faster or slower.

This phase-attraction mechanism serves synchronized timing. Increasing coupling
for an entrained/steady/flow participant instead increases its response to acoustic
context; it does not switch the participant into this clock.

Each onset carries a strength set by the voice's `rhythm_role`—beat 1.0, subdivision 0.7, accent 2.5, texture 0.85—and these strengths feed back into the production meter's drive. A recurring accent therefore drives the meter harder, allowing a downbeat (and eventually a measure) to be *induced* by the population rather than declared. There is no externally imposed grid anywhere in this loop.

The optional `measure_accent(amount)` couples detected measure structure back to onset strength.
Its multiplier is `1 + 0.35 * amount * confidence * cos(phase)`, evaluated at each actual onset.
The measure phase is measured relative to the circular mean of the observed accents, rather than
the arbitrary origin of the accumulated beat count. An undetected measure produces no modulation;
the default amount is zero. This path schedules no new onset and sets no beats-per-measure value.
Production `rhythm_observation` records expose the measure phase, frequency, confidence and ratio
used by the voices; listener observations remain separate.


### Sustained beat synchronization

Explicit `sustain()` plus `metric()`, in either order, retains a live Tone and uses
integer crossings of the existing CouplingClock as sounding opportunities. The
first Hold onset is immediate. Free-pitch GateSnap/Rearticulate defaults to
attacking a beat with a pending adopted destination and extending other beats,
equivalent to `tie(1)`. Explicit `tie(0)` re-attacks every legal crossing; other
explicit probabilities keep their meaning. Fixed pitch and Glide still attack
each crossing. Sustained entrained/steady/flow keeps omitted tie at 0, with cost-based
omission. The resolved default also applies while a live mode switch enters
the eligible range; tie remains initial-only. GateSnap and boundary-zero
Rearticulate defer the latest adoption to a legal crossing. Positive boundaries,
Base/Connect persistence, finite-duration rest, role strength, confirmed measure
emphasis and microtiming share the existing articulation rules. Synchronization
has no participation cost skip or committed waiting plan; suppressed crossings
advance the clock without replay or onset metabolism/learning.

The shared beat is acoustically confirmed. Before confirmation coupling is zero,
so the clock free-runs at the existing basin-centered rate with its independent
initial phase. This combination does not place an external beat. Fixed pitch is
supported as in sustained participation, while tie/persistence retain their
free-pitch scope. Implicit Coupled tuning and standalone presets are unchanged.


### Authored rest and dwell

`rest_on_habituation(w)` and `dwell(w)` are initial-only author rules, both disabled
by default (`w = 0`). They accept finite nonnegative f32 weights without an imposed
upper cap. They do not implement phrase perception or a principle of tonal stability.

For Voices, `rest_on_habituation(w)` requires explicit `sustain()` with `metric()`,
`entrained()`, `steady()` or `flow()`. A shared audio-only history of heard attacks
supplies habituation at the sounding pitch. With habituation enabled, the weight
scales the stochastic exit rate from a sounding span. Exit uses an existing legal
opportunity and releases the Tone; return accumulation starts after the audible
tail ends. Return depends on private recovery from the Voice's own attacks, and
resumes sound only at a legal opportunity that passes the existing policy. Unknown
observations suspend exit and return accumulation while private recovery continues.
This history is separate from the Landscape-level consonance erosion in §3.5.

`dwell(w)` requires free pitch with explicit `seek_consonance()`. It changes the
pitch-proposal clock according to the current environmental level $L$ after shared
habituation erosion:

$$ \Delta\phi = \Delta t\,\max\{0, 1+w(1-2L)\}. $$

High levels slow exploration, low levels accelerate it, and the clock stops when
the expression reaches zero. Pitch self-exclusion supplies $L$ when enabled and
available; otherwise the mixed effective field is used. Existing legal proposal
gates still apply. Adaptation, glide and private recovery keep physical time.

### Following executed self intervals

`follow_own_sequence(w)` is an initial-only, opt-in preference for free
`seek_consonance()` Voices with stepwise sustained participation (no glide, no
positive rearticulation boundary, no articulation persistence); the default
`w = 0` leaves behavior unchanged. Its predictor is a candidate model: PPM-Decay
(Harrison, Bianco, Chait and Pearce 2020) with the published analysis settings
(order 4, a 15-item buffer, long-term weight 0.6, half-life 3.5 s, retrieval noise
0.8). Only executed self onsets train it; rejected and pending targets do not.

Execution recording, interval encoding and the coupling to pitch choice are author
rules. Each executed frequency becomes the signed interval
`round(12 log2(f / f_previous))` in semitones, rounded to the nearest integer with ties away from zero; output pitch is not quantized. After
the existing candidate generation, each candidate's score gains
`w (log2 p(candidate) − log2 p(current target))` from one immutable retrieval of
the predictive distribution, so the current target receives zero and the field,
self-exclusion, habituation and movement terms are unchanged. The preference
changes which pitch is chosen from the sequence of executed intervals only; it does
not model rhythm, duration, phrase boundaries or harmonic context, and it does not
learn other Voices' sequences.

### Melodic continuity from range and mobility

Initial-only `melodic_continuity(w)` uses the same supported stepwise sustained
forms and defaults to zero. Its candidate is von Hippel (2000), *Redefining Pitch
Proximity*: the text-consistent AR(1) mean `mu = m + r*(last-m)` and variance
`v = s²*(1-r²)`. This interpretation restores mean and scale missing from the
printed distribution; it is not an author-approved erratum.

Finite-prefix sample statistics, committed self-On input and continuous log-density
scoring are author rules. With at least four onsets and certifiably positive
variance, a known prediction replaces movement cost and fixed-center gravity,
including their role in peak extraction, and scores the resulting legal candidates.
Unknown or failed predictions restore those two terms. SQ remains independent.
There is no new forgetting window, rhythm, harmony or phrase planner.

### Following heard tonal context

`follow_tonal_context(w)` is an initial-only, default-zero term for free stepwise
sustained `seek_consonance()` at 48 kHz. The candidate observation is the IPEM
sensory short-term memory model associated with Leman (2000) and described by
Bigand et al. (2014): 40-channel ANI, periodicity pitch and local/global memories
with 0.1/1.5-second half-lives. It receives actual habitat PCM, or `mix - own`
when self-exclusion is enabled; the active listener observes presentation PCM.

The sine probe, 1-cent lookup approximation, correlation readout and generation
coupling are author rules. They add `w*q` to existing candidates and the current
target baseline, and to the current sounding pitch's score before the dwell
sigmoid. An unknown query makes the selection use its existing path. The term
provides neither chord labels nor harmonic plans. The approximation's declared
operating range is [0.25,1], established at 0.25/0.5/1; the word still accepts all
finite nonnegative weights. Enabled initialization builds a shared table of about
1 GB, measured at about 80 seconds, before audio starts; disabled use builds none.

### Responding to local phrase boundaries

Initial-only `phrase_boundary(w)`, default zero, supports Sine Voices with explicit
sustained metric/entrained/steady/flow participation and discrete gate-snap or
rearticulation. Its candidate is Cambouropoulos (2001), LBDM, with pitch/IOI/rest
weights .25/.50/.25. Online normalization over a six-onset window and the response
are author rules. The input is the renderer's completed execution receipts,
separate from the committed commands used by SQ and ME.

After a confirmed local boundary becomes available, the first existing opportunity
consumes the response. A legal new Tone before expiry receives `1/(1+w*B)` gain
through its tail on both buses and self-exclusion capture. A veto does not transfer
the response to a later opportunity. Existing strength, release and clocks stay
in place. This detects a past local boundary; it does not plan a future ending.

### Expression over a known final section

`expressive_section(name, arch_weight, ritard_weight, callback)` assigns a loudness
arch and final slowing to a composer-declared last interval. Both weights default
to zero and saturate at one; zero/zero follows ordinary `section` behavior.
Supported scenes use restricted Once/WhileAlive or nonsustained synchronized
Coupled Gates/Field Voices. Flora, sustained participation and other unsupported
forms disable the scene-wide effect.

Todd (1992) single-layer intensity and the Friberg–Sundberg (1999) q=3 timing
curve are candidate kernels. Interval assignment, the amplitude floor, weight
interpolation and the frozen generation clock are author rules. One shared time
map moves crossings, new gate Offs and release starts; DSP, metabolism, perception,
memory and release tails keep physical time. Gain applies once before PCM routing.
The section is an initial form choice, not evidence of a perceived phrase or cadence.

### Selecting a listener profile

`listener_profile("sensory" | "tonal")` is a Director setting before scene actions
or scopes. Omission selects **sensory**, preserving the existing base behavior.
Tonal bundles SQ 0.05, ME 0.00076066292, PX 1 and final supported section
arch/ritard 1/1. Explicit Population weights, including zero, take precedence;
unsupported final forms and absent suitable sections are silently skipped.
An explicit `expressive_section`, including zero/zero, overrides inherited expression.

Profile selection makes no separate mechanism claim: each term retains its candidate
and author-rule labels. TC is excluded because of table initialization cost and its
limited approximation acceptance range. Individual axes remain zero by default;
selecting a profile does not broaden their supported Voice forms.

## 5.5 The Breath Oscillator: Kuramoto Articulation

Independent of onset scheduling, the `KuramotoCore` ArticulationCore entrains each voice's *envelope* (its breath) to the meter-derived theta band, using a mean-field Kuramoto phase step:

$$ K_{eff} = \omega_{target} \cdot K_{global} \cdot s_\theta \cdot |\theta_{mag}| \cdot \theta_\alpha \cdot g_{env} \cdot a_{env} $$

where $\omega_{target}$ is the theta band's angular frequency, $K_{global}$ the scene-wide coupling gain (`set_global_coupling`), $s_\theta$ the voice's theta sensitivity, $|\theta_{mag}|$ and $\theta_\alpha$ derive from beat confidence, and $g_{env}$, $a_{env}$ are the envelope gate and amplitude. Helper functions `kuramoto_k_eff()` and `kuramoto_phase_step()` are exposed for external simulation (paper experiments). The energy/vitality subsystems interact with this coupling:

*   **Rhythm Coupling Modes**: `TemporalOnly` (pure phase coupling) or `TemporalTimesVitality { lambda_v, v_floor }` (healthy agents synchronize more strongly).
*   **Rhythm Reward**: an optional `MetabolismRhythmReward` (`rho_t`, `AttackPhaseMatch`) grants a metabolic bonus for phase-matched onsets, linking rhythmic conformity to survival.
*   **Autonomous Attack**: self-triggered attacks when envelope-gate and confidence thresholds align.

## 5.6 Flora: Sound as a Growing Spectral Mass

Flora adds a composition mode alongside Populations. A Flora is a non-negative
spectral biomass over a bounded Log2Space extent. It grows from the base
consonance field before habituation and shares the habitat with Voices. Resource
depletion and recovery provide its band fatigue; Voices and the listener retain
habituation, which Flora's habitat audio still drives. Coupling between Flora and a
Population passes only through their radiated sound and the resulting Landscape.

Local consumer–resource rules govern growth, spread, consumption and recovery.
The Allee half-saturation density is fixed at 0.144 mass per octave and converted
to mass per bin on the Log2Space grid.
Each Flora's resource is consumed by its latent biomass and by other habitat
sound. Its actual habitat PCM is removed before external power is measured.
By default, habitat-routed Flora also removes its own PCM before the auditory
front end computes the base consonance and harmonicity used for growth and seeding.
Its H/R observer starts from the mixed stream's pre-birth history. Initial-only
`leave_self_out(false)` uses mixed C/H without this H/R observer; presentation-only
Flora also uses mixed C/H. Resource power always excludes own PCM. Placement uses
mixed base mass because no own sound exists yet. Missing self-excluded C/H is not
replaced by mixed C/H; the runtime's existing observation-gap stop rule applies.
The acoustic
reference is the mean square of a sine with nominal amplitude 0.03. Supported
FFT power is distributed across NSGT bins and integrated with the existing band
coefficients. Missing observations preserve initial resource state or the last
observed consumption, rather than counting as silence. Resources are per Flora;
a muted or presentation-only neighbour contributes no habitat power.
ERB cells decompose biomass into tracked tonal and residual lanes. Sine or resonant
tones turn those lanes into audio, with sample-rate fades and amplitude ramps.
Residual sound is disabled by default and can be enabled with `.residual(true)`.
Placement, updates, release and bus routing are available through the Experimental
Rhai surface; sample `17_flora.rhai` combines Flora with a
retuned Population.

Every Flora schedules tonal births and ordinary replacements at private
CouplingClock crossings when fresh audio confirms a beat. There is no Flora
`metric()` switch. A shared habitat audio-only meter uses the selected recurrence
path and enculturation priors, without internal onset commands. In a scene with
Flora placements it observes from run start, including before placement. Private
clocks begin with independent phases and the tempo-basin center (2 Hz without a
basin); fresh confirmed beats attract them with strength 0.95 weighted by confidence.
Unknown, stale or unconfirmed observations use continuous allocation and the
basin-centered rate without resetting phase.

`pace(k)` sets ecological speed relative to that private clock, initially or live.
Its default is 1, its lower bound is 1/64, and finite f32 values have no additional
upper cap. Let $f$ be the clock velocity in Hz before phase correction and $r$ the
selected opportunity ratio. If $k > 20$ and fresh confirmed beat evidence also has
positive subdivision confidence with ratio 2, 3 or 4, crossings of private phase
times that ratio allow action. Otherwise $r=1$ and actions use beat crossings.
Layer changes preserve phase. The ecological speed is

$$ p = \frac{f}{2}\min\{k,20r\}, \qquad \Delta t_{eco}=p\,\Delta t. $$

This saturates at one `RENEWAL_SEC` (10 ecological seconds) per nominal opportunity;
continuous fallback uses $r=1$. Growth, diffusion, seeding, resource recovery and
consumption share this time scale, as does retirement. Biomass and resources
remain continuous. During confirmed timing, tonal and residual amplitude targets
and their joint normalization update only at action crossings; supported targets
hold between crossings while the existing slew advances. Fallback updates targets
each hop. Waiting lanes reserve only current excess in their last window.
Support loss, separation conflicts and release retire lanes immediately, with a
finite metabolic tail. Transitions apply at sample ticks within the hop. Macro
fade bounds release, and explicit `fade(0)` ends immediately.

`pitch_apply_mode("glide")` is the default and lets tonal lanes follow supported
peaks. `pitch_apply_mode("rearticulate")` holds each lane's sounding frequency,
including its retirement tail. A peak more than 50 cents away requires retirement
and a new lane under the existing opportunity and capacity rules. Live mode changes
preserve sounding lanes; switching from glide holds their then-current frequency.
Residual noise keeps its centroid motion. Flora does not support `gate_snap`.

Initial-only `rest_on_habituation(w)` applies the span/gap author rule from §5.4
to a whole Flora, with default 0 and finite nonnegative weights without an imposed
upper cap. Heard habituation is weighted by the target power of sounding tonal
lanes. A gap suppresses tonal and residual births while existing lanes retire;
biomass and resources continue evolving. Return follows private recovery after
the tails end. Span/gap transitions use beat crossings, including the free-running
clock during fallback; subdivision opportunities affect allocation and amplitude
targets only.

These dynamics belong to the authored-rule tier. Their use of an auditory landscape
does not make the growth or succession rules a model grounded in auditory cognition.
The API stability tier, Experimental, is independent of that classification.

# 6. System Architecture and Implementation Details

Conchordal is implemented in Rust to satisfy the stringent requirements of real-time audio (latency < 10ms) alongside heavy numerical analysis (auditory filterbank, NSGT and convolution). The architecture uses a concurrent, lock-free design pattern.

## 6.1 Threading Model

The instrument `conchordal` and the separate offline `conchordal-render` binary
support an optional excitation renderer.
Enable it at the top level of the TOML configuration:

```toml
render_prototype = true
```

Omission or `false` selects the legacy renderer. The setting is a boolean and has
no decay, rise, kick-gain or motion-scale subsettings. The instrument binary
`conchordal` exposes no audio-file output. In either execution path, natural Finish
closes excitation and waits for retained body radiation to retire. Offline rendering
then closes the WAV. Headless execution exits; the GUI waits for user exit by default
and feeds silence to the ring after Finish, or exits when `wait_user_exit = false`.
This is generator completion; no additional device or ring-drain wait is introduced.

`[audio].output_gain_db` applies fixed gain to the device copy and the offline
WAV copy immediately before their output guards. The multiplier is computed at
startup and applied in place without allocation. Habitat analysis, the modeled
ListenerTwin's presentation input, JSONL telemetry and UI meters retain the
original bus levels. The +4.5 dB default is calibrated from all 17 bundled top-level
samples at three seeds against the peak limiter's 0.98 ceiling, rounded down to
0.5 dB steps; `0.0` restores unity gain. This calibration covers those renders,
not arbitrary performances. Selecting `limiter = "none"` keeps the gain.

The application creates four primary thread contexts, plus the GUI event loop:

1.  **Audio Thread (Real-Time Priority)**:
    *   Managed by `cpal` in `audio/output.rs`.
    *   **Constraint**: Must never block. No Mutexes, no memory allocation.
    *   **Responsibility**: Pops mono samples from a lock-free ring buffer and copies them to all output channels. The output guard is selected by `[audio].limiter`: `peak-limiter` (default), `soft-clip`, or `none`. It is applied in-place on the interleaved output. SoftClip preserves samples up to its knee and compresses larger samples strictly below the ceiling; its small-signal gain is unity. Limiter meters aggregate maxima over each reporting interval.

2.  **Analysis Thread (Background Priority)**:
    *   Defined in `core/analysis_worker.rs`, running `AnalysisStream` from `core/stream/analysis.rs`.
    *   **Responsibility**: Receives habitat-bus hops (time-domain chunks), runs the auditory front end for subjective intensity, and computes *both* the Harmonicity field (Sibling Projection) and the Roughness field (ERB-domain convolution). NSGT power and spectral history run separately when the consumers listed in §3.1.3 need them.
    *   **Update Cycle**: When analysis is complete, it sends the updated Landscape snapshot back to the worker thread via a bounded SPSC channel.

3.  **Listener-Analysis Thread**:
    *   Runs the `ListenerTwin` perception pipeline on presentation-bus hops.
    *   **Responsibility**: Models what an audience member perceives—including its own habituation field and the perception meter's beat confidence—for the UI, the headless report, and the DCC pressure coupler.

4.  **Worker Thread (Simulation Loop)**:
    *   Named `"worker"` in `runtime/wiring.rs`; the loop lives in `runtime/worker.rs`.
    *   **Responsibility**: Runs the main simulation loop. Each iteration: merges analysis results into the current Landscape, dispatches Conductor events, advances the Population (pitch retargeting, articulation, metabolism), renders audio via `ScheduleRenderer` (which processes `PhonationBatch` vectors of `ToneCmd` and maintains the `Tone` pool), drives the production `MeterNetwork` from habitat flux and the population's own onsets, and pushes mono samples into the ring buffer for the audio thread.

5.  **App/GUI Thread (Main)**:
    *   Runs the `eframe`/`egui` visualizer.
    *   **Responsibility**: Handles user input, visualizing the Landscape (`ui/plots.rs`), and displaying simulation metadata. It receives `UiFrame` snapshots from the worker thread via a bounded channel.

## 6.2 Data Flow

To maintain data consistency without locking the audio thread, Conchordal uses a multi-channel update strategy for the Landscape. Rendered audio is split across two buses: the **habitat bus** (the environment the ecosystem senses) and the **presentation bus** (what the audience hears). A drone can be routed to the habitat bus only—shaping the landscape without being presented.

1.  The **Worker Thread** renders audio per bus and sends each habitat hop to the **Analysis Thread**; presentation hops go to the **Listener-Analysis Thread**.
2.  The **Analysis Thread** runs the auditory subjective-intensity + Roughness + Harmonicity pipeline, with NSGT spectral power enabled for the consumers in §3.1.3, and sends the resulting raw `Landscape` snapshot back.
3.  The **Worker Thread** merges the analysis result into the current `Landscape`, recomputes the base Consonance representations, advances the ecology's `HabituationField`, and writes the effective views. A separate habituation state is advanced on listener-analysis results.
4.  The **Worker Thread** drives the production `MeterNetwork` with the habitat onset flux (`DorsalStream`) combined with the phonation onset strengths of habitat-routed voices; the resulting `MeterState` is projected into `landscape.rhythm` via `NeuralRhythms::from_meter_state`.
5.  The `Community` evaluates the effective Landscape for pitch selection, metabolism, spawn, respawn, and Voice lifecycle.
6.  When DCC coupling is enabled, `tension_pressure = tension_level * coupling_strength` produces a bounded temperature bonus that feeds the Voices' pitch search. Listener tension already includes resolvability; the coupler applies no second factor. The default coupling strength is zero, so this path is behaviorally inert unless explicitly enabled.
7.  The `PhonationEngine` emits `ToneCmd` batches; the `ScheduleRenderer` creates, updates, or releases `Tone` instances accordingly and renders the selected body response into each bus (Section 5.1.3).
8.  Rendered presentation audio is pushed into a lock-free ring buffer consumed by the **Audio Thread**.

This decoupled architecture ensures that the audio thread always sees a consistent stream of samples, even if the analysis thread lags slightly behind real-time. The analysis thread processes all hops in-order to maintain causal auditory and spectral history.

## 6.3 The Conductor: Scripting with Rhai

The Conductor module acts as the interface between the human artist and the ecosystem. It embeds the [Rhai](https://rhai.rs/) scripting language, exposing a tiered API for controlling the simulation.

The API follows four explicit lifetimes: a **PopulationSpec** is a reusable pre-placement definition, a **Population** is the stable population identity created by `place()`, a **Voice** is one living member, and the runtime **Community** aggregates all Populations sharing the Landscape. The authoritative, always-current reference is the Script Reference book (`docs/rhai_book`, published under `/docs/rhai/`); this section summarizes the conceptual tiers only. There is no `Species` object; PopulationSpec is the configuration unit.

### 6.3.1 PopulationSpec Configuration

A PopulationSpec begins with a preset and is refined through method chaining. It contains founder Voice defaults together with population-level lifecycle, viability, and respawn policy:

**Presets**: `sine()`, `harmonic()`, `saw()`, `square()`, `noise()`, `modal()`. Assign an existing PopulationSpec to another variable and apply builders to derive a modified value.

**Body**: `amp(v)`, `freq(v)`, `brightness(v)`, `spread(v)`, `unison(n)`, `modes(pattern)`, `adsr(a,d,s,r)`, `send(bus)` (habitat/presentation routing).

**Pitch**: `anchor()` (hold the voice at its pitch; implied by `freq(hz)`), `seek_consonance()` (climb the consonance terrain; the apply mode is then resolved automatically—sustained voices glide, re-attacking voices snap at onsets—unless overridden via `pitch_apply_mode("gate_snap"|"glide"|"rearticulate")`), `pitch_core("hill_climb"|"peak_sampler")`, `glide(v)`, `landscape_weight(v)`, `neighbor_step_cents(v)`, `tessitura_gravity(v)`, `temperature(v)` (one search-stochasticity knob shared by both cores; 0 settles greedily — this *is* the movement-tension knob: a hot search keeps a voice restless and straying off consonance, cooling it resolves), `move_cost(v)`, `proposal_interval(sec)`, `global_peaks(n)`, `ratio_candidates(n)`, plus peak-sampler knobs (`window_cents`, `top_k`, `sigma_cents`, `random_candidates`).

For a free-pitch Hold Voice without participation, `pitch_apply_mode("rearticulate")` with the
boundary omitted or zero couples each changed adopted target to one onset.
At its sample time the old Tone begins release and a new Tone attacks at the
new pitch. A positive `rearticulate_above(cents)` measures net displacement from
the last actual onset: strictly above the boundary attacks, at or below it glides
with `glide(tau)`, retaining that selection until the next adoption. Exact
Log2Space centers use bin-distance in cents; other endpoints use continuous log2
distance without rounding or epsilon. Earlier legal attacks in the same output
hop provide the reference, while cancelled requests do not update actual onset
memory. The boundary is a composer parameter; it defaults to zero, sanitizes
negative or nonfinite values to zero and imposes no upper limit. Outside
Rearticulate, a call warns and is ignored. Live boundary changes cause no attack
and affect the next adoption. Replacement Voices inherit the boundary, not the
parent's last-onset pitch. Anchored Voices remain locked.

Unchanged targets, rejected proposals and steps without proposals create no
extra onsets. Glide retains legato movement. Live mode changes preserve the
sounding Tone, create no onset by themselves and cancel unrendered articulation
requests. Free-pitch Gated Rearticulate is a script error at final placement,
independent of builder order, and before live writes. Use Hold with sustain, or
Glide/GateSnap for Gated phonation. Attacks and release tails follow the existing
synthesis, energy and habitat-perception paths.

Explicit `sustain()` with `entrained()`, `steady()` or `flow()`, in either builder order,
combines sustained sound and participation opportunities. Waiting or skipping
retains the Tone. The omitted mode is Glide: sounding releases the old Tone and
attacks at that sample's body pitch. In entrained/flow, a pitch adoption offers one legal candidate
using the unchanged costs, tie order and committed plans; unselected movement
glides, and coincident adoption and clock onset emit one attack. Steady offers no
adoption-time candidate and retains only its due-time opportunity.

Explicit GateSnap with free pitch instead holds one adopted target. Later
adoptions replace it; only a sounded participation opportunity consumes it,
releases the old Tone and attacks at the target pitch. With no pending target,
the current pitch is re-attacked. Waiting and skipping retain pitch and pending
target, without sliding or expiry. This mode offers no extra adoption-time
candidate and leaves participation costs and skip unchanged. At the same sample,
adoption precedes opportunity; a future adoption cannot affect an earlier onset.
Release, death and removal cancel pending targets. Each emitted attack enters
energy and acoustic paths once. Prediction retains the existing finite
representative footprint.

With sustained participation, explicit Rearticulate uses the last actual onset
pitch to classify each adopted target. At or below a positive distance boundary,
it cancels any old pending target and starts gliding immediately, without an
attack. Above the boundary, it stops the glide, holds body pitch and replaces the
single pending target. Only a sounded opportunity consumes that target, jumping
and attacking there; without a target it attacks at current body pitch. That pitch
can differ from the preceding onset during a glide. Both attacks update actual
onset memory. Waiting and skipping retain sound and pending state while existing
glides continue. Same-sample adoption precedes opportunity; no adoption-time
candidate is added, and the boundary does not filter clock attacks. Zero or omission
matches GateSnap, including zero-distance adoption. Without participation,
Rearticulate attacks at adoption; with participation, it attacks at sounded opportunities.

Live Glide ↔ GateSnap ↔ Rearticulate changes cancel pending attacks, selected glides
and unrendered control records, retaining the Tone and the controller's target.
Returning to Glide follows that target at the next control update; GateSnap and
Rearticulate wait for later adoptions. Repeating the same mode retains pending
state. A boundary change retains the selected behavior until the next adoption.
Replacement Voices inherit the resolved controls with fresh performance state.
Implicit WhileAlive or `while_alive()` alone does not select sustained participation.
Single presets and sustain with metric, pulse or repeat retain their existing
meaning. Participation presets remain initial-only.

With `rest()`, explicit sustained participation also retains an explicit finite
duration from `cycles(n)` or `adaptive_duration()` and its modifiers, independently
of builder order. Each sounded note ends with its scheduled normal Off while the
Voice remains alive. Tails are retained, and new attacks are prohibited until the
renderer confirms completion of all Tones with the same source ID and generation.
From that completion, `rest()` waits one intrinsic bodily period; `rest(periods)`
waits the composer's multiple of the period measured then. With a known period,
zero waits only for tail completion. Unknown periods or unfinished tails keep
waiting without a fallback or timeout. Negative and nonfinite values sanitize to
zero, with no upper limit on finite nonnegative values.

The end of the rest permits participation but emits no forced onset. Existing
candidates, costs and skip rules are retained, and intentional suppression is not
learned as a cost-based skip. GateSnap retains the latest pending target through
sounding, tails and rest until the next sounded opportunity; Glide does not update
released tails. Rearticulate processes adoptions through release and rest using
the same small-target cancellation and large-target deferral. The last actual
onset remains its reference until the next attack; body glides never retune
released tails. Lifecycle endings take precedence. `rest` and finite duration are
initial-only controls, inherited by replacements with fresh performance state.
Missing finite duration or timing outside explicit sustain plus entrained/steady/flow
is a placement error before writes. All three pitch application modes are available. Explicit `while_alive()`
clears finite intent. Absence of `rest` retains existing behavior. Every note in
this form has a finite end; per-note selection of sustain versus release is not
implemented.

Explicit `glide`, `pitch_core`, and `viability_scope` settings take precedence over preset defaults regardless of builder order. Enum parameters accept only their exact canonical names; invalid names are positioned errors, and ineffective settings produce positioned warnings.

**Crowding**: `avoid_neighbors(strength)` (default sigma) / `avoid_neighbors(strength, sigma_cents)`, `crowding_target(same, other)`, `leave_self_out(bool)`.

**Brain/Phonation**: `brain("entrain"|"seq"|"drone")`, `sustain()`, `repeat()`, `once()`, `pulse(rate)`, `pulse_lock(depth)`, `social(coupling)`; duration via `while_alive()`, `cycles(n)`, `adaptive_duration()`, `duration_range(min,max)`, `duration_curve(k,x0)`, `shorten_on_drop(gain)`; finite sustained participation with `rest()` / `rest(periods)`.

**Rhythm (Section 5.4)**: `metric()` selects shared-beat attraction; `entrained()`, `steady()` and `flow()` select acoustic participation. `entrainment(v)` changes the influence within that intention. `microtiming(v)` offsets the synchronized target only. `temporal_basin` shapes the production meter and initializes metric bodies at its center; `intrinsic_rate(hz)` sets the separate founder bodily scale of entrained/steady/flow, otherwise 2 Hz, and changes note lengths and rests in seconds together. Articulation controls remain separate: `rhythm_freq(v)`, `rhythm_coupling_vitality(lambda_v, v_floor)`, and `rhythm_reward(rho_t, "attack_phase_match")` (Entrain Hold with autonomous attacks enabled).

**Lifecycle/Viability**: `endurance(sec)`, optional `recovery(sec)`, `attack_cost_fraction(v)`, `attack_recharge_fraction(v)`, `consonance_viability(low, high)`, `dissonance_penalty(v)`, and `phonate_when_viable()` (withhold the first onset until the viability window opens).

**Respawn**: `respawn_random()`, `respawn_hereditary(sigma_oct)`, `respawn_consonance()`, `respawn_capacity(n)` (maximum living membership; founder count by default and never lower than it), `respawn_settle(placement)`, `respawn_min_consonance(v)`, `respawn_background_death_rate(v)`.

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

*   **Harmonic terrain**: `set_roughness_aversion(w)`, `set_pitch_objective("consonance"|"dissonance")`. Both change valuation, not the sensed $R_{01}$ / $H_{01}$.
*   **Temporal terrain**: `meter_stability(v)`, `temporal_basin(min_hz, max_hz)` (Section 4.4).
*   **Interaction**: `set_global_coupling(v)` scales agent interaction strength.

`harmonic_rhythm(N)` is an optional scene-wide initial setting for coordinating background changes. For positive integer N, one shared clock offers every Nth beat crossing to all Flora and eligible held, free-pitch Voices using rearticulation. Existing pitch selection and local readiness still decide whether to change; release and loss of support take priority. The clock follows completed habitat-audio observations and continues from its tempo prior while observations are unknown. Omission or 0 preserves the previous behavior. This is an **author rule** about change timing, with N chosen by the composer; it supplies no keys, chord tables or progressions, and makes no claim to a perceived bar length.

**Scenario Parsing**: Scenarios are loaded from `.rhai` files. This separation allows users to compose the "Macro-Structure" (the narrative arc, the changing laws of physics) while the "Micro-Structure" (the specific notes and rhythms) emerges from the agents' adaptation to those changes.

# 7. Case Studies: Analysis of Emergent Behavior

The following examples, derived from the `samples/` directory, illustrate how specific parameter configurations lead to complex musical behaviors.

## 7.1 Case Study: Three Timing Intentions (`samples/07_heartbeat.rhai`, `08_murmuration.rhai`, `09_rain.rhai`)

The three samples exercise the separate intentions described in Section 5.4.

1. **Metric** (`07_heartbeat.rhai`): explicit synchronization to the production beat. The dedicated downbeat Voice remains because synchronization is the subject of the demo.
2. **Entrained** (`08_murmuration.rhai`): bodily participation using acoustic context. Gated attacks use ordinary consonance recharge, without a phase-match reward.
3. **Flow** (`09_rain.rhai`): clustered bodily renewal with weak acoustic participation. Pitch still follows the consonance field.

Samples 08 and 09 have no dedicated synchronization accompaniment.

## 7.2 Case Study: Drift and Flow (`samples/research/drift_flow.rhai`)

This compact research fixture demonstrates live Population patches and repeated field-relative placement; it does not demonstrate autonomous drift.

1.  **Explicit patch**: A sustained anchor enters at C2 (65.41 Hz), while a second sustained Voice enters at C#3 (138.59 Hz). The script then patches that Voice directly to 220 Hz with `population.freq(220.0)` and releases it.
2.  **Field query**: Five one-Voice swarms are placed in sequence with `consonance(130.0).peak().range(1.0, 4.0)`, demonstrating repeated deterministic queries against the live field.
3.  **Terrain change**: The anchor is finally patched to F2 (87.31 Hz), changing the field after those placements.

Autonomous settling is demonstrated by `samples/05_settling.rhai`.

## 7.3 Case Study: Emergence and Resolution (`samples/12_emergence_and_resolution.rhai`)

The closing integration sample combines a habitat-only harmonic root, a metric colony, and a temporary non-metric flow Population. It has no dedicated beat-carrier Voice. The colony's own onsets drive the shared production meter, while the hidden sustained root supplies a pitch reference to the Landscape.

The colony enters at 2.3 s. At 11.7 s, the Scenario raises its search temperature and shifts the root upward; flow enters at 15.0 s. At 20.3 s, the Scenario cools the search, explicitly enables Glide, returns the root, and reduces flow amplitude. Flow is released at 23.6 s, followed by a further colony amplitude reduction at 24.9 s and the remaining releases. These are scheduled macro operations; individual pitch searches, onsets, viability, and consonance-biased respawn remain local processes.

“Tension” and “resolution” name the intended arc, not a verified perceptual outcome. The resulting harmonic development and closure require listening judgments for the rendered conditions.

# 8. Conclusion

Conchordal establishes a foundation for Bio-Mimetic Computational Audio. By replacing the rigid abstractions of music theory (notes, grids, BPM) with continuous physiological models (`Log2Space`, ERB bands, neural oscillation), it creates a system where music is not constructed, but grown.

The paper "Conchordal: Emergent Harmony via Direct Cognitive Coupling in a Psychoacoustic Landscape" (arXiv:2603.25637) validated the psychoacoustic landscape as an effective ALife terrain through controlled experiments demonstrating self-organization, selection, synchronization, and hereditary accumulation. These results confirm that the Roughness-Harmonicity-Consonance pipeline and the Kuramoto entrainment model produce musically coherent emergent behavior under a range of initial conditions.

Version 0.4.0 integrated the paper findings into the instrument and added the temporal axis of the architecture: the fixed rhythm filterbank was replaced by an emergent meter (a forced limit-cycle oscillator with corroborated acoustic recurrence and phase/pattern confidence), voice timing was expressed as synchronized (`metric`) or acoustic-participation (`entrained`, `flow`) intentions (Section 5.4), and the composer's temporal control was reduced to terrain priors (`meter_stability`, `temporal_basin`) that shape where a pulse forms without ever scheduling one. A dual-bus design separated the habitat (what the ecosystem senses) from the presentation (what the audience hears). Landscape habituation is enabled by default. A simulated `ListenerTwin` feedback path can raise pitch-search temperature; its coupling remains zero by default.

Version 0.5.0 expands sustained articulation, unifies pitch-movement and environment-viability self-exclusion, and improves scenario diagnostics, shutdown, and perception processing.

The technical architecture—anchored by the `Log2Space` coordinate system and the "Sibling Projection" algorithm—provides a robust mathematical foundation for this paradigm. The use of Rust ensures that these complex biological simulations can run in real-time, bridging the gap between ALife research and performative musical instruments.

Chapter 9 closes the document with a condensed ledger of the Manifesto's commitments: what is discharged and what remains open. The design decisions behind each row, and what the implementation taught back, are kept in the repository's design ledger rather than in this note.

# 9. Manifesto Correspondence and Open Problems

The Manifesto declares commitments; this chapter records which of them the current implementation discharges and which remain open. It describes the instrument as built. The design decisions, numerical registrations and revision history behind each row are kept in the repository's design ledger, [`docs/design-notes/technote-ledger.md`](https://github.com/ktakahashi74/conchordal/blob/main/docs/design-notes/technote-ledger.md), which preserves the former numbering of this chapter (§9.2–§9.4).

## 9.1 The Ledger

| Manifesto commitment | Mechanism | Section | Status |
| :--- | :--- | :--- | :--- |
| Generation without symbolic intermediaries | `Log2Space` + landscape; no note names, scales, or time signatures anywhere in the engine | §2–4 | Implemented |
| Frequency-axis terrain (cochlea/brainstem models) | Roughness, Harmonicity, Consonance kernels | §3 | Implemented |
| Temporal-axis terrain (neural oscillation) | Emergent meter: forced limit cycle, default onset recurrence or optional continuous T2 recurrence for tempo adaptation; fast input updates phase and intensity, and confidence does not certify phase-origin accuracy. `metric` keeps explicit synchronization; `entrained`, `steady` and `flow` use a bounded acoustic observer and a bodily participation policy; steady fixes due and chooses sound or skip | §4, §5.4 | Partial. Persistence of temporal relations across all conditions is open; measure-accent coupling is opt-in |
| Landscape variability (culture, individual, unknown principles) | consonance kernel coefficients, `set_roughness_aversion`, default-enabled habituation erosion | §3.4–3.5, §6.3.6 | Partial. Cultural tuning systems are not yet absorbed |
| Adaptation and expectation | Per-Voice `AdaptationContext`, ecology/listener `HabituationField`, short-term recurrence and energy forecasts feeding habitat participation | §3.5, §5 | Partial. Relational memory, phrase/scene expectation and action-value learning are open |
| Acoustic life: perception, metabolism, autonomy | The Voice: articulation-life cores, normalized energy, time-domain endurance/recovery, viability | §5 | Implemented |
| Population: niches, symbiosis, terrain deformation | Crowding, respawn, the closed loop | §5 | Implemented |
| No central conductor | Local perception only; one shared meter reads habitat recurrence and fast phase/intensity evidence, with optional continuous T2 recurrence | §4–5, §7 | Implemented. Dedicated beat carriers appear only in explicit synchronization demonstrations (sample 07) |
| Scenario as macro direction | Director terrain operations | §6.3.6 | Implemented as authorial composition |
| Temporal structure grounded in auditory cognition | Default-disabled `temporal_meter_t2` feeds whole-habitat continuous T2 recurrence into the production meter; group accents and Periodic arrival costs have separate default-disabled settings | §5.4 | Partial. Continuous recurrence, accent and arrival mechanisms are implemented. Production recurrence does not require group assignment and retains causal support and explicit unknowns; group arrival retains owner credit. Group-event numerical parameters have no defaults; an omitted arrival weight uses 1. Phrase perception and harmonic rhythm remain open |
| Authored spectral ecology | Flora consumer–resource dynamics, default own-PCM exclusion for C/H, beat/subdivision allocation and amplitude targets, ecological pace, lane pitch modes and optional habituation-driven rest | §5.6 | Implemented as author rules; the Rhai API is Experimental |
| Authored temporal behavior | Voice span/gap control from heard attacks and private recovery; environmental-level dwell changes pitch-search timing | §5.4 | Implemented as author rules, not phrase perception or tonal-stability principles |
| DCC stage two: biosignal closed loop | `ListenerTwin` pressure can feed pitch-search temperature when `[dcc]` coupling is enabled | §4.1, §6.2 | Simulated loop implemented, off by default; the physical biosignal loop is open |
| Cognition coupled to the sound actually presented | Separate presentation analysis; missing hops invalidate observations and suspend DCC pressure until a complete window is available | §6.2 | Implemented. Physical-device overload validation is open |
| Excitation and body radiation | Optional excitation renderer in the instrument and offline binary: immediate Sine/Harmonic response and Modal free decay, with separate Off and disposal | §5.1, §6.1 | Implemented as an opt-in; the default remains the legacy renderer |
| Music as a living performance | The instrument exposes no audio-file output; the separate `conchordal-render` binary supports offline study | §6 | Implemented as a binary boundary |
| Dissolution of roles; spatial landscapes; heredity of timbre; other domains | Hereditary respawn exists as assays | — | Horizon |

## 9.2 Open Problems

- Cultural tuning systems and individual hearing profiles as landscape parameters.
- Persistence of temporal relations across performance conditions for `entrained` and `flow`.
- Phrase perception, section structure and long-context memory. Authored rest and dwell do not implement these mechanisms.
- Harmonic rhythm as a mechanism coordinating pitch changes with metrical structure. Existing note-boundary gates and Flora lane opportunities do not implement it.
- Cognitive retention and interference, and stream identity in overlapping sound.
- The physical biosignal loop of DCC stage two.
- Real-time acceptance under device overload and large populations.
- Native Modal remains unsupported for body-aware birth/metabolism and representative-onset footprints. Legacy analytic Tone-energy forecasts remain unavailable in the excitation renderer.

# Appendix A: Key System Parameters

| Parameter | Module | Unit | Description |
| :--- | :--- | :--- | :--- |
| `bins_per_oct` | `Log2Space` | Int | Resolution of the frequency grid (typ. 48-96). |
| `sigma_cents` | `HarmonicityParams` | Cents | Width of harmonic peaks. Lower = stricter intonation. |
| `roughness_k` | `LandscapeParams` | Float | Saturation parameter for roughness mapping. Default: $(1/0.7) - 1 \approx 0.4286$ (so $x=1$ maps to $\approx 0.7$). |
| `kernel.a` | `ConsonanceKernel` | Float | Harmonicity coefficient (default 1.0). |
| `kernel.b` | `ConsonanceKernel` | Float | Roughness coefficient (default -1.35; negative penalizes roughness). |
| `roughness_aversion` | `LandscapeParams` | Float | Valuation weight on the roughness terms $b$, $c$ and density $\rho$; set by `set_roughness_aversion`. Default 1.0. |
| `kernel.c` | `ConsonanceKernel` | Float | Interaction coefficient (default 1.0; positive attenuates roughness penalty at high harmonicity). |
| `kernel.d` | `ConsonanceKernel` | Float | Bias term (default 0.0). |
| `beta` | `ConsonanceRepresentationParams` | Float | Sigmoid steepness for $C_{level01}$ (default 2.0). |
| `theta` | `ConsonanceRepresentationParams` | Float | Sigmoid threshold for $C_{level01}$ (default 0.0). |
| `consonance_density_roughness_gain` | `LandscapeParams` | Float | $\rho$ in density kernel $H(1-\rho R)$ (default 1.0). |
| `habituation.enabled` | `HabituationParams` | Bool | Enable Landscape erosion; configuration default true. Explicit false makes effective views bit-identical to base views. |
| `habituation.satiation_sec` | `HabituationParams` | Seconds | Time to reduce the discrepancy from a sustained-drive equilibrium by 90% (default 5.0). |
| `habituation.recovery_sec` | `HabituationParams` | Seconds | Time to recover 90% after drive withdrawal (default 8.0). |
| `habituation.ref_drive` | `HabituationParams` | Float | Half-saturation drive in the habituation transfer function (default 0.25). |
| `loudness_exp` | `LandscapeParams` | Float | Compressive exponent applied to the A-weighted critical-band context power in the subjective-intensity front end (§3.1.2; default 0.23). |
| `tau_ms` | `LandscapeParams` | Milliseconds | S6 subjective-intensity smoothing time constant (`[analysis].tau_ms`). |
| `stability` | `MeterShaping` | 0.0-1.0 | Beat attractor depth (`meter_stability`): scales entrainment forcing and tempo learning. |
| `basin_hz` | `MeterShaping` | Hz pair | Scene tempo region (`temporal_basin`): seeds and confines beat-frequency learning; its center initializes metric bodies independently of stability. |
| `coupling` | `CoupledTimingSpec` | 0.0-1.0 | Acoustic-context influence for participation; shared-beat attraction for synchronization (`entrainment`). |
| `flow_depth` | `CoupledTimingSpec` | 0.0-1.0 | Renewal clustering of free-running onsets (0 = regular). |
| `steady` | `CoupledTimingSpec` | boolean | Non-flow participation with fixed intrinsic period and due as its only candidate. |
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

$$
d_i=\frac{L_iP_i}{L_iP_i+r_{ref}}, \qquad
h_i(t+\Delta t)=a h_i(t)+(1-a)d_i, \quad a=e^{-\Delta t/\tau_{rise/fall}}
$$

$$
C_{score,i}^{eff}=\theta+(C_{score,i}-\theta)(1-h_i), \qquad
C_{density\_mass,i}^{eff}=C_{density\_mass,i}(1-h_i)
$$

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

$$
b = \frac{1}{T_e(1+p_d)}, \qquad
\Delta E_{basal} = -b\,[1+p_d(1-C_{level01}^{eff})]\,\Delta t
$$

$$
\Delta E_{recovery} = \frac{V(C^{eff})}{T_r}\,\Delta t, \qquad
\Delta E_{attack} = -f_c + f_r C_{level01}^{eff}m_r
$$

where $T_e$ is endurance, $T_r$ is recovery, $p_d$ is the dissonance penalty,
$V(C^{eff})$ is the viability-window signal, $f_c/f_r$ are attack cost/recharge
fractions, and $m_r$ is the bounded rhythm-reward multiplier.

**Harmonicity Projection (single-path Sibling Algorithm):**

$$ Roots[i]=\left(\sum_k k^{-\rho}A[i+\log_2(k)]\right)^{\gamma} $$

$$
H[i]=\max\left(0,\;\sum_m m^{-\rho}Roots[i-\log_2(m)]
-(1-w_{diag})\left(\sum_{k=1}^{K}k^{-2\rho}\right)A[i]\right)
$$

The implementation then applies the optional absolute-frequency TFS gate and, by default, peak-normalizes the scan.

**Roughness Convolution:**
$$ R_{shape}(z) = \int A(\tau) \cdot K_{rough}(z-\tau) d\tau $$
