# York Minster research room

The room approximates one source under the tower and one receiver in the
nave, 23.5 m apart. The 48 kHz FIR contains a unit impulse at sample zero and
a synthetic reflected field. Call `york_response()` at preparation and pass the
taps to `RoomKernel::new`. Habitat, self-returns and left presentation share this plan. Right presentation has its own FIR plan. For off, bypass the
kernel and allocate zero slots. Strict W/W reserves three shared buses plus one stream per simultaneous
(kind, ID, generation) read by self-sound, LOO or metabolism. The score scan
bounds live placements through release and adds overlapping predecessors.
For a regular synchronized fixed-gate clock with a bounded production beat,
each onset has an absolute Off deadline. A population with retirement bound D
hops needs at most D+1 cohorts of successors, capped by total possible births
in a finite score. A population release stops births and flushes retiring
notes. An old Tone can keep sounding on the shared buses after its generation
loses its last reader. Unknown gated deadlines use the finite-score bound.
The pool is prepared before opening WAV/JSONL. An unexpected shortage ends
the run before publishing that hop, preserving earlier complete hops.

The score can select the room with `room("off")`, `room("presentation")` or
`room("both")`. Its last finite declaration applies to the whole performance.
Otherwise `[room].mode` supplies the mode, with dry mono as the default.
The two wet modes require a 48 kHz output and use stereo presentation.
Without explicit sites, the audience in `presentation` hears every source at
12 m, the distance of the default placement (0, 12). Each ear receives a
unit direct tap plus its reflected response scaled by 12/23.5. This gives
an impulse direct-to-reverberant ratio of +0.68 dB. The habitat stays dry.
Without sites, `both` retains the measured-distance balance at 23.5 m
(−5.16 dB). With explicit sites the direct level depends on distance from
each source, and the audience reflection send remains 12/23.5.
For the same emitted signals, placing all sources at (0, 12) gives the
same audience balance as a score without sites in `presentation`. Explicit
sites also activate local ecological observations, which can change what
the sources play.
Only `both` applies the response to the habitat and needs strict source histories.
The startup diagnostic identifies the mode source and whether self-return is active.

## Measurement and attribution

The published octave T30 values come from the [University of York measurement
record](https://www.space-net.hosted.york.ac.uk/node/54/index.html).
The [OpenAIR York Minster recording](https://www.openair.hosted.york.ac.uk/?p=797)
is credited to openairlib.net, Audiolab, University of York, and Damian T.
Murphy, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The measurement was made on 13 January 2006. The W channel of
`Minster1_000_WX_48k.wav` supplies two scalar parameters. The crate and its
distribution contain zero recorded samples.

Run `python3 tests/research/room_measurement.py --bands` for an in-memory,
SHA-256-checked download, or supply a local WAV path. The direct window is
[p−120, p+121) around the absolute peak (p=3223). The next window ends at
p+3840 and the late window at p+351360, both exclusive. The first sample
outside the direct window reaching 10% of the direct peak occurs at offset
121 (2.521 ms). This is a window- and threshold-defined onset. The measurement
gives D_E = −5.161637 dB and C80 = −2.414685 dB. The published T30 table does
not specify a microphone channel. A fit to the single W channel can differ
from the published values, especially near the recording noise floor.

## Left response synthesis

The response has 351,360 taps (7.320 s), 1,405,440 bytes as f32, zero bytes of
coefficient files. Ten independent octave carriers start together at sample
121, with equal initial mean-square amplitudes, a common 96-sample half-cosine
attack, and amplitude envelopes whose per-sample ratio is
`10^(-3/(48000*T30))`. The envelope continues through 80 ms. Labels E and L
split the resulting reflection for analysis at sample 3840.

In ascending centre frequency and then ascending bin index, PCG XSL RR 128/64
(initial state 17, increment 1) supplies the high 53 bits for a uniform phase.
The 524288-point spectra have unit magnitude on each band's positive bins and
conjugate negative bins. Internal edges are geometric means of adjacent
centres, outer edges are 31.5/√2 and 16000√2 Hz. Rectangular windows include
their lower edge and exclude their upper edge. The inverse transform is scaled
by 1/M. Multiplication by M/√(2 × positive bin count) gives each complete
carrier unit RMS. Normalize the combined reflected energy to 10^(5.16/10),
then round each tap once to f32. The seed, arithmetic order and coefficient
hash are fixed in the Rust acceptance test.

The stream uses fixed-order f32 radix-2 butterflies with f64-computed twiddles
rounded to f32, half-spectrum history and uniform overlap-add. Add the direct
tap in the time domain. The last nonzero input at s has exclusive end s+N.

## Right presentation response

The receiver uses the three-dimensional diffuse-field head model of Jeub,
Dörbecker and Vary (2011), IEEE Signal Processing Letters 18, pp. 197–200.
Their two circular plates model head shadowing, and equation (22) retains the
sign of the cross spectrum. Section VI compares the model with artificial-head
measurements after removal of the direct sound and early reflections. Applying
this model to every reflected sample is a synthesis choice. The direct tap is
identical at the two ears. [Original paper, equations (15)–(22) and section VI](https://www.iks.rwth-aachen.de/fileadmin/publications/jeub11.pdf).

Use the experimental geometry in section VI: plate radius R = 0.075 m, plate
spacing d = 0.15 m and microphone spacing D = 0.17 m. Sound speed c = 340 m/s
comes from the authors’ reference calculation. Set k = 2πf/c, u = cos θ,
g = (D−d)/2 and l = √(r²+g²). With the common propagation factor cancelled,
the azimuth integral in (18) gives the Bessel function J₀. The transfer
functions and signed coherence reduce to

```text
H₂ = exp(i k D u/2)
H₁ = conj(H₂) − ½ exp(−i k d u/2) ∫₀ᴿ exp(−i k l) J₀(k r √(1−u²))
                           × [i k (u + g/l) + g/l²] r/l dr
γ  = 2 ∫₀¹ Re(H₁ conj(H₂)) du / ∫₀¹ (|H₁|² + |H₂|²) du
```

The cos θ substitution incorporates the solid-angle weight sin θ in (22).
The generator in `tests/research/room_coherence.py` uses order-64 Gauss–Legendre
quadrature on both coordinates. It checks convergence against order 96 and
midpoint interpolation error below 0.001 on the 25 Hz grid from 0 to 24 kHz.
The 961 little-endian f64 grid values occupy 7,688 bytes. Runtime preparation
linearly interpolates this table at each positive FFT bin. [Authors’ reference implementation](https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/publications/binaural_coherence_v1.zip).

For each occupied positive bin, regenerate the left unit phasor A from state
17 and draw B from a second PCG with initial state 31, the same transition
and the same band/bin order. Write γA + √(1−γ²)B, then its conjugate at the
negative bin. Apply the left transform, envelope, taper and total reflected
energy normalization independently to the right, then round to f32. The
native right tap SHA-256 is
`c06ba2ee89027a7fd6700ac3fb18495e8d7506b6d6b36b4f9702f39117f482e1`.
Both responses have their first reflection at 121 and their last tap at 351359.
Right D_E is −5.1600 dB and C80 is −2.3856 dB. On the late tail, octave T30 ranges
from 1.0256 to 7.2619 s and the minimum regression R² is 0.991929. The native
acceptance test applies the same decay and energy bounds to both ears.
The published room quantities describe a single receiver. Interaural coherence
is imposed by the head model.

## Frozen left response at 48 kHz

The native SHA-256 of little-endian f32 taps is
`46355804385b9ffbd73a6852b9fe3e2998574be6c42babcc63862f84c8d2710f`.
D_E is −5.1600 dB. C80 is −2.4290 dB (−0.0143 dB relative to W).
The first 10 ms of reflection has RMS 0.0103723. The full-response peak
magnitude is 42.6740 (+32.60 dB) on a 1,048,576-point FFT grid.

| Centre (Hz) | Published T30 (s) | Synthetic L (s) | L R² | Full FIR (s) | Measured W late (s) |
|---:|---:|---:|---:|---:|---:|
| 31.5 | 5.90 | 6.6092 | 0.981005 | 7.3714 | 10.2740 |
| 63 | 6.88 | 7.4028 | 0.995340 | 7.3556 | 8.5036 |
| 125 | 7.24 | 7.1692 | 0.998546 | 7.1433 | 8.4013 |
| 250 | 7.22 | 7.2727 | 0.997686 | 7.3086 | 8.4372 |
| 500 | 6.92 | 6.9582 | 0.999695 | 6.9866 | 8.4389 |
| 1000 | 6.10 | 6.1587 | 0.999622 | 6.1358 | 7.7472 |
| 2000 | 4.75 | 4.7979 | 0.999670 | 4.7566 | 6.0563 |
| 4000 | 3.26 | 3.2795 | 0.999901 | 3.2674 | 3.8061 |
| 8000 | 1.80 | 1.8220 | 0.999823 | 1.8096 | 1.7548 |
| 16000 | 1.03 | 1.0383 | 0.999886 | 1.0413 | 33.4766 |

The W-channel fit at 16 kHz has R² = 0.812. These recording fits describe
the finite W-channel window with its noise included. The synthesis targets
are the ten published values.

## Reproduction

The Rust `impulse_statistics` test evaluates the rounded taps independently
with rustfft in f64, zero-padding to the first power of two at least twice the
signal length. It applies the same defined frequency windows, integrates band
energy backwards over the finite signal, and fits −5 to −35 dB by ordinary
least squares. Its output includes the half-open regression sample ranges,
residual RMS in dB, and the full-response fits. Peak frequency response is the
maximum on this FFT grid. The onset RMS window is 480 samples starting at 121.

For relative costs build the ignored `core::room::tests::room_cost` library
test in release with `profile-alloc`, then run it alone with `--ignored
--exact --nocapture`. Run the instrument headless on
`tests/research/room_dry_load.rhai` with a profile and report, fixed 48 kHz and
512-sample hops. Compare nine-stream p99 with the hop budget minus the dry
worker p99. The following example reserves common output headroom for dry and wet research renders.

```toml
[audio]
output_gain_db = -36.0
```
