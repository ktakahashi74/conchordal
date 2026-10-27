# ME-1 numerical reference

The candidate is the text-consistent stationary AR(1) in von Hippel (2000),
*Redefining Pitch Proximity*, Music Perception 17(3):315–327, Eq. 1 and p.324.
The printed predictive distribution omits the mean offset and scale. We use
`mu = m + r*(last-m)` and `variance = s²*(1-r²)`, explicitly as an algebraic
interpretation, not an authorized erratum. Corpus fits are not fixtures here.

`python3 reference.py` regenerates 18 cases using Python Decimal at precision 80.
The reference integrates the Gaussian power series independently of Rust/libm.
Cases cover translation, doubled range scale, zero/negative/near-one correlation,
central bins and 7–8 sigma tails. Production uses only mean, variance and log-density
differences. Tests evaluate bins with libm erfc, retaining relative tail accuracy.

The equation assertions allow binary64 rounding scaled to mean and input variance.
CDF relative bounds include input-boundary rounding and standardization sensitivity,
`64*epsilon*(1+abs(mu)/sqrt(variance))`, for this registered bounded matrix. These
are numerical implementation checks, not a new musical tolerance. Prefix tests
compare online moments with a separate batch Pearson calculation, with error
scaling to operation count. Outward-rounded moment intervals, rather than that
comparison tolerance, determine runtime support. Three notes or fewer and longer
exactly affine adjacent sequences are unknown; near-perfect resolvable sequences
remain supported.

Consumer tests use the independent algebraic prefixes A=[60,60,62,60] and
B=[64,60,62,60], identical last two notes, candidate 60/62, baseline 60, flat field,
weight 1 and zero effective temperature. A chooses 60 and B chooses 62. The f32
rounding error must not bridge the existing improvement threshold 0.1. Native
proposal tests check replacement before peak extraction, legal mapping and
independent ME/SQ-1 numerical fallback. Actual phonation tests check raw Hz input,
report independence and same-hop causal ordering.

Figure 7's reported 19%/18% is a separate, unreproduced result: the paper does not
specify the finite-prefix variance/autocorrelation estimator. No estimator or free
coefficient is adjusted to fit it. This does not stand in for the equation checks.
