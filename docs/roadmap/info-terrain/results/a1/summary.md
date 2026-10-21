# Function A measurements

Code: `i11-stage1-baseline-32-g4e93573-dirty`. Registration: draft 5, including the token erratum.

Habitat analysis, [2,20) s, soundgen 3.0.0 aggregation; values are mean surprise / mean precision.

| Stimulus | Original | Extended |
|---|---:|---:|
| 1_sine | 0.511351 / 1.000000 | 0.038964 / 0.999768 |
| 2a_white | 0.543955 / 0.847436 | 0.503311 / 0.894259 |
| 2b_white_quiet | 0.543955 / 0.847436 | 0.503311 / 0.894259 |
| 3_pink | 0.574631 / 0.846355 | 0.501690 / 0.937195 |
| 4_periodic | 0.538939 / 0.511119 | 0.389037 / 0.603532 |
| 5_deviant | 0.722014 / 0.493280 | 0.367809 / 0.511096 |
| flora_sine_residual | 0.809719 / 0.891942 | 0.419402 / 0.865499 |
| flora_sine_only | 1.564376 / 0.977487 | 0.267866 / 0.943610 |
| flora_resonant_residual | 0.833656 / 0.856285 | 0.431811 / 0.861380 |
| 03_gravity | 1.239932 / 0.806488 | 0.324937 / 0.970789 |
| 09_rain | 0.736490 / 0.831609 | 0.365905 / 0.866340 |
| 12_emergence_and_resolution | 0.957415 / 0.799060 | 0.404524 / 0.883789 |

External comparison: Pearson r [95% paired cluster percentile CI].

| Version | Soundgen primary | OSF auxiliary |
|---|---:|---:|
| R-S | 0.532489 [0.377179, 0.664132] | 0.598206 [0.456083, 0.719536] |
| R-O | 0.509450 [0.381720, 0.628420] | 0.590285 [0.471642, 0.695132] |
| N-S | 0.493033 [0.335011, 0.628476] | 0.625938 [0.489437, 0.731239] |
| N-O | 0.541861 [0.389407, 0.658664] | 0.641308 [0.535199, 0.723497] |
| N-P-S | 0.273741 [0.109262, 0.397982] | 0.434730 [0.289683, 0.560410] |
| N-P-O | 0.303527 [0.115944, 0.452083] | 0.505549 [0.357329, 0.633573] |

| Contrast | Soundgen primary | OSF auxiliary |
|---|---:|---:|
| R-S minus N-S | 0.039455 [-0.093529, 0.170666] | -0.027733 [-0.137093, 0.087322] |
| N-S minus N-P-S | 0.219293 [0.131800, 0.312999] | 0.191208 [0.129703, 0.246184] |
| R-S minus R-O | 0.023039 [-0.025942, 0.073764] | 0.007921 [-0.029326, 0.047303] |

Common set: 300 stimuli, 25 prototypes. Each version excludes `vig_01.wav`, `vig_25.wav`, `vig_75.wav` (token join failure); no all-missing exclusions. Bootstrap: 10000 shared resamples, splitmix64 seed 20261020.

Gain pair: [2,20) s masks identical and every bin value within the registered tolerance. Additional full-input diagnostics retain four extended-surprise violations at 1.013333, 1.024000, 1.045333 and 1.120000 s. The unregistered exactly representable 1/32 gain probe is identical at every bin and hop.

Conventions: gravity receives a 6 s terminal wait to reach 20 s. Other samples render their full scenario, with the portion after 20 s excluded. R uses the registered approximation with symmetric Hann and raw, unnormalized one-sided band powers. OSF auxiliary follows the registered per-hop rule; the bundled R script final-window weight reuse is not reproduced. MP3 delay retained; fit uncertainty excluded from bootstrap CI.

Run all opt-in measurements:

```sh
cargo test --release --lib info_terrain_eval -- --ignored --nocapture
```

Individual filters: `info_terrain_eval_controls`, `info_terrain_eval_external`, `info_terrain_eval_renders`, `info_terrain_eval_wav`. The WAV test accepts `CONCHORDAL_INFO_TERRAIN_WAV=/path/to/input.wav`; absent that variable, it replays the first prepared WAV. Results are overwritten in this directory.

Validation: fmt and normal clippy passed; lib 1170 passed; full suite 1256 passed; release ignored harness 4 passed. All-target clippy stops on 16 existing findings in four untouched files, with none in touched files. Full test output/backtraces and exit status: repository-root `test_report.txt` and `test_status.txt`. See `verification.json` for details.
