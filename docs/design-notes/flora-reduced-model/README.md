# Flora reduced closed-loop model

Exploration code behind the consumer–resource proposal in
[`../flora.md`](../flora.md#dynamics-redesign).
It is not part of the build, has no tests, and its results are not acceptance evidence.

## Files

- `model.py`: the terrain, ported from `src/core`:
  - the harmonicity common-root projection (`harmonicity_kernel.rs`, default parameters);
  - the ERB-domain Sethares roughness with self exclusion, suppression dip and the
    1000 Hz / +0.25 ERB reference (`roughness_kernel.rs`, `psycho_state.rs`);
  - loudness compression `power^0.23` at each partial;
  - the consonance field level with the default kernel `(1, −1.35, 1, 0)`, β = 2, θ = 0.
- `sim2.py`: revision 2, the model the proposal cites. Consumer–resource dynamics with
  biomass-driven consumption, an exact resource update, per-octave constants,
  configurable resolution and substeps, and metrics on both the biomass and the rendered
  spectrum in real time.
- `experiments.py`: every run the proposal cites (reference, ablations, grid, seeds,
  lone, other drones, delay, numerics, rate, flat terrain, relative loudness, level
  scale). `sweep2.py`: grid driver for ad hoc sweeps.
- `results/experiments.jsonl`: one summary per run of `experiments.py`.
  `results/sequences.txt`: the rendered partials of the reference and retune runs.
- `sim.py`, `sweep.py`, `ablate.py`, `show2.py`: revision 1, kept because it holds the
  implemented baseline (`"model": "baseline"`) used for the validation run. Its
  consumption is driven by the rendered tonal power, which leaves non-sounding biomass
  unconsumed; revision 2 replaces it.

## What it models and omits

Flora covers 110–1760 Hz on a 55–4000 Hz space (96 bins/oct unless `bpo` is set),
split into 1-ERB cells. Each cell sounds its dominant local maximum with the excess over
the cell floor within ±2 bins (sine only, no residual). Flora hears its own partials and
an optional 8-partial drone one hop later (`delay_hops` adds more; `self_hear: false`
removes Flora from its own terrain).

Omitted: NSGT smearing and its band-dependent delay, the residual layer, lane births,
slews and minimum separation, moving Populations, routing, and every parameter not
named here. The tonal window is in bins, so results depend on `bpo`.

## Running

Requires Python 3 with numpy, for example in a virtual environment under `$TMPDIR`.
Run the scripts from this directory without `-I` so that `model.py` is importable.

```sh
python experiments.py > experiments.txt     # about 4.5 minutes on 9 workers
python sim2.py '{"seconds":90,"drone":110,"snap_every":3}'
```

## Metrics

States are sampled every 47 hops (0.5013 s) and stamped at the end of the hop. Both
ends of every comparison lie after a 20 s warm-up. Lags are rounded to the sampling
grid; recurrence candidates are chosen by time difference (at least 5 s).

`sim2.py` takes the substep count from the proposed rule (`rate · dt ≤ 0.05`,
`D · dt ≤ 0.25`) unless `substeps` is given, does not clip the biomass, and records the
smallest biomass value (`min_biomass_before_clip`); a value below −10⁻¹⁵ aborts the run.
`results/experiments.jsonl` keeps six significant digits.

- `snd_js_1s`, `snd_js_5s`, `snd_js_20s`: mean Jensen–Shannon distance (base 2) between
  rendered spectra about 1, 5 and 20 s apart. The rendered spectrum is the partial
  powers smoothed by a 0.25 ERB Gaussian. `bio_*` is the same on the biomass.
- `snd_recur_frac`: fraction of frames whose nearest frame at least 5 s earlier lies
  within JS distance 0.1.
- `n_eff_mean`: participation ratio of the rendered partial powers.
- `sounding_mean`: mean sum of rendered targets (fraction of the ceiling).
- `total_v_max`: largest total biomass.
- `lattice_frac`, `lattice_new_after15s`, `lattice_old_after15s`: time-mean power
  fraction within ±25 cents of integer multiples of `f0/2`, for the initial drone and,
  from 15 s after a retune, for the new and the old drone.

`classify` labels runs `silent`, `frozen`, `haze`, `recurrent`, `monopoly` or `moving`
by fixed thresholds. The labels are a reading aid, not registered criteria.
