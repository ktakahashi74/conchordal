# F4e PeakBiased parent-present birth registration (2026-09-27)

Status: fixed before acquisition. This covers one offline death and one refill opportunity with two living parent candidates. It extends the parent-absent F4e boundary without reinterpreting that result as parent coverage. It does not establish normal-runtime metabolic input or timing performance.

## Fixed scene and model

Use seed 7, 48,000 Hz, 512 samples per hop, population 7, frame 64, a Harmonic child template with brightness 0.9, and the accepted three-tone habitat analysis used by F4b. Spawn Voices 1 and 3 at 330 and 550 Hz as living potential parents; Voice 2 at 440 Hz dies. Set their Entrain energy to 0.35 and 0.65, then apply one real `commit_decided_control` with 0.01 seconds, a flat point level of 0.6, phase 0.1, retrigger enabled, and autonomous attack disabled. Save each resulting energy bit pattern and the full live parent pool (ID, actual base frequency bits, updated energy bits, generation) before choosing a parent. Death must leave exactly one refill slot, reserved child ID 4 and member index 3. Keep parent generations 0; the child generation must be 1.

Use `RespawnPolicy::PeakBiased` with the default parent-bias parameters except local-search radius and step both 0.1 semitone, a Field range of 380–520 Hz, and at most 16 peaks. From the fixed spawn seed, independently reproduce the production energy-weighted parent draw over the full pool before any candidate operation. Use the chosen parent's actual pitch for the Gaussian, same-band discount, and octave discount. Never substitute template pitch or a manually chosen parent.

## Prepared child body and decision

At every Log2 bin center in the range, spawn the exact reserved child with the selected parent ID and generation. Derive its actual body and representative Recipe, run the 72-hop subjective footprint against the accepted habitat, and compute body fitness score and level. Extract peaks from the resulting body-score scan. For every extracted peak, pre-enumerate its nonzero-radius local grid and independently evaluate the actual child body at each grid frequency. Check that every candidate Recipe identity carries reserved child ID and generation 1 and that the selected child has the same body, frequency, parent lineage, and Recipe identity as the evaluated candidate.

After the independent parent draw, reproduce all production RNG consumption in order, including any fallback draw. For each peak, calculate `max(body_score, 0)^scene_score_exponent` multiplied by the selected parent's Gaussian and same-band/octave discount factors. Select with independent `WeightedIndex`, then choose the highest body-score local point on the declared grid. Compare selected frequency and post-selection RNG probe. A poisoned point-score scan must not change the body-based candidate or choice when the body table is held fixed.

Use one threshold below the chosen body level and another above it. The lower case must create exactly one child and one spawn event; the higher case must create neither, with one opportunity counter increment but no child ID or member-index consumption. A following hop must not refill the same death again. Point-level poisoning must not override body-level threshold decisions. Reject stale parent pool/energy or generation, wrong child ID, changed template, stale epoch, and mismatched local Recipe before selection or child creation. Preserve every failed attempt as evidence; do not relax thresholds after acquisition.

Run the focused parent-present test and affected F4b/F4e tests with full output. Root owns the eventual full suite, source freeze, and timing measurements.
