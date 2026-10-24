# Auditory consumer validation

Ordinary tests cover the selected auditory front end, PCM self-exclusion,
onset perturbations, body history, reset/clone behavior and intermittent updates.
The V3 and V4 tests run without an output directory or acquisition files.

The ignored `acquire_current_v2_consumers` test evaluates the current front end
against a manifest's ideal component reference. It retains the V2 HillClimb,
respawn and range-local birth decisions, and cumulative Flora biomass signs
across the registered 47-hop window. It does not choose acceptance criteria.

```sh
export AUD_V2_INPUT=/path/to/registered-inputs
export AUD_V2_OUTPUT=/path/to/new-output
mkdir -p "$AUD_V2_OUTPUT"
cargo test --release --lib acquire_current_v2_consumers -- --ignored --nocapture
```

Use the repository's resource guard and Cargo wrapper on shared research hosts.
`AUD_V2_INPUT/manifest.json` supplies each fixture's PCM path and SHA256,
frequencies, amplitudes, onset hop, category, resolution, eight `judgment_hops`
and 47 `flora_hops`. Paths can be absolute or relative to the input directory.
Only complete hops are processed; any trailing samples are recorded separately.
Existing completed fixtures in the output are reused. Use a fresh directory
when inputs, parameters or source change. `AUD_V2_PILOT=1` runs only the
registered `onset_110.0` fixture for checking a changed acquisition pipeline.

Outputs are `consumer-rows.jsonl` (per judgment hop, including chosen/truth
pitches and component field errors) and `consumer-fixtures.jsonl` (counts,
three birth-distribution L1 sums, target counts and PCM provenance). The single
array entry is the current producer. For comparisons, acquire each source into
its own directory over the same immutable manifest; keep category and
resolution cells separate and split `two_tone` cells by their ERB interval.
Do not exclude near ties or treat the reference's nonzero errors as test failures.

The earlier multi-variant V2 tools remain in `auditory_validation_assay.rs` and
`auditory_shadow_tests.rs`. They retain the old NSGT front end and the S1,
narrow-context, Gaussian and direct-context controls they use. The independent
PCM/reference generator is `aud_m1_reference.py` (with an independently supplied
reference module). `aud_m1_v2_consumers` also creates the modal-body fixtures and
input metadata. `aud_m1_diagnosis.py` copies and classifies those inputs and
builds direct-PCM self-exclusion controls. `aud_m1_summarize.py` and
`aud_m1_boxcar_summary.py` summarize the retained legacy V2 output shapes.
Their historical criteria describe those acquisitions; they do not define
acceptance for a new front end.
