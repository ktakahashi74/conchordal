# Python scripts

Python は `scripts/.python-version` の 3.14.8、外部依存は `requirements.txt` の固定版を使用する。これらは CI とローカル検査で共通である。

```bash
python3.14 -m venv "$HOME/.cache/conchordal-test-tmp/python-venv"
"$HOME/.cache/conchordal-test-tmp/python-venv/bin/python" -m pip install -r scripts/requirements.txt
"$HOME/.cache/conchordal-test-tmp/python-venv/bin/python" -m unittest discover -s scripts/tests -p 'test_*.py'
```

作業用 venv はリポジトリ外に置いてもよい。重いジョブの一時出力は disk 上の `TMPDIR` に置き、[OOM 運用記録](../docs/operations/oom-prevention.md)に従う。

## 分類と参照

2026-10-17 の基準版にあった 84 本を、tests、CI、docs、他の scripts からのファイル名・module 名参照で調べた。下表の参照先は各 file を保持する根拠の一例であり、科学的受入の完了を意味しない。評価器には研究用比較器、取得器、運用ツールも含む。モデルはこれらから import される参照実装である。

### fixture の生成器（10 本）

| File | 用途 | 参照元の例 |
| --- | --- | --- |
| [generate_temporal_accent_ledger_fixtures.py](generate_temporal_accent_ledger_fixtures.py) | Finite ordered-delivery traces from the accent reference, including cap loss. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |
| [generate_temporal_feature_fixtures.py](generate_temporal_feature_fixtures.py) | Replay registered Python references as independent Rust feature fixtures. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |
| [generate_temporal_group_fixtures.py](generate_temporal_group_fixtures.py) | Independent Decimal oracle for saved-member maxima and fractional assignment. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |
| [generate_temporal_grouping_fixtures.py](generate_temporal_grouping_fixtures.py) | Decimal Pearson and set-based complete-link oracles on constructed inputs. | [scripts/generate_temporal_producer_fixtures.py](../scripts/generate_temporal_producer_fixtures.py) |
| [generate_temporal_lifecycle_fixtures.py](generate_temporal_lifecycle_fixtures.py) | Independent slot/clock/admission oracle, without acoustic matching arithmetic. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |
| [generate_temporal_period_fixtures.py](generate_temporal_period_fixtures.py) | Independent all-bin kernels and uncached summation for period-bank fixtures. | [docs/roadmap/temporal-dcc/feature-manifest.json](../docs/roadmap/temporal-dcc/feature-manifest.json) |
| [generate_temporal_producer_fixtures.py](generate_temporal_producer_fixtures.py) | Bundle/key producer oracle using dictionaries, immutable tuples and sets. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |
| [generate_temporal_proposal_fixtures.py](generate_temporal_proposal_fixtures.py) | Independent exact-rational supplied-key persistence and arbitration reference. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |
| [generate_temporal_ridge_fixtures.py](generate_temporal_ridge_fixtures.py) | Decimal oracle for masked ridge residuals and candidate secants. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |
| [generate_temporal_trajectory_fixtures.py](generate_temporal_trajectory_fixtures.py) | Independent rational peak/ownership reference for the first acoustic stage. | [docs/roadmap/temporal-dcc/m0-accent-ledger.json](../docs/roadmap/temporal-dcc/m0-accent-ledger.json) |

### 受入の評価器・取得・運用（52 本）

| File | 用途 | 参照元の例 |
| --- | --- | --- |
| [capture_temporal_dcc_baseline.py](capture_temporal_dcc_baseline.py) | Freeze actual worktree bytes and explicitly selected offline evidence for M0. | [scripts/tests/test_evaluate_temporal_dcc_baseline.py](../scripts/tests/test_evaluate_temporal_dcc_baseline.py) |
| [collect_temporal_body_prototypes.py](collect_temporal_body_prototypes.py) | Render the preregistered I10 diagnostic corpus and freeze actual body descriptors. | [docs/roadmap/temporal-dcc/i10-body-outcome.md](../docs/roadmap/temporal-dcc/i10-body-outcome.md) |
| [evaluate_acoustic_hypotheses.py](evaluate_acoustic_hypotheses.py) | Observe the chosen action prefix, compare acoustic hypotheses, then score futures. | [scripts/tests/test_evaluate_acoustic_hypotheses.py](../scripts/tests/test_evaluate_acoustic_hypotheses.py) |
| [evaluate_acoustic_state.py](evaluate_acoustic_state.py) | Compare sustained changes of acoustic distributions without naming phrases. | [scripts/tests/test_evaluate_acoustic_state.py](../scripts/tests/test_evaluate_acoustic_state.py) |
| [evaluate_action_conditioned.py](evaluate_action_conditioned.py) | Owned-action acoustic forecasts, with realized branches read only after issue. | [scripts/evaluate_acoustic_hypotheses.py](../scripts/evaluate_acoustic_hypotheses.py) |
| [evaluate_auditory_envelope.py](evaluate_auditory_envelope.py) | Causal auditory-band envelopes for research, without source or stream labels. | [scripts/evaluate_acoustic_hypotheses.py](../scripts/evaluate_acoustic_hypotheses.py) |
| [evaluate_beta.py](evaluate_beta.py) | Render reproducible beta comparisons with the Python standard library. | [scripts/evaluate_flow_amplitude.py](../scripts/evaluate_flow_amplitude.py) |
| [evaluate_chosen_outcomes.py](evaluate_chosen_outcomes.py) | Learn acoustic innovation scale from only the selected action's actual PCM. | [scripts/tests/test_evaluate_chosen_outcomes.py](../scripts/tests/test_evaluate_chosen_outcomes.py) |
| [evaluate_component_order.py](evaluate_component_order.py) | Compare causal timing/frequency predictions of unordered spectral-rise groups. | [scripts/tests/test_evaluate_component_order.py](../scripts/tests/test_evaluate_component_order.py) |
| [evaluate_component_relations.py](evaluate_component_relations.py) | Observe causal temporal relations between frequency channels in mixed audio. | [scripts/evaluate_auditory_envelope.py](../scripts/evaluate_auditory_envelope.py) |
| [evaluate_context_expectation.py](evaluate_context_expectation.py) | Research: causal continuous-feature expectations over changing contexts. | [scripts/evaluate_predictive_relations.py](../scripts/evaluate_predictive_relations.py) |
| [evaluate_context_scores.py](evaluate_context_scores.py) | Bounded distribution distances for frozen regional context forecasts. | [scripts/tests/test_evaluate_context_scores.py](../scripts/tests/test_evaluate_context_scores.py) |
| [evaluate_continuous_acoustic_state.py](evaluate_continuous_acoustic_state.py) | Research PCM state updates separated from requests for joint future forecasts. | [scripts/tests/test_evaluate_continuous_acoustic_state.py](../scripts/tests/test_evaluate_continuous_acoustic_state.py) |
| [evaluate_contour_observability.py](evaluate_contour_observability.py) | Audit waveform periodicity and missing-input semantics before runtime attachment. | [scripts/tests/test_evaluate_phrase_expectation.py](../scripts/tests/test_evaluate_phrase_expectation.py) |
| [evaluate_driven_acoustic_state.py](evaluate_driven_acoustic_state.py) | Causal Gaussian-driven acoustic candidates, not source or cognitive labels. | [scripts/acoustic_hypotheses.py](../scripts/acoustic_hypotheses.py) |
| [evaluate_flow_amplitude.py](evaluate_flow_amplitude.py) | Render a blinded, matched-seed flow-amplitude comparison for sample 12. | [scripts/tests/test_evaluate_flow_amplitude.py](../scripts/tests/test_evaluate_flow_amplitude.py) |
| [evaluate_frequency_events.py](evaluate_frequency_events.py) | Observe spectral-component power rises and trajectories in mixed audio, offline. | [scripts/evaluate_component_order.py](../scripts/evaluate_component_order.py) |
| [evaluate_grouping_cues.py](evaluate_grouping_cues.py) | Audit separate within-channel regularity and cross-channel coherence inputs. | [scripts/tests/test_evaluate_grouping_cues.py](../scripts/tests/test_evaluate_grouping_cues.py) |
| [evaluate_interval_order.py](evaluate_interval_order.py) | Observe and predict interval order from the existing Rust acoustic onset path. | [scripts/evaluate_component_order.py](../scripts/evaluate_component_order.py) |
| [evaluate_interval_retrieval.py](evaluate_interval_retrieval.py) | Research-only continuous interval retrieval with explicit, unfitted parameters. | [scripts/tests/test_evaluate_interval_retrieval.py](../scripts/tests/test_evaluate_interval_retrieval.py) |
| [evaluate_local_prediction.py](evaluate_local_prediction.py) | Compare causal local envelope histories without assigning acoustic sources. | [scripts/tests/test_evaluate_local_prediction.py](../scripts/tests/test_evaluate_local_prediction.py) |
| [evaluate_memory_constraints.py](evaluate_memory_constraints.py) | Reanalyse the published PPM-Decay task without treating it as an interval-memory fit. | [scripts/tests/test_evaluate_memory_constraints.py](../scripts/tests/test_evaluate_memory_constraints.py) |
| [evaluate_modal_continuation.py](evaluate_modal_continuation.py) | Audio-only modal continuation candidates, for offline research. | [scripts/tests/test_evaluate_modal_continuation.py](../scripts/tests/test_evaluate_modal_continuation.py) |
| [evaluate_perceptual_action.py](evaluate_perceptual_action.py) | Issue owned-action trajectories, observe native perceptual consequences, then score. | [scripts/evaluate_chosen_outcomes.py](../scripts/evaluate_chosen_outcomes.py) |
| [evaluate_phrase_expectation.py](evaluate_phrase_expectation.py) | Compare causal contour prediction on controlled audio, without driving the instrument. | [scripts/evaluate_acoustic_state.py](../scripts/evaluate_acoustic_state.py) |
| [evaluate_prediction_residuals.py](evaluate_prediction_residuals.py) | Past-only residual forecasts for frozen local energy predictions. | [scripts/evaluate_spectral_relations.py](../scripts/evaluate_spectral_relations.py) |
| [evaluate_predictive_relations.py](evaluate_predictive_relations.py) | Research: past-partner prediction gains conditional on own and background history. | [scripts/tests/test_evaluate_predictive_relations.py](../scripts/tests/test_evaluate_predictive_relations.py) |
| [evaluate_prospective_actions.py](evaluate_prospective_actions.py) | Prospective paired sound/wait consequences and isolated chosen-outcome updates. | [scripts/evaluate_selected_revision.py](../scripts/evaluate_selected_revision.py) |
| [evaluate_relational_contexts.py](evaluate_relational_contexts.py) | Score frozen context forecasts when their matching native sound becomes observed. | [scripts/tests/test_evaluate_continuation_stream.py](../scripts/tests/test_evaluate_continuation_stream.py) |
| [evaluate_residual_uncertainty.py](evaluate_residual_uncertainty.py) | Learn total residual variation from matching outcomes of frozen regional forecasts. | [scripts/tests/test_evaluate_residual_stream.py](../scripts/tests/test_evaluate_residual_stream.py) |
| [evaluate_resolution.py](evaluate_resolution.py) | Compare the three scripted resolution interventions in sample 12. | [scripts/evaluate_flow_amplitude.py](../scripts/evaluate_flow_amplitude.py) |
| [evaluate_rhythm_perception.py](evaluate_rhythm_perception.py) | Archive acoustic-only meter observations before changing temporal behavior. | [scripts/evaluate_rhythm_recurrence.py](../scripts/evaluate_rhythm_recurrence.py) |
| [evaluate_rhythm_recurrence.py](evaluate_rhythm_recurrence.py) | Test causal acoustic-event recurrence prediction without driving production. | [scripts/tests/test_evaluate_rhythm_recurrence.py](../scripts/tests/test_evaluate_rhythm_recurrence.py) |
| [evaluate_rt.py](evaluate_rt.py) | Profile sustained, dense-onset and overload/recovery workloads. | [scripts/evaluate_stage2.py](../scripts/evaluate_stage2.py) |
| [evaluate_selected_revision.py](evaluate_selected_revision.py) | Compare retained states and selected-evidence proposals on common future audio. | [scripts/tests/test_evaluate_selected_revision.py](../scripts/tests/test_evaluate_selected_revision.py) |
| [evaluate_spectral_change.py](evaluate_spectral_change.py) | Forecast the first spectral change and its direction, using only past audio. | [scripts/evaluate_acoustic_state.py](../scripts/evaluate_acoustic_state.py) |
| [evaluate_spectral_expectation.py](evaluate_spectral_expectation.py) | Research-only prediction of relative spectral mass in overlapping audio. | [scripts/evaluate_acoustic_state.py](../scripts/evaluate_acoustic_state.py) |
| [evaluate_spectral_observability.py](evaluate_spectral_observability.py) | Audit information discarded by three-band temporal energy observations. | [scripts/tests/test_evaluate_spectral_observability.py](../scripts/tests/test_evaluate_spectral_observability.py) |
| [evaluate_spectral_relations.py](evaluate_spectral_relations.py) | Prospective regional NSGT expectations conditioned on native frequency history. | [scripts/tests/test_evaluate_relational_contexts.py](../scripts/tests/test_evaluate_relational_contexts.py) |
| [evaluate_stage2.py](evaluate_stage2.py) | Archive matched-seed measure, habituation and practical DCC comparisons. | [scripts/tests/test_evaluate_stage2.py](../scripts/tests/test_evaluate_stage2.py) |
| [evaluate_structural_predictions.py](evaluate_structural_predictions.py) | Compare retained acoustic hypotheses after a chosen action, then score real audio. | [scripts/evaluate_prospective_actions.py](../scripts/evaluate_prospective_actions.py) |
| [evaluate_temporal_associations.py](evaluate_temporal_associations.py) | Research Hebbian associations from a continuous auditory history. | [scripts/tests/test_evaluate_temporal_associations.py](../scripts/tests/test_evaluate_temporal_associations.py) |
| [evaluate_temporal_coherence.py](evaluate_temporal_coherence.py) | Research: delayed multirate envelope coherence with explicit finite support. | [scripts/tests/test_evaluate_temporal_coherence.py](../scripts/tests/test_evaluate_temporal_coherence.py) |
| [evaluate_temporal_energy_footprints.py](evaluate_temporal_energy_footprints.py) | Offline quadrature audit against frozen counterfactual PCM energy windows. | [scripts/tests/test_temporal_energy_footprints.py](../scripts/tests/test_temporal_energy_footprints.py) |
| [evaluate_temporal_history.py](evaluate_temporal_history.py) | Research temporal history with scale-dependent temporal precision. | [scripts/evaluate_temporal_associations.py](../scripts/evaluate_temporal_associations.py) |
| [evaluate_temporal_waiting.py](evaluate_temporal_waiting.py) | Score observed waiting with the existing interval forecast, without inferring closure. | [scripts/tests/test_evaluate_temporal_waiting.py](../scripts/tests/test_evaluate_temporal_waiting.py) |
| [fit_temporal_body_prototypes.py](fit_temporal_body_prototypes.py) | Fit frozen descriptive body medoids from registered, actually rendered records. | [scripts/verify_temporal_body_prototypes.py](../scripts/verify_temporal_body_prototypes.py) |
| [locate_i11_first_divergence.py](locate_i11_first_divergence.py) | I11-1 §5.5(a): where a `body` run and a `proxy` run first choose differently, and why. | [scripts/tests/test_locate_i11_first_divergence.py](../scripts/tests/test_locate_i11_first_divergence.py) |
| [run_guarded.py](run_guarded.py) | Run a command in a verified, memory-limited systemd user scope. | [scripts/tests/test_run_guarded.py](../scripts/tests/test_run_guarded.py) |
| [summarize_i11_representative_gap.py](summarize_i11_representative_gap.py) | Distribution report for I11-1 §5.9: representative against actual onset footprints. | [scripts/tests/test_summarize_i11_representative_gap.py](../scripts/tests/test_summarize_i11_representative_gap.py) |
| [verify_i11_stage1.py](verify_i11_stage1.py) | Independent reference for I11-1 stage 1 (docs/roadmap/temporal-dcc/i11-onset-comparison.md §5.1-5.3). | [scripts/summarize_i11_representative_gap.py](../scripts/summarize_i11_representative_gap.py) |
| [verify_temporal_body_prototypes.py](verify_temporal_body_prototypes.py) | Replay the registered corpus through the normal prototype-enabled renderer. | [docs/roadmap/temporal-dcc/i10-body-outcome.md](../docs/roadmap/temporal-dcc/i10-body-outcome.md) |

### 参照の corpus・モデル（21 本）

| File | 用途 | 参照元の例 |
| --- | --- | --- |
| [acoustic_body.py](acoustic_body.py) | Conditional driven modal state, without a second resonant residual state. | [scripts/acoustic_revision.py](../scripts/acoustic_revision.py) |
| [acoustic_hypotheses.py](acoustic_hypotheses.py) | Competing local acoustic supports and amplitude evolutions. | [scripts/evaluate_acoustic_hypotheses.py](../scripts/evaluate_acoustic_hypotheses.py) |
| [acoustic_mixture.py](acoustic_mixture.py) | Conditional predictive mixtures, not probabilities of perceived source identity. | [scripts/acoustic_parameters.py](../scripts/acoustic_parameters.py) |
| [acoustic_parameters.py](acoustic_parameters.py) | Continuous passive acoustic parameters conditional on mode count and AR response. | [scripts/evaluate_structural_predictions.py](../scripts/evaluate_structural_predictions.py) |
| [acoustic_posterior.py](acoustic_posterior.py) | Conditional trajectory amplitudes and correlated AR residual state. | [scripts/acoustic_hypotheses.py](../scripts/acoustic_hypotheses.py) |
| [acoustic_revision.py](acoustic_revision.py) | Retained and freshly proposed conditional acoustic states. | [scripts/evaluate_selected_revision.py](../scripts/evaluate_selected_revision.py) |
| [acoustic_trajectory.py](acoustic_trajectory.py) | Causal acoustic chirp candidates, not identified perceptual sources. | [scripts/acoustic_hypotheses.py](../scripts/acoustic_hypotheses.py) |
| [context_paths.py](context_paths.py) | Shared finite context transitions; numerical hypotheses, not cognitive lifetimes. | [scripts/continuation_contexts.py](../scripts/continuation_contexts.py) |
| [continuation_contexts.py](continuation_contexts.py) | Joint continuation and context corrections conditional on the currently heard level. | [scripts/evaluate_relational_contexts.py](../scripts/evaluate_relational_contexts.py) |
| [evolving_regression.py](evolving_regression.py) | Continuous-time predictive deviation with retained coefficient uncertainty. | [scripts/evaluate_residual_uncertainty.py](../scripts/evaluate_residual_uncertainty.py) |
| [perceptual_sampling.py](perceptual_sampling.py) | Stratified integration of bounded auditory observations of whole trajectories. | [scripts/evaluate_prospective_actions.py](../scripts/evaluate_prospective_actions.py) |
| [perceptual_scores.py](perceptual_scores.py) | Distributional audit of bounded native perceptual fields. | [scripts/tests/test_evaluate_perceptual_scores.py](../scripts/tests/test_evaluate_perceptual_scores.py) |
| [ppm_decay_reference.py](ppm_decay_reference.py) | Offline reference for Harrison et al. (2020), ppm v0.1.1 (MIT). | [scripts/tests/test_evaluate_ppm_decay_reference.py](../scripts/tests/test_evaluate_ppm_decay_reference.py) |
| [relational_contexts.py](relational_contexts.py) | Conditional positive/zero forecasts with stationary, renewed or recurring contexts. | [scripts/evaluate_relational_contexts.py](../scripts/evaluate_relational_contexts.py) |
| [residual_regression.py](residual_regression.py) | Folded offset regression with learned, uncertain total predictive residual variance. | [scripts/evaluate_context_scores.py](../scripts/evaluate_context_scores.py) |
| [sample12.py](sample12.py) | Shared source contract and measurement windows for the current sample 12. | [scripts/evaluate_beta.py](../scripts/evaluate_beta.py) |
| [temporal_accent_reference.py](temporal_accent_reference.py) | Offline f64 accent admission, original-support cuts and bounded delivery. | [scripts/generate_temporal_accent_ledger_fixtures.py](../scripts/generate_temporal_accent_ledger_fixtures.py) |
| [temporal_credit_reference.py](temporal_credit_reference.py) | Research credit assignment from Goh, Ursekar & Howard (2022), section 2. | [scripts/tests/test_evaluate_temporal_credit.py](../scripts/tests/test_evaluate_temporal_credit.py) |
| [temporal_feature_reference.py](temporal_feature_reference.py) | Independent causal raw-feature arithmetic; no retained descriptor or search state. | [scripts/generate_temporal_feature_fixtures.py](../scripts/generate_temporal_feature_fixtures.py) |
| [uncertain_contexts.py](uncertain_contexts.py) | Folded observations with distinct coefficient, sign and context uncertainty. | [scripts/evaluate_relational_contexts.py](../scripts/evaluate_relational_contexts.py) |
| [uncertain_regression.py](uncertain_regression.py) | Nonnegative observations with explicit Gaussian residual and observation noise. | [scripts/evaluate_context_scores.py](../scripts/evaluate_context_scores.py) |

### 参照の無いもの（削除）

`capture_temporal_dcc_hardware.py` は、自身を除く追跡対象の tests、CI、docs、scripts からファイル名・module 名とも参照が無いため削除した。ほかの 83 本は参照があり保持した。過去の baseline inventory は履歴として変更しない。

## Fixture の生成

対応表と再生成手順は [tests/fixtures/README.md](../tests/fixtures/README.md) に記す。生成済み JSON は Rust の独立参照データであり、通常の test 実行で上書きしない。受入用 corpus の条件と評価基準は各参照先の登録に従う。
