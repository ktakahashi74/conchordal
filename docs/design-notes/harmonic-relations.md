# Harmonic Relations: Temporal Model and Generation Contract

Status: design adopted for implementation (HV-4, 2026-11-01). Implementation units HV-5 (steps 1-3) and a later unit for steps 4-6.
Scope: the temporal part of the author's decision to extend harmonic valuation beyond one consonance scalar. The vertical part is [harmonic-support.md](harmonic-support.md).

## 1. Purpose, scope, and evidence status

**[Author rule]** Implement temporal harmony as bounded, uncertain relations among observed acoustic states, voice trajectories, and elapsed times. Learn only from the current performance. Connect the resulting predictions to generation through an opt-in control.

**[Principle]** A support distribution is neither an audible-root estimate nor a chord label. Acoustic compatibility, prediction, arrival, competition reduction, and perceptual closure are distinct quantities.

**[Author rule]** The following mechanisms cover all fourteen assigned phenomena:

1. Extension preparation, retention, resolution: voice intervals linked to preceding and succeeding support states.
2. Avoid-note residence and exit: dwell duration, competition trajectory, and observed departure relations.
3. Flattened-fifth and altered relations: approach direction and destination-conditioned correspondence.
4. Suspensions: retained voice across a background change, followed by a separate movement.
5. Rootless persistence: remembered support distinguished from currently observed support.
6. Upper structures and multiple tonal streams: concurrent correspondence and context hypotheses.
7. Bass and common-tone retention: absolute register, source identity, and sustained intervals.
8. Diatonic substitution: acoustically different intermediate states compatible with shared surrounding relations.
9. Tritone substitution: alternative approaches to shared destinations, without an interval-name lookup.
10. Function, cadence, resolution: predicted destination, arrival, competition change, and boundary evidence.
11. Modulation and pivot: changing context responsibilities and states compatible with multiple contexts.
12. Voice leading and common tones: identity, overlap, displacement, crossing, and handover.
13. Passing, neighboring, and appoggiatura relations: preceding/following trajectories and residence time.
14. Harmonic rhythm and sequences: legal change opportunities, waiting times, and transformed recurrence.

**[Principle]** These mechanisms do not establish culture-specific classifications. L3 familiarity is outside this implementation. No fixed key profiles, chord-progressions, scores, generated stylistic priors, or corpus priors enter the mainline.

**[Author rule]** Unless explicitly marked otherwise, the algorithms below are author rules, not validated perceptual models.

## 2. Observations and interval construction

### 2.1 Observation record

**[Author rule]**

```text
Observation {
  timestamp, snapshot_id, support_coordinate_version
  p[], M, C, D
  validity_mask, measurement_error
  optional IPEM_local[], IPEM_global[]
  trajectories[]: source_id, pitch, amplitude, timbre, articulation
  events[]: onset, offset, legal_opportunity, observed_boundary
}
```

Read timestamps and p, M, C, D from HV-3. Read trajectories and onset/offset records from existing sources. IPEM and PS observations are optional; their absence does not activate those generators.

**[Principle]** When M is zero, p is undefined. Missing data, silence, and an observed uniform distribution are different states. Measurement errors must come from the producer, numerical analysis, or repeatability measurements; they are not tuning parameters.

### 2.2 Segmentation

**[Author rule]** Construct a shared temporal partition at:

- Every audio-hop boundary.
- Every timestamped onset or offset.
- Every existing pitch/spawn/background decision opportunity.
- Every externally observed phrase or section boundary.

Events sharing a timestamp produce one boundary with multiple event records. Events between hops divide the trajectory interval, but do not fabricate intermediate HV-3 observations: the support value remains a timestamped sample with its sampling uncertainty.

Merge adjacent intervals only when neither contains a structural event and their stored acoustic and trajectory records are exactly identical after the producer’s declared numerical representation. Otherwise retain both. No novelty threshold decides segmentation.

**[Author rule]** Matching uses the latest bounded sequence of *anchors*: intervals containing a structural event (an onset, an offset, an existing decision opportunity or an observed boundary). Support changes between structural events remain in the hop intervals' M, C and D and in the next anchor's stored shape; they do not create anchors, because the support shape changes slightly at almost every hop. A hop interval is not automatically a phrase, beat, or harmonic event.

## 3. Graph, capacities, and expiration

### 3.1 Structures

**[Author rule]**

```text
Node {
  id, start, end, observation_refs, trajectory_refs
  observed_boundary_flags, continuation_id
}

Edge {
  id, source_node, target_node, relation_type
  attributes[], errors[], validity_mask
}

Correspondence {
  source_episode_id, source_node_ids[], current_node_ids[]
  matched_edge_ids[], unmatched_edge_ids[]
  pitch_translation, time_scale, transformation_errors
  preserved_relations[], transformed_relations[]
  context_id, retrieval_weight, source_version
}

Episode {
  id, immutable_nodes_and_edges
  boundary_evidence, continuation_links
}

Context {
  id, member_episode_refs_and_responsibilities
  support_accumulator, feature_statistics
  recent_responsibility, version
}
```

Edge types distinguish order, overlap, relative pitch change, relative time, articulation, absolute register, absolute speed, timbre, and source continuity. Do not replace them with a similarity field.

### 3.2 Supported load

**[Author rule; resource provenance: supplied candidate-layer limits]** Adopt these engineering limits, without claiming perceptual justification:

```text
Tmax = 600 seconds of retained history
Umax = 38,400 structural-event/decision records in that history
Smax = 16 anchors in one matching context
Kmax = 16 active episode-context hypotheses
Hmax = 16 retained correspondence hypotheses per query
```

A record containing several event flags counts once; different sources at the same time retain separate records. A longer performance uses a rolling history. Counts above the declared event density exceed the supported load.

**[Principle]** For sample rate Fs, hop size h, declared maximum V voice sources and L lane sources:

```text
Jmax <= 1 + ceil(Tmax * Fs / h) + Umax
```

This bounds shared interval nodes before merging. Store source trajectories separately; do not copy p for every voice.

**[Author rule]** Keep only consecutive order edges, source-continuation edges, and simultaneous source-pair relations. For S = V + L:

```text
Emax <= Jmax * [1 + S + S(S-1)/2]
```

Nonadjacent relations are computed from paths during matching. Correspondences contain at most Smax matched anchors. Their storage is additional to Emax.

**[Author rule]** Candidate count A is derived from the existing consumers:

```text
Amax = sum over consumers(
  maximum evaluations per legal opportunity
  * maximum opportunities processed in one hop
)
```

Declare those maxima with V, L, and each selector’s existing proposal/scan bound. Do not assume that sixteen states means sixteen pitch candidates.

**[Author rule; orchestrator revision]** Storage of support shapes. Hop-boundary intervals keep only M, C, D, validity and references. A support shape p is stored only at anchors (intervals with a structural event, §2.2), in the producer's sparse form with its residual mass recorded, so retained support storage is bounded by Umax anchors rather than by every hop. Report the dense worst case and the measured sparse size separately.

### 3.3 Expiration

**[Author rule]** Expire data older than Tmax, then remove references to expired data. Capacity exhaustion before age expiration evicts the oldest complete resource chunk. Chunks contain at most Smax anchors; continuation links prevent chunk boundaries from becoming musical boundaries.

Subtract expired contributions from statistics. Never overwrite an episode with a transformed interpretation. New interpretations receive new versions.

Discarded correspondence weight becomes unknown mass. Invalid references cannot redirect to a reused storage slot.

## 4. Contexts

### 4.1 Sensory and support contexts

**[Candidate: Leman 2000; Bigand et al. 2014]** Existing IPEM local/global images retain their original representation and processing. The 0.1/1.5-second half-decay times and image correlation belong to that candidate, not to a general theory of harmonic prediction. [Model description](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2014.00094/full)

**[Author rule; time constants inherited from TC]** Maintain separate support-space accumulators:

```text
a_T = 2^(-dt/T)
R_T <- a_T R_T + (1-a_T) M p
mass_T = sum(R_T)
q_T = R_T / mass_T, when mass_T > 0
T in {0.1 s, 1.5 s}
```

This is an author extension to HV-3 support, not an IPEM reproduction. Publish mass with q; silence must reduce evidential strength even when normalized shape remains unchanged.

### 4.2 Episode-context formation

**[Author rule]** Existing observed boundaries close episodes. Resource chunking does not. Each context contains weighted references to observed chunks and their successors.

For a new completed chunk g, calculate compatibility `k(g,k)` with each context using §5. Context prior `π_k` is its retained responsibility mass divided by the total retained mass.

```text
b_k = π_k * k(g,k)
b_unknown = 1 - sum_k(b_k)
```

**[Author rule; orchestrator revision]** The context compatibility is the context's mixture likelihood, the same construction as §5.3 inside one context. With `w_e` the member responsibility of eligible endpoint e divided by the context's total member responsibility, and the alternative mappings of e sharing its weight equally before pruning,

```text
k(g,k) = sum_e w_e * sum_{m in mappings(e)} (1/|mappings(e)|) * c_m * k_m
```

Pruned mappings contribute zero and are not renormalized away; a context without eligible endpoints has compatibility zero. A maximum over members is not used: it would let a context absorb any chunk that fits one member, while the mixture keeps contexts coherent.

If unknown is strictly greater than every existing b_k, create a context from g. Otherwise add g with normalized responsibilities among compatible contexts. With no compatible context, retain it as unknown until context creation completes.

At Kmax, replace the context with the smallest retained responsibility; break ties by oldest last observation, then stable ID. Its episodes remain historical evidence until ordinary expiration.

**[Author rule]** Merge contexts only when their canonical relation signatures, transformation permissions, and predictive summaries are identical under the declared numerical representation. Combine membership without counting the same observation twice. Similarity alone does not justify merging.

**[Author rule]** Context responsibility is updated after scoring the new observation. Prediction always uses the preceding version. Simultaneous partial streams may support different contexts; no winner-takes-all tonal label is required.

## 5. Prediction, correspondence, scales, and unknown

### 5.1 Feature errors and scales

**[Author rule]** For each relation attribute d, retain count n, mean, and centered sum of squares over unexpired observations of that attribute and relation type:

```text
v_d = sum_i (x_i - mean_d)^2 / (n-1), if n >= 2
v_d = 0, otherwise

s_d² = error_current_d² + error_reference_d² + v_d
```

Use pre-update statistics. Error terms are propagated standard errors where available. If the producer supplies only bounds, use those bounds as conservative scales and mark that interpretation.

For `s_d=0`, identical values have zero residual; unequal values have infinite residual. Never insert an arbitrary epsilon to create perceptual tolerance.

**[Author rule]** Treat support shape as one relation block:

```text
d_p² = (1/2) * sum_b (sqrt(p_b)-sqrt(q_b))²
```

Its error and variance are computed in the same distance representation.

**[Author rule; orchestrator revision]** The support-shape variance is the sample variance in the Hellinger embedding, the direct generalization of the scalar variance. With `u_i = sqrt(p_i)` for the n unexpired anchors and `S = sum_i u_i`,

```text
v_p = sum_i (1/2)||u_i - S/n||^2 / (n-1)
    = n * (1 - ||S/n||^2) / (2*(n-1)),   n >= 2   (since ||u_i||^2 = 1)
v_p = 0,                                  otherwise
```

This equals half the mean pairwise squared Hellinger distance, as the scalar variance equals half the mean squared pairwise difference. Each anchor registers once when it is committed and is subtracted from `S` when it expires. Distances between consecutive anchors measure temporal smoothness, a different statistic, and the variance of pairwise distances measures their spread, not the dispersion of shapes; neither is used. M, C, D, timing, and movement remain separate blocks. Undefined blocks are omitted and reduce coverage.

### 5.2 Retrieval and correspondence

**[Author rule]** Search retained anchor sequences with a monotone dynamic program. An extension may:

- Match the next pair of compatible relation types.
- Leave an observed or historical anchor unmatched.
- Continue an independent source stream.

No crossing correspondence is allowed within one stream; crossings between physical pitch trajectories remain representable.

At each extension retain Hmax hypotheses, ordered by coverage-adjusted compatibility, then stable IDs. Match at most Smax anchors. Every hypothesis records omissions.

Fit pitch translation by error-weighted least squares over matched pitches. Fit positive time scale by error-weighted least squares over log duration ratios. Use only the matched prefix. Missing or zero-duration evidence disables that transformation. Keep identity and fitted-transform hypotheses separate.

### Orchestrator revision after HV-7

**[Author rule; supersedes exhaustive correspondence retrieval above]** Use bounded
proposals rather than a best exhaustive correspondence. The observed HV-7 background
batch cost of about 1.4 seconds rejects repeated whole-history dynamic programming
as the integration method. Keep the fourteen relation representations, six consumers,
composer word, causal controls, and zero-weight identity.

Extend a retained bank of at most Hmax hypotheses with each new observed anchor.
Retain matched and omitted relations, source identity, observed successors, prefix-only
transformation fits, and numerical uncertainty. Cache the already evaluated prefix;
do not rescore the entire history or prefix on each observation. Unsupported extensions,
expiration and pruning retain unknown mass. Querying never trains on a proposal.

Insert a committed anchor into a neighborhood index over the existing square-root
support embedding. Retrieve bounded endpoint proposals from that index instead of
scanning all eligible endpoints. Index ordering is an implementation of approximate
retrieval, not a new perceptual feature or harmony prior. Missing support retains an
explicit missing-shape route. All eligible past endpoints can support context evidence.
A predictive atom additionally requires an observed successor; a missing successor
retains its mass as unknown.
Identity and transformed source relations remain distinct.
Compatibility and unknown accounting apply to the proposed empirical neighborhood;
failed or pruned hypotheses are not renormalized into confidence.

Generation and context assignment share the bounded proposal evidence from the current
observed resource chunk (at most Smax anchors); an episode or resource-chunk boundary
starts a new incremental query without erasing historical evidence. A full resource
chunk closes on its Smax-th anchor, so its assignment uses that chunk's bank before
the next observation replaces it; this does not create a musical boundary or episode.
Source and context membership remain explicit; a change in retrieval is not permission
to erase retained observations or causal history. Acceptance uses the unchanged definition and causal-contrast tests, not equality to exhaustive search.

The index orders the existing square-root vectors by their projection onto the
normalized existing coordinate vector, then visits bounded neighbors on both sides.
This is approximate retrieval, not an exact Hellinger nearest-neighbor guarantee.
Missing-shape endpoints have a separate chronological index. The original full
compatibility kernel scores the retrieved proposals. Every proposed endpoint shares
its mass among its proposed mappings before pruning; context mixtures retain their
empirical membership prior and leave unproposed members unknown. Index expiration
compacts each retained array once per batch, rather than once per expired endpoint.

### 5.3 Compatibility and weights

**[Author rule]** For each matched relation block r:

```text
z_r² = mean over its valid attributes d(
  residual_d² / s_d²
)

k_j = exp[-(1/2) * mean_r(z_r²)]
c_j = matched_valid_relation_blocks / requested_relation_blocks
```

The factors one-half and squared standardized residuals define the author’s Gaussian-shaped kernel; they are not fitted psychoacoustic constants. Averaging prevents a densely represented block from acquiring accidental extra weight.

Let N be the number of eligible historical anchor endpoints searched. If endpoint i yields several alternative mappings, divide its empirical weight `1/N` equally among them before pruning.

```text
alpha_j = empirical_endpoint_weight_j * c_j * k_j
alpha_unknown = 1 - sum_retained_j(alpha_j)
```

Thus partial matches, weak matches, and pruning retain unknown mass instead of becoming confident predictions through renormalization.

### 5.4 Transported successors

**[Author rule]**

```text
P_hat(Y, tau | G_t)
  = alpha_unknown * UNKNOWN
    + sum_j alpha_j * K_j(Y, tau)
```

`K_j` is an empirical atom at the observed successor relation trajectory and waiting time, transported by correspondence j, with propagated measurement uncertainty.

Translate pitches; multiply durations by the inferred time scale. Transport support through mass-conserving interpolation in HV-3’s declared coordinate system. Out-of-range mass becomes unknown, not wrapped pitch-class mass.

Do not assume C, D, or timbre-dependent support are transposition-invariant. Transformed attributes lacking a certified transformation become unavailable. Preserve absolute register, speed, and timbre evidence alongside transformed relative relations.

**[Author rule]** A complete query is unknown when it has no valid matched relation blocks, no observed successors, or expired source data. Predictions are fixed before observation and updated only afterward. Unplayed proposals never enter training.

**[Candidate: Harrison et al. 2020]** PPM-Decay remains the existing SQ model with its existing settings. The continuous graph above is not claimed as PPM-Decay; introducing a learned discrete alphabet would be a separate author rule. [Original model](https://pmc.ncbi.nlm.nih.gov/articles/PMC7668605/)

## 6. Derived relations

All definitions in this section are **[Author rules]**.

- **Arrival:** observed state and arrival time overlap a previously issued prediction’s propagated error envelope. Record soft compatibility as well; variance alone is not a categorical tolerance.
- **Competition reduction:** `D_before-D_after`, with both M values and source changes attached. Silence or source deletion is not automatically resolution.
- **Retention:** a matched source interval continues with pitch change inside its measurement error; distinguish rearticulation from uninterrupted continuation.
- **Cadence candidate:** an arrival linked to observed retention, articulation change, boundary evidence, or subsequent inactivity. Preserve the evidence vector; do not synthesize a universal closure score.
- **Substitution:** distinct intermediate acoustic states match the same predecessor and successor relations. Online successor compatibility integrates over the predicted successor distribution. Without successor evidence, report only predecessor-compatible alternatives.
- **Pivot candidate:** one observed state has valid mappings into multiple context histories. Confirm a context transition only when later observations change the leading context responsibility; record its persistence duration without imposing an invented minimum.
- **Voice continuity:** preserve generator source identity separately from perceptual correspondence. Pitch proximity cannot silently exchange IDs at crossings. Handover, silence, and timbre changes remain explicit edges.
- **Retrospective ornament role:** assign passing, return, or delayed-approach relations only after the successor is observed. Before that point, retain alternatives.

## 7. Generation integration

### 7.1 HV-3 preview

**[Author rule]**

```text
preview(snapshot, action) -> {
  surrogate_p, surrogate_M, surrogate_C, surrogate_D
  baseline_id, candidate_id, validity, numerical_error
}
```

HV-3 evaluates the candidate’s acoustic surrogate `e + q_f`. The temporal layer neither reconstructs e from p nor subtracts a voice from normalized support.

Self-exclusion uses an observer that subtracts the source's actual habitat PCM before
auditory analysis. It is true PCM leave-self-out, not subtraction from normalized
support. Every result identifies its baseline observer. History always records the
sound actually observed.

For lane replacement, an insertion surrogate must not be described as the exact post-removal scene. Evaluate available surrogate attributes plus action timing and trajectory relations; leave unsupported replacement attributes unknown. Joint previews are used only where HV-3 explicitly supports them.

### Orchestrator revision after HV-6

**[Author rule; supersedes the additive-mixture preview in §7.1]** Preview is a
first-order approximation at the currently heard state. It does not predict mixed
PCM. HV-6's second revision rejected counterfactual mixture prediction because it
was expensive and failed to agree with the heard sound. The consumer needs a local
proposal, not an exact pre-hearing simulation.

Read partial positions `y_k` and densities `q_k` from the body's cached self-report.
With `r=Ae`, `R=sum(r)`, `a_y=A delta_y`, and log2 grid width `dx`:

```text
C_preview = C + sum_k q_k gC(y_k)
D_preview = D + sum_k q_k gD(y_k)
M_preview = M + dx * sum_k q_k sum(a_yk)
dp/de_y   = (a_y - p * sum(a_y)) / R, when R > 0
r_preview = r + sum_k q_k a_yk
p_preview = r_preview / sum(r_preview), when sum(r_preview) > 0
```

Use the same sparse projection columns and grid interpolation as the published
gradients. At zero baseline mass, a nonzero candidate supplies its own support
shape; zero total updated mass remains undefined. Mass outside the declared
coordinate range and residual mass become unknown under the existing temporal
transport rules, without wrapping. N reads HV-6's related
novelty at the partial positions. Resulting timbre, articulation, and other
attributes without a first-order representation remain unavailable and contribute
neutrally under §7.2. Lane insertion is not an exact replacement preview.

Each result carries the observer identity (mixed or actual-PCM self-excluded,
including source identity and generation), observation sample/time, snapshot ID,
coordinate version, validity and propagated numerical error. Definition acceptance
uses the limit of finite differences for vanishing additions, including the shared
silence-floor derivative already adopted by HV-6. Mixed-PCM agreement is not an
acceptance condition.

**[Orchestrator decision C]** Support amount is linear in density, so its additive
update and normalization are exact within the additive-density model. This
projective first-order update is nonnegative and normalized even for finite
additions; its tangent at zero addition is the derivative above. It avoids an
arbitrary signed-to-probability conversion and works at a silent baseline. Sparse
column addition and normalization are inexpensive, whereas exact C/D evaluation
would require per-candidate FFTs, so C/D retain the published first-order gradients.
This is the same local density-additivity premise as gC, not a mixed-PCM prediction
or power-domain recompression. No new constant, tolerance or weight is introduced.

### 7.2 Paired temporal consequences

**[Author rule, constrained by the adopted DCC §5 and §9.6 contract]** The
empirical successor kernel remains unchanged. Its compatibility is evidence,
not a preference or a replacement objective. Evaluate only attributes represented
by the first-order preview. Preserve unassigned mass without renormalizing it.
Require paired support at the same coordinate/reference; a missing side makes
the column neutral, not a reward for silence or missing evidence.

For the existing selector's winner `a0`, use two separate columns:

```text
relation(a, a0) = tanh((relation_score(a) - relation_score(a0)) / scale_relation)
continuation(a, a0) = tanh((continuation_score(a) - continuation_score(a0)) / scale_continuation)
utility(a) = existing_score(a)
             + follow_harmonic_relations * relation(a, a0)
             + follow_harmonic_continuation * continuation(a, a0)
```

The relation column compares destinations conditional on moving. It cannot
change a baseline retain/wait/no-spawn decision into a movement, or vice versa.
The continuation column compares the continuing and acting alternatives. Within
each class it uses the native representative so destination affinity is not
silently counted as a preference for retaining. Source trajectory pitch belongs
to the destination column; the continuation preview uses acoustic support or
empirical waiting-time support. No ordinal human closure model is claimed.

Freeze the development scales before acceptance. HV-9 uses the unchanged HV-8
calibration scene and seed 9736058528124221056, with passive recording of both
columns' raw paired differences. The complete 354.016-second WAV is byte-identical
to HV-8's saved calibration WAV. Collect every supported nonzero difference per
column, use linearly interpolated quartiles (type 7), and floor the IQR at 1e-6:

| Column | Nonzero differences | Q25 | Q75 | Frozen scale |
|---|---:|---:|---:|---:|
| Relation | 3199 | -0.006584158711807825 | -2.1158810627979262e-119 | 0.006584158711807825 |
| Continuation | 7132 | -0.00015730696271130468 | 6.518287843773661e-05 | 0.0002224898411490413 |

Zeros and unsupported comparisons do not fit a scale. The audition demo is not
used to tune it. The 1e-6 floor is §9.6's numerical protection, not a musical
threshold. The user-adopted HV-9 specialization applies this signed scaling to
both columns; it does not import §9.6's separately calibrated ordinal heads.

### 7.3 PeakSampler

Preserve native scores, temperature, candidate set and categorical random stream.
Obtain `a0` with a shadow of that original draw. Add the supported signed columns
to the scores **before** the existing softmax. With continuation zero, condition
the destination distribution on `a0`'s move/retain class. Thus destination support
cannot inflate the marginal probability of retention. Restore the original total
weight before reusing the existing draw, mapped to its native conditional
quantile within that action class. Roundoff at the cumulative tail returns a
positive-weight candidate in the same conditioned support. Unknown or zero contributions return the
original weights exactly. Spawn densities use their native log-mass scores at
unit temperature, equivalently multiplying mass by `exp(signed consequence)`.

### 7.4 HillClimb / Metropolis

Preserve the native greedy improvement, proposal pool, temperature and body RNG
sequence to obtain `a0`. With continuation zero its action class is fixed. For
stochastic destination choices, perform a conditional Metropolis comparison in
that class using the native score plus the signed consequences, with the existing
uniform finite-candidate proposal measure and Hastings exponential. Record the
additional keyed draw independently of the body stream; do not mix acceptance
probabilities or discard the native score. For a supported continuation class
change, compare its native representative to `a0` with the same temperature.
A greedy opportunity uses deterministic maximum utility. Existing PS rescaling
rescales the temporal addition and temperature together, preserving score units.

### 7.5 Deterministic paths and randomness

Both dispositions zero branch to the original implementation before requesting
preview or randomness. Ties preserve the baseline. Any native bodily cost,
habituation, novelty, salience and existing composer terms remain in the score;
a unit temporal disposition never deletes them. Unsupported columns contribute
zero independently. Dedicated conditional draws are keyed by the existing sample,
source identity/generation and prediction identity; they do not consume body RNG.

Flora lane matching retains the existing displacement score; lane competition
retains its current value score; birth choices retain their current resource
score. A hold/no-birth alternative that previously had no independent native
score uses the baseline's score (zero native difference), not a new penalty or
reward. Existing hard eligibility, rate, resource and finite-candidate limits
still apply. No candidate-table expansion or authored harmonic template is added.

### 7.6 Timing, spawn, lanes, and commit order

The six consumers remain Voice pitch, Voice next opportunity, initial spawn,
respawn, Flora lane movement/replacement and Flora birth/no-birth. An opportunity
is first admitted by its unchanged native eligibility. Only continuation can
then prefer waiting, retaining a lane or declining spawn. Waiting compares the
empirical time compatibility of the existing now/next opportunities; expired or
unknown time support is neutral. Relations alone always leave that decision to
the native rule, while they may change a moving destination.

Freeze the received prediction before evaluating candidates. Commit the selected
physical action through the original path. Only subsequently heard outcomes enter
history; unplayed previews and alternatives do not train the relation model.
All fourteen relation representations, source identities, unknown mass, causal
controls and declared load remain in scope. Identical candidate/reference pairs
have an exact zero consequence and require no preview. After continuation chooses
retention, the unselected movement class requires no destination evaluation.
These elisions change neither the utility of an eligible winner nor its draw.

### Orchestrator revision after HV-8 audition

HV-9 distinguishes a learned expectation from a preference for fulfilling it.
The hierarchy memo §1.2, §4.3, §5 and §7 does not prescribe minimum prediction
error as the goal of generation. Under ledger §9.3.56, in-performance expectation,
learning and habituation belong to layer 4; the use and valuation of their
consequences belong to layer 3. This is a division of authority, not a source
equation for a novelty reward. The algorithms in §7 remain author rules.

The former deterministic mixture had a concrete boundary: at weight one its
winner depends on L alone. The baseline winner, including the effect of shared
habituation on Voice scores, no longer contributes to its ranking. Categorical
and Metropolis paths retain their baseline probabilities or ratios, but can still
increase retention when retention has greater compatibility. For example,
`Q(hold) > P0(hold)` whenever positive `P0(hold)` has `L(hold) > E_P0[L]`.
This was a property of the former conditioning rule, whose contract was
inconsistent with DCC §5 and §9.6. The diagnostics and same-opportunity comparisons
are recorded in `.orchestration/units/HV-9/report.md` (machine-local evidence).

Shared habituation currently erodes a signed acoustic score toward its existing
neutral theta, and a nonnegative acoustic mass toward zero. It neither erases
heard relations nor supplies a preference over their predicted continuations.
Attack habituation supports the existing span/gap action rule and remains
separate. Flora's base resource/growth fields deliberately retain their raw
views. Therefore a common multiplication of every L, timing alternative and
no-spawn action by a pitch-bin habituation value is not already specified by
these mechanisms. A no-sound action has no such pitch coordinate. Applying
score erosion toward L's neutral value one would instead remove mismatch
penalties; replacing L by `1-L` would reward mismatch and would turn unknown
mass into the wrong endpoint.

The first HV-9 diagnosis left the replacement law unresolved. The orchestrator
subsequently adopted the following revision from the existing DCC contract;
that decision supersedes the earlier block and the HV-4 convex mixtures.

**Orchestrator revision after HV-8 audition — adopted additive revision.**
DCC §5 passes relation support/departure, continuation/ending support and
uncertainty separately. It does not make minimum prediction error the goal.
DCC §9.6 supplies signed authored dispositions, differences from the context-free
body default, paired-support neutrality and additive utility minus bodily cost.
Here the body default is the existing selector's winner `a0`, and its existing
score already contains the bodily and ecological costs, including habituation.
Therefore add the two scaled columns to that score; do not convexly mix winners,
probabilities or acceptance rates, and do not replace it by compatibility.

The original failure had two parts: the deterministic weight-one path removed
the existing score from ranking, and a single L conflated where to move with
whether to continue. Sections 7.2–7.6 separate those decisions. Positive relation
means support among moving destinations; negative means departure. Positive
continuation may increase hold/wait/lane retention/no-spawn; this is its declared
meaning, not the monotony regression for relation-only use. The comparison uses
the same HV-8 scenes and diagnostic event shapes, without a new audition.
The relation/preview algorithms and dispositions retain Author rule labels;
derivation of the integration contract does not upgrade them to a verified
neural mechanism. No further author decision or listening gate is required.

## 8. Composer control

**[Author rule]**

```text
follow_harmonic_relations(w)       // conditional destination support / departure
follow_harmonic_continuation(w)    // continue / end, wait / act
range of each: [-1,1]
default of each: 0
event type of each: cursor event
evidence label of each: Author rule
```

The independent signed bounds define authored dispositions, not convex-mixture
endpoints. Changing one preserves the other. Both zero retain bit identity with
the existing instrument. Weight changes do not reset memory. First activation
starts from retained observations; unavailable history remains unknown. Preserve
TC, PS, SQ, ME, PX, PE, `harmonic_rhythm` and all three vertical terms unchanged.

## 9. Cost and real-time execution

**[Principle]** At 48 kHz and hop 512, the deadline is `512/48000` seconds, approximately 10.67 ms.

**[Author rule]** The audio thread performs bounded snapshot access, consumer-requested previews, compatibility evaluation against Hmax atoms, and commits. It performs no allocation, graph search, lock wait, or context restructuring.

The worker performs segmentation finalization, correspondence search, statistics, context updates, expiration, and publication through preallocated versioned buffers.

**[Author rule]** Preallocate from the declared B support coordinates, Jmax, Emax, Hmax, Kmax, V, L, and Amax:

```text
support storage: O(Jmax * B + Kmax * B)
graph storage: O(Jmax + Emax)
correspondences: O(Hmax * Smax)
consumer scoring: O(Amax * Hmax * feature_cost)
```

Use producer-certified sparse support with retained residual mass where available. Dense worst-case memory must still be reported. This design does not require a one-cent all-frequency candidate table; it does not remove an enabled TC table’s existing cost.

**[Author rule]** Available audio budget equals the hop deadline minus measured existing worst-case work and the platform’s declared scheduling reserve. A supported-load claim requires measured preview and scoring costs within that remainder.

A prediction from a different query version is unusable. Worker lateness produces baseline behavior and a recorded miss; successful fallback is not successful temporal-model real-time acceptance.

### Orchestrator revision after HV-7

**[Author rule]** Derive the neighborhood and hypothesis work bounds from the declared
load's arrival rate and the worst measured background processing per observation batch.
Hmax remains the existing hypothesis ceiling. Test that ceiling as a structural bound;
it is not an arbitrary proposed operating value. Adopt only counts whose complete
indexed retrieval, extension, context update, expiration and publication fit the hop
arrival budget, including all timestamp groups arriving in that hop. Report the
measurement and arithmetic that determine the adopted counts. Do not hide late work
behind successful audio fallback or an unbounded shutdown drain.

There is one endpoint seed slot per retained hypothesis slot, so N = H <= Hmax;
there is no independent neighborhood-size control. For H = Hmax the implementation
proposes at most 3H retained extensions/omissions, H joint seeds, and 2H source
identity/transformed seeds per new anchor, then keeps H hypotheses. Indexed lookup
visits at most N + Smax entries. The report adopts the largest existing structural
ceiling that the complete measured hop can sustain, with its observed margin against
512/48000 seconds; it does not introduce a provisional lower count or a new tolerance.

The producer derives the expected query from exactly the structural records sent in
that hop's half-open interval. Records on its right boundary remain queued for the
next batch; they must not update the worker's query while disappearing from the
producer's expected-query calculation. Shutdown flushes the final actual boundary
without inventing a later observation. Exact query-version validation remains required.

Report the former 1.57 GB reservation by storage category. Separate the declared dense
worst-case capacity from the live sparse payload and its actual allocation. Sparse
storage must preserve every retained observation, identity and residual; capacity
expiry remains the existing rule. The index is updated at anchor commitment and
invalidated with its underlying evidence. The same ABBA acquisition reports audio
maximum hop, prediction misses, background backlog, startup and peak memory.

This revision replaces dense upfront payload reservation with background-owned sparse
payloads that grow with actual observations, within the unchanged retention limits.
It adds no quantization or support threshold. Allocation capacity, live payload bytes,
and process RSS are separate measurements; dense worst-case storage is still reported.

## 10. Acceptance

1. **[Author rule] Default identity:** bit-identical audio, events, and existing random-stream state at w=0, including configurations with other existing axes enabled.
2. **[Author rule] Acoustic trajectory contrasts:** distinguish preparation, retention, rearticulation, approach, arrival, competition change, silence, partial recurrence, concurrent streams, and context transitions without chord-name labels.
3. **[Author rule] Causal history contrasts:** hold current acoustic state, sensory-context inputs, candidate set, and baseline generator state fixed; change earlier ordering while preserving marginal exposure. Verify changes in prediction, choice, and context responsibility.
4. **[Author rule] Causal ablations:** freeze learning *before* divergent exposure; remove relevant correspondence edges; or construct shuffled histories with matched conditional relation statistics. The corresponding learned-history difference must disappear. An arbitrary shuffle need not erase every surviving pattern.
5. **[Author rule] Declared-load real time:** report maximum hop time, deadline misses, prediction misses, startup cost, and peak memory on the declared machine and load.
6. **[Author rule] Author listening:** once, on the final integrated configuration. Numerical checks establish specified distinctions and causality; they do not judge whether the result sounds better.

## 11. Implementation sequence

1. **[Author rule] Observation and bounded graph — independent of HV-3 completion.** Use injected observations. Complete when segmentation, identity, expiration, continuation, and capacity accounting are deterministic.
2. **[Author rule] Correspondence and prediction — independent.** Complete when partial/multiple-stream mappings, transformations, unknown mass, causal update order, and scale calculations pass structural contrasts.
3. **[Author rule] Contexts and derived relations — independent.** Complete when context birth, responsibility, merge, expiration, arrival, substitution, and pivot evidence behave according to the stated algorithms.
4. **[Author rule] HV-3 adapter — depends on its public observation contract and first-order preview.** Complete when coordinate versions, timestamps, actual-PCM baseline provenance, missing attributes, and numerical errors propagate correctly.
5. **[Author rule] Consumer integration — depends on the adapter.** Connect timing, both pitch selectors, deterministic paths, spawn, and lanes. Complete when zero-weight identity and ordered snapshot commitment hold for every consumer.
6. **[Author rule] Integrated acceptance.** Complete only after all fourteen phenomena have implemented representations and consumers, causal contrasts pass, declared-load deadlines hold, and the author has listened once. HV-8 supplied that audition; the adopted HV-9 correction uses recorded acceptance, with no repeat audition.

### Endless harmony observation reservations

Endless scenes publish vertical support scans from sample zero, including while all
three valuation weights are zero. The temporal observer is separately reserved by
`observe_harmonic_relations(max_sources)` before time advances. The positive integer
bounds simultaneous Voice and Flora lane sources, including retiring sources; its
ceiling is the existing structural-record capacity of 38,400, not a musical limit.
Both signed nonzero temporal controls require this declaration; missing or late
reservation is a positioned script error. Finite scenes warn and ignore the declaration,
retaining event pre-scanning for observer and source capacity. Observation does not
activate either valuation weight. The declaration is an Author rule and does not
change the existing 600-second rolling history or the supported event density.
