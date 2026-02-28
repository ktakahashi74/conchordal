# Conchordal Core Strengthening Proposal (Synced to Current Implementation)

## Scope
- Preserve `consonance -> survival` as the primary viability pressure.
- Strengthen rhythm autonomy without introducing symbolic grids (fixed meter/scale).
- Keep defaults behavior-identical.

## Rhythm Coupling
- Enum: `RhythmCouplingMode`
  - `TemporalOnly` (default)
  - `TemporalTimesVitality { lambda_v, v_floor }`

### Formula
- Temporal term (existing): `k_time = kuramoto_k_eff(...)`
- Vitality modulation (opt-in):
  - `g(v) = clamp01((v - v_floor) / (1 - v_floor))`
  - `k_eff = k_time * clamp(1 + lambda_v * g(v), 0, MAX_COUPLING_MULT)`
- Current constant: `MAX_COUPLING_MULT = 2.0`

Notes:
- If temporal drive is zero, coupling remains zero (vitality does not create coupling alone).
- Runtime core uses sanitized config values from boundary conversion.

## Rhythm Reward for Metabolism
- Configuration shape:
  - `rhythm_reward: Option<MetabolismRhythmReward>`
  - `None` means reward modulation is disabled.
- Struct: `MetabolismRhythmReward { rho_t, metric }`
- Metric enum (current): `RhythmRewardMetric::AttackPhaseMatch` only.
- Future metric candidates (e.g. `phase_lock`) are out of current scope.

### Metric
- At attack:
  - `T = clamp01(0.5 + 0.5 * cos(phase_err_at_attack))`

### Energy Rule
- Attack delta is applied as:
  - `delta = -action_cost + recharge_rate * clamp01(consonance) * recharge_multiplier`
  - `recharge_multiplier = clamp(1 + rho_t * T, 0, MAX_RECHARGE_MULT)`
- Equivalent interpretation: recharge is multiplied by `(1 + rho_t * T)`,
  but still gated by consonance because recharge already includes `clamp01(consonance)`.

Why this preserves consonance pressure:
- Extra reward only scales the recharge term.
- If consonance is zero, reward bonus is zero by construction.
- Rhythm alignment cannot bypass consonance-dependent viability.

## Rhai API (Current)
- `species.rhythm_coupling("temporal")`
- `species.rhythm_coupling_vitality(lambda_v, v_floor)`
- `species.rhythm_reward(rho_t, "attack_phase_match")`
- Group-handle variants with the same names are also available for draft groups.

## Default Behavior
- `rhythm_coupling = TemporalOnly`
- `rhythm_reward = None`
- Existing runtime behavior remains unchanged unless scenarios opt in.
