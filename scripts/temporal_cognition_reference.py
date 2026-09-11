"""Small f64 references for the temporal-DCC specification and M0 candidates.

These functions establish algebra and support conventions, not human validity or
the real-time implementation. They deliberately retain unsupported observations.
"""

import math
from dataclasses import dataclass, field


def mono_observation(frames, channels):
    """Downmix observed frames once; None means a missing time, not silence."""
    if not isinstance(channels, int) or channels < 1:
        raise ValueError("channels must be positive")
    mono = []
    for frame in frames:
        if frame is None:
            mono.append(None)
            continue
        if len(frame) != channels or not all(math.isfinite(x) for x in frame):
            raise ValueError("observed frames must contain finite samples in every channel")
        mono.append(math.fsum(frame) / channels)
    observed = [x for x in mono if x is not None]
    energy = math.fsum(x * x for x in observed) / len(observed) if observed else None
    return {"mono": mono, "observed_times": len(observed),
            "coverage": len(observed) / len(mono) if mono else 0.0,
            "energy": energy, "rms": math.sqrt(energy) if energy is not None else None}


def episode_log_availability(strength, elapsed_sec, interference, tau_sec, kappa):
    """Section 9.3 availability before its numerical retrieval floor."""
    values = [strength, elapsed_sec, interference, tau_sec, kappa]
    if (not all(math.isfinite(x) for x in values) or min(values) < 0
            or tau_sec == 0 or kappa == 0):
        raise ValueError("finite nonnegative state and positive time/interference scales required")
    if strength == 0:
        return -math.inf
    return math.log(strength) - elapsed_sec / tau_sec - interference / kappa


def recognition_probability(log_availabilities, match_scores, bias, epsilon_avail=1e-300):
    """The assay's stored-bank competition, with no cue-length renormalization."""
    if (len(log_availabilities) != len(match_scores) or not math.isfinite(bias)
            or not math.isfinite(epsilon_avail) or epsilon_avail <= 0):
        raise ValueError("aligned supported episodes, finite bias and positive floor required")
    floor = math.log(epsilon_avail)
    scores = []
    for available, match in zip(log_availabilities, match_scores):
        if not math.isfinite(match) or (not math.isfinite(available) and available != -math.inf):
            raise ValueError("invalid episode score")
        if available == -math.inf:
            continue
        hi, lo = max(available, floor), min(available, floor)
        scores.append(hi + math.log1p(math.exp(lo - hi)) + match)
    if not scores:
        return 0.0
    maximum = max(scores)
    log_return = maximum + math.log(math.fsum(math.exp(s - maximum) for s in scores))
    difference = log_return - bias
    if difference >= 0:
        return 1 / (1 + math.exp(-difference))
    odds = math.exp(difference)
    return odds / (1 + odds)


def competing_event_step(rates_per_sec, elapsed_sec):
    """One first-exit competition with physical time, not per-hop probabilities."""
    if (not math.isfinite(elapsed_sec) or elapsed_sec < 0
            or not all(math.isfinite(r) and r >= 0 for r in rates_per_sec)):
        raise ValueError("finite nonnegative rates and duration required")
    total = math.fsum(rates_per_sec)
    exited = -math.expm1(-total * elapsed_sec)
    return {"stay": math.exp(-total * elapsed_sec),
            "exit": [exited * r / total if total else 0.0 for r in rates_per_sec]}


def duration_hazard_increment(intercept, duration_coefficient, start_sec, elapsed_sec, multiplier=1.0):
    """Two-point Gauss-Legendre integration over one original observed hop.

    Intercept includes the frozen context dot product. The duration coefficient
    multiplies raw log(1+d/1s); standardized section coefficients are converted
    to that form before calling. Calibration scales each local integral. This
    function does not coarsen batched hops or integrate unknown gaps.
    """
    if (not all(math.isfinite(x) for x in (intercept, duration_coefficient, elapsed_sec, multiplier))
            or elapsed_sec < 0 or multiplier <= 0):
        raise ValueError("finite coefficients, nonnegative duration and positive multiplier required")
    if start_sec is None:
        return None
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError("known elapsed foreground duration must be finite and nonnegative")
    half = elapsed_sec / 2
    midpoint = start_sec + half
    values = []
    for duration in (midpoint - half / math.sqrt(3), midpoint + half / math.sqrt(3)):
        predictor = intercept + duration_coefficient * math.log1p(duration)
        values.append(max(predictor, 0) + math.log1p(math.exp(-abs(predictor))))
    integral = half * math.fsum(values) * multiplier
    if not math.isfinite(integral):
        raise ValueError("integrated hazard must remain finite")
    return integral


def gap_survival(intercept, duration_coefficient, start_sec, elapsed_sec, multiplier=1.0,
                 absolute_tolerance=1e-9, relative_tolerance=1e-7, max_evaluations=64):
    """Propagate a missing interval under the last supported context as a prior.

    Composite two-point Gauss-Legendre doubles panels within the evaluation cap.
    A fourth-derivative bound also guards convergence: two small quadrature sums
    alone can miss a narrow initial hazard. Bernstein coefficients bound the
    derivative polynomial over the endpoint sigmoid range. Endpoint evaluations
    count toward the same cap. No exit across missing sound has an observed type.
    """
    if (not all(math.isfinite(v) for v in (intercept, duration_coefficient, elapsed_sec,
                                          multiplier, absolute_tolerance, relative_tolerance))
            or elapsed_sec < 0 or multiplier <= 0 or min(absolute_tolerance, relative_tolerance) < 0
            or absolute_tolerance + relative_tolerance == 0
            or not isinstance(max_evaluations, int) or not 4 <= max_evaluations <= 64):
        raise ValueError("finite hazard, nonnegative tolerances and4..64 evaluations required")
    result = {"status": "unknown_start", "integrated_hazard": None, "log_survival": None,
              "retained_foreground_mass": 0.0, "unknown_current_mass": 1.0,
              "observed_exit_mass": 0.0, "evaluations": 0, "panels": 0,
              "last_integral_estimate": None, "doubling_difference": None,
              "error_bound": None, "tolerance": None}
    if start_sec is None:
        return result
    if not math.isfinite(start_sec) or start_sec < 0 or not math.isfinite(start_sec + elapsed_sec):
        raise ValueError("finite known start and elapsed endpoint required")
    if elapsed_sec == 0:
        result.update(status="prior_survival", integrated_hazard=0.0, log_survival=0.0,
                      retained_foreground_mass=1.0, unknown_current_mass=0.0,
                      last_integral_estimate=0.0, error_bound=0.0, doubling_difference=0.0,
                      tolerance=absolute_tolerance)
        return result
    result["status"] = "unresolved_budget"
    try:
        sigmoids = []
        for duration in (start_sec, start_sec + elapsed_sec):
            predictor = intercept + duration_coefficient * math.log1p(duration)
            result["evaluations"] += 1
            if not math.isfinite(predictor):
                raise OverflowError
            odds = math.exp(-abs(predictor))
            sigmoids.append(1 / (1 + odds) if predictor >= 0 else odds / (1 + odds))
        low, high = min(sigmoids), max(sigmoids)
        b = duration_coefficient
        # h''''(d) = P(sigmoid(a+b*log(1+d))) / (1+d)^4.
        coefficients = [0, b**4 - 6*b**3 + 11*b*b - 6*b,
                        -7*b**4 + 18*b**3 - 11*b*b, 12*b**4 - 12*b**3, -6*b**4]
        power = [math.fsum(coefficients[k] * math.comb(k, j) * low**(k-j) * (high-low)**j
                           for k in range(j, 5)) for j in range(5)]
        bernstein = [math.fsum(power[j] * math.comb(i, j) / math.comb(4, j)
                              for j in range(i + 1)) for i in range(5)]
        rounding = 64 * math.ulp(1.0) * math.fsum(abs(v) for v in coefficients)
        derivative_bound = (max(abs(v) for v in bernstein) + rounding) / (1 + start_sec)**4
        result["evaluations"] += 2
        previous = duration_hazard_increment(intercept, b, start_sec, elapsed_sec, multiplier)
        result.update(panels=1, last_integral_estimate=previous)
        panels = 2
        while result["evaluations"] + 2 * panels <= max_evaluations:
            width = elapsed_sec / panels
            integrals = []
            for panel in range(panels):
                result["evaluations"] += 2
                integrals.append(duration_hazard_increment(intercept, b, start_sec + panel * width,
                                                          width, multiplier))
            integral = math.fsum(integrals)
            difference = abs(integral - previous)
            error = (elapsed_sec * width**4 * derivative_bound * multiplier / 4320
                     + 128 * math.ulp(1.0) * integral)
            tolerance = absolute_tolerance + relative_tolerance * integral
            result.update(panels=panels, last_integral_estimate=integral,
                          doubling_difference=difference, error_bound=error, tolerance=tolerance)
            if max(difference, error) <= tolerance:
                result.update(status="prior_survival", integrated_hazard=integral,
                              log_survival=-integral, retained_foreground_mass=math.exp(-integral),
                              unknown_current_mass=-math.expm1(-integral))
                return result
            previous = integral
            panels *= 2
    except (OverflowError, ValueError):
        result["status"] = "unresolved_numeric"
    return result


def mixture_first_event(contexts, n_hops, n_types, snapshot_kind, snapshot_time):
    """Finite-hop first-event law from one frozen context/path-weight snapshot.

    Paths supply integrated hazards, conditional exit-type probabilities and
    required-history retention per original hop. Loss precedes every group's
    exit test; ties belong to the smallest stable group handle. Already resolved
    first events retain their scoring register despite later observation loss.
    Missing context/path weight remains unknown. This neither extracts features
    nor implements ancestry/register pruning or issue-snapshot retention.
    """
    if (not isinstance(n_hops, int) or n_hops < 0 or not isinstance(n_types, int) or n_types < 1
            or snapshot_kind not in ("issued_forecast", "heard_prefix_annotation")
            or not math.isfinite(snapshot_time)):
        raise ValueError("valid hop/type counts and an explicit finite scoring snapshot required")
    context_weights = [context["weight"] for context in contexts]
    if (not all(math.isfinite(w) and 0 <= w <= 1 for w in context_weights)
            or math.fsum(context_weights) > 1 + 1e-12):
        raise ValueError("context weights must be nonnegative with total at most one")
    initial_known = 0.0
    survival = [0.0] * (n_hops + 1)
    mid_unresolved = [0.0] * (n_hops + 1)
    events = [[0.0] * n_types for _ in range(n_hops)]
    for context in contexts:
        groups = context["groups"]
        handles = [group["handle"] for group in groups]
        if (not all(isinstance(handle, int) and handle >= 0 for handle in handles)
                or len(set(handles)) != len(handles)):
            raise ValueError("group handles must be unique nonnegative integers")
        prepared = []
        for group in sorted(groups, key=lambda group: group["handle"]):
            weights = [path["weight"] for path in group["paths"]]
            if (not all(math.isfinite(w) and 0 <= w <= 1 for w in weights)
                    or math.fsum(weights) > 1 + 1e-12):
                raise ValueError("conditional path weights must have total at most one")
            start_mass = 0.0
            kept = [0.0] * n_hops
            after_exit = [0.0] * n_hops
            exited = [[0.0] * n_types for _ in range(n_hops)]
            for path in group["paths"]:
                hazards, types, retention = (path[key] for key in
                                              ("hazard_increments", "exit_types", "history_retention"))
                if (not isinstance(path["known_at_start"], bool)
                        or any(len(values) != n_hops for values in (hazards, types, retention))
                        or not all(math.isfinite(h) and h >= 0 for h in hazards)
                        or not all(math.isfinite(r) and 0 <= r <= 1 for r in retention)):
                    raise ValueError("aligned finite hazards and history retention required")
                alive = path["weight"] if path["known_at_start"] else 0.0
                start_mass += alive
                for hop, (hazard, probabilities, retained) in enumerate(zip(hazards, types, retention)):
                    if (len(probabilities) != n_types
                            or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                            or not math.isclose(math.fsum(probabilities), 1, abs_tol=1e-12, rel_tol=0)):
                        raise ValueError("each exit-type vector must be a normalized distribution")
                    alive *= retained
                    kept[hop] += alive
                    mass = alive * -math.expm1(-hazard)
                    for kind, probability in enumerate(probabilities):
                        exited[hop][kind] += mass * probability
                    alive *= math.exp(-hazard)
                    after_exit[hop] += alive
            prepared.append((start_mass, kept, after_exit, exited))
        weight = context["weight"]
        previous = weight * math.prod(group[0] for group in prepared)
        initial_known += previous
        survival[0] += previous
        lost = 0.0
        for hop in range(n_hops):
            kept = [group[1][hop] for group in prepared]
            after_exit = [group[2][hop] for group in prepared]
            lost += max(0.0, previous - weight * math.prod(kept))
            mid_unresolved[hop + 1] += lost
            previous = weight * math.prod(after_exit)
            survival[hop + 1] += previous
            for index, group in enumerate(prepared):
                others = weight * math.prod(after_exit[:index]) * math.prod(kept[index + 1:])
                for kind, mass in enumerate(group[3][hop]):
                    events[hop][kind] += others * mass
    cdf = [0.0]
    for row in events:
        cdf.append(cdf[-1] + math.fsum(row))
    unresolved = [max(0.0, 1 - initial_known) + lost for lost in mid_unresolved]
    return {"snapshot": {"kind": snapshot_kind, "time": snapshot_time},
            "initial_known": initial_known, "survival": survival,
            "event_mass": events, "cdf": cdf, "mid_unresolved": mid_unresolved,
            "unresolved": unresolved,
            "type_distribution": [math.fsum(row[k] for row in events) for k in range(n_types)]
            + [survival[-1], unresolved[-1]]}


def ordinal_probabilities(predictor, cutpoints):
    """Five-level cumulative-logit head; constant beta is omitted/fixed at zero."""
    if (len(cutpoints) != 4 or not math.isfinite(predictor)
            or not all(math.isfinite(a) for a in cutpoints)
            or any(a >= b for a, b in zip(cutpoints, cutpoints[1:]))):
        raise ValueError("finite predictor and four strictly ordered cutpoints required")
    cumulative = [0.0]
    for point in cutpoints:
        x = point - predictor
        if x >= 0:
            cumulative.append(1 / (1 + math.exp(-x)))
        else:
            odds = math.exp(x)
            cumulative.append(odds / (1 + odds))
    cumulative.append(1.0)
    return [b - a for a, b in zip(cumulative, cumulative[1:])]


def timing_lookup(masses, position, periodic):
    """Interpolate bin mass at centers; linear overflow keeps its normalizing mass."""
    n = len(masses) if periodic else len(masses) - 1
    if (n < 2 or not math.isfinite(position)
            or not all(math.isfinite(m) and m >= 0 for m in masses)):
        raise ValueError("finite nonnegative bins and finite relative position required")
    total = math.fsum(masses)
    if total == 0 or (not periodic and not 0 <= position <= 4):
        return None
    coordinate = n * (position % 1 if periodic else position / 4) - 0.5
    if not periodic:
        coordinate = min(n - 1, max(0.0, coordinate))
    left = math.floor(coordinate)
    fraction = coordinate - left
    right = (left + 1) % n if periodic else min(n - 1, left + 1)
    left = left % n if periodic else left
    return ((1 - fraction) * masses[left] + fraction * masses[right]) / total


def known_overlap(group_masses):
    """Section 9.4 known yes/no/unknown competition, conditional on group paths."""
    for triple in group_masses:
        if (len(triple) != 3 or not all(math.isfinite(x) and x >= 0 for x in triple)
                or not math.isclose(math.fsum(triple), 1.0, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("each group must provide normalized yes/no/unknown mass")
    yes = 1 - math.prod(no + unknown for _, no, unknown in group_masses)
    no = math.prod(no for _, no, _ in group_masses)
    return (yes, no, max(0.0, 1 - yes - no))


def section_head_covariates(cumulative, recent, section_start, epoch_start, observed_time,
                            known_intervals, match_scores, match_support_end,
                            means, standard_deviations, forecast_time=None):
    """Assemble the 82 covariates from already validated path-local descriptors.

    The caller selects the actual section/generation lineage and distinct eligible
    episode scores. This does not extract descriptors or perform correspondence.
    """
    if len(cumulative) != 39 or len(recent) != 39 or len(means) != 82 or len(standard_deviations) != 82:
        raise ValueError("section descriptors require 39 entries and head scales require 82")
    if (not all(x is None or math.isfinite(x) for x in [*cumulative, *recent])
            or not all(math.isfinite(x) for x in [epoch_start, observed_time, *means, *standard_deviations, *match_scores])
            or any(x < 0 for x in standard_deviations) or observed_time < epoch_start):
        raise ValueError("finite supported values, nonnegative deviations and ordered epoch required")
    if section_start is None:
        return {"supported": False, "reason": "unknown_section_start", "covariates": None}
    if not math.isfinite(section_start) or section_start > observed_time:
        raise ValueError("section start must be a finite observed time")
    prediction_time = observed_time if forecast_time is None else forecast_time
    if not math.isfinite(prediction_time) or prediction_time < observed_time:
        raise ValueError("forecast cannot precede its supporting observation")
    start = max(epoch_start, section_start)
    if start == observed_time:
        return {"supported": False, "reason": "empty_section_window", "covariates": None}
    intervals = []
    for lo, hi in known_intervals:
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo or hi > observed_time:
            raise ValueError("known support must be ordered and available at observation time")
        if hi > start and lo < observed_time:
            intervals.append((max(lo, start), min(hi, observed_time)))
    union, last_end = [], start
    for lo, hi in sorted(intervals):
        if hi <= last_end:
            continue
        union.append(hi - max(last_end, lo))
        last_end = hi
    known_duration = math.fsum(union)
    missing_fraction = max(0.0, min(1.0, 1 - known_duration / (observed_time - start)))
    scores = sorted(match_scores, reverse=True)
    if match_support_end is not None and (not math.isfinite(match_support_end) or match_support_end > observed_time):
        raise ValueError("matcher support must be an available observation endpoint")
    fresh = match_support_end is not None and epoch_start <= match_support_end and observed_time - match_support_end < 0.5
    best = scores[0] if scores and fresh else None
    gap = scores[0] - scores[1] if len(scores) >= 2 and fresh else None
    delta = [r - c if r is not None and c is not None else None for c, r in zip(cumulative, recent)]
    raw = list(cumulative) + delta + [math.log1p(prediction_time - start), missing_fraction, best, gap]
    valid = [x is not None for x in raw]
    values = [(x - mean) / max(sd, 1e-6) if x is not None else 0.0
              for x, mean, sd in zip(raw, means, standard_deviations)]
    return {"supported": True, "covariates": values, "valid": valid, "raw": raw,
            "window": [start, observed_time], "known_duration": known_duration,
            "missing_fraction": missing_fraction, "forecast_time": prediction_time}


def executed_reference_credit(entries, observed_fraction, epoch, retained_keys, confirmed=True):
    """Freeze issue-time mass; lost handles/anchors never donate mass to survivors.

    Bin updates use the conditional distribution of supported anchors with this
    credit. Multiplying by unnormalized anchor bins would count coverage twice.
    Interference deltas are applied only to traces that actually exist.
    """
    if (not math.isfinite(observed_fraction) or not 0 <= observed_fraction <= 1
            or len(entries) > 16):
        raise ValueError("at most 16 references and observed fraction in [0,1] required")
    keys = [e["key"] for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("issue-time keys must already be coalesced")
    if (not all(math.isfinite(e["weight"]) and 0 <= e["weight"] <= 1
                and math.isfinite(e["anchor_coverage"]) and 0 <= e["anchor_coverage"] <= 1 for e in entries)
            or math.fsum(e["weight"] for e in entries) > 1 + 1e-12):
        raise ValueError("normalized inventory and valid anchor coverage required")
    credits = {e["key"]: observed_fraction * e["weight"] * e["anchor_coverage"]
               if confirmed and e["key"][0] == epoch and e["key"] in retained_keys else 0.0
               for e in entries}
    assigned = math.fsum(credits.values())
    interference = {key: math.fsum(value for other, value in credits.items() if other != key)
                    for key in retained_keys if key[0] == epoch}
    return {"credits": credits, "unassigned": max(0.0, 1 - assigned),
            "interference": interference}


def uniform_difference_cdf(value, outcome_interval, anchor_interval):
    """CDF of independent known uniform timestamps; point timestamps are allowed."""
    a, b = outcome_interval
    c, d = anchor_interval
    if (not all(math.isfinite(x) for x in [value, a, b, c, d]) or a > b or c > d):
        raise ValueError("ordered finite known timestamp intervals required")
    lower, upper = a - d, b - c
    if value < lower:
        return 0.0
    if value >= upper:
        return 1.0
    width_x, width_y = b - a, d - c
    position = value - lower
    if width_x == 0 or width_y == 0:
        return position / max(width_x, width_y)
    # Reflect the upper half to avoid cancellation near probability one.
    reflected = position > (width_x + width_y) / 2
    z = width_x + width_y - position if reflected else position
    small, large = sorted([width_x, width_y])
    if z <= small:
        cumulative = z * z / (2 * width_x * width_y)
    else:
        cumulative = (z - small / 2) / large
    return 1 - cumulative if reflected else cumulative


def integrate_timing_bins(outcome_interval, anchor_alternatives, periodic, bins=32):
    """Retain phase alternatives and known overflow; missing anchors stay unassigned.

    Each alternative supplies weight, interval (or None), and frozen period_sec.
    For linear timing, period_sec is the Voice's intrinsic period at issue.
    """
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("at least two bins required")
    masses = [0.0] * (bins if periodic else bins + 1)
    weights = [a["weight"] for a in anchor_alternatives]
    if (not all(math.isfinite(w) and w >= 0 for w in weights)
            or math.fsum(weights) > 1 + 1e-12):
        raise ValueError("anchor alternatives must retain normalized nonnegative weights")
    if outcome_interval is None:
        return {"bins": masses, "unsupported": 1.0}
    observed_anchor_weight = 0.0
    for alternative in anchor_alternatives:
        anchor = alternative["interval"]
        weight = alternative["weight"]
        if anchor is None or weight == 0:
            continue
        period = alternative["period_sec"]
        if not math.isfinite(period) or period <= 0:
            raise ValueError("frozen period must be finite and positive")
        uniform_difference_cdf(0, outcome_interval, anchor)
        low = outcome_interval[0] - anchor[1]
        high = outcome_interval[1] - anchor[0]
        observed_anchor_weight += weight
        if low == high:
            coordinate = low / period
            if periodic:
                index = min(bins - 1, int((coordinate % 1) * bins))
            elif 0 <= coordinate <= 4:
                index = min(bins - 1, int(coordinate * bins / 4))
            else:
                index = bins
            masses[index] += weight
            continue
        if periodic:
            for cycle in range(math.floor(low / period), math.floor(high / period) + 1):
                for index in range(bins):
                    left = (cycle + index / bins) * period
                    right = (cycle + (index + 1) / bins) * period
                    probability = (uniform_difference_cdf(right, outcome_interval, anchor)
                                   - uniform_difference_cdf(left, outcome_interval, anchor))
                    masses[index] += weight * max(0.0, probability)
        else:
            inside = []
            for index in range(bins):
                left, right = 4 * period * index / bins, 4 * period * (index + 1) / bins
                probability = (uniform_difference_cdf(right, outcome_interval, anchor)
                               - uniform_difference_cdf(left, outcome_interval, anchor))
                inside.append(max(0.0, probability))
                masses[index] += weight * max(0.0, probability)
            masses[-1] += weight * max(0.0, 1 - math.fsum(inside))
    return {"bins": masses, "unsupported": max(0.0, 1 - observed_anchor_weight)}


def private_bin_probabilities(strengths, elapsed_sec, interference, tau_sec, kappa):
    """Normalize conditional trace mass in log space; no support stays unknown.

    This does not define how occurrence contributions are accumulated or capped.
    The episode retrieval floor is not a pseudocount for an unseen timing bin.
    """
    if not (len(strengths) == len(elapsed_sec) == len(interference)):
        raise ValueError("aligned bin states required")
    values = [episode_log_availability(s, t, i, tau_sec, kappa)
              for s, t, i in zip(strengths, elapsed_sec, interference)]
    largest = max(values, default=-math.inf)
    if largest == -math.inf:
        return None
    masses = [math.exp(v - largest) for v in values]
    total = math.fsum(masses)
    return [m / total for m in masses]


def integrate_retained_timing_bins(outcome_interval, anchor_alternatives, periodic,
                                  tau_sec, bins=32):
    """Integrate bin support times exp(-(observed_end-outcome_time)/tau).

    Uniform timestamps are integrated before the bin cap. Unsupported anchor
    mass is unchanged by retention. A report's delivery time is not an input.
    """
    if not math.isfinite(tau_sec) or tau_sec <= 0:
        raise ValueError("positive finite time scale required")
    support = integrate_timing_bins(outcome_interval, anchor_alternatives, periodic, bins)
    if outcome_interval is None or outcome_interval[0] == outcome_interval[1]:
        return support
    origin, end = outcome_interval
    width = end - origin
    masses = [0.0] * len(support["bins"])
    for alternative in anchor_alternatives:
        anchor, weight = alternative["interval"], alternative["weight"]
        if anchor is None or weight == 0:
            continue
        c, d = anchor[0] - origin, anchor[1] - origin
        period = alternative["period_sec"]
        cycles = range(math.floor(-d / period), math.floor((width - c) / period) + 1) if periodic else [0]
        inside = [0.0] * bins
        for cycle in cycles:
            for index in range(bins):
                left = (cycle + index / bins) * period if periodic else 4 * period * index / bins
                right = left + period / bins * (1 if periodic else 4)
                if c == d:
                    lo, hi = max(0.0, c + left), min(width, c + right)
                    points = [lo, hi] if lo < hi else []
                else:
                    points = sorted({0.0, width, *(max(0.0, min(width, x)) for x in
                                                  [c + left, c + right, d + left, d + right])})
                for lo, hi in zip(points, points[1:]):
                    if c == d:
                        p_lo = p_hi = 1.0
                    else:
                        p_lo = max(0.0, min(d, lo - left) - max(c, lo - right)) / (d - c)
                        p_hi = max(0.0, min(d, hi - left) - max(c, hi - right)) / (d - c)
                    q = (hi - lo) / tau_sec
                    if q < 1e-3:
                        # These convergent moments avoid subtracting nearly equal numbers.
                        moment0, moment1, term = 0.0, 0.0, 1.0
                        for k in range(8):
                            moment0 += term / (k + 1)
                            moment1 += term / ((k + 1) * (k + 2))
                            term *= -q / (k + 1)
                    else:
                        moment0 = -math.expm1(-q) / q
                        moment1 = (1 - moment0) / q
                    value = ((hi - lo) / width * math.exp((hi - width) / tau_sec)
                             * (p_lo * moment0 + (p_hi - p_lo) * moment1))
                    inside[index] += max(0.0, value)
        for index, mass in enumerate(inside):
            masses[index] += weight * mass
        if not periodic:
            total = -math.expm1(-width / tau_sec) * tau_sec / width
            masses[-1] += weight * max(0.0, total - math.fsum(inside))
    return {"bins": masses, "unsupported": support["unsupported"]}


@dataclass
class PrivateTimingTrace:
    """A bounded retained-mass filter, separate from episode-strength renewal.

    The caller supplies sealed outcomes in (observed end, onset/release, ID)
    order. Replay, loss, or delivery reordering is resolved before this filter.
    This f64 reference has no claim about real-time allocation or latency.
    """

    epoch: int
    tau_sec: float
    kappa: float
    strength_max: float
    capacity: int = 16
    bins: int = 32
    traces: dict = field(default_factory=dict, init=False)
    last_order: tuple | None = field(default=None, init=False)

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in [self.tau_sec, self.kappa, self.strength_max])
                or not isinstance(self.capacity, int) or not 1 <= self.capacity <= 16
                or not isinstance(self.bins, int) or self.bins < 2):
            raise ValueError("positive parameters, capacity in [1,16], and at least two bins required")

    def probabilities(self, key, head):
        if head not in ("onset", "release"):
            raise ValueError("onset or release head required")
        if key not in self.traces:
            return None
        values = self.traces[key][head]
        largest = max(values)
        if largest == -math.inf:
            return None
        masses = [math.exp(v - largest) for v in values]
        total = math.fsum(masses)
        return [m / total for m in masses]

    def observe(self, outcome_id, head, outcome_interval, entries, observed_fraction,
                retained_keys, confirmed=True):
        """Apply one physical outcome, with all frozen reference alternatives.

        Entries contain key, weight and anchors. Keys end in periodic/nonperiodic.
        Age old bins, add integrated heard credit, cap each bin, then apply other
        reference credit to both heads once. Never reset accumulated interference.
        The returned pre-update view is diagnostic. Prequential scoring must use
        probabilities separately frozen at action issue, before intervening updates.
        """
        if head not in ("onset", "release") or not isinstance(outcome_id, int) or outcome_id < 0:
            raise ValueError("canonical nonnegative outcome ID and onset/release required")
        if any(len(e["key"]) != 3 or e["key"][2] not in ("periodic", "nonperiodic") for e in entries):
            raise ValueError("epoch, episode and coordinate-family keys required")
        supported, retained, credits_input = {}, {}, []
        for entry in entries:
            key = entry["key"]
            periodic = key[2] == "periodic"
            supported[key] = integrate_timing_bins(outcome_interval, entry["anchors"], periodic, self.bins)
            retained[key] = integrate_retained_timing_bins(
                outcome_interval, entry["anchors"], periodic, self.tau_sec, self.bins)
            credits_input.append({"key": key, "weight": entry["weight"],
                                  "anchor_coverage": 1 - supported[key]["unsupported"]})
        credit = executed_reference_credit(credits_input, observed_fraction, self.epoch,
                                           retained_keys, confirmed)
        if not confirmed or outcome_interval is None:
            return {**credit, "applied": False, "evicted": [], "removed": [], "before": {}}
        start, end = outcome_interval
        if not all(math.isfinite(t) for t in [start, end]) or start > end:
            raise ValueError("ordered finite outcome support required")
        order = (end, 0 if head == "onset" else 1, outcome_id)
        if self.last_order is not None and order <= self.last_order:
            raise ValueError("duplicate or out-of-order sealed outcome")
        removed = [key for key in self.traces if key not in retained_keys or key[0] != self.epoch]
        for key in removed:
            del self.traces[key]
        before = {key: self.probabilities(key, head) for key, r in credit["credits"].items() if r > 0}
        for key, r in credit["credits"].items():
            if r > 0 and key not in self.traces:
                count = self.bins + (key[2] == "nonperiodic")
                self.traces[key] = {"onset": [-math.inf] * count,
                                    "release": [-math.inf] * count, "end": end}
        for key, trace in self.traces.items():
            elapsed = end - trace["end"]
            r = credit["credits"].get(key, 0.0)
            support = 1 - supported[key]["unsupported"] if r > 0 else 0.0
            interference = credit["interference"].get(key, 0.0) / self.kappa
            for event_head in ("onset", "release"):
                for index, old in enumerate(trace[event_head]):
                    value = old - elapsed / self.tau_sec
                    added = r * retained[key]["bins"][index] / support if r > 0 and event_head == head else 0.0
                    if added > 0:
                        new = math.log(added)
                        largest, smallest = max(value, new), min(value, new)
                        value = min(math.log(self.strength_max), largest + math.log1p(math.exp(smallest - largest)))
                    trace[event_head][index] = value - interference
            trace["end"] = end
        evicted = []
        while len(self.traces) > self.capacity:
            availability = {}
            for key, trace in self.traces.items():
                values = trace["onset"] + trace["release"]
                largest = max(values)
                availability[key] = largest + math.log(math.fsum(math.exp(v - largest) for v in values))
            victim = min(self.traces, key=lambda key: (availability[key], key))
            del self.traces[victim]
            evicted.append(victim)
        self.last_order = order
        return {**credit, "applied": True, "evicted": evicted, "removed": removed, "before": before}


def periodic_observation_probability(probabilities, outcome_interval, anchor, period_sec):
    """Probability of a known interval under a piecewise-uniform phase density.

    This independent observation audit covers one periodic reference and one
    known anchor alternative, not a joint physical generator for many references.
    The query spans at most one period; zero-duration queries have zero mass.
    """
    if (len(probabilities) < 2 or not all(math.isfinite(q) and q >= 0 for q in probabilities)
            or not math.isclose(math.fsum(probabilities), 1, abs_tol=1e-12, rel_tol=0)
            or not math.isfinite(period_sec) or period_sec <= 0):
        raise ValueError("normalized probabilities and positive finite period required")
    width = outcome_interval[1] - outcome_interval[0]
    if not 0 <= width <= period_sec:
        raise ValueError("observed interval must span at most one period")
    timing = integrate_timing_bins(outcome_interval,
        [{"weight": 1, "interval": anchor, "period_sec": period_sec}], True, len(probabilities))
    return width * len(probabilities) / period_sec * math.fsum(
        q * p for q, p in zip(probabilities, timing["bins"]))


def timing_observation_kernel(window, records, anchors, periodic, bins=32):
    """Lift bin probabilities to one prespecified physical opportunity window.

    For each frozen anchor, draw a bin, then a uniform time in that bin's legal
    intersection with the window. An inaccessible bin uses the uniform window
    baseline. Integrate the frozen anchor prior without changing its weights.
    Records are disjoint known intervals with detection probabilities; the last
    output is missing detection, including window gaps. This is an assay law,
    not a new live action scheduler or a physical model of failed execution.
    """
    start, end = window
    if (not all(math.isfinite(v) for v in window) or start >= end
            or not isinstance(bins, int) or bins < 2):
        raise ValueError("finite positive opportunity duration and at least two bins required")
    duration = end - start
    canonical = []
    previous = start
    for record in records:
        left, right = record["interval"]
        detection = record.get("detection", 1.0)
        if (not all(math.isfinite(v) for v in [left, right, detection])
                or not previous <= left < right <= end or not 0 <= detection <= 1):
            raise ValueError("ordered disjoint known record intervals and detection in [0,1] required")
        canonical.append((left - start, right - start, detection))
        previous = right
    weights = [a["weight"] for a in anchors]
    if (not all(math.isfinite(v) and v >= 0 for v in weights)
            or math.fsum(weights) > 1 + 1e-12):
        raise ValueError("nonnegative frozen anchor weights summing to at most one required")
    count = bins if periodic else bins + 1
    baseline = [(right - left) * detection / duration for left, right, detection in canonical]
    baseline.append(max(0.0, 1 - math.fsum(baseline)))
    matrix = [[0.0] * count for _ in baseline]
    unreachable = [0.0] * count
    coverage = 0.0
    for alternative in anchors:
        weight, interval = alternative["weight"], alternative["interval"]
        if interval is None or weight == 0:
            continue
        a, b = interval[0] - start, interval[1] - start
        period = alternative["period_sec"]
        if not all(math.isfinite(v) for v in [a, b, period]) or a > b or period <= 0:
            raise ValueError("ordered known anchor support and positive frozen period required")
        coverage += weight
        for index in range(count):
            if periodic:
                cycles = range(math.floor(-b / period), math.floor((duration - a) / period) + 1)
                offsets = [((cycle + index / bins) * period, (cycle + (index + 1) / bins) * period)
                           for cycle in cycles]
            elif index < bins:
                offsets = [(4 * period * index / bins, 4 * period * (index + 1) / bins)]
            else:
                offsets = [(-math.inf, 0.0), (4 * period, math.inf)]
            # All lengths are affine between these anchor positions.
            points = {a, b}
            boundaries = {0.0, duration, *(x for row in canonical for x in row[:2])}
            for left, right in offsets:
                for edge in (left, right):
                    if math.isfinite(edge):
                        points.update(boundary - edge for boundary in boundaries if a < boundary - edge < b)
            points = sorted(points)
            segments = [(a, a)] if a == b else list(zip(points, points[1:]))
            for lo, hi in segments:
                segment_weight = weight if a == b else weight * (hi - lo) / (b - a)
                denominators = [math.fsum(max(0.0, min(duration, anchor + right) - max(0.0, anchor + left))
                                         for left, right in offsets) for anchor in (lo, hi)]
                d0, d1 = denominators
                if d0 == d1 == 0:
                    unreachable[index] += segment_weight
                    for row, probability in enumerate(baseline):
                        matrix[row][index] += segment_weight * probability
                    continue
                known = []
                for left, right, detection in canonical:
                    n0, n1 = [math.fsum(max(0.0, min(right, anchor + upper) - max(left, anchor + lower))
                                       for lower, upper in offsets) for anchor in (lo, hi)]
                    if d0 == 0:
                        mean = n1 / d1
                    elif d1 == 0:
                        mean = n0 / d0
                    else:
                        z = (d1 - d0) / (d1 + d0)
                        if abs(z) < 1e-3:
                            even = math.fsum(z ** (2 * k) / (2 * k + 1) for k in range(8))
                            odd = -math.fsum(z ** (2 * k + 1) / (2 * k + 3) for k in range(8))
                        else:
                            even = (math.log(d1) - math.log(d0)) / (2 * z)
                            odd = (1 - even) / z
                        mean = ((n0 + n1) * even + (n1 - n0) * odd) / (d0 + d1)
                    if not -1e-10 <= mean <= 1 + 1e-10:
                        raise ArithmeticError("invalid integrated observation probability")
                    known.append(detection * min(1.0, max(0.0, mean)))
                missing = 1 - math.fsum(known)
                if missing < -1e-10:
                    raise ArithmeticError("record probabilities exceed unit mass")
                for row, probability in enumerate([*known, max(0.0, missing)]):
                    matrix[row][index] += segment_weight * probability
    for row, probability in enumerate(baseline):
        for index in range(count):
            matrix[row][index] += max(0.0, 1 - coverage) * probability
    return {"window": tuple(window), "records": tuple(canonical), "periodic": periodic,
            "bins": bins, "matrix": matrix, "baseline": baseline,
            "anchor_coverage": coverage, "unreachable": unreachable}


def private_outcome_distribution(kernels, references, epoch, retained_keys):
    """One mixture over physical records, frozen before any outcome is observed.

    Kernels share a window and detector. Reference weights are issue-time z,
    never result-dependent credit or posterior mixture responsibilities. Empty
    trace heads use the stated uniform-bin assay prior, leaving live action
    support unknown. Lost reference weight stays in the common baseline.
    """
    if not kernels:
        raise ValueError("an opportunity/detector kernel is required, even for empty inventory")
    first = next(iter(kernels.values()))
    baseline = first["baseline"]
    for kernel in kernels.values():
        if kernel["window"] != first["window"] or kernel["records"] != first["records"]:
            raise ValueError("all references must predict the same physical observation records")
    weights = [r["weight"] for r in references]
    keys = [r["key"] for r in references]
    if (len(references) > 16 or len(set(keys)) != len(keys)
            or not all(math.isfinite(w) and 0 <= w <= 1 for w in weights)
            or math.fsum(weights) > 1 + 1e-12):
        raise ValueError("coalesced issue-time inventory of at most 16 normalized references required")
    probabilities = [0.0] * len(baseline)
    used_weight, effective_baseline, supported = 0.0, 0.0, 0.0
    for reference in references:
        key, weight = reference["key"], reference["weight"]
        if key[0] != epoch or key not in retained_keys or weight == 0:
            continue
        kernel = kernels[key]
        if kernel["periodic"] != (key[2] == "periodic"):
            raise ValueError("kernel and coordinate family must agree")
        count = len(kernel["matrix"][0])
        q = reference["probabilities"]
        if q is None:
            q = [1 / count] * count
        if (len(q) != count or not all(math.isfinite(p) and p >= 0 for p in q)
                or not math.isclose(math.fsum(q), 1, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("normalized prediction with the complete bin layout required")
        used_weight += weight
        supported += weight * kernel["anchor_coverage"]
        effective_baseline += weight * (1 - kernel["anchor_coverage"]
                                       + math.fsum(p * u for p, u in zip(q, kernel["unreachable"])))
        for row, masses in enumerate(kernel["matrix"]):
            probabilities[row] += weight * math.fsum(p * k for p, k in zip(q, masses))
    residual = max(0.0, 1 - used_weight)
    for row, probability in enumerate(baseline):
        probabilities[row] += residual * probability
    if not math.isclose(math.fsum(probabilities), 1, rel_tol=0, abs_tol=1e-9):
        raise ArithmeticError("physical-outcome distribution failed normalization")
    return {"probabilities": probabilities, "issue_reference_support": supported,
            "effective_baseline_weight": residual + effective_baseline}
