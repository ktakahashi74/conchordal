#!/usr/bin/env python3
"""Independent I11-2 arrival CDF and decision-report checks."""

import argparse
import hashlib
import json
import math
import statistics
import struct
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

from locate_i11_first_divergence import CANDIDATE_INPUTS, DECISION_INPUTS, decisions
from verify_i11_stage1 import hellinger, overlap

POINTS = 201
GRID_SECONDS = 0.02
ABS_INTEGRAL_TOL = 1e-11
MAX_DEPTH = 24
BASE_ABS_TOL = 1e-7
BASE_REL_TOL = 1e-5


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def same_f32(left, right):
    return struct.pack("<f", left) == struct.pack("<f", right)


def same_f64(left, right):
    return struct.pack("<d", left) == struct.pack("<d", right)


def same_value(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and isinstance(right, float):
        return same_f64(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_value(left[k], right[k]) for k in left)
    return left == right


def softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def features(elapsed, case):
    config = case["config"]
    context = case["context"]
    intervals = case["intervals"]
    values = [None] * 8
    for i in range(3):
        if intervals[0] is not None and intervals[i + 1] is not None:
            values[i] = math.log(intervals[i + 1] / intervals[0])
    peak = context["peak"]
    if peak is not None:
        phase = math.tau * ((elapsed / peak["period_seconds"]) % 1.0)
        values[3:7] = [math.log2(peak["period_seconds"]), peak["support"],
                       math.cos(phase), math.sin(phase)]
        if context["word"] is not None:
            values[7] = float(context["word"])
    out = [0.0] * 18
    out[0] = 1.0
    out[1] = math.log1p(elapsed)
    for i, value in enumerate(values):
        index = 2 + i if i < 3 else 5 + i
        missing = 5 + i if i < 3 else 10 + i
        out[index] = 0.0 if value is None else (value - config["means"][i]) / max(
            config["deviations"][i], 1e-6)
        out[missing] = float(value is None)
    return out


def hazard(elapsed, case):
    z = math.fsum(a * b for a, b in zip(case["config"]["coefficients"],
                                       features(elapsed, case)))
    return softplus(z)


def simpson(function, left, right, midpoint_value=None):
    middle = (left + right) / 2.0
    fm = function(middle) if midpoint_value is None else midpoint_value
    return (right - left) * (function(left) + 4.0 * fm + function(right)) / 6.0


def adaptive_simpson(function, left, right, tolerance, depth=MAX_DEPTH):
    whole = simpson(function, left, right)

    def refine(a, b, coarse, limit, remaining):
        mid = (a + b) / 2.0
        fine_left = simpson(function, a, mid)
        fine_right = simpson(function, mid, b)
        fine = fine_left + fine_right
        delta = fine - coarse
        if abs(delta) <= 15.0 * limit:
            return fine + delta / 15.0
        if remaining == 0:
            raise ValueError("independent Simpson reference did not converge")
        return (refine(a, mid, fine_left, limit / 2.0, remaining - 1)
                + refine(mid, b, fine_right, limit / 2.0, remaining - 1))

    return refine(left, right, whole, tolerance, depth)


def reference_cdf(case, last_point=POINTS - 1):
    config = case["config"]
    elapsed = case["elapsed_seconds"]
    if elapsed[0] != elapsed[1] or case["reset_unknown"]:
        return None, None
    if config["model"] == "periodic":
        if case["context"]["peak"] is None:
            return None, None
        period = case["context"]["peak"]["period_seconds"]
        wait = period - (elapsed[0] % period)
        return [float(k / 50.0 >= wait) for k in range(last_point + 1)], None
    if config["model"] != "hazard":
        raise ValueError("unknown arrival model")
    peak = case["context"]["peak"]
    phase_dependent = config["coefficients"][10] != 0 or config["coefficients"][11] != 0
    max_width = min(GRID_SECONDS, peak["period_seconds"] / 8.0) if peak and phase_dependent else GRID_SECONDS
    cumulative = [0.0]
    cdf = [0.0]
    total = 0.0
    for k in range(1, last_point + 1):
        left = elapsed[0] + (k - 1) / 50.0
        right = elapsed[0] + k / 50.0
        segments = math.ceil((right - left) / max_width)
        integral = 0.0
        for j in range(segments):
            a = left + (right - left) * j / segments
            b = left + (right - left) * (j + 1) / segments
            segment_tol = ABS_INTEGRAL_TOL * (b - a) / config["horizon_sec"]
            integral += adaptive_simpson(lambda t: hazard(t, case), a, b, segment_tol)
        total += integral
        cumulative.append(total)
        cdf.append(-math.expm1(-total))
    return cdf, cumulative


def verify_cdf_fixture(data):
    if data.get("schema") != "i11-2-cdf-fixture-v2":
        raise ValueError("unexpected CDF fixture schema")
    checks = []
    for case in data["cases"]:
        name = case["name"]
        config = case["config"]
        if len(config["coefficients"]) != 18 or len(config["means"]) != 8 or len(config["deviations"]) != 8:
            raise ValueError(f"{name}: invalid feature dimensions")
        if config["horizon_sec"] != 4.0:
            checks.append({"case": name, "check": "unsupported horizon",
                           "passed": case["cdf"] is None and case["unknown"] == "unsupported_horizon"
                           and case["evaluations"] == 0})
            continue
        if case["horizon_end"] - case["issued_at"] != 4 * case["sample_rate"]:
            checks.append({"case": name, "check": "invalid clock",
                           "passed": case["cdf"] is None and case["unknown"] == "invalid_clock"
                           and case["evaluations"] == 0})
            continue
        expected, integrated = reference_cdf(
            case, case.get("failed_interval", POINTS - 1)
            if case["unknown"] == "integral_failed" else POINTS - 1)
        legacy = case["legacy_full_interval_integral"]
        if config["model"] == "hazard":
            if legacy is None:
                checks.append({"case": name, "check": "legacy unavailable with unknown CDF",
                               "passed": expected is None})
            elif len(legacy) != POINTS or not same_f64(legacy[0], 0.0):
                checks.append({"case": name, "check": "legacy full-interval shape", "passed": False})
            else:
                for point, got in enumerate(legacy):
                    if got is None:
                        continue
                    reference = (integrated[point] if integrated is not None and point < len(integrated)
                                 else reference_cdf(case, point)[1][point])
                    checks.append({"case": name, "check": "legacy full-interval integral",
                                   "point": point, "passed": math.isfinite(got)
                                   and abs(got - reference) <= 1e-8 + 1e-6 * abs(reference)})
        elif legacy is not None:
            checks.append({"case": name, "check": "Periodic has no hazard integral", "passed": False})
        actual = case["cdf"]
        if expected is None:
            reason = ("reset_unknown" if case["reset_unknown"] else
                      "uncertain_elapsed" if case["elapsed_seconds"][0] != case["elapsed_seconds"][1] else
                      "nonperiodic")
            passed = actual is None and case["unknown"] == reason and case["evaluations"] == 0
            checks.append({"case": name, "check": "unknown", "passed": passed})
            continue
        if case["unknown"] == "integral_failed":
            failed = case.get("failed_interval")
            evaluations = case["interval_evaluations"]
            checks.append({"case": name, "check": "bounded integration failure",
                           "passed": actual is None and isinstance(failed, int)
                           and 1 <= failed < POINTS and evaluations is not None
                           and len(evaluations) == failed and evaluations[-1] == 62
                           and sum(evaluations) == case["evaluations"]})
            continue
        if actual is None or len(actual) != POINTS or case["unknown"] is not None:
            checks.append({"case": name, "check": "present and complete", "passed": False})
            continue
        checks.append({"case": name, "check": "CDF domain and monotonicity",
                       "passed": same_f64(actual[0], 0.0)
                       and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in actual)
                       and all(a <= b for a, b in zip(actual, actual[1:]))})
        for k, (got, want) in enumerate(zip(actual, expected)):
            bound = (1e-8 + 1e-6 * abs(integrated[k])) if integrated is not None else 0.0
            checks.append({"case": name, "check": "cdf", "point": k,
                           "passed": math.isfinite(got) and abs(got - want) <= bound})
        forecast_probability = case.get("forecast_probability")
        if forecast_probability is not None:
            checks.append({"case": name, "check": "whole-horizon probability",
                           "passed": len(forecast_probability) == 2
                           and all(math.isfinite(value) and abs(value - expected[-1])
                                   <= 1e-8 + 1e-6 * abs(integrated[-1] if integrated else 0.0)
                                   for value in forecast_probability)})
        if integrated is not None:
            actual_integrals = case["cumulative_hazard"]
            if actual_integrals is None or len(actual_integrals) != POINTS:
                checks.append({"case": name, "check": "cumulative hazard present", "passed": False})
            else:
                for k, (got, want) in enumerate(zip(actual_integrals, integrated)):
                    checks.append({"case": name, "check": "cumulative hazard", "point": k,
                                   "passed": math.isfinite(got)
                                   and abs(got - want) <= 1e-8 + 1e-6 * abs(want)})
            evaluations = case["interval_evaluations"]
            checks.append({"case": name, "check": "evaluation cap",
                           "passed": evaluations is not None and len(evaluations) == POINTS - 1
                           and all(2 <= n <= 62 and n % 2 == 0 for n in evaluations)
                           and sum(evaluations) == case["evaluations"]})
        else:
            checks.append({"case": name, "check": "periodic zero evaluations",
                           "passed": case["evaluations"] == 0})
    return checks


def verify_registered_cdf_fixture(data, plan):
    if data.get("schema") != "i11-2-cdf-fixture-v2":
        raise ValueError("unexpected registered CDF fixture schema")
    cases = {case["name"]: case for case in data["cases"]}
    required = {"constant_hazard", "phase_hazard", "periodic", "nonperiodic_hazard", "integral_failed"}
    if len(cases) != len(data["cases"]) or not required <= cases.keys():
        raise ValueError("registered CDF fixture cases missing or duplicated")
    primary = plan["primary_hazard"]
    phase = cases["phase_hazard"]["config"]
    if phase["model"] != "hazard" or any(
            not same_f64(a, b) for key in ("coefficients", "means", "deviations")
            for a, b in zip(phase[key], primary[key])) or not same_f64(
                phase["horizon_sec"], primary["horizon_sec"]):
        raise ValueError("phase fixture differs from frozen primary Hazard")
    constant = cases["constant_hazard"]["config"]
    if (constant["model"] != "hazard" or not same_f64(softplus(constant["coefficients"][0]), 2.0)
            or any(value != 0.0 for value in constant["coefficients"][1:])):
        raise ValueError("constant Hazard fixture differs from analytic control")
    if cases["periodic"]["config"]["model"] != "periodic":
        raise ValueError("Periodic control model differs")
    if cases["nonperiodic_hazard"]["context"]["peak"] is not None:
        raise ValueError("nonperiodic control has a peak")
    if not cases["integral_failed"].get("forced_out_of_estimator_period_range"):
        raise ValueError("forced integration failure is not identified")


def interpolate(cdf, sample, issued_at, step):
    q = min(max((sample - float(issued_at)) / step, 0.0), POINTS - 1.0)
    index = math.floor(q)
    if index == POINTS - 1:
        return cdf[index]
    return cdf[index] + (cdf[index + 1] - cdf[index]) * (q - index)


def base_cost(decision, candidate, distance, overlap_sum):
    cost = candidate["displacement_sq"]
    if distance is not None:
        cost += decision["coupling"] * distance
    if overlap_sum is not None:
        norm = max(sum(decision["own_band_energy"]) * sum(decision["footprint_power"]), 1e-12)
        cost += (decision["coupling"] * 6.0 * decision["overlap_sensitivity"]
                 * overlap_sum / norm)
    return cost


def close_base(actual, expected):
    return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= (
        BASE_ABS_TOL + BASE_REL_TOL * max(abs(actual), abs(expected)))


def binding_status(input_data, binding, now):
    issue = input_data["publication"]["issue"]
    key = input_data["current_profile_key"]
    if key is None:
        return "no_profile_key", None
    if input_data["publication"]["profile_key"] != key:
        return "profile_key_mismatch", None
    owner = input_data["owner"]
    if (binding["source_id"] != owner[0] or binding["source_generation"] != owner[1]
            or binding["body_generation"] != owner[2]):
        return "wrong_owner", None
    if binding["bus"] != 0 or key["bus"] != 0 or issue["bus"] != 0:
        return "wrong_bus", None
    prototype = binding["prototype"]
    if (key["lookup_version"] != 3 or binding["model_version"] != key["body_model_version"]
            or prototype >= len(key["groups"])):
        return "incompatible_model", None
    if key["epoch"] != input_data["current_epoch"]:
        return "wrong_epoch", None
    if binding["end"] > binding["available"] or binding["available"] > now:
        return "future_body", None
    if now - binding["end"] >= 24000:
        return "stale_body", None
    if (key["assignment_end"] > key["issued_at"] or key["issued_at"] > now
            or key["issued_at"] > issue["issued_at"]):
        return "future_table", None
    if now - key["assignment_end"] >= 24000:
        return "stale_table", None
    if key["routed_prototypes"] & (1 << prototype) == 0:
        return "unrouted_prototype", None
    handle = key["groups"][prototype]
    if handle is None:
        return "no_group", None
    if handle["bus"] != 0 or handle["epoch"] != issue["epoch"]:
        return "wrong_group_epoch", None
    for slot, group in enumerate(issue["groups"]):
        if group is not None and group["group"] == handle:
            return "self", slot
    return "retired_group", None


def arrival_selection(decision):
    trace = decision["arrival"]
    input_data = trace["input"]
    if input_data is None:
        return {"input_status": "no_publication", "self_status": "no_binding",
                "binding_checks": [None] * 8, "groups": [None] * 7, "eligible_count": 0,
                "known_decision": False}, None, []
    issue = input_data["publication"]["issue"]
    now = decision["now"]
    if issue["bus"] != 0:
        input_status = "wrong_bus"
    elif issue["epoch"] != input_data["current_epoch"]:
        input_status = "wrong_epoch"
    elif (issue["version"] != 1 or issue["sample_rate"] == 0
          or issue["sample_rate"] != decision["sample_rate"]
          or issue["horizon_end"] != issue["issued_at"] + 4 * issue["sample_rate"]
          or input_data["received_at"] > now or issue["issued_at"] > input_data["received_at"]):
        input_status = "invalid_clock"
    elif now >= issue["horizon_end"]:
        input_status = "expired"
    else:
        input_status = "ready"
    self_status = "bindings_overflow" if input_data["bindings_overflow"] else "no_binding"
    checks = [None] * 8
    self_slots = set()
    if not input_data["bindings_overflow"]:
        for slot, binding in enumerate(input_data["bindings"]):
            if binding is None:
                continue
            status, group_slot = binding_status(input_data, binding, now)
            checks[slot] = {"binding": binding, "status": status, "group_slot": group_slot}
            if group_slot is not None:
                self_slots.add(group_slot)
                self_status = "matched"
            elif self_status == "no_binding":
                self_status = status
    max_upper = max((candidate["at"] + decision["width"] for candidate in decision["candidates"]
                     if candidate is not None), default=-math.inf)
    group_checks = [None] * 7
    used = []
    for slot, group in enumerate(issue["groups"]):
        if group is None:
            continue
        handle = group["group"]
        forecast = group["forecast"]
        if input_status != "ready":
            status = input_status
        elif handle["bus"] != issue["bus"] or handle["epoch"] != issue["epoch"]:
            status = "wrong_group_epoch"
        elif any(previous is not None and previous["group"] == handle for previous in issue["groups"][:slot]):
            status = "duplicate_group"
        elif slot in self_slots:
            status = "self"
        elif not group["periodic"]:
            status = "nonperiodic"
        elif group["unknown"] is not None:
            status = group["unknown"]
        elif group["cdf"] is None:
            status = "no_cdf"
        elif (forecast is None or forecast["group"] != handle
              or forecast["source_start"] > forecast["source_end"]
              or forecast["source_end"] > forecast["available"]
              or forecast["available"] > forecast["issued_at"]
              or forecast["issued_at"] != issue["issued_at"]
              or forecast["horizon_end"] != issue["horizon_end"]
              or forecast["model"] != issue["model"] or forecast["version"] != issue["version"]
              or forecast["reset_unknown"]
              or forecast["elapsed_seconds"][0] != forecast["elapsed_seconds"][1]):
            status = "invalid_source"
        elif max_upper > float(issue["horizon_end"]):
            status = "outside_horizon"
        else:
            status = "used"
            used.append(group)
        group_checks[slot] = {"slot": slot, "status": status}
    return {"input_status": input_status, "self_status": self_status,
            "binding_checks": checks, "groups": group_checks,
            "eligible_count": len(used), "known_decision": bool(used)}, issue, used


def verify_decision(decision):
    """Return named failures; all new arrival arithmetic is compared by exact bits."""
    failed = []
    if "arrival" not in decision or "arrival_weight" not in decision:
        return ["missing arrival trace or weight"]
    if (len(decision["candidates"]) != 23 or not any(c is not None for c in decision["candidates"])
            or len(decision["footprint_power"]) != 16 or len(decision["footprint_delay"]) != 16
            or len(decision["own_band_energy"]) != 3
            or (decision["memory"] is not None
                and (len(decision["memory"]) != 2 or any(len(bands) != 3 for bands in decision["memory"])))):
        return ["candidate structure"]
    expected_selection, issue, used = arrival_selection(decision)
    for key, expected in expected_selection.items():
        if decision["arrival"][key] != expected:
            failed.append(f"arrival.{key}")
    costs = []
    for index, candidate in enumerate(decision["candidates"]):
        offset = index - 2
        shift = (offset * decision["width"] / 2.0 if offset < 0
                 else offset * decision["period_frames"] / 20.0)
        at = (max(decision["due_frame"], decision["earliest"]) if offset == 0
              else decision["due_frame"] + shift)
        if (candidate is None) != (at < decision["earliest"]):
            failed.append(f"candidate {index} grid")
        if candidate is None:
            continue
        if (len(candidate["external_energy"]) != 16
                or any(energy is not None and len(energy) != 3 for energy in candidate["external_energy"])
                or (candidate["pred_external_band_energy"] is not None
                    and (len(candidate["pred_external_band_energy"]) != 2
                         or any(len(bands) != 3 for bands in candidate["pred_external_band_energy"])))):
            failed.append(f"candidate {index} energy structure")
            continue
        if candidate["offset"] != offset or not same_f64(candidate["at"], at):
            failed.append(f"candidate {index} time")
        displacement = (at - decision["due_frame"]) / decision["period_frames"]
        if not same_f32(candidate["displacement_sq"], displacement * displacement):
            failed.append(f"candidate {index} displacement")
        predicted = candidate["pred_external_band_energy"]
        applies_context = (decision["onset_allowed"] and decision["memory"] is not None
                           and predicted is not None)
        distance = hellinger(decision["memory"], predicted) if applies_context else None
        if ((distance is None) != (candidate["context_distance"] is None)
                or (distance is not None and not close_base(candidate["context_distance"], distance))):
            failed.append(f"candidate {index} context")
        energies = candidate["external_energy"]
        applies_overlap = (decision["onset_allowed"] and decision["overlap_sensitivity"] != 0.0
                           and decision["forecast_available_through_frame"] is not None)
        overlap_sum = overlap(decision["own_band_energy"], decision["footprint_power"], energies) if applies_overlap else None
        if ((overlap_sum is None) != (candidate["overlap"] is None)
                or (overlap_sum is not None and not close_base(candidate["overlap"], overlap_sum))):
            failed.append(f"candidate {index} overlap")
        available = decision["forecast_available_through_frame"]
        for delay, energy in zip(decision["footprint_delay"], energies):
            if energy is not None and (available is None or at + delay < available):
                failed.append(f"candidate {index} external deadline")
        expected_base = base_cost(decision, candidate, distance, overlap_sum)
        if not close_base(candidate["base_cost"], expected_base):
            failed.append(f"candidate {index} base")
        probability = None
        if used:
            step = float(issue["sample_rate"]) / 50.0
            lower = max(candidate["at"] - decision["width"], float(issue["issued_at"]))
            upper = candidate["at"] + decision["width"]
            total = 0.0
            for group in used:
                cdf = group["cdf"]
                if (cdf is None or len(cdf) != POINTS or not same_f64(cdf[0], 0.0)
                        or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in cdf)
                        or not all(a <= b for a, b in zip(cdf, cdf[1:]))):
                    failed.append("used group has invalid CDF")
                    continue
                total += (interpolate(cdf, upper, issue["issued_at"], step)
                          - interpolate(cdf, lower, issue["issued_at"], step))
            probability = total / len(used)
        if ((candidate.get("arrival_p") is None) != (probability is None)
                or (probability is not None and not same_f64(candidate["arrival_p"], probability))):
            failed.append(f"candidate {index} probability")
        term = None if probability is None else f32(
            float(f32(decision["coupling"])) * decision["arrival_weight"] * (1.0 - probability))
        if ((candidate.get("arrival_term") is None) != (term is None)
                or (term is not None and not same_f32(candidate["arrival_term"], term))):
            failed.append(f"candidate {index} term")
        total_cost = f32(f32(candidate["base_cost"]) + term) if term is not None else f32(candidate["base_cost"])
        if not same_f32(candidate["cost"], total_cost):
            failed.append(f"candidate {index} total")
        costs.append((candidate["offset"], candidate["at"], f32(candidate["cost"]), f32(candidate["base_cost"])))
    selected = min(costs, key=lambda row: row[2])
    if (decision["selected_offset"] != selected[0] or not same_f64(decision["selected_at"], selected[1])
            or not same_f32(decision["selected_cost"], selected[2])
            or not same_f32(decision["selected_base_cost"], selected[3])):
        failed.append("selection")
    reference = decision["candidates"][2]
    if reference is not None and not same_f32(decision["reference_cost"], reference["cost"]):
        failed.append("reference cost")
    skip = (decision["onset_allowed"] and decision["forecast_available_through_frame"] is not None
            and sum(decision["own_band_energy"]) > 1e-12 and decision["overlap_sensitivity"] > 0
            and selected[3] > f32(1.0 + decision["skipped_cycles"]))
    if decision["skipped"] != skip:
        failed.append("skip from chosen base")
    return failed


def verify_report(path):
    rows = decisions(path)
    checks = []
    received_versions = {}
    for index, row in enumerate(rows):
        failures = verify_decision(row)
        input_data = row.get("arrival", {}).get("input")
        if input_data is not None:
            issue = input_data["publication"]["issue"]
            version = (issue["bus"], issue["epoch"], issue["issued_at"])
            received = input_data["received_at"]
            if version in received_versions and received_versions[version] != received:
                failures.append("first receipt changed for publication version")
            received_versions.setdefault(version, received)
        if failures:
            checks.append({"index": index, "now": row["now"], "voice_id": row["voice_id"],
                           "failures": failures})
    return {"decisions": len(rows), "failed_decisions": len(checks), "examples": checks[:10]}


T2_DECISION_INPUTS = DECISION_INPUTS + ("footprint_source", "footprint_power")
T2_CANDIDATE_INPUTS = CANDIDATE_INPUTS + ("overlap",)


def locate_arrival_divergence(off_path, on_path):
    off_rows, on_rows = decisions(off_path), decisions(on_path)
    first_prior_difference = None
    for index, (off, on) in enumerate(zip(off_rows, on_rows)):
        if (off["now"], off["voice_id"]) != (on["now"], on["voice_id"]):
            return {"status": "schedule_differs", "index": index}
        off_candidates, on_candidates = off["candidates"], on["candidates"]
        if (len(off_candidates) != 23 or len(on_candidates) != 23
                or any((a is None) != (b is None) for a, b in zip(off_candidates, on_candidates))
                or any(c is not None and c["offset"] != slot - 2
                       for candidates in (off_candidates, on_candidates)
                       for slot, c in enumerate(candidates))):
            return {"status": "candidate_structure_differs", "index": index}
        differing = [key for key in T2_DECISION_INPUTS if not same_value(off[key], on[key])]
        differing.extend(f"candidates.{key}" for key in T2_CANDIDATE_INPUTS
                         if any(a is not None and not same_value(a[key], b[key])
                                for a, b in zip(off_candidates, on_candidates)))
        if any(a is not None and not same_f32(a["cost"], b["base_cost"])
               for a, b in zip(off_candidates, on_candidates)):
            differing.append("candidates.off_cost_vs_on_base_cost")
        if differing and first_prior_difference is None:
            first_prior_difference = {"index": index, "inputs": differing}
        outputs = [key for key in ("selected_offset", "skipped") if off[key] != on[key]]
        if outputs:
            return {"status": "diverged", "index": index, "outputs": outputs,
                    "nonarrival_differences": differing,
                    "first_prior_input_difference": first_prior_difference,
                    "on_arrival_failures": verify_decision(on),
                    "arrival_used": on["arrival"]["eligible_count"] > 0}
    if len(off_rows) != len(on_rows):
        return {"status": "schedule_differs", "index": min(len(off_rows), len(on_rows)),
                "off_decisions": len(off_rows), "on_decisions": len(on_rows)}
    return {"status": "no_divergence", "compared": len(off_rows),
            "first_prior_input_difference": first_prior_difference}


PERFORMANCE_FIELDS = tuple(f"{part}_{stat}" for part in
                           ("population_us", "synthesis_us", "elapsed_us")
                           for stat in ("median", "p99", "max")) + ("over_budget_hops",)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifact_names(row):
    label = f"{row['sequence']:03d}-{row['case']}-{row['config']}-{row['mode']}-{row['rep']}"
    if "arm" in row:
        label += f"-{row['arm']}"
    if "pass_index" in row:
        label += f"-pass{row['pass_index']}"
    names = {"log": f"{label}.log"}
    if row["mode"] == "render":
        names.update(wav=f"{label}.wav", report=f"{label}.jsonl")
    else:
        names["profile"] = f"{label}-profile.json"
        if row["mode"] == "report":
            names["report"] = f"{label}.jsonl"
    return names


def run_profile(path):
    profile = json.loads(path.read_text())
    hops = profile["hops"]
    if (not hops or profile.get("truncated") is not False or profile.get("dropped_hops") != 0
            or profile["summary"].get("hop_count") != len(hops)):
        raise ValueError(f"incomplete hop profile: {path}")
    result = {}
    for part in ("population_us", "synthesis_us", "elapsed_us"):
        values = [h[part] for h in hops]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"invalid {part} timing: {path}")
        values.sort()
        result.update({f"{part}_median": values[int(0.5 * (len(values) - 1))],
                       f"{part}_p99": values[int(0.99 * (len(values) - 1))],
                       f"{part}_max": values[-1]})
    budget_count = profile["summary"]["over_budget_hops"]
    budget = profile["hop_budget_us"]
    if (type(budget_count) is not int or not 0 <= budget_count <= len(hops)
            or isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or budget <= 0
            or budget_count != sum(hop["elapsed_us"] > budget for hop in hops)):
        raise ValueError(f"invalid over-budget count: {path}")
    result["over_budget_hops"] = budget_count
    known = (profile.get("background") or {}).get("arrival_known_decisions")
    if known is not None and (type(known) is not int or known < 0):
        raise ValueError(f"invalid arrival consumption count: {path}")
    result["arrival_known_decisions"] = known
    return result


def reduced_profile(repetitions):
    if len(repetitions) != 3:
        raise ValueError("performance cell lacks three repetitions")
    return {key: (sum(row[key] for row in repetitions) if key == "over_budget_hops"
                  else statistics.median(row[key] for row in repetitions))
            for key in PERFORMANCE_FIELDS}


def verify_performance(root, plan):
    root = Path(root).resolve()
    ledger = json.loads((root / "runs.json").read_text())
    provenance_path = root / "acquisition-provenance.json"
    if (ledger.get("schema") != "conchordal/i11-stage2-runs/1"
            or ledger.get("provenance_sha256") != sha256_file(provenance_path)):
        raise ValueError("acquisition ledger provenance differs")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("schema") != "conchordal/i11-stage2-acquisition-provenance/1":
        raise ValueError("acquisition provenance schema differs")
    repo = Path(__file__).resolve().parents[1]
    frozen = repo / "docs/roadmap/temporal-dcc/i11-2-inputs"
    if plan != json.loads((frozen / "plan.json").read_text()):
        raise ValueError("verification plan differs from frozen registration")
    if (provenance["registration_freeze_sha256"] != sha256_file(frozen / "freeze.json")
            or provenance["plan_sha256"] != sha256_file(frozen / "plan.json")
            or provenance["candidate_manifest_sha256"] != sha256_file(provenance["candidate_manifest"])):
        raise ValueError("acquisition source seal differs")
    candidate_manifest = json.loads(Path(provenance["candidate_manifest"]).read_text())
    if candidate_manifest.get("schema") != "conchordal/i11-stage2-candidate-provenance/1":
        raise ValueError("candidate provenance schema differs")
    for version in ("original_none", "pre_stage2_off"):
        baseline = plan["baseline"][version]
        if sha256_file(repo / baseline["manifest_path"]) != baseline["manifest_sha256"]:
            raise ValueError(f"baseline manifest differs: {version}")
    for version, executables in provenance["binaries"].items():
        for executable, entry in executables.items():
            path = Path(entry["path"])
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"acquisition binary differs: {version}/{executable}")
            if version != "stage2_candidate":
                kind = "render" if executable == "conchordal-render" else "instrument"
                if entry["sha256"] != plan["baseline"][version][f"{kind}_binary_sha256"]:
                    raise ValueError(f"baseline binary registration differs: {version}/{executable}")
            elif entry["sha256"] != candidate_manifest["binaries"][executable]["sha256"]:
                raise ValueError(f"candidate binary provenance differs: {executable}")
    if tomllib is None:
        raise ValueError("performance verification requires Python 3.11 or newer (tomllib)")
    for config, entry in provenance["configs"].items():
        for path_key, sha_key in (("path", "sha256"), ("original", "original_sha256")):
            if sha256_file(entry[path_key]) != entry[sha_key]:
                raise ValueError(f"acquisition config differs: {config}/{path_key}")
        registered = f"inputs/config-{config}.toml"
        if (entry["original"] != str(frozen / registered)
                or entry["original_sha256"] != plan["input_sha256"][registered]):
            raise ValueError(f"original config registration differs: {config}")
        original = tomllib.loads(Path(entry["original"]).read_text())
        derived = tomllib.loads(Path(entry["path"]).read_text())
        original["temporal_action_profiles"]["file"] = provenance["profile"]["path"]
        if original != derived:
            raise ValueError(f"derived config differs beyond profile path: {config}")
    if sha256_file(provenance["profile"]["path"]) != provenance["profile"]["sha256"]:
        raise ValueError("acquisition action profile differs")
    if provenance["profile"]["sha256"] != plan["source_profile_sha256"]:
        raise ValueError("action profile registration differs")
    rows = ledger["runs"]
    if len(rows) != 852 or len(plan["run_order"]) != 852:
        raise ValueError("incomplete registered 852-run ledger")
    for registered, run in zip(plan["run_order"], rows):
        if run["registration"] != registered or run["exit"] != 0:
            raise ValueError(f"run metadata differs: {registered['sequence']}")
        expected_names = expected_artifact_names(registered)
        if set(run["artifacts"]) != set(expected_names):
            raise ValueError(f"run artifact set differs: {registered['sequence']}")
        case = registered["case"]
        if case in plan["existing_stage1_inputs"]:
            script = repo / plan["existing_stage1_inputs"][case]["registered_path"]
            script_sha = plan["existing_stage1_inputs"][case]["sha256"]
        else:
            script = frozen / "inputs" / f"{case}.rhai"
            script_sha = plan["input_sha256"][f"inputs/{case}.rhai"]
        if sha256_file(script) != script_sha:
            raise ValueError(f"run scenario differs: {case}")
        executable = "conchordal-render" if registered["mode"] == "render" else "conchordal"
        binary = provenance["binaries"][registered["binary"]][executable]["path"]
        config = provenance["configs"][registered["config"]]["path"]
        expected_command = [binary, str(script), "--config", config]
        if registered["mode"] == "render":
            expected_command.extend(("-o", str(root / expected_names["wav"]),
                                     "--report", str(root / expected_names["report"])))
        else:
            expected_command.extend(("--nogui", "--play=false", "--wait-user-start=false",
                                     "--wait-user-exit=false", "--profile",
                                     str(root / expected_names["profile"])))
            if registered["mode"] == "report":
                expected_command.extend(("--report", str(root / expected_names["report"])))
        if run.get("command") != expected_command:
            raise ValueError(f"run command differs: {registered['sequence']}")
        for kind, artifact in run["artifacts"].items():
            if artifact["path"] != expected_names[kind]:
                raise ValueError(f"run artifact path differs: {registered['sequence']}/{kind}")
            path = root / artifact["path"]
            if (not path.is_file() or not path.stat().st_size
                    or path.stat().st_size != artifact["bytes"]
                    or sha256_file(path) != artifact["sha256"]):
                raise ValueError(f"run artifact differs: {path}")
    floor_path = root / "aa-floor.json"
    if sha256_file(floor_path) != ledger.get("aa_floor_sha256"):
        raise ValueError("A/A floor seal differs")
    floor = json.loads(floor_path.read_text())
    if floor["schema"] != "conchordal/i11-stage2-aa-floor/1" or set(floor["floor"]) != set(PERFORMANCE_FIELDS):
        raise ValueError("A/A floor fields differ")
    prefix_sha = hashlib.sha256(json.dumps(rows[:600], sort_keys=True).encode()).hexdigest()
    if floor["source_ledger_prefix_sha256"] != prefix_sha:
        raise ValueError("A/A floor source ledger differs")
    aa = {}
    for run in rows[96:600]:
        registered = run["registration"]
        if registered["phase"] != "performance_aa":
            raise ValueError("A/A phase differs")
        key = (registered["pass_index"], registered["case"], registered["mode"], registered["arm"])
        aa.setdefault(key, {})[registered["rep"]] = run_profile(
            root / run["artifacts"]["profile"]["path"])
    if len(aa) != 168:
        raise ValueError("A/A arm count differs")
    calculated_floor = {field: 0 for field in PERFORMANCE_FIELDS}
    first_two_floor = {field: 0 for field in PERFORMANCE_FIELDS}
    aa_differences = []
    for pass_index in (1, 2, 3):
        for case in plan["acquisition"]["performance_cases"]:
            for mode in plan["acquisition"]["performance_report_modes"]:
                arms = []
                for arm in ("none-A", "none-B"):
                    reps = aa[(pass_index, case, mode, arm)]
                    if set(reps) != {0, 1, 2}:
                        raise ValueError(f"A/A repetitions differ: {case}/{mode}/pass{pass_index}")
                    arms.append(reduced_profile([reps[i] for i in range(3)]))
                difference = {}
                for field in PERFORMANCE_FIELDS:
                    difference[field] = abs(arms[0][field] - arms[1][field])
                    calculated_floor[field] = max(calculated_floor[field], difference[field])
                    if pass_index < 3:
                        first_two_floor[field] = max(first_two_floor[field], difference[field])
                aa_differences.append({"pass_index": pass_index, "case": case, "mode": mode,
                                       "absolute_difference": difference})
    aa_differences.sort(key=lambda item: (item["pass_index"], item["case"], item["mode"]))
    pass3_exceedances = {field: sum(item["pass_index"] == 3
                                    and item["absolute_difference"][field] > first_two_floor[field]
                                    for item in aa_differences) for field in PERFORMANCE_FIELDS}
    if (calculated_floor != floor["floor"] or aa_differences != floor["differences"]
            or pass3_exceedances != floor["pass3_exceedances_against_pass1_2"]):
        raise ValueError("saved A/A floor differs from measured profiles")
    profiles = {}
    for run in rows[600:]:
        registered = run["registration"]
        if registered["phase"] != "performance_comparison":
            raise ValueError("comparison phase differs")
        key = (registered["case"], registered["mode"], registered["config"])
        profiles.setdefault(key, {})[registered["rep"]] = run_profile(
            root / run["artifacts"]["profile"]["path"])
    checks = []
    for case in plan["acquisition"]["performance_cases"]:
        for mode in plan["acquisition"]["performance_report_modes"]:
            variants = {}
            for config in plan["acquisition"]["performance_variants"]:
                repetitions = profiles[(case, mode, config)]
                if set(repetitions) != {0, 1, 2}:
                    raise ValueError(f"repetitions differ: {case}/{mode}/{config}")
                variants[config] = [repetitions[i] for i in range(3)]
            if case in plan["new_pulse_cases"]:
                for rep, row in enumerate(variants["body-on"]):
                    checks.append({"case": case, "mode": mode, "rep": rep,
                                   "check": "arrival consumption", "passed":
                                   isinstance(row["arrival_known_decisions"], int)
                                   and row["arrival_known_decisions"] >= 1})
            on = reduced_profile(variants["body-on"])
            for baseline in ("none", "body-off"):
                other = reduced_profile(variants[baseline])
                for field in PERFORMANCE_FIELDS:
                    difference = abs(on[field] - other[field])
                    if field == "over_budget_hops":
                        tolerance = floor["floor"][field]
                        original = sorted(row[field] for row in variants["body-on"]) == sorted(
                            row[field] for row in variants[baseline])
                    elif field.endswith("_max"):
                        tolerance = max(1000.0, floor["floor"][field])
                        original = difference <= 1000.0
                    else:
                        tolerance = max(0.05 * other[field], floor["floor"][field])
                        original = difference <= 0.05 * other[field]
                    checks.append({"case": case, "mode": mode, "baseline": baseline,
                                   "field": field, "on": on[field], "baseline_value": other[field],
                                   "floor": floor["floor"][field], "difference": difference,
                                   "tolerance": tolerance,
                                   "passed": difference <= tolerance, "original_rule_passed": original})
    return {"checked": len(checks), "failed": sum(not c["passed"] for c in checks),
            "examples": [c for c in checks if not c["passed"]][:20],
            "checks": checks,
            "original_rule_failed": sum(not c.get("original_rule_passed", True) for c in checks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdf-fixture", type=Path)
    parser.add_argument("--report", type=Path, action="append")
    parser.add_argument("--pair", type=Path, nargs=2, action="append", metavar=("OFF", "ON"))
    parser.add_argument("--performance-root", type=Path)
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parents[1]
                        / "docs/roadmap/temporal-dcc/i11-2-inputs/plan.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not (args.cdf_fixture or args.report or args.pair or args.performance_root):
        parser.error("provide --cdf-fixture, --report, --pair, or --performance-root")
    result = {"schema": "conchordal/i11-stage2-verification/1"}
    failed = False
    if args.cdf_fixture:
        fixture = json.loads(args.cdf_fixture.read_text())
        verify_registered_cdf_fixture(fixture, json.loads(args.plan.read_text()))
        checks = verify_cdf_fixture(fixture)
        result["cdf"] = {"checked": len(checks), "failed": sum(not c["passed"] for c in checks),
                         "examples": [c for c in checks if not c["passed"]][:10]}
        failed |= result["cdf"]["failed"] > 0
    if args.report:
        result["reports"] = [{"path": str(path), **verify_report(path)} for path in args.report]
        failed |= any(r["decisions"] == 0 or r["failed_decisions"] for r in result["reports"])
    if args.pair:
        result["pairs"] = [{"off": str(off), "on": str(on), **locate_arrival_divergence(off, on)}
                           for off, on in args.pair]
        failed |= any(p["status"] not in ("diverged", "no_divergence")
                      or p["first_prior_input_difference"] is not None
                      or (p["status"] == "no_divergence" and p["compared"] == 0)
                      or (p["status"] == "diverged"
                          and (p["nonarrival_differences"] or p["on_arrival_failures"]
                               or not p["arrival_used"])) for p in result["pairs"])
    if args.performance_root:
        result["performance"] = verify_performance(
            args.performance_root, json.loads(args.plan.read_text()))
        failed |= result["performance"]["failed"] > 0
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n")
    print(serialized)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
