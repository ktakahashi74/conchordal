#!/usr/bin/env python3
"""Independently evaluate the frozen I12b two-edge assay."""

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
import sys
import tomllib
import wave
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTERED = REPO / "docs/roadmap/temporal-dcc/i12b-inputs"
I11_ROOT = REPO / "target/i11-stage2-20260924/acquisition"
sys.path.insert(0, str(REPO / "docs/roadmap/temporal-dcc/i11-2-inputs"))
sys.path.insert(0, str(REPO / "docs/roadmap/temporal-dcc/i11-inputs"))
from evaluate_i11_stage2_offline import (  # noqa: E402
    LEARNING_TYPES, REALTIME_EXACT, compare_candidates_bits, exact_report_pair,
    verify_stage1_report,
)
from verify_i11_stage2 import (  # noqa: E402
    PERFORMANCE_FIELDS, arrival_selection, f32, interpolate, run_profile,
    same_f32, same_f64, same_value, verify_decision, verify_report,
)

FREEZE_SHA = "3c9d157ce7e9f44206b7253f0dde1000033c847468edad7227bfd070fd22762b"
BASELINE_LEDGER_SHA = "5819c1fdbba7a726ab8ab2dc5fec78332da5e0a97eb4b479b92023d366305a29"
NONARRIVAL_KEYS = (
    "sample_rate", "due_frame", "period_frames", "width", "earliest", "coupling",
    "overlap_sensitivity", "onset_allowed", "memory", "own_band_energy",
    "sound_hold_sec", "sound_adsr", "footprint_source", "footprint_identity",
    "current_identity", "footprint_requested_at", "footprint_received_at",
    "footprint_d_samples", "footprint_truncated", "footprint_delay", "footprint_power",
    "forecast_observed_frame", "forecast_available_through_frame", "skipped_cycles",
)
CANDIDATE_KEYS = (
    "offset", "at", "displacement_sq", "context_distance", "pred_external_band_energy",
    "external_energy", "overlap", "base_cost",
)
REALTIME_OBSERVATION_KEYS = {"delivery_delay_us"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"missing or SHA-256 differs: {path}")
    return path


def read_report(path, kinds):
    result = {kind: [] for kind in kinds}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get("type")
            if kind in result:
                result[kind].append(row)
    return result


def publication_key(publication):
    issue = publication["issue"]
    return (issue["bus"], issue["epoch"], issue["issued_at"],
            json.dumps(publication["profile_key"], sort_keys=True, separators=(",", ":")))


def publication_coverage(observations, decisions):
    observed = {}
    observed_identity = {}
    for record in observations:
        observation = record["observation"]
        publication = (observation.get("arrival") or {}).get("issue")
        if publication is None:
            continue
        value = observation["arrival"]
        key = publication_key(value)
        identity = key[:3]
        if (observation["bus"] != publication["bus"]
                or observation["source_epoch"] != publication["epoch"]
                or publication["issued_at"] > observation["available_sample"]):
            return {"passed": False, "reason": "observed packet cut differs", "key": key}
        if key in observed and not same_value(observed[key]["publication"], value):
            return {"passed": False, "reason": "conflicting observed publication", "key": key}
        if identity in observed_identity and not same_value(observed_identity[identity], value):
            return {"passed": False, "reason": "same observed issue changed packet or profile key",
                    "identity": identity}
        observed_identity[identity] = value
        observed[key] = {"publication": value, "cut": observation["support_end_sample"]}
    seen = {}
    seen_identity = {}
    matched = 0
    missing = 0
    for index, decision in enumerate(decisions):
        input_data = decision["arrival"]["input"]
        if input_data is None:
            continue
        publication = input_data["publication"]
        key = publication_key(publication)
        identity = key[:3]
        value = (publication, input_data["received_at"])
        if identity in seen_identity and (
                not same_value(seen_identity[identity][0], value[0]) or
                seen_identity[identity][1] != value[1]):
            return {"passed": False, "reason": "same issue changed packet, profile key or receipt",
                    "identity": identity, "decision_index": index}
        seen_identity.setdefault(identity, value)
        if key in seen and (not same_value(seen[key][0], value[0]) or seen[key][1] != value[1]):
            return {"passed": False, "reason": "same packet changed or first receipt changed",
                    "key": key, "decision_index": index}
        seen.setdefault(key, value)
        if identity in observed_identity and not same_value(
                publication, observed_identity[identity]):
            return {"passed": False, "reason": "decision issue changed observed packet or profile key",
                    "identity": identity, "decision_index": index}
        if key in observed:
            if not same_value(publication, observed[key]["publication"]):
                return {"passed": False, "reason": "decision packet differs from observed issue",
                        "key": key, "decision_index": index}
            matched += 1
        else:
            missing += 1
    return {"passed": True, "observed_issues": len(observed),
            "decision_packets": matched + missing, "matched_observation": matched,
            "unobserved_due_to_report_decimation": missing,
            "unobserved_keys": [key for key in seen if key not in observed]}


def candidate_structure(left, right):
    a, b = left["candidates"], right["candidates"]
    if len(a) != 23 or len(b) != 23:
        return False
    for slot, (first, second) in enumerate(zip(a, b)):
        if (first is None) != (second is None):
            return False
        if first is not None and (first["offset"] != slot - 2 or second["offset"] != slot - 2):
            return False
    return True


def same_nonarrival_inputs(left, right):
    if (left["now"], left["voice_id"]) != (right["now"], right["voice_id"]):
        return False, "schedule"
    if not candidate_structure(left, right):
        return False, "candidate_structure"
    for key in NONARRIVAL_KEYS:
        if not same_value(left[key], right[key]):
            return False, key
    for index, (first, second) in enumerate(zip(left["candidates"], right["candidates"])):
        if first is None:
            continue
        for key in CANDIDATE_KEYS:
            if not same_value(first[key], second[key]):
                return False, f"candidates[{index}].{key}"
    return True, None


def nonarrival_differences(left, right):
    differences = [key for key in NONARRIVAL_KEYS if not same_value(left[key], right[key])]
    if not candidate_structure(left, right):
        differences.append("candidate_structure")
    for index, (first, second) in enumerate(zip(left["candidates"], right["candidates"])):
        if first is None or second is None:
            continue
        differences.extend(f"candidates[{index}].{key}" for key in CANDIDATE_KEYS
                           if not same_value(first[key], second[key]))
    return differences


def same_publication(left, right):
    first = left["arrival"]["input"]
    second = right["arrival"]["input"]
    if (first is None) != (second is None):
        return False
    if first is None:
        return True
    return same_value(first, second)


def recompute_selection(decision, packet=None):
    """Recompute P, f32 term/cost, and selection from recorded base inputs."""
    working = copy.deepcopy(decision)
    if packet is not None:
        if working["arrival"]["input"] is None:
            raise ValueError("full decision has no input to freeze")
        working["arrival"]["input"]["publication"] = copy.deepcopy(packet["publication"])
        working["arrival"]["input"]["received_at"] = packet["received_at"]
    status, issue, used = arrival_selection(working)
    if status["input_status"] != "ready":
        return {"valid": False, "reason": status["input_status"], "status": status}
    if not working["candidates"] or len(working["candidates"]) != 23:
        raise ValueError("candidate structure differs")
    costs = []
    probabilities = []
    terms = []
    for candidate in working["candidates"]:
        if candidate is None:
            costs.append(None)
            probabilities.append(None)
            terms.append(None)
            continue
        probability = None
        if used:
            step = issue["sample_rate"] / 50.0
            lower = max(candidate["at"] - working["width"], float(issue["issued_at"]))
            upper = candidate["at"] + working["width"]
            values = []
            for group in used:
                cdf = group["cdf"]
                if (cdf is None or len(cdf) != 201 or not same_f64(cdf[0], 0.0)
                        or any(not math.isfinite(v) or v < 0 or v > 1 for v in cdf)
                        or any(a > b for a, b in zip(cdf, cdf[1:]))):
                    raise ValueError("invalid used CDF")
                values.append(interpolate(cdf, upper, issue["issued_at"], step)
                              - interpolate(cdf, lower, issue["issued_at"], step))
            probability = sum(values) / len(values)
        term = (None if probability is None else
                f32(float(f32(working["coupling"])) * working["arrival_weight"] * (1.0 - probability)))
        cost = (f32(candidate["base_cost"]) if term is None else
                f32(f32(candidate["base_cost"]) + term))
        costs.append(cost)
        probabilities.append(probability)
        terms.append(term)
    present = [(index, cost) for index, cost in enumerate(costs) if cost is not None]
    if not present:
        raise ValueError("no candidate")
    selected_index = min(present, key=lambda row: row[1])[0]
    selected = working["candidates"][selected_index]
    base = f32(selected["base_cost"])
    skip = (working["onset_allowed"] and working["forecast_available_through_frame"] is not None
            and sum(working["own_band_energy"]) > 1e-12 and working["overlap_sensitivity"] > 0
            and base > f32(1.0 + working["skipped_cycles"]))
    return {"valid": True, "status": status, "eligible_count": len(used),
            "selected_offset": selected["offset"], "skipped": skip,
            "costs": costs, "probabilities": probabilities, "terms": terms}


def forward_pair(full_rows, cut_rows):
    if not full_rows or not cut_rows:
        return {"passed": False, "reason": "missing decisions"}
    first = None
    prior_input_difference = None
    for index, (full, cut) in enumerate(zip(full_rows, cut_rows)):
        aligned, field = same_nonarrival_inputs(full, cut)
        if not aligned and prior_input_difference is None:
            prior_input_difference = {"index": index, "field": field}
        if not same_publication(full, cut):
            return {"passed": False, "reason": "publication differs", "index": index}
        if not same_f64(full["arrival_weight"], 1.0) or not same_f64(cut["arrival_weight"], 0.0):
            return {"passed": False, "reason": "weight intervention differs", "index": index}
        if verify_decision(full) or verify_decision(cut):
            return {"passed": False, "reason": "invalid arrival arithmetic", "index": index}
        if (full["selected_offset"], full["skipped"]) != (cut["selected_offset"], cut["skipped"]):
            first = index
            break
    if first is None:
        return {"passed": False, "reason": "no decision divergence",
                "compared": min(len(full_rows), len(cut_rows)),
                "first_prior_input_difference": prior_input_difference}
    f, c = full_rows[first], cut_rows[first]
    used = f["arrival"]["eligible_count"] > 0 and c["arrival"]["eligible_count"] > 0
    return {"passed": used and prior_input_difference is None,
            "first_index": first, "now": f["now"], "voice_id": f["voice_id"],
            "outputs": [key for key in ("selected_offset", "skipped") if f[key] != c[key]],
            "known_on_both": used, "first_prior_input_difference": prior_input_difference}


def canonical_observation(value):
    if isinstance(value, dict):
        return {key: canonical_observation(member) for key, member in value.items()
                if key not in REALTIME_EXACT and not key.endswith(("_us", "_ns"))}
    if isinstance(value, list):
        return [canonical_observation(member) for member in value]
    return value


def forward_observation_prefix(full_report, cut_report, decision_now):
    left = read_report(full_report, ("temporal_observation",))["temporal_observation"]
    right = read_report(cut_report, ("temporal_observation",))["temporal_observation"]
    first = [item["observation"] for item in left
             if item["observation"]["support_end_sample"] <= decision_now]
    second = [item["observation"] for item in right
              if item["observation"]["support_end_sample"] <= decision_now]
    if not first or len(first) != len(second):
        return {"passed": False, "reason": "observation prefix missing or truncated",
                "counts": [len(first), len(second)]}
    for index, (a, b) in enumerate(zip(first, second)):
        if not same_value(canonical_observation(a), canonical_observation(b)):
            return {"passed": False, "reason": "period/grouping/forecast prefix differs",
                    "index": index, "bus": a.get("bus")}
    return {"passed": True, "compared": len(first)}


def packet_identity(input_data):
    if input_data is None:
        return None
    publication = input_data["publication"]
    issue = publication["issue"]
    lineage = [group["group"] if group is not None else None for group in issue["groups"]]
    return (issue["model"], issue["version"], issue["bus"], issue["epoch"],
            issue["sample_rate"], issue["issued_at"], issue["horizon_end"],
            lineage, publication["profile_key"])


def direct_packet_swap(full, backward):
    aligned, field = same_nonarrival_inputs(full, full)
    if not aligned:
        return {"passed": False, "reason": f"full malformed: {field}"}
    if (full["now"], full["voice_id"]) != (backward["now"], backward["voice_id"]):
        return {"passed": False, "reason": "decision time or voice differs"}
    first = full["arrival"]["input"]
    second = backward["arrival"]["input"]
    if first is None or second is None:
        return {"passed": False, "reason": "missing real publication"}
    if packet_identity(first) != packet_identity(second):
        return {"passed": False, "reason": "packet lineage or clock differs"}
    if not same_value(first["owner"], second["owner"]):
        return {"passed": False, "reason": "owner differs"}
    if first["current_epoch"] != second["current_epoch"]:
        return {"passed": False, "reason": "current epoch differs"}
    if not same_value(first["current_profile_key"], second["current_profile_key"]):
        return {"passed": False, "reason": "profile key differs"}
    if (not same_value(first["bindings"], second["bindings"])
            or first["bindings_overflow"] != second["bindings_overflow"]):
        return {"passed": False, "reason": "bindings differ"}
    if first["received_at"] > full["now"] or second["received_at"] > full["now"]:
        return {"passed": False, "reason": "packet not received at decision"}
    if verify_decision(full) or verify_decision(backward):
        return {"passed": False, "reason": "invalid source decision"}
    original = recompute_selection(full)
    swapped = recompute_selection(full, second)
    if not original["valid"] or not swapped["valid"]:
        return {"passed": False, "reason": "packet invalid at frozen decision",
                "full_status": original["status"]["input_status"],
                "backward_status": swapped["status"]["input_status"]}
    if original["eligible_count"] == 0:
        return {"passed": False, "reason": "full lacks known CDF"}
    if (original["selected_offset"], original["skipped"]) != (
            full["selected_offset"], full["skipped"]):
        return {"passed": False, "reason": "full replay differs"}
    terms_differ = any(a is not None and b is None or a is None and b is not None
                       or a is not None and b is not None and not same_f32(a, b)
                       for a, b in zip(original["terms"], swapped["terms"]))
    selected_differ = (original["selected_offset"], original["skipped"]) != (
        swapped["selected_offset"], swapped["skipped"])
    return {"passed": terms_differ and selected_differ,
            "reason": (None if terms_differ and selected_differ else
                       "no arrival term difference" if not terms_differ else
                       "no selection difference under packet intervention"),
            "terms_differ": terms_differ, "selected_differ": selected_differ,
            "full_selected": [original["selected_offset"], original["skipped"]],
            "swapped_selected": [swapped["selected_offset"], swapped["skipped"]],
            "full_eligible": original["eligible_count"],
            "swapped_eligible": swapped["eligible_count"],
            "full_packet_key": publication_key(first["publication"]),
            "swapped_packet_key": publication_key(second["publication"])}


def period_semantic(observation):
    period = observation.get("period")
    if period is None:
        return None
    return [{key: group.get(key) for key in ("peaks", "period_source")}
        if group is not None else None for group in period["groups"]]


def period_peaks(observation):
    period = observation.get("period")
    if period is None:
        return None
    return [group["peaks"] if group is not None else None for group in period["groups"]]


def raw_prefix(observation):
    period = observation.get("period")
    return {"bus": observation["bus"], "source_epoch": observation["source_epoch"],
            "sample_rate": observation["sample_rate"], "frame_id": observation["frame_id"],
            "support_start_sample": observation["support_start_sample"],
            "support_end_sample": observation["support_end_sample"],
            "available_sample": observation["available_sample"],
            "input_end_sample": observation["input_end_sample"],
            "acoustic": observation.get("acoustic"),
            "period_received_at": period["received_at"] if period else None,
            "period_ledgers": [g["ledger"] if g else None for g in period["groups"]]
            if period else None,
            "period_residual": period["residual"] if period else None}


def upstream_observation(observation):
    downstream = {"period_parameters", "period", "period_error", "arrival",
                  "action_profile_features", "context", "context_error",
                  "group_prototypes", "gesture", "gesture_error",
                  "period_accent_cut_deliveries"}
    return canonical_observation({key: value for key, value in observation.items()
                                  if key not in downstream})


def first_wav_difference(left_path, right_path, start_frame=0):
    with wave.open(str(left_path), "rb") as left, wave.open(str(right_path), "rb") as right:
        parameters = (left.getnchannels(), left.getsampwidth(), left.getframerate(),
                      left.getnframes(), left.getcomptype())
        if parameters != (right.getnchannels(), right.getsampwidth(), right.getframerate(),
                          right.getnframes(), right.getcomptype()):
            raise ValueError("WAV formats differ")
        frame_bytes = left.getnchannels() * left.getsampwidth()
        if start_frame < 0 or start_frame >= left.getnframes():
            return {"first_frame": None, "sample_rate": left.getframerate(),
                    "frames": left.getnframes()}
        left.setpos(start_frame)
        right.setpos(start_frame)
        consumed = start_frame
        while True:
            a, b = left.readframes(8192), right.readframes(8192)
            if a != b:
                offset = next(index for index, (x, y) in enumerate(zip(a, b)) if x != y)
                return {"first_frame": consumed + offset // frame_bytes,
                        "sample_rate": left.getframerate(), "frames": left.getnframes()}
            if not a:
                return {"first_frame": None, "sample_rate": left.getframerate(),
                        "frames": left.getnframes()}
            consumed += len(a) // frame_bytes


def arrival_real_clock(observation):
    period = observation.get("period")
    if period is None:
        return None
    clocks = []
    for group in period["groups"]:
        if group is None:
            clocks.append(None)
            continue
        forecast = group.get("forecast")
        clocks.append(None if forecast is None else {
            "group": forecast["group"], "last_accent": forecast["last_accent"],
            "elapsed_seconds": forecast["elapsed_seconds"],
            "reset_unknown": forecast["reset_unknown"]})
    return clocks


def observation_cuts(rows):
    cuts = {}
    for record in rows:
        observation = record["observation"]
        cut, bus = observation["support_end_sample"], observation["bus"]
        if bus in cuts.setdefault(cut, {}):
            raise ValueError(f"duplicate temporal observation at cut {cut}/bus{bus}")
        cuts[cut][bus] = observation
    return cuts


def affected_handles(left, right):
    a, b = left["period"], right["period"]
    handles = []
    if a is None or b is None:
        return handles
    if len(a["groups"]) != len(b["groups"]):
        raise ValueError("period group structure differs")
    for first, second in zip(a["groups"], b["groups"]):
        left_peaks = first["peaks"] if first is not None else None
        right_peaks = second["peaks"] if second is not None else None
        if not same_value(left_peaks, right_peaks):
            source = first or second
            if source is not None:
                handles.append(source["ledger"]["group"])
    return handles


def scan_period_observations(full_rows, cut_rows, packet_witnesses=()):
    full, cut = observation_cuts(full_rows), observation_cuts(cut_rows)
    first = None
    habitat = None
    last_joint_equal = None
    last_habitat_equal = None
    affected = {}
    habitat_affected = {}
    later_upstream = 0
    uncertain_cut_upstream = []
    witnesses = {}
    for witness in packet_witnesses:
        witnesses.setdefault(witness["issued_at"], []).append(witness)
    for sample in sorted(set(full) | set(cut) | set(witnesses)):
        for witness in witnesses.get(sample, []):
            bus = witness["bus"]
            if first is None:
                first = {"last_joint_equal_cut": last_joint_equal,
                         "first_observed_peak_cut": None,
                         "first_period_evidence_cut": sample, "buses": [bus]}
            if bus == 0 and habitat is None:
                habitat = {"last_equal_cut": last_habitat_equal,
                           "first_observed_peak_cut": None,
                           "first_period_evidence_cut": sample,
                           "clock_equal_at_observed_peak": None}
            for handle in witness["groups"]:
                affected[json.dumps(handle, sort_keys=True)] = handle
                if bus == 0:
                    habitat_affected[json.dumps(handle, sort_keys=True)] = handle
        if sample not in full and sample not in cut:
            continue
        left_bus, right_bus = full.get(sample), cut.get(sample)
        if (left_bus is None or right_bus is None or set(left_bus) != {0, 1}
                or set(right_bus) != {0, 1}):
            if first is None or habitat is None:
                return {"passed": False, "reason": "missing or unpaired observation cut",
                        "sample": sample}
            later_upstream += 1
            continue
        changes = [bus for bus in (0, 1) if not same_value(
            period_peaks(left_bus[bus]), period_peaks(right_bus[bus]))]
        prior_to_observed_period_change = first is None and not changes
        for bus in sorted(left_bus):
            left, right = left_bus[bus], right_bus[bus]
            upstream_equal = same_value(upstream_observation(left), upstream_observation(right))
            raw_equal = same_value(raw_prefix(left), raw_prefix(right))
            clock_equal = same_value(arrival_real_clock(left), arrival_real_clock(right))
            peak_equal = same_value(period_peaks(left), period_peaks(right))
            semantic_equal = same_value(period_semantic(left), period_semantic(right))
            if prior_to_observed_period_change and (
                    not upstream_equal or not raw_equal or not clock_equal):
                return {"passed": False, "reason": "upstream observation differs before first period peak",
                        "sample": sample, "bus": bus}
            if not peak_equal:
                if (type(right.get("period_accent_cut_deliveries")) is not int
                        or right["period_accent_cut_deliveries"] < 1):
                    return {"passed": False, "reason": "peak changed without real accent cut",
                            "sample": sample, "bus": bus}
                for handle in affected_handles(left, right):
                    affected[json.dumps(handle, sort_keys=True)] = handle
                    if bus == 0:
                        habitat_affected[json.dumps(handle, sort_keys=True)] = handle
                if bus == 0:
                    if habitat is None:
                        habitat = {"last_equal_cut": last_habitat_equal,
                                   "first_observed_peak_cut": sample,
                                   "first_period_evidence_cut": sample,
                                   "clock_equal_at_observed_peak": clock_equal}
                    elif habitat["first_observed_peak_cut"] is None:
                        habitat.update(first_observed_peak_cut=sample,
                                       clock_equal_at_observed_peak=clock_equal)
            elif prior_to_observed_period_change and not semantic_equal:
                return {"passed": False, "reason": "period source differs before peak",
                        "sample": sample, "bus": bus}
            if bus == 0 and habitat is None:
                last_habitat_equal = sample
            if first is not None and not upstream_equal:
                later_upstream += 1
            elif changes and first is None and not upstream_equal:
                uncertain_cut_upstream.append({"sample": sample, "bus": bus})
        if changes and first is None:
            first = {"last_joint_equal_cut": last_joint_equal,
                     "first_observed_peak_cut": sample,
                     "first_period_evidence_cut": sample, "buses": changes}
        elif changes and first["first_observed_peak_cut"] is None:
            first["first_observed_peak_cut"] = sample
        elif prior_to_observed_period_change:
            for bus in left_bus:
                a = canonical_observation({key: value for key, value in left_bus[bus].items()
                                           if key not in {"period_parameters", "period_accent_cut_deliveries"}})
                b = canonical_observation({key: value for key, value in right_bus[bus].items()
                                           if key not in {"period_parameters", "period_accent_cut_deliveries"}})
                if not same_value(a, b):
                    return {"passed": False, "reason": "downstream changed before period peak",
                            "sample": sample, "bus": bus}
            last_joint_equal = sample
    if (first is None or habitat is None or first["last_joint_equal_cut"] is None
            or habitat["last_equal_cut"] is None
            or first["first_observed_peak_cut"] is None
            or habitat["first_observed_peak_cut"] is None):
        return {"passed": False, "reason": "no witnessed all-bus and habitat period peak change",
                "first": first, "habitat": habitat}
    return {"passed": True, "first_all_bus": first, "first_habitat": habitat,
            "affected_handles": list(affected.values()),
            "habitat_affected_handles": list(habitat_affected.values()),
            "later_upstream_differences": later_upstream,
            "packet_period_witnesses": list(packet_witnesses),
            "upstream_differences_at_first_observed_peak": uncertain_cut_upstream,
            "observation_cut_counts": [len(full), len(cut)]}


def relevant_arrival_change(full, cut, affected, earliest_issue):
    left_input = full["arrival"]["input"]
    right_input = cut["arrival"]["input"]
    if left_input is None or right_input is None:
        return {"passed": False, "reason": "missing real publication"}
    first_issue = left_input["publication"]["issue"]
    second_issue = right_input["publication"]["issue"]
    if (first_issue["bus"] != 0 or first_issue["issued_at"] < earliest_issue
            or second_issue["issued_at"] < earliest_issue):
        return {"passed": False, "reason": "packet precedes habitat period window"}
    if (first_issue["bus"], first_issue["epoch"]) != (
            second_issue["bus"], second_issue["epoch"]):
        return {"passed": False, "reason": "publication bus or epoch differs"}
    affected_keys = {json.dumps(item, sort_keys=True) for item in affected}
    for slot, group in enumerate(first_issue["groups"]):
        if group is None or full["arrival"]["groups"][slot] is None:
            continue
        if full["arrival"]["groups"][slot]["status"] != "used":
            continue
        handle = group["group"]
        if json.dumps(handle, sort_keys=True) not in affected_keys:
            continue
        partner = next(((index, item) for index, item in enumerate(second_issue["groups"])
                        if item is not None and same_value(item["group"], handle)), None)
        if partner is None:
            return {"passed": False, "reason": "affected group lineage disappeared"}
        other_slot, other = partner
        other_check = cut["arrival"]["groups"][other_slot]
        status = other_check["status"] if other_check is not None else None
        if status in {"nonperiodic", "used"} and (
                group["periodic"] != other["periodic"]
                or not same_value(group["cdf"], other["cdf"])):
            return {"passed": True, "group": handle, "full_slot": slot,
                    "backward_slot": other_slot, "backward_status": status,
                    "periodic_changed": group["periodic"] != other["periodic"],
                    "cdf_changed": not same_value(group["cdf"], other["cdf"])}
    return {"passed": False, "reason": "no affected used group CDF/admission change"}


def backward_pair(full_report, backward_report, full_wav=None, backward_wav=None):
    full = read_report(full_report, ("temporal_observation", "participation_decision"))
    backward = read_report(backward_report, ("temporal_observation", "participation_decision"))
    if not full["temporal_observation"] or not backward["temporal_observation"]:
        return {"passed": False, "reason": "missing period observations"}
    publications = []
    for report in (full, backward):
        indexed_publications = {}
        for row in report["temporal_observation"]:
            publication = row["observation"].get("arrival")
            if publication is not None:
                indexed_publications[publication_key(publication)] = publication
        for row in report["participation_decision"]:
            received = row["arrival"]["input"]
            if received is not None:
                publication = received["publication"]
                indexed_publications[publication_key(publication)] = publication
        publications.append(indexed_publications)
    witnesses = []
    for key in publications[0].keys() & publications[1].keys():
        left, right = (index[key]["issue"] for index in publications)
        if any(left[field] != right[field] for field in (
                "model", "version", "bus", "epoch", "sample_rate", "issued_at", "horizon_end")):
            continue
        groups = [group["group"] for group in left["groups"] if group is not None
                  and any(other is not None and same_value(group["group"], other["group"])
                          and group["periodic"] != other["periodic"]
                          for other in right["groups"])]
        if groups:
            witnesses.append({"bus": left["bus"], "issued_at": left["issued_at"],
                              "publication_key": key, "groups": groups})
    witnesses.sort(key=lambda row: (row["issued_at"], row["bus"]))
    observations = scan_period_observations(full["temporal_observation"],
                                            backward["temporal_observation"], witnesses)
    if not observations["passed"]:
        return {"passed": False, "reason": "period observation chain invalid",
                "observations": observations}
    left_rows, right_rows = full["participation_decision"], backward["participation_decision"]
    if not left_rows or not right_rows:
        return {"passed": False, "reason": "missing decisions", "observations": observations}
    lower = observations["first_all_bus"]["last_joint_equal_cut"]
    habitat_lower = observations["first_habitat"]["last_equal_cut"]
    left_prefix = [row for row in left_rows if row["now"] < lower]
    right_prefix = [row for row in right_rows if row["now"] < lower]
    if not left_prefix or len(left_prefix) != len(right_prefix):
        return {"passed": False, "reason": "decision prefix missing or truncated",
                "counts": [len(left_prefix), len(right_prefix)]}
    for index, (left, right) in enumerate(zip(left_prefix, right_prefix)):
        if not same_value(left, right):
            return {"passed": False, "reason": "decision prefix changed before period window",
                    "index": index, "now": left["now"], "voice_id": left["voice_id"]}
    def indexed(rows):
        result = {}
        for row in rows:
            key = (row["now"], row["voice_id"])
            if key in result:
                raise ValueError(f"duplicate decision key: {key}")
            result[key] = row
        return result
    left_by_key, right_by_key = indexed(left_rows), indexed(right_rows)
    common = sorted(left_by_key.keys() & right_by_key.keys())
    unmatched = {"full_only": [key for key in left_by_key if key not in right_by_key],
                 "backward_only": [key for key in right_by_key if key not in left_by_key]}
    unmatched["full_only"].sort()
    unmatched["backward_only"].sort()
    first_schedule = (min(((key, arm) for arm, keys in unmatched.items() for key in keys),
                          default=None))
    first_selection = None
    total_candidates = []
    direct_candidates = []
    affected = observations["habitat_affected_handles"]
    for key in common:
        left, right = left_by_key[key], right_by_key[key]
        outputs = [field for field in ("selected_offset", "skipped")
                   if left[field] != right[field]]
        if outputs and first_selection is None:
            first_selection = {"now": key[0], "voice_id": key[1], "outputs": outputs,
                               "nonarrival_differences": nonarrival_differences(left, right)}
        if key[0] < habitat_lower:
            continue
        diagnostic = {"now": key[0], "voice_id": key[1], "outputs": outputs,
                      "nonarrival_differences": nonarrival_differences(left, right)}
        relevance = relevant_arrival_change(left, right, affected, habitat_lower)
        diagnostic["arrival_link"] = relevance
        if not relevance["passed"]:
            diagnostic["reason"] = relevance["reason"]
        elif not outputs:
            diagnostic["reason"] = "no selection difference"
        else:
            try:
                first, second = recompute_selection(left), recompute_selection(right)
                if not first["valid"] or not second["valid"]:
                    diagnostic["reason"] = "packet not ready at matched decision"
                elif first["eligible_count"] == 0:
                    diagnostic["reason"] = "full lacks known CDF"
                elif ((first["selected_offset"], first["skipped"]) !=
                      (left["selected_offset"], left["skipped"]) or
                      (second["selected_offset"], second["skipped"]) !=
                      (right["selected_offset"], right["skipped"])):
                    diagnostic["reason"] = "source selection replay differs"
                elif publication_key(left["arrival"]["input"]["publication"]) != (
                        publication_key(right["arrival"]["input"]["publication"])):
                    diagnostic["reason"] = "publication key or issue time differs"
                else:
                    terms_differ = any(
                        a is None and b is not None or a is not None and b is None or
                        a is not None and b is not None and not same_f32(a, b)
                        for a, b in zip(first["terms"], second["terms"]))
                    diagnostic.update({"terms_differ": terms_differ,
                                       "full_eligible": first["eligible_count"],
                                       "backward_eligible": second["eligible_count"]})
                    if not terms_differ:
                        diagnostic["reason"] = "no arrival term difference"
                    else:
                        diagnostic["passed"] = True
            except (KeyError, TypeError, ValueError, IndexError) as error:
                diagnostic["reason"] = f"invalid matched decision: {error}"
        total_candidates.append(diagnostic)
        if relevance["passed"]:
            try:
                direct = direct_packet_swap(left, right)
            except (KeyError, TypeError, ValueError, IndexError) as error:
                direct = {"passed": False, "reason": f"invalid packet intervention: {error}"}
            direct_candidates.append({"now": key[0], "voice_id": key[1], **direct})
    total = next((item for item in total_candidates if item.get("passed")), None)
    direct = next((item for item in direct_candidates if item["passed"]), None)
    wav = None
    wav_after_total = None
    if full_wav is not None and backward_wav is not None:
        wav = first_wav_difference(full_wav, backward_wav)
        if total is not None:
            wav_after_total = first_wav_difference(full_wav, backward_wav, total["now"])
    audio_valid = (wav is not None and wav["first_frame"] is not None
                   and wav["first_frame"] >= lower and wav_after_total is not None
                   and wav_after_total["first_frame"] is not None)
    return {"passed": total is not None and direct is not None and audio_valid,
            "observations": observations, "prefix_decisions": len(left_prefix),
            "first_selection_divergence": first_selection,
            "first_schedule_divergence": (None if first_schedule is None else
                                          {"now": first_schedule[0][0],
                                           "voice_id": first_schedule[0][1],
                                           "present_only_in": first_schedule[1]}),
            "first_total_effect": total, "total_candidates": len(total_candidates),
            "total_reasons": [item for item in total_candidates if not item.get("passed")],
            "direct_positive": direct, "direct_candidates": len(direct_candidates),
            "direct_reasons": [item for item in direct_candidates if not item["passed"]],
            "unmatched_decisions": unmatched, "first_wav_difference": wav,
            "first_wav_after_total": wav_after_total,
            "wav_timing": ("unresolved_within_period_observation_window" if wav is not None
                           and wav["first_frame"] is not None and
                           wav["first_frame"] <= observations["first_all_bus"]["first_period_evidence_cut"]
                           else "after_first_period_evidence" if wav is not None and
                           wav["first_frame"] is not None else "none")}


def reduced(repetitions):
    if set(repetitions) != {1, 2, 3}:
        raise ValueError("performance cell lacks exactly three repetitions")
    return {field: (sum(repetitions[rep][field] for rep in (1, 2, 3))
                    if field == "over_budget_hops" else
                    statistics.median(repetitions[rep][field] for rep in (1, 2, 3)))
            for field in PERFORMANCE_FIELDS}


def performance_cell(case, mode, profiles, floor):
    if set(profiles) != {"old-full", "full", "forward-cut", "backward-cut"}:
        raise ValueError(f"performance arm set differs: {case}/{mode}")
    measured = {arm: reduced(reps) for arm, reps in profiles.items()}
    eligible = []
    for arm, reps in profiles.items():
        if set(reps) != {1, 2, 3}:
            raise ValueError(f"performance repetitions differ: {case}/{mode}/{arm}")
        for rep, item in reps.items():
            count = item["period_accent_cut_deliveries"] if arm == "backward-cut" else item[
                "arrival_known_decisions"]
            eligible.append({"case": case, "mode": mode, "arm": arm, "rep": rep,
                             "count": count, "passed": type(count) is int and count >= 1})
    checks = []
    cuts = []
    for field in PERFORMANCE_FIELDS:
        old = measured["old-full"][field]
        new = measured["full"][field]
        width = floor[field]
        if field == "over_budget_hops":
            tolerance = width
            original = sorted(profiles["old-full"][rep][field] for rep in (1, 2, 3)) == sorted(
                profiles["full"][rep][field] for rep in (1, 2, 3))
        elif field.endswith("_max"):
            tolerance = max(1000.0, width)
            original = abs(new - old) <= 1000.0
        else:
            tolerance = max(0.05 * old, width)
            original = abs(new - old) <= 0.05 * old
        checks.append({"case": case, "mode": mode, "field": field,
                       "new_full": new, "old_full": old, "new_minus_old": new - old,
                       "old_floor": width, "upper_tolerance": tolerance,
                       "passed": new - old <= tolerance,
                       "old_absolute_with_floor": abs(new - old) <= tolerance,
                       "old_rule_without_floor": original})
        for arm in ("forward-cut", "backward-cut"):
            value = measured[arm][field]
            cuts.append({"case": case, "mode": mode, "arm": arm, "field": field,
                         "cut": value, "full": new, "cut_minus_full": value - new})
    return {"eligibility": eligible, "nonregression": checks, "cut_diagnostics": cuts,
            "reduced": measured}


def profile_measurement(path, arm):
    result = run_profile(Path(path))
    profile = json.loads(Path(path).read_text())
    counter = (profile.get("background") or {}).get("shared")
    if arm == "backward-cut":
        if not isinstance(counter, list) or not counter:
            raise ValueError("missing background shared bus0")
        value = counter[0].get("period_accent_cut_deliveries")
        if type(value) is not int or value < 0:
            raise ValueError("invalid backward cut count")
    else:
        value = None
        if isinstance(counter, list) and any(isinstance(bus, dict) and
                                              "period_accent_cut_deliveries" in bus
                                              for bus in counter):
            raise ValueError("cut count appears outside backward cut")
        if result["arrival_known_decisions"] is None:
            raise ValueError("missing arrival known count")
    return {**result, "period_accent_cut_deliveries": value,
            "background_diagnostics": profile.get("background")}


def cut_counter_trace(report_path, arm, profile_total=None):
    observations = read_report(report_path, ("temporal_observation",))["temporal_observation"]
    if not observations:
        raise ValueError("report lacks temporal observations for cut counter")
    by_bus = {}
    for row in observations:
        observation = row["observation"]
        bus = observation["bus"]
        value = observation.get("period_accent_cut_deliveries")
        if arm != "backward-cut":
            if "period_accent_cut_deliveries" in observation:
                raise ValueError("cut count appears outside backward cut report")
            continue
        if type(value) is not int or value < 0:
            raise ValueError("invalid backward cut report count")
        prior = by_bus.setdefault(bus, [])
        if prior and value < prior[-1]:
            raise ValueError("backward cut report count decreased")
        prior.append(value)
    if arm == "backward-cut":
        if not by_bus.get(0):
            raise ValueError("backward cut report lacks habitat observations")
        if profile_total is not None and by_bus[0][-1] > profile_total:
            raise ValueError("report count exceeds whole-run profile")
    return {"observation_count": len(observations),
            "last_by_bus": {str(bus): values[-1] for bus, values in by_bus.items()},
            "observed_habitat_cut": bool(by_bus.get(0) and by_bus[0][-1] > 0),
            "profile_whole_run": profile_total,
            "terminal_equality_claimed": False}


def performance_matrix(runs, floor):
    profiles = {}
    counters = []
    diagnostics = []
    for entry in runs.values():
        row = entry["registration"]
        if row["phase"] != "performance":
            continue
        key = (row["case"], row["mode"])
        arm = row["arm"]
        measurement = profile_measurement(entry["paths"]["profile"], arm)
        profiles.setdefault(key, {}).setdefault(arm, {})[row["rep"]] = measurement
        diagnostics.append({"case": row["case"], "mode": row["mode"], "arm": arm,
                            "rep": row["rep"], "background": measurement["background_diagnostics"]})
        if row["mode"] == "report":
            counters.append({"case": row["case"], "arm": arm, "rep": row["rep"],
                             **cut_counter_trace(entry["paths"]["report"], arm,
                                                 measurement["period_accent_cut_deliveries"])})
    if len(profiles) != 12:
        raise ValueError("performance matrix lacks 12 case/mode cells")
    output = [performance_cell(case, mode, arms, floor) for (case, mode), arms in sorted(profiles.items())]
    checks = [item for row in output for item in row["nonregression"]]
    cuts = [item for row in output for item in row["cut_diagnostics"]]
    eligible = [item for row in output for item in row["eligibility"]]
    if len(checks) != 120 or len(cuts) != 240 or len(eligible) != 144:
        raise ValueError("performance cell count differs")
    return {"nonregression": checks, "cut_diagnostics": cuts, "eligibility": eligible,
            "report_counter_checks": counters, "run_background_diagnostics": diagnostics,
            "nonregression_failed": sum(not item["passed"] for item in checks),
            "eligibility_failed": sum(not item["passed"] for item in eligible),
            "old_absolute_with_floor_failed": sum(not item["old_absolute_with_floor"] for item in checks),
            "old_rule_without_floor_failed": sum(not item["old_rule_without_floor"] for item in checks)}


def repo_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else REPO / path).resolve()
    if not path.is_relative_to(REPO):
        raise ValueError(f"sealed path outside repository: {value}")
    return path


def acquisition_sources(root, plan):
    freeze = json.loads(require_file(REGISTERED / "freeze.json", FREEZE_SHA).read_text())
    for relative, expected in freeze["files"].items():
        require_file(REPO / relative, expected)
    if digest(REGISTERED / "plan.json") != freeze["plan_sha256"]:
        raise ValueError("registered plan differs")
    provenance_path = root / "acquisition-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("schema") != "conchordal/i12b-acquisition-provenance/1"
            or provenance["registration_freeze_sha256"] != FREEZE_SHA
            or provenance["plan_sha256"] != freeze["plan_sha256"]):
        raise ValueError("acquisition provenance differs")
    seal = json.loads(require_file(provenance["candidate_seal"],
                                  provenance["candidate_seal_sha256"]).read_text())
    if seal["registration_freeze"]["sha256"] != FREEZE_SHA:
        raise ValueError("candidate seal registration differs")
    baseline = seal["baseline_ledger"]
    if (repo_path(baseline["path"]) != I11_ROOT / "runs.json"
            or baseline["sha256"] != BASELINE_LEDGER_SHA):
        raise ValueError("sealed old acquisition ledger identity differs")
    require_file(I11_ROOT / "runs.json", BASELINE_LEDGER_SHA)
    for relative, expected in seal["verifier_hashes"].items():
        require_file(repo_path(relative), expected)
    if seal["candidate_manifest"] != provenance["candidate_manifest"]:
        raise ValueError("candidate manifest seal differs")
    manifest_entry = seal["candidate_manifest"]
    manifest_path = require_file(repo_path(manifest_entry["path"]), manifest_entry["sha256"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_commit"] != provenance["candidate_source_commit"]:
        raise ValueError("candidate source commit differs")
    for relative, expected in manifest["source_files_sha256"].items():
        require_file(repo_path(relative), expected)
        require_file(manifest_path.parent / manifest["source_snapshot"] / relative, expected)
    old_manifest = require_file(REPO / plan["old_candidate"]["manifest_path"],
                                plan["old_candidate"]["manifest_sha256"])
    old = json.loads(old_manifest.read_text())
    for executable, entry in plan["old_candidate"]["binaries"].items():
        if old["binaries"][executable]["sha256"] != entry["sha256"]:
            raise ValueError("old candidate manifest differs")
        require_file(REPO / entry["path"], entry["sha256"])
    profile = provenance["profile"]
    if profile["sha256"] != plan["profile_sha256"]:
        raise ValueError("profile registration differs")
    require_file(profile["source_path"], profile["sha256"])
    if require_file(profile["path"], profile["sha256"]) != root / "action-profiles.bin":
        raise ValueError("profile copy outside acquisition")
    configs = provenance["configs"]
    if set(configs) != {row["config"] for row in plan["run_order"]}:
        raise ValueError("config set differs")
    for name, entry in configs.items():
        original = REGISTERED / name
        if require_file(entry["original"], plan["inputs_sha256"][name]) != original:
            raise ValueError(f"original config path differs: {name}")
        if entry["original_sha256"] != plan["inputs_sha256"][name]:
            raise ValueError(f"original config SHA declaration differs: {name}")
        derived = require_file(entry["path"], entry["sha256"])
        if derived != root / "run-configs" / original.name:
            raise ValueError(f"derived config path differs: {name}")
        values = tomllib.loads(original.read_text())
        values["temporal_action_profiles"]["file"] = profile["path"]
        if tomllib.loads(derived.read_text()) != values:
            raise ValueError(f"derived config value differs: {name}")
    binaries = provenance["binaries"]
    if set(binaries) != {"new-render", "new-instrument", "old-instrument"}:
        raise ValueError("binary role set differs")
    for role, executable in (("new-render", "conchordal-render"),
                             ("new-instrument", "conchordal"),
                             ("old-instrument", "conchordal")):
        entry = binaries[role]
        expected = (plan["old_candidate"]["binaries"][executable]["sha256"]
                    if role.startswith("old") else manifest["binaries"][executable]["sha256"])
        if entry["sha256"] != expected:
            raise ValueError(f"binary declaration differs: {role}")
        require_file(entry["source_path"], expected)
        if require_file(entry["path"], expected) != root / "binaries" / f"{role}-{executable}":
            raise ValueError(f"binary copy path differs: {role}")
    return provenance


def expected_paths(root, row):
    stem = f"{row['sequence']:03d}-{row['case']}-{row.get('arm', 'none')}-{row['mode']}-r{row['rep']}"
    paths = {"stdout": root / f"{stem}.stdout.log", "stderr": root / f"{stem}.stderr.log"}
    if row["phase"] == "performance":
        paths["profile"] = root / f"{stem}-profile.json"
    else:
        paths["wav"] = root / f"{stem}.wav"
    if row["mode"] == "report":
        paths["report"] = root / f"{stem}.jsonl"
    return paths


def expected_command(row, root, provenance):
    paths = expected_paths(root, row)
    scenario = (REGISTERED / row["scenario"] if row["scenario"].startswith("inputs/")
                else REPO / row["scenario"])
    command = [provenance["binaries"][row["binary_role"]]["path"], str(scenario),
               "--config", provenance["configs"][row["config"]]["path"]]
    if row["phase"] == "performance":
        command += ["--nogui", "--play=false", "--wait-user-start=false",
                    "--wait-user-exit=false", "--profile", str(paths["profile"])]
    else:
        command += ["-o", str(paths["wav"])]
    if "report" in paths:
        command += ["--report", str(paths["report"])]
    return command


def control_events(path, start, end):
    events = []
    expression = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4})\s+(\S+)")
    start = datetime.fromisoformat(start).replace(microsecond=0)
    end = datetime.fromisoformat(end)
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            match = expression.match(line)
            if not match:
                continue
            at = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S %z")
            if not start <= at <= end:
                continue
            if (match[2] in {"Sleep", "DarkWake", "Wake"} or "Using BATT" in line
                    or "Battery Power" in line or "Power Source" in line
                    or "Clamshell" in line or re.search(r"\blid\b", line, re.I)):
                events.append(line.rstrip())
    return events


def valid_power(row, pid):
    exits = row.get("command_exits") or {}
    assertions = row.get("pmset_assertions") or ""
    pid_text = str(pid)
    assertion = False
    lines = assertions.splitlines()
    for index, line in enumerate(lines):
        if not (re.search(r"\bpid\s+\d+\(caffeinate\):", line, re.I)
                and "PreventUserIdleSystemSleep" in line):
            continue
        stanza = [line]
        for detail in lines[index + 1:]:
            if re.search(r"^\s*pid\s+\d+\(", detail, re.I):
                break
            stanza.append(detail)
        if re.search(rf"\bon behalf of\b[^\n]*\bpid\s*{re.escape(pid_text)}\b",
                     "\n".join(stanza), re.I):
            assertion = True
            break
    return (set(exits) == {"pmset_batt", "ioreg", "pmset_assertions", "ps"}
            and all(type(value) is int and value == 0 for value in exits.values())
            and row.get("power_source") == "AC Power"
            and row.get("clamshell_state") == "No"
            and re.search(r"Now drawing from 'AC Power'", row.get("pmset_batt", "")) is not None
            and re.search(r'"AppleClamshellState"\s*=\s*No\b',
                          row.get("ioreg_clamshell", "")) is not None
            and row.get("build_processes") == []
            and row.get("caffeinate_for_python") is True
            and row.get("passed") is True and assertion)


def valid_run_pmset_audit(audit, saved, raw_log):
    return (audit.get("command") == ["pmset", "-g", "log"] and audit.get("exit") == 0
            and audit.get("events") == saved["pmset_events"]
            and type(audit.get("raw_bytes")) is int and audit["raw_bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", audit.get("raw_sha256", "")) is not None
            and saved["started_at"] <= audit.get("window_start", "")
            and audit.get("window_start", "") <= audit.get("window_end", "")
            and audit.get("window_end", "") <= saved["finished_at"]
            and control_events(raw_log, audit["window_start"], audit["window_end"])
            == audit["events"])


def acquired_runs(root, plan, provenance):
    ledger_path = root / "runs.json"
    ledger = json.loads(ledger_path.read_text())
    if (ledger.get("schema") != "conchordal/i12b-runs/1"
            or ledger.get("candidate_seal_sha256") != provenance["candidate_seal_sha256"]
            or ledger.get("provenance_sha256") != digest(root / "acquisition-provenance.json")
            or len(ledger.get("runs", [])) != 192
            or (root / "active-run.json").exists()):
        raise ValueError("incomplete or unsealed acquisition ledger")
    power_path = root / "power-states.jsonl"
    power = [json.loads(line) for line in power_path.read_text().splitlines()]
    if not ledger.get("control_sessions"):
        raise ValueError("missing power control session")
    sessions = []
    next_start = 0
    for session in ledger["control_sessions"]:
        status_path = root / session["path"] / "status.json"
        status = json.loads(require_file(status_path, session["status_sha256"]).read_text())
        if (session["status"] != "controlled" or status["controlled"] is not True
                or status["error"] is not None or status["pmset_control_events"]
                or status["power_start"] not in power or status["power_end"] not in power
                or status["starting_runs"] != next_start
                or not next_start < status["through_sequence"] <= 192
                or status["completed_runs"] != status["through_sequence"]):
            raise ValueError(f"power control session invalid: {session['path']}")
        next_start = status["through_sequence"]
        session_runs = [item for item in ledger["runs"] if
                        status["started_at"] <= item["started_at"] <= status["finished_at"]]
        if (not session_runs or not valid_power(status["power_start"], session_runs[0]["pid"])
                or not valid_power(status["power_end"], session_runs[-1]["pid"])):
            raise ValueError(f"power control record differs: {session['path']}")
        for label in ("before", "after"):
            require_file(root / session["path"] / f"pmset-{label}.log",
                         status[f"pmset_{label}_sha256"])
        if control_events(root / session["path"] / "pmset-after.log",
                          status["started_at"], status["finished_at"]) != []:
            raise ValueError(f"unreported block sleep/power event: {session['path']}")
        sessions.append((status, root / session["path"] / "pmset-after.log"))
    if next_start != 192:
        raise ValueError("power sessions do not cover all 192 runs")
    if len(plan["run_order"]) != 192:
        raise ValueError("registration run count differs")
    result = {}
    for row, saved in zip(plan["run_order"], ledger["runs"]):
        sequence = row["sequence"]
        if (saved.get("sequence") != sequence or saved.get("registration") != row
                or saved.get("command") != expected_command(row, root, provenance)
                or saved.get("status") != "completed" or saved.get("exit") != 0
                or saved.get("pmset_events") or saved.get("power_before") not in power
                or saved.get("power_after") not in power
                or not valid_power(saved["power_before"], saved["pid"])
                or not valid_power(saved["power_after"], saved["pid"])):
            raise ValueError(f"run or control differs: {sequence}")
        paths = expected_paths(root, row)
        if set(saved["artifacts"]) != set(paths):
            raise ValueError(f"artifact set differs: {sequence}")
        for kind, path in paths.items():
            entry = saved["artifacts"][kind]
            if (entry["path"] != path.relative_to(root).as_posix() or not path.is_file()
                    or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]
                    or (kind not in {"stdout", "stderr"} and entry["bytes"] <= 0)):
                raise ValueError(f"artifact differs: {sequence}/{kind}")
        pmset_entry = saved["pmset_log"]
        pmset = root / pmset_entry["path"]
        if (not pmset.is_file() or pmset.stat().st_size != pmset_entry["bytes"]
                or digest(pmset) != pmset_entry["sha256"]):
            raise ValueError(f"run power log differs: {sequence}")
        audit = json.loads(pmset.read_text())
        matching_sessions = [(status, raw) for status, raw in sessions
                             if status["started_at"] <= saved["started_at"]
                             and saved["finished_at"] <= status["finished_at"]]
        if len(matching_sessions) != 1:
            raise ValueError(f"run outside unique control session: {sequence}")
        status, raw_log = matching_sessions[0]
        if not valid_run_pmset_audit(audit, saved, raw_log):
            raise ValueError(f"run pmset audit differs from block raw log: {sequence}")
        case = row["case"]
        scenario = (REGISTERED / row["scenario"] if row["scenario"].startswith("inputs/")
                    else REPO / row["scenario"])
        expected_sha = (plan["inputs_sha256"][row["scenario"]]
                        if row["scenario"].startswith("inputs/") else
                        plan["existing12_none_references"][case]["sha256"])
        require_file(scenario, expected_sha)
        if row["phase"] == "performance" and saved.get("eligible") is not True:
            raise ValueError(f"profile eligibility differs: {sequence}")
        result[sequence] = {**saved, "paths": paths}
    return result, ledger


def prior_offline_runs(plan):
    ledger = json.loads((I11_ROOT / "runs.json").read_text())
    if ledger.get("schema") != "conchordal/i11-stage2-runs/1" or len(ledger["runs"]) != 852:
        raise ValueError("old I11-2 acquisition ledger incomplete")
    prior_provenance = I11_ROOT / "acquisition-provenance.json"
    if ledger["provenance_sha256"] != digest(prior_provenance):
        raise ValueError("old I11-2 acquisition provenance differs")
    provenance = json.loads(prior_provenance.read_text())
    if provenance["candidate_manifest_sha256"] != plan["old_candidate"]["manifest_sha256"]:
        raise ValueError("old I11-2 candidate identity differs")
    output = {}
    for saved in ledger["runs"][:96]:
        row = saved["registration"]
        case = row["case"]
        if (row["binary"], row["config"]) not in {
                ("stage2_candidate", "none"), ("stage2_candidate", "body-on")}:
            continue
        if row["config"] == "none" and case not in plan["existing12_none_references"]:
            continue
        if row["config"] == "body-on" and case not in plan["new6_cases"]:
            continue
        key = (case, row["config"])
        if key in output or saved["exit"] != 0:
            raise ValueError(f"old I11-2 run duplicate or failed: {key}")
        paths = {}
        for kind in ("wav", "report"):
            entry = saved["artifacts"][kind]
            path = I11_ROOT / entry["path"]
            if (not path.is_file() or path.stat().st_size != entry["bytes"]
                    or digest(path) != entry["sha256"]):
                raise ValueError(f"old I11-2 artifact differs: {key}/{kind}")
            paths[kind] = path
        output[key] = paths
    if len(output) != 18:
        raise ValueError(f"old I11-2 baseline pair count differs: {len(output)}")
    return output


def report_types(path):
    kinds = set()
    with Path(path).open() as stream:
        for line in stream:
            kinds.add(json.loads(line)["type"])
    return kinds


def offline_matrix(runs, plan):
    old = prior_offline_runs(plan)
    offline = {}
    for saved in runs.values():
        row = saved["registration"]
        if row["phase"] == "performance":
            continue
        key = (row["case"], row.get("arm", "none"), row["mode"])
        if key in offline:
            raise ValueError(f"duplicate offline cell: {key}")
        offline[key] = saved["paths"]
    if len(offline) != 48:
        raise ValueError("offline matrix lacks 48 cases")
    failures = []
    none = []
    for case in sorted(plan["existing12_none_references"]):
        current = offline[(case, "none", "report")]
        baseline = old[(case, "none")]
        learning = exact_report_pair(baseline["report"], current["report"], LEARNING_TYPES)
        same_wav = digest(baseline["wav"]) == digest(current["wav"])
        none.append({"case": case, "same_wav": same_wav, "learning": learning})
        if not same_wav or not learning["passed"]:
            failures.append(f"None bit regression: {case}")
    full_regression = []
    effects = []
    positives = {body: {"forward": False, "backward": False}
                 for body in ("sine", "harmonic", "modal")}
    for case in plan["new6_cases"]:
        baseline = old[(case, "body-on")]
        full = offline[(case, "full", "report")]
        kinds = report_types(baseline["report"]) | report_types(full["report"])
        kinds.discard("body_candidate_energy")
        required = {"participation_decision", "temporal_observation"}
        if not required <= kinds:
            failures.append(f"full report lacks causal core: {case}")
        comparison = exact_report_pair(baseline["report"], full["report"], sorted(kinds))
        candidates = compare_candidates_bits(baseline["report"], full["report"])
        same_wav = digest(baseline["wav"]) == digest(full["wav"])
        regression_passed = (same_wav and comparison["passed"] and candidates["matched"] > 0
                             and candidates["differing"] == 0 and candidates["bit_differing"] == 0)
        full_regression.append({"case": case, "passed": regression_passed,
                                "same_wav": same_wav, "report": comparison,
                                "candidates": candidates})
        if not regression_passed:
            failures.append(f"old/new full regression: {case}")

        pair = {arm: offline[(case, arm, "report")] for arm in
                ("full", "forward-cut", "backward-cut")}
        for arm in pair:
            report_result = verify_report(pair[arm]["report"])
            if report_result["decisions"] == 0 or report_result["failed_decisions"]:
                failures.append(f"invalid arrival report: {case}/{arm}")
            stage1 = verify_stage1_report(pair[arm]["report"], arrival_on=True)
            if stage1["decisions"] == 0 or stage1["failed_checks"]:
                failures.append(f"invalid base/footprint report: {case}/{arm}")
            causal_records = read_report(pair[arm]["report"],
                                         ("temporal_observation", "participation_decision"))
            coverage = publication_coverage(causal_records["temporal_observation"],
                                            causal_records["participation_decision"])
            pair[arm]["publication_coverage"] = coverage
            pair[arm]["cut_counter_trace"] = cut_counter_trace(pair[arm]["report"], arm)
            if not coverage["passed"]:
                failures.append(f"invalid observed publication linkage: {case}/{arm}")
        full_rows = read_report(pair["full"]["report"], ("participation_decision",))["participation_decision"]
        forward_rows = read_report(pair["forward-cut"]["report"],
                                   ("participation_decision",))["participation_decision"]
        forward = forward_pair(full_rows, forward_rows)
        forward_prefix = (forward_observation_prefix(pair["full"]["report"],
                                                     pair["forward-cut"]["report"],
                                                     forward["now"])
                          if forward.get("first_index") is not None else
                          {"passed": False, "reason": "no forward choice divergence"})
        forward["observation_prefix"] = forward_prefix
        forward["passed"] = forward["passed"] and forward_prefix["passed"]
        backward = backward_pair(pair["full"]["report"], pair["backward-cut"]["report"],
                                 pair["full"]["wav"], pair["backward-cut"]["wav"])
        same_mode = {}
        for arm in pair:
            rendered = offline[(case, arm, "no-report")]
            same_mode[arm] = digest(pair[arm]["wav"]) == digest(rendered["wav"])
            if not same_mode[arm]:
                failures.append(f"report toggles WAV: {case}/{arm}")
        forward_wav = digest(pair["full"]["wav"]) != digest(pair["forward-cut"]["wav"])
        backward_wav = digest(pair["full"]["wav"]) != digest(pair["backward-cut"]["wav"])
        body = case.split("-")[0]
        positives[body]["forward"] |= forward["passed"] and forward_wav
        positives[body]["backward"] |= backward["passed"] and backward_wav
        effects.append({"case": case, "forward": forward, "backward": backward,
                        "forward_wav_differs": forward_wav,
                        "backward_wav_differs": backward_wav,
                        "report_no_report_wav_equal": same_mode,
                        "publication_coverage": {arm: pair[arm]["publication_coverage"]
                                                 for arm in pair},
                        "cut_counter_trace": {arm: pair[arm]["cut_counter_trace"]
                                              for arm in pair}})
    for body, status in positives.items():
        for arm, passed in status.items():
            if not passed:
                failures.append(f"no positive {arm} effect for {body}")
    return {"none_regression": none, "full_regression": full_regression,
            "effect_cases": effects, "body_positive": positives,
            "failures": failures}


def evaluate(root):
    root = Path(root).resolve()
    plan = json.loads((REGISTERED / "plan.json").read_text())
    provenance = acquisition_sources(root, plan)
    runs, ledger = acquired_runs(root, plan, provenance)
    floor_path = REPO / plan["performance_gate"]["old_floor_path"]
    floor = json.loads(require_file(floor_path,
                                   plan["performance_gate"]["old_floor_sha256"]).read_text())
    if (floor["schema"] != "conchordal/i11-stage2-aa-floor/1"
            or set(floor["floor"]) != set(PERFORMANCE_FIELDS)):
        raise ValueError("old A/A floor schema or fields differ")
    offline = offline_matrix(runs, plan)
    performance = performance_matrix(runs, floor["floor"])
    failures = list(offline["failures"])
    if performance["nonregression_failed"]:
        failures.append(f"nonregression failed in {performance['nonregression_failed']}/120 fields")
    if performance["eligibility_failed"]:
        failures.append(f"performance eligibility failed in {performance['eligibility_failed']}/144 runs")
    return {"schema": "conchordal/i12b-evaluation/1", "acquisition_root": str(root),
            "registration_freeze_sha256": FREEZE_SHA,
            "candidate_seal_sha256": provenance["candidate_seal_sha256"],
            "provenance_sha256": digest(root / "acquisition-provenance.json"),
            "ledger_sha256": digest(root / "runs.json"), "run_count": len(ledger["runs"]),
            "existing_realtime_exact_exclusions": sorted(REALTIME_EXACT),
            "new_metadata_exclusions": [], "offline": offline,
            "performance": performance, "failures": failures,
            "passed": not failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"run_count": result["run_count"], "passed": result["passed"],
                      "failures": result["failures"]}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
