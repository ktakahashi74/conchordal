"""Offline f64 accent admission, original-support cuts and bounded delivery.

Ridge memberships and group associations are inputs, not Scenario identities.
These references do not implement NSGT, grouping inference or runtime scheduling.
"""

import copy
import math


def assigned_group_energy(bus_energy, spectrum, ridge_memberships, group_associations, residual):
    """Conserve mixture energy over normalized ridge and group assignments."""
    if (not spectrum or not ridge_memberships or len(ridge_memberships) != len(group_associations)
            or not group_associations[0]):
        raise ValueError("nonempty aligned spectrum, ridges and groups required")
    bins, groups = len(spectrum), len(group_associations[0])
    if (not isinstance(residual, int) or not 0 <= residual < groups
            or not math.isfinite(bus_energy) or bus_energy < 0
            or any(len(row) != bins for row in ridge_memberships)
            or any(len(row) != groups for row in group_associations)
            or not all(math.isfinite(x) and x >= 0 for x in spectrum)
            or not all(math.isfinite(x) and 0 <= x <= 1
                       for row in [*ridge_memberships, *group_associations] for x in row)):
        raise ValueError("finite nonnegative energy and aligned fractional memberships required")
    if any(not math.isclose(math.fsum(row), 1, rel_tol=0, abs_tol=1e-12)
           for row in [*zip(*ridge_memberships), *group_associations]):
        raise ValueError("ridge bin memberships and group association rows must sum to one")
    mass = math.fsum(spectrum)
    if bus_energy > 0 and mass == 0:
        return [{"energy": bus_energy if g == residual else 0., "spectrum": None}
                for g in range(groups)]
    output = []
    for g in range(groups):
        assigned = [bus_energy * (value / mass) * math.fsum(
            group_associations[i][g] * row[b] for i, row in enumerate(ridge_memberships))
                    if mass > 0 else 0. for b, value in enumerate(spectrum)]
        output.append({"energy": math.fsum(assigned), "spectrum": assigned})
    return output


def accent_at_cut(accent, observed_end):
    """Check an admitted cache record against its complete original evidence.

    time is the event-support endpoint used for ordering/window inventory. Timing
    kernels use event_interval instead; neither that endpoint nor a delivery time
    can replace the complete raw-support endpoint in a causal cut.
    """
    lo, hi = accent["event_interval"]
    time, raw_end, available = (accent[k] for k in ("time", "raw_support_end", "available_end"))
    if (not all(math.isfinite(v) for v in (lo, hi, time, raw_end, available, observed_end, accent["weight"]))
            or lo > hi or time != hi or hi > raw_end or raw_end > available
            or not 0 <= accent["weight"] <= 1):
        raise ValueError("ordered event, full raw evidence and availability endpoints required")
    return available <= observed_end


def accent_window(hops, means, standard_deviations, threshold=1., floor=1e-6):
    """Compare three saliences from four canonical, completely acquired raw hops.

    Sample intervals are integer and half-open. The peak retains the entire
    middle hop as its uncertain event interval. Raw support also includes the
    preceding differencing hop and the right comparison hop.
    """
    if (len(hops) != 4 or len(means) != 2 or len(standard_deviations) != 2
            or not all(math.isfinite(v) for v in [*means, *standard_deviations, threshold, floor])
            or any(v < 0 for v in standard_deviations) or threshold <= 0 or floor <= 0):
        raise ValueError("four raw hops and two finite development scales required")
    intervals = []
    rate = hops[0]["sample_rate"]
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError("positive integer sample rate required")
    for hop in hops:
        lo, hi = hop["sample_start"], hop["sample_end"]
        if (not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi <= lo
                or hop["sample_rate"] != rate or intervals and lo < intervals[-1][1]):
            raise ValueError("nonoverlapping canonical sample intervals on one clock required")
        intervals.append((lo, hi))
    lo, hi = intervals[0][0], intervals[-1][1]
    observed = 0
    for (a, b), hop in zip(intervals, hops):
        support = hop.get('known_sample_intervals', [(a, b)] if hop['observed'] else [])
        right = a
        for start, end in sorted(support):
            if (not isinstance(start, int) or not isinstance(end, int) or not a <= start <= end <= b):
                raise ValueError("known sample support must lie inside its canonical raw hop")
            observed += max(0, end - max(start, right))
            right = max(right, end)
    out = {"status": "unsupported", "acquisition_coverage": observed / (hi - lo),
           "detector_coverage": 0., "saliences": None, "accent": None,
           "raw_support_intervals": intervals, "raw_support_end": hi / rate}
    if (observed != hi - lo or not all(h["observed"] and h["association_known"] for h in hops)
            or any(intervals[i][1] != intervals[i+1][0] for i in range(3))
            or any(h[k] != hops[0][k] for h in hops for k in ("epoch", "generation", "association_handle", "grid_id"))
            or any(h["energy"] is None or h["spectrum"] is None for h in hops)):
        return out
    bins = len(hops[0]["spectrum"])
    if (bins == 0 or any(len(h["spectrum"]) != bins for h in hops)
            or not all(math.isfinite(h["energy"]) and h["energy"] >= 0 for h in hops)
            or not all(math.isfinite(v) and v >= 0 for h in hops for v in h["spectrum"])):
        raise ValueError("finite nonnegative assigned energies on one nonempty grid required")
    # Positive energy with no spectral evidence cannot support a flux comparison.
    if any(h["energy"] > 0 and math.fsum(h["spectrum"]) == 0 for h in hops):
        return out
    saliences, components = [], []
    for previous, current in zip(hops, hops[1:]):
        rise = max(0., math.log2(max(math.sqrt(current["energy"]), floor))
                   - math.log2(max(math.sqrt(previous["energy"]), floor)))
        flux = math.fsum(max(0., .5 * math.log2(max(b, floor*floor))
                             - .5 * math.log2(max(a, floor*floor)))
                         for a, b in zip(previous["spectrum"], current["spectrum"])) / bins
        components.append((rise, flux))
        saliences.append(math.fsum(max(0., (x - mean) / max(sd, 1e-6))
                                   for x, mean, sd in zip((rise, flux), means, standard_deviations)) / 2)
    out.update(detector_coverage=1., saliences=saliences, raw_components=components)
    if saliences[1] <= threshold:
        out["status"] = "below_threshold"
    elif not saliences[1] > saliences[0] or saliences[1] < saliences[2]:
        out["status"] = "not_local_peak"
    else:
        start, end = intervals[2]
        out["status"] = "admitted"
        out["accent"] = {"epoch": hops[2]["epoch"], "generation": hops[2]["generation"],
                         "id": (hops[2]["epoch"], hops[2]["generation"], start, end),
                         "event_interval": [start / rate, end / rate], "time": end / rate,
                         "raw_support_intervals": intervals, "sample_rate": rate,
                         "raw_support_end": hi / rate, "available_end": hi / rate,
                         "weight": min(1., saliences[1] - threshold)}
    return out


class AccentStream:
    """One generation's bounded four-hop detector with duplicate-read protection."""

    def __init__(self, epoch, generation, means, standard_deviations, threshold=1., floor=1e-6):
        self.epoch, self.generation = epoch, generation
        self.means, self.standard_deviations = list(means), list(standard_deviations)
        self.threshold, self.floor = threshold, floor
        self.hops = []

    def push(self, hop):
        if hop["epoch"] != self.epoch or hop["generation"] != self.generation:
            return None
        lo, hi = hop["sample_start"], hop["sample_end"]
        if (not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi <= lo
                or not isinstance(hop["sample_rate"], int) or hop["sample_rate"] <= 0):
            raise ValueError("positive canonical sample interval and rate required")
        if self.hops:
            if hop == self.hops[-1]:
                return None
            if lo < self.hops[-1]["sample_end"]:
                raise ValueError("stale or conflicting raw delivery requires upstream reconciliation")
        candidate = [*self.hops, copy.deepcopy(hop)]
        result = accent_window(candidate, self.means, self.standard_deviations, self.threshold, self.floor) if len(candidate) == 4 else None
        self.hops = candidate[-3:]
        return result


class AccentLedger:
    """Deliver once, retain a bounded retrospective bank and independent totals.

    Input is ordered detector admission, not arbitrary delayed packet order. A
    causal receiver may deliver later than raw evidence became available. Epoch
    or generation replacement creates a fresh ledger, never a recycled slot.
    """

    def __init__(self, epoch, generation, capacity=128, window_sec=32.):
        if not isinstance(capacity, int) or capacity <= 0 or not math.isfinite(window_sec) or window_sec <= 0:
            raise ValueError("positive registered bank capacity and window required")
        self.epoch, self.generation = epoch, generation
        self.capacity, self.window_sec = capacity, window_sec
        self.bank = []
        self.last = None
        self.observed_end = 0.
        self.capacity_evicted_through = None
        self.admission_count = 0
        self.admission_weight = 0.

    def deliver(self, accent, observed_end):
        if accent["epoch"] != self.epoch or accent["generation"] != self.generation:
            return None
        if not accent_at_cut(accent, observed_end):
            return None
        intervals, rate = accent["raw_support_intervals"], accent["sample_rate"]
        if (not isinstance(rate, int) or rate <= 0 or len(intervals) != 4
                or any(not isinstance(v, int) or v < 0 for pair in intervals for v in pair)
                or any(a >= b for a, b in intervals)
                or any(intervals[i][1] != intervals[i+1][0] for i in range(3))
                or accent["event_interval"] != [v / rate for v in intervals[2]]
                or accent["raw_support_end"] < intervals[-1][1] / rate
                or not math.isfinite(accent.get('raw_support_start', intervals[0][0] / rate))
                or not 0 <= accent.get('raw_support_start', intervals[0][0] / rate) <= intervals[0][0] / rate
                or accent["id"] != (self.epoch, self.generation, *intervals[2]) or accent["weight"] <= 0):
            raise ValueError("complete canonical detector provenance and stable accent ID required")
        if observed_end < self.observed_end:
            raise ValueError("delivery clock must not run backwards")
        # Stable IDs cannot acquire new credit by changing their evidence window.
        for old in ([self.last] if self.last is not None else []) + self.bank:
            if old['id'] == accent['id']:
                if old != accent:
                    raise ValueError("conflicting repeat of an admitted accent")
                return None
        key = (accent["raw_support_end"], accent["id"])
        if self.last is not None:
            if key <= (self.last["raw_support_end"], self.last["id"]):
                raise ValueError("stale admission requires upstream reconciliation, never new credit")
            if (accent['time'], accent['id']) <= (self.last['time'], self.last['id']):
                raise ValueError("canonical detector event order must be preserved")
        self.advance(observed_end)
        self.last = copy.deepcopy(accent)
        self.admission_count += 1
        self.admission_weight += accent["weight"]
        if accent["time"] >= observed_end - self.window_sec:
            if len(self.bank) == self.capacity:
                removed = self.bank.pop(0)
                self.capacity_evicted_through = max(removed["time"], self.capacity_evicted_through
                                                    if self.capacity_evicted_through is not None else -math.inf)
            self.bank.append(copy.deepcopy(accent))
        return {"sequence": self.admission_count, "delivered_at": observed_end,
                "accent": copy.deepcopy(accent), "weight": accent["weight"]}

    def advance(self, observed_end):
        if not math.isfinite(observed_end) or observed_end < self.observed_end:
            raise ValueError("finite monotone delivery clock required")
        self.observed_end = observed_end
        self.bank = [a for a in self.bank if a["time"] >= observed_end - self.window_sec]

    def snapshot(self, start):
        if not math.isfinite(start) or not self.observed_end - self.window_sec <= start <= self.observed_end:
            raise ValueError("retrospective query must fit the retained physical window")
        accents = [copy.deepcopy(a) for a in self.bank if start <= a["time"] <= self.observed_end]
        valid = self.capacity_evicted_through is None or self.capacity_evicted_through < start
        return {"window": [start, self.observed_end], "accents": accents, "capacity_valid": valid,
                "weight": math.fsum(a["weight"] for a in accents) if valid else None,
                "cumulative_admission_weight": self.admission_weight,
                "cumulative_admission_count": self.admission_count,
                "capacity_evicted_through": self.capacity_evicted_through}
