"""Offline f64 references for section correspondence and cached cue selection.

Inputs are observed path-local records from earlier stages, not Scenario labels.
The exhaustive record interface is an oracle, not the bounded production store.
"""

import math
import struct

from temporal_accent_reference import accent_at_cut


CATEGORIES = ("exact", "transformed", "contrast", "unmatched", "unresolved")
PREPARED_SECTION_BYTES = 1408
COMMIT_SLICES = ((0, 248), (432, 480), (536, 544), (576, 624))


def _commit_bytes(buffer):
    """Only sealed statistics/receipts; observation sums, cuts and accents stay live."""
    return b''.join(buffer[start:end] for start, end in COMMIT_SLICES)


def _activity_result(numerators, denominators, physical_valid_seconds, assignment_seconds, physical_window_seconds):
    coverage = [v / physical_window_seconds if physical_window_seconds > 0 else 0
                for v in physical_valid_seconds]
    values = [n / d if d > 0 and c >= 0.9 else None
              for n, d, c in zip(numerators, denominators, coverage)]
    return {"values": values, "valid": [v is not None for v in values],
            "numerators": numerators, "denominators": denominators,
            "physical_valid_seconds": physical_valid_seconds, "coverage": coverage,
            "assignment_seconds": assignment_seconds, "physical_window_seconds": physical_window_seconds}


def ending_descriptor(hops, accents, log2_bins, epoch, generation, epoch_start, generation_start,
                      span_start, support_end, means, standard_deviations, capacity_evicted_through=None):
    """Extract the six ending coordinates from original assigned acoustic hops.

    Energy/spectra already include acoustic assignment. Do not multiply them by
    assignment probability again. Raw rise/flux and admitted accents come from
    the earlier detector; this is not a waveform analyser or a new peak detector.
    """
    if (len(means) != 6 or len(standard_deviations) != 6 or not log2_bins
            or not all(math.isfinite(x) for x in [*means, *standard_deviations, *log2_bins,
                                                   epoch_start, generation_start, span_start, support_end])
            or any(sd < 0 for sd in standard_deviations)
            or max(epoch_start, generation_start, span_start) > support_end):
        raise ValueError("six finite scales, frequency bins and ordered ending support required")
    start = max(epoch_start, generation_start, span_start, support_end - 2)
    raw, coverage = [None] * 6, [0.0] * 6
    window_length = support_end - start
    if capacity_evicted_through is not None and (not math.isfinite(capacity_evicted_through)
                                                or capacity_evicted_through > support_end):
        raise ValueError("capacity history must be frozen at the ending support cut")
    canonical = {}
    for hop in hops:
        if hop["epoch"] != epoch or hop["generation"] != generation:
            continue
        lo, hi = hop["start"], hop["end"]
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("canonical acoustic hops require finite positive intervals")
        if hi > support_end or hi <= start or not hop["observed"] or not hop["association_known"]:
            continue
        key = (lo, hi)
        if key in canonical and canonical[key] != hop:
            raise ValueError("conflicting copies of one observed hop")
        canonical[key] = hop
    valid_seconds = [0.0] * 6
    spectrum = [0.0] * len(log2_bins)
    energy = rise = flux = 0.0
    last = start
    for (lo, hi), hop in sorted(canonical.items()):
        lo = max(start, lo)
        if lo < last:
            raise ValueError("canonical observed hops must not overlap")
        last = hi
        duration = hi - lo
        valid_seconds[5] += duration
        if hop["spectrum"] is not None:
            if (len(hop["spectrum"]) != len(log2_bins)
                    or not all(math.isfinite(v) and v >= 0 for v in hop["spectrum"])):
                raise ValueError("assigned spectrum must align with the log-frequency bins")
            for index, value in enumerate(hop["spectrum"]):
                spectrum[index] += duration * value
            valid_seconds[0] += duration
            valid_seconds[1] += duration
        for key, index in (("energy", 2), ("rise", 3), ("flux", 4)):
            value = hop[key]
            if value is None:
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError("assigned energy and positive change components must be nonnegative")
            valid_seconds[index] += duration
            if index == 2:
                energy += duration * value
            elif index == 3:
                rise += duration * value
            else:
                flux += duration * value
    if window_length > 0:
        coverage = [seconds / window_length for seconds in valid_seconds]
        mass = math.fsum(spectrum)
        if coverage[0] >= 0.9 and mass > 0:
            raw[0] = math.fsum(m * f for m, f in zip(spectrum, log2_bins)) / mass
            raw[1] = math.sqrt(math.fsum(m * (f - raw[0]) ** 2 for m, f in zip(spectrum, log2_bins)) / mass)
        if coverage[2] >= 0.9:
            raw[2] = math.log2(max(1e-6, math.sqrt(energy / valid_seconds[2])))
        for index, numerator in ((3, rise), (4, flux)):
            if coverage[index] >= 0.9:
                raw[index] = numerator / valid_seconds[index]
        kept = {}
        for accent in accents:
            if accent["epoch"] != epoch or accent["generation"] != generation:
                continue
            time, weight = accent["time"], accent["weight"]
            if not math.isfinite(time) or not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("admitted accents require finite time and fractional weight")
            if not accent_at_cut(accent, support_end) or not start <= time <= support_end:
                continue
            key = accent["id"]
            if key in kept and kept[key] != accent:
                raise ValueError("conflicting copies of one admitted accent")
            kept[key] = accent
        if coverage[5] >= 0.9 and (capacity_evicted_through is None or capacity_evicted_through < start):
            raw[5] = math.log1p(math.fsum(a["weight"] for a in kept.values()) / valid_seconds[5])
    scaled = [(v - mean) / max(sd, 1e-6) if v is not None else None
              for v, mean, sd in zip(raw, means, standard_deviations)]
    return {"window": [start, support_end], "raw": raw, "scaled": scaled,
            "valid": [v is not None for v in raw], "coverage": coverage,
            "epoch": epoch, "ending_generation": generation}


def correspondence_category(assignment, ending_descriptor, predecessor_descriptor):
    """Classify one assigned alternative without an extra matcher invocation."""
    status = assignment["status"]
    if status not in ("match", "no_memory") or assignment.get("bound_hit", False):
        return 4
    if (not assignment["supported"] or assignment.get("ambiguous_cutoff", False)
            or not assignment["search_nonempty"]):
        return 4
    cost = assignment.get("cost")
    if status == "match":
        if cost is None:
            return 4
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cached match cost must be finite and nonnegative")
        if cost <= 1:
            shifts = [assignment.get("frequency_shift_log2"), assignment.get("tempo_shift_log2")]
            if not all(value is None or math.isfinite(value) for value in shifts):
                raise ValueError("identified transformations must be finite")
            if cost > 0.25 or any(value is not None and abs(value) > 1 / 48 for value in shifts):
                return 1
            return 0 if all(value is not None for value in shifts) else 4
    if not all(assignment[name] for name in ("search_completed", "search_covered", "search_nonempty")):
        return 4
    if predecessor_descriptor is None:
        return 3
    if len(ending_descriptor) != 6 or len(predecessor_descriptor) != 6:
        raise ValueError("ending descriptors must contain six standardized coordinates")
    differences = []
    for value, previous in zip(ending_descriptor, predecessor_descriptor):
        if value is not None and previous is not None:
            if not math.isfinite(value) or not math.isfinite(previous):
                raise ValueError("supported ending descriptors must be finite")
            differences.append((value - previous) ** 2)
    if not differences:
        return 4
    return 2 if math.sqrt(math.fsum(differences) / len(differences)) > 1 else 3


def section_correspondence(records, known_intervals, epoch, section_start, observed_end, ring_size=4):
    """Thirty coordinates from canonical committed records, retaining unknowns.

    Each record has one assigned correspondence alternative on this local path.
    Alternative paths remain separate; their marginal category vectors must not
    create an independent cross-record joint distribution. Fractional membership
    multiplies valid acoustic assignment seconds once. Alias/retrieval records
    add no observation. The caller seals commits; conflicting duplicates fail.
    """
    if (not math.isfinite(section_start) or not math.isfinite(observed_end)
            or observed_end < section_start or not isinstance(ring_size, int) or ring_size < 1):
        raise ValueError("ordered finite section window and positive ring size required")
    intervals = []
    for lo, hi in known_intervals:
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo or hi > observed_end:
            raise ValueError("known audio/association intervals must precede the observation cut")
        intervals.append((lo, hi))
    canonical = {}
    for record in records:
        if record["epoch"] != epoch or record["record_kind"] != "observed_commit":
            continue
        start, end = record["start"], record["support_end"]
        weight, membership = record["assignment_seconds"], record["membership"]
        if (not all(math.isfinite(v) for v in (start, end, weight, membership)) or start > end
                or not 0 <= weight <= end - start + 1e-12 or not 0 <= membership <= 1):
            raise ValueError("finite observed support and fractional membership required")
        if end > observed_end or weight * membership == 0:
            continue
        key = record["occurrence_id"]
        if not isinstance(key, int) or key < 0:
            raise ValueError("occurrence IDs must be nonnegative integers")
        if key in canonical and canonical[key] != record:
            raise ValueError("conflicting commits for one occurrence require upstream reconciliation")
        canonical[key] = record
    ordered = sorted(canonical.values(), key=lambda r: (r["support_end"], r["start"], r["occurrence_id"]))
    selected, predecessor = [], None
    for record in ordered:
        categories = [0.0] * 5
        category = correspondence_category(record["assignment"], record["ending_descriptor"], predecessor)
        categories[category] = 1.0
        predecessor = record["ending_descriptor"]
        if record["start"] >= section_start:
            selected.append((record, record["assignment_seconds"] * record["membership"], categories))
    outputs = {}
    for name, entries in (("cumulative", selected), ("recent", selected[-ring_size:])):
        total = math.fsum(entry[1] for entry in entries)
        mass = [math.fsum(w * e[c] for _, w, e in entries) for c in range(5)]
        transitions = [0.0] * 25
        pair_total = pair_valid = 0.0
        for (previous, pw, pe), (current, cw, ce) in zip(entries, entries[1:]):
            pair_weight = min(pw, cw)
            pair_total += pair_weight
            lo, hi = previous["support_end"], current["start"]
            last, covered = lo, 0.0
            for left, right in sorted(intervals):
                left, right = max(lo, left), min(hi, right)
                if right > max(left, last):
                    covered += right - max(left, last)
                    last = right
            # Overlapping spans have no intervening gap; ordering still needs support.
            coverage = covered / (hi - lo) if hi > lo else 1.0
            if not previous["ordering_known"] or not current["ordering_known"] or coverage < 0.9:
                continue
            pair_valid += pair_weight
            for a in range(5):
                for b in range(5):
                    transitions[5 * a + b] += pair_weight * pe[a] * ce[b]
        valid_transition = pair_valid > 0 and pair_valid >= 0.9 * pair_total
        values = ([m / total for m in mass] if total > 0 else [None] * 5)
        values += [v / pair_valid for v in transitions] if valid_transition else [None] * 25
        outputs[name] = {"values": values, "valid": [v is not None for v in values],
                         "edge_numerators": mass, "edge_denominator": total,
                         "transition_numerators": transitions, "pair_weight": pair_total,
                         "valid_pair_weight": pair_valid, "excluded_pair_weight": pair_total - pair_valid,
                         "occurrence_ids": [r["occurrence_id"] for r, _, _ in entries]}
    return outputs


def section_activity(hops, accents, epoch, generation_handles, start, observed_end):
    """Nine coordinates from earlier-stage acoustic/articulation/history caches.

    This exhaustive oracle retains physical support and assignment-weighted
    seconds separately. Event density, periodic fraction and overlap use the
    specification's D_g denominator. Occupancies and timing means normalize
    their valid weights. Per-component physical coverage accompanies every value.
    History residuals are supplied normalized cached record-to-mode means; this
    function does not reconstruct uncertain timing records or admit new modes.
    """
    if not math.isfinite(start) or not math.isfinite(observed_end) or start > observed_end:
        raise ValueError("ordered finite section support required")
    canonical = {}
    for hop in hops:
        if hop["epoch"] != epoch or hop["generation"] not in generation_handles:
            continue
        lo, hi = hop["start"], hop["end"]
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("canonical acoustic hops require finite positive intervals")
        if hi > observed_end or hi <= start or not hop["observed"] or not hop["association_known"]:
            continue
        key = (lo, hi)
        if key in canonical and canonical[key] != hop:
            raise ValueError("conflicting copies or overlapping generations in one local path")
        canonical[key] = hop
    duration_weights = []
    numerators = [[] for _ in range(9)]
    valid_seconds = [[] for _ in range(9)]
    valid_weights = [[] for _ in range(9)]
    last = start
    for (lo, hi), hop in sorted(canonical.items()):
        lo = max(lo, start)
        if lo < last:
            raise ValueError("canonical local-path acoustic support cannot overlap")
        last = hi
        duration = hi - lo
        support = hop["assignment_support"]
        if not math.isfinite(support) or not 0 <= support <= 1:
            raise ValueError("acoustic assignment support must lie in[0,1]")
        weight = duration * support
        duration_weights.append(weight)
        values = [None] * 9
        state = hop["articulation"]
        if state is not None:
            if not isinstance(state, int) or not 0 <= state < 4:
                raise ValueError("known articulation must select one of four states")
            values[:4] = [float(state == i) for i in range(4)]
        values[4] = 0.0
        proposals = hop["periodic_proposals"]
        if hop["resolved"] and proposals is not None:
            values[5] = float(any(p["admitted"] and p["kind"] in ("integer", "word") for p in proposals))
        offsets, residuals, offset_weights, residual_weights = [], [], [], []
        for history in hop["timing_histories"] if hop["resolved"] else []:
            if not history["supported"] or history["direction"] not in ("outgoing", "within"):
                continue
            if history["direction"] == "within" and not history["periodic"]:
                raise ValueError("within-group section history must be periodic")
            ranking = history["ranking_support"]
            if not math.isfinite(ranking) or ranking < 0:
                raise ValueError("history ranking support must be finite and nonnegative")
            modes = history["modes"]
            if not modes or ranking == 0:
                continue
            if not all(math.isfinite(p) and math.isfinite(m) and m >= 0 for p, m in modes):
                raise ValueError("mode positions and peak masses must be finite")
            total = math.fsum(m for _, m in modes)
            if total == 0:
                continue
            distances = [(min(p % 1, 1 - p % 1) / 0.5 if history["periodic"] else abs(p) / 4) ** 2
                         for p, _ in modes]
            offsets.append(ranking * math.fsum(m * d for (_, m), d in zip(modes, distances)) / total)
            offset_weights.append(ranking)
            residual = history["residual_squared_mean"]
            if residual is not None:
                if not math.isfinite(residual) or residual < 0:
                    raise ValueError("cached normalized residual dispersion must be nonnegative")
                residuals.append(ranking * residual)
                residual_weights.append(ranking)
        if offset_weights:
            values[6] = math.fsum(offsets) / math.fsum(offset_weights)
        if residual_weights:
            values[7] = math.fsum(residuals) / math.fsum(residual_weights)
        bus, resolved = hop["bus_energy"], hop["resolved_energies"]
        if hop["resolved"] and bus is not None and resolved is not None:
            if (not math.isfinite(bus) or bus < 0
                    or not all(math.isfinite(e) and e >= 0 for e in resolved.values())):
                raise ValueError("assigned resolved and bus energies must be nonnegative")
            if hop["group_handle"] not in resolved:
                raise ValueError("resolved group must occur in its own assignment inventory")
            own = resolved[hop["group_handle"]]
            others = math.fsum(e for key, e in resolved.items() if key != hop["group_handle"])
            values[8] = float(bus > 0 and own / bus >= 0.01 and others / bus >= 0.01)
        for index, value in enumerate(values):
            if value is not None:
                numerators[index].append(weight * value)
                valid_seconds[index].append(duration)
                valid_weights[index].append(weight)
    counts = {}
    for accent in accents:
        if accent["epoch"] != epoch or accent["generation"] not in generation_handles:
            continue
        time, weight = accent["time"], accent["weight"]
        if not math.isfinite(time) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("admitted accent time and weight must be valid")
        if not accent_at_cut(accent, observed_end) or not start <= time <= observed_end:
            continue
        key = accent["id"]
        if key in counts and counts[key] != accent:
            raise ValueError("conflicting copies of one admitted accent")
        counts[key] = accent
    total_weight = math.fsum(duration_weights)
    sums = [math.fsum(values) for values in numerators]
    sums[4] = math.fsum(accent["weight"] for accent in counts.values())
    durations = [math.fsum(values) for values in valid_seconds]
    denominators = [math.fsum(values) for values in valid_weights]
    for index in (4, 5, 8):
        denominators[index] = total_weight
    return {**_activity_result(sums, denominators, durations, total_weight, observed_end - start),
            "unique_accent_count": len(counts), "window": [start, observed_end]}


def partition_span_activity(hops, accents, spans, epoch, generation_handles, observed_end):
    """Exhaustive oracle over conserved path-local support assignments.

    Each canonical hop/accent supplies span_shares before the record's scalar
    membership. Shares of the same physical interval or accent total at most one;
    unassigned mass is not redistributed. Missing time must have an explicit hop
    record and its potential ownership. This does not infer ownership from labels.
    """
    totals = {key: {"numerators": [0.] * 9, "denominators": [0.] * 9,
                    "physical_valid_seconds": [0.] * 9, "assignment_seconds": 0.,
                    "physical_window_seconds": 0., "unique_accents": set()} for key in spans}
    for key, span in spans.items():
        if (not isinstance(key, int) or key < 0 or not all(math.isfinite(span[k]) for k in ("start", "support_end"))
                or not span["start"] <= span["support_end"] <= observed_end):
            raise ValueError("finite completed span intervals and stable IDs required")
    canonical = {}
    for hop in hops:
        if hop["epoch"] != epoch or hop["generation"] not in generation_handles or hop["end"] > observed_end:
            continue
        key = (hop["start"], hop["end"])
        if key in canonical and canonical[key] != hop:
            raise ValueError("conflicting canonical support or ownership")
        canonical[key] = hop
    previous_end = None
    covered_intervals = []
    for (start, end), hop in sorted(canonical.items()):
        if (not all(math.isfinite(v) for v in (start, end)) or end <= start
                or previous_end is not None and start < previous_end):
            raise ValueError("nonoverlapping positive canonical hop intervals required")
        previous_end = end
        covered_intervals.append((start, end))
        shares = hop["span_shares"]
        if any(key not in spans or not math.isfinite(w) or not 0 <= w <= 1 for key, w in shares.items()):
            raise ValueError("finite nonnegative shares for registered spans required")
        cuts = sorted({start, end} | {max(start, min(end, spans[key][bound]))
                                      for key in shares for bound in ("start", "support_end")})
        raw = section_activity([hop], [], epoch, generation_handles, start, end)
        for lo, hi in zip(cuts, cuts[1:]):
            active = {key: w for key, w in shares.items()
                      if spans[key]["start"] <= lo < hi <= spans[key]["support_end"] and w > 0}
            if math.fsum(active.values()) > 1 + 1e-12:
                raise ValueError("span ownership duplicates a physical support interval")
            for key, share in active.items():
                scale = share * (hi - lo) / (end - start)
                total = totals[key]
                for field in ("numerators", "denominators", "physical_valid_seconds"):
                    for index, value in enumerate(raw[field]):
                        total[field][index] += scale * value
                total["assignment_seconds"] += scale * raw["assignment_seconds"]
                total["physical_window_seconds"] += share * (hi - lo)
    # An omitted interval must not silently disappear from the coverage denominator.
    for span in spans.values():
        known_clock = math.fsum(max(0, min(hi, span["support_end"]) - max(lo, span["start"]))
                               for lo, hi in covered_intervals)
        if not math.isclose(known_clock, span["support_end"] - span["start"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("span clock requires explicit observed or missing support records")
    seen = {}
    for accent in accents:
        if accent["epoch"] != epoch or accent["generation"] not in generation_handles or accent["time"] > observed_end:
            continue
        if not accent_at_cut(accent, observed_end):
            continue
        key = accent["id"]
        if key in seen:
            if seen[key] != accent:
                raise ValueError("conflicting canonical accent ownership")
            continue
        seen[key] = accent
        shares = accent["span_shares"]
        if (not all(math.isfinite(accent[k]) for k in ("time", "weight")) or not 0 <= accent["weight"] <= 1
                or any(key not in spans or not math.isfinite(w) or not 0 <= w <= 1 for key, w in shares.items())
                or math.fsum(shares.values()) > 1 + 1e-12):
            raise ValueError("one accent must have conserved finite span ownership")
        for key, share in shares.items():
            if not spans[key]["start"] <= accent["time"] <= spans[key]["support_end"]:
                raise ValueError("accent ownership must lie in its span support")
            if share > 0:
                totals[key]["numerators"][4] += accent["weight"] * share
                totals[key]["unique_accents"].add(accent["id"])
    return {key: {**_activity_result(t["numerators"], t["denominators"], t["physical_valid_seconds"],
                                    t["assignment_seconds"], t["physical_window_seconds"]),
                  "unique_accent_count": len(t["unique_accents"]),
                  "window": [spans[key]["start"], spans[key]["support_end"]]}
            for key, t in totals.items()}


class SectionRecord:
    """A640-byte numerical record; Python object/scratch overhead is separate.

    Fifty-four f64 sums and six f64 ending descriptors use120 f32-sized slots.
    Timestamps occupy u64 bookkeeping slots as f64 bit patterns. Two-f32 sum
    storage changed a90% validity decision in the exhaustive boundary fixture.
    """
    __slots__ = ("buffer",)
    fields = {"edge_numerators": tuple(range(5)), "transition_numerators": tuple(range(5, 30)),
              "pair_weight": (30,), "numerators": tuple(range(31, 40)),
              "denominators": (40, 40, 40, 40, 41, 42, 43, 44, 45),
              "physical_valid_seconds": (46, 46, 46, 46, 47, 48, 49, 50, 51),
              "assignment_seconds": (52,), "physical_window_seconds": (53,)}
    _layouts = {field: (8*indices[0], struct.Struct(f'<{indices[-1]-indices[0]+1}d'),
                       struct.Struct(f'<{len(indices)}d')) for field, indices in fields.items()}
    _activity_stored = struct.Struct('<23d')
    _activity_expanded = struct.Struct('<29d')
    _sealed_stored = struct.Struct('<31d')

    def __init__(self, buffer=None):
        self.buffer = bytearray(640) if buffer is None else bytearray(buffer)
        if len(self.buffer) != 640:
            raise ValueError("section record must contain exactly640 bytes")

    def block(self, field):
        offset, stored, expanded = self._layouts[field]
        values = stored.unpack_from(self.buffer, offset)
        if stored.size != expanded.size:
            values = (values[0],)*3 + values
        return list(values)

    def add(self, field, values, scale=1):
        offset, stored, expanded = self._layouts[field]
        if len(values)*8 != expanded.size or not math.isfinite(scale) or scale < 0:
            raise ValueError("aligned nonnegative finite statistics required")
        old = self.block(field)
        updates = []
        for previous, value in zip(old, values):
            if not math.isfinite(value) or value < 0:
                raise ValueError("section sufficient statistics must be finite and nonnegative")
            total = previous + scale * value
            if not math.isfinite(total):
                raise ValueError("section statistic exceeds the declared numeric representation")
            updates.append(total)
        packed = expanded.pack(*updates)
        if stored.size != expanded.size:
            # Compare rounded results, including signed zero, before coalescing.
            if not packed[:8] == packed[8:16] == packed[16:24] == packed[24:32]:
                raise ValueError("four articulation coordinates must share their support denominator")
            packed = packed[24:]
        self.buffer[offset:offset+stored.size] = packed

    def add_activity(self, activity, scale=1, physical_window_seconds=None):
        """Merge all activity fields atomically; cumulative callers include clock gaps."""
        if (not math.isfinite(scale) or scale < 0
                or any(len(activity[k]) != 9 for k in ('numerators', 'denominators', 'physical_valid_seconds'))):
            raise ValueError('aligned nonnegative finite activity statistics required')
        physical = activity['physical_window_seconds'] if physical_window_seconds is None else physical_window_seconds
        values = (*activity['numerators'], *activity['denominators'], *activity['physical_valid_seconds'],
                  activity['assignment_seconds'], physical)
        old = self._activity_stored.unpack_from(self.buffer, 248)
        old = old[:9] + (old[9],)*3 + old[9:15] + (old[15],)*3 + old[15:]
        updates = []
        for previous, value in zip(old, values):
            if not math.isfinite(value) or value < 0:
                raise ValueError('activity sufficient statistics must be finite and nonnegative')
            total = previous + scale * value
            if not math.isfinite(total):
                raise ValueError('activity statistic exceeds the declared numeric representation')
            updates.append(total)
        packed = self._activity_expanded.pack(*updates)
        if (not packed[72:80] == packed[80:88] == packed[88:96] == packed[96:104]
                or not packed[144:152] == packed[152:160] == packed[160:168] == packed[168:176]):
            raise ValueError('four articulation coordinates must share their support denominator')
        self.buffer[248:432] = packed[:72] + packed[96:144] + packed[168:]

    def integer(self, slot, value=None):
        if value is not None:
            if not isinstance(value, int) or not 0 <= value < 2**64:
                raise ValueError("unsigned stable bookkeeping value required")
            struct.pack_into('<Q', self.buffer, 512+8*slot, value)
        return struct.unpack_from('<Q', self.buffer, 512+8*slot)[0]

    def time(self, slot, value=None):
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("finite physical timestamp required")
            struct.pack_into('<d', self.buffer, 512+8*slot, value)
        return struct.unpack_from('<d', self.buffer, 512+8*slot)[0]

    def ending(self, values=None):
        if values is not None:
            if len(values) != 6 or any(v is not None and not math.isfinite(v) for v in values):
                raise ValueError("six finite or masked ending coordinates required")
            struct.pack_into('<6d', self.buffer, 432, *(0 if v is None else v for v in values))
            self.integer(12, sum(1 << i for i, v in enumerate(values) if v is not None))
        raw = struct.unpack_from('<6d', self.buffer, 432)
        mask = self.integer(12)
        return [v if mask & (1 << i) else None for i, v in enumerate(raw)]

    def snapshot(self):
        edge, transitions = self.block('edge_numerators'), self.block('transition_numerators')
        denominator, pair_valid = math.fsum(edge), math.fsum(transitions)
        pair_total = self.block('pair_weight')[0]
        values = [v / denominator for v in edge] if denominator > 0 else [None]*5
        values += ([v / pair_valid for v in transitions] if pair_valid > 0 and pair_valid >= 0.9*pair_total
                   else [None]*25)
        activity = _activity_result(self.block('numerators'), self.block('denominators'),
                                    self.block('physical_valid_seconds'), self.block('assignment_seconds')[0],
                                    self.block('physical_window_seconds')[0])
        return {"values": values + activity['values'], "valid": [v is not None for v in values] + activity['valid'],
                "edge_numerators": edge, "edge_denominator": denominator,
                "transition_numerators": transitions, "pair_weight": pair_total,
                "valid_pair_weight": pair_valid, "excluded_pair_weight": max(0, pair_total-pair_valid),
                "activity": activity}


class SectionHistory:
    """Bounded aggregation of canonical deltas and already sealed span assignments.

    The caller owns provisional spans, the commitment ledger and observed ordering
    coverage. Old replays are rejected after ring eviction; this class does not
    pretend that a bounded record can reconcile arbitrary historical revisions.
    """
    __slots__ = ('cumulative', 'ring', 'ring_size')

    def __init__(self, epoch, generation, section_id, start, ring_size=4):
        if not isinstance(ring_size, int) or ring_size not in (2, 4, 8):
            raise ValueError("registered recent windows are2/4/8 completed spans")
        self.cumulative, self.ring, self.ring_size = SectionRecord(), [], ring_size
        for slot, value in ((0, epoch), (1, generation), (2, section_id)):
            self.cumulative.integer(slot, value)
        for slot in (4, 5, 6, 7):
            self.cumulative.time(slot, start)

    def inherit(self, old_generation, new_generation):
        if old_generation != self.cumulative.integer(1) or old_generation == new_generation:
            raise ValueError("continuation must name its actual predecessor generation")
        child = SectionHistory(self.cumulative.integer(0), new_generation, self.cumulative.integer(2),
                               self.cumulative.time(4), self.ring_size)
        child.cumulative = SectionRecord(self.cumulative.buffer)
        child.cumulative.integer(1, new_generation)
        child.cumulative.integer(15, 0)
        child.ring = [SectionRecord(record.buffer) for record in self.ring]
        return child

    def observe(self, activity, epoch, generation, accent_deliveries=()):
        if epoch != self.cumulative.integer(0) or generation != self.cumulative.integer(1):
            return False
        start, end = activity['window']
        if start == self.cumulative.time(6) and end == self.cumulative.time(7):
            return False
        if not math.isfinite(start) or not math.isfinite(end) or start < self.cumulative.time(7) or end <= start:
            raise ValueError("new canonical nonoverlapping activity delta required")
        if not math.isclose(activity['physical_window_seconds'], end-start, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("cumulative observation needs physical support, not fractional span ownership")
        candidate = SectionRecord(self.cumulative.buffer)
        if accent_deliveries and activity['numerators'][4] != 0:
            raise ValueError("delivery accounting requires an accent-free acoustic delta")
        delivery_order = [d['sequence'] for d in accent_deliveries]
        if delivery_order != sorted(delivery_order) or len(delivery_order) != len(set(delivery_order)):
            raise ValueError("unique ordered delivery sequences required within one delta")
        for delivery in accent_deliveries:
            accent = delivery['accent']
            sequence = delivery['sequence']
            if (not isinstance(sequence, int) or sequence <= 0
                    or accent['epoch'] != epoch or accent['generation'] != generation
                    or not math.isfinite(delivery['delivered_at'])
                    or not start <= delivery['delivered_at'] <= end
                    or not accent_at_cut(accent, end) or delivery['weight'] != accent['weight']):
                raise ValueError("canonical generation-local accent delivery in this delta required")
            if sequence <= candidate.integer(15):
                continue
            candidate.integer(15, sequence)
            if accent['time'] >= candidate.time(4):
                credit = [0.] * 9
                credit[4] = accent['weight']
                candidate.add('numerators', credit)
        candidate.add_activity(activity, physical_window_seconds=end-candidate.time(7))
        candidate.time(5, end)
        candidate.time(6, start)
        candidate.time(7, end)
        self.cumulative = candidate
        return True

    def commit(self, record, activity, sequence, adjacency_coverage):
        prepared = self._prepare_commit(record, activity, sequence, adjacency_coverage)
        return False if prepared is None else self._apply_prepared(prepared)

    def _prepare_commit(self, record, activity, sequence, adjacency_coverage):
        """Validate and compute one commit without changing any visible state."""
        if record['epoch'] != self.cumulative.integer(0) or record['record_kind'] != 'observed_commit':
            return None
        if not isinstance(sequence, int) or sequence <= 0 or not math.isfinite(adjacency_coverage) or not 0 <= adjacency_coverage <= 1:
            raise ValueError("positive sealed sequence and known adjacency coverage required")
        key = (record['support_end'], record['start'], record['occurrence_id'])
        previous_key = (self.cumulative.time(9), self.cumulative.time(8), self.cumulative.integer(10))
        if sequence <= self.cumulative.integer(13):
            for prior in self.ring:
                if prior.integer(13) == sequence and (prior.time(5), prior.time(4), prior.integer(3)) == key:
                    return None
            raise ValueError("old or conflicting sealed replay requires upstream reconciliation")
        if (self.cumulative.integer(11) and key < previous_key or key[0] > self.cumulative.time(5)
                or record['start'] > record['support_end']):
            raise ValueError("commit order and original observed support must be preserved")
        membership, seconds = record['membership'], record['assignment_seconds']
        if (not math.isfinite(membership) or not 0 <= membership <= 1 or not math.isfinite(seconds)
                or not 0 <= seconds <= key[0]-key[1]+1e-12
                or not math.isclose(seconds, activity['assignment_seconds'], rel_tol=1e-9, abs_tol=1e-12)
                or activity['window'] != [record['start'], record['support_end']]):
            raise ValueError("same sealed span support and once-applied fractional membership required")
        if membership*seconds == 0:
            return None
        predecessor = self.cumulative.ending() if self.cumulative.integer(11) else None
        category = correspondence_category(record['assignment'], record['ending_descriptor'], predecessor)
        item = SectionRecord()
        sealed = [0.] * 31
        sealed[category] = 0. + 1 * (membership*seconds)
        item.add_activity(activity, membership)
        item.ending(record['ending_descriptor'])
        for slot, value in ((0, record['epoch']), (1, record.get('ending_generation', self.cumulative.integer(1))),
                            (2, self.cumulative.integer(2)), (3, record['occurrence_id']), (13, sequence),
                            (14, int(record['ordering_known']))):
            item.integer(slot, value)
        item.time(4, record['start'])
        item.time(5, record['support_end'])
        cumulative = SectionRecord(self.cumulative.buffer)
        ring = self.ring
        if record['start'] >= cumulative.time(4):
            if ring:
                prior_edges = ring[-1].block('edge_numerators')
                weight = min(math.fsum(prior_edges), membership*seconds)
                sealed[30] = 0. + 1 * weight
                if record['ordering_known'] and ring[-1].integer(14) and adjacency_coverage >= 0.9:
                    prior_category = max(range(5), key=prior_edges.__getitem__)
                    sealed[5+5*prior_category+category] = 0. + 1 * weight
        if any(not math.isfinite(v) or v < 0 for v in sealed):
            raise ValueError('sealed statistics must be finite and nonnegative')
        item.buffer[:248] = SectionRecord._sealed_stored.pack(*sealed)
        if record['start'] >= cumulative.time(4):
            old = SectionRecord._sealed_stored.unpack_from(cumulative.buffer)
            # Include zero additions to retain signed-zero and invalid-state behavior.
            totals = [previous + 1 * value for previous, value in zip(old, sealed)]
            if any(not math.isfinite(v) for v in totals):
                raise ValueError('section statistic exceeds the declared numeric representation')
            cumulative.buffer[:248] = SectionRecord._sealed_stored.pack(*totals)
        cumulative.buffer[432:480] = item.buffer[432:480]
        cumulative.integer(12, item.integer(12))
        cumulative.integer(3, item.integer(1))
        cumulative.integer(11, 1)
        cumulative.integer(10, record['occurrence_id'])
        cumulative.integer(13, sequence)
        cumulative.time(8, record['start'])
        cumulative.time(9, record['support_end'])
        prepared = bytearray(PREPARED_SECTION_BYTES)
        struct.pack_into('<4sI3Qd2H', prepared, 0, b'SP1\0', int(record['start'] >= cumulative.time(4)),
                         *(self.cumulative.integer(i) for i in range(3)),
                         self.cumulative.time(4), self.ring_size, len(ring))
        prepared[64:416] = _commit_bytes(self.cumulative.buffer)
        prepared[416:768] = _commit_bytes(cumulative.buffer)
        prepared[768:] = item.buffer
        return bytes(prepared)

    def _apply_prepared(self, prepared):
        """Apply trusted private preparation, retaining all intervening observations.

        This is an internal owner-thread interface, not an untrusted wire decoder.
        Only canonical observe calls may intervene; other commit state is bound.
        """
        if (type(prepared) not in (bytes, memoryview) or len(prepared) != PREPARED_SECTION_BYTES
                or type(prepared) is memoryview and (not prepared.readonly or not prepared.c_contiguous
                                                     or prepared.format != 'B')):
            raise ValueError('fixed immutable private section preparation required')
        tag, append, epoch, generation, section, start, ring_size, count = struct.unpack_from('<4sI3Qd2H', prepared)
        if (tag != b'SP1\0' or append not in (0, 1) or prepared[44:64] != bytes(20)
                or (epoch, generation, section) != tuple(self.cumulative.integer(i) for i in range(3))
                or start != self.cumulative.time(4) or ring_size != self.ring_size or count != len(self.ring)
                or prepared[64:416] != _commit_bytes(self.cumulative.buffer)):
            raise ValueError('prepared section predecessor, identity or sealed state changed')
        cumulative = SectionRecord(self.cumulative.buffer)
        offset = 416
        for lo, hi in COMMIT_SLICES:
            cumulative.buffer[lo:hi] = prepared[offset:offset+hi-lo]
            offset += hi-lo
        ring = list(self.ring)
        if append:
            ring.append(SectionRecord(prepared[768:]))
            ring = ring[-self.ring_size:]
        self.cumulative, self.ring = cumulative, ring
        return True

    def snapshot(self):
        recent = SectionRecord()
        for index, record in enumerate(self.ring):
            for field in SectionRecord.fields:
                if index == 0 and field in ('transition_numerators', 'pair_weight'):
                    continue
                recent.add(field, record.block(field))
        cumulative, recent = self.cumulative.snapshot(), recent.snapshot()
        physical = self.cumulative.block('physical_window_seconds')[0]
        known = self.cumulative.block('physical_valid_seconds')[4]
        return {'cumulative': cumulative, 'recent': recent,
                'missing_fraction': 1-known/physical if physical > 0 else None,
                'occurrence_ids': [r.integer(3) for r in self.ring],
                'record_payload_bytes': 640*(1+len(self.ring))}


def select_section_cue(spans, epoch, generation_handles, observed_end, epoch_start, window_sec=0.5):
    """Select at the reused matcher's observation cut, never its publication time.

    Supported intervals have stable generation handles. The supplied handle set
    must be the actual path lineage, not a set reconstructed from recycled slots.
    Return the selected whole bounded-prefix descriptor unchanged.
    """
    if (not all(math.isfinite(v) for v in (observed_end, epoch_start, window_sec))
            or epoch_start > observed_end or window_sec <= 0):
        raise ValueError("finite observation window required")
    candidates = []
    seen = {}
    for span in spans:
        if span["epoch"] != epoch or span["record_kind"] not in ("ongoing", "observed_commit"):
            continue
        key = span["occurrence_id"]
        if (not isinstance(key, int) or key < 0 or not math.isfinite(span["start"])
                or not math.isfinite(span["descriptor_support_end"])):
            raise ValueError("stable occurrence ID and finite start required")
        if key in seen:
            if seen[key] != span:
                raise ValueError("conflicting cue records require upstream reconciliation")
            continue
        seen[key] = span
        if span["start"] > observed_end or span["descriptor_support_end"] > observed_end:
            continue
        selection_start = max(epoch_start, observed_end - window_sec, span["start"])
        total = recent = 0.0
        last_end = None
        prior_end = -math.inf
        intervals = sorted(set(tuple(row) for row in span["support_intervals"]), key=lambda row: (row[1], row[2], row[0]))
        for generation, lo, hi, weight in intervals:
            if not all(math.isfinite(v) for v in (lo, hi, weight)) or hi < lo or not 0 <= weight <= 1:
                raise ValueError("finite ordered support with fractional weight required")
            if generation not in generation_handles or hi > observed_end:
                continue
            lo = max(lo, span["start"], epoch_start)
            if hi <= lo or weight == 0:
                continue
            if lo < prior_end:
                raise ValueError("cue support must be canonical nonoverlapping physical intervals")
            prior_end = hi
            total += (hi - lo) * weight
            recent += max(0.0, hi - max(lo, selection_start)) * weight
            last_end = hi
        if total > 0:
            candidates.append((span, recent, last_end))
    ongoing = [c for c in candidates if c[0]["record_kind"] == "ongoing" and c[1] > 0]
    if ongoing:
        selected = min(ongoing, key=lambda c: (-c[1], -c[2], -c[0]["start"], c[0]["occurrence_id"]))
        kind = "ongoing"
    else:
        committed = [c for c in candidates if c[0]["record_kind"] == "observed_commit"]
        if not committed:
            return None
        selected = min(committed, key=lambda c: (-c[2], -c[0]["start"], c[0]["occurrence_id"]))
        kind = "committed_fallback"
    span, support, last_end = selected
    return {"occurrence_id": span["occurrence_id"], "query_descriptor": span["query_descriptor"],
            "descriptor_support_end": span["descriptor_support_end"], "kind": kind,
            "selection_weighted_seconds": support, "last_observed_end": last_end,
            "query_observation_end": observed_end}


def section_retrieval_scores(query, selected_cue, observed_end):
    """Best acoustic transformation per eligible episode, with original evidence age."""
    if selected_cue is None:
        return {"scores": [], "reason": "no_supported_cue"}
    if (query["cue_occurrence_id"] != selected_cue["occurrence_id"]
            or query["observation_end"] != selected_cue["query_observation_end"]
            or query["support_end"] != selected_cue["descriptor_support_end"]):
        raise ValueError("section features must reuse the exact selected cue query")
    if (not math.isfinite(observed_end) or not math.isfinite(query["support_end"])
            or query["support_end"] > query["observation_end"] or query["observation_end"] > observed_end):
        raise ValueError("matcher support must precede the current observed cut")
    if observed_end - query["support_end"] >= 0.5:
        return {"scores": [], "reason": "stale_support", "flags": query["flags"]}
    episodes = {}
    for match in query["matches"]:
        if not match["eligible"] or not match["supported"]:
            continue
        value = match["acoustic_score"]
        if not math.isfinite(value):
            raise ValueError("supported acoustic scores must be finite")
        episode = match["episode_handle"]
        episodes[episode] = max(episodes.get(episode, -math.inf), value)
    if len(episodes) > 16:
        raise ValueError("reused query exceeds the registered16-episode inventory")
    scores = sorted(episodes.values(), reverse=True)
    return {"scores": scores, "best": scores[0] if scores else None,
            "gap": scores[0] - scores[1] if len(scores) > 1 else None,
            "support_end": query["support_end"], "flags": query["flags"]}
