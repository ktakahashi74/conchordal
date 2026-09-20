"""Analytic section-record, ending-window and exact-cue fixtures."""

import copy
import math
import struct
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import temporal_section_reference as ref
import temporal_cognition_reference as cognition


def assignment(status="match", cost=0.1, **changes):
    return {"status": status, "cost": cost, "supported": True, "search_completed": True,
            "search_covered": True, "search_nonempty": True, "frequency_shift_log2": 0,
            "tempo_shift_log2": 0, **changes}


def span(identifier, start, end, category=None, membership=1):
    return {"epoch": 1, "record_kind": "observed_commit", "occurrence_id": identifier,
            "start": start, "support_end": end, "assignment_seconds": end - start,
            "membership": membership, "ordering_known": True, "ending_descriptor": [0] * 6,
            "assignment": assignment() if category is None else category}


class CorrespondenceTests(unittest.TestCase):
    def test_exact_thresholds_and_known_failures_with_missing_transformations(self):
        self.assertEqual(ref.correspondence_category(assignment(cost=0.25, frequency_shift_log2=1 / 48), [], None), 0)
        self.assertEqual(ref.correspondence_category(assignment(cost=0.25001, tempo_shift_log2=None), [], None), 1)
        self.assertEqual(ref.correspondence_category(assignment(frequency_shift_log2=0.1, tempo_shift_log2=None), [], None), 1)
        self.assertEqual(ref.correspondence_category(assignment(tempo_shift_log2=None), [], None), 4)
        self.assertEqual(ref.correspondence_category(assignment(cost=1), [], None), 1)

    def test_missing_search_bound_or_unheard_state_never_becomes_nonrecurrence(self):
        for status in ("unknown", "pruned", "pending_unheard", "empty", "missed", "ambiguous_cutoff"):
            self.assertEqual(ref.correspondence_category(assignment(status=status, cost=4), [100] * 6, [0] * 6), 4)
        for changes in ({"cost": None}, {"bound_hit": True}, {"supported": False},
                        {"ambiguous_cutoff": True}, {"search_nonempty": False}):
            self.assertEqual(ref.correspondence_category(assignment(**changes), [100] * 6, [0] * 6), 4)
        for flag in ("search_completed", "search_covered", "search_nonempty"):
            self.assertEqual(ref.correspondence_category(assignment(status="no_memory", **{flag: False}),
                                                        [100] * 6, [0] * 6), 4)

    def test_departure_uses_common_standardized_coordinates_and_real_predecessor(self):
        no_memory = assignment(status="no_memory", cost=None)
        self.assertEqual(ref.correspondence_category(no_memory, [1] * 6, [0] * 6), 3)
        self.assertEqual(ref.correspondence_category(no_memory, [1.1, None, None, None, None, None], [0] * 6), 2)
        self.assertEqual(ref.correspondence_category(no_memory, [None] * 6, [0] * 6), 4)
        self.assertEqual(ref.correspondence_category(no_memory, [None] * 6, None), 3)
        self.assertEqual(ref.correspondence_category(assignment(cost=1.01), [2] * 6, [0] * 6), 2)

    def test_physical_weights_and_unresolved_adjacency_match_hand_sum(self):
        records = [span(1, 0, 2), span(2, 3, 4, assignment("unknown"), 0.5),
                   span(3, 5, 6, assignment(cost=0.5))]
        out = ref.section_correspondence(records, [(0, 6)], 1, 0, 6)["cumulative"]
        self.assertEqual(out["edge_denominator"], 3.5)
        self.assertEqual(out["edge_numerators"], [2, 1, 0, 0, 0.5])
        self.assertEqual(out["valid_pair_weight"], 1)
        self.assertEqual(out["values"][5 + 4], 0.5)
        self.assertEqual(out["values"][5 + 4 * 5 + 1], 0.5)
        self.assertEqual(out["values"][5 + 1], 0)
        self.assertAlmostEqual(sum(out["values"][:5]), 1)
        self.assertAlmostEqual(sum(out["values"][5:]), 1)

    def test_aliases_duplicates_future_and_other_epoch_add_no_credit(self):
        first, second = span(1, 0, 1), span(2, 2, 3, assignment(cost=0.5))
        expected = ref.section_correspondence([first, second], [(0, 3)], 1, 0, 3)
        alias = {**first, "record_kind": "alias", "assignment": assignment("unknown")}
        records = [second, first, copy.deepcopy(first), alias,
                   {**first, "record_kind": "retrieval"}, {**second, "epoch": 0}, span(3, 4, 5)]
        self.assertEqual(ref.section_correspondence(records, [(0, 3)], 1, 0, 3), expected)
        with self.assertRaises(ValueError):
            ref.section_correspondence([first, {**first, "membership": 0.5}], [(0, 3)], 1, 0, 3)

    def test_ring_section_start_pair_masks_and_real_predecessor(self):
        records = [span(i, 2 * i, 2 * i + 1, assignment(cost=0.5 if i % 2 else 0.1)) for i in range(6)]
        out = ref.section_correspondence(records, [(0, 11)], 1, 0, 11)
        self.assertEqual(out["recent"]["occurrence_ids"], [2, 3, 4, 5])
        self.assertEqual(out["recent"]["pair_weight"], 3)
        cut = ref.section_correspondence(records, [(0, 11)], 1, 9, 11)
        self.assertEqual(cut["cumulative"]["occurrence_ids"], [5])
        self.assertEqual(cut["cumulative"]["values"][5:], [None] * 25)
        records[5]["assignment"] = assignment(status="no_memory")
        records[5]["ending_descriptor"] = [2] * 6
        self.assertEqual(ref.section_correspondence(records, [(0, 11)], 1, 9, 11)["cumulative"]["values"][2], 1)
        incomplete = ref.section_correspondence(records[:3], [(0, 3.8), (4, 5)], 1, 0, 5)["cumulative"]
        self.assertEqual(incomplete["excluded_pair_weight"], 1)
        self.assertEqual(incomplete["values"][5:], [None] * 25)

    def test_overlapping_ties_use_supported_order_and_unknown_order_masks_pairs(self):
        records = [span(3, 0.5, 1, assignment(cost=0.5)), span(2, 0, 1), span(1, 0, 1)]
        out = ref.section_correspondence(records, [(0, 1)], 1, 0, 1)["cumulative"]
        self.assertEqual(out["occurrence_ids"], [1, 2, 3])
        self.assertAlmostEqual(sum(out["values"][5:]), 1)
        records[0]["ordering_known"] = False
        self.assertEqual(ref.section_correspondence(records, [(0, 1)], 1, 0, 1)["cumulative"]["values"][5:], [None] * 25)

    def test_alternative_paths_keep_correlated_category_pairs(self):
        exact = [span(1, 0, 1), span(2, 2, 3)]
        transformed = [span(1, 0, 1, assignment(cost=0.5)), span(2, 2, 3, assignment(cost=0.5))]
        a = ref.section_correspondence(exact, [(0, 3)], 1, 0, 3)["cumulative"]["values"][5:]
        b = ref.section_correspondence(transformed, [(0, 3)], 1, 0, 3)["cumulative"]["values"][5:]
        marginal = [(x + y) / 2 for x, y in zip(a, b)]
        self.assertEqual(marginal[0], 0.5)
        self.assertEqual(marginal[6], 0.5)
        self.assertEqual(marginal[1] + marginal[5], 0)


class EndingDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.hops = [{"epoch": 1, "generation": 4, "start": i / 2, "end": (i + 1) / 2,
                      "observed": True, "association_known": True, "energy": 4,
                      "spectrum": [1, 3], "rise": 2, "flux": 4} for i in range(4)]
        self.accents = [{"epoch": 1, "generation": 4, "id": i, "time": (i + .5) / 2, "weight": 0.5,
                         "event_interval": [i / 2, (i + .5) / 2],
                         "raw_support_end": (i + 1) / 2, "available_end": (i + 1) / 2}
                        for i in range(4)]
        self.arguments = {"log2_bins": [1, 3], "epoch": 1, "generation": 4, "epoch_start": 0,
                          "generation_start": 0, "span_start": 0, "support_end": 2,
                          "means": [0] * 6, "standard_deviations": [1] * 6}

    def test_pooled_spectrum_energy_time_means_and_once_counted_accents(self):
        out = ref.ending_descriptor(self.hops, self.accents, **self.arguments)
        expected = [2.5, math.sqrt(0.75), 1, 2, 4, math.log(2)]
        for actual, value in zip(out["raw"], expected):
            self.assertAlmostEqual(actual, value)
        self.assertEqual(out["coverage"], [1] * 6)
        self.assertEqual(out, ref.ending_descriptor(self.hops + [self.hops[0]], self.accents + [self.accents[0]], **self.arguments))

    def test_short_spans_and_ending_generation_do_not_borrow_predecessor_audio(self):
        self.hops[0]["energy"] = 1000
        self.hops[1]["energy"] = 1000
        args = {**self.arguments, "span_start": 1.5}
        out = ref.ending_descriptor(self.hops, self.accents, **args)
        self.assertEqual(out["window"], [1.5, 2])
        self.assertEqual(out["raw"][2], 1)
        self.hops[2]["generation"] = 3
        args = {**self.arguments, "generation_start": 1.5}
        self.assertEqual(ref.ending_descriptor(self.hops, self.accents, **args)["raw"][2], 1)
        self.assertEqual(ref.ending_descriptor(self.hops, self.accents, **{**args, "span_start": 2})["raw"], [None] * 6)

    def test_missing_components_silence_and_cap_loss_remain_distinct(self):
        self.hops[0]["flux"] = None
        self.hops[0]["spectrum"] = None
        out = ref.ending_descriptor(self.hops, self.accents, **self.arguments, capacity_evicted_through=0.5)
        self.assertEqual(out["raw"][:2], [None, None])
        self.assertIsNone(out["raw"][4])
        self.assertIsNone(out["raw"][5])
        self.assertEqual(out["raw"][2:4], [1, 2])
        for hop in self.hops:
            hop.update(energy=0, spectrum=[0, 0], flux=0, rise=0)
        silent = ref.ending_descriptor(self.hops, [], **self.arguments)
        self.assertEqual(silent["raw"][:2], [None, None])
        self.assertEqual(silent["raw"][2], math.log2(1e-6))
        self.assertEqual(silent["raw"][3:], [0, 0, 0])

    def test_original_cut_blocks_future_and_recycled_generation_and_applies_global_scales(self):
        expected = ref.ending_descriptor(self.hops, self.accents, **self.arguments)
        extras = [{**self.hops[0], "generation": 5, "energy": 1000},
                  {**self.hops[0], "epoch": 2, "energy": 1000},
                  {**self.hops[0], "start": 2, "end": 2.5, "energy": 1000}]
        self.assertEqual(expected, ref.ending_descriptor(self.hops + extras, self.accents, **self.arguments))
        scaled = ref.ending_descriptor(self.hops, self.accents, **{**self.arguments, "means": [1] * 6, "standard_deviations": [2] * 6})
        for raw, value in zip(expected["raw"], scaled["scaled"]):
            self.assertAlmostEqual((raw - 1) / 2, value)
        with self.assertRaises(ValueError):
            ref.ending_descriptor(self.hops, self.accents, **self.arguments, capacity_evicted_through=3)


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.hops = [{"epoch": 1, "generation": 4, "start": i / 4, "end": (i + 1) / 4,
                      "observed": True, "association_known": True, "assignment_support": 0.5,
                      "articulation": i, "periodic_proposals": [], "timing_histories": [],
                      "resolved": True, "group_handle": 7, "bus_energy": 10,
                      "resolved_energies": {7: 1}, "residual_energy": 9} for i in range(4)]
        self.accents = [{"epoch": 1, "generation": 4, "id": i, "time": (i + .5) / 4, "weight": 0.5,
                         "event_interval": [i / 4, (i + .5) / 4],
                         "raw_support_end": (i + 1) / 4, "available_end": (i + 1) / 4}
                        for i in range(4)]

    def test_occupancy_and_density_use_assignment_seconds_once(self):
        out = ref.section_activity(self.hops, self.accents * 2, 1, {4}, 0, 1)
        self.assertEqual(out["values"][:4], [0.25] * 4)
        self.assertEqual(out["assignment_seconds"], 0.5)
        self.assertEqual(out["values"][4], 4)
        self.assertEqual(out["unique_accent_count"], 4)
        self.assertEqual(out["values"][6:8], [None, None])
        self.assertEqual(out, ref.section_activity(self.hops * 2, self.accents, 1, {4}, 0, 1))

    def test_periodic_union_includes_nonuniform_word_but_not_best_period_alone(self):
        for hop in self.hops:
            hop["best_period"] = 0.5
        self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][5], 0)
        for kind, steps in [("integer", None), ("word", [1, 1]), ("word", [0.5, 1.5])]:
            for hop in self.hops:
                hop["periodic_proposals"] = [{"kind": kind, "steps": steps, "admitted": True}]
            self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][5], 1)
        self.hops[0]["periodic_proposals"] = None
        self.assertIsNone(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][5])

    def test_residual_energy_alone_never_supplies_resolved_overlap(self):
        self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][8], 0)
        for hop in self.hops:
            hop["resolved_energies"][8] = 0.1
        self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][8], 1)
        for hop in self.hops:
            hop["bus_energy"] = 0
            hop["resolved_energies"] = {7: 0, 8: 0}
        self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][8], 0)
        for hop in self.hops:
            hop["resolved"] = False
        result = ref.section_activity(self.hops, [], 1, {4}, 0, 1)
        self.assertEqual(result["values"][5:], [None] * 4)

    def test_all_supported_outgoing_and_within_histories_preserve_offsets_and_residuals(self):
        histories = [
            {"supported": True, "direction": "outgoing", "periodic": True, "ranking_support": 2,
             "modes": [(0.25, 1), (0.75, 1)], "residual_squared_mean": 0.01},
            {"supported": True, "direction": "outgoing", "periodic": False, "ranking_support": 1,
             "modes": [(1, 1)], "residual_squared_mean": 0.09},
            {"supported": True, "direction": "within", "periodic": True, "ranking_support": 1,
             "modes": [(0, 1)], "residual_squared_mean": 0.04},
            {"supported": True, "direction": "incoming", "periodic": True, "ranking_support": 1000,
             "modes": [(0, 1)], "residual_squared_mean": 0},
        ]
        for hop in self.hops:
            hop["timing_histories"] = histories
        out = ref.section_activity(self.hops, [], 1, {4}, 0, 1)
        self.assertAlmostEqual(out["values"][6], 0.140625)
        self.assertAlmostEqual(out["values"][7], 0.0375)
        for history in histories:
            history["modes"] = []
        self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][6:8], [None, None])

    def test_known_inactivity_missing_support_and_reused_slots_stay_distinct(self):
        for hop in self.hops:
            hop["articulation"] = 3
        self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"][:4], [0, 0, 0, 1])
        self.hops[0]["observed"] = False
        self.assertEqual(ref.section_activity(self.hops, [], 1, {4}, 0, 1)["values"], [None] * 9)
        self.assertEqual(ref.section_activity(self.hops, [], 1, {5}, 0, 1)["assignment_seconds"], 0)
        self.assertEqual(ref.section_activity(self.hops, [], 2, {4}, 0, 1)["values"], [None] * 9)


class CueTests(unittest.TestCase):
    @staticmethod
    def cue(identifier, start, intervals, kind="ongoing", epoch=1):
        return {"epoch": epoch, "record_kind": kind, "occurrence_id": identifier, "start": start,
                "support_intervals": intervals, "descriptor_support_end": max(row[2] for row in intervals),
                "query_descriptor": {"whole_prefix_for": identifier}}

    def test_weighted_recent_support_and_actual_generation_select_whole_prefix(self):
        old = self.cue(1, 0, [(3, 0, 9.5, 1), (4, 9.5, 10, 0.1)])
        new = self.cue(2, 9.75, [(4, 9.75, 10, 0.8)])
        wrong_generation = self.cue(3, 0, [(5, 9.5, 10, 1)])
        out = ref.select_section_cue([old, new, wrong_generation], 1, {3, 4}, 10, 0)
        self.assertEqual(out["occurrence_id"], 2)
        self.assertIs(out["query_descriptor"], new["query_descriptor"])
        self.assertAlmostEqual(out["selection_weighted_seconds"], 0.2)
        split = copy.deepcopy(new)
        split["support_intervals"] = [(4, 9.75, 9.875, 0.8), (4, 9.875, 10, 0.8)] * 2
        self.assertEqual(out, ref.select_section_cue([old, split], 1, {3, 4}, 10, 0))

    def test_support_ties_and_committed_fallback_use_timestamp_start_and_stable_id(self):
        earlier = self.cue(1, 9.5, [(4, 9.5, 9.75, 1)])
        later = self.cue(2, 9.75, [(4, 9.75, 10, 1)])
        tied = self.cue(0, 9.75, [(4, 9.75, 10, 1)])
        self.assertEqual(ref.select_section_cue([earlier, later, tied], 1, {4}, 10, 0)["occurrence_id"], 0)
        for candidate in (earlier, later, tied):
            candidate["record_kind"] = "observed_commit"
        self.assertEqual(ref.select_section_cue([earlier, later, tied], 1, {4}, 11, 0)["occurrence_id"], 0)
        self.assertIsNone(ref.select_section_cue([earlier, later, tied], 2, {4}, 11, 0))

    def test_stale_or_different_cue_query_cannot_supply_a_fresh_best_match(self):
        selected = ref.select_section_cue([self.cue(1, 9.5, [(4, 9.5, 10, 1)])], 1, {4}, 10, 0)
        query = {"cue_occurrence_id": 1, "observation_end": 10, "support_end": 10, "published_at": 1000,
                 "flags": {"truncated": True}, "matches": [
                     {"episode_handle": 1, "eligible": True, "supported": True, "acoustic_score": 1, "log_availability": 100},
                     {"episode_handle": 1, "eligible": True, "supported": True, "acoustic_score": 2, "log_availability": -100},
                     {"episode_handle": 2, "eligible": True, "supported": True, "acoustic_score": 1.75},
                     {"episode_handle": 3, "eligible": False, "supported": True, "acoustic_score": 100}]}
        result = ref.section_retrieval_scores(query, selected, 10.25)
        self.assertEqual(result["scores"], [2, 1.75])
        self.assertEqual(result["gap"], 0.25)
        self.assertEqual(result["flags"], query["flags"])
        self.assertEqual(ref.section_retrieval_scores(query, selected, 10.5)["scores"], [])
        with self.assertRaises(ValueError):
            ref.section_retrieval_scores({**query, "cue_occurrence_id": 2}, selected, 10.25)
        self.assertEqual(ref.section_retrieval_scores(query, None, 10.25)["scores"], [])


class SectionInputIntegrationTests(unittest.TestCase):
    def test_contiguous_commits_feed_cumulative_recent_and_causal_82_coordinate_head(self):
        records = [span(i, i, i + 1, assignment(cost=0.5 if i < 2 else 0.1)) for i in range(6)]
        histories = [{"supported": True, "direction": "within", "periodic": True,
                      "ranking_support": 1, "modes": [(0.25, 1)], "residual_squared_mean": 0.04}]
        hops = [{"epoch": 1, "generation": 4, "start": i, "end": i + 1,
                 "observed": True, "association_known": True, "assignment_support": 0.5,
                 "articulation": 0 if i < 2 else 1, "periodic_proposals": [{"kind": "word", "admitted": True}],
                 "timing_histories": histories, "resolved": True, "group_handle": 7,
                 "bus_energy": 2, "resolved_energies": {7: 1, 8: 1}} for i in range(6)]
        relations = ref.section_correspondence(records, [(0, 6)], 1, 0, 6)
        cumulative = relations["cumulative"]["values"] + ref.section_activity(hops, [], 1, {4}, 0, 6)["values"]
        recent = relations["recent"]["values"] + ref.section_activity(hops, [], 1, {4}, 2, 6)["values"]
        head = cognition.section_head_covariates(cumulative, recent, 0, 0, 6, [(0, 6)],
                                                [0.5, 0.4], 6, [0] * 82, [1] * 82, forecast_time=7)
        self.assertEqual(len(cumulative), 39)
        self.assertEqual(len(head["raw"]), 82)
        self.assertAlmostEqual(head["raw"][0], 4 / 6)
        self.assertAlmostEqual(head["raw"][39], 1 / 3)
        self.assertAlmostEqual(head["raw"][30], 1 / 3)
        self.assertAlmostEqual(head["raw"][69], -1 / 3)
        self.assertEqual(head["raw"][35:39], [1, 0.25, 0.04, 1])
        self.assertAlmostEqual(head["raw"][78], math.log(8))
        self.assertEqual(head["window"], [0, 6])
        self.assertEqual(head["raw"][79], 0)
        self.assertAlmostEqual(head["raw"][81], 0.1)


def cached_hop(start, end, state=0, support=0.5, **changes):
    return {"epoch": 1, "generation": 4, "start": start, "end": end,
            "observed": True, "association_known": True, "assignment_support": support,
            "articulation": state, "periodic_proposals": [], "timing_histories": [],
            "resolved": True, "group_handle": 7, "bus_energy": 10,
            "resolved_energies": {7: 1}, **changes}


class SpanPartitionTests(unittest.TestCase):
    def setUp(self):
        self.spans = {1: {"start": 0, "support_end": 2}, 2: {"start": 1, "support_end": 3}}
        self.hops = [cached_hop(0, 1, 0, span_shares={1: 0.25}),
                     cached_hop(1, 2, 1, span_shares={1: 0.25, 2: 0.75}),
                     cached_hop(2, 3, 2, span_shares={2: 0.75})]
        self.accents = [{"epoch": 1, "generation": 4, "id": 11, "time": 1.5,
                         "event_interval": [1.25, 1.5], "raw_support_end": 1.75, "available_end": 1.75,
                         "weight": 0.8, "span_shares": {1: 0.25, 2: 0.75}}]

    def test_overlapping_owned_support_matches_hand_calculated_seconds_and_once_only_accent(self):
        result = ref.partition_span_activity(self.hops, self.accents, self.spans, 1, {4}, 3)
        self.assertEqual(result[1]['assignment_seconds'], 0.25)
        self.assertEqual(result[2]['assignment_seconds'], 0.75)
        self.assertEqual(result[1]['physical_window_seconds'], 0.5)
        self.assertEqual(result[2]['physical_window_seconds'], 1.5)
        self.assertEqual(result[1]['values'][:4], [0.5, 0.5, 0, 0])
        self.assertEqual(result[2]['values'][:4], [0, 0.5, 0.5, 0])
        self.assertAlmostEqual(sum(s['numerators'][4] for s in result.values()), 0.8)
        self.assertAlmostEqual(result[1]['values'][4], 0.8)
        self.assertAlmostEqual(result[2]['values'][4], 0.8)
        self.assertEqual(result, ref.partition_span_activity(self.hops*2, self.accents*3, self.spans, 1, {4}, 3))

    def test_partition_clips_inside_a_complete_raw_hop_and_checks_overlap_pointwise(self):
        spans = {1: {"start": 0, "support_end": 0.5}, 2: {"start": 0.5, "support_end": 1}}
        hop = cached_hop(0, 1, span_shares={1: 1, 2: 1})
        result = ref.partition_span_activity([hop], [], spans, 1, {4}, 1)
        self.assertEqual([s['assignment_seconds'] for s in result.values()], [0.25, 0.25])
        with self.assertRaises(ValueError):
            ref.partition_span_activity([hop], [], {1: spans[1]}, 1, {4}, 0.5)
        spans[1]['support_end'] = 0.75
        with self.assertRaises(ValueError):
            ref.partition_span_activity([hop], [], spans, 1, {4}, 1)
        self.accents[0]['span_shares'] = {1: 0.8, 2: 0.8}
        with self.assertRaises(ValueError):
            ref.partition_span_activity(self.hops, self.accents, self.spans, 1, {4}, 3)

    def test_missing_clock_is_explicit_and_fractional_ownership_does_not_hide_input_loss(self):
        self.hops[1]['observed'] = False
        result = ref.partition_span_activity(self.hops, [], self.spans, 1, {4}, 3)
        self.assertEqual(result[1]['physical_window_seconds'], 0.5)
        self.assertEqual(result[1]['coverage'][:6], [0.5]*6)
        self.assertEqual(result[1]['values'], [None]*9)
        self.assertEqual(result[2]['values'], [None]*9)
        with self.assertRaises(ValueError):
            ref.partition_span_activity(self.hops[::2], [], self.spans, 1, {4}, 3)

    def test_epoch_generation_future_and_conflicting_aliases_cannot_add_support(self):
        expected = ref.partition_span_activity(self.hops, self.accents, self.spans, 1, {4}, 3)
        extras = [{**self.hops[0], "epoch": 2}, {**self.hops[0], "generation": 5},
                  {**self.hops[0], "start": 3, "end": 4}]
        self.assertEqual(expected, ref.partition_span_activity(self.hops+extras, self.accents, self.spans, 1, {4}, 3))
        with self.assertRaises(ValueError):
            ref.partition_span_activity(self.hops + [{**self.hops[0], "span_shares": {1: 0.5}}],
                                        self.accents, self.spans, 1, {4}, 3)

    def test_partition_ring_and_cumulative_have_distinct_support_and_reach_82_inputs(self):
        parts = ref.partition_span_activity(self.hops, self.accents, self.spans, 1, {4}, 3)
        history = ref.SectionHistory(1, 4, 100, 0)
        full = ref.section_activity(self.hops, self.accents, 1, {4}, 0, 3)
        history.observe(full, 1, 4)
        for i, membership in ((1, 0.5), (2, 0.25)):
            record = span(i, **{'start': self.spans[i]['start'], 'end': self.spans[i]['support_end']},
                          category=assignment(cost=0.1 if i == 1 else 0.5), membership=membership)
            record['assignment_seconds'] = parts[i]['assignment_seconds']
            history.commit(record, parts[i], i, 1)
        snapshot = history.snapshot()
        self.assertEqual(snapshot['missing_fraction'], 0)
        self.assertAlmostEqual(snapshot['recent']['activity']['assignment_seconds'], 0.3125)
        for value, expected in zip(snapshot['recent']['values'][30:35], [0.2, 0.5, 0.3, 0, 0.8]):
            self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(snapshot['cumulative']['values'][34], 0.8/1.5)
        head = cognition.section_head_covariates(snapshot['cumulative']['values'], snapshot['recent']['values'],
                                                0, 0, 3, [(0, 3)], [], None, [0]*82, [1]*82)
        self.assertEqual(len(head['raw']), 82)
        self.assertAlmostEqual(head['raw'][30], 1/3)
        self.assertAlmostEqual(head['raw'][69], 0.2-1/3)
        self.assertEqual(head['raw'][79], 0)


class SectionHistoryTests(unittest.TestCase):
    def test_sealed_statistics_keep_all_ordered_pairs_and_dense_zero_additions(self):
        alternatives = (assignment(), assignment(cost=.5), assignment('no_memory', cost=None),
                        assignment('no_memory', cost=None), assignment('unknown'))
        for prior_category in range(5):
            for category, alternative in enumerate(alternatives):
                for known, coverage in ((True, .9), (True, .8999999999), (False, 1.)):
                    with self.subTest(prior=prior_category, current=category, known=known, coverage=coverage):
                        history = ref.SectionHistory(1, 4, 100, 0)
                        activity = ref.section_activity([cached_hop(0, 1, support=.75)], [], 1, {4}, 0, 1)
                        history.observe(activity, 1, 4)
                        initial = [-0. if i % 3 == 0 else (i+1)/8 for i in range(31)]
                        struct.pack_into('<31d', history.cumulative.buffer, 0, *initial)
                        history.cumulative.ending([0.]*6)
                        history.cumulative.integer(11, 1)
                        prior = ref.SectionRecord()
                        struct.pack_into('<d', prior.buffer, 8*prior_category, .25)
                        prior.integer(14, 1)
                        history.ring = [prior]
                        record = span(2, 0, 1, alternative, .5)
                        record['assignment_seconds'] = .75
                        record['ordering_known'] = known
                        record['ending_descriptor'] = [2.]*6 if category == 2 else [-0., None, 0., None, 0., None]
                        before = bytes(history.cumulative.buffer)
                        prepared = history._prepare_commit(record, activity, 1, coverage)
                        self.assertEqual(bytes(history.cumulative.buffer), before)
                        self.assertTrue(history._apply_prepared(prepared))
                        additions = [0.]*31
                        additions[category], additions[30] = .375, .25
                        if known and coverage >= .9:
                            additions[5+5*prior_category+category] = .25
                        expected = struct.pack('<31d', *(a+b for a, b in zip(initial, additions)))
                        self.assertEqual(history.cumulative.buffer[:248], expected)
                        self.assertEqual(history.ring[-1].buffer[:248], struct.pack('<31d', *additions))
                        self.assertEqual(history.cumulative.buffer[248:432], before[248:432])
                        self.assertEqual(history.cumulative.buffer[552:576], before[552:576])
                        self.assertEqual(history.cumulative.ending(), record['ending_descriptor'])

    def test_sealed_statistic_overflow_or_inactive_nan_rejects_without_effects(self):
        activity = dict(numerators=[0.]*9, denominators=[1.]*9, physical_valid_seconds=[1.]*9,
                        assignment_seconds=1e308, physical_window_seconds=1e308, window=[0., 1e308])
        for index, bad in ((0, 1e308), (5, 1e308), (30, 1e308), (4, float('nan')), (29, float('nan'))):
            with self.subTest(index=index, bad=bad):
                history = ref.SectionHistory(1, 4, 100, 0)
                history.cumulative.time(5, 1e308)
                struct.pack_into('<d', history.cumulative.buffer, 8*index, bad)
                prior = ref.SectionRecord()
                struct.pack_into('<d', prior.buffer, 0, 1e308)
                prior.integer(14, 1)
                history.ring = [prior]
                before = bytes(history.cumulative.buffer), [bytes(r.buffer) for r in history.ring]
                with self.assertRaises(ValueError):
                    history._prepare_commit(span(1, 0, 1e308), activity, 1, 1)
                self.assertEqual((bytes(history.cumulative.buffer), [bytes(r.buffer) for r in history.ring]), before)

    def test_activity_merge_preserves_literal_field_values_and_other_record_bytes(self):
        activity = dict(numerators=[i/10 for i in range(9)], denominators=[.75]*4+[.5]*5,
                        physical_valid_seconds=[.9]*4+[.8]*5, assignment_seconds=.625, physical_window_seconds=1.)
        for physical in (None, 3.):
            with self.subTest(physical=physical):
                record = ref.SectionRecord(bytes([0xa5])*640)
                struct.pack_into('<54d', record.buffer, 0, *[i/8 for i in range(54)])
                expected = bytearray(record.buffer)
                for field in ('numerators', 'denominators', 'physical_valid_seconds', 'assignment_seconds', 'physical_window_seconds'):
                    values = activity[field] if isinstance(activity[field], list) else [activity[field]]
                    if field == 'physical_window_seconds' and physical is not None:
                        values = [physical]
                    for index, value in zip(ref.SectionRecord.fields[field], values):
                        previous = struct.unpack_from('<d', record.buffer, index*8)[0]
                        struct.pack_into('<d', expected, index*8, previous+.375*value)
                record.add_activity(activity, .375, physical)
                self.assertEqual(record.buffer, expected)
                self.assertEqual(len(record.buffer), 640)

    def test_activity_merge_rejects_late_invalid_values_without_partial_statistics(self):
        activity = dict(numerators=[1.]*9, denominators=[1.]*9, physical_valid_seconds=[1.]*9,
                        assignment_seconds=1., physical_window_seconds=1.)
        for field in activity:
            for invalid in (-1., float('inf'), float('nan')):
                for scale in (0., 1.):
                    with self.subTest(field=field, invalid=invalid, scale=scale):
                        values = copy.deepcopy(activity)
                        if isinstance(values[field], list):
                            values[field][-1] = invalid
                        else:
                            values[field] = invalid
                        record = ref.SectionRecord()
                        before = bytes(record.buffer)
                        with self.assertRaises(ValueError):
                            record.add_activity(values, scale)
                        self.assertEqual(bytes(record.buffer), before)
        record = ref.SectionRecord()
        record.add('physical_window_seconds', [sys.float_info.max])
        before = bytes(record.buffer)
        activity['physical_window_seconds'] = sys.float_info.max
        with self.assertRaises(ValueError):
            record.add_activity(activity)
        self.assertEqual(bytes(record.buffer), before)

    def test_activity_merge_preserves_rounded_aliases_and_signed_zero(self):
        activity = dict(numerators=[0.]*9, denominators=[0.]*9, physical_valid_seconds=[0.]*9,
                        assignment_seconds=0., physical_window_seconds=0.)
        for field in ('denominators', 'physical_valid_seconds'):
            with self.subTest(field=field):
                record = ref.SectionRecord()
                record.add(field, [2.**53]*9)
                before = bytes(record.buffer)
                values = copy.deepcopy(activity)
                values[field][:4] = [1., 0., 1., 0.]
                record.add_activity(values)
                self.assertEqual(bytes(record.buffer), before)
                values[field][0] = 2.
                with self.assertRaises(ValueError):
                    record.add_activity(values)
                self.assertEqual(bytes(record.buffer), before)
                struct.pack_into('<d', record.buffer, ref.SectionRecord.fields[field][0]*8, -0.)
                before = bytes(record.buffer)
                values = copy.deepcopy(activity)
                values[field][:4] = [-0., 0., -0., -0.]
                with self.assertRaises(ValueError):
                    record.add_activity(values)
                self.assertEqual(bytes(record.buffer), before)
                values[field][:4] = [-0.]*4
                record.add_activity(values)
                self.assertEqual(bytes(record.buffer), before)

    def test_block_updates_match_literal_offsets_and_preserve_other_bytes(self):
        for field, indices in ref.SectionRecord.fields.items():
            with self.subTest(field=field):
                record = ref.SectionRecord(bytes([0xa5])*640)
                struct.pack_into('<54d', record.buffer, 0, *[.25+i for i in range(54)])
                values = [(i+1)/8 for i in indices]
                expected = bytearray(record.buffer)
                for i, value in zip(indices, values):
                    old = struct.unpack_from('<d', record.buffer, 8*i)[0]
                    struct.pack_into('<d', expected, 8*i, old+.375*value)
                record.add(field, values, .375)
                self.assertEqual(record.buffer, expected)
                self.assertEqual(record.block(field), [struct.unpack_from('<d', expected, 8*i)[0] for i in indices])
                self.assertEqual(len(record.buffer), 640)

    def test_block_rejection_is_atomic_for_late_invalid_value_and_overflow(self):
        for field, indices in ref.SectionRecord.fields.items():
            for invalid in (-1., float('inf'), float('nan')):
                with self.subTest(field=field, invalid=invalid):
                    record = ref.SectionRecord()
                    before = bytes(record.buffer)
                    values = [1.]*len(indices)
                    values[-1] = invalid
                    with self.assertRaises(ValueError):
                        record.add(field, values)
                    self.assertEqual(bytes(record.buffer), before)
            record = ref.SectionRecord()
            maximum = sys.float_info.max
            record.add(field, [maximum]*len(indices))
            before = bytes(record.buffer)
            with self.assertRaises(ValueError):
                record.add(field, [maximum]*len(indices))
            self.assertEqual(bytes(record.buffer), before)

    def test_shared_denominators_compare_rounded_totals_and_signed_zero(self):
        for field in ('denominators', 'physical_valid_seconds'):
            with self.subTest(field=field):
                record = ref.SectionRecord()
                record.add(field, [2.**53]*9)
                before = bytes(record.buffer)
                record.add(field, [1., 0., 1., 0., 0., 0., 0., 0., 0.])
                self.assertEqual(bytes(record.buffer), before)
                with self.assertRaises(ValueError):
                    record.add(field, [2., 0., 0., 0., 0., 0., 0., 0., 0.])
                self.assertEqual(bytes(record.buffer), before)
                struct.pack_into('<d', record.buffer, ref.SectionRecord.fields[field][0]*8, -0.)
                before = bytes(record.buffer)
                with self.assertRaises(ValueError):
                    record.add(field, [-0., 0., -0., -0., 0., 0., 0., 0., 0.])
                self.assertEqual(bytes(record.buffer), before)
                record.add(field, [-0.]*9)
                self.assertEqual(bytes(record.buffer), before)

    def test_prepared_commit_preserves_later_observations_and_once_only_accents(self):
        for section_start in (0., 2.):
            with self.subTest(section_start=section_start):
                history = ref.SectionHistory(1, 4, 100, section_start)
                activity = ref.section_activity([cached_hop(0, 1, support=1)], [], 1, {4}, 0, 1)
                if section_start == 0:
                    history.observe(activity, 1, 4)
                record = span(1, 0, 1, membership=.375)
                serial = copy.deepcopy(history)
                before = bytes(history.cumulative.buffer)
                prepared = history._prepare_commit(record, activity, 1, 1)
                self.assertEqual(bytes(history.cumulative.buffer), before)
                self.assertEqual(history.ring, [])
                start = history.cumulative.time(5)
                accent = dict(epoch=1, generation=4, event_interval=[start, start+.1],
                              time=start+.1, raw_support_end=start+.2, available_end=start+.3, weight=.625)
                for i in range(2):
                    lo, hi = start+i, start+i+1
                    delta = ref.section_activity([cached_hop(lo, hi, articulation=2, support=1)], [], 1, {4}, lo, hi)
                    delivery = dict(sequence=1, accent=accent, weight=.625, delivered_at=lo+.5)
                    for target in (history, serial):
                        self.assertTrue(target.observe(delta, 1, 4, [delivery]))
                observed = history.cumulative.snapshot()
                self.assertTrue(history._apply_prepared(prepared))
                self.assertTrue(serial.commit(record, activity, 1, 1))
                self.assertEqual(history.cumulative.buffer, serial.cumulative.buffer)
                self.assertEqual([r.buffer for r in history.ring], [r.buffer for r in serial.ring])
                self.assertEqual(history.cumulative.snapshot()['values'][30:], observed['values'][30:])
                self.assertEqual(history.cumulative.block('numerators')[4], .625)
                self.assertEqual(history.cumulative.integer(15), 1)
                self.assertEqual(history.cumulative.time(5), start+2)
                self.assertEqual(len(history.ring), int(section_start == 0))
                frozen = bytes(history.cumulative.buffer)
                with self.assertRaises(ValueError):
                    history._apply_prepared(prepared)
                self.assertEqual(bytes(history.cumulative.buffer), frozen)

    def test_prepared_commit_rejects_changed_identity_or_sealed_predecessor(self):
        history = ref.SectionHistory(1, 4, 100, 0.)
        activity = ref.section_activity([cached_hop(0, 1, support=1)], [], 1, {4}, 0, 1)
        history.observe(activity, 1, 4)
        prepared = history._prepare_commit(span(1, 0, 1), activity, 1, 1)
        changes = (
            lambda h: h.cumulative.integer(0, 2),
            lambda h: h.cumulative.integer(1, 5),
            lambda h: h.cumulative.integer(2, 101),
            lambda h: h.cumulative.time(4, .25),
            lambda h: setattr(h, 'ring_size', 8),
            lambda h: h.commit(span(2, 0, 1), activity, 1, 1),
        )
        for i, change in enumerate(changes):
            with self.subTest(change=i):
                target = copy.deepcopy(history)
                change(target)
                before = bytes(target.cumulative.buffer), [bytes(r.buffer) for r in target.ring]
                with self.assertRaises(ValueError):
                    target._apply_prepared(prepared)
                self.assertEqual((bytes(target.cumulative.buffer), [bytes(r.buffer) for r in target.ring]), before)

    def test_packed_layout_retains_small_increments_and_preserves_ending_thresholds(self):
        record = ref.SectionRecord()
        self.assertEqual(len(record.buffer), struct.calcsize('<128f16Q'))
        record.add('edge_numerators', [1e7, 0, 0, 0, 0])
        for _ in range(10000):
            record.add('edge_numerators', [0.0001, 0, 0, 0, 0])
        self.assertAlmostEqual(record.block('edge_numerators')[0], 1e7+1, delta=1e-10+1e-12*(1e7+1))
        ending = [1+1e-8, None, -0.01, 0, 2, 3]
        record.ending(ending)
        restored = ref.SectionRecord(bytes(record.buffer))
        self.assertEqual(restored.ending(), ending)
        self.assertEqual(restored.snapshot(), record.snapshot())
        self.assertEqual(ref.correspondence_category(assignment('no_memory'), [2+1e-8, None, None, None, None, None],
                                                     restored.ending()), 3)

    def test_all_prefixes_and_recent_window_sizes_match_the_exhaustive_correspondence_oracle(self):
        for size in (2, 4, 8):
            history = ref.SectionHistory(1, 4, 100, 0, ring_size=size)
            records = []
            for i in range(20):
                hop = cached_hop(i, i+1, i%4, (i%5+1)/5)
                activity = ref.section_activity([hop], [], 1, {4}, i, i+1)
                history.observe(activity, 1, 4)
                record = span(i, i, i+1, assignment('unknown' if i%5 == 1 else 'match', cost=0.5 if i%3 else 0.1),
                              (i%3+1)/4)
                record['assignment_seconds'] = activity['assignment_seconds']
                record['ordering_known'] = i%7 != 2
                records.append(record)
                history.commit(record, activity, i+1, 1)
                actual = history.snapshot()
                expected = ref.section_correspondence(records, [(0, i+1)], 1, 0, i+1, size)
                for part in ('cumulative', 'recent'):
                    for value, reference in zip(actual[part]['values'][:30], expected[part]['values']):
                        if reference is None:
                            self.assertIsNone(value)
                        else:
                            self.assertAlmostEqual(value, reference, delta=1e-12)
                    self.assertAlmostEqual(actual[part]['pair_weight'], expected[part]['pair_weight'], delta=1e-12)
                self.assertEqual(actual['occurrence_ids'], expected['recent']['occurrence_ids'])
                self.assertLessEqual(actual['record_payload_bytes'], 640*(1+size))
                recent = records[-size:]
                weights = [r['assignment_seconds']*r['membership'] for r in recent]
                for state in range(4):
                    expected_occupancy = sum(w for r, w in zip(recent, weights) if r['occurrence_id']%4 == state)/sum(weights)
                    self.assertAlmostEqual(actual['recent']['values'][30+state], expected_occupancy, delta=1e-12)

    def test_duplicate_delta_sealed_replay_alias_and_out_of_order_do_not_gain_credit(self):
        history = ref.SectionHistory(1, 4, 100, 0)
        first = None
        for i in range(6):
            activity = ref.section_activity([cached_hop(i, i+1, support=1)], [], 1, {4}, i, i+1)
            record = span(i, i, i+1)
            history.observe(activity, 1, 4)
            before = bytes(history.cumulative.buffer)
            self.assertFalse(history.observe(copy.deepcopy(activity), 1, 4))
            self.assertEqual(bytes(history.cumulative.buffer), before)
            history.commit(record, activity, i+1, 1)
            before = history.snapshot()
            self.assertFalse(history.commit(record, activity, i+1, 1))
            self.assertFalse(history.commit({**record, 'record_kind': 'alias'}, activity, i+10, 1))
            self.assertEqual(history.snapshot(), before)
            if first is None:
                first = (record, activity)
        with self.assertRaises(ValueError):
            history.commit(*first, 1, 1)
        with self.assertRaises(ValueError):
            history.commit(*first, 7, 1)
        self.assertEqual(history.snapshot(), before)

    def test_actual_generation_inheritance_isolated_from_parent_fresh_slots_and_epochs(self):
        parent = ref.SectionHistory(1, 4, 100, 0)
        activity = ref.section_activity([cached_hop(0, 1, support=1)], [], 1, {4}, 0, 1)
        parent.observe(activity, 1, 4)
        parent.commit(span(1, 0, 1), activity, 1, 1)
        frozen = parent.snapshot()
        child = parent.inherit(4, 5)
        later = ref.section_activity([cached_hop(1, 2, generation=5)], [], 1, {5}, 1, 2)
        self.assertFalse(child.observe(later, 1, 4))
        self.assertFalse(child.observe(later, 2, 5))
        self.assertTrue(child.observe(later, 1, 5))
        self.assertEqual(parent.snapshot(), frozen)
        self.assertEqual(child.snapshot()['occurrence_ids'], [1])
        fresh = ref.SectionHistory(1, 5, 101, 1)
        self.assertEqual(fresh.snapshot()['occurrence_ids'], [])
        self.assertEqual(fresh.snapshot()['cumulative']['values'], [None]*39)
        with self.assertRaises(ValueError):
            parent.inherit(6, 5)

    def test_new_section_uses_real_predecessor_without_borrowing_a_record_or_crossing_pair(self):
        history = ref.SectionHistory(1, 4, 100, 2)
        previous_activity = ref.section_activity([cached_hop(0, 1, support=1)], [], 1, {4}, 0, 1)
        history.commit(span(1, 0, 1), previous_activity, 1, 1)
        activity = ref.section_activity([cached_hop(2, 3, support=1)], [], 1, {4}, 2, 3)
        history.observe(activity, 1, 4)
        record = span(2, 2, 3, assignment('no_memory'))
        record['ending_descriptor'] = [2]*6
        history.commit(record, activity, 2, 1)
        out = history.snapshot()
        self.assertEqual(out['occurrence_ids'], [2])
        self.assertEqual(out['cumulative']['values'][:5], [0, 0, 1, 0, 0])
        self.assertEqual(out['cumulative']['values'][5:30], [None]*25)
        self.assertEqual(out['missing_fraction'], 0)

    def test_missing_audio_coverage_is_independent_of_span_membership_and_old_generation(self):
        history = ref.SectionHistory(1, 4, 100, 0)
        for start, end in ((0, 1), (2, 3)):
            activity = ref.section_activity([cached_hop(start, end, support=0.1)], [], 1, {4}, start, end)
            history.observe(activity, 1, 4)
        self.assertAlmostEqual(history.snapshot()['missing_fraction'], 1/3)
        self.assertEqual(history.snapshot()['cumulative']['values'][30:], [None]*9)
        empty = ref.SectionHistory(2, 4, 100, 0)
        self.assertFalse(empty.observe(activity, 1, 4))
        self.assertIsNone(empty.snapshot()['missing_fraction'])

    def test_invalid_sealed_support_and_future_commit_leave_history_unchanged(self):
        history = ref.SectionHistory(1, 4, 100, 0)
        activity = ref.section_activity([cached_hop(0, 1, support=1)], [], 1, {4}, 0, 1)
        history.observe(activity, 1, 4)
        before = history.snapshot()
        for record in (span(1, 0, 2), {**span(1, 0, 1), 'assignment_seconds': 0.5},
                       {**span(1, 0, 1), 'membership': 2}):
            with self.assertRaises(ValueError):
                history.commit(record, activity, 1, 1)
            self.assertEqual(history.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
