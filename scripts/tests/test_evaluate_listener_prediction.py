"""Regression checks for prospective support, fallback and infinite-loss accounting."""
from _bootstrap import SCRIPTS

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from evaluate_listener_prediction import evaluate


class ListenerPredictionTests(unittest.TestCase):
    def fixture(self, version, known, baseline, coverage, outcome):
        q = None if baseline is None else [coverage * k + (1-coverage) * b
                                         for k, b in zip(known or [0]*5, baseline)]
        f = dict(version=version, start_sample=10, end_sample=60, reference_sample=10,
                 reference_bin=1, examples=version, p_known=known, marginal=baseline,
                 coverage_score=coverage, q=q, arrival_score=None if q is None else sum(q[1:4]))
        done = dict(forecast=f, outcome=outcome, observed_sample=60,
                    zero_likelihood=q is not None and q[outcome] == 0,
                    surprise_nats=-math.log(q[outcome]) if q and q[outcome] else None)
        return dict(type="listener_state", prediction=dict(forecast=f, completed=done, censored=None))

    def run_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return evaluate(path)

    def test_fallback_population_and_infinite_issued_loss_are_retained(self):
        cold = self.fixture(1, None, None, 0, 2)
        fallback = self.fixture(2, None, [.25, 0, .75, 0, 0], 0, 2)
        partial = self.fixture(3, [0, 0, 0, 1, 0], [.25, 0, .75, 0, 0], .2, 2)
        zero = self.fixture(4, [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], .5, 0)
        result = self.run_rows([cold, fallback, partial, zero, zero])
        self.assertEqual((result["completed"], result["supported"], result["unsupported"]), (4, 3, 1))
        self.assertEqual(result["known_unsupported"], 2)
        self.assertEqual(result["E0"]["infinite_log_losses"], 1)
        self.assertEqual(result["E0"]["mean_log_loss_nats"], "Infinity")
        self.assertEqual(result["E0"]["arrival_infinite_log_losses"], 1)
        self.assertTrue(result["registered_probability_rejection"])
        self.assertAlmostEqual(result["E0"]["brier"], (0.125 + .24 + 2) / 3)

    def test_censored_version_cannot_be_completed(self):
        row = self.fixture(1, None, None, 0, 0)
        f = row["prediction"]["forecast"]
        f["start_sample"] = 30
        cancelled = copy.deepcopy(row)
        cancelled["prediction"]["completed"] = None
        cancelled["prediction"]["censored"] = dict(forecast=f, observed_sample=20)
        result = self.run_rows([cancelled, cancelled])
        self.assertEqual((result["censored"], result["completed"]), (1, 0))
        with self.assertRaises(AssertionError):
            self.run_rows([cancelled, row])

    def test_impossible_support_is_rejected(self):
        row = self.fixture(1, None, [0, 1, 0, 0, 0], 0, 1)
        with self.assertRaises(AssertionError):
            self.run_rows([row])


if __name__ == "__main__":
    unittest.main()
